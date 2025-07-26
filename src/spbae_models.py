from dataclasses import dataclass

import numpy as np
from itertools import permutations, combinations

import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import numpy.linalg as nla
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

from sklearn.decomposition import NMF
from sklearn.cluster import k_means, KMeans

# my code
import util
from sbmf import sparse as search

####################################################################
############## Matrix factorization classes ########################
####################################################################

@dataclass
class SpBMF:

    temp: float = 1e-3

    def metrics(self, X):
        pass

    def EStep(self, X):
        """
        Update of the discrete parameters
        """
        return NotImplementedError

    def MStep(self, X):
        """
        Update of the continuous parameters
        """
        return NotImplementedError

    def grad_step(self, X):
        """
        One iteration of optimization
        """

        E = self.EStep()
        loss = self.MStep(E)

        return loss

    def initialize(self):

        raise NotImplementedError

@dataclass
class SparseSemiBMF(SpBMF):
    """
    Generalized BAE, taking any exponential family observation (in theory)
    """

    dim_hid: int
    tree_reg: float = 1e-2
    sparse_reg: float = 0
    weight_reg: float = 1e-2
    nonneg: bool = False
    fit_intercept: bool = True

    def __call__(self):
        return self.S.dot(self.W.T) + self.b

    def initialize(self, X, hot=True, batch_size=1, lr=1e-1, W_lr=0.1, b_lr=0.1):

        self.n, self.d = X.shape
        self.data = X/np.sqrt(np.mean(X**2))

        self.W_lr = W_lr
        self.b_lr = b_lr
        self.bsz = batch_size
        self.lr = lr

        self.b = np.zeros(self.d) # Initialize b

        if self.nonneg and hot: # Initialize with NMF
            nmf = NMF(self.dim_hid)
            Z = nmf.fit_transform(self.data)
            # print('Fit NMF')
            S = []
            kmn = KMeans(2, n_init=1)
            for i in range(Z.shape[1]):
                S.append(kmn.fit_predict(Z[:,[i]]))

            S = sprs.csr_array(np.array(S))
            StS = S.T@S/len(S) - np.outer(S.mean(0), S.mean(0))
            self.S = search.BiMat(S.indices, S.indptr, self.dim_hid)
            self.sampler = search.GaussianSampler(StS, self.lr)
            self.W = nmf.components_

        else:
            ## Initialize W
            self.W = np.random.randn(self.d, self.dim_hid)/np.sqrt(self.d)
            if self.nonneg:
                self.W[self.W < 0] = 0

            ## Initialize S 
            Mx = self.data@self.W
            S = sprs.csr_array(1*(Mx >= 0.5))
            StS = S.T@S/len(S) - np.outer(S.mean(0), S.mean(0))
            self.S = search.BiMat(S.indices, S.indptr, self.dim_hid)
            self.sampler = search.GaussianSampler(StS, self.lr)

    def EStep(self):

        WX = self.W.T@self.data.T
        WtW = self.W.T@self.W
        self.S = self.sampler.sample(self.S, WX, WtW, self.temp, self.tree_reg)

        return self.S

    def MStep(self, ES):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        N = ES.dot(self.W.T) + self.b
        WTW = self.W.T@self.W

        dXhat = (self.data - N)
        # dReg = self.gamma*self.W@np.sign(self.W.T@self.W)
        eta = np.trace(WTW)/np.sum(WTW**2)
        dReg = self.weight_reg*(self.W - eta*self.W@WTW)

        dW = ES.rdot(dXhat.T)/len(self.data)
        self.W += self.W_lr*(dW + dReg)

        if self.fit_intercept:
            db = dXhat.sum(0)/len(self.data)
            self.b += self.b_lr*db

        if self.nonneg:
            self.W[self.W<0] = 0
            self.b[self.b<0] = 0

        err = np.mean(dXhat**2)

        return err

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        N = self.S.dot(self.W.T) + self.b
        return np.mean((X[mask] - N[mask])**2)

@dataclass
class SparseKernelBMF(SpBMF):
    
    dim_hid: int
    tree_reg: float = 1e-2

    def initialize(self, X, alpha=2, beta=5, batch_size=1, lr=1e-1, scale_lr=1):

        self.n = len(X)      
        self.bsz = batch_size
        self.lr = lr
        self.scl_lr = scale_lr

        self.X = X

        self.data = X
        self.data *= np.sqrt(np.prod(X.shape)/np.sum(X**2))
        self.d = self.data.shape[1]

        ## Initialize S
        coding_level = np.random.beta(alpha, beta, self.dim_hid)/2
        num_active = np.floor(coding_level*len(X)).astype(int)

        Mx = self.X@np.random.randn(len(X.T),self.dim_hid)
        thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.dim_hid)]
        S = sprs.csr_array(1*(Mx >= thr))

        self.S = search.BiMat(S.indices, S.indptr, self.dim_hid)

        StS = S.T@S/len(S) - np.outer(S.mean(0), S.mean(0))
        StX = S.T@X/len(S) - np.outer(S.mean(0), X.mean(0))
        self.sampler = search.KernelSampler(StS, StX, self.bsz, self.lr)

        self.scl = 1

    def __call__(self):

        return self.scl*self.S.ker()

    def loss(self):
        """
        Compute the energy of the network, for a subset I
        """
        
        Kx = util.center(self.data@self.data.T)
        Ks = util.center(self.S.ker())
        dot = self.scl*np.sum(Kx*Ks)
        Qnrm = (self.scl**2)*np.sum(Ks**2)
        Knrm = np.sum(Kx**2)
        
        return Qnrm + Knrm - 2*dot
    
    def EStep(self):

        self.S = self.sampler.sample(
                                    self.S, 
                                    self.data, 
                                    self.scl, 
                                    self.temp, 
                                    self.tree_reg)
        K = self.S.ker()

        return util.center(K)

    def MStep(self, kerS):
        """
        Optimally scale S
        """
        
        dot = np.sum((self.data@self.data.T)*kerS)
        nrm = np.sum(kerS**2)

        self.scl += self.scl_lr*(dot/nrm - self.scl)
        
        return 1 + ((self.scl**2)*nrm - 2*self.scl*dot)/(self.n**2)

