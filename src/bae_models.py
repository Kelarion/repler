import os, sys, re
import pickle
from time import time
import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.distributions as dis
import torch.linalg as tla
import torch._dynamo
import numpy as np
from itertools import permutations, combinations
from tqdm import tqdm

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

torch._dynamo.config.suppress_errors = True 

from numba import njit
import math

# my code
import util
import pt_util
import bae_search
import students

####################################################################
############## Matrix factorization classes ########################
####################################################################

@dataclass
class BMF(nn.Module):

    def __post_init__(self):
        ## Certain methods expect a temp attribute to exist
        self.temp = 1e-5 
    
    def fit(self, *data, initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=True, **opt_args):

        if max_iter is None:
            max_iter = period*int(np.log(min_temp/initial_temp)/np.log(decay_rate))

        if verbose:
            pbar = tqdm(range(max_iter))

        en = []
        # mets = []
        self.initialize(*data, **opt_args)
        for it in range(max_iter):
            T = initial_temp*(decay_rate**(it//period))
            self.temp = T
            en.append(self.grad_step(*data))

            if verbose:
                pbar.update(1)

        return en

    def metrics(self, X):
        pass

    def EStep(self, X):
        """
        Update of the discrete parameters
        """
        return NotImplementedErro

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
class BiPCA(BMF):
    """
    Binary PCA 

    Simplest kind of BMF, with Gaussian observations and orthogonal weights

    Setting
        `weight_alg = 'exact', tree_reg=0`
    results in fast, closed-form updates, but will have poor performance
    on non-identifiable instances. 
    """

    dim_hid: int
    center: bool = False
    sparse_reg: float = 1e-2
    tree_reg: float = 0

    def initialize(self, X, alpha=2, beta=5, rank=None, pvar=1, W_init='pca'):

        self.n, self.d = X.shape

        if self.center:
            X_ = X 
        else:
            X_ = X - X.mean(0)

        # self.frac_var = np.cumsum(s**2)/np.sum(s**2)
        # if rank is None:
        #     r = np.min([len(s), np.sum(self.frac_var <= pvar)+1])
        # else:
        #     r = rank

        coding_level = np.random.beta(alpha_pr, beta_pr, self.dim_hid)/2
        self.prior_logits = -np.log(coding_level/(1-coding_level))

        ## Standardize data to O(1) fluctuations
        self.data = X_ # /np.sqrt(np.mean(X_**2))

        ## Initialise b
        if self.center:
            self.b = X.mean(0)
        else:
            self.b = np.zeros(self.d)

        ## Initialize W
        if W_init == 'pca':
            Ux,sx,Vx = la.svd(X_)
            self.W = Vx[:self.dim_hid].T
        else:
            self.W = sts.ortho_group.rvs(self.d)[:,:self.dim_hid]

        # self.W = Vx[:self.r].T 
        # self.scl = 
        self.scl = np.sqrt(np.mean(X_**2))

        ## Initialize S
        self.S = 1*(self.data@self.W - self.b@self.W > 0.5)

        # Mx = self.data@self.W
        # thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.r)]
        # self.S = 1*(Mx >= thr)

    def EStep(self):
        """
        Compute expectation of log-likelihood over S 

        X is a FloatTensor of shape (num_inp, dim_inp)
        """

        XW = (self.data@self.W - self.b@self.W)

        if self.beta > 1e-6:
            StS = 1.0*self.S.T@self.S
            newS = bae_search.bpca(XW, 1.0*self.S, self.scl, 
                StS=StS, N=self.n, 
                alpha=self.sparse_reg, beta=self.tree_reg, temp=self.temp,
                prior_logits=self.prior_logits)
        else:
            newS = bae_search.bpca(XW, 1.0*self.S, self.scl, 
                alpha=self.sparse_reg, beta=self.tree_reg, temp=self.temp,
                prior_logits=self.prior_logits)

        self.S = newS

        if self.center:
            return newS - newS.mean(0)
        else:
            return newS

    def MStep(self, ES):

        XS = self.data.T@ES-np.outer(self.b,ES.sum(0))
        U,s,V = la.svd(XS + 1e-6*np.eye(self.d,self.dim_hid), full_matrices=False)

        self.W = U@V
        self.scl = np.sum(s)/np.sum(ES**2)
        if not self.center:
            self.b = self.data.mean(0) - self.scl*self.W@ES.mean(0)

        return self.scl*np.sqrt(np.sum(ES**2))/np.sqrt(np.sum((self.data-self.b)**2))

    def __call__(self):
        return self.scl*self.S@self.W.T + self.b

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        return np.mean((self()[mask] - X[mask])**2)/np.mean(X[mask]**2)
        # return self.scl*np.sqrt(np.sum(self.S[mask]))/np.sqrt(np.sum((X-self.b)**2))

@dataclass
class SemiBMF(BMF):
    """
    Generalized BAE, taking any exponential family observation (in theory)
    """

    dim_hid: int
    nonneg: bool = False
    fit_intercept: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_reg: float = 1e-2

    def __call__(self):
        return self.S@self.W.T + self.b

    def initialize(self, X, S0=None, hot=False, W_lr=0.1, b_lr=0.1):

        self.n, self.d = X.shape
        self.data = X/np.sqrt(np.mean(X**2))

        self.W_lr = W_lr
        self.b_lr = b_lr

        self.b = np.zeros(self.d) # Initialize b

        if self.nonneg and hot: # Initialize with NMF
            nmf = NMF(self.dim_hid)
            Z = nmf.fit_transform(self.data)
            # print('Fit NMF')
            S = []
            kmn = KMeans(2, n_init=1)
            for i in range(Z.shape[1]):
                S.append(kmn.fit_predict(Z[:,[i]]))

            self.S = np.array(S).T
            self.W = nmf.components_

        else:
            ## Initialize W
            self.W = np.random.randn(self.d, self.dim_hid)/np.sqrt(self.d)
            if self.nonneg:
                self.W[self.W < 0] = 0

            ## Initialize S 
            Mx = self.data@self.W
            self.S = 1*(Mx >= 0.5)

    def EStep(self):

        # newS = binary_glm(self.data*1.0, oldS, self.W, self.b, steps=self.S_steps,
        #     beta=self.beta, temp=self.temp, lognorm=self.lognorm)

        XW = (self.data@self.W - self.b@self.W)
        WtW = self.W.T@self.W

        if self.tree_reg > 1e-6:
            StS = 1.0*self.S.T@self.S
            newS = bae_search.sbmf(XW, 1.0*self.S, WtW, 
                StS=StS, N=self.n, 
                beta=self.tree_reg, alpha=self.sparse_reg,
                temp=self.temp)
        else:
            newS = bae_search.sbmf(XW, 1.0*self.S, WtW, 
                beta=self.tree_reg, alpha=self.sparse_reg,
                temp=self.temp)

        self.S = newS

        return newS

    def MStep(self, ES):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        if self.weight_reg > 1e-3:
            N = ES@self.W.T + self.b

            WTW = self.W.T@self.W

            dXhat = (self.data - N)
            # dReg = self.gamma*self.W@np.sign(self.W.T@self.W)
            eta = np.trace(WTW)/np.sum(WTW**2)
            dReg = self.weight_reg*(self.W - eta*self.W@WTW)

            dW = dXhat.T@ES/len(self.data)
            self.W += self.W_lr*(dW + dReg)

            if self.fit_intercept:
                db = dXhat.sum(0)/len(self.data)
                self.b += self.b_lr*db

            if self.nonneg:
                self.W[self.W<0] = 0
                self.b[self.b<0] = 0

            err = np.mean(dXhat**2)

        elif self.nonneg:
            err = 0
            for i in range(self.d):
                what, rnorm = nnls(ES, self.data[:,i]-self.b[i])
                self.W[i] = what
                err += rnorm/self.d 

        return err

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        N = self.S@self.W.T + self.b
        return np.mean((X[mask] - N[mask])**2)


# class SemiBMF(BMF):
#     """
#     Generalized BAE, taking any exponential family observation (in theory)
#     """

#     def __init__(self, dim_hid, noise='gaussian', tree_reg=1e-2, weight_reg=1e-2, 
#         do_pca=False, nonneg=False, fit_intercept=False):

#         super().__init__()

#         self.has_data = False
#         self.reduce = do_pca

#         self.r = dim_hid

#         self.nonneg = nonneg

#         self.alpha = weight_reg
#         self.beta = tree_reg

#         ## rn only support three kinds of observation noise, because
#         ## other distributions have constraints on the natural params
#         ## (mostly non-negativity) which I don't want to deal with 
#         if noise == 'gaussian':
#             self.lognorm = bae_util.gaussian
#             self.mean = lambda x:x
#             self.likelihood = sts.norm
#             self.base = lambda x: (-x**2 - np.log(2*np.pi))/2

#         elif noise == 'poisson':
#             self.lognorm = bae_util.poisson
#             self.mean = np.exp
#             self.likelihood = sts.poisson
#             self.base = lambda x: -np.log(spc.factorial(x))

#         elif noise == 'bernoulli':
#             self.lognorm = bae_util.bernoulli
#             self.mean = spc.expit
#             self.likelihood = sts.bernoulli
#             self.base = lambda x: 0

#         # ## Initialization is better when it's data-dependent
#         # self.initialize(X_init, alpha, beta)

#     def __call__(self):
#         N = self.S@self.W.T + self.b
#         return self.likelihood(self.mean(N)).rvs()

#     def initialize(self, X, alpha=2, beta=5, rank=None, pvar=1, W_lr=0.1, b_lr=0.1):

#         self.n, self.d = X.shape
#         if self.reduce:

#             Ux,sx,Vx = la.svd(X-X.mean(0), full_matrices=False)
#             self.frac_var = np.cumsum(sx**2)/np.sum(sx**2)
#             if rank is None:
#                 r = np.min([len(sx), np.sum(self.frac_var <= pvar)+1])
#                 # r = np.max([dim_hid, np.sum(self.frac_var <= pvar)+1])
#             else:
#                 r = np.min([rank, np.sum(self.frac_var <= pvar)+1])

#             self.d = r

#             self.data = Ux[:,:r]@np.diag(sx[:r])
#             self.V = Vx[:r]
#         else:
#             self.data = X
#         self.has_data = True

#         self.W_lr = W_lr
#         self.b_lr = b_lr

#         ## Initialise b
#         self.b = np.zeros(self.d) # -X.mean(0)

#         ## Initialize W
#         self.W = np.random.randn(self.d, self.r)/np.sqrt(self.d)

#         ## Initialize S 
#         coding_level = np.random.beta(alpha, beta, self.r)/2
#         num_active = np.floor(coding_level*self.n).astype(int)

#         Mx = self.data@self.W
#         # thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.r)]
#         # self.S = 1*(Mx >= thr)
#         self.S = 1*(Mx >= 0.5)

#     def EStep(self):

#         # newS = binary_glm(self.data*1.0, oldS, self.W, self.b, steps=self.S_steps,
#         #     beta=self.beta, temp=self.temp, lognorm=self.lognorm)

#         XW = (self.data@self.W - self.b@self.W)
#         WtW = self.W.T@self.W

#         if self.beta > 1e-6:
#             StS = 1.0*self.S.T@self.S
#             newS = bae_search.sbmf(XW, 1.0*self.S, self.W.T@self.W, 
#                 StS=StS, N=self.n, 
#                 alpha=self.alpha, beta=self.beta, temp=self.temp)
#         else:
#             newS = bae_search.sbmf(XW, 1.0*self.S, self.scl, 
#                 alpha=self.alpha, beta=self.beta, temp=self.temp)

#         self.S = newS

#         return newS

#     def MStep(self, ES):
#         """
#         Maximise log-likelihood conditional on S, with p.r. regularization
#         """

#         for i in range(self.W_steps):

#             N = ES@self.W.T + self.b

#             WTW = self.W.T@self.W

#             dXhat = (self.data - self.mean(N))
#             # dReg = self.alpha*self.W@np.sign(self.W.T@self.W)
#             dReg = self.alpha*self.W@(np.eye(self.r) - WTW*np.trace(WTW)/np.sum(WTW**2))

#             dW = dXhat.T@ES/len(self.data)
#             db = dXhat.sum(0)/len(self.data)

#             self.W += self.W_lr*(dW + dReg)
#             self.b += self.b_lr*db

#         return np.mean(self.data*N - self.lognorm(N))

#     def loss(self, X, mask=None):
#         if mask is None:
#             mask = np.ones(X.shape) > 0
#         N = self.S@self.W.T + self.b
#         return -np.mean(X[mask]*N[mask] - self.lognorm(N[mask]) + self.base(X[mask]))

@dataclass
class KernelBMF(BMF):
    
    dim_hid: int
    sparse_reg: float = 0
    tree_reg: float = 1e-2

    def initialize(self, X, S0=None, alpha=2, beta=5, rank=None, pvar=1, scale_lr=1):

        self.scl_lr = scale_lr

        self.n = len(X)
        U,s,V = la.svd(X-X.mean(0), full_matrices=False)        
            
        self.frac_var = np.cumsum(s**2)/np.sum(s**2)
        if rank is None:
            r = np.min([len(s), np.sum(self.frac_var <= pvar)+1])
        else:
            r = rank

        # self.X = X
        self.X = U[:,:r]@np.diag(s[:r])@V[:r]
        self.U = U[:,s>1e-6]
        self.U = self.U[:,:r]
        
        ## Center the kernel of each item excluding that item
        # K = self.X@self.X.T
        # notI = (1 - np.eye(self.n))/(self.n-1) 
        # self.data = (K - K@notI - (K*notI).sum(0) + ((K@notI)*notI).sum(0)).T 

        self.data = U[:,:r]@np.diag(s[:r])
        self.data *= np.sqrt(r*self.n/np.sum(s[:r]**2))
        self.d = self.data.shape[1]

        ## Initialize S
        if S0 is None:
            coding_level = np.random.beta(alpha, beta, self.dim_hid)/2
            num_active = np.floor(coding_level*len(X)).astype(int)

            Mx = self.data@np.random.randn(self.d, self.dim_hid)
            thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.dim_hid)]

            self.S = np.array(1*(Mx >= thr))
        else:
            self.S = 1*S0

        self.scl = 1

    def __call__(self):

        return self.scl*util.center(self.S@self.S.T)

    def loss(self):
        """
        Compute the energy of the network, for a subset I
        """
        
        X = self.data
        S = self.S
        K = X@X.T
        dot = self.scl*np.sum(util.center(K)*(S@S.T))
        Qnrm = (self.scl**2)*np.sum(util.center((S@S.T))**2)
        Knrm = np.sum(util.center(K)**2)
        
        return Qnrm + Knrm - 2*dot
    
    def EStep(self):

        # newS = update_concepts_kernel(self.data, 1.0*self.S, 
        #     scl=self.scl, beta=self.beta, temp=self.temp, steps=self.steps)
        newS = bae_search.kerbmf(self.data, 1.0*self.S, scl=self.scl, 
            StS=self.S.T@self.S, StX = self.S.T@self.data, N=self.n,
            alpha=self.sparse_reg, beta=self.tree_reg, temp=self.temp)
        # newS = bae_search.kerbae(self.data, 1.0*self.S, scl=self.scl, 
        #     StS=self.S.T@self.S/self.n, StX = self.S.T@self.data/self.n,
        #     beta=self.tree_reg, temp=self.temp)

        # return newS
        self.S = newS
        return self.S

    def MStep(self, ES):
        """
        Optimally scale S
        """
        
        S_ = ES - ES.mean(0)
        
        dot = np.sum((self.data@self.data.T)*(S_@S_.T))
        nrm = np.sum((S_@S_.T)**2)

        self.scl += self.scl_lr*(dot/nrm - self.scl)
        
        return 1 + ((self.scl**2)*nrm - 2*self.scl*dot)/self.n


####################################################################
################ Binary autoencoder classes ########################
####################################################################

@dataclass(eq=False)
class BAE(students.NeuralNet):
    """
    A hybdrid of a Bernoulli VAE and SemiBMF 

    Minimizes the same loss as the BMF models, but S is parameterised by a 
    linear-threshold layer, S = f[MX + p], rather than stored non-parametrically.

    Also can include the entropy regularization used by the Bernoulli VAE.
    """
    
    dim_hid: int
    dim_inp: int
    beta: float = 1.0
    temp: float = 2/3
    tree_reg: float = 0
    sparse_reg: float = 1e-2
    weight_reg: float = 1e-2

    def __post_init__(self):
        super().__init__()

        self.q = nn.Linear(self.dim_inp, self.dim_hid) # x -> s
        self.p = nn.Linear(self.dim_hid, self.dim_inp) # s -> x'

        self.Cov = torch.zeros((self.dim_hid, self.dim_hid))

    def fit(self, *data, initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=True, **opt_args):

        if max_iter is None:
            max_iter = period*int(np.log(min_temp/initial_temp)/np.log(decay_rate))

        if verbose:
            pbar = tqdm(range(max_iter))

        en = []
        # mets = []
        self.initialize(*data, **opt_args)
        for it in range(max_iter):
            T = initial_temp*(decay_rate**(it//period))
            self.temp = T
            en.append(self.grad_step(*data))

            if verbose:
                pbar.update(1)

        return en

    #     self.init_weights()

    # def init_weights(self, scale=None):

    #     if scale is None:
    #         scl = 1

    #     if self.dim_hid > self.dim_inp:
    #         W = sts.ortho_group(self.dim_hid).rvs()[:,:self.dim_inp]
    #     else:
    #         W = sts.ortho_group(self.dim_inp).rvs()[:,:self.dim_hid].T

    #     with torch.no_grad():
    #         self.q.weight.copy_(torch.tensor(scl*W))
    #         self.p.weight.copy_(torch.tensor(scl*W.T))

    def initialize(self, dl, **opt_args):
        """
        Input should be a dataloader for the data, same as the input to grad_step

        Eventually, 'N' will be a hyperparmeter whose default is some large number,
        but that's not something I'm implementing yet
        """
        self.N = len(dl.dataset)
        self.init_optimizer(**opt_args)

        self.tree_lr = dl.batch_size/self.N

    def forward(self, X):
        # return self.p((torch.sign(self.q(X))+1)/2)
        return self.p(torch.sigmoid(self.q(X)/self.temp))
    
    def hidden(self, X):
        return (1+torch.sign(self.q(X)))/2

    def metrics(self, dl):
        pass

    def EStep(self, S, X):
        """
        Discrete search over S, initialised by q(s|x)

        Expects two pytorch tensors, returns a pytorch tensor
        """
        return NotImplementedError

    def MStep(self, S, X):
        """
        Continuous parameter loss function

        Expects two pytorch tensors, returns a scalar loss
        """
        return NotImplementedError

    def loss(self, batch):

        C0 = self.q(batch[0])

        ## Search over S
        S = self.EStep(C0, batch[0])

        ## Update continuous parameters
        qls = nn.BCEWithLogitsLoss()(C0, S)
        pls = self.MStep(S, batch[0])

        return pls + self.beta*qls 

@dataclass(eq=False)
class BinaryAutoencoder(BAE):
    """
    Most general BAE, without constraints on the readout weights
    """

    def EStep(self, S, X):

        with torch.no_grad():

            # if self.Cov.device.type != S.device.type:
            #     self.Cov = self.Cov.to(S.device)
            Sbin = 1.0*(S > 0)

            if S.device.type == 'cpu':
            # Convert to numpy since that's what Numba accepts
                Xnp = X.detach().numpy().astype(float)
                W = self.p.weight.data.detach().numpy().astype(float)
                b = self.p.bias.data.detach().numpy().astype(float)
                Snp = Sbin.data.detach().numpy().astype(float) 
                StS = self.Cov.detach().numpy().astype(float)
            else:
                Xnp = X.cpu().numpy().astype(float)
                W = self.p.weight.data.cpu().numpy().astype(float)
                b = self.p.bias.data.cpu().numpy().astype(float)
                Snp = Sbin.data.cpu().numpy().astype(float) 
                StS = self.Cov.cpu().numpy().astype(float)

            newS = bae_search.bae(
                XW=Xnp@W - b@W, S=Snp,                      # inputs
                StS=StS, WtW=W.T@W,                         # quadratic terms
                alpha=self.sparse_reg, beta=self.tree_reg,  # regualarization
                temp=self.temp,                             # temperature
                )                            
            newCov = (1-self.tree_lr)*StS + self.tree_lr*newS.T@newS/len(newS)

            newS = torch.tensor(newS, dtype=S.dtype, device=S.device)
            self.Cov = torch.tensor(newCov, 
                dtype=self.Cov.dtype, 
                device=self.Cov.device)

        return newS

    def MStep(self, S, X):

        loss = nn.MSELoss()(self.p(S), X)

        if self.weight_reg > 0:
            W = self.p.weight
            WtW = W.T@W
            loss -= self.weight_reg*(torch.sum(W**2)**2)/torch.sum(WtW**2)
            
        return loss 

@dataclass(eq=False)
class BernVAE(BAE):
    """
    Bernoulli VAE, i.e. a BAE without any search step
    """

    def EStep(self, S, X):

        with torch.no_grad():
            Sbin = 1*(S > 0)
            self.Cov += self.tree_lr*(Sbin.T@Sbin/len(Sbin) - self.Cov)

        return torch.sigmoid(S)

    def MStep(self, S, X):

        loss = nn.MSELoss()(self.p(S), X) 
        loss += self.sparse_reg*torch.mean(S)

        if self.weight_reg > 0:
            W = self.p.weight
            WtW = W.T@W
            loss -= self.weight_reg*(torch.sum(W**2)**2)/torch.sum(WtW**2)
            
        return loss 

############################################################
######### Jitted update of S ###############################
############################################################

# @njit
# def binary_glm(X, S, W, b, temp, alpha=0, beta=0, steps=1, STS=None, N=None, lognorm=bae_util.gaussian):
#     """
#     One gradient step on S

#     lognorm should always be gaussian, the others are all worse for some reason

#     TODO: figure out a good sparse implementation?
#     """

#     n, m = S.shape

#     if (STS is None) or (N is None):
#         StS = np.dot(S.T, S)
#         N = n 
#     else:
#         StS = 1*STS
#         N = 1*N
#     St1 = np.diag(StS)

#     ## Initial values
#     E = np.dot(S,W.T) + b  # Natural parameters
#     C  = np.dot(X,W)        # Constant term

#     for step in range(steps):
#         # for i in np.random.permutation(np.arange(n)):
#         # en = np.zeros((n,m))
#         for i in np.arange(n): 

#             if beta > 1e-6:
#                 ## Organize states
#                 s = S[i]
#                 St1 -= s
#                 StS -= np.outer(s,s)

#                 ## Regularization (more verbose because of numba reasons)
#                 D1 = StS
#                 D2 = St1[None,:] - StS
#                 D3 = St1[:,None] - StS
#                 D4 = (N-1) - St1[None,:] - St1[:,None] + StS

#                 best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
#                 best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
#                 best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
#                 best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

#                 R = (best1 - best2 - best3 + best4)*1.0
#                 r = (best2.sum(0) - best4.sum(0))*1.0

#             ## Hopfield update of s
#             # for j in np.random.permutation(np.arange(m)): # concept
#             for j in np.arange(m): 

#                 ## Compute linear terms
#                 dot = np.sum(lognorm(E[i] + (1-S[i,j])*W[:,j])) 
#                 dot -= np.sum(lognorm(E[i] - S[i,j]*W[:,j]))
#                 if beta > 1e-6:
#                     inhib = np.dot(R[j], S[i]) + r[j]
#                 else:
#                     inhib = 0

#                 ## Compute currents
#                 curr = (C[i,j] - beta*inhib - dot - alpha)/temp

#                 ## Apply sigmoid (overflow robust)
#                 if curr < -100:
#                     prob = 0.0
#                 elif curr > 100:
#                     prob = 1.0
#                 else:
#                     prob = 1.0 / (1.0 + math.exp(-curr))

#                 ## Update outputs
#                 sj = 1*(np.random.rand() < prob)
#                 ds = sj - S[i,j]
#                 S[i,j] = sj
                
#                 ## Update dot products
#                 E[i] += ds*W[:,j]

#                 # en[i,j] = np.sum(lognorm(E)) - 2*np.sum(C*S)

#             ## Update 
#             # S[i] = news
#             St1 += S[i]
#             StS += np.outer(S[i], S[i]) 

#     return S #, en

# @njit
# def update_concepts_asym_cntr(XW, S, scl, alpha, beta, temp, STS=None, N=None, steps=1):
#     """
#     One gradient step on S

#     TODO: figure out a good sparse implementation!
#     """

#     n, m = S.shape

#     if (STS is None) or (N is None):
#         StS = np.dot(S.T, S)
#         N = n
#     else:
#         StS = 1*STS
#         N = 1*N
#     St1 = np.diag(StS)

#     for step in range(steps):
#         for i in np.random.permutation(np.arange(n)):

#             ## Organize states
#             St1 -= S[i]

#             if beta >= 1e-6:
#                 StS -= np.outer(S[i], S[i])

#                 ## Regularization (more verbose because of numba reasons)
#                 D1 = StS
#                 D2 = St1[None,:] - StS
#                 D3 = St1[:,None] - StS
#                 D4 = (N-1) - St1[None,:] - St1[:,None] + StS

#                 best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
#                 best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
#                 best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
#                 best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

#                 R = (best1 - best2 - best3 + best4)*1.0
#                 r = (best2.sum(0) - best4.sum(0))*1.0

#             ## Hopfield update of s
#             for j in np.random.permutation(np.arange(m)): # concept

#                 inp = (2*XW[i,j] - scl*(N - 1 - 2*St1[j])/N - alpha)

#                 ## Compute linear terms
#                 if beta >= 1e-6:
#                     inhib = np.dot(R[j], S[i]) + r[j]
#                 else:
#                     inhib = 0

#                 ## Compute currents
#                 curr = (inp - beta*scl*inhib)/temp

#                 # ## Apply sigmoid (overflow robust)
#                 if curr < -100:
#                     prob = 0.0
#                 elif curr > 100:
#                     prob = 1.0
#                 else:
#                     prob = 1.0 / (1.0 + math.exp(-curr))

#                 ## Update outputs
#                 S[i,j] = 1*(np.random.rand() < prob)

#             ## Update 
#             # S[i] = news
#             St1 += S[i]
#             StS += np.outer(S[i], S[i]) 

#     return S #, en
