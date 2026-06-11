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
from torch.nn.utils import parametrize
from torch.nn import functional as F

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
import df_util
import pt_util
import bae_search
import students

####################################################################
############## Matrix factorization classes ########################
####################################################################

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu(1) + X.triu(1).T  # Return a symmetric matrix

    def right_inverse(self, A):
        return A.triu(1)

class ZeroDiag(nn.Module):
    def forward(self, X):
        return X.triu(1) + X.tril(-1)  # Return a symmetric matrix

    def right_inverse(self, A):
        return A.triu(1) + A.tril(-1)

@dataclass
class BMF:

    def __post_init__(self):
        ## Certain methods expect a temp attribute to exist
        self.temp = 1e-5 
        self.initialized = False

    def initialize(self, X, **args):

        self.init_params(X, **args)
        self.init_latents(X, **args)
        self.initialized = True
    
    def fit(self, *data, initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=True, **opt_args):

        if max_iter is None:
            max_iter = period*int(np.log(1e-4/initial_temp)/np.log(decay_rate))

        if verbose:
            pbar = tqdm(range(max_iter))

        en = []
        # mets = []
        if not self.initialized:
            self.initialize(*data, **opt_args)
        for it in range(max_iter):
            # print(self.S.sum())
            # print(self.scl)
            T = min_temp + initial_temp*(decay_rate**(it//period))
            self.temp = T
            en.append(self.grad_step(*data))

            if verbose:
                pbar.update(1)

        return en

    def sample(self, X, temp=None, n_samp=1, burnin=10, **args):
        """
        Generate posterior samples of S, given X and current parameters
        """
        if temp is not None:
            self.temp = temp

        samps = np.zeros((n_samp,len(X),self.dim_hid))

        S = 1.0*np.random.choice([0,1], size=(len(X), self.dim_hid))
        i = 0
        for n in range(n_samp*burnin):
            samp = self.EStep(S, X, inplace=False, **args)
            if not np.mod(n, burnin):
                samps[i] = 1*samp
                i += 1

        return samps

    def impute(self, X, mask):
        """
        Draw from posterior predictive of unmasked X values
        """

        Zhat = self.sample(X)

    def loglikelihood(self, X, Xhat):
        """
        The log-likelihood given the current parameters

        default is MSE
        """
        return np.mean((Xhat - X)**2)

    def EStep(self, S, X):
        """
        Update of the discrete parameters
        """
        return NotImplementedError

    def MStep(self, S, X):
        """
        Update of the continuous parameters
        """
        return NotImplementedError

    def grad_step(self, X):
        """
        One iteration of optimization
        """

        newS = self.EStep(self.S, X)
        loss = self.MStep(newS,X)

        return loss

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
    sparse_reg: float = 1e-2
    tree_reg: float = 0
    alpha_pr: float = 2
    beta_pr: float = 2
    fit_intercept: bool = True
    W_init: str = 'pca'

    def init_params(self, X):

        self.d = X.shape[1]
        if self.d < self.dim_hid: # then the rows are orthogonal
            self.tranpose = True

        if self.W_init == 'pca':
            Ux,sx,Vx = la.svd(X, full_matrices=False)
            self.W = Vx[:self.dim_hid].T
        else:
            s1 = np.max([self.d, self.dim_hid])
            s2 = np.min([self.d, self.dim_hid])
            self.W = sts.ortho_group.rvs(s1)[:,:s2]
            if self.transpose:
                self.W = self.W.T

        self.b = X.mean(0)
        self.scl = np.sqrt(np.mean((X-self.b)**2))

    def init_latents(self, X):

        coding_level = np.random.beta(self.alpha_pr, self.beta_pr, self.dim_hid)/2
        # self.prior_logits = -np.log(coding_level/(1-coding_level))
        self.prior_logits = np.ones(self.dim_hid)

        self.S = 1.0*(X@self.W - self.b@self.W > 0.5**self.scl)
        self.StS = self.S.T@self.S

    def EStep(self, S, X, inplace=True):
        """
        Compute expectation of log-likelihood over S 

        X is a FloatTensor of shape (num_inp, dim_inp)
        """

        XW = (X@self.W - self.b@self.W)

        newS = bae_search.bpca(XW, S, self.scl, 
            StS=self.StS, N=len(S), 
            alpha=self.sparse_reg, beta=self.tree_reg, temp=self.temp,
            prior_logits=self.prior_logits)

        return newS

    def MStep(self, ES, X):

        XS = X.T@ES - np.outer(self.b, ES.sum(0))
        U,s,V = la.svd(XS + 1e-6*np.eye(X.shape[1], self.dim_hid), full_matrices=False)

        self.W = U@V
        self.scl = np.sum(s)/np.sum(ES**2)
        if self.fit_intercept:
            self.b = X.mean(0) - self.scl*self.W@ES.mean(0)

        return np.mean((X - self.scl*ES@self.W.T - self.b)**2)

    def __call__(self, S):
        return self.scl*S@self.W.T + self.b

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
    fit_intercept: bool = True
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l2_reg: float = 1e-2
    weight_l1_reg: float = 0.0

    def __call__(self, S):
        return S@self.W.T + self.b

    def init_params(self, X,  W_lr=0.1, b_lr=0.1, scl_lr=0, hot_start=False):

        self.n, self.d = X.shape

        self.W_lr = W_lr
        self.b_lr = b_lr

        self.sigma_x = 1
        self.scl_lr = scl_lr

        self.b = np.zeros(self.d) # Initialize b
        if hot_start:
            nmf = NMF(n_components=self.dim_hid, alpha_W=self.sparse_reg, l1_ratio=1)
            nmf.fit(X)
            self.W = nmf.components_.T

        else:
            self.W = np.random.randn(self.d, self.dim_hid)/np.sqrt(self.d)
            if self.nonneg:
                self.W[self.W < 0] = 0

    def init_latents(self, X, **args):

            ## Initialize S 
            Mx = X@self.W
            self.S = 1.0*(Mx >= 0.5)

            self.StS = self.S.T@self.S

    def EStep(self, S, X, inplace=True):

        # newS = binary_glm(self.data*1.0, oldS, self.W, self.b, steps=self.S_steps,
        #     beta=self.beta, temp=self.temp, lognorm=self.lognorm)

        XW = (X@self.W - self.b@self.W)
        WtW = self.W.T@self.W

        newS = bae_search.sbmf(
            XW/self.sigma_x, 
            S, 
            WtW/self.sigma_x, 
            StS=self.StS, N=self.n, 
            beta=self.tree_reg, alpha=self.sparse_reg,
            temp=self.temp,
            inplace=inplace)

        return newS

    def MStep(self, ES, X):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        if self.weight_pr_reg > 1e-4:
            N = ES@self.W.T + self.b

            WTW = self.W.T@self.W

            dXhat = (X - N)
            # dReg = self.gamma*self.W@np.sign(self.W.T@self.W)
            eta = np.trace(WTW)/np.sum(WTW**2)
            dReg = self.weight_pr_reg*(self.W - eta*self.W@WTW)
            dReg -= self.weight_l2_reg*self.W
            dReg -= self.weight_l1_reg*np.sign(self.W)

            dW = dXhat.T@ES/len(X)
            self.W += self.W_lr*(dW + dReg)

            if self.fit_intercept:
                db = dXhat.sum(0)/len(X)    
                self.b += self.b_lr*db

            if self.nonneg:
                self.W[self.W<0] = 0
                self.b[self.b<0] = 0

            err = np.mean(dXhat**2)
            self.sigma_x += self.scl_lr*(err - self.sigma_x)

        elif self.nonneg:
            err = 0
            for i in range(self.d):
                what, rnorm = nnls(ES, X[:,i]-self.b[i])
                self.W[i] = what
                err += rnorm/self.d 

        else:            
            if self.fit_intercept:
                S_ = ES-ES.mean(0)
                X_ = X - X.mean(0)
            else:
                S_ = ES
                X_ = X

            U,s,V = la.svd(S_, full_matrices=False)
            shat = s/(s**2 + self.weight_l2_reg**2)
            self.W = X_.T@U@np.diag(shat)@V

            if self.fit_intercept:
                self.b = X.mean(0) - ES.mean(0)@self.W.T

            err = np.mean((X - self())**2)

        return [err, 1*ES, err/self.sigma_x + np.log(self.sigma_x)]

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        N = self.S@self.W.T + self.b
        return np.mean((X[mask] - N[mask])**2)


@dataclass
class SpikeNMF(BMF):
    """
    NMF with spike-and-slab prior
    """

    dim_hid: int
    nonneg: bool = True
    fit_intercept: bool = True
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    slab_prior: float = 1 
    weight_pr_reg: float = 1e-1
    weight_l1_reg: float = 0
    weight_l2_reg: float = 1e-2
    m_iters: int = 1

    def __call__(self, S):
        with torch.no_grad():
            Xhat = self.W(torch.FloatTensor(S)).numpy()
        return Xhat

    def init_params(self, X, hot_start=False, scl_lr=0, **opt_args):

        self.n, self.d = X.shape

        if hot_start:
            nmf = NMF(n_components=self.dim_hid, alpha_W=self.sparse_reg, l1_ratio=1)
            nmf.fit(X)
            Winit = nmf.components_.T

            dead = Winit.sum(0) == 0
            newW = np.random.randn(self.d, dead.sum())/np.sqrt(self.d)
            newW[newW < 0] = 0
            Winit[:,dead] = newW

        else:
            Winit = np.random.randn(self.d, self.dim_hid)/np.sqrt(self.d)
            if self.nonneg:
                Winit[Winit < 0] = 0

        self.sigma_x = 1 
        self.scl_lr = scl_lr

        self.W = nn.Linear(self.dim_hid, self.d, bias=self.fit_intercept)
        self.W.weight.data.copy_(torch.FloatTensor(Winit))
        if self.fit_intercept:
            self.W.bias.data.copy_(torch.zeros(self.d))

        self.optimizer = optim.SGD(self.W.parameters(), lr=0.1, **opt_args)

    def init_latents(self, X, **args):

        with torch.no_grad():
            Z = X@self.W.weight.numpy()
        self.S = 1.0*(Z > 0)
        self.Z = Z*self.S  ## Rectify multipliers

        self.StS = self.S.T@self.S

    def EStep(self, S, X, inplace=True, slab=True):

        with torch.no_grad():
            W = self.W.weight.numpy()
            b = self.W.bias.numpy()
            XW = (X@W - b@W)
            WtW = W.T@W

        newS, newZ = bae_search.snmf(
                               XW=XW, 
                               S=S,
                               Z=self.Z,
                               WtW=WtW, 
                               StS=self.StS, N=self.n, 
                               tau=self.slab_prior,
                               beta=self.tree_reg, 
                               alpha=self.sparse_reg,
                               sigma2=self.sigma_x,
                               temp=self.temp,
                               inplace=inplace
                               )

        if slab:
            return newS*newZ
        else:
            return newS

    def MStep(self, ES, X):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        Spt = torch.FloatTensor(ES)
        Xpt = torch.FloatTensor(X)
        ls = []
        for _ in range(self.m_iters):
            
            self.optimizer.zero_grad()
            
            Xhat = self.W(Spt)
            loss = torch.sum((Xpt - Xhat)**2)/len(Xpt)
            
            ls.append(loss.item())

            ## Regularization on W
            WtW = self.W.weight.T@self.W.weight
            loss -= self.weight_pr_reg*((torch.trace(WtW)**2)/torch.sum(WtW**2))#/self.dim_hid
            loss += self.weight_l1_reg*torch.sum(torch.abs(self.W.weight))#/self.dim_hid
            loss += self.weight_l2_reg*torch.trace(WtW)#/self.dim_hid
            
            loss.backward()
            
            self.optimizer.step()

            if self.nonneg:
                with torch.no_grad():
                    self.W.weight[self.W.weight<0] = 0
                    self.W.bias[self.W.bias<0] = 0

                    ## resample dead weights
                    dead = self.W.weight.sum(0) == 0
                    newW = torch.randn(self.d, dead.sum())/np.sqrt(self.d)
                    newW[newW < 0] = 0
                    self.W.weight[:,dead] = newW

        with torch.no_grad():
            new_sig = np.mean((X - Xhat.numpy())**2)
            self.sigma_x += self.scl_lr*(new_sig - self.sigma_x)

        return new_sig

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
    # alpha_pr: float = 2
    # beta_pr: float = 2
    uniform_scale: bool = True
    kernel_input: bool = False  # should we expect a kernel as input

    def init_params(self, X, scl_lr=1):

        # K = self.X@self.X.T
        # notI = (1 - np.eye(self.n))/(self.n-1) 
        # self.data = (K - K@notI - (K*notI).sum(0) + ((K@notI)*notI).sum(0)).T 

        self.n, self.d = X.shape
        self.scl_lr = scl_lr
        if self.kernel_input:
            X_ = util.center(X)
            # self.scl = np.mean(X_**2)
            if self.uniform_scale:
                self.scl = 1
            else:
                self.scl = np.ones(self.dim_hid)
            self.data_norm = np.sum(X_**2)
        else:
            X_ = X - X.mean(0)
            if self.uniform_scale:
                self.scl = np.mean(X_**2)
            else:
                self.scl = np.mean(X_**2) * np.ones(self.dim_hid)
            self.data_norm = np.sum((X_.T@X_)**2)

    def init_latents(self, X, **kwargs):

        # coding_level = np.random.beta(self.alpha_pr, self.beta_pr, self.dim_hid)/2
        # num_active = np.floor(coding_level*len(X)).astype(int)

        if self.kernel_input:
            l,V = la.eigh(X)
            Mx = np.diag(np.sqrt(l+1e-6))@V@np.random.randn(self.d, self.dim_hid)
        else:
            Mx = X@np.random.randn(self.d, self.dim_hid)
        # thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.dim_hid)]

        # self.S = np.array(1*(Mx >= thr))
        self.S = np.array(1.0*(Mx >= 0.5)) # make sure this is a float

        self.StS = self.S.T@self.S
        self.StX = self.S.T@X

    def __call__(self, S):

        return self.scl*util.center(S@S.T)

    def loss(self, X):
        """
        Compute the energy of the network, for a subset I
        """
        
        S = self.S 
        StS = self.StS - np.outer(np.diag(self.StS), np.diag(self.StS))/len(X)
        if self.kernel_input:
            dot = self.scl*np.sum(util.center(X)*(S@S.T))
            Knrm = np.sum(util.center(X)**2)
        else:
            K = X@X.T
            dot = self.scl*np.sum((S.T@X)**2)
            X_ = X - X.mean(0)
            Knrm = np.sum(X_.T@X_**2)

        Qnrm = (self.scl**2)*np.sum(StS**2)
            
        return (Qnrm + Knrm - 2*dot)/Knrm
    
    def EStep(self, S, X, inplace=True):

        if self.kernel_input: # the input is a kernel matrix
            newS = bae_search.kerbmf2(X, S, scl=self.scl, 
                StS=self.StS, N=self.n, inplace=inplace,
                alpha=self.sparse_reg, beta=self.tree_reg, temp=self.temp)

        else: # the input is a feature matrix
            newS = bae_search.kerbmf(X, S, scl=self.scl, 
                StS=self.StS, StX=self.StX, N=self.n, inplace=inplace,
                alpha=self.sparse_reg, beta=self.tree_reg, temp=self.temp)

        return newS

    def MStep(self, ES, X):
        """
        Optimally scale S
        """
        
        S_ = ES - ES.mean(0)
        StS = self.StS - np.outer(np.diag(self.StS), np.diag(self.StS))/len(X)
        StX = self.StX - np.outer(np.diag(self.StS), X.mean(0))

        if self.kernel_input:
            dot = np.sum(X*(S_@S_.T))
        else:
            dot = np.sum(StX**2)
        nrm = np.sum(StS**2)

        self.scl += self.scl_lr*(dot/nrm - self.scl)
        
        return 1 + ((self.scl**2)*nrm - 2*self.scl*dot)/self.data_norm


@dataclass
class KernelBMF2(BMF):
    
    dim_hid: int
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    uniform_scale: bool = True
    l1_reg: float = 0.0        # only if non-uniform
    kernel_input: bool = False  # should we expect a kernel as input

    def init_params(self, X, scl_lr=1):
        self.n, self.d = X.shape
        self.scl_lr = scl_lr
        
        if self.kernel_input:
            X_ = util.center(X)
            # Always initialize as a 1D array to satisfy the Numba signature
            self.scl = np.ones(self.dim_hid)
            self.data_norm = np.sum(X_**2)
        else:
            X_ = X - X.mean(0)
            # Always initialize as a 1D array to satisfy the Numba signature
            self.scl = np.ones(self.dim_hid)
            self.data_norm = np.sum((X_.T@X_)**2)

    def init_latents(self, X, **kwargs):
        if self.kernel_input:
            l, V = la.eigh(X)
            Mx = np.diag(np.sqrt(l+1e-6))@V@np.random.randn(self.d, self.dim_hid)
        else:
            Mx = X@np.random.randn(self.d, self.dim_hid)

        self.S = np.array(1.0*(Mx >= 0.5)) 
        self.StS = self.S.T@self.S
        self.StX = self.S.T@X

    def __call__(self, S):
        # Matrix representation of S D S' centered
        return np.einsum('...ik,k,...jk->...ij', S, self.scl, S)

    def loss(self, X):
        S = self.S 
        StS = self.StS - np.outer(np.diag(self.StS), np.diag(self.StS))/len(X)
        
        if self.kernel_input:
            V_dot = np.diag(S.T @ util.center(X) @ S)
            Knrm = np.sum(util.center(X)**2)
        else:
            V_dot = np.sum((S.T@X)**2, axis=1)
            X_ = X - X.mean(0)
            Knrm = np.sum(X_.T@X_**2)

        # Generalized inner products for diagonal D layout
        dot = np.sum(self.scl * V_dot)
        Qnrm = self.scl @ (StS**2) @ self.scl
            
        return (Qnrm + Knrm - 2*dot)/Knrm
    
    def EStep(self, S, X, inplace=True):
        if self.kernel_input: 
            newS = bae_search.kerbmf2(
                X, 
                S, scl=self.scl, 
                StS=self.StS, N=self.n, inplace=inplace,
                alpha=self.sparse_reg, beta=self.tree_reg, temp=self.temp)
        else: 
            newS = bae_search.kerbmf3(
                X, 
                S, 
                scl=self.scl, 
                StS=self.StS, StX=self.StX, N=self.n, inplace=inplace,
                alpha=self.sparse_reg, beta=self.tree_reg, temp=self.temp)
        return newS

    def MStep(self, ES, X):
            """
            Optimally scale S, enforcing non-negativity on the weights.
            """
            S_ = ES - ES.mean(0)
            StS = self.StS - np.outer(np.diag(self.StS), np.diag(self.StS))/len(X)
            StX = self.StX - np.outer(np.diag(self.StS), X.mean(0))

            if self.kernel_input:
                V_dot = np.diag(S_.T @ X @ S_)
            else:
                V_dot = np.sum(StX**2, axis=1)
                
            G = StS**2
            
            if self.uniform_scale:
                dot = np.sum(V_dot)
                nrm = np.sum(G)
                # V_dot and G are strictly positive, but defensive clipping is safe
                target = np.full(self.dim_hid, max(0.0, dot / nrm))
            else:
                # --- Projected Gradient Step with Exact Line Search ---
                # 1. Compute the gradient dL/dw
                grad = G @ self.scl - V_dot + self.l1_reg
                
                # 2. Compute the exact optimal step size for the quadratic form
                # Added 1e-8 to prevent division by zero if gradient is completely flat
                eta = np.dot(grad, grad) / (np.dot(grad, G @ grad) + 1e-8)
                
                # 3. Take the step and project onto the non-negative orthant
                target = np.maximum(0.0, self.scl - eta * grad)
            
            # Exponential Moving Average update using your existing scl_lr
            # (This mathematically preserves non-negativity since target >= 0 and scl_lr <= 1)
            self.scl += self.scl_lr * (target - self.scl)
            
            # Compute exact multi-feature normalization matching the loss surface
            Qnrm = self.scl @ G @ self.scl
            dot_total = np.sum(self.scl * V_dot)
            
            return 1 + (Qnrm - 2*dot_total)/self.data_norm


@dataclass
class CorrBMF(BMF):
    """
    Correlated BMF
    """

    dim_hid: int
    nonneg: bool = False
    fit_intercept: bool = True
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    pr_reg: float = 1e-2
    l2_reg: float = 1e-2
    m_iters: float = 1
    J_l1_reg: float = None 
    J_loss: str = 'rple'

    def __call__(self, S):
        return S@self.W.T + self.b

    def init_params(self, X, J_lr=1e-2, **opt_args):

        self.n, self.d = X.shape

        if self.J_l1_reg is None:
            self.J_l1_reg = np.sqrt(np.log(self.dim_hid**2 / 1e-3)/self.n)
            if self.J_loss == 'rple':
                self.J_l1_reg *= 0.2
            else:
                self.J_l1_reg *= 0.8

        self.b = nn.Parameter(torch.zeros(self.d)) # Initialize b

        ## Initialize W
        W = np.random.randn(self.d, self.dim_hid)/np.sqrt(self.d)
        if self.nonneg:
            W[W < 0] = 0
        self.W = nn.Parameter(torch.FloatTensor(W))

        ## Initialize J
        self.J = nn.Linear(self.dim_hid, self.dim_hid)
        self.J.weight.data.copy_(torch.zeros(self.dim_hid, self.dim_hid))
        self.J.bias.data.copy_(torch.zeros(self.dim_hid))
        parametrize.register_parametrization(self.J, "weight", ZeroDiag())

        # self.optimizer = optim.Adam([{"params":[self.W]},
        #                              {"params":self.J.parameters(), "lr":J_lr}], 
        #                              **opt_args)
        self.optimizer = optim.SGD([{"params":[self.W]},
                                    {"params":self.J.parameters(), "lr":J_lr}], 
                                    **opt_args)

    def init_latents(self, X, **args):
        ## Initialize S 
        Mx = (X@self.W).detach().numpy()
        self.S = 1*(Mx >= 0.5)

    def EStep(self, S, X, inplace=True):

        # newS = binary_glm(self.data*1.0, oldS, self.W, self.b, steps=self.S_steps,
        #     beta=self.beta, temp=self.temp, lognorm=self.lognorm)

        with torch.no_grad():
            XW = (X@self.W - self.b@self.W)
            WtW = self.W.T@self.W

            XW = XW.detach().numpy()
            WtW = WtW.detach().numpy()

            J = self.J.weight.detach().numpy()
            h = self.J.bias.detach().numpy()
            J = (J + J.T) / 2
            J = 4*J + 2*np.diag(h - 2*J.sum(1))

        if self.tree_reg > 1e-6:
            StS = 1.0*(S.T@S)
            newS = bae_search.sbmf(XW, 1.0*S, WtW + J, 
                StS=StS, N=self.n, 
                beta=self.tree_reg, alpha=self.sparse_reg,
                temp=self.temp)
        else:
            newS = bae_search.sbmf(XW, 1.0*S, WtW + J,
                beta=self.tree_reg, alpha=self.sparse_reg,
                temp=self.temp)

        return newS

    def MStep(self, ES, X):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        Spt = torch.FloatTensor(ES)
        Srbm = 2*Spt - 1
        ls = []
        for _ in range(self.m_iters):
            
            self.optimizer.zero_grad()
            
            Xhat = Spt@self.W.T + self.b
            loss = torch.sum((X - Xhat)**2)/len(X)
            
            ls.append(loss.item())

            ## Regularization on W
            WtW = self.W.T@self.W
            loss -= self.pr_reg*((torch.trace(WtW)**2)/torch.sum(WtW**2))/self.dim_hid
            loss += self.l2_reg*torch.trace(WtW)/self.dim_hid

            ## Inverse Ising via log-RISE
            pred = self.J(Srbm)
            if self.J_loss == 'logrise':
                loss += torch.sum(torch.logsumexp(-pred*Srbm, 1)) / len(X)
            else:
                loss += torch.sum(nn.Softplus()(-2*pred*Srbm)) / len(X)
            loss += self.J_l1_reg*torch.sum(torch.abs(self.J.weight))

            loss.backward()
            
            self.optimizer.step()

            if self.nonneg:
                with torch.no_grad():
                    self.W[self.W<0] = 0
                    self.b[self.b<0] = 0

        return [ls[-1], self.J.weight.detach().numpy(), self.J.bias.detach().numpy()]

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        N = self.S@self.W.T + self.b
        return np.mean((X[mask] - N[mask])**2)


@dataclass
class SCPD(BMF):
    """
    Sparse canonical polyadic decomposition
    """

    dim_hid: int
    nonneg: bool = False
    fit_intercept: bool = True
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    pr_reg: float = 1e-2
    l1_reg: float = 1e-2
    m_iters: int = 1            # how many gradient steps
    nmf_init: bool = False

    def __call__(self):
        with torch.no_grad():
            U = self.U.detach().numpy()
            V = self.V.detach().numpy()
            b = self.b.detach().numpy()

        Xhat = np.einsum('ck,tk,nk->ctn', self.S, U, V) + b
        return Xhat

    def initialize(self, X, **opt_args):

        self.n, self.t, self.d = X.shape # X is a 3d tensor
        X_ = X.numpy()

        if self.nmf_init: # Initialize with NMF
            U = np.zeros((self.t, self.dim_hid))
            V = np.zeros((self.d, self.dim_hid))
            if self.nonneg:
                nmf = NMF(self.dim_hid, l2_ratio=1, alpha_W=self.weight_l2_reg)

                Z = nmf.fit_transform(self.data - self.data.min())
            else:
                nmf = df_util.SemiNMF(self.dim_hid)
                nmf.fit(X_.reshape((self.n, -1)).T)
                Z = nmf.Z.T
                for k in range(self.dim_hid):
                    Wk = nmf.W[:,k].reshape((self.t, self.d))
                    Upca,s,Vpca = la.svd(Wk,full_matrices=False)
                    U[:,k] = Upca[:,0] * np.sqrt(s[0])
                    V[:,k] = Vpca[0]

            self.S = df_util.binarize(Z)

        else:
            ## Initialize W
            U = np.random.randn(self.t, self.dim_hid)/np.sqrt(self.t)
            V = np.random.randn(self.d, self.dim_hid)/np.sqrt(self.d)

            ## Initialize S 
            XW = (U[None]*((X_-X_.mean(0))@V)).sum(1)
            self.S = 1*(XW >= 0.5)

        if self.nonneg:
            U[U < 0] = 0
            V[V < 0] = 0

        self.U = nn.Parameter(torch.tensor(U))
        self.V = nn.Parameter(torch.tensor(V))

        self.b = nn.Parameter(X.mean(0))

        self.optimizer = optim.Adam([self.U,self.V,self.b], **opt_args)

    def EStep(self, S, X):

        # newS = binary_glm(self.data*1.0, oldS, self.W, self.b, steps=self.S_steps,
        #     beta=self.beta, temp=self.temp, lognorm=self.lognorm)

        with torch.no_grad():
            XW = (self.U[None]*((X-self.b)@self.V)).sum(1)
            WtW = (self.U.T@self.U)*(self.V.T@self.V)

            XW = XW.detach().numpy()
            WtW = WtW.detach().numpy()

        if self.tree_reg > 1e-6:
            StS = 1.0*S.T@S
            newS = bae_search.sbmf(XW, 1.0*S, WtW, 
                StS=StS, N=self.n, 
                beta=self.tree_reg, alpha=self.sparse_reg,
                temp=self.temp)
        else:
            newS = bae_search.sbmf(XW, 1.0*S, WtW, 
                beta=self.tree_reg, alpha=self.sparse_reg,
                temp=self.temp)

        return newS

    def MStep(self, ES, X):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        Spt = torch.tensor(ES)
        # ls = []
        for _ in range(self.m_iters):
            
            self.optimizer.zero_grad()
            
            Xhat = torch.einsum('ck,tk,nk->ctn', Spt, self.U, self.V)
            loss = torch.sum((X - Xhat - self.b)**2)/len(X)
            
            # ls.append(loss.item())
            
            VtV = self.V.T@self.V
            loss -= self.pr_reg*((torch.trace(VtV)**2)/torch.sum(VtV**2))/self.dim_hid
            loss += self.l1_reg*torch.sum(torch.abs(self.V))/self.dim_hid
            loss += self.l1_reg*torch.sum(torch.abs(self.U))/self.dim_hid

            loss.backward()
            
            self.optimizer.step()

            if self.nonneg:
                with torch.no_grad():
                    self.V[self.V<0] = 0
                    self.U[self.U<0] = 0
                    self.b[self.b<0] = 0

        return loss.item()

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        return np.mean((X[mask] - self()[mask])**2)

@dataclass
class ConvBMF(BMF):
    """
    Convolutional BMF
    """

    dim_hid: int
    kernel_size: int
    # rank: int = None
    nonneg: bool = False
    fit_intercept: bool = True
    sparse_reg: float = 1e-2
    seq_reg: float = 1e-2
    time_sparsity: float = 1e-1
    feature_sparsity: float = 1e-1
    pr_reg: float = 1e-2
    ker_reg: float = 1e-2
    gp_width: float = 0.1
    m_iters: int = 1            # how many gradient steps

    def __call__(self):
        with torch.no_grad():
            K = self.K.flip([2]).detach().numpy()
            b = self.b.detach().numpy()

            Xhat = F.conv1d(self.S, self.K.flip([2]), padding=self.pad) + self.b
        
        return Xhat

    def init_params(self, X, **opt_args):

        self.n, self.d, self.t = X.shape # X is a 3d tensor
        
        self.pad = self.kernel_size-1

        ## Initialize kernels
        t = np.linspace(0, 1, self.kernel_size)
        ker = util.RBF(self.gp_width)

        ## Initialize convolutions from GP
        K = util.gaussian_process(t[:,None], (self.d, self.dim_hid), ker)
        K /= np.sqrt(self.d*self.dim_hid)

        if self.nonneg:
            K[K < 0] = 0

        self.K = nn.Parameter(torch.FloatTensor(K))
        if self.fit_intercept:
            self.b = nn.Parameter(X.mean(0))
            self.optimizer = optim.Adam([self.K,self.b], **opt_args)

        else:
            self.b = torch.zeros(self.d, self.t)
            self.optimizer = optim.Adam([self.K], **opt_args)

    def init_latents(self, X, **opt_args):

        XW = F.conv1d(X-self.b, self.K.transpose(0,1), padding=0)
        self.S = 1*(XW >= 0)

    def EStep(self, S, X):

        with torch.no_grad():
            Kt = self.K.transpose(0,1)
            XW = F.conv1d(X-self.b, Kt, padding=0)
            WtW = F.conv1d(Kt, Kt, padding=self.pad)

            XW = XW.detach().numpy()
            WtW = WtW.detach().numpy()

            # filt =  np.ones(2*self.kernel_size-1)
            # XWreg = np.apply_along_axis(np.convolve, 0, XW, filt, mode='same')
            filt = np.ones(2 * self.kernel_size - 1)
            XW_smooth = np.apply_along_axis(
                lambda x: np.convolve(x, filt, mode='same'), 2, XW)   # axis 2 = time
            XWreg = XW_smooth.sum(axis=1, keepdims=True) - XW_smooth  # sum over c'≠c

            S_ = S.detach().numpy()

        newS = bae_search.convbmf(XW, S_, WtW, 
            beta=self.feature_sparsity, 
            alpha=self.time_sparsity,
            l1_reg=self.sparse_reg,
            temp=self.temp)

        return torch.FloatTensor(newS)

    def MStep(self, ES, X):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        ls = []
        for _ in range(self.m_iters):
            
            self.optimizer.zero_grad()

            Xhat = F.conv1d(ES, self.K.flip([2]), padding=self.pad) + self.b
            loss = torch.sum((X - Xhat)**2)/len(X)
            
            ls.append(1*loss.item())

            ## Orthogonality regularization on primitives
            ## Based on seq-NMF regularizer from Alex's paper
            filt = torch.ones(2*self.kernel_size-1)
            XW = F.conv1d(X-self.b, self.K.transpose(0,1), padding=0)
            Sfilt = F.conv1d(ES.T, filt.flip(), padding=self.pad)
            loss += self.seq_reg*torch.sum(torch.abs(XW@Sfilt))

            ls.append(1*loss.item())

            loss.backward()
            
            self.optimizer.step()

            if self.nonneg:
                with torch.no_grad():
                    self.K[self.K<0] = 0
                    self.b[self.b<0] = 0

        return [ls[-2]-ls[-1], ls[-2]]

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        return np.mean((X[mask] - self()[mask])**2)


@dataclass
class RRBMF(BMF):
    """
    Reduced rank BMF
    """

    dim_hid: int
    rank: int 
    nonneg: bool = False
    fit_intercept: bool = True
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    pr_reg: float = 1e-2
    l1_reg: float = 0.0
    l2_reg: float = 1e-2
    m_iters: int = 1            # how many gradient steps

    def __call__(self, S):
        with torch.no_grad():
        #     U = self.U.detach().numpy()
        #     V = self.V.detach().numpy()
        #     b = self.b.detach().numpy()

            beta = self.V@self.U.T
            Xhat = torch.einsum('...ck,knt->...ctn', torch.tensor(S), beta) + self.b
        return Xhat

    def init_params(self, X, U=None, scl_lr=0.0, opt_alg=optim.SGD, **opt_args):

        self.n, self.t, self.d = X.shape # X is a 3d tensor
        X_ = X.numpy()

        ## Initialize W
        if U is None:
            U = np.random.randn(self.t, self.rank)/np.sqrt(self.t*self.rank)
            fit_U = True
        else:
            fit_U = False
        V = np.random.randn(self.dim_hid, self.d, self.rank)/np.sqrt(self.d*self.rank)
        if self.nonneg:
            U[U < 0] = 0
            V[V < 0] = 0
        b = X_.mean(0)

        self.sigma_x = 1.0
        self.scl_lr = scl_lr

        if fit_U:
            self.U = nn.Parameter(torch.tensor(U))
        else:
            self.U = torch.tensor(U)
        self.V = nn.Parameter(torch.tensor(V))
        self.b = nn.Parameter(torch.tensor(b))

        if fit_U:
            params = [self.U,self.V,self.b]
        else:
            params = [self.V,self.b]

        self.optimizer = opt_alg(params,**opt_args)

    def init_latents(self, X, **kwargs):

        with torch.no_grad():
        ## Initialize S 
            XW = np.einsum('kdt,ntd->nk',self.V@self.U.T,X-self.b)
        self.S = 1.0*(XW > 0)
        self.StS = self.S.T@self.S

    def EStep(self, S, X, inplace=False):

        # newS = binary_glm(self.data*1.0, oldS, self.W, self.b, steps=self.S_steps,
        #     beta=self.beta, temp=self.temp, lognorm=self.lognorm)

        with torch.no_grad():
            beta = self.V@self.U.T

            XW = torch.einsum('kdt,ntd->nk',beta,X-self.b)
            WtW = torch.einsum('kdt,cdt', beta,beta)

            XW = XW.detach().numpy()
            WtW = WtW.detach().numpy()

        newS = bae_search.sbmf(
            XW / self.sigma_x,
            S,
            WtW / self.sigma_x,
            StS=self.StS, N=self.n, 
            beta=self.tree_reg, alpha=self.sparse_reg,
            temp=self.temp,
            inplace=inplace,
            )

        return newS

    def MStep(self, ES, X):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        Spt = torch.tensor(ES)
        # ls = []
        for _ in range(self.m_iters):
            
            self.optimizer.zero_grad()
            
            beta = self.V@self.U.T
            Xhat = torch.einsum('ck,knt->ctn', Spt, beta) + self.b
            loss = torch.sum((X - Xhat)**2) / (self.d * self.t)
            
            new_sig = loss.item() / len(X)
                
            VtV = self.V.swapaxes(1,2)@self.V
            trace = torch.einsum('kii->k', VtV)
            norm = torch.sum(VtV**2, axis=(1,2))
            loss -= self.pr_reg*torch.sum((trace**2)/norm) / (self.rank*self.dim_hid)
            loss += self.l1_reg*torch.mean(torch.abs(self.V))
            loss += self.l2_reg*torch.mean(beta**2)

            loss.backward()
            
            self.optimizer.step()

            with torch.no_grad():
                self.sigma_x += self.scl_lr*(new_sig - self.sigma_x)
                if self.nonneg:
                    self.V[self.V<0] = 0
                    self.U[self.U<0] = 0
                    self.b[self.b<0] = 0

        return new_sig

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        return np.mean((X[mask] - self()[mask])**2) / np.mean(X[mask]**2)

@dataclass
class SpikeRRNMF(BMF):
    """
    Reduced rank NMF with spike-and-slab prior on S.
    Continuous slab multipliers Z replace the hard binary activations
    passed to the M-step, leaving the decoder math identical.
    """

    dim_hid: int
    rank: int
    nonneg: bool = False
    fit_intercept: bool = True
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    slab_prior: float = 1.0      # <-- tau: prior rate on slab magnitude
    pr_reg: float = 1e-2
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    m_iters: int = 1

    def __call__(self, SZ):
        with torch.no_grad():
            beta = self.V @ self.U.T
            Xhat = torch.einsum('...ck,knt->...ctn', torch.tensor(SZ), beta) + self.b
        return Xhat

    def init_params(self, X, U=None, **opt_args):

        self.n, self.t, self.d = X.shape
        X_ = X.numpy()

        if U is None:
            U = np.random.randn(self.t, self.rank) / np.sqrt(self.t * self.rank)
            fit_U = True
        else:
            fit_U = False

        V = np.random.randn(self.dim_hid, self.d, self.rank) / np.sqrt(self.d * self.rank)
        if self.nonneg:
            U[U < 0] = 0
            V[V < 0] = 0

        if fit_U:
            self.U = nn.Parameter(torch.tensor(U))
        else:
            self.U = torch.tensor(U)
        self.V = nn.Parameter(torch.tensor(V))
        self.b = nn.Parameter(torch.tensor(X_.mean(0)))

        if fit_U:
            self.optimizer = optim.Adam([self.U, self.V, self.b], **opt_args)
        else:
            self.optimizer = optim.Adam([self.V, self.b], **opt_args)

    def init_latents(self, X, **args):

        X_ = X.numpy()
        with torch.no_grad():
            beta = self.V @ self.U.T
            XW = np.einsum('kdt,ntd->nk', beta.numpy(), X_ - self.b.numpy())

        self.S = 1.0 * (XW >= 0.5)
        self.Z = XW * self.S
        self.StS = self.S.T @ self.S

    def EStep(self, S, X, slab=True):

        with torch.no_grad():
            beta = self.V @ self.U.T
            XW  = torch.einsum('kdt,ntd->nk', beta, X - self.b).detach().numpy()
            WtW = torch.einsum('kdt,cdt->kc', beta, beta).detach().numpy()

        newS, newZ = bae_search.snmf(XW=XW, 
                                    S=S,
                                    Z=self.Z,
                                    WtW=WtW,
                                    tau=self.slab_prior,
                                    beta=self.tree_reg,
                                    alpha=self.sparse_reg,
                                    temp=self.temp,
                                    StS=self.StS, 
                                    N=self.n)

        self.Z = newZ  # keep slab state in sync

        return newS * newZ if slab else newS

    # MStep and loss are identical to RRBMF — no changes needed
    def MStep(self, ES, X):
        Spt = torch.tensor(ES)
        ls = []
        for _ in range(self.m_iters):
            self.optimizer.zero_grad()
            beta = self.V @ self.U.T
            Xhat = torch.einsum('ck,knt->ctn', Spt, beta)
            loss = torch.sum((X - Xhat - self.b) ** 2)
            ls.append(loss.item())
            VtV   = self.V.swapaxes(1, 2) @ self.V
            trace = torch.einsum('kii->k', VtV)
            norm  = torch.sum(VtV ** 2, dim=(1, 2))
            loss -= self.pr_reg * torch.sum((trace ** 2) / norm) / self.rank
            loss += self.l1_reg * torch.sum(torch.abs(self.V))
            loss += self.l2_reg * torch.sum(beta ** 2)
            loss.backward()
            self.optimizer.step()
            if self.nonneg:
                with torch.no_grad():
                    self.V[self.V < 0] = 0
                    self.U[self.U < 0] = 0
                    self.b[self.b < 0] = 0
        return ls[-1]

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        return np.mean((X[mask] - self()[mask]) ** 2) / np.mean(X[mask] ** 2)

@dataclass
class Buffet(BMF):
    """
    Linear gaussian binary feature model
    """

    dim_hid: int = np.inf
    ibp_alpha: float = 1.0
    sigma_x: float = 1.0
    sigma_w: float = 1.0
    sparse_reg: float = 0
    l1_reg: float = 0.1

    def __call__(self):
        return self.S@self.W

    def initialize(self, X):

        self.n, self.d = X.shape
        self.data = X #/np.sqrt(np.mean(X**2))

        if self.dim_hid < np.inf:
            self.infinite = False
        else:
            self.infinite = True
            k_init = np.random.poisson(self.ibp_alpha*np.sum(1/np.arange(1,self.n+1)))
            self.dim_hid = k_init

        ## Initialize W
        self.W = np.random.randn(self.d, self.dim_hid)/np.sqrt(self.d)

        ## Initialize S 
        Mx = self.data@self.W
        self.S = 1*(Mx >= 0.5)

        ## Initial posterior over W (i.e. ridge regression)
        self.covW = la.inv(self.S.T@self.S/self.sigma_x**2 + np.eye(self.dim_hid) / self.sigma_w**2)
        self.meanW = self.covW@(self.S.T@self.data / self.sigma_x**2)

    def EStep(self, S, X):

        newS = bae_search.gauss_bern(X, 1.0*S, self.covW, self.meanW,
            sigma_x=self.sigma_x, alpha=self.sparse_reg, temp=self.temp)

        return newS

    def MStep(self, ES, X):
        """
        There's no M-step because we maintain a posterior over W
        """
        self.W = self.meanW
        return np.mean((ES@self.W - X)**2)

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        N = self.S@self.W.T + self.b
        return np.mean((X[mask] - N[mask])**2)


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
    tree_reg: float = 0
    sparse_reg: float = 1e-2
    weight_reg: float = 1e-2

    def __post_init__(self):
        super().__init__()
        self.temp = 1e-5 

        self.q = nn.Linear(self.dim_inp, self.dim_hid) # x -> s
        self.p = nn.Linear(self.dim_hid, self.dim_inp) # s -> x'

        self.Cov = torch.zeros((self.dim_hid, self.dim_hid))

    def fit(self, *data, initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=True, **opt_args):

        if max_iter is None:
            max_iter = period*int(np.log(1e-4/initial_temp)/np.log(decay_rate))

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

        return torch.sigmoid(S/self.temp)

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


