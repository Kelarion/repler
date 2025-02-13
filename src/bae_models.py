import os, sys, re
import pickle
from time import time
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.distributions as dis
import torch.linalg as tla
import numpy as np
from itertools import permutations, combinations
# from tqdm import tqdm
import geoopt as geo

import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import numpy.linalg as nla
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

from numba import njit
import math

# my code
import util
import pt_util
import bae_util

####################################################################
############ Base PyTorch module for BAE ###########################
####################################################################

class BAE(nn.Module):

    def __init__(self):
        super().__init__()

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

    def grad_step(self, X, temp, use_sample=True, **opt_args):
        """
        One iteration of optimization
        """

        E = self.EStep(temp)
        if use_sample:
            loss = self.MStep(self.S, **opt_args)
        else:
            loss = self.MStep(E, **opt_args)

        return loss

    def initialize(self):

        raise NotImplementedError


####################################################################
################# Specific instances ###############################
####################################################################

class GaussBAE(BAE):
    """
    Simplest BAE, with Gaussian observations and orthogonal weights

    Setting
        `weight_alg = 'exact', tree_reg=0`
    results in fast, closed-form updates, but will have poor performance
    on non-identifiable instances. 
    """

    def __init__(self, dim_hid, tree_reg=0, sparse_reg=1e-1, 
        do_pca=False, center=True, chains=1):

        super().__init__()

        self.r = dim_hid
        self.k = chains

        self.alpha = sparse_reg
        self.beta = tree_reg 
        self.reduce = do_pca
        self.center = center

    def initialize(self, X, alpha=2, beta=5, rank=None, pvar=1, W_init='pca'):

        self.n, self.d = X.shape

        if self.center:
            X_ = X - X.mean(0)
        else:
            X_ = X

        if self.reduce:
            Ux,sx,Vx = la.svd(X_-X_.mean(0), full_matrices=False)
            self.frac_var = np.cumsum(sx**2)/np.sum(sx**2)
            if rank is None:
                r = np.min([len(sx), np.sum(self.frac_var <= pvar)+1])
                # r = np.max([dim_hid, np.sum(self.frac_var <= pvar)+1])
            else:
                r = np.min([rank, np.sum(self.frac_var <= pvar)+1])

            r = np.max([self.r, r]) # needs to have enough 'space' for each dimension

            self.d = r

            self.data = Ux[:,:r]@np.diag(sx[:r])
            self.V = Vx[:r]
        else:
            self.data = X_

        ## Initialise b
        self.b = np.zeros(self.d) # -X.mean(0)

        ## Initialize W
        if W_init == 'pca':
            X_-X_.mean(0)
            _, V = la.eigh((X_-X_.mean(0)).T@(X_-X_.mean(0)))
            if self.n > self.r:
                self.W = V[:,:self.r]
        else:
            self.W = sts.ortho_group.rvs(self.d)[:,:self.r]
        # self.W = Vx[:self.r].T 
        self.scl = 1

        ## Initialize S
        self.S = 1*(self.data@self.W > 1/2)
        # coding_level = np.random.beta(alpha, beta, self.r)/2
        # num_active = np.floor(coding_level*self.n).astype(int)

        # Mx = self.data@self.W
        # thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.r)]
        # self.S = 1*(Mx >= thr)

    def EStep(self, temp):
        """
        Compute expectation of log-likelihood over S 

        X is a FloatTensor of shape (num_inp, dim_inp)
        """

        if (self.beta > 1e-6) or self.center:
            if self.center:
                newS = update_concepts_asym_cntr(self.data, 1.0*self.S, self.W, self.scl, 
                    alpha=self.alpha, beta=self.beta, temp=temp)
            else:
                newS = update_concepts_asym(self.data-self.b, 1.0*self.S, self.W, self.scl, 
                    beta=self.beta, temp=temp)
        else:
            C = 2*(self.data-self.b)@self.W - self.scl - self.alpha
            newS = 1*(np.random.rand(self.n, self.r) > spc.expit(C/temp))

        self.S = newS

        if self.center:
            return newS - newS.mean(0)
        else:
            return newS

    def MStep(self, ES):

        U,s,V = la.svd((self.data-self.b).T@ES, full_matrices=False)

        self.W = U@V
        self.scl = np.sum(s)/np.sum(ES**2)
        if not self.center:
            self.b = (self.data - self.scl*ES@self.W.T).mean(0)

        return self.scl*np.sqrt(np.sum(ES**2))/np.sqrt(np.sum((self.data-self.b)**2))

    def __call__(self):
        return self.scl*self.S@self.W.T + self.b

    def loss(self, X):
        return self.scl*np.sqrt(np.sum(self.S))/np.sqrt(np.sum(X-self.b)**2)


class GeBAE(BAE):
    """
    Generalized BAE, taking any exponential family observation (in theory)
    """

    def __init__(self, dim_hid, noise='gaussian', tree_reg=1e-2, weight_reg=1e-2, 
        S_steps=1, W_steps=1, do_pca=False):

        super().__init__()

        self.has_data = False
        self.reduce = do_pca

        self.r = dim_hid

        self.S_steps = S_steps
        self.W_steps = W_steps

        self.alpha = weight_reg
        self.beta = tree_reg

        ## rn only support three kinds of observation noise, because
        ## other distributions have constraints on the natural params
        ## (mostly non-negativity) which I don't want to deal with 
        if noise == 'gaussian':
            self.lognorm = bae_util.gaussian
            self.mean = lambda x:x
            self.likelihood = sts.norm
            self.base = lambda x: (-x**2 - np.log(2*np.pi))/2

        elif noise == 'poisson':
            self.lognorm = bae_util.poisson
            self.mean = np.exp
            self.likelihood = sts.poisson
            self.base = lambda x: -np.log(spc.factorial(x))

        elif noise == 'bernoulli':
            self.lognorm = bae_util.bernoulli
            self.mean = spc.expit
            self.likelihood = sts.bernoulli
            self.base = lambda x: 0

        # ## Initialization is better when it's data-dependent
        # self.initialize(X_init, alpha, beta)

    def __call__(self):
        N = self.S@self.W.T + self.b
        return self.likelihood(self.mean(N)).rvs()

    def initialize(self, X, alpha=2, beta=5, rank=None, pvar=1):

        self.n, self.d = X.shape
        if self.reduce:

            Ux,sx,Vx = la.svd(X-X.mean(0), full_matrices=False)
            self.frac_var = np.cumsum(sx**2)/np.sum(sx**2)
            if rank is None:
                r = np.min([len(sx), np.sum(self.frac_var <= pvar)+1])
                # r = np.max([dim_hid, np.sum(self.frac_var <= pvar)+1])
            else:
                r = np.min([rank, np.sum(self.frac_var <= pvar)+1])

            self.d = r

            self.data = Ux[:,:r]@np.diag(sx[:r])
            self.V = Vx[:r]
        else:
            self.data = X
        self.has_data = True

        ## Initialise b
        self.b = np.zeros(self.d) # -X.mean(0)

        ## Initialize W
        self.W = np.random.randn(self.d, self.r)/np.sqrt(self.d)

        ## Initialize S 
        coding_level = np.random.beta(alpha, beta, self.r)/2
        num_active = np.floor(coding_level*self.n).astype(int)

        Mx = self.data@self.W
        # thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.r)]
        # self.S = 1*(Mx >= thr)
        self.S = 1*(Mx >= 0)

    def EStep(self, temp):

        oldS = 1.0*self.S
        newS = binary_glm(self.data*1.0, oldS, self.W, self.b, steps=self.S_steps,
            beta=self.beta, temp=temp, lognorm=self.lognorm)

        self.S = newS

        return newS

    def MStep(self, ES, W_lr=0.1, b_lr=0.1):
        """
        Maximise log-likelihood conditional on S, with p.r. regularization
        """

        for i in range(self.W_steps):

            N = ES@self.W.T + self.b

            WTW = self.W.T@self.W

            dXhat = (self.data - self.mean(N))
            # dReg = self.alpha*self.W@np.sign(self.W.T@self.W)
            dReg = self.alpha*self.W@(np.eye(self.r) - WTW*np.trace(WTW)/np.sum(WTW**2))

            dW = dXhat.T@ES/len(self.data)
            db = dXhat.sum(0)/len(self.data)

            self.W += W_lr*(dW + dReg)
            self.b += b_lr*db

        return np.mean(self.data*N - self.lognorm(N))

    def loss(self, X, mask=None):
        if mask is None:
            mask = np.ones(X.shape) > 0
        N = self.S@self.W.T + self.b
        return -np.mean(X[mask]*N[mask] - self.lognorm(N[mask]) + self.base(X[mask]))


class KernelBAE(BAE):
    
    def __init__(self, dim_hid, penalty=1e-2, steps=1, max_ctx=None, fix_scale=False):
        
        super().__init__()

        self.r = dim_hid
        self.initialized = False

        self.steps = steps
        self.beta = penalty
        self.max_ctx = max_ctx
        self.fix_scale = fix_scale

    def initialize(self, X, alpha=2, beta=5, rank=None, pvar=1):

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
        K = self.X@self.X.T
        notI = (1 - np.eye(self.n))/(self.n-1) 
        self.data = (K - K@notI - (K*notI).sum(0) + ((K@notI)*notI).sum(0)).T 

        ## Initialize S
        coding_level = np.random.beta(alpha, beta, self.r)/2
        num_active = np.floor(coding_level*len(X)).astype(int)

        Mx = self.X@np.random.randn(len(X.T),self.r)
        thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.r)]
        # self.S = sprs.csc_array(1*(Mx >= thr))
        self.S = np.array((Mx >= thr)*1)
        self.scl = 1

    def __call__(self):

        return self.scl*self.S@self.S.T

    def loss(self):
        """
        Compute the energy of the network, for a subset I
        """
        
        X = self.X
        S = self.S
        K = X@X.T
        dot = self.scl*np.sum(util.center(K)*(S@S.T))
        Qnrm = (self.scl**2)*np.sum(util.center((S@S.T))**2)
        Knrm = np.sum(util.center(K)**2)
        
        return Qnrm + Knrm - 2*dot
    
    def EStep(self, temp):

        newS = update_concepts_kernel(self.data, 1.0*self.S, 
            scl=self.scl, beta=self.beta, temp=temp)

        self.S = newS

        return self.S

    def MStep(self, ES):
        """
        Optimally scale S
        """
        
        S_ = ES - ES.mean(0)
        
        dot = np.sum((self.X@self.X.T)*(S_@S_.T))
        nrm = np.sum((S_@S_.T)**2)

        self.scl = dot/nrm
        
        return nrm - 2*dot

############################################################
######### Jitted update of S ###############################
############################################################

@njit
def binary_glm(X, S, W, b, beta, temp, steps=1, lognorm=bae_util.gaussian):
    """
    One gradient step on S

    TODO: figure out a good sparse implementation!
    """

    n, m = S.shape

    StS = np.dot(S.T, S)
    St1 = S.sum(0)

    ## Initial values
    E = np.dot(S,W.T) + b  # Natural parameters
    C = np.dot(X,W)        # Constant term

    for step in range(steps):
        for i in np.random.permutation(np.arange(n)):
        # en = np.zeros((n,m))
        # for i in np.arange(n): 

            if beta > 1e-6:
                ## Organize states
                s = S[i]
                St1 -= s
                StS -= np.outer(s,s)

                ## Regularization (more verbose because of numba reasons)
                D1 = StS
                D2 = St1[None,:] - StS
                D3 = St1[:,None] - StS
                D4 = (n-1) - St1[None,:] - St1[:,None] + StS

                best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
                best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
                best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
                best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

                R = (best1 - best2 - best3 + best4)*1.0
                r = (best2.sum(0) - best4.sum(0))*1.0

            ## Hopfield update of s
            for j in np.random.permutation(np.arange(m)): # concept

                ## Compute linear terms
                dot = np.sum(lognorm(E[i] + (1-S[i,j])*W[:,j])) 
                dot -= np.sum(lognorm(E[i] - S[i,j]*W[:,j]))
                if beta > 1e-6:
                    inhib = np.dot(R[j], S[i]) + r[j]
                else:
                    inhib = 0

                ## Compute currents
                curr = (C[i,j] - beta*inhib - dot)/temp

                ## Apply sigmoid (overflow robust)
                if curr < -100:
                    prob = 0.0
                elif curr > 100:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-curr))

                ## Update outputs
                sj = 1*(np.random.rand() < prob)
                ds = sj - S[i,j]
                S[i,j] = sj
                
                ## Update dot products
                E[i] += ds*W[:,j]

                # en[i,j] = np.sum(lognorm(E)) - 2*np.sum(C*S)

            ## Update 
            # S[i] = news
            St1 += S[i]
            StS += np.outer(S[i], S[i]) 

    return S #, en

@njit
def update_concepts_asym(X, S, W, scl, beta, temp, steps=1):
    """
    One gradient step on S

    TODO: figure out a good sparse implementation!
    """

    n, m = S.shape

    StS = np.dot(S.T, S)
    St1 = S.sum(0)

    ## Initial values
    C = np.dot(X,W)        # Constant term

    for step in range(steps):
        for i in np.random.permutation(np.arange(n)):
        # en = np.zeros((n,m))
        # for i in np.arange(n): 

            ## Organize states
            St1 -= S[i]
            StS -= np.outer(S[i],S[i])

            ## Regularization (more verbose because of numba reasons)
            D1 = StS
            D2 = St1[None,:] - StS
            D3 = St1[:,None] - StS
            D4 = (n-1) - St1[None,:] - St1[:,None] + StS

            best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
            best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
            best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
            best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

            R = (best1 - best2 - best3 + best4)*1.0
            r = (best2.sum(0) - best4.sum(0))*1.0

            ## Hopfield update of s
            for j in np.random.permutation(np.arange(m)): # concept

                ## Compute linear terms
                inhib = np.dot(R[j], S[i]) + r[j]

                ## Compute currents
                curr = (2*C[i,j] - beta*scl*inhib - scl)/temp

                # ## Apply sigmoid (overflow robust)
                if curr < -100:
                    prob = 0.0
                elif curr > 100:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-curr))

                ## Update outputs
                sj = 1*(np.random.rand() < prob)
                ds = sj - S[i,j]
                S[i,j] = sj

                # en[i,j] = np.sum(lognorm(E)) - 2*np.sum(C*S)

            ## Update 
            # S[i] = news
            St1 += S[i]
            StS += np.outer(S[i], S[i]) 

    return S #, en

@njit
def update_concepts_asym_cntr(X, S, W, scl, alpha, beta, temp, steps=1):
    """
    One gradient step on S

    TODO: figure out a good sparse implementation!
    """

    n, m = S.shape

    if beta >= 1e-6:
        StS = np.dot(S.T, S)
    St1 = S.sum(0)

    ## Initial values
    # C = np.dot(X,W)        # Constant term
    
    for step in range(steps):
        for i in np.random.permutation(np.arange(n)):

            ## Organize states
            St1 -= S[i]

            if beta >= 1e-6:
                StS -= np.outer(S[i], S[i])

                ## Regularization (more verbose because of numba reasons)
                D1 = StS
                D2 = St1[None,:] - StS
                D3 = St1[:,None] - StS
                D4 = (n-1) - St1[None,:] - St1[:,None] + StS

                best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
                best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
                best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
                best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

                R = (best1 - best2 - best3 + best4)*1.0
                r = (best2.sum(0) - best4.sum(0))*1.0

            ## Hopfield update of s
            for j in np.random.permutation(np.arange(m)): # concept

                # inp = (2*C[i,j] - scl*(n - 1 - 2*St1[j])/n - alpha)
                inp = (2*np.dot(X[i], W[:,j]) - scl*(n - 1 - 2*St1[j])/n - alpha)

                ## Compute linear terms
                if beta >= 1e-6:
                    inhib = np.dot(R[j], S[i]) + r[j]
                else:
                    inhib = 0

                ## Compute currents
                curr = (inp - beta*scl*inhib)/temp

                # ## Apply sigmoid (overflow robust)
                if curr < -100:
                    prob = 0.0
                elif curr > 100:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-curr))

                ## Update outputs
                S[i,j] = 1*(np.random.rand() < prob)

            # else:

            #     for j in range(m):

            #         # curr = (2*C[i,j] - scl*(n - 1 - 2*St1[j])/n - alpha)/temp
            #         inp = (2*np.dot(X[i], W[:,j]) - scl*(n - 1 - 2*St1[j])/n - alpha)
                    
            #         if curr < -100:
            #             prob = 0.0
            #         elif curr > 100:
            #             prob = 1.0
            #         else:
            #             prob = 1.0 / (1.0 + math.exp(-curr))

            #         ## Update outputs
            #         S[i,j] = 1*(np.random.rand() < prob)
                
            ## Update 
            # S[i] = news
            St1 += S[i]
            StS += np.outer(S[i], S[i]) 

    return S #, en

@njit
def update_concepts_kernel(K, S, scl, beta, temp, steps=1):
    """
    One gradient step on S

    TODO: figure out a good sparse implementation!
    """

    n, m = S.shape

    StS = np.dot(S.T, S)
    St1 = S.sum(0)

    ## Sparse (CSC) representation of S
    # aye, ridx = np.nonzero(S.T)     # get row indices (compress columns)
    # rptr = np.unique(aye, return_index=True)[1]
    # rptr = np.append(rptr, len(ridx))

    # c2r = np.argsort(rindices) # conversion to CSR (which is better for row slicing)
    # cindptr = np.unique(rindices[c2r], return_index=True)[1]
    # cindptr = np.append(cindptr, len(rindices))
    # cindices = aye[c2r]

    for i in np.random.permutation(np.arange(n)):

        idx = np.mod(np.arange(n)+i, n) ## item i goes first
        I = idx[1:]

        ## Organize data
        k = K[i,I]
        k0 = K[i,i]

        ## Organize states
        Sk = np.dot(S[I].T, k)
        s = S[i]
        # St1[s>0] -= 1
        # StS[s>0][:,s>0] -= 1
        St1 -= s
        StS -= np.outer(s,s)

        ## Regularization (more verbose because of numba reasons)
        # I1 = np.sign(2*StS - St1[None,:])
        # I2 = np.sign(2*StS - St1[:,None])
        # I3 = np.sign(St1[None,:] - St1[:,None])
        # I4 = np.sign(St1[None,:] + St1[:,None] - (n-1))

        # R = ((1*(I1<0)*(I2<0) - 1*(I1>0)*(I3<0) - 1*(I2>0)*(I3>0))*(I4<0))*1.0
        # r = ((I2>0)*(I3>0)*(I4<0)).sum(1)*1.0

        D1 = StS
        D2 = St1[None,:] - StS
        D3 = St1[:,None] - StS
        D4 = (n-1) - St1[None,:] - St1[:,None] + StS

        best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
        best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
        best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
        best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

        R = (best1 - best2 - best3 + best4)*1.0
        r = (best2.sum(0) - best4.sum(0))*1.0

        ## Recurrence
        t = (n-1)/n
        
        # Compute the rank-one terms
        s_ = St1/(n-1)
        u = 2*s_ - 1
        
        s_sc_ = s_*(1-s_)

        ## Constants
        # sx = Sx.sum()
        sx = np.dot(s_, s - s_)
        ux = 2*sx - s.sum() + s_.sum()
        
        # Form the threshold 
        h = t*((scl**2)*s_sc_.sum() - scl*k0)*u + 2*scl*Sk - beta*(scl**2)*r
        
        # Need to subtract the diagonal and add it back in
        Jii = 2*(n-1)*s_sc_ + t*u**2

        ## Hopfield update of s
        news = 1*s
        for step in range(steps):
            for j in range(m): # concept

                # rows = ridx[rptr[j]:rptr[j+1]]

                # Compute sparse dot product
                dot = 2*np.dot(StS[j], news - s_)
                dot -= 2*(n-1)*s_[j]*sx
                dot += t*u[j]*ux
                dot -= Jii[j]*news[j]
                dot += beta*np.dot(R[j], news)

                ## Compute currents
                curr = (h[j] - (scl**2)*Jii[j]/2 - (scl**2)*dot)/temp

                ## Apply sigmoid (overflow robust)
                if curr < -100:
                    prob = 0.0
                elif curr > 100:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-curr))
                
                ## Update outputs
                sj = 1*(np.random.rand() < prob)
                ds = sj - news[j]
                news[j] = sj
                
                ## Update dot products
                if np.abs(ds) > 0:
                    sx += ds*s_[j]
                    ux += ds*u[j]

        ## Update 
        S[i] = news
        St1 += news
        StS += np.outer(news, news) 

        ## Keep in sparsest form
        # flip = (St1 > (n//2))

        # StS[flip] = St1[None,:] - StS[flip]
        # St1[flip] = n - St1[flip]
        # StS[:,flip] = St1[:,None] - StS[:,flip]
    
    return S

