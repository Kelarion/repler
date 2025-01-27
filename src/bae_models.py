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

        E = self.EStep(X, temp)
        if use_sample:
            loss = self.MStep(X, self.S, **opt_args)
        else:
            loss = self.MStep(X, E, **opt_args)

        return loss

    def initialize(self, X=None, alpha=2, beta=5):

        ## Initialise b
        self.b = np.zeros(self.d) # -X.mean(0)

        ## Initialize W
        self.W = np.random.randn(self.d, self.r)

        ## Initialize S 
        coding_level = np.random.beta(alpha, beta, self.r)/2
        num_active = np.floor(coding_level*self.n).astype(int)

        if X is None:
            self.S = 1*(np.random.rand(self.n, self.r) > coding_level)
        else:
            n,d = X.shape
            assert n == self.n
            assert d == self.d

            Mx = X@self.W
            thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.r)]
            self.S = 1*(Mx >= thr)


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

    def __init__(self, num_inp, dim_inp, dim_hid, 
        tree_reg=0, X_init=None, alpha=2, beta=5):

        super().__init__()

        self.n = num_inp
        self.d = dim_inp
        self.r = dim_hid

        self.beta = tree_reg

        ## Parameters
        self.initialize(X_init, alpha, beta)

    def EStep(self, X, temp):
        """
        Compute expectation of log-likelihood over S 

        X is a FloatTensor of shape (num_inp, dim_inp)
        """

        oldS = 1*self.S
        newS = binary_glm(X-self.b, oldS, self.W, beta=self.beta, temp=temp, lognorm=self.lognorm)

        self.S = newS

        return newS

    def MStep(self, X, ES):

        U,s,V = tla.svd((X-self.b).T@ES, full_matrices=False)

        self.W = U@V
        self.scl = torch.sum(s)/torch.sum(ES)
        self.b = (X - self.scl*ES@self.W.T).mean(0)

        return self.scl*np.sqrt(np.sum(ES))/np.sqrt(np.sum(X-self.b))

class GeBAE(BAE):
    """
    Generalized BAE, taking any exponential family observation (in theory)
    """

    def __init__(self, num_inp, dim_inp, dim_hid, observations='gaussian', 
        tree_reg=1e-2, weight_reg=1e-2, X_init=None, alpha=2, beta=5, S_steps=1, W_steps=1):

        super().__init__()
        self.n = num_inp
        self.d = dim_inp
        self.r = dim_hid

        self.S_steps = S_steps
        self.W_steps = W_steps

        self.alpha = weight_reg
        self.beta = tree_reg

        ## rn only support three kinds of observation noise, because
        ## other distributions have constraints on the natural params
        ## (mostly non-negativity) which I don't want to deal with 
        if observations == 'gaussian':
            self.lognorm = bae_util.gaussian
            self.mean = lambda x:x
            self.likelihood = sts.norm

        elif observations == 'poisson':
            self.lognorm = bae_util.poisson
            self.mean = np.exp
            self.likelihood = sts.poisson

        elif observations == 'bernoulli':
            self.lognorm = bae_util.bernoulli
            self.mean = spc.expit
            self.likelihood = sts.bernoulli

        ## Initialization is better when it's data-dependent
        self.initialize(X_init, alpha, beta)

    def __call__(self):
        N = self.S@self.W.T + self.b
        return self.likelihood(self.mean(N)).rvs()

    def EStep(self, X, temp):

        oldS = 1.0*self.S
        newS = binary_glm(X*1.0, oldS, self.W, self.b, steps=self.S_steps,
            beta=self.beta, temp=temp, lognorm=self.lognorm)

        self.S = newS

        return newS

    def MStep(self, X, ES, W_lr=0.1, b_lr=0.1):

        for i in range(self.W_steps):
            N = ES@self.W.T + self.b

            dXhat = (X - self.mean(N))
            dReg = self.alpha*self.W@np.sign(self.W.T@self.W)

            dW = dXhat.T@ES/len(X)
            db = dXhat.sum(0)/len(X)

            self.W += W_lr*(dW - dReg)
            self.b += b_lr*db

        return np.mean(X*N - self.lognorm(N))

    def loss(self, X):
        N = self.S@self.W.T + self.b
        return np.mean(X*N - self.lognorm(N))

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

            idx = np.mod(np.arange(n)+i, n) ## item i goes first
            I = idx[1:]

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
                inhib = np.dot(R[j], S[i]) + r[j]

                ## Compute currents
                curr = (C[i,j] - beta*inhib - dot)/temp

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
                
                ## Update dot products
                E[i] += ds*W[:,j]

                # en[i,j] = np.sum(lognorm(E)) - 2*np.sum(C*S)

            ## Update 
            # S[i] = news
            St1 += S[i]
            StS += np.outer(S[i], S[i]) 

    return S #, en


