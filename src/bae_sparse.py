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
import bae_util
import bae_models

############################################################
### Models which represent data and parameters as sparse ###
### matrices throughout training ###########################
############################################################

# @dataclass(eq=False)
# class SparseBAE(bae_models.BAE):

#   tree_reg: float = 0.1
#     weight_reg: float = 0.1

#     def EStep(self, S, X):
#       """
#       S and X are sparse arrays
#       """

@dataclass
class SparseBMF(bae_models.BMF):

    dim_hid: int
    tree_reg: float = 1e-2
    pr_reg: float = 1e-2
    nonneg: bool = False

    def initialize(self, X, W_lr=0.1, b_lr=0.1):

        self.N, m = X.shape

        self.W_lr = W_lr
        self.b_lr = b_lr

        self.data = X

        # U,s,V = sprs.linalg.svds(X, k=self.dim_hid)
        # self.W = V.T
        # self.W = sprs.random_array((m, self.dim_hid)).todense()
        self.W = np.random.randn(m, self.dim_hid)/np.sqrt(m)
        # V = np.random.randn(self.N, self.dim_hid)/np.sqrt(self.N)
        # self.W = X.T@V - np.outer(X.mean(0), V.sum(0))
        self.b = np.zeros(m)
        S = X@self.W
        S = 1*(S > 0.5)

        self.S = 1.0*S #.todense()
        self.StS = 1.0*(S.T@S) #.todense()
        self.WtW = self.W.T@self.W

    def EStep(self):

        XW = (self.data@self.W - self.b@self.W)
        WtW = self.WtW #.todense()
        StS = 1.0*self.StS

        newS, StS = sparse_bmf(XW, 1.0*self.S, WtW, StS, self.N, 
            temp=self.temp, beta=self.tree_reg)
        
        self.S = newS
        self.StS = StS

        return newS

    def MStep(self, S):

        dpr = np.trace(self.WtW)/np.sum(self.WtW**2)

        dW = (self.data.T@S - self.W@self.StS - np.outer(self.b, S.sum(0)))/self.N
        dReg = self.W - dpr*self.W@self.WtW

        db = (self.data.mean(0) - self.W@S.mean(0))

        self.W += self.W_lr*(dW + self.pr_reg*dReg)
        if self.nonneg:
            self.W *= (self.W > 0)
        self.WtW = self.W.T@self.W

        self.b += self.b_lr*db

        return (dW**2).mean()

@dataclass
class SparseBiPCA(bae_models.BMF):

    dim_hid: int
    tree_reg: float = 1e-2
    sparse_reg: float = 1e-2
    center: bool = False

    def initialize(self, X, W_lr=0.1, b_lr=0.1):

        self.N, m = X.shape

        self.W_lr = W_lr
        self.b_lr = b_lr

        self.data = X

        U,s,V = sprs.linalg.svds(X, k=self.dim_hid)
        self.W = V.T
        self.scl = np.sum(X**2)/(m*self.N)
        # self.W = sprs.random_array((m, self.dim_hid)).todense()
        # self.W = np.random.randn(m, self.dim_hid)/np.sqrt(m)
        # V = np.random.randn(self.N, self.dim_hid)/np.sqrt(self.N)
        # self.W = X.T@V - np.outer(X.mean(0), V.sum(0))
        if self.center:
            self.b = X.mean(0)
        else:
            self.b = np.zeros(m)

        S = (X@self.W - self.b@self.W)/self.scl
        S = 1*(S > 0.5)

        self.S = 1.0*S #.todense()
        self.StS = 1.0*(S.T@S) #.todense()

    def EStep(self):

        XW = (self.data@self.W - self.b@self.W)
        StS = 1.0*self.StS

        if (self.tree_reg > 1e-6) or self.center:
            newS, StS = sparse_bca(XW, 1.0*self.S, StS, self.N, scl=self.scl,
                temp=self.temp, 
                alpha=self.sparse_reg, beta=self.tree_reg, 
                center=self.center)
        else:
            C = 2*(self.data@self.W - self.b@self.W) - self.scl - self.sparse_reg
            newS = 1*(np.random.rand(self.N, self.dim_hid) > spc.expit(C/self.temp))
            StS = newS.T@newS
        
        self.S = newS
        self.StS = StS

        if self.center:
            return newS - newS.mean(0)
        else:
            return newS

    def MStep(self, S):

        U,s,V = la.svd(self.data.T@S-np.outer(self.b,S.sum(0)), full_matrices=False)

        self.W = U@V
        self.scl = np.sum(s)/np.sum(S**2)

        if not self.center:
            self.b = self.data.mean(0) - self.scl*self.W@S.mean(0)

        nrm = np.sum(self.data**2) + self.N*self.b@(self.b - 2*self.data.mean(0))

        return self.scl*np.sqrt(np.sum(S**2))/np.sqrt(nrm)


############################################################
######### Sparse updates of S ##############################
############################################################
### These assume that the data X and the latents S are #####
### both sparse, in the compressed sparse row format #######
############################################################

# @njit
# def sprdot(Sind, Sptr, Xind, Xptr, Xval=None):
#   """
#   Sparse dot of binary S with sparse (and optionally binary) X
#   """

# @njit
# def sparse_reg(inds, ptr, val):

#   m = len(ptr)

#   data = []
#   indices = [] 
#   indptr = np.zeros(m+1)

#   for i in range(m):
#       for j in ptr[i:i+1]:
            

@njit
def sparse_bca(XW, S, StS, N, scl, alpha=0, beta=0, temp=0, center=False):
    """
    
    """

    n, m = S.shape

    St1 = np.diag(StS)

    for i in np.random.permutation(np.arange(n)):
    # en = np.zeros((n,m))
    # for i in np.arange(n): 

        ## Organize states
        if center or (beta>1e-6):
            St1 -= S[i]
        StS -= np.outer(S[i],S[i])

        ## Regularization (more verbose because of numba reasons)
        if beta > 1e-6:
            D1 = StS
            D2 = St1[None,:] - StS
            D3 = St1[:,None] - StS
            D4 = (N-1) - St1[None,:] - St1[:,None] + StS

            best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
            best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
            best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
            best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

            R = (best1 - best2 - best3 + best4)*1.0
            r = (best2.sum(0) - best4.sum(0))*1.0

        ## Hopfield update of s
        for j in np.random.permutation(np.arange(m)): # concept

            inp = (2*XW[i,j] - scl*(N - 1 - 2*St1[j])/N - alpha)
            ## Compute linear terms
            if beta > 1e-6:
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

            # en[i,j] = np.sum(lognorm(E)) - 2*np.sum(C*S)

        ## Update 
        # S[i] = news
        if center or (beta > 1e-6):
            St1 += S[i]
        StS += np.outer(S[i], S[i]) 

    return S, StS


@njit
def sparse_bmf(XW: np.ndarray, 
               S: np.ndarray, 
               WtW: np.ndarray, 
               StS: np.ndarray, 
               N: int, 
               temp: float,
               beta: float = 0.0):
    
    ## For now, all inputs are dense matrices, because trying to implement
    ## everything as sparse lineaar algebra is complicated and might not
    ## actually be that much more efficient, seing as we need to update S
    ## and StS on each iteration

    n,m = S.shape
    St1 = np.diag(StS)

    for i in np.random.permutation(np.arange(n)):
    # en = np.zeros((n,m))
    # for i in np.arange(n): 

        ## Organize states
        StS -= np.outer(S[i],S[i])
        if beta > 1e-6:
            St1 -= S[i]
            ## Regularization (more verbose because of numba reasons)
            D1 = StS
            D2 = St1[None,:] - StS
            D3 = St1[:,None] - StS
            D4 = (N-1) - St1[None,:] - St1[:,None] + StS
            
            best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
            best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
            best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
            best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

            R = (best1 - best2 - best3 + best4)*1.0
            r = (best2.sum(0) - best4.sum(0))*1.0

        ## Hopfield update of s
        for j in np.random.permutation(np.arange(m)): # concept
        # for j in np.arange(m): 

            ## Compute linear terms
            dot = np.dot(WtW[j],S[i]) + (0.5 - S[i,j])*WtW[j,j]

            if beta > 1e-6:
                inhib = np.dot(R[j],S[i]) + r[j]
            else:
                inhib = 0

            ## Compute currents
            curr = (XW[i,j] - beta*inhib - dot)/temp

            ## Apply sigmoid (overflow robust)
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            ## Update outputs
            S[i,j] = 1*(np.random.rand() < prob)

        ## Update 
        if beta > 1e-6:
            St1 += S[i]
        StS += np.outer(S[i], S[i]) 
        
    return S, StS

# @njit
# def sparse_bmf(XW, S, WtW, StS, N, beta=0):
#   """
#   S and WtW are lists of arrays

#   S is length 2 [indices, indptr] since it is binary
#   WtW is length 3 [indices, indptr, data]

#     Eventually it would be nice for StS to also be sparse, 
#     but that's a lot harder so will forget about it for now
#   """

#     inds, ptr = S
#     winds, wptr, wvals = WtW

#     m = len(wptr)
#     n = len(ptr)

#     St1 = np.diag(StS)

#     for i in np.random.permutation(np.arange(n)):
#     # en = np.zeros((n,m))
#     # for i in np.arange(n): 

#         if beta > 1e-6:
#             ## Organize states
#             s = inds[ptr[i]:ptr[i+1]]
#             St1[s] -= 1
#             StS[s][:,s] -= 1

#             ## Regularization (more verbose because of numba reasons)
#             D1 = StS
#             D2 = St1[None,:] - StS
#             D3 = St1[:,None] - StS
#             D4 = (N-1) - St1[None,:] - St1[:,None] + StS
            
#             best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
#             best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
#             best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
#             best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

#             R = (best1 - best2 - best3 + best4)*1.0
#             r = (best2.sum(0) - best4.sum(0))*1.0

#         ## Hopfield update of s
#         news = list(s)
#         for j in np.random.permutation(np.arange(m)): # concept

#             news.remove(j)

#             ## Compute linear terms
#             dot = 0
#             for k in range(wptr[j],wptr[j+1]):
#                 if winds[k] in news:
#                     dot += wvals[k]

#             if beta > 1e-6:
#                 inhib = np.sum(R[j,news]) + r[j]
#             else:
#                 inhib = 0

#             ## Compute currents
#             curr = (XW[i,j] - beta*inhib - dot)/temp

#             ## Apply sigmoid (overflow robust)
#             if curr < -100:
#                 prob = 0.0
#             elif curr > 100:
#                 prob = 1.0
#             else:
#                 prob = 1.0 / (1.0 + math.exp(-curr))

#             ## Update outputs
#             if (np.random.rand() < prob):
#                 news.append(j)

#         ## Update 
#         St1[news] += 1
#         StS[news][:,news] += 1

#         # care required for the pointers
#         dif = len(news) - len(s)
#         ptr[i+1:] += dif


#     return S #, en
