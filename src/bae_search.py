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
# from tqdm import tqdm

import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import numpy.linalg as nla
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

torch._dynamo.config.suppress_errors = True 

from numba import njit
import math

# my code
import util


##################################################################
######### Search functions go here ###############################
##################################################################

@njit
def sbmf(XW: np.ndarray, 
         S: np.ndarray, 
         WtW: np.ndarray, 
         temp: float,
         StS: Optional[np.ndarray]=None, 
         N: Optional[int]=None,
         beta: float = 0.0,
         alpha: float = 0.0):
    
    """
    One iteration of Semi Binary Matrix Factorization

    In order for this to be batched, this function takes certain outer
    product matrices (i.e. StS) as inputs, as well as the full dataset
    size N which may be larger (but not smaller) than len(S)
    """

    n,m = S.shape
    
    regularize = (beta > 1e-6)
    if not regularize:
        StS = np.eye(1) # need to do this for it to compile
        N = 1

    for i in np.random.permutation(np.arange(n)):
    # for i in range(n):

        for j in np.random.permutation(np.arange(m)):
        # for j in range(m):
            
            Sij = S[i,j]
            
            dot = 0.5*WtW[j,j]
            inhib = 0
            for k in range(m):
                Sik = S[i,k]
                
                if k != j:
                    dot += WtW[j,k] * Sik
                
                if regularize:
                    A = StS[j,k] - Sij*Sik
                    B = StS[j,j] - A - Sij
                    C = StS[k,k] - A
                    D = N - A - B - C
                    
                    # Simple conditional assignment
                    if A < min(B,C-1,D):
                        inhib += Sik
                    if B < min(A,C,D-1):
                        inhib += 1 - Sik 
                    if C <= min(A,B,D):
                        inhib -= Sik
                    if D <= min(A,B,C):
                        inhib -= 1 - Sik

            curr = (XW[i,j] - beta*inhib - dot - alpha)/temp

            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            ds = (np.random.rand() < prob) - Sij

            if regularize:
                StS[j,j] += ds
                for k in range(m):
                    if k != j:
                        StS[j,k] += S[i,k]*ds
                        StS[k,j] += S[i,k]*ds

            S[i,j] += ds
        
    return S


@njit
def bpca(XW: np.ndarray, 
         S: np.ndarray, 
         scl: float, 
         temp: float,
         StS: Optional[np.ndarray]=None, 
         N: Optional[int]=None,
         alpha: float = 1e-2,
         beta: float = 0.0, 
         steps=1):
    """
    One gradient step on S

    TODO: figure out a good sparse implementation!
    """

    n,m = S.shape
    
    regularize = (beta > 1e-6)
    if not regularize:
        StS = np.eye(1) # need to do this for it to compile
        N = 1

    for i in np.random.permutation(np.arange(n)):
    # for i in range(n):

        for j in np.random.permutation(np.arange(m)):
        # for j in range(m):
            
            Sij = S[i,j]
            
            inhib = 0
            if regularize:
                for k in range(m):
                    Sik = S[i,k]
                    
                    A = StS[j,k] - Sij*Sik
                    B = StS[j,j] - A - Sij
                    C = StS[k,k] - A
                    D = N - A - B - C
                    
                    ## Simple conditional assignment
                    if A < min(B,C-1,D):
                        inhib += Sik
                    if B < min(A,C,D-1):
                        inhib += 1 - Sik 
                    if C <= min(A,B,D):
                        inhib -= Sik
                    if D <= min(A,B,C):
                        inhib -= 1 - Sik
                    
            curr = (2*XW[i,j] - beta*inhib - scl - alpha)/temp

            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            ds = (np.random.rand() < prob) - Sij

            if regularize:
                StS[j,j] += ds
                for k in range(m):
                    if k != j:
                        StS[j,k] += S[i,k]*ds
                        StS[k,j] += S[i,k]*ds

            S[i,j] += ds
        
    return S


@njit
def kerbmf(X: np.ndarray,
           S: np.ndarray, 
           StX: np.ndarray, 
           StS: np.ndarray,
           N: int,
           scl: float,  
           temp: float,
           beta: float = 0.0):
    """
    One batch gradient step on S

    Assumes that X is centered *with respect to the full dataset* and that, 
    if supplied, StS and StX are also computed for the full dataset. 

    TODO: figure out a good sparse implementation?
    """

    n, m = S.shape
    n2, d = X.shape

    regularize = (beta > 1e-6)

    assert n == n2
    t = (N-1)/N

    for i in np.random.permutation(np.arange(n)):
    # for i in np.arange(n):

        for j in np.random.permutation(np.arange(m)):
        # for j in range(m): # concept
            Sij = S[i,j]
            S_j = (StS[j,j] - Sij)/(N-1)

            ## Inputs
            inp = 0    
            for k in range(d):
                inp += (2*StX[j,k]*X[i,k] + (1-2*Sij)*X[i,k]**2)/t

            ## Recurrence
            dot = t*(2*(N-2)*S_j*(1-S_j) + 1)*(1/2 - Sij)
            inhib = 0.0
            for k in range(m):                        
                Sik = S[i,k] 
                S_k = (StS[k,k]-Sik)/(N-1)
                
                dot += (2*(StS[j,k] - Sij*Sik) + t)*(Sik - S_k)
                dot -= 2*t*((N-2)*S_j*S_k + S_j + S_k)*(Sik - S_k)
                dot -= t*(1-S_k)*S_k*(2*S_j-1)

                if regularize:
                    A = StS[j,k] - Sij*Sik
                    B = StS[j,j] - A - Sij
                    C = StS[k,k] - A
                    D = N - A - B - C
                    
                    # Simple conditional assignment
                    if A < min(B,C-1,D):
                        inhib += Sik
                    if B < min(A,C,D-1):
                        inhib += (1 - Sik) 
                    if C <= min(A,B,D):
                        inhib -= Sik
                    if D <= min(A,B,C):
                        inhib -= (1 - Sik)

            ## Compute currents
            curr = (scl*inp - (scl**2)*(dot + beta*inhib))/temp

            ## Apply sigmoid (overflow robust)
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))
            
            ## Update outputs
            ds = (np.random.rand() < prob) - Sij
            S[i,j] += ds

            StS[j,j] += ds
            for k in range(m):
                if k != j:
                    StS[j,k] += S[i,k]*ds
                    StS[k,j] += S[i,k]*ds

            for k in range(d):
                StX[j,k] += X[i,k]*ds
    
    return S


# @njit(cache=True)
# def kerbmf(S: np.ndarray, 
#          S: np.ndarray,
#          StX: np.ndarray, 
#          StS: np.ndarray,
#            N: Optional[int]=None,
#            scl: float, 
#            beta: float, 
#            temp: float, 
#            steps: Optional[int]=1) -> np.ndarray:
#     """
#     One batch gradient step on S

#     Assumes that X is centered *with respect to the full dataset* and that, 
#     if supplied, StS and StX are also computed for the full dataset. 

#     TODO: figure out a good sparse implementation?
#     """

#     n, m = S.shape

#     for i in np.random.permutation(np.arange(n)):
#     # for i in np.arange(n):

#         ## Pick current item
#         t = (N-1)/N

#         x = X[i]
#         s = S[i]

#         ## Subtract current item
#         St1 -= s
#         StS -= np.outer(s,s)
#         StX -= np.outer(s,x)

#         ## Organize states
#         xtx = np.sum(x**2)
#         Sk = (np.dot(StX, x) + St1*xtx/(N-1))/t
#         k0 = xtx/(t**2)

#         ## Recurrence
#         # Compute the rank-one terms
#         s_ = St1/(N-1)
#         u = 2*s_ - 1

#         s_sc_ = s_*(1-s_)

#         ## Constants
#         # sx = Sx.sum()
#         sx = np.dot(s_, s - s_)
#         ux = 2*sx - s.sum() + s_.sum()
        
#         # Form the threshold 
#         h = t*((scl**2)*s_sc_.sum() - scl*k0)*u + 2*scl*Sk 
        
#         # Need to subtract the diagonal and add it back in
#         Jii = 2*(N-1)*s_sc_ + t*u**2

#         ## Hopfield update of s
#         for j in np.random.permutation(np.arange(m)):
#         # for j in range(m): # concept
#           Sij = S[i,j]

#             dot = t*u[j]*ux - 2*s_[j]*sx - Jii[j]*Sij
#             for k in range(m):
#                 Sik = S[i,k]
                
#                 dot += 2 * StS[j,k] * (Sik - s_[k])
                
#                 if regularize:
#                     A = StS[j,k] - Sij*Sik
#                     B = StS[k,k] - A - Sij
#                     C = StS[j,j] - A
#                     D = n - A - B - C
                    
#                     # Simple conditional assignment
#                     if (A < min(B, C, D)) or (D < min(A,B,C)):
#                         inhib += Sik
#                     elif (B < min(A,B,D)):
#                         inhib += 1 - Sik
#                     elif C < min(A,B,D):
#                         inhib -= 1 + Sik

#             # Compute sparse dot product
#             dot = 2*np.dot(StS[j], news - s_)
#             dot -= 2*(N-1)*s_[j]*sx
#             dot += t*u[j]*ux
#             dot -= Jii[j]*news[j]

#             ## Compute currents
#             curr = (h[j] - (scl**2)*Jii[j]/2 - (scl**2)*dot)/temp

#             ## Apply sigmoid (overflow robust)
#             if curr < -100:
#                 prob = 0.0
#             elif curr > 100:
#                 prob = 1.0
#             else:
#                 prob = 1.0 / (1.0 + math.exp(-curr))
            
#             ## Update outputs
#             sj = 1*(np.random.rand() < prob)
#             ds = sj - news[j]
#             news[j] = sj
            
#             ## Update dot products
#             if np.abs(ds) > 0:
#                 sx += ds*s_[j]
#                 ux += ds*u[j]

#         ## Update 
#         S[i] = news
#         St1 += news
#         StS += np.outer(news, news)
#         StX += np.outer(news, x)
    
#     return S
