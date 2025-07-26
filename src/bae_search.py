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
import bae_util

##################################################################
######### Search functions go here ###############################
##################################################################

#################################
######## Binary autoencoders ####
#################################

@njit
def bae(XW: np.ndarray, 
        S: np.ndarray, 
        WtW: np.ndarray, 
        temp: float,
        StS: Optional[np.ndarray]=None,
        beta: float = 0.0,
        alpha: float = 0.0):
    
    """
    Search function for the binary autoencoder
    """

    n,m = S.shape
    
    regularize = (beta > 1e-6)
    if not regularize:
        StS = np.eye(1) # need to do this for it to compile

    for i in np.random.permutation(np.arange(n)):
    # for i in range(n):

        for j in np.random.permutation(np.arange(m)):
        # for j in range(m):
                        
            dot = 0.5*WtW[j,j]
            inhib = 0
            for k in range(m):
                Sik = S[i,k]
                
                if k != j:
                    dot += WtW[j,k] * Sik
                
                if regularize:
                    A = StS[j,k] 
                    B = StS[j,j] - A 
                    C = StS[k,k] - A
                    D = 1 - A - B - C
                    
                    # Simple conditional assignment
                    if A < min(B,C,D):
                        inhib += Sik
                    if B < min(A,C,D):
                        inhib += 1 - Sik 
                    if C <= min(A,B,D):
                        inhib -= Sik
                    if D <= min(A,B,C):
                        inhib -= 1 - Sik

            curr = (XW[i,j] - dot - beta*inhib - alpha)/temp

            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            S[i,j] = 1*(np.random.rand() < prob)
        
    return S

@njit
def gebae(Xhat: np.ndarray, # same shape as X 
          XW: np.ndarray,   # same shape as S 
          S: np.ndarray, 
          W: np.ndarray,
          temp: float,
          lognorm: bae_util.gaussian,
          StS: Optional[np.ndarray]=None,
          beta: float = 0.0,
          alpha: float = 0.0):
    """
    Search function for the binary autoencoder
    """

    n,m = S.shape
    n,d = Xhat.shape
    
    regularize = (beta > 1e-6)
    if not regularize:
        StS = np.eye(1) # need to do this for it to compile

    for i in np.random.permutation(np.arange(n)):
    # for i in range(n):

        for j in np.random.permutation(np.arange(m)):
        # for j in range(m):
            
            Sij = S[i,j]

            ## Compute deltas
            dot = 0    
            for k in range(d):
                dot += lognorm(Xhat[i,k] + (1-Sij)*W[k,j])
                dot -= lognorm(Xhat[i,k] - Sij*W[k,j])

            inhib = 0
            if regularize:
                for k in range(m):

                    Sik = S[i,k] 
                    
                    A = StS[j,k] 
                    B = StS[j,j] - A 
                    C = StS[k,k] - A
                    D = 1 - A - B - C
                    
                    # Simple conditional assignment
                    if A < min(B,C,D):
                        inhib += Sik
                    if B < min(A,C,D):
                        inhib += 1 - Sik 
                    if C <= min(A,B,D):
                        inhib -= Sik
                    if D <= min(A,B,C):
                        inhib -= 1 - Sik

            curr = (XW[i,j] - dot - beta*inhib - alpha)/temp

            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            ## Update latents and natural parameters
            ds = 1*(np.random.rand() < prob) - Sij
            for k in range(d):
                Xhat[i,k] += ds*W[k,j]
            S[i,j] += ds
        
    return S

@njit
def kerbae(X: np.ndarray,
           S: np.ndarray, 
           StX: np.ndarray, 
           StS: np.ndarray,
           scl: float,  
           temp: float,
           beta: float = 0.0):
    """
    Kernel matching search function for binary autoencoders
    """

    n, m = S.shape
    n2, d = X.shape

    regularize = (beta > 1e-6)

    assert n == n2
    currs = np.zeros(S.shape)

    # for i in np.random.permutation(np.arange(n)):
    for i in np.arange(n):

        # for j in np.random.permutation(np.arange(m)):
        for j in range(m): # concept
            Sij = S[i,j]
            S_j = StS[j,j]

            ## Inputs
            inp = 0    
            for k in range(d):
                inp += 2*StX[j,k]*X[i,k]

            ## Recurrence
            dot = S_j*(1-S_j)*(1 - 2*Sij)
            inhib = 0.0
            for k in range(m):                        
                Sik = S[i,k] 
                S_k = StS[k,k]
                
                dot += 2*(StS[j,k] - S_j*S_k)*(Sik - S_k)

                if regularize:
                    A = StS[j,k] 
                    B = S_j - A
                    C = S_k - A
                    D = 1 - A - B - C
                    
                    # Simple conditional assignment
                    if A < min(B,C,D):
                        inhib += Sik
                    if B < min(A,C,D):
                        inhib += (1 - Sik) 
                    if C < min(A,B,D):
                        inhib -= Sik
                    if D < min(A,B,C):
                        inhib -= (1 - Sik)

            ## Compute currents
            # curr = (scl*inp - (scl**2)*dot - beta*inhib)/temp
            # curr = (scl*(inp - scl*dot)/N - beta*inhib)/temp
            curr = ((inp - scl*dot) - beta*inhib)/temp
            # curr = ((inp/scl - dot) - beta*inhib)/temp

            currs[i,j] = 1*curr

            ## Apply sigmoid (overflow robust)
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))
            
            # ## Update outputs
            # S[i,j] = 1*(np.random.rand() < prob)

            ## Update outputs
            ds = (np.random.rand() < prob) - Sij
            S[i,j] += ds

            ## Update covariance matrices
            StS[j,j] += ds
            for k in range(m):
                if k != j:
                    StS[j,k] += S[i,k]*ds/n
                    StS[k,j] += S[i,k]*ds/n

            for k in range(d):
                StX[j,k] += X[i,k]*ds/n
    
    return currs
    # return S
    # return allcurr 


#############################
######## Matrix factorization
#############################

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
         alpha: float = 0.0,
         beta: float = 0.0, 
         prior_logits: Optional[np.ndarray]=None,
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

    if prior_logits is None:    
        prior_logits = np.ones(m)

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
                    
            # curr = (scl*(2*XW[i,j] - scl) - alpha*prior_logits[j] - beta*inhib)/temp
            curr = (2*XW[i,j]/scl - 1 - alpha*prior_logits[j] - beta*inhib)/temp
            
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


# @njit
# def spkerbmf(X: np.ndarray,
#            S: np.ndarray, 
#            StX: np.ndarray, 
#            StS: np.ndarray,
#            N: int,
#            scl: float,  
#            temp: float,
#            alpha: float = 0.0,
#            beta: float = 0.0):
#     """
#     One batch gradient step on S

#     Assumes that X is centered *with respect to the full dataset* and that, 
#     if supplied, StS and StX are also computed for the full dataset. 

#     TODO: figure out a good sparse implementation?
#     """

#     n, m = S.shape
#     n2, d = X.shape

#     regularize = (beta > 1e-6)

#     currs = np.zeros(S.shape)
#     sign = 2*(np.diag(StS) < N//2)-1 # which concepts are flipped

#     assert n == n2
#     t = (N-1)/N

#     # for i in np.random.permutation(np.arange(n)):
#     for i in np.arange(n):

#         # for j in np.random.permutation(np.arange(m)):
#         for j in range(m): 
#             Sij = S[i,j]
#             S_j = (StS[j,j] - Sij)/(N-1)
#             # sign[j] = 2*(S_j < 0.5) - 1

#             # if sign[j] < 0: # take care of sign flips
#             #     S_j = 1 - S_j  # only flip the mean

#             ## Inputs
#             inp = 0 
#             for k in range(d):
#                 # inp += sign[j]*(2*StX[j,k]*X[i,k] + (1 - 2*Sij)*X[i,k]**2)/(t*(N-1))
#                 inp += (2*StX[j,k]*X[i,k] + (1 - 2*Sij)*X[i,k]**2)/(t*(N-1))
                
#             ## Recurrence
#             # dot = sign[j]*((N-2)*S_j*(1-S_j) + 1/2)*(1 - 2*Sij) / N
#             dot = ((N-2)*S_j*(1-S_j) + 1/2)*(1 - 2*Sij) / N
#             inhib = 0.0
#             for k in range(m):                        
#                 Sik = 1*S[i,k]
#                 S_k = (StS[k,k] - Sik)/(N-1)
#                 SjSk = (StS[j,k] - Sij*Sik)/(N-1) # raw second moment

#                 # if sign[j] < 0: # the order matters here
#                 #     SjSk = S_k - SjSk  
#                 # if sign[k] < 0:
#                 #     Sik = 1 - Sik # we can safely flip this
#                 #     S_k = 1 - S_k
#                 #     SjSk = S_j - SjSk 

#                 dot += 2*(SjSk + (1/2 - S_j - S_k - (N-2)*S_j*S_k)/N)*(Sik - S_k)
#                 dot -= (2*S_j-1)*(1-S_k)*S_k / N

#                 if regularize:
#                     A = SjSk
#                     B = S_j - SjSk
#                     C = S_k + Sik/(N-1) - SjSk
                    
#                     # Simple conditional assignment
#                     if A < min(B,C - 1/(N-1)):
#                         inhib += Sik
#                     if B < min(A,C):
#                         inhib += (1 - Sik) 
#                     if C <= min(A,B):
#                         inhib -= Sik

#             ## Compute currents
#             # curr = sign[j]*((inp - scl*dot) - beta*inhib - alpha)/temp
#             curr = ((inp - scl*dot) - beta*inhib - alpha)/temp
            
#             currs[i,j] = temp*curr

#             # ## Apply sigmoid (overflow robust)
#             # if curr < -100:
#             #     prob = 0.0
#             # elif curr > 100:
#             #     prob = 1.0
#             # else:
#             #     prob = 1.0 / (1.0 + math.exp(-curr))
            
#             # ## Update outputs
#             # ds = (np.random.rand() < prob) - Sij 
#             # S[i,j] += ds    # this is why we couldn't flip Sij

#             # ## Update covariance matrices
#             # StS[j,j] += ds
#             # for k in range(m):
#             #     if k != j:
#             #         StS[j,k] += S[i,k]*ds
#             #         StS[k,j] += S[i,k]*ds

#             # for k in range(d):
#             #     StX[j,k] += X[i,k]*ds
            
#             # sign[j] = 2*(StS[j,j] < N//2) - 1

#     return currs
#     # return S

@njit
def kerbmf(X: np.ndarray,
           S: np.ndarray, 
           StX: np.ndarray, 
           StS: np.ndarray,
           N: int,
           scl: float,  
           temp: float,
           alpha: float = 0.0,
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

    # currs = np.zeros(S.shape)

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
                
                dot += 2*(StS[j,k] - Sij*Sik + t*(1/2 - S_j - S_k - (N-2)*S_j*S_k))*(Sik - S_k)
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
            # curr = (scl*inp - (scl**2)*dot - beta*inhib)/temp
            # curr = (scl*(inp - scl*dot)/N - beta*inhib)/temp
            curr = ((inp - scl*dot)/(N-1) - beta*inhib - alpha)/temp
            # curr = ((inp/scl - dot)/N - beta*inhib)/temp

            # currs[i,j] = temp*curr

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
    # return currs