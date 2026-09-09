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
         alpha: float = 0.0,
         inplace=True):
    
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
            
            dot = 0.5*(WtW[j,j])
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

            if regularize and inplace:
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
           J: Optional[np.ndarray]=None,
           alpha: float = 0.0,
           beta: float = 0.0,
           inplace: bool = True):
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
            # dot = 0.0
            dot = t*(2*(N-2)*S_j*(1-S_j) + 1) * (1/2 - Sij)
            inhib = 0.0
            for k in range(m):                        
                Sik = S[i,k] 
                S_k = (StS[k,k]-Sik)/(N-1)
                
                dot += 2*(StS[j,k] - Sij*Sik + t*(0.5 - S_j - S_k - (N-2)*S_j*S_k))*(Sik - S_k)
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

            if inplace:
                StS[j,j] += ds
                for k in range(m):
                    if k != j:
                        StS[j,k] += S[i,k]*ds
                        StS[k,j] += S[i,k]*ds

                for k in range(d):
                    StX[j,k] += X[i,k]*ds
    
    return S
    # return currs

@njit
def kerbmf3(X: np.ndarray,
           S: np.ndarray, 
           StX: np.ndarray, 
           StS: np.ndarray,
           N: int,
           scl: np.ndarray,
           temp: float,
           J: Optional[np.ndarray]=None,
           alpha: float = 0.0,
           beta: float = 0.0,
           inplace: bool = True):

    n, m = S.shape
    n2, d = X.shape

    regularize = (beta > 1e-6)

    assert n == n2
    t = (N-1)/N
    
    for i in np.random.permutation(np.arange(n)):
        for j in np.random.permutation(np.arange(m)):
            Sij = S[i,j] 
            S_j = (StS[j,j] - Sij)/(N-1)
            pij = scl[j] # Extract the scale for the current feature

            ## Inputs
            inp = 0    
            for k in range(d):
                inp += (2*StX[j,k]*X[i,k] + (1-2*Sij)*X[i,k]**2)/t

            ## Recurrence
            # Calculate the k=j self-interaction offset
            pre_dot = t*(2*(N-2)*S_j*(1-S_j) + 1) * (1/2 - Sij)
            
            # Initialize dot_sum weighted by wj for the self-interaction term
            dot_sum = pij * pre_dot 
            
            inhib = 0.0
            for k in range(m):                        
                Sik = S[i,k] 
                S_k = (StS[k,k]-Sik)/(N-1)
                pik = scl[k] # Extract scale for the interacting feature
                
                loop_dot = 2*(StS[j,k] - Sij*Sik + t*(0.5 - S_j - S_k - (N-2)*S_j*S_k))*(Sik - S_k)
                loop_dot -= t*(1-S_k)*S_k*(2*S_j-1)
                
                # Weight the cross-interaction by wk
                dot_sum += pik * loop_dot

                if regularize:
                    A = StS[j,k] - Sij*Sik
                    B = StS[j,j] - A - Sij
                    C = StS[k,k] - A
                    D = N - A - B - C
                    
                    if A < min(B,C-1,D):
                        inhib += Sik
                    if B < min(A,C,D-1):
                        inhib += (1 - Sik) 
                    if C <= min(A,B,D):
                        inhib -= Sik
                    if D <= min(A,B,C):
                        inhib -= (1 - Sik)

            ## Compute currents
            # Detailed balance restored: wj * inp and wj * dot_sum
            curr = (pij * (inp - dot_sum)/(N-1) - beta*inhib - alpha)/temp

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

            if inplace:
                StS[j,j] += ds
                for k in range(m):
                    if k != j:
                        StS[j,k] += S[i,k]*ds
                        StS[k,j] += S[i,k]*ds

                for k in range(d):
                    StX[j,k] += X[i,k]*ds
    
    return S

@njit
def kerbmf2(K: np.ndarray,
            S: np.ndarray,
            StS: np.ndarray,
            N: int,
            scl: float,  
            temp: float,
            alpha: float = 0.0,
            beta: float = 0.0,
            inplace: bool = True):
    """
    same as kermbf but with the kernel directly given as input, so it has
    n^2 complexity rather than nd 
    """

    n, m = S.shape
    n2 = len(K)

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
            inp = (1-2*Sij)*K[i,i]/t
            for k in range(n):
                inp += 2*K[i,k]*S[k,j]/t

            ## Recurrence
            # dot = 0.0
            dot = t*(2*(N-2)*S_j*(1-S_j) + 1) * (1/2 - Sij)
            inhib = 0.0
            for k in range(m):                        
                Sik = S[i,k] 
                S_k = (StS[k,k]-Sik)/(N-1)
                
                dot += 2*(StS[j,k] - Sij*Sik + t*(0.5 - S_j - S_k - (N-2)*S_j*S_k))*(Sik - S_k)
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

            if inplace:
                StS[j,j] += ds
                for k in range(m):
                    if k != j:
                        StS[j,k] += S[i,k]*ds
                        StS[k,j] += S[i,k]*ds

    return S
    # return currs

@njit
def convbmf(XW: np.ndarray, 
            S: np.ndarray,
            WtW: np.ndarray, 
            temp: float,
            # R: Optional[np.ndarray]=None,
            # StS: Optional[np.ndarray]=None,
            # N: Optional[int]=None,
            beta: float = 0.0,
            alpha: float = 0.0,
            l1_reg: float = 0.0,
            seq_reg: float = 1e-2,
            ):
    
    """
    SBMF sampler specialized for convolutional weights, 
    i.e. dealing explicitly with time

    WtW should be size (k,k,t)
    """

    n,m,T = S.shape
    
    L = (WtW.shape[2] + 1) // 2
    mid = L - 1

    # if R is None:
    #     R = np.eye(1)
    #     reg = False

    indices = np.arange(m * T)

    regularize = (beta > 1e-6) | (alpha > 1e-6)

    for i in np.random.permutation(np.arange(n)):
    # for i in range(n):

        for j in np.random.permutation(indices):
        # for j in range(m):
            
            c = j // T
            t = j % T

            Sij = S[i,c,t]
            
            t_start = max(0, t - (L - 1))
            t_end = min(T, t + L)

            dot = 0.5*(WtW[c,c,mid])
            inhib = 0
            for c_prime in range(m):

                for t_prime in range(t_start, t_end):

                    if (c == c_prime) and (t==t_prime):
                        continue

                    q_idx = mid + t_prime - t
                    Sik = S[i,c_prime,t_prime]
                    
                    dot += WtW[c,c_prime,q_idx] * Sik
                    
                    if regularize:
                        if (c_prime == c):
                            inhib += alpha*Sik
                        if (t_prime == t):
                            inhib += beta*Sik

            curr = (XW[i,c,t] - inhib - dot - l1_reg)/temp

            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            ds = (np.random.rand() < prob) - Sij

            # if regularize:
            #     StS[c,c,t] += ds
            #     for c_prime in range(m):
            #         if c_prime != c:
            #             StS[c,c_prime,t] += S[i,c_prime,t]*ds
            #             StS[c_prime,c,t] += S[i,c_prime,t]*ds

            S[i,c,t] += ds
            
    return S


@njit
def snmf(XW: np.ndarray, 
         S: np.ndarray, 
         Z: np.ndarray,
         WtW: np.ndarray, 
         temp: float, 
         sigma2: float = 1.0,
         tau: float = 1e-2, 
         StS: Optional[np.ndarray]=None, 
         N: int = None,
         beta: float = 0.0, 
         alpha: float = 0.0,
         z_pos: str = 'sample',
         inplace: bool =True,
         ):
    """Partially collapsed element-wise sampler for S and Z."""
    n, m = S.shape
    regularize = (beta > 1e-6)

    if not regularize:
        StS = np.eye(1) # need to do this for it to compile
        N = 1

    if z_pos == 'map':
        zfunc = bae_util.mode_trunc_norm
    elif z_pos == 'mean':
        zfunc = bae_util.mean_trunc_norm
    elif z_pos == 'sample':
        zfunc = bae_util.sample_trunc_norm
    else:
        raise ValueError('`z_pos` makes no sense')

    for i in np.random.permutation(np.arange(n)):
        for j in np.random.permutation(np.arange(m)):
            Sij = S[i,j]
            E_dot_Wj = XW[i,j]
            inhib = 0.0
            
            for k in range(m):
                if k != j:
                    if S[i,k] > 0:
                        E_dot_Wj -= WtW[j,k] * Z[i,k]
                
                if regularize:
                    Sik = S[i,k]
                    A = StS[j,k] - Sij*Sik
                    B = StS[j,j] - A - Sij
                    C = StS[k,k] - A
                    D = N - A - B - C
                    
                    if A < min(B,C-1,D): inhib += Sik
                    if B < min(A,C,D-1): inhib += 1 - Sik 
                    if C <= min(A,B,D): inhib -= Sik
                    if D <= min(A,B,C): inhib -= 1 - Sik

            W_norm_sq = WtW[j,j]
            mu = (E_dot_Wj - tau * sigma2) / W_norm_sq
            nu = math.sqrt(sigma2 / W_norm_sq)
            
            Lambda_base = math.log(tau * nu * math.sqrt(2 * math.pi)) + (mu**2)/(2 * (sigma2/W_norm_sq)) + bae_util.log_ndtr(mu/nu)
            curr = (Lambda_base - beta*inhib - alpha) / temp
            
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            new_Sij = int(np.random.rand() < prob)
            
            if new_Sij == 1:
                Z[i,j] = zfunc(mu, nu)
            else:
                Z[i,j] = 0.0 

            if regularize and new_Sij != Sij and inplace:
                ds = new_Sij - Sij
                StS[j,j] += ds
                for k in range(m):
                    if k != j:
                        StS[j,k] += S[i,k] * ds
                        StS[k,j] = StS[j,k]

            S[i,j] = new_Sij
        
    return S, Z

###################################################
####### Accelerated samplers ######################
###################################################

@njit
def gauss_bern(X: np.ndarray,
               S: np.ndarray, 
               # StX: np.ndarray, 
               # StS: np.ndarray,
               Cov: np.ndarray,
               Mn: np.ndarray,
               sigma_x: float,  
               alpha: float = 0.0,
               temp: float = 1.0,
               ):
    """ 
    One step of accelerated gibbs sampling for the linear gaussian
    model with gaussian prior on weights/features 
    (Doshi-Velez and Ghahramani ICML 2009)
    """

    n, m = S.shape
    n2, d = X.shape

    assert n == n2

    StX = np.dot(S.T, X)
    s_ = np.dot(S.T, np.ones(n)) / n 

    # currs = np.zeros(S.shape)
    for i in np.random.permutation(np.arange(n)):
    # for i in np.arange(n):

        ## Update posterior of W removing item i
        CSi = np.dot(Cov, S[i])
        denom = (sigma_x**2 - np.dot(S[i], CSi))
        Cov  += np.outer(CSi, CSi) / denom
        Mn += np.outer(CSi, np.dot(CSi, StX/sigma_x**2))/denom 
        Mn -= np.dot(S[i], CSi)*np.outer(CSi, X[i]/sigma_x**2)/denom
        Mn -= np.outer(CSi, X[i])/sigma_x**2

        ## Precompute some things
        CSi = np.dot(Cov, S[i])
        mui = np.dot(Mn.T, S[i])
        ci = np.dot(S[i], CSi)

        for j in np.random.permutation(np.arange(m)):
        # for j in range(m): # concept
            Sij = S[i,j] 
            
            ## Means if Sij is 1 or 0
            mui_1 = mui + (1-Sij)*Mn[j] # O(d)
            ci_1 = ci + (1-Sij)*(2*CSi[j] + Cov[j,j])
            mui_0 = mui - Sij*Mn[j]     # O(d)
            ci_0 = ci - Sij*(2*CSi[j] - Cov[j,j])

            ## Likelihood
            dlik = 0 
            dlik -= np.sum((X[i] - mui_1)**2)/(sigma_x**2 + ci_1)
            dlik -= d*np.log(sigma_x**2 + ci_1)
            dlik += np.sum((X[i] - mui_0)**2)/(sigma_x**2 + ci_0)
            dlik += d*np.log(sigma_x**2 + ci_0)

            ## Compute posterior
            # curr = (0.5*dlik + np.log(s_[j]) - np.log(1-s_[j]))/temp
            curr = (0.5*dlik - alpha)/temp

            # currs[i,j] = 1*dlik

            ## Apply sigmoid 
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))
            
            ## Update outputs
            newSij = 1*(np.random.rand() < prob)
            ds = newSij - Sij
            if newSij:
                S[i,j] = 1 
                mui = 1*mui_1
                ci = 1*ci_1
            else:
                S[i,j] = 0
                mui = 1*mui_0
                ci = 1*ci_0

            ## Might as well do this while we're here?
            s_[j] += ds/n
            for k in range(d):
                StX[j,k] += X[i,k]*ds

        ## Update posterior of W
        CSi = np.dot(Cov, S[i]) 
        denom = (sigma_x**2 + np.dot(S[i], CSi))
        Cov -= np.outer(CSi, CSi) / denom
        Mn -= np.outer(CSi, np.dot(CSi, StX/sigma_x**2))/denom 
        Mn += np.outer(CSi, X[i])/sigma_x**2
    
    return S

