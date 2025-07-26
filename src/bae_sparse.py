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
from numba.typed import List
import math

# my code
import util
import bae_util
import bae_models

######################################################
###### Sparse search functions #######################
######################################################

# @njit
def spkerbmf(X: np.ndarray,
             S: list,                # list of index lists
             StX: np.ndarray, 
             StS: np.ndarray,
             scl: float,  
             temp: float,
             beta: float = 0.0):
    """
    Kernel matching search function for binary autoencoders
    """

    # n, m = S.shape
    m = len(StS)
    n, d = X.shape

    regularize = (beta > 1e-6)

    h = np.dot(StS, np.diag(StS)) - np.diag(StS)*np.sum(np.diag(StS)**2)

    # for i in np.random.permutation(np.arange(n)):
    for i in np.arange(n):

        # for j in np.random.permutation(np.arange(m)):
        for j in range(m): # concept
            # Sij = S[i,j]
            S_j = StS[j,j]

            ## Inputs
            inp = 0    
            for k in range(d):
                inp += 2*StX[j,k]*X[i,k]

            if j in S[i]:
                S[i].remove(j)

            ## Recurrence
            # dot = S_j*(1-S_j)*(1 - 2*Sij)
            dot = S_j*(1-S_j) - h[j]

            inhib = 0.0
            for k in S[i]:
                # Sik = S[i,k] 
                S_k = StS[k,k]

                dot += 2*(StS[j,k] - S_j*S_k)

                if regularize:
                    A = StS[j,k] 
                    B = S_j - A
                    C = S_k - A
                    
                    # Simple conditional assignment
                    if A < min(B,C):
                        inhib += 1
                    if B < min(A,C):
                        inhib -= 1
                    if C < min(A,B):
                        inhib -= 1

            ## Compute currents
            curr = (scl*inp - (scl**2)*dot - beta*inhib)/temp
            # curr = ((inp - scl*dot) - beta*inhib)/temp
            # curr = ((inp/scl - dot) - beta*inhib)/temp

            ## Apply sigmoid (overflow robust)
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))
            
            ## Update outputs
            # S[i,j] = 1*(np.random.rand() < prob)
            if np.random.rand() < prob:
                S[i].append(j)
    
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

    # for i in np.random.permutation(np.arange(n)):
    for i in np.arange(n):

        # for j in np.random.permutation(np.arange(m)):
        for j in range(m): # concept
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
            # curr = (scl*inp - (scl**2)*dot - beta*inhib)/temp
            # curr = (scl*(inp - scl*dot)/N - beta*inhib)/temp
            curr = ((inp - scl*dot)/N - beta*inhib)/temp
            # curr = ((inp/scl - dot)/N - beta*inhib)/temp

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

