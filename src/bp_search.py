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


@njit
def gecbm(Xhat: np.ndarray, 	# same shape as X 
		  X: np.ndarray,		# inputs
		  W: np.ndarray,		# input weights
          Yhat: np.ndarray,   	# same shape as Y
          Y: np.ndarray, 		# outputs
          M: np.ndarray,		# output weights
          S: np.ndarray, 		# initial conditions
          temp: float,
          lognorm: bae_util.gaussian,
          StS: Optional[np.ndarray]=None,
          beta: float = 0.0,
          alpha: float = 0.0, 
          gamma: float = 0.1,
          ):
    """
    Search function for the binary autoencoder
    """

    n,m = S.shape
    n,dx = X.shape
    n,dy = Y.shape

    regularize = (beta > 1e-6)
    if not regularize:
        StS = np.eye(1) # need to do this for it to compile

    XM = np.dot(X, M.T)
    YW = np.dot(Y, W)

    for i in np.random.permutation(np.arange(n)):
    # for i in range(n):

        for j in np.random.permutation(np.arange(m)):
        # for j in range(m):
            
            Sij = S[i,j]

            ## Compute deltas
            dot = 0    
            for k in range(dx):
                dot += lognorm(Xhat[i,k] + (1-Sij)*M[k,j])
                dot -= lognorm(Xhat[i,k] - Sij*M[k,j])
            for k in range(dy):
                dot += gamma*lognorm(Yhat[i,k] + (1-Sij)*W[k,j])
                dot -= gamma*lognorm(Yhat[i,k] - Sij*W[k,j])

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

            curr = (XM[i,j] + gamma*YW[i,j] - dot - beta*inhib - alpha)/temp

            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            ## Update latents and natural parameters
            ds = 1*(np.random.rand() < prob) - Sij
            for k in range(dx):
                Xhat[i,k] += ds*M[k,j]
            for k in range(dy):
            	Yhat[i,k] += ds*W[k,j]
            S[i,j] += ds
        
    return S



@njit
def kercbm(X: np.ndarray,
		   StX: np.ndarray, 
		   Y: np.ndarray,
		   StY: np.ndarray,
           S: np.ndarray, 
           StS: np.ndarray,
           N: int,
           scl: float,  
           temp: float,
           gamma: float = 0.1,
           beta: float = 0.0):
    """
    One batch gradient step on S

    Assumes that X is centered *with respect to the full dataset* and that, 
    if supplied, StS and StX are also computed for the full dataset. 

    TODO: figure out a good sparse implementation?
    """

    n, m = S.shape
    n2, dx = X.shape
    n3, dy = Y.shape

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
            for k in range(dx):
                inp += (2*StX[j,k]*X[i,k] + (1-2*Sij)*X[i,k]**2)/t
            for k in range(dy):
                inp += gamma*(2*StY[j,k]*Y[i,k] + (1-2*Sij)*Y[i,k]**2)/t

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

            for k in range(dx):
                StX[j,k] += X[i,k]*ds
            for k in range(dy):
                StY[j,k] += Y[i,k]*ds
    
    return S
