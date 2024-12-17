
import os, sys, re
import pickle
from time import time
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
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

from numba import njit
import math

# my code
import util
import df_util 

#############################################################################
############### Binary autoencoder with recurrent updates ###################
#############################################################################

class KernelBAE:
    
    def __init__(self, X, dim_hid, penalty=1e-2, rank=None, pvar=1, 
        steps=1, alpha=2, beta=5, max_ctx=None, fix_scale=True):
        
        self.n = len(X)
        self.r = dim_hid
        self.initialized = False

        self.scl = 1
        self.steps = steps
        self.beta = penalty
        self.max_ctx = max_ctx

        U,s,V = la.svd(X-X.mean(0), full_matrices=False)        
        
        self.frac_var = np.cumsum(s**2)/np.sum(s**2)
        if rank is None:
            r = np.sum(self.frac_var <= pvar + 1e-6)
        else:
            r = rank

        # self.X = X
        self.X = U[:,:r]@np.diag(s[:r])@V[:r]
        self.U = U[:,s>1e-6]
        self.U = self.U[:,:r]
        self.K = X@X.T
        
        ## Initialize S
        coding_level = np.random.beta(alpha, beta, dim_hid)/2
        num_active = np.floor(coding_level*len(X)).astype(int)
        
        Mx = self.X@np.random.randn(len(X.T),self.r)
        thr = -np.sort(-Mx, axis=0)[num_active, np.arange(dim_hid)]
        self.S = sprs.csc_array(1*(Mx >= thr))
        self.notS = sprs.csc_array(1*(Mx < thr))

        self.sum_s = self.S.sum(0)

        ## Penalty matrix
        self.StS = self.S.T@self.S
        # St1 = self.sum_s[None,:]

        # self.I1 = 2*StS - St1 
        # self.I2 = 2*Sts - St1.T 
        # self.I3 = St1 - St1.T 
        # self.I4 = St1 + St1.T - self.n 

    def grad_step(self, pause_schedule=False, **opt_args):
        
        if not self.initialized:
            self.schedule = Neal(**opt_args)
            self.initialized = True
        
        T = self.schedule(freeze=pause_schedule)
        
        for i in np.random.permutation(np.arange(self.n)):
            
            idx = np.mod(np.arange(self.n)+i, self.n) ## item i goes first
            
            olds = self.S[[i]].todense().squeeze()
            # news = self.recur(i, T, I=idx[1:self.max_ctx])
            news = self.slow_recur(i, T, I=idx[1:self.max_ctx])

            ## Update 
            self.sum_s += (news - olds)
            
            self.S[[i]] = news

            self.StS += np.outer(news,news) - np.outer(olds,olds)

            ## Keep in sparsest form
            flip = (self.sum_s > (self.n//2))

            oldS = self.S[:,flip]*1
            self.S[:,flip] = self.notS[:,flip]*1 
            self.notS[:,flip] = oldS

            self.sum_s[flip] = self.n - self.sum_s[flip]

        
    def init_optimizer(self, **opt_args):
        self.schedule = Neal(**opt_args)
        self.initialized = True

    def proj(self, rank=None, pvar=1):
        """
        Project S onto the span of the data
        Can truncate the basis to a particular rank 
        or percentage of variance explained
        """

        if rank is None:
            r = np.sum(self.frac_var <= pvar) 
        else:
            rank = r

        P = self.U[:,:r]

        S = self.S.tocsc()
        C = 2*(P@(P.T@self.S) + self.S.mean(0)) - 1
        self.S = sprs.lil_array(1*(C>0))
        self.sum_s = (C>0).sum(0)

    def energy(self, I=None):
        """
        Compute the energy of the network, for a subset I
        """
        
        if I is None:
            I = np.arange(self.n)
        
        X = self.X[I]
        S = self.S[I]
        K = X@X.T
        dot = self.scl*np.sum(util.center(K)*(S@S.T))
        Qnrm = (self.scl**2)*np.sum(util.center((S@S.T).todense())**2)
        Knrm = np.sum(util.center(K)**2)
        
        return Qnrm + Knrm - 2*dot

    def scaleX(self):
        """
        Optimally scale X
        """
        
        X_ = self.X - self.X.mean(0)
        S = self.S.tocsr()
        
        dot = np.sum((X_@X_.T)*(S@S.T))
        nrm = np.sum((X_@X_.T)**2)
        
        return np.sqrt(dot/nrm)
    
    def scaleS(self):
        """
        Optimally scale S
        """
        
        X = self.X
        S = self.S.tocsr()
        S_ = S - S.mean(0)
        
        dot = np.sum((X@X.T)*(S_@S_.T))
        nrm = np.sum((S_@S_.T)**2)
        
        return dot/nrm

    def recur(self, i, temp, I=None):
        """
        match S(I)*S(i) to X(I)*X(i) by updating s(i)
        """

        if I is None:
            I = np.mod(np.arange(self.n)+i, self.n)[1:]
        
        n = len(I)

        ## Organize data
        X = self.X[I]
        x = self.X[i]
        x_ = X.mean(0)
        
        # k = (X-x_)@(x-x_)
        # k0 = (x-x_)@(x-x_)
        K_ = self.K[I][:,I].mean(0)
        K__ = K_.mean()
        k_ = self.K[I,i].mean()
        k = self.K[I,i] - K_ - k_ + K__
        k0 = self.K[i,i] - 2*k_ + K__
        
        ## Organize states
        S = self.S[I].tocsc()         # i.e. S(X)
    
        ## Recurrent pass
        s0 = self.S[[i]].todense().squeeze()

        s = hopfield_loop(n, s0, S.indptr, S.indices, 
            k0, S.T@k, temp, self.scl, self.steps, beta=self.beta)

        return s

    def slow_recur(self, i, temp, I=None, debug=False):
        """
        For debugging/workshopping purposes

        Basically the pseudocode from the paper written literally, 
        which is much slower but easier to read and change
        """

        if I is None:
            I = np.mod(np.arange(self.n)+i, self.n)[1:]

        n = len(I)
            
        ## Organize data
        # X = self.X[I]
        # x = self.X[i]
        # x_ = X.mean(0)
        
        # k = (X-x_)@(x-x_)
        # k0 = (x-x_)@(x-x_)
        K_ = self.K[I][:,I].mean(0)
        K__ = K_.mean()
        k_ = self.K[I,i].mean()
        k = self.K[I,i] - K_ - k_ + K__
        k0 = self.K[i,i] - 2*k_ + K__
        
        ## Organize states
        S = self.S[I].tocsc()         # i.e. S(X)
        s = self.S[[i]].todense().squeeze()

        ## Energy
        # StS = self.StS - np.outer(s,s)
        # S1 = (self.sum_s - s)[None,:]
        StS = S.T@S
        S1 = S.sum(0)[None,:]
        scl = self.scl

        s_ = S1.squeeze()/n
        u = 2*s_ - 1
        t = (n/(n+1))
        J = (scl**2)*(2*StS - 2*n*np.outer(s_,s_) + t*np.outer(u,u))
        h = J@s_ + t*scl*(scl*(1-s_)@s_ - k0)*u + 2*scl*S.T@k
        Ji = np.diag(J)
        J = J*(1-np.eye(self.r))

        ## Regularization
        I1 = np.sign(2*StS - S1)
        I2 = np.sign(2*StS - S1.T)
        I3 = np.sign(S1 - S1.T)
        I4 = np.sign(S1 + S1.T - n)

        Q = (1*(I1<0)*(I2<0) - 1*(I1>0)*(I3<0) - 1*(I2>0)*(I3>0))*(I4<0)
        p = ((I2>0)*(I3>0)*(I4<0)).sum(1)

        A = J + self.beta*(scl**2)*Q
        b = h - 0.5*Ji - self.beta*(scl**2)*p

        ## Dynamics
        # Qs = Q@s
        # Js = J@s
        for i in range(self.r):

            # inhib = Qs[i] + p[i]
            # current = h[i] - 0.5*Ji[i] - Js[i]
            # inhib = Q[i]@s + p[i]
            # current = h[i] - 0.5*Ji[i] - J[i]@s 
            current = b[i] - A[i]@s

            # prob = spc.expit((current - self.beta*inhib)/temp)
            prob = spc.expit(current/temp)
            s[i] = np.random.binomial(1, prob)
            # ds = np.random.binomial(1, prob) - s[i]

            # s[i] += ds
            # Qs += Q[:,i]*ds
            # Js += J[:,i]*ds

        return s

class BAE:

    def __init__(self, X, dim_hid, rank=None, pvar=1,
        weight_class='ortho', weight_alg='svd', lr=1e-2,
        penalty=0, alpha=2, beta=5, fix_scale=False):

        self.n, _ = X.shape
        self.r = dim_hid
        self.initialized = False
        self.beta = penalty
        self.fix_scl = fix_scale

        self.weight_class = weight_class
        self.weight_alg = weight_alg
        self.lr = lr

        Ux,sx,Vx = la.svd(X-X.mean(0), full_matrices=False)        
        
        self.frac_var = np.cumsum(sx**2)/np.sum(sx**2)
        if rank is None:
            r = np.sum(self.frac_var <= pvar)+1 
        else:
            r = rank
        self.d = r

        # self.X = X - X.mean(0, keepdims=True)
        self.X = Ux[:,:r]@np.diag(sx[:r])
        self.U = Ux[:,sx>1e-6]
        self.U = self.U[:,:r]
        self.trXX = np.sum(sx[:r]**2)

        ## Initialize S 
        coding_level = np.random.beta(alpha, beta, dim_hid)/2
        num_active = np.floor(coding_level*len(X)).astype(int)
        
        Mx = (self.X)@np.random.randn(self.d,self.r)
        thr = -np.sort(-Mx, axis=0)[num_active, np.arange(dim_hid)]
        self.S = sprs.csr_array(1*(Mx >= thr))
        self.trSS = (self.S.T@self.S).trace()

        ## Initialise b
        self.b = np.zeros(self.d) # -X.mean(0)

        ## Initialize W
        if self.weight_class == 'ortho':
            U,s,V = la.svd(self.X.T@self.S, full_matrices=False)
            self.W = U@V
            if not self.fix_scl:
                self.scl = np.sum(s)/self.trSS
            else:
                self.scl = 1
        elif self.weight_class == 'real':
            self.W = (la.pinv(self.S)@self.X).T

        ## Regularization
        self.StS = self.S.T@self.S
        self.Ssum = self.S.sum(0)

    def grad_step(self, pause_schedule=False, **opt_args):

        if not self.initialized:
            self.schedule = Neal(**opt_args)
            self.initialized = True
        
        T = self.schedule(freeze=pause_schedule)

        # self.updateS(T)
        ES = self.updateS_reg(T) ## E step
        self.updateWb(ES)    ## M step
        # self.updateWb(self.S)

    def updateS(self, temp):

        C = (self.U@(self.U.T@self.S) + self.S.mean(0) - 0.5)
        current = self.beta*C + 2*(self.X - self.b)@self.W - self.scl
        prb = spc.expit(current/temp)
        self.S = sprs.csr_array(np.random.binomial(1, prb))
        self.trSS = (self.S.T@self.S).trace()

    def updateWb(self, ES):

        # eps = np.random.randn(self.n, self.d)*temp/10
        ## update weights
        if self.weight_class == 'ortho':

            if self.weight_alg == 'svd':
                # U,s,V = la.svd((self.X-self.b).T@self.S, full_matrices=False)
                U,s,V = la.svd((self.X-self.b).T@ES, full_matrices=False)
                self.W = U@V
                if not self.fix_scl:
                    # self.scl = np.sum(s)/self.trSS
                    self.scl = np.sum(s)/ES.sum()
            else:
                dW = (self.X-self.b).T@self.S
                self.W += self.lr*dW

        elif self.weight_class == 'real':
            self.W = (la.pinv(self.S)@(self.X - self.b)).T

        ## update biases
        self.b = (self.X - self.scl*ES@self.W.T).mean(0)

    def updateS_reg(self, temp):
        """
        Regularised update of S
        """

        StS = 1*self.StS
        Ssum = 1*self.Ssum
        C = 2*(self.X - self.b)@self.W - self.scl
        ES = np.zeros(C.shape) ## keep track of the expectation

        for it in np.random.permutation(range(self.n)):

            ## Organize states
            s = self.S[[it]].todense().squeeze()
            n = self.n-1

            ## Energy
            StS -= np.outer(s,s)
            Ssum -= s

            ## Regularization
            D1 = StS
            D2 = Ssum[None,:] - StS
            D3 = Ssum[:,None] - StS
            D4 = n - Ssum[None,:] - Ssum[:,None] + StS

            deez = np.array([D1,D2,D3,D4])
            best = 1*(deez == deez.min(0))
            best *= (best.sum(0)==1)

            Q = best[0] - best[1] - best[2] + best[3]
            p = (best[1].sum(0) - best[3].sum(0))
            # I1 = np.sign(2*StS - S1[None,:]) # si'sj - (1-si)'sj
            # I2 = np.sign(2*StS - S1[:,None]) # si'sj - si'(1-sj)
            # I3 = np.sign(S1[None,:] - S1[:,None]) # (1-si)'sj - si'(1-sj)
            # I4 = np.sign(S1[None,:] + S1[:,None] - n) # si'sj - (1-si)'(1-sj)
            # I5 = np.sign()

            # Q = (1*(I1<0)*(I2<0) - 1*(I1>0)*(I3<0) - 1*(I2>0)*(I3>0))*(I4<0)
            # p = ((I2>0)*(I3>0)*(I4<0)).sum(1)

            ## Dynamics 
            for i in np.random.permutation(range(self.r)):

                inhib = Q[i]@s + p[i]

                ES[it,i] = spc.expit((C[it,i] - self.beta*inhib)/temp)
                s[i] = np.random.binomial(1, ES[it,i])

            StS += np.outer(s,s)
            Ssum += s

            self.S[[it]] = s

        self.trSS = np.trace(StS)
        self.StS = StS
        self.Ssum = Ssum

        return ES

            ## maintain sparsity
            # flip = (S1 + s) > self.n//2
            # StS[flip] = S1[None,:] - StS[flip]
            # S1[flip] = self.n - S1[flip]
            # StS[:,flip] = S1[:,None] - StS[:,flip]

            # s[flip] = 1-s[flip]

    def energy(self):
        return np.sum((self.scl*self.S@self.W.T - self.X + self.b)**2)
        # bb = self.n*self.b@self.b
        # return 1 - self.scl*np.sqrt(self.trSS)/np.sqrt(self.trXX + bb)

    def init_optimizer(self, **opt_args):
        self.schedule = Neal(**opt_args)
        self.initialized = True


@dataclass
class Neal:
    """
    Annealing scheduler
    """
    
    decay_rate: float = 0.8
    period: int = 2
    initial: float = 10.0
    
    def __post_init__(self):
        self.t = 0
    
    def __call__(self, freeze=False):
        T =  self.initial*(self.decay_rate**(self.t//self.period))
        if not freeze:
            self.t += 1
        return T


#############################################################################
################## jit-ed functions #########################################
#############################################################################

@njit
def hopfield_loop(n, x, indptr, indices, k0, Sk, temp, scl=1, steps=1, beta=0):
    """
    Sparse binary + low-rank hopfield updates
    
    x <- sigmoid[h - J(x-x_)]
    
    where J = (2S'S + aa')*(1 - I) i.e. a Hebbian + rank-one term with 
    the diagonal subtracted to ensure convergence (and added back in h)
    
    Inputs:
    	n [int]: number of rows
        x [(dim,)-array]: vector to be MVMd
        indptr [(dim+1,)-array]: CSC style column index pointers
        indices [(nnz,)-array]: CSC style row indices
        k0 [float]: norm of new input
        Sk [(dim,)-array]: input to network
        temp [float]: temperature parameter
        scl [float]: scale term
        steps [int]: how many asynchronous passes
        
    Output:
        x [(dim,)-array]: output vector
    """

    ## TODO: keep track of energy to avoid re-computing
    
    ## It seems that the @njit decorator lets you use for
    ## loops in python, but you can only use numpy and 
    ## for some reason it is much faster when written flat
    
    m = len(indptr)-1     # number of columns  (categories)
    # n = np.max(indices)     # number of rows  (inputs)
    
    assert m == len(x)
    
    # Compute the first dot product
    Sx = np.zeros(n, dtype=float)
    s = np.zeros(m, dtype=float)
    for i in range(m):
        rows = indices[indptr[i]:indptr[i+1]]
        s[i] = len(rows)
        for j in rows:
            Sx[j] += (x[i] - s[i]/n)
    t = n/(n+1)
    
    # Compute the rank-one terms
    s_ = s/n
    u = 2*s_ - 1
    
    s_sc_ = s_*(1-s_)
    
    sx = Sx.sum()
    ux = 2*sx/n - x.sum() + s_.sum()
    
    # Form the threshold 
    h = t*((scl**2)*s_sc_.sum() - scl*k0)*u + 2*scl*Sk
    
    # Need to subtract the diagonal and add it back in
    Jii = 2*n*s_sc_ + t*u**2
    # Jii = 2*s*(1-s_) + t*u**2
    
    xout = 1*x
    for step in range(steps):
        for i in range(m):
            rows = indices[indptr[i]:indptr[i+1]]
            
            ## Compute sparse dot product
            dot = 0
            for j in rows:
                dot += 2*Sx[j]
            dot -= 2*s_[i]*sx
            dot += t*u[i]*ux
            dot -= Jii[i]*xout[i]
            
            ## Compute currents
            curr = (h[i] - (scl**2)*Jii[i]/2 - (scl**2)*dot - beta)/temp
        
            ## Apply sigmoid (overflow robust)
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))
            
            ## Update outputs
            xi = 1*(np.random.rand() < prob)
            dx = xi - xout[i]
            xout[i] = xi
            
            ## Update dot products
            if np.abs(dx) > 0:
                sx += dx*s[i]
                ux += dx*u[i]
                Sx[rows] += dx    # O(nnz)
        
    return xout


    