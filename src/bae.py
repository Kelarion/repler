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
from tqdm import tqdm
# import geoopt as geo

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
import bae_util

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
        self.fix_scale = fix_scale

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
        
        # Center the kernel of each item excluding that item
        K = self.X@self.X.T
        notI = (1 - np.eye(self.n))/(self.n-1) 
        self.K = (K - K@notI - (K*notI).sum(0) + ((K@notI)*notI).sum(0)).T 
        
        ## Initialize S
        coding_level = np.random.beta(alpha, beta, dim_hid)/2
        num_active = np.floor(coding_level*len(X)).astype(int)
        
        Mx = self.X@np.random.randn(len(X.T),self.r)
        thr = -np.sort(-Mx, axis=0)[num_active, np.arange(dim_hid)]
        # self.S = sprs.csc_array(1*(Mx >= thr))
        self.S = np.array((Mx >= thr)*1)
        # self.notS = sprs.csc_array(1*(Mx < thr))

        # self.sum_s = self.S.sum(0)

        ## Penalty matrix
        # self.StS = self.S.T@self.S

    def predict(self):

        return self.S@self.S.T

    def grad_step(self, pause_schedule=False, **opt_args):
        
        if not self.initialized:
            self.schedule = Neal(**opt_args)
            self.initialized = True
        
        T = self.schedule(freeze=pause_schedule)
        
        newS = update_concepts_kernel(self.K, self.S.astype(float), 
            scl=self.scl, beta=self.beta, temp=T)
        self.S = newS

        if not self.fix_scale:
            self.scl = self.scaleS()
        
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
        Qnrm = (self.scl**2)*np.sum(util.center((S@S.T))**2)
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
        S = self.S
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
        
        k = (X-x_)@(x-x_)
        k0 = (x-x_)@(x-x_)
        # K_ = self.K[I][:,I].mean(0)
        # K__ = K_.mean()
        # k_ = self.K[I,i].mean()
        # k = self.K[I,i] - K_ - k_ + K__
        # k0 = self.K[i,i] - 2*k_ + K__
        
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
        
        k = self.K[i,I]
        k0 = self.K[i,i]
        # k = (X-x_)@(x-x_)
        # k0 = (x-x_)@(x-x_)
        # K_ = self.K[I][:,I].mean(0)
        # K__ = K_.mean()
        # k_ = self.K[I,i].mean()
        # k = self.K[I,i] - K_ - k_ + K__
        # k0 = self.K[i,i] - 2*k_ + K__
        
        ## Organize states
        S = self.S[I].tocsc()         # i.e. S(X)
        s = self.S[[i]].todense().squeeze()

        ## Energy
        StS = self.StS - np.outer(s,s)
        S1 = (self.sum_s - s)[None,:]
        # StS = S.T@S
        # S1 = S.sum(0)[None,:]
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

        # ## Dynamics
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

        # s = regularized_hopfield(n, s, S.indptr, S.indices, k0, S.T@k, temp, 
        #     Q, p, self.scl, self.steps, beta=self.beta)

        return s

class BAE:

    def __init__(self, X, dim_hid, rank=None, pvar=1,
        weight_class='ortho', weight_alg='exact', lr=1e-2,
        penalty=0, alpha=2, beta=5, fix_scale=False):

        self.n, self.d = X.shape
        self.r = dim_hid
        self.initialized = False
        self.beta = penalty
        self.fix_scl = fix_scale

        self.prior_b = beta
        self.prior_a = alpha

        self.weight_class = weight_class
        self.weight_alg = weight_alg
        self.lr = lr

        Ux,sx,Vx = la.svd(X-X.mean(0), full_matrices=False)        
        
        self.frac_var = np.cumsum(sx**2)/np.sum(sx**2)
        if rank is None:
            r = np.min([len(sx), np.sum(self.frac_var <= pvar)+1])
            # r = np.max([dim_hid, np.sum(self.frac_var <= pvar)+1])
        else:
            r = rank
        # self.d = r

        self.X = X - X.mean(0, keepdims=True)
        # self.X = Ux[:,:r]@np.diag(sx[:r])
        self.U = Ux[:,sx>1e-6]
        self.U = self.U[:,:r]
        self.trXX = np.sum(sx[:r]**2)

        self.initialize()

    def predict(self):
        """
        Make predictions for items I
        """
        return self.S@self.W.T

    def initialize(self):

        ## Initialize S 
        coding_level = np.random.beta(self.prior_a, self.prior_b, self.r)/2
        num_active = np.floor(coding_level*self.n).astype(int)
        
        Mx = self.X@np.random.randn(self.d,self.r)
        thr = -np.sort(-Mx, axis=0)[num_active, np.arange(self.r)]
        # self.S = sprs.csr_array(1*(Mx >= thr))
        self.S = 1*(Mx >= thr)
        self.trSS = np.trace(self.S.T@self.S)
        # self.trSS = (self.S.T@self.S).trace()

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
            self.W = (la.pinv(self.S.todense())@self.X).T
            self.scl = 1

        ## Regularization
        self.StS = self.S.T@self.S
        self.Ssum = self.S.sum(0)

    def grad_step(self, pause_schedule=False, **opt_args):

        if not self.initialized:
            self.schedule = Neal(**opt_args)
            self.initialized = True
        
        T = self.schedule(freeze=pause_schedule)

        self.updateS(T)          ## "E step"
        # ES = self.updateS_reg(T) ## E step
        self.updateWb(ES)        ## "M step"
        # self.updateWb(self.S)

    # def updateS(self, temp):

    #     C = (self.U@(self.U.T@self.S) + self.S.mean(0) - 0.5)
    #     current = self.beta*C + 2*(self.X - self.b)@self.W - self.scl
    #     prb = spc.expit(current/temp)
    #     self.S = sprs.csr_array(np.random.binomial(1, prb))
    #     self.trSS = (self.S.T@self.S).trace()

    def updateS(self, temp):

        oldS = self.S.todense().astype(float)
        newS = update_concepts_asymmetric(self.X-self.b, oldS, self.scl*self.W, 
            beta=self.beta, temp=temp)

        self.S = newS
        self.trSS = np.trace(newS.T@newS)

    def updateWb(self, ES):

        ## update weights
        if self.weight_class == 'ortho':

            if self.weight_alg == 'exact':
                U,s,V = la.svd((self.X-self.b).T@ES, full_matrices=False)
                self.W = U@V
                if not self.fix_scl:
                    self.scl = np.sum(s)/ES.sum()
            else:
                ## Riemannian SGD over the weights
                ## Using Cayley algorithm of Li, Li, Todorovic (ICLR 2020)

                dW = -2*self.scl*(self.X-self.b).T@self.S
                Z = dW@W.T
                Z = dW@W.T - 0.5*(W@())

                self.W += self.lr*dW 

        elif self.weight_class == 'real':
            if self.weight_alg == 'exact':
                self.W = (la.pinv(self.S.todense())@(self.X - self.b)).T
            else:
                dW = (self.X-self.b).T@self.S
                self.W -= self.lr*dW

        ## update biases
        if self.weight_alg == 'exact':
            self.b = (self.X - self.scl*ES@self.W.T).mean(0)
        else:
            db = self.b + (self.X - self.scl*ES@self.W.T).mean(0)
            self.b -= lr*db

    def updateS_reg(self, temp):
        """
        Regularised update of S
        """

        StS = 1*self.StS
        Ssum = 1*self.Ssum
        C = 2*(self.X - self.b)@self.W - self.scl
        ES = np.zeros(C.shape) ## keep track of the expectation

        for it in np.random.permutation(range(self.n)):
        # for it in np.arange(self.n):

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
            # I1 = np.sign(2*StS - Ssum[None,:]) # si'sj - (1-si)'sj
            # I2 = np.sign(2*StS - Ssum[:,None]) # si'sj - si'(1-sj)
            # I3 = np.sign(Ssum[None,:] - Ssum[:,None]) # (1-si)'sj - si'(1-sj)
            # I4 = np.sign(Ssum[None,:] + Ssum[:,None] - n) # si'sj - (1-si)'(1-sj)

            # Q = (1*(I1<0)*(I2<0) - 1*(I1>0)*(I3<0) - 1*(I2>0)*(I3>0))*(I4<0)
            # p = ((I2>0)*(I3>0)*(I4<0)).sum(1)

            ## Dynamics 
            for i in np.random.permutation(range(self.r)):

                inhib = Q[i]@s + p[i]

                ES[it,i] = spc.expit((self.scl*C[it,i] - self.beta*inhib)/temp)
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


class RejectionBAE:

    def __init__(self, X, eps=1e-5, rmax=None, whiten=True, learn=False):
        """
        X has shape (num_data, dim_data)
        """

        U,s,V = la.svd(X-X.mean(0,keepdims=True), full_matrices=False)

        these = s>eps
        if rmax is not None:
            these[rmax:] = False

        self.P = U[:,these] # for random sampling
        self.Q = 1*self.P   # for energy function
        if not whiten:
            self.P = self.P@np.diag(s[these])

        # self.Q = U[:,(~these)]

        self.updateQ = learn

        self.N = len(X)
        self.r = sum(these)

    def step(self, S, t=1, beta=1):
        "Apply Hopfield update for t steps"

        for i in range(t):
            if self.updateQ:
                c = self.P@(self.P.T@S) + S.mean(0, keepdims=True)
            else:
                c = self.Q@(self.Q.T@S) + S.mean(0, keepdims=True)
            S = 2*np.random.binomial(1, spc.expit(beta*c)) - 1

        return S

    def sample(self, m):
        "Random normal vector in Im(A)"
        return self.P@np.random.randn(self.r, m)

    def learn(self, s, eta=0.1):
        "Avoid sampling w in the future"
        self.P -= (eta/(s.shape[1]+1e-6))*s@s.T@self.P/self.N

    def energy(self, s):
        return 1 - (((self.Q.T@s)**2).sum(0) + (s.sum(0)**2)/self.N)/self.N

    def fit(self, samps, max_iter=100, tol=1e-3, lr=0.0, S0=None,
        alpha=0.88, beta=1, period=10, verbose=False):

        if S0 is None:
            M = self.sample(samps)
            S = np.sign(M)
        else:
            S = 1*S0
            samps = S0.shape[1]

        t = np.zeros(samps)

        if verbose:
            pbar = tqdm(range(max_iter))

        basins = np.empty((self.N, 0))
        energy = np.empty((0, samps))
        emin = np.inf
        it = 0
        while (it<max_iter):

            ens = self.energy(S)
            energy = np.vstack([energy, ens])

            temp = alpha**(t//period)
            news = self.step(S, beta=beta/temp)

            stopped = (np.sum(S*news, axis=0) == self.N) 
            good = energy[it] - emin < tol
            bad = (np.abs(np.sum(S, axis=0)) == self.N)

            basins = np.hstack([basins, S[:,good&stopped&(~bad)]])

            neww = self.sample(np.sum(stopped))
            news[:,stopped] = np.sign(neww)
            energy[it,stopped] = np.nan

            self.learn(S[:,stopped], eta=lr)
            S = 1*news
            t[stopped] = 0
            t[~stopped] += 1

            basins = np.unique(basins*basins[[0]], axis=1)

            if np.sum(~bad)>0:
                emin = np.min([emin, np.min(ens[~bad])])

            if verbose:
                pbar.update(1)

            it += 1

        # ben = self.energy(basins)
        # basins = basins[:, ben-ben.min() < tol]

        return basins, energy, emin


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
    ## for some reason it only works when written flat
    
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


    
@njit
def regularized_hopfield(n, x, indptr, indices, k0, Sk, temp, R, r, scl=1, steps=1, beta=0):
    """
    
    """

    ## TODO: keep track of energy to avoid re-computing
    
    ## It seems that the @njit decorator lets you use for
    ## loops in python, but you can only use numpy and 
    ## for some reason it only works when written flat
    
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
    h = t*((scl**2)*s_sc_.sum() - scl*k0)*u + 2*scl*Sk - beta*(scl**2)*r
    
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
            dot -= beta*R[i][xout>0].sum()

            ## Compute currents
            curr = (h[i] - (scl**2)*Jii[i]/2 - (scl**2)*dot)/temp
        
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

        ## Regularization
        I1 = np.sign(2*StS - St1[None,:])
        I2 = np.sign(2*StS - St1[:,None])
        I3 = np.sign(St1[None,:] - St1[:,None])
        I4 = np.sign(St1[None,:] + St1[:,None] - (n-1))

        R = ((1*(I1<0)*(I2<0) - 1*(I1>0)*(I3<0) - 1*(I2>0)*(I3>0))*(I4<0))*1.0
        r = ((I2>0)*(I3>0)*(I4<0)).sum(1)*1.0

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
                dot += beta*(scl**2)*np.dot(R[j], news)

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


@njit
def update_concepts_asymmetric(X, S, W, beta, temp, steps=1, lognorm=bae_util.gaussian):
    """
    One gradient step on S

    TODO: figure out a good sparse implementation!
    """

    n, m = S.shape

    StS = np.dot(S.T, S)
    St1 = S.sum(0)

    ## Initial values
    E = np.dot(S,W.T)  # Natural parameters
    C = np.dot(X,W)    # Constant term

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
        # news = 1*s
        for step in range(steps):
            for j in range(m): # concept

                ## Compute linear terms
                dot = np.sum(lognorm(E[i] + (1-S[i,j])*W[:,j])) 
                dot -= np.sum(lognorm(E[i] - S[i,j]*W[:,j]))
                inhib = np.dot(R[j], S[i]) + r[j]

                ## Compute currents
                curr = (2*C[i,j] - beta*inhib - dot)/temp

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
