
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

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import numpy.linalg as nla
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

import networkx as nx

import cvxpy as cvx

# my code
import util
import df_util 


#############################################################################
################## Classes for approximate embedding ########################
#############################################################################

# class RBLS(BinaryKernel):
#     """
#     Recursive binary least squares

#     Inputs at each update should be uncentered kernel, 
#     this object will keep a running estimate of the mean
#     """

#     def __init__(self, K, max_dim=10, threshold=1e-4):

#         self.dmax = max_dim     # maximum dimension
#         self.pimin = threshold  # minimum norm

#         super().__init__(K[:max_dim,:max_dim])

#     def initfeats(self, K):

#         # self.Knrm = np.sum(util.center(K)**2)
#         self.K_ = K.mean(0) #/np.sqrt(Knrm)

#         ## initialize
#         S, pi = df_util.mindist(K)
#         these = pi >= self.pimin
#         S = S[:,these]
#         pi = pi[these]
#         self.i, self.s = np.where(S)
#         self.S = sprs.csr_array(S)
#         self.pi = pi 

#         self.dot = np.sum(util.center(K)*util.center(self.pred()))
#         self.nrm = np.sum(util.center(self.pred())**2)
#         self.knrm = np.sum(util.center(K)**2)

#     def iterate(self, k, k0, eps=0, verbose=False):

#         if verbose:
#             t0 = time()
#         x, x0, sig = self.fit(k, k0, eps=eps)
#         if verbose:
#             print(' ... Fit in %.3f s'%(time()-t0))

#         self.K_ = self.newK_
#         self.nrm = self.newnrm
#         self.dot = self.newdot
#         self.knrm = self.newknrm

#         self.split(x, x0, sig)
#         if verbose:
#             print(' ... Split in  %.3f s'%(time()-t0))

#         self.n += 1

#     def pred(self, n=None):

#         if n is None:
#             n = self.n
#         return (self.S@sprs.diags(self.pi)@self.S.T)[:n,:n].todense()

#     def splitcost(self, x, x0, k, k0):

#         k_ = k - self.K_ - k.mean() + self.K_.mean()
#         k0_ = k0 - 2*k.mean() + self.K_.mean()

#         ## Define constants
#         s_ = self.S.mean(0)
#         Pi = sprs.diags(self.pi)

#         t = self.n/(self.n + 1)

#         S_ = self.S - s_
#         sqt = np.sqrt(2*t)

#         ## Problem construction
#         A = np.block([[sqt*S_@Pi, np.zeros((self.n,1))], [-t*Pi@(1-2*s_), -t]])
#         b = np.concatenate([sqt*(k_ + S_@Pi@s_), [t*(s_@Pi@s_ - k0_)]])

#         p = np.append(x, x0)

#         return np.sum((A@p - b)**2)

#     def fit(self, k, k0, eps, beta=1e-3):
#         """
#         Assumes that k and k0 are uncentered w.r.t. K_n ...
#         """

#         ## Deal with centering
#         k_ = k - self.K_ - k.mean() + self.K_.mean()
#         k0_ = k0 - 2*k.mean() + self.K_.mean()

#         ## Define constants
#         s_ = self.S.mean(0)
#         Pi = sprs.diags(self.pi)
#         S_ = self.S - s_

#         t = self.n/(self.n + 1)
#         sqt = np.sqrt(2*t)
#         u = (1-s_)@Pi@s_

#         Aup = [sqt*S_@Pi, np.zeros((self.n,1)), -sqt*S_@Pi@s_[:,None]]
#         Alow = [-t*Pi@(1-2*s_), -t, -t*s_@Pi@s_]
#         A = np.block([Aup, Alow])
#         b = np.concatenate([sqt*k_, [-t*k0_]]) 

#         ## Construct problem
#         p = cvx.Variable(len(s_)+2) 

#         cost = cvx.sum_squares(A@p - b) + self.nrm*(p[-1]**2) - 2*self.dot*p[-1]
#         reg = np.append(np.min([s_, 1-s_], axis=0), [-1, 0])
#         reg = (1-eps)*reg + eps*np.random.randn(len(reg))
#         regcost = cvx.Minimize(cost + beta*reg@p)
#         # regcost = cvx.Minimize(cost)
#         prob = cvx.Problem(regcost, constraints=[p>=0, p[:-2]<=p[-1]])

#         Cmin = prob.solve(solver='ECOS')
#         pstar = p.value

#         # ## Define constants
#         # s_ = self.S.mean(0)
#         # Pi = sprs.diags(self.pi)

#         # S_ = self.S - s_
#         # sqt = np.sqrt(2*t)

#         # A = np.block([[sqt*S_@Pi, np.zeros((self.n,1))], [-t*Pi@(1-2*s_), -t]])
#         # b = np.concatenate([sqt*(k_ + S_@Pi@s_), [t*(s_@Pi@s_ - k0_)]])

#         # ## Problem construction
#         # p = cvx.Variable(len(s_)+1)

#         # cost = cvx.sum_squares(A@p - b)
#         # reg = np.append(np.min([s_, 1-s_], axis=0), -1)
#         # reg = (1-eps)*reg + eps*np.random.randn(len(reg))
#         # regcost = cvx.Minimize(cost + 1e-4*reg@p)
#         # # regcost = cvx.Minimize(cost)
#         # prob = cvx.Problem(regcost, constraints=[p>=0, p[:-1]<=1])

#         # Cmin = prob.solve()
#         # pstar = p.value

#         ##### Book-keeping

#         ## update kernel center
#         self.newK_ = self.K_ + (k - self.K_)/(self.n+1)
#         self.newK_ = np.append(self.newK_, (np.sum(k) + k0)/(self.n+1))

#         ## update running loss terms
#         sig = pstar[-1]
#         p_ = pstar[:-2]/sig - s_
#         kpred = sig*S_@Pi@p_
#         k0pred = sig*p_@Pi@(1-2*s_) + sig*u + pstar[-2]
#         self.newnrm = (sig**2)*self.nrm + 2*t*kpred@kpred + (t**2)*k0pred**2
#         self.newdot = sig*self.dot + 2*t*k_@kpred + (t**2)*k0_*k0pred
#         self.newknrm = self.knrm + 2*t*k_@k_ + (t**2)*k0_**2

#         return pstar[:-2]/sig, pstar[-2], pstar[-1]


# @dataclass
# class Cuts:
#     """
#     Stores information about the embeddings
#     """

#     N: int
#     pimin: float = 1e-3
#     beta: float = 1e-3
#     tol: float = 1e-3
#     eps: float = 0

#     def __post_init__(self):

#         S = util.F2(self.N-1)
#         S = np.hstack([np.zeros((len(S),1)), S]).T
#         self.i, self.s = np.where(S)
#         self.S = sprs.csr_array(S)

#         ovlp = S.T@S

#         self.issup = (ovlp == np.diag(ovlp)) - np.eye(S.shape[1])
#         self.numsup = self.issup.sum(0)

#         ## Mutual exclusivity (and no self-connections)
#         self.isopp = (ovlp == 0) + np.eye(S.shape[1])

#     def fit_pi(self, K):

#         S, pi = df_util.mindist(K, S=self.S, 
#             beta=self.beta, eps=self.eps, tol=self.tol)
#         self.i, self.s = np.where(S)
#         self.S = sprs.csr_array(S)
#         self.pi = pi

#         self.dot = np.sum(util.center(self.K)*util.center(self.pred()))
#         self.nrm = np.sum(util.center(self.pred())**2)
#         self.knrm = np.sum(util.center(self.K)**2)

#     def navigate(self, s, u, temp=1e-3):
#         """
#         Produce a path from s with initial utility u
#         """

#         path = []
#         s_t = 1*s

#         while 1:
#             ua = u*self.actions(s)
#             ua = ua*(ua>0)

#             if np.any(ua):
#                 a = np.argmax(ua)
#                 path.append(a)
#                 s_t[a] += 1
#             else:
#                 break

#         return path

#     def actions(self, s):
#         """
#         Produce feasible components at point s
#         """

#         ante = self.issup.T@s == self.numsup
#         poss = self.isopp@s == 0

#         feas = ante*poss

#         return feas

#     def kern(self):
#         return (self.S@sprs.diags(self.pi)@self.S.T).todense()

#     def split(self, x, x0, sig):

#         k = len(self.pi)

#         has = np.nonzero(x > 1e-6)[0]

#         split = np.nonzero(np.abs(x - x**2) > 1e-6)[0]
#         these = np.isin(self.s, split)
#         guys = np.unique(self.s[these], return_inverse=True)[1]

#         self.pi = self.pi*sig

#         ## Add new columns (split features)
#         self.pi = np.append(self.pi, self.pi[split]*(1-x[split]))
#         self.pi[split] = self.pi[split]*x[split]
#         self.s = np.append(self.s, guys+k)
#         self.i = np.append(self.i, self.i[these])

#         ## Add new row (new item)
#         self.s = np.append(self.s, has)
#         self.i = np.append(self.i, self.n*np.ones(len(has), int))

#         ## Remove columns below threshold
#         dead = np.where(self.pi < self.pimin)[0]
#         isdead = np.isin(self.s, dead)
#         self.pi = np.delete(self.pi, dead)
#         self.s = np.delete(self.s, isdead)
#         self.i = np.delete(self.i, isdead)
#         _, self.s = np.unique(self.s, return_inverse=True)

#         ## Add one-hot column if necessary
#         if x0 > 1e-6:
#             self.pi = np.append(self.pi, x0)
#             self.s = np.append(self.s, self.s.max()+1)
#             self.i = np.append(self.i, self.n)

#         self.S = sprs.coo_array((np.ones(len(self.i)), (self.i, self.s)),
#             shape=(self.n+1, len(self.pi)))

#     def fit(self, k_, k0_, eps=None, beta=None):
#         """
#         Assumes that k and k0 are centered w.r.t. K_n ...
#         """

#         ## Parse args
#         if eps is None:
#             eps = self.eps
#         if beta is None:
#             beta = self.beta

#         ## Define constants
#         s_ = self.S.mean(0)
#         Pi = sprs.diags(self.pi)
#         S_ = self.S - s_

#         t = self.n/(self.n + 1)
#         n = self.n
#         sqt = np.sqrt(2*t)
#         tn = (t**2 - 2*t/n)
#         u = (1-s_)@Pi@s_

#         A1 = [self.S@Pi, np.zeros((n,1)), -self.S@Pi@s_[:,None]]
#         A2 = [(1-s_)@Pi, [1], [0]]

#         A1 = sprs.block_array([A1])
#         A2 = np.concatenate(A2)
#         a = A1.sum(0)

#         A = np.vstack([sqt*(A1-a[None,:]/n), -t*(A2-a[None,:]/n)]) 
#         b = np.concatenate([sqt*k_, [-t*k0_]]) 

#         ## Construct problem
#         x = cvx.Variable(len(s_)+2) # x = [p_hat, pi_0, sigma]

#         qq = self.nrm 
#         kq = self.dot

#         # if n > 40:
#         # cost = 2*t*(cvx.sum_squares(A1@x - k_) - (1/n)*(a@x)**2)
#         # cost += (t**2)*((A2 - a/n)@x - k0_)**2
#         # cost += qq*(x[-1]**2) - 2*kq*x[-1]
#         # else:
        
#         cost = cvx.sum_squares(A@x - b) + qq*(x[-1]**2) - 2*kq*x[-1]  

#         # Q = A.T@A 
#         # c = A.T@b
#         # cost = cvx.quad_form(x, Q, assume_PSD=True) - 2*c@x + qq*(x[-1]**2) - 2*kq*x[-1]
        
#         reg = np.where(s_ < 0.5, self.pi, -self.pi)
#         reg = np.append(reg, [-1, 0])
#         reg = ((1-eps)*reg + eps*np.random.randn(len(reg)))
#         regcost = cvx.Minimize(cost + beta*reg@x)
#         prob = cvx.Problem(regcost, constraints=[x>=0, x[:-2]<=x[-1]])

#         Cmin = prob.solve(solver='CLARABEL')
#         xstar = x.value

#         ## Round within tolerance (dumb rounding)
#         sig = xstar[-1]
#         p = xstar[:-2]/sig 

#         # sort by slack
#         slack = np.round(p) - p
#         pidx = np.argsort(np.abs(slack))
#         piidx = np.argsort(self.pi)

#         # compute marginal loss of rounding each element
#         err = A@xstar - b 
#         cerr = np.cumsum(A[:,:-2][:,pidx]*slack, axis=1)
#         marg = (cerr**2).sum(0)

#         # round those below tolerance
#         these = pidx[marg/np.sqrt(self.knrm) < self.tol]
#         p[these] += slack[these]

#         ##### Book-keeping

#         ## update kernel center
#         self.newK_ = self.K_ + (k - self.K_)/(self.n+1)
#         self.newK_ = np.append(self.newK_, (np.sum(k) + k0)/(self.n+1))

#         ## update running loss terms
#         p_ = p - s_
#         kpred = sig*S_@Pi@p_
#         k0pred = sig*p_@Pi@(1-2*s_) + sig*u + xstar[-2]
#         self.newnrm = (sig**2)*self.nrm + 2*t*kpred@kpred + (t**2)*k0pred**2
#         self.newdot = sig*self.dot + 2*t*k_@kpred + (t**2)*k0_*k0pred
#         self.newknrm = self.knrm + 2*t*k_@k_ + (t**2)*k0_**2

#         return p, xstar[-2], sig


class RBLS:
    """
    Recursive binary least squares
    
    Class for optimizing Cuts

    Inputs at each update should be uncentered kernel, 
    this object will keep a running estimate of the mean
    """

    def __init__(self, K, max_dim=10, S0=None, reg='sparse',
        pimin=1e-3, tol=1e-4, beta=1e-3, eps=0, rmax=1000):

        self.pimin = pimin  # minimum norm
        self.tol = tol
        self.eps = eps # stochasticity
        self.beta = beta # regularization
        self.rmax = rmax

        self.reg = reg

        if S0 is not None:
            max_dim = len(S0)
        self.K = K[:max_dim, :max_dim]

        # self.Knrm = np.sum(util.center(K)**2)
        self.K_ = self.K.mean(0) #/np.sqrt(Knrm)
        self.n = len(self.K)

        ## initialize
        self.fit_pi(S0)

    def loss(self):
        return self.nrm/self.knrm - 2*self.dot/self.knrm + 1

    def iterate(self, k, k0, eps=None, beta=None, verbose=False, debug=False):

        t0 = time()
        t = []

        x, x0, sig = self.fit(k, k0, eps=eps, beta=beta)
        t.append(time()-t0)
        if verbose:
            print(' ... Fit in %.3f s (loss: %.3f, sig: %.3f)'%(time()-t0, self.loss(), sig))

        self.K = np.block([[self.K, k[:,None]],[k[None,:], k0]])
        self.K_ = self.newK_
        self.nrm = self.newnrm
        self.dot = self.newdot
        self.knrm = self.newknrm

        self.split(x, x0, sig)
        t.append(time()-t0)
        if verbose:
            print(' ... Split in  %.3f s (pi range:(%.3f, %.3f, %.3f))'%(time()-t0, self.pi.min(), np.median(self.pi), self.pi.max()))

        self.n += 1

        if debug:
            return t
        else:
            return None

        # self.fit_pi(self.S.todense())

    def pred(self, n=None):

        if n is None:
            n = self.n
        return (self.S@sprs.diags(self.pi)@self.S.T)[:n,:n].todense()

    def splitcost(self, x, x0, k, k0):

        k_ = k - self.K_ - k.mean() + self.K_.mean()
        k0_ = k0 - 2*k.mean() + self.K_.mean()
        p = np.append(x, x0)

        ## Define constants
        s_ = self.S.mean(0)
        Pi = sprs.diags(self.pi)

        t = self.n/(self.n + 1)
        n = self.n

        S_ = self.S - s_
        sqt = np.sqrt(2*t)

        ## Problem construction
        # A = np.block([[sqt*S_@Pi, np.zeros((n,1))], [-t*Pi@(1-2*s_), -t]])
        # b = np.concatenate([sqt*(k_ + S_@Pi@s_), [t*(s_@Pi@s_ - k0_)]])

        # cost = np.sum((A@p - b)**2)

        p = np.append(p, 1)

        A1 = sprs.block_array([[self.S@Pi, np.zeros((n,1)), -self.S@Pi@s_[:,None]]])
        A2 = np.concatenate([(1-s_)@Pi, [1], [0]])

        a = A1.sum(0)

        cost = 2*t*(np.sum((A1@p - k_)**2) - (1/n)*(a@p)**2)
        cost += (t**2)*((A2 - a/n)@p - k0_)**2

        return cost
        # return np.sum((A@p - b)**2)

    def merge(self, *others):

        ## merge features
        S_unq = util.spunique(sprs.hstack([o.S for o in others]))

        self.s = S_unq.col
        self.i = S_unq.row

        self.S = S_unq
        self.n = S_unq.shape[0]

        ## refit 
        self.refit(self.S.todense())

    def fit_pi(self, S=None):

        S, pi = df_util.mindist(self.K, S=S, 
            beta=self.beta, eps=self.eps, tol=self.tol)
        self.i, self.s = np.where(S)
        self.S = sprs.csr_array(S)
        self.pi = pi 

        self.dot = np.sum(util.center(self.K)*util.center(self.pred()))
        self.nrm = np.sum(util.center(self.pred())**2)
        self.knrm = np.sum(util.center(self.K)**2)

    def split(self, x, x0, sig):

        k = len(self.pi)

        has = np.nonzero(x > 1e-6)[0]

        split = np.nonzero(np.abs(x - x**2) > 1e-6)[0]
        these = np.isin(self.s, split)
        guys = np.unique(self.s[these], return_inverse=True)[1]

        self.pi = self.pi*sig

        ## Add new columns (split features)
        self.pi = np.append(self.pi, self.pi[split]*(1-x[split]))
        self.pi[split] = self.pi[split]*x[split]
        self.s = np.append(self.s, guys+k)
        self.i = np.append(self.i, self.i[these])

        ## Add new row (new item)
        self.s = np.append(self.s, has)
        self.i = np.append(self.i, self.n*np.ones(len(has), int))

        ## Remove columns below threshold
        over = np.argsort(np.argsort(-self.pi))>self.rmax
        dead = np.where((self.pi < self.pimin)|over)[0]
        isdead = np.isin(self.s, dead)
        self.pi = np.delete(self.pi, dead)
        self.s = np.delete(self.s, isdead)
        self.i = np.delete(self.i, isdead)
        _, self.s = np.unique(self.s, return_inverse=True)

        ## Add one-hot column if necessary
        if x0 > self.pimin:
            self.pi = np.append(self.pi, x0)
            self.s = np.append(self.s, self.s.max()+1)
            self.i = np.append(self.i, self.n)

        self.S = sprs.coo_array((np.ones(len(self.i)), (self.i, self.s)),
            shape=(self.n+1, len(self.pi)))

    def fit(self, k, k0, eps=None, beta=None):
        """
        Assumes that k and k0 are uncentered w.r.t. K_n ...
        """

        ## Parse args
        if eps is None:
            eps = self.eps
        if beta is None:
            beta = self.beta

        ## Deal with centering
        k_ = k - self.K_ - k.mean() + self.K_.mean()
        k0_ = k0 - 2*k.mean() + self.K_.mean()

        ## Define constants
        s_ = self.S.mean(0)
        Pi = sprs.diags(self.pi)
        S_ = self.S - s_

        t = self.n/(self.n + 1)
        n = self.n
        sqt = np.sqrt(2*t)
        tn = (t**2 - 2*t/n)
        u = (1-s_)@Pi@s_

        A1 = [self.S@Pi, np.zeros((n,1)), -self.S@Pi@s_[:,None]]
        A2 = [(1-s_)@Pi, [1], [0]]

        A1 = sprs.block_array([A1])
        A2 = np.concatenate(A2)
        a = A1.sum(0)

        A = np.vstack([sqt*(A1-a[None,:]/n), -t*(A2-a[None,:]/n)]) 
        b = np.concatenate([sqt*k_, [-t*k0_]]) 

        ## Construct problem
        x = cvx.Variable(len(s_)+2) # x = [p_hat, pi_0, sigma]

        qq = self.nrm 
        kq = self.dot
        
        # if n > 40:
        # cost = 2*t*(cvx.sum_squares(A1@x - k_) - (1/n)*(a@x)**2)
        # cost += (t**2)*((A2 - a/n)@x - k0_)**2
        # cost += qq*(x[-1]**2) - 2*kq*x[-1]
        # else:
        
        cost = cvx.sum_squares(A@x - b) + qq*(x[-1]**2) - 2*kq*x[-1]  

        # Q = A.T@A 
        # c = A.T@b
        # cost = cvx.quad_form(x, Q, assume_PSD=True) - 2*c@x + qq*(x[-1]**2) - 2*kq*x[-1]
        
        if self.reg == 'sparse': # simple sparsity regularization
            reg = np.where(s_ < 0.5, self.pi, -self.pi)
            reg = np.append(reg, [-1, 0])
        elif self.reg == 'node': # graph-based regularization
            r = self.S.sum(0)
            cn = spc.binom(self.n+1, 2)
            reg = r - (r**3)/cn 
            reg = np.append(reg, [0, 0])
        reg = ((1-eps)*reg + eps*np.random.randn(len(reg)))
        regcost = cvx.Minimize(cost/np.sqrt(self.knrm) + beta*reg@x)
        prob = cvx.Problem(regcost, constraints=[x>=0, x[:-2]<=x[-1]])

        Cmin = prob.solve(solver='CLARABEL')
        xstar = x.value

        ## Round within tolerance (dumb rounding)
        sig = xstar[-1]
        p = xstar[:-2]/sig 

        # sort by slack
        slack = np.round(p) - p
        pidx = np.argsort(np.abs(slack))
        piidx = np.argsort(self.pi)

        # compute marginal loss of rounding each element
        err = A@xstar - b 
        cerr = np.cumsum(A[:,:-2][:,pidx]*slack, axis=1)
        marg = (cerr**2).sum(0)

        # round those below tolerance
        these = pidx[marg/np.sqrt(self.knrm) < self.tol]
        p[these] += slack[these]

        ##### Book-keeping

        ## update kernel center
        self.newK_ = self.K_ + (k - self.K_)/(self.n+1)
        self.newK_ = np.append(self.newK_, (np.sum(k) + k0)/(self.n+1))

        ## update running loss terms
        p_ = p - s_
        kpred = sig*S_@Pi@p_
        k0pred = sig*p_@Pi@(1-2*s_) + sig*u + xstar[-2]
        self.newnrm = (sig**2)*self.nrm + 2*t*kpred@kpred + (t**2)*k0pred**2
        self.newdot = sig*self.dot + 2*t*k_@kpred + (t**2)*k0_*k0pred
        self.newknrm = self.knrm + 2*t*k_@k_ + (t**2)*k0_**2

        return p, xstar[-2], sig

