import os, sys, re
import pickle
from time import time
import copy

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal as ortho
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
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp

import networkx as nx

import cvxpy as cvx

# import line_profiler

# my code
import util
import df_util 
import df_models as mods

#############################################################################
################## Generic algorithm ########################################
#############################################################################

# @line_profiler.profile
def cuts(K, branches=12, eps=0.5, order='nearest', refit_period=10,
    verbose=False, bar=False, debug=False, return_mod=False,
    **kwargs):

    N = len(K)

    ## Optionally re-order items to try and minimize dead ends
    if order == 'nearest':
        idx = df_util.item_order(util.dot2dist(K))
    elif order == 'ball':
        idx = df_util.ball_order(util.dot2dist(K))
    elif order == 'given':
        idx = np.arange(N)
    idxinv = np.argsort(idx)
    K_ = K[idx,:][:,idx]

    ## Initialize each branch
    S = []
    ls = []
    for i in range(branches):
        S.append(mods.RBLS(K_, eps=eps*(i > 0), **kwargs))
        ls.append([S[i].loss()])

    if not verbose and bar:
        pbar = tqdm(range(S[0].n, S[0].N))

    ## add new items
    times = []
    feats = []
    t0 = time()
    for n in range(S[0].n, N):

        if verbose:
            print('Item %d at time %.3f'%(n, time()-t0))

        ## see how well we can do
        t = []
        r = []
        for i in range(branches):

            if verbose:
                print('... Chain %d with %d features:'%(i, len(S[i].pi)))

            tt = S[i].iterate(K_[n,:n], K[n,n], verbose=verbose, debug=debug)
            t.append(tt)
            r.append(len(S[i].pi))

            if not np.mod(n, refit_period):
                S[i].fit_pi(S[i].S.todense())

            ls[i].append(S[i].loss())

        times.append(t)
        feats.append(r)

        if not verbose and bar:
            pbar.update(1)

    ## Compile information across branches
    S_unq = util.spunique(sprs.hstack([s.S for s in S]))
    pi_unq = []
    for i in range(branches):
        match = util.hamming(S_unq, S[i].S)
        pis = np.where(match.min(1)==0, S[i].pi[match.argmin(1)], 0)
        pi_unq.append(np.squeeze(pis))

    S_unq.row = idx[S_unq.row]

    if debug:
        if return_mod:
            outs = (S, (times, feats, ls))
        else:
            outs = (S_unq, np.array(pi_unq), (times, feats, np.array(ls)))
    else:
        outs = (S_unq, np.array(pi_unq))

    return outs


#############################################################################
##################### Binary autoencoder ####################################
#############################################################################

# class Affine(np.ndarray):
#     """
#     Simple affine layer which includes manual gradients

#     just to avoid requiring pytorch
#     """

class sBAE:
    """
    Binary autoencoder with reconstruction loss
    """

    def __init__(self, X, dim_hid, eps=1e-5, tau=0,
        w_lr=1e-2, b_lr=None, rank=None, whiten=True, 
        learn_energy=False):
        """
        X has shape (num_data, dim_data)
        """

        U,s,V = la.svd(X-X.mean(0,keepdims=True), full_matrices=False)

        these = s>eps
        if rank is not None:
            these[rank:] = False

        self.N = len(X)     # number of inputs
        self.r = sum(these) # input/output dimension
        self.h = dim_hid    # hidden dimension

        ## Hyperparameters
        self.tau = tau
        self.w_lr = w_lr
        if b_lr is None:
            self.b_lr = w_lr
        else:
            self.b_lr = b_lr

        ## Data
        self.P = U[:,these] # for random sampling
        self.Q = 1*self.P   # for energy function
        if not whiten:
            self.P = self.P@np.diag(s[these])

        self.X = U[:,these]@np.diag(s[these]) # unwhitened inputs
        self.xnorm = np.sum(self.X**2)

        ## Parameters
        # self.W = np.random.randn(self.h, self.r)/np.sqrt(self.r)
        # self.b = np.zeros(self.r)
        self.W = nn.Linear(self.h, self.r)
        self.Wopt = optim.Adam(self.W.parameters(), lr=w_lr)
        self.p = np.zeros((self.N, self.h))
        # self.p = np.random.randn(self.h)/np.sqrt(self.h)
        self.M = np.random.randn(self.r, self.h)/np.sqrt(self.r)

        self.updateQ = learn_energy

    def step(self, H, t=1, beta=1):
        "Apply parameter updates for t steps"

        for i in range(t):
            self.Wopt.zero_grad()
            S = 2*np.random.binomial(1, spc.expit(beta*H))-1
            # Xhat = S@self.W + self.b[None,:]
            Xhat = self.W(torch.FloatTensor(S))

            ## update input weights
            if self.updateQ:
                self.M = self.P.T@S
            else:
                self.M = self.Q.T@S

            ## update input intercept
            # dS = (0.5*Xhat - self.X)@self.W.T
            with torch.no_grad():
                dS = (0.5*Xhat.numpy() - self.X)@self.W.weight.numpy()  
            self.p = self.tau*dS/self.N - S.mean(0, keepdims=True) + 1e-2

            ## update linear layer
            # reg = torch.tensor(self.N - np.abs(S.sum(0)))
            # l2 = torch.sum(self.W.weight**2, 0)@(reg/reg.sum())
            # loss = nn.MSELoss()(Xhat, torch.FloatTensor(self.X)) + 1e-2*l2
            # loss.backward()
            # self.Wopt.step()
            # ## update output weights
            # dW = S.T@err/self.N
            # self.W -= self.w_lr*dW
            loss = 0

            # ## update output intercept
            # db = err.mean(0)
            # self.b -= self.b_lr*db

        # return loss.item()
        return loss

    def current(self):
        # if self.updateQ:
        H = self.P@self.M - self.p
        # else:
        #     H = self.Q@self.M - self.p
        return H

    def resample(self, these):
        "Random normal vector in Im(A)"
        self.M[:,these] = np.random.randn(self.r, np.sum(these))/np.sqrt(self.r)
        # self.p[:,these] = np.zeros((self.N, np.sum(these)))
        self.p[:,these] = np.random.randn(np.sum(these))/np.sqrt(self.h)
        # self.W[these] = np.random.randn(np.sum(these), self.r)/np.sqrt(self.r)

    def learn(self, s, eta=0.1):
        "Avoid sampling w in the future"
        self.P -= (eta/(s.shape[1]+1e-6))*s@s.T@self.P/self.N

    def energy(self, s):
        return 1 - (((self.Q.T@s)**2).sum(0)/self.N + s.mean(0)**2)

    def fit(self, max_iter=100, tol=0.1, lr=0.0, 
        alpha=0.88, beta=5, period=10, verbose=False):

        t = np.zeros(self.h)

        if verbose:
            pbar = tqdm(range(max_iter))

        H = self.current() # hidden
        S = np.sign(H)

        energy = np.empty((0, self.h))
        mse = []
        keep = np.zeros(self.h) > 0
        emin = np.inf
        it = 0
        while (it<max_iter):

            ens = self.energy(S)
            energy = np.vstack([energy, ens])

            temp = alpha**(t//period)
            loss = self.step(H, beta=beta/temp)
            mse.append(loss)

            H = self.current()
            newS = np.sign(H)

            ## Resample poorly converged neurons
            stopped = (np.sum(S*newS, axis=0) == self.N) 
            meh = energy[it] - emin > tol
            bad = (np.abs(np.sum(S, axis=0)) >= self.N)

            dead = bad|(stopped&meh)
            self.resample(dead)
            keep = keep | ((~meh)&stopped&(~bad))

            ## Update if necessary
            self.learn(S[:,stopped], eta=lr)
            S = newS
            t[dead] = 0
            t[~dead] += 1

            if np.sum(~bad)>0:
                emin = np.min([emin, np.min(ens[~bad])])

            if verbose:
                pbar.update(1)

            it += 1

        H = self.current()
        S = np.sign(H)
        Sout = S[:,keep]
        Sout, grp = np.unique(Sout*Sout[[0]], axis=1, return_inverse=True)
        # Wout = util.group_sum(self.W[keep], grp, axis=0)

        return Sout, energy, np.array(mse)


class KBAE:
    """
    Binary autoencoder with reconstruction loss, 
    but entirely in kernel space
    """

    def __init__(self, X, dim_hid, eps=1e-5, tau=1,
        pi_lr=1e-2, rank=None, whiten=True, learn_energy=False):
        """
        X has shape (num_data, dim_data)
        """

        U,s,V = la.svd(X-X.mean(0,keepdims=True), full_matrices=False)

        these = s>eps
        if rank is not None:
            these[rank:] = False

        self.N = len(X)     # number of inputs
        self.r = sum(these) # input/output dimension
        self.h = dim_hid    # hidden dimension

        ## Hyperparameters
        self.tau = tau
        self.pi_lr = pi_lr

        ## Data
        self.P = U[:,these] # for random sampling
        self.Q = 1*self.P   # for energy function
        if not whiten:
            self.P = self.P@np.diag(s[these])

        self.X = self.P@np.diag(s[these]) # unwhitened inputs
        self.xnorm = np.sum((self.X@self.X.T)**2)

        ## Parameters
        self.H = self.P@np.random.randn(self.r, self.h)
        self.H += + np.random.randn(1,self.h)/np.sqrt(self.h)
        self.pi = np.zeros(self.h)

        self.updateQ = learn_energy

    def step(self, t=1, beta=1):
        "Apply parameter updates for t steps"

        for i in range(t):
            S = 2*np.random.binomial(1, spc.expit(beta*self.H))-1

            Sbar = S - S.mean(0, keepdims=True)
            Khat = Sbar@np.diag(self.pi)@Sbar.T

            ## update currents
            if self.updateQ:
                self.H = self.P@(self.P.T@S) + S.mean(0, keepdims=True)
            else:
                self.H = self.Q@(self.Q.T@S) + S.mean(0, keepdims=True)

            ## add reconstruction term
            dS = 0.5*Khat@S - self.X@(self.X.T@S)
            self.H -= (self.tau/self.xnorm)*dS@np.diag(self.pi)

            ## update pi
            norm = ((Sbar.T@Sbar)**2)@self.pi
            dot = ((Sbar.T@self.X)**2).sum(1)
            reg = np.min([(S>0).mean(0), (S<0).mean(0)], axis=0)
            self.pi -= self.pi_lr*((norm - 2*dot)/self.xnorm + 1e-3*reg)
            self.pi = self.pi*(self.pi > 0)

        return (self.pi@(norm-2*dot))/self.xnorm

    def resample(self, these):
        "Random normal vector in Im(A)"
        self.H[:,these] = self.P@np.random.randn(self.r, np.sum(these))
        self.pi[these] = 0

    def learn(self, s, eta=0.1):
        "Avoid sampling w in the future"
        self.P -= (eta/(s.shape[1]+1e-6))*s@s.T@self.P/self.N

    def energy(self, s):
        return 1 - (((self.Q.T@s)**2).sum(0)/self.N + s.mean(0)**2)

    def fit(self, max_iter=100, tol=0.1, lr=0.0, 
        alpha=0.88, beta=1, period=10, verbose=False):

        t = np.zeros(self.h)

        if verbose:
            pbar = tqdm(range(max_iter))

        S = np.sign(self.H)

        energy = np.empty((0, self.h))
        mse = []
        emin = np.inf
        it = 0
        while (it<max_iter):

            ens = self.energy(S)
            energy = np.vstack([energy, ens])

            temp = alpha**(t//period)
            loss = self.step(beta=beta/temp)
            mse.append(loss+1)

            newS = np.sign(self.H)

            ## Resample poorly converged neurons
            stopped = (np.sum(S*newS, axis=0) == self.N) 
            meh = energy[it] - emin > tol
            bad = (np.abs(np.sum(S, axis=0)) >= self.N)

            # dead = bad|(stopped&(meh|(self.pi<1e-3)))
            dead = bad|(stopped&meh)
            self.resample(dead)

            ## Update if necessary
            self.learn(S[:,stopped], eta=lr)
            S = newS
            t[dead] = 0
            t[~dead] += 1

            if np.sum(~bad)>0:
                emin = np.min([emin, np.min(ens[~bad])])

            if verbose:
                pbar.update(1)

            it += 1

        S = np.sign(self.H)
        en = self.energy(S)
        keep = en-en.min() < tol
        Sout = S[:,keep]
        Sout, grp = np.unique(Sout*Sout[[0]], axis=1, return_inverse=True)
        piout = util.group_sum(self.pi[keep], grp)

        return Sout, piout, energy, np.array(mse)


#############################################################################
########################## old stuff ########################################
#############################################################################


# def fit_x(S, pi, d, c=None):

#     n = len(d)

#     catmat = np.hstack([d[:,None], S@np.diag(pi)]) # I hate this 'None' shit

#     ## Average gradient of the nth moment

#     if c is None:
#         c = np.eye(len(S.T)+1)[0]

#     prog = lp(c=c, 
#               A_eq=catmat,
#               b_eq=np.ones(n), 
#               bounds=[[0, None]] + [[-1,1]]*len(pi))
    
#     return prog.x[1:], prog.x[0]


# def fit_pi(S, x, D):
#     """
    
#     """

#     n = len(S)

#     ## Increase scale to maximum permissible
#     S_kron = np.vstack([util.vec(util.outers(S)).T, S@np.diag(x)])
#     intcpt = np.repeat([[1],[-1]], [(n**2 - n)//2, n], axis=0)
#     S_intcpt = np.hstack([S_kron, intcpt])

#     vecD = util.vec(D)

#     A = np.block([[np.ones((1,len(S.T)+1)), 0],
#                   [S_intcpt, vecD[:,None] ] ])

#     prog = lp(c=-np.eye(len(A.T))[-1],
#               A_eq=A,
#               b_eq=np.ones(len(A)),
#               bounds=[0,1])
#     if prog.x is None:
#         raise DidntWorkError

#     new_pi = prog.x[:-1]
#     scale = prog.x[-1]

#     return new_pi, scale


# def fit_x_pi(S, D, c=None, eps=0):

#     n = len(D)-1
#     ns = len(S.T)

#     if c is None:
#         c = np.eye(2*(ns+1))[-1]
#         c += np.random.randn(2*(ns+1))*eps
#         # c = np.random.randn(2*(ns+1))

#     vecD = np.concatenate([util.vec(D[:n,:n]), D[:n,n]])
#     S_kron = util.vec(util.outers(S)).T

#     A_eq = la.block_diag(S_kron, S)
#     A_eq = np.block([[np.ones(len(S_kron)), -np.ones(n)], [A_eq.T], [vecD]]).T
#     A_eq = np.block([[A_eq], 
#                      [np.ones(ns+1), np.zeros(ns), 0]])
#     b_eq = np.ones(len(A_eq))

#     o = np.zeros((ns,1))
#     A_ub = np.block([[o, -np.eye(ns), -np.eye(ns), o],
#                      [o, -np.eye(ns), np.eye(ns), o]])
#     b_ub = np.zeros(len(A_ub))

#     bounds = [[0,None]]*(ns+1) + [[None,None]]*ns + [[0,None]]

#     prog = lp(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, 
#         bounds=bounds, method='highs')

#     if prog.x is None:
#         raise DidntWorkError

#     pi_fit = prog.x[:ns+1]
#     x_fit = prog.x[ns+1:-1]
#     a_fit = prog.x[-1]

#     return pi_fit, x_fit, a_fit


# def split_feats(S, pi, x):

#     keep = np.abs(pi) >= 1e-6
#     split = pi - np.abs(x) >= 1e-6     # fractional membership
#     n_new = np.sum(split&keep)

#     new_pi = np.concatenate([pi[(~split)&keep], 
#                              0.5*(pi[split&keep] + x[split&keep]), 
#                              0.5*(pi[split&keep] - x[split&keep])])

#     new_S = np.block([[S[:,(~split)&keep], S[:,split&keep], S[:,split&keep] ], 
#                       [np.sign(x[(~split)&keep]), np.ones(n_new), -np.ones(n_new)]])

#     return new_S, new_pi


# def oldcuts(K, chains=12, verbose=True, fixed_order=False, chain_eps=0.5):
#     """
#     Compute the cut decomposition, tentative name, of kernel K
#     """

#     ## useful definitions
#     zero_tol = 1e-6

#     s2 = np.array([[1,1],[-1,1]]) # base case

#     N = len(K)
#     inds = util.LexOrder()

#     d = 1 - K # squared distances

#     ## Choose order of items
#     if fixed_order:
#         order = np.arange(N)
#     else:
#         order = item_order(d, sign=1)
#     inv_order = np.argsort(order)

#     if verbose:
#         iterator = tqdm(range(3,N))
#     else:
#         iterator = range(3,N)

#     d = d[order,:][:,order]

#     ## Fit features
#     # S = 2*np.eye(3) - 1
#     S_r = [2*np.eye(3) - 1]*(1+chains) # random feature bank

#     for n in iterator:

#         # try:   
#             # pi, x, scale = fit_x_pi(S, d[:n+1,:n+1])
#             # new_S, new_pi = split_feats(S, pi[1:], x)

#         extinct = []
#         survivors = []
#         S_r_survived = []
#         for i in range(chains+1):
#             try:
#                 if i == 0:
#                     pi, x, scale = fit_x_pi(S_r[i], d[:n+1,:n+1])
#                     new_S_r, new_pi = split_feats(S_r[i], pi[1:], x)
#                 else:
#                     pi_r, x_r, scale_r = fit_x_pi(S_r[i], d[:n+1,:n+1], eps=chain_eps)
#                     new_S_r, _ = split_feats(S_r[i], pi_r[1:], x_r)

#                 S_r_survived.append(S_r[i]) 
#                 survivors.append(i)

#                 S_r[i] = np.block([(2*np.eye(n+1)-1)[:,[-1]], new_S_r])

#             except DidntWorkError:
#                 # exctinction
#                 extinct.append(i)
#                 # S_r[i] = np.hstack([S, S_r[i]])
#                 # pi_r, x_r, scale_r = fit_x_pi(S_r[i], d[:n+1,:n+1], eps=chain_eps)

#         # except:

#             # mass extinction

#         if len(extinct) > 0:
#             print('%d lines are extinct'%len(extinct))
#             if len(survivors) > 0:
#                 for i in extinct:
#                     if i == 0:
#                         S_cat = np.unique(np.hstack(S_r_survived), axis=1)
#                         pi, x_r, scale_r = fit_x_pi(S_cat, d[:n+1,:n+1])
#                         new_S_r, new_pi = split_feats(S_cat, pi[1:], x_r)
#                         S_r[i] = np.block([(2*np.eye(n+1)-1)[:,[-1]], new_S_r])
#                     else:
#                         j = np.random.choice(survivors)
#                         S_r[i] = S_r[j]

#             else:
#                 print('Mass extinction!')
#                 S_cat = np.unique(np.hstack(S_r), axis=1)

#                 for i in extinct:

#                     # pi, x, scale = fit_x_pi(S_cat, d[:n+1,:n+1])
#                     # new_S, new_pi = split_feats(S_cat, pi[1:], x)

#                     # for i in range(chains):
#                     if i == 0:
#                         pi, x_r, scale_r = fit_x_pi(S_cat, d[:n+1,:n+1])
#                         new_S_r, new_pi = split_feats(S_cat, pi[1:], x_r)
#                     else:
#                         pi_r, x_r, scale_r = fit_x_pi(S_cat, d[:n+1,:n+1], eps=chain_eps)
#                         new_S_r, _ = split_feats(S_cat, pi_r[1:], x_r)
                    
#                     S_r[i] = np.block([(2*np.eye(n+1)-1)[:,[-1]], new_S_r])

#         # print(new_S)
#         # S = np.block([(2*np.eye(n+1)-1)[:,[-1]], new_S])
#         pi = np.concatenate([[pi[0]], new_pi])

#     S_out = S_r[0][inv_order, :]

#     S_out = S_out[:,pi>zero_tol]
#     pi_out = pi[pi>zero_tol]

#     S_out = S_out[:, np.argsort(-pi_out)]
#     pi_out = pi_out[np.argsort(-pi_out)]

#     return S_out, pi_out


#############################################################################
################## Binary version ###########################################
#############################################################################

# def fit_bx(B, pi, k, eps=0):

#     prog = lp(c=pi, A_eq=B@np.diag(pi), b_eq=k, bounds=[0,1])

#     return prog.x

# def split_bfeats(B, pi, x):

#     has = x > 1e-6
#     strict = np.abs(x - x**2) <= 1e-6   # non-binary cuts

#     is_zero = strict&has 
#     is_one = strict&(~has)
#     split = ~strict

#     new_pi = np.concatenate([pi[is_zero], pi[is_one], 
#                                 pi[split]*x[split], pi[split]*(1-x[split])])
#     new_B = np.block([[B[:,is_zero], B[:,is_one], B[:,split], B[:,split]],
#                       [np.zeros(np.sum(is_zero)), np.ones(np.sum(is_one)), np.zeros(np.sum(split)), np.ones(np.sum(split)) ]])


#     return new_B, new_pi

# def BDF(K, chains=12, verbose=True, fixed_order=False, chain_eps=0.5):
#     """
#     Compute the cut decomposition, tentative name, of kernel K
#     """

#     ## useful definitions
#     zero_tol = 1e-6

#     s2 = np.array([[1,1],[-1,1]]) # base case

#     N = len(K)
#     inds = util.LexOrder()

#     d = 1 - K # squared distances

#     ## Choose order of items
#     if fixed_order:
#         order = np.arange(N)
#     else:
#         order = item_order(d, sign=1)
#     inv_order = np.argsort(order)

#     if verbose:
#         iterator = tqdm(range(3,N))
#     else:
#         iterator = range(3,N)

#     d = d[order,:][:,order]

#     ## Use first data point as the 'origin'
#     K_o = (d[[0],:] + d[:,[0]] - d)/4


    
# def oldBDF(K, sparse=True, in_cut=False, num_samp=None, 
#     zero_tol=1e-6, fixed_order=False):
#     """
#     Binary distance factorization (working name) of K. Using the non-integer 
#     formulation.
#     """
    
#     N = len(K)
#     inds = util.LexOrder()
#     sgn = (-1)**(sparse + 1) # + if sparse else -
    
#     ## "Project" into cut polytope, if it isn't already there
#     if not in_cut:
#         if num_samp is None:
#             C_ = gauss_projection(util.correlify(K))
#         else:
#             samps = np.sign(np.random.multivariate_normal(np.zeros(N), K, size=num_samp))
#             C_ = samps.T@samps/num_samp
#     else:
#         C_ = K
    
#     d = 1 - C_
    
#     if sparse:
#         alpha = np.sum(d)/np.sum(d**2)   # distance scale which minimizes correlations
#         # alpha = (1+np.max(d))/(N/2) # unproven: this is largest alpha which returns sparsest solution?
#     else:
#         if not np.mod(N, 2):
#             alpha = (N**2)/(d.sum()) # project onto balanced dichotomies
#     alpha = 1
    
#     C = 1 - alpha*d
    
#     if fixed_order:
#         orig=0
#     else:
#         orig = np.argmax(-1*sgn*np.sum(d, axis=0))
    
#     ### center around arbitrary (?) point
#     K_o = (C[orig,orig] - C[[orig],:] - C[:,[orig]] + C)/4
    
#     idx = np.arange(N)
#     if fixed_order:
#         first=1
#     else:
#         first = np.setdiff1d(idx, [orig])[np.argmax(-1*sgn*d[orig,:][np.setdiff1d(idx, [orig])])]

#     B = np.ones((1,1), dtype=int)
#     pi = np.array([K_o[first,first]])
    
#     included = [first]
#     remaining = np.setdiff1d(idx, [orig, first]).tolist()
    
#     while len(remaining) > 0:
#     # for n in tqdm(range())

#         ## Most "explainable" item <=> closest/furthest item
#         if fixed_order:
#             this_i = len(included)+1
#         else:
#             this_i = remaining[np.argmax(-1*sgn*np.sum(d[included,:][:,remaining], axis=0))]

#         ### new features
#         prog = lp(sgn*np.ones(len(pi)),
#                   A_ub=np.ones((1,len(pi))), b_ub=K_o[this_i,this_i],
#                   A_eq=B, b_eq=K_o[included,this_i],
#                   bounds=[[0,p] for p in pi],
#                   method='highs')
        
#         if prog.success:
#             x = prog.x
#         else:
#             raise RuntimeError('Inconsistency at %d items'%len(included))
        
#         ## Split clusters as necessary 
#         has = x > zero_tol
#         split = x < pi
        
#         new_B = np.block([[B, B[:,has&split]], 
#                           [has, np.zeros(sum(has&split))]])
#         new_pi = np.concatenate([np.where(has, x, pi), pi[has&split] - x[has&split]])
        
#         resid = K_o[this_i, this_i] - np.sum(x) # create new symbols to make up the distance
#         if resid > 0:
#             new_B = np.hstack([new_B, np.eye(len(new_B))[:,[-1]]])
#             new_pi = np.concatenate([new_pi, [resid]])

#         B = new_B
#         pi = new_pi
        
#         ### Discard clusters (find a good way to do this)
#         # S_unq, num_s = cull(S_unq, num_s, thresh=thresh)
        
#         included.append(this_i)
#         remaining.remove(this_i)
    
#     ## Fix the matrix
#     B = np.vstack([np.zeros(len(pi)), B])
#     trivial = (B.sum(0) == N) | (B.sum(0) == 0)
#     B = B[:,~trivial]
#     pi = pi[~trivial]/np.sum(pi[~trivial])
    
#     feat_idx = np.argsort(-pi)
    
#     ### fill in remaining difference vectors 
#     idx = np.append(orig, included)
#     idx_order = np.argsort(idx)
    
#     B = B[idx_order][:,feat_idx]
#     pi = pi[feat_idx]
    
#     B_full = np.vstack([np.mod(B[(i+1):] + B[i], 2) for i in range(N)])
    
#     pt_id = idx[idx_order]
#     ix = inds(np.concatenate([pt_id[(i+1):] for i in range(N)]), 
#               np.concatenate([np.ones(N-i-1)*i for i in range(N)]))
    
#     ### convert to 'canonical' form
#     vecs, pts = color_points(B_full, ix)
    
#     return vecs.T, pi


