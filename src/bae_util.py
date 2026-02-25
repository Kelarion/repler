import os, sys, re
import pickle
from time import time
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.distributions as dis
import torch.linalg as tla
import numpy as np
from itertools import permutations, combinations
from tqdm import tqdm

import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import numpy.linalg as nla
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

from sklearn import svm

from numba import njit
import math

# my code
import util
import df_util 

#############################################################
###### Annealing ############################################
#############################################################

@dataclass
class Neal:
    """
    Annealing scheduler
    """

    decay_rate: float = 0.8
    period: int = 2
    initial: float = 10.0

    def fit(self, model, *data, T_min=1e-4, max_iter=None, verbose=True, **opt_args):

        if max_iter is None:
            max_iter = self.period*int(np.log(T_min/self.initial)/ np.log(self.decay_rate))

        if verbose:
            pbar = tqdm(range(max_iter))

        en = []
        # mets = []
        model.initialize(*data, **opt_args)
        for it in range(max_iter):
            T = self.initial*(self.decay_rate**(it//self.period))
            model.temp = T
            en.append(model.grad_step(*data))
            # mets.append(model.metrics(*data))

            if verbose:
                pbar.update(1)

        return en # , mets

#     def cv_fit(self, model, X, T_min=1e-4, max_iter=None, verbose=False , 
#         draws=10, folds=10, **opt_args):

#         if max_iter is None:
#             max_iter = self.period*int(np.log(T_min/self.initial)/ np.log(self.decay_rate))

#         if verbose:
#             pbar = tqdm(range(draws))

#         ens = []
#         mods = []
#         for fold in range(draws):
            
#             model.initialize(X)

#             ## Mask
#             M = np.random.rand(*X.shape) < (1/folds)

#             ## Initialize masked values at random
#             X_M = X*1
#             X_M[M] = np.random.randn(M.sum())

#             ## Fit parameters and mask
#             for it in range(max_iter):
#                 T = self.initial*(self.decay_rate**(it//self.period))
#                 model.grad_step(X_M, T, **opt_args)
#                 X_M[M] = model()[M]

#             ens.append(model.loss(X, mask=M))
#             mods.append(model.S*1)

#             if verbose:
#                 pbar.update(1)

#         return ens, mods



#############################################################
###### Model comparison #####################################
#############################################################

def bicv(model, X, train_items, train_feats, n_iter=1, **opt_args):
    """
    Bi-crossvalidation (Owen and Perry, 2009; Fu and Perry, 2017)
    """

    A = X[train_items][:,train_feats]
    B = X[~train_items][:,train_feats]
    C = X[train_items][:,~train_feats]
    D = X[~train_items][:,~train_feats]

    ## Fit W and S from A
    _ = model.fit(A, verbose=False, **opt_args)
    S = 1*model.S       # model state: (S, W)
    ls = model.loss(A)

    ## Fit S' from W and B
    model.init_latents(B)   # re-initialize S
    for _ in range(n_iter):
        S_ = model.EStep(B) # model state: (S', W)

    ## Fit W' from S and C
    model.init_params(C)    # re-initialize (W,b)
    _ = model.MStep(S, C)   # model state: (S', W')

    ## Residual between S'W' and D
    return ls, model.loss(D) 


def fpcv(model, X, train_items, train_feats, classifier=svm.LinearSVC, **opt_args):

    A = X[train_items][:,train_feats]
    B = X[~train_items][:,train_feats]
    C = X[train_items][:,~train_feats]
    D = X[~train_items][:,~train_feats]

    ## Fit model on A
    _ = model.fit(A, verbose=False, **opt_args)
    S = 1*model.S
    ls = model.loss(A)

    ## Train classifier to predict S(A) from C
    k = S.shape[1]
    d = C.shape[1]

    W = np.zeros((k,d))
    b = np.zeros(k)
    for i in range(k):
        if S[:,i].sum() > 0:
            clf = classifier()
            clf.fit(C, S[:,i])

            W[i] = 1*clf.coef_[0]
            b[i] = 1*clf.intercept_[0]

    ## Predict labels for heldout points
    Spred = 1*(D@W.T + b > 0)

    ## Evaluate
    model.S = Spred
    return ls, model.loss(B)

def impcv(model, X, mask='random', folds=10, 
            initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=False, **init_args):
    """
    Imputation-based cross validation 

    the `fold` of the CV is the fraction of the data masked

    optimizer should already be initialized
    """

    if max_iter is None:
        max_iter = period*int(np.log(min_temp/initial_temp)/np.log(decay_rate))

    if verbose:
        pbar = tqdm(range(max_iter))

    n,d = X.shape
    idx = np.arange(n*d).reshape((n,d))
    ntest = int(n*d/folds)

    Z = 1*X

    these_idx = np.random.choice(idx.flatten(), ntest, replace=False)
    M = np.isin(idx, these_idx)

    ## Initialize masked values from `empirical distribution`
    Z[M] = np.random.choice(Z[~M], M.sum())

    model.initialize(Z, **init_args)
    en = []
    for it in range(max_iter):
        T = initial_temp*(decay_rate**(it//period))
        model.temp = T
        en.append(model.grad_step(Z))
        Z[M] = model()[M]

        if verbose:
            pbar.update()

    pred = model()
    test = np.mean((X[M] - pred[M])**2)/np.mean(X[M]**2)
    train = np.mean((X[~M] - pred[~M])**2)/np.mean(X[~M]**2)

    return train, test

def kerimpcv(model, X, mask='random', folds=10, diag=False,
            initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=False, **init_args):
    """
    Imputation-based cross validation 

    the `fold` of the CV is the fraction of the data masked

    optimizer should already be initialized
    """

    if max_iter is None:
        max_iter = period*int(np.log(min_temp/initial_temp)/np.log(decay_rate))

    if verbose:
        pbar = tqdm(range(draws))

    n,d = X.shape
    tot = spc.binom(n,2) + n*diag
    idx = util.LexOrder(diag=diag)

    ntest = int(tot/folds)

    K = util.center(X@X.T)
    Z = 1*K

    M = np.zeros((n,n)) > 0
    these_idx = np.random.choice(np.arange(tot), ntest, replace=False)
    M[idx.inv(these_idx)] = True

    ## Initialize masked values from `empirical distribution`
    Z[M] = np.random.choice(Z[~M], M.sum())
    Z[M.T] = Z[M]
    M = M + M.T # symmetrize

    model.initialize(util.center(Z), **init_args)
    en = []
    for it in range(max_iter):
        T = initial_temp*(decay_rate**(it//period))
        model.temp = T
        Z[M] = model()[M]
        en.append(model.grad_step(util.center(Z)))

    pred = model()
    test = np.mean((K[M] - pred[M])**2)/np.mean(K[M]**2)
    train = np.mean((K[~M] - pred[~M])**2)/np.mean(K[~M]**2)

    return train, test

def kercv(model, X, train_items, 
            initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=False, **init_args):
    """
    Cross validation procedure for the kernel MSE

    the `fold` of the CV is the fraction of the data masked
    """

    if max_iter is None:
        max_iter = period*int(np.log(min_temp/initial_temp)/np.log(decay_rate))

    if verbose:
        pbar = tqdm(range(draws))

    n,d = X.shape

    model.initialize(X, **init_args)
    
    Strn = model.S[train_items]
    Stst = model.S[~train_items]

    Xtrn = X[train_items]
    Xtst = X[test_items]

    # These are censored when inferring on test set
    Jtst = Stst.T@Stst
    Wtst = Stst.T@Xtst

    en = []
    for it in range(max_iter):
        T = initial_temp*(decay_rate**(it//period))
        model.temp = T

        ## Train set
        Strn = model.EStep(Xtrn, S0=Strn)

        model.StS -= Jtst # remove information from test set
        model.StX -= Wtst

        Stst = model.EStep()
        

        en.append(model.grad_step(Z))
        Z[M] = model()[M]

    pred = model()
    test = np.mean((X[M] - pred[M])**2)/np.mean(X[M]**2)
    train = np.mean((X[~M] - pred[~M])**2)/np.mean(X[~M]**2)

    return train, test

def multifit(model, X, chains=10, **neal_args):

    Xpr = []
    ls = []
    # neal = Neal(**neal_args)
    for it in tqdm(range(chains)):
        en = model.fit(X, **neal_args)
        # en = neal.fit(model, X, verbose=False)
        ls.append(en)
        Xpr.append(model())

    return np.mean(Xpr, axis=0), np.mean(ls, axis=0)

def splitfit(model, X, draws=100, **neal_args):

    N = len(X)

    neal = Neal(**neal_args)

    trn_loss = []
    tst_loss = []
    X_pred = np.zeros(X.shape)
    for draw in tqdm(range(draws)):

        idx = np.random.permutation(range(N))
        A = idx[:N//2]
        B = idx[N//2:]

        enA = neal.fit(model, X[A], verbose=False)
        ApredB = (X[B] - model.b)@model.W/model.scl
        SA = model.S*1
        WA = model.W*1
        sclA = model.scl*1
        bA = model.b*1

        enB = neal.fit(model, X[A], verbose=False)
        SB = model.S*1
        WB = model.W*1
        sclB = model.scl*1
        bB = model.b*1
        
        ## Predict heldout latents
        ApredB = (X[B] - bA)@WA/sclA
        BpredA = (X[A] - bB)@WB/sclB
        AhamB = df_util.permham(SB, 1*(ApredB > 0.5))
        BhamA = df_util.permham(SA, 1*(BpredA > 0.5))

        ## Predict heldout data
        X_pred[A] += (sclB*SA@WB.T + bB)/draws
        X_pred[B] += (sclA*SB@WA.T + bA)/draws

        trn_loss.append((enA[-1] + enB[-1])/2)
        tst_loss.append((AhamB.mean() + BhamA.mean())/2)

    return X_pred, trn_loss, tst_loss


#############################################################
###### Gradient descent #####################################
#############################################################


# class StiefelOptimizer:

#   def __init__(self, method='exact', batch_size=64, **sgd_args):

#       


#############################################################
###### Log-normalizers ######################################
#############################################################

@njit
def gaussian(nat):
    return 0.5*nat**2

@njit
def poisson(nat):
    return np.exp(nat)

@njit
def bernoulli(nat):
    return np.log(1+np.exp(nat))

@njit
def enby(nat):
    return -np.log(1-np.exp(nat))

#############################################################
###### Custom distributions #################################
#############################################################

## To Do:
class Enby(dis.ExponentialFamily):
    """
    Natural parameterisation of the negative binomial distribution, with
    fixed r (number of trials) parameters to allow exponential family form
    """

    def __init__(self, total_trials, log_prob):

        self.r = total_trials # hyperparameter
        self.eta = log_prob   # natural parameter 

    @property
    def _natural_params(self):
        """
        Abstract method for natural parameters. Returns a tuple of Tensors based
        on the distribution
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        """
        Abstract method for log normalizer function. Returns a log normalizer based on
        the distribution and input
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
        raise NotImplementedError

#################################################################
########### Sampling latents ####################################
#################################################################


