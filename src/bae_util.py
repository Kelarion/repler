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
    
    def __post_init__(self):
        self.t = 0
    
    def __call__(self, freeze=False):
        T =  self.initial*(self.decay_rate**(self.t//self.period))
        if not freeze:
            self.t += 1
        return T

    def fit(self, model, *data, T_min=1e-4, verbose=True, **opt_args):

        max_iter = self.period*int(np.log(T_min)/ np.log(self.decay_rate))

        en = []
        for it in tqdm(range(max_iter)):
            T = self()
            en.append(model.grad_step(*data, T, **opt_args))

        return en

#############################################################
###### Model comparison #####################################
#############################################################


def impcv(model, mask='random', folds=10, iters=100, draws=10):
    """
    Imputation-based cross validation 

    the `fold` of the CV is the fraction of the data masked

    optimizer should already be initialized
    """

    ens = []
    for fold in range(draws):
        ## Mask
        M = np.random.rand(*model.X.shape) < (1/folds)

        ## Initialize masked values at random
        X_orig = model.X*1
        model.X[M] = np.random.randn(M.sum())

        for it in range(iters):
            model.grad_step()
            model.X[M] = model.predict()[M]

        model.X = X_orig
        ens.append(model.energy())

    return ens

# def bcv(model, fold=1, iters=10):



# def pdmask(n,d,folds):
#     """
#     pseudo-diagonal mask for cross-validation (Wold 1978 Technometrics)
#     """




#############################################################
###### Gradient descent #####################################
#############################################################


# class StiefelOptimizer:

#   def __init__(self, method='exact', batch_size=64, **sgd_args):

#       


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


