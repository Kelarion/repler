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
import math

# my code
import util
import pt_util
import bae_util

####################################################################
############ Base PyTorch module for BAE ###########################
####################################################################

class BAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.initialized = False

    def EStep(self, X):
        """
        Update of the discrete parameters
        """
        return NotImplementedError

    def MStep(self, X):
        """
        Update of the continuous parameters
        """
        return NotImplementedError

    def grad_step(self, X, temp, opt_alg=geo.optim.RiemannianSGD, **opt_args):
        """
        One iteration of optimization
        """

        if not self.initialized:
            self.init_optimizer(opt_alg=opt_alg, **opt_args)

        E = self.EStep(X, temp)
        loss = self.MStep(X, E)

        return loss

    def init_optimizer(self, opt_alg=geo.optim.RiemannianSGD, **opt_args):
        self.optimizer = opt_alg(self.parameters(), **opt_args)
        self.initialized = True


####################################################################
################# Specific instances ###############################
####################################################################

class GaussBAE(BAE):
    """
    Simplest BAE, with Gaussian observations and orthogonal weights

    Setting
        `weight_alg = 'exact', tree_reg=0`
    results in fast, closed-form updates, but will have poor performance
    on non-identifiable instances. 
    """

    def __init__(self, num_inp, dim_inp, dim_hid, weight_alg='exact', 
        tree_reg=0, sparse_reg=1e-2):

        super().__init__()

        self.n = num_inp
        self.d = dim_inp
        self.r = dim_hid

        self.alpha = sparse_reg
        self.beta = tree_reg
        self.weight_alg = weight_alg
        if weight_alg not in ['exact', 'sgd']:
            raise ValueError('weight_alg must be either "exact" or "sgd"')

        ## Parameters
        W = sts.ortho_group.rvs(dim_inp)[:,:dim_hid]
        self.W = geo.ManifoldParameter(torch.FloatTensor(W), geo.Stiefel())
        self.b = geo.ManifoldParameter(torch.zeros(dim_inp), geo.Euclidean())
        self.scl = geo.ManifoldParameter(torch.ones(1), geo.Euclidean())

        ## Latents
        

        ## Latent distribution
        self.bern = dis.bernoulli.Bernoulli

    def forward(self, X, temp=1e-4, size=[]):
        return self.bern(logits=((X-self.b)@self.W)/temp).sample(size)

    def EStep(self, X, temp):
        """
        Compute expectation of log-likelihood over S 

        X is a FloatTensor of shape (num_inp, dim_inp)
        """

        ## Input signal (without regularization this is the whole step)
        C = 2*(X - self.b)@self.W - self.scl 
        if self.beta < 1e-6:
            return torch.special.expit(C/temp)

        ## Regularization 
        ES = torch.zeros(C.shape) 

        S = self.bern(logits=C/temp).sample()
        StS = S.T@S     
        St1 = S.sum(0)

        n = self.n - 1
        for it in np.random.permutation(range(self.n)):
            ## Current row
            s = S[it]

            ## Energy
            StS -= torch.outer(s,s)
            St1 -= s

            ## Make recurrent weight matrix
            D1 = StS
            D2 = St1[None,:] - StS
            D3 = St1[:,None] - StS
            D4 = n - St1[None,:] - St1[:,None] + StS

            deez = torch.stack([D1,D2,D3,D4])
            best = 1*(deez == deez.min(0)[0]).float()
            best *= (best.sum(0)==1)

            Q = best[0] - best[1] - best[2] + best[3]
            p = (best[1].sum(0) - best[3].sum(0))

            ## Dynamics 
            for i in np.random.permutation(range(self.r)):

                inhib = Q[i]@s + p[i]

                ES[it,i] = torch.special.expit((C[it,i] - self.beta*inhib)/temp)
                s[i] = self.bern(logits=ES[it,i]).sample()

            StS += torch.outer(s,s)
            St1 += s

            S[it] = s

        return ES
        # return S

    def MStep(self, X, ES):

        M = (X-self.b).T@ES
        if self.weight_alg == 'exact': ## Closed-form updates
            U,s,V = tla.svd(M, full_matrices=False)

            self.W.data = U@V
            self.scl.data = torch.sum(s)/torch.sum(ES)
            self.b.data = (X - self.scl*ES@self.W.T).mean(0)

            loss = 1 - torch.sum(s)/torch.sqrt(torch.sum(ES)*torch.trace((X-self.b).T@(X-self.b)))

        elif self.weight_alg == 'sgd':

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.scl*ES@self.W.T + self.b, X)
            loss.backward()
            self.optimizer.step()

            # ls = loss.item()

        return loss.item()


# class GeBAE(BAE):
#     """
#     Generalized BAE, taking any exponential family observation
#     """

#     def __init__(self, num_inp, dim_inp, dim_hid, obs=dis.normal.Normal, tree_reg=0):



