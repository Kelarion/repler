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
import pt_util
import bae_util
import bp_search
import students


####################################################################
################ Nonparametric models ##############################
####################################################################


####################################################################
################ Parametric models #################################
####################################################################


@dataclass(eq=False)
class ConceptBottleneck(students.NeuralNet):
    """
    A feedforward neural network, with binary bottleneck layer

    Yhat = W f[MX] + b

    which minimizes a mixture of the target error and input reconstruction

    """
    
    dim_inp: int
    dim_hid: int
    dim_out: int
    noise: str =  'gaussian'
    tied_weights: bool = True           # Tie forward and backward weights
    beta: float = 1.0                   # Strength of output loss
    temp: float = 1.0 
    tree_reg: float = 0
    sparse_reg: float = 0
    weight_reg: float = 0

    def __post_init__(self):
        super().__init__()

        # self.q = nn.Linear(self.dim_inp, self.dim_hid) # x -> s
        # self.p = nn.Linear(self.dim_hid, self.dim_out) # s -> y
        self.W = nn.Parameter(torch.empty(self.dim_out, self.dim_hid))    # s -> y
        self.b = nn.Parameter(torch.zeros(self.dim_out))
        if self.tied_weights                                            
            self.U = self.W.T
            self.d = self.b
        else:
            self.U = nn.Parameter(torch.empty(self.dim_hid, self.dim_out)) 
            self.d = nn.Parameter(torch.zeros(self.dim_out))

        self.V = nn.Parameter(torch.empty(self.dim_inp, self.dim_hid))    # x -> z
        self.q = nn.Parameter(torch.zeros(self.dim_hid))
        if self.tied_weights
            self.M = self.V.T                                       # s -> x'
            self.p = self.q
        else:
            self.M = nn.Parameter(torch.empty(self.dim_hid, self.dim_inp))
            self.p = nn.Parameter(torch.zeros(self.dim_hid))

        self.init_weights()

        self.Cov = torch.zeros((self.dim_hid, self.dim_hid))

        if self.noise == 'gaussian':
            self.lognorm = bae_util.gaussian
            self.obj = nn.MSELoss()

        elif self.noise == 'poisson':
            self.lognorm = bae_util.poisson
            self.obj = nn.PoissonNLLLoss()

        elif self.noise == 'bernoulli':
            self.lognorm = bae_util.bernoulli
            self.obj = nn.BCEWithLogitsLoss()

    def initialize(self, dl, **opt_args):
        """
        Input should be a dataloader for the data, same as the input to grad_step

        Eventually, 'N' will be a hyperparmeter whose default is some large number,
        but that's not something I'm implementing yet
        """
        self.N = len(dl.dataset)
        self.init_optimizer(**opt_args)

        self.tree_lr = dl.batch_size/self.N

    def forward(self, X):
        # return self.p((torch.sign(self.q(X))+1)/2)
        return torch.sigmoid((X@self.V+self.p)/self.temp)@self.W + self.b
    
    def hidden(self, X):
        return 1*(X@self.V+self.p > 0)

    def metrics(self, dl):
        pass

    def loss(self, batch):

        Z = batch[0]@self.V+self.p

        ## Search over S
        S = self.EStep(batch[0], Z, batch[1])

        ## Update continuous parameters
        qls = nn.BCEWithLogitsLoss()(Z, S)
        pls = self.MStep(S, batch[1])

        return pls + qls 

    def EStep(self, X, Z, Y):

        with torch.no_grad():
            
            S = 1.0*(Z > 0)
            Yhat = S@self.W.T + self.b
            Xhat = (S - self.p)@self.M

            if Z.device.type == 'cpu':
            # Convert to numpy since that's what Numba accepts
                Xnp = X.numpy().astype(float)
                Ynp = Y.numpy().astype(float)
                Xhat = Xhat.numpy().astype(float)
                Yhat = Yhat.numpy().astype(float)
                W = self.W.data.numpy().astype(float)
                M = self.M.data.numpy().astype(float)
                Snp = S.data.numpy().astype(float) 
                StS = self.Cov.numpy().astype(float)
            else:
                Xnp = X.cpu().numpy().astype(float)
                Ynp = Y.cpu().numpy().astype(float)
                Xhat = Xhat.cpu().numpy().astype(float)
                Yhat = Yhat.cpu().numpy().astype(float)
                W = self.W.data.cpu().numpy().astype(float)
                M = self.M.data.cpu().numpy().astype(float)
                Snp = S.data.cpu().numpy().astype(float) 
                StS = self.Cov.cpu().numpy().astype(float)

            newS = bp_search.gecbm(
                Xhat=Xhat, X=Xnp, W=W,                      # inputs
                Yhat=Yhat, Y=Ynp, M=M,                      # outputs
                S=Snp, StS=StS, gamma=self.beta,            # latents
                alpha=self.sparse_reg, beta=self.tree_reg,  # regualarization
                temp=self.temp, lognorm=self.lognorm        # noise distribution
                )                            
            newCov = (1-self.tree_lr)*StS + self.tree_lr*newS.T@newS/len(newS)

            newS = torch.tensor(newS, dtype=Z.dtype, device=Z.device)
            self.Cov = torch.tensor(newCov, 
                dtype=self.Cov.dtype, 
                device=self.Cov.device)

        return newS

    def MStep(self, X, S, Y):

        loss = self.obj(S, Y)

        if self.weight_reg > 0:
            WtW = self.W.T@self.W
            loss -= self.weight_reg*(torch.sum(self.W**2)**2)/torch.sum(WtW**2)
            
        return loss 

