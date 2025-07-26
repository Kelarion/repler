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
import cb_search
import students


####################################################################
################ Nonparametric models ##############################
####################################################################


####################################################################
################ Parametric models #################################
####################################################################


NOISE_PRM = {'gaussian': (bae_util.gaussian, nn.MSELoss()),
             'bernoulli': (bae_util.bernoulli, nn.BCEWithLogitsLoss()),
             'poisson': (bae_util.poisson, nn.PoissonNLLLoss())}


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
    inp_noise: str = 'gaussian'
    out_noise: str = 'gaussian'
    tied_weights: bool = False           # Tie forward and backward weights
    fit_intecept: bool = True
    beta: float = 0.1                   # Strength of output loss
    gamma: float = 1.0
    temp: float = 1.0 
    tree_reg: float = 0
    sparse_reg: float = 0
    weight_reg: float = 0

    def __post_init__(self):
        super().__init__()

        # self.q = nn.Linear(self.dim_inp, self.dim_hid) # x -> s
        # self.p = nn.Linear(self.dim_hid, self.dim_out) # s -> y
        self.W = nn.Parameter(torch.empty(self.dim_hid, self.dim_out))    # s -> y
        self.b = nn.Parameter(torch.zeros(self.dim_out))
        # if self.tied_weights:
        #     self.U = self.W.T
        #     self.d = self.b
        # else:
        #     self.U = nn.Parameter(torch.empty(self.dim_hid, self.dim_out)) 
        #     self.d = nn.Parameter(torch.zeros(self.dim_out))

        self.V = nn.Parameter(torch.empty(self.dim_inp, self.dim_hid))    # x -> z
        self.q = nn.Parameter(torch.zeros(self.dim_hid))
        if self.tied_weights:
            self.M = self.V.T                                       # s -> x'
            self.p = nn.Parameter(torch.zeros(self.dim_inp))
        else:
            self.M = nn.Parameter(torch.empty(self.dim_hid, self.dim_inp))
            self.p = nn.Parameter(torch.zeros(self.dim_inp))

        self.init_weights()

        self.Cov = torch.zeros((self.dim_hid, self.dim_hid))

        self.lnx, self.xobj = NOISE_PRM[self.inp_noise]
        self.lny, self.yobj = NOISE_PRM[self.out_noise]

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
        # return torch.sigmoid((X@self.V+self.q)/self.temp)@self.W + self.b
        return torch.sigmoid(X@self.V+self.q)@self.W + self.b
    
    def hidden(self, X):
        # return 1*(X@self.V+self.q > 0)
        return torch.sigmoid(X@self.V+self.q)

    def metrics(self, dl):
        pass

    def loss(self, batch):

        Z = batch[0]@self.V+self.q

        ## Search over S
        S = self.EStep(batch[0], Z, batch[1])

        ## Update continuous parameters
        qls = nn.BCEWithLogitsLoss()(Z, S)
        # pls = self.MStep(batch[0], S, batch[1])
        pls = self.xobj(S@self.M + self.p, batch[0])
        # pls += self.yobj(S@self.W + self.b, Y)
        pls += self.yobj(self(batch[0]), batch[1])

        return pls + self.beta*qls 

    def EStep(self, X, Z, Y):

        with torch.no_grad():
            
            S = 1.0*(Z > 0)
            Yhat = S@self.W + self.b
            Xhat = S@self.M + self.p

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

            newS = cb_search.gecbm(
                Xhat=Xhat, X=Xnp, M=M, lnx=self.lnx,        # inputs
                Yhat=Yhat, Y=Ynp, W=W, lny=self.lny,        # outputs
                S=Snp, StS=StS, gamma=self.gamma,           # latents
                alpha=self.sparse_reg, beta=self.tree_reg,  # regualarization
                temp=self.temp,                             # noise distribution
                )                            
            newCov = (1-self.tree_lr)*StS + self.tree_lr*newS.T@newS/len(newS)

            newS = torch.tensor(newS, dtype=Z.dtype, device=Z.device)
            self.Cov = torch.tensor(newCov, 
                dtype=self.Cov.dtype, 
                device=self.Cov.device)

        return newS

    # def MStep(self, X, S, Y):

    #     loss = self.beta*self.xobj(S@self.M + self.p, X)
    #     loss += self.yobj(S@self.W + self.b, Y)
    #     # if self.weight_reg > 0:
    #         # WtW = self.W.T@self.W
    #         # loss -= self.weight_reg*(torch.sum(self.W**2)**2)/torch.sum(WtW**2)
            
    #     return loss 

