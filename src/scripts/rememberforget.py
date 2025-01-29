CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
from dataclasses import dataclass, fields, field
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from sklearn.manifold import MDS

import networkx as nx
# import pydot 
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

from pypoman import compute_polytope_vertices, compute_polytope_halfspaces
import cvxpy as cvx
# import polytope as pc

# my code
import students
import assistants
import experiments as exp
import super_experiments as sxp
import util
import pt_util
import tasks
import plotting as dicplt

import distance_factorization as df
import df_util
import bae
import bae_models
import bae_util

#%%

## Rehashing my GRU sequential memory project with SueYeon from ages ago

class RNNClassifier(students.NeuralNet):
    
    def __init__(self, dim_inp, dim_out, dim_hidden, rnn_type=nn.RNN, **rnn_args):
        
        super().__init__()
        
        self.dim_inp = dim_inp
        self.dim_hid = dim_hidden
        self.dim_out = dim_out
        
        self.dec = nn.Linear(dim_hidden, dim_out)
        self.rnn = rnn_type(dim_inp, dim_hidden, **rnn_args, batch_first=True)

    def forward(self, X, H0=None):
        return self.dec(self.rnn(X, H0)[0])
        
    def loss(self, batch):
        y_hat = self(batch[0]).swapaxes(1,-1) 
        return nn.CrossEntropyLoss(ignore_index=-1)(y_hat, batch[1])
      
@dataclass
class SequenceCompletion(sxp.Task):
    
    samps: int
    num_tok: int
    min_length: int
    max_length: int
    min_delay: int = 0
    max_delay: int = 0
    reconstruct: bool = False   # include reconstruction in output?
    silent_delay: bool = True   # enforce no output during delay?
    num_groups: int = 1         # how many groups of equivalent tokens
    
    seed: int = 0
    
    def sample(self):
        
        np.random.seed(self.seed)
        
        ntok = self.num_tok*self.num_groups
        I = np.eye(ntok + 2)
        pad = nn.utils.rnn.pad_sequence
        
        Xs = []
        Ys = []
        for i in range(self.samps):
            
            ## Draw sequence
            L = np.random.choice(range(self.min_length, self.max_length+1))
            toks1 = np.random.choice(range(self.num_tok), size=L, replace=False)
            toks2 = toks1[np.random.permutation(range(L))]
            I1 = I[toks1]
            I2 = I[toks2]
            
            ## Add delays 
            D = np.random.choice(range(self.min_delay+1,self.max_delay+2), 2*L+1)
            T = np.cumsum(D) - D[0]
            
            ## Package data
            X = np.zeros((T[-1]+1, ntok + 2))
            X[T[:L]] = I1           # first sequence
            X[T[L],-2] = 1          # context cue
            X[T[L+1:-1]] = I2[:-1]  # second sequence
            X[T[-1],-1] = 1         # choice cue
            
            Y = np.ones((T[-1]+1))*(1*self.silent_delay - 1)
            if self.reconstruct:
                Y[T[:L]] = toks1 + 1*self.silent_delay
                Y[T[L+1:-1]] = toks2[:-1] + 1*self.silent_delay
            Y[-1] = toks2[-1] + 1*self.silent_delay
            
            Xs.append(torch.FloatTensor(X))
            Ys.append(torch.LongTensor(Y))
        
        return {'X': pad(Xs, padding_value=0, batch_first=True), 
                'Y': pad(Ys, padding_value=-1, batch_first=True)}


@dataclass(kw_only=True)
class RFRNN(sxp.PTModel):
    
    dim_hidden: int
    rnn_type: object
    rnn_args: dict = field(default_factory=dict)
    
    def init_network(self, X, Y):
        self.pbar = None
        return RNNClassifier(dim_inp=X.shape[-1], dim_out=Y.max()+1, 
                             dim_hidden=self.dim_hidden, 
                             rnn_type=self.rnn_type, **self.rnn_args)
    
    def loop(self, **data):
        
        if self.pbar is None:
            self.pbar = tqdm(range(self.epochs))
        self.pbar.update(1)


#%% Define network

net = RFRNN(dim_hidden=100, rnn_type=nn.GRU, epochs=100)

task = SequenceCompletion(samps=5000, num_tok=10, 
                          min_length=4, max_length=7, 
                          min_delay=2, max_delay=5, 
                          silent_delay=False, 
                          reconstruct=False)

this_exp = sxp.Experiment(task, net)

#%%

this_exp.run()

#%% Organize hidden states

samps = task.sample()
X = samps['X'].numpy()

## Organize according to cummulative tokens
ctx = X[...,-2].cumsum(-1)
cue = X[...,-2].argmax(-1)
toks = X.cumsum(1)

## The hypothesised memory state at each time point
in_trial = X[...,-1].cumsum(-1) < 1
mem = (X[...,:-2]*(1-2*ctx[...,None])).cumsum(1)

is_tok = X.max(-1)
which_tok = X.argmax(-1)
trl, inp_time = np.where(is_tok)

Z = net.net.rnn(samps['X'])[0]


#%% reps

C1 = nets[0].conv(this_exp.train_data[0][deez])
C = C1.detach().numpy()/(784)
Kc = np.einsum('iklm,jklm->ij',C,C)

Z1 = nets[0].ff.network[:2](torch.flatten(C1,1)).detach().numpy()/(784)
Z2 = nets[0].ff.network[:4](torch.flatten(C1,1)).detach().numpy()/(784)
Z3 = nets[0].ff(torch.flatten(C1,1)).detach().numpy()/(784)

Kc = np.einsum('iklm,jklm->ij',C,C)
Kz1 = Z1@Z1.T
Kz2 = Z2@Z2.T
Kz3 = Z3@Z3.T

#%% Factorize

baer = bae.BAE(Z1, 40, pvar=0.95)
baer.init_optimizer(decay_rate=0.95, period=2, initial=10)

en = []
for t in tqdm(range(400)):
    #r = np.sum(pvar< (0.8 + 0.2*(t//10)/10))
    baer.proj(pvar=0.95)
    #baer.scl = baer.scaleS()
    baer.grad_step()
    en.append(baer.energy())
    
S = baer.S.todense()

#%%

Sunq, counts = np.unique(np.mod(S+S[[0]],2), axis=1, return_counts=True)

# is_dec = util.qform(util.center(SRS_@SRS_.T), Sunq.T).squeeze() > 1e-7

S,pi = df_util.mindistX(Z1, Sunq, beta=1e-7)
S = S[:,np.argsort(-pi)]
pi = pi[np.argsort(-pi)]
    
