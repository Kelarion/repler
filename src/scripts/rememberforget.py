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
import scipy.special as spc
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
    train_lengths: list
    test_lengths: list
    min_delay: int = 0
    max_delay: int = 0
    sustained: bool = False     # output targets during whole delay period?
    mask_token: bool = True     # special token to cue the output?
    reconstruct: bool = False   # include reconstruction in output?
    silent_delay: bool = True   # enforce no output during delay?
    loud_delay: bool = False    # require output during entire delay period?
    num_groups: int = 1         # how many groups of equivalent tokens
    maintain_cue: bool = False
    seed: int = 0
    
    def draw_seq(self, L):
        
        ntok = self.num_tok*self.num_groups
        I = np.eye(ntok + 2)
        
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
        if self.mask_token:
            X[T[-1],-1] = 1         # choice cue
        
        if self.silent_delay:
            Y = self.num_tok*np.ones((T[-1]+1))
        else:
            Y = -1*np.ones((T[-1]+1))
        if self.reconstruct:
            if self.sustained:
                for t in range(L):
                    Y[T[t]:T[t+1]] = toks1[t]
                    Y[T[L+t]:T[L+t+1]] = toks2[t]
            else:
                Y[T[:L]] = toks1 
                Y[T[L+1:-1]] = toks2[:-1]
        if self.sustained:
            Y[T[-2]:] = toks2[-1]
        else:
            Y[-1] = toks2[-1] 
        
        return torch.FloatTensor(X), torch.LongTensor(Y)
    
    def sample(self, these_lengths=None):
        
        np.random.seed(self.seed)
        
        if these_lengths is None:
            these_lengths = self.train_lengths
        
        pad = nn.utils.rnn.pad_sequence
        
        Xs = []
        Ys = []
        # tXs = []
        # tYs = []
        for i in range(self.samps):
            
            ## Draw sequence
            L_trn = np.random.choice(these_lengths)
            Xtrn, Ytrn = self.draw_seq(L_trn)
            Xs.append(Xtrn)
            Ys.append(Ytrn)
            
            # L_tst = np.random.choice(self.test_lengths)
            # Xtst, Ytst = self.draw_seq(L_tst)
            # tXs.append(Xtst)
            # tYs.append(Ytst)
        
        return {'X': pad(Xs, padding_value=0, batch_first=True), 
                'Y': pad(Ys, padding_value=-1, batch_first=True),
                # 'Xtest': pad(Xs, padding_value=0, batch_first=True), 
                # 'Ytest': pad(Ys, padding_value=-1, batch_first=True),
                }

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
        
        # Yhat = self.net(data['X'])
        

#%% Define network

net = RFRNN(dim_hidden=100, rnn_type=nn.RNN, epochs=1000, 
            rnn_args={'nonlinearity': 'relu'})

task = SequenceCompletion(samps=5000, num_tok=8,
                          train_lengths=[3,5,7],
                          test_lengths=[2,4,6,8],
                          min_delay=2, max_delay=5,
                          silent_delay=True, 
                          reconstruct=False,
                          mask_token=False,
                          sustained=True)

this_exp = sxp.Experiment(task, net)

#%%

this_exp.run()

#%% Look at length generalization

n_samp = 100

test_loss = []
train_loss = []
for L in range(2,task.num_tok+1):
    
    ls = []
    for i in range(n_samp):
        x,y = task.draw_seq(L)
        yhat = net.net(x).argmax(1)
        # ls.append(net.net.loss((x[None],y[None])).item())
        ls.append((y == yhat)[y>=0].numpy().mean())
    
    if L in task.test_lengths:
        test_loss.append(np.mean(ls))
    else:
        train_loss.append(np.mean(ls))

plt.scatter(task.train_lengths, train_loss)
plt.scatter(task.test_lengths, test_loss)

#%% Organize hidden states

samps = task.sample(these_lengths=list(range(2, task.num_tok+1)))
X = samps['X'].numpy()
Y = samps['Y'].numpy()
Z = net.net.rnn(samps['X'])[0].detach().numpy().astype(float)

idx = np.argsort(X.sum((-1,-2))//2) # sort by length
X = X[idx]
Y = Y[idx]
Z = Z[idx]

## Organize according to cummulative tokens
ctx = X[...,-2].cumsum(-1)
cue = X[...,-2].argmax(-1)
toks = X.cumsum(1)

## The hypothesised memory state at each time point
# in_trial = X[...,-1].cumsum(-1) < 1
in_trial = np.fliplr(np.cumsum(np.fliplr(Y>=0),1)) > 0
mem = (X[...,:-2]*(1-2*ctx[...,None])).cumsum(1)

is_inp = X.max(-1) == 1
is_final = np.fliplr(np.diff(np.fliplr(in_trial), prepend=0)) == 1
is_cue = X[...,-2] == 1
is_precue = np.roll(is_cue, -1, axis=-1)
cuetime = X[...,-2].argmax(-1)
pretok = np.roll(X.max(-1), -1, axis=1) == 1
pretok[:,-1] = 0
which_tok = X.argmax(-1)
which_trl = np.arange(len(X))[:,None]*np.ones(X.shape[1])
time = np.arange(X.shape[1], dtype=int)[None,:]*np.ones((len(X),1), dtype=int)


#%%

Zswi = Z[is_precue]
Zfin = Z[is_final]

Z1 = Z[in_trial*(ctx==0)]
Z2 = Z[in_trial*(ctx==1)]

memunq, grp = np.unique( mem[is_precue], axis=0, return_inverse=True)
Zgrp = util.group_mean(Zswi, grp, axis=0)

#%% Plot individual trials

this_trial = 0

plt.imshow(Z[this_trial][in_trial[this_trial]], 'binary')
# plt.imshow(ZW[this_trial][in_trial[this_trial]], 'bwr')
# plt.imshow(Zdec[this_trial][in_trial[this_trial]], 'bwr')

cmap = cm.viridis
for t in time[this_trial][is_inp[this_trial]]:
    
    tok = which_tok[this_trial][t]
    if tok < task.num_tok:
        col = cmap(which_tok[this_trial][t]/task.num_tok)
        sty = '--'
    else:
        col = 'k'
        sty = '-'
    plt.plot(plt.xlim(), [t-0.5, t-0.5], sty, color=col, linewidth=2)

#%%
this_trial = 1

plt.plot(Z[this_trial][in_trial[this_trial]]@V[0])

cmap = cm.viridis
for t in time[this_trial][is_inp[this_trial]]:
    
    tok = which_tok[this_trial][t]
    if tok < task.num_tok:
        col = cmap(which_tok[this_trial][t]/task.num_tok)
        sty = '--'
    else:
        col = 'k'
        sty = '-'
    plt.plot([t, t], plt.ylim(), sty, color=col, linewidth=2)

# zees = []
# for t in range(X.shape[-1]):
#     zees.append(Z[in_trial][which_tok[in_trial] == t])


#%% Recurrent interventions

## Compute empirically how the recurrent activity affects concepts:
## w_b * <(F(xi - (1-2s_ai)w_a) - F(xi))>_i

# Z1 = Ztrl[ctx[in_trial]==0]

dS = []
zr = torch.zeros(12,1).T
for j in tqdm(range(mod.r)):    
    
    # deez = np.random.choice(range(len(Z1)), 10_000, replace=False)
    
    dZ = []
    # for i in deez:
    for i in range(len(Ztrl)):
        dz = (1 - 2*mod.S[i,j])*mod.W[:,j]*mod.scl
        before = net.net.rnn(zr, torch.FloatTensor(Ztrl[[i]]))[0]
        after = net.net.rnn(zr, torch.FloatTensor(Ztrl[[i]]+dz[None]))[0]
        dZ.append((after-before).detach().numpy())
        
    dZ = np.squeeze(np.mean(dZ, axis=0))
    dS.append(mod.W.T@dZ/(la.norm(dZ)*mod.scl))
    
dS = np.array(dS)

# #%% Intervene on hidden

# dL = []
# for j in range(len(W)):
#     row = []
#     for i in range(len(Z)):
        
#         sgn = 2*S[i,j]-1
#         # ystar = 2*R_[i,0]-1
        
#         before = nets[0].dec(torch.FloatTensor(Z[i])).detach().numpy()
#         after = nets[0].dec(torch.FloatTensor(Z[i] - sgn*W[j])).detach().numpy()
        
#         row.append((sgn*(after-before)))
#     dL.append(row)

# dL = np.squeeze(dL)

#%% centered

before = []
after = []

X = H@util.F2(3)
for _ in range(100):
    W = sts.ortho_group(3).rvs()
    S = 1*(H@X@W > 0)

    # ls = []
    before.append(np.sum((X - H@S@W.T)**2))
    for step in range(10):
        
        # ls.append(np.sum((X - H@S@W.T)**2))
        Ssum = wa.sum(0)
        for i in range(8):
            Ssum -= S[i]
            c = (2*X[i]@W - 7/8 + 2*Ssum/8)
            S[i] = 1*(c > 0)
            
            Ssum += S[i]
        
        W = df_util.krusty(X.T,S.T)
    
    after.append(np.sum((X - H@S@W.T)**2))
    
#%%

before = []
after = []

for _ in range(100):
    W = sts.ortho_group(3).rvs()
    S = S = 1*(X@W > 0)
    
    before.append(np.sum((X - S@W.T)**2))
    
    for step in range(10):
        
        S = 1*(2*X@W > 1)
        
        W = df_util.krusty(X.T,wa.T)
        
    after.append(np.sum((X - S@W.T)**2))
    
#%%

before = []
after = []

X = util.F2(3) + 0*np.random.randn(1,3)
for _ in range(100):
    
    W = sts.ortho_group(3).rvs()
    b = X.mean(0)
    S = 1*((X-b)@W > 0)
    
    before.append(np.sum((X - b - S@W.T)**2))
    
    for step in range(10):
        
        S = 1*(2*(X-b)@W > 1)
        
        W = df_util.krusty(wa.T, (X-b).T)
        b = (X - S@W.T).mean(0)
    
    after.append(np.sum((X - b - S@W.T)**2))