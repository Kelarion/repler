CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
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
import students as stud
import assistants
import experiments as exp
import super_experiments as sxp
import util
import pt_util
import tasks
import plotting as dicplt
import grammars as gram
import dichotomies as dics

#%%

class RelationalAttention(nn.Module):
    """
    My take on the 'relational attention' of the Abstractor -- same as normal
    attention but with random values
    """
    
    def __init__(self, width, heads, use_vals=False, **mha_args):
        super(RelationalAttention, self).__init__()
        
        self.use_vals = use_vals
        self.MHA = nn.MultiheadAttention(width, heads, batch_first=True, **mha_args)
    
    def forward(self, X, *args, **kwargs):
        """
        Assume X is shape (batch, len, dim)
        """
        
        if self.use_vals:
            V = X
        else:
            V = torch.randn(*X.shape)
        msk = nn.Transformer.generate_square_subsequent_mask(X.shape[-2])
        
        return self.MHA(X, X, V, attn_mask=msk, is_causal=True, *args, **kwargs)
        

class Abstractor(nn.Module):
    """
    
    """
    
    def __init__(self, depth, width, heads, use_vals, 
                 enc_class=None, dim_out=None, **attn_prms):
        super(Abstractor, self).__init__()
        
        self.depth = depth
        
        if enc_class is None:
            enc_class = nn.Linear
        
        self.att = nn.ModuleList()
        self.enc = nn.ModuleList()
        for l in range(depth):
            self.att.append(RelationalAttention(width, heads, **attn_prms))
            self.enc.append(enc_class(width, width))
        
        if dim_out is not None:
            self.dec = nn.Linear(width, dim_out)
        else:
            self.dec = None
        
        self.use_vals = use_vals
        
    def forward(self, X):
        """
        X is shape (num_seq, len_seq, dim) 
        """
        
        Z = self.posenc(X)
        for att, enc in zip(self.att, self.enc):
            Y, _ = att(Z, need_weights=False)
            Z = enc(Y)
        
        if self.dec is not None:
            Z = self.dec(Z)
        
        return Z
    
    def posenc(self, X):
        
        T = X.shape[-2]
        E = X.shape[-1]
        other_dims = tuple(range(len(X.shape[:-2])))
        
        t = np.arange(T)
        n = 10000
        
        P = np.vstack([[np.cos(t/n**(2*i/E)), np.sin(t/n**(2*i/E))] for i in range(E//2)])
        
        return X + torch.tensor(np.expand_dims(P.T, other_dims)).float()
    
def task2seq(X, Y, idx):
    """
    X is shape (num_item, dim_x)
    Y is shape (num_item, dim_y)
    idx is shape (time, ...)
    
    Convert input-output pairs into a sequence of inputs followed by outputs,
    where inputs and outputs occupy orthogonal dimenions
    """
    
    N, dx = X.shape
    if len(Y.shape) == 1:
        dy = 1
        Y = Y[:,None]
    else:
        _, dy = Y.shape
    T = len(idx)
    
    y_fill = np.hstack([np.zeros((N, dx)), Y])
    x_fill = np.hstack([X, np.zeros((N, dy))])
    
    X_seq = np.empty((2*T, *idx.shape[1:], dx+dy))
    X_seq[0::2,...] = x_fill[idx]
    X_seq[1::2, ...] = y_fill[idx]
    
    return X_seq
    
#%% Create task

seq_len = 32
num_ctx = 2
num_seq = 1000
dim_embed = 100

batch_size = 32

sequential = True

X = 2*util.F2(2) - 1
Y = X.prod(1)

cond = np.random.choice(range(4), size=(num_seq, num_ctx*seq_len))

first_basis = sts.ortho_group.rvs(dim_embed)[:,:3]

## sequential version
if sequential:
    XY = task2seq(X,Y,cond.T)
    # Y_seq = Y[np.repeat(cond, 2,axis=1)]
    # Y_seq[...,1::2] = 0
    
    b1 = np.repeat(first_basis[None,None,...], num_seq, axis=1)
    
    basis = sts.ortho_group.rvs(dim_embed, size=(num_ctx-1)*num_seq)
    basis = basis[...,:3].reshape((num_ctx-1, num_seq, dim_embed, -1))
    basis = np.repeat(np.concatenate([b1,basis], axis=0), 2*seq_len, axis=0)
    
    seq = np.squeeze(basis@XY[...,None])
    
    XY_seq = torch.tensor(seq.swapaxes(0, 1)).float()
    YY_seq = torch.tensor(Y[np.repeat(cond, 2,axis=1)] > 0)[...,None].float()

## input-output version
else:
    X_seq = X[cond]
    Y_seq = Y[cond]

    basis = sts.ortho_group.rvs(dim_embed, size=num_ctx*num_seq)
    basis = basis[...,:2].reshape((num_seq, num_ctx, dim_embed, -1))
    basis = np.repeat(basis, seq_len, axis=1)

    seq = np.squeeze(basis@X_seq[...,None])
    
    XY_seq = torch.tensor(seq).float()
    YY_seq = torch.tensor(Y_seq[...,None]>0).float()

#%% Package for transformer

dl = pt_util.batch_data(XY_seq, YY_seq, batch_size=batch_size)

#%% Define network
num_head = 5
depth = 1
use_vals = True

net = Abstractor(depth, dim_embed, num_head, use_vals, 
                 enc_class=stud.NewFeedforward,
                 dim_out=1)

#%% Train network

epochs = 500

optimizer = optim.Adam(net.parameters())

train_loss = []
for epoch in tqdm(range(epochs)):
    
    ls = 0
    for i,batch in enumerate(dl):
        optimizer.zero_grad()
        pred = net(batch[0])
        
        loss = nn.BCEWithLogitsLoss()(pred, batch[1])
        loss.backward()
        
        optimizer.step()
        
        ls += loss.item()
    
    train_loss.append(ls/(i+1))



