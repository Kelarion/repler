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

class Memba(nn.Module):
    
    def __init__(self, dim_inp, output, memory, **sr_args):
        
        super(Memba, self).__init__()
        
        # Modules
        self.SR = ShapeRotator(dim_inp, **sr_args)
        self.FF = output
        self.MM = memory
        
    def forward(self, X, Y=None):
        """
        For a single X,Y pair
        """
        
        with torch.no_grad(): # remapping and learning happens offline
            Rx = X + self.SR(X)
            
            if Y is not None:
                X_hat = self.MM(Rx, Y)      # retrieve memory
                self.SR.update(X, X_hat)    # update associations
                # self.MM.update(Rx, Y, X)   # write new memory
        
        Y_hat = self.FF(Rx)[0] # make prediction
        
        return Y_hat


class ShapeRotator(nn.Module):
    
    def __init__(self, dim_inp, memory, rule="ols", tau=1):
        """
        Recurrent weights for mapping new patterns onto old patterns
        
        currently accepted values for 'rule' are:
            - 'hebb'
            - 'ols'
            - 'procrustes'
        with default being procrustes
        
        Can optionally enforce orthonormality, in which case the updates
        implement recursive Procrustes. 
        """
        
        super(ShapeRotator, self).__init__()
        
        self.memory = memory
        
        # hyperparameters
        self.rule = rule
        self.tau = tau
        
        # weights
        self.W = torch.zeros(dim_inp, dim_inp) # actual weights
        
        # intermediate variables
        self.H = torch.zeros(dim_inp, dim_inp) # hebbian component
        self.R = 1e3*torch.eye(dim_inp) # correction (depends on rule) 
    
    def forward(self, X, Y=None):
        """
        Evaluate on inputs X, and if Y is provided 
        
        X is shape (..., dim)
        """
        
        Rx = X + torch.einsum('...ij,...j->...i', self.W, X)
        
        if Y is not None:
            X_hat = self.MM(Rx, Y)      # retrieve memory
            self.update(X, X_hat)    # update associations
        
        return Rx
    
    def update(self, A, B):
        """
        Update weights given new input-output pair, 
        A is pre (i.e. inputs)
        B is post (i.e. targets)
        """
        
        self.H = self.H + self.tau*torch.einsum('...i,...j->...ij',B,A)
        
        if self.rule == 'procrustes': # procrustes update
            
            U,S,V = torch.SVD(self.H) # can't find a simple online SVD ... 
            self.W = U@V.T
            
        elif self.rule == 'ols': # ordinary least squares update
            
            Ra = torch.einsum('...ij,...j->...i', self.R, A)
            alph = 1 + torch.einsum('...i,...i', Ra, A)
            dR = torch.einsum('...i,...j->...ij',Ra,Ra)
            self.R = self.R - dR/alph    # whitening matrix
            self.W = self.H@self.R
        
        elif self.rule == 'hebb': # normal hebbian learning
            self.W = self.H
        
        # elif self.rule == 'oja':
        #     self.W = self.H - torch.einsum('...i,...j->...ij', B**2, self.W)


class MagicMemory(nn.Module):
    
    def __init__(self, X, Y, permutations):
        """
        Stores a list of patterns, P_i, i = 1,...,N, and returns P_pi(i), where
        pi is a pre-specified permutation on [N]
        
        In reality, there's no one stopping you from making pi some arbitrary 
        assignment instead of a one-to-one permutation, only your conscience
        """
          
        super().__init__()
        
        self.keys = torch.cat([X,Y], dim=-1)
        self.values = X
        self.pi = permutations
    
    def forward(self, X, Y):
        """
        Find key which matches (X,Y), thresholded, 
        """
        
        XY = torch.cat([X,Y], dim=-1)
        
        norms = (self.keys**2).sum(-1)
        kq = torch.einsum('ij,...j->...i', self.keys, XY)
        d = norms + (XY**2).sum(-1) - 2*kq
        
        in_ball = (d <= 1e-2).any(dim=1, keepdim=True)
        idx = torch.argmin(d, dim=-1)
        
        out = in_ball*(self.values[self.pi[idx]])
        
        return out
    

class KeyValueMemory(nn.Module):
    
    def __init__(self, dim_key1, dim_key2, dim_out, thresh=0.1):
        """
        Key-value memory with two keys, currently just thresholds the sum for 
        retrieval but other rules are possible 
        """
        
        super(MagicMemory, self).__init__()
        
        self.x_keys = torch.empty(0, dim_key1)
        self.y_keys = torch.empty(0, dim_key2) 
        self.values = torch.empty(0, dim_out)
        
        self.threshold = thresh
    
    def forward(self, X, Y, noise_var=1e-4):
        """
        Assuming X and Y are shape (..., dim)
        """
        
        if len(self.y_keys) > 0:
            x_norms = (self.x_keys**2).sum(-1)
            y_norms = (self.y_keys**2).sum(-1)
            
            kq_y = torch.einsum('ij,...j->...i', self.y_keys, Y)
            kq_x = torch.einsum('ij,...j->...i', self.x_keys, X)
            
            # d_x = x_norms + (X**2).sum(-1, keepdim=True) - 2*kq_x
            d_y = y_norms + (Y**2).sum(-1) - 2*kq_y
            
            gate = (d_y < self.threshold)*(kq_x > 0)      # gating
            noise = torch.randn(gate.shape)*noise_var   # tie breaker
            
            if torch.any(gate):
                this_one = (gate*(kq_x + kq_y) + noise).argmax(-1) 
                x_hat = self.values[this_one]
            else:
                x_hat = torch.zeros(self.values.shape[1])
        else:
            x_hat = torch.zeros(self.values.shape[1])
        
        return x_hat
    
    def update(self, X, Y, V):
        """
        Append a single X,Y pair to memory
        """
        
        self.x_keys = torch.cat([self.x_keys, X], dim=0)
        self.y_keys = torch.cat([self.y_keys, Y], dim=0)
        self.values = torch.cat([self.values, V], dim=0)

class Transformer(stud.NeuralNet):
    
    def __init__(self, dim_inp, dim_out, width, abstractor=False,
                 num_head=1, att_depth=2, mlp_depth=1, max_len=64):
        
        super().__init__()
        
        self.enc = nn.Linear(dim_inp, width)
        
        att = [nn.MultiheadAttention(width+max_len, num_head, batch_first=True)]*att_depth
        self.att = nn.ModuleList(att)
        
        lyr = [width+max_len]*mlp_depth
        self.mlp = stud.NewFeedforward(*lyr)
        
        self.dec = nn.Linear(width+max_len, dim_out)
        
        self.sym = abstractor
        if abstractor:
            self.S = torch.randn(max_len, width)
        
        self.max_len = max_len
        
    def forward(self, X):
        """
        Shape of X is (..., len_seq, dim_inp)
        """
        
        X_enc = self.pos_enc(self.enc(X))
        
        msk = nn.Transformer.generate_square_subsequent_mask(X.shape[-2])
        
        att_args = {'attn_mask':msk, 
                    'is_causal':True, 
                    'need_weights':False}
        
        for l in range(len(self.att)):
            
            if self.sym:
                V = self.S[X.shape[-2]]
            else:
                V = X_enc
                
            X_enc = X_enc + self.att[i](X_enc, X_enc, V, **att_args)[0]
        
        Y_hat = self.dec(self.mlp(X_enc))
        
        return Y_hat
    
    def pos_enc(self, X):
        """
        One-hot positional encoding
        """
        
        # stupid awful shape broadcasting
        P = torch.eye(self.max_len)[:X.shape[-2]]
        P = P[(None,)*(len(X.shape) - 2)] # hate this None shit so much!
        P = P.expand(*(X.shape[:-1] + (self.max_len,)))
        
        return torch.cat([X, P], dim=-1)
    
    
    def hidden(self, X):
        """
        Returns transformer embedding in each layer
        """
        
        
        msk = nn.Transformer.generate_square_subsequent_mask(X.shape[-2])
        
        X_out = [self.pos_enc(self.enc(X))]
        for l in range(len(self.att)):
            
            if self.sym:
                V = self.S[X.shape[-2]]
            else:
                V = X_out[l]
                
            X_out.append(X_out[l] + self.att[i](X_out[l], X_out[l], V, attn_mask=msk, is_causal=True))
        
        return torch.stack(X_out)
    
    
    def loss(self, y, y_hat):
    
        return nn.BCEWithLogitsLoss()(y_hat, y)
        

class Samformer(stud.NeuralNet):
    
    def __init__(self, dim_inp, dim_out, width, num_head=2, 
                 bilinear=False, share_weights=True, enforce_chunk=True,
                 par_depth=1, par_act='Identity',
                 dec_depth=2, dec_width=None, dec_act='ReLU',
                 init_var=None, resid=False, **bl_args):
        
        super().__init__()
        
        ## arg parsing
        if init_var is None:
            init_var = 1/dim_inp
        
        self.num_head = num_head
        self.width = width
        
        dim_head = dim_inp//num_head
        if enforce_chunk: # assumes dim_inp is divisible by num_head
            idx = np.reshape(np.arange(dim_inp), (num_head, dim_head))
            # basis = init_var*torch.randn(num_head, width, dim_head)
            # self.P = nn.Parameter(basis @ np.eye(dim_inp)[idx])
            self.P = torch.tensor(np.eye(dim_inp)[idx]).float()
        else:
            # self.P = nn.Parameter(init_var*torch.randn(num_head, width, dim_inp))
            self.P = nn.Parameter(init_var*torch.randn(num_head, dim_head, dim_inp))
        
        if dec_width is None:
            dec_width = self.width
        
        head_layers = (dim_head,) + (self.width,)*(par_depth)
        self.FF = stud.NewFeedforward(*head_layers, nonlinearity=par_act) # same network for each head
        
        if bilinear:
            self.dec = Mattention(width, dec_width, dim_out, **bl_args)
        else:
            dec_layers = (width*self.num_head,) + (dec_width,)*(dec_depth-1) + (dim_out,)
            dec_nonlin = [dec_act]*(dec_depth-1) + [None]
            self.dec = stud.NewFeedforward(*dec_layers, nonlinearity=dec_nonlin) 
        
    def forward(self, X):
        """
        X is shape (...,dim_x)
        """
        
        proj = torch.einsum('kij,...j->...ki',self.P, X)
        
        Z = self.FF(proj)
        Y_hat = self.dec(Z.flatten(start_dim=-2, end_dim=-1))
        
        return Y_hat
    
    def loss(self, y, y_hat):
        return nn.BCEWithLogitsLoss()(y_hat, y)
    
    
class CoRN(stud.NeuralNet):
    """
    My implementation of CoRelNet with multiple heads
    """
    
    def __init__(self, dim_inp, num_inp, width, dim_out, 
                 depth=1, num_head=1, enc_act='Identity', 
                 dec_depth=1, dec_width=None, dec_act='Identity', 
                 train_R=True, rank=None, iden_init=False):
        
        super().__init__()
        
        ## Input de-concatenating
        idx = np.reshape(np.arange(dim_inp*num_inp), (num_inp, dim_inp))
        self.P = torch.tensor(np.eye(dim_inp*num_inp)[idx]).float()
        
        ## Encoder 
        if depth > 0:
            enc_layers = (dim_inp,) + (width,)*(depth)
            self.FF = stud.NewFeedforward(*enc_layers, nonlinearity=enc_act) 
        else:
            self.FF = nn.Identity()
            width = dim_inp
        
        ## Bilinear (psd)
        if rank is None:
            rank = width
        if iden_init:
            R_ = torch.eye(width).repeat(num_head,1,1)
        else:
            R_ = torch.randn(num_head, rank, width)/np.sqrt(width*rank)
        if train_R:
            self.R = nn.Parameter(R_)
        else:
            self.R = R_
        
        ## Decoder
        if dec_width is None:
            dec_width = width
        
        self.W = nn.Parameter(torch.randn(num_head, num_inp, num_inp)/num_inp)
        # self.W = nn.Parameter(torch.randn(num_inp, num_inp)/num_inp)
        
        dec_layers = (num_head,) + (dec_width,)*(dec_depth-1) + (dim_out,)
        dec_nonlin = [dec_act]*(dec_depth-1) + [None]
        self.dec = stud.NewFeedforward(*dec_layers, nonlinearity=dec_nonlin) 
    
    def forward(self, X):
        
        ## Encode
        proj = torch.einsum('kij,...j->...ki',self.P, X) 
        Z = self.FF(proj)
        
        ## Dot products
        RZ = torch.einsum('hij,...kj->...hki', self.R, Z) 
        ZRZ = torch.einsum('...hik,...hjk->...hij', RZ, RZ) # dot product matrix

        ## 
        C = torch.einsum('hij,...hij->...h', self.W, ZRZ)
        # C = torch.einsum('ij,...hij->...h', self.W, ZRZ)
        Y_hat = self.dec(C)
        
        return Y_hat
    
    def loss(self, y, y_hat):
        return nn.BCEWithLogitsLoss()(y_hat, y)
        


class Mattention(stud.NeuralNet):
    
    def __init__(self, dim_x, width, dim_out, dim_y=None, 
                 nonlinearity=None, bias=False, resid=False, 
                 symmetric=False, psd=False, diagonal=False,
                 train_bl=True, train_dec=True):
        """
        Bilinear layer
        
        Given two patterns x, y, computes 
        
        o = w@z
        z_k = x.T@R_k@y
        
        for k = 1,...,width
        
        If R is constrained to be psd, dim_y is the maximum rank of R
        """
        
        super().__init__()
        
        if nonlinearity is None:
            self.f = nn.Identity()
        else:
            self.f = nonlinearity
            
        if dim_y is None:
            dim_y = dim_x
        elif symmetric and (dim_y != dim_x):
            raise Exception('Cannot be symmetric if dim_x != dim_y')
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.width = width
        self.is_sym = symmetric
        self.is_psd = psd
        self.is_diag = diagonal
        self.resid = resid
       
        # self.R = nn.Bilinear(dim_x, dim_y, width, bias=bias)
        if diagonal:
            R = torch.randn(width, dim_x, dim_y)/np.sqrt(dim_x*dim_y)
        if train_bl:
            self.R = nn.Parameter(R)
        else:
            self.R = R
            
        self.dec = nn.Linear(width, dim_out)
        
        self.dec.weight.requires_grad = train_dec
        self.dec.bias.requires_grad = train_dec
    
    def forward(self, XY):
        """
        Inputs [X Y] concatenated along the final dimension
        X and Y are shape (...,dim), so XY is shape (..., dim_x + dim_y)
        """

        X = XY[...,:self.dim_x]
        Y = XY[...,self.dim_x:]
        if (not self.is_psd) and (Y.shape[-1] != self.dim_y):
            raise ValueError('Dimension of input must be dim_x + dim_y')
        elif self.is_psd and (Y.shape[-1] != self.dim_x):
            raise ValueError('Dimension of input must be 2*dim_x for psd matrix')
        
        outers = torch.einsum('...i,...j->...ij', X, Y)
        
        if self.is_sym:
            R = self.R + self.R.swapaxes(-1,-2)
            if self.resid:
                R = R + torch.eye(self.dim_x)
        elif self.is_psd:
            R = torch.einsum('kin,kjn->kij', self.R, self.R)
            if self.resid:
                R = R + torch.eye(self.dim_x)
        else:
            R = self.R
        
        xRy = torch.einsum('kij,...ij->...k', R, outers)
        
        return self.dec(self.f(xRy))
    
    def loss(self, y, y_hat):
        return nn.BCEWithLogitsLoss()(y_hat, y)



class TransferTask:

    def __init__(self, X, Y, num_ctx, dim_embed=None, ctx_angle=0, ordered=False):
        """
        Given a prototypical set of input-output pairs, generate many instances
        in orthogonal subspaces.
        
        Generates num_ctx embedding matrices, A1, ..., A2 such that
        
        """
        
        self.X = X  # prototypical X
        if len(Y.shape) == 1:
            self.Y = Y[:,None]
        else:
            self.Y = Y  # prototypical Y
        
        self.ordered = ordered
        self.num_ctx = num_ctx
        self.nX, self.dX = X.shape
        self.dY = self.Y.shape[-1]
        
        self.nCond = self.nX*num_ctx
        
        X_ctx = la.block_diag(*((self.X,)*num_ctx))
        Y_ctx = np.tile(self.Y, [num_ctx,1])

        if dim_embed is not None:
            
            # basis = sts.ortho_group.rvs(dim_embed)[:,:(self.dX*num_ctx)]
            basis = sts.ortho_group.rvs(dim_embed, size=num_ctx)[...,:self.dX]
            basis = basis.swapaxes(-1,-2).reshape((self.dX*num_ctx,-1))
            X_ctx = X_ctx@basis
            
            self.dX = dim_embed
        
        self.X_ctx = X_ctx
        self.Y_ctx = Y_ctx

    def draw_sequence(self, num_seq, len_seq=None, train_set=None, test_set=None, 
                      test_ctx=None, p_ctx=None, ctx_per_seq=None, transformer=False,
                      heldout_ctx=[], num_test_seq=None):

        # parse args        
        if train_set is None:
            train_set = list(range(self.nX))
        if test_set is None:
            test_set = np.setdiff1d(list(range(self.nX)), train_set)
        elif type(test_set) is list:
            test_set = np.array(test_set)
        
        if ctx_per_seq is None:
            ctx_per_seq = self.num_ctx
        
        if p_ctx is None:
            p_ctx = np.ones(self.num_ctx - len(heldout_ctx))/(self.num_ctx - len(heldout_ctx))
            
        if test_ctx is None:
            test_ctx = list(range(self.num_ctx))
        
        if num_test_seq is None:
            num_test_seq = num_seq
        
        if self.ordered:
            if len_seq is not None:
                Warning('Task is ordered but len_seq was supplied, overwriting')
            len_seq = self.nX
        
        T = len_seq*ctx_per_seq
        
        ## Training set
        # going to go through a lot of effort to vectorize, for some reason
        
        train_ctx = np.setdiff1d(range(self.num_ctx), heldout_ctx)
            
        # draw context
        ctx = np.random.choice(train_ctx, size=(num_seq, ctx_per_seq), p=p_ctx)
        ctx = np.repeat(ctx, len_seq, axis=1)
        
        # draw condition
        if self.ordered:
            cond = np.repeat(np.arange(self.nX), ctx_per_seq)
            cond = np.repeat(cond[None,:], num_seq, axis=0)
            trn_cond = np.repeat(train_set, ctx_per_seq)
            trn_cond = np.repeat(trn_cond[None,:], num_seq, axis=0)
        else:
            cond = np.random.choice(np.arange(self.nX), size=(num_seq, T))
            trn_cond = np.random.choice(train_set, size=(num_seq, T))
        
        cond = np.where(np.isin(ctx, test_ctx), trn_cond, cond) + self.nX*ctx
        
        ## Test set
        if transformer:
            
            y_fill = np.hstack([np.zeros((self.nCond, self.dX)), self.Y_ctx])
            x_fill = np.hstack([self.X_ctx, np.zeros((self.nCond, self.dY))])
            
            X_seq = np.empty((num_seq, 2*T, self.dX+self.dY))
            X_seq[:,0::2,...] = x_fill[cond]
            X_seq[:,1::2, ...] = y_fill[cond]
            
            Y_seq = np.repeat(self.Y_ctx[cond], 2, axis=1)
            
            # draw test sequences
            
            if ctx_per_seq > 1:
                ctx_juxta = np.random.choice(train_ctx, size=(num_test_seq,1), p=p_ctx)
                ctx_test = np.random.choice(heldout_ctx, size=(num_test_seq, ctx_per_seq-1))
                ctx_test = np.hstack([ctx_juxta, ctx_test])
            else:
                ctx_test = np.random.choice(heldout_ctx, size=(num_test_seq, ctx_per_seq))
                
            ctx_test = np.repeat(ctx_test, len_seq, axis=1)
            
            if self.ordered:
                cond_test = np.repeat(np.arange(self.nX), ctx_per_seq)
                cond_test = np.repeat(cond_test[None,:], num_test_seq, axis=0)
            else:
                cond_test = np.random.choice(np.arange(self.nX), size=(num_test_seq, T))
            cond_test = cond_test + ctx_test*self.nX
            
            X_tst = np.empty((num_test_seq, 2*T, self.dX+self.dY))
            X_tst[:,0::2,...] = x_fill[cond_test]
            X_tst[:,1::2, ...] = y_fill[cond_test]
            
            Y_tst = np.repeat(self.Y_ctx[cond_test], 2, axis=1)
            
        else:
        
            X_seq = self.X_ctx[cond]
            Y_seq = self.Y_ctx[cond]
            # seq = (self.X_ctx[cond], self.Y_ctx[cond])
        
            tst_cond = np.tile(test_set, len(test_ctx)) 
            tst_cond += self.nX*np.repeat(test_ctx, len(test_set))
        
            X_tst = self.X_ctx[tst_cond]
            Y_tst = self.Y_ctx[tst_cond]
        
        return (X_seq, Y_seq), (X_tst, Y_tst)
        
    def analog(self, idx):
        """
        Analog of item idx in the first context
        """
        
        return np.mod(idx, self.nX)
        

#%% XOR task

seq_len = 16
num_seq = 5000
dim_embed = 128
num_ctx = 256

alpha = 1

ctx_angle = 0

test_set = [0,1]

# X = np.hstack([util.F2(2), 1-util.F2(2)]) 
X = 2*util.F2(2)-1
Y = X.prod(1)
# X = np.hstack([X, Y[:,None]])
# Y = X[:,0]

p_ctx = 1/np.arange(1, num_ctx-len(test_set) + 1)**(alpha)
p_ctx = p_ctx/np.sum(p_ctx)

task = TransferTask(X, Y, num_ctx, dim_embed=dim_embed, ordered=True)

(X_seq, Y_seq), (X_tst, Y_tst) = task.draw_sequence(num_seq, seq_len, ctx_per_seq=5,
                                                    transformer=True, heldout_ctx=test_set,
                                                    num_test_seq=10, p_ctx=p_ctx)

#%%

dl = pt_util.batch_data(torch.tensor(X_seq).float(), 
                        torch.tensor(Y_seq > 0).float(), 
                        batch_size=64)

net = Transformer(X_seq.shape[-1], 1, 128, mlp_depth=1, max_len=X_seq.shape[1])

loss = []
test_loss = []
for epoch in tqdm(range(1000)):
    ls = net.grad_step(dl)
    
    y_hat = net(torch.tensor(X_tst).float())
    tst_ls = net.loss(torch.tensor(Y_tst>0).float(), y_hat)
    
    loss.append(ls)
    test_loss.append(tst_ls.item())
    

#%% Transverse patterning
n_item = 3
num_ctx = 2
seq_len = 300
num_seq = 10
dim_embed = None

left, right = np.where(np.ones((n_item,n_item))-np.eye(n_item))
adj = np.abs(left - right) < 2 # train on adjacent items

train_set = np.where(adj)[0]

num_cond = len(left)
allcond = np.arange(num_cond)

X = np.hstack([np.eye(n_item)[left], np.eye(n_item)[right]])    # dim_X x num_cond
Y = np.roll(np.eye(3),1,axis=1)[left, right]                    # dim_Y x num_cond

task = TransferTask(X, Y, num_ctx, dim_embed=dim_embed)

(X_seq, Y_seq), (X_tst, Y_tst) = task.draw_sequence(num_seq, seq_len,
                                                    train_set=train_set,
                                                    test_ctx=[1])

n = X_seq.shape[-1]
n_tot = n_item*num_ctx
is_left = np.tile(range(n_item), num_ctx) + n_tot*np.repeat(range(num_ctx), n_item)
is_right = np.tile(range(n_item,2*n_item), num_ctx) + n_tot*np.repeat(range(num_ctx), n_item)

LR = np.eye(n)[np.concatenate([is_left, is_right])]

X_seq = X_seq@LR.T
X_tst = X_tst@LR.T
X_ctx = task.X_ctx@LR.T

#%% Same-different

n_item = 8
n_test = 4
seq_len = 256
num_seq = 10
# dim_embed = 100
dim_embed = None
num_ctx = 1

left, right = np.where(np.ones((n_item,n_item)))
X = np.hstack([np.eye(n_item)[left], np.eye(n_item)[right]])
Y = left == right

train_set = np.where((left<n_item-n_test)*(right<n_item-n_test))[0]
test_set = np.where((left>=n_item-n_test)*(right>=n_item-n_test))[0]

task = TransferTask(X, Y, num_ctx, dim_embed=dim_embed)

(X_seq, Y_seq), (X_tst, Y_tst) = task.draw_sequence(num_seq, seq_len,
                                                    train_set=train_set,
                                                    test_set=test_set)


n = X_seq.shape[-1]
n_tot = n_item*num_ctx
is_left = np.tile(range(n_item), num_ctx) + n_tot*np.repeat(range(num_ctx), n_item)
is_right = np.tile(range(n_item,2*n_item), num_ctx) + n_tot*np.repeat(range(num_ctx), n_item)

# LR = np.stack([np.eye(n)[is_left], np.eye(n)[is_right]])
# P = torch.tensor(LR).float()
LR = np.eye(n)[np.concatenate([is_left, is_right])]

X_seq = X_seq@LR.T
X_tst = X_tst@LR.T
X_ctx = task.X_ctx@LR.T

#%% 

# dl_trn = pt_util.batch_data(torch.Tensor(X_seq[:seq_len]), torch.Tensor(Y_seq[:seq_len]))
# dl_tst = pt_util.batch_data(torch.Tensor(X_seq[seq_len:]), torch.Tensor(Y_seq[seq_len:]))

W_lr_1 = 0.1
J_lr_1 = 0.1

W_lr_2 = 0.1
J_lr_2 = 0.1

y_tst = torch.tensor(Y_tst>0).float()
x_tst = torch.tensor(X_tst).float()

ncond = len(task.X_ctx)
MM = MagicMemory(torch.tensor(task.X_ctx), 
                 torch.tensor(task.Y_ctx),
                 task.analog(np.arange(ncond)))

losses = []
for x_t, y_t in zip(X_seq, Y_seq):

    x_t = torch.tensor(x_t).float()
    y_t = torch.tensor(y_t>0).float()
    
    test_net = stud.ShallowNetwork(X_seq.shape[-1], 100, 1, pt_util.RayLou(), stud.Bernoulli)
    
    ## My network
    # SR = ShapeRotator(X_seq.shape[-1], MM, rule="ols")
    
    train_loss = []
    test_loss = []
    for t in tqdm(range(num_ctx*seq_len)):
        
        ## My network
        
        # MM = MagicMemory(, values, permutations)
        
        ## Normal MLP
        test_net.initialized = False # need to reset the learning rates
        
        if t > seq_len:
            W_lr = W_lr_2
            J_lr = J_lr_2
        else:
            W_lr = W_lr_1
            J_lr = J_lr_1
            
        dl = pt_util.batch_data(x_t[[t]], y_t[[t]], shuffle=False)
        train_loss.append(test_net.grad_step(dl, W_lr=W_lr, J_lr=J_lr))
        
        pred = test_net(x_tst)[0].detach()
        test_loss.append(-test_net.p_targ.distr(pred).log_prob(y_tst).mean().item())
    
    losses.append([train_loss, test_loss])

plt.plot(np.mean(losses,axis=0)[0,:seq_len])
plt.plot(np.mean(losses,axis=0)[0,seq_len:])


#%% Compare models

epochs = 500
noise = 0

losses = []
sam_losses = []
best_sam_loss = np.inf
best_sam = None
best_init = None
for x_t, y_t in tqdm(zip(X_seq, Y_seq)):

    x_t = torch.Tensor(x_t).float() + torch.randn(x_t.shape)*noise
    y_t = torch.Tensor(y_t>0).float()
    
    sam_net = Samformer(X_seq.shape[-1], 1, 
                        width=50, 
                        dec_width=100)
    net = Transformer(X_seq.shape[-1], 1, 128, mlp_depth=1, max_len=1000)
    # net = Mattention(n_item*num_ctx, 
    #                   width=100, dim_out=1, 
    #                   psd=False, resid=False)
    # net = CoRN(n_item*num_ctx, 2, 50, 1, depth=2, num_head=1,
    #             iden_init=True, train_R=False, enc_act='ReLU',
    #             dec_depth=2, dec_width=100, dec_act='ReLU')
    
    init_W = 1*sam_net.FF.network.layer0.weight.data.numpy()
    init_V = 1*sam_net.dec.network.layer0.weight.data.numpy()
    
    trn_loss = []
    tst_loss = []
    trn_sam = []
    tst_sam = []
    
    for ctx in range(num_ctx):
        
        dl = pt_util.batch_data(x_t[seq_len*ctx:seq_len*(ctx+1)], 
                                y_t[seq_len*ctx:seq_len*(ctx+1)],
                                shuffle=True, batch_size=32)
        for i in range(epochs):
            with torch.no_grad():
                pred = net(torch.tensor(X_tst).float())
                tst_loss.append(net.loss(torch.tensor(Y_tst).float(), pred))
                
                sam_pred = sam_net(torch.tensor(X_tst).float())
                tst_sam.append(sam_net.loss(torch.tensor(Y_tst).float(), sam_pred))
                
            trn_loss.append(net.grad_step(dl))
            trn_sam.append(sam_net.grad_step(dl))
    
    if tst_sam[-1] < best_sam_loss:
        best_sam_loss = tst_sam[-1]
        best_sam = sam_net
        best_init = [init_W, init_V]
    
    losses.append([trn_loss, tst_loss])
    sam_losses.append([trn_sam, tst_sam])


plt.figure()
plt.subplot(121)
cols = ['b','r','g']
for ctx in range(num_ctx):
    plt.plot(np.mean(losses,axis=0)[0,epochs*ctx:epochs*(ctx+1)], cols[ctx])
    plt.plot(np.array(losses)[:,0,epochs*ctx:epochs*(ctx+1)].T, cols[ctx], alpha=0.5)
    plt.plot(np.mean(losses,axis=0)[1,epochs*ctx:epochs*(ctx+1)], cols[ctx]+'--')
    plt.plot(np.array(losses)[:,1,epochs*ctx:epochs*(ctx+1)].T, cols[ctx]+'--', alpha=0.5)

plt.subplot(122)
for ctx in range(num_ctx):
    plt.plot(np.mean(sam_losses,axis=0)[0,epochs*ctx:epochs*(ctx+1)], cols[ctx])
    plt.plot(np.array(sam_losses)[:,0,epochs*ctx:epochs*(ctx+1)].T, cols[ctx], alpha=0.5)
    plt.plot(np.mean(sam_losses,axis=0)[1,epochs*ctx:epochs*(ctx+1)], cols[ctx]+'--')
    plt.plot(np.array(sam_losses)[:,1,epochs*ctx:epochs*(ctx+1)].T, cols[ctx]+'--', alpha=0.5)



