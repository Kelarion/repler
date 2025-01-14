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

class WMFFN(stud.NeuralNet):
    
    def __init__(self, dim_inp, wm_dims, width, dim_out, depth=1,
                 wm_width=None, wm_depth=1, **ff_args):
        """
        Normal MLP but the first wm_dims dimensions are first passed through 
        a "working memory" network
        """
        
        super().__init__()
        
        if wm_width is None:
            wm_width = width
        
        wm_layers = [wm_dims] + [wm_width]*wm_depth
        self.wm = stud.NewFeedforward(*wm_layers, **ff_args)
        
        mlp_layers = [dim_inp-wm_dims + wm_width] + [width]*depth
        self.mlp = stud.NewFeedforward(*mlp_layers)
        
        self.dec = nn.Linear(width, dim_out)
        asdsdadasd
        self.wm_dim = wm_dims
        
    def forward(self, x):
        """
        X is shape (..., dim_x)
        """
        
        mem = self.wm(x[...,:self.wm_dim])
        
        x_mem = torch.cat([mem, x[...,self.wm_dim:]], dim=-1)
        y_hat = self.dec(self.mlp(x_mem))
        
        return y_hat
    
    def loss(self, y, y_hat):
        
        return nn.BCEWithLogitsLoss()(y_hat, y)
        
class RegularizedMLP(stud.SimpleMLP):
    """
    MLP regularized to maximize participation ratio in all layers
    """
    
    def __init__(self, beta=1, **mlp_args):
        
        self.beta = beta
        
        super().__init__(**mlp_args)
        
    def loss_args(self, batch):
        
        Z = self.hidden(batch[0])
        
        return (Z,)
    
    def loss(self, batch):
        
        z = self.hidden(batch[0])
        y_hat = self.dec(z[-1])
        y = batch[1]
        
        Kz = torch.einsum('...ik,...jk->...ij', z, z)
        pr = torch.sum(Kz**2, dim=(-1,-2))/(torch.einsum('...ii', Kz)**2)
        
        L = -self.obs.distr(y_hat).log_prob(y).mean()
        L = L + self.beta*pr.mean()
        
        return L
    
# class PBTask(sxp.FeedforwardExperiment):
    
#     def __init__(self, inps, outs):
        
#         super().__init__(inps, outs)
        
#     def 

#%% Define task
dim_embed = 100
xor_signal = 0
noise = 0.3

X = util.F2(3) # dimensions are (upper, lower, cue)
Y_targ = X[:,0]*X[:,2] + X[:,1]*(1-X[:,2])
Y_swp =  X[:,0]*(1-X[:,2]) + X[:,1]*(X[:,2])
# Y = np.stack([Y_targ, Y_swp, X[:,2]]).T
Y = Y_targ[:,None]

X_ = 2*X - 1
Y_ = 2*Y - 1

basis = sts.ortho_group.rvs(dim_embed)[:,:4]
M = np.hstack([X_[:,:2], xor_signal*X_[:,:2].prod(1, keepdims=True), X_[:,[2]]])
XM = M @ basis.T

cond = np.random.choice(range(8), size=5000)
X_noise = XM[cond] + np.random.randn(len(cond), dim_embed)*noise
Y_noise = Y[cond]

#%%

nepoch = 100
N = 100
nonlin = 'ReLU'
# nonlin = 'Tanh'
# noise = 0.1
    
net = RegularizedMLP(dim_inp=dim_embed, width=N, dim_out=Y.shape[-1], p_targ=stud.Bernoulli, 
                     beta=0, train_dec=False)
# net = stud.SimpleMLP(dim_inp=dim_embed, width=N, dim_out=1, p_targ=stud.Bernoulli)

dl = pt_util.batch_data(torch.tensor(X_noise).float(), 
                        torch.tensor(Y_noise).float(), batch_size=128)

loss = []
err = []
swp_err = []
Ws = []
for epoch in tqdm(range(nepoch)):
    
    with torch.no_grad():
        pred = net(torch.tensor(XM).float()).detach().numpy()
    
        y_hat = 2*spc.expit(pred) - 1 # soft sign 
    
        # err.append(Y_[:,0] @ y_hat[:,0])
        # swp_err.append(Y_[:,1] @ y_hat[:,0])
        err.append((2*Y_targ-1) @ y_hat[:,0])
        swp_err.append((2*Y_swp-1) @ y_hat[:,0])
        
        Ws.append(basis[:,[0,1,3]].T@(net.enc.network.layer0.weight.data.numpy()))
        
    ls = net.grad_step(dl)
    loss.append(ls)
    
# plt.plot(loss)
plt.plot(err, marker='o')
plt.plot(swp_err, marker='o')

#%%
# plt.plot(err)
# plt.plot(swp_err)

# inps = tasks.BinaryLabels(SRS_null.T)
# outs = tasks.BinaryLabels(R_null.T)

inps = tasks.LinearExpansion(tasks.BinaryLabels(2*X.T - 1), 10, noise_var=1)
# # inps = tasks.BinaryLabels(2*X.T - 1)
outs = tasks.BinaryLabels(Y.T)

this_exp = sxp.FeedforwardExperiment(inps, outs)

# nets = this_exp.initialize_network(WMFFN, 
#                                     width=N,
#                                     wm_dims=2,
#                                     wm_width=100,
#                                     depth=5,
#                                     wm_depth=5,
#                                     nonlinearity=nonlin,
#                                     num_init=10)

nets = this_exp.initialize_network(RegularizedMLP,
                                   train_dec=False,
                                    width=128,
                                    p_targ=stud.Bernoulli,
                                    depth=1,
                                    activation='ReLU',
                                    beta=0,
                                    num_init=10)

this_exp.train_network(nets, skip_metrics=False, verbose=True,
                        nepoch=100, metric_period=1)




