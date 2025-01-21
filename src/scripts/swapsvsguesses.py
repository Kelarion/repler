CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
from time import time
from dataclasses import dataclass

sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
import itertools as itt
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from sklearn.manifold import MDS

import networkx as nx
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

# my code
import students as studs 
import assistants
import experiments as exp
import super_experiments as sxp
import util
import pt_util
import tasks
import plotting as tpl
import dichotomies as dics

import distance_factorization as df
import df_util
import df_models as mods
import bae

#%%

class WMRNN(studs.NeuralNet):
    
    def __init__(self, dim_inp, dim_hid, dim_out,
                 loss=nn.MSELoss(), rnn_type=nn.RNN, **rnn_args):

        self.net = rnn_type(**rnn_args)

        self.obs = loss
    
    def forward(self, X, H0=None):
        """
        X is shape (len_seq, batch, input_size)
        """
        
        return self.net(X, H0)
    
    def loss(self, batch):
        """
        Assumes batch contains (inps, targs)
        """
        
        Yhat = self(batch[0])
        return self.obs(Yhat, batch[1])
    
@dataclass
class WMTask(sxp.Task):
    """
    Working memory task in which inps are mapped to outs after a delay
    
    inps (num_inp_step, dim_inp, num_cond): Input sequence
    outs (num_out_step, dim_out, num_cond): Output sequence
    trials: Number of random draws 
    inter_delay: Average number of time steps between inputs or outputs
    intra_delay: Average number of time steps between inputs and outputs
    jitter: Range of random delays
    """
    
    inps: np.ndarray
    outs: np.ndarray 
    trials: int
    inter_delay: int = 0
    intra_delay: int = None
    jitter: int = 0
    
    def sample(self):
        
        Tx, Dx, Nx = self.inps.shape
        Ty, Dy, Ny = self.outs.shape
        
        assert Nx == Ny
        
        if self.intra_delay is None:
            self.intra_delay = self.inter_delay
        
        ## Draw random delays
        jitr = np.max([self.inter_delay-self.jitter, 0])
        delay_xx = np.random.choice(self.inter_delay + np.arange(-jitr,jitr+1), 
                                    size=(Tx,self.trials))
        delay_yy = np.random.choice(self.inter_delay + np.arange(-jitr,jitr+1), 
                                    size=(Ty,self.trials))
        
        jitr = np.max([self.intra_delay-self.jitter, 0])
        delay_xy = np.random.choice(self.intra_delay + np.arange(-jitr,jitr+1), 
                                    size=self.trials)
        
        ## Fill data array
        cond = np.random.choice(range(Nx), size=self.trials)
        condx = np.repeat(cond[None,:], Tx, axis=0)
        condy = np.repear(cond[None,:], Ty, axis=0)
        
        X = np.zeros((self.trials, Tx + np.max(np.sum(delay_xx,0)), Dx))
        Y = np.zeros((self.trials, Ty + np.max(np.sum(delay_yy,0)), Dy))
        
        X[condx, np.cumsum(delay_xx,0),:] = self.inps[np.arange(Tx),:,condx]
        Y[condy, np.cumsum(delay_yy,0),:] = self.outs[np.arange(Ty),:,condy]

    
    
#%% Generate data
grid_size = 24
grid_dim = 2
neur_dim = 100

X = util.grid_vecs(grid_size, grid_dim)
y = np.sign(X[0])

Z = util.gaussian_process(X.T, neur_dim, kernel=util.RBF(1))

#%% Plot "behavior"

_,axs = plt.subplots(1,3)

## Extrapolation
clf = svm.LinearSVC()

y_ext = np.zeros(len(y))

clf.fit(Z.T[X[1]<0], y[X[1]<0])
y_ext[X[1]>=0] = clf.predict(Z.T[X[1]>=0])

clf.fit(Z.T[X[1]>=0], y[X[1]>=0])
y_ext[X[1]<0] = clf.predict(Z.T[X[1]<0])

axs[0].scatter(X[0], X[1], c=y_ext, cmap='bwr')
axs[0].set_title('SVM')
# tpl.square_axis(axs[0])

## Kernel-logit
K = util.center(Z.T@Z)
K -= np.diag(np.diag(K))

y_logit = np.zeros(len(y))
y_logit[X[1]>=0] = spc.expit(K[X[1]>=0][:,X[1]<0]@y[X[1]<0])
y_logit[X[1]<0] = spc.expit(K[X[1]<0][:,X[1]>=0]@y[X[1]>=0])

axs[1].scatter(X[0], X[1], c=y_logit, cmap='bwr')
axs[1].set_title('All points')

# Weight points according to prototype distance
ptype = np.exp(-(X[1]**2 + (np.abs(X[0])-1)**2)/0.01)
ptype /= np.sum(ptype)

y_logit = spc.expit(K@np.diag(ptype)@y)

axs[2].scatter(X[0], X[1], c=y_logit, cmap='bwr')
axs[2].set_title('Prototypes')