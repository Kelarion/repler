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

#%%

# labs = np.array([[1,1,0,0,0,0],
#                  [1,0,1,0,0,0],
#                  [0,1,0,0,0,0],
#                  [0,0,1,0,0,0],
#                  [0,0,0,1,0,0],
#                  [0,0,0,0,1,0],
#                  [0,0,0,0,0,1],
#                  [1,0,0,1,0,0],
#                  [1,0,0,0,0,1],
#                  [1,0,0,0,1,0]])

labs = np.array([[1,0],
                 [0,0],
                 [1,0],
                 [0,0],
                 [1,0],
                 [0,1],
                 [1,1],
                 [0,1],
                 [1,1],
                 [0,1]])

#%% Define network

nepoch = 300

# this_exp = exp.Cifar10(torch.tensor(labs))
this_exp = exp.MNIST(torch.tensor(labs))

nets = this_exp.initialize_network(students.ConvNet,
                                   inp_res=28,
                                   conv_width=16,
                                   ff_width=100,
                                   kern=5,
                                   depth=3,
                                   activation='ReLU',
                                   num_init=1)

this_exp.train_network(nets, skip_rep_metrics=True, verbose=True, nepoch=nepoch)

#%% Take a subset and compute kernels

cond = this_exp.train_conditions.detach()
deez = np.argsort(cond)[np.concatenate([np.arange(100)+i*6000 for i in range(10)])]


X = np.squeeze(this_exp.train_data[0][deez].numpy())
X_ = (X - X.mean((-1,-2),keepdims=True))/255
Y = this_exp.train_data[1][deez].numpy()

Kx = np.einsum('ikl,jkl->ij',X_,X_)
Ky = Y@Y.T

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
    
