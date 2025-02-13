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

#%% Rank robustness for dense categories

N = 32
dim = 100
snr = [0,10,20]

Strue = util.F2(int(np.log2(N)))
# Strue = (1+df_util.schurcats(N, 0.5))/2
rtrue = Strue.shape[1]

neal = bae_util.Neal(decay_rate=0.98, period=2, initial=5)

for i,this_snr in enumerate(snr):
    
    X = df_util.noisyembed(Strue, dim, this_snr, scl=1e-4)
    cln = util.nbs(X, Strue)
    
    nbs = []
    dh = []
    for r in tqdm(range(1, 2*rtrue+1)):
        
        # mod = bae_models.KernelBAE(r, penalty=1, fix_scale=False)
        # mod = bae_models.GaussBAE(r, tree_reg=1, sparse_reg=0, center=False)
        mod = bae_models.GaussBAE(r, tree_reg=0, sparse_reg=0.1, center=True)
        
        en = neal.fit(mod, X, verbose=False)
        
        dh.append(np.mean(len(mod.S) - np.abs((2*mod.S-1).T@(2*Strue-1)).max(1)))
        nbs.append(util.nbs(X, mod.S))
    
    plt.subplot(2,1,1)
    plt.plot(range(1,2*rtrue+1), nbs)
    if i == len(snr)-1:
        plt.plot([rtrue, rtrue], plt.ylim(), 'k--')
    plt.plot(plt.xlim(), [cln,cln], 'k--')
    plt.ylim([0,1])
    
    plt.subplot(2,1,2)
    plt.plot(range(1,2*rtrue+1), dh)
    if i == len(snr)-1:
        plt.plot([rtrue, rtrue], plt.ylim(), 'k--')


#%% Rank robustness for hierarchical categories

N = 32
dim = 100
snr = 30

Strue = df_util.randtree_feats(N, bmin=2, bmax=4)
X = df_util.noisyembed(Strue, dim, snr, scl=1e-4)
cln = util.nbs(X, Strue)
rtrue = Strue.shape[1]

neal = bae_util.Neal(decay_rate=0.98, period=2, initial=10)

nbs = []
dist_to_mod = []
dist_to_true = []
for r in tqdm(range(1, 2*rtrue+1)):
    
    mod = bae_models.KernelBAE(r, penalty=1, fix_scale=False)
    # mod = bae_models.GaussBAE(r, tree_reg=1, sparse_reg=0, center=False)
    # mod = bae_models.GaussBAE(r, tree_reg=0, sparse_reg=0.1, center=True)
    
    en = neal.fit(mod, X, verbose=False)
    
    dist_to_mod.append(np.mean(N - np.abs((2*mod.S-1).T@(2*Strue-1)).max(1)))
    dist_to_true.append((N - np.abs((2*mod.S-1).T@(2*Strue-1)).max(0)))
    nbs.append(util.nbs(X, mod.S))

dist_to_mod = np.array(dist_to_mod)
dist_to_true = np.array(dist_to_true)

plt.subplot(2,1,1)
plt.plot(range(1,2*rtrue+1), nbs)
plt.plot([rtrue, rtrue], plt.ylim(), 'k--')
plt.plot(plt.xlim(), [cln,cln], 'k--')
plt.ylim([0,1])

plt.subplot(2,1,2)
plt.plot(range(1,2*rtrue+1), dist_to_mod)
plt.plot([rtrue, rtrue], plt.ylim(), 'k--')


#%% Sum/product of projection matrices

N = 32
dim = 100
snr = 0
chains = 10

# Strue = util.F2(int(np.log2(N)))
Strue = df_util.schurcats(N, 0.5)
X = df_util.noisyembed(Strue, dim, snr, scl=1e-4)
cln = util.nbs(X, Strue)

neal = bae_util.Neal(decay_rate=0.98, period=2, initial=5)

S = []
W = []
b = []
scl = []
for _ in tqdm(range(chains)):
    
    mod = bae_models.GaussBAE(Strue.shape[1], sparse_reg=1e-2)
    # mod = bae_models.GaussBAE(Strue.shape[1], sparse_reg=1e-2, center=False)

    en = neal.fit(mod, X, verbose=False)
    
    S.append(mod.S)
    W.append(mod.W)
    b.append(mod.b)
    scl.append(mod.scl)

D = np.array([[df_util.permham(2*S[i]-1, 2*S[j]-1).mean() for i in range(chains)] for j in range(chains)])
plt.imshow(D, 'binary')

#%% Train/test split for points

N = 32
dim = 100
snr = [0,5,10]
draws = 500

Strue = util.F2(int(np.log2(N)))
# Strue = df_util.schurcats(N, 0.5)
rtrue = Strue.shape[1]

neal = bae_util.Neal(decay_rate=0.98, period=2, initial=5)

hams = []
for this_snr in snr:

    X = df_util.noisyembed(Strue, dim, this_snr, scl=1e-4)
    
    ham = []
    
    for draw in tqdm(range(draws)):
            
        idx = np.random.permutation(range(N))
        trn = idx[:N//2]
        tst = idx[N//2:]
        
        trnmod = bae_models.GaussBAE(Strue.shape[1], sparse_reg=1e-2)
        tstmod = bae_models.GaussBAE(Strue.shape[1], sparse_reg=1e-2)
        # mod = bae_models.GaussBAE(Strue.shape[1], sparse_reg=1e-2, center=False)
        
        en = neal.fit(trnmod, X[trn], verbose=False)
        en = neal.fit(tstmod, X[tst], verbose=False)
        
        pred = (X[tst] - X[trn].mean(0))@trnmod.W
        
        ham.append(df_util.permham(tstmod.S, 1*(pred > 0)).mean())
    
    plt.hist(ham, density=True, bins=20, alpha=0.5)
    
    hams.append(ham)
    
plt.legend(snr, title='logSNR')

#%% Train/test split for structured Gaussian mixture



