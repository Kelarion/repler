CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/bernardi_data/'
 
import os, sys, re
import pickle
from time import time
import math
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

import matplotlib.pyplot as plt

import networkx as nx
import cvxpy as cvx

from numba import njit

# my code
import util
import df_util
import old_bae
import bae_util
import old_bae_models
import plotting as tpl
import anime

import bae_models

#%%

bern_hpc = pkl.load(open(SAVE_DIR+'HPC_dump.pck','rb'))
bern_pfc = pkl.load(open(SAVE_DIR+'DLPFC_dump.pck','rb'))
bern_acc = pkl.load(open(SAVE_DIR+'ACC_dump.pck','rb'))

Z_hpc = np.stack([np.mean(z,axis=0) for z in bern_hpc.values()])
Z_pfc = np.stack([np.mean(z,axis=0) for z in bern_pfc.values()])
Z_acc = np.stack([np.mean(z,axis=0) for z in bern_acc.values()])

#%% Robust fitting

folds = 10

ntrl = 1000//folds

hpc = []
pfc = []
acc = []
for fold in range(folds):
    
    i0 = ntrl*fold
    i1 = ntrl*(fold+1)
    
    hpc.append(np.vstack([z[i0:i1] for z in bern_hpc.values()]))
    pfc.append(np.vstack([z[i0:i1] for z in bern_pfc.values()]))
    acc.append(np.vstack([z[i0:i1] for z in bern_acc.values()]))
    
    # hpc.append(np.stack([np.mean(z[i0:i1],axis=0) for z in bern_hpc.values()]))
    # pfc.append(np.stack([np.mean(z[i0:i1],axis=0) for z in bern_pfc.values()]))
    # acc.append(np.stack([np.mean(z[i0:i1],axis=0) for z in bern_acc.values()]))
    
cond = np.repeat(np.arange(8), ntrl)

# %% 
# X_ = hpc[0]
# X_ = pfc[0]
# X_ = acc[0] 
# X_ = Z_hpc
X_ = Z_pfc
# X_ = Z_acc

# mod = old_bae_models.BiPCA(11, tree_reg=1e-2, sparse_reg=1e-2)
# mod = old_bae_models.KernelBMF(6, tree_reg=1e-2)
# mod = old_bae_models.SemiBMF(3, weight_pr_reg=1, tree_reg=1e-1, sparse_reg=1e-2,
#                          fit_intercept=True, nonneg=True)
# neal = bae_util.Neal(0.9, period=50, initial=10)
# neal = bae_util.Neal(1, 1, 1e-4)
# en = neal.fit(mod, this, W_lr=1e-1, b_lr=1e-1)
# en = neal.fit(mod, util.group_mean(this[0], cond, axis=0), pvar=0.9)


mod = bae_models.JBMF(3,
                         nonneg=True,
                         # nonneg=False,
                         # fit_intercept=False,
                         tree_reg=0,
                         weight_pr_reg=1,
                         weight_l2_reg=1e-2,
                         weight_l1_reg=0,
                         sparse_reg=0,
                         # J_loss='mle',
                         J_loss='rple',
                         # J_l1_reg=1e-3,
                         J_lr=1e-4,
                         # J_lr=0,
                         # slab=True,
                         # slab_prior=0.1,
                         )


en = mod.fit(X_ / X_.std(),
             period=100,
             initial_temp=100,
             decay_rate=0.88, 
             min_temp=1, 
             scl_lr=1e-4,
             lr=1e-2,
             hot_start=False,
             )


# en = mod.fit(this / this.std(), decay_rate=0.88, min_temp=1, initial_temp=10, period=50)


samps = mod.sample(X_ / X_.std(), n_samp=1000)

# S = (mod.S + (mod.S.mean(0)>0.5))%2
# S = mod.S
# S = S[cond]
plt.imshow(samps.mean(0))
# plt.imshow(util.group_mean(samps.mean(0), cond, axis=0))

#%%



#%% 

U,s,V = la.svd(this-this.mean(0), full_matrices=False)
X = this@V[:3].T

sunq,grp = np.unique(S, axis=0, return_inverse=True)
E,H = df_util.allpaths(sunq, ovlp=S.T@S/len(S), thr=1e-2)

W_ = df_util.krusty((sunq[grp]-sunq[grp].mean(0)).T, (this-this.mean(0)).T)
W = W_@V[:3].T
b = V[:3]@(this.mean(0) - W_.T@sunq[grp].mean(0))

tpl.plotcube(E.T, H, W)
tpl.scatter3d(X-b, ax=plt.gca(), c=cond)
# tpl.scatter3d(X-b, ax=plt.gca())

#%%




