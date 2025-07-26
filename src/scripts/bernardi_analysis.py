CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
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
import bae
import bae_util
import bae_models
import plotting as tpl
import anime

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
# this = hpc[0]
this = pfc[0]
# this = acc[0]
# this = Z_hpc
# this = Z_pfc
# this = Z_acc

# mod = bae_models.BiPCA(11, tree_reg=1e-2, sparse_reg=1e-2)
# mod = bae_models.KernelBMF(6, tree_reg=1e-2)
mod = bae_models.SemiBMF(3, weight_reg=1, tree_reg=1e-1, sparse_reg=1e-2,
                         fit_intercept=True, nonneg=True)
neal = bae_util.Neal(0.9, period=50, initial=10)
# neal = bae_util.Neal(1, 1, 1e-4)
en = neal.fit(mod, this, W_lr=1e-1, b_lr=1e-1)
# en = neal.fit(mod, util.group_mean(this[0], cond, axis=0), pvar=0.9)

# S = (mod.S + (mod.S.mean(0)>0.5))%2
S = mod.S
# S = S[cond]

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




