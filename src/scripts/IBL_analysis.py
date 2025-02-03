
CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle as pkl
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
import bae_models
import bae_util
import plotting as tpl
import anime

#%%

X = pkl.load(open('C:/Users/mmall/Downloads/X_forMatteo.pck','rb'))
trl = pkl.load(open('C:/Users/mmall/Downloads/trials_forMatteo.pck','rb'))

neurs = np.hstack([x for x in X.values()])
area = np.concatenate([i*np.ones(len(x.T)) for i,x in enumerate(X.values())])

neurz = (neurs-neurs.mean(0,keepdims=True))/(neurs.std(0,keepdims=True)+1e-12)

#%%

CV = []
TRN = []
best = []

for this_area in range(29):

    Z = neurz[:,area==this_area]
    
    cv = []
    trn = []
    Es = []
    for k in tqdm(range(2,16)):
        mod = bae.BAE(Z, k, penalty=0.1)
        mod.init_optimizer(decay_rate=0.98, period=1)
        for it in range(500):
            mod.grad_step()
        trn.append(mod.energy())  
        Es.append(mod.S.todense())  
    
        mod = bae.BAE(Z, k, penalty=0.1)
        cv.append(bae_util.impcv(mod, folds=10, iters=500, draws=10, 
                                 decay_rate=0.98, period=1))

    TRN.append(trn)
    CV.append(np.mean(cv, axis=1))
    best.append(Es[np.argmin(np.mean(cv, axis=1))])
    
    
#%%

_,axs = plt.subplots(5,6)

for this_area in range(29):
    j = this_area//5
    i = np.mod(this_area,5)
    
    wa = neurz[:,area==this_area]@neurz[:,area==this_area].T
    axs[i,j].imshow(wa,'bwr', vmin=-np.abs(wa).max(), vmax=np.abs(wa).max())
    axs[i,j].set_title(list(X.keys())[this_area])
    
    axs[i,j].set_xticks([])
    axs[i,j].set_yticks([])
    axs[i,j].set_xlim([-0.5,15.5])
    axs[i,j].set_ylim([15.5,-0.5])
    
    plt.autoscale(False)
    axs[i,j].plot([7.5,7.5],[0,16],'k--')
    axs[i,j].plot([0,16],[7.5,7.5],'k--')

#%% Single trials

this_area = 'MOs'
this_session = 6

Z = []
labels = []
for key, value in trl[this_area][this_session].items():
    Z.append(value)
    labels.append(np.repeat(np.array(key)[None], len(value), axis=0))

Z = np.vstack(Z)
Y = np.vstack(labels)
Y = np.stack([np.unique(y, return_inverse=True)[1] for y in Y.T]).T

unq, cond = np.unique(Y, axis=0, return_inverse=True)

zZ = (Z-Z.mean(0,keepdims=True))/(Z.std(0,keepdims=True)+1e-12)

#%%

cv = []
eses = []
neal = bae_util.Neal(decay_rate=0.98, initial=1, period=2)
for k in tqdm(range(2, 29)):
    mod = bae_models.GeBAE(k, tree_reg=0.1, weight_reg=1e-1)
    ba = neal.cv_fit(mod, zZ, draws=10, folds=10)
    cv.append(np.mean(ba[0]))
    eses.append(ba[1])
    
