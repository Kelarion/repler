CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/willet_bci/' 

import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/sca/')

import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
 
from sklearn import svm, discriminant_analysis, manifold, linear_model
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import scipy.io as scio

import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
# import cvxpy as cvx

# my code
import util
import df_util
import pt_util
import bae
import bae_models
import bae_search
import bae_util
import plotting as tpl

#%%

data = scio.loadmat(LOAD_DIR + '/Datasets/t5.2019.05.08/singleLetters.mat')

#%%

neur = data['neuralActivityTimeSeries']
go = data['goPeriodOnsetTimeBin'].squeeze()
letter = np.hstack(data['characterCues'].squeeze())

# T = 200
T = 150

gauss = np.exp(-np.linspace(-2,2,11)**2)

X = []
for t in go:
    
    x_let = neur[t:t+T]
    x_smth = np.apply_along_axis(np.convolve,0,x_let, gauss / gauss.sum(), mode='same')
    
    X.append(x_smth)

X = np.array(X)
Xgrp = util.group_mean(X, letter, axis=0)

n, t, c = Xgrp.shape

letters = np.unique(letter)
singles = np.char.str_len(letters) == 1
baseline = np.where(letters == 'doNothing')[0]

#%%

this = 'h'

i = np.where(letters==this)[0][0]

Xcat = Xgrp[singles].reshape((-1, c))

d = util.yuke(Xgrp[i], Xgrp[singles])

Nit = (d < np.sort(util.yuke(Xgrp[i], Xcat), axis=1)[:,101][None,:,None])


K = (1 / (1 + util.yuke(Xgrp[i], Xgrp[singles])))*Nit

plt.imshow(K[12])

#%%
# letters = []
# X = []
# for k in data.keys():
#     let = re.findall('neuralActivityCube_(.+)', k) 
    
#     if len(let) > 0:
#         letters.append(let[0])
#         X.append(data[k].mean(0))

# X = np.array(X)
# letters = np.char.array(letters)
# single_letters = np.char.str_len(letters) == 1

# n, t, c = X.shape 
# Xpt = torch.FloatTensor(X).transpose(1,2)

#%%

Xpt = torch.FloatTensor(Xgrp).transpose(1,2)

#%%

# kays = [2,5,10,15,20]
kays = [2,3,4,5,6]
# kays = [10]
els = [10, 20, 30]

args = {'nonneg':True,
        'time_sparsity': 1,
        'feature_sparsity': 1,
        'sparse_reg': 1e-2,
        'pr_reg': 1e-3,
        'gp_width': 0.05,
        'fit_intercept': False,
        }
# args = {'nonneg':True,
#         'sparse_reg': 2,
#         'pr_reg': 1e-3,
#         'gp_width': 0.05,
#         # 'fit_intercept': False,
#         }

opt_args = {'initial_temp': 10,
            'decay_rate': 0.9,
            'period': 10,
            'lr': 1e-1,
            }
# opt_args = {'initial_temp': 1,
#             'decay_rate': 1,
#             'max_iter': 500,
#             # 'period': 10,
#             'lr': 1e-2,
#             }

n_run = 1

trn = np.zeros((len(kays), len(els)))
tst = np.zeros((len(kays), len(els)))
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
        for j,l in enumerate(els):
            
            # mod = bae_models.BiPCA(k, sparse_reg=1e-4)
            mod = bae_models.ConvBMF(k,l,**args)
            # mod = bae_models.ConvNMF(k,l,**args)
            
            # wa,ba = bae_util.impcv(mod, X, verbose=True, **opt_args)
            # wa,ba = bae_util.impcv(mod, Xpt[singles], verbose=False, **opt_args)
            wa,ba = bae_util.impcv(mod, torch.FloatTensor(data['NEURAL'][None]), verbose=False, **opt_args)
            
            trn[i,j] += wa.numpy()
            tst[i,j] += ba.numpy()

plt.plot(kays, trn)
plt.plot(kays, tst, '--')
# plt.plot(kays, np.mean(trn,axis=0))
# plt.plot(kays, np.mean(tst, axis=0))

#%%

