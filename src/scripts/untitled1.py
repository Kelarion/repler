CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'

import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)

sys.path.append('C:/Users/mmall/OneDrive/Documents/github/multi_object_memory_2025/phys_modeling/')
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/multi_object_memory_2025/')

import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

import torch

from sklearn import svm, discriminant_analysis, manifold, linear_model
from sklearn.cluster import KMeans
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc

from phys_modeling.training.dataset import Dataset
from phys_modeling.training.trial_filters import TriangleTrialFilter
from phys_modeling.training.unit_filters import UnitFilter
from phys_modeling import constants

import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
# import cvxpy as cvx

import pandas as pd

import h5py

from pathlib import Path

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

subj = 'Elgar'
# subj = 'Perle'
# sess = '2022-06-01'
# sess = '2022-05-30'
# phase = 'delay'
# phase = 'precue'
# phase = 'cue'
phase = 'stimulus'

all_neurs = {}
is_pfc = []
for sess in os.listdir(constants.SELECTIVITY_DIR / f'{subj}/'):

    df_behavior_ring = pd.read_csv(constants.BEHAVIOR_DIR_RING)
    df_behavior_ring = df_behavior_ring[
            (df_behavior_ring.subject == subj)
            & (df_behavior_ring.session == sess)
        ]    
    if len(df_behavior_ring) > 0:
        print('Ring session')
        continue

    dset = Dataset(subj, sess, phase, UnitFilter(), TriangleTrialFilter(), 3)

    ## Neural
    
    X = dset.data['neural'].sum(1)
    mavg = np.apply_along_axis(np.convolve, 0, X, np.ones(100)/100)
    deez = mavg[100:-100].min(0) > 0
    
    X = X[:,deez]
    X /= np.sqrt(np.mean((X-X.mean(0))**2))
    
    area = (dset.df_units['probe']=='s0').to_numpy()[deez]

    ## Trial types
    
    these = ['object_0_id','object_1_id', 'object_2_id', 
     'object_0_location','object_1_location', 'object_2_location']
    
    labs = dset.df_behavior[these].apply(
        lambda x: [ord(o) - ord('a') if type(o) == str else o for o in x]
        )
    labs = labs.to_numpy()
    
    sort_pos = np.zeros((len(labs),9))
    targs = np.zeros(len(labs), dtype=int)
    for trl, lbs in enumerate(labs):
        targs[trl] = int(lbs[dset.df_behavior['target_object_index'][trl]])
        for i in range(3):
            if np.isnan(lbs[i]):
                continue
            j = int(lbs[i])
            j_ = int(lbs[i+3])
            sort_pos[trl, j*3:(j+1)*3] = np.eye(3)[j_]
    
    # unqs, idx = np.unique(sort_pos, axis=0, return_inverse=True)
    
    # Make a string representation of each trial type
    names = np.array(['a','b','c'])
    uncs = sort_pos.reshape((-1,3,3))
    stims = np.char.array(np.where(uncs.sum(1), names[uncs.argmax(1)], '-'))
    stims = stims[:,0] + stims[:,1] + stims[:,2]
    
    targs = np.char.array(names[targs])
    
    labels = stims + '|' + targs 
    
    unqs, idx = np.unique(labels, return_inverse=True)
    
    ## Prepare for pseudopop
    all_neurs[sess] = {}
    for lab in unqs:
        all_neurs[sess][lab] = X[labels==lab]
    
    is_pfc.append(area)

is_pfc = np.concatenate(is_pfc)

#%% make pseudopop

# ntrls = []
# nneur = []
# for this_sess in all_neurs.values():
#     ntrls.append( [len(foo) for foo in this_sess.values()] )
#     nneur.append( [len(foo.T) for foo in this_sess.values()] )

# ntrls = np.array(ntrls)
# nneur = np.array(nneur)

# ## how many trials to sample from each condition?
# n_samp = np.round(ntrls.mean(0)).astype(int)
# # n_samp = ntrls.min(0)

# pp = {}
# for this_sess in all_neurs.values():
#     for i, (lab, rep) in enumerate(this_sess.items()):
        
#         ix = np.random.choice(range(len(rep)), n_samp[i])
#         X_cond = rep[ix]
        
#         if lab in pp.keys():
#             pp[lab] = np.hstack([pp[lab], X_cond])
#         else:
#             pp[lab] = X_cond

# X = np.vstack([foo for foo in pp.values()])
# X_grp = np.stack([foo.mean(0) for foo in pp.values()])
# idx = np.concatenate([np.ones(len(wa))*i for i,wa in enumerate(pp.values())])

#%%

# frac = 0.8 
frac = 1.0

ntrls = []
nneur = []
for this_sess in all_neurs.values():
    ntrls.append( [len(foo) for foo in this_sess.values()] )
    nneur.append( [len(foo.T) for foo in this_sess.values()] )

ntrls = np.array(ntrls)
nneur = np.array(nneur)

## how many trials to sample from each condition?
# ntrnsamp = np.round(frac*ntrls.mean(0)).astype(int)
# ntstsamp = np.round((1-frac)*ntrls.mean(0)).astype(int)
ntrnsamp = int(np.round(frac*ntrls.mean())) * np.ones(ntrls.shape[1], dtype=int)
ntstsamp = int(np.round((1-frac)*ntrls.mean())) * np.ones(ntrls.shape[1], dtype=int)
# n_samp = ntrls.min(0)

ntrn = np.floor(frac*ntrls).astype(int)

pp_trn = {}
pp_tst = {}
for i, this_sess in enumerate(all_neurs.values()):
    for j, (lab, rep) in enumerate(this_sess.items()):
        
        trnset = np.random.choice(range(ntrls[i,j]), ntrn[i,j], replace=False)
        tstset = np.setdiff1d(range(ntrls[i,j]), trnset)
        
        ix_trn = np.random.choice(trnset, ntrnsamp[j])
        X_trn = rep[ix_trn]
    
        ix_tst = np.random.choice(tstset, ntstsamp[j])
        X_tst = rep[ix_tst]
        
        if lab in pp_trn.keys():
            pp_trn[lab] = np.hstack([pp_trn[lab], X_trn])
            pp_tst[lab] = np.hstack([pp_tst[lab], X_tst])
        else:
            pp_trn[lab] = X_trn
            pp_tst[lab] = X_tst

Xtrn = np.vstack([foo for foo in pp_trn.values()])
Xtst = np.vstack([foo for foo in pp_tst.values()])
Xtrn_grp = np.stack([foo.mean(0) for foo in pp_trn.values()])
Xtst_grp = np.stack([foo.mean(0) for foo in pp_tst.values()])
idx = np.concatenate([np.ones(len(wa), dtype=int)*i for i,wa in enumerate(pp_trn.values())])

# Xtrn_grp = Xtrn_grp / np.sqrt(np.mean((Xtrn_grp - Xtrn_grp.mean(0))**2))
# Xtst_grp = Xtst_grp / np.sqrt(np.mean((Xtst_grp - Xtst_grp.mean(0))**2))

labels = np.char.array(list(pp_trn.keys()))

stims = labels.partition('|')[:,0]
targ = labels.partition('|')[:,-1]
targ_pos = stims.find(targ)

#%%

trn = np.concatenate([np.random.choice(np.where(idx==i)[0], n_samp[i]//2, replace=False) for i in np.unique(idx)])
tst = np.arange(len(idx))[~np.isin(np.arange(len(idx)),trn)]

#%%

# kays = [2,5,10,50,100,150,200,250]
# kays = [5,10,50,100,200,300,400]
# kays = []
kays = np.arange(2,15)

args = {'nonneg':True,
        'sparse_reg': 3,
        'tree_reg': 1e-2,
        'weight_pr_reg': 1e-1,
        # 'fit_intercept': False,
        }

opt_args = {'initial_temp': 100,
            'decay_rate': 0.9,
            'period': 10,
            }

n_run = 50

trn = []
tst = []
for _ in range(n_run):
    tr = []
    ts = []
    for k in tqdm(kays):
        
        # mod = bae_models.BiPCA(k, sparse_reg=1e-4)
        mod = bae_models.SemiBMF(k, **args)
        
        # wa,ba = bae_util.impcv(mod, X, verbose=True, **opt_args)
        wa,ba = bae_util.impcv(mod, Xtrn_grp[:,is_pfc], verbose=False, **opt_args)
        
        tr.append(1*wa)
        ts.append(1*ba)
    trn.append(tr)
    tst.append(ts)

plt.plot(kays, np.mean(trn,axis=0))
plt.plot(kays, np.mean(tst, axis=0))

#%%

n_chain = 20
# k = 50
k = 8

args = {'nonneg':True,
        'sparse_reg': 3,
        'tree_reg': 1e-2,
        'weight_pr_reg': 1e-1,
        # 'fit_intercept': False,
        }

opt_args = {'initial_temp': 100,
            'decay_rate': 0.9,
            'period': 50,
            }

mods = [bae_models.SemiBMF(k, **args) for i in range(n_chain)]
ens = [m.fit(Xtrn_grp[:,is_pfc], **opt_args) for m in mods]

allW = np.hstack([m.W for m in mods])
allS = np.hstack([m.S for m in mods])

#%%

K = k

kmn = KMeans(K)
# kmn.fit(allW.T)
kmn.fit(allS.T)

avg = util.group_mean(allS, kmn.labels_)
quality = np.argsort(np.log2(avg+1e-6).mean(0))

plt.imshow(avg[:,quality], cmap='binary')

#%%

xdi = np.argsort(np.argsort(quality)[kmn.labels_])
plt.imshow(allW[:,xdi]) 
plt.vlines(np.where(np.diff(np.sort(np.argsort(quality)[kmn.labels_]))), ymin=0, ymax=plt.ylim()[0], linestyle='solid')

#%%



