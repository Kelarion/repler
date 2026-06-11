CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/courelis_data/data/' 

import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)

import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

import pandas as pd

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

import h5py

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

def zscore(x, center=True):
    return (x - center*x.mean(0))/(x.std(0) + 1e-6)

#%%

with open(LOAD_DIR + "inference_absent_sessions.txt", 'r') as f:
    noinf = f.readlines()[0].strip().split(',')
with open(LOAD_DIR + "inference_present_sessions.txt", 'r') as f:
    yesinf = f.readlines()[0].strip().split(',')

area_names = {1:'Amy',
              2:'dAcc',
              3:'Hpc',
              4:'preSMA',
              9:'vmPFC',
              12:'vTC'}

#%%

neurons = []
subject = []
session = []
area = []
condition = []
inf_present = []

inf_sess = {}
for subj in os.listdir(LOAD_DIR):
    
    if not os.path.isdir(LOAD_DIR + subj):
        continue
    
    all_sess = np.unique([re.findall("_(\d+)", s)[0] for s in os.listdir(LOAD_DIR + f"/{subj}")])
    # if len(all_sess) < 2:
    #     print(f"{subj} not enough")
    #     continue
    
    inf_sess[subj] = []
    
    for sess in all_sess:
        
        neur = scio.loadmat(LOAD_DIR + f"/{subj}/neural_{sess}.mat")['X']
        beh = pd.read_csv(LOAD_DIR + f"/{subj}/behav_{sess}.csv")
        areas = scio.loadmat(LOAD_DIR + f"/{subj}/region_{sess}.mat")['region'].squeeze()

        neur[...,0] = zscore(neur[...,0], center=False)
        # neur[...,0] -= neur[...,1]

        filt = (beh.iscorrect == 1)
        
        conds = np.stack([beh.context[filt], beh.stim_id[filt]]).T
        # conds = np.stack([beh.stim_id, beh.context, beh.iscorrect]).T
        
        unqs, idx = np.unique(conds, axis=0, return_inverse=True)
        
        if (f"{subj}_{sess}" in yesinf):
            inf_sess[subj].append(sess)
        
        if len(unqs) < 8:
            print(f"{subj}_{sess} not enough")
            continue
        
        for i,lab in enumerate(unqs):
            for this_area in np.unique(areas):
                
                if (f"{subj}_{sess}" in yesinf):
                    inf_present.append(1)
                else:
                    inf_present.append(0)
                
                neurons.append(neur[filt][idx==i][:,areas==this_area,0])
                condition.append(tuple(lab))
                session.append(sess)
                subject.append(subj)
                area.append(area_names[this_area])

subj_type = np.array([1*('1' in inf_sess[s]) if ('2' in inf_sess[s]) else -1 for s in subject])
neurons = np.array(neurons, dtype=object)
condition = np.array(condition)
subject = np.array(subject)
area = np.array(area)
inf_present = np.array(inf_present)

#%%

# filt = (subj_type == subj_group) * (inf_present == 1)
filt = (inf_present == 1)

pp = util.ppop([np.mean(x,axis=0, keepdims=True) for x in neurons[filt]], 
               condition[filt], 
               session=(subject+session)[filt],
               subsets=area[filt], 
               independent=True, 
               K=1,
               )

# pp = util.ppop(neurons[filt], 
#                condition[filt], 
#                session=(subject+session)[filt],
#                subsets=area[filt], 
#                independent=True,
#                K=13,
#                )

X = pp['data'][:,pp['neur_labels']=='Hpc']

#%%

# # subj_group = 1
# only_inference = True

# # filt = (subj_type == subj_group) * (inf_present == 1)
# filt = (inf_present == 1)

# pp = util.ppop(neurons[filt], 
#                condition[filt], 
#                session=(subject+session)[filt],
#                subsets=area[filt], 
#                independent=True, 
#                K=13,
#                )


# pp = util.ppop(neurons[filt], condition[filt], subsets=area[filt])

# # pool_subjects = False
# pool_subjects = True

# if pool_subjects:
#     concat = {f"{k1}_{k2}":v2 for k1,v1 in all_neurs.items() for k2,v2 in v1.items()}
#     pp_trn, pp_tst = util.ppop(concat)
    
#     Xtrn = np.vstack([foo for foo in pp_trn.values()])
#     Xtst = np.vstack([foo for foo in pp_tst.values()])
#     Xtrn_grp = np.stack([foo.mean(0) for foo in pp_trn.values()])
#     Xtst_grp = np.stack([foo.mean(0) for foo in pp_tst.values()])
#     idx = np.concatenate([np.ones(len(wa), dtype=int)*i for i,wa in enumerate(pp_trn.values())])

# else:
#     Xtrn = {}
#     Xtst = {}
#     Xtrn_avg = {}
#     Xtst_avg = {}
#     for subj in all_neurs.keys():
#         pp_trn, pp_tst = util.ppop(all_neurs[subj])

#         Xtrn[subj] = np.vstack([foo for foo in pp_trn.values()])
#         Xtst[subj] = np.vstack([foo for foo in pp_tst.values()])
#         Xtrn_grp = np.stack([np.median(foo,0) for foo in pp_trn.values()])
#         Xtst_grp = np.stack([np.median(foo,0) for foo in pp_tst.values()])

#%%

X_ = X
# X_ = X - X.mean(0)
# X_ = (X / X.std(0)) - (X / X.std(0)).mean(0)

mod = bae_models.SemiBMF(4,
                         nonneg=True, 
                         sparse_reg=5e-1,
                         weight_pr_reg=1, 
                         tree_reg=1e-1, 
                         weight_l2_reg=1,
                         )

# mod = bae_models.SpikeNMF(5,
#                          nonneg=True, 
#                          sparse_reg=1e-1, 
#                          weight_pr_reg=1, 
#                          tree_reg=1e-1, 
#                          weight_l2_reg=1,
#                          )

# mod = bae_models.KernelBMF2(4,
#                            sparse_reg=0.5,
#                            tree_reg=1,
#                            uniform_scale=False,
#                            # l1_reg=1e-3,
#                            )


en = mod.fit(X_,# / X_.std(), 
             initial_temp=100,
             decay_rate=0.9,
             min_temp=1, 
             period=50, 
             scl_lr=1e-3,
             # scl_lr=1e-2,
             )

plt.figure()
samps = mod.sample(X_ , n_samp=1000)
# samps = mod.sample(X / X.std(), n_samp=1000, slab=False)

# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.imshow(samps.mean(0))
# plt.imshow(samps.mean(0)[:,np.argsort(mod.scl)])

#%%


# kays = [2,3,4,5,10,15,20]
kays = [2,3,4,5,6,7,8,9,10]
# kays = [10]

args = {
        'nonneg':True,
        # 'nonneg': False,
        'weight_pr_reg': 1,
        # 'weight_pr_reg': 0,
        'tree_reg': 1e-1,
        'sparse_reg': 1e-1,
        # 'tree_reg': 0,
        # 'weight_l1_reg': 1e-3,
        'weight_l2_reg': 1,
        # 'fit_intercept': True,
        # 'fit_intercept': False,
        }

# args = {
#         'nonneg':True,
#         # 'nonneg': False,
#         'weight_pr_reg': 1,
#         # 'weight_pr_reg': 0,
#         'tree_reg': 1e-1,
#         'sparse_reg': 1e-1,
#         # 'tree_reg': 0,
#         # 'weight_l1_reg': 1e-3,
#         'weight_l1_reg': 1e-1,
#         # 'weight_l2_reg': 1e-2,
#         # 'fit_intercept': True,
#         # 'fit_intercept': False,
#         'slab_prior': 1,
#         }

opt_args = {'initial_temp': 100,
            'decay_rate': 0.8,
            'period': 50,
            # 'hot_start': True,
            'hot_start': False,
            'scl_lr': 1e-3,
            'min_temp': 1,
            # 'lr': 1e-1,
            }

n_run = 50

trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
        
        mod = bae_models.SemiBMF(k,**args)
        # mod = bae_models.SpikeNMF(k,**args)
        
        wa,ba = bae_util.impcv(mod, X/X.std(), verbose=False, n_sample=10, folds=20, **opt_args)
        
        trn[i] += wa / n_run
        tst[i] += ba / n_run

plt.plot(kays, trn)
plt.plot(kays, tst, '--')
# plt.plot(kays, np.mean(trn,axis=0))
# plt.plot(kays, np.mean(tst, axis=0))

