CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/mckenzie_data/' 

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
    return (x - center*x.mean(0))/np.sqrt(np.mean((x-x.mean(0))**2, axis=0))

#%%

# binsize = None
binsize = 0.5

starttime = 0.0

# center = True
center = False

stims = np.array(['A', 'B', 'C', 'D'])
place = np.array(['1', '2', '3', '4'])
context = np.array(['w', 'e'])

sess_type = 'CDEF'

neurons = []
condition = []
animal = []
session = []

for subj in os.listdir(LOAD_DIR):
    for sess in os.listdir(f"{LOAD_DIR}/{subj}"):
        
        if sess_type not in sess:
            continue
        
        neu = pd.read_hdf(f"{LOAD_DIR}/{subj}/{sess}/spikes.h5")
        beh = pd.read_hdf(f"{LOAD_DIR}/{subj}/{sess}/events.h5")
        
        t0 = beh['time'].to_numpy()
        t1 = beh['duration'].to_numpy()
        
        deez = ((~beh['exclude'])*(beh['correct'])).to_numpy()
        # deez = ((~beh['exclude'])).to_numpy()
        # deez = ((beh['correct'])).to_numpy()
        
        ctx = 1*(beh['context'].to_numpy() == 'east')
        pos = beh['position'].to_numpy()
        stim = beh['odor'].to_numpy()
        # correct = beh['correct'].to_numpy()
    
        if binsize is None:
            dt = np.min([t1[deez], 2*np.ones(np.sum(deez))], axis=0)
        else:
            dt = binsize
            deez = deez * ((starttime + binsize) < t1)
        
        unqs, idx = np.unique([ctx[deez], pos[deez], stim[deez]], axis=1, return_inverse=True)
        if len(unqs.T) < 16:
            print('bad')
            continue
        
        labs = context[unqs[0]] + place[unqs[1]-1] + stims[unqs[2]-1]
        
        bins = np.array([starttime+t0[deez], starttime+t0[deez]+dt]).T.flatten()
            
        X = []
        for neur in range(len(neu)):
            X.append(np.histogram(neu['spikes'][neur], bins)[0][::2] )
        X = np.array(X).T
        
        X = zscore(X[:,X.sum(0)>0], center=center)
        
        for i,lab in enumerate(labs):
            neurons.append(X[idx==i])
            condition.append(lab)
            animal.append(subj)
            session.append(sess)

neurons = np.array(neurons, dtype=object)
condition = np.array(condition)
animal = np.array(animal)
session = np.array(session)

# unqs = [labs.startswith('w'),
#    

#%%

pp = util.ppop(neurons, 
               condition, 
               animal+session,
               independent=True,
               jagged=False,
               K=6)


#%%

pp = util.ppop([np.median(x,axis=0, keepdims=True) for x in neurons], 
               condition, 
               animal+session,
               independent=True, 
               K=1)
# pp = util.ppop([np.mean(x,axis=0, keepdims=True) for x in neurons], 
#                condition, 
#                animal+session,
#                independent=True, 
#                K=1)

Xmd = pp['data']

#%%

## Parallelism, try other train/test combinations
## different times in the trial
## how is trial duration defined?

# binsize = None  # use the full event duration
binsize = 0.5

starttime = 0.0

# center = True
center = False

stims = np.array(['A', 'B', 'C', 'D'])
place = np.array(['1', '2', '3', '4'])
context = np.array(['w', 'e'])

all_neurs = {}

sess_type = 'CDEF'

for animal in os.listdir(LOAD_DIR):
    for sess in os.listdir(f"{LOAD_DIR}/{animal}"):
        
        if sess_type not in sess:
            continue
        
        neu = pd.read_hdf(f"{LOAD_DIR}/{animal}/{sess}/spikes.h5")
        beh = pd.read_hdf(f"{LOAD_DIR}/{animal}/{sess}/events.h5")
        
        t0 = beh['time'].to_numpy()
        t1 = beh['duration'].to_numpy()
        
        deez = ((~beh['exclude'])*(beh['correct'])).to_numpy()
        # deez = ((~beh['exclude'])).to_numpy()
        # deez = ((beh['correct'])).to_numpy()
        
        ctx = 1*(beh['context'].to_numpy() == 'east')
        pos = beh['position'].to_numpy()
        stim = beh['odor'].to_numpy()
        # correct = beh['correct'].to_numpy()
    
        if binsize is None:
            dt = t1[deez]
        else:
            dt = binsize
            deez = deez * ((starttime + binsize) < t1)
        
        unqs, idx = np.unique([ctx[deez], pos[deez], stim[deez]], axis=1, return_inverse=True)
        if len(unqs.T) < 16:
            continue
        
        labs = context[unqs[0]] + place[unqs[1]-1] + stims[unqs[2]-1]
        
        bins = np.array([starttime+t0[deez], starttime+t0[deez]+dt]).T.flatten()
            
        X = []
        for neur in range(len(neu)):
            X.append(np.histogram(neu['spikes'][neur], bins)[0][::2] / dt)
        X = np.array(X).T
        
        X = zscore(X[:,X.sum(0)>0], center=center)
        
        all_neurs[animal+'_'+sess] = {}
        for i,lab in enumerate(labs):
            all_neurs[animal+'_'+sess][lab] = X[idx==i]

Xmd = np.hstack([np.stack([np.median(v, axis=0) for v in n.values()]) for n in all_neurs.values()])
# Xmd = np.hstack([np.stack([np.mean(v, axis=0) for v in n.values()]) for n in all_neurs.values()])

X_ = (Xmd) / la.norm((Xmd-Xmd.mean(0)), axis=1, keepdims=True)

labs = np.char.array(list(list(all_neurs.values())[0].keys()))

# unqs = [labs.startswith('w'),
#         ]

Xmd = Xmd[:,Xmd.std(0)>0]

#%% Reproduce the dendrogram, even if it's the wrong one

# Xmd = np.hstack([np.stack([np.median(v, axis=0) for v in n.values()]) for n in all_neurs.values()])

X_ = (Xmd-Xmd.mean(0)) / la.norm((Xmd-Xmd.mean(0)), axis=1, keepdims=True)

agg = AgglomerativeClustering(n_clusters=None,
                              distance_threshold=0,
                              compute_full_tree=True)

agg.fit(X_)
# agg.fit(1-util.correlify(util.center(Xmd@Xmd.T)))

S = np.eye(len(X_))
for pair in agg.children_:
    S = np.hstack([S, S[:,pair].sum(1, keepdims=True)])

from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(agg, labels=labs)

#%%

# X_ = Xmd
X_ = Xmd / Xmd.std(0)
# X_ = Xmd - Xmd.mean(0)
# X_ = (Xmd / Xmd.std(0)) - (Xmd / Xmd.std(0)).mean(0)

mod = bae_models.SemiBMF(7,
                         nonneg=True, 
                         sparse_reg=1e-2,
                         weight_pr_reg=1, 
                         tree_reg=1e-0,
                         weight_l2_reg=1,
                         )
# mod = bae_models.SpikeNMF(4,
#                          nonneg=True, 
#                          sparse_reg=1e-2,
#                          weight_pr_reg=1,
#                          tree_reg=1e-1,
#                          weight_l2_reg=1,
#                          )

# mod = bae_models.KernelBMF2(6,
#                            sparse_reg=1e-1,
#                            tree_reg=1,
#                            uniform_scale=False,
#                            )

en = mod.fit(X_ / X_.std(), 
             initial_temp=100, 
             decay_rate=0.9, 
             min_temp=1, 
             period=50, 
             scl_lr=1e-3,
             )

samps = mod.sample(X_ / X_.std(), n_samp=1000)
# samps = mod.sample(X_ / X_.std(), n_samp=1000, slab=False)

# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.imshow(samps.mean(0))

#%%

value = labs.startswith('w')*(labs.count('A')+labs.count('C')) + labs.startswith('e')*((labs.count('B')+labs.count('D')))
# value = 1*(np.isin(unqs[2], [1,3])*(unqs[0]==1)) + 1*(np.isin(unqs[2], [2,4])*(unqs[0]==0))
pos = np.eye(4)[unqs[1]-1].T
odor = np.eye(4)[unqs[2]-1].T
context = unqs[0]


#%%

# kays = [2,3,4,5,10,15,20]
kays = [2,3,4,5,6,7,8,9,10]
# kays = [10]

# args = {
#         'nonneg':True,
#         # 'nonneg': False,
#         'weight_pr_reg': 1e-1,
#         # 'weight_pr_reg': 0,
#         'tree_reg': 1e-2,
#         'sparse_reg': 1e-2,
#         # 'tree_reg': 0,
#         # 'weight_l1_reg': 1e-3,
#         # 'weight_l2_reg': 1e-2,
#         # 'fit_intercept': True,
#         # 'fit_intercept': False,
#         }

args = {
        'nonneg':True,
        # 'nonneg': False,
        'weight_pr_reg': 1e-1,
        # 'weight_pr_reg': 0,
        'tree_reg': 1e-2,
        'sparse_reg': 1e-2,
        # 'tree_reg': 0,
        # 'weight_l1_reg': 1e-3,
        'weight_l1_reg': 0,
        # 'weight_l2_reg': 1e-2,
        # 'fit_intercept': True,
        # 'fit_intercept': False,
        'slab_prior': 1,
        }

opt_args = {'initial_temp': 10,
            'decay_rate': 0.9,
            'period': 10,
            # 'hot_start': True,
            'hot_start': False,
            # 'lr': 1e-1,
            }

n_run = 50

trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
         
        # mod = bae_models.SemiBMF(k,**args)
        mod = bae_models.SpikeNMF(k,**args)
        
        # wa,ba = bae_util.impcv(mod, X, verbose=True, **opt_args)
        # wa,ba = bae_util.impcv(mod, Xpt[singles], verbose=False, **opt_args)
        wa,ba = bae_util.impcv(mod, X_, verbose=False, **opt_args)
        
        trn[i] += wa / n_run
        tst[i] += ba / n_run

plt.plot(kays, trn)
plt.plot(kays, tst, '--')
# plt.plot(kays, np.mean(trn,axis=0))
# plt.plot(kays, np.mean(tst, axis=0))

#%%

n_chain = 20
k = 6

args = {'nonneg':True,
        # 'sparse_reg': 3,
        'tree_reg': 1e-2,
        'sparse_reg': 1e-2,
        'weight_pr_reg': 1e-2,
        'weight_l2_reg': 1e-2,
        # 'fit_intercept': False,
        }

opt_args = {'initial_temp': 100,
            'decay_rate': 0.9,
            'period': 50,
            }

# mods = [bae_models.SpikeNMF(k, **args) for i in range(n_chain)]
mods = [bae_models.SemiBMF(k, **args) for i in range(n_chain)]
ens = [m.fit(X_, **opt_args) for m in mods]

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



#%%

# frac = 0.8 
frac = 1.0


ntrls = []
nneur = []
for this_sess in all_neurs.values():
    ntrls.append( [foo.shape[0] for foo in this_sess.values()] )
    nneur.append( [foo.shape[-1] for foo in this_sess.values()] )

these_sess = [n for n in ntrls if len(n) == np.max([len(n) for n in ntrls])]

ntrls = np.array(these_sess)
# nneur = np.array(nneur)

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
            pp_trn[lab] = np.concatenate([pp_trn[lab], X_trn], axis=-1)
            pp_tst[lab] = np.concatenate([pp_tst[lab], X_tst], axis=-1)
        else:
            pp_trn[lab] = X_trn
            pp_tst[lab] = X_tst

Xtrn = np.vstack([foo for foo in pp_trn.values()])
Xtst = np.vstack([foo for foo in pp_tst.values()])
Xtrn_grp = np.stack([foo.mean(0) for foo in pp_trn.values()])
Xtst_grp = np.stack([foo.mean(0) for foo in pp_tst.values()])
idx = np.concatenate([np.ones(len(wa), dtype=int)*i for i,wa in enumerate(pp_trn.values())])

#%%

order = util.LexOrder()

ps = np.zeros((int(spc.binom(16,2)), int(spc.binom(16,2)))) * np.nan

this_rep = X_
# this_rep = np.repeat(np.eye(4), 4, axis=0)
# this_rep = wa@np.diag([1.0,1.0,3,3,3,3]) + np.random.randn(*wa.shape)*0.1

for ij in range(len(ps)):
    i,j = order.inv(ij)
    
    x = this_rep[i] - this_rep[j]
    
    for kl in range(len(ps)):
        
        k,l = order.inv(kl)
        
        if len(np.unique([i,j,k,l])) < 4:
            continue
        
        y = this_rep[k] - this_rep[l]
        
        ps[ij,kl] = x@y / np.sqrt(x@x * y@y)

#%%
diffs, ix = np.unique(np.mod(mod.S[aye] + mod.S[jay], 2)[:,[2,5,3,0,4,1]], axis=0, return_inverse=True)

#%%

diff_strings = []
for d in diffs:

    diff_string = ''
    for i in np.where(d)[0]:
        diff_string += string.ascii_lowercase[i]
    diff_strings.append(diff_string)

#%%
plt.figure()
plt.imshow(ps[np.argsort(ix),:][:,np.argsort(ix)], 'bwr', vmin=-1, vmax=1)

foo = np.where(np.diff(np.sort(ix), prepend=0))[0] - 0.5
foo = np.concatenate([foo, [len(ps)-0.5]])

plt.vlines(foo, len(ps), 0, 'k')
plt.hlines(foo, len(ps), 0, 'k')

plt.xlim([0, len(ps)])
plt.ylim([0, len(ps)])

plt.xticks(foo - np.diff(foo, prepend=0)/2, labels=diff_strings)
plt.yticks(foo - np.diff(foo, prepend=0)/2, labels=diff_strings)

#%%
ccgp = np.zeros((4,4))
for i in range(4):
    trainset = labs.endswith(str(i+1))[idx]
    clf.fit(Xtrn[trainset], value[idx][trainset])
    for j in range(4):
        perf = clf.score(Xtrn[labs.endswith(str(j+1))[idx]], value[idx][labs.endswith(str(j+1))[idx]])
        ccgp[i,j] = 1*perf

plt.imshow(ccgp, 'binary', vmin=0, vmax=1)