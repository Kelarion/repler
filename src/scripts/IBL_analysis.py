
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
names = list(X.keys())

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

draws = 500
ranks = list(range(2,8))

X = neurz[:,area==1]

N = len(X)    

hams = []
neal = bae_util.Neal(decay_rate=0.98, period=2, initial=5)
for r in ranks:
    
    ham = []
    
    for draw in tqdm(range(draws)):
            
        idx = np.random.permutation(range(N))
        trn = idx[:N//2]
        tst = idx[N//2:]
        
        trnmod = bae_models.GaussBAE(r, sparse_reg=1e-2)
        tstmod = bae_models.GaussBAE(r, sparse_reg=1e-2)
        # trnmod = bae_models.GaussBAE(r, center=False, tree_reg=0.1, sparse_reg=1e-2)
        # tstmod = bae_models.GaussBAE(r, center=False, tree_reg=0.1, sparse_reg=1e-2)
        # mod = bae_models.GaussBAE(Strue.shape[1], sparse_reg=1e-2, center=False)
        
        en = neal.fit(trnmod, X[trn], verbose=False)
        en = neal.fit(tstmod, X[tst], verbose=False)
        
        pred = (X[tst] - X[trn].mean(0))@trnmod.W
        
        ham.append(df_util.permham(tstmod.S, 1*(pred > 0)).mean())
    
    # plt.hist(ham, density=True, bins=20, alpha=0.5)
    
    hams.append(ham)

#%%

ranks = list(range(2,17))

thisX = neurz[:,area==1]

N = len(X)    

neal = bae_util.Neal(decay_rate=0.98, period=2, initial=5)
S = []
# cv = []
for r in ranks:
    
    mod = bae_models.GaussBAE(r, center=False, tree_reg=0.1, sparse_reg=1e-2)
    en = neal.fit(mod, thisX)
    S.append(mod.S*1)
    # ens, esses = neal.cv_fit(mod, thisX, verbose=True)
    # cv.append(np.mean(ens))
    
allS = np.hstack(S)
Sunq, idx = np.unique(np.mod(allS+allS[[0]], 2),axis=1, return_inverse=True)    
r = np.repeat(ranks, ranks)

I = np.zeros((len(Sunq.T), len(ranks)))
I[idx, r-r.min()] = 1

hams = []
for i,Si in enumerate(S):
    
    for Sj in S[i+1:]:
        hams.append(df_util.permham(Si,Sj))


#%%

# thisX = neurz[:,area==1]
thisX = Xavg

# r = 100
r = 16

S = []
W = []
Xpred = []
scl = []
for _ in range(10):
    
    # mod = bae_models.KernelBAE(r, penalty=1)
    mod = bae_models.GaussBAE(r, center=False, tree_reg=0.1, sparse_reg=1e-2)
    en = neal.fit(mod, thisX)
    S.append(mod.S*1)
    W.append(mod.W*1)
    Xpred.append(mod())
    scl.append(mod.scl)
    
allS = np.hstack(S)
Sunq,cnt = np.unique(np.mod(allS+(allS.mean(0)>0.5),2),axis=1, return_counts=True)

hams = np.zeros((10,10))
for i,Si in enumerate(S):
    for j,Sj in enumerate(S[i+1:]):
        
        Siunq = np.unique(np.mod(Si+Si[[0]],2), axis=1)
        Sjunq = np.unique(np.mod(Sj+Sj[[0]],2), axis=1)
        
        hams[i,i+j+1] = df_util.permham(Siunq, Sjunq).sum()

#%%
def meanprediction(X, r, draws, **kwargs):
    
    Xpred = []
    for _ in range(draws):
        
        mod = bae_models.GaussBAE(r, **kwargs)
        en = neal.fit(mod, X, verbose=False)
        Xpred.append(mod())
    
    return np.mean(Xpred, axis=0)

#%%

this_area = 'MOs'

Xtrn = []
Xtst = []
for sess in trl[this_area]:
    
    xtrn = []
    xtst = []
    for cnd in sess.keys():
        
        T = len(sess[cnd])
        idx = np.isin(range(T),np.random.choice(range(T), T//2, replace=False))
        xtrn.append(sess[cnd][idx].mean(0))
        xtst.append(sess[cnd][~idx].mean(0))
    
    Xtrn.append(np.array(xtrn))
    Xtst.append(np.array(xtst))

Xtrn = np.hstack(Xtrn)
Xtst = np.hstack(Xtst)

#%%

mod = bae_models.GaussBAE(12, center=False, tree_reg=0.1)
neal = bae_util.Neal(decay_rate=0.98)
en = neal.fit(mod, Xtrn)


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

# this_area = 'MOs'
this_area = 'VISp'
# this_area = 'ACAd'
this_session = 7

# Zall = []
# for this_session in range(len(trl[this_area])):
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

# cv = []
# eses = []
# neal = bae_util.Neal(decay_rate=0.98, initial=1, period=2)
# for k in tqdm(range(2, 29)):
#     mod = bae_models.GeBAE(k, tree_reg=0.1, weight_reg=1e-1)
#     ba = neal.cv_fit(mod, zZ, draws=10, folds=10)
#     cv.append(np.mean(ba[0]))
#     eses.append(ba[1])
    
#%%

plt.imshow(util.correlify(util.center(Z@Z.T)), 'bwr', vmin=-1, vmax=1)
# plt.imshow(util.center(Z@Z.T))

deez = np.where(np.abs(np.diff(cond))>0)[0]

for d in deez:
    plt.plot([d,d],plt.xlim(),'k--')
    plt.plot(plt.xlim(), [d,d],'k--')
    
#%% Single-trial data with all variables

fils = os.listdir(SAVE_DIR+'/IBL_sessions_lorenzo/')

align_to = 'time_from_onset'
# align_to = 'time_from_response'
# align_to = 'time_from_firstmove'

bins = 10
t0 = 0
dt = 0.1

neur = {}
meta = {}

tossed = []
pids = []
for fil in fils: 
    
    match = re.findall('sessions_([a-zA-Z]+)_*', fil)
    if len(match) != 1:
        continue
    
    data = pkl.load(open(SAVE_DIR+'/IBL_sessions_lorenzo/'+fil,'rb'))
    
    Xsess = []
    metasess = {'context':[],
                'movement':[],
                'whisking_left':[],
                'whisking_right':[],
                'response':[],
                'reward':[],
                'choice':[],
                'position':[],
                'contrast':[],
                'response_time':[],
                }
    
    for dat in data:
        
        trl_time = dat[align_to]
        
        # mask = (dat[align_to] >= t0)*(dat[align_to] <= t0 + dt)
        mask = (dat['response_time'] >= 0.1)*(dat['response_time'] <= 0.8)
        # mask *= (dat['context'] == 0.5)
        mask *= (dat['context'] < 0.5) | (dat['context'] > 0.5)
        trial = dat['trial'][mask]
        
        X = []
        for i in range(bins):
            times = (trl_time >= t0 + i*dt)*(trl_time <= t0 + (i+1)*dt)
            X.append(util.group_mean(dat['raster'][mask*times], dat['trial'][mask*times], axis=0))
        X = np.stack(X).swapaxes(0, 1)
        # mavg = np.apply_along_axis(np.convolve, 0, X, np.ones(50)/50)
        # deez = mavg[50:-50].min(0) > 0
        
        # Xsess.append(X[:,deez])
        # tossed.append(1 - deez.mean())
        
        Xsess.append(X)
        
        pids.append(dat['PID'])
        
        for key in metasess.keys():
            metasess[key].append(util.group_mean(dat[key][mask], trial, axis=0))
            
    neur[match[0]] = Xsess
    meta[match[0]] = metasess

areas = neur.keys()


#%%

neurons = []
conditions = []
area = []
others = []
session = []

excl_keys = ['whisking_left', 'whisking_right', 'context','contrast','position']
# excl_keys = ['whisking_left', 'whisking_right', 'choice','contrast','position']
zkeys = []

for this_area, these_sess in neur.items():
    
    for this_sess, X  in enumerate(these_sess):
        
        ## Lorenzo's processing
        wl = meta[this_area]['whisking_left'][this_sess]
        wr = meta[this_area]['whisking_right'][this_sess]
        ctx = meta[this_area]['context'][this_sess]
        ctr = meta[this_area]['contrast'][this_sess]
        pos = meta[this_area]['position'][this_sess]
        cho = meta[this_area]['choice'][this_sess]
        
        Y = np.stack([1*((wl+wr) > np.median(wl+wr)),
                      1*(ctx>0.5),
                      1*(pos>0),
                      1*(ctr>0.13),
                      ]).T
        
        # Y = np.stack([1*((wl+wr) > np.median(wl+wr)),
        #               1*(cho>0),
        #               1*(pos>0),
        #               1*(ctr>0.13),
        #               ]).T
        
        Z = []
        for key in meta[this_area].keys():
            if key not in excl_keys:
                Z.append(meta[this_area][key][this_sess])
                if key not in zkeys:
                    zkeys.append(key)
        Z = np.array(Z).T
        
        Yunq, cond, cnt = np.unique(Y, axis=0, return_inverse=True, return_counts=True)
        
        excl = (ctx==0.5)*(ctr<0.06)
        # excl = (ctr<0.06)
        
        if (len(Yunq) < 16) or np.any(cnt < 5) or X.shape[1] < 1:
            continue
        
        for thisy, cnd in zip(Yunq, np.unique(cond)):
            
            deez = (cond == cnd)*(~excl)
            
            neurons.append(X[deez])
            conditions.append(thisy)
            area.append(this_area)
            others.append(Z[deez])
            session.append(this_sess)
                
neurons = np.array(neurons, dtype=object)
conditions = np.array(conditions)
session = np.array(session)
area = np.array(area)
others = np.array(others, dtype=object)

#%%

# pp = util.ppop([np.median(x,axis=0, keepdims=True) for x in neurons], 
#                conditions, 
#                session=session,
#                subsets=area, 
#                independent=True, 
#                K=1,
#                )

pp = util.ppop(neurons, 
               conditions, 
               session=session,
               subsets=area, 
               independent=True, 
               K=5,
               )


#%%

## check: ORBvl, RSP, FRP, AId/p/v, PL, VISa

# X_ = pp['data'][...,pp['neur_labels']=='AId']
X_ = pp['data'][...,pp['neur_labels']=='MOs'].squeeze()
# X_ = pp['data'][...,pp['neur_labels']=='MOp']
# X_ = pp['data'][:,pp['neur_labels']=='ACAd']
# X_ = pp['data'][:,pp['neur_labels']=='PL']
# X_ = pp['data'][...,pp['neur_labels']=='RSPd']

# kays = [2,3,4,5,10,15,20]
kays = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# kays = [10]

args = {
        'nonneg':True,
        # 'nonneg': False,
        'weight_pr_reg': 1e-1,
        # 'weight_pr_reg': 0,
        'tree_reg': 1e-2,
        'sparse_reg': 1,
        # 'tree_reg': 0,
        # 'weight_l1_reg': 1e-3,
        # 'weight_l2_reg': 1e-2,
        # 'fit_intercept': True,
        # 'fit_intercept': False,
        }

# args = {
#         'nonneg':True,
#         # 'nonneg': False,
#         'weight_pr_reg': 1e-1,
#         # 'weight_pr_reg': 0,
#         'tree_reg': 1e-2,
#         'sparse_reg': 1e-2,
#         # 'tree_reg': 0,
#         # 'weight_l1_reg': 1e-3,
#         'weight_l1_reg': 0,
#         # 'weight_l2_reg': 1e-2,
#         # 'fit_intercept': True,
#         # 'fit_intercept': False,
#         'slab_prior': 1,
#         }

opt_args = {'initial_temp': 10,
            'decay_rate': 0.9,
            'period': 10,
            # 'hot_start': True,
            'hot_start': False,
            # 'lr': 1e-1,
            'min_temp': 1,
            }

n_run = 20

trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
         
        mod = bae_models.SemiBMF(k,**args)
        # mod = bae_models.SpikeNMF(k,**args)
        
        # wa,ba = bae_util.impcv(mod, X, verbose=True, **opt_args)
        # wa,ba = bae_util.impcv(mod, Xpt[singles], verbose=False, **opt_args)
        wa,ba = bae_util.impcv(mod, X_/X_.std(), verbose=False, n_sample=100, **opt_args)
        
        trn[i] += wa / n_run
        tst[i] += ba / n_run

plt.plot(kays, trn)
plt.plot(kays, tst, '--')


#%%

mod = bae_models.BiPCA(12, center=False, tree_reg=0.1)
neal = bae_util.Neal(decay_rate=0.98)
en = neal.fit(mod, Xtrn)


