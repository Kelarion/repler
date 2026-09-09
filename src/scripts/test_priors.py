CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/concepts/sbmf/')

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
from matplotlib import cm

import networkx as nx
import cvxpy as cvx

# my code
import util
import df_util
import pt_util
import old_bae
import old_bae_models
import bae_util
import plotting as tpl
import experiments as exp
import bae_priors as nbp   # parallel classes merged into bae_priors (n_chains)
import bae_models

import bae_experiments as nbx


#%%

mod = bae_models.BiPCA(43,
                           tree_reg=0,
                           sparse_reg=1,
                           J_lr=1e-3,
                           # slab=True, ## multiplicative noise
                           n_chains=1,
                           )

en = mod.fit(X / X.std(),
             # period=100,
             # initial_temp=50,
             # decay_rate=0.88,
             initial_temp=0,
             decay_rate=1,
             min_temp=1,
             max_iter=600,
             # scl_lr=1e-3,
             # lr=1e-1,
             # hot_start=False,
             )

mod.collapse()

samps = mod.sample(X_  / X_.std(), n_samp=100)

#%%

nepoch = 150
N = 100
K = 5
# temp = 1e-2

task = df_util.UndirectedModel(K, 'grid', {'d':2})
# task = df_util.UndirectedModel(K, 'tree')
task = df_util.UndirectedModel(K, 'cat', {'d':2}, blowup=3)

n_run = 5

recon = np.zeros((3,3,20))
betas = np.zeros((3,3,20))
for _ in tqdm(range(n_run)):
    # for i,l0_reg in enumerate([0.05, 0.2, 0.5]):
    for i,temp in enumerate([1e-2, 0.5, 1]):
        for j,beta_lr in enumerate([1e-3, 1e-2, 1e-1]):
            for k,l0_reg in enumerate(np.linspace(1e-2, 1, 20)):
            
                S = task.sample(N, temp) ## figure 2
                # S = np.unique(task.sample(10000, 1e-2), axis=0)[np.random.choice(range(36), 1000)]
                # S = np.repeat(np.unique(task.sample(10000, 1e-2), axis=0), 30, axis=0) ## figure 1
                
                prior = nbp.MRFPrior(n_chains=1,
                                     l0_reg=l0_reg,
                                     beta_init=1,
                                     beta_lr=beta_lr,
                                     # beta_mle=False,
                                     beta_mle=True,
                                     )
                
                prior.init_params(S[None])
                
                Jnrm = la.norm(task.J)
                
                ls = []
                for t in range(nepoch):
                    
                    # prior.J_temp = 0.88**t
                
                    Jmod = prior.J
                    ls.append(np.sum(task.J*Jmod)/(Jnrm*la.norm(Jmod) + 1e-6))
                    
                    S = task.sample(N, temp) ## commented out for figure 1
                
                    prior.learn(S[None])
                
                recon[i,j,k] += ls[-1] / n_run
                betas[i,j,k] += prior.beta[0] / n_run
            
# plt.subplot(1,2,1)
# plt.plot(ls)
# plt.ylabel('J recovery (cosine sim)')
# plt.subplot(1,2,2)
# plt.plot(betas)
# plt.ylabel('beta')

# plt.scatter(prior.h, spc.logit(task.sample(1000, temp=1e-2).mean(0)))

    #%%

cols = ['r','g','b']
styl = ['-', '--', '-.']

# data = {'vals': recon.reshape((-1, 20)),
        # 'x': }

# for i,l0_reg in enumerate([0.05, 0.2, 0.5]):
for i,temp in enumerate([1e-2, 0.5, 1]):
    for j,beta_lr in enumerate([1e-3, 1e-2, 1e-1]):
        plt.plot(np.linspace(1e-2, 1, 20), recon[i,j], color=cols[i], linestyle=styl[j])

plt.xlabel('l0_reg')
plt.ylabel('J reconstruction')


#%%

N = 1000
K = 5
# K = 5
# K = 10
temp = 1e-2
# temp = 1

task = nbx.StructuredCats(1, 15, 10,
                        # spec=f"grid{K}(d=2)",
                        spec=f"tree{K}",
                        # spec=f"cat{K}x3",
                        # spec='cat3x2+tree3',
                        N=N,
                        temp=temp,
                        )

data = task.sample()

#%%

k = data['Strue'][0].shape[1]
X = data['X'][0] / data['X'][0].std()

W = sts.ortho_group(data['X'][0].shape[1]).rvs()[:,:k]
b = X.mean(0)

S = 1*(data['X'][0]@W > 0.5)

prior = nbp.MRFPrior(n_chains=1,
                     l0_reg=0.1,
                     beta_init=1,
                     beta_lr=1e-1,
                     beta_mle=False,
                     # beta_mle=True,
                     h_l2_reg=1e-2,
                     )
 
# prior = nbp.ParallelBoltzmannPriorNP(J_lr=1e-1)

prior.init_params(S[None])

lamb = 1
scl = 1

lr = 1e-2
kap = 1e-2

foo = []
ba = []
wa = []
ga = []

R = []
D = []

bae_search._build_dense_search(score, aux_update, prior, diag_gram=False, debug=False,parallel=False):

#%%

# for t in [1, 0.95, 0.9, 0.85, 0.8]:
for t in [2, 1.75, 1.5, 1.25, 1]:

    for _ in range(50):
    
        J,h = prior.coupling()
        
        foo.append(S.mean(0))
        # wa.append(np.abs(J).sum())
        ba.append(h[0])
        wa.append(prior.beta[0])
        
        mse = np.mean((S@W.T - X - b)**2)
        
        D.append(1*mse)
        R.append()
        
        lamb *= np.exp(lr*(mse/np.mean(X**2) - kap))
        # scl *= np.exp(lr*(mse - scl))
        
        # S, _ = search(X@W, S, 1*S, W.T@W, S.T@S, len(S), t, 0, 0, 0, 1/np.sqrt(2*lamb), J[0], h[0])
        S, _ = search(X@W, S, 1*S, W.T@W, S.T@S, len(S), t, 0, 0, 0, 1/np.sqrt(2*lamb), J[0], h[0])
        # S, _ = search(X@W, S, 1*S, W.T@W, S.T@S, len(S), t, 1, 0, 0, scl, J[0], h[0])
        
        W = df_util.krusty(S.T, X.T).T
        
        prior.learn(S[None])
    

#%%

X_ = data['X'][0]
conds = np.unique(data['Strue'][0], axis=0, return_inverse=True)[1]

# mod = bae_models.JBMF(dim_hid=9,
#                         nonneg=False, 
#                         sparse_reg=1,
#                         tree_reg=10,
#                         weight_pr_reg=1,
#                         weight_l2_reg=1e-1,
#                         J_lr=1e-3,
#                         # J_l1_reg=0.0,
#                         )

mod = bae_models.BiPCA(dim_hid=data['Strue'][0].shape[-1],
                           # sparse_reg=1e-1,
                           sparse_reg=1e-1,
                           tree_reg=10,
                           # tree_reg=1e-1,
                           J_lr=1e-2,
                           # J_loss='mle',
                           # J_l1_reg=0.2,
                           # J_prior='mrf',
                           n_chains=8,
                           )

# mod = nbp.ParallelBiPCA(dim_hid=data['Strue'][0].shape[-1],
#                            sparse_reg=1,
#                            tree_reg=0,
#                            J_lr=1e-2,
#                            n_chains=8,
#                            # J_loss='mle',
#                            # J_l1_reg=0.1,
#                            )

en = mod.fit(X_ / X_.std(), 
             decay_rate=0.88,
             min_temp=1,
             initial_temp=10,
             period=50,
             scl_lr=1e-2,
             lr=1e-1,
             # prior_temp=1,
             # prior_min_temp=1e-2,
             # max_iter=1000,
             )

samps = mod.sample(X_ / X_.std(), n_samp=100)

mod.collapse()

# best = mod.best_chain(X_ / X_.std())
# S_ = 1*(samps.mean(0) > 0.9)[best]
# W_ = mod.operator.W[best]
# J_ = mod.latent_prior.J_W[best]
# h_ = mod.latent_prior.J_h[best]

S_ = 1*(samps.mean(0) > 0.9)
W_ = mod.operator.W
J_ = mod.latent_prior.J_W
h_ = mod.latent_prior.J_h

#%%

ls = []
for t in 10**np.linspace(-2, 0, 50):
    mod.latent_prior.temp = t
    ls.append(mod.ppll(X_ / X_.std(), n_samp=10).mean())

#%%

c = mod.n_chains

k = 1
for i in range(c):
    for j in range(c):
        plt.subplot(c, c, k)
        plt.imshow(np.abs(mod.operator.W[i].T@mod.operator.W[j]), 'binary', vmin=0, vmax=1)
        
        k += 1

#%%
aye, jay = df_util.permham_idx(data['Strue'][0], S_)

WtW_ = data['Wtrue'][0][:,aye].T@W_[:,jay]
sgn = np.sign(np.diag(WtW_))

plt.figure()

plt.subplot(2, 3, 1)
# plt.imshow(samps.mean(0)[:,i].reshape((2,5)))
Shat = util.group_mean(S_, conds, axis=0)
tpl.matshow(np.mod(Shat[:,jay] + (1-sgn)/2, 2), cmap='binary', color=(0.5,0.5,0.5))

plt.subplot(2,3, 4)
plt.plot(en)

# samps = mod.sample(X_  / X_.std(), n_samp=100)
# Xhat = mod(samps)
# ll = np.round(mod.loglikelihood(X_/X_.std(), Xhat).mean(),2)
# r2 = np.round(np.mean((X_/X_.std() - Xhat.mean(0))**2) / np.mean((X_/X_.std())**2),2)
# plt.title(f"{r2}, {ll}")

plt.subplot(2,3,2)
StS_ = ((2*data['Strue'][0][:,aye]-1).T@(2*S_[:,jay]-1))/N
plt.imshow(np.abs(StS_), 
           'binary', vmin=0, vmax=1)

plt.subplot(2, 3, 5)
plt.imshow(np.abs(WtW_),'binary', vmin=0, vmax=1)

plt.subplot(2, 3, 3)
plt.imshow(data['Jtrue'][0], 'bwr', vmin=-1, vmax=1)

Jeff = (J_ + J_.T) / 2
mask = df_util.binarize(np.abs(util.vec(Jeff))[None], axis=1).squeeze()
# thresh = np.mean(np.abs(util.vec(Jeff))[np.abs(util.vec(Jeff))> 1e-2]) * 0.1
# mask = np.abs(util.vec(Jeff)) > thresh
Jeff *= util.mat(mask)

plt.subplot(2, 3, 6)
plt.imshow(Jeff[jay][:,jay]*np.outer(sgn,sgn) + np.diag(h_[jay]*sgn), 
           'bwr', vmin=-1, vmax=1)

#%%

X_ = data['X'][0]

# kays = [2,3,4,5,10,15,20]
kays = [2,3,4,5,6,7,8,9]
# kays = [10]

# args = {
#         'nonneg':True,
#         # 'nonneg': False,
#         'weight_pr_reg': 1,
#         # 'weight_pr_reg': 0,
#         # 'tree_reg': 1e-1,
#         # 'sparse_reg': 1,
#         'sparse_reg': 0,
#         'tree_reg': 0,
#         # 'weight_l1_reg': 1e-3,
#         'weight_l2_reg': 1e-1,
#         # 'J_lr': 1e-4,
#         'J_lr': 0,
#         # 'slab': True,
#         'slab': False,
#         # 'fit_intercept': True,
#         # 'fit_intercept': False,
#         }

args = {
        # 'tree_reg': 1e-1,
        # 'sparse_reg': 1,
        'sparse_reg': 1,
        'tree_reg': 0,
        'J_lr': 1e-3,
        # 'J_lr': 0,
        # 'slab': True,
        # 'fit_scl': False,
        'slab': False,
        # 'fit_intercept': True,
        # 'fit_intercept': False,
        'n_chains': 8,
        }

opt_args = {'initial_temp': 100,
            'decay_rate': 0.88,
            'period': 100,
            'hot_start': True,
            # 'hot_start': False,
            # 'scl_lr': 0,
            'scl_lr': 1e-3,
            'min_temp': 1,
            # 'min_temp': 1e-4,
            # 'lr': 1e-2,
            'lr':1e-1,
            }

n_run = 1

trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
full = np.zeros(len(kays))
sigs = []
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
        
        # mod = bae_models.JBMF(k,**args)
        mod = bae_models.BiPCA(k, **args)
        
        # wa,ba = bae_util.loocv(mod, X_/X_.std(), n_sample=10, **opt_args)
        wa,ba = bae_util.impcv(mod, X_/X_.std(), verbose=False, seed=0, n_sample=1, folds=10, max_folds=1, **opt_args)
        # wa,ba = bae_util.gabriel_bicv(mod, X_/X_.std(), n_samp=10, **opt_args)
        
        mod = bae_models.BiPCA(k, **args)
        en = mod.fit(X_ / X_.std(), verbose=False, **opt_args)
        
        trn[i] += np.mean(wa) / n_run
        tst[i] += np.mean(ba) / n_run
        full[i] += mod.ppll(X_/X_.std()).mean()
        # ens.append(en)
        sigs.append(mod.sigma_x)

plt.plot(kays, trn)
plt.plot(kays, tst, '--')
plt.plot(kays, full)


#%%

# smoke test: recover a planted sign Ising model with beta scheduled, not fit
# rng = np.random.default_rng(0)
# n, M = 15, 4000
# beta_true = 0.6

# J_true = np.zeros((n, n))
# for _ in range(18):
#     i, j = rng.choice(n, size=2, replace=False)
#     J_true[i, j] = J_true[j, i] = rng.choice([-1.0, 1.0])
# h_true = 0.3 * rng.standard_normal(n)

# S = rng.choice([-1.0, 1.0], size=(M, n))
# for _ in range(80):
#     for i in range(n):
#         f = S @ J_true[i] - J_true[i, i] * S[:, i] + h_true[i]
#         p = 1.0 / (1.0 + np.exp(-4.0 * beta_true * f))
#         S[:, i] = np.where(rng.random(M) < p, 1.0, -1.0)

# off = ~np.eye(n, dtype=bool)
# nz = J_true[off] != 0

# def report(tag, J_hat, h_hat):
#     tp = int(((J_hat[off] != 0) & nz).sum())
#     fp = int(((J_hat[off] != 0) & ~nz).sum())
#     sa = (J_hat[off][nz] == J_true[off][nz]).mean()
#     hc = np.corrcoef(h_hat, h_true)[0, 1]
#     print(f"{tag}: TP={tp}/{int(nz.sum())} FP={fp} sign-acc={sa:.2f} "
#           f"corr(h, h_true)={hc:.2f}")

# # beta ramped up to the true scale (in practice: to discretize_rple's beta0)
# bs = np.linspace(0.2, beta_true, 150)
# # J_hat, h_hat, hist = mrf.fit_sign_ising(S, n_sweeps=150, beta_schedule=bs,
#                                     # seed=1, verbose=True)
# prior = nbp.MRFPrior(tree_reg=0, J_lam=None)


# prior = nbp.MRFPrior()

# prior.init_params(S)

# Jnrm = la.norm(task.J)

# ls = []
# for _ in range(nepoch):
    
#     Jmod = prior.J_W + prior.J_W.T
#     ls.append(np.sum(task.J*Jmod)/(Jnrm*la.norm(Jmod)))
#     prior.learn(task.sample(N, temp))

# plt.plot(ls)


# report("cold start, beta ramp", J_hat, h_hat)

