CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/dandisets/' 

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

import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
# import cvxpy as cvx

import h5py

from dandi.download import download as dandi_download
from pynwb import NWBHDF5IO
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

from scipy.signal import resample_poly
from math import gcd

def zscore(x, center=True):
    # return (x - center*x.mean(0))/np.sqrt(np.mean((x-x.mean(0))**2, axis=0))
    return (x - center*x.mean(0)) / x.std(0)

def common_average_reference(data):
    """data: (n_samples, n_channels)"""
    return data - data.mean(axis=1, keepdims=True)

def get_gaussian_filter_params(n_bands=8, cf_min=85, cf_max=175, Q=16):
    """
    Logarithmically spaced center frequencies, semi-logarithmically increasing bandwidths.
    """
    center_freqs = np.logspace(np.log10(cf_min), np.log10(cf_max), n_bands)
    # Semi-log bandwidths: linearly scale with log of center frequency
    log_cfs = np.log10(center_freqs)
    log_min, log_max = log_cfs[0], log_cfs[-1]
    # Normalize to get bandwidths that scale semi-logarithmically
    bw_min, bw_max = cf_min / (Q+1), cf_max / (Q-1)  # approximate scaling
    bandwidths = bw_min + (bw_max - bw_min) * (log_cfs - log_min) / (log_max - log_min)
    # bandwidths = center_freqs / Q
    
    return center_freqs, bandwidths

def extract_high_gamma(data, sfreq, target_sfreq=200, n_bands=8, cf_min=85, cf_max=175, Q=16):
    """
    Full pipeline: CAR -> bandpass -> Hilbert -> analytic amplitude -> average -> downsample.
    
    Parameters
    ----------
    data        : (n_samples, n_channels)
    sfreq       : original sampling frequency (Hz)
    target_sfreq: output sampling frequency (Hz)
    
    Returns
    -------
    hg_power    : (n_samples_resampled, n_channels)
    """
    data_car = common_average_reference(data)
    
    # downsample
    print('Downsampling')
    up = 2*target_sfreq
    down = sfreq
    common = gcd(int(up), int(down))
    data_ds = resample_poly(data_car, int(up) // common, int(down) // common, axis=0)    
    
    # 2. Get filter parameters
    center_freqs, bandwidths = get_gaussian_filter_params(n_bands, cf_min, cf_max, Q)
    
    # 3. Extract analytic amplitude from each band and average
    amplitudes = np.zeros_like(data_ds)
    
    print('FFT')
    n_samples = data_ds.shape[0]
    freqs = np.fft.fftfreq(n_samples, d=1.0/up)
    data_fft = np.fft.fft(data_ds, axis=0)
    mask = (1*(freqs >= 0)[:,None] + 1*(freqs > 0)[:,None])
    
    print('bp + Hilbert')
    for cf, bw in tqdm(zip(center_freqs, bandwidths)):

        # Gaussian kernel in frequency domain (one-sided, applied to positive freqs)
        gaussian = np.exp(-0.5 * ((freqs - cf) / bw) ** 2)
        
        # FFT, filter, IFFT
        filtered_fft = data_fft * gaussian[:, None] * mask

        # Analytic amplitude via Hilbert transform
        analytic = np.fft.ifft(filtered_fft, axis=0)
        amplitudes += np.abs(analytic)
    
    hg_power = amplitudes / n_bands  # average across bands
    
    # 4. Downsample to target_sfreq
    up = target_sfreq
    down = 2*target_sfreq
    common = gcd(int(up), int(down))
    hg_power_ds = resample_poly(hg_power, int(up) // common, int(down) // common, axis=0)    
    
    return hg_power_ds, np.linspace(0, len(hg_power_ds)/up, len(hg_power_ds))

#%%

## Consonant conversion table
cons = {'b': 'b',
        'd': 'd',
        'f': 'f',
        'g': 'g',
        'h': 'h',
        'k': 'k',
        'l': 'l',
        'm': 'm',
        'n': 'n',
        'p': 'p',
        'r': 'r',
        's': 's',
        'sh': '\u017f',
        't': 't',
        'th': '\u03B8',
        'v': 'v',
        'w': 'w',
        'y': 'j',
        'z': 'z'}

articulators = {'labial': ['b', 'p', 'm', 'w', 'f', 'v'],
                'dorsal': ['g', 'k', 'j', 'h', 'r'],
                'coronal': ['t','d', 'l', 'n', 's', 'z', '\u03B8', '\u017f'],
                }

def baby2ipa(chars):
    
    ipas = np.chararray(len(chars), unicode=True, itemsize=2)
    
    ## Go vowel by vowel
    for i, s in enumerate(chars.split('a')):
        if len(s) > 1:
            ipas[i] = cons[s[0]] + 'a' 
            
    for i, s in enumerate(chars.split('e')):
        if len(s) > 1:
            ipas[i] = cons[s[0]] + 'i' 
            
    for i, s in enumerate(chars.split('o')):
        if len(s) > 1:
            ipas[i] = cons[s[0]] + 'u' 
    
    return ipas 


#%%

dset = '000019' 

t0 = -0.05
t1 = 0.25

ds_freq = 200

id0 = int(t0*ds_freq)
id1 = int(t1*ds_freq)

T = np.abs(id0) + np.abs(id1)

subj = os.listdir(f"{LOAD_DIR}/{dset}")[1]

neurs = {}
locs = []
eids = []
isbad = []
these_sess = np.unique([re.findall('ses-([^_.]+)', sess)[0] for sess in os.listdir(f"{LOAD_DIR}/{dset}/{subj}/")])

for j,sess in enumerate(these_sess):
    
    pathtoNWBFile = f"{LOAD_DIR}/{dset}/{subj}/{subj}_ses-{sess}.nwb"
    
    nwbBasePath = Path(pathtoNWBFile)

    io = NWBHDF5IO(str(nwbBasePath), mode='r')
    nwb = io.read()
    
    sfreq = np.round(nwb.acquisition['ElectricalSeries'].rate)
    times = np.array(nwb.intervals['trials'].cv_transition_time.data)
    conds = np.array(nwb.intervals['trials'].condition.data)
    
    bad = np.array(nwb.electrodes['bad'].data)
    print('%d bad sites'%(np.sum(bad)))
    
    loc = np.array(nwb.electrodes['location'].data)
    eid = np.array(nwb.electrodes['id'].data)
    
    ecog = np.array(nwb.acquisition['ElectricalSeries'].data)
    hg_power_ds, t = extract_high_gamma(ecog, sfreq, target_sfreq=ds_freq, Q=32)
    
    # Z = (hg_power_ds-hg_power_ds.mean(0))/np.sqrt(((hg_power_ds-hg_power_ds.mean(0))**2).mean(0))
    # Z = (hg_power_ds)/np.sqrt(((hg_power_ds-hg_power_ds.mean(0))**2).mean(0))
    Z = zscore(hg_power_ds, center=False)
    
    idx = np.abs(t[None] - times[:,None]).argmin(1)
    
    Xcntr = np.zeros((len(conds), T, hg_power_ds.shape[1]))
    for i,dt in enumerate(np.arange(id0, id1)):
        Xcntr[:,i] = Z[idx+dt]
        
    chars, ids = np.unique(conds, return_inverse=True)
    labels = baby2ipa(np.char.array(chars, unicode=True))
        
    neurs[sess] = {}
    for lab in labels:
        neurs[sess][lab] = Xcntr[labels[ids]==lab]
    
    locs.append(loc)
    eids.append(eid)
    isbad.append(bad)

locs = np.array(locs)
eids = np.array(eids)
isbad = np.array(isbad)

deez = ~isbad.max(0)

locs = locs[0][deez]
eids = eids[0][deez]

#%%

# frac = 0.8 
frac = 1.0

ntrls = []
nneur = []
for this_sess in neurs.values():
    ntrls.append( [len(foo) for foo in this_sess.values()] )
    nneur.append( [len(foo.T) for foo in this_sess.values()] )

ntrls = np.array(ntrls[:-1])
nneur = np.array(nneur[:-1])

# ## how many trials to sample from each condition?
# # ntrnsamp = np.round(frac*ntrls.mean(0)).astype(int)
# # ntstsamp = np.round((1-frac)*ntrls.mean(0)).astype(int)
# ntrnsamp = int(np.round(frac*ntrls.mean())) * np.ones(ntrls.shape[1], dtype=int)
# ntstsamp = int(np.round((1-frac)*ntrls.mean())) * np.ones(ntrls.shape[1], dtype=int)
# # n_samp = ntrls.min(0)

# ntrn = np.floor(frac*ntrls).astype(int)

# pp_trn = {}
# pp_tst = {}
# for i, this_sess in enumerate(neurs.values())[:-1]:
#     for j, (lab, rep) in enumerate(this_sess.items()):
        
#         trnset = np.random.choice(range(ntrls[i,j]), ntrn[i,j], replace=False)
#         tstset = np.setdiff1d(range(ntrls[i,j]), trnset)
        
#         ix_trn = np.random.choice(trnset, ntrnsamp[j])
#         X_trn = rep[ix_trn]
    
#         ix_tst = np.random.choice(tstset, ntstsamp[j])
#         X_tst = rep[ix_tst]
        
#         if lab in pp_trn.keys():
#             pp_trn[lab] = np.hstack([pp_trn[lab], X_trn])
#             pp_tst[lab] = np.hstack([pp_tst[lab], X_tst])
#         else:
#             pp_trn[lab] = X_trn
#             pp_tst[lab] = X_tst

#%%

## Reduced rank regression weights
def rr_regression(X, S, r, lam_l1=1e-2, lam_pr=1e-2, lam_l2=1e-2, max_iter=100, nonneg=False):
    """
    Reduced rank regression with shared latent dynamics
    """
    
    c, t, n = X.shape
    c2, k = S.shape
    
    assert c == c2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    U_init = np.random.randn(t,r) / np.sqrt(t*r)
    V_init = np.random.randn(k,n,r) / np.sqrt(n*r)
    b_init = X.mean(0,keepdims=True).swapaxes(1,2)
    
    if nonneg:
        U_init[U_init<0] = 0
        V_init[V_init<0] = 0 
        b_init[b_init<0] = 0
        
    V = nn.Parameter(torch.FloatTensor(V_init).to(device))
    U = nn.Parameter(torch.FloatTensor(U_init).to(device)) 
    b = nn.Parameter(torch.FloatTensor(b_init).to(device))
    
    Xpt = torch.FloatTensor(X).to(device)
    Spt = torch.FloatTensor(np.hstack([S, np.ones((c,1))])).to(device)
    
    # optimizer = optim.LBFGS([U, V, b], lr=1., max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None)
    optimizer = optim.Adam([U, V, b], lr=0.1)
    
    for _ in tqdm(range(max_iter)):
    # def closure():
        optimizer.zero_grad()
        
        beta = torch.cat([V@U.T, b], 0)
        Xhat = torch.einsum('ck,knt->ctn', Spt, beta)
        loss = torch.sum((Xpt - Xhat)**2)
        
        VtV = V.swapaxes(1,2)@V
        trace = torch.einsum('kii->k', VtV)
        norm = torch.sum(VtV**2, axis=(1,2))
        loss -= lam_pr*torch.sum((trace**2)/norm) / r
        
        loss += lam_l1*torch.sum(torch.abs(V))
        loss += lam_l2*torch.sum(beta**2)
        
        loss.backward()
        
        # closure_calls[0] += 1
        # print(f"step: {closure_calls[0]} total_loss: {loss.item()}")
        # return loss
        
        optimizer.step()
        
        with torch.no_grad():
            if nonneg:
                U[U<0] = 0
                V[V<0] = 0 
                b[b<0] = 0
        
    # closure_calls = [0]
    # optimizer.step(closure)
    
    return U.detach().cpu().numpy(), V.detach().cpu().numpy(), b.detach().cpu().numpy()

## Vanilla SCA and SCA w/ masking

def multilinear_regression(X, S, lam_l1=1e-2, lam_pr=1e-2, 
                           autoencode=False, max_iter=100,
                           fit_intercept=True):

    c, t, n = X.shape
    c2, k = S.shape
    m = min(n, k)
    
    assert c == c2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if k < min(t,n):
        U_init,s,V_init = la.svd(X.mean(0) - X.mean((0,1)), full_matrices=False)
    else:
        U_init = np.random.randn(t,k)
        V_init = np.random.randn(k,n)
    
    V = nn.Parameter(torch.FloatTensor(V_init[:k].T).to(device)) # shape n x k 
    if autoencode:
        U = nn.Parameter(torch.FloatTensor(V_init[:k].T).to(device)) # shape n x k
    else:
        U = nn.Parameter(torch.FloatTensor(U_init[:,:k]).to(device)) # shape t x k
    
    Xpt = torch.FloatTensor(X).to(device)
    Spt = torch.FloatTensor(S).to(device)
    
    optimizer = optim.Adam([U, V], lr=0.1)
    
    ls = []
    for _ in tqdm(range(max_iter)):
        
        optimizer.zero_grad()
        
        if autoencode:
            Utilde = torch.einsum('nk,ctn->ctk', U, Xpt)
            Xhat = torch.einsum('ck,ctk,nk->ctn', Spt, Utilde, V)
        else:
            Xhat = torch.einsum('ck,tk,nk->ctn', Spt, U, V)
        loss = torch.mean((Xpt - Xhat)**2)    
        
        ls.append(loss.item())
        
        VtV = V.T@V
        loss -= lam_pr*((torch.trace(VtV)**2)/torch.sum(VtV**2))/m
        
        if autoencode:
            loss += lam_l1*torch.mean(torch.abs(Utilde))
        else:
            loss += lam_l1*torch.mean(torch.abs(U))
        
        loss.backward()
        
        optimizer.step()
    
    return U.detach().cpu().numpy(), V.detach().cpu().numpy(), ls

        
#%%

pop = {}

for this_sess in neurs.values():
    if len(this_sess.values()) < 57:
        continue
    for lab, rep in this_sess.items():
        if lab in pop.keys():
            pop[lab] = np.vstack([pop[lab], rep])
        else:
            pop[lab] = rep
            
X = np.array([v.mean(0) for v in pop.values()])
labels = np.char.array([k for k in pop.keys()])
cv_idx = np.concatenate([np.ones(len(v), dtype=int)*i for i,v in enumerate(pop.values())])

X = X[...,deez]

consonant = labels.strip('aui')
vowel = labels.strip(consonant)

vowel_feats = np.stack([vowel==v for v in ['a','i','u']])*1
consonant_feats = np.array([np.isin(consonant, v) for v in articulators.values()])*1

S = np.vstack([vowel_feats, consonant_feats]).T

#%%

X_ = X
# X_ = X / X.std((0,1))
# X_ = Xmd / Xmd.std(0)

mod = bae_models.RRBMF(16,
                       2,
                       nonneg=True, 
                       sparse_reg=1e-2,
                       pr_reg=1e-2,
                       l1_reg=0,
                       tree_reg=5,
                       l2_reg=1e-1,
                       )

en = mod.fit(torch.tensor(X_ / X_.std()),
             initial_temp=100, 
             decay_rate=0.9, 
             min_temp=1, 
             period=50, 
             lr=1e-1,
             # scl_lr=1e-3,
             # opt_alg=optim.Adam,
             )

samps = mod.sample(torch.tensor(X_ / X_.std()), n_samp=1000)

plt.imshow(samps.mean(0)) 


#%%

# X_ = (X[:,0] - X[:,0].mean(0))

agg = AgglomerativeClustering(n_clusters=None,
                              distance_threshold=0,
                              compute_full_tree=True)

agg.fit(X_)

S = np.eye(len(X_))
for pair in agg.children_:
    S = np.hstack([S, S[:,pair].sum(1, keepdims=True)])


#%%

rrr_data = {'1': {'Xall': Xrrr, 
                  'yall': X, 
                  'setup': {'mean_y_TN': X.mean(0), 
                            'std_y_TN': np.sqrt(((X - X.mean(0))**2).mean(0)),
                            'mean_X_Tv': Xrrr.mean(0),
                            'std_X_Tv': np.sqrt(((Xrrr - Xrrr.mean(0))**2).mean(0)),
                            } ,
                  } 
            }

RRR_p = dict(n_comp_list=[1,2,3,6], l2_list=[10, 25,], lr=1.0)
CV_p = dict(nsplit=3, test_size=0.3, stratify_by=[None])

#%%

n_chain = 20
# k = 50
k = 6

args = {
        'l1_reg': 1e-3,
        'pr_reg': 1e-4,
        'tree_reg': 10,
        'm_iters': 5,
        'nmf_init': True,
        }

opt_args = {'initial_temp': 100,
            'decay_rate': 0.95,
            'period': 10,
            'lr': 0.02,
            }

mods = [bae_models.SCPD(k, **args) for i in range(n_chain)]
ens = [m.fit(torch.tensor(X), **opt_args) for m in mods]

# allW = np.hstack([m.W for m in mods])
allS = np.hstack([m.S for m in mods])

#%%

K = k*2

kmn = KMeans(K)
# kmn.fit(allW.T)
kmn.fit(allS.T)

avg = util.group_mean(allS, kmn.labels_)
quality = np.argsort(np.log2(avg+1e-6).mean(0))

plt.imshow(avg[:,quality], cmap='binary')

#%%

# kays = [3,6,9,12,15]
kays = [10,20,30,40,50]
# kays = [10]
ars = [2,3,4,5]

# args = {'nonneg':True,
#         'time_sparsity': 1,
#         'feature_sparsity': 2,
#         'pr_reg': 1e-3,
#         'gp_width': 0.05,
#         # 'fit_intercept': False,
#         }
args = {'nonneg':True,
        'sparse_reg': 0,
        'tree_reg': 1e-1,
        'pr_reg': 1e-3,
        # 'fit_intercept': False,
        }

opt_args = {'initial_temp': 10,
            'decay_rate': 0.9,
            'period': 10,
            'lr': 1e-1,
            'folds': 5,
            }
# opt_args = {'initial_temp': 1,
#             'decay_rate': 1,
#             'max_iter': 500,
#             # 'period': 10,
#             'lr': 1e-2,
#             }

n_run = 1

trn = np.zeros((len(kays), len(ars)))
tst = np.zeros((len(kays), len(ars)))
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
        for j,r in enumerate(ars):
            
            # mod = bae_models.BiPCA(k, sparse_reg=1e-4)
            # mod = bae_models.ConvBMF(k,r,**args)
            # mod = bae_models.ConvNMF(k,l,**args)
            mod = bae_models.RRBMF(k,r,**args)
            
            # wa,ba = bae_util.impcv(mod, X, verbose=True, **opt_args)
            wa,ba = bae_util.impcv(mod, torch.tensor(X), verbose=False, **opt_args)
            
            trn[i,j] += wa.numpy()
            tst[i,j] += ba.numpy()

plt.plot(kays, trn)
plt.plot(kays, tst, '--')
# plt.plot(kays, np.mean(trn,axis=0))
# plt.plot(kays, np.mean(tst, axis=0))