CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/tian_data/' 

import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/sca/')
sys.path.append(LOAD_DIR + 'code/')

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
from sklearn.model_selection import cross_val_score
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
import old_bae
import old_bae_models
import old_bae_search
import bae_util
import plotting as tpl

import bae_models 

#%%

import warnings
from scipy.interpolate import interp1d

def get_consensus_signal(data, times, new_times):
    """
    Interpolates and averages K signals to find a consensus signal,
    and calculates the standard deviation at each point as a mismatch metric.
    
    Parameters:
    - data: np.ndarray of shape (N, K, T)
    - times: np.ndarray of shape (K, T) containing the time labels for each K.
    - new_times: np.ndarray of shape (newT,) with the target time labels.
    
    Returns:
    - consensus: np.ndarray of shape (N, newT) (The mean signal)
    - mismatch: np.ndarray of shape (N, newT) (The standard deviation)
    - counts: np.ndarray of shape (N, newT) (Number of overlapping signals)
    """
    N, K, T = data.shape
    newT = len(new_times)
    
    # Initialize a temporary array with NaNs to hold our interpolated data
    interp_data = np.full((N, K, newT), np.nan)
    
    for k in range(K):
        interpolator = interp1d(
            times[k], 
            data[:, k, :], 
            axis=-1, 
            bounds_error=False, 
            fill_value=np.nan    
        )
        interp_data[:, k, :] = interpolator(new_times)
        
    # Calculate counts before the warning block (np.isnan doesn't throw warnings)
    counts = np.sum(~np.isnan(interp_data), axis=1)
        
    # Average and standard deviation across the K dimension (axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        consensus = np.nanmean(interp_data, axis=1)
        
        # ddof=0 is default (population std). If you prefer sample std, set ddof=1
        mismatch = np.nanstd(interp_data, axis=1)
        
    # Optional: If you want standard deviation to be NaN where count is 1 
    # (since you can't have a mismatch with only 1 signal), uncomment this:
    # mismatch[counts <= 1] = np.nan
        
    return consensus, mismatch, counts


#%%

animal = 'Diego'
date = 231122
region = 'PMv'

nogrid = True

dat = pkl.load(open(LOAD_DIR + f"fig6/DFallpa-{animal}-{date}.pkl",'rb'))

pa = dat['pa'][np.where(dat['bregion'] == region)[0][0]]
trls = pa.Xlabels['trials']

allX = []
which_area = []
for i in range(len(dat)):
    allX.append(dat['pa'][i].X)
    which_area.append(np.repeat(dat['bregion'][i], len(dat['pa'][i].X)))
allX = np.concatenate(allX, axis=0)
which_area = np.concatenate(which_area)

if nogrid:
    deez = trls['task_kind'] != 'prims_on_grid'
    # deez = trls['task_kind'] == 'character'
    trls = trls[deez]
    allX = allX[:,deez]

#%%

region = 'PMv'

P = allX[which_area=='PMv'][:,trls.task_kind == 'prims_single']

#%%

## Figure out which strokes were part of the same trial
chars, cidx = np.unique(trls['character'], return_inverse=True)
trial_idx = np.cumsum(np.diff(cidx, prepend=-1) != 0)

## Stitch together activity from the same trial
dt = 0.01   # pretty sure all the time bins are 10 ms 

X_trl = []
trl_times = []
stroke_times = []
stroke_idx = []
targ_strokes = []
real_strokes = []
errors = []
for this_trl in np.unique(trial_idx):
    
    stroks = trls['event_time'][trial_idx==this_trl].to_numpy()
    
    ## get times across the whole character
    t = pa.Times[None] + stroks[:,None]
    trlX = allX[:,trial_idx==this_trl]
    
    ## stitch
    newt = np.arange(t.min(), t.max(), dt)
    
    newX, err, cnts = get_consensus_signal(trlX, t, newt)
    normalized_err = np.nanmean(err[cnts>1] / newX[cnts>1])
    
    strok_names = np.unique(trls['charclust_shape_seq'][trial_idx == this_trl])
    assert (len(strok_names) == 1)
    
    true_stroks = trls['shape'][trial_idx==this_trl].to_numpy()
    ix = np.argsort(stroks)
    
    if np.diff(stroks[ix], prepend=stroks[ix][0]).max() > (pa.Times[-1] - pa.Times[0]):
        continue

    X_trl.append(newX)
    trl_times.append(newt - np.min(stroks))
    stroke_times.append(stroks)
    targ_strokes.append(strok_names[0])
    real_strokes.append(true_stroks)
    stroke_idx.append(ix)
    errors.append(normalized_err)

num_stroke = np.array([len(s) for s in stroke_times])

#%%

all_strokes = np.unique(np.concatenate(real_strokes))
strok2num = {s:i for i,s in enumerate(all_strokes)}

K = len(all_strokes)

labs = [np.eye(K)[[strok2num[s] for s in sks]].max(0) for sks in real_strokes]
labs = np.array(labs)

y = np.array([strok2num[s] for s in trls['shape'].to_numpy()])
curr_lab = np.array([np.eye(K)[i] for i in y])

first_lab = np.array([strok2num[s[i][0]] for i,s in zip(stroke_idx, real_strokes)])
last_lab = np.array([strok2num[s[i][-1]] for i,s in zip(stroke_idx, real_strokes)])

#%% per-time decoding
# region = 'preSMA'
# region = 'PMv'
region = 'SMA'

deez = num_stroke > 1

# this_y = first_lab[trial_idx-1]
this_y = curr_lab

perf = np.zeros((len(pa.Times), this_y.shape[-1]))
for t in tqdm(range(len(pa.Times))):
    
    for k in range(this_y.shape[-1]):
        clf = svm.LinearSVC(class_weight='balanced')
        
        if this_y.sum(0)[k] < 6:
            continue
        # perf[t,k] = np.mean(cross_val_score(clf, X[...,t].T, labs[trial_idx-1,k]))
        # perf[t,k] = np.mean(cross_val_score(clf, X[...,t].T, curr_lab[:,k]))
        perf[t,k] = np.mean(cross_val_score(clf, allX[which_area==region][...,t].T, this_y[:,k]))

plt.plot(pa.Times, perf[:,this_y.sum(0)>6].mean(1))

#%%

region = 'PMv'

perf = []
for k in np.where(deez)[0]:
    
    perf.append(np.mean(cross_val_score(clf, Xpre[:,which_area==region], labs[:,k])))


#%%


animal = 'Diego'

neurons = []


for fil in os.listdir(LOAD_DIR + 'fig6'):
    
    if f"DFallpa-{animal}" not in fil:
        continue

    dat = pkl.load(open(LOAD_DIR + f"fig6/{fil}",'rb'))
    
    pa = dat['pa'][np.where(dat['bregion'] == region)[0][0]]
    trls = pa.Xlabels['trials']
    
    allX = []
    which_area = []
    for i in range(len(dat)):
        allX.append(dat['pa'][i].X)
        which_area.append(np.repeat(dat['bregion'][i], len(dat['pa'][i].X)))
    allX = np.concatenate(allX, axis=0)
    which_area = np.concatenate(which_area)
    
    if nogrid:
        # deez = trls['task_kind'] != 'prims_on_grid'
        deez = trls['task_kind'] == 'character'
        trls = trls[deez]
        allX = allX[:,deez]

    ## Figure out which strokes were part of the same trial
    chars, cidx = np.unique(trls['character'], return_inverse=True)
    trial_idx = np.cumsum(np.diff(cidx, prepend=-1) != 0)
    
    # region = 'dlPFC'
    # region = 'PMv'
    region = 'preSMA'
    
    ## Stitch together activity from the same trial
    dt = 0.01   # pretty sure all the time bins are 10 ms 
    
    X_trl = []
    stroke_times = []
    stroke_idx = []
    targ_strokes = []
    real_strokes = []
    errors = []
    for this_trl in np.unique(trial_idx):
            
        stroks = trls['event_time'][trial_idx==this_trl].to_numpy()
        
        ## get times across the whole character
        t = pa.Times[None] + stroks[:,None]
        trlX = allX[:,trial_idx==this_trl]
        
        ## stitch
        newt = np.arange(t.min(), t.max(), dt)
        
        newX, err, cnts = get_consensus_signal(trlX, t, newt)
        normalized_err = np.nanmean(err[cnts>1] / newX[cnts>1])
        
        strok_names = np.unique(trls['charclust_shape_seq'][trial_idx == this_trl])
        assert (len(strok_names) == 1)
        
        true_stroks = trls['shape'][trial_idx==this_trl].to_numpy()
        ix = np.argsort(stroks)
        
        if np.diff(stroks[ix], prepend=stroks[ix][0]).max() > (pa.Times[-1] - pa.Times[0]):
            continue
    
        X_trl.append(newX)
        stroke_times.append(stroks)
        targ_strokes.append(strok_names[0])
        real_strokes.append(true_stroks)
        stroke_idx.append(ix)
        errors.append(normalized_err)

    num_stroke = np.array([len(s) for s in stroke_times])

#%% ============================================================
#  Character-vs-primitive neural template matching
#  ------------------------------------------------------------
#  For each character trial, take the stitched-together neural activity
#  (get_consensus_signal, exactly as above) and compare it to each
#  prims_single ("prim") trial's stroke-aligned activity by:
#     conv over time (per neuron) -> sum over neurons -> max over time.
#  Done separately per brain region. Result per (animal, date, region) is
#  an (n_character x n_primitive) matrix.
#
#  Only the 4 dates that have BOTH neural (fig6) and drawing (fig2kp) data.
#  This computation itself needs only the firing rates, but we restrict to
#  these dates so each char/prim row can later be tied to its drawing via
#  trialcode (+ stroke_index for the multi-stroke characters).
# ============================================================

from scipy.signal import fftconvolve

animal = 'Diego'
dates_with_drawings = [231122, 231128, 231129, 231201]
dt = 0.01            # all time bins are 10 ms
SIM_MODE = 'conv'    # 'conv' = convolution (as requested); 'xcorr' = cross-correlation
ZSCORE = True        # z-score each neuron's timeseries (over time) before matching


def load_allpa(animal, date):
    """Load one session's DFallpa and stack neurons across all brain regions.
    Returns (allX, which_area, trls, Times) where allX is (n_neuron, n_trial, n_time),
    which_area labels each neuron's region, trls is the shared per-stroke label table."""
    dat = pkl.load(open(LOAD_DIR + f"fig6/DFallpa-{animal}-{date}.pkl", 'rb'))
    pa0 = dat['pa'][0]
    trls = pa0.Xlabels['trials']
    allX, which_area = [], []
    for i in range(len(dat)):
        allX.append(dat['pa'][i].X)
        which_area.append(np.repeat(dat['bregion'][i], len(dat['pa'][i].X)))
    allX = np.concatenate(allX, axis=0)
    which_area = np.concatenate(which_area)
    return allX, which_area, trls, pa0.Times


def stitch_characters(allX_char, trls_char, Times, dt=0.01):
    """Stitch the strokes of each character trial into one continuous timeseries,
    exactly as in the cell above. Returns a list of (n_neuron, T_char) arrays, an
    aligned list of (T_char,) time vectors (the neural time-in-trial, on the same
    continuous trial clock as the drawings' third column), plus aligned label lists
    (one entry per character trial)."""
    _, cidx = np.unique(trls_char['character'], return_inverse=True)
    trial_idx = np.cumsum(np.diff(cidx, prepend=-1) != 0)
    span = Times[-1] - Times[0]

    X_trl, X_t, char_names, char_targ, char_tc = [], [], [], [], []
    for this_trl in np.unique(trial_idx):
        sel = trial_idx == this_trl
        stroks = trls_char['event_time'][sel].to_numpy()

        t = Times[None] + stroks[:, None]          # absolute time of each stroke's window
        trlX = allX_char[:, sel]
        newt = np.arange(t.min(), t.max(), dt)     # neural time-in-trial (trial clock)
        newX, _, _ = get_consensus_signal(trlX, t, newt)

        ix = np.argsort(stroks)
        if np.diff(stroks[ix], prepend=stroks[ix][0]).max() > span:
            continue                                # gap too big -> dropped strokes, skip

        strok_names = np.unique(trls_char['charclust_shape_seq'][sel])
        X_trl.append(newX)
        X_t.append(newt)
        char_targ.append(strok_names[0])
        char_names.append(trls_char['character'][sel].to_numpy()[0])
        char_tc.append(trls_char['trialcode'][sel].to_numpy()[0])
    return X_trl, X_t, char_names, char_targ, char_tc


def _zscore_time(A):
    """z-score along the last (time) axis, per neuron, ignoring NaNs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(A, axis=-1, keepdims=True)
        sd = np.nanstd(A, axis=-1, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)                     # avoid /0 for flat neurons
    return (A - mu) / sd


def conv_match_matrix(C_list, P, mask, mode='conv', zscore=True):
    """C_list: list of (n_neuron, T_c) char timeseries; P: (n_neuron, n_prim, T_p);
    mask: bool over neurons selecting one region.
    Returns (n_char, n_prim): sum_over_neurons( conv(C[n], P[n]) ), max over time.
    If zscore, each neuron's timeseries is z-scored over time first (so the match
    reflects waveform shape, not overall firing rate)."""
    Pr = P[mask].transpose(1, 0, 2)                       # (n_prim, Nr, T_p)
    if zscore:
        Pr = _zscore_time(Pr)
    Pr = np.nan_to_num(Pr)
    if mode == 'xcorr':
        Pr = Pr[:, :, ::-1]                               # flip time -> cross-correlation
    n_prim = Pr.shape[0]
    M = np.full((len(C_list), n_prim), np.nan)
    for i, C in enumerate(C_list):
        Cr = C[mask]                                      # (Nr, T_c)
        if zscore:
            Cr = _zscore_time(Cr)
        Cr = np.nan_to_num(Cr)
        A = np.broadcast_to(Cr, (n_prim,) + Cr.shape)     # (n_prim, Nr, T_c)
        conv = fftconvolve(A, Pr, mode='full', axes=-1)   # (n_prim, Nr, T_c+T_p-1)
        M[i] = conv.sum(1).max(-1)                         # sum neurons, max over time
    return M

def load_drawings(animal, date):
    """trialcode -> list of stroke arrays (each (Npts,3) = [x,y,time]), from the DS."""
    DS = pkl.load(open(LOAD_DIR + f"fig2kp/DS_before_cluster/{animal}-{date}.pkl", 'rb'))
    by_tc = {tc: list(g['strok']) for tc, g in DS.Dat.groupby('trialcode')}
    return by_tc


def stitch_drawing(strokes):
    """Concatenate strokes in temporal order -> (sum_Npts, 3) [x,y,time].
    Pen-up gaps between strokes remain as jumps / time-gaps (not interpolated)."""
    if not strokes:
        return None
    strokes = sorted(strokes, key=lambda s: s[0, 2])   # order by first timestamp
    return np.concatenate(strokes, axis=0)


savepath = SAVE_DIR + f"tian_char_prim_convmatch-{animal}-{SIM_MODE}-z={ZSCORE}.pkl"

try: 
    results = pkl.load(open(savepath,'rb'))
except:
    
    results = {}
    for date in dates_with_drawings:
        print(f"=== {animal}-{date} ===", flush=True)
        allX, which_area, trls, Times = load_allpa(animal, date)
    
        # --- primitive (prims_single) trials: one stroke each, stroke-aligned over Times ---
        prim_mask = (trls['task_kind'] == 'prims_single').to_numpy()
        P = allX[:, prim_mask]                                # (n_neuron, n_prim, T_p)
        prim_labels = trls['shape'][prim_mask].to_numpy()
        prim_tc = trls['trialcode'][prim_mask].to_numpy()
        # neural time-in-trial for each prim, on the same continuous trial clock as the
        # drawings' third column: stroke-aligned Times shifted by the stroke's event_time
        prim_evt = trls['event_time'][prim_mask].to_numpy()
        prim_t = Times[None] + prim_evt[:, None]             # (n_prim, T_p)
    
        # --- character trials: stitch strokes into one continuous timeseries ---
        char_mask = (trls['task_kind'] == 'character').to_numpy()
        X_trl, X_t, char_names, char_targ, char_tc = stitch_characters(
            allX[:, char_mask], trls[char_mask], Times, dt=dt)
    
        print(f"  n_character_trials={len(X_trl)}  n_primitive_trials={P.shape[1]}")
    
        regions = list(dict.fromkeys(which_area))            # preserve region order
        M_by_region = {}
        for region in regions:
            M_by_region[region] = conv_match_matrix(
                X_trl, P, which_area == region, mode=SIM_MODE, zscore=ZSCORE)
            print(f"    {region}: matrix {M_by_region[region].shape}")
    
        by_tc = load_drawings(animal, date)
    
        prim_draw = np.array(
            [stitch_drawing(by_tc.get(tc)) for tc in prim_tc], dtype=object)
        char_draw = np.array(
            [stitch_drawing(by_tc.get(tc)) for tc in char_tc], dtype=object)
    
        results[(animal, date)] = dict(
            char_X=X_trl, prim_X=P,
            char_t=X_t, prim_t=prim_t,                      # neural time-in-trial (trial clock)
            which_area=which_area,
            M=M_by_region, regions=regions,
            char_names=np.array(char_names),
            char_targ=np.array(char_targ, dtype=object),   # tuples of varying length
            char_draw=char_draw,
            char_trialcode=np.array(char_tc),
            prim_labels=prim_labels, prim_trialcode=prim_tc,
            prim_draw=prim_draw,
        )
        del allX, P, X_trl
    
    # Save
    with open(savepath, 'wb') as f:
        pkl.dump(results, f)
    print("saved ->", savepath)

#%%

Pzs = []
labs = []
for date in dates_with_drawings:
    
    unqs, idx = np.unique(results[(animal, date)]['prim_labels'].astype(str), return_inverse=True)
    P = util.group_mean(results[(animal, date)]['prim_X'], idx, axis=1)
    Pzs.append(P[results[(animal, date)]['which_area'] == 'PMv'])
    labs.append(unqs)
    
P = np.concatenate(Pzs, axis=0)

#%%

# X_ = P.transpose((1,2,0))
# X_ = out['PFC']['X'].transpose((0,2,1))[:12]

X_ = P[...,(times > -0.15)*(times < 0)].mean(-1).T

X_ /= X_.std((0,1), keepdims=True)


# mod = old_bae_models.SemiBMF(16,
#                          nonneg=True, 
#                          sparse_reg=1e-2,
#                          weight_pr_reg=1, 
#                          tree_reg=1,
#                          weight_l2_reg=1,
#                          )

mod = bae_models.JBMF(3,
                         nonneg=True,
                         # nonneg=False,
                         # fit_intercept=False,
                         tree_reg=1,
                         weight_pr_reg=1,
                         weight_l2_reg=1e-1,
                         weight_l1_reg=0,
                         sparse_reg=0,
                         # J_loss='mle',
                         J_loss='rple',
                         # J_l1_reg=1e-3,
                         J_lr=1e-4,
                         # J_lr=0,
                         slab=True,
                         # slab_prior=0.1,
                         )


# mod = bae_models.RRBMF(dim_hid=5,
#                             rank=3,
#                             nonneg=True, 
#                             # nonneg=False,
#                             sparse_reg=0,
#                             tree_reg=1,
#                             weight_pr_reg=1,
#                             weight_l2_reg=1e-1,
#                             )

# mod = bae_models.JRRBMF(dim_hid=5,
#                             rank=3,
#                             nonneg=True, 
#                             # nonneg=False,
#                             sparse_reg=1e-1,
#                             tree_reg=0,
#                             weight_pr_reg=1,
#                             weight_l2_reg=1e-3,
#                             J_lr=1e-4,
#                             )

# mod = bae_models.SCPD(dim_hid=4,
#                             nonneg=True, 
#                             # nonneg=False,
#                             sparse_reg=1,
#                             tree_reg=1,
#                             weight_pr_reg=1,
#                             weight_l1_reg=1e-2,
#                             )

# mod = bae_models.JSCPD(dim_hid=6,
#                             nonneg=True,
#                             # nonneg=False,
#                             sparse_reg=1,
#                             tree_reg=10,
#                             weight_pr_reg=1,
#                             # weight_l1_reg=1e-3,
#                             weight_l2_reg=1e-1,
#                             # J_lr=1e-4,
#                             # J_lr=0,
#                             # J_loss='logrise',
#                             # slab=True,
#                             )

en = mod.fit(X_ / X_.std(),
             period=100,
             initial_temp=100,
             decay_rate=0.88, 
             min_temp=1, 
             scl_lr=1e-4,
             lr=1e-2,
             # hot_start=False,
             )

samps = mod.sample(X_ / X_.std(), n_samp=100)
# samps = mod.sample(X_ / X_.std(), n_samp=1000, slab=False)

# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.figure()

plt.subplot(2,2,1)
plt.imshow(samps.mean(0))

plt.subplot(2,2,2)
plt.plot(en)

plt.subplot(2,2,3)
plt.imshow(mod.operator.W.T@mod.operator.W)

plt.subplot(2,2,4)
plt.imshow(mod.latent_prior.J_W + mod.latent_prior.J_W.T, 'bwr', vmin=-1, vmax=1)

#%%

foo, cond = np.unique(samps.mean(0) > 0.5, axis=0, return_inverse=True)

k = 4

print(np.mean(dics.compute_ccgp(X_, cond, 
                          1*foo[:,k][cond], svm.LinearSVC())))

print(np.mean(dics.compute_ccgp(X_, cond, 
                          1*np.random.permutation(foo[:,k])[cond], svm.LinearSVC())))


#%%

k = 2

# u = mod.operator.U.detach()[:,k]

# unqs = np.unique(results[(animal, date)]['prim_labels'])

for j,thisone in enumerate(labs[0]):
    
    # if samps.mean(0)[j,k] < 0.9:
        # continue
    plt.subplot(4,4,j+1)
    for res in results.values():
        
        for i in np.where(res['prim_labels'] == thisone)[0]:
            
            tdraw = res['prim_draw'][i][:,2]
            tneur = res['prim_t'][i]
            
            ix = np.abs(tdraw[None] - tneur[:,None]).argmin(0)
            
            plt.scatter(res['prim_draw'][i][:,0], 
                        res['prim_draw'][i][:,1],
                        # c=u[ix],
                        c=['r','b'][int(samps.mean(0)[j,k] > 0.9)],
                        alpha=0.5)
            
    plt.title(thisone)

#%%

# kays = [2,3,4,5,10,15,20]
kays = [2,3,4,5,6,7,8,9,10]
# kays = [10]

args = {
        'nonneg':True,
        # 'nonneg': False,
        'weight_pr_reg': 1,
        # 'weight_pr_reg': 0,
        # 'tree_reg': 1e-1,
        # 'sparse_reg': 1,
        'sparse_reg': 0,
        'tree_reg': 0,
        # 'weight_l1_reg': 1e-3,
        'weight_l2_reg': 1e-1,
        # 'J_lr': 1e-4,
        'J_lr': 0,
        # 'slab': True,
        'slab': False,
        # 'fit_intercept': True,
        # 'fit_intercept': False,
        }

opt_args = {'initial_temp': 100,
            'decay_rate': 0.88,
            'period': 50,
            'hot_start': True,
            # 'hot_start': False,
            'scl_lr': 1e-3,
            'min_temp': 1,
            'lr': 1e-2,
            }

n_run = 50

trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
        
        mod = bae_models.JBMF(k,**args)
        # mod = old_bae_models.SpikeNMF(k,**args)
        
        wa,ba = bae_util.impcv(mod, X_/X_.std(), verbose=False, n_sample=10, folds=10, **opt_args)
        # wa,ba = bae_util.gabriel_bicv(mod, X_/X_.std(), n_samp=10, **opt_args)
        
        trn[i] += np.mean(wa) / n_run
        tst[i] += np.mean(ba) / n_run

plt.plot(kays, trn)
plt.plot(kays, tst, '--')
# plt.plot(kays, np.mean(trn,axis=0))
# plt.plot(kays, np.mean(tst, axis=0))


#%%

S = 1*(samps.mean(0) > 0.5)

rep = X_
# rep = Usc
# signal = 1
# noise = 0
# rep = signal*S@np.random.randn(6, X_.shape[-1]) + noise*np.random.randn(*X_.shape)

ps = np.zeros((mod.dim_hid,mod.dim_hid))
for i in range(mod.dim_hid):
    for j in range(mod.dim_hid):
        
        edges = 1*((util.yuke(S) == 1) * (S[:,[i]] != S[:,[i]].T))
        aye_i, jay_i = np.where(np.triu(edges))
        
        edges = 1*((util.yuke(S) == 1) * (S[:,[j]] != S[:,[j]].T))
        aye_j, jay_j = np.where(np.triu(edges))
        
        cs = util.cosine_sim((rep[aye_i] - rep[jay_i]).T, (rep[aye_j] - rep[jay_j]).T)
        
        if i == j:
            ps[i,j] = np.sum(np.triu(cs, k=1)) / spc.binom(6,2)
        else:
            ps[i,j] = np.trace(cs) / 6


plt.imshow(ps, 'bwr', vmin=-0.4, vmax=0.4)


