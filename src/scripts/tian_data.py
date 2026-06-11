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
import bae
import bae_models
import bae_search
import bae_util
import plotting as tpl

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
date = 231206
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
    # deez = trls['task_kind'] != 'prims_on_grid'
    deez = trls['task_kind'] == 'character'
    trls = trls[deez]
    allX = allX[:,deez]

#%%

## Figure out which strokes were part of the same trial
chars, cidx = np.unique(trls['character'], return_inverse=True)
trial_idx = np.cumsum(np.diff(cidx, prepend=-1) != 0)

# region = 'dlPFC'
# region = 'PMv'
region = 'preSMA'

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
