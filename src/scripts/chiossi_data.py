from __future__ import annotations

CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/chiossi_data/' 

import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/hpc-hierarchy')

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

from func_behaviour import load_behaviour, get_trials_percat
import func_basics as basics
import func_tracking as tracking
from class_cluster import ClusterPopulation
import pickle

#%%
import numpy as np
import pandas as pd


def get_dig_events(
    lwhl,
    reward_coords,
    behaviour,
    whl_shifts,
    whl_rate=50,
    radius=12.0,
    speed_thresh=6.0,
    stable_check_frames=100,
    has_initial_sleep=True,
):
    """
    Extract the onset frame of each digging event for a single recording day.

    Parameters
    ----------
    lwhl : ndarray, shape (T,)
        Linearised position trace in cm. Untracked frames should be <= 0.
    reward_coords : dict
        Reward positions, e.g. {'U': 300, 'A': 100, 'B': 180}.
        Keys must match the column names in `behaviour`.
    behaviour : DataFrame
        One row per trial (training + probe).  Required columns:
        ``Type`` ('Learning'/'Visible'/'Probe'), ``Context``, ``Start``,
        ``Correct``, and one binary column per key in ``reward_coords``
        (1 = dug that well, 0 = did not).
    whl_shifts : ndarray
        Cumulative end-frame indices (already in whl_rate units) for every
        recorded segment.  Normal layout (``has_initial_sleep=True``):

            shifts[0]       end of sleep-1  = start of training trial 1
            shifts[1..T]    end of training trials 1..T
            shifts[T+1]     end of sleep-2  = start of probe trial 1
            shifts[T+2..]   end of probe trials

        For the missing-sleep session (jc250 / 050221), pass
        ``has_initial_sleep=False``; shifts then start at end of trial 1.
    whl_rate : int
        Tracking sample rate in Hz (default 50). Used only for ``time_s``.
    radius : float
        Distance in cm from a reward position counted as "at the well" (12).
    speed_thresh : float
        Maximum speed in cm/s counted as stationary / digging (6).
    stable_check_frames : int
        Look-ahead window (frames) for the stability check (100 ≈ 2 s).
        An entry is accepted only if the mean position over the next
        ``stable_check_frames`` frames is still inside the reward zone
        and mean speed is < 10 cm/s (mirrors dig_start in func_mazetime.py).

    Returns
    -------
    DataFrame
        One row per (trial, dug well).  Columns: trial_idx, trial_type,
        context, start_side, correct, well, frame, time_s.
        ``frame`` is the whl-frame of first stable entry (NaN if not found).
        ``time_s`` = frame / whl_rate.
    """
    # ── Speed from position differences, 10-frame centred rolling mean ──
    dx    = np.diff(lwhl)
    valid = (lwhl[:-1] > 0) & (lwhl[1:] > 0)
    raw   = np.append(np.where(valid, np.abs(dx) * whl_rate, np.nan), np.nan)
    speed = pd.Series(raw).rolling(10, center=True, min_periods=1).mean().to_numpy(copy=True)
    speed[lwhl <= 0] = np.nan

    # ── Session layout helpers ───────────────────────────────────────────
    beh     = behaviour.reset_index(drop=True)
    n_train = int(beh["Type"].isin(["Learning", "Visible"]).sum())
    wells   = list(reward_coords)

    records = []
    for i, row in beh.iterrows():
        ttype = row["Type"]

        if ttype in ("Learning", "Visible"):
            if has_initial_sleep:
                t0, t1 = int(whl_shifts[i]), int(whl_shifts[i + 1])
            else:
                t0 = int(whl_shifts[i - 1]) if i > 0 else 0
                t1 = int(whl_shifts[i])
        elif ttype == "Probe":
            j      = i - n_train
            offset = n_train + 1 if has_initial_sleep else n_train
            t0, t1 = int(whl_shifts[offset + j]), int(whl_shifts[offset + j + 1])
        else:
            continue

        t1 = min(t1, len(lwhl))

        for well in wells:
            if not row.get(well, 0):
                continue

            rpos = reward_coords[well]
            cand = ((lwhl > rpos - radius) & (lwhl < rpos + radius)
                    & (speed < speed_thresh)).copy()
            cand[:t0] = False
            cand[t1:] = False

            for entry in np.where(np.diff(cand.astype(np.int8)) > 0)[0]:
                end = min(entry + stable_check_frames, len(lwhl))
                if (rpos - radius < np.nanmean(lwhl[entry:end]) < rpos + radius
                        and np.nanmean(speed[entry:end]) < 10.0):
                    records.append(dict(
                        trial_idx  = i,
                        trial_type = ttype,
                        context    = row["Context"],
                        start_side = row["Start"],
                        correct    = int(row["Correct"]),
                        well       = well,
                        frame      = float(entry),
                        time_s     = float(entry) / whl_rate,
                    ))
                    break

    return pd.DataFrame(records)

def population_vectors(
    spike_times,
    spike_ids,
    lwhl,
    whl_shifts,
    behaviour,
    pos_edges,
    speed=None,
    speed_thres=3.0,
    whl_rate=50,
    dir_filter=True,
    neuron_ids=None,
):
    """
    Compute per-trial, per-position firing-rate population vectors.

    Reproduces the representation fed into hierarchical clustering in
    Chiossi et al. 2025 (PNAS), specifically `popvec_perpos_fromedges`
    from func_popanalysis.py with direction filtering enabled.

    Each row of the output is one (trial × position bin) sample — a
    population vector of mean firing rates (Hz) across all neurons, for
    frames when the animal was in that position bin during that trial,
    moving above threshold, and (if dir_filter=True) moving in the
    direction that matches the trial's start side (i.e. the outbound pass).

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Spike timestamps in WHL-frame units (50 Hz). If loading from the
        raw .res file (20 kHz) convert first:
            spike_times_whl = (res_times * whl_rate / res_rate).astype(int)
    spike_ids : ndarray, shape (n_spikes,)
        Neuron ID for each spike, parallel to spike_times.
    lwhl : ndarray, shape (T,)
        Linearised position trace in cm. Untracked frames should be <= 0.
    whl_shifts : ndarray
        Cumulative end-frame indices (whl-frame units) for each recorded
        segment. For training trial i (0-based):
            start = whl_shifts[i],  end = whl_shifts[i+1]
        (i.e. whl_shifts[0] = end of sleep-1 = start of trial 0).
    behaviour : DataFrame
        One row per training trial with columns Start ('L'/'R'),
        Context ('A'/'B'), Correct (0/1), error_type.
    pos_edges : array-like, shape (npos+1,)
        Position bin edges in cm, e.g. np.arange(0, max_pos+bin_size, bin_size).
    speed : ndarray, optional
        Pre-computed speed in cm/s aligned with lwhl. Computed from lwhl
        if not provided.
    speed_thres : float
        Minimum speed (cm/s) for a frame to be included (default 3).
    whl_rate : int
        Tracking rate in Hz (default 50).
    dir_filter : bool
        If True (default), keep only frames where movement direction matches
        the trial's start side — i.e. the outbound pass of each trial.
    neuron_ids : array-like, optional
        Neuron IDs to include, in order. Defaults to sorted unique values
        in spike_ids (typically you'd pass only pyramidal-cell IDs here).

    Returns
    -------
    pop_vec_flat : ndarray, shape (n_samples, n_neurons)
        Firing rates in Hz. n_samples = ntrials × npos_bins minus any
        (trial, position) pairs the animal never visited.
        NaN cells (unvisited bins) are dropped as rows, not zero-filled,
        matching the exclude_untracked=True default in the original code.
    labels : DataFrame, shape (n_samples, 5)
        One row per sample, aligned with pop_vec_flat. Columns:
        trial_idx, position_bin, context, start_side, correct.
    """
    beh      = behaviour.reset_index(drop=True)
    ntrials  = len(beh)
    pos_edges = np.asarray(pos_edges)
    npos     = len(pos_edges) - 1

    if neuron_ids is None:
        neuron_ids = np.array(sorted(np.unique(spike_ids)))
    neuron_ids = np.asarray(neuron_ids)
    n_neurons  = len(neuron_ids)

    # ── Speed ─────────────────────────────────────────────────────────
    if speed is None:
        dx    = np.diff(lwhl)
        valid = (lwhl[:-1] > 0) & (lwhl[1:] > 0)
        raw   = np.append(np.where(valid, np.abs(dx) * whl_rate, np.nan), np.nan)
        speed = (pd.Series(raw)
                   .rolling(10, center=True, min_periods=1)
                   .mean()
                   .to_numpy(copy=True))
        speed[lwhl <= 0] = np.nan

    speed_ok = np.where(speed > speed_thres)[0]

    # ── Direction filter (applied globally, then restricted per trial) ─
    # For each frame, mark whether movement matches that trial's start side.
    # +1 = rightward (increasing position), -1 = leftward.
    if dir_filter:
        mov_dir = np.sign(
            pd.Series(np.where(lwhl > 0, lwhl, np.nan))
              .diff(5).fillna(0).to_numpy()
        )
        side_map   = {'L': -1, 'R': 1}
        trial_filt = np.zeros(len(lwhl))
        for t in range(ntrials):
            t0, t1 = int(whl_shifts[t]), int(whl_shifts[t + 1])
            trial_filt[t0:t1] = side_map[beh.iloc[t]['Start']]
        dir_ok = np.where(mov_dir * trial_filt == 1)[0]

    # ── Per-neuron spike index ─────────────────────────────────────────
    spk = {nid: spike_times[spike_ids == nid] for nid in neuron_ids}

    # ── Main loop ─────────────────────────────────────────────────────
    pop_vec = np.full((ntrials, npos, n_neurons), np.nan)

    for p in range(npos):
        # Frames in this position bin, above speed threshold
        in_bin = np.intersect1d(
            np.where((lwhl > pos_edges[p]) & (lwhl <= pos_edges[p + 1]))[0],
            speed_ok
        )
        
        if dir_filter:
            in_bin = np.intersect1d(in_bin, dir_ok)

        for ni, nid in enumerate(neuron_ids):
            # Spikes that landed in this position bin (globally)
            spk_in_bin = np.intersect1d(in_bin, spk[nid])
            
            for t in range(ntrials):
                t0, t1    = int(whl_shifts[t]), int(whl_shifts[t + 1])
                time_in   = np.sum((in_bin    > t0) & (in_bin    < t1)) / whl_rate
                if time_in == 0:
                    continue  # stays NaN → row will be dropped
                n_spikes  = np.sum((spk_in_bin > t0) & (spk_in_bin < t1))
                pop_vec[t, p, ni] = n_spikes / time_in

    # ── Flatten to (ntrials × npos, n_neurons) ────────────────────────
    pop_vec_flat = pop_vec.reshape(-1, n_neurons)

    # ── Labels (built before row removal so indices align) ────────────
    labels = pd.DataFrame([
        {'trial_idx':    t,
         'position_bin': p,
         'context':      beh.iloc[t]['Context'],
         'start_side':   beh.iloc[t]['Start'],
         'correct':      int(beh.iloc[t]['Correct'])}
        for t in range(ntrials)
        for p in range(npos)
    ])

    # ── Drop (trial, pos) pairs the animal never visited ──────────────
    keep = ~np.isnan(pop_vec_flat[:, 0])
    return pop_vec_flat[keep], labels[keep].reset_index(drop=True)


def zscore(x, center=True):
    return (x - center*x.mean(0))/np.sqrt(np.mean((x-x.mean(0))**2, axis=0))


#%%

rec_rate = 24000    #as in the shifts file
res_rate = 20000
whl_rate = 50

dt = 0.5

neurons = []
subject = []
session = []
condition = []
celltype = []
perf = []

for animal in ['jc233', 'jc243', 'jc250', 'jc253', 'jc259']:
    
    coords = pd.read_csv(f"{LOAD_DIR}/{animal}/{animal}-reward.coord", delimiter=' ')
    coords = {k: coords[k].item() for k in coords.keys()}
    
    for date in tqdm(os.listdir(f"{LOAD_DIR}/{animal}")):
        if animal in date:
            continue 
        
        beh = pd.read_csv(f"{LOAD_DIR}/behaviour/{animal}-{date}.csv")
        
        for fil in os.listdir(f"{LOAD_DIR}/{animal}/{date}"):
            
            if 'clu' in fil:
                with open(f"{LOAD_DIR}/{animal}/{date}/{fil}") as f:
                    neurid = np.array([int(x) for x in f])
                
            elif 'res' in fil:
                with open(f"{LOAD_DIR}/{animal}/{date}/{fil}") as f:
                    times = np.array([int(x) for x in f])
            elif 'des' in fil:
                neurtype = np.genfromtxt(f"{LOAD_DIR}/{animal}/{date}/{fil}", dtype='str')
            elif 'lwhl' in fil:
                whl = np.loadtxt(f"{LOAD_DIR}/{animal}/{date}/{fil}")
            elif 'shifts' in fil:
                shifts = np.loadtxt(f"{LOAD_DIR}/{animal}/{date}/{fil}")
            
        N = neurid[0] - 1
        neurid = neurid[1:]
        deez = neurid > 1
        
        neurid = neurid[deez] - 2
        times = times[deez] / res_rate
        
        whl_shifts = (shifts / (rec_rate/whl_rate)).astype(int)
        
        digs = get_dig_events(whl, coords, beh, whl_shifts, whl_rate=whl_rate)
        digs = digs.sort_values('time_s')
        
        start = digs['time_s'].to_numpy()
        correct = ((digs['well'] == digs['context']) | (digs['well'] == 'U'))
        deez = (correct * (digs['trial_type'] == 'Learning')).to_numpy()
        
        conds = digs[['start_side', 'context', 'well']][deez].to_numpy()
        
        unqs, idx = np.unique(conds.sum(1), return_inverse=True)
        
        bins = np.array([start[deez], start[deez]+dt]).T.flatten()
        
        if len(unqs) < 8:
            print('bad')
            continue
            
        X = []
        for neur in range(N):
            X.append(np.histogram(times[neurid==neur], bins)[0][::2] )
        X = np.array(X).T
        
        X = zscore(X[:,(X.sum(0)>0)*(neurtype=='p1')], center=False)
        
        for i,lab in enumerate(unqs):
            neurons.append(X[idx==i])
            condition.append(lab)
            subject.append(animal)
            session.append(date)
            perf.append(np.mean(correct[digs['trial_type'] == 'Learning']))
            # celltype.append(neurtype)

neurons = np.array(neurons, dtype=object)
condition = np.array(condition)
subject = np.array(subject)
session = np.array(session)
perf = np.array(perf)
# celltype = np.concatenate(celltype)

#%%

pp = util.ppop([np.median(x,axis=0, keepdims=True) for x in neurons[perf > 0.9]], 
               condition[perf > 0.9], 
               (subject+session)[perf > 0.9],
               subsets=subject[perf > 0.9],
               independent=True, 
               K=1)


#%%



