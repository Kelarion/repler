CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/hunt_data/' 

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
import old_bae
import old_bae_models
import old_bae_search
import bae_util
import plotting as tpl



#%%

# timebins of Cue1_MatrixRaw, relative to cue-1 onset (matches pre_cue=199,
# post_cue=600 in InfoGathering_Preprocess_Neurons.m -> neuron_matrix.m)
TIMEBINS = np.arange(-199, 601)  # length 800


# ----------------------------------------------------------------------------
# neuron_matrix.m  (only needed if you re-epoch raw spikes yourself; the
# processed files already contain Cue1_MatrixRaw, so this is for completeness)
# ----------------------------------------------------------------------------
def neuron_matrix(pre_period, start_code, post_period, timestamps):
    """Faithful port of neuron_matrix.m.

    Returns an (nTrials x pre+post+1) raster with 1000 at each spike (so it
    reads directly in Hz at 1 kHz). Divide by 1000 to get a binary raster,
    which is what the pipeline stores as *_MatrixRaw.

    start_code : (nTrials,) event time per trial, ms
    timestamps : sequence of length nTrials, each a 1-D array of spike times, ms
    """
    n = len(timestamps)
    width = int(pre_period + post_period + 1)
    out = np.zeros((n, width))
    for a in range(n):
        ts = np.asarray(timestamps[a]).ravel()
        lo = start_code[a] - pre_period
        hi = start_code[a] + post_period
        sel = ts[(ts >= lo) & (ts <= hi)]
        # MATLAB: temp - (start-pre) + 1  (1-indexed) -> 0-indexed: temp - lo
        idx = np.rint(sel - lo).astype(int)
        out[a, idx] = 1000
    return out


# ----------------------------------------------------------------------------
# per-unit condition-mean vector  (the core of calculate_RSA_matrices_updown.m)
# ----------------------------------------------------------------------------
_KEYS = ("Lreg", "Rreg", "P_first", "First_pos", "Cue1_MatrixRaw", "brain_region")

# brain_region codes from InfoGathering_Preprocess_Neurons.m / the *_area tables
REGION_NAMES = {1: "ACC", 2: "DLPFC", 3: "OFC"}  # 4/5/6 -> "Other"


def load_unit(path):
    """Load only the arrays needed for the RSA representation from a Unit####.mat."""
    m = scio.loadmat(path, variable_names=_KEYS)
    return {
        "Lreg": m["Lreg"].astype(np.int64),                 # (nTrials, 4)
        "Rreg": m["Rreg"].astype(np.int64),                 # (nTrials, 4)
        "P_first": m["P_first"].ravel().astype(np.int64),   # (nTrials,)
        "First_pos": m["First_pos"].ravel().astype(np.int64),
        "Cue1_MatrixRaw": m["Cue1_MatrixRaw"].astype(np.float64),  # (nTrials, 800)
        "brain_region": int(np.ravel(m["brain_region"])[0]),       # 1/2/3/4/5/6
    }


def unit_condition_vector(unit, tin=100, tout=500, split_updown=True):
    """Mean spike count per condition for one unit -> (40,) or (20,) vector.

    Mirrors calculate_RSA_matrices_updown.m exactly:
      multmat (C x nTrials) one-hot assignment of each trial to its cue-1
      condition, row-normalised by number of presentations, times the per-trial
      spike count in (tin, tout) ms.
    """
    Lreg, Rreg = unit["Lreg"], unit["Rreg"]
    P = unit["P_first"]
    Up = unit["First_pos"] % 2            # 1 = cue 1 on top row, 0 = bottom
    notP, notUp = 1 - P, 1 - Up
    L1, R1 = Lreg[:, 0], Rreg[:, 0]       # cue-1 rank if on left / right, else 0
    nTr = Lreg.shape[0]

    if split_updown:
        # 8 groups x 5 ranks = 40, offsets 0,5,...,35
        groups = [
            L1 * P * Up,    L1 * notP * Up,    R1 * P * Up,    R1 * notP * Up,
            L1 * P * notUp, L1 * notP * notUp, R1 * P * notUp, R1 * notP * notUp,
        ]
        n_cond = 40
    else:
        # collapse over top/bottom: 4 groups x 5 ranks = 20
        groups = [L1 * P, L1 * notP, R1 * P, R1 * notP]
        n_cond = 20

    multmat = np.zeros((n_cond, nTr))
    for g, off in zip(groups, range(0, n_cond, 5)):
        tr = np.nonzero(g > 0)[0]
        multmat[g[tr] - 1 + off, tr] = 1.0

    nPres = multmat.sum(axis=1)
    nz = nPres > 0
    multmat[nz] /= nPres[nz, None]        # row -> averaging operator
    # rows with nPres == 0 are already all-zero

    win = (TIMEBINS > tin) & (TIMEBINS < tout)
    nSpikes = unit["Cue1_MatrixRaw"][:, win].sum(axis=1)   # (nTrials,)
    return multmat @ nSpikes               # (n_cond,)


# ----------------------------------------------------------------------------
# row labels: the task variables defining each condition (same block ordering
# the matrix uses, so row_labels[i] describes row i of unitmat exactly)
# ----------------------------------------------------------------------------
def condition_labels(split_updown=True):
    """Structured array of length 40 (or 20) labelling each condition row.

    Fields: side ('L'/'R'), attribute ('Prob'/'Mag'), rank (1-5), and, when
    split_updown, row ('Top'/'Bottom'); plus a human-readable 'label' string.
    """
    if split_updown:
        dt = [("side", "U1"), ("attribute", "U4"), ("row", "U6"),
              ("rank", "i8"), ("label", "U24")]
        recs = [(s, a, r, k, f"{s}-{a}-{r}-{k}")
                for r in ("Top", "Bottom")        # blocks 0-19 Top, 20-39 Bottom
                for s in ("L", "R")
                for a in ("Prob", "Mag")
                for k in range(1, 6)]
    else:
        dt = [("side", "U1"), ("attribute", "U4"), ("rank", "i8"),
              ("label", "U16")]
        recs = [(s, a, k, f"{s}-{a}-{k}")
                for s in ("L", "R")
                for a in ("Prob", "Mag")
                for k in range(1, 6)]
    return np.array(recs, dtype=dt)

# ----------------------------------------------------------------------------
# stack units -> (C x N) representation matrix, with row and column labels
# ----------------------------------------------------------------------------
def build_unitmat(unit_paths, tin=100, tout=500, split_updown=True):
    """Return (unitmat, row_labels, col_labels).

    unitmat    : (C x N) representation matrix, C = 40 (or 20)
    row_labels : structured array of length C, task variables per condition
    col_labels : (N,) array of brain-region names per unit ('ACC'/'DLPFC'/
                 'OFC'/'Other'), read from each file's saved brain_region
    """
    cols, regions = [], []
    for p in unit_paths:
        u = load_unit(p)
        cols.append(unit_condition_vector(u, tin, tout, split_updown))
        regions.append(REGION_NAMES.get(u["brain_region"], "Other"))
    unitmat = np.stack(cols, axis=1)               # (C, N)
    row_labels = condition_labels(split_updown)    # (C,)
    col_labels = np.array(regions)                 # (N,)
    return unitmat, row_labels, col_labels


# ----------------------------------------------------------------------------
# FigS4_RSA_split.m corrcoef step:  RSAmat = corrcoef( normalise(unitmat(:,ok),1)' )
# ----------------------------------------------------------------------------
def rsa_matrix(unitmat):
    """(C x N) condition-mean matrix -> (C x C) RSA correlation matrix.

    Drops never-spiking units, z-scores each unit across conditions (the
    `normalise(...,1)` step), then correlates conditions across the population.
    The per-unit z-score puts every neuron on a common scale before the
    cross-neuron correlation; the corrcoef itself is what RSA reports.
    """
    ok = ~np.all(unitmat == 0, axis=0)     # exclude cells that never spike
    U = unitmat[:, ok]
    Z = (U - U.mean(axis=0, keepdims=True)) / U.std(axis=0, keepdims=True)
    # rows of Z are conditions; np.corrcoef(Z) correlates conditions across units
    return np.corrcoef(Z)                   # (C x C)


#%%

paths = [f"{LOAD_DIR}/MonkeyF/LTH_processed_data/{p}" for p in os.listdir(f"{LOAD_DIR}/MonkeyF/LTH_processed_data/")]
X, conds, area = build_unitmat(paths, split_updown=False)
        
conds = np.array([c.tolist() for c in conds])
_, ix = np.unique(conds[:,[1,2]], axis=0, return_inverse=True)

stim = np.eye(10)[ix]
rank = conds[:,2].astype(int)

#%%

this_area = 'OFC'
# this_area = 'ACC'
# this_area = 'DLPFC'

X_ = X[:,area==this_area] / X[:,area==this_area].std(0, keepdims=True)
X_ = X_ / X_.std(1, keepdims=True)

# X_ -= X_.mean(0)

mod = old_bae_models.SemiBMF(12,
                         nonneg=True, 
                         tree_reg=1,
                         weight_pr_reg=1,
                         weight_l2_reg=1e-2,
                         weight_l1_reg=0,
                         sparse_reg=1,
                         )

# mod = old_bae_models.SemiBMF(4,
#                          nonneg=True,
#                          tree_reg=0,
#                          weight_pr_reg=1,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=0,
#                          sparse_reg=1,
#                          )

# mod = old_bae_models.SpikeNMF(11,
#                          nonneg=True, 
#                          sparse_reg=1,
#                          weight_pr_reg=1,
#                          tree_reg=0,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=0,
#                          )

# mod = old_bae_models.SpikeNMF(4,
#                          nonneg=True, 
#                          sparse_reg=1,
#                          weight_pr_reg=1,
#                          tree_reg=0,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=0,
#                          )

# mod = old_bae_models.KernelBMF2(12,
#                            sparse_reg=1,
#                            tree_reg=100,
#                            uniform_scale=False,
#                            # l1_reg=0,
#                            )

en = mod.fit(X_ ,
             period=100, 
             initial_temp=10,
             decay_rate=0.9, 
             min_temp=1, 
             scl_lr=1e-4,
             )

# en = mod.fit(X_ / X_.std(),
#              period=100, 
#              initial_temp=10,
#              decay_rate=0.9, 
#              min_temp=1e-4, 
#              scl_lr=0,
#              )


samps = mod.sample(X_ , n_samp=1000)
# samps = mod.sample(X_ / X_.std(), n_samp=1000, slab=False)
# 
# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.imshow(samps.mean(0))

#%%


# kays = [2,3,4,5,10,15,20]
# kays = [2,3,4,5,6,7,8,9,10]
kays = [2,4,6,8,10,12,14,16]
# kays = [10]

args = {
        'nonneg':True,
        # 'nonneg': False,
        'weight_pr_reg': 1,
        # 'weight_pr_reg': 0,
        'tree_reg': 0,
        'sparse_reg': 1,
        # 'tree_reg': 0,
        'weight_l1_reg': 0,
        'weight_l2_reg': 1e-2,
        # 'fit_intercept': True,
        # 'fit_intercept': False,
        }

# args = {
#         'nonneg':True,
#         # 'nonneg': False,
#         'weight_pr_reg': 1,
#         # 'weight_pr_reg': 0,
#         'tree_reg': 0,
#         'sparse_reg': 1,
#         # 'tree_reg': 0,
#         # 'weight_l1_reg': 1e-3,
#         'weight_l1_reg': 0,
#         'weight_l2_reg': 1e-2,
#         # 'fit_intercept': True,
#         # 'fit_intercept': False,
#         'slab_prior': 1,
#         }

opt_args = {'initial_temp': 10,
            'decay_rate': 0.9,
            'period': 50,
            'hot_start': True,
            'min_temp': 1,
            # 'hot_start': False,
            'scl_lr': 1e-4,
            # 'scl_lr': 0,
            # 'lr': 1e-1,
            }

n_run = 3
modclass = old_bae_models.SemiBMF
# modclass = old_bae_models.SpikeNMF

full = np.zeros(len(kays))
trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
         
        # mod = old_bae_models.SemiBMF(k,**args)
        mod = modclass(k,**args)
        
        # wa,ba = bae_util.impcv(mod, X_ / X_.std(), folds=10, n_sample=1000, verbose=False, **opt_args)
        # wa, ba = bae_util.loocv(mod, X_ / X_.std(), n_sample=10, **opt_args)
        wa, ba = bae_util.gabriel_bicv(mod, X_/X_.std(), n_samp=1000, **opt_args)
        
        trn[i] += np.mean(wa) / n_run
        tst[i] += np.mean(ba) / n_run
        
        mod = modclass(k,**args)
        en = mod.fit(X_ / X_.std(), verbose=False, **opt_args)
        samps = mod.sample(X_ / X_.std(), n_samp=10)
        full[i] += mod.loglikelihood(X_ / X_.std(), mod(samps)).mean() / n_run

plt.plot(kays, trn)
plt.plot(kays, tst, '--')

#%%




#%%
# =============================================================================
# Automated search: preprocessing × model × hyperparameters
# =============================================================================

def preprocess_X(X, area_mask,
                 center_neurons: bool = True,
                 zscore_neurons: bool = True,
                 center_conditions: bool = False,
                 zscore_conditions: bool = True,
                 clip_percentile=None,
                 global_scale: bool = True):
    """
    Preprocess (C x N) condition-by-neuron matrix.
    area_mask : boolean (N,) selecting neurons to include.
    Returns (C x N') float array.
    """
    Xa = X[:, area_mask].astype(float)
    ok = np.any(Xa != 0, axis=0)
    Xa = Xa[:, ok]
    if center_neurons:
        Xa -= Xa.mean(0, keepdims=True)
    if zscore_neurons:
        s = Xa.std(0, keepdims=True)
        Xa /= np.where(s > 1e-9, s, 1.0)
    if clip_percentile is not None:
        p = np.percentile(np.abs(Xa), clip_percentile)
        Xa = np.clip(Xa, -p, p)
    if center_conditions:
        Xa -= Xa.mean(1, keepdims=True)
    if zscore_conditions:
        s = Xa.std(1, keepdims=True)
        Xa /= np.where(s > 1e-9, s, 1.0)
    if global_scale:
        g = Xa.std()
        if g > 1e-9:
            Xa /= g
    return Xa


def task_regressors(conds):
    """
    Build a dict of normalised regressors for all task variables
    and their pairwise conjunctions (side × attr × rank thresholds).
    """
    side = (conds['side'] == 'L').astype(float)
    attr = (conds['attribute'] == 'Prob').astype(float)
    rank = conds['rank'].astype(float)

    tv = {
        'side':      side,
        'attr':      attr,
        'rank':      rank,
        'side*attr': side * attr,
        'rank*attr': rank * attr,
        'rank*side': rank * side,
        'rank*s*a':  rank * side * attr,
    }
    for thr in range(2, 5):
        tv[f'rank>={thr}'] = (rank >= thr).astype(float)

    for s_lbl, s_val in [('L', 1.0), ('R', 0.0)]:
        for a_lbl, a_val in [('Pr', 1.0), ('Mg', 0.0)]:
            grp = ((side == s_val) & (attr == a_val)).astype(float)
            key = f'{s_lbl}_{a_lbl}'
            tv[key]           = grp
            tv[f'rk|{key}']   = rank * grp
            for thr in range(2, 5):
                tv[f'r>={thr}|{key}'] = (rank >= thr).astype(float) * grp

    # exact-rank / stimulus regressors (e.g. r==1|Pr → 10000 00000 10000 00000)
    for k in [1, 2]:
        v_k = (rank == k).astype(float)
        tv[f'rank=={k}']  = v_k
        tv[f'r=={k}|Pr']  = v_k * attr
        tv[f'r=={k}|Mg']  = v_k * (1.0 - attr)
        for s_lbl, s_val in [('L', 1.0), ('R', 0.0)]:
            for a_lbl, a_val in [('Pr', 1.0), ('Mg', 0.0)]:
                grp = ((side == s_val) & (attr == a_val)).astype(float)
                tv[f'r=={k}|{s_lbl}_{a_lbl}'] = v_k * grp

    out = {}
    for name, v in tv.items():
        v = v - v.mean()
        sd = v.std()
        if sd > 1e-9:
            out[name] = v / sd
    return out


def interp_score(S, tv):
    """
    Mean (over all K latents, including dead ones) of best R² with any
    task variable from `tv`.  Range [0, 1]; higher = more interpretable.
    """
    K = S.shape[1]
    per_k = np.zeros(K)
    for k in range(K):
        s = S[:, k].astype(float)
        s -= s.mean()
        if np.abs(s).max() < 1e-9:
            continue
        s /= s.std()
        per_k[k] = max(float(np.corrcoef(s, v)[0, 1]**2)
                       for v in tv.values())
    return float(per_k.mean())


# ---- fitting schedules -------------------------------------------------------

# Quick schedule for broad search (~300 iters, fast exploration)
_QUICK = dict(period=10, initial_temp=50, decay_rate=0.9,
              min_temp=5, max_iter=300, verbose=False)

# Full schedule for focused refinement (mirrors original usage, ~6550 iters)
_FULL  = dict(period=50, initial_temp=100, decay_rate=0.9,
              min_temp=1, verbose=False)


def _make_and_fit(Xp, model_class, cfg, sched):
    """Instantiate model from cfg dict, fit, and return it."""
    if model_class is old_bae_models.SemiBMF:
        mod = old_bae_models.SemiBMF(
            cfg['dim_hid'],
            nonneg        = cfg['nonneg'],
            sparse_reg    = cfg['sparse_reg'],
            tree_reg      = cfg['tree_reg'],
            weight_pr_reg = cfg['weight_pr_reg'],
            weight_l1_reg = cfg['weight_l1_reg'],
            weight_l2_reg = 0,
        )
        mod.fit(Xp, **sched, scl_lr=cfg.get('scl_lr', 1e-3))

    elif model_class is old_bae_models.SpikeNMF:
        mod = old_bae_models.SpikeNMF(
            cfg['dim_hid'],
            nonneg        = cfg.get('nonneg', True),
            sparse_reg    = cfg['sparse_reg'],
            tree_reg      = cfg['tree_reg'],
            weight_pr_reg = cfg['weight_pr_reg'],
            weight_l1_reg = cfg.get('weight_l1_reg', 0.0),
            weight_l2_reg = cfg.get('weight_l2_reg', 0.1),
            slab_prior    = cfg.get('slab_prior', 1.0),
        )
        mod.fit(Xp, **sched, scl_lr=cfg.get('scl_lr', 1e-3))

    elif model_class is old_bae_models.KernelBMF2:
        mod = old_bae_models.KernelBMF2(
            cfg['dim_hid'],
            sparse_reg    = cfg['sparse_reg'],
            tree_reg      = cfg['tree_reg'],
            uniform_scale = cfg['uniform_scale'],
        )
        mod.fit(Xp, **sched, scl_lr=1.0)

    elif model_class is old_bae_models.BiPCA:
        # BiPCA uses closed-form SVD in the M-step; no learning rate needed
        mod = old_bae_models.BiPCA(
            cfg['dim_hid'],
            sparse_reg = cfg['sparse_reg'],
            tree_reg   = cfg['tree_reg'],
        )
        mod.fit(Xp, **sched)

    else:
        raise ValueError(f'Unknown model: {model_class}')
    return mod


def _get_S(mod):
    """Return binary S matrix. SpikeNMF stores S*Z (continuous); threshold at 0."""
    if isinstance(mod, old_bae_models.SpikeNMF):
        return (mod.S > 0).astype(float)
    return mod.S.copy()


def _fit_n(Xp, tv, model_class, cfg, sched, n_runs, base_seed=None,
           score_fn=None):
    """Fit n_runs times; return (best_S, mean_score, std_score, best_seed)."""
    if score_fn is None:
        score_fn = interp_score
    scores, best_S, best_sc, best_seed = [], None, -1.0, None
    for i in range(n_runs):
        seed = None if base_seed is None else base_seed + i
        if seed is not None:
            np.random.seed(seed)
        try:
            mod = _make_and_fit(Xp, model_class, cfg, sched)
            S   = _get_S(mod)
            sc  = score_fn(S, tv)
            scores.append(sc)
            if sc > best_sc:
                best_sc, best_S, best_seed = sc, S, seed
        except Exception:
            pass
    if not scores:
        return None, 0.0, 0.0, None
    return best_S, float(np.mean(scores)), float(np.std(scores)), best_seed


def broad_search(X, conds, area, n_trials=300, n_runs=3, seed=0):
    """
    Random search over preprocessing × model class × hyperparameters.
    Returns list of result dicts sorted descending by mean_score.
    """
    rng    = np.random.default_rng(seed)
    tv     = task_regressors(conds)
    models = [old_bae_models.SemiBMF, old_bae_models.KernelBMF2, old_bae_models.BiPCA]
    results = []

    for _ in tqdm(range(n_trials), desc='broad search'):

        this_area = ['OFC', 'ACC', 'DLPFC'][int(rng.integers(3))]
        mask = (area == this_area)
        if mask.sum() < 5:
            continue

        clip_idx = int(rng.choice(3, p=[0.5, 0.25, 0.25]))
        prep = dict(
            center_neurons    = bool(rng.integers(2)),
            zscore_neurons    = bool(rng.integers(2)),
            center_conditions = bool(rng.integers(2)),
            zscore_conditions = bool(rng.integers(2)),
            clip_percentile   = [None, 95, 99][clip_idx],
            global_scale      = True,
        )

        model_class = models[int(rng.integers(3))]
        dim_hid     = int(rng.choice([3, 4, 5, 6, 7, 8]))

        if model_class is old_bae_models.SemiBMF:
            cfg = dict(
                dim_hid       = dim_hid,
                nonneg        = bool(rng.integers(2)),
                sparse_reg    = float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.0])),
                tree_reg      = float(rng.choice([0.01, 0.05, 0.1, 0.5, 1.0])),
                weight_pr_reg = float(rng.choice([0.1, 0.5, 1.0, 5.0])),
                weight_l1_reg = float(rng.choice([0.0, 0.01, 0.05, 0.1])),
            )
        elif model_class is old_bae_models.KernelBMF2:
            cfg = dict(
                dim_hid       = dim_hid,
                sparse_reg    = float(rng.choice([0.0, 0.1, 0.5, 1.0])),
                tree_reg      = float(rng.choice([0.01, 0.1, 0.5, 1.0])),
                uniform_scale = bool(rng.integers(2)),
            )
        else:  # BiPCA
            cfg = dict(
                dim_hid    = dim_hid,
                sparse_reg = float(rng.choice([0.0, 0.1, 0.5, 1.0])),
                tree_reg   = float(rng.choice([0.01, 0.1, 0.5, 1.0])),
            )

        try:
            Xp = preprocess_X(X, mask, **prep)
        except Exception:
            continue

        trial_seed = int(rng.integers(2**31))
        best_S, mean_sc, std_sc, best_seed = _fit_n(
            Xp, tv, model_class, cfg, _QUICK, n_runs, base_seed=trial_seed)
        if best_S is None:
            continue

        results.append(dict(
            area        = this_area,
            model       = model_class.__name__,
            model_class = model_class,
            **cfg,
            **prep,
            mean_score  = mean_sc,
            std_score   = std_sc,
            best_S      = best_S,
            seed        = best_seed,
        ))

    results.sort(key=lambda r: r['mean_score'], reverse=True)
    return results


def focused_search(X, conds, area, top_results, n_runs=8, seed=0):
    """Refit top configs with full annealing schedule and more random seeds."""
    rng       = np.random.default_rng(seed)
    tv        = task_regressors(conds)
    prep_keys = ['center_neurons', 'zscore_neurons', 'center_conditions',
                 'zscore_conditions', 'clip_percentile', 'global_scale']
    cfg_keys  = ['dim_hid', 'nonneg', 'sparse_reg', 'tree_reg',
                 'weight_pr_reg', 'weight_l1_reg', 'uniform_scale']
    refined = []

    for r in tqdm(top_results, desc='focused search'):
        mask = (area == r['area'])
        prep = {k: r[k] for k in prep_keys if k in r}
        cfg  = {k: r[k] for k in cfg_keys  if k in r}
        try:
            Xp = preprocess_X(X, mask, **prep)
        except Exception:
            continue
        trial_seed = int(rng.integers(2**31))
        best_S, mean_sc, std_sc, best_seed = _fit_n(
            Xp, tv, r['model_class'], cfg, _FULL, n_runs, base_seed=trial_seed)
        if best_S is None:
            continue
        refined.append({**r, 'mean_score': mean_sc,
                        'std_score': std_sc, 'best_S': best_S,
                        'seed': best_seed})

    refined.sort(key=lambda r: r['mean_score'], reverse=True)
    return refined


def semibmf_search(X, conds, area, n_trials=50, n_runs=5, seed=0):
    """
    Random search for SemiBMF with the full annealing schedule.

    SemiBMF was absent from broad_search results because the quick schedule
    (300 iters) is too short for it to converge.  Key constraint: do NOT
    center neurons — the existing working example in hunt_data.py uses only
    std-normalization, which preserves the mean firing rate that nonneg latents
    rely on to represent above-baseline (stimulus-specific) activations.
    """
    rng     = np.random.default_rng(seed)
    tv      = task_regressors(conds)
    results = []

    for _ in tqdm(range(n_trials), desc='SemiBMF search'):
        this_area = ['OFC', 'ACC', 'DLPFC'][int(rng.integers(3))]
        mask = (area == this_area)
        if mask.sum() < 5:
            continue

        prep = dict(
            center_neurons    = False,           # preserve mean for nonneg latents
            zscore_neurons    = True,
            center_conditions = bool(rng.integers(2)),
            zscore_conditions = True,
            clip_percentile   = None,
            global_scale      = True,
        )

        dim_hid = int(rng.choice([3, 4, 5]))

        cfg = dict(
            dim_hid       = dim_hid,
            nonneg        = True,
            sparse_reg    = float(rng.choice([0.1, 0.5, 1.0, 2.0, 5.0])),
            tree_reg      = float(rng.choice([0.0, 0.01, 0.1, 0.5, 1.0])),
            weight_pr_reg = float(rng.choice([0.1, 0.5, 1.0, 5.0])),
            weight_l1_reg = float(rng.choice([0.0, 0.01, 0.05, 0.1])),
            scl_lr        = float(rng.choice([1e-4, 5e-4, 1e-3])),
        )

        try:
            Xp = preprocess_X(X, mask, **prep)
        except Exception:
            continue

        trial_seed = int(rng.integers(2**31))
        best_S, mean_sc, std_sc, best_seed = _fit_n(
            Xp, tv, old_bae_models.SemiBMF, cfg, _FULL, n_runs, base_seed=trial_seed)
        if best_S is None:
            continue

        results.append(dict(
            area        = this_area,
            model       = 'SemiBMF',
            model_class = old_bae_models.SemiBMF,
            **cfg,
            **prep,
            mean_score  = mean_sc,
            std_score   = std_sc,
            best_S      = best_S,
            seed        = best_seed,
        ))

    results.sort(key=lambda r: r['mean_score'], reverse=True)
    return results


def stimulus_score(S, tv):
    """
    Alternative to interp_score that rewards sparse, stimulus-like latents.
    For each latent, the best R² among *exact-rank* regressors (r==k|group)
    counts double vs. the global mean used by interp_score.
    """
    stim_keys = {k for k in tv if k.startswith('r==') or k.startswith('rank==')}
    K = S.shape[1]
    per_k = np.zeros(K)
    for k in range(K):
        s = S[:, k].astype(float)
        s -= s.mean()
        if np.abs(s).max() < 1e-9:
            continue
        s /= s.std()
        best_all  = max(float(np.corrcoef(s, v)[0, 1]**2) for v in tv.values())
        best_stim = max(
            (float(np.corrcoef(s, tv[n])[0, 1]**2) for n in stim_keys),
            default=0.0)
        per_k[k] = 0.5 * best_all + 0.5 * best_stim
    return float(per_k.mean())


def stimulus_search(X, conds, area, n_trials=60, n_runs=5, seed=0):
    """
    SemiBMF search tuned to find stimulus-specific (sparse, rank-exact) features.

    Key differences from semibmf_search:
      - tree_reg always 0 (no penalty on single-rank bumps)
      - higher sparse_reg (encourages few active conditions per latent)
      - scoring uses stimulus_score (rewards exact-rank regressors)
      - K restricted to 2–4 (small K forces representation of rare patterns)
    """
    rng     = np.random.default_rng(seed)
    tv      = task_regressors(conds)
    results = []

    for _ in tqdm(range(n_trials), desc='stimulus search'):
        this_area = ['OFC', 'ACC', 'DLPFC'][int(rng.integers(3))]
        mask = (area == this_area)
        if mask.sum() < 5:
            continue

        prep = dict(
            center_neurons    = False,
            zscore_neurons    = True,
            center_conditions = bool(rng.integers(2)),
            zscore_conditions = True,
            clip_percentile   = None,
            global_scale      = True,
        )

        dim_hid = int(rng.choice([2, 3, 4]))

        cfg = dict(
            dim_hid       = dim_hid,
            nonneg        = True,
            sparse_reg    = float(rng.choice([1.0, 2.0, 5.0, 10.0])),
            tree_reg      = 0.0,
            weight_pr_reg = float(rng.choice([0.1, 0.5, 1.0, 5.0])),
            weight_l1_reg = float(rng.choice([0.0, 0.01, 0.05, 0.1])),
            scl_lr        = float(rng.choice([1e-4, 5e-4, 1e-3])),
        )

        try:
            Xp = preprocess_X(X, mask, **prep)
        except Exception:
            continue

        trial_seed = int(rng.integers(2**31))
        scores, best_S, best_sc, best_seed = [], None, -1.0, None
        for i in range(n_runs):
            seed_i = trial_seed + i
            np.random.seed(seed_i)
            try:
                mod = _make_and_fit(Xp, old_bae_models.SemiBMF, cfg, _FULL)
                S   = _get_S(mod)
                sc  = stimulus_score(S, tv)
                scores.append(sc)
                if sc > best_sc:
                    best_sc, best_S, best_seed = sc, S, seed_i
            except Exception:
                pass

        if not scores:
            continue

        results.append(dict(
            area        = this_area,
            model       = 'SemiBMF',
            model_class = old_bae_models.SemiBMF,
            **cfg,
            **prep,
            mean_score  = float(np.mean(scores)),
            std_score   = float(np.std(scores)),
            best_S      = best_S,
            seed        = best_seed,
        ))

    results.sort(key=lambda r: r['mean_score'], reverse=True)
    return results


def spikenmf_search(X, conds, area, n_trials=60, n_runs=5, seed=0):
    """
    SpikeNMF search targeting stimulus-specific features in OFC.

    SpikeNMF differs from SemiBMF in two key ways:
      1. Spike-and-slab prior: continuous Z multiplier absorbs per-condition
         magnitude variation (e.g., same stimulus fires at different rates
         when presented left vs right), which the binary-only SemiBMF cannot.
      2. Dead weight recovery: W columns that collapse to zero are randomly
         reinitialized, preventing latent collapse.

    Scored with stimulus_score to reward rank-exact patterns.
    mod.S stores S*Z (continuous); binarized via _get_S before scoring/storage.
    """
    rng     = np.random.default_rng(seed)
    tv      = task_regressors(conds)
    results = []

    for _ in tqdm(range(n_trials), desc='SpikeNMF search'):
        this_area = ['OFC', 'ACC', 'DLPFC'][int(rng.integers(3))]
        mask = (area == this_area)
        if mask.sum() < 5:
            continue

        prep = dict(
            center_neurons    = False,
            zscore_neurons    = True,
            center_conditions = bool(rng.integers(2)),
            zscore_conditions = True,
            clip_percentile   = None,
            global_scale      = True,
        )

        dim_hid = int(rng.choice([3, 4, 5]))

        cfg = dict(
            dim_hid       = dim_hid,
            nonneg        = True,
            sparse_reg    = float(rng.choice([0.005, 0.01, 0.05, 0.1, 0.5])),
            tree_reg      = float(rng.choice([0.0, 0.05, 0.1, 0.5])),
            weight_pr_reg = float(rng.choice([0.1, 0.5, 1.0, 5.0])),
            weight_l1_reg = 0.0,
            weight_l2_reg = float(rng.choice([0.0, 0.05, 0.1, 0.5])),
            slab_prior    = float(rng.choice([0.5, 1.0, 2.0])),
        )

        try:
            Xp = preprocess_X(X, mask, **prep)
        except Exception:
            continue

        trial_seed = int(rng.integers(2**31))
        best_S, mean_sc, std_sc, best_seed = _fit_n(
            Xp, tv, old_bae_models.SpikeNMF, cfg, _FULL, n_runs,
            base_seed=trial_seed, score_fn=stimulus_score)
        if best_S is None:
            continue

        results.append(dict(
            area        = this_area,
            model       = 'SpikeNMF',
            model_class = old_bae_models.SpikeNMF,
            **cfg,
            **prep,
            mean_score  = mean_sc,
            std_score   = std_sc,
            best_S      = best_S,
            seed        = best_seed,
        ))

    results.sort(key=lambda r: r['mean_score'], reverse=True)
    return results


def describe_latents(S, conds):
    """
    Print a compact text view of each latent as a binary string per
    (side×attr) group, annotated with top correlates.

    Example output:
      L_Pr  L_Mg  R_Pr  R_Mg   top correlates
      Lat 0: 00111  00011  00001  00000   rank>=3(0.91) rk|L_Pr(0.88)
      Lat 1: 11111  00000  11111  00000   attr(1.00)
    """
    tv = task_regressors(conds)
    K  = S.shape[1]

    groups = [
        ('L_Pr', (conds['side'] == 'L') & (conds['attribute'] == 'Prob')),
        ('L_Mg', (conds['side'] == 'L') & (conds['attribute'] == 'Mag')),
        ('R_Pr', (conds['side'] == 'R') & (conds['attribute'] == 'Prob')),
        ('R_Mg', (conds['side'] == 'R') & (conds['attribute'] == 'Mag')),
    ]

    # header
    print('         ' + '  '.join(f'{lbl}' for lbl, _ in groups)
          + '   top correlates')

    for k in range(K):
        parts = [f'  Lat {k}: ']
        for _, mask in groups:
            sort_idx = np.argsort(conds['rank'][mask])
            vals = S[mask][sort_idx, k].astype(int)
            parts.append(''.join(str(v) for v in vals))

        s = S[:, k].astype(float) - S[:, k].mean()
        if np.abs(s).max() > 1e-9:
            s /= s.std()
            cors = {n: float(np.corrcoef(s, v)[0, 1]**2)
                    for n, v in tv.items()}
            top3 = sorted(cors.items(), key=lambda x: -x[1])[:3]
            ann = '  '.join(f'{n}({r2:.2f})' for n, r2 in top3 if r2 > 0.05)
        else:
            ann = '(dead)'

        print('  '.join(parts) + '   ' + ann)


def describe_result(r, conds, rank=None):
    """Print a one-block summary of a search result including latent patterns."""
    prefix = f'[{rank}] ' if rank is not None else ''
    seed_str = f'  seed={r["seed"]}' if r.get('seed') is not None else ''
    print(f"{prefix}{r['model']:<12} area={r['area']}  K={r['dim_hid']}  "
          f"score={r['mean_score']:.3f}±{r['std_score']:.3f}{seed_str}")

    print(f"  prep: cN={r['center_neurons']} zN={r['zscore_neurons']} "
          f"cC={r['center_conditions']} zC={r['zscore_conditions']} "
          f"clip={r['clip_percentile']}")

    if r['model'] == 'SemiBMF':
        print(f"  hyp: nonneg={r['nonneg']}  sp={r['sparse_reg']:.2f}  "
              f"tree={r['tree_reg']:.2f}  wpr={r['weight_pr_reg']:.2f}  "
              f"wl1={r['weight_l1_reg']:.2f}  lr={r.get('scl_lr', 1e-3):.0e}")
    elif r['model'] == 'KernelBMF2':
        print(f"  hyp: sp={r['sparse_reg']:.2f}  tree={r['tree_reg']:.2f}  "
              f"uniform_scale={r['uniform_scale']}")
    elif r['model'] == 'BiPCA':
        print(f"  hyp: sp={r['sparse_reg']:.2f}  tree={r['tree_reg']:.2f}")
    elif r['model'] == 'SpikeNMF':
        print(f"  hyp: nonneg={r.get('nonneg', True)}  sp={r['sparse_reg']:.3f}  "
              f"tree={r['tree_reg']:.2f}  wpr={r['weight_pr_reg']:.2f}  "
              f"wl2={r.get('weight_l2_reg', 0.1):.2f}  "
              f"slab={r.get('slab_prior', 1.0):.2f}")

    describe_latents(r['best_S'], conds)


def plot_latents(result, conds, title_prefix=''):
    """
    Two-panel figure:
      Left  — binary latent matrix (conditions sorted by side/attr/rank)
      Right — |correlation| with each task variable
    """
    S    = result['best_S']
    tv   = task_regressors(conds)
    C, K = S.shape

    # Sort: L < R outer, Prob < Mag middle, rank 1→5 inner
    order = np.lexsort([conds['rank'],
                        (conds['attribute'] == 'Mag').astype(int),
                        (conds['side'] == 'R').astype(int)])

    tv_keys = list(tv.keys())
    cors = np.zeros((K, len(tv_keys)))
    for k in range(K):
        s = S[:, k].astype(float)
        s -= s.mean()
        if np.abs(s).max() < 1e-9:
            continue
        s /= s.std()
        for j, v in enumerate(tv.values()):
            cors[k, j] = float(np.corrcoef(s, v)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(15, max(3, K * 0.6 + 1.5)),
                              gridspec_kw={'width_ratios': [1, 3]})

    ax = axes[0]
    ax.imshow(S[order].T, aspect='auto', cmap='Blues',
              interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Condition  (L_Pr · L_Mg · R_Pr · R_Mg, each rank 1→5)')
    ax.set_ylabel('Latent')
    ax.set_xticks(range(C))
    ax.set_xticklabels([conds['label'][i] for i in order],
                        rotation=90, fontsize=6)
    ax.set_title(f"{title_prefix}{result['model']}  area={result['area']}"
                 f"  K={result['dim_hid']}\n"
                 f"score={result['mean_score']:.3f}±{result['std_score']:.3f}",
                 fontsize=9)

    ax = axes[1]
    im = ax.imshow(np.abs(cors), aspect='auto', cmap='hot_r',
                   vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(len(tv_keys)))
    ax.set_xticklabels(tv_keys, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(K))
    ax.set_ylabel('Latent')
    ax.set_title('|Correlation| with task variables', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02)

    plt.tight_layout()
    return fig


#%%
# --- Phase 1: broad random search ---
broad_results = broad_search(X, conds, area, n_trials=300, n_runs=3, seed=42)

print('\n=== Top 15 (broad search, quick schedule) ===\n')
for i, r in enumerate(broad_results[:15]):
    describe_result(r, conds, rank=i+1)
    print()

#%%
# --- Phase 2: focused refinement of top 20 ---
refined = focused_search(X, conds, area, broad_results[:20], n_runs=8)

print('\n=== Top 10 (focused, full schedule) ===\n')
for i, r in enumerate(refined[:10]):
    describe_result(r, conds, rank=i+1)
    print()

#%%
# --- Visualize top 4 ---
for i, r in enumerate(refined[:4]):
    fig = plot_latents(r, conds, title_prefix=f'Rank {i+1}: ')
    plt.show()

#%%

