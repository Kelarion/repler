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
import bae
import bae_models
import bae_search
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
        
#%%

this_area = 'OFC'
# this_area = 'ACC'
# this_area = 'DLPFC'

X_ = X[:,area==this_area] / X[:,area==this_area].std(0, keepdims=True)
X_ = X_ / X_.std(1, keepdims=True)

# X_ -= X_.mean(0)

mod = bae_models.SemiBMF(7,
                         nonneg=True, 
                         tree_reg=1e-1,
                         weight_pr_reg=1,
                         weight_l2_reg=0,
                         weight_l1_reg=1e-2,
                         sparse_reg=1,
                         )

# mod = bae_models.SpikeNMF(3,
#                          nonneg=True, 
#                          sparse_reg=1,
#                          weight_pr_reg=1e-2,
#                          tree_reg=1e-1,
#                          weight_l2_reg=0,
#                          weight_l1_reg=1e-1,
#                          )

# mod = bae_models.KernelBMF2(5,
#                            sparse_reg=1,
#                            tree_reg=100,
#                            uniform_scale=True,
#                            # l1_reg=0,
#                            )

en = mod.fit(X_ / X_.std(), 
             period=50, 
             initial_temp=100,
             decay_rate=0.9, 
             min_temp=1, 
             scl_lr=1e-3,
             )

samps = mod.sample(X_ / X_.std(), n_samp=1000)

samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.imshow(samps.mean(0))

#%%


