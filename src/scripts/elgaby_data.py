CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/elgaby_data/Data/'

import os, sys, re, glob, hashlib
from collections import defaultdict
sys.path.append(CODE_DIR)

import numpy as np
import pandas as pd
import pickle as pkl

import torch

import matplotlib.pyplot as plt

# my code
import util
import old_bae
import old_bae_models
import old_bae_search
import bae_util
import plotting as tpl

import bae_models

#%%
# ----------------------------------------------------------------------------
# El-Gaby et al. (2024) Nature -- "A cellular basis for mapping behavioural
# structure" (s41586-024-08145-x). Mice run a 3x3 maze (nodes 1-9, graph in
# MetaData/Edge_grid.npy); 4 reward ports A->B->C->D must be visited in a fixed
# cyclic order. The rewarded nodes change between tasks. Recordings are from
# medial frontal cortex (Neuropixels / Cambridge NeuroTech), Kilosort 2.5/3.
#
# Per-file naming:  <var>_<mouse>_<date>[ _<date2> ]_<task_idx>.npy
#   single day  = ab03_01092023          (~3 tasks/day)
#   double day  = ab03_01092023_02092023 (two days spike-sorted together by
#                 concatenation -> the SAME neurons tracked across up to 6 tasks;
#                 this is the only across-day neuron matching in the release).
#
# On disk (Data/):
#   Neuronal_activity/Awake/Neuron_raw_<key>_<i>.npy : (n_neurons, n_bins),
#       spike counts in 25 ms bins, starting at the first reward-A of the task.
#   Maze location/Location_raw_<key>_<i>.npy : (n_bins,) maze position --
#       1-9 = nodes, 10-21 = the 12 edges (in transit), NaN = untracked.
#   Trial_times/trialtimes_<key>_<i>.npy : (n_trials, n_states+1), reward-arrival
#       times in MILLISECONDS -> // 25 to get 25 ms bin indices. For ABCD the 5
#       columns are arrival bins of [A, B, C, D, A_next]; col4==col0 of next row.
#   Tasks/Task_data_<key>.npy : reward-node sequence per task (often pickled with
#       a custom class and unreadable -- we derive nodes from trialtimes+Location).
#
# Representation built here (option discussed with author):
#   condition = (task state in {A,B,C,D}) x (reward-port node in 1..9) = 36 rows,
#   value     = goal-progress-normalised firing profile of the state's leg
#               (90 bins; bin 0 = just after the previous reward, bin 89 = this
#               reward reached). Pooled into a 36 x n_neuron x 90 pseudopopulation
#               across all sessions/animals, NaN where a neuron never had that
#               reward at that node. Average over the time axis -> 36 x n_neuron.
#
# A reward port is fixed within a task, so each task fills exactly 4 of the 36
# rows for its neurons; with <=6 tasks/neuron the matrix is intrinsically sparse
# (a reward slot only ever visits <=6 of 9 nodes) -- NaNs are expected and are
# handled natively by the impute-CV / masked BMF tools (bae_util.impcv).
# ----------------------------------------------------------------------------

STATES = ['A', 'B', 'C', 'D']          # task states / reward identities
NODES = np.arange(1, 10)               # 3x3 maze nodes
N_BINS = 90                            # goal-progress bins per state leg
BIN_MS = 25                            # ms per neuronal bin

# leg whose arrival is column `k` belongs to which state index (A=0..D=3):
#   col1->B, col2->C, col3->D, col4->A(next)
_END_COL_TO_STATE = {1: 1, 2: 2, 3: 3, 4: 0}


#%%
# ----------------------------------- helpers --------------------------------

def list_sessions(load_dir=LOAD_DIR):
    """Map recording-group key -> sorted list of task indices, from Neuron files.
    key is the file stem minus the trailing _<task_idx> (so it keeps the
    mouse and one or two dates)."""
    groups = defaultdict(set)
    for f in glob.glob(os.path.join(load_dir, 'Neuronal_activity', 'Awake',
                                    'Neuron_raw_*.npy')):
        # some files have a stray trailing underscore (..._0_.npy) shared across
        # the Neuron/Trial_times/Location dirs -- rstrip it before parsing.
        stem = os.path.basename(f)[len('Neuron_raw_'):-4].rstrip('_')
        parts = stem.split('_')
        groups['_'.join(parts[:-1])].add(parts[-1])
    return {k: sorted(v, key=int) for k, v in groups.items()}


def is_double(key):
    """True for a concatenated two-day key (mouse_date1_date2)."""
    return key.count('_') >= 2


def dedup_groups(groups):
    """One sort per physical neuron: prefer the double-day (concatenated) file,
    and keep a single-day file only for dates not covered by any double day.
    Takes and returns a {key: idxs} dict."""
    covered = set()                       # (mouse, date) handled by a double day
    for key in groups:
        if is_double(key):
            mouse, d1, d2 = key.split('_')
            covered |= {(mouse, d1), (mouse, d2)}
    keep = {}
    for key, idxs in groups.items():
        if is_double(key) or tuple(key.split('_')) not in covered:
            keep[key] = idxs
    return dict(sorted(keep.items()))


def _find(load_dir, subdir, prefix, key, idx):
    """Locate a file, tolerating the stray-trailing-underscore variant."""
    for cand in (f'{prefix}_{key}_{idx}.npy', f'{prefix}_{key}_{idx}_.npy'):
        p = os.path.join(load_dir, subdir, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{prefix}_{key}_{idx}')


def load_session(key, idx, load_dir=LOAD_DIR):
    """Return (neuron_raw (n_neurons, n_bins), trial_bins (n_trials, ncol),
    location (n_bins,)) for one task file. trial_bins already // 25."""
    nr = np.load(_find(load_dir, 'Neuronal_activity/Awake', 'Neuron_raw', key, idx))
    tt = np.load(_find(load_dir, 'Trial_times', 'trialtimes', key, idx))
    loc = np.load(_find(load_dir, 'Maze location', 'Location_raw', key, idx))
    return nr, tt // BIN_MS, loc


def _mode_node(loc, b):
    """Most common maze node (1-9) in a +/-2 bin window around bin b; None if untracked."""
    w = loc[max(0, b - 2):b + 3]
    w = w[(w >= 1) & (w <= 9)]
    if not len(w):
        return None
    v, c = np.unique(w.astype(int), return_counts=True)
    return int(v[np.argmax(c)])


def reward_nodes(tt, loc):
    """Fixed port node of each state A,B,C,D for one task (mode over trials).
    Returns dict state_idx -> node or None."""
    out = {}
    for s in range(4):                    # arrival columns 0..3 == A,B,C,D
        nodes = [_mode_node(loc, int(x)) for x in tt[:, s] if 0 <= int(x) < len(loc)]
        nodes = [n for n in nodes if n is not None]
        if nodes:
            v, c = np.unique(nodes, return_counts=True)
            out[s] = int(v[np.argmax(c)])
        else:
            out[s] = None
    return out


def normalise_block(seg, num_bins=N_BINS):
    """Time-warp a leg to num_bins via goal-progress, vectorised over neurons.
    seg: (n_neurons, L) spike counts -> (n_neurons, num_bins) mean per bin.
    Mirrors the paper's normalise(): legs shorter than num_bins are upsampled
    10x first so every bin is populated."""
    L = seg.shape[1]
    if L < num_bins:
        seg = np.repeat(seg, 10, axis=1) / 10.0
        L = seg.shape[1]
    idx = np.minimum((np.arange(L) * num_bins // L), num_bins - 1)   # bin of each sample
    starts = np.searchsorted(idx, np.arange(num_bins))               # idx is non-decreasing
    sums = np.add.reduceat(seg, starts, axis=1)
    counts = np.diff(np.append(starts, L))
    return sums / counts[None, :]


def _leg_path(loc_seg):
    """Sequence of maze NODES (1-9) traversed in a leg, in order, with consecutive
    repeats collapsed. Edges (codes 10-21, 'in transit') and untracked (NaN) bins
    are skipped, so this is the route of grid cells the animal actually occupied
    from the previous reward to this one (first entry = start node near the last
    reward, last entry = the node of the reward being approached)."""
    loc_seg = np.asarray(loc_seg)
    nodes = loc_seg[(loc_seg >= 1) & (loc_seg <= 9)].astype(int)
    if not len(nodes):
        return np.empty(0, dtype=int)
    keep = np.ones(len(nodes), dtype=bool)
    keep[1:] = nodes[1:] != nodes[:-1]
    return nodes[keep]


#%%
# --------------------------- per-session loading ----------------------------

def warp_session(nr, tt, num_bins=N_BINS, loc=None, with_traj=False):
    """Goal-progress-warp every reward leg of one task (NOT trial-averaged).

    Same segmentation as the paper's `raw_to_norm` (Basic_analysis.ipynb) -- each
    trial is cut at its reward-arrival times into n_states legs and each leg is
    time-warped to `num_bins` goal-progress bins -- but the legs are kept
    individually instead of being averaged over trials. Legs come out in
    chronological order (trial-major, then task order A->B, B->C, C->D, D->A).

    nr : (n_neurons, n_bins) raw 25 ms spike counts (Neuron_raw).
    tt : (n_trials, n_states+1) reward-arrival BIN indices (already // 25); leg j
         of trial ti is the segment tt[ti, j] -> tt[ti, j+1].
    loc: (n_bins,) Location_raw for the same task (1-9 nodes, 10-21 edges, NaN);
         only needed when with_traj=True.
    with_traj: also return the un-warped duration and grid-node path of each leg.

    Returns
    -------
    legs   : (n_reward, num_bins, n_neurons) -- one row per individual reward leg.
    states : (n_reward,) int, each leg's state index (0..n_states-1) i.e. which
             reward it approaches; legs that can't be cut (bad times) are dropped,
             so use this rather than assuming a fixed cycle.
    durations : (n_reward,) float -- only when with_traj=True; the real (un-warped)
             time of each leg in MILLISECONDS (= raw bins spanned x 25 ms), i.e. how
             long the animal took to get from the previous reward to this one.
    paths  : (n_reward,) object -- only when with_traj=True; each entry is the
             ordered sequence of maze nodes (1-9) traversed in that leg (see
             `_leg_path`).
    """
    n_states = tt.shape[1] - 1
    legs, states, durations, paths = [], [], [], []
    for ti in range(tt.shape[0]):
        for j in range(n_states):
            a, e = int(tt[ti, j]), int(tt[ti, j + 1])
            if e <= a or a < 0 or e > nr.shape[1]:
                continue
            legs.append(normalise_block(nr[:, a:e], num_bins))   # (n_neuron, 90)
            states.append(j)
            if with_traj:
                durations.append((e - a) * BIN_MS)               # ms
                paths.append(_leg_path(loc[a:e]) if loc is not None
                             else np.empty(0, dtype=int))
    if not legs:
        out = (np.empty((0, num_bins, nr.shape[0])), np.empty(0, dtype=int))
        if with_traj:
            out += (np.empty(0), np.empty(0, dtype=object))
        return out
    out = (np.stack(legs).transpose(0, 2, 1), np.array(states, dtype=int))
    if with_traj:
        parr = np.empty(len(paths), dtype=object)
        parr[:] = paths                                          # 1-D object array
        out += (np.array(durations, dtype=float), parr)
    return out


def load_all_sessions(load_dir=LOAD_DIR, n_states=None, min_trials=2,
                      with_ids=False, verbose=False):
    """Load every individual task recording as an (n_reward, 90, n_neuron) array
    of goal-progress-warped reward legs (NOT trial-averaged).

    Iterates ALL Neuron_raw files -- every (group, task) file is one session. The
    single- and double-day sorts of a date are BOTH present here, so the same
    physical neuron can recur across sessions; wrap `list_sessions` in
    `dedup_groups` (or filter the returned ids) if you want one sort per neuron.

    Parameters
    ----------
    n_states  : keep only sessions with this many reward legs per trial (4 = ABCD,
                5 = ABCDE, 2 = the 2-reward control); None keeps all.
    min_trials: skip sessions with fewer than this many trials.
    with_ids  : also return parallel lists of (key, idx) ids and per-leg state
                indices.

    Returns
    -------
    sessions : list of (n_reward, 90, n_neuron) float arrays, one per session
               (n_reward = number of reward legs in that session).
    ids      : list of (key, idx)            -- only when with_ids=True.
    leg_states : list of (n_reward,) int arrays giving each leg's state index
               -- only when with_ids=True.
    """
    sessions, ids, leg_states = [], [], []
    for key, idxs in list_sessions(load_dir).items():
        for idx in idxs:
            try:
                nr, tt, _ = load_session(key, idx, load_dir)
            except FileNotFoundError:
                continue                         # missing Location/trialtimes
            if tt.ndim != 2 or tt.shape[0] < min_trials or tt.shape[1] < 2:
                continue
            if n_states is not None and tt.shape[1] - 1 != n_states:
                continue
            legs, st = warp_session(nr, tt)
            if len(legs) == 0:
                continue
            sessions.append(legs)
            ids.append((key, idx))
            leg_states.append(st)
            if verbose:
                print(f'{key}_{idx}: {legs.shape}')
    return (sessions, ids, leg_states) if with_ids else sessions


def _uf_find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:           # path compression
        parent[x], x = root, parent[x]
    return root


def load_combined_legs(load_dir=LOAD_DIR, animals=None, n_states=None,
                       min_trials=2, with_meta=True, verbose=True):
    """All reward legs of each PHYSICALLY-DISTINCT sorted population, combined into
    as few arrays as possible with exact duplicates removed.

    Why content-based: a single-day file and its `_<d1>_<d2>` double-day file store
    BYTE-IDENTICAL task arrays when they are the same sort, but NOT always -- e.g.
    me11's 05/06 double (46 units) matches day-6 only, not the standalone day-5
    sort (33 units); the 07/08 double (24 units) matches neither single (54, 46).
    So neuron identity is decided by actual spike content, not by filenames:
    keys are merged (union-find) iff they share a byte-identical `Neuron_raw`
    array, which also guarantees the SAME neuron row-order across them. Within each
    merged population every distinct task is taken once (exact-duplicate arrays --
    the single-vs-double redundancy -- dropped), then all its legs are warped
    (`warp_session`) and stacked. Sorts that happen to be re-derived from the same
    raw recording (e.g. day-5's 33- and 46-unit versions) are different unit sets,
    so they remain separate populations -- that is not double-counting.

    Parameters
    ----------
    animals : keep only these mice (e.g. ['me11']); None = all. The full set loads
              every Neuron_raw file once to fingerprint it (~GBs of IO); filter by
              animal to work incrementally.
    n_states: keep only tasks with this many reward legs (4 = ABCD, 5 = ABCDE,
              2 = control); None mixes all task types for a population.
    min_trials: skip tasks with fewer than this many trials.
    with_meta: also return a list of per-population metadata dicts.

    Returns
    -------
    pops : list of (n_legs, 90, n_neuron) float arrays, one per distinct population
           (neuron = column, constant identity down the array).
    meta : list of dicts {keys, n_neuron, n_legs, leg_state, leg_session,
           leg_duration, leg_path}, only when with_meta=True. Per-leg, in array
           order: `leg_state` = state index approached; `leg_session` = source
           (key, idx); `leg_duration` = real inter-reward time in MILLISECONDS
           (un-warped); `leg_path` = object array of the maze nodes (1-9) traversed
           from the previous reward to this one (consecutive repeats collapsed).
    """
    groups = list_sessions(load_dir)
    if animals is not None:
        animals = set(animals)
        groups = {k: v for k, v in groups.items() if k.split('_')[0] in animals}

    parent = {}                                   # union-find over keys
    seen_hash_key = {}                            # task hash -> first key seen
    hash_legs = {}                                # task hash -> (legs, states)
    hash_rep = {}                                 # task hash -> (key, idx) source
    key_hashes = defaultdict(list)               # key -> its task hashes
    n_files = 0
    for key, idxs in groups.items():
        parent.setdefault(key, key)
        for idx in idxs:
            try:
                nr, tt, loc = load_session(key, idx, load_dir)
            except FileNotFoundError:
                continue
            if tt.ndim != 2 or tt.shape[0] < min_trials or tt.shape[1] < 2:
                continue
            if n_states is not None and tt.shape[1] - 1 != n_states:
                continue
            n_files += 1
            h = hashlib.sha1(np.ascontiguousarray(nr).tobytes()).digest()
            key_hashes[key].append(h)
            if h in seen_hash_key:                 # identical task -> same sort
                ra, rb = _uf_find(parent, key), _uf_find(parent, seen_hash_key[h])
                parent[ra] = rb
            else:
                seen_hash_key[h] = key
                hash_legs[h] = warp_session(nr, tt, loc=loc, with_traj=True)
                hash_rep[h] = (key, idx)
            if verbose and n_files % 50 == 0:
                print(f'  fingerprinted {n_files} files...')

    comps = defaultdict(list)                      # root key -> member keys
    for key in key_hashes:
        comps[_uf_find(parent, key)].append(key)

    pops, meta = [], []
    for root in sorted(comps):
        uniq = {h for key in comps[root] for h in key_hashes[key]}     # distinct tasks
        # chronological-ish order: by source (key, idx)
        uniq = sorted(uniq, key=lambda h: (hash_rep[h][0], int(hash_rep[h][1])))
        legs = np.concatenate([hash_legs[h][0] for h in uniq], axis=0)
        states = np.concatenate([hash_legs[h][1] for h in uniq])
        durations = np.concatenate([hash_legs[h][2] for h in uniq])
        paths = np.concatenate([hash_legs[h][3] for h in uniq])   # 1-D object arrays
        sess = np.array([hash_rep[h] for h in uniq for _ in range(len(hash_legs[h][1]))],
                        dtype=object)
        pops.append(legs)
        meta.append(dict(keys=sorted(comps[root]), n_neuron=legs.shape[2],
                         n_legs=legs.shape[0], leg_state=states, leg_session=sess,
                         leg_duration=durations, leg_path=paths))
        if verbose:
            print(f'[{len(pops)}] {sorted(comps[root])[0]} (+{len(comps[root])-1}): '
                  f'{legs.shape[2]} neurons, {legs.shape[0]} legs, '
                  f'{len(uniq)} tasks')

    return (pops, meta) if with_meta else pops


#%%
# --------------------- Figure-5 anchoring-GLM regressors --------------------
# Reconstructs the design matrix of El-Gaby Fig 5g (Code/Figure5_Regression.ipynb,
# cells 15+21): for every 25 ms time bin, a (location x phase x task-lag) tensor.
#   location = 9 maze nodes  |  phase = 3 thirds of goal progress within a state
#   task-lag = num_states*num_phases (=12 for ABCD): #phase-steps since the animal
#              was last at that node in that phase.
# A per-(node,phase) "activity bump" is seeded at lag 0 when the animal is at that
# node in that phase, and advances one lag every phase change -- so a cell tuned to
# (node, phase, lag) fires at a fixed task-lag after passing a node at a phase.
# The regression target is the RAW spike count per bin (Poisson GLM); time is
# continuous (all trials concatenated), NOT trial-averaged or goal-progress-warped.


def _phase_state_arrays(tt, n_bins, num_phases=3):
    """Per-bin task phase (0..num_phases-1) and state index over the trial span.

    The span is [tt[0,0], tt[-1,-1]) in 25 ms bins (the first reward to the last).
    Each leg tt[ti,j]->tt[ti,j+1] is split into `num_phases` equal-progress thirds;
    within a leg the paper's goal-progress is linear in time, so equal-time thirds
    reproduce its phase (matches cell 26's `phase_norm_mean`). Bins not inside any
    leg are left -1.

    Returns (start, end, phase, state), phase/state length end-start.
    """
    tt = np.asarray(tt)
    n_states = tt.shape[1] - 1
    start = int(tt[0, 0])
    end = min(int(tt[-1, -1]), int(n_bins))
    T = max(0, end - start)
    phase = np.full(T, -1, dtype=int)
    state = np.full(T, -1, dtype=int)
    for ti in range(tt.shape[0]):
        for j in range(n_states):
            a, e = max(int(tt[ti, j]), start), min(int(tt[ti, j + 1]), end)
            if e <= a:
                continue
            rel = np.arange(e - a)
            phase[a - start:e - start] = np.minimum(rel * num_phases // (e - a),
                                                    num_phases - 1)
            state[a - start:e - start] = j
    return start, end, phase, state


def _anchor_bump(nodes, phase, num_nodes=9, num_phases=3, num_lags=12):
    """Activity-bump regressor construction (port of Figure5_Regression cell 15,
    multiple_bumps=True), vectorised over the 9x3 modules.

    nodes : (T,) maze-node index 0..num_nodes-1, or -1 where edge/untracked.
    phase : (T,) task phase 0..num_phases-1.
    Returns regressors (T, num_nodes, num_phases, num_lags).
    """
    T = len(nodes)
    M = np.zeros((num_nodes, num_phases, num_lags))
    out = np.zeros((T, num_nodes, num_phases, num_lags))
    prev_phase, prev_node = -1, -2
    for t in range(T):
        nd, ph = int(nodes[t]), int(phase[t])
        valid = 0 <= nd < num_nodes
        if ph != prev_phase:                       # move_phase: advance every bump
            if valid:
                M[nd, ph, 0] = 1.0                 # spatial/phase input at lag 0
            M[:] = np.roll(M, 1, axis=2)
            M[:, :, 1] = 0.0                       # lag-1 kept only if re-driven now
            if valid:
                M[nd, ph, 1] = 1.0
        elif valid and nd != prev_node:            # move_location, same phase
            M[nd, ph, 1] = 1.0
        out[t] = M
        prev_phase, prev_node = ph, nd
    return np.roll(out, -1, axis=3)                # undo the 1-step lag (cell 15)


def anchor_regressors(nr, tt, loc, num_phases=3):
    """Fig-5 anchoring-GLM data + design for ONE task (continuous 25 ms bins).

    Returns
    -------
    X : (T, num_nodes*num_phases*num_lags) float32 -- flattened (node, phase, lag)
        regressors; reshape to (T, 9, num_phases, num_states*num_phases) to recover
        the conjunction axes.
    Y : (T, n_neuron) float32 -- raw spike counts (the GLM target).
    info : dict(location (T,) node code 1-9 / NaN at edges, phase (T,), state (T,),
        start) -- `location` NaN marks the bins the paper drops; `phase` is for the
        per-neuron preferred-phase subset.
    """
    n_states = tt.shape[1] - 1
    num_nodes = 9
    num_lags = n_states * num_phases
    start, end, phase, state = _phase_state_arrays(tt, nr.shape[1], num_phases)
    loc_seg = np.asarray(loc[start:end], dtype=float)
    nodes = np.where((loc_seg >= 1) & (loc_seg <= num_nodes),
                     loc_seg - 1, -1).astype(int)
    reg = _anchor_bump(nodes, phase, num_nodes, num_phases, num_lags)
    X = reg.reshape(reg.shape[0], -1).astype(np.float32)
    Y = nr[:, start:end].T.astype(np.float32)
    location = np.where(nodes >= 0, nodes + 1.0, np.nan)
    return X, Y, dict(location=location, phase=phase, state=state, start=start)


def load_regression_data(load_dir=LOAD_DIR, animals=None, min_trials=2,
                         num_phases=3, drop_repeat_tasks=True,
                         with_meta=True, verbose=True):
    """Fig-5 anchoring-GLM data + regressors for each physically-distinct sorted
    population, combined into as few arrays as possible (same content-based
    union-find as `load_combined_legs`).

    Per population, the population's ABCD tasks are concatenated along continuous
    time (all 25 ms bins of each task's trial span -- NOT trial-averaged, NOT warped)
    into a single design:
      X : (T_total, 9*num_phases*num_lags) anchoring regressors (= 324 for ABCD)
      Y : (T_total, n_neuron) raw spike counts (columns = the population's tracked
          neurons, identity constant down the array -- this is exactly the
          within-recording-day neuron sharing the paper relies on).
    To reproduce the paper's leave-one-task-out fit: drop bins where
    `meta['bin_location']` is NaN (edges/untracked), subset each neuron to its
    preferred phase via `meta['bin_phase']`, and CV over `meta['bin_task']`
    (indexes `meta['task_keys']`), fitting PoissonRegressor(alpha=1) per neuron.

    Parameters
    ----------
    animals : keep only these mice; None = all (loads every Neuron_raw file -- the
              designs are large, so filter by animal to work incrementally).
    drop_repeat_tasks : keep only the first task of each distinct reward-node layout
              within a population (mirrors the paper's `non_repeat_ses`).
    num_phases : phases per state leg (3 in the paper).

    Returns
    -------
    pops : list of (Y, X) tuples, one per distinct population.
    meta : list of dicts {keys, n_neuron, n_tasks, T, bin_location, bin_phase,
           bin_state, bin_task, task_keys}, only when with_meta=True. The bin_*
           arrays are length T_total and aligned with the rows of X/Y.
    """
    groups = list_sessions(load_dir)
    if animals is not None:
        animals = set(animals)
        groups = {k: v for k, v in groups.items() if k.split('_')[0] in animals}

    parent = {}                                   # union-find over keys
    seen_hash_key = {}                            # task hash -> first key seen
    hash_data = {}                                # task hash -> (X, Y, info)
    hash_rep = {}                                 # task hash -> (key, idx, structure)
    key_hashes = defaultdict(list)
    n_files = 0
    for key, idxs in groups.items():
        parent.setdefault(key, key)
        for idx in idxs:
            try:
                nr, tt, loc = load_session(key, idx, load_dir)
            except FileNotFoundError:
                continue
            if tt.ndim != 2 or tt.shape[0] < min_trials or tt.shape[1] != 5:
                continue                          # ABCD only (4 states -> 324 regs)
            n_files += 1
            h = hashlib.sha1(np.ascontiguousarray(nr).tobytes()).digest()
            key_hashes[key].append(h)
            if h in seen_hash_key:                # identical task -> same sort
                ra, rb = _uf_find(parent, key), _uf_find(parent, seen_hash_key[h])
                parent[ra] = rb
            else:
                seen_hash_key[h] = key
                hash_data[h] = anchor_regressors(nr, tt, loc, num_phases)
                struct = tuple(reward_nodes(tt, loc).get(s) for s in range(4))
                hash_rep[h] = (key, idx, struct)
            if verbose and n_files % 25 == 0:
                print(f'  built {n_files} task designs...')

    comps = defaultdict(list)                      # root key -> member keys
    for key in key_hashes:
        comps[_uf_find(parent, key)].append(key)

    pops, meta = [], []
    for root in sorted(comps):
        uniq = {h for key in comps[root] for h in key_hashes[key]}     # distinct tasks
        uniq = sorted(uniq, key=lambda h: (hash_rep[h][0], int(hash_rep[h][1])))
        if drop_repeat_tasks:                      # one task per reward layout
            seen_struct, kept = set(), []
            for h in uniq:
                if hash_rep[h][2] in seen_struct:
                    continue
                seen_struct.add(hash_rep[h][2])
                kept.append(h)
            uniq = kept
        X = np.concatenate([hash_data[h][0] for h in uniq], axis=0)
        Y = np.concatenate([hash_data[h][1] for h in uniq], axis=0)
        loc_b = np.concatenate([hash_data[h][2]['location'] for h in uniq])
        ph_b = np.concatenate([hash_data[h][2]['phase'] for h in uniq])
        st_b = np.concatenate([hash_data[h][2]['state'] for h in uniq])
        task_b = np.concatenate([np.full(len(hash_data[h][1]), ti, dtype=int)
                                 for ti, h in enumerate(uniq)])
        task_keys = [(hash_rep[h][0], hash_rep[h][1]) for h in uniq]
        pops.append((Y, X))
        meta.append(dict(keys=sorted(comps[root]), n_neuron=Y.shape[1],
                         n_tasks=len(uniq), T=Y.shape[0], bin_location=loc_b,
                         bin_phase=ph_b, bin_state=st_b, bin_task=task_b,
                         task_keys=task_keys))
        if verbose:
            print(f'[{len(pops)}] {sorted(comps[root])[0]} (+{len(comps[root])-1}): '
                  f'{Y.shape[1]} neurons, T={Y.shape[0]}, {len(uniq)} tasks, '
                  f'X{X.shape}')

    return (pops, meta) if with_meta else pops


#%%
# ------------------------------ build pseudopop -----------------------------

def build_group(key, idxs, load_dir=LOAD_DIR):
    """Accumulate one recording group's neurons into (4, 9, n_neurons, N_BINS)
    sums and (4, 9, n_neurons) counts over all its ABCD tasks/trials. Returns
    (X_block, count_block) with X_block = mean profile (NaN where count==0)."""
    n_neuron = None
    sums = counts = None
    for idx in idxs:
        try:
            nr, tt, loc = load_session(key, idx, load_dir)
        except FileNotFoundError:
            continue                       # session missing Location/trialtimes
        if tt.ndim != 2 or tt.shape[1] != 5 or tt.shape[0] < 2:
            continue                       # ABCD only (5 columns); skip ABCDE/other
        if n_neuron is None:
            n_neuron = nr.shape[0]
            sums = np.zeros((4, 9, n_neuron, N_BINS))
            counts = np.zeros((4, 9, n_neuron))
        elif nr.shape[0] != n_neuron:
            continue                       # neuron set must match within a group
        nodes = reward_nodes(tt, loc)
        for ti in range(tt.shape[0]):
            for end_col, state in _END_COL_TO_STATE.items():
                node = nodes[state]
                if node is None:
                    continue
                a, e = int(tt[ti, end_col - 1]), int(tt[ti, end_col])
                if e <= a or a < 0 or e > nr.shape[1]:
                    continue
                prof = normalise_block(nr[:, a:e])      # (n_neuron, 90)
                sums[state, node - 1] += prof
                counts[state, node - 1] += 1
    if n_neuron is None:
        return None, None
    with np.errstate(invalid='ignore'):
        X = sums / counts[..., None]
    X[counts == 0] = np.nan
    return X, counts


def build_pseudopop(groups=None, load_dir=LOAD_DIR, animals=None, verbose=True):
    """Pool all ABCD recordings into a 36 x n_neuron x 90 pseudopopulation.

    Returns
    -------
    X : (36, n_neuron, 90) float, NaN where a neuron never had that reward at
        that node. Rows ordered state-major: row = state_idx*9 + (node-1).
    cond_df : DataFrame (state, node, label) describing the 36 rows.
    neuron_group : (n_neuron,) object array, source recording-group key per column.
    counts : (36, n_neuron) number of legs averaged into each cell (0 where NaN).
    """
    if groups is None:
        groups = dedup_groups(list_sessions(load_dir))
    if animals is not None:
        animals = set(animals)
        groups = {k: v for k, v in groups.items() if k.split('_')[0] in animals}

    Xs, cnts, src = [], [], []
    for gi, (key, idxs) in enumerate(groups.items()):
        Xb, cb = build_group(key, idxs, load_dir)
        if Xb is None:
            continue
        nn = Xb.shape[2]
        Xs.append(Xb.reshape(36, nn, N_BINS))
        cnts.append(cb.reshape(36, nn))
        src.append(np.array([key] * nn, dtype=object))
        if verbose:
            filled = np.mean(np.any(~np.isnan(Xb.reshape(36, nn, N_BINS)), axis=2))
            print(f'[{gi + 1}/{len(groups)}] {key}: {nn} neurons, '
                  f'{filled * 100:.0f}% of 36 conds filled')

    X = np.concatenate(Xs, axis=1)
    counts = np.concatenate(cnts, axis=1)
    neuron_group = np.concatenate(src)

    cond = [(STATES[s], int(n)) for s in range(4) for n in NODES]
    cond_df = pd.DataFrame(cond, columns=['state', 'node'])
    cond_df['label'] = [f'{s}@{n}' for s, n in cond]
    return X, cond_df, neuron_group, counts


#%%

# X, cond_df, ng, counts = build_pseudopop(animals=['ab03'])


# full build across all animals + save
# X, cond_df, ng, counts = build_pseudopop()
# out = dict(X=X, cond_df=cond_df, neuron_group=ng, counts=counts,
#            states=STATES, nodes=NODES, n_bins=N_BINS)
# obs = np.any(~np.isnan(X), axis=2)                  # (36, n_neuron) observed?

# with open(SAVE_DIR + 'elgaby_state_x_location_pseudopop.pkl', 'wb') as f:
#     pkl.dump(out, f)

pops, meta = load_combined_legs(with_meta=True, verbose=True, n_states=4)

#%%

X_ = pops[0]
# X_ = X.transpose((0,2,1))
# mask = np.isnan(X_)
# X_ = np.where(np.isnan(X_), np.nanmean(X_), X_)

mod = bae_models.JRRBMF(dim_hid=9,
                            rank=3, 
                            nonneg=True, 
                            sparse_reg=1,
                            tree_reg=10,
                            weight_pr_reg=1e-1, 
                            weight_l2_reg=1e-1, 
                            J_lr=1e-3,
                            )

en = mod.fit(X_ / X_.std(), 
             decay_rate=0.88,
             min_temp=1,
             initial_temp=10,
             period=100,
             scl_lr=1e-3,
             lr=0.1,
             # mask=mask,
             )

samps = mod.sample(X_ / X_.std(), n_samp=100)

#%%


X_ = util.group_mean(pops[0][0],pops[0][1], axis=0)

# mod = bae_models.JBMF(dim_hid=324,
#                           nonneg=True, 
#                           sparse_reg=1,
#                           tree_reg=10,
#                           weight_pr_reg=1e-1, 
#                           weight_l2_reg=1e-1, 
#                           J_lr=1e-3,
#                           )

mod = bae_models.SemiBMF(dim_hid=324,
                              nonneg=True, 
                              sparse_reg=1,
                              tree_reg=10,
                              weight_pr_reg=1e-1, 
                              weight_l2_reg=1e-1, 
                              )

en = mod.fit(X_ / X_.std(), 
             decay_rate=0.88,
             min_temp=1,
             initial_temp=10,
             period=10,
             scl_lr=1e-3,
             # lr=0.1,
             # mask=mask,
             )

# samps = mod.sample(X_ / X_.std(), n_samp=100)


