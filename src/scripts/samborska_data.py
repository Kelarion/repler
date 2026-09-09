CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/samborska_data/'

import os, sys
sys.path.append(CODE_DIR)

import numpy as np
import pandas as pd
import scipy.io as scio
import pickle as pkl
import pandas as pd

from sklearn import svm, discriminant_analysis, manifold, linear_model
from sklearn.cluster import KMeans, AgglomerativeClustering

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
# ----------------------------------------------------------------------------
# Samborska et al. (2022) Nat Neurosci -- HP/PFC recordings, "generalising the
# structure of problems" two-step-ish task. Figure 4 (RSA / SVD / decoding).
#
# Each region's .mat loads to two parallel cell arrays, one entry per session:
#   data['Data'][0][s] : firing rates, shape (n_trials, n_neurons, n_time)
#   data['DM'][0][s]   : trial-info "design matrix", shape (n_trials, 10)
# Different neurons are recorded in different sessions, so there is *no* single
# trial x neuron matrix over all cells -- Figure 4 pools neurons across sessions
# into a condition x neuron pseudo-population (build_pseudopop below).
#
# Two versions on disk:
#   raw : data_recordings/{HP,PFC}.mat                 (63 time bins; Fig 4B regression)
#   dlc : data_recordings/after PCA/{HP,PFC}_dlc_pca.mat (64 time bins; movement
#         variance removed -- this is what main-figure panels 4C-H use)
# ----------------------------------------------------------------------------

# DM (design-matrix) columns, recovered from the paper's function code.
# 0 and 3 are binary flags whose meaning is not used by any Fig-4 function.
DM_COLS = ['col0', 'choice', 'reward', 'col3', 'block',
           'task', 'a_poke', 'b_poke', 'i_poke', 'const']
# choice: A==1, B==0 ; reward: 1/0 ; task: physical poke configuration (1,2,3)

# Time axis (bins -> event), from the paper's plotting xticks
# (functions_cells_regress.py: _xticks; functions_cells_ext.py: tick_).
# Full 7 anchors, labelled ['-1','-0.5','Init','Ch','R','+0.5','+1']:
#   raw (63 bins): bins [0, 12,   24, 35, 42, 49, 63]
#   dlc (64 bins): bins [0, 12.5, 25, 38, 45, 51, 64]
# The Init->Choice->Reward interval is time-warped to a fixed number of bins, so
# only the two flanks carry physical seconds: the pre-Init flank is real seconds
# relative to Init, the post-Reward flank real seconds relative to Reward; the
# warped middle (Init..Reward, including Choice) has no seconds value.
EVENTS_RAW = {'minus1s': 0, 'init': 24, 'choice': 35, 'reward': 42, 'plus1s': 63}
EVENTS_DLC = {'minus1s': 0, 'init': 25, 'choice': 38, 'reward': 45, 'plus1s': 64}

ANCHORS_RAW = {'minus1s': 0, 'minus0.5s': 12,   'init': 24, 'choice': 35,
               'reward': 42, 'plus0.5s': 49,    'plus1s': 63}
ANCHORS_DLC = {'minus1s': 0, 'minus0.5s': 12.5, 'init': 25, 'choice': 38,
               'reward': 45, 'plus0.5s': 51,    'plus1s': 64}


def trial_time_axis(n_time, dlc=None):
    """Time-in-trial axis for the Fig-4 firing-rate window.

    Parameters
    ----------
    n_time : number of time bins in the firing-rate arrays (63 raw / 64 dlc).
    dlc : pick the raw vs movement-corrected anchor bins. If None (default),
        inferred from n_time (>=64 -> dlc).

    Returns
    -------
    times_sec : (n_time,) float array. Real seconds in the two flanks -- relative
        to Init before Init, relative to Reward after Reward -- and NaN across the
        time-warped Init->Choice->Reward interval, which has no physical time.
    event_bins : dict {label: bin index} of the paper's 7 tick anchors
        ('minus1s','minus0.5s','init','choice','reward','plus0.5s','plus1s').
        These are the things to index/plot by (cf. the paper's xticks).
    """
    if dlc is None:
        dlc = n_time >= 64
    anchors = ANCHORS_DLC if dlc else ANCHORS_RAW

    bins = np.arange(n_time)
    times_sec = np.full(n_time, np.nan)
    init, reward = anchors['init'], anchors['reward']
    # pre-Init flank: linear, Init = 0 s (anchored by -1 s, -0.5 s, Init)
    pre = bins <= init
    times_sec[pre] = np.interp(
        bins[pre], [anchors['minus1s'], anchors['minus0.5s'], anchors['init']],
        [-1.0, -0.5, 0.0])
    # post-Reward flank: linear, Reward = 0 s (anchored by Reward, +0.5 s, +1 s)
    post = bins >= reward
    times_sec[post] = np.interp(
        bins[post], [anchors['reward'], anchors['plus0.5s'], anchors['plus1s']],
        [0.0, 0.5, 1.0])
    return times_sec, dict(anchors)

_FILES = {
    ('HP', False):  'data_recordings/HP.mat',
    ('PFC', False): 'data_recordings/PFC.mat',
    ('HP', True):   'data_recordings/after PCA/HP_dlc_pca.mat',
    ('PFC', True):  'data_recordings/after PCA/PFC_dlc_pca.mat',
}


def task_ind(task, a_pokes, b_pokes):
    """Port of helper_functions.task_ind: consistent task IDs (1,2,3) defined by
    the relative geometry of the A/B pokes rather than the raw 'task' column."""
    taskid = np.zeros(len(task))
    taskid[b_pokes == 10 - a_pokes] = 1
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3),
                         np.logical_or(b_pokes == 7, b_pokes == 8))] = 2
    taskid[np.logical_or(b_pokes == 1, b_pokes == 9)] = 3
    return taskid


def load_region(region='PFC', dlc=True, load_dir=LOAD_DIR):
    """Load one region as per-session arrays + trial-info DataFrames.

    Returns
    -------
    frs : list of (n_trials, n_neurons, n_time) float arrays, one per session.
    dms : list of pandas.DataFrame, one per session, columns = DM_COLS, plus a
          'taskid' column (geometry-consistent task id from task_ind).
    """
    mat = scio.loadmat(os.path.join(load_dir, _FILES[(region, dlc)]))
    n_sess = mat['Data'].shape[1]
    frs, dms = [], []
    for s in range(n_sess):
        fr = np.asarray(mat['Data'][0][s], dtype=float)
        dm = pd.DataFrame(np.asarray(mat['DM'][0][s], dtype=float), columns=DM_COLS)
        dm['taskid'] = task_ind(dm['task'].values, dm['a_poke'].values, dm['b_poke'].values)
        frs.append(fr)
        dms.append(dm)
    return frs, dms


def build_pseudopop(frs, dms, time_resolved=True, include_init=True, conditions=None,
                    dlc=None):
    """Pool neurons across sessions into a condition x neuron (x time) matrix.

    For each session and each condition, average the firing rate over the trials
    of that session matching the condition; then concatenate neurons across all
    sessions along the neuron axis. A condition with no trials in a session gives
    NaN for that session's neurons.

    Parameters
    ----------
    frs, dms : output of load_region.
    time_resolved : if True return (n_cond, n_neurons, n_time); else average the
        whole trial window away and return (n_cond, n_neurons).
    include_init : if True (default) add one Initiation condition per task -- all
        trials of that task, with choice/reward unspecified -- giving the 15
        conditions used in the paper's Fig-4 RSA (12 choice x reward x task + 3 init).
    conditions : explicit list of (taskid, choice, reward) tuples defining the rows;
        use choice=reward=None for an init/all-trials row. Overrides include_init.
        Default is the 12 task x choice{A=1,B=0} x reward{1,0} combinations
        (+ the 3 init rows when include_init).
    dlc : passed to trial_time_axis to label the time bins (raw vs dlc anchors);
        None (default) infers from the number of time bins.

    Returns
    -------
    X : float array, NaN where a session lacked a condition.
    cond_df : DataFrame describing each row (taskid, choice, reward, label);
        choice/reward are NaN for init rows.
    neuron_session : int array (n_neurons,) giving the source session of each neuron.
    times_sec : (n_time,) time-in-trial in seconds for the time axis -- real
        seconds on the pre-Init / post-Reward flanks, NaN across the warped
        Init->Choice->Reward middle (see trial_time_axis). All NaN when
        time_resolved=False (no time axis).
    event_bins : dict {label: bin index} of the trial-event / tick anchors
        (init, choice, reward, +/- 0.5 s, +/- 1 s) -- the timing of initiation,
        choice and outcome in bins. Same dict regardless of time_resolved.
    """
    if conditions is None:
        conditions = [(t, c, r) for t in (1, 2, 3)
                      for c in (1, 0) for r in (1, 0)]
        if include_init:
            conditions += [(t, None, None) for t in (1, 2, 3)]

    n_time = frs[0].shape[2]
    neuron_session = np.concatenate([np.full(fr.shape[1], s)
                                     for s, fr in enumerate(frs)])
    n_neurons = len(neuron_session)
    n_cond = len(conditions)

    if time_resolved:
        X = np.full((n_cond, n_neurons, n_time), np.nan)
    else:
        X = np.full((n_cond, n_neurons), np.nan)

    col = 0
    for fr, dm in zip(frs, dms):
        nn = fr.shape[1]
        choice = dm['choice'].values
        reward = dm['reward'].values
        taskid = dm['taskid'].values
        for ci, (t, c, r) in enumerate(conditions):
            mask = (taskid == t)
            if c is not None:                # init rows leave choice/reward free
                mask &= (choice == c) & (reward == r)
            sel = np.where(mask)[0]
            if len(sel) == 0:
                continue
            if time_resolved:
                X[ci, col:col + nn, :] = fr[sel].mean(0)         # mean over trials
            else:
                X[ci, col:col + nn] = fr[sel].mean((0, 2))       # mean over trials & time
        col += nn

    cond_df = pd.DataFrame(conditions, columns=['taskid', 'choice', 'reward'])
    cond_df['label'] = [
        'T%d_init' % t if c is None
        else 'T%d_%s_%s' % (t, 'A' if c else 'B', 'r' if r else 'nr')
        for t, c, r in conditions]

    times_sec, event_bins = trial_time_axis(n_time, dlc=dlc)
    if not time_resolved:                    # time axis collapsed away
        times_sec = np.full(n_time, np.nan)
    return X, cond_df, neuron_session, times_sec, event_bins


def rsa_matrix(frs, dms, t_start=36, t_end=40, init_start=24, init_stop=26):
    """Reproduce the exact 15 x neuron matrix used by the paper's Fig-4 RSA
    (functions_cells_rsa.extract_trials): per-neuron condition means in a window
    [t_start:t_end] around choice/outcome, plus a per-task Initiation row.

    Row order matches the paper:
      a1r a1nr a2r a2nr a3r a3nr  i1 i3 i2  b3r b3nr b2r b2nr b1r b1nr
    """
    n_neurons = sum(fr.shape[1] for fr in frs)
    rows = {k: np.zeros(n_neurons) for k in
            ['a1r', 'a1nr', 'a2r', 'a2nr', 'a3r', 'a3nr',
             'i1', 'i2', 'i3',
             'b1r', 'b1nr', 'b2r', 'b2nr', 'b3r', 'b3nr']}
    col = 0
    for fr, dm in zip(frs, dms):
        nn = fr.shape[1]
        choice = dm['choice'].values; reward = dm['reward'].values
        taskid = dm['taskid'].values
        sl = slice(col, col + nn)
        for t in (1, 2, 3):
            for c, ab in ((1, 'a'), (0, 'b')):
                for r, rn in ((1, 'r'), (0, 'nr')):
                    sel = np.where((choice == c) & (reward == r) & (taskid == t))[0]
                    rows['%s%d%s' % (ab, t, rn)][sl] = fr[sel][:, :, t_start:t_end].mean((0, 2))
            tt = np.where(taskid == t)[0]
            rows['i%d' % t][sl] = fr[tt][:, :, init_start:init_stop].mean((0, 2))
        col += nn
    order = ['a1r', 'a1nr', 'a2r', 'a2nr', 'a3r', 'a3nr',
             'i1', 'i3', 'i2',
             'b3r', 'b3nr', 'b2r', 'b2nr', 'b1r', 'b1nr']
    return np.vstack([rows[k] for k in order]), order


#%% demo: build the Fig-4 (movement-corrected) pseudo-populations and save


out = {}
for region in ['HP', 'PFC']:
    frs, dms = load_region(region, dlc=False)
    X, cond_df, neuron_session, times_sec, event_bins = build_pseudopop(
        frs, dms, time_resolved=True, dlc=False)
    Xt, *_ = build_pseudopop(frs, dms, time_resolved=False)
    rsa, rsa_order = rsa_matrix(frs, dms)

    n_neurons = X.shape[1]
    print('%s (dlc): %d sessions, %d neurons, %d time bins'
          % (region, len(frs), n_neurons, frs[0].shape[2]))
    print('   pseudopop  X (cond x neuron x time):', X.shape,
          '| NaN frac %.3f' % np.isnan(X).mean())
    print('   collapsed  X (cond x neuron):       ', Xt.shape)
    print('   Fig-4 RSA matrix (15 x neuron):     ', rsa.shape)

    out[region] = {'X': X, 'X_flat': Xt, 'conditions': cond_df,
                   'neuron_session': neuron_session,
                   'rsa': rsa, 'rsa_order': rsa_order,
                   'times_sec': times_sec, 'event_bins': event_bins,
                   'events': EVENTS_DLC}

# with open(os.path.join(SAVE_DIR, 'samborska_fig4_pseudopop.pkl'), 'wb') as f:
#     pkl.dump(out, f)
# print('saved -> %ssamborska_fig4_pseudopop.pkl' % SAVE_DIR)


#%%

# X_ = X[:,area==this_area] / X[:,area==this_area].std(0, keepdims=True)
# X_ = X_ / X_.std(1, keepdims=True)

X_ = out['PFC']['rsa']
# X_ = 1*out['HP']['rsa']

# X_ -= X_.mean(0)

mod = old_bae_models.SemiBMF(5,
                         nonneg=True,
                         tree_reg=1e-3,
                         weight_pr_reg=1,
                         weight_l2_reg=1e-2,
                         weight_l1_reg=0,
                         sparse_reg=1e-1,
                         )

# mod = old_bae_models.SemiBMF(7,
#                          nonneg=False,
#                          tree_reg=1,
#                          weight_pr_reg=1,
#                          weight_l2_reg=0,
#                          weight_l1_reg=1e-2,
#                          sparse_reg=1,
#                          )

# mod = old_bae_models.SpikeNMF(7,
#                          nonneg=False, 
#                          sparse_reg=1,
#                          weight_pr_reg=1,
#                          tree_reg=1,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=0,
#                          )

# mod = bae_models.JBMF(dim_hid=324,
#                           nonneg=True, 
#                           sparse_reg=1,
#                           tree_reg=10,
#                           weight_pr_reg=1e-1, 
#                           weight_l2_reg=1e-1, 
#                           J_lr=1e-3,
#                           )


# mod = bae_models.JBMF(4,
#                          # nonneg=True,
#                          nonneg=False,
#                          # fit_intercept=False,
#                          tree_reg=0,
#                          weight_pr_reg=1,
#                          weight_l2_reg=1e-3,
#                          weight_l1_reg=1e-2,
#                          sparse_reg=1e-2,
#                          # J_loss='mle',
#                          J_loss='rple',
#                          # J_l1_reg=1e-3,
#                          J_lr=1e-3,
#                          # slab=True,
#                          # slab_prior=0.1,
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
#                            sparse_reg=75,
#                            tree_reg=1,
#                            # uniform_scale=False,
#                            uniform_scale=True,
#                            # l1_reg=0,
#                            )

en = mod.fit(X_ / X_.std(),
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


samps = mod.sample(X_ / X_.std(), n_samp=1000)
# samps = mod.sample(X_ / X_.std(), n_samp=1000, slab=False)

# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.imshow(samps.mean(0))

#%%

X_ = out['HP']['X'].transpose((0,2,1))[:12]
# X_ = out['PFC']['X'].transpose((0,2,1))[:12]

X_ /= X_.std((0,1), keepdims=True)

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
#                             sparse_reg=0,
#                             tree_reg=0,
#                             weight_pr_reg=1e-1, 
#                             weight_l2_reg=1e-1, 
#                             J_lr=1e-3,
#                             )

# mod = bae_models.SCPD(dim_hid=4,
#                             nonneg=True, 
#                             # nonneg=False,
#                             sparse_reg=1,
#                             tree_reg=1,
#                             weight_pr_reg=1,
#                             weight_l1_reg=1e-2,
#                             )

mod = bae_models.JSCPD(dim_hid=5,
                            nonneg=True, 
                            # nonneg=False,
                            sparse_reg=1e-1,
                            tree_reg=0,
                            weight_pr_reg=10,
                            weight_l1_reg=1e-3,
                            weight_l2_reg=1e-1,
                            J_lr=1e-2,
                            # J_lr=0,
                            # J_loss='logrise',
                            # slab=True,
                            n_chains=8,
                            )

en = mod.fit(X_ / X_.std(),
             period=50,
             initial_temp=100,
             decay_rate=0.88, 
             min_temp=1, 
             scl_lr=1e-3,
             lr=1e-3,
             hot_start=False,
             # hot_start=True,
             )

mod.collapse()

samps = mod.sample(X_ / X_.std(), n_samp=100)
# samps = mod.sample(X_ / X_.std(), n_samp=1000, slab=False)

# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.figure()

plt.subplot(2,3,1)
tpl.matshow(samps.mean(0), cmap='binary', color=(0.5,0.5,0.5))

plt.subplot(2,3,2)
plt.imshow(mod.latent_prior.J_W + mod.latent_prior.J_W.T, 'bwr', vmin=-1, vmax=1)

plt.subplot(2,3,3)
plt.imshow((mod.operator.V.T@mod.operator.V).detach())

plt.subplot(2,3,4)
cols = cm.Set1(range(mod.dim_hid))

for i in range(mod.dim_hid):
    plt.plot(mod.operator.U.detach()[:,i], c=cols[i])

plt.legend(range(5), title='concept')
plt.ylim(plt.ylim())
plt.xlim(plt.xlim())

tims = []
evs = []
for e,t in out['HP']['events'].items():
    plt.vlines(t, plt.ylim()[0], plt.ylim()[1], color='k', linestyle='--')
    
    tims.append(t)
    evs.append(e)
    
plt.gca().set_xticks(tims)
plt.gca().set_xticklabels(evs)

plt.subplot(2,3,5)
plt.plot(en)

#%%

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
        'nonneg': True,
        'weight_pr_reg':10,
        'weight_l1_reg':1e-3,
        'weight_l2_reg':1e-1,
        # 'tree_reg': 1e-1,
        # 'sparse_reg': 1,
        'sparse_reg': 1,
        'tree_reg': 0,
        'J_lr': 1e-2,
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
            'period': 50,
            # 'hot_start': True,
            'hot_start': False,
            # 'scl_lr': 0,
            'scl_lr': 1e-3,
            'min_temp': 1,
            # 'min_temp': 1e-4,
            # 'lr': 1e-2,
            'lr':1e-3,
            }

n_run = 1

trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
ens = []
sigs = []
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
        
        # mod = bae_models.JBMF(k,**args)
        # mod = bae_models.BiPCA(k, **args)
        mod = bae_models.JSCPD(k, **args)
        
        # wa,ba = bae_util.loocv(mod, X_/X_.std(), n_sample=10, **opt_args)
        wa,ba = bae_util.impcv(mod, X_/X_.std(), verbose=False, seed=0, n_sample=10, folds=10, max_folds=1, **opt_args)
        # wa,ba = bae_util.gabriel_bicv(mod, X_/X_.std(), n_samp=10, **opt_args)
        
        trn[i] += np.mean(wa) / n_run
        tst[i] += np.mean(ba) / n_run
        ens.append(en)
        sigs.append(mod.sigma_x)

plt.plot(kays, trn)
plt.plot(kays, tst, '--')



