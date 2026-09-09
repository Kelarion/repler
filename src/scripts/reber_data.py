CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/reber_data/'

import os, sys
sys.path.append(CODE_DIR)

import numpy as np
import pickle as pkl
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.distance import pdist, squareform

import util
import df_util
import bae_util
import bae_models
import new_bae_models
import plotting as tpl

# ----------------------------------------------------------------------------
# Reber, Bausch, Mackay, Boström, Elger & Mormann (2019) -- "Representation of
# Abstract Semantic Knowledge in Populations of Human Single Neurons in the
# Medial Temporal Lobe". Human MTL single-unit recordings (epilepsy patients)
# while they viewed 100 images = 10 semantic categories x 10 exemplars.
#
# This file extracts the population representation that GENERATES the Figure 3
# RSA panels (similarity matrices / MDS / dendrograms), one matrix per MTL
# region, in (stimulus x neuron) form ready for the BMF factorization models.
#
# ---- What Figure 3 actually plots (see figure3orS2.m) ----------------------
# The per-region representation is `zvals(units_in_region, :)` -- an
# (n_units x 100 stimuli) matrix of z-scored firing rates (one number per
# unit-stimulus pair; z-scoring method per the paper). The RSA is then
#       S = pdist(zvals(idx,:)', 'correlation')          % 1 - Pearson R
# i.e. correlation distance BETWEEN the 100 stimuli, using the region's units
# as the feature dimension. So:
#       observations = 100 stimuli   (the things being compared)
#       features     = neurons       (the representation axis)
# We store X_region with shape (100, n_units) = zvals[idx].T, which is exactly
# the matrix the similarity/MDS/dendrogram operate on (their pdist is over rows
# of zvals[idx]', = rows of our X_region).
#
# Regions (regions.mat / cluster_lookup.regionname): AM (amygdala),
# HC (hippocampus), EC (entorhinal), PHC (parahippocampal). Units in 'other'
# (outside MTL, n=119) are NOT reported in the paper and are dropped here.
# Figure 3 = ALL units ('all'); Figure S2 = first session per patient.
#
# ---- Loading note ----------------------------------------------------------
# zvals.mat is a v7.3 (HDF5) file whose `cluster_lookup` is a MATLAB *table*,
# which scipy / h5py / mat73 cannot decode. Run the one-time MATLAB helper
#       export_for_python.m   (in LOAD_DIR)
# first; it flattens the table columns we need (regionname, subjid, sessid,
# consider_rs) and re-saves everything as scipy-readable reber_export.mat.
# ----------------------------------------------------------------------------

REGIONS = ['AM', 'HC', 'EC', 'PHC']      # MTL regions, Fig 3 panel order
N_STIM = 100
N_CAT = 10                                # 10 categories x 10 exemplars

# which units to include, mirroring figure3orS2.m:
#   'all'        -> Figure 3   (every unit in the region)
#   'firstsess'  -> Figure S2  (only the first session recorded per patient)
#   'responsive' -> region units with >=1 responsive stimulus (consider_rs)
WHICH_UNITS = 'all'


def load_reber(load_dir=LOAD_DIR):
    """Load the flattened export. Returns a dict of numpy arrays / label lists."""
    import scipy.io as sio
    f = os.path.join(load_dir, 'reber_export.mat')
    if not os.path.exists(f):
        raise FileNotFoundError(
            f'{f} not found -- run export_for_python.m in MATLAB first '
            '(see the loading note at the top of this file).')
    d = sio.loadmat(f)
    unwrap = lambda a: np.array([str(x[0]) if len(x) else '' for x in a.ravel()])
    return {
        'zvals':       d['zvals'].astype(float),          # (n_units, 100)
        'regionname':  unwrap(d['regionname']),           # (n_units,)
        'subjid':      d['subjid'].ravel().astype(int),   # (n_units,)
        'sessid':      d['sessid'].ravel().astype(int),   # (n_units,)
        'consider_rs': d['consider_rs'].astype(bool),     # (n_units, 100)
        'cat_lookup':  unwrap(d['cat_lookup']),           # (100,) category per stim
        'stim_lookup': unwrap(d['stim_lookup']),          # (100,) image name per stim
    }


def first_session_mask(subjid, sessid):
    """Boolean over units: True for units from each patient's earliest session
    (reproduces the Fig S2 'firstSessionPerPatient' selection)."""
    keep = np.zeros(len(subjid), dtype=bool)
    for s in np.unique(subjid):
        first = sessid[subjid == s].min()
        keep[(subjid == s) & (sessid == first)] = True
    return keep


def build_region_matrices(data, which=WHICH_UNITS):
    """Per region -> (100 stimuli x n_units) z-score matrix (the Fig 3 RSA input).

    Returns (mats, counts) where mats[region] has shape (100, n_units_region)."""
    rn = data['regionname']
    base = np.ones(len(rn), dtype=bool)
    if which == 'firstsess':
        base = first_session_mask(data['subjid'], data['sessid'])
    elif which == 'responsive':
        base = data['consider_rs'].any(axis=1)

    mats, counts = {}, {}
    for r in REGIONS:
        idx = (rn == r) & base
        mats[r] = data['zvals'][idx].T          # (100, n_units)
        counts[r] = int(idx.sum())
    return mats, counts


# ----------------------------------------------------------------------------
#%% ---- load + build the four region matrices, save a pickle ----------------

data = load_reber()

cats, cat_idx = np.unique(data['cat_lookup'], return_inverse=True)  # 10 names, idx in 0..9
assert len(cats) == N_CAT and len(cat_idx) == N_STIM

mats, counts = build_region_matrices(data, WHICH_UNITS)
print(f'units per region ({WHICH_UNITS}):',
      {r: counts[r] for r in REGIONS})       # Fig 3: AM 1392, HC 1863, EC 828, PHC 831

bundle = {
    'X': mats,                       # region -> (100 stimuli x n_units) z-scores
    'regions': REGIONS,
    'cat_lookup': data['cat_lookup'],    # (100,) category name per stimulus
    'stim_lookup': data['stim_lookup'],  # (100,) image name per stimulus
    'cat_names': cats,                   # (10,) the category names
    'cat_idx': cat_idx,                  # (100,) integer category label 0..9
    'which_units': WHICH_UNITS,
}
out = os.path.join(SAVE_DIR, f'reber_region_zvals_{WHICH_UNITS}.pkl')
# pkl.dump(bundle, open(out, 'wb'))
# print('saved ->', out)


#%% ---- sanity check: reproduce the Fig 3 RSA (1 - R) per region ------------
# Each panel should show the block-diagonal (within-category) structure of the
# paper's Fig 3 similarity matrices. caxis in the paper is [0.6 1.1].

fig, axs = plt.subplots(1, 4, figsize=(15, 4))
for ax, r in zip(axs, REGIONS):
    Xr = mats[r]                              # (100, n_units)
    D = squareform(pdist(Xr, 'correlation'))  # 100x100, 1 - Pearson R
    im = ax.imshow(D, cmap='jet', vmin=0.6, vmax=1.1)
    same = cat_idx[:, None] == cat_idx[None, :]
    iu = ~np.eye(N_STIM, dtype=bool)
    win = D[same & iu].mean(); btw = D[~same & iu].mean()
    ax.set_title(f'{r} (n={counts[r]})\nwin={win:.2f} btw={btw:.2f}')
    for i in range(1, N_CAT):                 # category block boundaries
        ax.axhline(i*N_CAT - 0.5, color='k', lw=0.5)
        ax.axvline(i*N_CAT - 0.5, color='k', lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs, fraction=0.02, label='1 - R')
fig.suptitle(f'Reber Fig 3 RSA ({WHICH_UNITS} units)')
plt.show()


#%% ===========================================================================
# Factorize a region. X = (100 stimuli x n_units): each STIMULUS gets a binary
# latent code S (100 x k); W (k x n_units) are the per-neuron loadings. This
# asks what binary "concept" structure the population imposes on the 100 images
# (cf. the category blocks in the RSA / the dendrograms in Fig 3).
# (To instead factorize neurons by their tuning, use X_ = mats[r] -- i.e. the
#  transpose, n_units x 100 -- but the RSA in Fig 3 is over stimuli, so default
#  observations = stimuli.)
# =============================================================================

r = 'AM'
X_ = mats[r]                          # (100, n_units)
X_ = X_ - X_.mean(0, keepdims=True)   # mean-center neurons (RSA uses correlation)

mod = bae_models.SemiBMF(10,
                         nonneg=True,
                         tree_reg=1,
                         weight_pr_reg=1,
                         weight_l2_reg=1e-2,
                         weight_l1_reg=1e-2,
                         sparse_reg=1e-1,
                         )

en = mod.fit(X_ / X_.std(),
             period=100,
             initial_temp=10,
             decay_rate=0.88,
             min_temp=1,
             scl_lr=1e-3,
             hot_start=True,
             )

# look at the learned binary codes, grouped by the 10 categories
samps = mod.sample(X_ / X_.std(), n_samp=500)
S = samps.mean(0)                     # (100, k) posterior mean code per stimulus
order = np.argsort(cat_idx)
plt.figure(figsize=(6, 8))
plt.imshow(S[order], aspect='auto', cmap='magma')
plt.yticks(np.arange(5, N_STIM, N_CAT), cats, fontsize=7)
plt.xlabel('latent dim'); plt.title(f'{r}: binary codes over stimuli')
plt.colorbar(label='P(on)')
plt.show()


#%% ---- model selection over k with imputation CV (impcv), per region -------
# impcv is the primary model-selection tool (masks entries, refits imputing
# them, scores held-out vs train loglik). Sweep latent dim per region.

kays = [2, 3, 4, 5, 6, 8, 10, 12, 15]

args = dict(nonneg=True, weight_pr_reg=1, tree_reg=1,
            sparse_reg=1e-1, weight_l1_reg=1e-2, weight_l2_reg=1e-2)
opt_args = dict(initial_temp=10, decay_rate=0.88, period=50,
                hot_start=True, min_temp=1, scl_lr=1e-4)
n_run = 5

for r in REGIONS:
    X_ = mats[r] - mats[r].mean(0, keepdims=True)
    trn = np.zeros(len(kays)); tst = np.zeros(len(kays))
    for _ in range(n_run):
        for i, k in enumerate(tqdm(kays, desc=r, leave=False)):
            mod = bae_models.SemiBMF(k, **args)
            wa, ba = bae_util.impcv(mod, X_, folds=10, n_sample=100,
                                    verbose=False, **opt_args)
            trn[i] += np.mean(wa) / n_run
            tst[i] += np.mean(ba) / n_run
    plt.plot(kays, tst, '-o', label=r)
plt.xlabel('latent dim k'); plt.ylabel('held-out loglik (impcv)')
plt.legend(); plt.title('Reber MTL: BMF model selection per region')
plt.show()
