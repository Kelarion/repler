CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/lin_data/'

import os, sys
sys.path.append(CODE_DIR)

import numpy as np
import scipy.special as spc
import pickle as pkl
from tqdm import tqdm 

import matplotlib.pyplot as plt
from matplotlib import cm

import util
import df_util
import bae_util
import bae_models
import new_bae_models
import plotting as tpl

# ----------------------------------------------------------------------------
# Lin, Nieder & Jacob (2023) Sci Adv -- "The neuronal implementation of
# representational geometry in primate PFC". Delayed-match-to-numerosity task
# with a distracting numerosity. We try to replicate Figure 3 (effect of
# distraction on sample representations).
#
# The release is laconic: two pickles, a README, no code.
#   MeanFiringRate.pkl : (467 neurons, 16 conditions, 45 time bins)
#       16 conditions = 4 sample numerosities x 4 distractor numerosities.
#       (distractor=0 / blank control trials are NOT included here.)
#       45 bins x 100 ms = 4.5 s trial, Gaussian binned (100 ms step, 50 ms sigma),
#       aligned to fixation onset.
#   FixationPeriod.pkl : finer (25 ms) rates, used only for Fig 5 autocorr.
#
# Trial structure (Methods) -> bin mapping (100 ms bins, fixation-aligned):
#   Fixation 500 ms  bins 0-4
#   Sample   500 ms  bins 5-9    (S)
#   Memory1 1000 ms  bins 10-19  (M1)
#   Distract 500 ms  bins 20-24  (D)
#   Memory2 1000 ms  bins 25-34  (M2)
#   Test    1000 ms  bins 35-44
#
# ---- This file: pipeline + Fig 3F-H (PCA), the parameter-free panels, as a
#      validation checkpoint before implementing SCA (Fig 3C-E) and demixing
#      (Fig 3A-B).
#
# NORMALIZATION: Methods z-score each tensor entry by the SD ACROSS TRIALS
# within each condition (reduced to one within-class SD scalar per neuron).
# MeanFiringRate has no single trials, BUT FixationPeriod.pkl does: it's a list
# of 467 (n_trials, 180) single-trial arrays @25 ms over the whole 4.5 s trial
# (n_trials varies 88-421). The fixation epoch (first 500 ms = first 20 bins) is
# pre-stimulus and identical across conditions, so its across-trial SD is a clean
# per-neuron noise SD ~= the within-class SD the paper divides by. Caveat: there
# are no condition labels, and naive division blows up the loadings of low-rate
# neurons (one near-zero-SD neuron drives SI -> 150), so it needs floor/soft-norm.
# Empirically, for these PCA panels the closest match to the paper's Fig 3H SI
# (1.55 / 1.44) is NO z-scoring at all -- raw, mean-centered M2 gives ~1.4 / 1.4,
# whereas per-neuron SD-across-conditions over-Gaussianizes it (~1.06 / 1.11).
# So NORM='none' is the default; 'fix' (soft fixation-SD) is provided too.
NORM = 'none'   # 'none' | 'fix' | 'cond'
# ----------------------------------------------------------------------------

# epoch -> bins
EPOCHS = {'Fix': (0, 5), 'S': (5, 10), 'M1': (10, 20),
          'D': (20, 25), 'M2': (25, 35), 'Test': (35, 45)}

N_SAMP = 4
N_DIST = 4

# Reshape order of the 16 conditions into (sample, distractor). The README does
# not specify; assume sample-major (condition = sample*4 + distractor). Flip this
# if the Fig 3F heatmaps come out transposed relative to the paper.
SAMPLE_MAJOR = True


def sparsity_index(x):
    """SI = kurtosis/3 (Eq 7-8). Gaussian -> 1, heavy-tailed -> >1."""
    x = x - x.mean()
    m2 = np.mean(x**2)
    m4 = np.mean(x**4)
    return (m4 / m2**2) / 3.0


def si_over_angles(loadings2d, n_ang=180):
    """SI of the loading distribution projected onto every axis in a 2D plane.
    loadings2d: (n_neurons, 2). Returns (angles, SI) for the rose/polar inset."""
    ang = np.linspace(0, np.pi, n_ang)
    si = np.array([sparsity_index(loadings2d @ np.array([np.cos(a), np.sin(a)]))
                   for a in ang])
    return ang, si


# ----------------------------------------------------------------------------
# Sparse component analysis (Eq 11). X (n instances x p neurons) ~= U V^T with
# U (n x k) unit-norm SC *activities* and V (p x k) sparse neuronal *loadings*:
#   Loss = ||X - U V^T||_F^2 + alpha*||V||_1 + beta*||V||_F^2,   ||u_i|| = 1
# (the per-component sums in Eq 11 are just elementwise L1 / Frobenius on V.)
# Solved by alternating minimization: V-step = elementwise elastic-net coordinate
# descent (closed-form soft-threshold); U-step = least squares + column renorm.
# beta=0.01 (paper); alpha & k by twofold CV (EM-imputation, mirroring impcv).
# ----------------------------------------------------------------------------

def _soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def sca(X, k, alpha, beta=0.01, n_restarts=8, n_iter=150,
        n_sweep=40, tol=1e-8, seed=0):
    n, p = X.shape
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        U = rng.standard_normal((n, k))
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V = np.zeros((p, k))
        prev = np.inf
        for _ in range(n_iter):
            # V-step: vectorized elastic-net coordinate descent over neurons
            G = U.T @ U                      # (k,k)
            B = X.T @ U                      # (p,k) = <x_j, u_m>
            for _ in range(n_sweep):
                for m in range(k):
                    rho = B[:, m] - V @ G[:, m] + V[:, m] * G[m, m]
                    V[:, m] = _soft(rho, alpha / 2.0) / (G[m, m] + beta)
            # U-step: least squares + unit-norm columns
            VtV = V.T @ V
            U = X @ V @ np.linalg.pinv(VtV)
            nrm = np.linalg.norm(U, axis=0, keepdims=True)
            nrm[nrm == 0] = 1.0
            U /= nrm
            loss = (np.sum((X - U @ V.T) ** 2)
                    + alpha * np.abs(V).sum() + beta * np.sum(V ** 2))
            if prev - loss < tol:
                prev = loss
                break
            prev = loss
        if best is None or prev < best[2]:
            best = (U, V, prev)
    return best  # (U, V, loss)


def sca_cv(X, k, alpha, folds=2, em_iter=12, seed=0, **kw):
    """Twofold EM-imputation CV: mask entries, refit imputing them, score held-out."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    fold = rng.integers(0, folds, size=(n, p))
    errs = []
    for f in range(folds):
        test = fold == f
        Xf = X.copy()
        Xf[test] = X[~test].mean()
        for _ in range(em_iter):
            U, V, _ = sca(Xf, k, alpha, seed=seed, **kw)
            R = U @ V.T
            Xf = np.where(test, R, X)
        errs.append(np.mean((R[test] - X[test]) ** 2))
    return np.mean(errs)


#%%
# ---- load + build the M2-averaged condition x neuron matrix ----------------

X = pkl.load(open(LOAD_DIR + 'MeanFiringRate.pkl', 'rb'))  # (467, 16, 45)
n_neur, n_cond, n_time = X.shape
assert n_cond == N_SAMP * N_DIST

# per-neuron normalization scalar (see NORM note in header)
if NORM == 'fix':
    fix = pkl.load(open(LOAD_DIR + 'FixationPeriod.pkl', 'rb'))  # list of (trials,180)@25ms
    fix_sd = np.array([a[:, :20].std() for a in fix])            # noise SD over fixation epoch
    nrm = fix_sd + np.median(fix_sd)                             # dPCA-style soft normalization
elif NORM == 'cond':
    nrm = X.reshape(n_neur, -1).std(axis=1)                      # SD across conditions+time
else:
    nrm = np.ones(n_neur)                                        # raw (mean-centering only)
Xn = X / nrm[:, None, None]

# average firing rates across the second memory delay (M2), per Fig 3C-H
m2a, m2b = EPOCHS['M2']
M2 = Xn[:, :, m2a:m2b].mean(axis=2)          # (467, 16)

# data matrix for PCA: instances (conditions) x neurons, columns mean-centered
D = M2.T                                       # (16, 467)
# D = D - D.mean(axis=0, keepdims=True)


#%%
# ---- confirm the 16-condition ordering (temporal disambiguation) -----------
# Reshape (467,16)->(467,4,4) as index = i0*4 + i1, and ask how population
# variance splits between the two reshape indices in each epoch. The distractor
# only appears at bin 20, so BEFORE that (S, M1) any tuning must be to sample.
# Result: i0 carries ~2x i1's variance during S/M1, then i1 jumps to match it
# at D/M2 -> i0 = SAMPLE (maintained throughout), i1 = DISTRACTOR (onset at D).
# Hence condition = sample*4 + distractor, i.e. SAMPLE_MAJOR = True.
for nm, (a, b) in EPOCHS.items():
    A = Xn[:, :, a:b].mean(2)
    A = A - A.mean(1, keepdims=True)
    G = A.reshape(n_neur, N_SAMP, N_DIST)
    v0 = (G.mean(2) ** 2).sum(); v1 = (G.mean(1) ** 2).sum()
    print(f'  {nm:5s}: var(i0/sample)={v0:7.2f}  var(i1/distractor)={v1:7.2f}'
          f'  ratio={v0/v1:4.2f}')


#%%
# ---- PCA via SVD (Eq 6): X = U S V^T ---------------------------------------
#   scores  = U[:, :k] * S[:k]  -> (16, k)  representational geometry  (Fig 3F,G)
#   loadings = V[:, :k]         -> (467, k) neuronal loadings          (Fig 3H)

U, S, Vt = np.linalg.svd(D, full_matrices=False)
k = 2
scores = U[:, :k] * S[:k]                      # (16, k)
loadings = Vt[:k].T                            # (467, k)

ev = (S**2) / np.sum(S**2)
print('explained variance, first 5 PCs:', np.round(ev[:5], 3))

# reshape scores to (sample, distractor) grid
if SAMPLE_MAJOR:
    grid = scores.reshape(N_SAMP, N_DIST, k)   # [sample, distractor, pc]
else:
    grid = scores.reshape(N_DIST, N_SAMP, k).transpose(1, 0, 2)


#%%
# ---- Fig 3F: PC1 / PC2 activity over the 4x4 sample x distractor grid -------

fig, axs = plt.subplots(1, k, figsize=(7, 3.2))
for i in range(k):
    ax = axs[i]
    im = ax.imshow(grid[:, :, i], origin='upper', aspect='auto', cmap='viridis')
    ax.set_title(f'PC{i+1}')
    ax.set_xlabel('Distractor'); ax.set_ylabel('Sample')
    ax.set_xticks(range(N_DIST)); ax.set_xticklabels(range(1, N_DIST+1))
    ax.set_yticks(range(N_SAMP)); ax.set_yticklabels(range(1, N_SAMP+1))
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle('Fig 3F  (PCA, M2-averaged)')
fig.tight_layout()


#%%
# ---- Fig 3G: representational geometry in PC space -------------------------
# blue grid = sample, red grid = distractor; arrows = best linear sample/distractor axes

samp_idx = np.repeat(np.arange(N_SAMP), N_DIST) if SAMPLE_MAJOR else np.tile(np.arange(N_SAMP), N_DIST)
dist_idx = np.tile(np.arange(N_DIST), N_SAMP) if SAMPLE_MAJOR else np.repeat(np.arange(N_DIST), N_SAMP)

fig, ax = plt.subplots(figsize=(5, 5))
blues = cm.Blues(np.linspace(0.4, 1, N_SAMP))
reds = cm.Reds(np.linspace(0.4, 1, N_DIST))
# connect along sample (fixed distractor) and along distractor (fixed sample)
for d in range(N_DIST):
    pts = scores[dist_idx == d]
    order = np.argsort(samp_idx[dist_idx == d])
    ax.plot(pts[order, 0], pts[order, 1], '-', color='steelblue', alpha=0.4, lw=1)
for s in range(N_SAMP):
    pts = scores[samp_idx == s]
    order = np.argsort(dist_idx[samp_idx == s])
    ax.plot(pts[order, 0], pts[order, 1], '-', color='indianred', alpha=0.4, lw=1)
ax.scatter(scores[:, 0], scores[:, 1], c=blues[samp_idx],
           edgecolors=reds[dist_idx], s=120, linewidths=2.5, zorder=3)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('Fig 3G  geometry')
ax.set_aspect('equal'); fig.tight_layout()


#%%
# ---- Fig 3H: neuronal loadings + SI over all axes --------------------------

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                      wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

ax.scatter(loadings[:, 0], loadings[:, 1], s=12, c='gray', alpha=0.6)
ax.set_xlabel('Loading on PC1'); ax.set_ylabel('Loading on PC2')
ax_top.hist(loadings[:, 0], bins=40, color='gray'); ax_top.axis('off')
ax_right.hist(loadings[:, 1], bins=40, orientation='horizontal', color='gray')
ax_right.axis('off')

print('SI on PC1:', round(sparsity_index(loadings[:, 0]), 2),
      ' SI on PC2:', round(sparsity_index(loadings[:, 1]), 2))

# polar inset: SI as a function of axis orientation
ang, si = si_over_angles(loadings)
axp = fig.add_axes([0.62, 0.62, 0.3, 0.3], projection='polar')
axp.plot(np.r_[ang, ang + np.pi], np.r_[si, si], color='magenta')
axp.set_title('SI(axis)', fontsize=8)
fig.suptitle('Fig 3H  (PCA)')

plt.show()


#%%
# ===========================================================================
# SCA on the M2-averaged data (Fig 3C-E). Same X matrix as PCA (D, 16 x 467).
# Paper found k=2 SCs here; alpha by twofold CV grid search.
# ===========================================================================

k_sca = 2
alpha_grid = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
cv = np.array([sca_cv(D, k_sca, a, n_restarts=4, seed=0) for a in alpha_grid])
alpha = alpha_grid[np.argmin(cv)]
print('alpha CV MSE:', dict(zip(alpha_grid.round(3), cv.round(4))))
print('selected alpha =', alpha)

Usc, Vsc, _ = sca(D-D.mean(0), k_sca, alpha, n_restarts=16, seed=0)   # U:(16,2) act, V:(467,2) load
sca_ev = 1 - np.sum((D - D.mean(0) - Usc @ Vsc.T) ** 2) / np.sum(D ** 2)
print('SCA reconstruction EV (k=2):', round(sca_ev, 3))
print('SI on SC1:', round(sparsity_index(Vsc[:, 0]), 2),
      ' SI on SC2:', round(sparsity_index(Vsc[:, 1]), 2),
      '  (paper Fig 3E: 2.37 / 1.63)')


#%%
# ---- ordering diagnostic ---------------------------------------------------
# The two candidate orderings (sample-major vs distractor-major) are transposes
# of the 4x4 grid. For each, fit an additive model  a_sample + b_distractor  to
# every SC's activity and report how much variance it captures (grid-ness) and
# how the variance splits between the sample and distractor marginals. The factor
# carrying the larger, cleaner gradient should be the (behaviourally relevant)
# sample numerosity -- that fixes SAMPLE_MAJOR.

def grid_report(act_16, name):
    print(f'\n[{name}]')
    for major in (True, False):
        g = (act_16.reshape(N_SAMP, N_DIST) if major
             else act_16.reshape(N_DIST, N_SAMP).T)            # rows=sample,cols=distr
        rmarg = g.mean(1); cmarg = g.mean(0); gm = g.mean()
        add = rmarg[:, None] + cmarg[None, :] - gm             # additive prediction
        r2 = 1 - np.sum((g - add) ** 2) / np.sum((g - gm) ** 2)
        print(f'  SAMPLE_MAJOR={major!s:5}  additive R2={r2:.3f}  '
              f'range(sample)={np.ptp(rmarg):.2f}  range(distr)={np.ptp(cmarg):.2f}')

for i in range(k_sca):
    grid_report(Usc[:, i], f'SC{i+1}')


#%%
# ---- Fig 3C: SC1 / SC2 activity over the 4x4 sample x distractor grid -------

if SAMPLE_MAJOR:
    grid_sc = Usc.reshape(N_SAMP, N_DIST, k_sca)
else:
    grid_sc = Usc.reshape(N_DIST, N_SAMP, k_sca).transpose(1, 0, 2)

fig, axs = plt.subplots(1, k_sca, figsize=(7, 3.2))
for i in range(k_sca):
    ax = axs[i]
    im = ax.imshow(grid_sc[:, :, i], origin='upper', aspect='auto', cmap='viridis')
    ax.set_title(f'SC{i+1}'); ax.set_xlabel('Distractor'); ax.set_ylabel('Sample')
    ax.set_xticks(range(N_DIST)); ax.set_xticklabels(range(1, N_DIST+1))
    ax.set_yticks(range(N_SAMP)); ax.set_yticklabels(range(1, N_SAMP+1))
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle('Fig 3C  (SCA, M2-averaged)'); fig.tight_layout()


#%%
# ---- Fig 3D: representational geometry in SC space -------------------------

fig, ax = plt.subplots(figsize=(5, 5))
for d in range(N_DIST):
    pts = Usc[dist_idx == d]; order = np.argsort(samp_idx[dist_idx == d])
    ax.plot(pts[order, 0], pts[order, 1], '-', color='steelblue', alpha=0.4, lw=1)
for s in range(N_SAMP):
    pts = Usc[samp_idx == s]; order = np.argsort(dist_idx[samp_idx == s])
    ax.plot(pts[order, 0], pts[order, 1], '-', color='indianred', alpha=0.4, lw=1)
ax.scatter(Usc[:, 0], Usc[:, 1], c=blues[samp_idx],
           edgecolors=reds[dist_idx], s=120, linewidths=2.5, zorder=3)
ax.set_xlabel('SC1'); ax.set_ylabel('SC2'); ax.set_title('Fig 3D  geometry')
ax.set_aspect('equal'); fig.tight_layout()


#%%
# ---- Fig 3E: neuronal loadings on the two SCs + SI -------------------------

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                      wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
ax.scatter(Vsc[:, 0], Vsc[:, 1], s=12, c='gray', alpha=0.6)
ax.set_xlabel('Loading on SC1'); ax.set_ylabel('Loading on SC2')
ax_top.hist(Vsc[:, 0], bins=40, color='gray'); ax_top.axis('off')
ax_right.hist(Vsc[:, 1], bins=40, orientation='horizontal', color='gray'); ax_right.axis('off')
ang, si = si_over_angles(Vsc)
axp = fig.add_axes([0.62, 0.62, 0.3, 0.3], projection='polar')
axp.plot(np.r_[ang, ang + np.pi], np.r_[si, si], color='magenta')
axp.set_title('SI(axis)', fontsize=8)
fig.suptitle('Fig 3E  (SCA)')

plt.show()

#%%

# X_ = D - D.mean(0)
X_ = D
# X_ = D / D.std(0)

# mod = bae_models.SemiBMF(8,
#                          nonneg=True, 
#                          tree_reg=1,

#                          weight_pr_reg=1,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=0,
#                          sparse_reg=1,
#                          )

# mod = bae_models.SemiBMF(4,
#                          nonneg=True,
#                          # nonneg=False,
#                          # tree_reg=1e-1,
#                          tree_reg=0,
#                          weight_pr_reg=1,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=1e-2,
#                          # sparse_reg=1e-2,
#                          sparse_reg=0,
#                          # fit_intercept=False,
#                          )

mod = new_bae_models.JBMF(6,
                         nonneg=True,
                         # nonneg=False,
                         # fit_intercept=False,
                         tree_reg=0,
                         weight_pr_reg=1,
                         weight_l2_reg=1e-1,
                         weight_l1_reg=1e-3,
                         sparse_reg=1e-2,
                         # J_loss='mle',
                         J_loss='rple',
                         # J_l1_reg=0.2,
                         J_lr=1e-4,
                         # slab=True,
                         # slab_prior=1,
                         )

# mod = bae_models.SpikeNMF(11,
#                          nonneg=True, 
#                          sparse_reg=1,
#                          weight_pr_reg=1,
#                          tree_reg=0,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=0,
#                          )

# mod = bae_models.SpikeNMF(2,
#                          nonneg=False, 
#                          sparse_reg=0,
#                          weight_pr_reg=0,
#                          tree_reg=0,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=1e-2,
#                          fit_intercept=False,
#                          slab_prior=1,
#                          )

# mod = bae_models.KernelBMF2(12,
#                            sparse_reg=1,
#                            tree_reg=100,
#                            uniform_scale=False,
#                            # l1_reg=0,
#                            )

en = mod.fit(X_  / X_.std(),
             period=100, 
             initial_temp=100,
             decay_rate=0.88,
             min_temp=1, 
             scl_lr=1e-4,
             # W_lr=0.05,
             # b_lr=0.05,
             lr=1e-2,
             # hot_start=False,
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
# samps = mod.sample(X_ / X_.std(), n_samp=1000, slab=False)
# 
# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

for k in range(mod.dim_hid):
    
    plt.subplot(1, mod.dim_hid, k+1)
    plt.imshow(samps.mean(0)[:,k].reshape((4,4)))

#%%

# kays = [2,3,4,5,10,15,20]
kays = [1,2,3,4,5,6,7,8,9,10]
# kays = [2,4,6,8,10,12,14,16]
# kays = [10]

args = {
        'nonneg':True,
        # 'nonneg': False,
        'weight_pr_reg': 1,
        # 'weight_pr_reg': 0,
        'tree_reg': 1,
        'sparse_reg': 1e-1,
        # 'tree_reg': 0,
        # 'weight_l1_reg': 0,
        'weight_l1_reg': 1e-2,
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
            'decay_rate': 0.88,
            'period': 50,
            'hot_start': True,
            'min_temp': 1,
            # 'hot_start': False,
            'scl_lr': 1e-4,
            # 'scl_lr': 0,
            # 'lr': 1e-1,
            }

n_run = 20
modclass = bae_models.SemiBMF
# modclass = bae_models.SpikeNMF

full = np.zeros(len(kays))
trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
         
        # mod = bae_models.SemiBMF(k,**args)
        mod = modclass(k,**args)
        
        wa,ba = bae_util.impcv(mod, X_, folds=10, n_sample=100, verbose=False, **opt_args)
        # wa, ba = bae_util.loocv(mod, X_ / X_.std(), n_sample=10, **opt_args)
        # wa, ba = bae_util.gabriel_bicv(mod, X_/X_.std(), n_samp=1000, **opt_args)
        
        trn[i] += np.mean(wa) / n_run
        tst[i] += np.mean(ba) / n_run
        
        mod = modclass(k,**args)
        en = mod.fit(X_ , verbose=False, **opt_args)
        samps = mod.sample(X_ , n_samp=10)
        full[i] += mod.loglikelihood(X_ , mod(samps)).mean() / n_run

plt.plot(kays, trn)
plt.plot(kays, tst, '--')

#%%

rep = X_
# rep = Usc
signal = 0
noise = 1
rep = signal*S@np.random.randn(6, X_.shape[-1]) + noise*np.random.randn(*X_.shape)

ps = np.zeros((6,6))
for i in range(6):
    for j in range(6):
        
        edges = 1*((util.yuke(S) == 1) * (S[:,[i]] != S[:,[i]].T))
        aye_i, jay_i = np.where(np.triu(edges))
        
        edges = 1*((util.yuke(S) == 1) * (S[:,[j]] != S[:,[j]].T))
        aye_j, jay_j = np.where(np.triu(edges))
        
        cs = util.cosine_sim((rep[aye_i] - rep[jay_i]).T, (rep[aye_j] - rep[jay_j]).T)
        
        if i == j:
            ps[i,j] = np.sum(np.triu(cs, k=1)) / spc.binom(len(cs),2)
        else:
            ps[i,j] = np.trace(cs) / len(cs)


plt.figure()
plt.imshow(ps, 'bwr', vmin=-1, vmax=1)

#%%

plt.scatter(Usc[aye_i,0], Usc[aye_i,1])
plt.scatter(Usc[jay_i,0], Usc[jay_i,1])


plt.quiver(Usc[jay_i,0], Usc[jay_i,1], 
           Usc[aye_i,0]-Usc[jay_i,0], 
           Usc[aye_i,1]-Usc[jay_i,1], 
           scale=1, scale_units='xy',
           angles='xy')

#%%
plt.scatter(Usc[aye_j,0], Usc[aye_j,1])
plt.scatter(Usc[jay_j,0], Usc[jay_j,1])


plt.quiver(Usc[jay_j,0], Usc[jay_j,1], 
           Usc[aye_j,0]-Usc[jay_j,0], 
           Usc[aye_j,1]-Usc[jay_j,1], 
           scale=1, scale_units='xy',
           angles='xy')

#%%



