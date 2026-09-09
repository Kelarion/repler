"""
Guan et al. 2023 (J. Neural Eng. 20 036020) -- Figure 8.

    "Compositional coding of individual finger movements in human posterior
     parietal cortex and motor cortex enables ten-finger decoding"

DANDI:000252 -- "Finger_RL: human intracortical recordings during attempted
finger movements of right and left hands".

Subject / participant mapping (verified from the NWB metadata, NOT the folder
names -- they do not match the paper's initials):

    sub-P1  ==  participant "NS"  (tetraplegic woman)
                one Utah array in PPC at PC-IP (junction of postcentral and
                intraparietal sulci).  10 sessions.   -> this is "NS-PPC".
    sub-N1  ==  participant "JJ"  (tetraplegic man)
                two arrays: MC (hand knob) + PPC (SPL).  2 sessions.
                -> "JJ-MC" and "JJ-PPC".

Figure 8 is the *NS-PPC* representation, i.e. sub-P1, PPC units, 10 sessions,
during the "ten-finger press task, with delay" (the contralateral/ipsilateral
= bilateral finger task).

This script builds, for each session, the neural representation of the 10
finger movements (Lt,Li,Lm,Lr,Lp,Rt,Ri,Rm,Rr,Rp) and reproduces:

    Fig 8(a)  cross-validated squared Mahalanobis (crossnobis) RDM, 10x10,
              averaged over the 10 sessions.
    Fig 8(b)  matching (same finger-type, different hand) vs non-matching
              finger-pair distances.
    Fig 8(d)  2-D MDS of the representation, showing the factorized geometry
              (parallel left->right hand vectors), with S.E. ellipses.
              <-- the panel of interest.

              IMPORTANT reproduction note: the paper's literal recipe is
              "MDS each session, then Generalized Procrustes to align across
              sessions".  That does NOT reproduce the published geometry from
              these single-session distances -- the left<->right (hand) offset
              is a small, sign-unstable fraction of each session's variance, so
              independent per-session 2D MDS fixes that weak axis by noise and
              GPA averaging cancels it, collapsing the hand dimension to ~0.
              The distances themselves are fine (Fig 8a reproduces); the
              structure only survives when MDS is run on the session-AVERAGED
              RDM (which is what Fig 8d "corresponds to distances (a)" means).
              We therefore take the consensus geometry from the mean RDM and
              estimate the S.E. ellipses with a leave-one-session-out jackknife.
              See `mds_consensus`.

Methods details used (from the paper):
  * Movement-execution ("Go") analysis window = the 500 ms window starting
    200 ms after the Go cue:  [go_on_time + 0.2, go_on_time + 0.7].
  * Firing rate = spike count in the window / window duration.
  * Cross-validated (squared) Mahalanobis distance with multivariate noise
    normalisation (rsatoolbox-style crossnobis); cross-validation folds are
    the experimental Runs so that E[d^2]=0 for identical patterns.
"""

CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/dandisets/'

import os, sys, re
import warnings
import pickle as pkl
sys.path.append(CODE_DIR)

from tqdm import tqdm 

import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
import scipy.special as spc
import scipy.stats as sts
from scipy.linalg import orthogonal_procrustes
from itertools import combinations

from sklearn.manifold import MDS
from sklearn.covariance import ledoit_wolf

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pynwb import NWBHDF5IO

import util
import df_util
import bae_util
import old_bae_models
import bae_models
import plotting as tpl

import bae_models

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DSET = '000252'
SUBJ = 'sub-P1'          # == participant "NS", single PPC (PC-IP) array
GROUP = 'PPC'            # electrode group to keep

GO_T0 = 0.2              # window start, s after Go cue
GO_T1 = 0.7             # window end,   s after Go cue

# canonical finger ordering: left hand thumb->pinky, then right hand thumb->pinky
FINGER_TYPES = ['t', 'i', 'm', 'r', 'p']
FINGER_ORDER = ['L' + f for f in FINGER_TYPES] + ['R' + f for f in FINGER_TYPES]
FINGER_IDX = {f: i for i, f in enumerate(FINGER_ORDER)}
NICE_LABEL = {'t': 'thumb', 'i': 'index', 'm': 'middle', 'r': 'ring', 'p': 'pinky'}

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def session_files(subj=SUBJ):
    """Sorted list of (session_id, path) for a subject."""
    d = f"{LOAD_DIR}/{DSET}/{subj}"
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.nwb'):
            continue
        sess = re.findall(r'ses-([0-9\-]+)', fn)[0]
        out.append((sess, os.path.join(d, fn)))
    return out


def load_session(path, group=GROUP, t0=GO_T0, t1=GO_T1):
    """
    Load one session and build the single-trial firing-rate representation.

    Returns dict with:
        X        : (n_trials, n_units) firing rates in the Go window
        finger   : (n_trials,) canonical finger index 0..9 (see FINGER_ORDER)
        run      : (n_trials,) run id (used as the crossnobis CV partition)
        n_units  : int
    """
    io = NWBHDF5IO(path, mode='r')
    nwb = io.read()

    df = nwb.trials.to_dataframe()

    # keep only units on the requested array
    grp = np.array([g.name if hasattr(g, 'name') else g
                    for g in nwb.units['electrode_group'][:]])
    keep = np.where(grp == group)[0]

    go = df['go_on_time'].to_numpy()
    bins0 = go + t0
    bins1 = go + t1
    dur = t1 - t0

    n_units = len(keep)
    n_tri = len(df)
    X = np.zeros((n_tri, n_units))
    for j, u in enumerate(keep):
        st = np.asarray(nwb.units['spike_times'][u])
        # count spikes per trial window
        lo = np.searchsorted(st, bins0)
        hi = np.searchsorted(st, bins1)
        X[:, j] = (hi - lo) / dur

    finger = np.array([FINGER_IDX[f] for f in df['finger']])
    run = df['Run'].to_numpy().astype(int)

    io.close()
    return {'X': X, 'finger': finger, 'run': run, 'n_units': n_units}


# ---------------------------------------------------------------------------
# cross-validated (squared) Mahalanobis distance  (crossnobis)
# ---------------------------------------------------------------------------

def _noise_precision(X, labels):
    """Ledoit-Wolf shrunk inverse noise covariance from within-condition residuals."""
    resid = np.zeros_like(X)
    for c in np.unique(labels):
        m = labels == c
        resid[m] = X[m] - X[m].mean(0, keepdims=True)
    cov, _ = ledoit_wolf(resid)
    # symmetric prewhitening matrix E s.t. E @ E.T = inv(cov)
    return nla.pinv(cov)


def crossnobis_rdm(X, labels, folds, n_cond=10):
    """
    Cross-validated squared Mahalanobis distance matrix (Walther et al. 2016).

    d^2_jk = mean_{a != b}  (mu_j^a - mu_k^a)^T  Sigma^-1  (mu_j^b - mu_k^b)

    where a,b index CV partitions (`folds`), mu_j^a is the mean pattern of
    condition j within partition a, and Sigma is the shrinkage noise covariance
    estimated across all trials.  Cross-validation over partitions makes the
    estimate unbiased: E[d^2]=0 for statistically identical patterns.
    """
    prec = _noise_precision(X, labels)          # Sigma^-1
    n_neur = X.shape[1]                          # for per-neuron normalisation (paper's N)
    uf = np.unique(folds)
    # per-fold, per-condition mean patterns
    mus = {}
    for a in uf:
        for c in range(n_cond):
            m = (folds == a) & (labels == c)
            if m.sum() == 0:
                mus[(a, c)] = None
            else:
                mus[(a, c)] = X[m].mean(0)

    D = np.zeros((n_cond, n_cond))
    fold_pairs = [(a, b) for a in uf for b in uf if a != b]
    for j, k in combinations(range(n_cond), 2):
        vals = []
        for a, b in fold_pairs:
            if mus[(a, j)] is None or mus[(a, k)] is None:
                continue
            if mus[(b, j)] is None or mus[(b, k)] is None:
                continue
            da = mus[(a, j)] - mus[(a, k)]
            db = mus[(b, j)] - mus[(b, k)]
            vals.append(da @ prec @ db)
        d = np.mean(vals) if vals else 0.0
        D[j, k] = D[k, j] = d / n_neur          # unitless^2 / neuron
    return D


# ---------------------------------------------------------------------------
# MDS + cross-session alignment
#
# NOTE on reproducing Fig 8(d) faithfully -- see the module docstring / the
# `mds_consensus` docstring below.  The paper's literal recipe ("MDS on each
# session, then Generalized Procrustes to align") does NOT reproduce the
# published geometry from these single-session distances, because the hand
# (left<->right) offset is a small, sign-unstable fraction of each session's
# variance.  Independent per-session 2D MDS fixes that weak axis by noise, and
# GPA averaging then cancels it, collapsing the hand dimension to ~0.  The
# factorized geometry only survives when MDS is computed on the *session-
# averaged* RDM (which is what Fig 8d "corresponds to distances (a)" means).
# We therefore take the consensus geometry from the mean RDM and estimate the
# S.E. ellipses with a leave-one-session-out jackknife.
# ---------------------------------------------------------------------------

def rdm_to_mds(D, n_comp=2, seed=0):
    """2-D metric MDS from a (possibly slightly negative) squared-distance RDM."""
    dist = np.sqrt(np.clip(D, 0, None))
    np.fill_diagonal(dist, 0.0)
    mds = MDS(n_components=n_comp, dissimilarity='precomputed',
              random_state=seed, normalized_stress=False, n_init=12, max_iter=800)
    return mds.fit_transform(dist)


def _standardize(Y):
    """center and scale to unit Frobenius norm."""
    Y = Y - Y.mean(0, keepdims=True)
    n = nla.norm(Y)
    return Y / n if n > 0 else Y


def procrustes_align(Y, ref):
    """Align config Y to reference `ref` with rotation/reflection + isotropic scale."""
    Y = _standardize(Y)
    R, _ = orthogonal_procrustes(Y, ref)      # min ||Y R - ref||, R orthogonal (incl. reflection)
    Yr = Y @ R
    denom = np.sum(Yr * Yr)
    sc = np.sum(Yr * ref) / denom if denom > 0 else 1.0
    return Yr * sc


def mds_consensus(mean_rdm, per_session_rdms, seed=0):
    """
    Consensus 2-D geometry (Fig 8d) with leave-one-session-out jackknife S.E.

    Returns
        consensus  : (n_cond, 2) MDS of the session-averaged RDM (the template).
        jack       : (n_sess, n_cond, 2) jackknife embeddings, each = MDS of the
                     mean RDM over all-but-one session, aligned to `consensus`.
        se         : (n_cond, 2) jackknife standard error per point/axis.
    """
    consensus = _standardize(rdm_to_mds(mean_rdm, seed=seed))
    rdms = np.asarray(per_session_rdms)
    n = len(rdms)
    jack = np.stack([procrustes_align(rdm_to_mds(np.delete(rdms, i, 0).mean(0), seed=seed),
                                      consensus)
                     for i in range(n)])
    # jackknife S.E.:  sqrt( (n-1)/n * sum_i (x_i - xbar)^2 )
    se = np.sqrt((n - 1) / n * np.sum((jack - consensus) ** 2, 0))
    return consensus, jack, se


def cov_ellipse(mean_xy, points, ax, **kw):
    """Draw the (jackknife) S.E. ellipse for one point from its jackknife cloud."""
    d = points - mean_xy
    n = len(points)
    cse = (n - 1) / n * (d.T @ d)             # jackknife covariance of the mean
    vals, vecs = nla.eigh(cse)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(mean_xy, w, h, angle=ang, **kw))


# ---------------------------------------------------------------------------
# pseudo-population (Fig 8e decoding / CCGP)
#
# Paper methods (2.7 neuron-dropping, 2.11 factorized coding): neurons are
# aggregated across sessions into a "pseudo-population".  Because the two hands
# / ten fingers are the *same* conditions in every session, single trials are
# combined *by within-finger order* -- e.g. each session's first right-thumb
# trial is combined into one pseudo-trial, each session's second right-thumb
# trial into the next, and so on.  Neurons from all sessions are concatenated
# along the unit axis.  For NS this yields 100 pseudo-trials (10 fingers x 10
# reps) x 1114 units.  If sessions had unequal trial counts, only the first
# min-over-sessions trials of each finger are used (their "first 96 trials"
# rule).
# ---------------------------------------------------------------------------

def build_pseudopop(sessions, n_cond=10, rng=None):
    """
    Aggregate neurons across sessions into a pseudo-population.

    Parameters
    ----------
    sessions : list of per-session dicts with keys 'X' (n_trials, n_units),
               'finger' (n_trials,) canonical finger idx, and 'sess'.
    rng      : optional int seed or np.random.Generator.  If given, the
               within-finger trial order of *each* session is permuted before
               pooling -> one random pseudo-population resample.  (The trial
               pairing across sessions is arbitrary, so drawing different
               permutations is how you generate many pseudo-population
               instantiations for resampling / the permutation null.)  If None,
               trials are pooled in their natural within-session appearance
               order (the paper's literal recipe).

    Returns dict:
        X            : (n_cond*reps, n_units_total) pseudo-population firing rates
        finger       : (n_cond*reps,) canonical finger index 0..n_cond-1
        hand         : (n_cond*reps,) 0 = Left, 1 = Right
        ftype        : (n_cond*reps,) finger-type index 0..4  (t,i,m,r,p)
        rank         : (n_cond*reps,) within-finger trial rank; use as the CV
                       fold id so each physical trial stays in one fold.
        n_units      : int, total pooled units
        unit_session : (n_units_total,) session index each unit came from
        sessions     : list of session ids
    """
    if rng is not None and not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    n_sess = len(sessions)

    # within-finger trial indices per session (optionally permuted)
    sel = []                      # sel[s][c] = array of trial rows for finger c
    reps = np.inf
    for s in sessions:
        per_c = []
        for c in range(n_cond):
            idx = np.where(s['finger'] == c)[0]        # appearance order
            if rng is not None:
                idx = idx[rng.permutation(len(idx))]
            per_c.append(idx)
            reps = min(reps, len(idx))
        sel.append(per_c)
    reps = int(reps)

    # pool
    finger = np.repeat(np.arange(n_cond), reps)
    rank = np.tile(np.arange(reps), n_cond)
    rows = []
    for c in range(n_cond):
        for r in range(reps):
            rows.append(np.concatenate([sessions[s]['X'][sel[s][c][r]]
                                        for s in range(n_sess)]))
    X = np.asarray(rows)

    unit_session = np.concatenate([np.full(s['X'].shape[1], i)
                                   for i, s in enumerate(sessions)])
    return {
        'X': X,
        'finger': finger,
        'hand': (finger >= n_cond // 2).astype(int),   # first half = Left
        'ftype': finger % (n_cond // 2),
        'rank': rank,
        'n_units': X.shape[1],
        'unit_session': unit_session,
        'sessions': [s['sess'] for s in sessions],
    }


# ---------------------------------------------------------------------------
# CCGP (Fig 8e): cross-condition generalization of hand / finger-type decoders
# ---------------------------------------------------------------------------

def _fit_acc(Xtr, ytr, Xte, yte, C=1.0):
    """Linear-SVM train/test accuracy (fresh z-scoring from the training set)."""
    from sklearn.svm import LinearSVC
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    clf = LinearSVC(C=C, dual='auto', max_iter=5000)
    clf.fit((Xtr - mu) / sd, ytr)
    return clf.score((Xte - mu) / sd, yte)


def finger_ccgp(pp, C=1.0):
    """
    Finger-type CCGP: train the 5-way finger-type decoder on one hand, test on
    the other hand (both train/test directions, averaged).  High accuracy =>
    finger-type code is shared across hands (factorized).
    """
    X, ft, hand = pp['X'], pp['ftype'], pp['hand']
    accs = []
    for htr, hte in [(0, 1), (1, 0)]:
        accs.append(_fit_acc(X[hand == htr], ft[hand == htr],
                             X[hand == hte], ft[hand == hte], C))
    return np.mean(accs)


def hand_ccgp(pp, C=1.0):
    """
    Hand CCGP: train the Left-vs-Right decoder on a subset of finger-types and
    test on the held-out finger-type (leave-one-finger-type-out, averaged).
    High accuracy => hand code generalizes across fingers (factorized).
    """
    X, ft, hand = pp['X'], pp['ftype'], pp['hand']
    accs = []
    for t in np.unique(ft):
        tr, te = ft != t, ft == t
        accs.append(_fit_acc(X[tr], hand[tr], X[te], hand[te], C))
    return np.mean(accs)


def ccgp_permtest(pp, stat_fn, n_perm=1001, C=1.0, seed=0):
    """
    Permutation null for a CCGP statistic (paper: N=1001 label shuffles).
    Returns (observed, null_samples, p_value).  Labels are shuffled *within*
    the grouping the decoder generalizes across so the null respects structure:
    finger_ccgp shuffles finger-type within hand; hand_ccgp shuffles hand
    within finger-type.
    """
    rng = np.random.default_rng(seed)
    obs = stat_fn(pp, C=C)
    if stat_fn is finger_ccgp:
        lab, grp = 'ftype', 'hand'
    else:
        lab, grp = 'hand', 'ftype'
    null = np.empty(n_perm)
    for i in range(n_perm):
        sh = dict(pp)
        y = pp[lab].copy()
        for gv in np.unique(pp[grp]):
            m = pp[grp] == gv
            y[m] = rng.permutation(y[m])
        sh[lab] = y
        null[i] = stat_fn(sh, C=C)
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return obs, null, p


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

#%%  load all NS-PPC sessions and compute per-session representations + RDMs


files = session_files(SUBJ)
print(f"{SUBJ}: {len(files)} sessions")

sessions = []      # per-session dicts
rdms = []          # per-session 10x10 crossnobis RDM

for sess, path in files:
    S = load_session(path)
    D = crossnobis_rdm(S['X'], S['finger'], S['run'], n_cond=len(FINGER_ORDER))

    # condition-mean representation (10 x n_units): the "neural representation"
    cond_mean = np.stack([S['X'][S['finger'] == c].mean(0)
                          for c in range(len(FINGER_ORDER))])
    cond_median = np.stack([np.median(S['X'][S['finger'] == c], axis=0)
                          for c in range(len(FINGER_ORDER))])

    S.update(sess=sess, rdm=D, cond_mean=cond_mean, cond_median=cond_median)
    sessions.append(S)
    rdms.append(D)
    print(f"  {sess}: {S['n_units']} PPC units, {len(S['finger'])} trials")

rdms = np.array(rdms)
mean_rdm = rdms.mean(0)

# consensus geometry for Fig 8d, with leave-one-session-out jackknife S.E.
# (see mds_consensus docstring for why this replaces literal per-session GPA)
mds_mean, mds_jack, mds_se = mds_consensus(mean_rdm, rdms, seed=0)

# package the representation for downstream use
representation = {
    'finger_order': FINGER_ORDER,
    'subject': SUBJ,
    'region': 'NS-PPC',
    'sessions': [s['sess'] for s in sessions],
    'cond_mean': [s['cond_mean'] for s in sessions],  # list of (10, n_units); n_units varies by session
    'cond_median': [s['cond_median'] for s in sessions],
    'rdm_per_session': rdms,
    'rdm_mean': mean_rdm,
    'mds_mean': mds_mean,         # (10,2) consensus MDS of the mean RDM
    'mds_jack': mds_jack,         # (n_sess,10,2) leave-one-out jackknife embeddings
    'mds_se': mds_se,             # (10,2) jackknife S.E. per point/axis
}
with open(f"{SAVE_DIR}/guan_fig8_NS-PPC.pkl", 'wb') as f:
    pkl.dump(representation, f)
print(f"saved -> {SAVE_DIR}/guan_fig8_NS-PPC.pkl")

# -----------------------------------------------------------------------
#%%  Figure 8(a): mean crossnobis RDM
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(mean_rdm, cmap='viridis')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(FINGER_ORDER, rotation=90); ax.set_yticklabels(FINGER_ORDER)
ax.set_title('Fig 8(a)  NS-PPC crossnobis distances\n(mean over sessions)')
fig.colorbar(im, ax=ax, label=r'$d^2$ (unitless$^2$/neuron)')
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/guan_fig8a_rdm.png", dpi=150)

# -----------------------------------------------------------------------
#%%  Figure 8(b): matching vs non-matching finger pairs
# -----------------------------------------------------------------------
def hand(i):  return FINGER_ORDER[i][0]
def ftype(i): return FINGER_ORDER[i][1]

match_vals, nonmatch_vals = [], []
for D in rdms:
    for j, k in combinations(range(10), 2):
        if hand(j) == hand(k):
            continue                       # only cross-hand pairs
        if ftype(j) == ftype(k):
            match_vals.append(D[j, k])
        else:
            nonmatch_vals.append(D[j, k])
match_vals = np.array(match_vals); nonmatch_vals = np.array(nonmatch_vals)
diff = nonmatch_vals.mean() - match_vals.mean()
print(f"Fig 8b: matching mean={match_vals.mean():.2f}, "
      f"non-matching mean={nonmatch_vals.mean():.2f}, "
      f"difference={diff:.2f}  (paper: 1.56)")

fig, ax = plt.subplots(figsize=(4, 4.4))
for x, v, c in [(0, match_vals, 'tab:blue'), (1, nonmatch_vals, 'tab:red')]:
    ax.scatter(np.full_like(v, x) + np.random.uniform(-.08, .08, len(v)),
               v, s=10, alpha=.4, color=c)
    ax.hlines(v.mean(), x - .25, x + .25, color='k', lw=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(['matching', 'non-matching'])
ax.set_ylabel(r'cross-hand $d^2$')
ax.set_title('Fig 8(b)')
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/guan_fig8b_pairs.png", dpi=150)

# -----------------------------------------------------------------------
#%%  Figure 8(d): consensus MDS geometry of the NS-PPC representation
# -----------------------------------------------------------------------
# The paper plots only index/middle/ring "for visual clarity"; set
# PLOT_FINGERS to FINGER_TYPES to show all ten.
PLOT_FINGERS = ['i', 'm', 'r']

colors = {'t': '#d62728', 'i': '#00bfff', 'm': '#2ca02c',
          'r': '#e6b800', 'p': '#ff7f0e'}
fig, ax = plt.subplots(figsize=(5.2, 6))

for ft in PLOT_FINGERS:
    for hnd, marker in [('L', '^'), ('R', 'o')]:   # triangle=L, circle=R (paper convention)
        i = FINGER_IDX[hnd + ft]
        ax.scatter(*mds_mean[i], s=200, marker=marker,
                   color=colors[ft], edgecolor='k', zorder=3)
        cov_ellipse(mds_mean[i], mds_jack[:, i, :], ax,
                    facecolor=colors[ft], alpha=0.25, edgecolor='none')
        ax.annotate(hnd + ft, mds_mean[i], textcoords='offset points',
                    xytext=(7, 5), fontsize=10)

# left->right (hand) vectors per finger-type: parallel/identical => factorized
for ft in PLOT_FINGERS:
    l = FINGER_IDX['L' + ft]; r = FINGER_IDX['R' + ft]
    ax.annotate('', xy=mds_mean[l], xytext=mds_mean[r],
                arrowprops=dict(arrowstyle='->', color=colors[ft], lw=2))

ax.set_aspect('equal')
ax.set_title('Fig 8(d)  NS-PPC\nconsensus MDS; triangles=L, circles=R; '
             'ellipses=jackknife S.E.')
ax.set_xlabel('MDS 1'); ax.set_ylabel('MDS 2')
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[ft],
                  markeredgecolor='k', markersize=10, label=NICE_LABEL[ft])
           for ft in PLOT_FINGERS]
ax.legend(handles=handles, loc='best', fontsize=8)
fig.tight_layout()
fig.savefig(f"{SAVE_DIR}/guan_fig8d_mds.png", dpi=150)
print(f"saved figures -> {SAVE_DIR}/guan_fig8[a,b,d]_*.png")

plt.show()

#%%
pp = build_pseudopop(sessions, n_cond=10)

deez = pp['X'].sum(0) > 0

# trainset = np.concatenate([np.where(pp['finger']==i)[0][np.random.permutation(10)[:5]] for i in range(10)])
# testset = np.setdiff1d(np.arange(100), trainset)

# Xtrn = util.group_mean(pp['X'][trainset], pp['finger'][trainset], axis=0)[:,deez]
# Xtst = util.group_mean(pp['X'][testset], pp['finger'][testset], axis=0)[:,deez]

#%%

# X_ = Xtrn
X_ = pp['X']
# X_ = np.hstack(representation['cond_mean'])
# X_ = np.hstack(representation['cond_median'])
X_ = X_[:,X_.sum(0) > 0]

# X_ = X_ - X_.mean(0)

# X_ = util.embed(util.center(X_@X_.T))

# X_ /= X_.std(0)

# X_ = util.embed(-util.center(representation['rdm_mean']))
# X_ = -util.center(representation['rdm_mean']) * 5

# mod = bae_models.JBMF(5,
#                          nonneg=True,
#                          # nonneg=False,
#                          # fit_intercept=False,
#                          tree_reg=0,
#                          weight_pr_reg=1,
#                          weight_l2_reg=1e-1,
#                          weight_l1_reg=0,
#                          sparse_reg=1,
#                          # J_loss='mle',
#                          # J_l1_reg=0.1,
#                          # J_lr=1e-3,
#                          J_lr=0,
#                          # slab=True,
#                          # slab_prior=0.1,
#                          )

mod = bae_models.BiPCA(4,
                           # fit_intercept=False,
                           tree_reg=0,
                           sparse_reg=1,
                           # J_loss='mle',
                           # J_l1_reg=0.2,
                           J_lr=1e-3,
                           # J_lr=0,
                           # fit_scl=False,
                           # slab=True,
                           # saem=True,
                           # gamma=1e-1,
                           n_chains=8,
                           )

# mod = bae_models.KernelBMF(5,
#                                sparse_reg=1,
#                                tree_reg=0,
#                                # uniform_scale=False,
#                                # l1_reg=10,
#                                uniform_scale=True,
#                                kernel_input=True,
#                                J_lr=1e-12,
#                                )
# mod = old_bae_models.KernelBMF2(5,
#                                sparse_reg=0,
#                                tree_reg=0,
#                                # uniform_scale=False,
#                                # l1_reg=10,
#                                uniform_scale=True,
#                                )

en = mod.fit(X_ / X_.std(),
             period=100,
             initial_temp=50,
             decay_rate=0.88, 
             min_temp=1, 
             scl_lr=1e-3,
             lr=1e-1,
             # hot_start=False,
             )

mod.collapse()

samps = mod.sample(X_  / X_.std(), n_samp=100)
# samps = mod.sample(X_ / X_.std(), n_samp=100, slab=False)

# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.figure()

for i in range(mod.dim_hid):
    
    plt.subplot(2, mod.dim_hid, i+1)
    # plt.imshow(samps.mean(0)[:,i].reshape((2,5)))
    tpl.matshow(samps.mean(0)[:,i].reshape((2,5)), cmap='binary', color=(0.5,0.5,0.5))

plt.subplot(2,mod.dim_hid, mod.dim_hid+1)
plt.plot(en)

# samps = mod.sample(X_  / X_.std(), n_samp=100)
Xhat = mod(samps)
ll = np.round(mod.loglikelihood(X_/X_.std(), Xhat).mean(),2)
r2 = np.round(np.mean((X_/X_.std() - Xhat.mean(0))**2) / np.mean((X_/X_.std())**2),2)
plt.title(f"{r2}, {ll}")

plt.subplot(2,mod.dim_hid, mod.dim_hid+2)
plt.imshow(mod.operator.W.T@mod.operator.W)

plt.subplot(2,mod.dim_hid, mod.dim_hid+3)
plt.imshow(mod.latent_prior.J_W + mod.latent_prior.J_W.T + 2*np.diag(mod.latent_prior.J_h), 
           'bwr', vmin=-1, vmax=1)

#%%

# X_ = (samps.mean(0)-samps.mean((0,1)))@sts.ortho_group.rvs(10)[:samps.shape[-1]] + np.random.randn(10,10)*0.1

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
        # 'tree_reg': 1e-1,
        # 'sparse_reg': 1,
        'sparse_reg': 1,
        'tree_reg': 0,
        'J_lr': 1e-3,
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
            'hot_start': True,
            # 'hot_start': False,
            # 'scl_lr': 0,
            'scl_lr': 1e-3,
            'min_temp': 1,
            # 'min_temp': 1e-4,
            # 'lr': 1e-2,
            'lr':1e-1,
            }

n_run = 1

trn = np.zeros(len(kays))
tst = np.zeros(len(kays))
ens = []
sigs = []
for _ in range(n_run):
    for i,k in tqdm(enumerate(kays)):
        
        # mod = bae_models.JBMF(k,**args)
        mod = bae_models.BiPCA(k, **args)
        
        # wa,ba = bae_util.loocv(mod, X_/X_.std(), n_sample=10, **opt_args)
        wa,ba = bae_util.impcv(mod, X_/X_.std(), verbose=False, seed=0, n_sample=1, folds=10, max_folds=1, **opt_args)
        # wa,ba = bae_util.gabriel_bicv(mod, X_/X_.std(), n_samp=10, **opt_args)
        
        trn[i] += np.mean(wa) / n_run
        tst[i] += np.mean(ba) / n_run
        ens.append(en)
        sigs.append(mod.sigma_x)

plt.plot(kays, trn)
plt.plot(kays, tst, '--')

#%%

from sklearn.svm import LinearSVC
import dichotomies as dics

# rep = sessions[0]['X']
# cond = sessions[0]['finger']
pp = build_pseudopop(sessions, n_cond=10)

for k in range(mod.dim_hid):
    
    ccgp = np.mean(dics.compute_ccgp(pp['X'], pp['finger'], 
                              Sm[:,k][pp['finger']], LinearSVC()))
    control = np.mean(dics.compute_ccgp(pp['X'], pp['finger'], 
                              1*np.random.permutation(Sm[:,k])[pp['finger']], 
                              LinearSVC()))
    print(f"{np.round(ccgp,2)}, {np.round(control,2)}")

#%%

finger = np.eye(5)[np.mod(pp['finger'], 5)]
hand = 1*(pp['finger'] < 5)

clf = LinearSVC()

hand_ccgp = np.zeros((5,5))

for i in range(5):
    clf.fit(pp['X'][finger[:,i]>0], hand[finger[:,i]>0])
    
    for j in range(5):
        hand_ccgp[i,j] = clf.score(pp['X'][finger[:,j]>0], hand[finger[:,j]>0])

finger_ccgp = np.zeros((10,10))
for i in range(5):
    for j in range(5):
        if j == i:
            continue
        trn = finger[:,[i,j]].sum(1)>0
        clf.fit(pp['X'][trn], finger[trn, i])
        
        for k in range(5):
            for l in range(5):
                if k == l:
                    continue

#%%

# S_ = S
S_ = 1*(samps.mean(0) > 0.9)

rep = X_
# rep = Usc
signal = 0
noise = 1
rep = signal*S_@np.random.randn(S_.shape[-1], X_.shape[-1]) + noise*np.random.randn(*X_.shape)

ps = np.zeros((S_.shape[-1],S_.shape[-1]))
for i in range(S_.shape[-1]):
    for j in range(S_.shape[-1]):
        
        edges = 1*((util.yuke(S_) == 1) * (S_[:,[i]] != S_[:,[i]].T))
        aye_i, jay_i = np.where(np.triu(edges))
        
        edges = 1*((util.yuke(S_) == 1) * (S_[:,[j]] != S_[:,[j]].T))
        aye_j, jay_j = np.where(np.triu(edges))
        
        cs = util.cosine_sim((rep[aye_i] - rep[jay_i]).T, (rep[aye_j] - rep[jay_j]).T)
        
        if i == j:
            ps[i,j] = np.sum(np.triu(cs, k=1)) / spc.binom(len(cs),2)
        else:
            ps[i,j] = np.trace(cs) / len(cs)

plt.imshow(ps, 'bwr', vmin=-1, vmax=1)

#%%



