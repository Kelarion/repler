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

import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
from scipy.linalg import orthogonal_procrustes
from itertools import combinations

from sklearn.manifold import MDS
from sklearn.covariance import ledoit_wolf

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pynwb import NWBHDF5IO

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
# main pipeline
# ---------------------------------------------------------------------------

#%%  load all NS-PPC sessions and compute per-session representations + RDMs

if __name__ == '__main__':

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

        S.update(sess=sess, rdm=D, cond_mean=cond_mean)
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
