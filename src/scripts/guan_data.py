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
    Fig 8(d)  2-D MDS of the representation, per session, aligned across
              sessions with Generalized Procrustes analysis (with scaling),
              with S.E. ellipses across sessions.   <-- the panel of interest.

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
# MDS + Generalized Procrustes alignment across sessions
# ---------------------------------------------------------------------------

def rdm_to_mds(D, n_comp=2, seed=0):
    """2-D metric MDS from a (possibly slightly negative) squared-distance RDM."""
    dist = np.sqrt(np.clip(D, 0, None))
    np.fill_diagonal(dist, 0.0)
    mds = MDS(n_components=n_comp, dissimilarity='precomputed',
              random_state=seed, normalized_stress=False, n_init=8, max_iter=500)
    return mds.fit_transform(dist)


def _standardize(Y):
    """center and scale to unit Frobenius norm."""
    Y = Y - Y.mean(0, keepdims=True)
    n = nla.norm(Y)
    return Y / n if n > 0 else Y


def generalized_procrustes(configs, n_iter=50, tol=1e-9):
    """
    Generalized Procrustes analysis with scaling (and reflection) for a list of
    (n_points, dim) configurations sharing the same point ordering.

    Returns (aligned_list, mean_config).
    """
    aligned = [_standardize(Y) for Y in configs]
    ref = _standardize(np.mean(aligned, 0))
    prev = np.inf
    for _ in range(n_iter):
        new = []
        for Y in aligned:
            R, s = orthogonal_procrustes(Y, ref)   # min ||Y R - ref||, R orthogonal (incl. reflection)
            Yr = Y @ R
            # optimal isotropic scale to match ref
            denom = np.sum(Yr * Yr)
            sc = np.sum(Yr * ref) / denom if denom > 0 else 1.0
            new.append(Yr * sc)
        aligned = new
        ref_new = _standardize(np.mean(aligned, 0))
        err = nla.norm(ref_new - ref)
        ref = ref_new
        if abs(prev - err) < tol:
            break
        prev = err
    return aligned, ref


def cov_ellipse(xy, ax, n_std=1.0, **kw):
    """Draw an ellipse for the covariance of a set of 2-D points (S.E. => pass S.E. points)."""
    if len(xy) < 2:
        return
    cov = np.cov(xy.T)
    vals, vecs = nla.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    e = Ellipse(xy.mean(0), w, h, angle=ang, **kw)
    ax.add_patch(e)


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

#%%  load all NS-PPC sessions and compute per-session representations + RDMs

if __name__ == '__main__':

    files = session_files(SUBJ)
    print(f"{SUBJ}: {len(files)} sessions")

    sessions = []      # per-session dicts
    rdms = []          # per-session 10x10 crossnobis RDM
    embeds = []        # per-session 2-D MDS embedding (10x2)

    for sess, path in files:
        S = load_session(path)
        D = crossnobis_rdm(S['X'], S['finger'], S['run'], n_cond=len(FINGER_ORDER))
        emb = rdm_to_mds(D, seed=0)

        # condition-mean representation (10 x n_units): the "neural representation"
        cond_mean = np.stack([S['X'][S['finger'] == c].mean(0)
                              for c in range(len(FINGER_ORDER))])

        S.update(sess=sess, rdm=D, mds=emb, cond_mean=cond_mean)
        sessions.append(S)
        rdms.append(D)
        embeds.append(emb)
        print(f"  {sess}: {S['n_units']} PPC units, {len(S['finger'])} trials")

    rdms = np.array(rdms)
    mean_rdm = rdms.mean(0)

    # align embeddings across sessions (Fig 8d)
    aligned, mean_emb = generalized_procrustes(embeds)
    aligned = np.array(aligned)         # (n_sess, 10, 2)

    # package the representation for downstream use
    representation = {
        'finger_order': FINGER_ORDER,
        'subject': SUBJ,
        'region': 'NS-PPC',
        'sessions': [s['sess'] for s in sessions],
        'cond_mean': [s['cond_mean'] for s in sessions],  # list of (10, n_units); n_units varies by session
        'rdm_per_session': rdms,
        'rdm_mean': mean_rdm,
        'mds_aligned': aligned,       # (n_sess,10,2), GPA-aligned
        'mds_mean': mean_emb,         # (10,2)
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
    #%%  Figure 8(d): MDS of the NS-PPC representation (GPA aligned)
    # -----------------------------------------------------------------------
    n_sess = aligned.shape[0]
    # standard error across sessions (per finger, per axis)
    se = aligned.std(0, ddof=1) / np.sqrt(n_sess)

    colors = {'t': '#d62728', 'i': '#1f77b4', 'm': '#2ca02c',
              'r': '#9467bd', 'p': '#ff7f0e'}
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # draw hand subspace vectors (left->right) and finger-type structure
    for i, f in enumerate(FINGER_ORDER):
        ft = f[1]
        pts = aligned[:, i, :]                 # per-session locations of this finger
        ax.scatter(*mean_emb[i], s=120,
                   marker='o' if f[0] == 'L' else 's',
                   color=colors[ft], edgecolor='k', zorder=3)
        cov_ellipse(pts, ax, n_std=1.0, facecolor=colors[ft], alpha=0.25, edgecolor='none')
        ax.annotate(f, mean_emb[i], textcoords='offset points',
                    xytext=(6, 6), fontsize=9)

    # left->right vectors per finger-type (should look parallel/identical if factorized)
    for ft in FINGER_TYPES:
        l = FINGER_IDX['L' + ft]; r = FINGER_IDX['R' + ft]
        ax.annotate('', xy=mean_emb[r], xytext=mean_emb[l],
                    arrowprops=dict(arrowstyle='->', color=colors[ft], lw=1.5, alpha=.7))

    ax.set_aspect('equal')
    ax.set_title('Fig 8(d)  NS-PPC finger representation (2-D MDS)\n'
                 'circles=left hand, squares=right hand; ellipses=S.E. over sessions')
    ax.set_xlabel('MDS 1'); ax.set_ylabel('MDS 2')
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[ft],
                      markeredgecolor='k', markersize=10, label=NICE_LABEL[ft])
               for ft in FINGER_TYPES]
    ax.legend(handles=handles, loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/guan_fig8d_mds.png", dpi=150)
    print(f"saved figures -> {SAVE_DIR}/guan_fig8[a,b,d]_*.png")

    plt.show()
