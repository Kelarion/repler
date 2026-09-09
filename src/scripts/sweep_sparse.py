"""
Sweep: BMF recovery vs N, across generative temperature, structure, and the
regularizers tree_reg / pr_reg.  Models: new_bae_models SemiBMF & SpikeNMF
(numpy) plus an NMF baseline.

Fixed: annealing min_temp = 1 (posterior-sampling regime); the swept temperature
is the *generative* Gibbs temp of SparseStructured.  Everything non-negative,
slab on, rank = true latent dim.  Data rescaled by X.std() for every model.
Metrics (S averaged over posterior samples):
    hamming  fraction of mismatched S entries, on features matched by permyuke_idx,
             binarized noise-robustly (lower better)
    nbs      latent-subspace recovery (normalized Bures similarity, higher better)
    feat     W recovery: matched-feature relative MSE, scale-normalized by an
             optimal global alpha (lower better, 0=perfect)
    entropy  mean per-element binary entropy of the posterior marginals <S>
             over the posterior samples (bits; posterior uncertainty)

Results pickle to sweep_sparse_results.pkl (list of per-config dicts).
    python sweep_sparse.py
"""

import sys, pickle, time, itertools
sys.path.insert(0, 'C:/Users/mmall/OneDrive/Documents/github/repler/src/')

from functools import partial

import numpy as np
from sklearn.decomposition import NMF

import experiments as exp
import new_bae_models as nbm
import df_util as d
import util

# ---- sweep grid -------------------------------------------------------------
STRUCTURES = {
    'categorical': [{'struct': 'categorical', 'K': 8}],
    'cat_blowup3': [{'struct': 'categorical', 'K': 3, 'blowup': 3}],  # 9 dims
    'tree':        [{'struct': 'tree', 'K': 4}],     # ~8-10 dims after clique blowup
}
NS         = [32, 64, 128, 256]            # x-axis
GEN_TEMPS  = [1e-4, 1e-1, 1.0]             # generative Gibbs temperature
TREE_REGS  = [0.0, 0.1, 1.0, 10.0]
PR_REGS    = [0.0, 0.1, 1.0]
SPARSE_REGS = [0.0, 0.1, 1.0]
N_DRAWS, N_SAMP = 3, 12
SNR, RATIO, SLAB = 14.0, 4, True

# annealing: min_temp ALWAYS 1 (sampling regime); learn sigma_x; converge.
FIT = dict(initial_temp=10, decay_rate=0.9, period=8, min_temp=1.0,
           max_iter=None, scl_lr=1e-3, hot_start=True, verbose=False)
# spike-and-slab is a FLAG, not a class: SemiBMF(slab=True) is what used to
# be SpikeNMF, so the second entry is the same class with the slab on.
BMF = {'SemiBMF': nbm.SemiBMF,
       'SpikeNMF': partial(nbm.SemiBMF, slab=True)}


def _metrics(samps, Strue, Wtrue, Wrec, want_entropy=True):
    """Score recovery.  Features are matched by an MSE-minimizing assignment
    (permyuke_idx) and the aligned binary S columns compared with a plain Hamming.
    Samples are binarized noise-robustly (2-means) and metrics averaged over draws.
        hamming  fraction of mismatched S entries on matched columns (lower=better)
        nbs      latent-subspace recovery (normalized Bures sim, higher=better)
        feat     W recovery: matched-feature relative MSE, scale-normalized by a
                 single optimal global alpha (lower=better, 0=perfect).  Global
                 (not per-column) normalization removes the X.std() rescaling
                 offset but still penalizes broad/mis-shaped features -- unlike
                 per-column cosine, which is too lenient on broad NMF features.
        entropy  mean per-element binary entropy of the posterior marginals <S>
    """
    Sbin = (Strue > 0).astype(float)
    aye, jay = d.permyuke_idx(Wtrue, Wrec, norm=True)  # scale-robust correspondence
    order = np.argsort(aye)
    jcol = np.asarray(jay)[order]                      # recovered col per true col
    Wt, Wr = Wtrue[:, np.asarray(aye)[order]], Wrec[:, jcol]
    alpha = (Wt * Wr).sum() / ((Wr ** 2).sum() + 1e-12)            # optimal global scale
    feat = float(((Wt - alpha * Wr) ** 2).sum() / ((Wt ** 2).sum() + 1e-12))
    Sb = d.binarize(samps, axis=1)                     # (n_samp, N, K), robust
    ham = float(np.mean([(Sbin != sb[:, jcol]).mean() for sb in Sb]))
    nb = float(np.mean([util.nbs(Sbin, sb) for sb in Sb]))
    if want_entropy:
        p = Sb.mean(0)
        H = -(p * np.log2(p + 1e-9) + (1 - p) * np.log2(1 - p + 1e-9))
        ent = float(H.mean())
    else:
        ent = np.nan
    return ham, nb, feat, ent


def fit_bmf(ModelCls, K, X, Strue, Wtrue, tree, pr, sparse):
    Xs = X / X.std()                           # rescale (helps convergence)
    m = ModelCls(K, nonneg=True, weight_pr_reg=pr, tree_reg=tree,
                 sparse_reg=sparse, weight_l2_reg=1e-2)
    m.operator.resample_dead = True            # recover columns that die under nonneg
    m.fit(Xs, **FIT)
    return _metrics(m.sample(Xs, n_samp=N_SAMP), Strue, Wtrue, m.operator.W)


def fit_nmf(K, X, Strue, Wtrue):
    Xs = X / X.std()                           # same rescaling as the BMF models
    nmf = NMF(K, init='nndsvda', max_iter=500)
    Z = nmf.fit_transform(Xs)
    return _metrics(Z[None], Strue, Wtrue, nmf.components_.T, want_entropy=False)


def agg(rows):
    """rows = list of (hamming, nbs, feat, entropy) per draw -> mean/std dict."""
    a = np.array(rows, float)
    out = {}
    for i, k in enumerate(['hamming', 'nbs', 'feat', 'entropy']):
        out[k + '_mean'] = float(np.nanmean(a[:, i]))
        out[k + '_std'] = float(np.nanstd(a[:, i]))
    return out

#%%
def run():
    results = []
    for sname, blocks in STRUCTURES.items():
        for temp in GEN_TEMPS:
            for N in NS:
                task = exp.SparseStructured(samps=N_DRAWS, snr=SNR, ratio=RATIO,
                                            seed=0, nonneg=True, blocks=blocks,
                                            N=N, temp=temp, slab=SLAB)
                data = task.sample()
                draws = list(zip(data['X'], data['Strue'], data['Wtrue']))

                # BMF models: full tree_reg x pr_reg x sparse_reg grid
                for mname, M in BMF.items():
                    for tree, pr, sparse in itertools.product(TREE_REGS, PR_REGS,
                                                              SPARSE_REGS):
                        per = []
                        for X, S, W in draws:
                            try:
                                per.append(fit_bmf(M, S.shape[1], X, S, W,
                                                   tree, pr, sparse))
                            except Exception:
                                pass
                        if per:
                            results.append(dict(structure=sname, model=mname,
                                gen_temp=temp, N=N, tree_reg=tree, pr_reg=pr,
                                sparse_reg=sparse, **agg(per)))
                    print(f"[{sname} T={temp:g} N={N}] {mname} done")

                # NMF baseline: no tree_reg / pr_reg / sparse_reg
                per = []
                for X, S, W in draws:
                    try:
                        per.append(fit_nmf(S.shape[1], X, S, W))
                    except Exception:
                        pass
                if per:
                    results.append(dict(structure=sname, model='NMF', gen_temp=temp,
                        N=N, tree_reg=np.nan, pr_reg=np.nan, sparse_reg=np.nan,
                        **agg(per)))

    with open('sweep_sparse_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nsaved sweep_sparse_results.pkl ({len(results)} configs)")
    return results


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"done in {time.time()-t0:.0f}s")
