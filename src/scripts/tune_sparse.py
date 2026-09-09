"""
Hyperparameter tuning for BMF models on SparseStructured data.

Uses the faster numpy models in new_bae_models.  For each (data condition, model)
it sweeps the annealing regime and the regularizers and reports the configs that
recover best.  Knobs swept:

    min_temp   1.0  (posterior-sampling regime) vs 1e-2 (low-temperature / MAP)
    pr_reg     participation-ratio weight reg (high values emphasized)
    tree_reg   hierarchy prior on the latents
    sparse_reg sparsity prior on the latents

Everything non-negative.  Metrics (rank = true latent dim):
    permham  S-support recovery, normalized permutation Hamming (lower better),
             averaged over N_SAMP posterior draws (sampling-mode fit)
    cka      representational similarity of S S^T   (higher better, sample-avg)
    permcos  W recovery, permutation-invariant column cosine (higher better)

    python tune_sparse.py
"""

import sys, os, pickle, time, itertools
sys.path.insert(0, 'C:/Users/mmall/OneDrive/Documents/github/repler/src/')

from functools import partial

import numpy as np

import experiments as exp
import new_bae_models as nbm
import df_util as d
import util

# ---- data conditions (the hard, slab cases; nonneg throughout) --------------
DATA = {
    'mixed+slab':       dict(blocks=[{'struct': 'categorical', 'K': 4},
                                     {'struct': 'none', 'K': 4}], slab=True),
    'categorical+slab': dict(blocks=[{'struct': 'categorical', 'K': 8}], slab=True),
}
N, TEMP, SNR, RATIO = 128, 1.0, 14.0, 4
N_DRAWS, N_SAMP = 3, 10

# ---- model hyperparameter grid ----------------------------------------------
MIN_TEMPS   = [1.0, 1e-2]        # sampling regime vs low-temperature
PR_REGS     = [0.1, 1.0, 10.0]   # participation-ratio weight reg (high emphasized)
TREE_REGS   = [0.0, 0.1, 1.0]
SPARSE_REGS = [0.0, 0.1, 1.0]

# spike-and-slab is a FLAG, not a class: SemiBMF(slab=True) is what used to
# be SpikeNMF, so the second entry is the same class with the slab on.
MODELS = {'SemiBMF': nbm.SemiBMF,
          'SpikeNMF': partial(nbm.SemiBMF, slab=True)}


def fit_score(ModelCls, K, X, Strue, Wtrue, min_temp, pr, tree, sparse):
    Xs = X / X.std()
    m = ModelCls(K, nonneg=True, weight_pr_reg=pr, tree_reg=tree,
                 sparse_reg=sparse, weight_l2_reg=1e-2)
    m.fit(Xs, initial_temp=10, decay_rate=0.9, period=8, min_temp=min_temp,
          max_iter=None, scl_lr=1e-2, hot_start=True, verbose=False)

    pcos = float(np.mean(d.permcos(Wtrue, m.operator.W)))   # W: point estimate
    Sbin = (Strue > 0).astype(float)
    samps = m.sample(Xs, n_samp=N_SAMP)                     # S: average over draws
    ph = np.mean([np.mean(d.permham(Sbin, (s > 0).astype(float), norm=True))
                  for s in samps])
    ck = np.mean([util.cka(Sbin @ Sbin.T,
                           (s > 0).astype(float) @ (s > 0).astype(float).T)
                  for s in samps])
    return float(ph), float(ck), pcos


def run():
    results = []
    grid = list(itertools.product(MIN_TEMPS, PR_REGS, TREE_REGS, SPARSE_REGS))
    for dname, dcfg in DATA.items():
        task = exp.SparseStructured(samps=N_DRAWS, snr=SNR, ratio=RATIO, seed=0,
                                    nonneg=True, N=N, temp=TEMP, **dcfg)
        data = task.sample()
        K = data['Strue'][0].shape[1]

        for mname, M in MODELS.items():
            rows = []
            for min_temp, pr, tree, sparse in grid:
                ph, ck, pc = [], [], []
                for Xd, Sd, Wd in zip(data['X'], data['Strue'], data['Wtrue']):
                    try:
                        a, b, c = fit_score(M, K, Xd, Sd, Wd, min_temp, pr, tree, sparse)
                        ph.append(a); ck.append(b); pc.append(c)
                    except Exception:
                        pass
                if not ph:
                    continue
                row = dict(data=dname, model=mname, min_temp=min_temp,
                           pr=pr, tree=tree, sparse=sparse,
                           permham=np.mean(ph), cka=np.mean(ck), permcos=np.mean(pc))
                rows.append(row); results.append(row)

            # permcos saturates ~1 at this SNR; rank by the discriminating metric
            rows.sort(key=lambda r: r['permham'])
            best_ph = rows[0]
            print(f"\n=== {dname} | {mname} (K={K}) -- top 5 by S-recovery (permham) ===")
            print(f"  {'min_T':>6} {'pr':>5} {'tree':>5} {'spar':>5} "
                  f"{'permham':>8} {'cka':>6} {'permcos':>8}")
            for r in rows[:5]:
                print(f"  {r['min_temp']:>6.2g} {r['pr']:>5.1f} {r['tree']:>5.2g} "
                      f"{r['sparse']:>5.2g} {r['permham']:>8.3f} {r['cka']:>6.3f} "
                      f"{r['permcos']:>8.3f}")
            print(f"  best permham: min_T={best_ph['min_temp']:.2g} pr={best_ph['pr']} "
                  f"tree={best_ph['tree']} sparse={best_ph['sparse']} "
                  f"-> permham={best_ph['permham']:.3f} permcos={best_ph['permcos']:.3f}")

    with open('tune_sparse_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nsaved tune_sparse_results.pkl")
    return results


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"done in {time.time()-t0:.0f}s")
