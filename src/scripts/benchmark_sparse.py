"""
Benchmark BMF models on data from experiments.SparseStructured.

Sweeps over the *data* parameters -- graph structure, number of observations N,
Gibbs temperature, and whether a continuous (slab) multiplier is applied -- and
measures how well each model recovers the true latent support.  Everything is
non-negative (data + models), which works best in practice.

Self-contained (doesn't use the server_utils sweep machinery): for each data
config it draws `n_draws` datasets, fits each model at the true rank, and scores
recovery with permutation-invariant Hamming (support) and CKA.  Results print as
a table and are pickled.

    python benchmark_sparse.py
"""

import sys, os, pickle, time, itertools
sys.path.insert(0, 'C:/Users/mmall/OneDrive/Documents/github/repler/src/')

import numpy as np

import experiments as exp
import bae_models
import df_util as d
import util

# ---------------------------------------------------------------- sweep grid --
STRUCTURES = {                                   # each -> SparseStructured.blocks
    'none':        [{'struct': 'none', 'K': 8}],            # independent bits
    'categorical': [{'struct': 'categorical', 'K': 8}],     # one winner-take-all block
    'mixed':       [{'struct': 'categorical', 'K': 4},      # union of two blocks
                    {'struct': 'none', 'K': 4}],
}
NS     = [64, 128, 256]      # number of observations
TEMPS  = [1.0, 2.0]          # Gibbs temperature (hotter = noisier dependencies)
SLABS  = [False, True]       # continuous Gamma(8,1/8) multiplier on the latents
N_DRAWS = 3                  # datasets per config (mean +/- std)
N_SAMP  = 20                 # posterior samples per fit (metrics averaged over them)
SNR    = 14.0                # embedding log-SNR (dB)
RATIO  = 4                   # observed dim = RATIO * latent dim

# Posterior-sampling fit: anneal down to T=1 (the sampling regime) and learn
# sigma_x (scl_lr>0) so the effective temperature is calibrated to the data
# scale.  Run to convergence (max_iter=None).  Because the fit stays in the
# sampling regime, the latent S is a *draw*, not a point estimate -- so metrics
# are computed on individual posterior samples and averaged (see `recover`).
SCHED = dict(initial_temp=10.0, decay_rate=0.9, period=8, min_temp=1.0,
             max_iter=None, scl_lr=1e-2, hot_start=True, verbose=False)

MODELS = {
    'SemiBMF':  lambda K: bae_models.SemiBMF(K, nonneg=True, tree_reg=0.0,
                    sparse_reg=0.0, weight_pr_reg=0.1, weight_l2_reg=1e-2),
    'SpikeNMF': lambda K: bae_models.SpikeNMF(K, nonneg=True, tree_reg=0.0,
                    sparse_reg=0.0, weight_pr_reg=0.1, weight_l2_reg=1e-2),
}


def recover(model, X, Strue):
    """Fit at the true rank (posterior-sampling mode), then score recovery of the
    latent support on N_SAMP posterior draws and average -- the fit doesn't give a
    point estimate, so a single S is noisy."""
    Xs = X / X.std()
    model.fit(Xs, **SCHED)
    Sbin = (Strue > 0).astype(float)                 # true support (slab -> binarize)
    samps = model.sample(Xs, n_samp=N_SAMP)          # (N_SAMP, N, K) draws at T=1
    ph, ck = [], []
    for s in samps:
        sb = (s > 0).astype(float)                   # binarize each posterior sample
        ph.append(np.mean(d.permham(Sbin, sb, norm=True)))   # lower = better
        ck.append(util.cka(Sbin @ Sbin.T, sb @ sb.T))        # higher = better
    return float(np.mean(ph)), float(np.mean(ck))


def run():
    results = []
    grid = list(itertools.product(STRUCTURES, NS, TEMPS, SLABS))
    print(f"{len(grid)} data configs x {len(MODELS)} models x {N_DRAWS} draws\n")
    print(f"{'structure':>11} {'N':>4} {'temp':>4} {'slab':>5} {'model':>9} "
          f"{'permham':>16} {'cka':>16}")

    for sname, N, temp, slab in grid:
        task = exp.SparseStructured(
            samps=N_DRAWS, snr=SNR, ratio=RATIO, seed=0, nonneg=True,
            blocks=STRUCTURES[sname], N=N, temp=temp, slab=slab)
        data = task.sample()

        for mname, make in MODELS.items():
            ph, ck = [], []
            for Xd, Sd in zip(data['X'], data['Strue']):
                try:
                    p, c = recover(make(Sd.shape[1]), Xd, Sd)   # rank = true dim
                    ph.append(p); ck.append(c)
                except Exception as e:
                    print(f"    !! {sname} N={N} {mname}: {type(e).__name__}: {e}")
            if not ph:
                continue
            row = dict(structure=sname, N=N, temp=temp, slab=slab, model=mname,
                       permham_mean=np.mean(ph), permham_std=np.std(ph),
                       cka_mean=np.mean(ck), cka_std=np.std(ck))
            results.append(row)
            print(f"{sname:>11} {N:>4} {temp:>4.1f} {str(slab):>5} {mname:>9} "
                  f"{row['permham_mean']:>7.3f}+-{row['permham_std']:<6.3f} "
                  f"{row['cka_mean']:>7.3f}+-{row['cka_std']:<6.3f}")

    with open('benchmark_sparse_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nsaved benchmark_sparse_results.pkl")
    return results


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"done in {time.time()-t0:.0f}s")
