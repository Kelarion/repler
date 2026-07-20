"""
Validate bae_util.bicv / gabriel_bicv as a rank selector, across true ranks.

Same synthetic generator as test_impcv_rank.py (experiments.SparseStructured,
struct='none' -> exact-rank-K binary latents, orthogonal embedding at a given
log-SNR).  For each true rank we sweep the model rank and report the held-out
BiCV log-likelihood; a good selector should PEAK at K_true (BiCV's block
hold-out gives a genuine complexity penalty, unlike element-wise imputation CV).
"""

import numpy as np

import bae_models
import bae_util
import df_util
import experiments as exp

# repo bug work-around (see test_impcv_rank.py): boltzmann->boltzman + transpose
df_util.boltzmann = lambda J, h, **kw: df_util.boltzman(J, h, **kw).T

# anneal down into the posterior-sampling regime (min_temp=1) and run to
# convergence (no max_iter).  A too-short / too-cold anneal leaves the NMSE
# stuck regardless of rank, killing the rank signal (see dbg_fit.py).
SCHED = dict(initial_temp=10.0, decay_rate=0.95, period=10, min_temp=1.0, max_iter=None)


def make_data(K_true, N=200, snr=8.0, seed=0):
    ratio = max(2, round(40 / K_true))          # keep d ~ 40 across K_true
    task = exp.SparseStructured(samps=1, snr=snr, ratio=ratio, orth=True, seed=seed,
                                nonneg=True, K=K_true, N=N, temp=1.0,
                                struct='none', kwargs={})
    X = task.sample()['X'][0]
    return X / X.std()      # scale so min_temp=1 sits in the committed regime


def model_K(K):
    return bae_models.SemiBMF(K, nonneg=True, tree_reg=0.0, sparse_reg=0.0,
                              weight_pr_reg=1e-2, weight_l2_reg=1e-2, weight_l1_reg=0.0)


def bicv_curve(X, ranks, train_frac=2 / 3, n_blk=4):
    """Mean/std held-out BiCV log-lik per model rank, over n_blk random blocks
    that train on ~train_frac x train_frac of the matrix.  The train block must
    be large enough to recover the rank -- a 1/2 x 1/2 (2x2 grid) block starves
    the fit and flattens the signal; ~2/3 works."""
    n, d = X.shape
    mean = np.zeros(len(ranks))
    std = np.zeros(len(ranks))
    for i, K in enumerate(ranks):
        scores = []
        for _ in range(n_blk):
            R = np.random.rand(n) < train_frac
            F = np.random.rand(d) < train_frac
            mod = model_K(K)
            _, te = bae_util.bicv(mod, X, R, F, fold_iter=100, hot_start=True,
                                  scl_lr=1e-2, **SCHED)   # fit sigma_x
            scores.append(te)
        mean[i], std[i] = np.mean(scores), np.std(scores)
    return mean, std


def knee(ranks, m):
    lo, hi = m[0], m.max()
    if hi - lo < 1e-9:
        return ranks[0]
    thr = lo + 0.95 * (hi - lo)
    return next(K for K, v in zip(ranks, m) if v >= thr)


def one_se(ranks, m, s):
    i = int(np.argmax(m))
    thr = m[i] - s[i]
    return next(K for K, v in zip(ranks, m) if v >= thr)


def report(K_true, ranks, mean, std):
    k_arg = ranks[int(np.argmax(mean))]
    k_knee = knee(ranks, mean)
    k_1se = one_se(ranks, mean, std)
    print(f"\n=== K_true = {K_true} ===")
    print(f"  {'rank':>4} {'bicv_test_LL':>22}")
    for K, m, s in zip(ranks, mean, std):
        tag = '  <-- true' if K == K_true else ''
        pk = ' max' if K == k_arg else '    '
        print(f"  {K:>4} {m:>14.4f}+-{s:<6.3f}{pk}{tag}")
    print(f"  selectors:  argmax -> {k_arg}   elbow(95%) -> {k_knee}   "
          f"1-SE -> {k_1se}   (true {K_true})")
    return k_arg, k_knee, k_1se


if __name__ == "__main__":
    summary = {}
    for K_true in [3, 5]:
        X = make_data(K_true, N=200, snr=8.0, seed=0)
        ranks = list(range(1, K_true + 3))
        mean, std = bicv_curve(X, ranks, train_frac=2 / 3, n_blk=4)
        summary[K_true] = report(K_true, ranks, mean, std)

    print("\n" + "=" * 60)
    print(f"  {'K_true':>8} {'argmax':>8} {'elbow':>8} {'1-SE':>8}")
    for K_true, (k_arg, k_knee, k_1se) in summary.items():
        hit = '  argmax hits' if k_arg == K_true else ''
        print(f"  {K_true:>8} {k_arg:>8} {k_knee:>8} {k_1se:>8}{hit}")
