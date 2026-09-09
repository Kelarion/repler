"""
Validate bae_util.impcv as a rank selector on synthetic data.

Data: experiments.SparseStructured with struct='none' -> S is N x K_true of
independent uniform bits (true rank = K_true, exactly), embedded as
X = S @ W.T + noise with an orthogonal W at a controllable log-SNR (dB).

Test: for each candidate rank K we run impcv (random-mask imputation CV) and
record the held-out log-likelihood.  A good rank selector should have the test
log-likelihood rise until K = K_true and then plateau / fall.  We report the
full curve and where the test LL is maximised.
"""

import numpy as np

import old_bae_models
import bae_util
import df_util
import experiments as exp

# NOTE: experiments.SparseStructured.gen_latents calls `df_util.boltzmann`, but
# the function is actually spelled `boltzman`; and `boltzman` returns shape
# (K, n_samp), whereas gen_data expects the latents as (n_samp, K).  Both are
# bugs in the repo.  We patch the seam (alias + transpose) so the class runs.
df_util.boltzmann = lambda J, h, **kw: df_util.boltzman(J, h, **kw).T


# ---- fitting schedule handed to impcv (anneal T down, then evaluate) --------
# OPT = dict(initial_temp=5.0, decay_rate=0.9, period=5, min_temp=1e-2,
#            max_iter=400, hot_start=True)
OPT = dict(initial_temp=10.0, decay_rate=0.88, period=10, min_temp=1,
           max_iter=None, hot_start=True)


def make_data(K_true, N, ratio, snr, seed):
    """One (X, Strue) draw of exact rank K_true."""
    task = exp.SparseStructured(
        samps=1, snr=snr, ratio=ratio, orth=True, seed=seed, nonneg=False,
        K=K_true, N=N, temp=1.0, struct='none', kwargs={})
    d = task.sample()
    return d['X'][0], d['Strue'][0]


def impcv_heldout(model, X, folds):
    """bae_util.impcv's fit, scored with the genuinely held-out prediction Z[M]
    (the imputed reconstruction, produced without ever seeing X[M]).  Returns the
    mean held-out and (in-sample) train log-likelihood per entry."""
    o = OPT
    ntest = int(np.prod(X.shape) / folds)

    Z = 1.0 * X
    foo = np.random.randn(*X.shape)
    M = (foo < np.sort(foo.flatten())[ntest])
    Z[M] = np.random.choice(Z[~M], M.sum())

    model.initialize(Z, hot_start=o['hot_start'])
    for it in range(o['max_iter']):
        model.temp = o['min_temp'] + o['initial_temp'] * (o['decay_rate'] ** (it // o['period']))
        ES, _ = model.grad_step(Z)
        Z[M] = model(ES)[M]

    pred = model(ES)                                     # reconstruction from final latents
    train = np.mean(model.loglikelihood(X[~M], pred[~M]))  # observed entries
    test = np.mean(model.loglikelihood(X[M], pred[M]))     # held-out entries (no leak)
    return train, test


def rowcv(model, X, holdout_frac=0.25, n_sample=5):
    """Row hold-out CV (the loocv axis, but a K-fold split for speed).

    Fit (W, b) on the training rows only; W is therefore genuinely held out
    from the test rows.  Fold the test rows in (infer their latents from the
    fixed W) and score their reconstruction.  Over-rank should make W overfit
    the training rows and hurt held-out reconstruction -> a real peak at K_true.
    """
    n, d = X.shape
    te = np.random.rand(n) < holdout_frac
    Xtr, Xte = X[~te], X[te]

    model.fit(Xtr, initial_temp=OPT['initial_temp'], decay_rate=OPT['decay_rate'],
              period=OPT['period'], min_temp=OPT['min_temp'],
              max_iter=OPT['max_iter'], hot_start=OPT['hot_start'], verbose=False)

    pred_tr = model(model.sample(Xtr, n_samp=n_sample))
    pred_te = model(model.sample(Xte, n_samp=n_sample))
    train = np.mean(model.loglikelihood(Xtr, pred_tr))
    test = np.mean(model.loglikelihood(Xte, pred_te))
    return train, test


def cv_curve(cv_fn, X, ranks, n_rep, **kw):
    """Mean/std of train & held-out test log-lik over n_rep repeats, per rank.
    `cv_fn(model, X, **kw) -> (train, test)`."""
    trn = np.zeros((n_rep, len(ranks)))
    tst = np.zeros((n_rep, len(ranks)))
    for r in range(n_rep):
        for i, K in enumerate(ranks):
            mod = old_bae_models.SemiBMF(K, nonneg=False, tree_reg=0.0, sparse_reg=0.0,
                                     weight_pr_reg=1e-2, weight_l2_reg=1e-2,
                                     weight_l1_reg=0.0)
            a, b = cv_fn(mod, X, **kw)
            trn[r, i] = a
            tst[r, i] = b
    return trn, tst


def knee(ranks, m):
    """Smallest rank reaching 95% of the (rank1 -> best) test-LL gain (elbow)."""
    lo, hi = m[0], m.max()
    if hi - lo < 1e-9:
        return ranks[0]
    thr = lo + 0.95 * (hi - lo)
    return next(K for K, v in zip(ranks, m) if v >= thr)


def one_se(ranks, m, s):
    """One-standard-error rule: smallest rank within 1 SE of the best test-LL.
    The standard CV choice when the curve plateaus rather than peaks."""
    i_best = int(np.argmax(m))
    thr = m[i_best] - s[i_best]
    return next(K for K, v in zip(ranks, m) if v >= thr)


def report(name, ranks, K_true, trn, tst):
    mtr = trn.mean(0)
    mts, sts = tst.mean(0), tst.std(0)
    k_arg = ranks[int(np.argmax(mts))]
    k_knee = knee(ranks, mts)
    k_1se = one_se(ranks, mts, sts)
    print(f"\n=== {name}  (K_true = {K_true}) ===")
    print(f"  {'rank':>4} {'train_LL':>14} {'heldout_test_LL':>22}")
    for i, K in enumerate(ranks):
        tag = '  <-- true' if K == K_true else ''
        pk = ' max' if K == k_arg else '    '
        print(f"  {K:>4} {mtr[i]:>14.4f} {mts[i]:>14.4f}+-{sts[i]:<6.3f}{pk}{tag}")
    print(f"  selectors:  argmax -> {k_arg}   elbow(95%) -> {k_knee}   "
          f"1-SE -> {k_1se}     (true {K_true})")
    return k_arg, k_knee, k_1se


if __name__ == "__main__":
    K_true = 4
    N = 150
    ranks = list(range(1, 9))
    snr = 8.0
    results = {}

    # Sweep the masking fraction: with element-wise masking and many observed
    # entries per row, extra latents barely overfit, so the test LL plateaus.
    # Heavier masking leaves fewer observations per row, which should make extra
    # rank genuinely overfit and the held-out LL peak at K_true.
    X, S = make_data(K_true, N, ratio=8, snr=snr, seed=0)
    sng = np.linalg.svd(S, compute_uv=False)
    print(f"[snr={snr} dB]  X shape={X.shape}  Strue rank={int((sng > 1e-9).sum())}")

    # element-wise masking (impcv) vs row hold-out (loocv axis)
    trn_e, tst_e = cv_curve(impcv_heldout, X, ranks, n_rep=8, folds=5)
    results['impcv (element, 20%)'] = report("impcv: element-wise mask 20%",
                                             ranks, K_true, trn_e, tst_e)

    trn_r, tst_r = cv_curve(rowcv, X, ranks, n_rep=8, holdout_frac=0.25, n_sample=5)
    results['rowcv (rows, 25%)'] = report("rowcv: row hold-out 25%",
                                          ranks, K_true, trn_r, tst_r)

    print("\n" + "=" * 64)
    print(f"  {'method':>26} {'argmax':>8} {'elbow':>8} {'1-SE':>8}   (true K={K_true})")
    for name, (k_arg, k_knee, k_1se) in results.items():
        print(f"  {name:>26} {k_arg:>8} {k_knee:>8} {k_1se:>8}")
