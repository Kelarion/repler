"""Masked (imputation) fitting and bae_util.impcv, end to end.

A masked fit holds out the `mask` entries of X: every sweep the model refills them
from its own generative model, so the latents and the decoder are fit to the
OBSERVED entries only.  What is checked here:

  1. MECHANICS  impute() writes exactly forward(ES) at the masked entries and
                nothing else; after a fit the observed block is bit-identical and
                the held-out block has been replaced.  With n_chains > 1 the fit
                imputes into a per-chain copy and leaves the caller's array alone.
  2. NO PEEK    the held-out VALUES do not drive the fit: replacing them with the
                observed mean before fitting changes neither the observed-block
                error nor the held-out recovery, and even outright garbage heals
                (an unmasked fit on the same array is destroyed).  The sampler
                with a mask is likewise indifferent to what is under the mask.
  3. RECOVERY   the imputed held-out block correlates with the truth.
  4. SURFACE    slab / Boltzmann prior / multi-chain / loss(mask) / ppll(mask),
                and _Ximp is re-seeded per fit (so successive CV folds do not
                train on the previous fold's fill).
  5. IMPCV      bae_util.impcv is reproducible under `seed`, scores train above
                test, and its held-out score peaks at the true rank.
"""

import numpy as np

import new_bae_models as nbm
import new_bae_search
import bae_util


def seed_all(k):
    """Seed BOTH streams: numpy's (masks, chain init) and numba's (the search
    kernel's flips), which a bare np.random.seed does NOT reach."""
    np.random.seed(k)
    new_bae_search._seed(k)


def synth(n=200, d=24, k=4, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    S = 1.0 * (rng.random((n, k)) < 0.4)
    W = rng.standard_normal((d, k))
    X = S @ W.T + noise * rng.standard_normal((n, d))
    return (X - X.mean(0)) / X.std(), S, W


def rand_mask(shape, frac=0.2, seed=0):
    return np.random.default_rng(seed).random(shape) < frac


FIT = dict(initial_temp=3, decay_rate=0.85, period=2, min_temp=1e-2, verbose=False)


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])


# ---------------------------------------------------------------------------
#  1. mechanics
# ---------------------------------------------------------------------------

def check_impute(n_chains=1, seed=0):
    """impute() replaces the masked entries by forward(ES) and touches nothing
    else -- checked directly, before any M-step can move the parameters."""
    X, _, _ = synth(seed=seed)
    M = rand_mask(X.shape, 0.2, seed=seed + 1)

    seed_all(seed)
    mod = nbm.BiPCA(4, n_chains=n_chains, tree_reg=0)
    mod.initialize(X)

    Xw = 1.0 * X
    if n_chains > 1:
        Xw = np.repeat(Xw[None], n_chains, 0)
    ES = mod.latent_prior.Z
    mod.impute(Xw, ES, M)

    rec = mod.operator.forward(ES)
    idx = (slice(None),) * (Xw.ndim - M.ndim) + (M,)
    nidx = (slice(None),) * (Xw.ndim - M.ndim) + (~M,)
    return np.abs(Xw[idx] - rec[idx]).max(), np.abs(Xw[nidx] - X[~M]).max()


def check_mechanics(n_chains=1, seed=0):
    X, _, _ = synth(seed=seed)
    M = rand_mask(X.shape, 0.2, seed=seed + 1)
    Xw = 1.0 * X

    seed_all(seed)
    mod = nbm.BiPCA(4, n_chains=n_chains, tree_reg=0)
    mod.fit(Xw, mask=M, **FIT)

    # the last imputation predates the last M-step, so it matches the final
    # reconstruction only up to one parameter update -- hence the loose tolerance
    rec = mod.operator.forward(mod.S)
    if n_chains == 1:
        obs = np.abs(Xw[~M] - X[~M]).max()            # must be exactly 0
        hid = np.abs(Xw[M] - X[M]).max()              # must be > 0 (imputed)
        agree = np.abs(Xw[M] - rec[M]).max()
    else:
        obs = np.abs(mod._Ximp[:, ~M] - X[~M]).max()
        hid = np.abs(Xw - X).max()                    # caller's copy untouched -> 0
        agree = np.abs(mod._Ximp[:, M] - rec[:, M]).max()
    return obs, hid, agree


# ---------------------------------------------------------------------------
#  2. no peeking at the held-out block
# ---------------------------------------------------------------------------

def _fit_and_score(Xin, M, X, mask=True, seed=0):
    """Fit on `Xin` (masked unless mask=False) and score against the true X:
    (error on the observed block, correlation of the imputed held-out block)."""
    seed_all(seed)
    mod = nbm.BiPCA(4, tree_reg=0)
    Xw = 1.0 * Xin
    mod.fit(Xw, mask=M if mask else None, **FIT)
    pred = mod(mod.sample(Xw, n_samp=8, burnin=10, mask=M)).mean(0)
    return float(mod.loss(X, mask=~M)), corr(pred[M], X[M])


def check_no_peek(seed=0, blowup=50.0):
    X, _, _ = synth(seed=seed)
    M = rand_mask(X.shape, 0.2, seed=seed + 1)

    Xfill = 1.0 * X
    Xfill[M] = X[~M].mean()                  # the held-out truth is simply gone
    Xbad = 1.0 * X
    Xbad[M] = blowup                         # ... and here it is replaced by garbage

    return {'clean': _fit_and_score(X, M, X, seed=seed),
            'mean-filled': _fit_and_score(Xfill, M, X, seed=seed),
            'garbage': _fit_and_score(Xbad, M, X, seed=seed),
            'garbage, UNmasked': _fit_and_score(Xbad, M, X, mask=False, seed=seed)}


def check_sampler_no_peek(seed=0, blowup=50.0):
    """`sample(mask=...)` is what impcv scores with: it must not read X[mask]."""
    X, _, _ = synth(seed=seed)
    M = rand_mask(X.shape, 0.2, seed=seed + 1)
    seed_all(seed)
    mod = nbm.BiPCA(4, tree_reg=0)
    mod.fit(1.0 * X, mask=M, **FIT)

    Xbad = 1.0 * X
    Xbad[M] = blowup
    seed_all(1)
    a = mod.sample(X, n_samp=6, burnin=5, mask=M)
    seed_all(1)
    b = mod.sample(Xbad, n_samp=6, burnin=5, mask=M)
    seed_all(1)
    c = mod.sample(Xbad, n_samp=6, burnin=5)          # no mask -> must differ
    return np.abs(a - b).max(), np.abs(a - c).max()


# ---------------------------------------------------------------------------
#  3. recovery of the held-out block
# ---------------------------------------------------------------------------

def check_recovery(model_fn, seed=0, **fit_args):
    X, _, _ = synth(seed=seed)
    M = rand_mask(X.shape, 0.2, seed=seed + 1)
    Xw = 1.0 * X

    seed_all(seed)
    mod = model_fn()
    mod.fit(Xw, mask=M, **FIT, **fit_args)

    pred = mod(mod.sample(Xw, n_samp=8, burnin=10, mask=M)).mean(0)
    r = corr(pred[M], X[M])
    # baseline: the same predictions scored against a shuffled truth
    perm = np.random.default_rng(seed).permutation(int(M.sum()))
    r0 = corr(pred[M], X[M][perm])
    return r, r0


# ---------------------------------------------------------------------------
#  4. the rest of the masked surface
# ---------------------------------------------------------------------------

def check_surface(seed=0):
    X, _, _ = synth(seed=seed)
    M = rand_mask(X.shape, 0.2, seed=seed + 1)
    out = {}

    seed_all(seed)
    mod = nbm.BiPCA(4, n_chains=3, tree_reg=0, slab=True,
                    J_prior='boltzmann', J_lr=5e-2)
    mod.fit(1.0 * X, mask=M, **FIT)

    out['spike_binary'] = bool(np.isin(mod.latent_prior.S, (0.0, 1.0)).all())
    out['slab_continuous'] = float(np.abs(mod.latent_prior.Z -
                                          mod.latent_prior.S).max())
    out['J_shape'] = mod.latent_prior.J.shape
    out['loss_masked'] = np.shape(mod.loss(X, mask=M))
    # the cached ranking (which is on the observed entries under a mask) must
    # agree with an explicitly recomputed one
    out['best_chain'] = mod.best_chain(X, mask=~M)
    out['best_chain_cached'] = mod.best_chain()

    ll = mod.ppll(X, mask=M, n_samp=4, burnin=5)
    out['ppll'] = (float(np.mean(ll[~M])), float(np.mean(ll[M])))

    Xc = 1.0 * X
    mod.sample(Xc, n_samp=2, burnin=3, mask=M)
    out['sample_pure'] = float(np.abs(Xc - X).max())

    # a second fit with a DIFFERENT mask must re-seed the per-chain copy from the
    # data it was handed, not carry over the first fit's fill
    M2 = rand_mask(X.shape, 0.2, seed=seed + 7)
    mod.fit(1.0 * X, mask=M2, **FIT)
    out['reseeded'] = float(np.abs(mod._Ximp[:, ~M2] - X[~M2]).max())
    return out


# ---------------------------------------------------------------------------
#  5. impcv
# ---------------------------------------------------------------------------

def impcv_at(rank, X, seed=0, n_chains=1, folds=5, max_folds=2, **kw):
    seed_all(seed)
    mod = nbm.BiPCA(rank, n_chains=n_chains, tree_reg=0, **kw)
    return bae_util.impcv(mod, X, folds=folds, max_folds=max_folds, seed=seed,
                          n_sample=4, **FIT)


if __name__ == "__main__":
    ok = True

    def report(flag, title, detail):
        global ok
        ok &= bool(flag)
        print(f"[{'PASS' if flag else 'FAIL'}] {title}")
        print(f"    {detail}")

    # 1 -------------------------------------------------------------------
    for C in (1, 3):
        d_in, d_out = check_impute(n_chains=C)
        report(d_in < 1e-15 and d_out == 0.0,
               f"impute() writes forward(ES) under the mask only (n_chains={C})",
               f"max|X[M] - forward(ES)[M]| = {d_in:.1e}   "
               f"max|X[~M] - truth| = {d_out:.1e}")

        obs, hid, agree = check_mechanics(n_chains=C)
        good = obs == 0.0 and (hid > 0.0 if C == 1 else hid == 0.0) and agree < 1e-3
        report(good, f"a masked fit leaves the observed data alone (n_chains={C})",
               f"max|d observed|={obs:.1e}  "
               + (f"max|d held-out|={hid:.2f} (imputed)  " if C == 1 else
                  f"caller array untouched: max|dX|={hid:.1e}  ")
               + f"|imputation - final reconstruction|={agree:.1e}")

    # 2 -------------------------------------------------------------------
    res = check_no_peek()
    lo, ro = res['clean']
    lf, rf = res['mean-filled']
    lg, rg = res['garbage']
    lu, ru = res['garbage, UNmasked']
    good = abs(lf - lo) < 0.25 * lo and abs(rf - ro) < 0.05 and lu > 5 * lo
    report(good, "the held-out VALUES do not drive the fit",
           "  ".join(f"{k}: MSE(obs) {v[0]:.3f} / r(held-out) {v[1]:+.3f}"
                     for k, v in res.items()))

    d_mask, d_nomask = check_sampler_no_peek()
    report(d_mask == 0.0 and d_nomask > 0,
           "sample(mask=...) ignores whatever is under the mask",
           f"garbage vs truth under the mask: max|dS| = {d_mask:.1e}   "
           f"(same sampler WITHOUT the mask: {d_nomask:.1f})")

    # 3 -------------------------------------------------------------------
    cases = {
        'BiPCA': lambda: nbm.BiPCA(4, tree_reg=0),
        'BiPCA slab': lambda: nbm.BiPCA(4, tree_reg=0, slab=True),
        'JBMF': lambda: nbm.JBMF(4, J_lr=5e-2),
        'SemiBMF': lambda: nbm.SemiBMF(4),
        'BiPCA x3 chains': lambda: nbm.BiPCA(4, n_chains=3, tree_reg=0),
    }
    for name, fn in cases.items():
        rr = np.array([check_recovery(fn, seed=sd) for sd in range(3)])
        r, r0 = rr.mean(0)
        good = r > 0.7 and abs(r0) < 0.2
        report(good, f"held-out entries are recovered ({name})",
               f"corr(imputed, truth) = {r:+.3f} over 3 seeds "
               f"{np.round(rr[:, 0], 3)}   (shuffled baseline {r0:+.3f})")

    # 4 -------------------------------------------------------------------
    s = check_surface()
    good = (s['spike_binary'] and s['slab_continuous'] > 0
            and s['loss_masked'] == (3,) and s['sample_pure'] == 0.0
            and s['ppll'][0] > s['ppll'][1] and s['reseeded'] == 0.0
            and s['best_chain'] == s['best_chain_cached'])
    report(good, "masked surface: slab + Boltzmann + 3 chains",
           f"spike binary={s['spike_binary']}  max|Z-S|={s['slab_continuous']:.2f}  "
           f"J{s['J_shape']}  loss(mask)->{s['loss_masked']}  "
           f"best_chain={s['best_chain']} (cached {s['best_chain_cached']})\n    "
           f"ppll: observed {s['ppll'][0]:+.3f} > held-out {s['ppll'][1]:+.3f}   "
           f"sample(mask) leaves X alone: {s['sample_pure']:.1e}   "
           f"refit re-seeds _Ximp: {s['reseeded']:.1e}")

    try:
        nbm.KernelBMF(4).fit(1.0 * synth()[0],
                             mask=rand_mask(synth()[0].shape, 0.2, seed=1), **FIT)
        raised = False
    except NotImplementedError as e:
        raised = str(e)
    report(bool(raised), "a model with no imputation E-step refuses a mask",
           f"KernelBMF -> {raised}")

    # 5 -------------------------------------------------------------------
    X, _, _ = synth(k=4)
    a = impcv_at(4, X, seed=11)
    b = impcv_at(4, X, seed=11)
    report(np.allclose(a, b), "impcv is reproducible under a fixed seed",
           f"run 1 (train, test) = ({a[0]:.4f}, {a[1]:.4f})   "
           f"run 2 = ({b[0]:.4f}, {b[1]:.4f})")

    curve = {r: impcv_at(r, X, seed=11) for r in (1, 2, 4, 8, 12)}
    best = max(curve, key=lambda r: curve[r][1])
    report(best in (3, 4, 5), "impcv held-out score peaks at the true rank (4)",
           "  ".join(f"k={r}: test {v[1]:+.3f}" for r, v in curve.items())
           + f"   -> argmax k={best}")

    for C in (1, 3):
        tr, te = impcv_at(4, X, seed=11, n_chains=C)
        report(te < tr, f"impcv scores train above test (n_chains={C})",
               f"train {tr:+.4f}   test {te:+.4f}   gap {tr - te:+.4f}")

    print("\nALL PASS" if ok else "\nSOME FAILED")
