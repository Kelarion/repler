"""Correctness check: bae_models.KernelBMF vs old_bae_models.KernelBMF2.

KernelBMF2 is the one model that does NOT fit the LinearGaussianBMF skeleton: it
factorizes a Gram matrix K ~ center(S diag(scl) S^T), quadratic in S.  So it
subclasses BMF directly and carries its own diagonal scale `scl` + running
StX = S^T X -- but it still reuses the LatentPrior (S / StS / Z + the shared
additive S-prior plugin) and only swaps in a kernel-specific likelihood field
(bae_search.make_kernel_search, ported from old_bae_search.kerbmf3 / kerbmf2).

The E-step is the numba kerbmf3 (RNG -> numba_seed); the scale M-step is
deterministic.  We sync S / StS / StX / scl, run identical anneals (explicit
schedule, since the two BMF bases have different fit defaults) and assert
latents / scale / StX / reconstruction / energy match bit-for-bit.

Plus a kernel-vs-feature equivalence check: the two input forms compute the
identical E-step field when K == Xc Xc^T, so a seeded sweep must produce the
same spike.
"""

import numpy as np
from numba import njit

import old_bae_models
import bae_models


@njit
def numba_seed(s):
    np.random.seed(s)


def make_data(n=60, d=15, K=4, seed=0):
    rng = np.random.default_rng(seed)
    S = (rng.random((n, K)) < 0.4).astype(float)
    W = rng.standard_normal((K, d))
    X = S @ W + 0.3 * rng.standard_normal((n, d))
    return X - X.mean(0)                               # centered feature matrix


def sync(orig, port):
    """Copy orig's initialized state into port so both start identical.  The port
    now divides the E-step field by sigma2 (input-scale-aware) instead of the
    original's /(N-1); forcing sigma2 = N-1 recovers the original exactly, so the
    bit-for-bit port check still holds modulo that (deliberate) normalization knob."""
    port.latent_prior.S = orig.S.copy()
    port.latent_prior.StS = orig.StS.copy()
    port.latent_prior.Z = orig.S.copy()               # unused by the kernel E-step
    port.StX = orig.StX.copy()
    port.scl = orig.scl.copy()
    port.sigma2 = float(port.n - 1)                   # reproduce the original /(N-1)


SCHED = dict(initial_temp=2.0, decay_rate=0.85, period=3, min_temp=1e-3)


def run_case(name, X, dim_hid=4, max_iter=120, seed=0,
             uniform_scale=True, l1_reg=0.0, sparse_reg=0.0, tree_reg=1e-2):
    orig = old_bae_models.KernelBMF2(dim_hid, sparse_reg=sparse_reg, tree_reg=tree_reg,
                                 uniform_scale=uniform_scale, l1_reg=l1_reg)
    port = bae_models.KernelBMF(dim_hid, sparse_reg=sparse_reg, tree_reg=tree_reg,
                                     uniform_scale=uniform_scale, l1_reg=l1_reg)

    orig.initialize(X)
    port.initialize(X)
    sync(orig, port)

    numba_seed(seed)
    en_o = orig.fit(X, max_iter=max_iter, verbose=False, **SCHED)
    numba_seed(seed)
    en_p = port.fit(X, max_iter=max_iter, verbose=False, **SCHED)

    S_eq = np.array_equal(orig.S, port.S)
    scl_eq = np.allclose(orig.scl, port.scl, rtol=0, atol=1e-12)
    stx_eq = np.array_equal(orig.StX, port.StX)
    rec_eq = np.allclose(orig(orig.S), port(port.S), rtol=0, atol=1e-10)
    en_o, en_p = np.asarray(en_o), np.asarray(en_p)
    en_close = np.allclose(en_o, en_p, rtol=0, atol=1e-9)

    ok = S_eq and scl_eq and stx_eq and rec_eq and en_close
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    spike S identical : {S_eq}")
    print(f"    scale scl close   : {scl_eq}   (max|d|={np.abs(orig.scl - port.scl).max():.2e})")
    print(f"    StX identical     : {stx_eq}")
    print(f"    reconstruction    : {rec_eq}")
    print(f"    energy trace close: {en_close}   (max|d|={np.abs(en_o - en_p).max():.2e})")
    return ok


def run_kernel_equiv(name, X, scl, dim_hid=4, seed=1, tree_reg=1e-2, sparse_reg=0.1):
    """The feature (X) and kernel (K = Xc Xc^T) E-steps compute the identical
    field -- for ANY diagonal scale scl -- so one seeded sweep from the same state
    must give the same spike.  This is how the (per-feature) kernel-input branch
    is verified, since the original old_bae_search.kerbmf2 is scalar-only."""
    K = X @ X.T

    feat = bae_models.KernelBMF(dim_hid, sparse_reg=sparse_reg,
                                     tree_reg=tree_reg, kernel_input=False)
    kern = bae_models.KernelBMF(dim_hid, sparse_reg=sparse_reg,
                                     tree_reg=tree_reg, kernel_input=True)
    feat.initialize(X)
    kern.initialize(K)

    # identical starting latents/scale (StX only matters for the feature form)
    S0 = feat.S.copy()
    for m in (feat, kern):
        m.latent_prior.S = S0.copy()
        m.latent_prior.StS = (S0.T @ S0).copy()
        m.scl = scl.copy()
    feat.StX = S0.T @ X
    feat.temp = kern.temp = 0.7

    numba_seed(seed)
    feat.EStep(X, feat.S)
    numba_seed(seed)
    kern.EStep(K, kern.S)

    ok = np.array_equal(feat.S, kern.S)
    print(f"[{'PASS' if ok else 'FAIL'}] kernel-vs-feature field equivalence ({name})")
    print(f"    spike S identical : {ok}   (agreement {np.mean(feat.S == kern.S):.5f})")
    return ok


def run_logodds(X, dim_hid=4, seed=2):
    """Verify the kernel-input search log-odds directly (flip-and-compare method):
    for each element flip it, compute the EXACT change in the model's energy /
    log-likelihood, and compare to the log-odds the search emits (extracted via
    the search's `currents` mode).  A correct field is EXACTLY proportional to
    d(loglik) -- the constant is the likelihood precision 1/sigma2; a non-constant
    ratio would mean the field is wrong.  The kernel is fed as the double-centered
    Gram (as the E-step centers it), matching the loss/MStep."""
    import bae_search as nbs
    Kc = X @ X.T                                   # double-centered (X col-centered)
    n, m = len(X), dim_hid
    rng = np.random.default_rng(seed)
    S = 1.0 * (rng.random((n, m)) < 0.5)
    scl = rng.uniform(0.5, 2.0, m)
    N = n
    sig2 = np.sum(Kc ** 2) / N ** 2                # <K^2>, as the model uses

    def energy(Sm):                                # model's S-dependent energy
        StS = Sm.T @ Sm
        StS_c = StS - np.outer(np.diag(StS), np.diag(StS)) / N
        Qnrm = scl @ (StS_c ** 2) @ scl
        V_dot = np.diag(Sm.T @ Kc @ Sm)
        return Qnrm - 2 * np.sum(scl * V_dot)

    dLL = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            S1 = S.copy(); S1[i, j] = 1.0
            S0 = S.copy(); S0[i, j] = 0.0
            dLL[i, j] = -(energy(S1) - energy(S0)) / 2

    zJ, zh = np.zeros((m, m)), np.zeros(m)
    kern = nbs.make_kernel_search(nbs.PRIOR_PLAIN, kernel_input=True, currents=True)
    feat = nbs.make_kernel_search(nbs.PRIOR_PLAIN, kernel_input=False, currents=True)
    ok = np.zeros((n, m)); of = np.zeros((n, m))
    kern(Kc, S.copy(), (S.T @ S).copy(), np.zeros((m, 0)), N, scl, sig2, 1.0, 0.0, 0.0, zJ, zh, ok, False)
    feat(X,  S.copy(), (S.T @ S).copy(), (S.T @ X).copy(), N, scl, sig2, 1.0, 0.0, 0.0, zJ, zh, of, False)

    rel = np.nanstd(ok / dLL) / abs(np.nanmean(ok / dLL))
    proportional = rel < 1e-9
    kf = np.allclose(ok, of)
    ok_all = proportional and kf
    print(f"[{'PASS' if ok_all else 'FAIL'}] kernel log-odds == d(loglik)  (flip-and-compare)")
    print(f"    LO / dLL constant : {proportional}   (ratio std/mean={rel:.1e})")
    print(f"    kernel == feature : {kf}")
    return ok_all


def run_sample(X, dim_hid=4):
    """The kernel model has no Z, so it overrides BMF.sample to walk an S-only
    Gibbs chain -- the inherited sample would pass a positional Z the kernel
    EStep does not accept.  Check both input modes run and return binary spikes
    of the right shape."""
    ok = True
    for name, data, kin in [("feature", X, False), ("kernel", X @ X.T, True)]:
        m = bae_models.KernelBMF(dim_hid, sparse_reg=0.1, tree_reg=0.05,
                                     kernel_input=kin)
        m.initialize(data)
        samps = m.sample(data, n_samp=3, burnin=2)
        good = (samps.shape == (3, len(data), dim_hid)
                and np.all((samps == 0) | (samps == 1)))
        ok = ok and good
        print(f"    sample ({name:7s}) : shape {samps.shape}, binary={good}")
    print(f"[{'PASS' if ok else 'FAIL'}] sample() runs (spike-only, no Z)")
    return ok


def run_logodds_scaling(dim_hid=4):
    """The sigma2 normalization (E-step divides the field by <K^2>=data_norm/N^2)
    should hold the log-odds std at O(1) regardless of N, d, and the INPUT SCALE --
    so the sigmoid stays responsive (neither saturated nor a coin flip).  Sweep all
    three and assert the std stays in a bounded band; also assert exact invariance
    to the input scale (identical std when the kernel is multiplied by a constant)."""
    import bae_search as nbs
    kern = nbs.make_kernel_search(nbs.PRIOR_PLAIN, kernel_input=True, currents=True)

    def std_logodds(Kc, N, m, seed=1):
        S = 1.0 * (np.random.default_rng(seed).random((N, m)) < 0.5)
        StS = S.T @ S
        StS_c = StS - np.outer(np.diag(StS), np.diag(StS)) / N
        scl = np.full(m, max(0.0, np.sum(np.diag(S.T @ Kc @ S)) / np.sum(StS_c ** 2)))
        sig2 = np.sum(Kc ** 2) / N ** 2                      # <K^2>, as the model uses
        out = np.zeros((N, m))
        kern(Kc, S.copy(), StS.copy(), np.zeros((m, 0)), N, scl, sig2, 1.0, 0.0, 0.0,
             np.zeros((m, m)), np.zeros(m), out, False)
        return np.std(out)

    def gram(N, d, cstd, seed=0):
        rng = np.random.default_rng(seed)
        Xin = cstd * rng.standard_normal((N, d)); Xin -= Xin.mean(0)
        K = Xin @ Xin.T
        return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()

    stds, scale_inv = [], True
    for N in [40, 160, 640]:
        for d in [20, 320]:
            base = None
            for cstd in [0.1, 1.0, 10.0]:               # 10^4 range in kernel entries
                s = std_logodds(gram(N, d, cstd), N, dim_hid)
                stds.append(s)
                if base is None:
                    base = s
                elif not np.isclose(s, base, rtol=1e-6):
                    scale_inv = False                    # must be exactly input-scale-free
    lo, hi = min(stds), max(stds)
    bounded = 0.1 < lo and hi < 10.0                     # O(1): no saturate / coin-flip
    ok = bounded and scale_inv
    print(f"[{'PASS' if ok else 'FAIL'}] log-odds std is O(1) vs N, d, input scale")
    print(f"    std range over N,d : [{lo:.2f}, {hi:.2f}]  (bounded O(1): {bounded})")
    print(f"    invariant to input scale : {scale_inv}")
    return ok


if __name__ == "__main__":
    X = make_data()
    results = []
    results.append(run_case("uniform scale, tree", X,
                            uniform_scale=True, sparse_reg=0.1, tree_reg=0.05))
    results.append(run_case("uniform scale, no-tree", X,
                            uniform_scale=True, sparse_reg=0.3, tree_reg=0.0))
    results.append(run_case("per-feature scale (l1)", X,
                            uniform_scale=False, l1_reg=0.05,
                            sparse_reg=0.1, tree_reg=0.05))
    results.append(run_case("per-feature, strong-tree", X,
                            uniform_scale=False, l1_reg=0.0,
                            sparse_reg=0.2, tree_reg=0.3))
    results.append(run_kernel_equiv("uniform scl", X, np.ones(4)))
    results.append(run_kernel_equiv("non-uniform scl", X,
                                    np.random.default_rng(3).uniform(0.5, 5.0, 4)))
    results.append(run_logodds(X))
    results.append(run_logodds_scaling())
    results.append(run_sample(X))
    print("\nALL PASS" if all(results) else "\nSOME FAILED")
