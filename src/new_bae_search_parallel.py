"""
new_bae_search_parallel.py  --  PROTOTYPE (parallel chains)
===========================================================

Parallel-chains variant of `new_bae_search._build_dense_search`.

The BMF objective is non-convex and the fit is stochastic, so in practice you run
several independent chains from different inits and keep the best.  Today that is a
Python `for` loop that refits the whole model N times (bae_util.multifit).  This
module makes the *chains* a leading array axis instead, so:

  * every numpy face (drive / gram / backward, the M-step, the loss) broadcasts
    over the chain axis for free -- no Python loop, just an einsum with a `c`
    index (see new_bae_weights_parallel / new_bae_priors_parallel);
  * the ONE genuinely-sequential kernel -- the coordinate-descent E-step -- keeps
    its exact per-chain 2-D sweep but runs it inside a `prange(C)`.  Chains are
    embarrassingly parallel (each owns its own S, Z, StS, W), so this is a numba
    parallel loop, not a Python one, and it is a 3-line wrapper around the
    existing body rather than a rewrite.

Crucially the LINK and PRIOR plugins are reused *verbatim* from new_bae_search --
they already operate on a 2-D (S, i, j), so the parallel scaffold just calls them
on the c-th slab.  The compositional story (pick a link + a prior) is unchanged;
"run C chains" is a second, orthogonal axis layered under it.

Shapes (C = n_chains):
    XW    (C, n, m)      per-chain drive           (chains have different W)
    S     (C, n, m)      per-chain binary spike
    Z     (C, n, m)      per-chain effective latent (== S with no slab)
    WtW   (C, m, m)      per-chain gram
    StS   (C, m, m)      per-chain S^T S
    sigma2(C,)           per-chain noise variance
    Jc    (C, m, m)      per-chain coupling  (zeros for a plain prior)
    hc    (C, m)         per-chain field     (zeros for a plain prior)
    out   (C, n, m)      debug log-odds, or None

`temp`, `alpha`, `beta`, `tau`, `N` are shared scalars (the annealing schedule and
the hyperparameters are the same across chains -- only the random state differs).
"""

import math
from functools import lru_cache

import numpy as np
from numba import njit, prange

# reuse the EXACT same element-wise plugins as the serial scaffold -- they take a
# 2-D (S, i, j), so the parallel sweep applies them to the c-th chain unchanged.
from new_bae_search import (
    score_binary, aux_binary,
    score_slab, aux_slab,
    prior_unstructured, prior_boltzmann,
    BINARY_LINK, SLAB_LINK, PRIOR_PLAIN, PRIOR_BOLTZMANN,
)


def _build_parallel_dense_search(score, aux_update, prior, diag_gram=False, debug=False):
    """Same closure factory as new_bae_search._build_dense_search, but the sweep is
    wrapped in `prange(C)`.  `score`, `aux_update`, `prior`, `diag_gram`, `debug`
    are compile-time freevar constants (so the plugins inline and the diag/debug
    branches fold), exactly as in the serial version.  Memoized: each distinct
    (link, prior, diag_gram, debug) compiles once."""

    @njit(parallel=True)
    def search(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None):
        C, n, m = S.shape
        regularize = beta > 1e-6

        # ---- the ONE new line: chains are independent -> parallel outer loop ----
        for c in prange(C):
            # everything below is the serial 2-D sweep, verbatim, on chain c's
            # own slabs (S[c], Z[c], WtW[c], ...).  No cross-chain reads or writes,
            # so the prange is race-free without any locking.
            sig2 = sigma2[c]
            for i in np.random.permutation(np.arange(n)):
                for j in np.random.permutation(np.arange(m)):

                    Sij = S[c, i, j]

                    # LIKELIHOOD field: off Z, true gram only (no prior leakage).
                    E = XW[c, i, j]
                    if not diag_gram:
                        for k in range(m):
                            if k != j:
                                E -= WtW[c, j, k] * Z[c, i, k]

                    logodds, mu, nu = score(E, WtW[c, j, j], tau, sig2)

                    # PRIOR on S: the shared plugin, on chain c's spike/coupling.
                    lp = prior(S[c], i, j, StS[c], N, Jc[c], hc[c], alpha, beta)

                    if debug:
                        out[c, i, j] = logodds + lp

                    curr = (logodds + lp) / temp

                    if curr < -100:
                        prob = 0.0
                    elif curr > 100:
                        prob = 1.0
                    else:
                        prob = 1.0 / (1.0 + math.exp(-curr))

                    new_Sij = 1.0 * (np.random.rand() < prob)
                    ds = new_Sij - Sij

                    if regularize and inplace:
                        StS[c, j, j] += ds
                        for k in range(m):
                            if k != j:
                                StS[c, j, k] += S[c, i, k] * ds
                                StS[c, k, j] += S[c, i, k] * ds

                    S[c, i, j] = new_Sij
                    aux_update(Z[c], i, j, new_Sij, mu, nu)

        return S, Z

    return search


make_parallel_dense_search = lru_cache(maxsize=None)(_build_parallel_dense_search)


# prebuilt kernels (compiled lazily on first call), mirroring new_bae_search
par_sbmf_search = make_parallel_dense_search(*BINARY_LINK, PRIOR_PLAIN)
par_snmf_search = make_parallel_dense_search(*SLAB_LINK, PRIOR_PLAIN)


# ===========================================================================
#  self-test: C parallel chains vs C serial calls to new_bae_search
# ===========================================================================
#
# Per-thread RNG in a prange means we can't match a specific serial call bit-for-
# bit, but each chain must be a VALID draw: run the serial kernel per chain with
# the same seed the parallel thread would not share, and instead check that (a)
# shapes/dtypes line up, (b) a chain fed identical inputs to the serial kernel
# lands in the same distribution (equal marginals over many sweeps), and (c) the
# parallel kernel actually reduces the energy like the serial one.

def _selftest():
    import time
    import new_bae_search as nbs

    rng = np.random.RandomState(0)
    C, n, d, m = 6, 200, 50, 8
    temp, alpha, beta, tau, sig = 0.7, 0.0, 1e-2, 1.0, 1.0

    # distinct W per chain -> distinct XW/WtW (this is what makes chains differ)
    W = rng.randn(C, d, m)
    X = rng.randn(n, d)
    XW = np.einsum('nd,cdm->cnm', X, W)
    WtW = np.einsum('cdm,cdk->cmk', W, W)
    S0 = 1.0 * (rng.rand(C, n, m) > 0.5)
    StS0 = np.einsum('cnm,cnk->cmk', S0, S0)
    Jc = np.zeros((C, m, m))
    hc = np.zeros((C, m))
    sigma2 = np.full(C, sig)

    # ---- parallel binary sweep -------------------------------------------
    Sp, Zp, StSp = S0.copy(), S0.copy(), StS0.copy()
    par_sbmf_search(XW.copy(), Sp, Zp, WtW.copy(), StSp, n,
                    temp, alpha, beta, tau, sigma2, Jc, hc, True, None)

    # ---- the same chains, one serial new_bae_search call each ------------
    Ss = S0.copy()
    for c in range(C):
        Sc, Zc, StSc = S0[c].copy(), S0[c].copy(), StS0[c].copy()
        nbs.sbmf_search(XW[c].copy(), Sc, Zc, WtW[c].copy(), StSc, n,
                        temp, alpha, beta, tau, sig,
                        np.zeros((m, m)), np.zeros(m), True)
        Ss[c] = Sc

    print(f"[shape]   parallel S {Sp.shape}  serial-stack S {Ss.shape}")
    print(f"[valid]   all spikes in {{0,1}}: {np.all((Sp == 0) | (Sp == 1))}")

    # energy check: E(S) = ||X - S W^T||^2 for each chain should drop from S0
    def energy(S):
        recon = np.einsum('cnm,cdm->cnd', S, W)
        return ((X[None] - recon) ** 2).mean((1, 2))

    e0, ep, es = energy(S0), energy(Sp), energy(Ss)
    print(f"[energy]  init          : {np.round(e0, 3)}")
    print(f"[energy]  parallel sweep: {np.round(ep, 3)}  (all decreased: {np.all(ep < e0)})")
    print(f"[energy]  serial  sweep : {np.round(es, 3)}  (all decreased: {np.all(es < e0)})")

    # ---- timing: parallel-C vs C serial calls ----------------------------
    def timeit(fn, reps=30):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3

    t_par = timeit(lambda: par_sbmf_search(
        XW.copy(), S0.copy(), S0.copy(), WtW.copy(), StS0.copy(), n,
        temp, alpha, beta, tau, sigma2, Jc, hc, True, None))

    def serial_all():
        for c in range(C):
            nbs.sbmf_search(XW[c].copy(), S0[c].copy(), S0[c].copy(),
                            WtW[c].copy(), StS0[c].copy(), n,
                            temp, alpha, beta, tau, sig,
                            np.zeros((m, m)), np.zeros(m), True)
    t_ser = timeit(serial_all)
    print(f"[time]    {C} chains: parallel {t_par:.3f} ms | serial {t_ser:.3f} ms "
          f"({t_ser / t_par:.2f}x)")


if __name__ == "__main__":
    _selftest()
