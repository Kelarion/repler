"""
bae_search.py  --  the compiled E-step
======================================

The compositional search: one
shared coordinate-descent *scaffold* + two injected, inlined "link" plugins,
so that the spike-only kernel (sbmf) and the spike-and-slab kernel (snmf) are
two instances of the SAME kernel rather than two hand-written copies.

The point being tested: inlining is preserved by specializing the kernel when
the configuration is known (a closure factory closing over the link), instead
of passing the link as a runtime argument (which would be an indirect,
un-inlinable call in the n*m*m hot loop).

A "link" is two element-wise pieces:
    score(E, wjj, tau, sigma2)            -> (logodds, mu, nu)   [no RNG]
    aux_update(Z, Seff, i, j, s, mu, nu)  -> None                [RNG + stores]
`score` runs before the Bernoulli flip; `aux_update` runs after (and is the
only place the slab draws its continuous Z), so the RNG-draw order matches the
originals exactly.

Two latent arrays, not three: `S` is the binary spike (drives the tree/StS
combinatorics) and `Z` is the *effective continuous latent* read by the field
accumulation.  The slab keeps the invariant Z==0 wherever the spike is off, so Z
already equals S*magnitude -- the original snmf's `if S[i,k]>0` gate is redundant
(a - 0.0 == a).  A no-slab model just initializes Z = S (the effective latent is
the spike) and aux_update keeps Z == S.

The scaffold owns everything shared: the permuted i,j loop, the neighbour-field
accumulation from the gram (off Z), the tree/sparsity A/B/C/D regularizer, the
StS bookkeeping (off S), and the sigmoid flip.

One factory builds the dense E-step kernel for any number of chains:
`make_dense_search(*link, prior, diag_gram=, debug=, parallel=)`.  Both the serial
(2-D state, plain @njit) and the chain-batched (leading chain axis + `prange`)
builds run the SAME per-chain sweep body (`_make_dense_sweep`); `parallel` just
picks whether it runs directly or inside `prange(C)`.  The operator
(bae_weights) passes parallel=(n_chains > 1), so a single chain keeps the exact
serial RNG stream (bit-for-bit vs old_bae_search.sbmf / snmf).

Run `python bae_search.py` (with the real src on PYTHONPATH) to compare the
serial scaffold against old_bae_search.sbmf / old_bae_search.snmf and the parallel one
against C serial calls, and to time both.
"""

import math
from functools import lru_cache

import numpy as np
from numba import njit, prange

import bae_util


# ---------------------------------------------------------------------------
#  Link plugins  (marked inline='always' so LLVM folds them into the loop)
# ---------------------------------------------------------------------------
#
# Binary spike (recovers old_bae_search.sbmf): the field is linear, magnitude is
# fixed at 1, nothing extra is sampled.

@njit(inline='always')
def score_binary(E, wjj, tau, sigma2):
    # logodds at the Gaussian precision 1/sigma2 (sbmf's XW/sigma2, WtW/sigma2);
    # tau unused.  sigma2 == 1 -> exactly E - 0.5*wjj (a/1.0 is exact).
    return (E - 0.5 * wjj) / sigma2, 0.0, 0.0


@njit(inline='always')
def aux_binary(Z, i, j, s, mu, nu):
    Z[i, j] = s                             # effective latent == the spike


# Spike-and-slab (recovers old_bae_search.snmf): the field is the partially
# collapsed marginal over the continuous magnitude, and when the spike turns on
# we draw that magnitude from a truncated normal.

@njit(inline='always')
def score_slab(E, wjj, tau, sigma2):
    # wjj is the TRUE gram diagonal ||W[:,j]||^2 (the prior on S is now a separate
    # additive term, no longer folded into WtW), so the only degeneracy left is a
    # genuinely dead weight column (wjj -> 0) or an overflowing W (wjj/E/sigma2 ->
    # inf/NaN).  Either makes nu = sqrt(sigma2/wjj) and mu = .../wjj non-finite,
    # which (a) divides by zero in this kernel and (b) feeds NaN/inf magnitudes to
    # the truncated-normal sampler.  In that case the column carries no usable
    # signal, so report the spike as strongly OFF with a bounded, well-defined
    # (mu, nu): the flip turns off, no magnitude is drawn, nothing blows up.  A
    # no-op when wjj is a healthy positive precision, so well-behaved fits are
    # unchanged.
    if (not (math.isfinite(wjj) and wjj > 1e-12)
            or not (math.isfinite(sigma2) and sigma2 > 0.0)
            or not math.isfinite(E)):
        return -1e18, 0.0, 1.0
    mu = (E - tau * sigma2) / wjj
    nu = math.sqrt(sigma2 / wjj)
    logodds = (math.log(tau * nu * math.sqrt(2 * math.pi))
               + (mu ** 2) / (2 * (sigma2 / wjj))
               + bae_util.log_ndtr(mu / nu))
    return logodds, mu, nu


@njit(inline='always')
def aux_slab(Z, i, j, s, mu, nu):
    if s > 0.5:
        Z[i, j] = bae_util.sample_trunc_norm(mu, nu)
    else:
        Z[i, j] = 0.0                       # maintains Z==0 off-spike


# ---------------------------------------------------------------------------
#  Prior plugins  (the latent prior's additive contribution to the log-odds)
# ---------------------------------------------------------------------------
#
# The conditional log-odds for the spike S_ij factors as
#     score(E, wjj)                          # LIKELIHOOD, integrates the slab Z
#   + [ -alpha - beta*inhib(S,StS) + sum_k A_jk S_ik + c_j ]   # PRIOR on S
# The bracket is purely additive and *linear in the binary spikes S_ik*, so it is
# its own plugin -- contributed by the LatentPrior, mirroring how the slab link is
# contributed by the slab.  It bundles every prior term: sparsity (-alpha), the
# tree/StS combinatorial reg (-beta*inhib), and -- for a Boltzmann prior -- the
# Ising conditional (the coupling Jc, field hc).  Crucially it reads the binary
# spike S_ik (NOT the magnitude Z_ik) and never touches wjj: folding the coupling
# into WtW (`WtW+J`) only reproduces this in the binary kernel, where score is
# affine and Z==S; for the slab it couples to the magnitude and corrupts the Z
# posterior.  Same signature for every plugin (the unstructured one ignores Jc/hc,
# which the model passes as zeros), so the factory can inject either.

@njit(inline='always')
def _tree_inhib(S, i, j, StS, N, beta):
    # the A/B/C/D StS combinatorics, summed over ALL k (incl. k==j), verbatim from
    # old_bae_search.snmf -- so the unstructured prior reproduces the old scaffold bit
    # for bit.  Gated on beta so a no-tree model pays nothing.
    if not (beta > 1e-6):
        return 0.0
    m = S.shape[1]
    Sij = S[i, j]
    inhib = 0.0
    for k in range(m):
        Sik = S[i, k]
        A = StS[j, k] - Sij * Sik
        B = StS[j, j] - A - Sij
        C = StS[k, k] - A
        D = N - A - B - C
        if A < min(B, C - 1, D):
            inhib += Sik
        if B < min(A, C, D - 1):
            inhib += 1 - Sik
        if C <= min(A, B, D):
            inhib -= Sik
        if D <= min(A, B, C):
            inhib -= 1 - Sik
    return inhib


@njit(inline='always')
def prior_unstructured(S, i, j, StS, N, Jc, hc, alpha, beta, prior_temp):
    """LatentPrior: sparsity + tree/StS reg.  No coupling (Jc, hc, prior_temp
    unused -- an unstructured prior has no temperature-scaled term)."""
    return -alpha - beta * _tree_inhib(S, i, j, StS, N, beta)


@njit(inline='always')
def prior_boltzmann(S, i, j, StS, N, Jc, hc, alpha, beta, prior_temp):
    """BoltzmannPrior: sparsity + tree + the Ising conditional, additive and linear
    in the binary spikes S_ik.  Jc is the symmetric {0,1}-coupling (zero diagonal)
    and hc the {0,1}-field, i.e. (2J, 2h) for a prior log p(S) ~ S'JS + 2h'S, so
    this accumulates exactly its conditional log-odds 2*(J S + h)_j.  Both are
    sigma2-independent: J is learned from the spike statistics alone, so the E-step
    applies it at face value.

    `prior_temp` is the prior temperature: the coupling contribution is divided by
    it, so high prior_temp = weak structure.  This is the ONLY place it acts, so it
    tempers how strongly the structure biases the sampling of S without touching
    what the structure learns, and the fit loop can anneal it freely (see
    bae_priors.TempSchedule).  The sparsity/tree terms are NOT scaled (they are
    separate regularizers, not part of the temperature-controlled Boltzmann prior)."""
    lp = -alpha - beta * _tree_inhib(S, i, j, StS, N, beta)
    m = S.shape[1]
    coup = hc[j]
    for k in range(m):
        if k != j:
            coup += Jc[j, k] * S[i, k]
    return lp + coup / prior_temp


# ---------------------------------------------------------------------------
#  The scaffold factory
# ---------------------------------------------------------------------------
#
# Closes over (score, aux_update, prior) so they are compile-time freevar
# constants -> inlinable.  Memoized so each distinct (link, prior) compiles the
# kernel exactly once (the "construct the search when the class is created" step).
# The scaffold owns only what is genuinely shared: the permuted i,j loop, the
# LIKELIHOOD field (off Z, true gram), the Bernoulli flip, and the StS
# bookkeeping.  The likelihood is `score`, the magnitude draw `aux_update`, and
# the whole additive S-prior `prior` -- the prior runs its own k-loop, which
# numba inlines into the sweep (no call overhead, no second pass kept around).

@lru_cache(maxsize=None)
def _make_dense_sweep(score, aux_update, prior, diag_gram=False, debug=False):
    """ONE chain's 2-D coordinate-descent sweep -- the shared body both the serial
    and the parallel dense search run.  Operates on 2-D state (S/Z/WtW/StS/XW),
    scalar sigma2, and a 2-D-or-None `out`; the serial search calls it directly and
    the parallel one calls it per chain inside `prange(C)`, so the sweep logic lives
    in exactly one place.

    `score`, `aux_update`, `prior`, `diag_gram`, `debug` are compile-time freevar
    constants: the link plugins (marked inline='always') fold into the loop, and the
    diag_gram / debug branches are pruned by LLVM (no runtime cost).

    * `diag_gram` -- set when the operator's gram is diagonal (orthonormal W,
      WtW == I, e.g. BiPCA): then sum_k WtW[j,k] Z[i,k] is exactly XW[i,j], so the
      O(m) neighbour loop is pure waste and is eliminated entirely.
    * `debug` -- samples EXACTLY as normal but also records each element's log-odds
      (logodds + prior, pre-temp) into `out` as it sweeps, so you can watch the
      current magnitudes over a real fit.  Off (the normal path) never touches `out`.
    Memoized, so the serial and parallel builds of the same (link, prior, diag_gram,
    debug) share one compiled sweep."""

    @njit
    def sweep(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
              Jc, hc, inplace, out, prior_temp):
        n, m = S.shape
        regularize = beta > 1e-6

        for i in np.random.permutation(np.arange(n)):
            for j in np.random.permutation(np.arange(m)):

                Sij = S[i, j]

                # LIKELIHOOD field: off Z, true gram only (no prior leakage).
                # With a diagonal gram the neighbour sum vanishes -> skip the loop.
                E = XW[i, j]
                if not diag_gram:
                    for k in range(m):
                        if k != j:
                            E -= WtW[j, k] * Z[i, k]

                logodds, mu, nu = score(E, WtW[j, j], tau, sigma2)

                # PRIOR on S: additive, owned by the latent prior (sparsity +
                # tree + any coupling), linear in the binary spikes.  prior_temp
                # scales the structured (coupling) part only.
                lp = prior(S, i, j, StS, N, Jc, hc, alpha, beta, prior_temp)

                if debug:
                    out[i, j] = logodds + lp     # record, then sample as usual

                curr = (logodds + lp) / temp

                if curr < -100:
                    prob = 0.0
                elif curr > 100:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-curr))

                new_Sij = 1.0 * (np.random.rand() < prob)
                ds = new_Sij - Sij

                # StS bookkeeping (the prior's state; maintained by the scaffold)
                if regularize and inplace:
                    StS[j, j] += ds
                    for k in range(m):
                        if k != j:
                            StS[j, k] += S[i, k] * ds
                            StS[k, j] += S[i, k] * ds

                S[i, j] = new_Sij
                aux_update(Z, i, j, new_Sij, mu, nu)

        return S, Z

    return sweep


def _build_dense_search(score, aux_update, prior, diag_gram=False, debug=False,
                        parallel=False):
    """The dense E-step kernel, in one factory.  `parallel` (a compile-time flag)
    selects between two thin wrappers around the shared `_make_dense_sweep` body:

      parallel=False -- a plain @njit over 2-D state (S/Z/... shaped (n,m)/(m,m),
                        sigma2 a scalar).  This is the single-chain path; it keeps
                        the exact serial RNG stream, so it stays bit-for-bit equal
                        to old_bae_search.sbmf / snmf.
      parallel=True  -- an @njit(parallel=True) that runs the SAME sweep per chain
                        inside `prange(C)` over 3-D state (leading chain axis C,
                        sigma2 (C,)).  Chains are embarrassingly parallel (each owns
                        its own S/Z/StS/W), so the prange is race-free without locks.

    Two wrappers rather than one because numba types a function on its argument
    shapes: a 2-D and a 3-D `S` cannot share one compiled body, and routing a single
    chain through `prange` would change its RNG stream (per-thread RNG).  The sweep
    LOGIC is shared, though -- only the outer iteration (nothing vs prange over
    chains) and the per-chain indexing differ.  Memoized on
    (link, prior, diag_gram, debug, parallel) via make_dense_search, so each variant
    compiles once.  The operator (bae_weights) passes parallel=(n_chains > 1)."""

    sweep = _make_dense_sweep(score, aux_update, prior, diag_gram, debug)

    if parallel:
        @njit(parallel=True)
        def search(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
                   Jc, hc, inplace=True, out=None, prior_temp=None):
            C = S.shape[0]
            for c in prange(C):
                # `prior_temp` is per chain here ((C,), like sigma2) -- the prior
                # temperature may differ across chains (e.g. AdaptiveTemp with a
                # per-chain loss), so each chain divides its coupling by its own.
                # `out` is (C,n,m) under debug (else None); the if/else is pruned
                # since `debug` is a compile-time constant, so None is never indexed.
                if debug:
                    sweep(XW[c], S[c], Z[c], WtW[c], StS[c], N, temp, alpha, beta,
                          tau, sigma2[c], Jc[c], hc[c], inplace, out[c], prior_temp[c])
                else:
                    sweep(XW[c], S[c], Z[c], WtW[c], StS[c], N, temp, alpha, beta,
                          tau, sigma2[c], Jc[c], hc[c], inplace, out, prior_temp[c])
            return S, Z
    else:
        @njit
        def search(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
                   Jc, hc, inplace=True, out=None, prior_temp=1.0):
            sweep(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
                  Jc, hc, inplace, out, prior_temp)
            return S, Z

    search._dense_sweep = sweep    # exposed for the inlining self-test
    return search


make_dense_search = lru_cache(maxsize=None)(_build_dense_search)


# ---------------------------------------------------------------------------
#  The kernel scaffold factory  (Gram factorization -- NOT a linear operator)
# ---------------------------------------------------------------------------
#
# KernelBMF factorizes a Gram/kernel matrix: K ~ center(S diag(scl) S^T).  The
# reconstruction is QUADRATIC in the binary latents S, so there is no linear
# operator (no forward/drive/gram) and it cannot ride make_dense_search.  But the
# conditional log-odds still factors the same way,
#     curr = ( LIKELIHOOD field  +  additive S-prior ) / temp ,
# and the additive S-prior is *exactly* the same plugin the dense models use
# (sparsity + tree combinatorics + any Boltzmann coupling).  So this scaffold
# reuses `prior` verbatim -- a plain LatentPrior reproduces old_bae_search.kerbmf3
# bit-for-bit, and a BoltzmannPrior gives KernelBMF a structured Ising prior for
# free.  Only the LIKELIHOOD field is kernel-specific and lives here.
#
# The field is the response of <data, S diag(scl) S^T> to flipping S_ij, weighted
# by the per-feature scale scl[j] (self) and scl[k] (neighbours), verbatim from
# old_bae_search.kerbmf3 (feature input) / kerbmf2 (kernel input).  `kernel_input`
# (compile-time) selects the input: a feature matrix X (n, d), for which the
# running StX = S^T X is maintained live and the field costs O(n d); or a
# precomputed kernel K (n, n), which needs no StX and costs O(n^2).  The two forms
# compute the identical field when K = X X^T.

def _build_kernel_search(prior, kernel_input=False, currents=False, debug=False):
    # `currents` (compile-time) turns the kernel into a diagnostic: instead of
    # sampling, it writes the raw per-element log-odds (like + prior, pre-temp)
    # into `out` and does NOT flip -- so every (i,j) is scored against the SAME
    # fixed S.  Used by test_kernelbmf2_port.py to check the field against the
    # exact change in log-likelihood.
    #
    # `debug` (compile-time) is the live-fit counterpart: it samples EXACTLY as the
    # normal path (flips, updates StS) but ALSO records each element's log-odds into
    # `out` as it sweeps -- so you can watch the current magnitudes evolve over a
    # real fit (out[i,j] is the log-odds at the moment (i,j) was resampled).  Both
    # off (the normal sampling path) never touches `out`, so callers pass a dummy.

    @njit
    def search(data, S, StS, StX, N, scl, sigma2, temp, alpha, beta, Jc, hc, out,
               inplace=True, prior_temp=1.0):
        n, m = S.shape
        t = (N - 1) / N

        for i in np.random.permutation(np.arange(n)):
            for j in np.random.permutation(np.arange(m)):

                Sij = S[i, j]
                S_j = (StS[j, j] - Sij) / (N - 1)
                pij = scl[j]

                # LIKELIHOOD field: response of <data, S diag(scl) S^T> to a flip.
                # feature form uses the live StX = S^T X; kernel form reads K
                # directly.  Identical value when K == X X^T.
                if kernel_input:
                    inp = (1 - 2 * Sij) * data[i, i] / t
                    for k in range(n):
                        inp += 2 * data[i, k] * S[k, j] / t
                else:
                    inp = 0.0
                    d = StX.shape[1]
                    for k in range(d):
                        inp += (2 * StX[j, k] * data[i, k]
                                + (1 - 2 * Sij) * data[i, k] ** 2) / t

                # normalizer recurrence off StS (the ||center(S diag(scl) S^T)||^2
                # term), each neighbour weighted by its scale scl[k] -- kerbmf3.
                dot = pij * (t * (2 * (N - 2) * S_j * (1 - S_j) + 1) * (0.5 - Sij))
                for k in range(m):
                    Sik = S[i, k]
                    S_k = (StS[k, k] - Sik) / (N - 1)
                    term = 2 * (StS[j, k] - Sij * Sik
                                + t * (0.5 - S_j - S_k - (N - 2) * S_j * S_k)) * (Sik - S_k)
                    term -= t * (1 - S_k) * S_k * (2 * S_j - 1)
                    dot += scl[k] * term

                # Divide the LIKELIHOOD field by the entry-noise variance sigma2
                # (the Gaussian precision, exactly like the dense scaffold's
                # (E - 0.5 wjj)/sigma2), NOT by a fixed N.  The raw field
                # pij*(inp - dot) has magnitude ~ <K^2> (the mean squared kernel
                # entry) independent of N, so with sigma2 = <K^2> = data_norm/N^2
                # the log-odds is O(1) in N, d and the input scale -- which keeps
                # the sigmoid out of both the saturated and the coin-flip regimes.
                # (The old /(N-1) fixed sigma2 = N-1, an input-scale-blind noise
                # level that made the log-odds ~ <K^2>/N: tiny for O(1)-entry
                # kernels, huge for raw Grams.)
                like = pij * (inp - dot) / sigma2

                # additive S-prior (sparsity + tree + coupling): the SHARED plugin
                # (prior_temp scales the structured coupling part only)
                lp = prior(S, i, j, StS, N, Jc, hc, alpha, beta, prior_temp)

                if currents:
                    out[i, j] = like + lp        # diagnostic: score, do not flip
                    continue

                if debug:
                    out[i, j] = like + lp        # record, then sample as usual

                curr = (like + lp) / temp

                if curr < -100:
                    prob = 0.0
                elif curr > 100:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-curr))

                ds = (1.0 * (np.random.rand() < prob)) - Sij
                S[i, j] += ds

                if inplace:
                    StS[j, j] += ds
                    for k in range(m):
                        if k != j:
                            StS[j, k] += S[i, k] * ds
                            StS[k, j] += S[i, k] * ds
                    if not kernel_input:
                        d = StX.shape[1]
                        for k in range(d):
                            StX[j, k] += data[i, k] * ds

        return S

    return search


make_kernel_search = lru_cache(maxsize=None)(_build_kernel_search)

# A model composes a "link" -- the (score, aux_update) pair from the slab side --
# with a "prior" -- the additive S-prior from the LatentPrior.  no-slab models use
# BINARY_LINK, slab models SLAB_LINK; unstructured priors use PRIOR_PLAIN,
# Boltzmann-structured ones PRIOR_BOLTZMANN.
BINARY_LINK = (score_binary, aux_binary)
SLAB_LINK = (score_slab, aux_slab)
PRIOR_PLAIN = prior_unstructured
PRIOR_BOLTZMANN = prior_boltzmann

# prebuilt kernels (compiled lazily on first call); unstructured prior here, the
# structured ones are composed by the model from its prior's plugin.  The `par_*`
# variants are the chain-batched (prange) builds the operators use when n_chains>1.
sbmf_search = make_dense_search(*BINARY_LINK, PRIOR_PLAIN)
snmf_search = make_dense_search(*SLAB_LINK, PRIOR_PLAIN)
par_sbmf_search = make_dense_search(*BINARY_LINK, PRIOR_PLAIN, parallel=True)
par_snmf_search = make_dense_search(*SLAB_LINK, PRIOR_PLAIN, parallel=True)


# ===========================================================================
#  self-test  (vs old_bae_search.sbmf / old_bae_search.snmf)
# ===========================================================================

@njit
def _seed(s):
    np.random.seed(s)


def _selftest():
    import time
    import old_bae_search

    rng = np.random.RandomState(0)
    n, d, m = 200, 50, 8
    W = rng.randn(d, m)
    b = rng.randn(d)
    X = rng.randn(n, d)
    XW = (X - b) @ W
    WtW = W.T @ W
    S0 = 1.0 * (rng.rand(n, m) > 0.5)
    temp, alpha, beta = 0.7, 0.0, 1e-2
    SEED = 12345

    def StS_of(S):
        return S.T @ S

    # ---- BINARY: unified vs sbmf -----------------------------------------
    S_a, StS_a = S0.copy(), StS_of(S0)
    _seed(SEED)
    old_bae_search.sbmf(XW.copy(), S_a, WtW.copy(), temp=temp,
                    StS=StS_a, N=n, beta=beta, alpha=alpha, inplace=True)

    S_b, Z_b, StS_b = S0.copy(), S0.copy(), StS_of(S0)   # no-slab: Z = S
    zeroJ, zeroh = np.zeros((m, m)), np.zeros(m)
    _seed(SEED)
    sbmf_search(XW.copy(), S_b, Z_b, WtW.copy(), StS_b, n,
                temp, alpha, beta, 1.0, 1.0, zeroJ, zeroh, True)

    bin_equal = np.array_equal(S_a, S_b)
    bin_frac = float(np.mean(S_a == S_b))
    print(f"[binary]  S array_equal vs sbmf : {bin_equal}   "
          f"(agreement {bin_frac:.5f})")

    # ---- SLAB: unified vs snmf -------------------------------------------
    Z0 = 1.0 * S0 * np.abs(rng.randn(n, m))
    tau, sigma2 = 1.0, 1.0

    S_c, Z_c, StS_c = S0.copy(), Z0.copy(), StS_of(S0)
    _seed(SEED)
    old_bae_search.snmf(XW.copy(), S_c, Z_c, WtW.copy(), temp=temp,
                    sigma2=sigma2, tau=tau, StS=StS_c, N=n,
                    beta=beta, alpha=alpha, inplace=True)

    S_d = S0.copy()
    Z_d = Z0.copy()                          # Z0 already zeroed off-spike
    StS_d = StS_of(S0)
    _seed(SEED)
    snmf_search(XW.copy(), S_d, Z_d, WtW.copy(), StS_d, n,
                temp, alpha, beta, tau, sigma2, zeroJ, zeroh, True)

    slab_S_equal = np.array_equal(S_c, S_d)
    slab_Z_equal = np.allclose(Z_c, Z_d)
    slab_Z_exact = np.array_equal(Z_c, Z_d)
    print(f"[slab]    S array_equal vs snmf : {slab_S_equal}")
    print(f"[slab]    Z array_equal vs snmf : {slab_Z_exact}   "
          f"(allclose {slab_Z_equal})")

    # ---- inlining evidence: scan the compiled LLVM IR --------------------
    def link_calls(kernel):
        llvm = kernel.inspect_llvm()
        txt = "\n".join(llvm.values())
        # any 'call' line that references the link helpers by name
        hits = [ln.strip() for ln in txt.splitlines()
                if " call " in ln and ("score_" in ln or "aux_" in ln
                                       or "log_ndtr" in ln
                                       or "sample_trunc_norm" in ln)]
        return hits

    print(f"[inolg]   binary kernel: residual link calls = "
          f"{len(link_calls(sbmf_search._dense_sweep))}")
    print(f"[inolg]   slab   kernel: residual link calls = "
          f"{len(link_calls(snmf_search._dense_sweep))}")

    # ---- timing -----------------------------------------------------------
    def timeit(fn, reps=50):
        fn()                                   # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3   # ms/sweep

    t_sbmf = timeit(lambda: old_bae_search.sbmf(
        XW.copy(), S0.copy(), WtW.copy(), temp=temp,
        StS=StS_of(S0), N=n, beta=beta, alpha=alpha, inplace=True))
    t_bin = timeit(lambda: sbmf_search(
        XW.copy(), S0.copy(), S0.copy(), WtW.copy(),
        StS_of(S0), n, temp, alpha, beta, 1.0, 1.0, zeroJ, zeroh, True))
    print(f"[time]    sbmf {t_sbmf:.3f} ms  | unified-binary {t_bin:.3f} ms  "
          f"({t_bin / t_sbmf:.2f}x)")

    t_snmf = timeit(lambda: old_bae_search.snmf(
        XW.copy(), S0.copy(), Z0.copy(), WtW.copy(), temp=temp,
        sigma2=sigma2, tau=tau, StS=StS_of(S0), N=n,
        beta=beta, alpha=alpha, inplace=True))
    t_slab = timeit(lambda: snmf_search(
        XW.copy(), S0.copy(), Z0.copy(), WtW.copy(),
        StS_of(S0), n, temp, alpha, beta, tau, sigma2, zeroJ, zeroh, True))
    print(f"[time]    snmf {t_snmf:.3f} ms  | unified-slab   {t_slab:.3f} ms  "
          f"({t_slab / t_snmf:.2f}x)")


def _selftest_parallel():
    """C parallel chains vs C serial calls.  Per-thread RNG in a prange means we
    can't match a specific serial call bit-for-bit, so we check that (a)
    shapes/dtypes line up, (b) every spike is a valid {0,1} draw, and (c) the
    parallel sweep reduces the per-chain energy like the serial one."""
    import time

    rng = np.random.RandomState(0)
    C, n, d, m = 6, 200, 50, 8
    temp, alpha, beta, tau, sig = 0.7, 0.0, 1e-2, 1.0, 1.0

    W = rng.randn(C, d, m)                 # distinct W per chain -> chains differ
    X = rng.randn(n, d)
    XW = np.einsum('nd,cdm->cnm', X, W)
    WtW = np.einsum('cdm,cdk->cmk', W, W)
    S0 = 1.0 * (rng.rand(C, n, m) > 0.5)
    StS0 = np.einsum('cnm,cnk->cmk', S0, S0)
    Jc = np.zeros((C, m, m))
    hc = np.zeros((C, m))
    sigma2 = np.full(C, sig)
    prior_temp = np.ones(C)          # per-chain prior temperature ((C,), like sigma2)

    Sp, Zp, StSp = S0.copy(), S0.copy(), StS0.copy()
    par_sbmf_search(XW.copy(), Sp, Zp, WtW.copy(), StSp, n,
                    temp, alpha, beta, tau, sigma2, Jc, hc, True, None, prior_temp)

    Ss = S0.copy()
    for c in range(C):
        Sc, Zc, StSc = S0[c].copy(), S0[c].copy(), StS0[c].copy()
        sbmf_search(XW[c].copy(), Sc, Zc, WtW[c].copy(), StSc, n,
                    temp, alpha, beta, tau, sig,
                    np.zeros((m, m)), np.zeros(m), True)
        Ss[c] = Sc

    print(f"[par shape]   parallel S {Sp.shape}  serial-stack S {Ss.shape}")
    print(f"[par valid]   all spikes in {{0,1}}: {np.all((Sp == 0) | (Sp == 1))}")

    def energy(S):
        recon = np.einsum('cnm,cdm->cnd', S, W)
        return ((X[None] - recon) ** 2).mean((1, 2))

    e0, ep, es = energy(S0), energy(Sp), energy(Ss)
    print(f"[par energy]  init          : {np.round(e0, 3)}")
    print(f"[par energy]  parallel sweep: {np.round(ep, 3)}  "
          f"(all decreased: {np.all(ep < e0)})")
    print(f"[par energy]  serial  sweep : {np.round(es, 3)}  "
          f"(all decreased: {np.all(es < e0)})")

    def timeit(fn, reps=30):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3

    t_par = timeit(lambda: par_sbmf_search(
        XW.copy(), S0.copy(), S0.copy(), WtW.copy(), StS0.copy(), n,
        temp, alpha, beta, tau, sigma2, Jc, hc, True, None, prior_temp))

    def serial_all():
        for c in range(C):
            sbmf_search(XW[c].copy(), S0[c].copy(), S0[c].copy(),
                        WtW[c].copy(), StS0[c].copy(), n,
                        temp, alpha, beta, tau, sig,
                        np.zeros((m, m)), np.zeros(m), True)
    t_ser = timeit(serial_all)
    print(f"[par time]    {C} chains: parallel {t_par:.3f} ms | serial {t_ser:.3f} "
          f"ms ({t_ser / t_par:.2f}x)")


if __name__ == "__main__":
    _selftest()
    print()
    _selftest_parallel()
