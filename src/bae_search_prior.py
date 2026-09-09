"""
bae_search_prior.py  --  PROTOTYPE (factorization sketch)
=============================================================

Extends the compositional search of bae_search.py with a THIRD injected
component, so the discrete E-step is now three orthogonal plugins instead of two:

    score(E, wjj, tau, sigma2)            -> (logodds, mu, nu)   [LIKELIHOOD]
    aux_update(Z, i, j, s, mu, nu)        -> None                [SLAB magnitude]
    prior(S, i, j, StS, N, Jc, hc, alpha, beta) -> float         [PRIOR on S]

The motivation is the spike-and-slab + Boltzmann derivation: the conditional
log-odds for the spike S_ij factors as

    logodds(S_ij) = Lambda(E, wjj)            # integrated-Z LIKELIHOOD ratio
                  + [ -alpha                   # sparsity
                      - beta * inhib(S,StS)    # tree / StS combinatorial reg
                      + sum_k A_jk S_ik + c_j ]# Boltzmann (Ising) conditional

The bracket is the PRIOR on S.  It is *purely additive* and *linear in the
binary spikes S_ik*, and it must NOT pass through the (nonlinear) slab
likelihood Lambda.  The old code folded the Ising coupling into WtW (`WtW + J`),
which is only valid for the binary kernel where Lambda is affine and Z == S; for
the slab it (1) couples the prior to the continuous magnitude Z_ik instead of the
spike S_ik, and (2) corrupts wjj (the Z-posterior precision), distorting both the
marginal likelihood and the sampled magnitude.  See the derivation in the chat.

So the prior becomes its own plugin, contributed by the LatentPrior component
(it owns sparsity, the tree/StS reg, and -- for a Boltzmann prior -- the
coupling).  The operator still owns the scaffold (the WtW likelihood field, off
Z) and the StS bookkeeping; the slab still owns score/aux.  Crucially `score`
now always receives the TRUE gram diagonal WtW[j,j], so wjj can never go
non-positive (which is also what fixed the slab hang).

Run `python bae_search_prior.py` (real src on PYTHONPATH) for the validation:
  [A] no-coupling slab reproduces bae_search.snmf_search,
  [B] the new additive Boltzmann log-odds matches a brute-force marginalization,
  [C] the old `WtW + J` folding does NOT.
"""

import math
from functools import lru_cache

import numpy as np
from numba import njit

import bae_util


# ---------------------------------------------------------------------------
#  LIKELIHOOD link plugins  (unchanged from bae_search.py)
# ---------------------------------------------------------------------------

@njit(inline='always')
def score_binary(E, wjj, tau, sigma2):
    return (E - 0.5 * wjj) / sigma2, 0.0, 0.0


@njit(inline='always')
def aux_binary(Z, i, j, s, mu, nu):
    Z[i, j] = s


@njit(inline='always')
def score_slab(E, wjj, tau, sigma2):
    # wjj is now ALWAYS the true gram diagonal ||W[:,j]||^2 (the prior no longer
    # leaks into it), so the only degeneracy left is a genuinely dead column.
    if not (math.isfinite(wjj) and wjj > 1e-12) \
       or not (math.isfinite(sigma2) and sigma2 > 0.0) or not math.isfinite(E):
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
        Z[i, j] = 0.0


# ---------------------------------------------------------------------------
#  PRIOR plugins  (the new seam: owned by the LatentPrior component)
# ---------------------------------------------------------------------------
#
# Each returns the FULL additive prior contribution to the log-odds for S_ij,
# doing its own k-loop.  Same fixed signature so the factory can inject either;
# the coupling args (Jc, hc) are ignored by the unstructured prior (pass zeros).
#
# `inhib` is copied verbatim from the existing scaffold (the A/B/C/D StS
# combinatorics, summed over ALL k including k==j, exactly as old_bae_search.snmf),
# so the unstructured prior reproduces the old behaviour bit-for-bit.

@njit(inline='always')
def _tree_inhib(S, i, j, StS, N, beta):
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
def prior_unstructured(S, i, j, StS, N, Jc, hc, alpha, beta):
    """LatentPrior: sparsity + tree/StS reg.  No coupling."""
    return -alpha - beta * _tree_inhib(S, i, j, StS, N, beta)


@njit(inline='always')
def prior_boltzmann(S, i, j, StS, N, Jc, hc, alpha, beta):
    """BoltzmannPrior: sparsity + tree + the Ising conditional, ADDITIVE and
    linear in the binary spikes S_ik (Jc = symmetric {0,1}-coupling with zero
    diagonal, hc = {0,1}-field).  This is the corrected slab coupling: it reads
    S_ik (not Z_ik) and never touches wjj."""
    lp = -alpha - beta * _tree_inhib(S, i, j, StS, N, beta)
    m = S.shape[1]
    coup = hc[j]
    for k in range(m):
        if k != j:
            coup += Jc[j, k] * S[i, k]
    return lp + coup


# ---------------------------------------------------------------------------
#  The scaffold factory  (now closes over THREE plugins)
# ---------------------------------------------------------------------------

def _build_dense_search(score, aux_update, prior):

    @njit
    def search(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True):
        n, m = S.shape
        regularize = beta > 1e-6

        for i in np.random.permutation(np.arange(n)):
            for j in np.random.permutation(np.arange(m)):

                Sij = S[i, j]

                # LIKELIHOOD field: off Z, true gram only (no prior leakage)
                E = XW[i, j]
                for k in range(m):
                    if k != j:
                        E -= WtW[j, k] * Z[i, k]

                logodds, mu, nu = score(E, WtW[j, j], tau, sigma2)

                # PRIOR on S: additive, owned by the latent prior
                lp = prior(S, i, j, StS, N, Jc, hc, alpha, beta)

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

    return search


make_dense_search = lru_cache(maxsize=None)(_build_dense_search)

# Composable bundles.  LINK = (score, aux) [slab/operator side];
# PRIOR = the additive S-prior [latent-prior side].
BINARY_LINK = (score_binary, aux_binary)
SLAB_LINK = (score_slab, aux_slab)
PRIOR_PLAIN = prior_unstructured
PRIOR_BOLTZMANN = prior_boltzmann

# A model composes one LINK with one PRIOR:
snmf_plain_search = make_dense_search(*SLAB_LINK, PRIOR_PLAIN)
snmf_boltz_search = make_dense_search(*SLAB_LINK, PRIOR_BOLTZMANN)
sbmf_plain_search = make_dense_search(*BINARY_LINK, PRIOR_PLAIN)
sbmf_boltz_search = make_dense_search(*BINARY_LINK, PRIOR_BOLTZMANN)


# ===========================================================================
#  validation
# ===========================================================================

@njit
def _seed(s):
    np.random.seed(s)


def _validate():
    import bae_search as old          # the current 2-component prototype
    from scipy.integrate import quad

    rng = np.random.RandomState(0)
    n, d, m = 120, 40, 6
    W = rng.randn(d, m)
    b = rng.randn(d)
    X = rng.randn(n, d)
    XW = (X - b) @ W
    WtW = W.T @ W
    S0 = 1.0 * (rng.rand(n, m) > 0.5)
    Z0 = S0 * np.abs(rng.randn(n, m))      # magnitudes, zero off-spike
    temp, alpha, beta, tau, sigma2 = 0.7, 0.3, 1e-2, 0.5, 1.0
    SEED = 2024
    zeroJ = np.zeros((m, m))
    zeroh = np.zeros(m)

    # ---- [A] no-coupling slab == existing snmf_search --------------------
    Sa, Za, StSa = S0.copy(), Z0.copy(), (S0.T @ S0).copy()
    _seed(SEED)
    old.snmf_search(XW.copy(), Sa, Za, WtW.copy(), StSa, n,
                    temp, alpha, beta, tau, sigma2, True)

    Sb, Zb, StSb = S0.copy(), Z0.copy(), (S0.T @ S0).copy()
    _seed(SEED)
    snmf_plain_search(XW.copy(), Sb, Zb, WtW.copy(), StSb, n,
                      temp, alpha, beta, tau, sigma2, zeroJ, zeroh, True)
    print(f"[A] no-coupling slab vs snmf_search: "
          f"S exact={np.array_equal(Sa, Sb)}  "
          f"Z exact={np.array_equal(Za, Zb)}  "
          f"(max|dZ|={np.max(np.abs(Za - Zb)):.2e})")

    # ---- a symmetric {0,1} Ising coupling (zero diagonal) + field --------
    A = rng.randn(m, m) * 0.5
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0.0)
    c = rng.randn(m) * 0.5

    # ---- [B] new additive log-odds  vs  brute-force marginalization ------
    # pick a handful of elements and compare three numbers each.
    def field_and_wjj(i, j, WW, Zin):
        E = XW[i, j] - sum(WW[j, k] * Zin[i, k] for k in range(m) if k != j)
        return E, WW[j, j]

    def brute_loglik_ratio(E, wjj):
        # log integral_0^inf exp((zE - .5 z^2 wjj)/sigma2) tau e^{-tau z} dz
        integ = quad(lambda z: math.exp((z * E - 0.5 * z * z * wjj) / sigma2
                                        - tau * z),
                     0, np.inf)[0]
        return math.log(tau * integ)

    def ising_cond(i, j):
        return c[j] + sum(A[j, k] * S0[i, k] for k in range(m) if k != j)

    # The old folding's J_bin that reproduces (A, c) in the BINARY case:
    #   A_jk = -J_bin[j,k],  c_j = -0.5 J_bin[j,j]
    Jbin = -2.0 * A.copy()
    np.fill_diagonal(Jbin, -2.0 * c)
    WtW_fold = WtW + Jbin

    print("[B/C]  element |  brute (correct) |  new (score+prior) |  old (WtW+J)")
    for (i, j) in [(3, 1), (10, 4), (50, 2), (77, 5)]:
        E, wjj = field_and_wjj(i, j, WtW, Z0)
        brute = brute_loglik_ratio(E, wjj) + ising_cond(i, j)

        new = score_slab(E, wjj, tau, sigma2)[0] \
            + prior_boltzmann(S0, i, j, S0.T @ S0, n, A, c, alpha=0.0, beta=0.0)

        Ef, wjjf = field_and_wjj(i, j, WtW_fold, Z0)
        old_fold = score_slab(Ef, wjjf, tau, sigma2)[0]   # prior folded in, no separate term

        print(f"        ({i:2d},{j}) | {brute:14.6f}  | {new:14.6f}    "
              f"| {old_fold:12.6f}   "
              f"(new-brute={new - brute:+.2e}, old-brute={old_fold - brute:+.2e})")


if __name__ == "__main__":
    _validate()
