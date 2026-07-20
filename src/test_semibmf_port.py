"""Correctness check: new_bae_models.SemiBMF vs bae_models.SemiBMF.

Strategy: the discrete E-step (bae_search.sbmf) draws from Numba's *own* RNG,
which Python's np.random.seed does not touch -- so we seed it from inside a
jitted function.  We sync both models to a byte-identical starting state, run
identical anneals, and assert the latents / weights / energies match exactly.
"""

import numpy as np
from numba import njit

import bae_models
import new_bae_models


@njit
def numba_seed(s):
    np.random.seed(s)


def make_data(n=60, d=40, K=5, seed=0):
    rng = np.random.default_rng(seed)
    Strue = (rng.random((n, K)) < 0.4).astype(float)
    Wtrue = rng.standard_normal((d, K))
    return Strue @ Wtrue.T + 0.3 * rng.standard_normal((n, d))


def sync(orig, port):
    """Copy orig's initialized state into port so both start identical.
    (The port keeps W and b on the operator, and the latents on latent_prior.)"""
    port.operator.W = orig.W.copy()
    port.operator.b = orig.b.copy()
    port.latent_prior.S = orig.S.copy()
    port.latent_prior.StS = orig.StS.copy()
    port.sigma_x = orig.sigma_x


def run_case(name, X, max_iter=300, seed=0, **kw):
    orig = bae_models.SemiBMF(5, **kw)
    port = new_bae_models.SemiBMF(5, **kw)

    orig.initialize(X)
    port.initialize(X)
    sync(orig, port)

    numba_seed(seed)
    en_o = orig.fit(X, max_iter=max_iter, verbose=False)
    numba_seed(seed)
    en_p = port.fit(X, max_iter=max_iter, verbose=False)

    S_eq = np.array_equal(orig.S, port.S)
    W_eq = np.array_equal(orig.W, port.operator.W)
    b_eq = np.array_equal(orig.b, port.operator.b)
    err_o = np.array([e[0] for e in en_o])    # original MStep returns [err, ES, energy]
    err_p = np.array(en_p)                     # port MStep returns the scalar err
    en_eq = np.array_equal(err_o, err_p)
    loss_eq = np.isclose(orig.loss(X), port.loss(X), atol=0, rtol=0)

    ok = S_eq and W_eq and b_eq and en_eq and loss_eq
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    S identical : {S_eq}")
    print(f"    W identical : {W_eq}   (max|dW|={np.abs(orig.W - port.operator.W).max():.2e})")
    print(f"    b identical : {b_eq}")
    print(f"    energy trace identical : {en_eq}   "
          f"(max|d|={np.abs(err_o - err_p).max():.2e})")
    print(f"    final loss : orig={orig.loss(X):.8f}  port={port.loss(X):.8f}")
    return ok


if __name__ == "__main__":
    X = make_data()
    results = []
    # default path (participation-ratio M-step branch)
    results.append(run_case("default", X,
                            nonneg=False, sparse_reg=0.5, tree_reg=0.1,
                            weight_pr_reg=0.5, weight_l2_reg=1e-2, weight_l1_reg=0.0))
    # nonneg + l1 (exercises the projection + l1 grad term)
    results.append(run_case("nonneg+l1", X,
                            nonneg=True, sparse_reg=0.2, tree_reg=0.05,
                            weight_pr_reg=1.0, weight_l2_reg=1e-2, weight_l1_reg=0.05))
    # tree_reg = 0 (turns off the StS regularizer path inside sbmf)
    results.append(run_case("no-tree", X,
                            nonneg=False, sparse_reg=1.0, tree_reg=0.0,
                            weight_pr_reg=0.5, weight_l2_reg=1e-2, weight_l1_reg=0.0))

    print("\nALL PASS" if all(results) else "\nSOME FAILED")
