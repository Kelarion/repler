"""Imputation E-step check for new_bae_models.

With a boolean `mask` (True = held-out / missing), the E-step refills the masked
entries of the working array with a draw from the generative model given the
current latents, in place, and hands the completed X to the M-step -- the
bae_util.impcv `Z[M] = model(ES)[M]` loop folded into the E-step.  Here we hold
out a fraction of a low-rank matrix, fit with the mask, and check the held-out
cells are recovered (MSE down, correlation with truth up).

Note: the fill is the conditional MEAN forward(ES), with no observation noise
added, so the imputation is an EM-style completion rather than a posterior draw.
"""

import numpy as np
from numba import njit

import new_bae_models as nbm


@njit
def numba_seed(s):
    np.random.seed(s)


def make_low_rank(n=120, d=40, K=5, noise=0.1, nonneg=False, seed=0):
    rng = np.random.default_rng(seed)
    S = (rng.random((n, K)) < 0.4).astype(float)
    W = np.abs(rng.standard_normal((d, K))) if nonneg else rng.standard_normal((d, K))
    return S @ W.T + noise * rng.standard_normal((n, d))


def run_case(name, model_factory, K=5, hold=0.15, scl_lr=0.1, max_iter=300, seed=0,
             nonneg=False):
    rng = np.random.default_rng(seed)
    X0 = make_low_rank(K=K, nonneg=nonneg, seed=seed)
    mask = rng.random(X0.shape) < hold

    Z = X0.copy()
    Z[mask] = rng.choice(Z[~mask], mask.sum())          # empirical-distribution init fill
    init_mse = np.mean((Z[mask] - X0[mask]) ** 2)

    numba_seed(seed); np.random.seed(seed)
    model = model_factory(K)
    model.fit(Z, mask=mask, max_iter=max_iter, verbose=False, scl_lr=scl_lr)

    # Z was filled IN PLACE during fitting -> held-out cells now hold the imputation
    imp_mse = np.mean((Z[mask] - X0[mask]) ** 2)
    corr = np.corrcoef(Z[mask], X0[mask])[0, 1]

    ok = (imp_mse < 0.5 * init_mse) and (corr > 0.8)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    held-out cells           : {mask.sum()} ({100 * mask.mean():.0f}%)")
    print(f"    masked MSE  init -> imputed : {init_mse:.3f} -> {imp_mse:.3f}")
    print(f"    corr(imputed, truth)       : {corr:.3f}")
    print(f"    learned sigma_x            : {model.sigma_x:.4f}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(run_case(
        "SemiBMF (affine)",
        lambda K: nbm.SemiBMF(K, tree_reg=0.0, sparse_reg=0.0, weight_l2_reg=1e-3)))
    results.append(run_case(
        "SemiBMF (spike-and-slab, nonneg)",
        lambda K: nbm.SemiBMF(K, slab=True, nonneg=True, tree_reg=0.0,
                              sparse_reg=0.0, weight_l2_reg=1e-3),
        nonneg=True))
    print("\nALL PASS" if all(results) else "\nSOME FAILED")
