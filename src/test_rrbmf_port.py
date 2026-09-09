"""Correctness check: bae_models.RRBMF vs old_bae_models.RRBMF.

Same idea as test_spikenmf_port.py.  The reduced-rank operator works on a 3-D
data tensor X (n, t, d); the E-step is the dense sbmf (Numba RNG -> numba_seed),
the M-step is autograd SGD on (U, V, b) (deterministic).  We sync U/V/b, the
spike S (and Z == S, since there is no slab), and StS, run identical anneals and
assert latents / weights / reconstructions / energies match bit-for-bit.

NOTE: assumes old_bae_models.RRBMF.EStep defaults to inplace=True (StS tracked live),
like every other model.  The refactor standardizes on inplace=True; the older
inplace=False froze StS for the whole fit, in which case only the tree_reg==0
cases would match.
"""

import numpy as np
import torch
from numba import njit

import old_bae_models
import bae_models


@njit
def numba_seed(s):
    np.random.seed(s)


def make_data(n=50, t=8, d=12, K=4, rank=2, seed=0):
    rng = np.random.default_rng(seed)
    S = (rng.random((n, K)) < 0.4).astype(float)
    U = rng.standard_normal((t, rank))
    V = rng.standard_normal((K, d, rank))
    beta = V @ U.T                                   # (K, d, t)
    X = np.einsum('ck,knt->ctn', S, beta) + 0.3 * rng.standard_normal((n, t, d))
    return torch.tensor(X)                            # (n, t, d) double


def sync(orig, port):
    """Copy orig's initialized state into port so both start identical."""
    port.operator.U.data.copy_(orig.U.data)
    port.operator.V.data.copy_(orig.V.data)
    port.operator.b.data.copy_(orig.b.data)
    port.latent_prior.S = orig.S.copy()
    port.latent_prior.StS = orig.StS.copy()
    port.latent_prior.Z = orig.S.copy()              # no slab: effective latent == spike
    port.sigma_x = orig.sigma_x


def run_case(name, X, dim_hid=4, rank=2, max_iter=150, seed=0, lr=1e-2,
             nonneg=False, sparse_reg=0.0, tree_reg=1e-2,
             pr_reg=1e-2, l1_reg=0.0, l2_reg=1e-2):
    orig = old_bae_models.RRBMF(dim_hid, rank, nonneg=nonneg,
                            sparse_reg=sparse_reg, tree_reg=tree_reg,
                            pr_reg=pr_reg, l1_reg=l1_reg, l2_reg=l2_reg)
    port = bae_models.RRBMF(dim_hid, rank=rank, nonneg=nonneg,
                                sparse_reg=sparse_reg, tree_reg=tree_reg,
                                weight_pr_reg=pr_reg, weight_l1_reg=l1_reg,
                                weight_l2_reg=l2_reg)

    # the optimizer is built at initialize() time; pass the same lr to both
    # (the original otherwise falls back to optim.SGD's default lr).
    orig.initialize(X, lr=lr)
    port.initialize(X, lr=lr)
    sync(orig, port)

    numba_seed(seed)
    en_o = orig.fit(X, max_iter=max_iter, verbose=False)
    numba_seed(seed)
    en_p = port.fit(X, max_iter=max_iter, verbose=False)

    Uo, Up = orig.U.detach().numpy(), port.operator.U.detach().numpy()
    Vo, Vp = orig.V.detach().numpy(), port.operator.V.detach().numpy()
    bo, bp = orig.b.detach().numpy(), port.operator.b.detach().numpy()

    S_eq = np.array_equal(orig.S, port.S)
    U_eq = np.array_equal(Uo, Up)
    V_eq = np.array_equal(Vo, Vp)
    b_eq = np.array_equal(bo, bp)
    rec_eq = np.array_equal(orig(orig.S).detach().numpy(), port(port.S))
    en_o, en_p = np.asarray(en_o), np.asarray(en_p)
    en_close = np.allclose(en_o, en_p, rtol=0, atol=1e-9)

    ok = S_eq and U_eq and V_eq and b_eq and rec_eq and en_close
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    spike S identical : {S_eq}")
    print(f"    U identical       : {U_eq}   (max|dU|={np.abs(Uo - Up).max():.2e})")
    print(f"    V identical       : {V_eq}   (max|dV|={np.abs(Vo - Vp).max():.2e})")
    print(f"    b identical       : {b_eq}   (max|db|={np.abs(bo - bp).max():.2e})")
    print(f"    reconstruction    : {rec_eq}")
    print(f"    energy trace close: {en_close}   (max|d|={np.abs(en_o - en_p).max():.2e})")
    return ok


if __name__ == "__main__":
    X = make_data()
    results = []
    results.append(run_case("free (no nonneg)", X,
                            nonneg=False, sparse_reg=0.2, tree_reg=0.05,
                            pr_reg=0.5, l1_reg=0.05, l2_reg=1e-2))
    results.append(run_case("default (nonneg)", X,
                            nonneg=True, sparse_reg=0.1, tree_reg=0.1,
                            pr_reg=0.1, l1_reg=0.0, l2_reg=1e-2))
    results.append(run_case("no-tree (nonneg)", X,
                            nonneg=True, sparse_reg=0.5, tree_reg=0.0,
                            pr_reg=0.1, l1_reg=0.0, l2_reg=1e-2))
    results.append(run_case("strong-tree (nonneg)", X,
                            nonneg=True, sparse_reg=0.3, tree_reg=0.3,
                            pr_reg=0.2, l1_reg=0.01, l2_reg=1e-2))
    print("\nALL PASS" if all(results) else "\nSOME FAILED")
