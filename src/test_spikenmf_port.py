"""Correctness check: new_bae_models.SpikeNMF vs bae_models.SpikeNMF.

Same idea as test_semibmf_port.py, with two extra RNG sources to pin down:
  * snmf (the slab E-step) draws from Numba's RNG  -> numba_seed
  * the dead-weight resampler draws from torch's RNG -> torch.manual_seed
We sync W (the nn.Linear), the spike S, the slab Z and StS, then run identical
anneals and assert latents / weights / reconstructions / energies match.
"""

import numpy as np
import torch
from numba import njit

import bae_models
import new_bae_models


@njit
def numba_seed(s):
    np.random.seed(s)


def make_data(n=60, d=40, K=5, seed=0):
    rng = np.random.default_rng(seed)
    Strue = (rng.random((n, K)) < 0.4).astype(float)
    Wtrue = np.abs(rng.standard_normal((d, K)))
    return Strue @ Wtrue.T + 0.3 * rng.standard_normal((n, d))


def sync(orig, port):
    """Copy orig's initialized state into port so both start identical."""
    port.operator.W.weight.data.copy_(orig.W.weight.data)
    if orig.W.bias is not None:
        port.operator.W.bias.data.copy_(orig.W.bias.data)
    port.latent_prior.S = orig.S.copy()
    port.latent_prior.Z = orig.Z.copy()  # the latent_prior (slab=True) owns S, Z, StS
    port.latent_prior.StS = orig.StS.copy()
    port.sigma_x = orig.sigma_x


def recon(model, S):
    return model(S)                       # __call__ -> torch decoder on S


def run_case(name, X, max_iter=200, seed=0, **kw):
    orig = bae_models.SpikeNMF(5, **kw)
    port = new_bae_models.SpikeNMF(5, **kw)

    orig.initialize(X)
    port.initialize(X)
    sync(orig, port)

    numba_seed(seed); torch.manual_seed(seed)
    en_o = orig.fit(X, max_iter=max_iter, verbose=False)
    numba_seed(seed); torch.manual_seed(seed)
    en_p = port.fit(X, max_iter=max_iter, verbose=False)

    Wo = orig.W.weight.detach().numpy()
    Wp = port.operator.W.weight.detach().numpy()
    bo = orig.W.bias.detach().numpy()
    bp = port.operator.W.bias.detach().numpy()

    S_eq = np.array_equal(orig.S, port.S)
    Z_eq = np.array_equal(orig.Z, port.latent_prior.Z)
    W_eq = np.array_equal(Wo, Wp)
    b_eq = np.array_equal(bo, bp)
    rec_eq = np.array_equal(recon(orig, orig.S), recon(port, port.S))
    en_o = np.asarray(en_o); en_p = np.asarray(en_p)
    en_eq = np.array_equal(en_o, en_p)

    ok = S_eq and Z_eq and W_eq and b_eq and rec_eq and en_eq
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    spike S identical : {S_eq}")
    print(f"    slab  Z identical : {Z_eq}   (max|dZ|={np.abs(orig.Z - port.latent_prior.Z).max():.2e})")
    print(f"    W identical       : {W_eq}   (max|dW|={np.abs(Wo - Wp).max():.2e})")
    print(f"    b identical       : {b_eq}")
    print(f"    reconstruction    : {rec_eq}")
    print(f"    energy trace       : {en_eq}   (max|d|={np.abs(en_o - en_p).max():.2e})")
    print(f"    final sigma_x : orig={orig.sigma_x:.8f}  port={port.sigma_x:.8f}")
    return ok


if __name__ == "__main__":
    X = make_data()
    results = []
    # default: nonneg=True (exercises clamp + dead-weight torch RNG path)
    results.append(run_case("default (nonneg)", X,
                            nonneg=True, sparse_reg=0.1, tree_reg=0.1,
                            slab_prior=1.0, weight_pr_reg=0.1,
                            weight_l1_reg=0.0, weight_l2_reg=1e-2))
    # nonneg=False: no clamp / no torch RNG, isolates the autograd + slab path
    results.append(run_case("free (no nonneg)", X,
                            nonneg=False, sparse_reg=0.2, tree_reg=0.05,
                            slab_prior=2.0, weight_pr_reg=0.5,
                            weight_l1_reg=0.05, weight_l2_reg=1e-2))
    # tree_reg = 0: turns off the StS regularizer branch inside snmf
    results.append(run_case("no-tree", X,
                            nonneg=True, sparse_reg=0.5, tree_reg=0.0,
                            slab_prior=1.0, weight_pr_reg=0.1,
                            weight_l1_reg=0.0, weight_l2_reg=1e-2))

    print("\nALL PASS" if all(results) else "\nSOME FAILED")
