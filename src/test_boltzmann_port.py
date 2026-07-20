"""Correctness check: new_bae_priors.BoltzmannPrior vs bae_models.JBMF.

The structured prior is the *J* side of JBMF: the `couple` recurrence (WtW + J)
fed to the search, and the inverse-Ising pseudolikelihood step on J.  Because the
loss is separable in (W, J), JBMF's J update depends only on (J, ES) -- not on W
-- so we can verify the ported prior in isolation:

  * feed an identical ES to JBMF.MStep and BoltzmannPrior.learn and check the J
    parameters evolve bit-for-bit (the pseudolikelihood port);
  * with the same J, check couple(WtW) == JBMF's WtW + J construction.

The J update is deterministic (no RNG in the autograd path), so no seeding of the
J step is needed; ES is fixed.  Tested for both J_loss = 'rple' and 'logrise'.
"""

import numpy as np
import torch

import bae_models
import new_bae_priors


def make_data(n=60, d=40, K=5, seed=0):
    rng = np.random.default_rng(seed)
    Strue = (rng.random((n, K)) < 0.4).astype(float)
    Wtrue = rng.standard_normal((d, K))
    return Strue @ Wtrue.T + 0.3 * rng.standard_normal((n, d))


def jbmf_J_matrix(jbmf):
    """Replicate JBMF.EStep's J construction (bae_models.py:1004-1007)."""
    J = jbmf.J.weight.detach().numpy()
    h = jbmf.J.bias.detach().numpy()
    J = (J + J.T) / 2
    J = 4 * J + 2 * np.diag(h - 2 * J.sum(1))
    return J


def run_case(name, J_loss, X, dim_hid=5, n_steps=8, seed=0):
    Xt = torch.FloatTensor(X)            # JBMF works in torch (W, b are Parameters)
    orig = bae_models.JBMF(dim_hid, J_loss=J_loss)
    orig.initialize(Xt)

    port = new_bae_priors.BoltzmannPrior(J_loss=J_loss)
    port.init_params(X, dim_hid)         # the prior only needs X.shape[0]

    # both auto-derive J_l1_reg from (dim_hid, n) the same way
    l1_eq = np.isclose(orig.J_l1_reg, port.J_l1_reg, atol=0, rtol=0)

    # both initialize J (weight + bias) to zero -> identical start
    Wj0 = np.array_equal(orig.J.weight.detach().numpy(),
                         port.J.weight.detach().numpy())
    bj0 = np.array_equal(orig.J.bias.detach().numpy(),
                         port.J.bias.detach().numpy())

    # --- the pseudolikelihood: identical ES -> identical J trajectory --------
    rng = np.random.default_rng(seed + 1)
    ES = (rng.random((len(X), dim_hid)) < 0.4).astype(float)

    traj_ok = True
    max_dev = 0.0
    for _ in range(n_steps):
        orig.MStep(ES, Xt)       # updates W and J (J part is what we compare)
        port.learn(ES)           # updates J only
        dW = np.abs(orig.J.weight.detach().numpy()
                    - port.J.weight.detach().numpy()).max()
        db = np.abs(orig.J.bias.detach().numpy()
                    - port.J.bias.detach().numpy()).max()
        max_dev = max(max_dev, dW, db)
        traj_ok &= np.array_equal(orig.J.weight.detach().numpy(),
                                  port.J.weight.detach().numpy())
        traj_ok &= np.array_equal(orig.J.bias.detach().numpy(),
                                  port.J.bias.detach().numpy())

    # --- the coupling: the additive (Jc, hc) reconstruct JBMF's folded J ------
    # The prior is no longer folded into WtW; coupling() returns the additive
    # (Jc, hc) with Jc = -offdiag(Jbin), hc = -0.5*diag(Jbin).  Rebuilding Jbin
    # from (Jc, hc) and comparing to JBMF's J matrix verifies the equivalence.
    Jc, hc = port.coupling()
    Jbin_rebuilt = -Jc.copy()
    np.fill_diagonal(Jbin_rebuilt, -2.0 * hc)
    couple_eq = np.allclose(Jbin_rebuilt, jbmf_J_matrix(orig))

    ok = l1_eq and Wj0 and bj0 and traj_ok and couple_eq
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    J_l1_reg identical : {l1_eq}   (orig={orig.J_l1_reg:.6f} port={port.J_l1_reg:.6f})")
    print(f"    J init identical   : weight={Wj0} bias={bj0}")
    print(f"    J trajectory ({n_steps} steps) identical : {traj_ok}   (max|dJ|={max_dev:.2e})")
    print(f"    coupling rebuilds J: {couple_eq}")
    return ok


if __name__ == "__main__":
    X = make_data()
    results = []
    results.append(run_case("rple (softplus pseudolikelihood)", 'rple', X))
    results.append(run_case("logrise", 'logrise', X))
    print("\nALL PASS" if all(results) else "\nSOME FAILED")
