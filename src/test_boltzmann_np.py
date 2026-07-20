"""Correctness check: new_bae_priors.BoltzmannPriorNP (pure numpy) vs torch.

BoltzmannPriorNP replaces the torch optimizer step with a hand-derived gradient.
Two checks:

  1. GRADIENT (float64): the analytic (dL/dJ_W, dL/dJ_h) from `_grads` matches
     torch autograd of the identical pseudolikelihood loss to ~1e-12.  This is
     the real proof the derivation is right.

  2. TRAJECTORY (float64): from J = 0, feeding an identical ES sequence to
     BoltzmannPriorNP and to a float64 autograd-SGD reference, the J parameters
     stay identical -- the manual-gradient SGD loop is correct.

A third, informational, number compares against the *actual* torch BoltzmannPrior
(float32).  It tracks for rple but drifts for logrise: logrise's larger J_l1
makes the non-smooth L1 sign() term flip for near-zero J entries between float32
and float64, each flip costing ~J_l1*lr.  That is float precision on a kink, not
a gradient error (which check 1 rules out at ~1e-16).

Both J_loss = 'rple' and 'logrise'.
"""

import numpy as np
import torch
import torch.nn.functional as F

import new_bae_priors


def make_data(n=60, d=40, K=5, seed=0):
    rng = np.random.default_rng(seed)
    Strue = (rng.random((n, K)) < 0.4).astype(float)
    Wtrue = rng.standard_normal((d, K))
    return Strue @ Wtrue.T + 0.3 * rng.standard_normal((n, d))


def torch_grads(J_W, J_h, ES, J_loss, J_l1):
    """Autograd reference: grad of the same loss, diagonal zeroed (ZeroDiag)."""
    W = torch.tensor(J_W, dtype=torch.float64, requires_grad=True)
    h = torch.tensor(J_h, dtype=torch.float64, requires_grad=True)
    S = torch.tensor(2.0 * ES - 1.0, dtype=torch.float64)
    pred = S @ W.T + h
    if J_loss == 'logrise':
        loss = torch.sum(torch.logsumexp(-pred * S, 1)) / len(ES)
    else:
        loss = torch.sum(F.softplus(-2 * pred * S)) / len(ES)
    loss = loss + J_l1 * torch.sum(torch.abs(W))
    loss.backward()
    gW = W.grad.numpy().copy()
    np.fill_diagonal(gW, 0.0)
    return gW, h.grad.numpy().copy()


def grad_check(J_loss, dim_hid=5, n=60, seed=1):
    rng = np.random.default_rng(seed)
    J_W = rng.standard_normal((dim_hid, dim_hid))
    np.fill_diagonal(J_W, 0.0)                          # zero-diag like ZeroDiag
    J_h = rng.standard_normal(dim_hid)
    ES = (rng.random((n, dim_hid)) < 0.4).astype(float)
    J_l1 = 0.07

    p = new_bae_priors.BoltzmannPriorNP(J_loss=J_loss)
    p.J_W, p.J_h, p.J_l1_reg = J_W.copy(), J_h.copy(), J_l1
    gW_np, gh_np = p._grads(ES)
    gW_t, gh_t = torch_grads(J_W, J_h, ES, J_loss, J_l1)

    dW = np.abs(gW_np - gW_t).max()
    dh = np.abs(gh_np - gh_t).max()
    return dW, dh


def traj_check(J_loss, X, dim_hid=5, n_steps=8, seed=2):
    npp = new_bae_priors.BoltzmannPriorNP(J_loss=J_loss)
    npp.init_params(X, dim_hid)
    J_l1, lr = npp.J_l1_reg, npp.J_lr

    rng = np.random.default_rng(seed)
    ES = (rng.random((len(X), dim_hid)) < 0.4).astype(float)

    # float64 autograd-SGD reference (same algorithm, gradients via torch)
    W = np.zeros((dim_hid, dim_hid)); h = np.zeros(dim_hid)
    # float32 actual torch prior (the thing being replaced) -- informational
    tor = new_bae_priors.BoltzmannPrior(J_loss=J_loss); tor.init_params(X, dim_hid)

    f64_dev = 0.0
    for _ in range(n_steps):
        gW, gh = torch_grads(W, h, ES, J_loss, J_l1)
        W = W - lr * gW; h = h - lr * gh
        npp.learn(ES)
        tor.learn(ES)
        f64_dev = max(f64_dev, np.abs(W - npp.J_W).max(),
                      np.abs(h - npp.J_h).max())

    f32_dev = max(np.abs(tor.J.weight.detach().numpy() - npp.J_W).max(),
                  np.abs(tor.J.bias.detach().numpy() - npp.J_h).max())
    return f64_dev, f32_dev


if __name__ == "__main__":
    X = make_data()
    ok = True
    for J_loss in ('rple', 'logrise'):
        dW, dh = grad_check(J_loss)
        f64_dev, f32_dev = traj_check(J_loss, X)
        passed = dW < 1e-9 and dh < 1e-9 and f64_dev < 1e-9
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {J_loss}")
        print(f"    grad vs autograd (float64)    : max|dJ_W|={dW:.2e}  max|dJ_h|={dh:.2e}")
        print(f"    trajectory vs float64 ref      : max|dJ| over 8 steps = {f64_dev:.2e}")
        print(f"    (info) vs float32 torch prior  : max|dJ| = {f32_dev:.2e}")
    print("\nALL PASS" if ok else "\nSOME FAILED")
