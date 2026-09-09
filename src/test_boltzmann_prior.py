"""Correctness checks for bae_priors.BoltzmannPrior (the {0,1} Ising prior).

The prior is  log p(S) ~ S' J S + 2 h' S  over S in {0,1}^m, with J symmetric and
zero-diagonal, so the conditional log-odds of S_j is 2*(J S + h)_j and the
additive (Jc, hc) that `coupling()` hands the search is exactly (2J, 2h).  This is
the convention of minimal_structured_bipca.StructuredBiPCA.

Checks:
  1. COUPLING     coupling() reproduces the model's own conditional log-odds, and
                  the compiled search applies it with the right SIGN end to end.
  2. GRADIENT     the analytic pseudolikelihood gradients ('rple', 'logrise')
                  match torch autograd of the identical loss to ~1e-12.
  3. SAMPLER      for a small enumerable model the Gibbs and GWG samplers'
                  moments match the exact enumerated ones.
  4. MLE          the moment-matching gradient vanishes at the truth, and
                  learning from J = 0 recovers a planted coupling.
  5. REFIT        the end-of-fit refit selects the true support and nothing else.
  6. CHAINS       n_chains > 1 reproduces the serial result chain by chain.
"""

import itertools
import numpy as np
import torch

import bae_priors as nbp


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def random_ising(m, seed=0, jscale=0.15, hscale=0.3):
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((m, m)) * jscale
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    return J, rng.standard_normal(m) * hscale


def exact_moments(J, h):
    """Exact <S>, <SS'> and the state distribution of log p(S) ~ S'JS + 2h'S."""
    m = len(h)
    states = np.array(list(itertools.product([0.0, 1.0], repeat=m)))    # (2^m, m)
    E = np.einsum('si,ij,sj->s', states, J, states) + 2 * states @ h
    w = np.exp(E - E.max())
    w /= w.sum()
    return w @ states, states.T @ (w[:, None] * states), states, w


def prior_at(J, h, **kw):
    """A BoltzmannPrior holding (J, h) directly, with its MLE particles seeded."""
    p = nbp.BoltzmannPrior(J_l1_reg=0.0, **kw)
    p.J, p.h = J.copy(), h.copy()
    p.state = 1.0 * np.random.choice([0, 1], size=(p.mle_n_samp, len(h)))
    return p


def draw_exact(J, h, n, seed=0):
    """Exact i.i.d. draws from the enumerated model."""
    _, _, states, w = exact_moments(J, h)
    idx = np.random.default_rng(seed).choice(len(states), size=n, p=w)
    return states[idx]


# ---------------------------------------------------------------------------
#  1. coupling
# ---------------------------------------------------------------------------

def check_coupling(m=6, seed=0):
    J, h = random_ising(m, seed, jscale=1.0, hscale=1.0)
    p = prior_at(J, h)
    Jc, hc = p.coupling()
    S = 1.0 * (np.random.default_rng(seed).random(m) > 0.5)
    want = 2 * (J @ S + h)                       # the model's conditional log-odds
    got = hc + (Jc - np.diag(np.diag(Jc))) @ S   # what prior_boltzmann accumulates
    return np.abs(want - got).max()


def check_search_sign(seed=0):
    """A strong +/- coupling must induce +/- correlation in the sampled codes."""
    import bae_models as nbm
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((300, 12))
    X /= X.std()
    out = []
    for sign in (+3.0, -3.0):
        np.random.seed(seed)
        mod = nbm.BiPCA(4, tree_reg=0, sparse_reg=0, J_prior='boltzmann', J_lr=1e-2)
        mod.initialize(X)
        mod.latent_prior.J[:] = 0.0
        mod.latent_prior.J[0, 1] = mod.latent_prior.J[1, 0] = sign
        mod.latent_prior.h[:] = 0.0
        mod.temp = 1
        S = mod.sample(X, n_samp=40, burnin=5)
        out.append(np.corrcoef(S[..., 0].ravel(), S[..., 1].ravel())[0, 1])
    return out


# ---------------------------------------------------------------------------
#  2. pseudolikelihood gradients vs autograd
# ---------------------------------------------------------------------------

def torch_pl_grads(J, h, S, J_loss, h_ridge):
    """Autograd reference for the SAME mean-per-element loss.  J is symmetric, so
    the gradient w.r.t. the symmetric parameterization is G + G^T."""
    Jt = torch.tensor(J, dtype=torch.float64, requires_grad=True)
    ht = torch.tensor(h, dtype=torch.float64, requires_grad=True)
    St = torch.tensor(S, dtype=torch.float64)
    n, m = S.shape
    pred = St @ Jt + ht
    if J_loss == 'logrise':
        loss = torch.logsumexp(-(2 * St - 1) * pred, dim=1).sum() / (n * m)
    else:
        logit = 2 * pred
        loss = (torch.nn.functional.softplus(logit) - St * logit).sum() / (n * m)
    loss = loss + 0.5 * h_ridge * (ht ** 2).sum()
    loss.backward()
    gJ = Jt.grad.numpy() + Jt.grad.numpy().T
    np.fill_diagonal(gJ, 0.0)
    return gJ, ht.grad.numpy()


def check_pl_grads(J_loss, m=5, n=60, seed=1):
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((m, m))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    h = rng.standard_normal(m)
    S = 1.0 * (rng.random((n, m)) < 0.4)

    p = nbp.BoltzmannPrior(J_loss=J_loss)
    gJ, gh = p._pl_grads(S, J, h)
    gJ_t, gh_t = torch_pl_grads(J, h, S, J_loss, p.h_l2_reg)
    return np.abs(gJ - gJ_t).max(), np.abs(gh - gh_t).max()


# ---------------------------------------------------------------------------
#  3-4. samplers and the MLE gradient
# ---------------------------------------------------------------------------

def check_sampler(sampler, m=6, seed=0, n_samp=20000):
    J, h = random_ising(m, seed)
    mean_x, second_x, _, _ = exact_moments(J, h)
    np.random.seed(0)
    p = prior_at(J, h, J_loss='mle', mle_n_samp=n_samp, sampler=sampler)
    p._advance(60)
    mean_s, second_s = p._stash_moments()
    return np.abs(mean_s - mean_x).max(), np.abs(second_s - second_x).max()


def check_grad_at_truth(sampler, m=6, seed=1, n_data=8000):
    J, h = random_ising(m, seed)
    S = draw_exact(J, h, n_data, seed=123)
    np.random.seed(1)
    p = prior_at(J, h, J_loss='mle', mle_n_samp=8000, sampler=sampler)
    p._advance(20)                                   # equilibrate the particles
    gJ, gh = p._grads(S)
    return np.abs(gJ).max(), np.abs(gh).max()


def check_mle_learning(sampler, m=6, seed=2, n_data=8000, steps=400):
    J, h = random_ising(m, seed)
    S = draw_exact(J, h, n_data, seed=7)
    np.random.seed(2)
    p = nbp.BoltzmannPrior(J_loss='mle', J_lr=0.05, J_l1_reg=0.0, mle_n_samp=2000,
                           mle_gibbs_steps=2, sampler=sampler)
    p.init_params(S)
    for _ in range(steps):
        p.learn(S)
    off = ~np.eye(m, dtype=bool)
    return float(np.sum(p.J[off] * J[off])
                 / (np.linalg.norm(p.J[off]) * np.linalg.norm(J[off]) + 1e-12))


# ---------------------------------------------------------------------------
#  5-6. refit and chains
# ---------------------------------------------------------------------------

def check_refit(m=6, n_data=8000, strength=1.5, seed=3):
    J = np.zeros((m, m))
    J[0, 1] = J[1, 0] = strength
    J[2, 3] = J[3, 2] = -strength
    S = draw_exact(J, np.zeros(m), n_data, seed=seed)

    p = nbp.BoltzmannPrior(J_lr=0.5, J_l1_reg=0.0)
    p.init_params(S)
    for _ in range(2000):
        p.learn(S)
    p.refit(S)
    true_sup = 1.0 * (J != 0)
    return float(np.abs(p.support - true_sup).max()), p.J[0, 1], p.J[2, 3]


def check_chains(m=5, n=80, C=3, seed=4):
    """A chain-batched prior must step each chain exactly like a serial one."""
    rng = np.random.default_rng(seed)
    S = 1.0 * (rng.random((C, n, m)) < 0.4)

    multi = nbp.BoltzmannPrior(n_chains=C, J_lr=0.3, J_l1_reg=1e-3)
    multi.init_params(S)
    serial = [nbp.BoltzmannPrior(J_lr=0.3, J_l1_reg=1e-3) for _ in range(C)]
    for c, s in enumerate(serial):
        s.init_params(S[c])

    dev = 0.0
    for _ in range(20):
        multi.learn(S)
        for c, s in enumerate(serial):
            s.learn(S[c])
            dev = max(dev, np.abs(multi.J[c] - s.J).max(),
                      np.abs(multi.h[c] - s.h).max())
    Jc_m, hc_m = multi.coupling()
    Jc_s, hc_s = serial[0].coupling()
    dev = max(dev, np.abs(Jc_m[0] - Jc_s).max(), np.abs(hc_m[0] - hc_s).max())
    return dev


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok = True

    d = check_coupling()
    p_ = d < 1e-12
    ok &= p_
    print(f"[{'PASS' if p_ else 'FAIL'}] coupling() == the model's conditional log-odds")
    print(f"    max|d(logodds)| = {d:.2e}")

    pos, neg = check_search_sign()
    p_ = pos > 0.2 and neg < -0.2
    ok &= p_
    print(f"[{'PASS' if p_ else 'FAIL'}] the search applies the coupling with the right sign")
    print(f"    corr(S0,S1): J>0 -> {pos:+.3f}   J<0 -> {neg:+.3f}")

    for J_loss in ('rple', 'logrise'):
        dJ, dh = check_pl_grads(J_loss)
        p_ = dJ < 1e-10 and dh < 1e-10
        ok &= p_
        print(f"[{'PASS' if p_ else 'FAIL'}] {J_loss} gradient vs torch autograd")
        print(f"    max|dJ|={dJ:.2e}  max|dh|={dh:.2e}")

    for sampler in ('gibbs', 'gwg'):
        dm, dC = check_sampler(sampler)
        p_ = dm < 0.03 and dC < 0.03
        ok &= p_
        print(f"[{'PASS' if p_ else 'FAIL'}] {sampler} sampler vs exact moments")
        print(f"    max|d<S>|={dm:.3f}   max|d<SS'>|={dC:.3f}")

        gJ, gh = check_grad_at_truth(sampler)
        p_ = gJ < 0.05 and gh < 0.05
        ok &= p_
        print(f"[{'PASS' if p_ else 'FAIL'}] {sampler} MLE gradient at truth ~ 0")
        print(f"    max|dL/dJ|={gJ:.3f}   max|dL/dh|={gh:.3f}")

        cos = check_mle_learning(sampler)
        p_ = cos > 0.8
        ok &= p_
        print(f"[{'PASS' if p_ else 'FAIL'}] {sampler} MLE learning recovers the coupling")
        print(f"    cosine(J_hat, J_true) = {cos:.3f}")

    dsup, j01, j23 = check_refit()
    p_ = dsup == 0.0 and j01 > 0 and j23 < 0
    ok &= p_
    print(f"[{'PASS' if p_ else 'FAIL'}] refit() selects exactly the true support")
    print(f"    support error = {dsup:.0f};  J[0,1]={j01:+.3f} (true +)  "
          f"J[2,3]={j23:+.3f} (true -)")

    dev = check_chains()
    p_ = dev < 1e-12
    ok &= p_
    print(f"[{'PASS' if p_ else 'FAIL'}] n_chains > 1 matches the serial prior")
    print(f"    max|dJ| over 20 steps = {dev:.2e}")

    print("\nALL PASS" if ok else "\nSOME FAILED")
