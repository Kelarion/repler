"""Correctness check: BoltzmannPriorNP J_loss='mle' (sampled moment matching).

The MLE gradient is stochastic, so instead of bit-for-bit we check the two
things that must hold:

  1. SAMPLER: for a small Ising prior (m=6, enumerable), the Gibbs sampler's
     moments <s> and <ss'> match the exact enumerated moments (to sampling
     error).  This validates `sample` / `_sweep` and the parametrization.

  2. GRADIENT AT TRUTH: with the prior set to the true (J_sym, h) and data drawn
     from that same model, the MLE gradient is ~0 -- data and model moments
     agree.  This validates the gradient assembly (moment matching) and signs.

  3. LEARNING: from J=0, a few hundred MLE steps on data from a known model drive
     J_sym toward the truth (positive cosine similarity) and shrink the moment
     mismatch.  A coarse end-to-end sanity check.
"""

import itertools
import numpy as np

import new_bae_priors


def random_ising(m, seed=0, jscale=0.15, hscale=0.3):
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((m, m)) * jscale
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    h = rng.standard_normal(m) * hscale
    return J, h


def exact_moments(J, h):
    """Exact <s> and <ss'> under P(s) ~ exp(0.5 s'Js + h's), s in {-1,+1}^m."""
    m = len(h)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=m)))   # (2^m, m)
    E = 0.5 * np.einsum('si,ij,sj->s', states, J, states) + states @ h
    w = np.exp(E - E.max()); w /= w.sum()
    mean = w @ states
    second = states.T @ (w[:, None] * states)
    return mean, second, states, w


def prior_at(J, h, **kw):
    """A BoltzmannPriorNP holding (J, h) directly (bypassing init's zeros)."""
    p = new_bae_priors.BoltzmannPriorNP(J_loss='mle', **kw)
    p.J_W = J.copy(); p.J_h = h.copy(); p.J_l1_reg = 0.0
    p.sigma = np.random.choice([-1.0, 1.0], size=(p.mle_n_samp, len(h)))
    return p


def check_sampler(sampler, m=6, seed=0):
    J, h = random_ising(m, seed)
    mean_x, second_x, _, _ = exact_moments(J, h)

    np.random.seed(0)
    p = prior_at(J, h, sampler=sampler)
    S = p.sample(n_samp=40000, burn=80)          # {0,1}
    s = 2 * S - 1
    mean_s = s.mean(0)
    second_s = s.T @ s / len(s)
    return np.abs(mean_s - mean_x).max(), np.abs(second_s - second_x).max()


def check_grad_at_truth(sampler, m=6, seed=1, n_data=8000):
    J, h = random_ising(m, seed)
    _, _, states, w = exact_moments(J, h)

    rng = np.random.default_rng(123)
    idx = rng.choice(len(states), size=n_data, p=w)       # exact-sample the data
    ES = (states[idx] + 1) / 2                             # {0,1}

    np.random.seed(1)
    p = prior_at(J, h, mle_n_samp=8000, sampler=sampler)
    p._advance(p.sigma, 20)                               # equilibrate the particles
    gW, gh = p._grads(ES)
    return np.abs(gW).max(), np.abs(gh).max()


def check_learning(sampler, m=6, seed=2, n_data=8000, steps=400):
    J, h = random_ising(m, seed)
    _, second_x, states, w = exact_moments(J, h)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(states), size=n_data, p=w)
    ES = (states[idx] + 1) / 2

    np.random.seed(2)
    p = new_bae_priors.BoltzmannPriorNP(J_loss='mle', J_lr=0.05, mle_n_samp=2000,
                                        mle_gibbs_steps=2, sampler=sampler)
    p.init_params(ES, m)
    for _ in range(steps):
        p.learn(ES)

    Jsym = (p.J_W + p.J_W.T) / 2
    off = ~np.eye(m, dtype=bool)
    cos = (np.sum(Jsym[off] * J[off])
           / (np.linalg.norm(Jsym[off]) * np.linalg.norm(J[off]) + 1e-12))
    # moment mismatch on the learned model vs the data
    s = 2 * p.sample(n_samp=20000, burn=80) - 1
    mism = np.abs(s.T @ s / len(s) - second_x).max()
    return cos, mism


if __name__ == "__main__":
    ok = True
    for sampler in ('gibbs', 'gwg'):
        print(f"--- sampler = {sampler} ---")

        dm, dC = check_sampler(sampler)
        s_ok = dm < 0.03 and dC < 0.03
        ok &= s_ok
        print(f"[{'PASS' if s_ok else 'FAIL'}] sampler vs exact moments")
        print(f"    max|d<s>|={dm:.3f}   max|d<ss'>|={dC:.3f}")

        gW, gh = check_grad_at_truth(sampler)
        g_ok = gW < 0.05 and gh < 0.05
        ok &= g_ok
        print(f"[{'PASS' if g_ok else 'FAIL'}] gradient at truth ~ 0")
        print(f"    max|dL/dJ_W|={gW:.3f}   max|dL/dJ_h|={gh:.3f}")

        cos, mism = check_learning(sampler)
        l_ok = cos > 0.85 and mism < 0.06
        ok &= l_ok
        print(f"[{'PASS' if l_ok else 'FAIL'}] learning recovers the coupling")
        print(f"    cos(J_sym, J_true)={cos:.3f}   moment mismatch={mism:.3f}")

    print("\nALL PASS" if ok else "\nSOME FAILED")
