"""The variational log-evidence (BMF.elbo) and the chain selection built on it.

With n_chains > 1 the chains end up with different decoders, different noise
variances and -- with a structured prior -- different couplings, so a
reconstruction MSE does not compare them.  `best_chain` therefore ranks on

    ELBO = E_q[log p(X | S)] + E_q[log p(S)] + H[q],     q = prod_ij Bern(P_ij)

with q the fixed point of the model's OWN conditionals at temperature 1.  Checks:

  1. LOG-ODDS   model.logodds (the numpy mirror of the search kernel) equals the
                exact difference of the joint log-density under a spike flip --
                for the binary link, with sparsity, a coupling and a prior
                temperature all switched on.
  2. SLAB       the collapsed slab log-odds and the magnitude moments m1, m2
                match numerical quadrature of the same integral.
  3. BOUND      on a single row of data (2^m enumerable) the ELBO is a genuine
                lower bound on log p(x), and a tight one.
  4. PARTITION  the mean-field log Z is exact for an independent prior.
  5. SELECTION  the ELBO survives collapse() unchanged, and it sees a cost the
                MSE structurally cannot (a mis-scaled noise variance leaves the
                reconstruction error untouched).  The held-out comparison of the
                two criteria is printed as INFO: which chain generalizes better
                is noisy, and not something to assert.
"""

import itertools
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

import new_bae_models as nbm
import new_bae_priors as nbp
import new_bae_search


def seed_all(k):
    np.random.seed(k)
    new_bae_search._seed(k)


def synth(n=150, d=14, k=4, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    S = 1.0 * (rng.random((n, k)) < 0.4)
    W = rng.standard_normal((d, k))
    X = S @ W.T + noise * rng.standard_normal((n, d))
    return (X - X.mean(0)) / X.std()


FIT = dict(initial_temp=3, decay_rate=0.85, period=2, min_temp=1e-2, verbose=False)


def fitted(model_fn, X, seed=0, **fit_args):
    seed_all(seed)
    mod = model_fn()
    mod.fit(1.0 * X, **FIT, **fit_args)
    return mod


# ---------------------------------------------------------------------------
#  the joint log-density, written out independently of the model's own methods
# ---------------------------------------------------------------------------

def exact_logZ(J, g):
    """log sum_S exp(S'JS + g'S) by enumeration (m must be small)."""
    st = np.array(list(itertools.product([0.0, 1.0], repeat=len(g))))
    return logsumexp(np.einsum('si,ij,sj->s', st, J, st) + st @ g)


def joint_logp(mod, X, S, logZ=None):
    """log p(X, S) for the fitted model: Gaussian data term + the tempered prior
    (sparsity + coupling), normalizer included."""
    s2 = float(np.asarray(mod.sigma_x))
    resid = X - mod.operator.forward(S)
    data = (-0.5 * (resid ** 2).sum() / s2
            - 0.5 * resid.size * np.log(2 * np.pi * s2))
    J, g = mod.latent_prior._energy_params()
    energy = np.einsum('nj,jk,nk->', S, J, S) + (S * g).sum()
    Z = exact_logZ(J, g) if logZ is None else logZ
    return data + energy - len(S) * Z


# ---------------------------------------------------------------------------
#  1. the log-odds mirror
# ---------------------------------------------------------------------------

def check_logodds(seed=0, n_probe=40):
    X = synth(k=4, seed=seed)
    mod = fitted(lambda: nbm.JBMF(4, tree_reg=0, sparse_reg=0.3, J_lr=5e-2,
                                  slab=False), X, seed=seed, scl_lr=0.1)
    mod.latent_prior.temp = 1.7            # the prior temperature must be honored
    S = 1.0 * mod.S
    got = mod.logodds(X, S)

    J, g = mod.latent_prior._energy_params()
    Z = exact_logZ(J, g)
    rng = np.random.default_rng(seed)
    dev = 0.0
    for _ in range(n_probe):
        i = rng.integers(len(S))
        j = rng.integers(S.shape[1])
        S1, S0 = 1.0 * S, 1.0 * S
        S1[i, j], S0[i, j] = 1.0, 0.0
        want = joint_logp(mod, X, S1, Z) - joint_logp(mod, X, S0, Z)
        dev = max(dev, abs(want - got[i, j]))
    return dev, float(np.abs(got).mean())


# ---------------------------------------------------------------------------
#  2. the slab link against quadrature
# ---------------------------------------------------------------------------

def check_slab_link(seed=1, n_probe=12):
    """The collapsed log-odds is log int_0^inf tau e^{-tau z} exp((E z - g z^2/2)
    / sigma2) dz, and (m1, m2) are that density's first two moments."""
    rng = np.random.default_rng(seed)
    prior = nbp.LatentPrior(slab=True, slab_prior=1.3)
    dev_lo = dev_m1 = dev_m2 = 0.0
    for _ in range(n_probe):
        E = rng.normal(0, 3)
        g = abs(rng.normal(1.5, 0.5))
        s2 = abs(rng.normal(1.0, 0.3)) + 0.2
        tau = prior.slab_prior

        lo, m1, m2 = (float(v) for v in prior.link_moments(
            np.array([[E]]), np.array([[g]]), s2))

        def w(z):
            return tau * np.exp(-tau * z + (E * z - 0.5 * g * z ** 2) / s2)

        Z0 = quad(w, 0, np.inf)[0]
        dev_lo = max(dev_lo, abs(lo - np.log(Z0)))
        dev_m1 = max(dev_m1, abs(m1 - quad(lambda z: z * w(z), 0, np.inf)[0] / Z0))
        dev_m2 = max(dev_m2, abs(m2 - quad(lambda z: z * z * w(z), 0, np.inf)[0] / Z0))
    return dev_lo, dev_m1, dev_m2


# ---------------------------------------------------------------------------
#  3. the bound itself
# ---------------------------------------------------------------------------

def check_bound(model_fn, m=6, seed=2):
    """On ONE row the marginal log p(x) = log sum_S p(x, S) is enumerable, so the
    ELBO can be checked for what it claims to be."""
    X = synth(k=m, d=12, seed=seed)
    mod = fitted(model_fn, X, seed=seed, scl_lr=0.1)

    J, g = mod.latent_prior._energy_params()
    Z = exact_logZ(J, g)
    states = np.array(list(itertools.product([0.0, 1.0], repeat=m)))

    gaps = []
    for row in range(6):
        x = X[row:row + 1]
        joint = np.array([joint_logp(mod, x, st[None], Z) for st in states])
        exact = logsumexp(joint)
        gaps.append(exact - mod.elbo(x))
    return np.array(gaps)


# ---------------------------------------------------------------------------
#  5. selection
# ---------------------------------------------------------------------------

def check_collapse(C=3, seed=4):
    """A chain's ELBO must survive collapse() into a serial model unchanged."""
    X = synth(seed=seed)
    mod = fitted(lambda: nbm.BiPCA(4, tree_reg=0, n_chains=C, slab=True,
                                   J_prior='boltzmann', J_lr=5e-2),
                 X, seed=seed, scl_lr=0.1)
    per_chain = np.asarray(mod.elbo(X))
    b = mod.best_chain()
    mod.collapse(X)
    return b, float(per_chain[b]), mod.elbo(X), float(per_chain.max())


def check_criterion(C=3, seed=5):
    """The concrete thing MSE gets wrong.  Take the chain the MSE crowns and give
    it a badly scaled noise variance: the reconstruction error is IDENTICAL (loss
    never looks at sigma), so the old criterion still picks it, while the evidence
    sees what that variance costs and moves on."""
    X = synth(seed=seed)
    mod = fitted(lambda: nbm.BiPCA(4, tree_reg=0, n_chains=C,
                                   J_prior='boltzmann', J_lr=5e-2),
                 X, seed=seed, scl_lr=0.1)
    before, loss_before = np.asarray(mod.elbo(X)), np.asarray(mod.loss(X))
    mse_pick = int(np.argmin(loss_before))

    mod.sigma_x = np.asarray(mod.sigma_x, dtype=float).copy()
    mod.sigma_x[mse_pick] *= 50.0
    after, loss_after = np.asarray(mod.elbo(X)), np.asarray(mod.loss(X))
    return (mse_pick, before, after, float(np.abs(loss_after - loss_before).max()),
            int(np.argmin(loss_after)), mod.best_chain(X))


def check_generalization(C=4, seeds=range(6), hold=0.2):
    """Does the ELBO pick better chains than the MSE?  Fit with a hold-out mask,
    then score each chain's held-out imputation; report which criterion picks the
    chain with the better held-out log-likelihood."""
    elbo_ll, mse_ll, best_ll = [], [], []
    for sd in seeds:
        X = synth(seed=sd)
        M = np.random.default_rng(sd + 100).random(X.shape) < hold
        seed_all(sd)
        mod = nbm.BiPCA(4, tree_reg=0, n_chains=C)
        mod.fit(1.0 * X, mask=M, **FIT, scl_lr=0.1)

        S = mod.sample(1.0 * X, n_samp=8, burnin=10, mask=M, per_chain=True)
        ll = []
        for c in range(C):
            op = mod.operator.to_serial(c)
            pred = op.forward(S[:, c]).mean(0)
            s2 = float(np.atleast_1d(mod.sigma_x)[c])
            ll.append(float(np.mean(-0.5 * (X[M] - pred[M]) ** 2 / s2
                                    - 0.5 * np.log(2 * np.pi * s2))))
        ll = np.array(ll)
        elbo_ll.append(ll[int(np.argmax(mod.chain_score))])
        mse_ll.append(ll[int(np.argmin(mod.chain_loss))])
        best_ll.append(ll.max())
    return np.array(elbo_ll), np.array(mse_ll), np.array(best_ll)


if __name__ == "__main__":
    ok = True

    def report(flag, title, detail):
        global ok
        ok &= bool(flag)
        print(f"[{'PASS' if flag else 'FAIL'}] {title}")
        print(f"    {detail}")

    dev, scale = check_logodds()
    report(dev < 1e-8,
           "logodds == the exact joint-density difference under a spike flip",
           f"max|d| over 40 probes = {dev:.2e}   (mean |logodds| = {scale:.2f})")

    dlo, dm1, dm2 = check_slab_link()
    report(max(dlo, dm1, dm2) < 1e-7,
           "the slab link matches numerical quadrature",
           f"max|d logodds| = {dlo:.2e}   max|d m1| = {dm1:.2e}   "
           f"max|d m2| = {dm2:.2e}")

    for name, fn in (('plain', lambda: nbm.SemiBMF(6, tree_reg=0)),
                     ('sparse', lambda: nbm.SemiBMF(6, tree_reg=0, sparse_reg=0.5)),
                     ('boltzmann', lambda: nbm.JBMF(6, tree_reg=0, J_lr=5e-2))):
        gaps = check_bound(fn)
        report(gaps.min() > -1e-9 and gaps.max() < 2.0,
               f"the ELBO is a lower bound on log p(x), and tight ({name})",
               f"log p(x) - ELBO over 6 rows: "
               f"[{gaps.min():+.4f}, {gaps.max():+.4f}] nats")

    m = 8
    g = np.random.default_rng(0).standard_normal(m)
    mf = float(nbp.LatentPrior._mf_logZ(np.zeros((m, m)), g))
    ex = exact_logZ(np.zeros((m, m)), g)
    report(abs(mf - ex) < 1e-9, "the mean-field log Z is exact when J = 0",
           f"mean field {mf:.6f}   exact {ex:.6f}")

    b, want, got, best = check_collapse()
    report(abs(want - got) < 1e-8 and want == best,
           "collapse() keeps the winning chain's ELBO",
           f"chain {b}: {want:.4f} before, {got:.4f} after "
           f"(and it is the argmax, {best:.4f})")

    pick, before, after, dloss, mse_after, best = check_criterion()
    report(dloss < 1e-12 and after[pick] < before[pick] - 1
           and mse_after == pick and best != pick,
           "the ELBO sees a cost the MSE cannot (a mis-scaled noise variance)",
           f"chain {pick} sigma x50: loss unchanged (max|d| = {dloss:.1e}, still "
           f"argmin {mse_after})\n    "
           f"ELBO {before[pick]:.1f} -> {after[pick]:.1f}, so best_chain moves "
           f"to {best}")

    e, s, top = check_generalization(seeds=range(8))
    print("[INFO] held-out ll of the selected chain, 8 seeds x 4 chains "
          "(noisy -- not asserted)")
    print(f"    ELBO {e.mean():+.4f}   MSE {s.mean():+.4f}   "
          f"best possible {top.mean():+.4f}\n    "
          f"per seed  ELBO {np.round(e, 3)}\n              MSE  {np.round(s, 3)}")

    print("\nALL PASS" if ok else "\nSOME FAILED")
