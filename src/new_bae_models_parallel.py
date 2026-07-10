"""
new_bae_models_parallel.py  --  PROTOTYPE (parallel chains)
===========================================================

Parallel-chains variant of new_bae_models.{BMF, LinearGaussianBMF, SemiBMF}.

Motivation: the BMF objective is non-convex and the fit is stochastic, so you want
several independent chains from different inits and keep the best.  Today that is
bae_util.multifit -- a Python `for` loop that refits the entire model N times.
Here the chains are a leading array axis C instead:

  * the numpy operator + prior carry a `c` axis and broadcast (no Python loop);
  * the coordinate-descent E-step runs all chains in a numba prange;
  * fit() advances all C chains on the SAME annealing schedule, and at the end
    `select_best` / `collapse` pick the lowest-loss chain.

What the chain axis touches, and nothing else:
  latent state  S / Z / StS   (C, n, m) / (C, m, m)   -- on the prior
  weights       W / b         (C, d, m) / (C, d)      -- on the operator
  noise         sigma_x       (C,)                    -- on the model
  loss          (C,)                                  -- argmin picks the winner

Everything else (the annealing loop, the E/M-step orchestration, the composition
of operator + prior) is byte-for-byte the serial structure.  This file implements
the SemiBMF path (ParallelAffineOperator + ParallelLatentPrior, no slab); other
models are the same assembly with a different operator, once those operators grow
a chain axis the same mechanical way.

Design choice -- one schedule, C chains -- matches multi-start: all chains see the
same temperature at iteration t; only their random flips and their inits differ.
(If you wanted per-chain schedules -- e.g. a temperature ladder / parallel
tempering -- `temp` becomes a (C,) array passed into the kernel and read as
temp[c].  Left out here to keep the prototype simple.)
"""

import numpy as np
from dataclasses import dataclass, field

from new_bae_priors_parallel import ParallelLatentPrior
from new_bae_weights_parallel import ParallelAffineOperator, ParallelProcrustes


@dataclass
class ParallelBMF:
    """Annealing fit loop over C chains at once.  Mirrors new_bae_models.BMF; the
    only difference is that grad_step advances a whole batch of chains and the
    loss it records is a (C,) vector."""

    def __post_init__(self):
        self.temp = 1
        self.initialized = False

    def initialize(self, X, **args):
        self.init_params(X, **args)
        self.init_latents(X, **args)
        self.initialized = True

    def fit(self, *data, initial_temp=10, decay_rate=0.88, period=10,
            min_temp=1, max_iter=None, verbose=True, **opt_args):
        if max_iter is None:
            max_iter = period * int(np.log(1e-4 / initial_temp) / np.log(decay_rate))
        if verbose:
            from tqdm import tqdm
            pbar = tqdm(range(max_iter))

        en = []                                   # per-iteration (C,) loss vectors
        if not self.initialized:
            self.initialize(*data, **opt_args)
        for it in range(max_iter):
            self.temp = min_temp + initial_temp * (decay_rate ** (it // period))
            _, ls = self.grad_step(*data)
            en.append(ls)
            if verbose:
                pbar.update(1)

        self.chain_loss = np.asarray(en[-1])      # final per-chain loss
        return np.asarray(en)                     # (max_iter, C)


@dataclass
class ParallelLinearGaussianBMF(ParallelBMF):
    """Shared linear-Gaussian skeleton, chain-batched.  Same seams as
    new_bae_models.LinearGaussianBMF (operator + latent_prior), everything sized
    with a leading chain axis."""

    dim_hid: int = None
    n_chains: int = 8
    fit_intercept: bool = True
    operator: object = None
    latent_prior: object = None
    m_iters: int = 1

    @property
    def S(self):
        return self.latent_prior.S               # (C, n, m)

    def __call__(self, S):
        return self.operator.forward(S)          # (C, n, d)

    def loss(self, X, mask=None):
        """Per-chain reconstruction MSE -> (C,)."""
        N = self(self.S)                         # (C, n, d)
        if mask is None:
            return ((X[None] - N) ** 2).mean((1, 2))
        return ((X[None] - N) ** 2)[:, mask].mean(1)

    # ---- init: operator + prior, then compose the parallel search ----------
    def init_params(self, X, hot_start=True, scl_lr=0, **opt_args):
        self.n = len(X)
        self.sigma_x = np.ones(self.n_chains)    # (C,) per-chain noise
        self.scl_lr = scl_lr
        self.operator.init_params(X, self.dim_hid, hot_start=hot_start, **opt_args)
        self.operator.build_search(self.latent_prior.link,
                                   self.latent_prior.prior_plugin)

    def init_latents(self, X, **args):
        self.latent_prior.init_latents(self.operator.drive(X))

    # ---- E-step (chain-batched) -------------------------------------------
    def EStep(self, X, S, Z=None):
        if Z is None:
            Z = 1.0 * S
        lp = self.latent_prior
        XW = self.operator.drive(X)              # (C, n, m)
        WtW = self.operator.gram()               # (C, m, m)
        Jc, hc = lp.coupling()                   # (C, m, m), (C, m)
        self.operator.search(
            XW, S, Z, WtW, lp.StS, self.n, self.temp,
            lp.sparse_reg, lp.tree_reg, lp.slab_prior, self.sigma_x, Jc, hc,
            True, None, lp.temp)                 # trailing prior_temp (shared scalar)
        return Z if lp.slab else S

    def grad_step(self, X, mask=None):
        lp = self.latent_prior
        newES = self.EStep(X, lp.S, lp.Z)
        loss = self.MStep(X, newES)
        return newES, loss

    # ---- M-step (chain-batched) -------------------------------------------
    def MStep(self, X, S, Z=None):
        if Z is None:
            Z = 1 * S
        for _ in range(self.m_iters):
            resid = self.operator.backward(Z, X)             # (C, n, d)
            self.latent_prior.learn(S)
        err = (resid ** 2).mean((1, 2))                      # (C,)
        self.sigma_x += self.scl_lr * (err - self.sigma_x)
        return err

    # ---- pick the winning chain -------------------------------------------
    def best_chain(self, X=None):
        """Index of the lowest-loss chain (uses the cached final loss unless X is
        given, in which case it is recomputed)."""
        loss = self.chain_loss if X is None else self.loss(X)
        return int(np.argmin(loss))

    def reconstruct_best(self, X=None):
        c = self.best_chain(X)
        return self.operator.forward(self.S)[c]              # (n, d)

    def collapse(self, X=None):
        """Squeeze the model down to its winning chain: S/W/b/sigma_x lose the
        chain axis, so downstream code that expects the serial (2-D S, 2-D W)
        shapes works unchanged.  Returns self."""
        c = self.best_chain(X)
        lp, op = self.latent_prior, self.operator
        lp.S, lp.Z, lp.StS = lp.S[c], lp.Z[c], lp.StS[c]
        op.W, op.b = op.W[c], op.b[c]
        if hasattr(op, 'scl'):                   # Procrustes carries a per-chain scale
            op.scl = float(np.atleast_1d(op.scl)[c])
        self.sigma_x = float(np.atleast_1d(self.sigma_x)[c])
        self.n_chains = 1
        return self


@dataclass
class ParallelSemiBMF(ParallelLinearGaussianBMF):
    """Parallel-chains SemiBMF: ParallelAffineOperator + ParallelLatentPrior.
    Constructor mirrors new_bae_models.SemiBMF with an added `n_chains`."""

    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l2_reg: float = 1e-2
    weight_l1_reg: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.operator = ParallelAffineOperator(n_chains=self.n_chains,
                                               fit_intercept=self.fit_intercept,
                                               nonneg=self.nonneg,
                                               pr_reg=self.weight_pr_reg,
                                               l1_reg=self.weight_l1_reg,
                                               l2_reg=self.weight_l2_reg)
        self.latent_prior = ParallelLatentPrior(n_chains=self.n_chains,
                                                sparse_reg=self.sparse_reg,
                                                tree_reg=self.tree_reg)


@dataclass
class ParallelBiPCA(ParallelLinearGaussianBMF):
    """Parallel-chains BiPCA: ParallelProcrustes (orthonormal W + scalar scale) +
    ParallelLatentPrior.  Mirrors new_bae_models.BiPCA with an added n_chains.

    Same story as ParallelSemiBMF -- the chains are a leading axis and the winner
    is picked at the end -- but the operator is the Procrustes one, so every chain
    keeps orthonormal weights and rides the shared binary link with diag_gram=True.
    sigma_x IS the modelled noise variance sigma^2 handed to the search: fixed at 1
    by default (scl_lr=0), estimated from residual MSE if init/fit is given
    scl_lr > 0.  The chains differ because init_params seeds a shared PCA hot-start
    and jitters+re-orthonormalizes each chain's W independently.

    Only the plain (J_lr == 0) LatentPrior path is prototyped; the Boltzmann prior
    would drop in as ParallelBoltzmannPrior exactly like the serial BiPCA, with a
    per-chain coupling axis (see new_bae_priors_parallel's note)."""

    sparse_reg: float = 1e-2
    tree_reg: float = 0
    fit_intercept: bool = True
    fit_scl: bool = True
    slab: bool = False
    slab_prior: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.operator = ParallelProcrustes(n_chains=self.n_chains,
                                           fit_intercept=self.fit_intercept,
                                           fit_scl=self.fit_scl)
        self.latent_prior = ParallelLatentPrior(n_chains=self.n_chains,
                                                sparse_reg=self.sparse_reg,
                                                tree_reg=self.tree_reg,
                                                slab=self.slab,
                                                slab_prior=self.slab_prior)


# ===========================================================================
#  demo: C parallel chains vs the serial multi-start it replaces
# ===========================================================================

def _demo():
    import time
    import bae_util
    import new_bae_models as nbm

    rng = np.random.RandomState(0)
    n, d, m = 200, 60, 6
    Wtrue = rng.randn(d, m)
    Strue = 1.0 * (rng.rand(n, m) > 0.5)
    X = Strue @ Wtrue.T + 0.1 * rng.randn(n, d)

    C = 8
    fit_kw = dict(initial_temp=5, decay_rate=0.85, period=5, verbose=False)

    # warm up the numba parallel kernel so the timed fit excludes the one-time
    # (parallel=True) JIT compile -- otherwise the first parallel fit is dominated
    # by compilation, not by the sweep.
    ParallelSemiBMF(dim_hid=m, n_chains=C, tree_reg=0.0, sparse_reg=0.0).fit(
        X, max_iter=1, verbose=False)

    # ---- parallel: C chains in one fit -----------------------------------
    par = ParallelSemiBMF(dim_hid=m, n_chains=C, tree_reg=0.0, sparse_reg=0.0)
    t0 = time.perf_counter()
    en = par.fit(X, **fit_kw)
    t_par = time.perf_counter() - t0
    losses = par.loss(X)
    best = par.best_chain(X)
    print(f"[parallel] per-chain final loss: {np.round(losses, 4)}")
    print(f"[parallel] best chain {best} -> loss {losses[best]:.4f}   "
          f"(worst {losses.max():.4f})   [{t_par:.2f}s for {C} chains]")

    # ---- serial multi-start: C independent fits --------------------------
    nbm.SemiBMF(dim_hid=m, tree_reg=0.0, sparse_reg=0.0).fit(
        X, max_iter=1, verbose=False)                     # warm up serial JIT too
    t0 = time.perf_counter()
    serial_losses = []
    for _ in range(C):
        mod = nbm.SemiBMF(dim_hid=m, tree_reg=0.0, sparse_reg=0.0)
        mod.fit(X, **fit_kw)
        serial_losses.append(mod.loss(X))
    t_ser = time.perf_counter() - t0
    serial_losses = np.array(serial_losses)
    print(f"[serial]   per-chain final loss: {np.round(serial_losses, 4)}")
    print(f"[serial]   best {serial_losses.min():.4f}   "
          f"[{t_ser:.2f}s for {C} fits]  ->  parallel is {t_ser / t_par:.2f}x faster")

    # ---- collapse to the winner: serial shapes restored ------------------
    par.collapse(X)
    recon = par.S @ par.operator.W.T + par.operator.b        # plain 2-D affine map
    print(f"[collapse] S {par.S.shape}  W {par.operator.W.shape}  "
          f"recon MSE {((X - recon) ** 2).mean():.4f}")


if __name__ == "__main__":
    _demo()
