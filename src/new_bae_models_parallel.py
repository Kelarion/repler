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

from typing import Optional

import new_bae_priors as nbp                  # TempSchedule: the prior-temp schedule
from new_bae_priors_parallel import ParallelLatentPrior, ParallelBoltzmannPriorNP
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
            min_temp=1, prior_schedule=None, max_iter=None, verbose=True,
            mask=None, **opt_args):
        # Signature parity with new_bae_models.BMF.fit: the *prior* temperature
        # (latent_prior.temp) runs on its own schedule -- `prior_schedule.update()`
        # once per iteration, defaulting to a constant 1.0 (nbp.TempSchedule); only
        # a structured prior reads it, and only for sampling S.  `mask` switches on
        # imputation -- each chain fills the masked entries from its own
        # reconstruction inside grad_step.  One schedule (model temp AND prior temp)
        # is shared across all C chains.
        if max_iter is None:
            max_iter = period * int(np.log(1e-4 / initial_temp) / np.log(decay_rate))

        if prior_schedule is None:
            prior_schedule = nbp.TempSchedule()
        sched_prior = hasattr(self, 'latent_prior')

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(range(max_iter))

        en = []                                   # per-iteration (C,) loss vectors
        if not self.initialized:
            self.initialize(*data, **opt_args)
        for it in range(max_iter):
            T = min_temp + initial_temp * (decay_rate ** (it // period))
            self.temp = T
            if sched_prior:
                self.latent_prior.temp = prior_schedule.update(
                    it=it, max_iter=max_iter, temp=T,
                    loss=en[-1] if en else None)
            _, ls = self.grad_step(*data, mask=mask)
            en.append(ls)
            if verbose:
                pbar.update(1)

        self.chain_loss = np.asarray(en[-1])      # final per-chain loss
        return np.asarray(en)                     # (max_iter, C)

    def sample(self, X, temp=None, n_samp=1, burnin=10, **args):
        """Gibbs-sample the latents for the rows of X, all C chains at once
        (new_bae_models.BMF.sample, chain-batched).  Fresh chain state shaped
        (C, len(X), m) -- NOT the prior's persistent S/Z -- so the prior's StS
        stays frozen (inplace=False); the search walks one Gibbs chain per (chain,
        sample) from a random init.  Returns (n_samp, C, len(X), m)."""
        if temp is not None:
            self.temp = temp

        C, m = self.n_chains, self.dim_hid
        samps = np.zeros((n_samp, C, len(X), m))

        S = 1.0 * np.random.choice([0, 1], size=(C, len(X), m))
        Z = 1.0 * S
        i = 0
        for it in range(n_samp * burnin):
            samp = self.EStep(X, S, Z, inplace=False, **args)
            if not np.mod(it, burnin):
                samps[i] = 1 * samp
                i += 1

        return samps


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
    debug: bool = False                          # record per-iteration E-step log-odds
    saem: bool = False                           # smooth the sufficient stat (SAEM)
    gamma: float = 1.0

    @property
    def S(self):
        return self.latent_prior.S               # (C, n, m)

    def __call__(self, S):
        return self.operator.forward(S)          # (C, n, d)

    def loss(self, X, mask=None):
        """Per-chain reconstruction MSE -> (C,).  A boolean (n, d) `mask` scores
        only the selected entries (e.g. held-out entries for imputation quality)."""
        N = self(self.S)                         # (C, n, d)
        if mask is None:
            return ((X[None] - N) ** 2).mean((1, 2))
        return ((X[None] - N) ** 2)[:, mask].mean(1)

    def loglikelihood(self, X, Xhat):
        """Per-element Gaussian log-likelihood, chain-batched -> (C, n, d).
        (new_bae_models.LinearGaussianBMF.loglikelihood; sigma_x is (C,).)"""
        Xb = X if X.ndim == 3 else X[None]
        sig = self.sigma_x[:, None, None]
        dot = 0.5 * ((Xhat - Xb) ** 2) / sig
        lnrm = (0.5 * np.log(self.sigma_x) + 0.5 * np.log(2 * np.pi))[:, None, None]
        return -(dot + lnrm)

    # ---- init: operator + prior, then compose the parallel search ----------
    def init_params(self, X, hot_start=True, scl_lr=0, **opt_args):
        self.n = len(X)
        self.sigma_x = np.ones(self.n_chains)    # (C,) per-chain noise
        self.scl_lr = scl_lr
        self.outs = []                           # per-iteration log-odds if debug
        self._Ximp = None                        # per-chain imputation copy (lazy)
        self.operator.init_params(X, self.dim_hid, hot_start=hot_start, **opt_args)
        self.operator.build_search(self.latent_prior.link,
                                   self.latent_prior.prior_plugin, debug=self.debug)

    def init_latents(self, X, **args):
        self.latent_prior.init_latents(self.operator.drive(X))
        # SAEM: seed the running expected sufficient stat one gamma-step from the
        # uniform-prior mean (0.5) toward the initial effective latent, per chain.
        if self.saem:
            self.ES = 0.5 + self.gamma * (self.latent_prior.Z - 0.5)

    # ---- E-step (chain-batched) -------------------------------------------
    # Full signature parity with new_bae_models.LinearGaussianBMF.EStep: `inplace`
    # (False freezes the prior's StS -- used by `sample`), `debug` records the
    # per-element log-odds into `out`, SAEM smooths the sufficient stat, and a
    # `mask` triggers per-chain imputation (returns the imputed data alongside ES).
    def EStep(self, X, S, Z=None, mask=None, inplace=True, slab=True, **kwargs):
        if Z is None:
            Z = 1.0 * S
        lp = self.latent_prior
        XW = self.operator.drive(X)              # (C, n, m)
        WtW = self.operator.gram()               # (C, m, m)
        Jc, hc = lp.coupling()                   # (C, m, m), (C, m)
        out = np.zeros(S.shape) if self.debug else None
        self.operator.search(
            XW, S, Z, WtW, lp.StS, self.n, self.temp,
            lp.sparse_reg, lp.tree_reg, lp.slab_prior, self.sigma_x, Jc, hc,
            inplace, out, lp.temp)               # trailing prior_temp (shared scalar)
        if self.debug and inplace:
            self.outs.append(out)
        ES = Z if slab else S                    # Z == S for a no-slab prior

        if self.saem and inplace:
            self.ES += self.gamma * (ES - self.ES)
            ES = self.ES

        if mask is None:
            return ES
        return ES, self.impute(X, ES, mask)

    def impute(self, X, ES, mask):
        """X[:, mask] <- a per-chain sample from p(X | latents) = N(forward(ES),
        sigma_x).  X is the PER-CHAIN working copy (C, n, d) -- each chain fills the
        masked holes with its OWN reconstruction -- mutated in place and returned."""
        Xhat = self.operator.forward(ES)         # (C, n, d)
        X[:, mask] = Xhat[:, mask]
        return X

    def grad_step(self, X, mask=None):
        lp = self.latent_prior
        if mask is None:
            newES = self.EStep(X, lp.S, lp.Z)
            loss = self.MStep(X, newES)
        else:
            # keep a per-chain copy of the data so chains impute independently; the
            # observed entries stay at the true X, the masked ones are refined each
            # sweep.  Seeded once from the passed X (broadcast over chains).
            if self._Ximp is None:
                self._Ximp = np.repeat(np.asarray(X, float)[None], self.n_chains, 0)
            newES, self._Ximp = self.EStep(self._Ximp, lp.S, lp.Z, mask=mask)
            loss = self.MStep(self._Ximp, newES)
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
    def best_chain(self, X=None, mask=None):
        """Index of the lowest-loss chain (uses the cached final loss unless X is
        given, in which case it is recomputed).  Pass the OBSERVED-entry `mask` when
        imputing to rank chains by their fit to the observed data only."""
        loss = self.chain_loss if X is None else self.loss(X, mask=mask)
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

    Optional structured (Boltzmann/Ising) prior, matching serial BiPCA: leave
    J_lr == 0 for the plain per-chain LatentPrior, or set J_lr > 0 to swap in
    ParallelBoltzmannPriorNP, which learns an independent Ising coupling per chain
    (each chain fits the prior to its own spike configuration).  The chains all
    ride the same PRIOR_BOLTZMANN kernel -- it already reads Jc[c] / hc[c]."""

    sparse_reg: float = 1e-2
    tree_reg: float = 0
    fit_intercept: bool = True
    fit_scl: bool = True

    J_l1_reg: Optional[float] = None
    J_loss: str = 'rple'
    J_lr: float = 0
    slab: bool = False
    slab_prior: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.operator = ParallelProcrustes(n_chains=self.n_chains,
                                           fit_intercept=self.fit_intercept,
                                           fit_scl=self.fit_scl)
        if self.J_lr == 0:                       # plain per-chain Bernoulli prior
            self.latent_prior = ParallelLatentPrior(n_chains=self.n_chains,
                                                    sparse_reg=self.sparse_reg,
                                                    tree_reg=self.tree_reg,
                                                    slab=self.slab,
                                                    slab_prior=self.slab_prior)
        else:                                    # structured Ising prior per chain
            self.latent_prior = ParallelBoltzmannPriorNP(
                n_chains=self.n_chains,
                sparse_reg=self.sparse_reg,
                tree_reg=self.tree_reg,
                J_l1_reg=self.J_l1_reg,
                J_loss=self.J_loss,
                J_lr=self.J_lr,
                slab=self.slab,
                slab_prior=self.slab_prior,
                sampler='gibbs')


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
