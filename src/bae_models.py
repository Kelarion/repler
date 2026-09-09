"""
bae_models.py  --  the models
=================================

A refactor of `old_bae_models.py` that pulls the structure shared by every BMF model
into a few composable pieces, so a change to shared behaviour is made once
instead of copy-pasted into ~10 classes.  Three modules:

  bae_priors.py   the latent side (one slot: `latent_prior`)
      LatentPrior      owns the latent state (S, StS, Z), its init, the
                       sparsity/tree regularizers, the additive S-prior it
                       injects into the search, and the search `link`;
                       `slab=True` makes it spike-and-slab
      BoltzmannPrior   pairwise Ising prior in the {0,1} coding -- droppable
                       onto any model, slab included
      MRFPrior         the same with a sign-constrained ({-1,0,+1}) coupling

  bae_weights.py  the weight side (one slot: `operator`)
      LinearOperator   L's faces (forward / drive / gram), the search kernel, and
                       `backward` (one M-step update of its own parameters)
        AffineOperator   L(S) = S W^T + b             (numpy)
        Procrustes       L(S) = scl S W^T + b, W orthonormal (BiPCA)
        TorchMatrixOp    L(S) = W(S)                  (nn.Linear / autograd)
        CPOperator       L(S) = einsum CP             (SCPD)
        ReducedRankOp    L(S) = reduced-rank tensor   (RRBMF)
        ConvOperator     L(S) = conv1d(S, K)          (ConvBMF)

  bae_models.py   (this file)
      BMF                annealing fit loop + Gibbs sampling
      LinearGaussianBMF  X_hat = L(S) with Gaussian likelihood: holds an operator
                         + a latent_prior and implements the ONE E-step / M-step /
                         init that serve every model below.

Every linear-Gaussian model is then just its __post_init__ -- the assembly of an
operator + a latent_prior.  The E-step is identical once L exposes three faces:

    forward(S)   reconstruction          L(S)            (-> __call__)
    drive(X)     adjoint on residual     L*(X - b)       (-> XW)
    gram()       operator metric         L*L             (-> WtW)

KernelBMF is the exception that proves the rule: its reconstruction
K ~ center(S diag(scl) S^T) is QUADRATIC in S, so it is not a LinearOperator and
subclasses BMF directly -- but it still composes a `latent_prior` unchanged, so a
structured prior drops onto it for free.

The update equations follow minimal_structured_bipca.StructuredBiPCA: binary and
spike-and-slab likelihood log-odds, an additive {0,1} Ising prior learned from the
BINARY spike, a scale-free (log-space) observation-variance update, and the
optional end-of-fit prior refit + fixed-prior refinement exposed as `fit` options.

Multiple chains are carried by a single `n_chains` field threaded into the
operator and the prior (n_chains == 1 is the ordinary serial 2-D model); the few
shape-dependent methods branch on it, the rest broadcast.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# pure orchestration -- no torch / scipy here; the numerics live in the operators
# and priors this module composes.
import bae_search
import bae_priors as nbp
from bae_priors import LatentPrior, BoltzmannPrior, MRFPrior
from bae_weights import (
    LinearOperator,
    AffineOperator, Procrustes, TorchMatrixOp, CPOperator, ReducedRankOp, ConvOperator,
)

_PRIORS = {'none': LatentPrior, 'boltzmann': BoltzmannPrior, 'mrf': MRFPrior}


def make_prior(kind='none', n_chains=1, sparse_reg=0.0, tree_reg=1e-2,
               slab=False, slab_prior=1.0, **prior_args):
    """Build a latent prior from the fields every model exposes.  `kind` is
    'none' (LatentPrior), 'boltzmann' (BoltzmannPrior) or 'mrf' (MRFPrior);
    `prior_args` are forwarded to the structured classes only."""
    common = dict(n_chains=n_chains, sparse_reg=sparse_reg, tree_reg=tree_reg,
                  slab=slab, slab_prior=slab_prior)
    if kind == 'none':
        return LatentPrior(**common)
    return _PRIORS[kind](**common, **prior_args)


def _sigmoid(x):
    """Stable logistic (no scipy in this module -- see the header)."""
    return 0.5 * (1 + np.tanh(0.5 * x))


def _bern_entropy(P):
    """Elementwise entropy of independent Bernoulli(P), in nats."""
    p = np.clip(P, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def _center_kernel(K):
    """Double-centering of a Gram matrix (== util.center_kernel), inlined so this
    module stays scipy/torch-free."""
    return (K - K.mean(-2, keepdims=True) - K.mean(-1, keepdims=True)
            + K.mean((-1, -2), keepdims=True))


# ===========================================================================
#  Training-time probes
# ===========================================================================
#
# A `probe` names something to record once per fit iteration: either a dotted
# attribute path walked from the model ('sigma_x', 'latent_prior.J', 'operator.W')
# or a callable model -> value.  `fit(..., probes=[...])` collects them into
# self.history = {name: [per-iteration snapshot]}.  Arrays are COPIED, so the
# history holds each iteration's state rather than a live alias.

def _snapshot(v):
    if hasattr(v, 'detach'):                       # torch tensor / nn.Parameter
        v = v.detach()
        v = v.cpu() if hasattr(v, 'cpu') else v
        return np.asarray(v).copy()
    if isinstance(v, np.ndarray):
        return v.copy()
    return v


def _probe_value(model, probe):
    if callable(probe):
        return _snapshot(probe(model))
    obj = model
    for attr in probe.split('.'):
        obj = getattr(obj, attr)
    return _snapshot(obj)


def _as_probe_dict(probes):
    """Normalize `probes` to {name: probe}; a bare list keys each entry by its own
    name (a callable's __name__, so lambdas want an explicit dict)."""
    if probes is None:
        return {}
    if isinstance(probes, dict):
        return probes
    return {(p if isinstance(p, str) else getattr(p, '__name__', repr(p))): p
            for p in probes}


# ===========================================================================
#  Base class: the fit loop
# ===========================================================================

@dataclass
class BMF:
    """Annealing fit loop + Gibbs sampling."""

    def __post_init__(self):
        self.temp = 1
        self.initialized = False

    def initialize(self, X, **args):
        self.init_params(X, **args)
        self.init_latents(X, **args)
        self.initialized = True

    def fit(self, *data, initial_temp=0, decay_rate=1.0, period=10, min_temp=1,
            prior_schedule=None, prior_delay=50, prior_refit=False,
            refine_iters=0, refine_sweeps=2, refine_lr_scale=0.5,
            max_iter=600, verbose=True, mask=None, probes=None, **opt_args):
        """Anneal the model temperature geometrically, taking one E-step + one
        M-step per iteration.

        prior_schedule   object with `update(**inputs) -> float`, called once per
                         iteration to set latent_prior.temp.  The prior temperature
                         is a SAMPLING knob only (the search divides the coupling's
                         log-odds by it), so a schedule here changes the search, not
                         the fitted coupling.  Default: pinned at 1.
        prior_delay      iterations to run before the prior starts learning, so the
                         coupling is fit to codes the decoder has already shaped.
        prior_refit      after the main loop, refit the prior's parameters from the
                         final spikes (support selection + debias + one temperature;
                         see BoltzmannPrior.refit).  No-op for an unstructured prior.
        refine_iters     iterations of a closing FIXED-prior refinement: the prior's
                         parameters are frozen while the codes and decoder settle at
                         min_temp, with `refine_sweeps` E-steps per M-step and every
                         learning rate scaled by `refine_lr_scale`.
        mask             boolean array over the data; the masked entries are imputed
                         from the generative model each iteration (the working array
                         in `data` is refined in place).
        probes           extra quantities to track (see _probe_value) -> self.history.
        """
        if max_iter is None:
            max_iter = period * int(np.log(1e-4 / initial_temp) / np.log(decay_rate))
        if prior_schedule is None:
            prior_schedule = nbp.TempSchedule()
        sched_prior = hasattr(self, 'latent_prior')

        probes = _as_probe_dict(probes)
        self.history = {name: [] for name in probes}

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(range(max_iter))

        en = []
        if not self.initialized:
            self.initialize(*data, **opt_args)
        # A masked fit owns its imputations: re-seed the per-chain working copy from
        # THIS call's data, so a refit (a new CV fold, say) starts from the observed
        # entries rather than the previous call's fill.
        if mask is not None:
            self._Ximp = None
        for it in range(max_iter):
            self.temp = min_temp + initial_temp * (decay_rate ** (it // period))
            if sched_prior:
                self.latent_prior.temp = prior_schedule.update(
                    it=it, max_iter=max_iter, temp=self.temp,
                    loss=en[-1] if en else None)
            _, ls = self.grad_step(*data, mask=mask, learn_prior=it >= prior_delay)
            en.append(ls)

            for name, probe in probes.items():
                self.history[name].append(_probe_value(self, probe))
            if verbose:
                pbar.update(1)

        if prior_refit and sched_prior:
            self.latent_prior.refit(self.S)
        if refine_iters:
            en += self.refine(*data, n_iter=refine_iters, sweeps=refine_sweeps,
                              lr_scale=refine_lr_scale, temp=min_temp, mask=mask)

        # multi-chain: cache the final per-chain (C,) selection score (and the
        # matching reconstruction loss) so best_chain / collapse can pick the
        # winner without recomputing.  Inert for a single chain.  Under a mask
        # the fit's own energy scores each chain against ITS OWN imputations,
        # which is not comparable across chains -- score the observed entries.
        if getattr(self, 'n_chains', 1) > 1:
            keep = None if mask is None else ~np.asarray(mask)
            self.chain_loss = (np.asarray(en[-1]) if mask is None
                               else np.asarray(self.loss(data[0], mask=keep)))
            self.chain_score = np.asarray(self._chain_score(data[0], mask=keep))
        return en

    def refine(self, *data, n_iter=1, sweeps=1, lr_scale=1.0, temp=1.0, mask=None):
        """Settle the codes and the decoder with the prior's parameters FROZEN, at
        a fixed temperature and reduced learning rates.  Returns the losses."""
        self.temp = temp
        if lr_scale:
            self._scale_lrs(lr_scale)
        en = []
        for _ in range(n_iter):
            for _ in range(sweeps - 1):
                self._latent_sweep(*data)
            en.append(self.grad_step(*data, mask=mask, learn_prior=False)[1])
        if lr_scale:
            self._scale_lrs(1 / lr_scale)
        return en

    def _chain_score(self, X, mask=None, **args):
        """Per-chain model-selection score, HIGHER is better (hook for
        best_chain).  The base score is the negative reconstruction loss;
        LinearGaussianBMF overrides it with a variational log-evidence."""
        return -np.asarray(self.loss(X))

    def _scale_lrs(self, factor):
        """Multiply every learning rate this model owns (hook for `refine`)."""
        return None

    def _latent_sweep(self, X, **args):
        """One extra E-step on the persistent latents (hook for `refine`)."""
        return self.EStep(X, self.S)

    def sample(self, X, temp=None, n_samp=1, burnin=10, per_chain=False, mask=None,
               **args):
        """Draw latent samples for the rows of X by walking a FRESH Gibbs chain
        (not the prior's persistent state, which is sized to the training set).

        With n_chains > 1 every chain is sampled at once; by default only the best
        chain's draws are returned, shape (n_samp, len(X), dim_hid) -- the same
        contract as a single-chain model.  per_chain=True keeps them all,
        (n_samp, C, len(X), dim_hid).

        With a `mask` the chain conditions on an IMPUTED X: the masked entries are
        refilled from the generative model before the first sweep and after every
        one, so the latents never see the held-out values.  A copy is used, so the
        caller's X is never mutated."""
        if temp is not None:
            self.temp = temp

        C = getattr(self, 'n_chains', 1)
        multi = C > 1
        batch = (C,) if multi else ()
        samps = np.zeros((n_samp,) + batch + (len(X), self.dim_hid))
        S = 1.0 * np.random.choice([0, 1], size=batch + (len(X), self.dim_hid))
        Z = 1.0 * S

        if mask is None:
            Xw = X
        else:
            Xw = 1.0 * np.asarray(X, float)
            Xw = np.repeat(Xw[None], C, 0) if multi else Xw
            # fill the holes from the CHAIN's own initial latents before the first
            # sweep reads them, so the walk is strictly independent of X[mask]
            Xw = self.impute(Xw, Z, mask)

        i = 0
        for n in range(n_samp * burnin):
            out = self.EStep(Xw, S, Z, inplace=False, mask=mask, **args)
            samp = out[0] if mask is not None else out   # (ES, Ximp) when masking
            if not np.mod(n+1, burnin):
                samps[i] = 1 * samp
                i += 1

        if multi and not per_chain:
            return samps[:, self.best_chain()]           # (n_samp, len(X), dim_hid)
        return samps

    def grad_step(self, X, mask=None, learn_prior=True):
        # Imputation is the linear-Gaussian E-step's job; a model that inherits
        # this base step has no way to fill the holes, so refuse the mask rather
        # than silently fitting (and cross-validating) on the full data.
        if mask is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support masked (imputation) fits")
        newS = self.EStep(X, self.S)
        return newS, self.MStep(X, newS, learn_prior=learn_prior)


# ===========================================================================
#  Shared linear-Gaussian skeleton
# ===========================================================================

@dataclass
class LinearGaussianBMF(BMF):
    """X_hat = L(S) with Gaussian noise.  Holds an operator + a latent_prior and
    implements everything common: reconstruction, MSE loss, Gaussian loglik,
    sign-of-drive latent init, and the single templated E-step / M-step that drive
    every model here (dense, Procrustes, CP, reduced-rank and conv alike)."""

    dim_hid: int
    fit_intercept: bool = True
    operator: LinearOperator = None                                 # set by subclass
    latent_prior: LatentPrior = field(default_factory=LatentPrior)  # the latent side
    m_iters: int = 1                             # M-step updates per grad_step
    debug: bool = False                          # record per-iteration E-step log-odds

    saem: bool = False                           # smooth the sufficient statistic
    gamma: float = 1.0                           # SAEM step size

    # n_chains == 1 (default) is the ordinary serial model with NO chain axis.
    # n_chains > 1 composes the chain-batched operator + prior (a leading axis C on
    # S / W / sigma_x / loss), shares one annealing schedule across chains, and lets
    # best_chain / collapse pick the winner.
    n_chains: int = 1

    @property
    def _multi(self):
        return self.n_chains > 1

    @property
    def S(self):
        """The binary spike (the latent state lives on the prior)."""
        return self.latent_prior.S

    # ---- reconstruction / loss / likelihood -------------------------------
    # The PUBLIC single-model faces -- __call__, loglikelihood, sample -- present
    # the BEST chain when multi, so downstream code that treats the model as one
    # winner works whether it was fit with 1 chain or many.  `loss` is the one
    # exception: it stays PER-CHAIN, which is what best_chain ranks on.
    def __call__(self, S):
        return (self.operator if not self._multi else self._best_operator()).forward(S)

    def _best_operator(self):
        """The winning chain as a serial operator (cached)."""
        b = self.best_chain()
        if getattr(self, '_best_op', None) is None or self._best_op_c != b:
            self._best_op, self._best_op_c = self.operator.to_serial(b), b
        return self._best_op

    def loss(self, X, mask=None):
        """MSE reconstruction loss: a scalar for one chain, per-chain (C,) for many.
        A boolean `mask` (matching X's trailing shape) scores only those entries.
        The reduction spans every axis but the chain axis, so 2-D data (n, d) and
        3-D data (n, t, d) both work."""
        sq = (X - self.operator.forward(self.S)) ** 2      # PER-CHAIN when multi
        if mask is not None:
            mask = np.asarray(mask)
            sq = sq.reshape(sq.shape[:-mask.ndim] + (-1,))[..., mask.ravel()]
        return sq.mean(axis=tuple(range(1, sq.ndim))) if self._multi else sq.mean()

    def loglikelihood(self, X, Xhat):
        """Per-element Gaussian log-likelihood.  `Xhat` is a single-model
        reconstruction (from __call__, i.e. the best chain when multi), so this uses
        that chain's scalar noise variance."""
        sig = (self.sigma_x if not self._multi
               else float(np.atleast_1d(self.sigma_x)[self.best_chain()]))
        return -(0.5 * ((Xhat - X) ** 2) / sig + 0.5 * np.log(2 * np.pi * sig))

    def ppll(self, X, mask=None, n_samp=1, **samp_args):
        """Posterior-predictive log-likelihood, per element: score X under the
        Monte-Carlo posterior-predictive MEAN reconstruction E_S[forward(S)].  With
        a boolean `mask` the masked entries are imputed along the sampling chain, so
        the latents condition on an imputed X while the loglik is scored against the
        ORIGINAL X -- held-out scores are loglik[..., mask]."""
        samps = self.sample(X, n_samp=n_samp, mask=mask, **samp_args)
        return self.loglikelihood(X, self(samps).mean(0))

    # ---- variational log-evidence (what ranks the chains) -----------------
    #
    # The fit leaves each chain with its own decoder, its own noise variance and
    # (with a structured prior) its own coupling, so a reconstruction error does
    # not compare them: it ignores how much of the fit was bought with prior
    # parameters, and it ignores how sharply the posterior is peaked.  The score
    # below is the ordinary mean-field bound
    #
    #     log p(X) >= E_q[log p(X | S)] + E_q[log p(S)] + H[q],   q = prod Bern(P)
    #
    # with q obtained by iterating the model's OWN conditionals -- the same
    # log-odds the search samples from, mirrored in numpy (link plugin from the
    # prior, drive/gram from the operator) -- to a fixed point at temperature 1.
    # Caveats, both documented where they arise: the tree/StS regularizer is not
    # a potential and is left out of log p(S) (see LatentPrior._energy_params),
    # and with slab=True the magnitudes enter at their conditional moments rather
    # than through a variational factor of their own, so the value is an
    # approximate evidence rather than a strict bound.

    def _sigma2(self):
        """The observation variance, broadcast over a (..., n, m) latent array."""
        s2 = np.asarray(self.sigma_x, dtype=float)
        return s2[..., None, None] if self._multi else float(s2)

    def _field(self, Zbar, drive):
        """(E, gjj): the leave-one-out likelihood field
        E_ij = drive_ij - sum_{k != j} G_jk Zbar_ik and the gram diagonal -- the
        vectorized mirror of the kernel's inner loop, for all (i, j) at once."""
        G = self.operator.gram()
        d = np.arange(G.shape[-1])
        gjj = G[..., d, d][..., None, :]
        return drive - Zbar @ G + Zbar * gjj, gjj

    def logodds(self, X, S=None):
        """The full conditional log-odds of every spike given all the others,
        (..., n, m): the likelihood link plus the prior's additive term, exactly
        what the search samples from at temperature 1.  `S` defaults to the
        model's current spike; a mean-field P may be passed in its place (every
        term is linear in the latent, so that gives E_q of the same quantity)."""
        lp = self.latent_prior
        S = lp.S if S is None else S
        Zbar = S if not lp.slab else S * self._magnitudes(X, S)
        E, gjj = self._field(Zbar, self.operator.drive(X))
        return lp.link_moments(E, gjj, self._sigma2())[0] + lp.logodds(S)

    def _magnitudes(self, X, S):
        """E[Z | spike on] at the current state -- the slab's conditional mean."""
        E, gjj = self._field(S * 0.0, self.operator.drive(X))
        return self.latent_prior.link_moments(E, gjj, self._sigma2())[1]

    def mean_field(self, X, n_iter=25, damp=0.5):
        """q(S) = prod_ij Bernoulli(P_ij), from the model's own conditional
        log-odds at temperature 1, iterated (damped) to a fixed point.  Returns
        (P, m1, m2, gjj): the spike probabilities, the first two moments of the
        magnitude given a spike (1 for a binary link) and the gram diagonal."""
        lp = self.latent_prior
        drive = self.operator.drive(X)
        # start from the fit's own mode when it is the right size, else from the
        # same sign-of-drive rule that seeds a fresh chain
        P = (1.0 * lp.S if np.shape(lp.S)[-2] == drive.shape[-2]
             else 1.0 * (drive >= 0))
        m1 = np.ones(P.shape)
        m2 = m1
        for _ in range(n_iter):
            E, gjj = self._field(P * m1, drive)
            ll, m1, m2 = lp.link_moments(E, gjj, self._sigma2())
            P = damp * P + (1 - damp) * _sigmoid(ll + lp.logodds(P))
        return P, m1, m2, gjj

    def elbo(self, X, mask=None, n_iter=25, damp=0.5):
        """Variational lower bound on log p(X) in nats: a scalar for one chain,
        per-chain (C,) for many.  A boolean `mask` scores only those entries (the
        same convention as `loss`).  Higher is better -- this is what best_chain
        ranks on."""
        P, m1, m2, gjj = self.mean_field(X, n_iter=n_iter, damp=damp)
        Zbar = P * m1
        var = P * m2 - Zbar ** 2                  # Var_q[S * Z], elementwise
        s2 = np.asarray(self.sigma_x, dtype=float)

        sq = (X - self.operator.forward(Zbar)) ** 2
        keep = 1.0
        if mask is not None:
            mask = np.asarray(mask)
            sq = sq.reshape(sq.shape[:-mask.ndim] + (-1,))[..., mask.ravel()]
            keep = float(mask.mean())
        n_obs = sq[0].size if self._multi else sq.size
        sse = sq.sum(axis=tuple(range(1, sq.ndim))) if self._multi else sq.sum()
        # the latent variance spreads over every feature, so the masked version
        # is scaled by the observed fraction rather than dropped entry by entry
        sse = sse + keep * (var * gjj).sum((-2, -1))

        data = -0.5 * sse / s2 - 0.5 * n_obs * np.log(2 * np.pi * s2)
        out = (data + self.latent_prior.log_prob(P, m1)
               + _bern_entropy(P).sum((-2, -1)))
        return out if self._multi else float(out)

    def _chain_score(self, X, mask=None, **args):
        return np.asarray(self.elbo(X, mask=mask, **args))

    # ---- initialization ---------------------------------------------------
    def init_params(self, X, hot_start=True, scl_lr=0, **opt_args):
        # Only the row count is used downstream (the E-step's StS prior), so len(X)
        # serves 2-D (n, d) and 3-D (n, t, d) data alike.
        self.n = len(X)
        self.sigma_x = np.ones(self.n_chains) if self._multi else 1
        self.scl_lr = scl_lr                      # lr of the noise-variance update
        self.outs = []                            # per-iteration log-odds if debug
        self._Ximp = None                         # per-chain imputation copy (lazy)
        self._best_op = None                      # cached best-chain operator (lazy)

        self.operator.init_params(X, self.dim_hid, hot_start=hot_start, **opt_args)
        # Compose the search: the operator's scaffold x the prior's link (which IS
        # the slab-vs-no-slab choice) x the prior's additive S-prior.  Compiled once.
        self.operator.build_search(self.latent_prior.link,
                                   self.latent_prior.prior_plugin, debug=self.debug)

    def init_latents(self, X, **args):
        self.latent_prior.init_latents(self.operator.drive(X))
        # SAEM: seed the running expected sufficient statistic one gamma-step from
        # the uniform prior mean (0.5) toward the initial effective latent, so
        # gamma == 1 hands the first M-step exactly what stochastic EM would.
        if self.saem:
            self.ES = 0.5 + self.gamma * (self.latent_prior.Z - 0.5)

    # ---- the one E-step that serves every linear-Gaussian model -----------
    # The latent component owns the spike S, the effective latent Z (== S with no
    # slab, the magnitudes when slab=True), the StS bookkeeping and the regularizer
    # scalars; the operator owns the search, already compiled with the prior's link.
    # The E-step operates on the SUPPLIED (S, Z) -- grad_step passes the prior's
    # persistent latents, `sample` a fresh chain shaped to the rows of X -- and
    # updates both in place.
    def EStep(self, X, S, Z=None, mask=None, inplace=True, **kwargs):
        
        if Z is None:
            Z = 1.0 * S           # the effective latent of a no-slab prior, and a
                                  # shape-matched seed the slab overwrites
        lp = self.latent_prior
        Jc, hc = lp.coupling()                    # additive S-prior (zeros if none)
        out = np.zeros(S.shape) if self.debug else None
        # A schedule may return a scalar (shared) or a per-chain array; coerce to
        # what the compiled kernel expects.
        prior_temp = np.asarray(lp.temp, dtype=float)
        prior_temp = (np.ascontiguousarray(np.broadcast_to(prior_temp, (self.n_chains,)))
                      if self._multi else float(prior_temp))

        self.operator.search(
            self.operator.drive(X), S, Z, self.operator.gram(), lp.StS, self.n,
            self.temp, lp.sparse_reg, lp.tree_reg, lp.slab_prior, self.sigma_x,
            Jc, hc, inplace, out, prior_temp)
        if self.debug and inplace:
            self.outs.append(out)

        ES = Z
        # SAEM: fold the fresh draw into the running sufficient statistic and hand
        # the smoothed estimate downstream.  Guarded by `inplace` so `sample` never
        # touches the model-bound ES.
        if self.saem and inplace:
            self.ES += self.gamma * (ES - self.ES)
            ES = self.ES
        return ES if mask is None else (ES, self.impute(X, ES, mask))

    def impute(self, X, ES, mask):
        """X[mask] <- forward(ES), the conditional MEAN of p(X | latents) (EM-style
        imputation -- no observation noise is added).  Mutates X in place and
        returns it.  Multi-chain: X is the per-chain working copy and each chain
        fills its own holes (the mask indexes the trailing axes)."""
        idx = (slice(None),) * (X.ndim - np.ndim(mask)) + (mask,)
        X[idx] = self.operator.forward(ES)[idx]
        return X

    def _latent_sweep(self, X, mask=None):
        lp = self.latent_prior
        return self.EStep(X, lp.S, lp.Z)

    # ---- the fit loop operates on the prior's persistent latents ----------
    def grad_step(self, X, mask=None, learn_prior=True):
        lp = self.latent_prior
        if mask is not None and self._multi:
            # multi-chain imputation: a PER-CHAIN copy of the data, so chains impute
            # independently (observed entries stay at the true X, masked ones are
            # refined each sweep).  Seeded once, broadcast over chains.
            if self._Ximp is None:
                self._Ximp = np.repeat(np.asarray(X, float)[None], self.n_chains, 0)
            X = self._Ximp
        if mask is None:
            ES = self.EStep(X, lp.S, lp.Z)
        else:
            ES, X = self.EStep(X, lp.S, lp.Z, mask=mask)   # X imputed in place
        return ES, self.MStep(X, ES, lp.S, learn_prior=learn_prior)

    # ---- the one M-step that serves every linear-Gaussian model -----------
    def MStep(self, X, ES, S=None, learn_prior=True):
        """One update of every parameter, then the reconstruction energy.

        `ES` is the EFFECTIVE latent the operator reconstructs from (continuous
        under a slab); `S` is the BINARY spike the prior learns from -- they differ
        exactly when slab=True, and passing the spike is what keeps the coupling an
        Ising model over binary codes.  It defaults to the prior's current spike.

        The operator's `backward` does one in-place update of its own parameters and
        the prior's `learn` one update of its own; they factor because the Gaussian
        loss is separable in (W, J).  The noise variance is the third, independent
        update: a SCALE-FREE partial step in log space, driven by the ratio of the
        residual energy to the current variance (so it is invariant to the data's
        overall scale)."""
        if S is None:
            S = self.latent_prior.S

        for _ in range(self.m_iters):
            resid = self.operator.backward(ES, X)        # operator parameter update
            if learn_prior:
                self.latent_prior.learn(S)               # prior parameter update
        sq = resid ** 2
        err = sq.mean(axis=tuple(range(1, sq.ndim))) if self._multi else float(sq.mean())

        drive = np.clip(err / np.maximum(self.sigma_x, 1e-12) - 1, -5.0, 5.0)
        self.sigma_x = self.sigma_x * np.exp(self.scl_lr * drive)
        return err

    def _scale_lrs(self, factor):
        self.scl_lr *= factor
        self.operator.scale_lrs(factor)

    # ---- multi-chain: pick / extract the winning chain --------------------
    def best_chain(self, X=None, mask=None):
        """Index of the winning chain (0 for a single chain): the one with the
        highest variational log-evidence (see `elbo`), NOT the lowest MSE -- the
        chains differ in their prior parameters and their noise variance, so a
        raw reconstruction error does not compare them.  Uses the score cached at
        the end of `fit` unless X is given; pass the observed-entry `mask` when
        imputing, to rank chains on the observed data only."""
        if not self._multi:
            return 0
        return int(np.argmax(self.chain_score if X is None
                             else self._chain_score(X, mask=mask)))

    def reconstruct_best(self, X=None):
        """The winning chain's forward(S), with the chain axis dropped."""
        N = self.operator.forward(self.S)
        return N if not self._multi else N[self.best_chain(X)]

    def collapse(self, X=None):
        """Reduce a multi-chain model to its winning chain: swap in that chain's
        SERIAL operator + prior and 2-D state, so it becomes an ordinary
        single-chain model.  No-op for a single chain.  Returns self."""
        if not self._multi:
            return self
        c = self.best_chain(X)
        self.operator = self.operator.to_serial(c)
        self.latent_prior = self.latent_prior.to_serial(c)
        self.sigma_x = float(np.atleast_1d(self.sigma_x)[c])
        self.n_chains = 1
        self._Ximp = None
        self.operator.build_search(self.latent_prior.link,
                                   self.latent_prior.prior_plugin, debug=self.debug)
        return self

    # ---- guards for components with no chain-batched port -----------------
    def _no_multichain(self, name):
        if self._multi:
            raise NotImplementedError(
                f"{name} has no multi-chain (n_chains > 1) port yet: its operator is "
                f"not chain-batched.  Use n_chains=1, or run several fits and keep "
                f"the best (bae_util.multifit).")


# ===========================================================================
#  Concrete models  (the operator choice + the prior choice, nothing else)
# ===========================================================================
#
# `J_prior` picks the latent side for every model below:
#   'none'      -- independent Bernoulli (LatentPrior).
#   'boltzmann' -- the Ising prior with a continuous coupling fit by SGD on a
#                  pseudolikelihood (or by MLE); J_lr == 0 falls back to 'none',
#                  so the historical J_lr-only interface still works.
#   'mrf'       -- the SIGN-CONSTRAINED Ising prior, coupling entries in {-1,0,+1}
#                  resampled by an annealed Gibbs/ICM sweep instead of descended.
#                  It has no learning rate, so J_lr does not gate it.
# The J* fields below are the structured prior's constructor arguments; they are
# inert under J_prior='none'.

@dataclass
class SemiBMF(LinearGaussianBMF):
    """AffineOperator (L(S) = S W^T + b, numpy) + any latent prior.  The operator's
    `backward` is the gradient step (reconstruction VJP + its weight penalty);
    everything else is inherited, so the model body is the assembly below.
    JBMF is this class with J_prior='boltzmann'."""

    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l2_reg: float = 1e-2
    weight_l1_reg: float = 0.0
    slab: bool = False
    slab_prior: float = 1.0
    J_prior: str = 'none'
    J_l1_reg: Optional[float] = 2e-4
    J_loss: str = 'rple'
    J_lr: float = 1e-2

    def __post_init__(self):
        super().__post_init__()
        self.operator = AffineOperator(n_chains=self.n_chains,
                                       fit_intercept=self.fit_intercept,
                                       nonneg=self.nonneg,
                                       pr_reg=self.weight_pr_reg,
                                       l1_reg=self.weight_l1_reg,
                                       l2_reg=self.weight_l2_reg)
        self.latent_prior = _boltzmann_or_plain(self)


@dataclass
class JBMF(SemiBMF):
    """SemiBMF with the structured (Ising) prior switched on: the coupling and its
    pseudolikelihood step live entirely in the prior, so this is a default flip."""

    J_prior: str = 'boltzmann'


@dataclass
class BiPCA(LinearGaussianBMF):
    """Binary PCA: a Procrustes operator (orthonormal W with one learned scale,
    Gaussian observations) + any latent prior.  The M-step follows
    minimal_structured_bipca: a relaxed polar update of W on centered data, a
    log-space update of the scale, and the closed-form intercept."""

    sparse_reg: float = 1e-2
    tree_reg: float = 0
    fit_intercept: bool = True
    fit_scl: bool = True
    W_lr: float = 0.25           # relaxation of the orthogonal-Procrustes update
    scale_lr: float = 0.10       # log-space relaxation of the decoder scale
    slab: bool = False
    slab_prior: float = 1.0

    J_prior: str = 'boltzmann'
    J_l1_reg: Optional[float] = 2e-4
    J_loss: str = 'rple'
    J_lr: float = 0
    J_beta: float = 1.0          # 'mrf' only: inverse temperature = the SCALE of
                                 # the {-1,0,+1} coupling (fit and applied).
    J_temp: float = 0.0          # 'mrf' only: sampler temperature over J (<=0 = ICM)
    J_sweeps: int = 1            # 'mrf' only: Gibbs sweeps over J per M-step

    def __post_init__(self):
        super().__post_init__()
        self.operator = Procrustes(n_chains=self.n_chains,
                                   fit_intercept=self.fit_intercept,
                                   fit_scl=self.fit_scl,
                                   W_lr=self.W_lr, scale_lr=self.scale_lr)
        common = dict(n_chains=self.n_chains, sparse_reg=self.sparse_reg,
                      tree_reg=self.tree_reg, slab=self.slab,
                      slab_prior=self.slab_prior)
        if self.J_prior == 'mrf':
            # MRFPrior's sparsity knob is l0_reg: the same role and default scale as
            # J_l1_reg, but a per-entry cost in nats rather than an L1 strength.
            self.latent_prior = make_prior('mrf', **common,
                                           l0_reg=self.J_l1_reg,
                                           beta_init=self.J_beta,
                                           beta_lr=self.J_lr,
                                           J_temp=self.J_temp,
                                           J_sweeps=self.J_sweeps)
        else:
            self.latent_prior = _boltzmann_or_plain(self)


@dataclass
class SCPD(LinearGaussianBMF):
    """Sparse canonical-polyadic decomposition: a CPOperator
    (L(S) = einsum('ck,tk,nk->ctn', S, U, V) + b over a 3-D tensor X of shape
    (n, t, d)) + any latent prior, with the shared autograd M-step.  Shares the
    entire E-step with RRBMF; only the operator's drive/gram/backward differ.
    `dim_hid` is the CP rank.  JSCPD is this class with J_prior='boltzmann'."""

    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l1_reg: float = 1e-2
    slab: bool = False
    slab_prior: float = 1.0
    J_prior: str = 'none'
    J_l1_reg: Optional[float] = 2e-4
    J_loss: str = 'rple'
    J_lr: float = 1e-2

    def __post_init__(self):
        super().__post_init__()
        self.operator = CPOperator(n_chains=self.n_chains,
                                   fit_intercept=self.fit_intercept,
                                   nonneg=self.nonneg,
                                   pr_reg=self.weight_pr_reg,
                                   l1_reg=self.weight_l1_reg)
        self.latent_prior = _boltzmann_or_plain(self)


@dataclass
class JSCPD(SCPD):
    """SCPD with the structured (Ising) prior switched on."""

    J_prior: str = 'boltzmann'


@dataclass
class RRBMF(LinearGaussianBMF):
    """Reduced-rank BMF: a ReducedRankOp (L(S) = einsum('ck,knt->ctn', S, V@U.T) + b
    over a 3-D tensor X of shape (n, t, d)) + any latent prior, with the shared
    autograd M-step.  Shares the entire E-step with SCPD.  JRRBMF is this class
    with J_prior='boltzmann'."""

    rank: int = None
    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l1_reg: float = 0.0
    weight_l2_reg: float = 1e-2
    slab: bool = False
    slab_prior: float = 1.0
    J_prior: str = 'none'
    J_l1_reg: Optional[float] = 2e-4
    J_loss: str = 'rple'
    J_lr: float = 1e-2

    def __post_init__(self):
        super().__post_init__()
        self.operator = ReducedRankOp(n_chains=self.n_chains,
                                      rank=self.rank,
                                      fit_intercept=self.fit_intercept,
                                      nonneg=self.nonneg,
                                      pr_reg=self.weight_pr_reg,
                                      l1_reg=self.weight_l1_reg,
                                      l2_reg=self.weight_l2_reg)
        self.latent_prior = _boltzmann_or_plain(self)


@dataclass
class JRRBMF(RRBMF):
    """RRBMF with the structured (Ising) prior switched on."""

    J_prior: str = 'boltzmann'


@dataclass
class ConvBMF(LinearGaussianBMF):
    """Convolutional operator: the same linear-Gaussian template in a
    translation-invariant space, so it plugs in a ConvOperator (which carries the
    convbmf search) and otherwise inherits everything."""

    kernel_size: int = None

    def __post_init__(self):
        super().__post_init__()
        self._no_multichain("ConvBMF")   # ConvOperator has no chain-batched port

    def init_params(self, X, **opt_args):
        self.operator = ConvOperator()
        # init GP kernels K, b + operator.optimizer (old_bae_models.ConvBMF:1231)
        ...


def _boltzmann_or_plain(model):
    """The latent prior for a model exposing the standard J* fields: a
    BoltzmannPrior when asked for and J_lr > 0, an unstructured LatentPrior
    otherwise (so the historical J_lr == 0 fallback still works)."""
    common = dict(n_chains=model.n_chains, sparse_reg=model.sparse_reg,
                  tree_reg=model.tree_reg, slab=model.slab,
                  slab_prior=model.slab_prior)
    if model.J_prior == 'none' or model.J_lr == 0:
        return make_prior('none', **common)
    return make_prior('boltzmann', **common, J_l1_reg=model.J_l1_reg,
                      J_loss=model.J_loss, J_lr=model.J_lr)


# ===========================================================================
#  Kernel factorization  (the one model that is NOT linear-Gaussian)
# ===========================================================================

@dataclass
class KernelBMF(BMF):
    """Gram/kernel factorization: K ~ center(S diag(scl) S^T).

    The one model here that does not fit the LinearGaussianBMF skeleton -- the
    reconstruction is QUADRATIC in S rather than an affine map, so there is no
    LinearOperator (no forward / drive / gram) and it subclasses BMF directly,
    owning its own diagonal scale `scl` and the running StX = S^T X.

    It still uses the LatentPrior structure unchanged: the prior owns S / StS / Z
    and supplies the SAME additive S-prior plugin the dense models use, so dropping
    in a BoltzmannPrior gives this a structured Ising prior for free.  Only the
    E-step's likelihood field (bae_search.make_kernel_search) and the scale
    M-step are kernel-specific.

    uniform_scale / l1_reg shape the scale M-step (uniform closed form vs projected
    gradient), not the E-step, which is always per-feature.  `kernel_input=True`
    takes a precomputed kernel instead of a feature matrix.  A slab prior has no
    meaning here (no per-element magnitude), so slab is always False."""

    dim_hid: int
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    uniform_scale: bool = True
    l1_reg: float = 0.0             # only used by the non-uniform scale M-step
    kernel_input: bool = False

    J_prior: str = 'none'
    J_l1_reg: Optional[float] = 2e-4
    J_loss: str = 'rple'
    J_lr: float = 0

    debug: bool = False             # record per-iteration E-step log-odds
    n_chains: int = 1               # single-chain; kept for the shared
                                    # prior/probe plumbing that reads it

    def __post_init__(self):
        super().__post_init__()
        self.slab, self.slab_prior = False, 1.0
        self.latent_prior = _boltzmann_or_plain(self)

    @property
    def S(self):
        return self.latent_prior.S

    # ---- params: the diagonal scale + data norm; compile the kernel search --
    def init_params(self, X, lr=1, **opt_args):
        self.n, self.d = X.shape
        self.lr = lr
        self.scl = np.ones(self.dim_hid)             # diagonal D (always a vector)
        if self.kernel_input:
            X_ = _center_kernel(X)
            self.data_norm = np.sum(X_ ** 2)
        else:
            X_ = X - X.mean(0)
            self.data_norm = np.sum((X_.T @ X_) ** 2)
        # sigma2 = mean squared (centered) kernel entry = data_norm / N^2, so the
        # E-step log-odds is O(1) regardless of N, d and input scale.  Fixed at the
        # data scale -- annealing is temp's job, not sigma2's.
        self.sigma2 = self.data_norm / self.n ** 2
        self._search = bae_search.make_kernel_search(
            self.latent_prior.prior_plugin, kernel_input=self.kernel_input,
            debug=self.debug)
        self._dummy_out = np.zeros((1, 1))           # unused by the sampling path
        self.outs = []

    # ---- latents: the prior owns S/StS/Z; StX is the model's likelihood state
    def init_latents(self, X, **kwargs):
        if self.kernel_input:
            l, V = np.linalg.eigh(X)
            Mx = np.diag(np.sqrt(l + 1e-6)) @ V @ np.random.randn(self.d, self.dim_hid)
        else:
            Mx = X @ np.random.randn(self.d, self.dim_hid)
        # Standardize the random projection before thresholding so the init coding
        # level is SCALE-INVARIANT (a raw 0.5 offset collapses to all-zero on small
        # inputs and to ~dense on large ones).
        Mx = Mx / (Mx.std() + 1e-12)
        self.latent_prior.init_latents(Mx - 0.5)
        self.StX = self.S.T @ X                      # running S^T X (kernel state)

        # Scale-match scl to the init S (the uniform M-step target) instead of 1:
        # with the sigma2-normalized E-step the self-energy term is ~scl^2/sigma2, so
        # an off-scale scl collapses S on the first sweep, before the M-step can
        # correct it.  This target scales WITH the data.
        lp = self.latent_prior
        StS_c = lp.StS - np.outer(np.diag(lp.StS), np.diag(lp.StS)) / self.n
        if self.kernel_input:
            S_ = self.S - self.S.mean(0)
            V_dot = np.diag(S_.T @ X @ S_)
        else:
            StX_c = self.StX - np.outer(np.diag(lp.StS), X.mean(0))
            V_dot = np.sum(StX_c ** 2, axis=1)
        denom = np.sum(StS_c ** 2)
        self.scl = np.full(self.dim_hid,
                           max(0.0, np.sum(V_dot) / denom) if denom > 0 else 1.0)

    def __call__(self, S):
        return np.einsum('...ik,k,...jk->...ij', S, self.scl, S)

    # ---- E-step: the shared kernel scaffold with the prior's plugin --------
    def EStep(self, X, S, inplace=True):
        lp = self.latent_prior
        Jc, hc = lp.coupling()                       # zeros for a plain prior
        if self.kernel_input:
            data, StX = _center_kernel(X), np.zeros((self.dim_hid, 0))
        else:
            data, StX = X, self.StX
        out = np.zeros(S.shape) if self.debug else self._dummy_out
        self._search(data, S, lp.StS, StX, self.n, self.scl, self.sigma2, self.temp,
                     lp.sparse_reg, lp.tree_reg, Jc, hc, out, inplace,
                     float(np.asarray(lp.temp)))
        if self.debug and inplace:
            self.outs.append(out)
        return S

    # BMF.sample threads a Z through EStep for the slab models; the kernel E-step
    # reads only the binary spike, so walk an S-only chain here.
    def sample(self, X, temp=None, n_samp=1, burnin=10, **args):
        if temp is not None:
            self.temp = temp
        samps = np.zeros((n_samp, len(X), self.dim_hid))
        S = 1.0 * np.random.choice([0, 1], size=(len(X), self.dim_hid))
        i = 0
        for n in range(n_samp * burnin):
            samp = self.EStep(X, S, inplace=False, **args)
            if not np.mod(n+1, burnin):
                samps[i] = 1 * samp
                i += 1
        return samps

    def _scale_lrs(self, factor):
        self.lr *= factor

    # ---- M-step: optimally scale D (+ the prior's own update) --------------
    def MStep(self, X, ES, S=None, learn_prior=True):
        """Update the diagonal scale scl: uniform closed form when uniform_scale,
        else a projected-gradient step with L1 and non-negativity.  The prior's
        `learn` runs alongside, on the BINARY spike (which is what ES is here --
        the kernel model has no slab)."""
        if S is None:
            S = self.latent_prior.S
        lp = self.latent_prior
        StS = lp.StS - np.outer(np.diag(lp.StS), np.diag(lp.StS)) / len(X)
        StX = self.StX - np.outer(np.diag(lp.StS), X.mean(0))

        if self.kernel_input:
            S_ = ES - ES.mean(0)
            V_dot = np.diag(S_.T @ X @ S_)
        else:
            V_dot = np.sum(StX ** 2, axis=1)
        G = StS ** 2

        if self.uniform_scale:
            target = np.full(self.dim_hid, max(0.0, np.sum(V_dot) / np.sum(G)))
        else:
            grad = G @ self.scl - V_dot + self.l1_reg
            eta = np.dot(grad, grad) / (np.dot(grad, G @ grad) + 1e-8)
            target = np.maximum(0.0, self.scl - eta * grad)
        self.scl += self.lr * (target - self.scl)

        if learn_prior:
            lp.learn(S)

        Qnrm = self.scl @ G @ self.scl
        return 1 + (Qnrm - 2 * np.sum(self.scl * V_dot)) / self.data_norm

    def loss(self, X):
        lp = self.latent_prior
        S = lp.S
        StS = lp.StS - np.outer(np.diag(lp.StS), np.diag(lp.StS)) / len(X)
        if self.kernel_input:
            Kc = _center_kernel(X)
            V_dot, Knrm = np.diag(S.T @ Kc @ S), np.sum(Kc ** 2)
        else:
            X_ = X - X.mean(0)
            V_dot, Knrm = np.sum((S.T @ X) ** 2, axis=1), np.sum(X_.T @ X_ ** 2)
        dot = np.sum(self.scl * V_dot)
        Qnrm = self.scl @ (StS ** 2) @ self.scl
        return (Qnrm + Knrm - 2 * dot) / Knrm
