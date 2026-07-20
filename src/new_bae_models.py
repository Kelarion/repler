"""
new_bae_models.py  --  ILLUSTRATION ONLY (design sketch, partially ported)
==========================================================================

A refactor of `bae_models.py` that pulls the structure shared by every BMF
model into a few composable pieces, so that a change to the shared behaviour
is made *once* instead of being copy-pasted into ~10 classes.

The machinery is split across three modules:

  new_bae_priors.py   the latent side (one slot: latent_prior)
      LatentPrior      owns the latent state (S, StS, Z), its init, the
                       sparsity/tree regularizers, the recurrence (`couple`) and
                       the search `link`; `slab=True` makes it spike-and-slab
      BoltzmannPrior   structured (Ising) prior -- the `J` from JBMF, now
                       droppable onto any model (and slab=True works on it too)

  new_bae_weights.py  the weight side
      LinearOperator   dataclass base: shared structural fields (fit_intercept /
                       nonneg / resample_dead), L's faces (forward/drive/gram),
                       the search kernel, and `backward` (one M-step update).
                       Each subclass adds its own learning-rate + weight-
                       regularization fields and implements the reg inline:
        AffineOperator   L(S) = S W^T + b             (numpy)
        TorchMatrixOp    L(S) = W(S)                  (nn.Linear / autograd)
        CPOperator       L(S) = einsum CP             (SCPD)
        ReducedRankOp    L(S) = reduced-rank tensor   (RRBMF)
        ConvOperator     L(S) = conv1d(S, K)          (ConvBMF)

  new_bae_models.py   (this file) the models
      BMF                annealing fit loop + Gibbs sampling
      LinearGaussianBMF  X_hat = L(S) + b with Gaussian likelihood: holds an
                         operator + a latent_prior, and implements the *one*
                         E-step / M-step / init that serve every model below.

Every linear-Gaussian model is then just its __post_init__ -- the assembly of an
operator + a latent_prior.  init_params, EStep, MStep, __call__, loss,
loglikelihood and init_latents are ALL inherited; nothing model-specific is left.
To recombine: pick an operator (AffineOperator / TorchMatrixOp / CP / ReducedRank
/ Conv) and a latent component (LatentPrior or BoltzmannPrior, each with
`slab=True/False` -- spike-and-slab is just a flag, not a separate slab).  The operator owns
its parameters, intercept, init, forward, search and `backward` (its one M-step
update); the model just orchestrates.

The E-step is identical once you have L's three faces:

    forward(S)   reconstruction          L(S)            (-> __call__)
    drive(X)     adjoint on residual     L*(X - b)       (-> XW)
    gram()       operator metric         L*L             (-> WtW)

(this is why RRBMF and SCPD literally share `bae_search.sbmf`).  Adding a
structured prior, or any new shared regularizer, to *all* of them means editing
one component, not ten classes.

KernelBMF2 is the exception that proves the rule: its reconstruction K ~
center(S diag(scl) S^T) is QUADRATIC in S, so it is NOT a LinearOperator and
subclasses BMF directly, carrying its own scale + a kernel-specific likelihood
field (new_bae_search.make_kernel_search).  But it STILL composes a latent_prior
unchanged -- reusing the shared additive S-prior plugin -- so a BoltzmannPrior
drops onto it for free.  This is what "use the LatentPrior structure without the
LinearGaussian framework" looks like.

STATUS: SemiBMF, SpikeNMF, RRBMF and KernelBMF2 are full ports, verified
bit-for-bit against the originals (test_semibmf_port.py, test_spikenmf_port.py,
test_rrbmf_port.py, test_kernelbmf2_port.py).  The other model bodies are still
illustrative stubs (`...`) with the original source lines referenced for a
mechanical port.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# this module is pure orchestration -- no torch / scipy here; all the numerics
# (params, init, forward/backward) live in the operators and priors it composes.
import new_bae_search
import new_bae_priors as nbp
from new_bae_priors import LatentPrior, BoltzmannPrior, MRFPrior
from new_bae_weights import (
    LinearOperator,
    AffineOperator, Procrustes, TorchMatrixOp, CPOperator, ReducedRankOp, ConvOperator,
)
# Every operator and prior now supports multiple chains through its own `n_chains`
# field (n_chains == 1 is the ordinary serial 2-D component; n_chains > 1 carries a
# leading chain axis).  There is no separate Parallel class -- a model just passes
# n_chains through to the single component class (see `_chained`).


def _center_kernel(K):
    """Double-centering of a Gram matrix (== util.center / util.center_kernel),
    inlined so this module stays scipy/torch-free (KernelBMF2 is the one model
    with its own model-side numerics, since it is not a LinearOperator)."""
    return (K - K.mean(-2, keepdims=True) - K.mean(-1, keepdims=True)
            + K.mean((-1, -2), keepdims=True))


# ===========================================================================
#  Training-time probes  (track any quantity over a fit, mirror of TempSchedule)
# ===========================================================================
#
# A `probe` names something to record once per fit iteration.  It is either
#   * a dotted attribute path (str): walked from the model, e.g. 'sigma_x',
#     'temp', 'latent_prior.temp', 'latent_prior.J_W', 'latent_prior.S',
#     'operator.W', 'operator.scl'; or
#   * a callable model -> value, for a derived quantity the model doesn't store
#     directly (e.g. lambda m: m.latent_prior.J_W.std()).
# `fit(..., probes=[...])` collects them into `self.history` = {name: [per-iter]},
# so nothing about the tracked set is baked into the model -- it is chosen per fit,
# exactly like the prior_schedule.  Values are SNAPSHOTTED (arrays copied) so the
# history holds each iteration's state, not a live alias to the array the fit keeps
# mutating in place.

def _snapshot(v):
    """Copy a probed value so history holds the state AT that iteration, not a live
    alias.  torch tensors / nn.Parameters -> detached numpy copies (duck-typed, so
    this module stays torch-free); numpy arrays -> copies; scalars pass through."""
    if hasattr(v, 'detach'):                       # torch tensor / nn.Parameter
        v = v.detach()
        v = v.cpu() if hasattr(v, 'cpu') else v
        return np.asarray(v).copy()
    if isinstance(v, np.ndarray):
        return v.copy()
    return v


def _probe_value(model, probe):
    """Evaluate one probe: a callable is called with the model; a string is a dotted
    attribute path walked from the model.  Result is snapshotted."""
    if callable(probe):
        return _snapshot(probe(model))
    obj = model
    for attr in probe.split('.'):
        obj = getattr(obj, attr)
    return _snapshot(obj)


# ===========================================================================
#  Base class: fit loop only (unchanged from bae_models.BMF)
# ===========================================================================

@dataclass
class BMF:
    """Annealing fit loop + Gibbs sampling.  Identical to bae_models.BMF."""

    def __post_init__(self):
        self.temp = 1
        self.initialized = False

    def initialize(self, X, **args):
        self.init_params(X, **args)
        self.init_latents(X, **args)
        self.initialized = True

    def fit(self, *data, initial_temp=10, decay_rate=0.88, period=10,
            min_temp=1, prior_schedule=None, max_iter=None, verbose=True,
            mask=None, probes=None, **opt_args):
        # The *prior* temperature (latent_prior.temp) runs on its OWN schedule,
        # decoupled from the model temp above: `prior_schedule` is any object with
        # `update(**inputs) -> float`, called once per iteration.  The default
        # (new_bae_priors.TempSchedule) returns 1.0 throughout, i.e. the structured
        # prior at face value; nbp.GeomAnneal(...) anneals it like the model temp,
        # nbp.ConstantTemp(t) pins it elsewhere.  Only the structured prior reads
        # it, and only when sampling S -- never when learning J -- so a schedule
        # here changes the search, not the fitted coupling.
        #
        # `probes` tracks any extra quantity over the fit (see _probe_value): a list
        # of dotted attribute paths and/or callables, or a {name: path-or-callable}
        # dict.  Each is recorded once per iteration (post-update) into `self.history`
        # = {name: [per-iter snapshots]}, e.g.
        #   probes=['sigma_x', 'latent_prior.temp', 'latent_prior.J_W']
        # No tracking is baked into the model -- it is chosen per fit, like the
        # prior_schedule -- and array values are snapshotted (copied) each iteration.

        if max_iter is None:
            max_iter = period * int(np.log(1e-4 / initial_temp) / np.log(decay_rate))

        if prior_schedule is None:
            prior_schedule = nbp.TempSchedule()
        sched_prior = hasattr(self, 'latent_prior')

        # normalize probes to a {name: probe} dict; a bare list keys each path/
        # callable by its own name (a callable's __name__, so lambdas want a dict).
        if probes is None:
            probes = {}
        elif not isinstance(probes, dict):
            probes = {(p if isinstance(p, str) else getattr(p, '__name__', repr(p))): p
                      for p in probes}
        self.history = {name: [] for name in probes}

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(range(max_iter))

        en = []
        if not self.initialized:
            self.initialize(*data, **opt_args)
        for it in range(max_iter):
            T = min_temp + initial_temp * (decay_rate ** (it // period))
            self.temp = T
            if sched_prior:
                self.latent_prior.temp = prior_schedule.update(
                    it=it, max_iter=max_iter, temp=T,
                    loss=en[-1] if en else None)
            # with a mask, grad_step imputes the masked entries in place (the
            # working array in `data` is refined across iterations).
            _, ls = self.grad_step(*data, mask=mask)
            en.append(ls)

            # record the tracked quantities for this iteration (post-update state)
            for name, probe in probes.items():
                self.history[name].append(_probe_value(self, probe))

            if verbose:
                pbar.update(1)

        # multi-chain: cache the final per-chain (C,) loss so best_chain / collapse
        # can pick the winner without recomputing.  Inert for a single chain.
        if getattr(self, 'n_chains', 1) > 1:
            self.chain_loss = np.asarray(en[-1])
        return en

    def sample(self, X, temp=None, n_samp=1, burnin=10, per_chain=False, mask=None, **args):
        # Fresh chain state, shaped to the rows of X (NOT the prior's persistent
        # S/Z, which are sized to the training set).  EStep updates S and Z in
        # place every sweep (inplace=False only freezes the prior's StS), so this
        # walks a single Gibbs chain from a random init; Z starts as a binary copy
        # and, for a slab, the search fills in continuous magnitudes as it sweeps.
        #
        # With n_chains > 1 every state carries a leading chain axis, so all chains
        # are sampled at once.  By DEFAULT only the best chain's draws are returned,
        # shape (n_samp, len(X), dim_hid) -- the same contract as a single-chain
        # model, so downstream code (e.g. new_bae_experiments.NewBMF, which scores
        # `sample` output and reconstructs `self(samps)`) works unchanged.  Pass
        # per_chain=True to keep every chain's draws, (n_samp, C, len(X), dim_hid)
        # -- needed when the caller wants each chain's samples (e.g. to score a
        # per-chain imputation).
        if temp is not None:
            self.temp = temp

        C = getattr(self, 'n_chains', 1)
        multi = C > 1
        if multi:
            samps = np.zeros((n_samp, C, len(X), self.dim_hid))
            S = 1.0 * np.random.choice([0, 1], size=(C, len(X), self.dim_hid))
        else:
            samps = np.zeros((n_samp, len(X), self.dim_hid))
            S = 1.0 * np.random.choice([0, 1], size=(len(X), self.dim_hid))
        Z = 1.0 * S

        # With a mask the chain conditions on an IMPUTED X: the masked entries are
        # refilled from the generative model each sweep (EStep -> impute), so the
        # latents never see the held-out values.  Work on a COPY (per-chain when
        # multi, matching impute's (C, ...) indexing) so the caller's X is never
        # mutated; the burn-in washes out the true values left in the seed copy.
        if mask is not None:
            Xw = (np.repeat(np.asarray(X, float)[None], C, 0) if multi
                  else 1.0 * np.asarray(X, float))
        else:
            Xw = X

        i = 0
        for n in range(n_samp * burnin):
            out = self.EStep(Xw, S, Z, inplace=False, mask=mask, **args)
            samp = out[0] if mask is not None else out   # (ES, Ximp) when masking
            if not np.mod(n, burnin):
                samps[i] = 1 * samp
                i += 1

        if multi and not per_chain:
            return samps[:, self.best_chain()]        # (n_samp, len(X), dim_hid)
        return samps

    def grad_step(self, X, mask=None):
        newS = self.EStep(X, self.S)
        loss = self.MStep(X, newS)
        return newS, loss

# ===========================================================================
#  Shared linear-Gaussian skeleton
# ===========================================================================
#
# Holds the operator + the two prior/penalty components and implements
# everything common: reconstruction, MSE loss, Gaussian loglik, sign-of-drive
# latent init, and the single templated E-step that drives every model here
# (dense, CP, reduced-rank and conv alike).

@dataclass
class LinearGaussianBMF(BMF):

    dim_hid: int
    fit_intercept: bool = True
    operator: LinearOperator = None                                 # set by subclass
    latent_prior: LatentPrior = field(default_factory=LatentPrior)  # the latent side
    m_iters: int = 1                             # M-step updates per grad_step
    debug: bool = False                          # record per-iteration E-step log-odds

    saem: bool = False
    gamma: float = 1.0

    kappa: float = 1e-3

    # Run n_chains independent chains at once and keep the best (the fit is
    # stochastic and non-convex).  n_chains == 1 (default) is the ordinary serial
    # model with NO chain axis -- every array is 2-D exactly as before.  When
    # n_chains > 1 the concrete model composes the chain-batched operator + prior
    # (a leading axis C on S / W / sigma_x / loss), the fit shares one annealing
    # schedule across chains, and best_chain / collapse pick the winner.  Only the
    # few shape-dependent methods below branch on it; EStep / init_latents are shared.
    n_chains: int = 1

    @property
    def _multi(self):
        return self.n_chains > 1

    # the latent state lives on the prior now; expose the spike for the base
    # BMF fit loop / loss (which read self.S).
    @property
    def S(self):
        return self.latent_prior.S

    # ---- shared reconstruction / loss / likelihood ------------------------
    # The PUBLIC single-model faces -- __call__ (reconstruction), loglikelihood and
    # sample -- present the BEST chain when multi, so downstream code that treats the
    # model as one winner (e.g. new_bae_experiments.NewBMF: self(samps),
    # loglikelihood(X, self(samps))) works whether it was fit with 1 chain or many.
    # `loss` is the one exception -- it stays PER-CHAIN so best_chain can rank them --
    # and the internal fit path uses operator/prior directly, never __call__.
    def __call__(self, S):
        if not self._multi:
            return self.operator.forward(S)
        return self._best_operator().forward(S)   # best chain's serial reconstruction

    def _best_operator(self):
        """The winning chain as a serial operator (cached), for best-chain forward."""
        b = self.best_chain()
        if getattr(self, '_best_op', None) is None or self._best_op_c != b:
            self._best_op, self._best_op_c = self.operator.to_serial(b), b
        return self._best_op

    def loss(self, X, mask=None):
        """MSE reconstruction loss.  Single chain -> scalar (unchanged); multi-chain
        -> per-chain (C,).  A boolean `mask` (matching X's shape) scores only the
        selected entries.  The reduction spans every axis but the chain axis, so it
        works for 2-D data (n,d) and 3-D data (n,t,d) alike."""
        N = self.operator.forward(self.S)          # PER-CHAIN when multi (not __call__)
        if not self._multi:
            if mask is None:
                mask = np.ones(X.shape) > 0
            return np.mean((X[mask] - N[mask]) ** 2)
        sq = (X[None] - N) ** 2                              # (C, ...)
        if mask is None:
            return sq.mean(axis=tuple(range(1, N.ndim)))
        return sq[:, mask].mean(1)

    def loglikelihood(self, X, Xhat):
        """Per-element Gaussian log-likelihood.  `Xhat` is a single-model
        reconstruction (from __call__, i.e. the best chain when multi), so this uses
        that chain's scalar noise variance -- the shapes match a serial model."""
        sig = self.sigma_x if not self._multi else \
            float(np.atleast_1d(self.sigma_x)[self.best_chain()])
        dot = 0.5 * ((Xhat - X) ** 2) / sig
        lnrm = 0.5 * np.log(sig) + 0.5 * np.log(2 * np.pi)
        return -(dot + lnrm)

    def ppll(self, X, mask=None, n_samp=1, **samp_args):
        """Posterior-predictive log-likelihood, per element.

        Draw n_samp latent samples from the posterior and score the Gaussian
        loglik of X under the Monte-Carlo posterior-predictive MEAN reconstruction
        Xhat = E_S[forward(S)] ~= mean over draws.  With a boolean `mask` (X's
        shape) the masked entries are imputed from the generative model along the
        sampling chain, so the latents condition on an imputed X while the loglik
        is scored against the ORIGINAL X -- held-out scores are loglik[..., mask],
        observed ones loglik[..., ~mask].  Returns loglikelihood's per-element
        shape (best chain when multi-chain)."""
        samps = self.sample(X, n_samp=n_samp, mask=mask, **samp_args)
        Xhat = self(samps).mean(0)          # MC posterior-predictive mean over draws
        return self.loglikelihood(X, Xhat)

    def init_latents(self, X, **args):
        self.latent_prior.init_latents(self.operator.drive(X))
        # SAEM: seed the running expected sufficient stat as one gamma-step from
        # the uniform prior mean (0.5) toward the initial effective latent, i.e.
        # ES_0 = 0.5 + gamma * (S0 - 0.5).  With gamma == 1 this is just S0, so
        # the first M-step sees exactly what plain stochastic EM would.  (Z holds
        # the effective latent for both slab and no-slab priors.)
        if self.saem:
            self.ES = 0.5 + self.gamma * (self.latent_prior.Z - 0.5)

    # ---- the one E-step that serves every linear-Gaussian model -----------
    # Fully compositional now: the latent component owns the spike S, the
    # effective latent Z (== S with no slab, the magnitudes when slab=True),
    # the StS bookkeeping and the regularizer/tau scalars; the operator owns the
    # search (already compiled with the prior's link).  The slab-vs-no-slab
    # branch is gone -- it was decided once, when the prior picked the link.
    # The E-step operates on the SUPPLIED (S, Z): the fit loop (grad_step) passes
    # the prior's *persistent* latents, while `sample` passes a fresh chain state
    # shaped to the rows of X.  Both are updated in place.  Z defaults to a binary
    # copy of S -- which is exactly the effective latent for a no-slab model, and a
    # shape-matched seed for a slab (the search overwrites it with the continuous
    # magnitudes on the first sweep) -- so the default always matches S's shape.
    # StS, by contrast, stays on the prior (frozen during sampling).
    def EStep(self, X, S, Z=None, mask=None, inplace=True, slab=True, **kwargs):

        if Z is None:
            Z = 1.0 * S

        lp = self.latent_prior
        XW = self.operator.drive(X)
        WtW = self.operator.gram()                    # TRUE gram (no prior folded in)
        Jc, hc = lp.coupling()                         # additive S-prior (zeros if none)
        # in debug mode the operator's search records each element's log-odds into
        # `out` (sized to S, so `sample`'s fresh chain works too); snapshot it per
        # fit sweep so self.outs[t] holds iteration t's current magnitudes.
        out = np.zeros(S.shape) if self.debug else None
        self.operator.search(
            XW, S, Z, WtW, lp.StS, self.n, self.temp,
            lp.sparse_reg, lp.tree_reg, lp.slab_prior, self.sigma_x, Jc, hc,
            inplace, out, lp.temp)
        if self.debug and inplace:
            self.outs.append(out)
        ES = Z if slab else S

        # SAEM: fold the fresh sample into the running expected sufficient stat
        # and hand the smoothed estimate downstream instead of the raw draw.
        # Guarded by `inplace` so `sample` (fresh inplace=False chain) never
        # touches the model-bound ES.  The mask/impute branch below then also
        # imputes from the smoothed ES, keeping the E-step self-consistent.
        if self.saem and inplace:
            self.ES += self.gamma * (ES - self.ES)
            ES = self.ES

        if mask is None:
            return ES

        return ES, self.impute(X, ES, mask)

    def impute(self, X, ES, mask):
        """X[mask] <- a sample from the generative model p(X | latents) =
        N(forward(ES), sigma_x).  Mutates X in place and returns it.  Multi-chain:
        X is the PER-CHAIN working copy (C, ...) and each chain fills the masked
        holes with its OWN reconstruction (the mask indexes the trailing axes)."""
        Xhat = self.operator.forward(ES)
        if not self._multi:
            X[mask] = Xhat[mask]
        else:
            X[:, mask] = Xhat[:, mask]
        return X

    # ---- the fit loop operates on the prior's persistent latents ----------
    # grad_step is the one caller that wants to advance the prior's *own* state:
    # it feeds the persistent spike S and the persistent effective latent Z (which
    # the slab maintains across sweeps as continuous magnitudes) so they are
    # updated in place.  `sample` is the other caller; it supplies a fresh (S, Z).
    def grad_step(self, X, mask=None):
        lp = self.latent_prior
        if mask is None:
            newES = self.EStep(X, lp.S, lp.Z)
            loss = self.MStep(X, newES)
        elif not self._multi:
            newES, X = self.EStep(X, lp.S, lp.Z, mask=mask)   # X imputed in place
            loss = self.MStep(X, newES)
        else:
            # multi-chain imputation: keep a PER-CHAIN copy of the data so chains
            # impute independently (observed entries stay at the true X; masked ones
            # are refined each sweep).  Seeded once, broadcast over chains.
            if self._Ximp is None:
                self._Ximp = np.repeat(np.asarray(X, float)[None], self.n_chains, 0)
            newES, self._Ximp = self.EStep(self._Ximp, lp.S, lp.Z, mask=mask)
            loss = self.MStep(self._Ximp, newES)
        return newES, loss

    # ---- the one M-step that serves every linear-Gaussian model -----------
    def MStep(self, X, S, Z=None):
        """Each operator's `backward` does one in-place update of *its* parameters
        (analytic-VJP gradient for numpy operators, autograd for torch ones, or a
        closed-form solve).  The prior's `learn` does one update of *its* params
        (e.g. JBMF's coupling J) right alongside, exactly as JBMF interleaves the
        W and J steps -- they factor because the Gaussian loss is separable in
        (W, J), so two SGD steps equal JBMF's one joint step.  sigma_x is the
        third independent update.  So this body is the same for SemiBMF / SpikeNMF
        / JBMF / SCPD / RRBMF / ConvBMF; only the components differ."""

        if Z is None:
            Z = 1*S

        for _ in range(self.m_iters):
            resid = self.operator.backward(Z, X)    # one operator parameter update
            self.latent_prior.learn(S)              # one prior parameter update (e.g. J)
        if not self._multi:
            err = float(np.mean(resid ** 2))
        else:
            err = (resid ** 2).mean(axis=tuple(range(1, resid.ndim)))   # (C,)
        # self.sigma_x += self.scl_lr * (err - self.sigma_x)   # independent update
        self.sigma_x *= np.exp(self.scl_lr * (err - self.sigma_x))   # independent update
        # self.sigma_x *= np.exp(-self.scl_lr*(err - self.kappa))
        # self.sigma_x *= np.exp(-self.scl_lr*np.log(err / self.kappa))
        return err

    def init_params(self, X, hot_start=True, scl_lr=0, **opt_args):
        # only the row count (n) is used downstream (the E-step's StS prior); the
        # feature dim isn't, so len(X) serves both 2-D (n, d) and 3-D (n, t, d)
        # data -- which is why even the reduced-rank/conv models share this.
        self.n = len(X)
        self.sigma_x = np.ones(self.n_chains) if self._multi else 1   # (C,) per chain
        self.scl_lr = scl_lr
        self.outs = []                            # per-iteration log-odds if debug
        self._Ximp = None                         # per-chain imputation copy (lazy)
        self._best_op = None                      # cached best-chain operator (lazy)

        self.operator.init_params(X, self.dim_hid, hot_start=hot_start, **opt_args)

        # Compose the search: operator scaffold x the latent component's link.
        # That link IS the slab-vs-no-slab choice (the prior's `link` property:
        # BINARY_LINK when slab=False, SLAB_LINK when slab=True).  Compiled once.
        # debug=True compiles the variant that records the log-odds into `out`.
        self.operator.build_search(self.latent_prior.link,
                                   self.latent_prior.prior_plugin, debug=self.debug)

    # A model's __post_init__ composes its components by passing `n_chains` straight
    # into the single (chain-aware) operator / prior class: n_chains == 1 builds the
    # ordinary serial 2-D component, n_chains > 1 the chain-batched one -- same class,
    # same keyword arguments, only the leading chain axis differs.

    def _no_multichain(self, name):
        """Guard for models whose operator has no chain-batched port yet."""
        if self._multi:
            raise NotImplementedError(
                f"{name} has no multi-chain (n_chains > 1) port yet: its operator is "
                f"not chain-batched.  Use n_chains=1, or run several fits and keep the "
                f"best (bae_util.multifit).")

    def _only_multichain(self, name):
        """The mirror guard: for components that exist ONLY chain-batched, so a
        single chain has no serial class to fall back to."""
        if not self._multi:
            raise NotImplementedError(
                f"{name} is chain-batched only (there is no serial version of it), so "
                f"it needs n_chains > 1.  n_chains=2 is the cheapest way to get the "
                f"same thing as a single-chain fit.")

    # ---- multi-chain: pick / extract the winning chain --------------------
    def best_chain(self, X=None, mask=None):
        """Index of the lowest-loss chain (0 for a single chain).  Uses the cached
        final loss unless X is given, then recomputes; pass the observed-entry
        `mask` when imputing to rank chains by their fit to the observed data."""
        if not self._multi:
            return 0
        loss = self.chain_loss if X is None else self.loss(X, mask=mask)
        return int(np.argmin(loss))

    def reconstruct_best(self, X=None):
        """The winning chain's reconstruction forward(S), with the chain axis dropped
        (a single chain just returns forward(S))."""
        if not self._multi:
            return self.operator.forward(self.S)
        return self.operator.forward(self.S)[self.best_chain(X)]

    def collapse(self, X=None):
        """Reduce a multi-chain model to its winning chain: swap in that chain's
        SERIAL operator + prior (via their to_serial(c)) and 2-D state, so the model
        becomes an ordinary single-chain model (n_chains == 1) that downstream 2-D
        code uses unchanged.  No-op for a single chain.  Returns self."""
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


# ===========================================================================
#  Concrete models  (what's left is the operator choice + its learning rule)
# ===========================================================================

@dataclass
class SemiBMF(LinearGaussianBMF):
    """FULL PORT of bae_models.SemiBMF.

    AffineOperator (numpy) + the shared MStep -- the operator's `backward` does
    the gradient update (recon VJP + the weight penalty), and sigma_x is the
    independent update.  EStep / __call__ / loss / loglikelihood / init_latents /
    MStep are all inherited; only init_params (operator setup) is model-specific.
    Bit-for-bit equivalent to the original on the gradient path (test_semibmf_port).

    Constructor is a drop-in for the original; the weight scalars route into the
    AffineOperator (which now owns its own regularization) and the sparse/tree
    scalars into the LatentPrior.  The original's NNLS / SVD M-step branches are
    AffineOperator.backward `method` options (sketched).
    """

    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l2_reg: float = 1e-2
    weight_l1_reg: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.operator = AffineOperator(n_chains=self.n_chains,
                                      fit_intercept=self.fit_intercept,
                                      nonneg=self.nonneg,
                                      pr_reg=self.weight_pr_reg,
                                      l1_reg=self.weight_l1_reg,
                                      l2_reg=self.weight_l2_reg)
        self.latent_prior = LatentPrior(n_chains=self.n_chains,
                                          sparse_reg=self.sparse_reg,
                                          tree_reg=self.tree_reg)

    # init_params + MStep inherited: AffineOperator.init_params builds W, b and
    # AffineOperator.backward does the gradient update.  The whole model body is
    # now just the component assembly in __post_init__.


@dataclass
class JBMF(LinearGaussianBMF):
    """Matrix operator + a *structured* latent prior.  This is the headline:
    JBMF is now just SemiBMF's skeleton with `latent_prior = BoltzmannPrior`.
    The `WtW + J` recurrence and the inverse-Ising pseudolikelihood both live in
    the prior; init_params / EStep / MStep are all inherited.

    NOTE: the BoltzmannPrior (the J coupling + pseudolikelihood) is ported
    bit-for-bit from bae_models.JBMF and verified in test_boltzmann_port.py.  The
    decoder (W) side here is a TorchMatrixOp, whose weight penalty uses this
    refactor's convention (no /dim_hid normalization) rather than JBMF's, so the
    *prior* is exact while the W M-step follows the operator's convention."""

    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l1_reg: float = 0.0
    weight_l2_reg: float = 1e-2
    J_l1_reg: Optional[float] = None
    J_loss: str = 'rple'
    J_lr: float = 1e-2
    slab: bool = False
    slab_prior: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        # self.operator = TorchMatrixOp(fit_intercept=self.fit_intercept,
        #                               nonneg=self.nonneg,
        #                               pr_reg=self.weight_pr_reg,
        #                               l1_reg=0.0,
        #                               l2_reg=self.weight_l2_reg)
        self.operator = AffineOperator(n_chains=self.n_chains,
                                      fit_intercept=self.fit_intercept,
                                      nonneg=self.nonneg,
                                      pr_reg=self.weight_pr_reg,
                                      l1_reg=self.weight_l1_reg,
                                      l2_reg=self.weight_l2_reg,
                                      resample_dead=True)
        self.latent_prior = nbp.BoltzmannPriorNP(n_chains=self.n_chains,
                                          sparse_reg=self.sparse_reg,
                                          tree_reg=self.tree_reg,
                                          J_l1_reg=self.J_l1_reg,
                                          J_loss=self.J_loss,
                                          J_lr=self.J_lr,
                                          slab=self.slab,
                                          slab_prior=self.slab_prior,
                                          sampler='gibbs')

    # init_params + EStep + MStep inherited: the prior's init_params builds J +
    # its optimizer, couple folds WtW + J into the search, and learn does the
    # pseudolikelihood step alongside the operator's W step.


@dataclass
class BiPCA(LinearGaussianBMF):
    """PORT of bae_models.BiPCA (binary PCA: Gaussian obs, orthonormal weights).

    Three latent priors, picked by `J_prior`:
      'none'      -- plain per-element Bernoulli (LatentPrior).
      'boltzmann' -- the structured Ising prior with a continuous coupling fit by
                     SGD on a pseudolikelihood (BoltzmannPriorNP).  Switched on by
                     J_lr > 0, which is what selects it; J_lr == 0 falls back to
                     'none', so the historical J_lr-only interface still works.
      'mrf'       -- the SIGN-CONSTRAINED Ising prior: coupling entries in
                     {-1,0,+1}, resampled by an annealed Gibbs/ICM sweep instead of
                     descended (MRFPrior).  It has no learning rate, so J_lr does
                     not gate it -- naming it is what turns it on.  Its sparsity
                     knob is J_l1_reg (passed through as MRFPrior's J_lam: same
                     role, same default scale, but a per-entry cost in nats rather
                     than an L1 strength).  Chain-batched only -- see the guard.
    """

    sparse_reg: float = 1e-2
    tree_reg: float = 0
    fit_intercept: bool = True
    fit_scl: bool = True

    J_prior: str = 'boltzmann'   # 'none' | 'boltzmann' | 'mrf' (see class docstring)
    J_l1_reg: Optional[float] = None
    J_loss: str = 'rple'
    J_lr: float = 0
    J_beta: float = 1.0          # 'mrf' only: inverse temperature = the scale of the
                                 # {-1,0,+1} coupling (fit and applied).  Distinct
                                 # from the prior temperature, which anneals.
    J_temp: float = 0.0          # 'mrf' only: sampler temperature over J (<=0 = ICM)
    J_sweeps: int = 1            # 'mrf' only: Gibbs sweeps over J per M-step
    slab: bool = False
    slab_prior: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.operator = Procrustes(n_chains=self.n_chains,
                                      fit_intercept=self.fit_intercept,
                                      fit_scl=self.fit_scl)
        if self.J_prior == 'mrf':
            # MRFPrior is chain-batched only: its state carries the chain axis
            # unconditionally, so a single-chain (2-D) operator's search would be
            # handed a (1,m,m) coupling.  Hence the guard, and n_chains passed through.
            self._only_multichain('BiPCA(J_prior="mrf")')
            self.latent_prior = MRFPrior(n_chains=self.n_chains,
                                         sparse_reg=self.sparse_reg,
                                         tree_reg=self.tree_reg,
                                         l0_reg=self.J_l1_reg,
                                         beta_init=self.J_beta,
                                         beta_lr=self.J_lr,
                                         J_temp=self.J_temp,
                                         J_sweeps=self.J_sweeps,
                                         slab=self.slab,
                                         slab_prior=self.slab_prior)
        elif self.J_prior == 'none' or self.J_lr == 0:
            self.latent_prior = LatentPrior(n_chains=self.n_chains,
                                              sparse_reg=self.sparse_reg,
                                              tree_reg=self.tree_reg,
                                              slab=self.slab,
                                              slab_prior=self.slab_prior)
        else:
            self.latent_prior = nbp.BoltzmannPriorNP(n_chains=self.n_chains,
                                              sparse_reg=self.sparse_reg,
                                              tree_reg=self.tree_reg,
                                              J_l1_reg=self.J_l1_reg,
                                              J_loss=self.J_loss,
                                              J_lr=self.J_lr,
                                              slab=self.slab,
                                              slab_prior=self.slab_prior,
                                              sampler='gibbs')


@dataclass
class SCPD(LinearGaussianBMF):
    """PORT of bae_models.SCPD (sparse canonical-polyadic decomposition).

    A CPOperator (L(S) = einsum('ck,tk,nk->ctn', S, U, V) + b over a 3-D data
    tensor X of shape (n, t, d)) + a plain LatentPrior, with the shared autograd
    M-step.  Shares the *entire* sbmf E-step with RRBMF; only the operator's
    drive/gram (and its `backward`) differ.  Like SemiBMF/RRBMF the whole model
    body is the component assembly in __post_init__: init_params / EStep / MStep /
    __call__ / loss / loglikelihood are ALL inherited (dim_hid is the CP rank).

    Constructor mirrors the original: the weight scalars route into the CPOperator
    (participation-ratio on V + L1 on U,V, each /dim_hid as in the original) and
    the sparse/tree scalars into the LatentPrior.  The operator's M-step optimizer
    is Adam, as in bae_models.SCPD.initialize:1144 (override via opt_alg=)."""

    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l1_reg: float = 1e-2
    slab: bool = False
    slab_prior: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.operator = CPOperator(n_chains=self.n_chains,
                                      fit_intercept=self.fit_intercept,
                                      nonneg=self.nonneg,
                                      pr_reg=self.weight_pr_reg,
                                      l1_reg=self.weight_l1_reg)
        self.latent_prior = LatentPrior(n_chains=self.n_chains,
                                          sparse_reg=self.sparse_reg,
                                          tree_reg=self.tree_reg,
                                          slab=self.slab,
                                          slab_prior=self.slab_prior)

@dataclass
class JSCPD(LinearGaussianBMF):
    """PORT of bae_models.SCPD (sparse canonical-polyadic decomposition).

    A CPOperator (L(S) = einsum('ck,tk,nk->ctn', S, U, V) + b over a 3-D data
    tensor X of shape (n, t, d)) + a plain LatentPrior, with the shared autograd
    M-step.  Shares the *entire* sbmf E-step with RRBMF; only the operator's
    drive/gram (and its `backward`) differ.  Like SemiBMF/RRBMF the whole model
    body is the component assembly in __post_init__: init_params / EStep / MStep /
    __call__ / loss / loglikelihood are ALL inherited (dim_hid is the CP rank).

    Constructor mirrors the original: the weight scalars route into the CPOperator
    (participation-ratio on V + L1 on U,V, each /dim_hid as in the original) and
    the sparse/tree scalars into the LatentPrior.  The operator's M-step optimizer
    is Adam, as in bae_models.SCPD.initialize:1144 (override via opt_alg=)."""

    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-1
    weight_l1_reg: float = 0.0
    weight_l2_reg: float = 1e-2
    J_l1_reg: Optional[float] = None
    J_loss: str = 'rple'
    J_lr: float = 1e-2
    slab: bool = False
    slab_prior: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.operator = CPOperator(n_chains=self.n_chains,
                                      fit_intercept=self.fit_intercept,
                                      nonneg=self.nonneg,
                                      pr_reg=self.weight_pr_reg,
                                      l1_reg=self.weight_l1_reg)
        self.latent_prior = nbp.BoltzmannPriorNP(n_chains=self.n_chains,
                                          sparse_reg=self.sparse_reg,
                                          tree_reg=self.tree_reg,
                                          J_l1_reg=self.J_l1_reg,
                                          J_loss=self.J_loss,
                                          J_lr=self.J_lr,
                                          slab=self.slab,
                                          slab_prior=self.slab_prior,
                                          sampler='gibbs')


@dataclass
class RRBMF(LinearGaussianBMF):
    """FULL PORT of bae_models.RRBMF.

    A ReducedRankOp (L(S) = einsum('ck,knt->ctn', S, V@U.T) + b over a 3-D data
    tensor X of shape (n, t, d)) + a plain LatentPrior, with the shared autograd
    M-step.  Shares the *entire* sbmf E-step with SCPD; only the operator's
    drive/gram (and its `backward`) differ.  Like SemiBMF/SpikeNMF the whole model
    body is just the component assembly in __post_init__: init_params / EStep /
    MStep / __call__ / loss / loglikelihood are ALL inherited -- the shared
    init_params uses len(X), so the 3-D data tensor needs no special-casing.

    Constructor mirrors the original: the weight scalars route into the
    ReducedRankOp (which owns its participation-ratio / L1 / L2 regularization)
    and the sparse/tree scalars into the LatentPrior.
    """

    rank: int = None
    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l1_reg: float = 0.0
    weight_l2_reg: float = 1e-2
    slab: bool = False
    slab_prior: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.operator = ReducedRankOp(n_chains=self.n_chains,
                                      rank=self.rank,
                                      fit_intercept=self.fit_intercept,
                                      nonneg=self.nonneg,
                                      pr_reg=self.weight_pr_reg,
                                      l1_reg=self.weight_l1_reg,
                                      l2_reg=self.weight_l2_reg)
        self.latent_prior = LatentPrior(n_chains=self.n_chains,
                                          sparse_reg=self.sparse_reg,
                                          tree_reg=self.tree_reg,
                                          slab=self.slab,
                                          slab_prior=self.slab_prior)


@dataclass
class JRRBMF(LinearGaussianBMF):
    """
    """

    rank: int = None
    nonneg: bool = False
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l1_reg: float = 0.0
    weight_l2_reg: float = 1e-2
    slab: bool = False
    slab_prior: float = 1.0
    J_l1_reg: Optional[float] = None
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
        self.latent_prior = nbp.BoltzmannPriorNP(n_chains=self.n_chains,
                                          sparse_reg=self.sparse_reg,
                                          tree_reg=self.tree_reg,
                                          J_l1_reg=self.J_l1_reg,
                                          J_loss=self.J_loss,
                                          J_lr=self.J_lr,
                                          slab=self.slab,
                                          slab_prior=self.slab_prior,
                                          sampler='gibbs')


# ===========================================================================
#  Kernel factorization  (the one model that is NOT linear-Gaussian)
# ===========================================================================

@dataclass
class KernelBMF(BMF):
    """PORT of bae_models.KernelBMF2 -- Gram/kernel factorization.

    The one model here that does NOT fit the LinearGaussianBMF skeleton.  It
    reconstructs a *kernel* K ~ center(S diag(scl) S^T), which is QUADRATIC in the
    binary latents S rather than an affine map L(S) + b -- so there is no
    LinearOperator (no forward / drive / gram) and it subclasses BMF directly,
    owning its own diagonal scale `scl` (the D) and the running StX = S^T X.

    But it still uses the LatentPrior structure -- the entire latent side is
    reused unchanged.  The prior owns S / StS / Z and their init, and (the point)
    supplies the SAME additive S-prior plugin the dense models use: a plain
    LatentPrior reproduces the original's sparse_reg/tree_reg bit-for-bit
    (test_kernelbmf2_port.py), and dropping in a BoltzmannPriorNP gives KernelBMF2
    a structured Ising prior for free -- exactly the parsimony dividend the
    refactor is for.  Only the E-step's LIKELIHOOD field (quadratic, in
    new_bae_search.make_kernel_search) and the scale M-step are kernel-specific.

    Constructor mirrors bae_models.KernelBMF2: sparse/tree route into the
    LatentPrior; uniform_scale / l1_reg stay on the model -- they shape the scale
    M-step (uniform closed-form vs projected-gradient), not the E-step, which is
    always per-feature (kerbmf3).  `kernel_input=True` takes a precomputed kernel
    instead of a feature matrix (and, like the feature form, supports the
    non-uniform diagonal scale -- the kernel field is weighted by scl[j]/scl[k]
    exactly as kerbmf3, generalizing the scalar-only bae_search.kerbmf2).  Pass
    any LatentPrior subclass as `latent_prior` to override the default (e.g. a
    BoltzmannPriorNP for the structured prior); a slab prior is rejected, since
    the kernel model has no per-element magnitude.
    """

    dim_hid: int
    sparse_reg: float = 0
    tree_reg: float = 1e-2
    uniform_scale: bool = True
    l1_reg: float = 0.0             # only used by the non-uniform scale M-step
    kernel_input: bool = False
    
    J_l1_reg: Optional[float] = None
    J_loss: str = 'rple'
    J_lr: float = 0

    debug: bool = False             # record per-iteration E-step log-odds into self.outs

    def __post_init__(self):
        super().__post_init__()
        if self.J_lr == 0:
            self.latent_prior = LatentPrior(sparse_reg=self.sparse_reg,
                                            tree_reg=self.tree_reg,
                                            slab=False)
        
        else:
            self.latent_prior = nbp.BoltzmannPriorNP(sparse_reg=self.sparse_reg,
                                       tree_reg=self.tree_reg,
                                       J_l1_reg=self.J_l1_reg,
                                       J_loss=self.J_loss,
                                       J_lr=self.J_lr,
                                       slab=False,
                                       sampler='gibbs')

    # expose the prior's spike for the base BMF fit loop (which reads self.S)
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
        # sigma2 = mean squared (centered) kernel entry = <K^2> = data_norm/N^2:
        # the Gaussian entry-noise variance the E-step divides the likelihood field
        # by, so the log-odds is O(1) regardless of N, d and input scale (the raw
        # field ~ <K^2>, so dividing by <K^2> cancels the input magnitude).  Fixed
        # at the data scale -- annealing is temp's job, not sigma2's.
        self.sigma2 = self.data_norm / self.n ** 2
        # compose the search: the kernel scaffold x the prior's plugin (so a
        # structured prior just swaps prior_plugin -- no new kernel).
        self._search = new_bae_search.make_kernel_search(
            self.latent_prior.prior_plugin, kernel_input=self.kernel_input,
            debug=self.debug)
        self._dummy_out = np.zeros((1, 1))           # unused by the sampling path
        self.outs = []                               # per-iteration log-odds if debug

    # ---- latents: the prior owns S/StS/Z; StX is the model's likelihood state
    def init_latents(self, X, **kwargs):
        if self.kernel_input:
            l, V = np.linalg.eigh(X)
            Mx = np.diag(np.sqrt(l + 1e-6)) @ V @ np.random.randn(self.d, self.dim_hid)
        else:
            Mx = X @ np.random.randn(self.d, self.dim_hid)
        # Standardize the random projection before thresholding so the init coding
        # level is SCALE-INVARIANT: the original 1*(Mx >= 0.5) collapses to all-zero
        # when the input (hence Mx) is small, and to ~dense when it is large.
        # Dividing by Mx's own scale makes the 0.5 offset (the original's mild
        # sparsity) relative, so init coding no longer depends on the input
        # magnitude.  StS/Z then come from the prior.
        Mx = Mx / (Mx.std() + 1e-12)
        self.latent_prior.init_latents(Mx - 0.5)
        self.StX = self.S.T @ X                      # running S^T X (kernel state)

        # Scale-match scl to the init S (the uniform M-step target) instead of the
        # default 1.  With the sigma2-normalized E-step the self-energy term is
        # ~scl^2/sigma2; if scl is far from the data scale (e.g. scl=1 on a small-
        # magnitude kernel) it swamps the data term and collapses S on the very
        # first sweep, before the M-step can correct scl.  This target scales WITH
        # the data (<K,SS^T>/||SS^T||^2), so the fit is scale-invariant.
        lp = self.latent_prior
        StS_c = lp.StS - np.outer(np.diag(lp.StS), np.diag(lp.StS)) / self.n
        if self.kernel_input:
            S_ = self.S - self.S.mean(0)
            V_dot = np.diag(S_.T @ X @ S_)
        else:
            StX_c = self.StX - np.outer(np.diag(lp.StS), X.mean(0))
            V_dot = np.sum(StX_c ** 2, axis=1)
        denom = np.sum(StS_c ** 2)
        self.scl = np.full(self.dim_hid, max(0.0, np.sum(V_dot) / denom) if denom > 0 else 1.0)

    def __call__(self, S):
        # center(S diag(scl) S^T) -- the reconstruction (bae_models.KernelBMF2:861)
        return np.einsum('...ik,k,...jk->...ij', S, self.scl, S)

    # ---- E-step: the shared kernel scaffold with the prior's plugin --------
    def EStep(self, X, S, inplace=True):
        lp = self.latent_prior
        Jc, hc = lp.coupling()                       # zeros for a plain prior
        if self.kernel_input:
            data = _center_kernel(X)
            StX = np.zeros((self.dim_hid, 0))
        else:
            data = X
            StX = self.StX
        # in debug mode the search records each element's log-odds into `out` (sized
        # to S so `sample`'s fresh chain works too); snapshot it per fit sweep so
        # self.outs[t] holds iteration t's current magnitudes over all (i, j).
        out = np.zeros(S.shape) if self.debug else self._dummy_out
        self._search(data, S, lp.StS, StX, self.n, self.scl, self.sigma2, self.temp,
                     lp.sparse_reg, lp.tree_reg, Jc, hc, out, inplace, lp.temp)
        if self.debug and inplace:
            self.outs.append(out)
        return S

    # ---- Gibbs sampling: spike only (no Z -- the kernel model has no slab) --
    # BMF.sample threads a Z through EStep for the slab models; the kernel EStep
    # reads only the binary spike S, so we override to walk an S-only chain.
    # Same structure as the base: a fresh chain of len(X) rows, inplace=False so
    # the prior's persistent StS is frozen while the field uses the trained params.
    def sample(self, X, temp=None, n_samp=1, burnin=10, **args):
        if temp is not None:
            self.temp = temp

        samps = np.zeros((n_samp, len(X), self.dim_hid))
        S = 1.0 * np.random.choice([0, 1], size=(len(X), self.dim_hid))
        i = 0
        for n in range(n_samp * burnin):
            samp = self.EStep(X, S, inplace=False, **args)
            if not np.mod(n, burnin):
                samps[i] = 1 * samp
                i += 1

        return samps

    # ---- M-step: optimally scale D (+ the prior's own update) --------------
    def MStep(self, X, ES):
        """Update the diagonal scale scl (bae_models.KernelBMF2.MStep): uniform
        closed-form when uniform_scale, else a projected-gradient step with L1 and
        non-negativity.  The prior's `learn` runs alongside (no-op for a plain
        prior; the inverse-Ising step for a BoltzmannPrior), then sigma is the
        normalized residual energy."""
        lp = self.latent_prior
        S_ = ES - ES.mean(0)
        StS = lp.StS - np.outer(np.diag(lp.StS), np.diag(lp.StS)) / len(X)
        StX = self.StX - np.outer(np.diag(lp.StS), X.mean(0))

        if self.kernel_input:
            V_dot = np.diag(S_.T @ X @ S_)
        else:
            V_dot = np.sum(StX ** 2, axis=1)
        G = StS ** 2

        if self.uniform_scale:
            dot = np.sum(V_dot)
            nrm = np.sum(G)
            target = np.full(self.dim_hid, max(0.0, dot / nrm))
        else:
            grad = G @ self.scl - V_dot + self.l1_reg
            eta = np.dot(grad, grad) / (np.dot(grad, G @ grad) + 1e-8)
            target = np.maximum(0.0, self.scl - eta * grad)

        self.scl += self.lr * (target - self.scl)

        lp.learn(ES)                                 # prior parameter update

        Qnrm = self.scl @ G @ self.scl
        dot_total = np.sum(self.scl * V_dot)
        return 1 + (Qnrm - 2 * dot_total) / self.data_norm

    def loss(self, X):
        lp = self.latent_prior
        S = lp.S
        StS = lp.StS - np.outer(np.diag(lp.StS), np.diag(lp.StS)) / len(X)
        if self.kernel_input:
            V_dot = np.diag(S.T @ _center_kernel(X) @ S)
            Knrm = np.sum(_center_kernel(X) ** 2)
        else:
            V_dot = np.sum((S.T @ X) ** 2, axis=1)
            X_ = X - X.mean(0)
            Knrm = np.sum(X_.T @ X_ ** 2)
        dot = np.sum(self.scl * V_dot)
        Qnrm = self.scl @ (StS ** 2) @ self.scl
        return (Qnrm + Knrm - 2 * dot) / Knrm


@dataclass
class ConvBMF(LinearGaussianBMF):
    """Convolutional operator: the same linear-Gaussian template in a
    translation-invariant space, so it plugs in a ConvOperator (which carries
    the convbmf search) and otherwise inherits everything."""

    kernel_size: int = None

    def __post_init__(self):
        super().__post_init__()
        self._no_multichain("ConvBMF")   # ConvOperator has no chain-batched port yet

    def init_params(self, X, **opt_args):
        self.operator = ConvOperator()
        # init GP kernels K, b + operator.optimizer (bae_models.ConvBMF:1231)
        ...

    # MStep inherited; ConvOperator.backward carries the seq-NMF orthogonality reg.

