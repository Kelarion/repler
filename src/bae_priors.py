"""
bae_priors.py  --  the latent side of the BMF stack
=======================================================

The latent component is one slot on a model, `latent_prior`.  It owns the latent
state (S, StS, Z), its initialization, the search `link`, and whatever prior
parameters it has of its own:

  LatentPrior     independent Bernoulli: sparsity (alpha) + tree (beta) penalty.
  BoltzmannPrior  pairwise Ising prior with a continuous, symmetric coupling, fit
                  by pseudolikelihood ('rple' / 'logrise') or by maximum
                  likelihood ('mle', persistent fantasy particles).
  MRFPrior        the same prior with the coupling constrained to {-1, 0, +1},
                  resampled by the annealed Gibbs/ICM kernels in mrf_samplers.

`slab=True` on any of them makes it spike-and-slab (the search link becomes
SLAB_LINK and the effective latent Z carries continuous magnitudes) -- the slab is
orthogonal to the prior's structure, so it is a flag, not a subclass.

CONVENTION.  Everything here works in the {0,1} coding of the reference
implementation (minimal_structured_bipca.StructuredBiPCA): the conditional
log-odds of a spike under the Boltzmann prior is

    logit_j = 2 * (sum_k J[j,k] S_k + h[j]),      J symmetric, zero diagonal,

so `coupling()` -- the additive (Jc, hc) the search adds to the likelihood
log-odds -- is simply (2J, 2h).  J and h are learned from the BINARY spike, never
from the slab magnitudes.

Multiple chains are carried by a leading axis: every array here is written
shape-agnostically (batch dims in front of the trailing (n, m) / (m, m) / (m,)),
so n_chains == 1 is the ordinary 2-D case with no special-casing.
"""

import itertools
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from scipy.special import expit, logsumexp, xlogy, log_ndtr
from scipy.optimize import minimize_scalar

import df_util
import bae_search
import mrf_samplers as mrf              # sign-constrained Ising kernels (numba)


def _swap(A):
    """Transpose the trailing two axes (chain-axis-agnostic `.T`)."""
    return A.swapaxes(-1, -2)


def _zero_diag(A):
    """Zero the trailing diagonal of a (..., m, m) array, in place."""
    d = np.arange(A.shape[-1])
    A[..., d, d] = 0.0
    return A


@lru_cache(maxsize=None)
def _all_states(m):
    """Every binary pattern of length m, (2^m, m) -- cached per m."""
    return np.array(list(itertools.product([0.0, 1.0], repeat=m)))


def _bernoulli_entropy(P):
    """Elementwise entropy of independent Bernoulli(P), in nats (0 at P = 0 or 1)."""
    return -(xlogy(P, P) + xlogy(1 - P, 1 - P))


def _categorical(logits):
    """Sample one index per row from softmax(logits) (Gumbel-max), vectorized."""
    g = -np.log(-np.log(np.random.rand(*logits.shape)))
    return np.argmax(logits + g, axis=-1)


# ===========================================================================
#  Schedules for the prior temperature
# ===========================================================================
#
# The prior temperature (LatentPrior.temp) is a pure SAMPLING knob: it divides the
# structured prior's contribution to the flip log-odds inside the search and
# touches nothing else, so moving it during a fit cannot bias what the prior
# LEARNS.  A schedule is any object with `update(**inputs) -> float`, called once
# per iteration by BMF.fit with (it, max_iter, temp, loss).

@dataclass
class TempSchedule:
    """Default: prior temperature pinned at 1.0 (structure at face value)."""

    def update(self, **inputs):
        return 1.0


@dataclass
class ConstantTemp(TempSchedule):
    """A fixed temperature: structure held permanently weak (>1) or sharp (<1)."""

    temp: float = 1.0

    def update(self, **inputs):
        return self.temp


@dataclass
class GeomAnneal(TempSchedule):
    """The model temperature's geometric schedule applied to the prior: start with
    the structure switched off and hand it over to the data as the fit cools."""

    initial: float = 10.0
    decay_rate: float = 0.88
    period: int = 10
    min_temp: float = 1.0

    def update(self, it=0, **inputs):
        return self.min_temp + self.initial * (self.decay_rate ** (it // self.period))


@dataclass
class AdaptiveTemp(TempSchedule):
    """Loss-adaptive: drive the model's reconstruction MSE toward `kappa` by
    nudging the inverse temperature lambda,

        lambda <- lambda * exp(alpha * EMA(kappa - err)),      temp = 1 / lambda
    """

    alpha: float = 1e-3          # learning rate on the inverse temperature
    kappa: float = 1e-3          # target reconstruction MSE
    gamma: float = 0.9           # exponential moving average
    lambda_init: float = 1.0     # initial inverse temperature

    def __post_init__(self):
        self._lam = self.lambda_init
        self._D = self.kappa

    def update(self, it=0, loss=None, **inputs):
        if it == 0:                                   # fresh fit -> re-seed
            self._lam, self._D = self.lambda_init, self.kappa
        if loss is not None:
            self._D = self.gamma * self._D + (1 - self.gamma) * (self.kappa - np.array(loss))
            self._lam *= np.exp(self.alpha * self._D)
        return np.where(self._lam <= 1e-7, np.inf, 1 / self._lam)


# ===========================================================================
#  LatentPrior -- independent Bernoulli, and the latent state itself
# ===========================================================================

@dataclass
class LatentPrior:
    """Independent Bernoulli prior (sparsity + tree penalty) and the owner of the
    latent state: S (binary spike), StS, Z (the effective latent the operator
    reads -- S itself, or the continuous magnitudes when slab=True).

    n_chains > 1 puts a leading chain axis on every state array."""

    n_chains: int = 1            # >1 runs C independent chains
    sparse_reg: float = 0.0      # alpha in the kernels
    tree_reg: float = 1e-2       # beta  in the kernels
    slab: bool = False           # spike-and-slab?  picks the link + effective Z
    slab_prior: float = 1.0      # tau (the slab's exponential rate); slab only
    temp: float = 1.0            # prior temperature (see TempSchedule); inert here

    @property
    def _multi(self):
        return self.n_chains > 1

    # `link` and `prior_plugin` are computed properties, so a subclass inherits
    # slab support for free and there is no __post_init__ to collide with.
    @property
    def link(self):
        return (bae_search.SLAB_LINK if self.slab
                else bae_search.BINARY_LINK)

    @property
    def prior_plugin(self):
        return bae_search.PRIOR_PLAIN

    # ---- parameters (a structured subclass builds its coupling here) ------
    def init_params(self, S):
        return None

    # ---- latent state + its initialization -------------------------------
    def init_latents(self, drive):
        """Seed the spike from the sign of the operator's drive, plus StS and the
        effective latent Z.  `drive` is (..., n, m); with n_chains > 1 it differs
        across chains (each has its own W), which is what makes them explore
        different basins."""
        self.S = 1.0 * (drive >= 0)
        self.StS = _swap(self.S) @ self.S
        self.Z = (drive * self.S) if self.slab else self.S
        self.init_params(self.S)

    # ---- what the E-step pulls from the latent side ----------------------
    def coupling(self):
        """The additive (Jc, hc) the search adds to the spike log-odds:
        sum_k Jc[j,k] S_ik + hc[j], LINEAR IN THE BINARY SPIKES and never folded
        into WtW.  No structure here, so zeros."""
        shape = self.S.shape[:-2] + (self.S.shape[-1],)
        return np.zeros(shape + shape[-1:]), np.zeros(shape)

    # ---- the same prior as an energy (chain selection; see BMF.elbo) -----
    #
    # Everything the search adds to the spike log-odds that IS a normalizable
    # prior comes from `coupling()` plus the sparsity penalty, so the three
    # methods below are written once here in terms of coupling() and inherited
    # unchanged by every structured subclass.  Writing the prior as
    #
    #     log p(S) = sum_i [ S_i' J S_i + g' S_i ] - n * log Z(J, g)
    #
    # with J = Jc/(2*temp) (zero diagonal) and g = hc/temp - sparse_reg recovers
    # exactly the plugin's conditional log-odds, 2*(J S_i)_j + g_j.
    #
    # NOT included: the tree/StS penalty (`tree_reg`).  It is a combinatorial
    # regularizer, not a potential -- its contribution to the log-odds is not the
    # difference of any energy and it has no partition function -- so an ELBO
    # built on it would not be a bound on anything.  With tree_reg > 0 the
    # selection score therefore scores the probabilistic part of the model only.

    def _energy_params(self):
        """(J, g): this prior as `sum_i S_i' J S_i + g' S_i`, tempered, with the
        sparsity penalty folded into the field."""
        Jc, hc = self.coupling()
        T = np.asarray(self.temp, dtype=float)        # scalar or per-chain
        return (_zero_diag(Jc / (2 * T[..., None, None])),
                hc / T[..., None] - self.sparse_reg)

    def logodds(self, S):
        """The prior's additive contribution to the spike log-odds, (..., n, m):
        the vectorized mirror of the search's prior plugin (minus the tree term).
        Linear in S, so passing a mean-field P gives E_q of the same quantity."""
        J, g = self._energy_params()
        return 2 * (S @ J) + g[..., None, :]

    def log_prob(self, P, m1=None):
        """E_q[log p(S)] for q = prod_ij Bernoulli(P_ij), summed over units and
        rows -- normalizer included, so chains with different couplings are
        comparable.  A point estimate S is just q with P in {0, 1}.  With a slab,
        `m1` (the conditional mean magnitude) adds the magnitudes' own Exp(tau)
        density, log p(Z | S=1) = log tau - tau*Z."""
        J, g = self._energy_params()
        e = (np.einsum('...nj,...jk,...nk->...', P, J, P)
             + (P * g[..., None, :]).sum((-2, -1)))
        lp = e - P.shape[-2] * self._logZ(J, g)
        if self.slab and m1 is not None:
            tau = self.slab_prior
            lp = lp + (P * (np.log(tau) - tau * m1)).sum((-2, -1))
        return lp

    def link_moments(self, E, gjj, sigma2):
        """Numpy mirror of the search's LINK plugin (bae_search.score_*), for
        a whole array at once: given the leave-one-out field E (..., n, m) and the
        gram diagonal, return

          (logodds, m1, m2) -- the likelihood's contribution to the spike
          log-odds, and the first two moments of the magnitude GIVEN a spike.

        Binary link: the magnitude is fixed at 1, so m1 = m2 = 1.  Slab: the
        magnitude is TruncNormal(mu, nu) on [0, inf) and the log-odds is the same
        collapsed (Z-integrated) expression the kernel uses."""
        if not self.slab:
            one = np.ones(np.broadcast_shapes(E.shape, np.shape(gjj)))
            return (E - 0.5 * gjj) / sigma2, one, one

        tau = self.slab_prior
        mu = (E - tau * sigma2) / gjj
        nu = np.sqrt(sigma2 / gjj) * np.ones_like(mu)
        a = mu / nu
        logodds = (np.log(tau * nu * np.sqrt(2 * np.pi)) + 0.5 * a ** 2
                   + log_ndtr(a))          # == mu^2 / (2 * sigma2/gjj), as in the kernel
        # inverse Mills ratio -> the truncated normal's moments on [0, inf)
        lam = np.exp(-0.5 * a ** 2 - 0.5 * np.log(2 * np.pi) - log_ndtr(a))
        return logodds, mu + nu * lam, mu ** 2 + nu ** 2 + mu * nu * lam

    exact_logZ_units = 14        # enumerate 2^m states up to this many units

    @classmethod
    def _logZ(cls, J, g):
        """log Z of the prior's energy, per batch element.  EXACT (enumeration of
        all 2^m spike patterns) while m is small -- which is the normal case, a
        handful of latent units -- because the naive mean-field bound below is
        biased low by an amount that GROWS with the coupling strength, and chains
        with different couplings are exactly what this has to compare.  Above
        `exact_logZ_units` there is no cheap exact answer and mean field is used."""
        m = np.shape(g)[-1]
        if m > cls.exact_logZ_units:
            return cls._mf_logZ(J, g)
        states = _all_states(m)                                # (2^m, m)
        E = (np.einsum('sj,...jk,sk->...s', states, J, states)
             + g @ states.T)
        return logsumexp(E, axis=-1)

    @staticmethod
    def _mf_logZ(J, g, n_iter=500, damp=0.5, tol=1e-11):
        """log Z of `sum_j g_j S_j + sum_jk J_jk S_j S_k` by naive mean field:
        iterate p_j = sigmoid(2 (J p)_j + g_j) to its fixed point and return that
        distribution's free energy.  EXACT when J == 0 (the independent case,
        where it reduces to sum_j log(1 + e^g_j)); a lower bound otherwise."""
        p = expit(g)
        for _ in range(n_iter):
            new = expit(2 * np.einsum('...jk,...k->...j', J, p) + g)
            step = np.abs(new - p).max()
            p = damp * p + (1 - damp) * new
            if step < tol:
                break
        return (np.einsum('...j,...jk,...k->...', p, J, p) + (g * p).sum(-1)
                + _bernoulli_entropy(p).sum(-1))

    # ---- parameter updates -----------------------------------------------
    def learn(self, S):
        """One M-step update of the prior's own parameters from the BINARY spike.
        No-op without structure."""
        return None

    def refit(self, S):
        """Optional end-of-fit refit of the prior's parameters (see
        BoltzmannPrior.refit).  No-op without structure."""
        return None

    def to_serial(self, c):
        """Chain c as an ordinary single-chain (n_chains=1) prior with 2-D state."""
        lp = LatentPrior(sparse_reg=self.sparse_reg, tree_reg=self.tree_reg,
                         slab=self.slab, slab_prior=self.slab_prior, temp=self.temp)
        self._copy_state(lp, c)
        return lp

    def _chain(self, c):
        """Selector pulling chain c out of a state array (identity when serial)."""
        return (lambda a: a[c]) if self._multi else (lambda a: a)

    def _copy_state(self, lp, c):
        """Copy this prior's latent state into `lp`, taking chain c when multi."""
        pick = self._chain(c)
        lp.S, lp.Z = pick(self.S).copy(), pick(self.Z).copy()
        lp.StS = pick(self.StS).copy()


# ===========================================================================
#  BoltzmannPrior -- pairwise Ising prior in the {0,1} coding
# ===========================================================================

@dataclass
class BoltzmannPrior(LatentPrior):
    """Pairwise (Ising) prior over the spikes, log p(S) ~ S' J S + 2 h' S with J
    symmetric and zero-diagonal, so the conditional log-odds of S_j is
    2*(J S + h)_j and `coupling()` is just (2J, 2h).

    Attaching `latent_prior=BoltzmannPrior(...)` to any LinearGaussianBMF gives it
    JBMF's structured prior without touching the model class.  Three fitting modes,
    picked by `J_loss`:

      'rple'    symmetric nodewise pseudolikelihood (the reference's `_rple_step`)
      'logrise' the logsumexp (interaction-screening) pseudolikelihood
      'mle'     true maximum likelihood -- moment matching against persistent
                fantasy particles (PCD), drawn by `sampler` ('gibbs' | 'gwg').

    All three share one parameter step: an L1 proximal (soft-threshold) on the
    coupling, symmetrization, and a zero diagonal.  Everything is written
    shape-agnostically, so n_chains > 1 fits one coupling per chain with the same
    code.  J and h are learned from the BINARY spike -- never from slab
    magnitudes -- and are applied at face value (`temp` scales only the search)."""

    J_l1_reg: Optional[float] = 2e-4   # L1 on the coupling (proximal)
    J_loss: str = 'rple'               # 'rple' | 'logrise' | 'mle'
    J_lr: float = 0.3
    h_l2_reg: float = 1e-4             # ridge on the field
    mle_n_samp: int = 100              # persistent fantasy particles ('mle')
    mle_gibbs_steps: int = 1           # sampler sweeps per learn ('mle', PCD)
    sampler: str = 'gibbs'             # 'gibbs' | 'gwg' (gibbs-with-gradients)
    gwg_temp: float = 2.0              # GWG proposal temperature

    # end-of-fit refit (`refit`): reset, select a support under a strong L1,
    # debias on that support, then rescale the whole coupling by one temperature.
    refit_l1: Optional[float] = None   # None -> 0.142/m, the reference's default
    refit_select_steps: int = 350
    refit_debias_steps: int = 250
    refit_select_lr: float = 0.4
    refit_debias_lr: float = 0.15

    @property
    def prior_plugin(self):
        return bae_search.PRIOR_BOLTZMANN

    def init_params(self, S):
        batch, m = S.shape[:-2], S.shape[-1]
        self.J = np.zeros(batch + (m, m))
        self.h = np.zeros(batch + (m,))
        if self.J_loss == 'mle':
            self.state = 1.0 * np.random.choice(
                [0, 1], size=batch + (self.mle_n_samp, m))
            self._stash_moments()

    def coupling(self):
        return 2 * self.J, 2 * self.h

    # ---- one gradient + one proximal step --------------------------------
    def _pl_grads(self, S, J, h):
        """(dL/dJ, dL/dh) of the mean-per-element pseudolikelihood at (J, h).
        'rple' and 'logrise' differ only in dL/dpred; the symmetrization, the
        1/(n*m) normalization, the field ridge and the zero diagonal are shared."""
        n, m = S.shape[-2:]
        pred = S @ J + h[..., None, :]
        if self.J_loss == 'logrise':
            sgn = 2 * S - 1
            B = -sgn * pred
            G = -sgn * np.exp(B - logsumexp(B, axis=-1, keepdims=True)) / (n * m)
        else:                                            # 'rple'
            G = 2 * (expit(2 * pred) - S) / (n * m)
        gJ = _swap(S) @ G
        return _zero_diag(gJ + _swap(gJ)), G.sum(-2) + self.h_l2_reg * h

    def _grads(self, S):
        """(dL/dJ, dL/dh) for the configured J_loss, at the current parameters."""
        if self.J_loss != 'mle':
            return self._pl_grads(S, self.J, self.h)
        # maximum likelihood: match the data and model moments of log p(S) ~
        # S'JS + 2h'S.  Model moments come from the persistent particles, advanced
        # mle_gibbs_steps sweeps per call (PCD).
        n = S.shape[-2]
        mean_d, second_d = S.mean(-2), _swap(S) @ S / n
        mean_m, second_m = self._model_moments()
        return _zero_diag(second_m - second_d), 2 * (mean_m - mean_d)

    def _step(self, J, h, gJ, gh, lr, l1=0.0, support=None):
        """One descent step + L1 proximal + symmetrize + zero diagonal, optionally
        confined to `support` (a {0,1} mask, for the debias pass of `refit`)."""
        if support is not None:
            gJ = gJ * support
        J = J - lr * gJ
        if l1:
            J = np.sign(J) * np.maximum(np.abs(J) - lr * l1, 0.0)
        J = _zero_diag((J + _swap(J)) / 2)
        if support is not None:
            J = J * support
        return J, h - lr * gh

    def learn(self, S):
        self.J, self.h = self._step(self.J, self.h, *self._grads(S),
                                    self.J_lr, l1=self.J_l1_reg)

    # ---- end-of-fit refit (the reference's _final_prior_fit) --------------
    def refit(self, S):
        """Discard the path-dependent (J, h) and refit them from the final spikes:
        polarity-normalize the codes, select a support under a strong L1, debias on
        that support, map back to the decoder's coordinates, then fit one scalar
        temperature for the whole coupling."""
        m = S.shape[-1]
        l1 = 0.142 / m if self.refit_l1 is None else self.refit_l1

        # Polarity-normalize: flip the columns that are on more than half the time,
        # so the L1 selects couplings rather than marginals.
        flip = S.mean(-2) > 0.5                                   # (..., m)
        codes = np.where(flip[..., None, :], 1 - S, S)

        J, h = np.zeros_like(self.J), np.zeros_like(self.h)
        for _ in range(self.refit_select_steps):
            J, h = self._step(J, h, *self._pl_grads(codes, J, h),
                              self.refit_select_lr, l1=l1)
        support = 1.0 * (np.abs(J) > 1e-8)
        for _ in range(self.refit_debias_steps):
            J, h = self._step(J, h, *self._pl_grads(codes, J, h),
                              self.refit_debias_lr, support=support)

        # Back to the decoder's (non-flipped) coordinates (h uses the FLIPPED J).
        off = 1.0 * flip
        sgn = 1 - 2 * off
        h = sgn * (h + (J @ off[..., None])[..., 0])
        J = sgn[..., :, None] * J * sgn[..., None, :]

        self.support = support
        self.prior_scale = self._fit_prior_scale(S, J, h)
        self.J = self.prior_scale[..., None, None] * J
        self.h = self.prior_scale[..., None] * h

    def _fit_prior_scale(self, S, J, h):
        """The single temperature that best explains the spikes under (J, h): the
        pseudolikelihood-optimal exp(x) with x in [-2, 2], one value per chain."""
        scale = np.ones(S.shape[:-2])
        for idx in np.ndindex(scale.shape):
            Si, Ji, hi = S[idx], J[idx], h[idx]

            def objective(log_scale):
                logit = 2 * np.exp(log_scale) * (Si @ Ji + hi)
                return float(np.mean(np.logaddexp(0, logit) - Si * logit))

            scale[idx] = np.exp(minimize_scalar(
                objective, bounds=(-2, 2), method='bounded').x)
        return scale

    # ---- sampling the prior (the MLE model moments, and standalone draws) --
    def _sweep(self, state):
        """One in-place Gibbs sweep over the {0,1} particles: node j turns on with
        probability sigmoid(2*(J[j].S + h[j]))."""
        for j in range(self.h.shape[-1]):
            field = np.einsum('...sk,...k->...s', state, self.J[..., j, :])
            p = expit(2 * (field + self.h[..., j, None]))
            state[..., j] = 1.0 * (np.random.rand(*p.shape) < p)
        return state

    def _gwg_flip(self, state, field):
        """One gibbs-with-gradients step (Grathwohl et al. 2021, sec 4): a single
        gradient-informed flip per particle with a Metropolis-Hastings correction.
        Flipping S_j changes log p by exactly d_j = 2*(1 - 2 S_j)*(J S + h)_j (the
        energy is quadratic with zero diagonal), so q(j|S) = softmax(d/gwg_temp).
        `field` (= J S + h) is carried along and rank-1 corrected after a flip, so
        each step costs O(n_samp * m) rather than another matmul."""
        ns = len(state)
        ar = np.arange(ns)
        d = 2 * (1 - 2 * state) * field
        logits = d / self.gwg_temp
        j = _categorical(logits)
        lp_f = logits[ar, j] - logsumexp(logits, axis=-1)

        prop = state.copy()
        prop[ar, j] = 1 - state[ar, j]
        field_p = field + (1 - 2 * state[ar, j])[:, None] * self.J[j]
        logits_r = 2 * (1 - 2 * prop) * field_p / self.gwg_temp
        lp_r = logits_r[ar, j] - logsumexp(logits_r, axis=-1)

        acc = np.log(np.random.rand(ns)) < d[ar, j] + lp_r - lp_f
        state[acc], field[acc] = prop[acc], field_p[acc]
        return state

    def _advance(self, n_sweeps):
        """Advance the persistent particles by n_sweeps sweep-equivalents (a GWG
        sweep = m gradient-informed flips).  The coupling is used at face value, so
        the particles come from the same model the MLE gradient matches."""
        if self.sampler == 'gwg':
            if self._multi:
                raise NotImplementedError(
                    "the gwg sampler is not chain-batched; use sampler='gibbs' "
                    "with n_chains > 1")
            field = self.state @ self.J + self.h
            for _ in range(n_sweeps * self.state.shape[-1]):
                self._gwg_flip(self.state, field)
        else:
            for _ in range(n_sweeps):
                self._sweep(self.state)
        return self.state

    def _model_moments(self):
        self._advance(self.mle_gibbs_steps)
        return self._stash_moments()

    def _stash_moments(self):
        """(mean, <SS'>) of the fantasy particles; also stash mean/covariance."""
        ns = self.state.shape[-2]
        mean = self.state.mean(-2)
        second = _swap(self.state) @ self.state / ns
        self.model_mean = mean
        self.model_cov = second - mean[..., :, None] * mean[..., None, :]
        return mean, second

    def sample(self, n_samp=1, **gibbs_args):
        """Draw {0,1} samples from the prior alone, via df_util.gibbs (which takes
        exactly the additive (Jc, hc) that `coupling` returns)."""
        Jc, hc = self.coupling()
        if not self._multi:
            return df_util.gibbs(Jc, hc, temp=self.temp, n_samp=n_samp,
                                 **gibbs_args).T
        return np.stack([df_util.gibbs(Jc[c], hc[c], temp=self.temp,
                                       n_samp=n_samp, **gibbs_args).T
                         for c in range(self.n_chains)])

    def to_serial(self, c):
        lp = BoltzmannPrior(
            sparse_reg=self.sparse_reg, tree_reg=self.tree_reg, slab=self.slab,
            slab_prior=self.slab_prior, temp=self.temp, J_l1_reg=self.J_l1_reg,
            J_loss=self.J_loss, J_lr=self.J_lr, h_l2_reg=self.h_l2_reg,
            mle_n_samp=self.mle_n_samp, mle_gibbs_steps=self.mle_gibbs_steps,
            sampler=self.sampler, gwg_temp=self.gwg_temp)
        self._copy_state(lp, c)
        pick = self._chain(c)
        lp.J, lp.h = pick(self.J).copy(), pick(self.h).copy()
        if self.J_loss == 'mle':
            lp.state = pick(self.state).copy()
            lp._stash_moments()
        return lp


# ===========================================================================
#  MRFPrior -- the same prior with a sign-constrained coupling
# ===========================================================================

@dataclass
class MRFPrior(LatentPrior):
    """Sign-constrained Ising prior, fit by the annealed Gibbs/ICM sampler in
    mrf_samplers.  State (trailing shape; a leading chain axis when n_chains > 1):

      J  (m, m)   symmetric, zero diagonal, entries in {-1, 0, +1}
      h  (m,)     unconstrained field
      beta scalar the prior's inverse temperature -- the coupling's SCALE

    `beta` and `temp` are separate knobs and the split matters: J carries no scale
    of its own, so beta IS the model being fit, while temp is the search's prior
    temperature, which the fit loop anneals.

    The mrf kernels are 2-D numba loops, so chains are looped over in Python (C is
    small and each call is O(m^2 n) inside numba)."""

    beta_init: float = 1.0          # inverse temperature (mrf's convention)
    l0_reg: Optional[float] = 0.3   # per-sample cost (nats) of a nonzero entry
    J_temp: float = 0.0             # SAMPLER temperature over J (mrf's `temp`)
    J_sweeps: int = 1               # gibbs_sweep_J passes per learn()
    fit_h: bool = True              # fit the field before each J sweep
    h_newton: bool = False          # newton_h instead of grad_h
    h_lr: float = 1.0               # grad_h's lr, in units of 1/beta^2
    h_l2_reg: float = 1e-2          # mrf's lam_h (0.5*lam_h*h^2 ridge)
    h_newton_steps: int = 3         # mrf's n_steps (both fitters)
    beta_mle: bool = True
    beta_lr: float = 1e-3
    beta_bsz: int = 100             # monte carlo batch size for the beta gradient

    @property
    def prior_plugin(self):
        return bae_search.PRIOR_BOLTZMANN

    def init_params(self, S):
        batch, m = S.shape[:-2], S.shape[-1]
        # Rebuilt (not scaled in place) on every init_latents, so a multi-start
        # fit does not compound the scaling across restarts.
        self.J_lam = self.l0_reg * np.sqrt(np.log(m ** 2 / 1e-3) / S.shape[-2])
        self.J = np.zeros(batch + (m, m))               # {-1, 0, +1}, zero diagonal
        self.h = np.zeros(batch + (m,))
        self.beta = np.full(batch, float(self.beta_init))
        self._n_changed = np.zeros(batch, dtype=int)

    def coupling(self):
        """The (Jc, hc) reproducing this prior's conditional log-odds
        beta*(J.sigma + h) with sigma = 2S - 1."""
        return (2.0 * self.beta[..., None, None] * self.J,
                self.beta[..., None] * (self.h - self.J.sum(-1)))

    def learn(self, S):
        """Resample the discrete coupling from the BINARY spikes.  The fields F are
        rebuilt per call because S changes every M-step (build_fields is O(m^2 n),
        the same order as one sweep)."""
        spins = 2.0 * S - 1.0
        for idx in np.ndindex(self.beta.shape):
            J, h = self.J[idx], self.h[idx]
            St = np.ascontiguousarray(spins[idx].T)     # (m, n), mrf's layout
            F = mrf.build_fields(J, h, St)
            for _ in range(self.J_sweeps):
                if self.fit_h:
                    self._fit_h(J, h, St, F, self.beta[idx])
                self._n_changed[idx] = mrf.gibbs_sweep_J(
                    J, St, F, self.J_lam * self.beta[idx], self.beta[idx], self.J_temp)
                self.beta[idx] = self._fit_beta(J, h, St, F, self.beta[idx])

    def _fit_h(self, J, h, St, F, beta):
        """One pass of the field fitter over every node; mutates h and F in place."""
        if self.h_newton:
            mrf.newton_h(J, h, St, F, beta, self.h_l2_reg, self.h_newton_steps)
        else:
            mrf.grad_h(J, h, St, F, beta, self.h_l2_reg, self.h_newton_steps, self.h_lr)

    def _fit_beta(self, J, h, St, F, beta):
        if self.beta_mle:
            return mrf.mle_beta(J, h, beta, St, F, lr=self.beta_lr,
                                n_samp=self.beta_bsz, J_lam=0)
        return mrf.ple_beta(J, h, beta, St, F, lr=self.beta_lr, J_lam=0)

    def objective(self, S):
        """Per-sample objective (mean log pseudolikelihood - lam*nnz(J)), the
        monitoring quantity of fit_sign_ising's history."""
        spins = 2.0 * S - 1.0
        out = np.empty(self.beta.shape)
        for idx in np.ndindex(self.beta.shape):
            St = np.ascontiguousarray(spins[idx].T)
            F = mrf.build_fields(self.J[idx], self.h[idx], St)
            out[idx] = mrf.objective(self.J[idx], St, F, self.J_lam, self.beta[idx])
        return out if self._multi else float(out)

    def sample(self, n_samp=1, **gibbs_args):
        """Draw {0,1} samples from the prior, via mrf_samplers.gibbs_samp."""
        samps = [mrf.gibbs_samp(self.beta[idx] * self.J[idx],
                                self.beta[idx] * self.h[idx],
                                temp=self.temp, n_samp=n_samp, **gibbs_args).T
                 for idx in np.ndindex(self.beta.shape)]
        samps = np.stack(samps) if self._multi else samps[0]
        return (1 + samps) / 2

    def to_serial(self, c):
        """Chain c as a serial BoltzmannPrior.  The continuous coupling carries its
        own scale, so beta is folded in: this prior's conditional log-odds
        beta*(J sigma + h) equals the Boltzmann one 2*(J' S + h') at
        J' = beta*J and h' = (beta/2)*(h - J.sum(-1))."""
        lp = BoltzmannPrior(
            sparse_reg=self.sparse_reg, tree_reg=self.tree_reg, slab=self.slab,
            slab_prior=self.slab_prior, temp=self.temp, J_l1_reg=self.J_lam)
        self._copy_state(lp, c)
        idx = (c,) if self._multi else ()
        lp.J = self.beta[idx] * self.J[idx]
        lp.h = 0.5 * self.beta[idx] * (self.h[idx] - self.J[idx].sum(-1))
        return lp
