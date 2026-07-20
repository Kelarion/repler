"""
new_bae_priors.py  --  ILLUSTRATION ONLY (design sketch)
========================================================

The *latent side* of the refactored BMF stack (see new_bae_models.py for the
overview): the latent component is one slot, `latent_prior`, which owns the
latent state (S, StS, Z), its initialization, and the search `link`.

  * LatentPrior     -- independent Bernoulli prior (sparsity + tree penalty);
                       the latent component.  `slab=True` makes it spike-and-slab
                       (link = SLAB_LINK, effective latent Z = continuous mult.).
  * BoltzmannPrior  -- structured (pairwise/Ising) prior, the `J` coupling lifted
                       out of JBMF; drop it onto any model for free.  Inherits the
                       slab flag, so BoltzmannPrior(slab=True) is a structured
                       spike-and-slab with no extra code.

"Has a slab" is just the `slab` flag on whichever prior -- not a separate class.

These depend only on bae_search / new_bae_search (plus torch for the structured
BoltzmannPrior), not on the operators or models.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parametrize

import df_util
import bae_search
import new_bae_search

class ZeroDiag(nn.Module):
    """Parametrization holding J's diagonal at zero (no self-coupling).
    Copied from bae_models.ZeroDiag so this module stays model-independent."""
    def forward(self, X):
        return X.triu(1) + X.tril(-1)

    def right_inverse(self, A):
        return A.triu(1) + A.tril(-1)


def _sigmoid(x):
    """Numerically stable logistic, branch-masked to avoid exp overflow."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _logsumexp(a):
    """Row-wise log-sum-exp over axis 1 (a is (n, m) -> (n,))."""
    am = a.max(1)
    return am + np.log(np.exp(a - am[:, None]).sum(1))


def _categorical(logits):
    """Sample one index per row from softmax(logits), vectorized (Gumbel-max)."""
    g = -np.log(-np.log(np.random.rand(*logits.shape)))
    return np.argmax(logits + g, axis=1)


# ===========================================================================
#  Schedules for the prior temperature
# ===========================================================================
#
# The prior temperature (LatentPrior.temp) is a pure sampling knob -- it divides
# the structured prior's coupling contribution inside the search and touches
# nothing else -- so it can be moved around freely during a fit without changing
# what the prior LEARNS.  A schedule is any object with
#
#     update(**inputs) -> float
#
# which the BMF fit loop calls once per iteration:  prior.temp = sched.update(...).
# The inputs are keyword-only and forward-compatible (take **inputs and ignore what
# you do not need); today the loop passes `it` (iteration), `max_iter`, `temp` (the
# model temperature this iteration) and `loss` (last iteration's, None on the first),
# which is enough for both open-loop and loss-adaptive schedules.

@dataclass
class TempSchedule:
    """The default schedule: prior temperature pinned at 1.0, i.e. the structured
    prior applied at face value for the whole fit.  Subclass and override `update`
    (or use ConstantTemp / GeomAnneal below) for anything else."""

    def update(self, **inputs):
        return 1.0


@dataclass
class ConstantTemp(TempSchedule):
    """A fixed temperature other than 1 -- the structured prior held permanently
    weak (temp > 1) or sharp (temp < 1)."""

    temp: float = 1.0

    def update(self, **inputs):
        return self.temp


@dataclass
class GeomAnneal(TempSchedule):
    """Geometric annealing, the model temperature's schedule applied to the prior:
    temp = min_temp + initial * decay_rate ** (it // period).  Starts the fit with
    the structure switched off (high temp = the coupling barely biases the flips of
    S) and hands it over to the data as the fit cools."""

    initial: float = 10.0
    decay_rate: float = 0.88
    period: int = 10
    min_temp: float = 1.0

    def update(self, it=0, **inputs):
        return self.min_temp + self.initial * (self.decay_rate ** (it // self.period))


# ===========================================================================
#  Priors over the binary latents  (the whole latent side of a model)
# ===========================================================================
#
# A LatentPrior is the model's *latent component*: it owns the latent state and
# everything the E-step needs from the latent side --
#
#   * the regularizer scalars (`sparse_reg`=alpha, `tree_reg`=beta) and the slab
#     rate (`slab_prior`=tau; ignored unless the link is the slab link);
#   * `link`           -- the (score, aux_update) pair the operator compiles into
#     its search; a computed property of the `slab` flag (BINARY_LINK / SLAB_LINK).
#     This is what makes "has a slab" just `slab=True` -- no separate component.
#   * `init_latents`   -- builds the spike S, its StS, and the effective latent Z
#     (Z == S for a plain prior) from the operator's drive;
#   * `couple(WtW, S)` -- the recurrence the search sees (identity here; a
#     structured prior folds in its coupling the way JBMF adds `WtW + J`).
#
# The state lives here: self.S (binary spike), self.StS, self.Z (effective
# continuous latent the operator reads).  A new shared regularizer is added here
# once and is then available to every model.

@dataclass
class LatentPrior:
    """Independent Bernoulli prior: sparsity + tree (hierarchy) penalty, and the
    latent component owning S / StS / Z.  `slab=True` turns ANY prior (this one
    or a structured subclass) into a spike-and-slab variant -- the slab is
    orthogonal to the prior's structure, so it is a flag here, not a subclass."""

    sparse_reg: float = 0.0      # alpha in the kernels
    tree_reg: float = 1e-2       # beta  in the kernels
    slab: bool = False           # spike-and-slab?  picks the link + effective Z
    slab_prior: float = 1.0      # tau; only read when slab is True
    temp: float = 1.0            # prior temperature (annealed separately from the
                                 # model temp, see TempSchedule).  It is a SAMPLING
                                 # knob only: the sole thing it touches is the search,
                                 # which divides the coupling's log-odds contribution
                                 # by it (high temp = weak structure when drawing S).
                                 # The prior's params are NOT rescaled by it -- J is
                                 # learned from the spikes at face value -- so the
                                 # temperature cannot bias what the coupling learns.
                                 # Inert for a plain prior.

    # The search link is a *computed* property of the slab flag, not a class
    # attribute -- so a structured subclass (BoltzmannPrior) inherits slab support
    # for free, and there is no __post_init__ to collide with the subclass's own.
    @property
    def link(self):
        return (new_bae_search.SLAB_LINK if self.slab
                else new_bae_search.BINARY_LINK)

    # The additive S-prior the search injects (sparsity + tree here; a structured
    # subclass overrides this to add its coupling).  The seam mirrors `link`: the
    # operator owns the scaffold, the slab owns the link, the prior owns this.
    @property
    def prior_plugin(self):
        return new_bae_search.PRIOR_PLAIN

    # ---- parameter init (the prior's own params, e.g. a coupling J) -------
    def init_params(self, S0, **opt_args):
        """Hook: build the prior's parameters + optimizer.  No-op by default;
        BoltzmannPrior builds its J here.  Called from the model's init_params."""
        return None

    # ---- latent state + its initialization (the "latent side" of init) ----
    def init_latents(self, drive):
        """Build the spike from the sign of the operator's drive, plus StS and
        the effective latent Z that the operator reads.  With a slab Z is the
        rectified continuous multiplier (drive*S, zero off-spike, so it doubles
        as the effective S*Z); without, Z is the spike itself.  State on `self`."""
        self.S = 1.0 * (drive >= 0)
        self.StS = self.S.T @ self.S
        self.Z = (drive * self.S) if self.slab else self.S

        self.init_params(self.S)

    # ---- what the E-step pulls from the latent side -----------------------
    def coupling(self):
        """The structured prior's coupling Jc (m,m) and field hc (m,) that the
        search adds to the spike log-odds -- ADDITIVELY and linear in the binary
        spikes (sum_k Jc[j,k] S_ik + hc[j]), never folded into WtW.  No structure
        here, so zeros (the search's coupling loop then contributes nothing)."""
        m = self.S.shape[1]
        return np.zeros((m, m)), np.zeros(m)

    def learn(self, ES):
        """Hook: one update of the prior's parameters in the M-step (e.g. the
        inverse-Ising step on J).  No-op by default."""
        return None


@dataclass
class BoltzmannPrior(LatentPrior):
    """Structured (pairwise/Ising) prior -- the `J` coupling lifted out of JBMF.

    Attaching `latent_prior=BoltzmannPrior(...)` to *any* LinearGaussianBMF gives
    it JBMF's structured prior without touching the model class.  `couple` folds
    the Ising coupling into the search recurrence (WtW + J), and `learn` fits J by
    inverse-Ising pseudolikelihood (RPLE softplus, or log-RISE).  Inherits the
    `slab` flag, so BoltzmannPrior(slab=True) is a structured spike-and-slab.

    Ported bit-for-bit from bae_models.JBMF (init_params / EStep / MStep); the J
    update factors out of the decoder's M-step because the loss is separable in
    (W, J), so it is a self-contained SGD step here.
    """

    J_l1_reg: Optional[float] = None
    J_loss: str = 'rple'          # 'rple' (softplus) | 'logrise'
    J_lr: float = 1e-2

    # ---- the prior's parameters: the coupling J (bae_models.JBMF.init_params) --
    def init_params(self, S0, **opt_args):

        n, dim_hid = S0
        if self.J_l1_reg is None:                        # JBMF:959-964
            self.J_l1_reg = np.sqrt(np.log(dim_hid ** 2 / 1e-3) / n)
            self.J_l1_reg *= 0.2 if self.J_loss == 'rple' else 0.8

        self.J = nn.Linear(dim_hid, dim_hid)             # JBMF:975-978
        self.J.weight.data.copy_(torch.zeros(dim_hid, dim_hid))
        self.J.bias.data.copy_(torch.zeros(dim_hid))
        parametrize.register_parametrization(self.J, "weight", ZeroDiag())

        self.optimizer = optim.SGD(self.J.parameters(), lr=self.J_lr, **opt_args)

    @property
    def prior_plugin(self):
        return new_bae_search.PRIOR_BOLTZMANN

    # ---- the structured prior the search adds (bae_models.JBMF.EStep) -------
    # The old binary code folded `Jbin` into WtW; the slab needs it ADDITIVE and
    # on the spikes, so we split the SAME Jbin into (Jc, hc) by the binary
    # equivalence: the embedded prior term was (-sum_k Jbin[j,k] S_ik - 0.5
    # Jbin[j,j]), i.e. coupling Jc[j,k] = -Jbin[j,k] (off-diagonal) and field
    # hc[j] = -0.5 Jbin[j,j].  No /sigma2: the J is learned from the spikes alone
    # (`learn` never sees sigma2), so the prior is applied at face value.
    def coupling(self):
        with torch.no_grad():
            J = self.J.weight.detach().numpy()
            h = self.J.bias.detach().numpy()
        Jsym = (J + J.T) / 2                              # JBMF:1006-1007
        Jbin = 4 * Jsym + 2 * np.diag(h - 2 * Jsym.sum(1))
        Jc = -Jbin.copy()
        np.fill_diagonal(Jc, 0.0)
        hc = -0.5 * np.diag(Jbin).copy()
        return Jc, hc

    # ---- inverse-Ising pseudolikelihood step on J (bae_models.JBMF.MStep) --
    def learn(self, ES):
        Spt = torch.FloatTensor(ES)
        Srbm = 2 * Spt - 1                               # JBMF:1028
        self.optimizer.zero_grad()
        pred = self.J(Srbm)                              # JBMF:1045
        if self.J_loss == 'logrise':
            loss = torch.sum(torch.logsumexp(-pred * Srbm, 1)) / len(ES)
        else:                                            # 'rple'
            loss = torch.sum(nn.Softplus()(-2 * pred * Srbm)) / len(ES)
        loss = loss + self.J_l1_reg * torch.sum(torch.abs(self.J.weight))
        loss.backward()
        self.optimizer.step()


@dataclass
class BoltzmannPriorNP(BoltzmannPrior):
    """Pure-numpy BoltzmannPrior: the SAME structured prior, but the inverse-Ising
    step is a hand-derived gradient + manual SGD -- no torch, hence no per-
    iteration numpy<->torch round-trip.  Pair it with a numpy operator
    (AffineOperator) for a fully-numpy structured-prior model.

    The coupling lives in plain arrays J_W (zero-diagonal, == JBMF's J.weight) and
    J_h (== bias).  The prior is the spin Ising model P(s) ~ exp(0.5 s'.J_sym.s +
    h'.s) with spins s = 2*S-1 and J_sym = (J_W + J_W.T)/2.  A single `_grads`
    computes the data-fit gradient for the chosen J_loss; the L1 on the coupling
    and the zero-diagonal projection are then applied once, for every mode:

      'rple'    : G = (-2/n) s * sigmoid(-2 pred s)           [pseudolikelihood]
      'logrise' : G = (-1/n) s * softmax_features(-pred s)    [pseudolikelihood]
                  with pred = s @ J_W.T + J_h, then dL/dJ_W = G.T @ s,
                  dL/dJ_h = G.sum(0).  (J_W is in general asymmetric here.)
      'mle'     : the true maximum-likelihood gradient (moment matching),
                  dL/dJ_W = <ss'>_model - <ss'>_data,  dL/dJ_h = <s>_model - <s>_data.
                  Data moments are exact; model moments come from persistent
                  particles (advanced `mle_gibbs_steps` sweeps per learn -- PCD),
                  drawn by `sampler`: 'gibbs' (df_util.gibbs in spins) or 'gwg'
                  (gibbs-with-gradients -- gradient-informed single-flip proposals,
                  q(i|s) = softmax(-2 s*(Js+h) / gwg_temp), with MH correction).
                  J_W stays symmetric under the MLE updates, so the sampler uses it
                  directly.  A `sample` method exposes the sampler; self.model_mean
                  / self.model_cov hold the maintained moments.

    All modes then add  J_l1_reg * sign(J_W)  to dL/dJ_W (so L1 works for MLE too).
    The pseudolikelihood gradients are verified against torch autograd, and the
    mle sampler + moments against exact enumeration, in test_boltzmann_*.py.
    """

    mle_n_samp: int = 100         # persistent fantasy particles for the model moments
    mle_gibbs_steps: int = 1      # sampler sweeps per learn (PCD)
    sampler: str = 'gibbs'        # 'gibbs' | 'gwg' (gibbs-with-gradients)
    gwg_temp: float = 2.0         # GWG proposal temperature
    # `temp` (the prior temperature) is inherited from LatentPrior and is NOT read
    # anywhere in this class: it is handed to the search (which divides this prior's
    # coupling log-odds by it) and nowhere else, so it tempers the sampling of S
    # only.  J_W / J_h are used at face value by _grads (learning) and _advance (the
    # fantasy particles), which keeps the coupling that is LEARNED independent of the
    # temperature at which S is drawn.  Put it on a schedule with TempSchedule.

    def init_params(self, S0, **opt_args):
        n, dim_hid = S0.shape
        if self.J_l1_reg is None:
            self.J_l1_reg = np.sqrt(np.log(dim_hid ** 2 / 1e-3) / n)
            if self.J_loss == 'mle':
                self.J_l1_reg = 0.0                       # no default sparsity for MLE
            elif self.J_loss == 'rple':
                self.J_l1_reg *= 0.2 
            else:
                self.J_l1_reg *= 0.8
        else:
            self.J_l1_reg *= np.sqrt(np.log(dim_hid ** 2 / 1e-3) / n)

        self.J_W = np.zeros((dim_hid, dim_hid))   # zero diagonal, kept so by ZeroDiag
        self.J_h = np.zeros(dim_hid)
        if self.J_loss == 'mle':                          # persistent particles (spins)
            self.sigma = np.random.choice([-1.0, 1.0], size=(self.mle_n_samp, dim_hid))
            self.model_mean = self.sigma.mean(0)
            self.model_cov = np.cov(self.sigma, rowvar=False)

    # (prior_plugin == PRIOR_BOLTZMANN inherited from BoltzmannPrior.)  Same split
    # of the folded Jbin into the additive (Jc, hc), now off the numpy J_W / J_h.
    def coupling(self):
        Jsym = (self.J_W + self.J_W.T) / 2
        Jbin = 4 * Jsym + 2 * np.diag(self.J_h - 2 * Jsym.sum(1))
        Jc = -Jbin.copy()
        np.fill_diagonal(Jc, 0.0)
        hc = -0.5 * np.diag(Jbin).copy()
        return Jc, hc

    # ---- one gradient routine for all three J_loss modes -------------------
    def _grads(self, ES, beta=1):
        """(dL/dJ_W, dL/dJ_h) for the current J_loss.  The L1 on the coupling and
        the ZeroDiag (zero-diagonal) projection are applied once, for every mode,
        at the end -- so L1 regularization is available to MLE as well."""
        if self.J_loss == 'mle':
            # maximum likelihood: match the data and model moments of the Ising
            # prior.  Model moments come from the persistent Gibbs particles.
            s_d = 2.0 * ES - 1.0
            m_d, C_d = s_d.mean(0), s_d.T @ s_d / len(ES)
            m_m, C_m = self._model_moments()
            gW, gh = C_m - C_d, m_m - m_d                 # <..>_model - <..>_data
        else:
            # pseudolikelihood: rple (softplus) or logrise (logsumexp)
            Srbm = 2.0 * ES - 1.0
            pred = Srbm @ self.J_W.T + self.J_h
            if self.J_loss == 'logrise':
                B = -pred * Srbm
                P = np.exp(B - B.max(1, keepdims=True))
                P /= P.sum(1, keepdims=True)              # softmax over features
                G = (-1.0 / len(ES)) * Srbm * P
            else:                                         # 'rple'
                G = (-2.0 / len(ES)) * Srbm * _sigmoid(-2.0 * pred * Srbm)
            gW, gh = G.T @ Srbm, G.sum(0)

        gW = gW + self.J_l1_reg * np.sign(self.J_W)       # shared L1 on the coupling
        np.fill_diagonal(gW, 0.0)                         # ZeroDiag parametrization
        return gW, gh

    def learn(self, ES):
        gW, gh = self._grads(ES)
        self.J_W -= self.J_lr * gW                        # manual SGD step
        self.J_h -= self.J_lr * gh

    # ---- sampling the Ising prior for the MLE model moments ---------------
    # The MLE keeps J_W symmetric (it starts at 0 and every update -- the
    # symmetric moment difference, the L1 sign, the diagonal zeroing -- preserves
    # symmetry), so the samplers use J_W directly with no symmetrisation.
    def _sweep(self, sig, J, h):
        """One in-place Gibbs sweep over spins sig (n_samp, m): node i gets
        s_i = +1 w.p. sigmoid(2*(J[i].s + h_i)) (= df_util.gibbs in spins)."""
        for i in range(len(h)):
            p = _sigmoid(2.0 * (sig @ J[i] + h[i]))
            sig[:, i] = np.where(np.random.rand(sig.shape[0]) < p, 1.0, -1.0)
        return sig

    def _gwg_flip(self, sig, field, J, temp):
        """One gibbs-with-gradients step (Grathwohl et al. 2021, sec 4): a single
        gradient-informed flip per particle with a Metropolis-Hastings correction.

        The local field grad f(s) = J s + h is the only matrix-vector product, and
        the per-flip change estimate d_i = -2 s_i (Js+h)_i is *exact* here (f is
        quadratic with zero diagonal), so the proposal is q(i|s) = softmax(d/temp).
        `field` (= J s + h) is passed in and updated in place after accepted flips
        (a rank-1 correction), so each flip costs O(n*m), not another matmul."""
        n, m = sig.shape
        ar = np.arange(n)
        d = -2.0 * sig * field                        # exact Df for flipping each i
        logits_f = d / temp
        i = _categorical(logits_f)                    # proposed coord per particle
        lp_f = logits_f[ar, i] - _logsumexp(logits_f)

        delta = -2.0 * sig[ar, i]                     # s'_i - s_i  (the flip)
        field_p = field + delta[:, None] * J[i]       # field after the flip (rank-1)
        prop = sig.copy(); prop[ar, i] = -sig[ar, i]
        d_p = -2.0 * prop * field_p
        logits_r = d_p / temp
        lp_r = logits_r[ar, i] - _logsumexp(logits_r)

        la = d[ar, i] + lp_r - lp_f                   # Df(exact) + log q(i|s') - log q(i|s)
        acc = np.log(np.random.rand(n)) < la
        sig[acc] = prop[acc]
        field[acc] = field_p[acc]
        return sig

    def _advance(self, sig, n_sweeps):
        """Advance spins `sig` by n_sweeps sweep-equivalents of the chosen sampler
        (a GWG sweep = m gradient-informed flips, comparable to a Gibbs sweep).
        The coupling is used at face value -- the prior temperature never enters
        here, so the fantasy particles are drawn from the SAME Ising model the MLE
        gradient is matching moments of."""
        if self.sampler == 'gwg':
            field = sig @ self.J_W + self.J_h         # grad f; recompute (J changed)
            for _ in range(n_sweeps * sig.shape[1]):
                self._gwg_flip(sig, field, self.J_W, self.gwg_temp)
        else:
            for _ in range(n_sweeps):
                self._sweep(sig, self.J_W, self.J_h)
        return sig

    # def sample(self, n_samp=1, burn=10):
    #     """Samples (in {0,1}) from the prior alone, fresh random init -- the
    #     standalone analogue of df_util.gibbs (uses the chosen sampler)."""
    #     sig = np.random.choice([-1.0, 1.0], size=(n_samp, len(self.J_h)))
    #     self._advance(sig, burn)
    #     return (sig + 1.0) / 2.0
    def sample(self, n_samp=1, **gibbs_args):
        """Draw {0,1} samples from each chain's prior, via df_util.gibbs.

        Returns (C, n_samp, m).  Chains loop in Python -- gibbs is 2-D per chain."""
        J, h = self.coupling()
        return df_util.gibbs(J, h, n_samp=n_samp, **gibbs_args).T

    def _model_moments(self):
        """Advance the persistent fantasy particles `mle_gibbs_steps` sweeps and
        return (mean, <ss'>); also stash the maintained model mean/cov."""
        self._advance(self.sigma, self.mle_gibbs_steps)
        m = self.sigma.mean(0)
        C = self.sigma.T @ self.sigma / len(self.sigma)
        self.model_mean, self.model_cov = m, C - np.outer(m, m)
        return m, C


