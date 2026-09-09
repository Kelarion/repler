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
import mrf_samplers as mrf              # sign-constrained Ising kernels (numba)

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


@dataclass
class AdaptiveTemp(TempSchedule):
    """Loss-adaptive prior temperature: a control law that drives the model's
    reconstruction MSE toward a target `kappa` by nudging the prior's INVERSE
    temperature lambda = 1/temp,

        lambda <- max(0, lambda + alpha * (err - kappa)),     temp = 1 / lambda
    """

    alpha: float = 1e-3          # learning rate on the inverse temperature
    kappa: float = 1e-3          # target reconstruction MSE
    gamma: float = 0.9           # exponential moving average
    lambda_init: float = 1.0     # initial inverse temperature (temp = 1 / lambda_init)

    def __post_init__(self):
        self._lam = self.lambda_init
        self._D = self.kappa

    def update(self, it=0, loss=None, **inputs):
        if it == 0:
            self._lam = self.lambda_init      # fresh fit -> re-seed the state
            self._D = self.kappa
        if loss is not None:
            self._D = self.gamma*self._D + (1-self.gamma)*(self.kappa - np.array(loss))
            self._lam *= np.exp(self.alpha * self._D)
        return np.where(self._lam <= 1e-7, np.inf, 1/self._lam)
        # return np.inf if self._lam <= 0.0 else 1.0 / self._lam

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
    orthogonal to the prior's structure, so it is a flag here, not a subclass.

    Supports multiple chains via `n_chains` (default 1): a single chain keeps 2-D
    state (S / StS / Z), while n_chains > 1 carries a leading chain axis C on every
    state array -- the same numpy math with one extra einsum index."""

    n_chains: int = 1            # >1 runs C independent chains (leading axis on S)
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

    @property
    def _multi(self):
        return self.n_chains > 1

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
        as the effective S*Z); without, Z is the spike itself.  State on `self`.

        Single chain: drive is (n, m) and the state is 2-D.  Multi-chain: drive is
        the operator's per-chain drive (C, n, m) -- distinct across chains because
        each has its own W, which is exactly what makes the chains explore different
        basins -- and StS is the chain-batched S^T S."""
        self.S = 1.0 * (drive >= 0)
        if self._multi:
            self.StS = self.S.transpose(0, 2, 1) @ self.S     # (C, m, m), batched BLAS
        else:
            self.StS = self.S.T @ self.S
        self.Z = (drive * self.S) if self.slab else self.S

        self.init_params(self.S)

    # ---- what the E-step pulls from the latent side -----------------------
    def coupling(self):
        """The structured prior's coupling Jc (m,m) and field hc (m,) that the
        search adds to the spike log-odds -- ADDITIVELY and linear in the binary
        spikes (sum_k Jc[j,k] S_ik + hc[j]), never folded into WtW.  No structure
        here, so zeros (the search's coupling loop then contributes nothing).
        Multi-chain: chain-axis shapes so the kernel can index Jc[c] / hc[c]."""
        if self._multi:
            C, _, m = self.S.shape
            return np.zeros((C, m, m)), np.zeros((C, m))
        m = self.S.shape[1]
        return np.zeros((m, m)), np.zeros(m)

    def learn(self, ES):
        """Hook: one update of the prior's parameters in the M-step (e.g. the
        inverse-Ising step on J).  No-op by default."""
        return None

    def to_serial(self, c):
        """Return chain c as an ordinary single-chain (n_chains=1) LatentPrior with
        2-D state.  No-op-shaped for a single chain (c == 0)."""
        lp = LatentPrior(sparse_reg=self.sparse_reg, tree_reg=self.tree_reg,
                         slab=self.slab, slab_prior=self.slab_prior, temp=self.temp)
        if self._multi:
            lp.S, lp.Z, lp.StS = self.S[c].copy(), self.Z[c].copy(), self.StS[c].copy()
        else:
            lp.S, lp.Z, lp.StS = self.S.copy(), self.Z.copy(), self.StS.copy()
        return lp


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
        # S0 is the spike: (n, m) for a single chain, (C, n, m) with n_chains > 1.
        # Multi-chain fits ONE coupling per chain (leading axis C on J_W / J_h and
        # the MLE particles); the numpy math below is otherwise the serial code.
        if self._multi:
            C, n, m = S0.shape
            self.m = m
            self._diag = np.arange(m)                     # for per-chain diagonal ops
        else:
            n, m = S0.shape
        if self.J_l1_reg is None:
            self.J_l1_reg = np.sqrt(np.log(m ** 2 / 1e-3) / n)
            if self.J_loss == 'mle':
                self.J_l1_reg = 0.0                       # no default sparsity for MLE
            elif self.J_loss == 'rple':
                self.J_l1_reg *= 0.2
            else:
                self.J_l1_reg *= 0.8
        else:
            self.J_l1_reg *= np.sqrt(np.log(m ** 2 / 1e-3) / n)

        if self._multi:
            self.J_W = np.zeros((C, m, m))                # per-chain, zero diagonal
            self.J_h = np.zeros((C, m))
            if self.J_loss == 'mle':                      # persistent particles (spins)
                self.sigma = np.random.choice(
                    [-1.0, 1.0], size=(C, self.mle_n_samp, m))
                self._stash_moments()
        else:
            self.J_W = np.zeros((m, m))   # zero diagonal, kept so by ZeroDiag
            self.J_h = np.zeros(m)
            if self.J_loss == 'mle':                      # persistent particles (spins)
                self.sigma = np.random.choice([-1.0, 1.0], size=(self.mle_n_samp, m))
                self.model_mean = self.sigma.mean(0)
                self.model_cov = np.cov(self.sigma, rowvar=False)

    # (prior_plugin == PRIOR_BOLTZMANN inherited from BoltzmannPrior.)  Same split
    # of the folded Jbin into the additive (Jc, hc), now off the numpy J_W / J_h.
    def coupling(self):
        if not self._multi:
            Jsym = (self.J_W + self.J_W.T) / 2
            Jbin = 4 * Jsym + 2 * np.diag(self.J_h - 2 * Jsym.sum(1))
            Jc = -Jbin.copy()
            np.fill_diagonal(Jc, 0.0)
            hc = -0.5 * np.diag(Jbin).copy()
            return Jc, hc
        # multi-chain: the same folded-Jbin -> additive (Jc, hc) split, per chain;
        # the kernel reads Jc[c] / hc[c] so no kernel change is needed.
        Jsym = (self.J_W + self.J_W.transpose(0, 2, 1)) / 2          # (C,m,m)
        Jbin = 4 * Jsym + 2 * self._diag_embed(self.J_h - 2 * Jsym.sum(2))
        Jc = -Jbin.copy()
        Jc[:, self._diag, self._diag] = 0.0
        hc = -0.5 * Jbin[:, self._diag, self._diag].copy()          # (C,m)
        return Jc, hc

    def _diag_embed(self, v):
        """(C, m) -> (C, m, m) with v on each chain's diagonal (multi-chain only)."""
        out = np.zeros(v.shape[:1] + (self.m, self.m))
        out[:, self._diag, self._diag] = v
        return out

    # ---- one gradient routine for all three J_loss modes -------------------
    def _grads(self, ES, beta=1):
        """(dL/dJ_W, dL/dJ_h) for the current J_loss.  The L1 on the coupling and
        the ZeroDiag (zero-diagonal) projection are applied once, for every mode,
        at the end -- so L1 regularization is available to MLE as well.  ES is
        (n, m) for a single chain, (C, n, m) with n_chains > 1 (batched over chains,
        the same expressions with a leading `c` axis)."""
        if not self._multi:
            if self.J_loss == 'mle':
                # maximum likelihood: match the data and model moments of the Ising
                # prior.  Model moments come from the persistent Gibbs particles.
                s_d = 2.0 * ES - 1.0
                m_d, C_d = s_d.mean(0), s_d.T @ s_d / len(ES)
                m_m, C_m = self._model_moments()
                gW, gh = C_m - C_d, m_m - m_d             # <..>_model - <..>_data
            else:
                # pseudolikelihood: rple (softplus) or logrise (logsumexp)
                Srbm = 2.0 * ES - 1.0
                pred = Srbm @ self.J_W.T + self.J_h
                if self.J_loss == 'logrise':
                    B = -pred * Srbm
                    P = np.exp(B - B.max(1, keepdims=True))
                    P /= P.sum(1, keepdims=True)          # softmax over features
                    G = (-1.0 / len(ES)) * Srbm * P
                else:                                     # 'rple'
                    G = (-2.0 / len(ES)) * Srbm * _sigmoid(-2.0 * pred * Srbm)
                gW, gh = G.T @ Srbm, G.sum(0)

            gW = gW + self.J_l1_reg * np.sign(self.J_W)   # shared L1 on the coupling
            np.fill_diagonal(gW, 0.0)                     # ZeroDiag parametrization
            return gW, gh

        # multi-chain, batched over the leading axis
        C, n, m = ES.shape
        if self.J_loss == 'mle':
            s_d = 2.0 * ES - 1.0
            m_d = s_d.mean(1)                                       # (C,m)
            C_d = s_d.transpose(0, 2, 1) @ s_d / n                  # (C,m,m)
            m_m, C_m = self._model_moments()
            gW, gh = C_m - C_d, m_m - m_d                           # model - data
        else:
            Srbm = 2.0 * ES - 1.0                                   # (C,n,m)
            pred = Srbm @ self.J_W.transpose(0, 2, 1) + self.J_h[:, None, :]
            if self.J_loss == 'logrise':
                B = -pred * Srbm
                P = np.exp(B - B.max(2, keepdims=True))
                P /= P.sum(2, keepdims=True)                        # softmax over feats
                G = (-1.0 / n) * Srbm * P
            else:                                                   # 'rple'
                G = (-2.0 / n) * Srbm * _sigmoid(-2.0 * pred * Srbm)
            gW = G.transpose(0, 2, 1) @ Srbm                        # (C,m,m)
            gh = G.sum(1)                                           # (C,m)

        gW = gW + self.J_l1_reg * np.sign(self.J_W)                 # shared L1
        gW[:, self._diag, self._diag] = 0.0                        # ZeroDiag
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
        """One in-place Gibbs sweep over spins.  Single chain: sig (n_samp, m),
        node i gets s_i = +1 w.p. sigmoid(2*(J[i].s + h_i)) (= df_util.gibbs in
        spins).  Multi-chain: sig (C, n_samp, m), all chains at once with the chain
        axis broadcast inside each node's local-field product."""
        if not self._multi:
            for i in range(len(h)):
                p = _sigmoid(2.0 * (sig @ J[i] + h[i]))
                sig[:, i] = np.where(np.random.rand(sig.shape[0]) < p, 1.0, -1.0)
            return sig
        C, ns, m = sig.shape
        for i in range(m):
            field = np.einsum('csk,ck->cs', sig, J[:, i, :]) + h[:, i, None]
            p = _sigmoid(2.0 * field)                              # (C, n_samp)
            flip = np.random.rand(C, ns) < p
            sig[:, :, i] = np.where(flip, 1.0, -1.0)
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
            if self._multi:
                raise NotImplementedError(
                    "the gwg sampler is not chain-batched; use sampler='gibbs' "
                    "with n_chains > 1")
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
        """Draw {0,1} samples from the prior, via df_util.gibbs.  Single chain ->
        (n_samp, m); multi-chain -> (C, n_samp, m) (chains loop in Python -- gibbs
        is 2-D per chain)."""
        if not self._multi:
            J, h = self.coupling()
            return df_util.gibbs(J, h, n_samp=n_samp, **gibbs_args).T
        Jc, hc = self.coupling()
        return np.stack([df_util.gibbs(Jc[c], hc[c], temp=self.temp,
                                       n_samp=n_samp, **gibbs_args).T
                         for c in range(len(self.J_W))])

    def _model_moments(self):
        """Advance the persistent fantasy particles `mle_gibbs_steps` sweeps and
        return (mean, <ss'>); also stash the maintained model mean/cov."""
        self._advance(self.sigma, self.mle_gibbs_steps)
        if not self._multi:
            m = self.sigma.mean(0)
            C = self.sigma.T @ self.sigma / len(self.sigma)
            self.model_mean, self.model_cov = m, C - np.outer(m, m)
            return m, C
        return self._stash_moments()

    def _stash_moments(self):
        """Per-chain (mean, <ss'>) of the fantasy particles (multi-chain)."""
        ns = self.sigma.shape[1]
        m = self.sigma.mean(1)                                     # (C,m)
        C = self.sigma.transpose(0, 2, 1) @ self.sigma / ns        # (C,m,m) = <ss'>
        self.model_mean = m
        self.model_cov = C - m[:, :, None] * m[:, None, :]
        return m, C

    def to_serial(self, c):
        """Chain c as an ordinary single-chain (n_chains=1) BoltzmannPriorNP: that
        chain's 2-D latent state and its learned coupling J_W / J_h (and MLE
        particles).  No-op-shaped for an already-serial prior (c == 0)."""
        lp = BoltzmannPriorNP(
            sparse_reg=self.sparse_reg, tree_reg=self.tree_reg, slab=self.slab,
            slab_prior=self.slab_prior, temp=self.temp, J_l1_reg=self.J_l1_reg,
            J_loss=self.J_loss, J_lr=self.J_lr, mle_n_samp=self.mle_n_samp,
            mle_gibbs_steps=self.mle_gibbs_steps, sampler=self.sampler)
        if self._multi:
            lp.S, lp.Z, lp.StS = self.S[c].copy(), self.Z[c].copy(), self.StS[c].copy()
            lp.J_W, lp.J_h = self.J_W[c].copy(), self.J_h[c].copy()
            if self.J_loss == 'mle':
                lp.sigma = self.sigma[c].copy()
                lp.model_mean = self.model_mean[c].copy()
                lp.model_cov = self.model_cov[c].copy()
        else:
            lp.S, lp.Z, lp.StS = self.S.copy(), self.Z.copy(), self.StS.copy()
            lp.J_W, lp.J_h = self.J_W.copy(), self.J_h.copy()
            if self.J_loss == 'mle':
                lp.sigma = self.sigma.copy()
                lp.model_mean = self.model_mean.copy()
                lp.model_cov = self.model_cov.copy()
        return lp


@dataclass
class MRFPrior(LatentPrior):
    """Sign-constrained Ising prior, fit by the annealed Gibbs/ICM sampler in
    mrf_samplers.  Supports multiple chains via `n_chains` (default 1) like the
    other priors: a single chain keeps 2-D state (J (m,m), h (m,), scalar beta),
    and n_chains > 1 carries a leading chain axis.

    State (single chain / multi-chain):

      J  (m,m) / (C,m,m)   symmetric, zero diagonal, entries in {-1, 0, +1}
      h  (m,)  / (C,m)     unconstrained field
      beta scalar / (C,)   the prior's inverse temperature (the coupling's scale)

    Inherits from LatentPrior: the latent state (S / Z / StS) and its init, the
    `link` property (so slab=True works here for free), sparse_reg / tree_reg, and
    `temp`.  Overrides prior_plugin (-> PRIOR_BOLTZMANN, the coupling kernel),
    init_params, coupling, learn, and to_serial; adds objective and sample.

    `beta` and `temp` are separate knobs, and the split matters.  J's entries are
    in {-1,0,+1}, so J carries no scale of its own -- beta IS the scale of the
    coupling, i.e. the model being fit.  temp is the search's prior temperature,
    which the fit loop anneals.  Tying the two (beta = 1/temp) therefore made the
    annealing schedule silently rescale the model, which is why beta is now its
    own field, externally scheduled exactly as in mrf_samplers.
    """

    beta_init: float = 1.0          # the prior's INVERSE TEMPERATURE: the scale of
                                    # the coupling, in mrf_samplers' convention
                                    # (its conditional is sigmoid(beta*f)).  Both
                                    # fit and applied -- see coupling() and learn().
    l0_reg: Optional[float] = 0.3   # per-sample cost (nats) of a nonzero entry
    J_temp: float = 0.0             # SAMPLER temperature over J (mrf's `temp`)
    J_sweeps: int = 1               # gibbs_sweep_J passes per learn() call
    fit_h: bool = True              # fit the field before each J sweep
    h_newton: bool = False          # newton_h instead of grad_h.
    h_lr: float = 1.0               # grad_h's lr, in units of 1/beta^2 (see grad_h)
    h_l2_reg: float = 1e-2          # mrf's lam_h (0.5*lam_h*h^2 ridge).
    h_newton_steps: int = 3         # mrf's n_steps (both fitters)
    beta_mle: bool = True
    beta_lr: float = 1e-3
    beta_bsz: int = 100             # monte carlo batch size for beta gradient

    @property
    def prior_plugin(self):
        return new_bae_search.PRIOR_BOLTZMANN      # the coupling kernel; see coupling()

    # ---- the prior's params: the discrete coupling ------------------------
    def init_params(self, S0, **opt_args):
        # S0 is the spike: (n, m) for a single chain, (C, n, m) with n_chains > 1.
        n, m = S0.shape[-2:]
        self.m = m
        # init_params is re-run on every init_latents, hence on every restart of a
        # multi-start fit -- so J_lam is rebuilt from the stashed multiplier rather
        # than scaled in place (`self.J_lam *= scale` would compound per restart, as
        # BoltzmannPriorNP's J_l1_reg does).
        self.J_lam = self.l0_reg * np.sqrt(np.log(m ** 2 / 1e-3) / n)  # mrf's lam

        if self._multi:
            C = self.n_chains
            self.J = np.zeros((C, m, m))                   # {-1,0,+1}, zero diagonal
            self.h = np.zeros((C, m))
            self.beta = np.ones(C) * self.beta_init
            self._n_changed = np.zeros(C, dtype=int)       # last sweep's #flips
        else:
            self.J = np.zeros((m, m))                      # {-1,0,+1}, zero diagonal
            self.h = np.zeros(m)
            self.beta = float(self.beta_init)              # scalar inverse temperature
            self._n_changed = 0

    # ---- the structured prior the search adds ------------------------------
    def coupling(self):
        """The (Jc, hc) that make new_bae_search.prior_boltzmann reproduce THIS
        prior's conditional log-odds, beta*(J.sigma + h).  Single chain: (m,m)/(m,);
        multi-chain: (C,m,m)/(C,m)."""
        if not self._multi:
            Jc = 2.0 * self.beta * self.J                # (m,m); symmetric, zero diag
            hc = self.beta * (self.h - self.J.sum(1))    # (m,); field + 2S-1 shift
            return Jc, hc
        Jc = 2.0 * self.beta[:, None, None] * self.J     # (C,m,m); symmetric, zero diag
        hc = self.beta[:, None] * (self.h - self.J.sum(2))  # (C,m); field + 2S-1 shift
        return Jc, hc

    # ---- one annealed sampler pass over J (and h) --------------------------
    # No chain batching in the numba layer: the mrf kernels are loops over the m
    # nodes and M samples, so each chain is a separate 2-D call.  A single chain is
    # one call; multi loops over C (which is ~8), each O(m^2 n) inside numba.
    def learn(self, ES):
        """Resample the discrete coupling from the spike statistics ES ((n,m) single
        chain, (C,n,m) multi).  The fields F are rebuilt per call because ES changes
        every M-step (build_fields is O(m^2 n), the same order as one sweep)."""
        spins = np.where(ES != 0, 1.0, -1.0)              # spike -> {-1,+1}; `!= 0`

        if not self._multi:
            St = np.ascontiguousarray(spins.T)            # (m, n), mrf's layout
            F = mrf.build_fields(self.J, self.h, St)
            for _ in range(self.J_sweeps):
                if self.fit_h:
                    self._fit_h(self.J, self.h, St, F, self.beta)
                self._n_changed = mrf.gibbs_sweep_J(
                    self.J, St, F, self.J_lam * self.beta, self.beta, self.J_temp)
                self.beta = self._fit_beta(self.J, self.h, St, F, self.beta)
            return

        for c in range(len(self.J)):
            St = np.ascontiguousarray(spins[c].T)         # (m, n), mrf's layout
            F = mrf.build_fields(self.J[c], self.h[c], St)
            for _ in range(self.J_sweeps):
                if self.fit_h:
                    self._fit_h(self.J[c], self.h[c], St, F, self.beta[c])
                self._n_changed[c] = mrf.gibbs_sweep_J(
                    self.J[c], St, F, self.J_lam * self.beta[c], self.beta[c],
                    self.J_temp)

                self.beta[c] = self._fit_beta(self.J[c], self.h[c], St, F, self.beta[c])

    def _fit_h(self, J, h, St, F, beta):
        """One pass of the field fitter over every node; mutates h and F in place."""
        if self.h_newton:
            mrf.newton_h(J, h, St, F, beta, self.h_l2_reg, self.h_newton_steps)
        else:
            mrf.grad_h(J, h, St, F, beta, self.h_l2_reg, self.h_newton_steps,
                       self.h_lr)

    def _fit_beta(self, J, h, St, F, beta):
        if self.beta_mle:
            return mrf.mle_beta(J, h, beta, St, F,
                                lr=self.beta_lr, n_samp=self.beta_bsz, J_lam=0)
        else:
            return mrf.ple_beta(J, h, beta, St, F, lr=self.beta_lr, J_lam=0)

    def objective(self, ES):
        """Per-sample objective (mean log pseudolikelihood - lam*nnz(J)), the
        monitoring quantity of fit_sign_ising's history.  Single chain -> scalar;
        multi-chain -> (C,)."""
        spins = np.where(ES != 0, 1.0, -1.0)
        if not self._multi:
            St = np.ascontiguousarray(spins.T)
            F = mrf.build_fields(self.J, self.h, St)
            return mrf.objective(self.J, St, F, self.J_lam, self.beta)
        out = np.empty(len(self.J))
        for c in range(len(self.J)):
            St = np.ascontiguousarray(spins[c].T)
            F = mrf.build_fields(self.J[c], self.h[c], St)
            out[c] = mrf.objective(self.J[c], St, F, self.J_lam, self.beta[c])
        return out

    # ---- sampling the prior itself (its generative model) ------------------
    def sample(self, n_samp=1, **gibbs_args):
        """Draw {0,1} samples from the prior, via mrf_samplers.gibbs_samp.  Single
        chain -> (n_samp, m); multi-chain -> (C, n_samp, m) (chains loop in Python,
        gibbs is 2-D per chain)."""
        if not self._multi:
            g = mrf.gibbs_samp(self.beta * self.J, self.beta * self.h,
                               temp=self.temp, n_samp=n_samp, **gibbs_args).T
            return (1 + g) / 2
        samps = []
        for c in range(len(self.J)):
            samps.append(mrf.gibbs_samp(self.beta[c] * self.J[c], self.beta[c] * self.h[c],
                                        temp=self.temp, n_samp=n_samp, **gibbs_args).T)
        return (1 + np.stack(samps)) / 2

    # ---- collapse to the winning chain's serial prior ----------------------
    def to_serial(self, c):
        """Convert chain c into the equivalent SERIAL prior.  The serial prior has
        no beta, its coupling being continuous and carrying its own scale, so beta
        is folded into J_W / J_h here (the serial spin model exp(0.5 s'J_sym s +
        h's) has conditional log-odds 2*(J_W s + J_h), matching beta*(J s + h) at
        J_W = 0.5*beta*J).
        CAVEAT: the collapsed prior's coupling() is BoltzmannPriorNP's (the old
        JBMF-folding one), so its *E-step* applies -1x this coupling and -0.5x this
        field.  Collapse to inspect / sample / read off J, not to keep fitting,
        until BoltzmannPriorNP.coupling() is reconciled."""
        lp = BoltzmannPriorNP(
            sparse_reg=self.sparse_reg, tree_reg=self.tree_reg, slab=self.slab,
            slab_prior=self.slab_prior, temp=self.temp, J_l1_reg=self.J_lam,
            sampler='gibbs')
        if self._multi:
            lp.S, lp.Z, lp.StS = self.S[c].copy(), self.Z[c].copy(), self.StS[c].copy()
            lp.J_W, lp.J_h = 0.5 * self.beta[c] * self.J[c], 0.5 * self.beta[c] * self.h[c]
        else:
            lp.S, lp.Z, lp.StS = self.S.copy(), self.Z.copy(), self.StS.copy()
            lp.J_W, lp.J_h = 0.5 * self.beta * self.J, 0.5 * self.beta * self.h
        return lp


