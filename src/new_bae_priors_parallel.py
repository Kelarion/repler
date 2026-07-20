"""
new_bae_priors_parallel.py  --  PROTOTYPE (parallel chains)
===========================================================

Parallel-chains variant of new_bae_priors.LatentPrior.  The whole latent side is
identical to the serial one except that every state array carries a leading chain
axis C, and the two numpy operations that build it (the sign-of-drive spike and
S^T S) become chain-batched einsums -- no Python loop over chains.

  S    (C, n, m)     per-chain binary spike
  Z    (C, n, m)     per-chain effective latent (== S with no slab)
  StS  (C, m, m)     per-chain S^T S

Scope of the prototype: the plain, no-slab LatentPrior (which backs SemiBMF).
`slab=True` and the structured BoltzmannPrior are straightforward extensions and
are sketched at the bottom -- the only real question there is per-chain J, which
is just another leading axis on J_W / J_h (C, m, m) / (C, m).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import new_bae_search_parallel as nbsp
import new_bae_priors as nbp                 # serial priors: collapse (to_serial) targets
from new_bae_priors import _sigmoid          # reuse the verified stable logistic
import mrf_samplers as mrf                   # sign-constrained Ising kernels (numba)
import df_util                               # annealed Gibbs sampler (df_util.gibbs)


@dataclass
class ParallelLatentPrior:
    """Independent Bernoulli prior (sparsity + tree), replicated across C chains.
    Same contract as new_bae_priors.LatentPrior -- link / prior_plugin / init /
    coupling / learn -- with a leading chain axis on the state."""

    n_chains: int = 8
    sparse_reg: float = 0.0
    tree_reg: float = 1e-2
    slab: bool = False           # prototype: no-slab only (see note at bottom)
    slab_prior: float = 1.0
    temp: float = 1.0            # prior temperature (shared across chains): a pure
                                 # sampling knob, handed to the search to scale the
                                 # structured coupling's log-odds and read nowhere
                                 # else (the coupling is LEARNED at face value).
                                 # Inert for a plain prior, but threaded so the kernel
                                 # API matches the serial one
                                 # (new_bae_priors.LatentPrior.temp).

    @property
    def link(self):
        # same choice as the serial prior; the parallel scaffold takes the same
        # (score, aux_update) pair (imported straight through from new_bae_search).
        return nbsp.SLAB_LINK if self.slab else nbsp.BINARY_LINK

    @property
    def prior_plugin(self):
        return nbsp.PRIOR_PLAIN

    def init_params(self, S0, **opt_args):
        return None

    # ---- latent state, chain-batched --------------------------------------
    def init_latents(self, drive):
        """`drive` is the operator's per-chain drive, shape (C, n, m) -- distinct
        across chains because each chain has its own W, which is exactly what makes
        the chains explore different basins.  Spike / StS / Z are the serial code
        with a `c` einsum axis."""
        self.S = 1.0 * (drive >= 0)                       # (C, n, m)
        self.StS = self.S.transpose(0, 2, 1) @ self.S     # (C, m, m), batched BLAS
        self.Z = (drive * self.S) if self.slab else self.S
        self.init_params(self.S)

    def coupling(self):
        """Zeros for the plain prior; the scaffold's coupling loop then adds
        nothing.  Chain-axis shapes so the kernel can index Jc[c] / hc[c]."""
        C, _, m = self.S.shape
        return np.zeros((C, m, m)), np.zeros((C, m))

    def learn(self, ES):
        return None

    # ---- collapse to the winning chain's serial prior ----------------------
    def to_serial(self, c):
        """Return chain c as an ordinary new_bae_priors.LatentPrior (2-D state)."""
        lp = nbp.LatentPrior(sparse_reg=self.sparse_reg, tree_reg=self.tree_reg,
                             slab=self.slab, slab_prior=self.slab_prior, temp=self.temp)
        lp.S, lp.Z, lp.StS = self.S[c].copy(), self.Z[c].copy(), self.StS[c].copy()
        return lp


@dataclass
class ParallelBoltzmannPriorNP(ParallelLatentPrior):
    """Parallel-chains port of new_bae_priors.BoltzmannPriorNP -- the structured
    (pairwise/Ising) prior, one coupling PER CHAIN.  Backs ParallelBiPCA's J_lr>0
    path exactly as BoltzmannPriorNP backs the serial BiPCA's.

    Each chain fits the prior to ITS OWN spike configuration S[c], so the coupling
    carries a leading chain axis:

      J_W  (C, m, m)   per-chain coupling (zero diagonal, kept so by ZeroDiag)
      J_h  (C, m)      per-chain field

    Everything is the serial numpy math with a `c` axis, no Python loop over chains
    in the hot path (the Gibbs sweep loops over the m nodes, not the chains):
      * coupling()  -- the additive (Jc, hc) split, batched -> (C,m,m), (C,m); the
        kernel already reads Jc[c] / hc[c] per chain, so no kernel change.
      * learn(ES)   -- one inverse-Ising gradient step, batched over chains.  The
        pseudolikelihood modes ('rple'/'logrise') are exact (no sampler); 'mle'
        matches data/model moments with persistent per-chain Gibbs particles (PCD).
    The pseudolikelihood gradients are the same expressions verified against torch
    autograd in the serial test_boltzmann_*.py, here evaluated for all C chains at
    once.  (GWG sampling is not ported -- BiPCA hardcodes sampler='gibbs'.)"""

    J_l1_reg: Optional[float] = None
    J_loss: str = 'rple'           # 'rple' (softplus) | 'logrise' | 'mle'
    J_lr: float = 1e-2
    mle_n_samp: int = 100          # persistent fantasy particles per chain (MLE)
    mle_gibbs_steps: int = 1       # sampler sweeps per learn (PCD)
    sampler: str = 'gibbs'         # 'gibbs' only (gwg unported; BiPCA uses gibbs)

    @property
    def prior_plugin(self):
        return nbsp.PRIOR_BOLTZMANN

    # ---- the prior's params: per-chain coupling (BoltzmannPriorNP.init_params) --
    def init_params(self, S0, **opt_args):
        C, n, m = S0.shape                                # S0 is the (C,n,m) spike
        self.m = m
        self._diag = np.arange(m)                         # for per-chain diagonal ops
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

        self.J_W = np.zeros((C, m, m))                    # per-chain, zero diagonal
        self.J_h = np.zeros((C, m))
        if self.J_loss == 'mle':                          # persistent particles (spins)
            self.sigma = np.random.choice(
                [-1.0, 1.0], size=(C, self.mle_n_samp, m))
            self._stash_moments()

    # ---- the structured prior the search adds, batched (BoltzmannPriorNP.coupling)
    # Same folded-Jbin -> additive (Jc, hc) split as the serial code, per chain.
    def coupling(self):
        Jsym = (self.J_W + self.J_W.transpose(0, 2, 1)) / 2          # (C,m,m)
        Jbin = 4 * Jsym + 2 * self._diag_embed(self.J_h - 2 * Jsym.sum(2))
        Jc = -Jbin.copy()
        Jc[:, self._diag, self._diag] = 0.0
        hc = -0.5 * Jbin[:, self._diag, self._diag].copy()          # (C,m)
        return Jc, hc

    def _diag_embed(self, v):
        """(C, m) -> (C, m, m) with v on each chain's diagonal."""
        out = np.zeros(v.shape[:1] + (self.m, self.m))
        out[:, self._diag, self._diag] = v
        return out

    # ---- one inverse-Ising gradient step for all C chains ------------------
    def _grads(self, ES):
        """(dL/dJ_W, dL/dJ_h) per chain for the current J_loss.  ES is (C,n,m).
        L1 on the coupling + the zero-diagonal projection are applied once at the
        end, for every mode (so L1 works for MLE too), exactly as the serial one."""
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
        self.J_W -= self.J_lr * gW                                  # manual SGD
        self.J_h -= self.J_lr * gh

    # ---- sampling the per-chain Ising prior for the MLE model moments ------
    # J_W stays symmetric under the MLE updates (starts at 0; the symmetric moment
    # difference, the L1 sign and the diagonal zeroing all preserve symmetry), so
    # the Gibbs sweep uses J_W directly.  The sweep loops over the m nodes; the chain
    # axis is broadcast inside each node's local-field product.
    def _sweep(self, sig, J, h):
        """One in-place Gibbs sweep over per-chain spins sig (C, n_samp, m):
        node i gets s_i = +1 w.p. sigmoid(2*(J[c,i].s + h[c,i])), all chains at once."""
        C, ns, m = sig.shape
        for i in range(m):
            field = np.einsum('csk,ck->cs', sig, J[:, i, :]) + h[:, i, None]
            p = _sigmoid(2.0 * field)                              # (C, n_samp)
            flip = np.random.rand(C, ns) < p
            sig[:, :, i] = np.where(flip, 1.0, -1.0)
        return sig

    def _advance(self, sig, n_sweeps):
        """The coupling is used at face value: the prior temperature tempers the
        sampling of S in the search only, never the particles the MLE gradient
        matches moments of (new_bae_priors.BoltzmannPriorNP._advance)."""
        if self.sampler != 'gibbs':
            raise NotImplementedError(
                "ParallelBoltzmannPriorNP only ports the 'gibbs' sampler "
                "(BiPCA hardcodes it); 'gwg' is left as an extension.")
        for _ in range(n_sweeps):
            self._sweep(sig, self.J_W, self.J_h)
        return sig

    def _model_moments(self):
        """Advance the persistent per-chain particles and return (mean, <ss'>)."""
        self._advance(self.sigma, self.mle_gibbs_steps)
        m, C = self._stash_moments()
        return m, C

    def _stash_moments(self):
        ns = self.sigma.shape[1]
        m = self.sigma.mean(1)                                     # (C,m)
        C = self.sigma.transpose(0, 2, 1) @ self.sigma / ns        # (C,m,m) = <ss'>
        self.model_mean = m
        self.model_cov = C - m[:, :, None] * m[:, None, :]
        return m, C

    # def sample(self, n_samp=1, burn=10):
    #     """Draw {0,1} samples from each chain's prior (fresh random init)."""
    #     C, m = self.J_h.shape
    #     sig = np.random.choice([-1.0, 1.0], size=(C, n_samp, m))
    #     self._advance(sig, burn)
    #     return (sig + 1.0) / 2.0
    def sample(self, n_samp=1, **gibbs_args):
        """Draw {0,1} samples from each chain's prior, via df_util.gibbs.

        Returns (C, n_samp, m).  Chains loop in Python -- gibbs is 2-D per chain."""
        Jc, hc = self.coupling()
        return np.stack([df_util.gibbs(Jc[c], hc[c], temp=self.temp,
                                       n_samp=n_samp, **gibbs_args).T
                         for c in range(len(self.J))])

    # ---- collapse to the winning chain's serial prior ----------------------
    def to_serial(self, c):
        """Return chain c as an ordinary new_bae_priors.BoltzmannPriorNP: the 2-D
        latent state and that chain's learned coupling J_W / J_h (and MLE particles)."""
        lp = nbp.BoltzmannPriorNP(
            sparse_reg=self.sparse_reg, tree_reg=self.tree_reg, slab=self.slab,
            slab_prior=self.slab_prior, temp=self.temp, J_l1_reg=self.J_l1_reg,
            J_loss=self.J_loss, J_lr=self.J_lr, mle_n_samp=self.mle_n_samp,
            mle_gibbs_steps=self.mle_gibbs_steps, sampler=self.sampler)
        lp.S, lp.Z, lp.StS = self.S[c].copy(), self.Z[c].copy(), self.StS[c].copy()
        lp.J_W, lp.J_h = self.J_W[c].copy(), self.J_h[c].copy()
        if self.J_loss == 'mle':
            lp.sigma = self.sigma[c].copy()
            lp.model_mean = self.model_mean[c].copy()
            lp.model_cov = self.model_cov[c].copy()
        return lp


@dataclass
class MRFPrior(ParallelLatentPrior):
    """Sign-constrained parallel-chains Ising prior, fit by the annealed Gibbs/ICM
    sampler in mrf_samplers.  

    State (per chain, one leading chain axis as everywhere in this module):

      J  (C, m, m)   symmetric, zero diagonal, entries in {-1, 0, +1}
      h  (C, m)      unconstrained field

    Inherits from ParallelLatentPrior: the latent state (S / Z / StS) and its init,
    the `link` property (so slab=True works here for free), sparse_reg / tree_reg,
    and `temp`.  Overrides prior_plugin (-> PRIOR_BOLTZMANN, the coupling kernel),
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
    h_l2_reg: float = 1e-2           # mrf's lam_h (0.5*lam_h*h^2 ridge).  
    h_newton_steps: int = 3         # mrf's n_steps (both fitters)
    beta_mle: bool = True
    beta_lr: float = 1e-3
    beta_bsz: int = 100             # monte carlo batch size for beta gradient

    @property
    def prior_plugin(self):
        return nbsp.PRIOR_BOLTZMANN      # the coupling kernel; see coupling()

    # ---- the prior's params: the discrete coupling ------------------------
    def init_params(self, S0, **opt_args):
        C, n, m = S0.shape                                # S0 is the (C,n,m) spike
        self.m = m
        # init_params is re-run on every init_latents, hence on every restart of a
        # multi-start fit -- so J_lam is rebuilt from the stashed multiplier rather
        # than scaled in place (`self.J_lam *= scale` would compound per restart, as
        # BoltzmannPriorNP's J_l1_reg does).
        self.J_lam = self.l0_reg * np.sqrt(np.log(m ** 2 / 1e-3) / n)  # mrf's lam

        self.J = np.zeros((C, m, m))                      # {-1,0,+1}, zero diagonal
        self.h = np.zeros((C, m))
        self.beta = np.ones(C)*self.beta_init
        self._n_changed = np.zeros(C, dtype=int)          # last sweep's #flips

    # ---- the structured prior the search adds ------------------------------
    def coupling(self):
        """The (Jc, hc) that make new_bae_search.prior_boltzmann reproduce THIS
        prior's conditional log-odds, beta*(J.sigma + h)."""
        Jc = 2.0 * self.beta[:,None,None] * self.J            # (C,m,m); symmetric, zero diagonal
        hc = self.beta[:,None] * (self.h - self.J.sum(2))  # (C,m); field + the 2S-1 shift
        return Jc, hc

    # ---- one annealed sampler pass over J (and h), per chain ---------------
    # No chain batching here: the mrf kernels are numba loops over the m nodes and
    # M samples, so each chain is a separate call.  The Python loop is over C only
    # (C is ~8), and each call is O(m^2 n) inside numba.
    def learn(self, ES):
        """Resample each chain's discrete coupling from the spike statistics ES
        (C, n, m).  The fields F are rebuilt per call because ES changes every
        M-step (build_fields is O(m^2 n), the same order as one sweep)."""
        spins = np.where(ES != 0, 1.0, -1.0)              # spike -> {-1,+1}; `!= 0`

        for c in range(len(self.J)):
            St = np.ascontiguousarray(spins[c].T)         # (m, n), mrf's layout
            F = mrf.build_fields(self.J[c], self.h[c], St)
            for _ in range(self.J_sweeps):
                if self.fit_h:
                    self._fit_h(self.J[c], self.h[c], St, F, self.beta[c])
                self._n_changed[c] = mrf.gibbs_sweep_J(
                    self.J[c], St, F, self.J_lam*self.beta[c], self.beta[c], self.J_temp)

                self.beta[c] = self._fit_beta(self.J[c], self.h[c], St, F, self.beta[c])

                # mod_samps = self.sample(self.beta_bsz)
                # mod_logs = np.mean(util.qform(Jc[0], mod_samps).squeeze() + mod_samps@hc[0])
                # task_logs = np.mean(util.qform(Jc[0], ES).squeeze() + ES@hc[0])

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
        """Per-chain per-sample objective (mean log pseudolikelihood - lam*nnz(J)),
        the monitoring quantity of fit_sign_ising's history."""
        spins = np.where(ES != 0, 1.0, -1.0)
        out = np.empty(len(self.J))
        for c in range(len(self.J)):
            St = np.ascontiguousarray(spins[c].T)
            F = mrf.build_fields(self.J[c], self.h[c], St)
            out[c] = mrf.objective(self.J[c], St, F, self.J_lam, self.beta[c])
        return out

    # ---- sampling the prior itself (its generative model) ------------------
    def sample(self, n_samp=1, **gibbs_args):
        """Draw {0,1} samples from each chain's prior, via df_util.gibbs.

        Returns (C, n_samp, m).  Chains loop in Python -- gibbs is 2-D per chain."""
        samps = []
        for c in range(len(self.J)):
            samps.append(mrf.gibbs_samp(self.beta[c]*self.J[c], self.beta[c]*self.h[c], 
                                        temp=self.temp, n_samp=n_samp, **gibbs_args).T)
        return (1+np.stack(samps))/2

    # ---- collapse to the winning chain's serial prior ----------------------
    def to_serial(self, c):
        """Convert chain c into the equivalent SERIAL prior.  The serial prior has
        no beta, its coupling being continuous and carrying its own scale, so beta
        is folded into J_W / J_h here (the serial spin model exp(0.5 s'J_sym s +
        h's) has conditional log-odds 2*(J_W s + J_h), matching beta*(J s + h) at
        J_W = 0.5*beta*J).
        CAVEAT: the collapsed prior's coupling() is the serial class's, i.e. the old
        JBMF-folding one, so its *E-step* applies -1x this coupling and -0.5x this
        field (see coupling()).  Collapse to inspect / sample / read off J, not to
        keep fitting, until BoltzmannPriorNP.coupling() is reconciled."""
        lp = nbp.BoltzmannPriorNP(
            sparse_reg=self.sparse_reg, tree_reg=self.tree_reg, slab=self.slab,
            slab_prior=self.slab_prior, temp=self.temp, J_l1_reg=self.J_lam,
            sampler='gibbs')
        lp.S, lp.Z, lp.StS = self.S[c].copy(), self.Z[c].copy(), self.StS[c].copy()
        lp.J_W, lp.J_h = 0.5 * self.beta[c] * self.J[c], 0.5 * self.beta[c] * self.h[c]
        return lp


# ---------------------------------------------------------------------------
#  NOTE -- slab, and what is left of the Boltzmann port
# ---------------------------------------------------------------------------
#
#  * slab=True  needs nothing here: init_latents already branches, the link
#    property already returns SLAB_LINK, and the parallel scaffold already threads
#    Z / mu / nu per chain.  The only caveat is that the truncated-normal draw in
#    aux_slab runs inside the prange -- fine, numba's RNG is thread-local.
#
#  * ParallelBoltzmannPriorNP (above) ports the structured prior: per-chain J_W
#    (C,m,m) / J_h (C,m), coupling() and the rple/logrise/mle learn() batched over
#    chains, and persistent per-chain Gibbs particles for MLE.  The E-step needs no
#    new kernel -- prior_boltzmann already reads Jc[c] / hc[c].  Only the GWG
#    sampler is unported (BiPCA hardcodes sampler='gibbs'); batching its
#    _categorical / _logsumexp over a chain axis is the one remaining piece.
