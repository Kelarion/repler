"""
new_bae_weights.py  --  ILLUSTRATION ONLY (design sketch)
=========================================================

The *weight side* of the refactored BMF stack (see new_bae_models.py for the
overview): the (affine) operator L that maps latents to the reconstruction.

  * LinearOperator  -- a dataclass base.  Holds the shared structural options
                       (fit_intercept / nonneg / resample_dead) which inherit to
                       every operator, and defines L's faces (forward / drive /
                       gram), the discrete-search kernel for L's space, and
                       `backward`: one in-place M-step update of the operator's
                       own parameters.
  * subclasses      -- AffineOperator (numpy), TorchMatrixOp (nn.Linear/autograd),
                       CPOperator, ReducedRankOp, ConvOperator.  Each adds its own
                       learning-rate and weight-regularization hyperparameters as
                       dataclass fields, and implements its regularization inline
                       in `backward` -- the penalty is in general specific to the
                       operator (participation-ratio on W for the matrix ops,
                       per-rank on V for reduced-rank, seq-NMF for conv).

The data-dependent parameters (W, b, the optimizer) are created in init_params.
These have no dependency on the priors or the models; the models compose them.

Every operator supports MULTIPLE CHAINS through a single `n_chains` field
(default 1): n_chains == 1 is the ordinary serial operator with 2-D parameters
(W (d,m), b (d)), and n_chains > 1 carries a leading chain axis C (W (C,d,m),
b (C,d)) and runs C independent chains at once -- the same numpy/torch math with
one extra leading index, dispatched to the chain-batched (`prange`) search.  So
there is no separate Parallel operator; `to_serial(c)` collapses a multi-chain
operator to the winning chain's 2-D one.  (TorchMatrixOp and ConvOperator are the
two operators with no chain-batched port yet; they reject n_chains > 1.)
"""

import numpy as np
import scipy.linalg as la
import scipy.stats as sts
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import bae_search
import new_bae_search
from new_bae_search import make_dense_search


def init_affine(X, dim_hid, nonneg, hot_start, fit_intercept, resample_dead=False):
    """Shared (W, b) initialization for the affine operators (numpy arrays):
    SVD hot-start with the pos/neg split when nonneg, else random; optional
    dead-column resampling.  AffineOperator stores these directly; TorchMatrixOp
    wraps them in an nn.Linear.  This is the only piece of `init_params` the two
    matrix operators share -- everything else is storage/optimizer setup."""
    n, d = X.shape
    if hot_start:
        U, s, V = la.svd(X - X.mean(0), full_matrices=False)
        if nonneg:
            kpos = int(np.ceil(dim_hid / 2))
            kneg = int(np.floor(dim_hid / 2))
            W = np.vstack([V[:kpos] * (V[:kpos] > 0),
                           -V[:kneg] * (V[:kneg] < 0)]).T
        else:
            W = V[:dim_hid].T
        # W /= np.sqrt(d)
        b = X.mean(0) if fit_intercept else np.zeros(d)
        if resample_dead:
            dead = W.sum(0) == 0
            newW = np.random.randn(d, dead.sum()) / np.sqrt(d)
            newW[newW < 0] = 0
            W[:, dead] = newW
    else:
        W = np.random.randn(d, dim_hid) / np.sqrt(d)
        if nonneg:
            W[W < 0] = 0
        b = np.zeros(d)
    return W, b


# ===========================================================================
#  The (affine) operator L  (this is what makes tensors fit in)
# ===========================================================================
#
# Every linear-Gaussian model reconstructs X_hat = L(S) + b.  The operator is a
# dataclass that owns its structural options (inherited) AND its own weight-
# regularization hyperparameters (operator-specific).  The data-dependent
# parameters (W, b, the optimizer) are created in init_params.  Faces:
#   forward(S)    L(S) + b          reconstruction            (-> __call__)
#   drive(X)      L*(X - b)         E-step input  XW
#   gram()        L*L               E-step recurrence WtW
#   search(...)   the discrete kernel for L's space (sbmf / convbmf)
#   backward(S,X) ONE in-place M-step update of the operator's own parameters
#                 (incl. its weight regularization), returning the residual.

@dataclass
class LinearOperator:
    """Abstract affine map S -> X_hat = L(S) + b.  Dataclass base: the shared
    structural options below inherit to every operator; each child adds its own
    learning-rate / regularization fields.  Updates *independent* of the operator
    (sigma_x, a structured prior's J) are driven by the model's MStep."""

    n_chains: int = 1                # >1 runs C independent chains (leading axis)
    fit_intercept: bool = True
    nonneg: bool = False             # constrain parameters >= 0 (clamped in project)
    resample_dead: bool = True      # re-randomize columns that die under nonneg

    @property
    def _multi(self):
        return self.n_chains > 1

    def _reject_multi(self, name):
        """Guard for operators with no chain-batched port yet (torch autograd
        operators whose parameters aren't chain-batched)."""
        if self._multi:
            raise NotImplementedError(
                f"{name} has no multi-chain (n_chains > 1) port yet; use n_chains=1, "
                f"or run several fits and keep the best (bae_util.multifit).")

    def forward(self, S):  raise NotImplementedError      # L(S) + b
    def drive(self, X):    raise NotImplementedError      # L*(X - b)
    def gram(self):        raise NotImplementedError      # L*L

    # Whether this operator's dense scaffold has a diagonal gram (orthonormal W):
    # Procrustes sets it, so the neighbour loop is skipped.  Base dense ops don't.
    _diag_gram = False

    def build_search(self, link, prior, debug=False):
        """Compile this operator's E-step kernel with `link` and `prior` plugged
        in.  This is the composition step, run once at model init: the operator
        supplies the scaffold (its space's likelihood field + StS bookkeeping),
        `link` the slab likelihood (BINARY_LINK / SLAB_LINK), and `prior` the
        latent_prior's additive S-prior (PRIOR_PLAIN / PRIOR_BOLTZMANN).
        Memoized, so each distinct (scaffold, link, prior, debug) compiles once.
        `debug=True` compiles the variant that records per-element log-odds.

        A single chain compiles the serial (plain-@njit, 2-D) scaffold; n_chains>1
        compiles the chain-batched (`prange`, 3-D) one -- same factory, same
        link/prior plugins, only the `parallel` flag (and the outer chain axis)
        differ.  ConvOperator overrides this."""
        self._kernel = make_dense_search(*link, prior, diag_gram=self._diag_gram,
                                         debug=debug, parallel=self._multi)

    def search(self, XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None, prior_temp=1.0):
        """Run one discrete E-step.  S is the binary spike (tree/StS), Z the
        effective continuous latent (field accumulation; Z == S with no slab);
        Jc/hc are the structured prior's coupling/field (zeros if unstructured),
        scaled by `prior_temp` (the prior temperature; 1.0 = unscaled).
        S and Z are updated in place.  `out` (debug builds only) receives the
        per-element log-odds; None on the normal path."""
        return self._kernel(XW, S, Z, WtW, StS, N, temp, alpha, beta,
                            tau, sigma2, Jc, hc, inplace, out, prior_temp)

    def backward(self, S, X):
        """One in-place M-step update of this operator's parameters (incl. its
        weight regularization); returns the residual X - forward(S)."""
        raise NotImplementedError


@dataclass
class AffineOperator(LinearOperator):
    """L(S) = S W^T + b (affine, numpy).  Backs SemiBMF / BiPCA / JBMF.

    Its weight regularization (participation-ratio + L1/L2) is the analytic
    gradient, applied inline in `backward`.  The M-step is a gradient step whose
    reconstruction gradient is the VJP of `forward`; closed-form solvers
    (NNLS / ridge-SVD) are available as `method` options.  W, b set in init_params.
    """

    pr_reg: float = 1e-2             # participation-ratio (maximised)
    l1_reg: float = 0.0
    l2_reg: float = 1e-2
    init_jitter: float = 0.1         # multi-chain: per-chain perturbation of the
                                     # shared hot-start (so the chains start apart)

    # ---- faces --------------------------------------------------------------
    # Single chain: W (d, m), b (d,), plain numpy.  Multi-chain: W (C, d, m),
    # b (C, d), batched np.matmul (`@`) so each chain dispatches to BLAS (einsum
    # with a `c` index would fall back to a non-BLAS loop -- several times slower).
    def forward(self, S):
        if not self._multi:
            return S @ self.W.T + self.b
        return S @ self.Wt + self.b[:, None, :]           # (C,n,m)@(C,m,d)->(C,n,d)

    def drive(self, X):
        if not self._multi:
            return X @ self.W - self.b @ self.W
        bW = self.b[:, None, :] @ self.W                  # (C,1,d)@(C,d,m)->(C,1,m)
        return X @ self.W - bW                            # X shared (n,d) -> (C,n,m)

    def gram(self):
        if not self._multi:
            return self.W.T @ self.W
        return self.Wt @ self.W                           # (C,m,d)@(C,d,m)->(C,m,m)

    @property
    def Wt(self):
        return self.W.transpose(0, 2, 1)                  # (C, m, d), multi only

    def init_params(self, X, dim_hid, hot_start=True, lr=0.1):
        if not self._multi:
            self.W, self.b = init_affine(X, dim_hid, self.nonneg, hot_start,
                                         self.fit_intercept, self.resample_dead)
            self.lr = lr
            return
        # multi-chain: a shared SVD hot-start plus independent per-chain jitter, so
        # each chain's drive (hence the prior's spike init) differs and the chains
        # explore different basins.  hot_start=False gives fully random per-chain W.
        C = self.n_chains
        n, d = X.shape
        if hot_start:
            _, _, V = la.svd(X - X.mean(0), full_matrices=False)
            W0 = V[:dim_hid].T                                     # (d, m)
            W = np.repeat(W0[None], C, axis=0)                     # (C, d, m)
            W = W + self.init_jitter * np.random.randn(C, d, dim_hid) / np.sqrt(d)
            b0 = X.mean(0) if self.fit_intercept else np.zeros(d)
            b = np.repeat(b0[None], C, axis=0)                     # (C, d)
        else:
            W = np.random.randn(C, d, dim_hid) / np.sqrt(d)
            b = np.zeros((C, d))
        if self.nonneg:
            W[W < 0] = 0
        self.W, self.b, self.lr = W, b, lr

    def _reg_grad(self):             # d(penalty)/dW : participation-ratio + L1/L2
        if not self._multi:
            WtW = self.W.T @ self.W
            eta = np.trace(WtW) / np.sum(WtW ** 2)
            dReg = self.pr_reg * (self.W - eta * self.W @ WtW)
            dReg -= self.l2_reg * self.W
            dReg -= self.l1_reg * np.sign(self.W)
            return dReg
        WtW = self.gram()                                          # (C, m, m)
        tr = np.trace(WtW, axis1=1, axis2=2)                       # (C,)
        eta = tr / (WtW ** 2).sum((1, 2))                          # (C,)
        dReg = self.pr_reg * (self.W - eta[:, None, None] * (self.W @ WtW))
        dReg -= self.l2_reg * self.W
        dReg -= self.l1_reg * np.sign(self.W)
        return dReg

    def backward(self, S, X, method='grad'):
        if not self._multi:
            resid = X - self.forward(S)                   # = dXhat
            if method == 'grad':
                n = len(S)
                dW = resid.T @ S / n                       # VJP of forward w.r.t. W
                self.W += self.lr * (dW + self._reg_grad())
                if self.fit_intercept:
                    self.b += self.lr * (resid.sum(0) / n)
                self.project()
            elif method == 'nnls':
                ...  # per-column NNLS, pure-nonneg case (bae_models.SemiBMF.MStep:360)
            elif method == 'svd':
                ...  # ridge-SVD / orthogonal-Procrustes (BiPCA) (:367, BiPCA:227)
            return resid
        # multi-chain gradient step per chain (broadcast VJP + reg).  X is the
        # SHARED data (n, d), or a PER-CHAIN working copy (C, n, d) when the model
        # is imputing masked entries -- broadcasting handles both.
        Xb = X if X.ndim == 3 else X[None]                         # (C,n,d)/(1,n,d)
        resid = Xb - self.forward(S)                               # (C, n, d)
        n = S.shape[1]
        dW = S.transpose(0, 2, 1) @ resid                          # (C,m,n)@(C,n,d)
        dW = dW.transpose(0, 2, 1) / n                             # VJP wrt W, per chain
        self.W += self.lr * (dW + self._reg_grad())
        if self.fit_intercept:
            self.b += self.lr * (resid.sum(1) / n)
        self.project()
        return resid

    def project(self):
        if not self.nonneg:
            return
        if not self._multi:
            self.W[self.W < 0] = 0
            self.b[self.b < 0] = 0
            if self.resample_dead:
                d = self.W.shape[0]
                dead = self.W.sum(0) == 0
                newW = np.random.randn(d, dead.sum()) / np.sqrt(d)
                newW[newW < 0] = 0
                self.W[:, dead] = newW
        else:
            self.W[self.W < 0] = 0
            self.b[self.b < 0] = 0

    def to_serial(self, c):
        """Return chain c as a single-chain (n_chains=1) AffineOperator."""
        op = AffineOperator(fit_intercept=self.fit_intercept, nonneg=self.nonneg,
                            resample_dead=self.resample_dead, pr_reg=self.pr_reg,
                            l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        if self._multi:
            op.W, op.b = self.W[c].copy(), self.b[c].copy()
        else:
            op.W, op.b = self.W.copy(), self.b.copy()
        op.lr = self.lr
        return op


@dataclass
class Procrustes:
    """L(S) = scl * S W^T + b with W ORTHONORMAL (W^T W = I).  Backs BiPCA.

    Deliberately NOT a LinearOperator: it shares none of that base's machinery --
    no participation-ratio / L1 / L2 weight penalty (an orthonormal W has a fixed
    spectrum, so those regularizers are meaningless) and it cannot be non-negative
    (non-negative + orthonormal columns force a permutation matrix).  Instead it
    duck-types the operator interface the model needs (forward / drive / gram /
    build_search / search / backward / init_params) and carries its own scalar
    scale `scl`.

    It reuses the SHARED dense binary search rather than a bespoke kernel.  For a
    linear-Gaussian model X = scl S W^T + b + N(0, sigma_x) the S_ij flip log-odds
    is  (scl/sigma_x) XW_ij - (scl^2/sigma_x) sum_{k!=j} S_ik WtW_jk
        - 0.5 (scl^2/sigma_x) WtW_jj,
    which is EXACTLY the scaffold's binary link (E - 0.5 wjj)/sigma2 once XW and
    WtW are pre-scaled by scl and scl^2 -- so `drive` and `gram` carry those
    factors and the operator hands the true noise variance sigma_x through as
    sigma2.  (This corrects the old bae_search.bpca, whose 2*XW/scl - 1 dropped the
    Gaussian 1/2 and implicitly fixed sigma^2 = scl^2/2.)  Because W is orthonormal
    the gram is diagonal, so the search is compiled with diag_gram=True and the
    neighbour loop is skipped.  The M-step is the closed-form orthogonal-Procrustes
    solve (SVD of X^T ES), not a gradient step; W, scl, b set in init_params.
    (bae_models.BiPCA:162.)
    """

    n_chains: int = 1
    fit_intercept: bool = True
    fit_scl: bool = True
    init_jitter: float = 0.1         # multi-chain: per-chain perturbation of the
                                     # shared hot-start (followed by a polar project)

    @property
    def _multi(self):
        return self.n_chains > 1

    # ---- faces.  Single chain: W (d,m), b (d,), scalar scl.  Multi-chain:
    # W (C,d,m), b (C,d), per-chain scl (C,) entering as scl[:,None,None] /
    # scl[:,None,None]**2 -- the pre-scaling the shared binary link needs.
    def forward(self, S):
        if not self._multi:
            return self.scl * (S @ self.W.T) + self.b
        return self.scl[:, None, None] * (S @ self.Wt) + self.b[:, None, :]

    def drive(self, X):
        if not self._multi:
            return self.scl * (X @ self.W - self.b @ self.W)      # scl*XW
        bW = self.b[:, None, :] @ self.W                          # (C,1,d)@(C,d,m)
        return self.scl[:, None, None] * (X @ self.W - bW)        # scl*XW -> (C,n,m)

    def gram(self):
        if not self._multi:
            return (self.scl ** 2) * (self.W.T @ self.W)          # scl^2*WtW
        return (self.scl[:, None, None] ** 2) * (self.Wt @ self.W)

    @property
    def Wt(self):
        return self.W.transpose(0, 2, 1)                          # (C, m, d), multi only

    def init_params(self, X, dim_hid, hot_start=True, lr=1.0):
        self.dim_hid = dim_hid
        self.d = d = X.shape[1]
        transpose = d < dim_hid                          # rows orthonormal instead
        b0 = X.mean(0)
        if not self._multi:
            # hot_start is accepted for the shared init_params signature but ignored:
            # W is always seeded from the data (PCA) or a random orthonormal frame.
            if hot_start:
                # Economy SVD only yields min(N, d) right-singular vectors; when the
                # data rank is below dim_hid (i.e. N < dim_hid) that is too few to
                # fill dim_hid orthonormal columns, and W would silently come out
                # narrow -- desyncing the operator from the model/prior dim_hid and
                # blowing up the Procrustes M-step.  Fall back to the full SVD there
                # (N is small then, so the full U is cheap) so the extra columns are
                # the orthonormal complement of the data subspace.
                full = min(X.shape) < dim_hid
                _, _, Vx = la.svd(X, full_matrices=full)
                self.W = Vx[:dim_hid].T
            else:
                s1, s2 = max(d, dim_hid), min(d, dim_hid)
                self.W = sts.ortho_group.rvs(s1)[:, :s2]
                if transpose:
                    self.W = self.W.T
            self.lr = lr
            self.b = b0
            self.scl = np.sqrt(np.mean((X - b0) ** 2)) if self.fit_scl else 1
            return
        # multi-chain: shared hot-start + independent per-chain jitter, but W must be
        # orthonormal PER CHAIN, so the jitter is followed by a polar projection
        # (nearest orthonormal frame = U V^T of its SVD).  The chain loop here is a
        # one-shot init cost, not the per-iteration hot path, so it stays a plain loop.
        C = self.n_chains
        W = np.empty((C, d, dim_hid))
        if hot_start:
            full = min(X.shape) < dim_hid
            _, _, Vx = la.svd(X, full_matrices=full)
            W0 = Vx[:dim_hid].T                            # (d, m), orthonormal cols
            for c in range(C):
                Wj = W0 + self.init_jitter * np.random.randn(*W0.shape) / np.sqrt(d)
                U, _, V = la.svd(Wj, full_matrices=False)  # polar factor -> orthonormal
                W[c] = U @ V
        else:
            s1, s2 = max(d, dim_hid), min(d, dim_hid)
            for c in range(C):
                Wc = sts.ortho_group.rvs(s1)[:, :s2]
                W[c] = Wc.T if transpose else Wc
        self.W = W
        self.lr = lr
        self.b = np.repeat(b0[None], C, axis=0)            # (C, d)
        if self.fit_scl:
            self.scl = np.full(C, np.sqrt(np.mean((X - b0) ** 2)))
        else:
            self.scl = np.ones(C)

    # Orthonormal W -> diagonal gram, so compile the binary link with the neighbour
    # loop skipped (diag_gram=True).  Same (score, aux, prior) machinery as every
    # other dense model -- the scale lives entirely in the pre-scaled drive/gram.
    # Serial (2-D) or chain-batched (prange) build, picked by n_chains.
    def build_search(self, link, prior, debug=False):
        self._kernel = make_dense_search(*link, prior, diag_gram=True,
                                         debug=debug, parallel=self._multi)

    def search(self, XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None, prior_temp=1.0):
        return self._kernel(XW, S, Z, WtW, StS, N, temp, alpha, beta,
                            tau, sigma2, Jc, hc, inplace, out, prior_temp)

    def backward(self, S, X):
        """Closed-form orthogonal-Procrustes M-step (bae_models.BiPCA.MStep:227):
        W <- polar factor of X^T ES, scl the mean singular value, b the residual
        mean, each relaxed by lr.  Returns the post-update residual so the model's
        MStep recovers BiPCA's mean((X - scl ES W^T - b)**2).  Multi-chain: the same
        solve batched over chains via np.linalg.svd (which vectorizes leading axes),
        with X either the shared data (n, d) or a per-chain working copy (C, n, d)."""
        ES = S
        if not self._multi:
            XS = X.T @ ES - np.outer(self.b, ES.sum(0))
            # ridge sized to XS's ACTUAL columns (== ES.shape[1]); with a consistent
            # init this equals self.dim_hid, but keying off ES keeps the SVD
            # well-posed regardless of how many latent columns the E-step handed back.
            U, s, V = la.svd(XS + 1e-6 * np.eye(X.shape[1], ES.shape[1]),
                             full_matrices=False)
            self.W = U @ V
            if self.fit_scl:
                self.scl += self.lr * (np.sum(s) / np.sum(ES ** 2) - self.scl)
            if self.fit_intercept:
                self.b += self.lr * (X.mean(0) - self.scl * self.W @ ES.mean(0) - self.b)
            # grouped exactly as bae_models.BiPCA.MStep (scl*ES)@W.T so the returned
            # residual -- and the model's mean(resid**2) energy -- is bit-for-bit equal.
            return X - self.scl * ES @ self.W.T - self.b
        # multi-chain: unify the three data summaries the solve needs so shared
        # (n, d) and per-chain (C, n, d) data both flow through by broadcasting.
        if X.ndim == 3:
            Xt, Xmean, Xb = X.transpose(0, 2, 1), X.mean(1), X          # (C,d,n)/(C,d)/(C,n,d)
        else:
            Xt, Xmean, Xb = X.T[None], X.mean(0)[None], X[None]         # (1,d,n)/(1,d)/(1,n,d)
        XtES = np.matmul(Xt, ES)                                        # (C,d,m)
        bOuter = self.b[:, :, None] * ES.sum(1)[:, None, :]            # b_c (sum_i ES_c)^T
        XS = XtES - bOuter + 1e-6 * np.eye(self.d, self.dim_hid)[None]
        U, s, V = np.linalg.svd(XS, full_matrices=False)               # batched over chains
        self.W = U @ V                                                 # (C, d, m), orthonormal
        if self.fit_scl:
            self.scl += self.lr * (s.sum(1) / (ES ** 2).sum((1, 2)) - self.scl)
        if self.fit_intercept:
            WeS = (self.W @ ES.mean(1)[:, :, None])[:, :, 0]           # (C, d)
            self.b += self.lr * (Xmean - self.scl[:, None] * WeS - self.b)
        return Xb - self.scl[:, None, None] * (ES @ self.Wt) - self.b[:, None, :]

    def to_serial(self, c):
        """Return chain c as a single-chain (n_chains=1) Procrustes."""
        op = Procrustes(fit_intercept=self.fit_intercept, fit_scl=self.fit_scl)
        if self._multi:
            op.W, op.b = self.W[c].copy(), self.b[c].copy()
            op.scl = float(np.atleast_1d(self.scl)[c])
        else:
            op.W, op.b = self.W.copy(), self.b.copy()
            op.scl = self.scl
        op.lr, op.dim_hid, op.d = self.lr, self.dim_hid, self.d
        return op


@dataclass
class TorchMatrixOp(LinearOperator):
    """L(S) = W(S) with a torch nn.Linear (bias = intercept).  Backs the autograd
    models (SpikeNMF, ...).  Its regularization is applied inline to the autograd
    loss in `backward`; 'backward' really is autograd.  nn.Linear + optimizer set
    in init_params.
    """

    # lr: float = 0.1
    pr_reg: float = 1e-1
    l1_reg: float = 0.0
    l2_reg: float = 1e-2

    def _np(self):
        W = self.W.weight.detach().numpy()
        b = (self.W.bias.detach().numpy() if self.W.bias is not None
             else np.zeros(W.shape[0]))
        return W, b

    def forward(self, S):
        with torch.no_grad():
            return self.decode(torch.FloatTensor(S)).numpy()

    def drive(self, X):                        # == SpikeNMF.EStep (X@W - b@W)
        W, b = self._np()
        return X @ W - b @ W

    def gram(self):
        W, _ = self._np()
        return W.T @ W

    def decode(self, S_t):                     # differentiable torch forward
        return self.W(S_t)

    def init_params(self, X, dim_hid, hot_start=False, **opt_args):
        self._reject_multi("TorchMatrixOp")
        Winit, binit = init_affine(X, dim_hid, self.nonneg,
                                   hot_start, self.fit_intercept, self.resample_dead)
        d = X.shape[1]
        self.W = nn.Linear(dim_hid, d, bias=self.fit_intercept)
        self.W.weight.data.copy_(torch.FloatTensor(Winit))
        if self.fit_intercept:
            self.W.bias.data.copy_(torch.FloatTensor(binit))
        # self.optimizer = optim.SGD(self.W.parameters(), lr=self.lr, **opt_args)
        self.optimizer = optim.SGD(self.W.parameters(), **opt_args)

    def _reg_loss(self, loss):                 # add penalty to the autograd loss
        W = self.W.weight
        WtW = W.T @ W
        loss = loss - self.pr_reg * ((torch.trace(WtW) ** 2) / torch.sum(WtW ** 2))
        loss = loss + self.l1_reg * torch.sum(torch.abs(W))
        loss = loss + self.l2_reg * torch.trace(WtW)
        return loss

    def backward(self, S, X):
        """One autograd M-step on W: minimise ||X - decode(S)||^2 + penalty(W)."""
        Spt = torch.FloatTensor(S)
        Xpt = torch.FloatTensor(X)
        self.optimizer.zero_grad()
        Xhat = self.decode(Spt)
        loss = torch.sum((Xpt - Xhat) ** 2) / len(Xpt)
        loss = self._reg_loss(loss)
        loss.backward()
        self.optimizer.step()
        self.project()
        return X - Xhat.detach().numpy()

    def project(self):
        if self.nonneg:
            with torch.no_grad():
                W = self.W.weight
                W[W < 0] = 0
                self.W.bias[self.W.bias < 0] = 0
                if self.resample_dead:
                    d = W.shape[0]
                    dead = W.sum(0) == 0
                    newW = torch.randn(d, dead.sum()) / np.sqrt(d)
                    newW[newW < 0] = 0
                    W[:, dead] = newW


@dataclass
class CPOperator(LinearOperator):
    """L(S) = einsum('ck,tk,nk->ctn', S, U, V) + b.  Backs SCPD (bae_models.SCPD).

    Params (set in init_params): self.U (t, dim_hid), self.V (d, dim_hid), self.b,
    torch; self.optimizer.  drive/gram detach to numpy for sbmf, so this shares
    the inherited sbmf `search` with ReducedRankOp; `backward` is the same
    autograd step as TorchMatrixOp with a CP decode + PR-on-V / L1-on-U,V penalty.

    Multi-chain: chains become a leading axis P on the nn.Parameters (U/V/b gain a
    C axis) and the einsum faces gain a `P` (chain) index.  ONE optimizer covers
    the whole batch -- the chains' parameters are independent, so the gradient of
    the summed per-chain loss w.r.t. chain P's params is exactly chain P's own
    gradient (no cross-talk, no lr rescaling).  The numpy faces (drive/gram) keep
    the standard batched shapes (C,n,k)/(C,k,k), so the parallel dense search is
    reused unchanged.
    """

    pr_reg: float = 1e-1
    l1_reg: float = 0.0
    l2_reg: float = 1e-2
    init_jitter: float = 0.1         # multi-chain: per-chain hot-start perturbation

    def _npar(self):                               # multi-chain: detached numpy factors
        with torch.no_grad():
            b = self.b.detach().numpy()
        return self.U.detach().numpy(), self.V.detach().numpy(), b

    def forward(self, S):                          # bae_models.SCPD.__call__:1086
        if not self._multi:
            with torch.no_grad():
                U, V = self.U.detach().numpy(), self.V.detach().numpy()
                b = self.b.detach().numpy()
            return np.einsum('...ck,tk,nk->...ctn', S, U, V) + b
        U, V, b = self._npar()
        return np.einsum('Pck,Ptk,Pnk->Pctn', S, U, V) + b[:, None]   # (C,n,t,d)

    def drive(self, X):                             # bae_models.SCPD.EStep:1145
        if not self._multi:
            with torch.no_grad():
                Xt = torch.as_tensor(np.asarray(X), dtype=self.b.dtype)
                XW = (self.U[None] * ((Xt - self.b) @ self.V)).sum(1)
            return XW.detach().numpy()
        U, V, b = self._npar()                                        # (C,t,k)/(C,d,k)/(C,t,d)
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xtb = Xnp[None] - b[:, None]                                  # (C,n,t,d)
        Cc, n, t, d = Xtb.shape
        proj = (Xtb.reshape(Cc, n * t, d) @ V).reshape(Cc, n, t, -1)  # (C,n,t,k)
        return (U[:, None] * proj).sum(2)                            # (C,n,k)

    def gram(self):                                 # bae_models.SCPD.EStep:1146
        if not self._multi:
            with torch.no_grad():
                WtW = (self.U.T @ self.U) * (self.V.T @ self.V)
            return WtW.detach().numpy()
        U, V, _ = self._npar()
        return (U.transpose(0, 2, 1) @ U) * (V.transpose(0, 2, 1) @ V)   # (C,k,k)

    def decode(self, S_t):                          # bae_models.SCPD.MStep:1175
        if not self._multi:
            return torch.einsum('...ck,tk,nk->...ctn', S_t, self.U, self.V) + self.b
        return torch.einsum('Pck,Ptk,Pnk->Pctn', S_t, self.U, self.V) + self.b[:, None]

    def init_params(self, X, dim_hid, hot_start=False, opt_alg=optim.Adam, **opt_args):
        # data is a 3-D tensor (n, t, d); CP factors U (t, dim_hid), V (d, dim_hid)
        # and a per-(t, d) intercept b.  The latents (S) are initialised by the
        # prior, so -- unlike bae_models.SCPD.initialize:1102 -- this sets only the
        # operator's parameters.  The original's nmf_init branch is not ported;
        # the factors are random (its `else` branch, SCPD.initialize:1126).
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        n, t, d = Xnp.shape
        self.dim_hid, self.t, self.d = dim_hid, t, d

        if not self._multi:
            V, _ = init_affine(X.reshape((-1, d)), dim_hid, self.nonneg,
                                hot_start, self.fit_intercept, self.resample_dead)
            b = X.mean(0)
            if hot_start:
                U = ((X - b) @ V).mean(0)
            else:
                U = np.random.randn(t, dim_hid) / np.sqrt(t)
            if self.nonneg:
                U[U < 0] = 0
                V[V < 0] = 0
            self.U = nn.Parameter(torch.tensor(U))
            self.V = nn.Parameter(torch.tensor(V))
            self.b = (nn.Parameter(torch.tensor(b)) if self.fit_intercept
                      else torch.tensor(b))
            params = [self.U, self.V, self.b] if self.fit_intercept else [self.U, self.V]
            self.optimizer = opt_alg(params, **opt_args)
            return
        # multi-chain: each chain draws its factors independently (a shared PCA
        # hot-start plus per-chain jitter, or fully random per chain) so chains differ.
        self._opt_alg, self._opt_args = opt_alg, opt_args
        C = self.n_chains
        b0 = Xnp.mean(0)                                              # (t, d)
        Us = np.empty((C, t, dim_hid)); Vs = np.empty((C, d, dim_hid))
        for c in range(C):
            V, _ = init_affine(Xnp.reshape((-1, d)), dim_hid, self.nonneg,
                               hot_start, self.fit_intercept, self.resample_dead)
            if hot_start:                                            # deterministic -> jitter
                V = V + self.init_jitter * np.random.randn(d, dim_hid) / np.sqrt(d)
                U = ((Xnp - b0) @ V).mean(0)
                U = U + self.init_jitter * np.random.randn(t, dim_hid) / np.sqrt(t)
            else:                                                    # random -> already differ
                U = np.random.randn(t, dim_hid) / np.sqrt(t)
            if self.nonneg:
                U[U < 0] = 0; V[V < 0] = 0
            Us[c], Vs[c] = U, V
        bs = np.repeat(b0[None], C, axis=0)                          # (C, t, d)
        self.U = nn.Parameter(torch.tensor(Us))
        self.V = nn.Parameter(torch.tensor(Vs))
        self.b = (nn.Parameter(torch.tensor(bs)) if self.fit_intercept
                  else torch.tensor(bs))
        params = [self.U, self.V, self.b] if self.fit_intercept else [self.U, self.V]
        self.optimizer = opt_alg(params, **opt_args)

    def backward(self, S, X):                       # autograd; PR on V + L1 on U,V (:1180)
        # one autograd M-step on (U, V, b): data loss ||X - decode(S)||^2 / n, then
        # the CP penalty -- participation-ratio on V (maximised) + L1 on U and V,
        # each /dim_hid exactly as bae_models.SCPD.MStep:1187-1190.  Returns the
        # pre-step residual so the model's MStep recovers mean((X - Xhat)**2).
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xt = torch.as_tensor(Xnp, dtype=self.b.dtype)
        Spt = torch.as_tensor(np.asarray(S), dtype=self.b.dtype)
        if not self._multi:
            self.optimizer.zero_grad()
            Xhat = self.decode(Spt)
            loss = torch.sum((Xt - Xhat) ** 2) / len(Xt)
            VtV = self.V.T @ self.V
            loss = loss - self.pr_reg * ((torch.trace(VtV) ** 2) / torch.sum(VtV ** 2))
            loss = loss + self.l1_reg * torch.sum(torch.abs(self.V))
            loss = loss + self.l1_reg * torch.sum(torch.abs(self.U))
            loss = loss + self.l2_reg * torch.sum(self.V ** 2)
            loss = loss + self.l2_reg * torch.sum(self.U ** 2)
            loss.backward()
            self.optimizer.step()
            self.project()
            return Xnp - Xhat.detach().numpy()
        # multi-chain: summed per-chain loss (independent params -> per-chain grad).
        Xtb = Xt if Xt.ndim == 4 else Xt[None]                       # broadcast leading C
        self.optimizer.zero_grad()
        Xhat = self.decode(Spt)                                      # (C,n,t,d)
        loss = torch.sum((Xtb - Xhat) ** 2) / Xhat.shape[1]          # /n, summed over chains
        VtV = self.V.transpose(1, 2) @ self.V                        # (C,k,k)
        tr = torch.diagonal(VtV, dim1=1, dim2=2).sum(1)              # (C,) trace per chain
        nrm = torch.sum(VtV ** 2, dim=(1, 2))                        # (C,)
        loss = loss - self.pr_reg * torch.sum((tr ** 2) / nrm)
        loss = loss + self.l1_reg * (torch.sum(torch.abs(self.V)) + torch.sum(torch.abs(self.U)))
        loss = loss + self.l2_reg * (torch.sum(self.V ** 2) + torch.sum(self.U ** 2))
        loss.backward()
        self.optimizer.step()
        self.project()
        return (Xnp if Xnp.ndim == 4 else Xnp[None]) - Xhat.detach().numpy()

    def project(self):                              # clamp U, V, b >= 0 (SCPD.MStep:1196)
        if self.nonneg:
            with torch.no_grad():
                self.U[self.U < 0] = 0
                self.V[self.V < 0] = 0
                self.b[self.b < 0] = 0

    def to_serial(self, c):
        """Return chain c as a single-chain (n_chains=1) CPOperator."""
        op = CPOperator(fit_intercept=self.fit_intercept, nonneg=self.nonneg,
                        resample_dead=self.resample_dead, pr_reg=self.pr_reg,
                        l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        op.dim_hid, op.t, op.d = self.dim_hid, self.t, self.d
        with torch.no_grad():
            if self._multi:
                U, V, b = self.U.detach()[c].clone(), self.V.detach()[c].clone(), self.b.detach()[c].clone()
                opt_alg, opt_args = self._opt_alg, self._opt_args
            else:
                U, V, b = self.U.detach().clone(), self.V.detach().clone(), self.b.detach().clone()
                opt_alg, opt_args = type(self.optimizer), {}
        op.U, op.V = nn.Parameter(U), nn.Parameter(V)
        op.b = nn.Parameter(b) if self.fit_intercept else b
        params = [op.U, op.V, op.b] if self.fit_intercept else [op.U, op.V]
        try:
            op.optimizer = opt_alg(params, **opt_args)
        except TypeError:
            op.optimizer = optim.Adam(params)
        return op


@dataclass
class ReducedRankOp(LinearOperator):
    """L(S) = einsum('ck,knt->ctn', S, beta) + b with beta = V @ U.T.  Backs RRBMF.

    Params (set in init_params): self.U (t, rank), self.V (dim_hid, d, rank),
    self.b, torch.  drive/gram match RRBMF.EStep and feed the *same* sbmf as
    CPOperator -- only these faces differ, which is why RRBMF and SCPD share the
    entire E-step.
    """

    rank: int = None
    pr_reg: float = 1e-2
    l1_reg: float = 0.0
    l2_reg: float = 1e-2

    def _beta(self):                                # single chain: (dim_hid, d, t)
        return self.V @ self.U.T

    def _beta_np(self):                             # multi-chain: (C, k, d, t)
        return np.einsum('Pkdr,Ptr->Pkdt', self.V.detach().numpy(),
                         self.U.detach().numpy())

    def _beta_torch(self):                          # multi-chain, differentiable
        return torch.einsum('Pkdr,Ptr->Pkdt', self.V, self.U)

    def forward(self, S):                           # bae_models.RRBMF.__call__:1347
        if not self._multi:
            with torch.no_grad():
                Xhat = torch.einsum('...ck,knt->...ctn', torch.tensor(S),
                                    self._beta()) + self.b
            return Xhat.detach().numpy()
        beta = self._beta_np()
        b = self.b.detach().numpy()
        return np.einsum('Pck,Pknt->Pctn', S, beta) + b[:, None]     # (C,n,t,d)

    def drive(self, X):                             # bae_models.RRBMF.EStep:1407
        if not self._multi:
            with torch.no_grad():
                Xt = torch.as_tensor(np.asarray(X), dtype=self.b.dtype)
                XW = torch.einsum('kdt,ntd->nk', self._beta(), Xt - self.b)
            return XW.detach().numpy()
        beta = self._beta_np()                                       # (C,k,d,t)
        b = self.b.detach().numpy()
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xtb = Xnp[None] - b[:, None]                                 # (C,n,t,d)
        return np.einsum('Pkdt,Pntd->Pnk', beta, Xtb)               # (C,n,k)

    def gram(self):                                 # bae_models.RRBMF.EStep:1408
        if not self._multi:
            with torch.no_grad():
                beta = self._beta()
                WtW = torch.einsum('kdt,cdt', beta, beta)
            return WtW.detach().numpy()
        beta = self._beta_np()
        return np.einsum('Pkdt,Pcdt->Pkc', beta, beta)              # (C,k,k)

    def decode(self, S_t):                          # bae_models.RRBMF.MStep:1437
        if not self._multi:
            return torch.einsum('ck,knt->ctn', S_t, self._beta()) + self.b
        return torch.einsum('Pck,Pknt->Pctn', S_t, self._beta_torch()) + self.b[:, None]

    def init_params(self, X, dim_hid, hot_start=False, U=None,
                    opt_alg=optim.SGD, **opt_args):
        # data is a 3-D tensor (n, t, d).  b is the per-(t, d) intercept; as in the
        # original RRBMF it is always fit (fit_intercept is not wired on this op).
        # `U` may be passed to *fix* the temporal factor (then it is not optimized).
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        n, t, d = Xnp.shape
        self.dim_hid, self.t, self.d = dim_hid, t, d

        if not self._multi:
            fit_U = U is None
            if fit_U:
                U = np.random.randn(t, self.rank) / np.sqrt(t * self.rank)
            V = np.random.randn(dim_hid, d, self.rank) / np.sqrt(d * self.rank)
            if self.nonneg:
                U[U < 0] = 0
                V[V < 0] = 0
            b = Xnp.mean(0)
            self.U = nn.Parameter(torch.tensor(U)) if fit_U else torch.tensor(U)
            self.V = nn.Parameter(torch.tensor(V))
            self.b = nn.Parameter(torch.tensor(b))
            params = [self.U, self.V, self.b] if fit_U else [self.V, self.b]
            self.optimizer = opt_alg(params, **opt_args)
            return
        # multi-chain: each chain's U, V drawn independently (the serial init is
        # already random, so no jitter is needed).
        self._opt_alg, self._opt_args = opt_alg, opt_args
        C = self.n_chains
        self._fit_U = U is None
        if self._fit_U:
            Us = np.random.randn(C, t, self.rank) / np.sqrt(t * self.rank)
        else:
            Us = np.repeat(np.asarray(U)[None], C, axis=0)           # fixed temporal factor
        Vs = np.random.randn(C, dim_hid, d, self.rank) / np.sqrt(d * self.rank)
        if self.nonneg:
            Us[Us < 0] = 0; Vs[Vs < 0] = 0
        bs = np.repeat(Xnp.mean(0)[None], C, axis=0)                 # (C, t, d)
        self.U = nn.Parameter(torch.tensor(Us)) if self._fit_U else torch.tensor(Us)
        self.V = nn.Parameter(torch.tensor(Vs))
        self.b = nn.Parameter(torch.tensor(bs))
        params = [self.U, self.V, self.b] if self._fit_U else [self.V, self.b]
        self.optimizer = opt_alg(params, **opt_args)

    def _reg_loss(self, loss, beta):
        # per-rank participation-ratio on V (maximised) + L1 on V + L2 on beta,
        # normalized exactly as bae_models.RRBMF.MStep:1449-1454.
        VtV = self.V.swapaxes(1, 2) @ self.V              # (dim_hid, rank, rank)
        trace = torch.einsum('kii->k', VtV)
        norm = torch.sum(VtV ** 2, axis=(1, 2))
        loss = loss - self.pr_reg * torch.sum((trace ** 2) / norm) / (self.rank * self.dim_hid)
        loss = loss + self.l1_reg * torch.mean(torch.abs(self.V))
        loss = loss + self.l2_reg * torch.mean(beta ** 2)
        return loss

    def backward(self, S, X):                       # autograd; per-rank PR on V + L1/L2 (:1442)
        # one autograd M-step on (U, V, b); data loss normalized by d*t (not n),
        # matching the original.  Returns the pre-step residual so the model's
        # MStep recovers RRBMF's new_sig = mean((X - Xhat)**2).
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xt = torch.as_tensor(Xnp, dtype=self.b.dtype)
        Spt = torch.as_tensor(np.asarray(S), dtype=self.b.dtype)
        if not self._multi:
            self.optimizer.zero_grad()
            beta = self._beta()
            Xhat = torch.einsum('ck,knt->ctn', Spt, beta) + self.b
            loss = torch.sum((Xt - Xhat) ** 2) / (self.d * self.t)
            loss = self._reg_loss(loss, beta)
            loss.backward()
            self.optimizer.step()
            self.project()
            return Xnp - Xhat.detach().numpy()
        # multi-chain: summed per-chain loss (independent params -> per-chain grad).
        Xtb = Xt if Xt.ndim == 4 else Xt[None]
        self.optimizer.zero_grad()
        beta = self._beta_torch()                                    # (C,k,d,t)
        Xhat = torch.einsum('Pck,Pknt->Pctn', Spt, beta) + self.b[:, None]
        loss = torch.sum((Xtb - Xhat) ** 2) / (self.d * self.t)      # summed over chains
        # per-rank participation ratio on V + L1 on V + L2 on beta, PER CHAIN then
        # summed (a plain mean over the batched array would divide the per-chain
        # gradient by an extra factor C -- weakening the penalty).
        VtV = self.V.transpose(2, 3) @ self.V                        # (C,k,r,r)
        trace = torch.einsum('Pkii->Pk', VtV)                        # (C,k)
        norm = torch.sum(VtV ** 2, dim=(2, 3))                       # (C,k)
        pr = torch.sum((trace ** 2) / norm, dim=1) / (self.rank * self.dim_hid)   # (C,)
        loss = loss - self.pr_reg * pr.sum()
        loss = loss + self.l1_reg * torch.abs(self.V).mean(dim=(1, 2, 3)).sum()
        loss = loss + self.l2_reg * (beta ** 2).mean(dim=(1, 2, 3)).sum()
        loss.backward()
        self.optimizer.step()
        self.project()
        return (Xnp if Xnp.ndim == 4 else Xnp[None]) - Xhat.detach().numpy()

    def project(self):
        if not self.nonneg:
            return
        with torch.no_grad():
            self.V[self.V < 0] = 0
            if (not self._multi) or self._fit_U:
                self.U[self.U < 0] = 0
            self.b[self.b < 0] = 0

    def to_serial(self, c):
        """Return chain c as a single-chain (n_chains=1) ReducedRankOp."""
        op = ReducedRankOp(rank=self.rank, fit_intercept=self.fit_intercept,
                           nonneg=self.nonneg, resample_dead=self.resample_dead,
                           pr_reg=self.pr_reg, l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        op.dim_hid, op.t, op.d = self.dim_hid, self.t, self.d
        with torch.no_grad():
            if self._multi:
                U, V, b = self.U.detach()[c].clone(), self.V.detach()[c].clone(), self.b.detach()[c].clone()
                fit_U, opt_alg, opt_args = self._fit_U, self._opt_alg, self._opt_args
            else:
                U, V, b = self.U.detach().clone(), self.V.detach().clone(), self.b.detach().clone()
                fit_U, opt_alg, opt_args = True, type(self.optimizer), {}
        op.U = nn.Parameter(U) if fit_U else U
        op.V = nn.Parameter(V)
        op.b = nn.Parameter(b)
        params = [op.U, op.V, op.b] if fit_U else [op.V, op.b]
        try:
            op.optimizer = opt_alg(params, **opt_args)
        except TypeError:
            op.optimizer = optim.SGD(params, lr=1e-2)
        return op


@dataclass
class ConvOperator(LinearOperator):
    """L(S) = conv1d(S, K) + b.  Backs ConvBMF (bae_models.ConvBMF).

    Params (set in init_params): self.K (dim_hid, d, kernel_size), self.b,
    self.pad, torch.  In the translation-invariant space the metric `gram` is
    banded over time-lags and the sparsity splits into feature vs time terms, so
    this operator overrides `search` and its regularizers are conv-specific.
    """

    kernel_size: int = None
    gp_width: float = 0.1
    feature_sparsity: float = 1e-1
    time_sparsity: float = 1e-1
    sparse_reg: float = 1e-2
    seq_reg: float = 1e-2

    def forward(self, S):                           # bae_models.ConvBMF.__call__:1222
        with torch.no_grad():
            Xhat = F.conv1d(S, self.K.flip([2]), padding=self.pad) + self.b
        return Xhat.detach().numpy()

    def drive(self, X):                             # bae_models.ConvBMF.EStep:1266
        with torch.no_grad():
            Kt = self.K.transpose(0, 1)
            XW = F.conv1d(X - self.b, Kt, padding=0)
        return XW.detach().numpy()                  # (n, dim_hid, t)

    def gram(self):                                 # bae_models.ConvBMF.EStep:1267
        with torch.no_grad():
            Kt = self.K.transpose(0, 1)
            WtW = F.conv1d(Kt, Kt, padding=self.pad)
        return WtW.detach().numpy()                 # (dim_hid, dim_hid, 2*ks-1)

    def build_search(self, link, prior, debug=False):
        # Conv lives in its own translation-invariant scaffold (a make_conv_search
        # factory, TODO) -- the time-banded gram + feature/time sparsity don't fit
        # the dense scaffold, so the dense link/prior bundles don't apply here.
        # (debug recording of the log-odds is not wired for the conv scaffold yet.)
        self._link = link
        self._prior = prior

    def search(self, XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None, prior_temp=1.0):
        # convbmf takes the conv-specific feature/time-sparsity scalars; Z/StS/N/
        # tau/sigma2/Jc/hc/prior_temp/out unused in this stub.  (ConvBMF.EStep:1281.)
        return bae_search.convbmf(
            XW, S, WtW, temp=temp,
            beta=self.feature_sparsity,
            alpha=self.time_sparsity,
            l1_reg=self.sparse_reg)

    def decode(self, S_t):                          # bae_models.ConvBMF.MStep:1299
        return F.conv1d(S_t, self.K.flip([2]), padding=self.pad) + self.b

    def init_params(self, X, dim_hid, hot_start=False, **opt_args):
        self._reject_multi("ConvOperator")
        ...   # init GP kernels K (dim_hid, d, ks), b, self.pad + self.optimizer (ConvBMF:1231)

    def backward(self, S, X):                       # autograd; seq-NMF orthogonality (:1304)
        ...                                         # same loop as TorchMatrixOp.backward
        return X - self.forward(S)

    def project(self):
        ...                                         # clamp K, b >= 0 (:1317)
