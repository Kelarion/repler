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

    fit_intercept: bool = True
    nonneg: bool = False             # constrain parameters >= 0 (clamped in project)
    resample_dead: bool = True      # re-randomize columns that die under nonneg

    def forward(self, S):  raise NotImplementedError      # L(S) + b
    def drive(self, X):    raise NotImplementedError      # L*(X - b)
    def gram(self):        raise NotImplementedError      # L*L

    # Which discrete-search scaffold this operator's space uses.  The dense ops
    # (Affine / CP / ReducedRank) all share new_bae_search's coordinate-descent
    # scaffold -- spike-only and spike-and-slab are just two *links* into it, so
    # the slab no longer owns a search.  ConvOperator overrides this.
    _search_factory = staticmethod(new_bae_search.make_dense_search)

    def build_search(self, link, prior, debug=False):
        """Compile this operator's E-step kernel with `link` and `prior` plugged
        in.  This is the composition step, run once at model init: the operator
        supplies the scaffold (its space's likelihood field + StS bookkeeping),
        `link` the slab likelihood (BINARY_LINK / SLAB_LINK), and `prior` the
        latent_prior's additive S-prior (PRIOR_PLAIN / PRIOR_BOLTZMANN).
        Memoized, so each distinct (scaffold, link, prior, debug) compiles once.
        `debug=True` compiles the variant that records per-element log-odds."""
        self._kernel = self._search_factory(*link, prior, debug=debug)

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

    def forward(self, S):  return S @ self.W.T + self.b
    def drive(self, X):    return X @ self.W - self.b @ self.W
    def gram(self):        return self.W.T @ self.W

    def init_params(self, X, dim_hid, hot_start=True, lr=0.1):
        self.W, self.b = init_affine(X, dim_hid, self.nonneg,
                                     hot_start, self.fit_intercept, self.resample_dead)

        self.lr = lr

    def _reg_grad(self):             # d(penalty)/dW : participation-ratio + L1/L2
        WtW = self.W.T @ self.W
        eta = np.trace(WtW) / np.sum(WtW ** 2)
        dReg = self.pr_reg * (self.W - eta * self.W @ WtW)
        dReg -= self.l2_reg * self.W
        dReg -= self.l1_reg * np.sign(self.W)
        return dReg

    def backward(self, S, X, method='grad'):
        resid = X - self.forward(S)                       # = dXhat
        if method == 'grad':
            n = len(S)
            dW = resid.T @ S / n                           # VJP of forward w.r.t. W
            self.W += self.lr * (dW + self._reg_grad())
            if self.fit_intercept:
                self.b += self.lr * (resid.sum(0) / n)
            self.project()
        elif method == 'nnls':
            ...      # per-column NNLS, pure-nonneg case (bae_models.SemiBMF.MStep:360)
        elif method == 'svd':
            ...      # ridge-SVD / orthogonal-Procrustes (BiPCA) (:367, BiPCA:227)
        return resid

    def project(self):
        if self.nonneg:
            self.W[self.W < 0] = 0
            self.b[self.b < 0] = 0
            if self.resample_dead:
                d = self.W.shape[0]
                dead = self.W.sum(0) == 0
                newW = np.random.randn(d, dead.sum()) / np.sqrt(d)
                newW[newW < 0] = 0
                self.W[:, dead] = newW


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

    fit_intercept: bool = True
    fit_scl: bool = True

    def forward(self, S):  return self.scl * (S @ self.W.T) + self.b
    def drive(self, X):    return self.scl * (X @ self.W - self.b @ self.W)      # scl*XW
    def gram(self):        return (self.scl ** 2) * (self.W.T @ self.W)          # scl^2*WtW

    def init_params(self, X, dim_hid, hot_start=True, lr=1.0):
        # hot_start is accepted for the shared init_params signature but ignored:
        # W is always seeded from the data (PCA) or a random orthonormal frame.
        self.dim_hid = dim_hid
        self.d = X.shape[1]
        transpose = self.d < dim_hid                          # rows orthonormal instead
        if hot_start:
            # Economy SVD only yields min(N, d) right-singular vectors; when the
            # data rank is below dim_hid (i.e. N < dim_hid) that is too few to fill
            # dim_hid orthonormal columns, and W would silently come out narrow --
            # desyncing the operator from the model/prior dim_hid and blowing up the
            # Procrustes M-step.  Fall back to the full SVD there (N is small then,
            # so the full U is cheap) so the extra columns are the orthonormal
            # complement of the data subspace.
            full = min(X.shape) < dim_hid
            _, _, Vx = la.svd(X, full_matrices=full)
            self.W = Vx[:dim_hid].T
        else:
            s1 = max(self.d, dim_hid)
            s2 = min(self.d, dim_hid)
            self.W = sts.ortho_group.rvs(s1)[:, :s2]
            if transpose:
                self.W = self.W.T
        self.lr = lr
        self.b = X.mean(0)
        if self.fit_scl:
            self.scl = np.sqrt(np.mean((X - self.b) ** 2))
            # self.scl = 1e-3
        else:
            self.scl = 1

    # Orthonormal W -> diagonal gram, so compile the binary link with the neighbour
    # loop skipped (diag_gram=True).  Same (score, aux, prior) machinery as every
    # other dense model -- the scale lives entirely in the pre-scaled drive/gram.
    def build_search(self, link, prior, debug=False):
        self._kernel = new_bae_search.make_dense_search(*link, prior,
                                                        diag_gram=True, debug=debug)

    def search(self, XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None, prior_temp=1.0):
        return self._kernel(XW, S, Z, WtW, StS, N, temp, alpha, beta,
                            tau, sigma2, Jc, hc, inplace, out, prior_temp)

    def backward(self, S, X):
        """Closed-form orthogonal-Procrustes M-step (bae_models.BiPCA.MStep:227):
        W <- polar factor of X^T ES, scl the mean singular value, b the residual
        mean, each relaxed by lr.  Returns the post-update residual so the model's
        MStep recovers BiPCA's mean((X - scl ES W^T - b)**2)."""
        ES = S
        XS = X.T @ ES - np.outer(self.b, ES.sum(0))
        # ridge sized to XS's ACTUAL columns (== ES.shape[1]); with a consistent
        # init this equals self.dim_hid, but keying off ES keeps the SVD well-posed
        # regardless of how many latent columns the E-step handed back.
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
    """

    pr_reg: float = 1e-1
    l1_reg: float = 0.0
    l2_reg: float = 1e-2

    def forward(self, S):                          # bae_models.SCPD.__call__:1086
        with torch.no_grad():
            U, V = self.U.detach().numpy(), self.V.detach().numpy()
            b = self.b.detach().numpy()
        return np.einsum('...ck,tk,nk->...ctn', S, U, V) + b

    def drive(self, X):                             # bae_models.SCPD.EStep:1145
        with torch.no_grad():
            Xt = torch.as_tensor(np.asarray(X), dtype=self.b.dtype)
            XW = (self.U[None] * ((Xt - self.b) @ self.V)).sum(1)
        return XW.detach().numpy()

    def gram(self):                                 # bae_models.SCPD.EStep:1146
        with torch.no_grad():
            WtW = (self.U.T @ self.U) * (self.V.T @ self.V)
        return WtW.detach().numpy()

    def decode(self, S_t):                          # bae_models.SCPD.MStep:1175
        return torch.einsum('...ck,tk,nk->...ctn', S_t, self.U, self.V) + self.b

    def init_params(self, X, dim_hid, hot_start=False, opt_alg=optim.Adam, **opt_args):
        # data is a 3-D tensor (n, t, d); CP factors U (t, dim_hid), V (d, dim_hid)
        # and a per-(t, d) intercept b.  The latents (S) are initialised by the
        # prior, so -- unlike bae_models.SCPD.initialize:1102 -- this sets only the
        # operator's parameters.  The original's nmf_init branch is not ported;
        # the factors are random (its `else` branch, SCPD.initialize:1126).
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        n, t, d = Xnp.shape
        self.dim_hid, self.t, self.d = dim_hid, t, d

        V, _ = init_affine(X.reshape((-1, d)), dim_hid, self.nonneg,
                            hot_start, self.fit_intercept, self.resample_dead)
        b = X.mean(0)

        if hot_start:
            U = ((X-b)@V).mean(0)
        else:
            U = np.random.randn(t, dim_hid) / np.sqrt(t)

        if self.nonneg:
            U[U < 0] = 0
            V[V < 0] = 0

        self.U = nn.Parameter(torch.tensor(U))
        self.V = nn.Parameter(torch.tensor(V))
        self.b = nn.Parameter(torch.tensor(b)) if self.fit_intercept else torch.tensor(b)

        params = [self.U, self.V, self.b] if self.fit_intercept else [self.U, self.V]
        # self.optimizer = opt_alg(params, **{'lr': self.lr, **opt_args})
        self.optimizer = opt_alg(params, **opt_args)

    def backward(self, S, X):                       # autograd; PR on V + L1 on U,V (:1180)
        # one autograd M-step on (U, V, b): data loss ||X - decode(S)||^2 / n, then
        # the CP penalty -- participation-ratio on V (maximised) + L1 on U and V,
        # each /dim_hid exactly as bae_models.SCPD.MStep:1187-1190.  Returns the
        # pre-step residual so the model's MStep recovers mean((X - Xhat)**2).
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xt = torch.as_tensor(Xnp, dtype=self.b.dtype)
        Spt = torch.as_tensor(np.asarray(S), dtype=self.b.dtype)
        self.optimizer.zero_grad()
        Xhat = self.decode(Spt)
        loss = torch.sum((Xt - Xhat) ** 2) / len(Xt)
        VtV = self.V.T @ self.V
        loss = loss - self.pr_reg * ((torch.trace(VtV) ** 2) / torch.sum(VtV ** 2)) 
        loss = loss + self.l1_reg * torch.sum(torch.abs(self.V))
        loss = loss + self.l1_reg * torch.sum(torch.abs(self.U))
        loss = loss + self.l2_reg * torch.sum(self.V**2) 
        loss = loss + self.l2_reg * torch.sum(self.U**2) 
        loss.backward()
        self.optimizer.step()
        self.project()
        return Xnp - Xhat.detach().numpy()

    def project(self):                              # clamp U, V, b >= 0 (SCPD.MStep:1196)
        if self.nonneg:
            with torch.no_grad():
                self.U[self.U < 0] = 0
                self.V[self.V < 0] = 0
                self.b[self.b < 0] = 0


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

    def _beta(self):
        return self.V @ self.U.T                    # (dim_hid, d, t)

    def forward(self, S):                           # bae_models.RRBMF.__call__:1347
        with torch.no_grad():
            Xhat = torch.einsum('...ck,knt->...ctn', torch.tensor(S), self._beta()) + self.b
        return Xhat.detach().numpy()

    def drive(self, X):                             # bae_models.RRBMF.EStep:1407
        with torch.no_grad():
            Xt = torch.as_tensor(np.asarray(X), dtype=self.b.dtype)
            XW = torch.einsum('kdt,ntd->nk', self._beta(), Xt - self.b)
        return XW.detach().numpy()

    def gram(self):                                 # bae_models.RRBMF.EStep:1408
        with torch.no_grad():
            beta = self._beta()
            WtW = torch.einsum('kdt,cdt', beta, beta)
        return WtW.detach().numpy()

    def decode(self, S_t):                          # bae_models.RRBMF.MStep:1437
        return torch.einsum('ck,knt->ctn', S_t, self._beta()) + self.b

    def init_params(self, X, dim_hid, hot_start=False, U=None,
                    opt_alg=optim.SGD, **opt_args):
        # data is a 3-D tensor (n, t, d).  b is the per-(t, d) intercept; as in the
        # original RRBMF it is always fit (fit_intercept is not wired on this op).
        # `U` may be passed to *fix* the temporal factor (then it is not optimized).
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        n, t, d = Xnp.shape
        self.dim_hid, self.t, self.d = dim_hid, t, d

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
        self.optimizer.zero_grad()
        beta = self._beta()
        Xhat = torch.einsum('ck,knt->ctn', Spt, beta) + self.b
        loss = torch.sum((Xt - Xhat) ** 2) / (self.d * self.t)
        loss = self._reg_loss(loss, beta)
        loss.backward()
        self.optimizer.step()
        self.project()
        return Xnp - Xhat.detach().numpy()

    def project(self):
        if self.nonneg:
            with torch.no_grad():
                self.V[self.V < 0] = 0
                self.U[self.U < 0] = 0
                self.b[self.b < 0] = 0


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
        ...   # init GP kernels K (dim_hid, d, ks), b, self.pad + self.optimizer (ConvBMF:1231)

    def backward(self, S, X):                       # autograd; seq-NMF orthogonality (:1304)
        ...                                         # same loop as TorchMatrixOp.backward
        return X - self.forward(S)

    def project(self):
        ...                                         # clamp K, b >= 0 (:1317)
