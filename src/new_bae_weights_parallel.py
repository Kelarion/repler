"""
new_bae_weights_parallel.py  --  PROTOTYPE (parallel chains)
============================================================

Parallel-chains variant of new_bae_weights.AffineOperator.  This is the file that
best shows the payoff of the "chains = a leading axis" idea: every face and the
whole M-step are the serial code with one extra einsum index `c`.  There is NO
Python loop over chains anywhere in the numpy math -- broadcasting does it.

  W  (C, d, m)     per-chain weights
  b  (C, d)        per-chain intercept

  forward(S)   einsum('cnm,cdm->cnd', S, W) + b        (C, n, d)
  drive(X)     einsum('nd,cdm->cnm', X, W) - (b@W)     (C, n, m)   X shared
  gram()       einsum('cdm,cdk->cmk', W, W)            (C, m, m)
  backward     resid VJP + reg, all with a `c` axis

The chains must start DIFFERENT or they collapse to one; init_params seeds a
shared SVD hot-start and adds independent per-chain jitter (so drive, hence the
prior's spike init, differs per chain).  Set `hot_start=False` for fully random
per-chain W.
"""

import numpy as np
import scipy.linalg as la
import scipy.stats as sts
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

import new_bae_search_parallel as nbsp
# the serial operators are the collapse targets: to_serial(c) returns the winning
# chain's params wrapped in the ordinary (2-D / non-batched) operator.
import new_bae_weights as nbw


@dataclass
class ParallelAffineOperator:
    """L_c(S) = S W_c^T + b_c for C chains at once (numpy, broadcast).  Duck-types
    the operator interface that ParallelLinearGaussianBMF needs; same regularizer
    (participation-ratio + L1/L2) as new_bae_weights.AffineOperator, per chain."""

    n_chains: int = 8
    fit_intercept: bool = True
    nonneg: bool = False
    resample_dead: bool = False
    pr_reg: float = 1e-2
    l1_reg: float = 0.0
    l2_reg: float = 1e-2
    init_jitter: float = 0.1     # per-chain perturbation of the shared hot-start

    # ---- faces (chain-batched) --------------------------------------------
    # Use batched np.matmul (`@`), NOT einsum: matmul treats the last two axes as
    # the matrices and broadcasts the leading chain axis, dispatching EACH chain to
    # BLAS -- so this is the multithreaded gemm the serial code gets, just batched.
    # (einsum with a `c` index falls back to the naive non-BLAS loop and is several
    # times slower for the same FLOPs -- the whole point of "broadcast, don't loop"
    # is lost if the broadcast is not a BLAS call.)
    def forward(self, S):
        return S @ self.Wt + self.b[:, None, :]            # (C,n,m)@(C,m,d)->(C,n,d)

    def drive(self, X):
        # X shared (n, d) broadcasts against W (C, d, m) -> (C, n, m)
        bW = (self.b[:, None, :] @ self.W)                 # (C,1,d)@(C,d,m)->(C,1,m)
        return X @ self.W - bW

    def gram(self):
        return self.Wt @ self.W                            # (C,m,d)@(C,d,m)->(C,m,m)

    @property
    def Wt(self):
        return self.W.transpose(0, 2, 1)                   # (C, m, d)

    # ---- init: shared hot-start + independent per-chain jitter -------------
    def init_params(self, X, dim_hid, hot_start=True, lr=0.1):
        C = self.n_chains
        n, d = X.shape
        if hot_start:
            _, _, V = la.svd(X - X.mean(0), full_matrices=False)
            W0 = V[:dim_hid].T                                  # (d, m)
            W = np.repeat(W0[None], C, axis=0)                 # (C, d, m)
            W = W + self.init_jitter * np.random.randn(C, d, dim_hid) / np.sqrt(d)
            b0 = X.mean(0) if self.fit_intercept else np.zeros(d)
            b = np.repeat(b0[None], C, axis=0)                # (C, d)
        else:
            W = np.random.randn(C, d, dim_hid) / np.sqrt(d)
            b = np.zeros((C, d))
        if self.nonneg:
            W[W < 0] = 0
        self.W, self.b, self.lr = W, b, lr

    # ---- the discrete E-step scaffold (parallel dense search) --------------
    def build_search(self, link, prior, debug=False):
        self._kernel = nbsp.make_parallel_dense_search(*link, prior, debug=debug)

    def search(self, XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None, prior_temp=1.0):
        return self._kernel(XW, S, Z, WtW, StS, N, temp, alpha, beta,
                            tau, sigma2, Jc, hc, inplace, out, prior_temp)

    # ---- M-step: one gradient step per chain (broadcast VJP + reg) ---------
    def _reg_grad(self):
        WtW = self.gram()                                      # (C, m, m)
        tr = np.trace(WtW, axis1=1, axis2=2)                   # (C,)
        nrm = (WtW ** 2).sum((1, 2))
        eta = tr / nrm                                         # (C,)
        WWtW = self.W @ WtW                                    # (C,d,m)@(C,m,m)
        dReg = self.pr_reg * (self.W - eta[:, None, None] * WWtW)
        dReg -= self.l2_reg * self.W
        dReg -= self.l1_reg * np.sign(self.W)
        return dReg

    def backward(self, S, X):
        # X is the SHARED data (n, d) in the normal fit, or a PER-CHAIN working copy
        # (C, n, d) when the model is imputing masked entries (each chain fills its
        # own holes).  Broadcasting handles both once X carries a leading axis.
        Xb = X if X.ndim == 3 else X[None]                     # (C, n, d) or (1, n, d)
        resid = Xb - self.forward(S)                           # (C, n, d)
        n = S.shape[1]
        dW = S.transpose(0, 2, 1) @ resid                      # (C,m,n)@(C,n,d)->(C,m,d)
        dW = dW.transpose(0, 2, 1) / n                         # VJP wrt W, per chain
        self.W += self.lr * (dW + self._reg_grad())
        if self.fit_intercept:
            self.b += self.lr * (resid.sum(1) / n)
        self.project()
        return resid

    def project(self):
        if self.nonneg:
            self.W[self.W < 0] = 0
            self.b[self.b < 0] = 0

    # ---- collapse to the winning chain's serial operator -------------------
    def to_serial(self, c):
        """Return chain c as an ordinary (2-D) new_bae_weights.AffineOperator."""
        op = nbw.AffineOperator(fit_intercept=self.fit_intercept, nonneg=self.nonneg,
                                resample_dead=self.resample_dead, pr_reg=self.pr_reg,
                                l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        op.W, op.b, op.lr = self.W[c].copy(), self.b[c].copy(), self.lr
        return op


@dataclass
class ParallelProcrustes:
    """Parallel-chains variant of new_bae_weights.Procrustes (backs ParallelBiPCA).

    L_c(S) = scl_c * S W_c^T + b_c with W_c ORTHONORMAL (W_c^T W_c = I), for C
    chains at once.  Like the serial one it is deliberately NOT a LinearOperator:
    orthonormal W has a fixed spectrum (no pr/l1/l2 penalty) and cannot be
    non-negative.  It duck-types the operator interface and carries a per-chain
    scalar scale scl (C,).

      W    (C, d, m)   per-chain orthonormal weights
      b    (C, d)      per-chain intercept
      scl  (C,)        per-chain scale

    It rides the SHARED binary link exactly as the serial Procrustes: drive and
    gram pre-scale by scl / scl^2 so the scaffold's (E - 0.5 wjj)/sigma2 is the
    exact linear-Gaussian flip log-odds, and orthonormal W -> diagonal gram means
    the parallel kernel is compiled with diag_gram=True (neighbour loop skipped).
    The M-step is the closed-form orthogonal-Procrustes SVD solve, batched over
    chains via np.linalg.svd (which vectorizes over leading axes)."""

    n_chains: int = 8
    fit_intercept: bool = True
    fit_scl: bool = True
    init_jitter: float = 0.1     # per-chain perturbation of the shared hot-start

    # ---- faces (chain-batched) --------------------------------------------
    # Same batched-matmul discipline as ParallelAffineOperator: the leading chain
    # axis broadcasts and each chain dispatches to BLAS.  The only extra is the
    # per-chain scalar scl, which enters as scl[:, None, None] (drive/forward) or
    # scl[:, None, None]**2 (gram) -- the pre-scaling the shared binary link needs.
    def forward(self, S):
        return self.scl[:, None, None] * (S @ self.Wt) + self.b[:, None, :]

    def drive(self, X):
        bW = self.b[:, None, :] @ self.W                   # (C,1,d)@(C,d,m)->(C,1,m)
        return self.scl[:, None, None] * (X @ self.W - bW)  # scl * XW  -> (C,n,m)

    def gram(self):
        return (self.scl[:, None, None] ** 2) * (self.Wt @ self.W)   # scl^2 * WtW

    @property
    def Wt(self):
        return self.W.transpose(0, 2, 1)                   # (C, m, d)

    # ---- init: shared hot-start + independent per-chain jitter -------------
    # W must be orthonormal PER CHAIN, so jitter is followed by a polar projection
    # (nearest orthonormal frame = U V^T of its SVD).  The chain loop here is a
    # one-shot init cost, not the per-iteration hot path, so it stays a plain loop.
    def init_params(self, X, dim_hid, hot_start=True, lr=1.0):
        C = self.n_chains
        self.dim_hid = dim_hid
        self.d = d = X.shape[1]
        transpose = d < dim_hid                            # rows orthonormal instead
        b0 = X.mean(0)
        W = np.empty((C, d, dim_hid))
        if hot_start:
            _, _, Vx = la.svd(X, full_matrices=False)
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

    # ---- the discrete E-step scaffold (diagonal gram -> skip neighbour loop)
    def build_search(self, link, prior, debug=False):
        self._kernel = nbsp.make_parallel_dense_search(*link, prior,
                                                       diag_gram=True, debug=debug)

    def search(self, XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None, prior_temp=1.0):
        return self._kernel(XW, S, Z, WtW, StS, N, temp, alpha, beta,
                            tau, sigma2, Jc, hc, inplace, out, prior_temp)

    # ---- M-step: closed-form orthogonal Procrustes, batched over chains ----
    def backward(self, S, X):
        ES = S                                             # (C, n, m)
        # X shared (n, d) or per-chain (C, n, d) when imputing; unify the three data
        # summaries the solve needs so both shapes flow through by broadcasting.
        if X.ndim == 3:
            Xt, Xmean, Xb = X.transpose(0, 2, 1), X.mean(1), X          # (C,d,n)/(C,d)/(C,n,d)
        else:
            Xt, Xmean, Xb = X.T[None], X.mean(0)[None], X[None]         # (1,d,n)/(1,d)/(1,n,d)
        XtES = np.matmul(Xt, ES)                           # (C,d,m)
        bOuter = self.b[:, :, None] * ES.sum(1)[:, None, :]   # b_c (sum_i ES_c)^T
        XS = XtES - bOuter + 1e-6 * np.eye(self.d, self.dim_hid)[None]
        U, s, V = np.linalg.svd(XS, full_matrices=False)   # batched over chains
        self.W = U @ V                                     # (C, d, m), orthonormal
        if self.fit_scl:
            self.scl += self.lr * (s.sum(1) / (ES ** 2).sum((1, 2)) - self.scl)
        if self.fit_intercept:
            WeS = (self.W @ ES.mean(1)[:, :, None])[:, :, 0]   # (C, d)
            self.b += self.lr * (Xmean - self.scl[:, None] * WeS - self.b)
        # grouped as (scl*ES)@W^T so the returned residual -- and the model's
        # mean(resid**2) energy -- matches the serial Procrustes per chain.
        return Xb - self.scl[:, None, None] * (ES @ self.Wt) - self.b[:, None, :]

    # ---- collapse to the winning chain's serial operator -------------------
    def to_serial(self, c):
        """Return chain c as an ordinary new_bae_weights.Procrustes."""
        op = nbw.Procrustes(fit_intercept=self.fit_intercept, fit_scl=self.fit_scl)
        op.W, op.b, op.lr = self.W[c].copy(), self.b[c].copy(), self.lr
        op.scl = float(np.atleast_1d(self.scl)[c])
        op.dim_hid, op.d = self.dim_hid, self.d
        return op


# ===========================================================================
#  Torch-autograd operators over 3-D data (n, t, d), chain-batched
# ===========================================================================
#
# CPOperator / ReducedRankOp are torch operators: their M-step is an autograd step
# through an optimizer, not an analytic VJP.  Chains become a leading axis on the
# nn.Parameters (U / V / b gain a C axis), the einsum faces gain a `P` (chain)
# index, and ONE optimizer covers the whole batch -- because the chains' parameters
# are independent, the gradient of the summed per-chain loss w.r.t. chain P's params
# is exactly chain P's own gradient, so a single summed loss trains all chains at
# once with no cross-talk (SGD/Adam are per-parameter, so no lr rescaling).  The
# numpy faces (drive / gram) that feed the search keep the standard batched shapes
# (C,n,k)/(C,k,k), so the parallel dense search is reused unchanged.


@dataclass
class ParallelCPOperator:
    """Parallel-chains CPOperator: L_P(S) = einsum('ck,tk,nk->ctn', S, U_P, V_P) + b_P
    over a 3-D data tensor X (n, t, d), for C chains at once (backs SCPD/JSCPD).

      U  (C, t, k)     per-chain temporal factor
      V  (C, d, k)     per-chain feature factor
      b  (C, t, d)     per-chain intercept

    Chains start different because init draws each chain's factors independently
    (a shared PCA hot-start plus per-chain jitter, or fully random per chain)."""

    n_chains: int = 8
    fit_intercept: bool = True
    nonneg: bool = False
    resample_dead: bool = True
    pr_reg: float = 1e-1
    l1_reg: float = 0.0
    l2_reg: float = 1e-2
    init_jitter: float = 0.1

    def _npar(self):
        with torch.no_grad():
            b = self.b.detach().numpy()
        return self.U.detach().numpy(), self.V.detach().numpy(), b

    # ---- faces (chain-batched); numpy faces feed the standard search shapes ----
    def forward(self, S):
        U, V, b = self._npar()
        return np.einsum('Pck,Ptk,Pnk->Pctn', S, U, V) + b[:, None]   # (C,n,t,d)

    def drive(self, X):
        U, V, b = self._npar()                                        # (C,t,k)/(C,d,k)/(C,t,d)
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xtb = Xnp[None] - b[:, None]                                  # (C,n,t,d)
        Cc, n, t, d = Xtb.shape
        proj = (Xtb.reshape(Cc, n * t, d) @ V).reshape(Cc, n, t, -1)  # (C,n,t,k)
        return (U[:, None] * proj).sum(2)                            # (C,n,k)

    def gram(self):
        U, V, _ = self._npar()
        return (U.transpose(0, 2, 1) @ U) * (V.transpose(0, 2, 1) @ V)   # (C,k,k)

    def decode(self, S_t):
        return torch.einsum('Pck,Ptk,Pnk->Pctn', S_t, self.U, self.V) + self.b[:, None]

    def build_search(self, link, prior, debug=False):
        self._kernel = nbsp.make_parallel_dense_search(*link, prior, debug=debug)

    def search(self, XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None, prior_temp=1.0):
        return self._kernel(XW, S, Z, WtW, StS, N, temp, alpha, beta,
                            tau, sigma2, Jc, hc, inplace, out, prior_temp)

    def init_params(self, X, dim_hid, hot_start=False, opt_alg=optim.Adam, **opt_args):
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        n, t, d = Xnp.shape
        self.dim_hid, self.t, self.d = dim_hid, t, d
        self._opt_alg, self._opt_args = opt_alg, opt_args
        C = self.n_chains
        b0 = Xnp.mean(0)                                              # (t, d)
        Us = np.empty((C, t, dim_hid)); Vs = np.empty((C, d, dim_hid))
        for c in range(C):
            V, _ = nbw.init_affine(Xnp.reshape((-1, d)), dim_hid, self.nonneg,
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
        self.b = nn.Parameter(torch.tensor(bs)) if self.fit_intercept else torch.tensor(bs)
        params = [self.U, self.V, self.b] if self.fit_intercept else [self.U, self.V]
        self.optimizer = opt_alg(params, **opt_args)

    def backward(self, S, X):
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xt = torch.as_tensor(Xnp, dtype=self.b.dtype)                # (n,t,d) or (C,n,t,d)
        Spt = torch.as_tensor(np.asarray(S), dtype=self.b.dtype)     # (C,n,k)
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
        return (Xnp if Xnp.ndim == 4 else Xnp[None]) - Xhat.detach().numpy()   # (C,n,t,d)

    def project(self):
        if self.nonneg:
            with torch.no_grad():
                self.U[self.U < 0] = 0
                self.V[self.V < 0] = 0
                self.b[self.b < 0] = 0

    def to_serial(self, c):
        op = nbw.CPOperator(fit_intercept=self.fit_intercept, nonneg=self.nonneg,
                            resample_dead=self.resample_dead, pr_reg=self.pr_reg,
                            l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        op.dim_hid, op.t, op.d = self.dim_hid, self.t, self.d
        with torch.no_grad():
            U, V = self.U.detach()[c].clone(), self.V.detach()[c].clone()
            b = self.b.detach()[c].clone()
        op.U, op.V = nn.Parameter(U), nn.Parameter(V)
        op.b = nn.Parameter(b) if self.fit_intercept else b
        params = [op.U, op.V, op.b] if self.fit_intercept else [op.U, op.V]
        op.optimizer = self._opt_alg(params, **self._opt_args)
        return op


@dataclass
class ParallelReducedRankOp:
    """Parallel-chains ReducedRankOp: L_P(S) = einsum('ck,knt->ctn', S, beta_P) + b_P
    with beta_P = V_P @ U_P^T over a 3-D data tensor X (n, t, d), for C chains
    (backs RRBMF/JRRBMF).

      U  (C, t, rank)              per-chain temporal factor
      V  (C, dim_hid, d, rank)     per-chain low-rank feature factor
      b  (C, t, d)                 per-chain intercept

    Chains start different because each chain's U, V are drawn independently at
    random (the serial init is already random, so no jitter is needed)."""

    rank: int = None
    n_chains: int = 8
    fit_intercept: bool = True
    nonneg: bool = False
    resample_dead: bool = True
    pr_reg: float = 1e-2
    l1_reg: float = 0.0
    l2_reg: float = 1e-2

    def _beta_np(self):
        return np.einsum('Pkdr,Ptr->Pkdt', self.V.detach().numpy(),
                         self.U.detach().numpy())                     # (C,k,d,t)

    def _beta_torch(self):
        return torch.einsum('Pkdr,Ptr->Pkdt', self.V, self.U)

    def forward(self, S):
        beta = self._beta_np()
        b = self.b.detach().numpy()
        return np.einsum('Pck,Pknt->Pctn', S, beta) + b[:, None]     # (C,n,t,d)

    def drive(self, X):
        beta = self._beta_np()                                       # (C,k,d,t)
        b = self.b.detach().numpy()
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xtb = Xnp[None] - b[:, None]                                 # (C,n,t,d)
        return np.einsum('Pkdt,Pntd->Pnk', beta, Xtb)               # (C,n,k)

    def gram(self):
        beta = self._beta_np()
        return np.einsum('Pkdt,Pcdt->Pkc', beta, beta)              # (C,k,k)

    def decode(self, S_t):
        return torch.einsum('Pck,Pknt->Pctn', S_t, self._beta_torch()) + self.b[:, None]

    def build_search(self, link, prior, debug=False):
        self._kernel = nbsp.make_parallel_dense_search(*link, prior, debug=debug)

    def search(self, XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
               Jc, hc, inplace=True, out=None, prior_temp=1.0):
        return self._kernel(XW, S, Z, WtW, StS, N, temp, alpha, beta,
                            tau, sigma2, Jc, hc, inplace, out, prior_temp)

    def init_params(self, X, dim_hid, hot_start=False, U=None,
                    opt_alg=optim.SGD, **opt_args):
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        n, t, d = Xnp.shape
        self.dim_hid, self.t, self.d = dim_hid, t, d
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

    def backward(self, S, X):
        Xnp = np.asarray(X.numpy() if torch.is_tensor(X) else X)
        Xt = torch.as_tensor(Xnp, dtype=self.b.dtype)
        Spt = torch.as_tensor(np.asarray(S), dtype=self.b.dtype)     # (C,n,k)
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
        if self.nonneg:
            with torch.no_grad():
                self.V[self.V < 0] = 0
                if self._fit_U:
                    self.U[self.U < 0] = 0
                self.b[self.b < 0] = 0

    def to_serial(self, c):
        with torch.no_grad():
            U, V = self.U.detach()[c].clone(), self.V.detach()[c].clone()
            b = self.b.detach()[c].clone()
        op = nbw.ReducedRankOp(rank=self.rank, fit_intercept=self.fit_intercept,
                               nonneg=self.nonneg, resample_dead=self.resample_dead,
                               pr_reg=self.pr_reg, l1_reg=self.l1_reg, l2_reg=self.l2_reg)
        op.dim_hid, op.t, op.d = self.dim_hid, self.t, self.d
        op.U = nn.Parameter(U) if self._fit_U else U
        op.V = nn.Parameter(V)
        op.b = nn.Parameter(b)
        params = [op.U, op.V, op.b] if self._fit_U else [op.V, op.b]
        op.optimizer = self._opt_alg(params, **self._opt_args)
        return op
