"""Minimal single-chain structured binary PCA / spike-and-slab EM.

Winning choices from the matched benchmark:

* orthonormal decoder with an intercept and one learned decoder scale;
* persistent one-sweep Gibbs E steps at temperature one;
* a permanent per-active-bit penalty;
* optional exponential slabs, analytically collapsed when sampling the spikes;
* scale-free partial updates of the observation variance;
* delayed, symmetric pseudolikelihood updates of J and h;
* a final reset, proximal-L1 support fit, debias, and temperature rescale; and
* a short fixed-prior decoder refinement.

Only NumPy and SciPy are required. Input should normally be globally scaled,
for example ``X = X / X.std()``. The implementation intentionally uses one
chain and keeps binary spikes separate from continuous slab magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_ndtr, ndtr, ndtri


def _polar(matrix):
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ vt


@dataclass
class StructuredBiPCA:
    n_components: int
    slab: bool = False
    sparse_reg: float = 1.0
    slab_rate: float = 1.0

    n_iter: int = 600
    prior_delay: int = 50
    J_lr: float = 0.3
    J_l1: float = 2e-4
    final_J_l1: float | None = None
    final_selection_steps: int = 350
    final_debias_steps: int = 250
    refinement_iters: int = 40

    W_lr: float = 0.25
    scale_lr: float = 0.10
    variance_lr: float = 0.03
    initial_variance: float = 1.0
    seed: int = 0

    def _initialize(self, X):
        if self.n_components > X.shape[1]:
            raise ValueError("n_components must not exceed the data dimension")
        # Matches the winning comparison's same-PCA condition.
        _, _, vt = np.linalg.svd(
            X, full_matrices=min(X.shape) < self.n_components,
        )
        self.W_ = vt[:self.n_components].T
        self.b_ = X.mean(axis=0)
        self.decoder_scale_ = float(np.sqrt(np.mean((X-self.b_)**2)))
        self.variance_ = float(self.initial_variance)

        drive = self.decoder_scale_*(X-self.b_)@self.W_
        self.S_ = (drive >= 0).astype(float)
        self.Z_ = np.maximum(drive, 0.0)*self.S_ if self.slab else self.S_.copy()
        self.J_ = np.zeros((self.n_components, self.n_components))
        self.h_ = np.zeros(self.n_components)

    def _likelihood_log_odds(self, X):
        projected = (X-self.b_)@self.W_
        variance = max(self.variance_, 1e-12)
        scale = max(self.decoder_scale_, 1e-12)
        if not self.slab:
            odds = (scale*projected-0.5*scale**2)/variance
            return odds, projected

        # If z~Exp(tau), then u=scale*z~Exp(tau/scale).
        rate = self.slab_rate/scale
        sigma = np.sqrt(variance)
        argument = projected/sigma-rate*sigma
        odds = (
            np.log(rate)+0.5*np.log(2*np.pi)+np.log(sigma)
            +0.5*(projected/sigma)**2-rate*projected
            +0.5*(rate*sigma)**2+log_ndtr(argument)
        )
        return odds, projected

    def _sample_slabs(self, projected, S, rng):
        scale = max(self.decoder_scale_, 1e-12)
        variance = max(self.variance_, 1e-12)
        mean = projected/scale-self.slab_rate*variance/scale**2
        sd = np.sqrt(variance)/scale
        lower_cdf = ndtr(-mean/sd)
        probability = lower_cdf+(1-lower_cdf)*rng.random(mean.shape)
        probability = np.clip(probability, np.finfo(float).tiny, 1-1e-12)
        Z = np.maximum(mean+sd*ndtri(probability), 0.0)
        return S*Z

    def _gibbs(self, X, S, rng, sweeps=1):
        """Persistent collapsed spike updates followed by active slab draws."""
        S = np.asarray(S, dtype=float).copy()
        data_logit, projected = self._likelihood_log_odds(X)
        prior_logit = 2*(S@self.J_+self.h_)

        for _ in range(sweeps):
            for node in rng.permutation(self.n_components):
                probability = expit(
                    data_logit[:, node]+prior_logit[:, node]-self.sparse_reg
                )
                updated = (rng.random(len(X)) < probability).astype(float)
                change = updated-S[:, node]
                S[:, node] = updated
                prior_logit += 2*change[:, None]*self.J_[node][None, :]

        Z = self._sample_slabs(projected, S, rng) if self.slab else S.copy()
        return S, Z

    @staticmethod
    def _rple_gradient(S, J, h, h_ridge=1e-4):
        """Symmetric nodewise pseudolikelihood in the {0,1} convention."""
        n, k = S.shape
        logit = 2*(S@J+h)
        error = expit(logit)-S
        loss = float(np.mean(np.logaddexp(0, logit)-S*logit))
        grad_h = 2*error.mean(axis=0)/k+h_ridge*h
        grad_J = 2*(error.T@S+S.T@error)/(n*k)
        np.fill_diagonal(grad_J, 0.0)
        return loss, grad_J, grad_h

    @classmethod
    def _rple_step(
        cls, S, J, h, *, learning_rate, l1=0.0, support=None,
    ):
        loss, grad_J, grad_h = cls._rple_gradient(S, J, h)
        if support is not None:
            grad_J *= support
        J = J-learning_rate*grad_J

        upper = np.triu_indices(len(J), 1)
        if l1:
            values = J[upper]
            J[upper] = np.sign(values)*np.maximum(
                np.abs(values)-learning_rate*l1, 0.0,
            )
        J[(upper[1], upper[0])] = J[upper]
        np.fill_diagonal(J, 0.0)
        if support is not None:
            J *= support
        h = h-learning_rate*grad_h
        return J, h, loss

    def _decoder_step(
        self, X, Z, *, W_lr=None, scale_lr=None, variance_lr=None,
    ):
        W_lr = self.W_lr if W_lr is None else W_lr
        scale_lr = self.scale_lr if scale_lr is None else scale_lr
        variance_lr = self.variance_lr if variance_lr is None else variance_lr

        xbar, zbar = X.mean(axis=0), Z.mean(axis=0)
        Xc, Zc = X-xbar, Z-zbar
        proposed_W = _polar(Xc.T@Zc)
        self.W_ = _polar((1-W_lr)*self.W_+W_lr*proposed_W)

        denominator = max(np.sum(Zc**2), 1e-12)
        target_scale = np.sum((Xc@self.W_)*Zc)/denominator
        target_scale = float(np.clip(target_scale, 1e-4, 100.0))
        self.decoder_scale_ = float(np.exp(
            (1-scale_lr)*np.log(max(self.decoder_scale_, 1e-12))
            +scale_lr*np.log(target_scale)
        ))
        self.b_ = xbar-self.decoder_scale_*self.W_@zbar

        residual = X-self.b_-self.decoder_scale_*Z@self.W_.T
        mse = float(np.mean(residual**2))
        direction = np.clip(mse/max(self.variance_, 1e-12)-1, -5.0, 5.0)
        self.variance_ = float(self.variance_*np.exp(variance_lr*direction))
        return mse

    def _final_prior_fit(self):
        """Reset path-dependent J,h; select support, debias, then rescale."""
        k = self.n_components
        l1 = 0.142/k if self.final_J_l1 is None else self.final_J_l1
        codes = self.S_.copy()
        flip = codes.mean(axis=0) > 0.5
        codes[:, flip] = 1-codes[:, flip]
        J, h = np.zeros((k, k)), np.zeros(k)
        for _ in range(self.final_selection_steps):
            J, h, _ = self._rple_step(
                codes, J, h, learning_rate=0.4, l1=l1,
            )
        support = (np.abs(J) > 1e-8).astype(float)
        for _ in range(self.final_debias_steps):
            J, h, _ = self._rple_step(
                codes, J, h, learning_rate=0.15, support=support,
            )

        # Return to the decoder's original, non-flipped coordinate system.
        offset = flip.astype(float)
        sign = 1-2*offset
        J_original = sign[:, None]*J*sign[None, :]
        h_original = sign*(h+J@offset)

        def objective(log_scale):
            scale = np.exp(log_scale)
            logit = 2*scale*(self.S_@J_original+h_original)
            return float(np.mean(
                np.logaddexp(0, logit)-self.S_*logit
            ))

        result = minimize_scalar(objective, bounds=(-2, 2), method="bounded")
        temperature_scale = float(np.exp(result.x))
        self.J_ = temperature_scale*J_original
        self.h_ = temperature_scale*h_original
        self.support_ = support
        self.prior_scale_ = temperature_scale

    def fit(self, X):
        """Fit one model and return self."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        rng = np.random.default_rng(self.seed)
        self._initialize(X)
        self.history_ = []

        for iteration in range(self.n_iter):
            self.S_, self.Z_ = self._gibbs(X, self.S_, rng)
            mse = self._decoder_step(X, self.Z_)
            prior_loss = np.nan
            if iteration >= self.prior_delay:
                self.J_, self.h_, prior_loss = self._rple_step(
                    self.S_, self.J_, self.h_,
                    learning_rate=self.J_lr, l1=self.J_l1,
                )
            if iteration % 20 == 0 or iteration == self.n_iter-1:
                self.history_.append({
                    "iteration": iteration,
                    "mse": mse,
                    "variance": self.variance_,
                    "decoder_scale": self.decoder_scale_,
                    "cardinality": float(self.S_.sum(axis=1).mean()),
                    "prior_loss": float(prior_loss),
                })

        self._final_prior_fit()

        # Fixed-prior refinement: J,h stay fixed while codes and decoder adjust.
        for _ in range(self.refinement_iters):
            self.S_, self.Z_ = self._gibbs(X, self.S_, rng, sweeps=2)
            self._decoder_step(
                X, self.Z_, W_lr=0.15, scale_lr=0.05, variance_lr=0.015,
            )
        return self

    def infer(self, X, *, sweeps=30, seed=None):
        """Return one posterior draw as ``(binary_spikes, effective_latents)``."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed+1 if seed is None else seed)
        S = (rng.random((len(X), self.n_components)) < 0.1).astype(float)
        return self._gibbs(X, S, rng, sweeps=sweeps)

    def sample(self, X, *, n_samples=100, burnin=20, thin=5, seed=None):
        """Return separate arrays ``(S_samples, Z_samples)`` after real burn-in."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed+2 if seed is None else seed)
        S = (rng.random((len(X), self.n_components)) < 0.1).astype(float)
        for _ in range(burnin):
            S, Z = self._gibbs(X, S, rng)
        S_samples, Z_samples = [], []
        for _ in range(n_samples):
            for _ in range(thin):
                S, Z = self._gibbs(X, S, rng)
            S_samples.append(S.copy())
            Z_samples.append(Z.copy())
        return np.stack(S_samples), np.stack(Z_samples)

    def reconstruct(self, Z):
        """Decode effective binary or slab latents into observation space."""
        return self.b_+self.decoder_scale_*np.asarray(Z)@self.W_.T


if __name__ == "__main__":
    # Tiny smoke example; replace X with globally scaled observations.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12))
    X /= X.std()
    model = StructuredBiPCA(
        4, slab=False, n_iter=20, prior_delay=5, refinement_iters=2,
    ).fit(X)
    S, Z = model.infer(X[:10])
    print(S.shape, model.reconstruct(Z).shape, model.history_[-1])
