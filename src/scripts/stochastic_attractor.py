import numpy as np
from scipy.special import i0, iv
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from tqdm import tqdm

CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
import sys
sys.path.append(CODE_DIR)
import util
import plotting as tpl

np.random.seed(0)

#%%
def solve_trig_poly(a_coeffs, b_coeffs, tol=1e-5):
    """
    Solves sum(a_n cos(nx) + b_n sin(nx)) = 0 for x in [0, 2*pi)
    a_coeffs: list of a_n from n=0 to N
    b_coeffs: list of b_n from n=0 to N (b_0 is usually 0)
    """
    N = len(a_coeffs)
    
    # Initialize complex polynomial coefficients array of size 2N + 1
    # np.roots expects coefficients from highest degree down to constant
    poly_coeffs = np.zeros(2 * N + 1, dtype=complex)
    
    # Center term (n=0)
    poly_coeffs[N] = 0
    
    for n in range(1,N+1):
        an = a_coeffs[n-1]
        bn = b_coeffs[n-1]
        
        # Term for z^(N+n)
        poly_coeffs[N - n] += (an - 1j * bn) / 2
        
        # Term for z^(N-n)
        poly_coeffs[N + n] += (an + 1j * bn) / 2
        
    # Find all roots of the algebraic polynomial
    z_roots = np.roots(poly_coeffs)
    
    # Filter for roots on the unit circle (where magnitude is approx 1)
    # These correspond to real-valued x
    real_x_roots = []
    for z in z_roots:
        if abs(abs(z) - 1.0) < tol:
            # Extract the angle x and normalize to [0, 2pi)
            x = np.angle(z) % (2 * np.pi)
            real_x_roots.append(x)
            
    # Remove any duplicates caused by numerical precision
    if real_x_roots:
        real_x_roots = np.unique(np.round(real_x_roots, decimals=5))
        
    return real_x_roots

# #%%
# # ── Parameters ────────────────────────────────────────────────────────────────
# kappa_k = 5.0     # kernel concentration
# alpha   = 1.0     # attraction strength
# beta    = 10.0     # noise amplitude in R^n (brute force)
# n       = 500     # ambient dimension
# N_grid  = 1000    # theta grid resolution for projection
# dt      = 5e-3
# T       = 20.0
# n_steps = int(T / dt)
# N_traj  = 6000     # independent trajectories
# # N_traj  = 1     # independent trajectories

# # Effective noise for diagonalized system.
# # The brute-force noise f'(th)^T * beta*dW ~ beta*sqrt(n*k11) * dW_1D,
# # and H ~ n*(-q_tilde''), so d(hat_theta) ~ beta/sqrt(n) * dW_1D / (-q_tilde'').
# # Hence beta_eff = beta / sqrt(n).
# beta_eff = beta / np.sqrt(n)

# # ── Kernel eigenvalues: lambda_m = I_m(kappa) / (exp(kappa) - I_0(kappa)) ───
# denom_k = np.exp(kappa_k) - i0(kappa_k)
# k11     = kappa_k * np.exp(kappa_k) / denom_k   # = 2*sum_{m>=1} m^2*lambda_m

# M = 1
# while iv(M + 1, kappa_k) / denom_k > 1e-8:
#     M += 1
# modes = np.arange(1, M + 1)
# lam_m = iv(modes, kappa_k) / denom_k     # (M,)

# print(f"kappa={kappa_k}, alpha={alpha}, beta={beta}, n={n}")
# print(f"beta_eff = {beta_eff:.4f}, k11 = {k11:.4f}, M = {M}")
# print(f"lambda_m = {np.round(lam_m, 5)}")
# print(f"Expected hat_theta std at T=1 (near-manifold approx): "
#       f"{beta_eff / np.sqrt(k11):.3f} rad\n")

# # ── Sample GP paths on grid ───────────────────────────────────────────────────
# theta_grid = np.linspace(0, 2 * np.pi, N_grid, endpoint=False)
# dth   = theta_grid[:, None] - theta_grid[None, :]
# K_mat = (np.exp(kappa_k * np.cos(dth)) - i0(kappa_k)) / denom_k
# K_mat += 1e-9 * np.eye(N_grid)
# L_ch  = np.linalg.cholesky(K_mat)
# # F[i, j] = j-th independent GP draw evaluated at theta_grid[i]
# # f(theta)^T f(theta') / n  ->  k(theta, theta') by LLN
# F       = L_ch @ np.random.randn(N_grid, n)   # (N_grid, n), unnormalised
# F_norm2 = np.sum(F ** 2, axis=1)              # (N_grid,)

# # Vectorised grid-search projection
# def project(x_arr):
#     xn2 = np.sum(x_arr ** 2, axis=1)                             # (N_traj,)
#     D   = F_norm2[None, :] - 2 * (x_arr @ F.T) + xn2[:, None]  # (N_traj, N_grid)
#     idx = np.argmin(D, axis=1)
#     return idx, theta_grid[idx]

# # ── Initial condition ─────────────────────────────────────────────────────────
# theta0 = np.pi
# idx0   = np.argmin(np.abs(theta_grid - theta0))
# x0_bf  = F[idx0].copy()    # start on manifold
# #%%
# # ════════════════════════════════════════════════════════════════════════════
# # BRUTE-FORCE  (vectorised over N_traj trajectories)
# # ════════════════════════════════════════════════════════════════════════════
# print("Running brute-force...")
# x_bf          = np.tile(x0_bf, (N_traj, 1))
# idx_bf, th_bf = project(x_bf)
# # th_dyn = th_bf.copy()

# bf_thetas = []
# # dyn_thetas = []
# for step in range(n_steps):
#     if step % 1000 == 0:
#         print(f"  step {step}/{n_steps}")
#     bf_thetas.append(1*th_bf)
#     fv    = F[idx_bf]
#     dW    = np.random.randn(N_traj, n) * np.sqrt(dt)
#     x_bf += alpha * (fv - x_bf) * dt + beta * dW
#     idx_bf, th_bf = project(x_bf)

# theta_bf_final = th_bf.copy()

# # ════════════════════════════════════════════════════════════════════════════
# # DIAGONALIZED SYSTEM
# #
# # q_tilde(theta,t) = x(t)^T f(theta) / n  (normalised overlap)
# #
# # Fourier expansion:
# #   q_tilde(theta) = sum_{m=1}^M [a_m(t)*cos(m*theta) + b_m(t)*sin(m*theta)]
# #   Near manifold: a_m -> 2*lambda_m*cos(m*hat_theta),
# #                  b_m -> 2*lambda_m*sin(m*hat_theta)
# #   => -q_tilde''(hat_theta) -> 2*sum_m m^2*lambda_m = k11  ✓
# #
# # SDEs:
# #   da_m = alpha*(2*lam_m*cos(m*th) - a_m)*dt + beta_eff*sqrt(2*lam_m)*dWc_m
# #   db_m = alpha*(2*lam_m*sin(m*th) - b_m)*dt + beta_eff*sqrt(2*lam_m)*dWs_m
# #
# #   d(hat_theta) = [beta_eff / (-q_tilde'')] o dW_eff   [Stratonovich]
# #   where dW_eff = sum_m m*sqrt(2*lam_m)*(-sin(m*th)*dWc + cos(m*th)*dWs)
# #   has variance k11*dt — collapsing to a single 1D BM.
# #
# # Scheme: Stratonovich-Heun for hat_theta, Euler-Maruyama for modes.
# # The SAME dWc, dWs are shared between both updates (correlated noise sources).
# # ════════════════════════════════════════════════════════════════════════════
# print("\nRunning diagonalized system...")

# #%%
# hat_theta = np.full(N_traj, theta_grid[idx0])
# mth0 = modes[None, :] * hat_theta[:, None]
# a_m  = 2 * lam_m[None, :] * np.cos(mth0)
# b_m  = 2 * lam_m[None, :] * np.sin(mth0)

# def q(theta, a, b):
#     """-q_tilde''(theta) = sum_m (a_m*cos + b_m*sin). Returns (N_traj,)."""
#     mth = modes * theta[..., None]
#     return np.sum((a * np.cos(mth) + b * np.sin(mth)), axis=-1)

# def qp(theta, a, b):
#     """-q_tilde'(theta) = sum_m m*(-a_m*sin + b_m*cos). Returns (N_traj,)."""
#     mth = modes * theta[..., None]
#     return np.sum(modes[None, :] * (-a * np.sin(mth) + b * np.cos(mth)), axis=-1)

# def neg_qpp(theta, a, b):
#     """-q_tilde''(theta) = sum_m m^2*(a_m*cos + b_m*sin). Returns (N_traj,)."""
#     mth = modes * theta[..., None]
#     return np.sum(modes[None, :] ** 2 * (a * np.cos(mth) + b * np.sin(mth)), axis=-1)


# thetas = []
# coefs = []
# for step in range(n_steps):
#     if step % 1000 == 0:
#         nq = neg_qpp(hat_theta, a_m, b_m)
#         print(f"  step {step}/{n_steps},"
#               f" mean(-q'')={nq.mean():.3f}, min(-q'')={nq.min():.3f}")

#     thetas.append(hat_theta*1)
#     coefs.append([a_m*1, b_m*1])

#     # Shared mode noise increments
#     dWc = np.random.randn(N_traj, M) * np.sqrt(dt)
#     dWs = np.random.randn(N_traj, M) * np.sqrt(dt)

#     # Effective 1D noise for hat_theta (var = k11*dt)
#     mth    = modes[None, :] * hat_theta[:, None]
#     dW_eff = np.sum(modes[None, :] * np.sqrt(2 * lam_m[None, :]) *
#                     (-np.sin(mth) * dWc + np.cos(mth) * dWs), axis=1)

#     # Stratonovich-Heun step for hat_theta
#     nq_curr    = neg_qpp(hat_theta, a_m, b_m)
#     sigma_curr = beta_eff / nq_curr
#     th_pred    = hat_theta + sigma_curr * dW_eff
#     sigma_pred = beta_eff / neg_qpp(th_pred, a_m, b_m)
#     hat_theta  = (hat_theta + 0.5 * (sigma_curr + sigma_pred) * dW_eff) % (2 * np.pi)

#     # Euler-Maruyama for modes (drift uses theta before Heun step)
#     a_m += (alpha * (2 * lam_m[None, :] * np.cos(mth) - a_m) * dt
#             + beta_eff * np.sqrt(2 * lam_m[None, :]) * dWc)
#     b_m += (alpha * (2 * lam_m[None, :] * np.sin(mth) - b_m) * dt
#             + beta_eff * np.sqrt(2 * lam_m[None, :]) * dWs)

# theta_diag_final = hat_theta.copy()


# #%%

# C = np.array(coefs[::10]).squeeze()

# fig, axs = plt.subplots(1,5)

# for m in range(5):
#     axs[m].plot(2*lam_m[m]*np.cos(theta_grid), 2*lam_m[m]*np.sin(theta_grid), 'k--')
#     axs[m].plot(C[:,0,m], C[:,1,m])

#     axs[m].set_xlim([-1,1])
#     axs[m].set_ylim([-1,1])
#     tpl.square_axis(axs[m])

# #%%

# sols = [[solve_trig_poly(c[1,i]*np.arange(1,M+1), -c[0,i]*np.arange(1,M+1)) for i in range(c.shape[1])] for c in C]

# crits = np.concatenate(sols)
# tims = np.concatenate([i*np.ones(len(s)) for i,s in enumerate(sols)])
# curvs = np.concatenate([-neg_qpp(s, C[i,0], C[i,1]) for i,s in enumerate(sols)])
# vals = np.concatenate([q(s, C[i,0], C[i,1]) for i,s in enumerate(sols)])
# amax = np.array([s[np.argmax(q(s, C[i,0], C[i,1]))] for i,s in enumerate(sols)])

# #%%
# plt.scatter(tims, crits-np.pi, c=np.sign(curvs), cmap='bwr', s=1)

# plt.scatter(np.arange(len(amax)), amax-np.pi, 
#             marker='o', facecolors='none', edgecolors='b')

# plt.plot(np.array(thetas) - np.pi, 'k--')

#%%
"""
GPCurveSimulator
================
Simulates the stochastic dynamics of the projection theta_hat(t) of a noise-
driven process onto a GP curve, in the INFINITE-n LIMIT.

The full-dimensional system is:

    dx = alpha * (f(theta_hat) - x) dt + beta dW,   x in R^n
    theta_hat(t) = argmin_theta ||f(theta) - x(t)||^2

In the infinite-n limit this is equivalent to tracking the Fourier
coefficients of the normalised overlap

    q_t(theta) = x(t)^T f(theta) / n

via the 2M-dimensional SDE (modes m = 1, ..., M):

    da_m = alpha * (2 lambda_m cos(m theta_hat) - a_m) dt
           + beta * sqrt(2 lambda_m) dW_m^c

    db_m = alpha * (2 lambda_m sin(m theta_hat) - b_m) dt
           + beta * sqrt(2 lambda_m) dW_m^s

where lambda_m = I_m(kappa) / (exp(kappa) - I_0(kappa)) are the kernel
eigenvalues and the 2M noise sources are independent standard Brownian
motions.  theta_hat is recovered each step as the global argmax of q_t(theta)
by finding the zeros of the trigonometric polynomial

    q_t'(theta) = sum_m  m * (-a_m sin(m theta) + b_m cos(m theta)) = 0

via np.roots applied to the equivalent algebraic polynomial (degree 2M),
filtering for roots on the unit circle, and selecting the maximum.

Parameters
----------
kappa : float   — kernel concentration (controls smoothness and M)
alpha : float   — mean-reversion strength
beta  : float   — noise amplitude (effective, already incorporates 1/sqrt(n))
"""

import numpy as np
from scipy.special import i0, iv


class GPCurveSimulator:
    """Infinite-n eigenbasis simulator for GP-curve attraction dynamics."""

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, kappa: float, alpha: float, beta: float,
                 lam_thresh: float = 1e-10):
        """
        Parameters
        ----------
        kappa       : kernel concentration
        alpha       : mean-reversion strength
        beta        : effective noise amplitude
        lam_thresh  : eigenvalues below this threshold are dropped
        """
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.beta  = float(beta)

        # ── Kernel eigenvalues ────────────────────────────────────────────────
        denom      = np.exp(kappa) - i0(kappa)
        self._denom = denom
        self.k11   = kappa * np.exp(kappa) / denom

        M = 1
        while iv(M + 1, kappa) / denom > lam_thresh:
            M += 1
        self.M     = M
        self.modes = np.arange(1, M + 1)                   # (M,)
        self.lam_m = iv(self.modes, kappa) / denom         # (M,)
        self._noise_std = self.beta * np.sqrt(2 * self.lam_m)  # (M,)

        # ── Polynomial coefficient builder (precomputed index arrays) ─────────
        # For trajectory i, P_i(z) = sum_m m[(b_m+ia_m)/2 * z^{M-m}
        #                                    +(b_m-ia_m)/2 * z^{M+m}]  (wait:)
        # np.roots convention: coeffs[0]*z^{2M} + ... + coeffs[2M] = 0
        # z^{M+m} corresponds to index 2M-(M+m) = M-m
        # z^{M-m} corresponds to index 2M-(M-m) = M+m
        self._idx_hi = M - self.modes   # indices for z^{M+m} (higher powers)
        self._idx_lo = M + self.modes   # indices for z^{M-m} (lower powers)

    # ── Trig-polynomial argmax ────────────────────────────────────────────────

    def q(self, theta, a, b):
        """-q_tilde''(theta) = sum_m (a_m*cos + b_m*sin). Returns (N_traj,)."""
        mth = self.modes * theta[..., None]
        return np.sum((a * np.cos(mth) + b * np.sin(mth)), axis=-1)

    def _modes(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mth = self.modes * theta[..., None]      # (N, M)
        a   = 2 * self.lam_m[None] * np.cos(mth)    # (N, M)
        b   = 2 * self.lam_m[None] * np.sin(mth)    # (N, M)
        return a, b

    def _poly_argmax(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Exact global argmax of q_t(theta) = sum_m[a_m cos(m*th)+b_m sin(m*th)]
        via np.roots on the degree-2M companion polynomial of q_t'(theta)=0.

        a, b  : (n_traj, M)
        Returns hat_theta : (n_traj,) in [0, 2*pi)
        """
        M, modes = self.M, self.modes
        n_traj   = len(a)

        # Build all polynomial coefficient arrays at once: (n_traj, 2M+1)
        # Coefficient of z^{M+m}: m*(b_m + i*a_m)/2  => stored at index M-m
        # Coefficient of z^{M-m}: m*(b_m - i*a_m)/2  => stored at index M+m
        coeffs = np.zeros((n_traj, 2*M + 1), dtype=complex)
        for k, m in enumerate(modes):
            coeffs[:, M - m] += m * (b[:, k] + 1j * a[:, k]) / 2
            coeffs[:, M + m] += m * (b[:, k] - 1j * a[:, k]) / 2
        # coeffs[:, M] = 0  (no z^M term — no constant in q_t')

        hat = np.empty(n_traj)

        for i in range(n_traj):
            # Roots of degree-2M polynomial
            roots = np.roots(coeffs[i])

            # Keep only roots on (or very near) the unit circle
            on_unit = roots[np.abs(np.abs(roots) - 1.0) < 0.05]

            if len(on_unit) == 0:
                # Degenerate case: fall back to 0
                hat[i] = 0.0
                continue

            thetas = np.angle(on_unit) % (2 * np.pi)   # (K,)

            # Evaluate q_t and q_t'' at each candidate
            mth = modes[:, None] * thetas[None, :]      # (M, K)
            cos_mth = np.cos(mth)
            sin_mth = np.sin(mth)
            qt_vals = np.sum(a[i, :, None] * cos_mth
                           + b[i, :, None] * sin_mth, axis=0)   # (K,)
            qtpp    = -np.sum(modes[:, None]**2 * (
                               a[i, :, None] * cos_mth
                             + b[i, :, None] * sin_mth), axis=0)  # (K,)

            # Restrict to local maxima (q_t'' < 0); fall back to all if none
            is_max = qtpp < 0
            if not is_max.any():
                is_max = np.ones(len(thetas), dtype=bool)

            hat[i] = thetas[is_max][np.argmax(qt_vals[is_max])]

        return hat

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate(self,
                 n_traj: int,
                 T: float,
                 dt: float,
                 theta0: float = 0.0,
                 return_full: bool = False,
                 seed: int | None = None) -> dict:
        """
        Simulate n_traj independent trajectories from theta0 to time T.

        Parameters
        ----------
        n_traj      : number of independent trajectories
        T           : total simulation time
        dt          : Euler-Maruyama time step
        theta0      : starting value of theta_hat (on-manifold initial condition)
        return_full : if True, store full (n_traj, n_steps+1) trajectory array;
                      if False (default), store only final (n_traj,) values
        seed        : RNG seed for reproducibility

        Returns
        -------
        dict with keys:
            'theta_hat' : (n_traj,) or (n_traj, n_steps+1) depending on return_full
            't'         : (n_steps+1,) time array  [only if return_full=True]
            'n_steps'   : int
            'M'         : number of modes
            'k11'       : kernel second spectral moment
        """
        n_steps    = int(T / dt)
        alpha      = self.alpha
        noise_std  = self._noise_std   # beta * sqrt(2*lam_m), (M,)
        rng        = np.random.default_rng(seed)

        # ── Initial condition: on the manifold at theta0 ──────────────────────
        # Near-manifold: a_m ≈ 2*lam_m*cos(m*theta0), b_m ≈ 2*lam_m*sin(m*theta0)
        
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.ndim == 0:
            theta0 = np.full(n_traj, float(theta0))
        elif theta0.shape != (n_traj,):
            raise ValueError(...)
        a_m, b_m = self._modes(theta0)

        hat_theta = self._poly_argmax(a_m, b_m)                  # (n_traj,)

        if return_full:
            traj    = np.empty((n_traj, n_steps + 1))
            traj[:, 0] = hat_theta
            
            cos_cf   = np.empty((n_traj, n_steps + 1, len(self.modes)))
            sin_cf   = np.empty((n_traj, n_steps + 1, len(self.modes)))
            
            cos_cf[:, 0] = 1*a_m
            sin_cf[:, 0] = 1*b_m

        sqdt = np.sqrt(dt)

        for step in tqdm(range(n_steps)):
            # Independent Brownian increments for each mode (2M total)
            dWc = rng.standard_normal((n_traj, self.M)) * sqdt   # (n_traj, M)
            dWs = rng.standard_normal((n_traj, self.M)) * sqdt

            # Euler-Maruyama update
            a_eq, b_eq = self._modes(hat_theta)
            a_m  += (alpha * (a_eq - a_m) * dt + noise_std[None, :] * dWc)
            b_m  += (alpha * (b_eq - b_m) * dt + noise_std[None, :] * dWs)

            # Recover theta_hat via exact trig-poly root finding
            hat_theta = self._poly_argmax(a_m, b_m)

            if return_full:
                traj[:, step + 1] = hat_theta
                cos_cf[:, step + 1] = 1*a_m
                sin_cf[:, step + 1] = 1*b_m

        result = {'n_steps': n_steps, 'M': self.M, 'k11': self.k11}
        if return_full:
            result['theta_hat'] = traj
            result['a_m'] = cos_cf
            result['b_m'] = sin_cf
        else:
            result['theta_hat'] = hat_theta

        return result

    def __repr__(self) -> str:
        return (f"GPCurveSimulator(kappa={self.kappa}, alpha={self.alpha}, "
                f"beta={self.beta}, M={self.M}, k11={self.k11:.4f})")
    
#%%

n_samp = 1000


sim = GPCurveSimulator(kappa=5, alpha=20, beta=1, lam_thresh=1e-5)

result = sim.simulate(n_traj=n_samp, T=5.0, dt=0.001, theta0=np.pi, 
                      # np.random.rand(n_samp)*2*np.pi
                      return_full=True, seed=0)


#%%

k = 0

a_p, b_p = sim._modes(result['theta_hat'])

plt.plot(result['a_m'][0,:,k].T, result['b_m'][0,:,k].T)
plt.plot(a_p[...,k].T, b_p[...,k].T, 'k-')

for t in range(result['n_steps']):
    if not t % 10:
        a_m = result['a_m'][0,t,k]
        b_m = result['b_m'][0,t,k]
        plt.plot([a_m, a_p[0,t,k]],[b_m,b_p[0,t,k]])
        
#%%

k = 0

lam = np.sqrt(sim.lam_m[k])

plt.plot(result['a_m'][:10][...,k].T, result['b_m'][:10][...,k].T)
# plt.scatter(result['a_m'][:,0,k], result['b_m'][:,0,k], marker='o')
# plt.scatter(result['a_m'][:,-1,k], result['b_m'][:,-1,k], marker='*')
x = np.linspace(-np.pi, np.pi)
plt.plot(np.cos(x)*lam, np.sin(x)*lam, 'k--')

# plt.plot(result['a_m'][:10][...,k].T, result['b_m'][:10][...,k].T)

#%%

t0 = result['theta_hat'][:,0]
tfin = result['theta_hat'][:,-1]

crit = 2*np.arccos((np.sqrt(1 + 4*sim.kappa**2) - 1)/(2*sim.kappa))
guess = np.abs(tfin - np.pi) > crit

targ = sim.q(t0[:,None], result['a_m'], result['b_m'])
actual = sim.q(tfin[:,None], result['a_m'], result['b_m'])

plt.plot(targ[guess].mean(0))
plt.plot(actual[guess].mean(0))
plt.plot(targ[~guess].mean(0))
plt.plot(actual[~guess].mean(0))

#%%

n_samp = 1000
T = 1
dt = 1e-3

# alpha = 20
# alpha = 0.05
alpha = 0.1
beta = 0.2

targ = np.zeros((4, int(T/dt)+1, n_samp))
actual = np.zeros((4, int(T/dt)+1, n_samp))
resp = np.zeros((4, n_samp))
for i,kap in enumerate([1e-5, 1e-1, 1, 5]):
    
    sim = GPCurveSimulator(kappa=kap, alpha=20, beta=1, lam_thresh=1e-5)
    
    result = sim.simulate(n_traj=n_samp, T=T, dt=dt, theta0=np.pi, # np.random.rand(n_samp)*2*np.pi
                          return_full=True, seed=0)

    t0 = result['theta_hat'][:,0]
    tfin = result['theta_hat'][:,-1]
    
    crit = 2*np.arccos((np.sqrt(1 + 4*sim.kappa**2) - 1)/(2*sim.kappa))
    guess = np.abs(tfin - np.pi) > crit
    
    
    baseline = sim.q(np.array([0]), result['a_m'], result['b_m']).T
    targ[i] = sim.q(t0[:,None], result['a_m'], result['b_m']).T - baseline
    actual[i] = sim.q(tfin[:,None], result['a_m'], result['b_m']).T - baseline
    resp[i] = 1*tfin

#%%




