"""
Gibbs / ICM sampler for a sign-constrained Ising model over signed spins
    s in {-1, +1}^n,
with J symmetric, zero-diagonal, off-diagonal entries constrained to
{-1, 0, +1}, and h in R^n unconstrained (matching the RPLE machinery's
spin convention).  The diagonal of J contributes only the constant
trace(J) to the energy, so it is dropped from inference entirely and can
be set post hoc to whatever the face-structure application wants.

`beta` IS DEFINED BY THE CONDITIONAL, which is the only thing this module
ever evaluates (everything here is pseudolikelihood -- the joint's partition
function never appears):
    p(s_i = +1 | s_-i) = sigmoid( beta*f_i ),
    f_i = sum_{j != i} J_ij s_j + h_i.

The joint that implies is p(s) ~ exp( (beta/4) * (s' J s + 2 h' s) ).  You can
have a clean joint or a clean conditional, not both: s'Js counts each pair
twice, and flipping s_i moves it by 2, so the two differ by exactly 4.  Since
every routine here reads the conditional and none reads the joint, the 4 is
spent there.  (Historically it sat in the conditional as sigmoid(4*beta*f);
if you compare against anything written against that convention, its beta is
this one over 4.)

Objective (everything per-sample, nothing extensive in M):
    O(J, h) = (1/M) sum_{i,m} log sigmoid(beta*s_i^m f_i^m)
              - lam * (number of nonzero off-diagonal entries of J)

`lam` is the per-sample cost (in nats) of a nonzero entry -- the direct
analogue of the per-sample L1 strength in the RPLE fitter, so the same
scaling heuristics apply (e.g. lam ~ c*sqrt(log(n^2/delta)/M)).  Under the
spike-and-slab reading, lam = log((1-pi)/(pi/2)) / M.

State maintained incrementally:
    F : (n, M) field matrix, F[i, m] = f_i^m.

Note vs. the 0/1-convention version: with signed spins every datapoint
enters every entry's conditional (there is no "inactive spin" sparsity), so
an entry update costs O(M) and a sweep O(n^2 M) softplus evaluations --
fine under numba for n ~ 1e2, M ~ 1e4.

All hot functions are @numba.njit(cache=True) and mutate J, h, F in place.
"""

import numpy as np
from numba import njit
import math

# ----------------------------------------------------------------------
# numerics
# ----------------------------------------------------------------------

@njit(cache=True, inline='always', fastmath=True)
def _softplus(x):
    # branch-free, numerically stable: max(x,0) + log1p(exp(-|x|))
    ax = np.abs(x)
    return 0.5 * (x + ax) + np.log1p(np.exp(-ax))


@njit(cache=True, inline='always')
def _sigmoid(x):
    if x >= 0.0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


# ----------------------------------------------------------------------
# fields
# ----------------------------------------------------------------------

@njit(cache=True)
def build_fields(J, h, St):
    """F[i, m] = sum_{j != i} J_ij St[j, m] + h_i.

    St is the TRANSPOSED spin matrix, shape (n, M), C-contiguous, entries in
    {-1, +1} -- this layout makes every inner loop a contiguous stream."""
    n, M = St.shape
    F = np.empty((n, M), dtype=np.float64)
    for i in range(n):
        for m in range(M):
            F[i, m] = h[i]
        for j in range(n):
            if j != i and J[i, j] != 0.0:
                Jij = J[i, j]
                for m in range(M):
                    F[i, m] += Jij * St[j, m]
    return F


# ----------------------------------------------------------------------
# per-entry conditional (per-sample scaled)
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _offdiag_logcond(i, j, J, St, F, lam, beta, ell):
    """
    ell[t] <- per-sample log conditional of J_ij at v = t-1 in {-1,0,+1}:

      ell(v) = -lam*|v|
               - (1/M) sum_m [ softplus(-b(t_i^m + (v-cur) sig_m))
                             + softplus(-b(t_j^m + (v-cur) sig_m)) ],

    with t_i^m = s_i^m F_i^m (current fit terms), sig_m = s_i^m s_j^m, and
    cur the current value of J_ij.  Uses log sigmoid(a) = -softplus(-a).
    Single pass over m computing all three candidates.
    """
    M = St.shape[1]
    cur = J[i, j]
    inv_M = 1.0 / M
    d0 = -1.0 - cur
    d1 = -cur
    d2 = 1.0 - cur

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for m in range(M):
        si = St[i, m]
        sj = St[j, m]
        sig = si * sj
        ti = si * F[i, m]
        tj = sj * F[j, m]
        s0 += _softplus(-beta * (ti + d0 * sig)) + _softplus(-beta * (tj + d0 * sig))
        s1 += _softplus(-beta * (ti + d1 * sig)) + _softplus(-beta * (tj + d1 * sig))
        s2 += _softplus(-beta * (ti + d2 * sig)) + _softplus(-beta * (tj + d2 * sig))
    ell[0] = -lam - inv_M * s0
    ell[1] = -inv_M * s1
    ell[2] = -lam - inv_M * s2


@njit(cache=True, inline='always')
def _sample_trinary(ell, temp):
    """Draw t in {0,1,2} from probs ~ exp(ell/temp); temp <= 0 -> argmax."""
    best = 0
    for t in range(1, 3):
        if ell[t] > ell[best]:
            best = t
    if temp <= 0.0:
        return best
    zmax = ell[best]
    tot = 0.0
    for t in range(3):
        ell[t] = np.exp((ell[t] - zmax) / temp)
        tot += ell[t]
    u = np.random.random() * tot
    c = 0.0
    for t in range(3):
        c += ell[t]
        if u <= c:
            return t
    return 2


# ----------------------------------------------------------------------
# one sweep over the off-diagonal entries of J
# ----------------------------------------------------------------------

# @njit(cache=True)
# def gibbs_samp(J, h, s, temp=1, n_samp=1, t0=10, decay_rate=0.8, period=2):

#     burn = period*int(np.log(1e-4/t0)/np.log(decay_rate))

#     n = len(h)
#     # s = np.random.choice([0,1], size=(n, n_samp))
#     for t in range(burn):
#         T = temp + t0*(decay_rate**(t//period))   # anneal a decaying offset down to `temp`
#         for i in range(n):
#             curr = (np.dot(J[i],s) + h[i]) / T
#             curr = np.minimum(np.maximum(curr, -100.0), 100.0)
#             p = 1.0 / (1.0 + np.exp(-curr))
#             s[i] = 1.0*(np.random.rand(n_samp) < p)

#     return s

@njit(cache=True)
def gibbs_samp(J, h, temp=1, n_samp=1, t0=10, decay_rate=0.8, period=2):
    """Gibbs sampler for a signed-spin MRF, s in {-1, +1}.
    """
    burn = period*int(np.log(1e-4/t0)/np.log(decay_rate))

    n = len(h)
    s = 2.0*(np.random.rand(n, n_samp) < 0.5) - 1.0   # random +-1 start
    for t in range(burn):
        T = temp + t0*(decay_rate**(t//period))   # anneal a decaying offset down to `temp`
        for i in range(n):
            curr = (np.dot(J[i], s) + h[i]) / T
            curr = np.minimum(np.maximum(curr, -100.0), 100.0)   # elementwise clip
            p = 1.0 / (1.0 + np.exp(-curr))                      # P(s_i = +1)
            s[i] = 2.0*(np.random.rand(n_samp) < p) - 1.0

    return s

# @njit(cache=True)
# def gibbs_samp(J, h, temp=1, n_samp=1, t0=10, decay_rate=0.8, period=2):
#     burn = period*int(np.log(1e-4/t0)/np.log(decay_rate))
#     n = len(h)
#     s = 2.0*(np.random.rand(n, n_samp) < 0.5) - 1.0
#     for t in range(burn):
#         T = temp + t0*(decay_rate**(t//period))
#         for i in range(n):
#             curr = np.clip((J[i]@s + h[i])/T, -100, 100)         # no factor 2
#             p = 1.0 / (1.0 + np.exp(-curr))
#             s[i] = 2.0*(np.random.rand(n_samp) < p) - 1.0

#     return s

@njit(cache=True)
def gibbs_sweep_J(J, St, F, lam, beta, temp):
    """
    Resample every off-diagonal pair (i < j) from its exact conditional in
    random order; mutates J and F in place.  Returns #entries changed.

    Parameters
    ----------
    J    : (n, n) float64, symmetric, zero diagonal, entries in {-1, 0, 1}
    St   : (n, M) float64 C-contiguous, TRANSPOSED spins in {-1, +1}
    F    : (n, M) float64 from build_fields(J, h, St); kept in sync
    lam  : per-sample cost (nats) of a nonzero entry
    beta : inverse temperature of the model (fixed; schedule it externally)
    temp : sampler temperature in per-sample nats; <= 0 gives ICM
    """
    n = J.shape[0]
    M = St.shape[1]
    n_pairs = n * (n - 1) // 2
    order = np.arange(n_pairs)
    np.random.shuffle(order)
    ell = np.empty(3, dtype=np.float64)
    n_changed = 0

    for e in range(n_pairs):
        idx = order[e]
        i = 0
        rem = idx
        row_len = n - 1
        while rem >= row_len:
            rem -= row_len
            i += 1
            row_len -= 1
        j = i + 1 + rem

        cur = J[i, j]
        _offdiag_logcond(i, j, J, St, F, lam, beta, ell)
        t = _sample_trinary(ell, temp)
        new = float(t - 1)
        if new != cur:
            d = new - cur
            J[i, j] = new
            J[j, i] = new
            for m in range(M):
                F[i, m] += d * St[j, m]
                F[j, m] += d * St[i, m]
            n_changed += 1
    return n_changed


# ----------------------------------------------------------------------
# h: per-row ascent on a concave 1-D objective (h is safe to fit; it does
# not rescale J).  Both fitters maximize the same thing,
#     O(h_i) = (1/M) sum_m log sigmoid(b s_i^m f_i^m) - 0.5*lam_h*h_i^2,
# and both mutate h and F in place.  grad_h is the DEFAULT for callers;
# newton_h is kept for comparison and for well-posed problems.  Why:
#
#   The 1-D problem is concave but, at lam_h = 0, NOT bounded -- if node i's
#   spin is constant given its neighbours (separable: s_i^m f_i^m > 0 for
#   every m), the maximizer is h_i = +-inf.  That is common at low prior
#   temperature, where the fields are sharp enough to pin a node.
#   * newton_h then walks off at a CONSTANT rate: as h grows both g and hess
#     decay like exp(-b h), so step = g/hess -> 1/b every iteration, forever.
#     Worse, once the sigmoids saturate hess underflows toward 0 while g does
#     not, so a single step can jump by ~1e17.  The `hess >= -1e-300` guard
#     never fires: hess -> 0 from below, so it stays "valid" all the way down.
#   * grad_h self-limits: the same exp(-b h) decay makes the STEP shrink, so
#     h creeps like log(t)/b -- unbounded in theory, bounded in any real run.
#     Measured on a separable node (b = 2.4, 50 steps): newton_h reaches
#     h = 21.3 and climbing linearly; grad_h reaches 1.5.
# ----------------------------------------------------------------------

@njit(cache=True)
def grad_h(J, h, St, F, beta, lam_h=0.0, n_steps=3, lr=1.0):
    """
    Per-row gradient ascent on h_i.  `lr` is in units of 1/b^2 -- the curvature
    at the unsaturated optimum is ~b^2/4, so lr=1 is a step of about 4x Newton's
    there, and lr <~ 2 is stable.  Mutates h and F in place.
    """
    n, M = F.shape
    inv_M = 1.0 / M
    step_scale = lr / (beta * beta)
    for i in range(n):
        for _ in range(n_steps):
            g = -lam_h * h[i]
            for m in range(M):
                a = beta * St[i, m] * F[i, m]
                g += inv_M * beta * St[i, m] * _sigmoid(-a)
            step = step_scale * g
            h[i] += step
            for m in range(M):
                F[i, m] += step
            if np.abs(step) < 1e-10:
                break

@njit(cache=True)
def newton_h(J, h, St, F, beta, lam_h=0.0, n_steps=3):
    """
    Per-row Newton steps on h_i.  UNSAFE at lam_h = 0 on separable rows -- see
    the note above; prefer grad_h unless you know h is well determined.
    Mutates h and F in place.
    """
    n, M = F.shape
    inv_M = 1.0 / M
    for i in range(n):
        for _ in range(n_steps):
            g = -lam_h * h[i]
            hess = -lam_h
            for m in range(M):
                a = beta * St[i, m] * F[i, m]
                sg = _sigmoid(-a)                     # d/da of log sigmoid(a)
                g += inv_M * beta * St[i, m] * sg
                hess -= inv_M * beta * beta * sg * (1.0 - sg)
            if hess >= -1e-300:
                break
            step = g / hess
            h[i] -= step
            for m in range(M):
                F[i, m] -= step
            if np.abs(step) < 1e-10:
                break


# ---------------------------------------------------------------------
# beta: uniform scaling
# can either learn with the pseudolikelihood or monte carlo MLE
# ---------------------------------------------------------------------

@njit(cache=True)
def ple_beta(J, h, beta, St, F, lr=1e-3, J_lam=0):
    """One exponentiated-gradient step on the per-sample pseudolikelihood.
    RETURNS the new beta (floats cannot be mutated in place in njit)."""
    n, M = F.shape
    g = 0.0
    for i in range(n):
        for m in range(M):
            a = St[i, m]*F[i, m]
            g += a*_sigmoid(-beta*a)
    g /= n*M                       # per-sample, per-spin: lr transfers across sizes
    
    return beta*np.exp(lr*(g - J_lam*np.sum(np.abs(J))))

@njit(cache=True)
def mle_beta(J, h, beta, St, F, lr=1e-3, n_samp=0, J_lam=0):
    """One exponentiated-gradient step on the joint likelihood via moment matching.
    Statistic T(s) = (s'Js + 2h's)/4 = (sum_i s_i(f_i + h_i))/4.  RETURNS beta."""
    n, M = St.shape
    if n_samp <= 0:
        n_samp = 4*n

    mod = gibbs_samp(beta*J, beta*h, n_samp=n_samp)
    Fm = build_fields(J, h, mod)
    t_d = 0.0
    for i in range(n):
        for m in range(M):
            t_d += St[i, m]*(F[i, m] + h[i])
    t_m = 0.0
    for i in range(n):
        for m in range(n_samp):
            t_m += mod[i, m]*(Fm[i, m] + h[i])
    g = 0.25*(t_d/M - t_m/n_samp)/n

    return beta*np.exp(lr*(g - J_lam*np.sum(np.abs(J))))

# ----------------------------------------------------------------------
# monitoring
# ----------------------------------------------------------------------

@njit(cache=True)
def objective(J, St, F, lam, beta):
    """Per-sample objective: mean log pseudolikelihood minus lam * nnz(J)."""
    n, M = F.shape
    ll = 0.0
    for i in range(n):
        for m in range(M):
            ll -= _softplus(-beta * St[i, m] * F[i, m])
    ll /= M
    nnz = 0
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                nnz += 1
    return ll - lam * nnz


# ----------------------------------------------------------------------
# warm start from the RPLE fit
# ----------------------------------------------------------------------

def discretize_rple(J_W, J_h, temp=1.0, thr=None):
    """
    Convert an RPLE fit (spin convention, conditional log-odds
    2*(J_W s + J_h)/temp, symmetrized W afterwards) into (J0, h0, beta0)
    for this model, whose conditional log-odds are beta*(J0 s + h0).
    Matching gives  2*W_sym/temp ~ beta*J,  2*J_h/temp ~ beta*h, so

        beta0 = median(2*|W_sym_ij| / temp) over detected edges
        J0    = sign(W_sym) where |W_sym| > thr, else 0
        h0    = 2*J_h / (beta0*temp)

    beta0 is a suggested *scale* for your external beta schedule (e.g. its
    endpoint), not a fitted parameter.
    """
    W = np.asarray(J_W, dtype=np.float64)
    W = (W + W.T) / 2.0
    n = W.shape[0]
    offmask = ~np.eye(n, dtype=bool)
    mags = np.abs(W[offmask])
    if thr is None:
        big = mags[mags > 0.1 * max(mags.max(), 1e-12)]
        thr = 0.5 * np.median(big) if big.size else 0.5
    sel = (np.abs(W) > thr) & offmask
    beta0 = float(2.0 * np.median(np.abs(W)[sel]) / temp) if sel.any() else 1.0
    J0 = np.where(sel, np.sign(W), 0.0)
    h0 = 2.0 * np.asarray(J_h, dtype=np.float64) / (beta0 * temp)
    return J0, h0, beta0


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def fit_sign_ising(S, lam=None, beta_schedule=None, temp_schedule=None,
                   n_sweeps=150, J_init=None, h_init=None,
                   fit_h=True, lam_h=0.0, h_lr=1.0, h_newton=False,
                   seed=None, verbose=False):
    """
    Annealed fit with externally scheduled beta.

    S may be in {-1,+1} or {0,1} (auto-converted to spins).
    lam defaults to sqrt(log(n^2 * 1e3) / M), mirroring the RPLE default.
    beta_schedule defaults to a ramp 1.0 -> 4.0 (you will generally want to
    supply your own, e.g. ending at the scale suggested by discretize_rple).
    temp_schedule defaults to per-sample geomspace(0.5, 0.005) with a final
    10% of ICM sweeps.
    h is fit by grad_h; `h_newton=True` switches back to newton_h, which is
    faster to converge but walks off on separable rows (see the note there).

    Returns (J, h, history); history rows are
    (sweep, temp, beta, n_changed, objective).
    """
    S = np.asarray(S, dtype=np.float64)
    if S.min() >= 0.0:
        S = 2.0 * S - 1.0
    M, n = S.shape
    St = np.ascontiguousarray(S.T)
    if seed is not None:
        np.random.seed(seed)
        _seed_numba(seed)

    if lam is None:
        lam = np.sqrt(np.log(n ** 2 * 1e3) / M)
    if beta_schedule is None:
        beta_schedule = np.linspace(1.0, 4.0, n_sweeps)
    if temp_schedule is None:
        n_anneal = int(0.9 * n_sweeps)
        temp_schedule = np.concatenate([
            np.geomspace(0.5, 0.005, n_anneal),
            np.zeros(n_sweeps - n_anneal),
        ])

    J = np.zeros((n, n)) if J_init is None else np.ascontiguousarray(J_init, dtype=np.float64)
    np.fill_diagonal(J, 0.0)
    h = np.zeros(n) if h_init is None else np.ascontiguousarray(h_init, dtype=np.float64)
    F = build_fields(J, h, St)

    history = []
    for t in range(n_sweeps):
        beta = float(beta_schedule[min(t, len(beta_schedule) - 1)])
        temp = float(temp_schedule[min(t, len(temp_schedule) - 1)])
        if fit_h:
            if h_newton:
                newton_h(J, h, St, F, beta, lam_h)
            else:
                grad_h(J, h, St, F, beta, lam_h, 3, h_lr)
        n_changed = gibbs_sweep_J(J, St, F, lam, beta, temp)
        obj = objective(J, St, F, lam, beta)
        history.append((t, temp, beta, n_changed, obj))
        if verbose and (t % max(1, n_sweeps // 10) == 0 or t == n_sweeps - 1):
            print(f"sweep {t:4d}  temp={temp:7.4f}  beta={beta:6.3f}  "
                  f"changed={n_changed:4d}  obj={obj:.5f}")
    return J, h, np.array(history)


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


if __name__ == "__main__":
    # smoke test: recover a planted sign Ising model with beta scheduled, not fit
    rng = np.random.default_rng(0)
    n, M = 15, 4000
    beta_true = 2.4                 # conditional is sigmoid(beta*f) -- was 4*0.6

    J_true = np.zeros((n, n))
    for _ in range(18):
        i, j = rng.choice(n, size=2, replace=False)
        J_true[i, j] = J_true[j, i] = rng.choice([-1.0, 1.0])
    h_true = 0.3 * rng.standard_normal(n)

    S = rng.choice([-1.0, 1.0], size=(M, n))
    for _ in range(80):
        for i in range(n):
            f = S @ J_true[i] - J_true[i, i] * S[:, i] + h_true[i]
            p = 1.0 / (1.0 + np.exp(-beta_true * f))
            S[:, i] = np.where(rng.random(M) < p, 1.0, -1.0)

    off = ~np.eye(n, dtype=bool)
    nz = J_true[off] != 0

    def report(tag, J_hat, h_hat):
        tp = int(((J_hat[off] != 0) & nz).sum())
        fp = int(((J_hat[off] != 0) & ~nz).sum())
        sa = (J_hat[off][nz] == J_true[off][nz]).mean()
        hc = np.corrcoef(h_hat, h_true)[0, 1]
        print(f"{tag}: TP={tp}/{int(nz.sum())} FP={fp} sign-acc={sa:.2f} "
              f"corr(h, h_true)={hc:.2f}")

    # beta ramped up to the true scale (in practice: to discretize_rple's beta0)
    bs = np.linspace(0.8, beta_true, 150)
    J_hat, h_hat, hist = fit_sign_ising(S, n_sweeps=150, beta_schedule=bs,
                                        seed=1, verbose=True)
    report("cold start, beta ramp", J_hat, h_hat)
    J_n, h_n, _ = fit_sign_ising(S, n_sweeps=150, beta_schedule=bs, seed=1,
                                 h_newton=True)
    report("   same, h by newton  ", J_n, h_n)
    print(f"max|h|: grad {np.abs(h_hat).max():.3f}  newton {np.abs(h_n).max():.3f}"
          f"  (true {np.abs(h_true).max():.3f})")
