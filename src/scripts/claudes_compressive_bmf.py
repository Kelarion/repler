"""
Compressive Boltzmann-prior binary matrix factorization -- exact k=10 sandbox.

Everything from our conversation in one file, faithful to the runs (same seeds
and hyperparameters, so the numbers reproduce):

  Model:   X ~ N(s W^T, sigma^2 I),  s in {0,1}^k,
           p_theta(s) = exp(theta . T(s)) / Z,  T(s) = ({s_i s_j}_{i<j}, {s_i})
           (natural params theta <-> energy convention s^T J s + 2 h^T s via
            theta_pair(ij) = 2 J_ij, theta_field(i) = 2 h_i; ground states =
            MAXIMIZERS of s^T J s + 2 h^T s, i.e. p ~ exp(+f))
  Ground truth: s uniform over the 11 ground states of the given (J, h)
           (the tree-path codebook), N=2000, D_obs=30, sigma=1 (held fixed).

  Objective per sample:  D + kappa * KL[q || p_theta]  (+ lam * E|s|)  (+ gamma * H(p_theta))
    E-step (exact):  q(s|x) prop p_theta(s) * exp((loglik - lam|s|)/kappa)
    M-step W:        weighted least squares (kappa-free)
    M-step theta:    moment matching to aggregated q (kappa drops out);
                     optional explicit entropy penalty via grad H = -Cov_p(T) theta

  Entry points (bottom of file):
    sweep_fixed_kappa()   -- traces the R-D frontier; slopes -dD/dR bracket kappa
    sweep_geco(Dstars)    -- GECO dual ascent on log kappa; lands at D = D*
    run_k0()              -- kappa=0 hard EM, no prior (codebook explodes)
    run_sparse(lam)       -- unit usage cost, cold starts (gauge-fixing check)
    analyze(...)          -- cross-tab vs true states, subtree connectivity,
                             cube-graph (Hamming-1 edges), permham

Requires: numpy, scipy, matplotlib.
"""
import numpy as np
from itertools import product, combinations
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
import scipy.stats as sts
import numpy.linalg as nla

#%%
# ----------------------------------------------------------------------------
# ground truth and data
# ----------------------------------------------------------------------------
rng = np.random.default_rng(0)
k = 10
S = np.array(list(product([0, 1], repeat=k)), dtype=float)          # (1024, k)

J_gt = np.array([[ 0,-1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [-1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                 [ 1, 0, 0,-1, 0, 0, 0, 0, 0, 0],
                 [ 1, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 1, 0, 0, 0,-1, 1, 1, 0, 0],
                 [ 0, 1, 0, 0,-1, 0, 0, 0, 1, 1],
                 [ 0, 0, 0, 0, 1, 0, 0,-1, 0, 0],
                 [ 0, 0, 0, 0, 1, 0,-1, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 1, 0, 0, 0,-1],
                 [ 0, 0, 0, 0, 0, 1, 0, 0,-1, 0]], float)
h_gt = np.array([0., 0., -1, -1, -1, -1, -1, -1, -1, -1])

f_gt = np.einsum('si,ij,sj->s', S, J_gt, S) + 2 * S @ h_gt
GS = S[np.isclose(f_gt, f_gt.max())]         # 11 tree-path ground states
nGS = len(GS)

N = 2000
Dm = 30
sigma = 0.1 # observation noise
z = rng.integers(0, nGS, size=N)             # true state index per sample
s_true = GS[z]
# W_gt = rng.normal(size=(Dm, k))
W_gt = sts.ortho_group(Dm).rvs()[:,:k]
X = s_true @ W_gt.T + sigma * rng.normal(size=(N, Dm))
xx = (X ** 2).sum(1)

pairs = list(combinations(range(k), 2))
T = np.concatenate([np.stack([S[:, i] * S[:, j] for i, j in pairs], 1), S], 1)
nstats = T.shape[1]                          # 45 pairs + 10 fields = 55
usage = S.sum(1)                             # |s| per state

#%%
# ----------------------------------------------------------------------------
# model pieces
# ----------------------------------------------------------------------------
def prior_logp(theta):
    g = T @ theta
    return g - logsumexp(g)

def loglik(W):
    """log p(x|s) up to an x-only constant; (N, 1024)."""
    C = S @ W.T
    return (X @ C.T - 0.5 * (C ** 2).sum(1)[None, :]) / sigma ** 2

def estep(W, theta, kappa, lam=0.0):
    logq = kappa * prior_logp(theta)[None, :] + (loglik(W) - lam * usage[None, :])
    logq -= logsumexp(logq, 1, keepdims=True)
    return np.exp(logq)

# def wstep(Q):
    
#     w = Q.sum(0)
#     A = S.T @ (S * w[:, None]) + 1e-6 * np.eye(k)
#     return (X.T @ (Q @ S)) @ np.linalg.inv(A)

def wstep(ES):
    
    M = np.einsum('...ij,...ik->...jk', X, ES)
    U,s,V = nla.svd(M, full_matrices=False)
    A = U@V
    scl = np.sum(s)/np.sum(ES**2)
    
    return scl*A

def theta_ascent(theta, mu_data, nsteps, gok=0.0, lr=0.5, lam=1e-4, alpha=1e-2):
    """Maximize mu_data.theta - A(theta) - gok*H(p_theta) - lam/2 |theta|^2.
    gok = gamma/kappa. grad H = -Cov_p(T) theta (exact by enumeration here;
    in a sampling setting, estimate Cov from the negative-phase chains)."""
    for _ in range(nsteps):
        g = T @ theta
        p = np.exp(g - logsumexp(g))
        mu = p @ T
        grad = mu_data - mu - lam * theta - alpha*np.sign(theta)
        if gok > 0:
            v = T @ theta
            grad += gok * (T.T @ (p * v) - mu * (mu @ theta))   # +gok*Cov(T)theta
        theta = theta + lr * grad
    return theta

def fit(kappa, lam=0.0, gamma=0.0, alpha=1e-2, W=None, theta=None, iters=300, seed=None):
    """Fixed-multiplier fit. kappa enters ONLY the E-step temper."""
    # r = np.random.default_rng(seed)
    # W = r.normal(size=(Dm, k)) * 0.3 if W is None else W
    W = sts.ortho_group(Dm).rvs()[:,:k] if W is None else W
    theta = np.zeros(nstats) if theta is None else theta
    gok = gamma / (kappa + 1e-10)
    for _ in range(iters):
        Q = estep(W, theta, kappa, lam)
        # W = wstep(Q)
        W = wstep(Q@S)
        theta = theta_ascent(theta, (Q.sum(0) / N) @ T, 60, gok=gok, alpha=alpha)
    for _ in range(3):                                   # polish moment matching
        Q = estep(W, theta, kappa, lam)
        theta = theta_ascent(theta, (Q.sum(0) / N) @ T, 1500, gok=gok, alpha=alpha)
    return W, theta

def fit_geco(Dstar, W=None, theta=None, iters=350, eta=0.02, alpha=0.9, k0=1.0,
             lam=0.0, gamma=0.0, l1_reg=1e-2):
    """Constrained fit: min KL s.t. D <= D*, dual ascent on log kappa.
    log kappa <- log kappa + eta * (D* - D_ma); D_ma = moving-avg E_q[distortion]."""
    
    theta = np.zeros(nstats) if theta is None else theta
    W = sts.ortho_group(Dm).rvs()[:,:k] if W is None else W
    
    kappa, Dma = k0, None
    ktraj, Dtraj = [], []
    for _ in range(iters):
        ll = loglik(W)
        logq = prior_logp(theta)[None, :] + (ll - lam * usage[None, :]) / kappa
        logq -= logsumexp(logq, 1, keepdims=True)
        Q = np.exp(logq)
        Dhat = (-(Q * ll).sum(1) + xx / (2 * sigma ** 2)).mean()
        Dma = Dhat if Dma is None else alpha * Dma + (1 - alpha) * Dhat
        # W = wstep(Q)
        W = wstep(Q@S)
        theta = theta_ascent(theta, (Q.sum(0) / N) @ T, 60, gok=gamma / kappa, alpha=l1_reg)
        kappa = float(np.clip(kappa * np.exp(eta * (Dstar - Dma)), 1e-2, 1e4))
        ktraj.append(kappa); Dtraj.append(Dhat)
    for _ in range(3):
        Q = estep(W, theta, kappa, lam)
        theta = theta_ascent(theta, (Q.sum(0) / N) @ T, 1200, gok=gamma / kappa, alpha=l1_reg)
    return W, theta, kappa, np.array(ktraj), np.array(Dtraj)

def fit_k0(seed, iters=80):
    """kappa=0: pure distortion, hard nearest-codeword EM over all 1024 codes."""
    r = np.random.default_rng(seed)
    W = r.normal(size=(Dm, k)) * 0.3
    for _ in range(iters):
        ll = loglik(W)
        mapc = ll.argmax(1)
        Q = np.zeros((N, 1024)); Q[np.arange(N), mapc] = 1.0
        W = wstep(Q)
    D = (-(Q * ll).sum(1) + xx / (2 * sigma ** 2)).mean()
    return W, Q, D

# ----------------------------------------------------------------------------
# metrics and structure analysis
# ----------------------------------------------------------------------------
def permham(Sm, Zm, norm=False):
    """Permutation- and flip-invariant Hamming distance."""
    S_ = 2 * Sm - 1 if np.all(np.abs(Sm ** 2 - Sm) < 1e-6) else Sm
    Z_ = 2 * Zm - 1 if np.all(np.abs(Zm ** 2 - Zm) < 1e-6) else Zm
    dH = len(Sm) - np.abs(S_.T @ Z_)
    if norm:
        dH = dH / np.maximum(np.sum(Sm > 0, axis=0), 1)[:, None]
    aye, jay = linear_sum_assignment(dH)
    return (dH[aye, jay])[np.argsort(aye)]

def nmi(a, b):
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    Cm = np.zeros((len(ua), len(ub))); np.add.at(Cm, (ia, ib), 1)
    P = Cm / Cm.sum(); px, py = P.sum(1), P.sum(0); nz = P > 0
    MI = (P[nz] * np.log(P[nz] / (px[:, None] * py[None, :])[nz])).sum()
    Hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    Hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return MI / np.sqrt(Hx * Hy) if Hx > 0 and Hy > 0 else 0.0

def metrics(W, theta, kappa, lam=0.0):
    Q = estep(W, theta, kappa, lam)
    logp = prior_logp(theta); p = np.exp(logp)
    C = S @ W.T
    sq = xx[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    D = (Q * sq).sum(1).mean() / (2 * sigma ** 2)
    lQ = np.log(np.maximum(Q, 1e-300))
    R = np.where(Q > 0, Q * (lQ - logp[None, :]), 0.0).sum(1).mean()
    qbar = Q.mean(0)
    Hqbar = -(qbar * np.log(np.maximum(qbar, 1e-300))).sum()
    EHqx = -(np.where(Q > 0, Q * lQ, 0.0)).sum(1).mean()
    Hp = -(p * logp).sum()
    mapc = Q.argmax(1)
    return dict(kappa=kappa, D=D, R=R, I=Hqbar - EHqx, Hp=Hp, Hqbar=Hqbar,
                EHqx=EHqx, ncodes=len(np.unique(mapc)), nmi=nmi(mapc, z),
                idgap=R - (Hp - EHqx), Q=Q, p=p, mapc=mapc)

def name(s):
    idx = np.flatnonzero(s).tolist()
    return '{' + ','.join(map(str, idx)) + '}' if idx else 'empty'
gs_names = [name(g) for g in GS]
GSint = GS.astype(int)
tree_edges = [(i, j) for i in range(nGS) for j in range(i + 1, nGS)
              if (GSint[i] != GSint[j]).sum() == 1]      # 10 edges: the true tree

def connected_subtree(labels):
    labels = set(labels)
    if len(labels) <= 1:
        return True
    seen, stack = {next(iter(labels))}, [next(iter(labels))]
    while stack:
        a = stack.pop()
        for (i, j) in tree_edges:
            for u, v in ((i, j), (j, i)):
                if u == a and v in labels and v not in seen:
                    seen.add(v); stack.append(v)
    return seen == labels

def analyze(tag, Q):
    """Cross-tab MAP codes vs true states; subtree check; cube-graph; permham."""
    mapc = Q.argmax(1)
    codes, inv = np.unique(mapc, return_inverse=True)
    heavy = codes[np.bincount(inv) >= 20]
    print(f"\n=== {tag}: {len(codes)} MAP codes ({len(heavy)} heavy) ===")
    all_conn = True
    for c in heavy:
        sel = mapc == c
        cnt = np.bincount(z[sel], minlength=nGS)
        support = [j for j in range(nGS) if cnt[j] >= max(5, 0.05 * sel.sum())]
        conn = connected_subtree(support); all_conn &= conn
        parts = "  ".join(f"{gs_names[j]}:{cnt[j]}" for j in np.argsort(cnt)[::-1]
                          if cnt[j] > 4)
        print(f"  {name(S[c]):>12s} n={sel.sum():4d} subtree={'Y' if conn else 'N'}"
              f" <- {parts}")
    print(f"  all heavy clusters connected subtrees of true tree: {all_conn}")
    Ch = S[heavy].astype(int)
    dH = (Ch[:, None, :] != Ch[None, :, :]).sum(-1)
    print(f"  learned codebook: support={int(Ch.sum())} (true 22), "
          f"H1-edges={int((dH == 1).sum() // 2)} (true 10)")
    ph = permham(s_true, S[mapc])
    print(f"  permham per true feature: {[int(v) for v in ph]}  total={int(ph.sum())}")

# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------
def sweep_fixed_kappa(kappas=(2., 4., 8., 16., 32., 64.)):
    best = None
    for sd in (1, 2, 3):                    # kappa=1 = exact MLE, cold restarts
        W, th = fit(1.0, iters=350, seed=sd)
        m = metrics(W, th, 1.0)
        if best is None or m['D'] + m['R'] < best[0]['D'] + best[0]['R']:
            best = (m, W, th)
    m1, W1, th1 = best
    res, par = {1.0: m1}, {1.0: (W1, th1)}
    Wc, thc = W1, th1
    for kap in kappas:                      # continuation upward
        Wc, thc = fit(kap, W=Wc.copy(), theta=thc.copy(), iters=200)
        res[kap] = metrics(Wc, thc, kap)
        par[kap] = (Wc.copy(), thc.copy())
    for kp in sorted(res):
        m = res[kp]
        print(f"kappa={kp:5.1f}  D={m['D']:6.2f}  R={m['R']:.3f}  I={m['I']:.3f}"
              f"  H(pJ)={m['Hp']:.3f}  #MAP={m['ncodes']:3d}  NMI={m['nmi']:.3f}")
    return res, par

def sweep_geco(Dstars=(16., 18., 21.5, 25., 31.), W0=None, th0=None):
    if W0 is None:                          # common warm start: kappa=1 MLE
        W0, th0 = fit(1.0, iters=350, seed=1)
    out = {}
    for Ds in Dstars:
        W, th, kap, ktraj, Dtraj = fit_geco(Ds, W0.copy(), th0.copy())
        m = metrics(W, th, kap)
        out[Ds] = dict(W=W, th=th, kappa=kap, ktraj=ktraj, Dtraj=Dtraj, m=m)
        print(f"D*={Ds:5.1f}: D={m['D']:6.2f}  R={m['R']:.3f}  kappa*={kap:7.2f}"
              f"  #MAP={m['ncodes']:2d}  H(pJ)={m['Hp']:.3f}")
    return out

def run_k0():
    best = min((fit_k0(sd) for sd in (1, 2)), key=lambda t: t[2])
    print(f"kappa=0: D={best[2]:.2f} (noise floor = {Dm / 2:.1f})")
    analyze("kappa=0 (no prior)", best[1])

def run_sparse(lam=1.0, kappa=2.0):
    best = None
    for sd in (1, 2, 3):
        W, th = fit(kappa, lam=lam, iters=300, seed=sd)
        m = metrics(W, th, kappa, lam)
        obj = m['D'] + lam * (m['Q'] @ usage).mean() + kappa * m['R']
        if best is None or obj < best[0]:
            best = (obj, m)
    m = best[1]
    print(f"lam={lam:g}, kappa={kappa:g}: D={m['D']:.2f} R={m['R']:.3f} "
          f"H(pJ)={m['Hp']:.3f}")
    analyze(f"unit cost lam={lam:g}", m['Q'])
    return m

#%%
if __name__ == "__main__":
    print(f"{nGS} ground states, log = {np.log(nGS):.3f} nats; "
          f"codebook: {[name(g) for g in GS]}")
    print("\n-- fixed-kappa frontier --")
    res, par = sweep_fixed_kappa()
    print("\n-- structure at MLE --")
    analyze("kappa=1 (MLE)", res[1.0]['Q'])
    print("\n-- GECO --")
    sweep_geco(W0=par[1.0][0].copy(), th0=par[1.0][1].copy())
    print("\n-- check (2): kappa=0 --")
    run_k0()
    print("\n-- check (1): unit usage cost --")
    run_sparse(lam=1.0)
