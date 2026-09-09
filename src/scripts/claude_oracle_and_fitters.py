"""
Companion to compressive_bmf.py:

1) run_oracle_test(): the exact procedure behind the "truth is a stable fixed
   point" observation -- initialize EM at the true coding and see if the
   objective retains it.

2) Three theta fitters on a common interface (codebook / weighted states -> theta):
     theta_momentmatch : global max-ent fit (what the EM M-step does)
     theta_min_l1      : global min ||theta||_1 witness LP (all 2^k constraints)
     theta_hinge_rple  : per-site single-flip margin LPs with L1 (RPLE-shaped)

3) random_graph_experiment(): draw random sparse integer Ising models with
   degenerate ground states, hand each exact GS set to the fitters, and measure
   face recovery (GS set match), its relation to H1-connectivity of the GS set,
   and support recovery vs the generating J (exploratory: at T=0 the generating
   theta is identifiable only up to the face's normal cone).
"""
import numpy as np
import importlib.util
from itertools import combinations
from scipy.special import logsumexp
from scipy.optimize import linprog, linear_sum_assignment

spec = importlib.util.spec_from_file_location("cb", "claude_compressive_bmf.py")
cb = importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
S, T, k, nstats, pairs = cb.S, cb.T, cb.k, cb.nstats, cb.pairs
X, xx, N, Dm, sigma = cb.X, cb.xx, cb.N, cb.Dm, cb.sigma
s_true, z, usage = cb.s_true, cb.z, cb.usage

#%%
# ---------------------------------------------------------------- fitters ----
def theta_momentmatch(wts_over_states, nsteps=4000, lr=0.5, l2=1e-4):
    """Global max-ent fit to state weights (the EM M-step). wts: (1024,), sums to 1."""
    mu = wts_over_states @ T
    theta = np.zeros(nstats)
    for _ in range(nsteps):
        g = T @ theta
        p = np.exp(g - logsumexp(g))
        theta += lr * (mu - p @ T - l2 * theta)
    return theta

def theta_min_l1(C):
    """Global LP: min ||theta||_1 s.t. f equal on codebook C, gap >= 1 elsewhere."""
    code_idx = [int(np.flatnonzero((S == C[i]).all(1))[0]) for i in range(len(C))]
    other = np.setdiff1d(np.arange(len(S)), code_idx)
    A_eq = np.hstack([T[code_idx[1:]] - T[code_idx[0]],
                      -(T[code_idx[1:]] - T[code_idx[0]])])
    A_ub = np.hstack([T[other] - T[code_idx[0]], -(T[other] - T[code_idx[0]])])
    res = linprog(np.ones(2 * nstats), A_ub=A_ub, b_ub=-np.ones(len(other)),
                  A_eq=A_eq, b_eq=np.zeros(len(code_idx) - 1),
                  bounds=[(0, None)] * 2 * nstats, method='highs')
    if res.status != 0:
        return None
    return res.x[:nstats] - res.x[nstats:]

def theta_hinge_rple(C, margin=1.0):
    """Per-site single-flip margin LPs (hinge-RPLE). Returns (J, h, max_row_asym).
    Site i's constraints involve only (J_i., h_i):
      f(c) - f(c^ei) = (2c_i-1) * 2 * (sum_j J_ij c_j + h_i)
      flip leaves C -> >= margin; flip stays in C -> equality (flat direction)."""
    Jr = np.zeros((k, k)); hr = np.zeros(k)
    inC = {tuple(c.astype(int)) for c in C}
    for i in range(k):
        A_ub, A_eq = [], []
        for c in C:
            coef = np.append(np.delete(c, i), 1.0)
            flip = c.copy(); flip[i] = 1 - c[i]
            if tuple(flip.astype(int)) in inC:
                A_eq.append(coef)
            else:
                A_ub.append(-(2 * c[i] - 1) * coef)
        n = k
        if not A_ub:
            # no gap constraints at this site; homogeneous equalities -> row = 0
            continue
        A_ub = np.array(A_ub)
        Aub2 = np.hstack([A_ub, -A_ub])
        if A_eq:
            A_eq = np.array(A_eq)
            Aeq2, beq = np.hstack([A_eq, -A_eq]), np.zeros(len(A_eq))
        else:
            Aeq2, beq = None, None
        res = linprog(np.ones(2 * n), A_ub=Aub2, b_ub=-margin * np.ones(len(A_ub)),
                      A_eq=Aeq2, b_eq=beq, bounds=[(0, None)] * 2 * n, method='highs')
        if res.status != 0:
            return None, None, np.inf
        x = res.x[:n] - res.x[n:]
        Jr[i, np.arange(k) != i] = x[:-1]; hr[i] = x[-1]
    return (Jr + Jr.T) / 2.0, hr, float(np.abs(Jr - Jr.T).max())

def gs_set(J, h):
    f = np.einsum('si,ij,sj->s', S, J, S) + 2 * S @ h
    return set(np.flatnonzero(f >= f.max() - 1e-9))

def theta_to_Jh(theta):
    J = np.zeros((k, k))
    for a, (i, j) in enumerate(pairs):
        J[i, j] = J[j, i] = theta[a] / 2.0
    return J, theta[len(pairs):] / 2.0

# ---------------------------------------------------- oracle (true-W) test ----
def oracle_init():
    """W by ridge regression of X on the true latents; theta by moment matching
    the exact empirical codebook frequencies."""
    W0 = (X.T @ s_true) @ np.linalg.inv(s_true.T @ s_true + 1e-6 * np.eye(k))
    wts = np.zeros(len(S))
    for i in range(len(cb.GS)):
        idx = int(np.flatnonzero((S == cb.GS[i]).all(1))[0])
        wts[idx] = (z == i).mean()
    th0 = theta_momentmatch(wts, nsteps=3000)
    return W0, th0

def run_oracle_test(kappa=2.0, lam_u=1.0, iters=150):
    """EM from the oracle init under the given config; report whether the
    objective retains the true coding, and its penalized objective value
    (compare to cold starts under the same config)."""
    W, theta = oracle_init()
    for _ in range(iters):
        Q = cb.estep(W, theta, kappa, lam_u)
        W = cb.wstep(Q)
        theta = cb.theta_ascent(theta, (Q.sum(0) / N) @ T, 60)
    for _ in range(3):
        Q = cb.estep(W, theta, kappa, lam_u)
        theta = cb.theta_ascent(theta, (Q.sum(0) / N) @ T, 1500)
    m = cb.metrics(W, theta, kappa, lam_u)
    obj = m['D'] + lam_u * (m['Q'] @ usage).mean() + kappa * m['R']
    ph = cb.permham(s_true, S[m['mapc']])
    print(f"oracle-init, kappa={kappa:g}, lam_u={lam_u:g}: obj={obj:.2f} "
          f"D={m['D']:.2f} R={m['R']:.3f} #MAP={m['ncodes']} "
          f"permham_total={int(ph.sum())}")
    return W, theta, m, obj

# ------------------------------------------------ random-graph experiment ----
def h1_connected(C):
    Ci = C.astype(int); n = len(Ci)
    dH = (Ci[:, None, :] != Ci[None, :, :]).sum(-1)
    seen, stack = {0}, [0]
    while stack:
        a = stack.pop()
        for b in range(n):
            if dH[a, b] == 1 and b not in seen:
                seen.add(b); stack.append(b)
    return len(seen) == n

def random_instance(r):
    J = np.zeros((k, k))
    for (i, j) in pairs:
        if r.random() < 0.25:
            J[i, j] = J[j, i] = r.choice([-2, -1, 1, 2])
    h = r.choice([-2, -1, 0, 1], size=k).astype(float)
    f = np.einsum('si,ij,sj->s', S, J, S) + 2 * S @ h
    gs = np.flatnonzero(f >= f.max() - 1e-9)
    return J, h, S[gs]

def support_pr(J_est, J_true, tol=1e-8):
    pe = {(i, j) for (i, j) in pairs if abs(J_est[i, j]) > tol}
    pt = {(i, j) for (i, j) in pairs if abs(J_true[i, j]) > 0}
    tp = len(pe & pt)
    P = tp / max(len(pe), 1); R = tp / max(len(pt), 1)
    return P, R

def random_graph_experiment(n_inst=30, seed=0, gs_range=(4, 40)):
    r = np.random.default_rng(seed)
    rows, tries = [], 0
    while len(rows) < n_inst and tries < 4000:
        tries += 1
        J, h, C = random_instance(r)
        if not (gs_range[0] <= len(C) <= gs_range[1]):
            continue
        conn = h1_connected(C)
        want = {int(np.flatnonzero((S == C[i]).all(1))[0]) for i in range(len(C))}
        Jh, hh, asym = theta_hinge_rple(C)
        ok_h = (Jh is not None) and (gs_set(Jh, hh) == want)
        P_h, R_h = support_pr(Jh, J) if Jh is not None else (np.nan, np.nan)
        th_g = theta_min_l1(C)
        if th_g is not None:
            Jg, hg = theta_to_Jh(th_g)
            ok_g = gs_set(Jg, hg) == want
            P_g, R_g = support_pr(Jg, J)
        else:
            ok_g, P_g, R_g = False, np.nan, np.nan
        rows.append(dict(nGS=len(C), conn=conn, feas_h=Jh is not None, ok_h=ok_h,
                         asym=asym, P_h=P_h, R_h=R_h, ok_g=ok_g, P_g=P_g, R_g=R_g))
    rows_ = rows
    def agg(sel, name):
        sub = [r_ for r_ in rows_ if sel(r_)]
        if not sub:
            print(f"  {name}: none"); return
        print(f"  {name}: n={len(sub)}  hinge feasible {np.mean([r_['feas_h'] for r_ in sub]):.2f}"
              f"  hinge exact-GS {np.mean([r_['ok_h'] for r_ in sub]):.2f}"
              f"  global exact-GS {np.mean([r_['ok_g'] for r_ in sub]):.2f}"
              f"  hinge supp P/R {np.nanmean([r_['P_h'] for r_ in sub]):.2f}/"
              f"{np.nanmean([r_['R_h'] for r_ in sub]):.2f}")
    print(f"random-graph experiment: {len(rows_)} instances "
          f"(#GS range {min(r_['nGS'] for r_ in rows_)}-{max(r_['nGS'] for r_ in rows_)}, "
          f"{sum(r_['conn'] for r_ in rows_)} H1-connected)")
    agg(lambda r_: True, "all           ")
    agg(lambda r_: r_['conn'], "H1-connected  ")
    agg(lambda r_: not r_['conn'], "H1-disconnected")
    return rows_

if __name__ == "__main__":
    print("-- oracle (true-W) test, config A --")
    run_oracle_test(2.0, 1.0)
    print("\n-- fitters on the true codebook --")
    Jh, hh, asym = theta_hinge_rple(cb.GS)
    print(f"hinge-RPLE: GS-exact={gs_set(Jh, hh) == set(np.flatnonzero(np.isclose(cb.f_gt, cb.f_gt.max())))}, "
          f"supp P/R vs J_gt = {support_pr(Jh, cb.J_gt)}")
    print("\n-- random graphs --")
    random_graph_experiment(30, seed=0)
