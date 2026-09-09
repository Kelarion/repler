"""Checks for bae_models.BiPCA (Procrustes operator on the SHARED binary link).

BiPCA no longer uses the bespoke old_bae_search.bpca kernel.  For the linear-Gaussian
model X = scl S W^T + b + N(0, sigma^2) with orthonormal W the S_ij flip log-odds
is  (scl/sigma^2) XW_ij - 0.5 (scl^2/sigma^2),  which the scaffold's binary link
(E - 0.5 wjj)/sigma2 reproduces once XW, WtW are pre-scaled by scl, scl^2 (done in
Procrustes.drive/gram).  We check:

  1. the algebra: scaled binary link == the corrected log-odds, and it reduces to
     the OLD bpca form (2 XW/scl - 1) exactly when sigma^2 = scl^2/2 (the implicit
     convention the old kernel baked in, along with a dropped Gaussian 1/2);
  2. empirically: the shared binary kernel (pre-scaled, diag_gram=True, sigma^2 =
     scl^2/2) reproduces old_bae_search.bpca's E-step sweep;
  3. end-to-end: the model fits -- loss decreases and W stays orthonormal -- both
     with fixed sigma^2 = 1 and with sigma^2 estimated from the residual (scl_lr>0).
"""

import numpy as np
from numba import njit

import old_bae_search
import bae_search as nbs
import bae_models


@njit
def numba_seed(s):
    np.random.seed(s)


def make_data(n=60, d=40, K=5, seed=0):
    rng = np.random.default_rng(seed)
    Strue = (rng.random((n, K)) < 0.4).astype(float)
    Wtrue = rng.standard_normal((d, K))
    return Strue @ Wtrue.T + 0.3 * rng.standard_normal((n, d))


def test_algebra():
    rng = np.random.RandomState(0)
    XW = rng.randn(200)
    scl = 1.7
    sig2 = 0.6
    corrected = (scl / sig2) * XW - 0.5 * (scl ** 2 / sig2)
    scaffold = (scl * XW - 0.5 * scl ** 2) / sig2          # (E - 0.5 wjj)/sigma2, wjj=scl^2
    ok1 = np.allclose(corrected, scaffold, atol=0, rtol=1e-12)

    # under sigma^2 = scl^2/2 the corrected log-odds IS the old bpca field
    sig2_bpca = scl ** 2 / 2
    corrected_bpca = (scl / sig2_bpca) * XW - 0.5 * (scl ** 2 / sig2_bpca)
    old_bpca = 2 * XW / scl - 1
    ok2 = np.allclose(corrected_bpca, old_bpca, atol=0, rtol=1e-12)

    print(f"[{'PASS' if ok1 and ok2 else 'FAIL'}] algebra")
    print(f"    scaled binary link == corrected log-odds : {ok1}")
    print(f"    reduces to old bpca at sigma^2=scl^2/2   : {ok2}")
    return ok1 and ok2


def test_kernel_matches_bpca(seed=0):
    """One E-step sweep: shared binary kernel (pre-scaled, diag_gram) vs bpca."""
    rng = np.random.RandomState(1)
    n, d, m = 80, 30, 6
    Q, _ = np.linalg.qr(rng.randn(d, m))                   # orthonormal W
    b = rng.randn(d)
    X = rng.randn(n, d)
    XWraw = (X - b) @ Q                                    # unscaled drive
    scl = 1.3
    sig2 = scl ** 2 / 2                                    # convention that matches bpca
    S0 = 1.0 * (rng.rand(n, m) > 0.5)
    temp, alpha, beta = 0.7, 0.4, 1e-2
    zeroJ, zeroh = np.zeros((m, m)), np.zeros(m)

    kernel = nbs.make_dense_search(*nbs.BINARY_LINK, nbs.PRIOR_PLAIN, diag_gram=True)
    WtW = (scl ** 2) * (Q.T @ Q)                           # scl^2 * I

    S_k, Z_k, StS_k = S0.copy(), S0.copy(), (S0.T @ S0).copy()
    numba_seed(seed)
    kernel(scl * XWraw.copy(), S_k, Z_k, WtW.copy(), StS_k, n,
           temp, alpha, beta, 1.0, sig2, zeroJ, zeroh, True)

    S_b, StS_b = S0.copy(), (S0.T @ S0).copy()
    numba_seed(seed)
    old_bae_search.bpca(XWraw.copy(), S_b, scl, temp=temp, StS=StS_b, N=n,
                    alpha=alpha, beta=beta)

    frac = float(np.mean(S_k == S_b))
    ok = np.array_equal(S_k, S_b)
    print(f"[{'PASS' if ok else 'FAIL'}] kernel matches bpca (sigma^2=scl^2/2)")
    print(f"    S identical vs bpca : {ok}   (agreement {frac:.5f})")
    return ok


def test_end_to_end():
    X = make_data()
    allok = True
    for name, scl_lr in [("fixed sigma^2=1", 0.0), ("estimated sigma^2", 0.2)]:
        m = bae_models.BiPCA(5, sparse_reg=1e-2, tree_reg=0.05)
        m.initialize(X, scl_lr=scl_lr)
        numba_seed(0)
        en = m.fit(X, initial_temp=1, decay_rate=0.8, period=2, min_temp=1e-4,
                   max_iter=200, verbose=False)
        en = np.asarray(en, dtype=float)
        WtW = m.operator.W.T @ m.operator.W
        ortho = np.abs(WtW - np.eye(5)).max()
        decreased = en[-1] < en[0]
        ok = decreased and ortho < 1e-6 and np.isfinite(en).all()
        allok = allok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] end-to-end ({name})")
        print(f"    energy {en[0]:.4f} -> {en[-1]:.4f} (decreased {decreased}); "
              f"max|WtW-I|={ortho:.1e}; sigma_x={m.sigma_x:.4f}")
    return allok


if __name__ == "__main__":
    results = [test_algebra(), test_kernel_matches_bpca(), test_end_to_end()]
    print("\nALL PASS" if all(results) else "\nSOME FAILED")
