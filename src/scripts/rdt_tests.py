CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/concepts/sbmf/')

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd

import networkx as nx
import cvxpy as cvx

# my code
import util
import df_util
import pt_util
import old_bae
import old_bae_models
import bae_util
import plotting as tpl
import experiments as exp
import bae_priors as nbp
import bae_models
import bae_search as nbs

import bae_experiments as nbx

import numpy as np
from scipy.special import logsumexp, xlogy

#%%

def enumerate_codes(k, coding="01"):
    """All 2^k latent states as a (2^k, k) float array.

    coding : "01"  -> entries in {0, 1}
             "pm1" -> entries in {-1, +1}

    Must match the convention used by your sampler. The {0,1} <-> {-1,+1}
    reparameterisation changes both J and h; a mismatch here produces
    plausible-looking but wrong numbers.
    """
    idx = np.arange(2 ** k, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(k)[None, :]) & 1).astype(np.float64)
    if coding == "01":
        return bits
    elif coding == "pm1":
        return 2.0 * bits - 1.0
    raise ValueError("coding must be '01' or 'pm1'")


def exact_rd(
    X,
    W,
    J,
    h,
    beta=1.0,
    sigma_x=None,
    lam=None,
    coding="01",
    zero_diagonal=True,
    chunk=4096,
    check_identity=True,
):
    """Exact rate, distortion and entropy terms by enumeration.

    Parameters
    ----------
    X : (M, D) data.
    W : (D, k) loadings, x ~ W s.
    J : (k, k) couplings. Symmetrised internally.
    h : (k,) fields, entering the energy as 2 h's.
    beta : global inverse temperature multiplying the prior energy.
    sigma_x : observation noise sd. Either this or `lam` must be given.
    lam : 1 / (2 sigma_x^2). Overrides sigma_x if both given.
    coding : "01" or "pm1"; must match the sampler.
    zero_diagonal : with {0,1} coding s_i^2 = s_i, so a nonzero diagonal of J
        is silently a field contribution and double-counts against h.
        Leave True unless you have deliberately folded it in already.
    chunk : rows of X processed at once; controls the peak (chunk, 2^k) array.
    check_identity : assert  E_x KL(q_i||p_J) == I(x;s) + KL(qbar||p_J).

    Returns
    -------
    dict with (all in nats unless noted):
        distortion            E_q ||x - Ws||^2, averaged over items
        distortion_per_dim    the above / D  -- compare against kappa
        distortion_plugin     ||x - W m_i||^2 averaged (the biased-low version)
        distortion_variance   tr(W'W Sigma_i) averaged (the part plug-in drops)
        mutual_information    I(x;s) = H[qbar] - mean_i H[q_i]   <-- the true rate
        mutual_information_bits
        elbo_rate             E_x KL(q_i || p_J)  <-- the rate in your objective
        kl_qbar_prior         KL(qbar || p_J)     <-- the bound gap
        H_qbar                H[qbar]
        H_post_mean           mean_i H[q_i]
        H_prior               H[p_J]
        log_Z                 log partition function of the prior
        n_effective_codes     exp(H[qbar]), the perplexity of the codebook
        magnetisations        (k,) aggregate posterior means
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64).ravel()

    M, D = X.shape
    k = W.shape[1]
    if W.shape[0] != D:
        raise ValueError(f"W is {W.shape}, expected ({D}, k)")
    if J.shape != (k, k):
        raise ValueError(f"J is {J.shape}, expected ({k}, {k})")
    if h.shape != (k,):
        raise ValueError(f"h is {h.shape}, expected ({k},)")

    if lam is None:
        if sigma_x is None:
            raise ValueError("give sigma_x or lam")
        lam = 1.0 / (2.0 * sigma_x ** 2)

    J = 0.5 * (J + J.T)
    if zero_diagonal:
        J = J - np.diag(np.diag(J))

    # ---- prior over all 2^k states -------------------------------------
    S = enumerate_codes(k, coding)                      # (C, k)
    C = S.shape[0]
    energy = beta * (np.einsum("ci,ij,cj->c", S, J, S)
                     + 2.0 * S @ h)                     # (C,)
    log_Z = logsumexp(energy)
    log_prior = energy - log_Z                          # (C,)
    prior = np.exp(log_prior)
    H_prior = -xlogy(prior, prior).sum()

    # ---- reconstruction of every code ----------------------------------
    R = S @ W.T                                         # (C, D)
    r_sq = np.einsum("cd,cd->c", R, R)                  # (C,)
    WtW = W.T @ W                                       # (k, k)

    # ---- accumulate over data in chunks --------------------------------
    qbar = np.zeros(C)
    H_post_sum = 0.0
    dist_sum = 0.0
    dist_plugin_sum = 0.0
    cross_entropy_sum = 0.0     # -sum_s q_i(s) log p_J(s)

    for start in range(0, M, chunk):
        Xc = X[start:start + chunk]
        n = Xc.shape[0]

        # squared distances, (n, C)
        d2 = (np.einsum("nd,nd->n", Xc, Xc)[:, None]
              - 2.0 * (Xc @ R.T)
              + r_sq[None, :])
        np.maximum(d2, 0.0, out=d2)

        log_q = log_prior[None, :] - lam * d2
        log_q -= logsumexp(log_q, axis=1, keepdims=True)
        q = np.exp(log_q)                               # (n, C)

        qbar += q.sum(axis=0)
        H_post_sum += -xlogy(q, q).sum()                # -sum q log q
        dist_sum += np.einsum("nc,nc->", q, d2)
        cross_entropy_sum += -(q @ log_prior).sum()

        # plug-in distortion from the posterior mean code
        m = q @ S                                       # (n, k)
        resid = Xc - m @ W.T
        dist_plugin_sum += np.einsum("nd,nd->", resid, resid)

    qbar /= M
    H_qbar = -xlogy(qbar, qbar).sum()
    H_post_mean = H_post_sum / M
    distortion = dist_sum / M
    distortion_plugin = dist_plugin_sum / M
    distortion_variance = distortion - distortion_plugin

    mutual_information = H_qbar - H_post_mean
    elbo_rate = cross_entropy_sum / M - H_post_mean
    kl_qbar_prior = -(qbar @ log_prior) - H_qbar

    if check_identity:
        gap = abs(elbo_rate - (mutual_information + kl_qbar_prior))
        assert gap < 1e-8 * max(1.0, abs(elbo_rate)), f"identity broken: {gap}"

    return {
        "distortion": distortion,
        "distortion_per_dim": distortion / D,
        "distortion_plugin": distortion_plugin,
        "distortion_variance": distortion_variance,
        "mutual_information": mutual_information,
        "mutual_information_bits": mutual_information / np.log(2.0),
        "elbo_rate": elbo_rate,
        "kl_qbar_prior": kl_qbar_prior,
        "H_qbar": H_qbar,
        "H_post_mean": H_post_mean,
        "H_prior": H_prior,
        "log_Z": log_Z,
        "n_effective_codes": float(np.exp(H_qbar)),
        "magnetisations": qbar @ S,
        "lam": lam,
        "sigma_x": float(np.sqrt(1.0 / (2.0 * lam))),
    }


def rd_sweep(X, W, J, h, lams, **kw):
    """Trace an RD curve by sweeping lambda. Returns (rates, distortions, rows)."""
    rows = [exact_rd(X, W, J, h, lam=l, **kw) for l in lams]
    return (np.array([r["mutual_information"] for r in rows]),
            np.array([r["distortion_per_dim"] for r in rows]),
            rows)

#%%

N = 1000
K = 5
# K = 5
# K = 10
temp = 1e-2
# temp = 1

task = nbx.StructuredCats(1, 15, 10,
                        # spec=f"grid{K}(d=2)",
                        spec=f"tree{K}",
                        # spec=f"cat{K}x3",
                        # spec='cat3x2+tree3',
                        N=N,
                        temp=temp,
                        )

data = task.sample()


# J = data['Jtrue'][0] - np.diag(np.diag(data['Jtrue'][0]))
# h = np.diag(data['Jtrue'][0])


# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     k, D, M = 8, 20, 500

#     W_true = rng.normal(size=(D, k)) / np.sqrt(D)
#     J_true = np.zeros((k, k))
#     for a in range(0, k - 1, 2):          # paired couplings
#         J_true[a, a + 1] = J_true[a + 1, a] = 1.0
#     h_true = np.zeros(k)

#     S_all = enumerate_codes(k)
#     e = np.einsum("ci,ij,cj->c", S_all, J_true, S_all) + 2.0 * S_all @ h_true
#     p = np.exp(e - logsumexp(e))
#     S_draw = S_all[rng.choice(len(S_all), size=M, p=p)]
#     X = S_draw @ W_true.T + 0.1 * rng.normal(size=(M, D))

    # print(f"{'sigma_x':>9} {'I (bits)':>10} {'D/dim':>10} "
    #       f"{'KL(qb|p)':>10} {'n_codes':>9}")
    # for sx in [0.5, 0.3, 0.2, 0.1, 0.05]:
    #     r = exact_rd(X, W_true, J_true, h_true, beta=1.0, sigma_x=sx)
    #     print(f"{sx:9.3f} {r['mutual_information_bits']:10.3f} "
    #           f"{r['distortion_per_dim']:10.5f} {r['kl_qbar_prior']:10.4f} "
    #           f"{r['n_effective_codes']:9.2f}")

#%%

for sx in [0.5, 0.3, 0.2, 0.1, 0.05]:
    r = exact_rd(data['X'][0], data['Wtrue'][0], J, h, beta=1.0, sigma_x=sx)
    print(f"{sx:9.3f} {r['mutual_information_bits']:10.3f} "
          f"{r['distortion_per_dim']:10.5f} {r['kl_qbar_prior']:10.4f} "
          f"{r['n_effective_codes']:9.2f}")
   
#%%
X_ = data['X'][0]


mod = bae_models.BiPCA(dim_hid=data['Strue'][0].shape[-1],
                           # sparse_reg=1e-1,
                           sparse_reg=1,
                           tree_reg=1e-1,
                           # tree_reg=0,
                           J_lr=1e-2,
                           # J_loss='mle',
                           J_l1_reg=1.0,
                           J_prior='mrf',
                           n_chains=2,
                           # kappa=kap,
                           )

en = mod.fit(X_ / X_.std(), 
             decay_rate=0.88,
             min_temp=1,
             initial_temp=10,
             period=50,
             scl_lr=1e-3,
             lr=1,
             # prior_temp=1e-1,
             # prior_min_temp=1e-1,
             # max_iter=1000,
             prior_schedule=nbp.ConstantTemp(5e-2),
             # prior_schedule=bae_priors.GeomAnneal(10, 0.88, period=50, min_temp=1e-1),
             # prior_schedule=nbp.AdaptiveTemp(1e-1, kappa=0.15, gamma=0.1),
             probes=['latent_prior.beta', 'latent_prior.temp', 'sigma_x'],
             )

# exact_rd(X_/X_.std()-mod.operator.b, 
#              mod.operator.W*mod.operator.scl, J, h, 
#              sigma_x=np.sqrt(mod.sigma_x))

#%%

# kap = 5e-2

X_ = data['X'][0]

plt.figure()
RD = []
permham = []
J_mag = []
for kap in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    foo = []
    ba = []
    wa = []
    for _ in range(5):
        mod = bae_models.BiPCA(dim_hid=data['Strue'][0].shape[-1],
                                   # sparse_reg=1e-1,
                                   sparse_reg=1,
                                   # tree_reg=10,
                                   tree_reg=0,
                                   J_lr=1e-2,
                                   # J_loss='mle',
                                   # J_l1_reg=0.2,
                                   # J_prior='mrf',
                                   n_chains=2,
                                   # kappa=kap,
                                   )
        
        en = mod.fit(X_ / X_.std(), 
                     decay_rate=0.88,
                     min_temp=1,
                     initial_temp=10,
                     period=50,
                     scl_lr=1e-2,
                     lr=1e-1,
                     # prior_temp=1,
                     # prior_min_temp=1e-2,
                     # max_iter=1000,
                     )
        
        mod.collapse()
        
        J,h = mod.latent_prior.coupling()
        
        foo.append(exact_rd(X_/X_.std()-mod.operator.b, 
                     mod.operator.W*mod.operator.scl, J, h, 
                     sigma_x=np.sqrt(mod.sigma_x)))
        
        # R.append(r['mutual_information'])
        # D.append(r['distortion_per_dim'])
    
        # foo.append(np.sqrt(mod.sigma_x))
        # ba.append(r['n_effective_codes'])
        wa.append(np.max(np.abs(J)))
        ba.append(df_util.permham(mod.S, data['Strue'][0]).mean())
    
    
    RD.append(pd.DataFrame(foo).drop('magnetisations', axis=1).mean())
    # plt.scatter(R, D)
    # sigs.append(foo)
    # unqs.append(ba)
    J_mag.append(wa)
    permham.append(ba)

RD = pd.DataFrame(RD)    

# r = exact_rd(X_/X_.std(), mod.operator.W, J, h, lam=mod.operator.scl)

#%%

k = data['Strue'][0].shape[1]
X = data['X'][0] / data['X'][0].std()

W = sts.ortho_group(data['X'][0].shape[1]).rvs()[:,:k]
b = X.mean(0)

S = 1*(data['X'][0]@W > 0.5)

prior = nbp.MRFPrior(n_chains=1,
                     l0_reg=0.1,
                     beta_init=1,
                     beta_lr=1e-1,
                     beta_mle=False,
                     # beta_mle=True,
                     h_l2_reg=1e-2,
                     )
 
# prior = nbp.BoltzmannPriorNP(J_lr=1e-1, J_loss='mle')

prior.init_params(S)

lamb = 1
scl = 1

lr = 1e-2
kap = 1e-2

RD = []

search = nbs._build_dense_search(nbs.score_binary, nbs.aux_binary, nbs.prior_boltzmann, diag_gram=True, parallel=False)

#%%

# for t in [1, 0.95, 0.9, 0.85, 0.8]:
for t in [2, 1.75, 1.5, 1.25, 1]:

    for _ in range(50):
    
        J,h = prior.coupling()
        
        # foo.append(S.mean(0))
        # wa.append(np.abs(J).sum())
        # ba.append(h[0])
        # wa.append(prior.beta[0])
        
        mse = np.mean((S@W.T - X - b)**2)
        
        # D.append(1*mse)
        # R.append()
        
        # lamb *= np.exp(lr*(mse/np.mean(X**2) - kap))
        # scl *= np.exp(lr*(mse - scl))
        
        # S, _ = search(X@W, S, 1*S, W.T@W, S.T@S, len(S), t, 0, 0, 0, 1/np.sqrt(2*lamb), J[0], h[0])
        # S, _ = search(X@W, S, 1*S, W.T@W, S.T@S, len(S), t, 0, 0, 0, 1/np.sqrt(2*lamb), J, h)
        S, _ = search(X@W, S, 1*S, W.T@W, S.T@S, len(S), t, 1, 0, 0, 1000, J, h)
        
        W = df_util.krusty(S.T, X.T).T
        
        prior.learn(S)
        
        RD.append(exact_rd(X-b, W, J, h, sigma_x=1000))
        
RD = pd.DataFrame(RD)

#%%

plt.plot(np.stack(RD.magnetisations))

