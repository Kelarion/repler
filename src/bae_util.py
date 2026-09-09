import os, sys, re
import pickle
import inspect
from time import time
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.distributions as dis
import torch.linalg as tla
import numpy as np
from itertools import permutations, combinations
from tqdm import tqdm

import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import numpy.linalg as nla
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

from sklearn import svm

from numba import njit
import math

# my code
import util
import df_util 

#############################################################
###### Annealing ############################################
#############################################################

@dataclass
class Neal:
    """
    Annealing scheduler
    """

    decay_rate: float = 0.8
    period: int = 2
    initial: float = 10.0

    def fit(self, model, *data, T_min=1e-4, max_iter=None, verbose=True, **opt_args):

        if max_iter is None:
            max_iter = self.period*int(np.log(T_min/self.initial)/ np.log(self.decay_rate))

        if verbose:
            pbar = tqdm(range(max_iter))

        en = []
        # mets = []
        model.initialize(*data, **opt_args)
        for it in range(max_iter):
            T = self.initial*(self.decay_rate**(it//self.period))
            model.temp = T
            en.append(model.grad_step(*data))
            # mets.append(model.metrics(*data))

            if verbose:
                pbar.update(1)

        return en # , mets

#     def cv_fit(self, model, X, T_min=1e-4, max_iter=None, verbose=False , 
#         draws=10, folds=10, **opt_args):

#         if max_iter is None:
#             max_iter = self.period*int(np.log(T_min/self.initial)/ np.log(self.decay_rate))

#         if verbose:
#             pbar = tqdm(range(draws))

#         ens = []
#         mods = []
#         for fold in range(draws):
            
#             model.initialize(X)

#             ## Mask
#             M = np.random.rand(*X.shape) < (1/folds)

#             ## Initialize masked values at random
#             X_M = X*1
#             X_M[M] = np.random.randn(M.sum())

#             ## Fit parameters and mask
#             for it in range(max_iter):
#                 T = self.initial*(self.decay_rate**(it//self.period))
#                 model.grad_step(X_M, T, **opt_args)
#                 X_M[M] = model()[M]

#             ens.append(model.loss(X, mask=M))
#             mods.append(model.S*1)

#             if verbose:
#                 pbar.update(1)

#         return ens, mods



#############################################################
###### Model comparison #####################################
#############################################################

def _anneal_temps(initial_temp=1, decay_rate=0.8, period=2, min_temp=1e-4, max_iter=None):
    """The temperature schedule used by BMF.fit, as an explicit list, so the
    BiCV fold-in E-step can reuse the exact same annealing."""
    if max_iter is None:
        max_iter = period*int(np.log(1e-4/initial_temp)/np.log(decay_rate))
    return [min_temp + initial_temp*(decay_rate**(it//period)) for it in range(max_iter)]


def _latent_first(method):
    """True if `method` takes the latents before the data matrix (the old
    bae_models signature EStep(S, X) / MStep(ES, X)), False for the new
    new_bae_models signature EStep(X, S) / MStep(X, S).  Discriminates on the
    first non-self parameter name: 'X' means data-first (new), anything else
    (S, ES, ...) means latent-first (old)."""
    params = [p for p in inspect.signature(method).parameters
              if p not in ('self', 'kwargs')]
    return not (params and params[0] == 'X')


def _estep(model, S, X):
    """model.EStep called with the right argument order for either signature."""
    if _latent_first(model.EStep):
        return model.EStep(S, X)
    return model.EStep(X, S)


def _mstep(model, ES, X):
    """model.MStep called with the right argument order for either signature."""
    if _latent_first(model.MStep):
        return model.MStep(ES, X)
    return model.MStep(X, ES)


def bicv(model, X, train_items, train_feats,
         initial_temp=1, decay_rate=0.8, period=2, min_temp=1e-4, max_iter=None,
         fold_iter=50, n_samp=10, **init_args):
    """
    Bi-cross-validation (Owen & Perry, 2009; Fu & Perry, 2017), probabilistic form.

    Gabriel block hold-out.  With train rows R = train_items and train cols
    F = train_feats (boolean masks), the data splits into four blocks:

        Xfit  = X[R , F ]   fit block: full E/M anneal -> (S_R, W_F, b_F)
        Xrow  = X[R̄, F ]   row fold-in: infer S_R̄ via the E-step, W frozen
        Xcol  = X[R , F̄]   col fold-in: infer W_F̄ via the M-step, S frozen
        Xtest = X[R̄, F̄]   held out: predicted by S_R̄ W_F̄^T + b_F̄, then scored

    The held-out block touches none of the three fits (S_R̄ never saw cols F̄,
    W_F̄ never saw rows R̄), so its log-likelihood carries a genuine complexity
    penalty -- unlike element-wise imputation CV, where each held-out entry can
    lean on the rest of its own row.  Freezing is automatic: EStep updates S
    only (the row fold-in) and MStep updates W only (the col fold-in).

    Returns (train_ll, test_ll): mean log-likelihood per entry.

    NOTE: assumes tree_reg = 0 (no structured prior).  With tree_reg > 0 the
    fold-in EStep needs an StS/N for the held-out rows and a choice about whether
    the coupling prior is frozen from Xfit or re-formed on R̄ -- left for later.
    For a spike-and-slab model the train-row latents below should be the
    effective product S*Z (the EStep return already gives this for the fold-in).
    """

    R, F = train_items, train_feats
    Xfit  = X[R][:,  F]
    Xrow  = X[~R][:, F]
    Xcol  = X[R][:, ~F]
    Xtest = X[~R][:, ~F]

    temps = _anneal_temps(initial_temp, decay_rate, period, min_temp, max_iter)

    # 1. fit the factorization on the train block
    model.initialized = False
    model.fit(Xfit, initial_temp=initial_temp, decay_rate=decay_rate, period=period,
              min_temp=min_temp, max_iter=max_iter, verbose=False, **init_args)
    S_R = 1*model.S                # train-row latents (see NOTE for spike-and-slab)
    sig = model.sigma_x
    train_ll = np.mean(model.loglikelihood(Xfit, model(S_R)))

    # 2. row fold-in: infer the held-out rows' latents from Xrow with the decoder
    #    (W_F, b_F) frozen.  EStep updates S only; its return is the effective
    #    latent (S, or S*Z for a slab), annealed on the same schedule as the fit.
    #    At min_temp (the posterior-sampling regime) a single draw is noisy, so
    #    take the posterior-predictive mean over n_samp draws at the final temp.
    model.init_latents(Xrow)
    for T in temps:
        model.temp = T
        ES_row = _estep(model, model.S, Xrow)
    model.temp = min_temp
    S_Rbar = np.zeros_like(ES_row, dtype=float)
    for _ in range(n_samp):
        S_Rbar = S_Rbar + _estep(model, model.S, Xrow)
    S_Rbar /= n_samp

    # 3. col fold-in: fit the held-out cols' weights from Xcol with the latents
    #    frozen at S_R.  MStep updates W only; with S fixed this is convex, so
    #    iterate it to convergence.  Force a random (exact-shape) W init: the SVD
    #    hot-start under-sizes W when the held-out block has fewer columns than
    #    dim_hid, and the init is irrelevant here since MStep refits W anyway.
    model.init_params(Xcol, **{**init_args, 'hot_start': False})
    for _ in range(fold_iter):
        _mstep(model, S_R, Xcol)

    # 4. predict and score the held-out block.  The decoder is now the col
    #    fold-in (W_F̄, b_F̄); fix sigma at the fit value for cross-rank scale.
    model.sigma_x = sig
    Xhat = model(S_Rbar)           # S_R̄ W_F̄^T + b_F̄
    test_ll = np.mean(model.loglikelihood(Xtest, Xhat))

    return train_ll, test_ll


def gabriel_bicv(model, X, k_row=2, k_col=2, reps=1, seed=None, **bicv_args):
    """Average BiCV over a k_row x k_col Gabriel grid (each block held out once),
    optionally repeated over random row/col partitions.  A balanced 2x2 grid is
    the usual Owen-Perry choice (hold out half the rows x half the cols).
    Returns mean (train_ll, test_ll).

    Pass the same `seed` for every rank to compare ranks under common random
    numbers: identical row/col partitions across ranks, so the partition noise
    cancels in rank-to-rank differences instead of jittering the curve.  The
    caller's global RNG state is restored on exit."""
    n, d = X.shape
    rng_state = _seed_all(seed) if seed is not None else None
    trn, tst = [], []
    try:
        for _ in range(reps):
            rfolds = np.array_split(np.random.permutation(n), k_row)
            cfolds = np.array_split(np.random.permutation(d), k_col)
            for rf in rfolds:
                for cf in cfolds:
                    R = np.ones(n, bool); R[rf] = False    # held-out rows = rf
                    F = np.ones(d, bool); F[cf] = False    # held-out cols = cf
                    a, b = bicv(model, X, R, F, **bicv_args)
                    trn.append(a)
                    tst.append(b)
    finally:
        if rng_state is not None:
            np.random.set_state(rng_state)
    return np.mean(trn), np.mean(tst)


def fpcv(model, X, train_items, train_feats, classifier=svm.LinearSVC, **opt_args):

    A = X[train_items][:,train_feats]
    B = X[~train_items][:,train_feats]
    C = X[train_items][:,~train_feats]
    D = X[~train_items][:,~train_feats]

    ## Fit model on A
    _ = model.fit(A, verbose=False, **opt_args)
    S = 1*model.S
    ls = model.loss(A)

    ## Train classifier to predict S(A) from C
    k = S.shape[1]
    d = C.shape[1]

    W = np.zeros((k,d))
    b = np.zeros(k)
    for i in range(k):
        if S[:,i].sum() > 0:
            clf = classifier()
            clf.fit(C, S[:,i])

            W[i] = 1*clf.coef_[0]
            b[i] = 1*clf.intercept_[0]

    ## Predict labels for heldout points
    Spred = 1*(D@W.T + b > 0)

    ## Evaluate
    model.S = Spred
    return ls, model.loss(B)

def loocv(model, X, n_sample=1, **opt_args):

    idx = np.arange(len(X))
    
    trn = []
    tst = []
    for i in idx:

        Xi = X[idx == i]
        Xnoti = X[idx != i]

        model.initialized = False
        en = model.fit(Xnoti, **opt_args, verbose=False)

        trn_samps = model.sample(Xnoti, n_samp=n_sample)
        tst_samps = model.sample(Xi, n_samp=n_sample)

        trn.append(model.loglikelihood(Xnoti, model(trn_samps)))
        tst.append(model.loglikelihood(Xi, model(tst_samps)))

    return trn, tst

@njit
def _seed_numba(s):
    # numba keeps its OWN global RNG, untouched by Python's np.random.seed; the
    # E-step search kernels (bae_search / new_bae_search) draw their stochastic
    # flips from it, so this must be seeded too for runs to be paired across ranks.
    np.random.seed(s)


def _seed_all(seed):
    """Seed BOTH RNGs (Python-side np.random for the mask/imputation/sampling,
    numba-side for the annealing flips) and return the saved Python RNG state so
    the caller's stream can be restored.  Numba's RNG has no Python-level
    get/set_state, so its stream is left advanced -- acceptable for CV."""
    state = np.random.get_state()
    np.random.seed(seed)
    _seed_numba(seed)
    return state


def _kfold_assignment(shape, folds, mask='random'):
    """Assign every entry of an array with the given `shape` to one of `folds`
    disjoint, (near-)balanced folds -> integer fold-id array.  `mask` may instead
    be a precomputed assignment reused across ranks: a boolean array (its True
    entries = a single hold-out fold, id 0, everything else id -1 = never held
    out) or an integer fold-id array (returned as-is).  Uses the *global* numpy
    RNG so a `seed` set by the caller makes the partition reproducible."""
    if isinstance(mask, np.ndarray):
        if mask.dtype == bool:
            return np.where(mask, 0, -1)
        return mask.astype(int)

    N = int(np.prod(shape))
    fold_ids = np.empty(N, dtype=int)
    for f, idx in enumerate(np.array_split(np.random.permutation(N), folds)):
        fold_ids[idx] = f
    return fold_ids.reshape(tuple(shape))


def impcv(model, X, mask='random', folds=10, n_sample=1, seed=None, max_folds=None, **opt_args):
    """
    Imputation-based K-fold cross validation.
    """

    n_folds = max(1, int(round(folds)))
    n_eval = n_folds if max_folds is None else min(n_folds, int(max_folds))

    # common random numbers: seed the global RNG so the partition + imputation +
    # sampling are identical across ranks, then restore the caller's stream so the
    # surrounding sweep RNG is left undisturbed.
    rng_state = _seed_all(seed) if seed is not None else None

    try:
        fold_ids = _kfold_assignment(X.shape, n_folds, mask=mask)

        train, test = [], []
        for f in range(n_eval):
            M = (fold_ids == f)
            if not M.any():
                continue
            Ximp = 1.0*X

            en = model.fit(Ximp, mask=M, **opt_args)

            # sample WITH the mask: the held-out entries are refilled from the
            # model each sweep, so the scoring latents never see them.  (With
            # n_chains > 1 the fit imputes into its own per-chain copy, so
            # `Ximp` still holds the true held-out values here.)
            pred = model(model.sample(Ximp, n_samp=n_sample, mask=M))
            loglik = model.loglikelihood(X, pred)
            test.append(np.mean(loglik[...,M]))
            train.append(np.mean(loglik[...,~M]))
            # test.append(np.mean((X[M] - pred.mean(0)[M])**2) / np.mean(X[M]**2))
            # train.append(np.mean((X[~M] - pred.mean(0)[~M])**2) / np.mean(X[~M]**2))

    finally:
        if rng_state is not None:
            np.random.set_state(rng_state)

    return np.mean(train), np.mean(test)

def oldimpcv(model, X, mask='random', folds=10, n_sample=1, seed=None, max_folds=None,
            initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=False, **init_args):
    """
    Imputation-based K-fold cross validation.

    Every entry of X is held out in exactly one of `folds` disjoint folds (so the
    held-out fraction per fold is 1/folds).  For each fold: mask that fold, refit
    the model while imputing the masked entries (EM-style: replace them with the
    current reconstruction each sweep), then score the held-out block.  Returns
    the mean (train_ll, test_ll) over the folds actually evaluated.

    Model selection across rank is only meaningful under *common random numbers*:
    pass the same `seed` (and, if you like, the same precomputed `mask`) for every
    rank so all ranks see the SAME fold partition, the SAME imputation init and the
    SAME sampling noise.  Then the mask/annealing noise is a common offset that
    cancels in rank-to-rank differences, instead of an independent jitter per rank
    that makes the test-loss-vs-rank curve chaotic.  Without a seed each call draws
    a fresh partition (true K-fold, but not paired across ranks).

    Args:
        folds:      K -- number of disjoint folds (held-out fraction = 1/K).
        max_folds:  if set, only evaluate this many of the K folds (cost control);
                    the partition is still the fixed K-fold one, so it stays
                    reproducible/paired across ranks.  None -> all K folds.
        seed:       seed the global numpy RNG for the whole routine (partition,
                    imputation init, Gibbs sampling), then restore the caller's RNG
                    state on exit.  This is the common-random-numbers knob.
        mask:       'random' (default) -> build the partition from the RNG; or a
                    precomputed assignment array (bool = one hold-out fold, or int
                    fold-ids) to reuse an identical partition across ranks.

    optimizer should already be initialized
    """

    if max_iter is None:
        max_iter = period*int(np.log(1e-4/initial_temp)/np.log(decay_rate))

    n_folds = max(1, int(round(folds)))
    n_eval = n_folds if max_folds is None else min(n_folds, int(max_folds))

    # common random numbers: seed the global RNG so the partition + imputation +
    # sampling are identical across ranks, then restore the caller's stream so the
    # surrounding sweep RNG is left undisturbed.
    rng_state = _seed_all(seed) if seed is not None else None

    try:
        fold_ids = _kfold_assignment(X.shape, n_folds, mask=mask)

        if verbose:
            pbar = tqdm(range(n_eval*max_iter))

        train, test = [], []
        for f in range(n_eval):
            M = (fold_ids == f)
            if not M.any():
                continue

            Z = 1*X
            ## Initialize masked values from the `empirical distribution`
            Z[M] = np.random.choice(Z[~M], M.sum())

            en = []
            model.initialize(Z, **init_args)
            for it in range(max_iter):
                T = min_temp + initial_temp*(decay_rate**(it//period))
                model.temp = T
                _, ls = model.grad_step(Z)   # one E and M step
                ES = model.sample(Z, n_samp=n_sample)
                Z[M] = model(ES).mean(0)[M]           # optimize wrt masked values
                en.append(1*ls)
                if verbose:
                    pbar.update()

            pred = model(model.sample(Z, n_samp=n_sample))
            loglik = model.loglikelihood(X, pred)
            test.append(np.mean(loglik[...,M]))
            train.append(np.mean(loglik[...,~M]))
            # test.append(np.mean((X[M] - pred.mean(0)[M])**2) / np.mean(X[M]**2))
            # train.append(np.mean((X[~M] - pred.mean(0)[~M])**2) / np.mean(X[~M]**2))

    finally:
        if rng_state is not None:
            np.random.set_state(rng_state)

    return np.mean(train), np.mean(test), en


def kerimpcv(model, X, mask='random', folds=10, diag=False,
            initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=False, **init_args):
    """
    Imputation-based cross validation 

    the `fold` of the CV is the fraction of the data masked

    optimizer should already be initialized
    """

    if max_iter is None:
        max_iter = period*int(np.log(min_temp/initial_temp)/np.log(decay_rate))

    if verbose:
        pbar = tqdm(range(draws))

    n,d = X.shape
    tot = spc.binom(n,2) + n*diag
    idx = util.LexOrder(diag=diag)

    ntest = int(tot/folds)

    K = util.center(X@X.T)
    Z = 1*K

    M = np.zeros((n,n)) > 0
    these_idx = np.random.choice(np.arange(tot), ntest, replace=False)
    M[idx.inv(these_idx)] = True

    ## Initialize masked values from `empirical distribution`
    Z[M] = np.random.choice(Z[~M], M.sum())
    Z[M.T] = Z[M]
    M = M + M.T # symmetrize

    model.initialize(util.center(Z), **init_args)
    en = []
    for it in range(max_iter):
        T = initial_temp*(decay_rate**(it//period))
        model.temp = T
        Z[M] = model()[M]
        en.append(model.grad_step(util.center(Z)))

    pred = model()
    test = np.mean((K[M] - pred[M])**2) #/np.mean(K[M]**2)
    train = np.mean((K[~M] - pred[~M])**2) #/np.mean(K[~M]**2)

    return train, test

def kercv(model, X, train_items, 
            initial_temp=1, decay_rate=0.8, period=2,
            min_temp=1e-4, max_iter=None, verbose=False, **init_args):
    """
    Cross validation procedure for the kernel MSE

    the `fold` of the CV is the fraction of the data masked
    """

    if max_iter is None:
        max_iter = period*int(np.log(min_temp/initial_temp)/np.log(decay_rate))

    if verbose:
        pbar = tqdm(range(draws))

    n,d = X.shape

    model.initialize(X, **init_args)
    
    Strn = model.S[train_items]
    Stst = model.S[~train_items]

    Xtrn = X[train_items]
    Xtst = X[test_items]

    # These are censored when inferring on test set
    Jtst = Stst.T@Stst
    Wtst = Stst.T@Xtst

    en = []
    for it in range(max_iter):
        T = initial_temp*(decay_rate**(it//period))
        model.temp = T

        ## Train set
        Strn = model.EStep(Xtrn, S0=Strn)

        model.StS -= Jtst # remove information from test set
        model.StX -= Wtst

        Stst = model.EStep()
        

        en.append(model.grad_step(Z))
        Z[M] = model()[M]

    pred = model()
    test = np.mean((X[M] - pred[M])**2)/np.mean(X[M]**2)
    train = np.mean((X[~M] - pred[~M])**2)/np.mean(X[~M]**2)

    return train, test

def multifit(model, X, chains=10, **neal_args):

    Xpr = []
    ls = []
    # neal = Neal(**neal_args)
    for it in tqdm(range(chains)):
        en = model.fit(X, **neal_args)
        # en = neal.fit(model, X, verbose=False)
        ls.append(en)
        Xpr.append(model())

    return np.mean(Xpr, axis=0), np.mean(ls, axis=0)

def splitfit(model, X, draws=100, **neal_args):

    N = len(X)

    neal = Neal(**neal_args)

    trn_loss = []
    tst_loss = []
    X_pred = np.zeros(X.shape)
    for draw in tqdm(range(draws)):

        idx = np.random.permutation(range(N))
        A = idx[:N//2]
        B = idx[N//2:]

        enA = neal.fit(model, X[A], verbose=False)
        ApredB = (X[B] - model.b)@model.W/model.scl
        SA = model.S*1
        WA = model.W*1
        sclA = model.scl*1
        bA = model.b*1

        enB = neal.fit(model, X[A], verbose=False)
        SB = model.S*1
        WB = model.W*1
        sclB = model.scl*1
        bB = model.b*1
        
        ## Predict heldout latents
        ApredB = (X[B] - bA)@WA/sclA
        BpredA = (X[A] - bB)@WB/sclB
        AhamB = df_util.permham(SB, 1*(ApredB > 0.5))
        BhamA = df_util.permham(SA, 1*(BpredA > 0.5))

        ## Predict heldout data
        X_pred[A] += (sclB*SA@WB.T + bB)/draws
        X_pred[B] += (sclA*SB@WA.T + bA)/draws

        trn_loss.append((enA[-1] + enB[-1])/2)
        tst_loss.append((AhamB.mean() + BhamA.mean())/2)

    return X_pred, trn_loss, tst_loss


#############################################################
###### Gradient descent #####################################
#############################################################


# class StiefelOptimizer:

#   def __init__(self, method='exact', batch_size=64, **sgd_args):

#       


#############################################################
###### Log-normalizers ######################################
#############################################################

@njit
def gaussian(nat):
    return 0.5*nat**2

@njit
def poisson(nat):
    return np.exp(nat)

@njit
def bernoulli(nat):
    return np.log(1+np.exp(nat))

@njit
def enby(nat):
    return -np.log(1-np.exp(nat))

#############################################################
###### Numba helpers ########################################
#############################################################

@njit
def log_ndtr(x: float) -> float:
    """Robust log standard normal CDF for Numba."""
    if x > -5.0:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)))
    else:
        return -0.5 * x**2 - math.log(-x) - 0.5 * math.log(2 * math.pi)

@njit
def sample_trunc_norm(mu: float, nu: float) -> float:
    """Samples from a normal distribution truncated to [0, inf).

    Hardened against the degenerate inputs that show up in spike-and-slab fits:
    when a weight column dies (gram diagonal -> 0) or a structured prior drives
    the effective precision non-positive, the caller's nu = sqrt(sigma2/wjj) and
    mu = .../wjj come through as inf/NaN.  A NaN mu fails `mu >= 0.0`, falls into
    the mu<0 branch, and its accept test `log(rand()) <= NaN` is never true -- so
    the original unbounded `while True` spins forever inside the (numba) E-step
    and forces a kernel restart.  Here a non-positive/non-finite nu or non-finite
    mu has no proper truncated normal, so we return the mode (max(mu, 0)); and
    both rejection loops are capped at `max_tries` with the same mode fallback, so
    the sampler can never stall the search.  For well-conditioned inputs the loops
    accept far inside the cap, leaving behaviour identical to before."""
    max_tries = 10000          # local constant (keep the 2-arg signature: snmf
                               # dispatches sample/mode/mean through one njit
                               # function pointer, so all three must match arity)
    # Degenerate scale/location -> no proper truncated normal; return the mode.
    if not (nu > 0.0) or not math.isfinite(mu):
        return mu if (math.isfinite(mu) and mu > 0.0) else 0.0

    if mu >= 0.0:
        for _ in range(max_tries):
            z = mu + nu * np.random.randn()
            if z >= 0.0:
                return z
        return mu                                    # mode fallback (mu >= 0)
    else:
        alpha_opt = (-mu + math.sqrt(mu**2 + 4 * nu**2)) / (2 * nu**2)
        z_max = mu + alpha_opt * nu**2
        log_rho_max = -(z_max - mu)**2 / (2 * nu**2) + alpha_opt * z_max
        for _ in range(max_tries):
            z = np.random.exponential(1.0 / alpha_opt)
            log_rho = -(z - mu)**2 / (2 * nu**2) + alpha_opt * z
            if math.log(np.random.rand()) <= log_rho - log_rho_max:
                return z
        return 0.0                                   # mode fallback (mu < 0)

@njit
def mode_trunc_norm(mu: float, nu: float) -> float:
    """Mode of a normal distribution truncated to [0, inf)."""
    if mu >= 0.0:
        return mu
    else:
        return 0.0

@njit
def mean_trunc_norm(mu: float, nu: float) -> float:
    """Mean of a normal distribution truncated to [0, inf)."""
    
    alpha = mu / nu

    Z = math.sqrt(2 * math.pi) * 0.5 * ( 1 + math.erf(alpha / math.sqrt(2.0)))
    return mu + nu * math.exp(-0.5 * alpha**2) / Z


#############################################################
###### Custom distributions #################################
#############################################################

## To Do:
class Enby(dis.ExponentialFamily):
    """
    Natural parameterisation of the negative binomial distribution, with
    fixed r (number of trials) parameters to allow exponential family form
    """

    def __init__(self, total_trials, log_prob):

        self.r = total_trials # hyperparameter
        self.eta = log_prob   # natural parameter 

    @property
    def _natural_params(self):
        """
        Abstract method for natural parameters. Returns a tuple of Tensors based
        on the distribution
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        """
        Abstract method for log normalizer function. Returns a log normalizer based on
        the distribution and input
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
        raise NotImplementedError

#################################################################
########### Sampling latents ####################################
#################################################################


