import os, sys, re
import pickle

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

import networkx as nx
import cvxpy as cvx

# my code
import util

###############################################
######### Generally useful ####################
###############################################


def in_conv(A, x):
    """
    Check whether x is in the convex hull of columns of A
    """

    A_eq = np.vstack([np.ones(len(A.T)), A])
    b_eq = np.concatenate([[1], x])

    return lp(c=np.random.randn(len(A.T)), A_eq=A_eq, b_eq=b_eq).success

def split(S, pi, x, x0):

    n, k = S.shape
    if sprs.issparse(S):
        i,s = S.nonzero()
    else:
        i, s = np.where(S)

    has = np.nonzero(x > 1e-6)[0]

    split = np.nonzero(np.abs(x - x**2) > 1e-6)[0]
    these = np.isin(s, split)
    guys = np.unique(s[these], return_inverse=True)[1]

    ## Add new columns (split features)
    pi_out = np.append(pi, pi[split]*(1-x[split]))
    pi_out[split] = pi_out[split]*x[split]
    s_out = np.append(s, guys+k)
    i_out = np.append(i, i[these])

    ## Add new row (new item)
    s_out = np.append(s_out, has)
    i_out = np.append(i_out, n*np.ones(len(has), int))

    ## Add one-hot column if necessary
    if x0 > 1e-6:
        pi_out = np.append(pi_out, x0)
        s_out = np.append(s_out, s_out.max()+1)
        i_out = np.append(i_out, n)

    S_out = sprs.coo_array((np.ones(len(i_out)), (i_out, s_out)),
        shape=(n+1, len(pi_out)))

    return S_out, pi_out

def allsplit(S):

    n,d = S.shape

    return np.block([[S, S, np.zeros((n,1))], [np.ones(d), np.zeros(d), 1]])

def splitpred(S, pi, x, x0):

    S_out, pi_out = split(S, pi, x, x0)

    return (S_out@sprs.diags(pi_out)@S_out.T).todense()

def krusty(X, Y):
    """
    Procrustes regression, AX ~ Y where A'A = I

    X is shape (...,d,N) and Y is shape (...,f,N)
    A is shape (...,f,d)

    requires that d <= f
    """ 

    M = np.einsum('...ij,...kj->...ik', X, Y)
    U,s,V = nla.svd(M, full_matrices=False)
    A = U@V

    return A

def permham(S, Z):
    """
    Compute the permutation-invariant hamming distance between S and Z

    Assumes that S and Z are sign matrices, i.e. +/- 1

    S is (num_item, num_S_features)
    Z is (num_item, num_Z_features)

    outputs an (num_S_features,)-sized vector of hamming distances
    """

    dH = len(S) - np.abs(S.T@Z)
    aye,jay = util.unbalanced_assignment(dH, one_sided=True)

    return dH[aye,jay]

def treecorr(S):
    return np.min([S.T@S, (1-S).T@S, S.T@(1-S), (1-S).T@(1-S)], axis=0)

class DidntWorkError(Exception):
    pass

###############################################
######### Preprocessing #######################
###############################################


def bounds(s, idx):
    """
    
    """

    lb = -np.ones(len(s))*np.inf
    ub = np.ones(len(s))*np.inf

    lb[idx[s[idx]>0]] = 0
    ub[idx[s[idx]<0]] = 0
    
    return lb, ub

def sign_kernel(c):
    """
    random expansion kernel
    similar to cho and saul, but for the sign function
    """
    return 1 - 2*np.arccos(c)/np.pi

def porthant(c):
    return 1/2 + np.arcsin(c)/np.pi

def acosker(c):
    "from Cho and Saul (2009)"
    return 1 - np.arccos(c)/np.pi

def recenter(K):
    "recenter kernel so that first point is the origin"
    d = util.dot2dist(K)
    return d[1:,[0]] + d[[0],1:] - d[1:,1:]

def merge_balls(d, thrs=1e-3):

    n = len(d)

    remaining = set(range(n))
    clqs = []
    while len(remaining) > 0:

        # d_rmn = d[list(remaining),:][:,list(remaining)]
        clq = util.find_clique(d < thrs, first=list(remaining)[0])
        remaining -= set(clq)
        # print(remaining)

        clqs.append(np.eye(n)[clq].sum(0))

    return np.array(clqs)

def fit_tri(K):
    """
    Get unique cut decomposition of triangle K
    """

    cut3 = cut_vertices(3)[1:]
    vecK = util.vec(K)
    n = len(vecK)

    alpha = (1+np.sum(vecK))/(np.sum(vecK)- n)

    pi = la.inv(cut3)@(alpha + vecK*(1-alpha))

    return pi

def get_triangles(K):
    """
    Binary embeddings of all triangles in K
    """

    N = len(K)

    cut3 = cut_vertices(3)[1:]

    tri = util.vec(np.array([K[idx,:][:,idx] for idx in combinations(range(N),3)]))
    n = tri.shape[1]

    alpha = (1+(tri.sum(1,keepdims=True)))/(tri.sum(1, keepdims=True) - n)

    pies = la.inv(cut3)@(alpha + tri*(1-alpha)).T
    items = np.array([idx for idx in combinations(range(N),3)])

    return pies.T, items


def enumerate_cuts(K=None, d=None):
    """
    Enumerte all solutions to the cut decomposition, scales extremely poorly
    """

    zero_tol = 1e-6

    if K is None and d is None:
        raise ValueError('Must supply an argument')
    elif d is None:
        d = util.dot2dist(K)

    N = len(d)
    if N > 10:
        raise ValueError('This is gonna take too long, boss')

    S_all = cut_vertices(N)[1:] # don't include 1
    ncut = len(S_all)

    A_eq = np.block([[np.ones(len(S_all)), 0], [S_all.T, util.vec(d)[:,None]]])
    b_eq = np.ones(len(S_all.T)+1)

    A_ub = np.vstack([A_eq, -A_eq, -np.eye(ncut+1)])
    b_ub = np.concatenate([b_eq, -b_eq, np.zeros(ncut+1)])

    Eses = []
    pies = []
    for vert in compute_polytope_vertices(A_ub, b_ub):
        x = vert[:-1]

        pi = x[x > zero_tol]
        S = util.mat(S_all[x > zero_tol])[:,0,:].T

        pies.append(pi)
        Eses.append(S)

    return Eses, pies

###############################################
######### Typical geometries ##################
###############################################

def cube(N):
    """
    Hypercube
    """

    X = (2*util.F2(N) - 1)
    return X@X.T/N

def cube_feats(N):
    """
    Hypercube
    """

    X = (2*util.F2(N) - 1)
    return X

def cross(N):
    """
    Cross polytope
    """
    X = np.vstack([np.eye(N), -np.flipud(np.eye(N))])
    return X@X.T

def cross_feats(N):
    """
    Cross polytope
    """
    X = np.vstack([np.eye(N), -np.flipud(np.eye(N))])
    return X

def simplex(N):
    """
    Simplex
    """
    return np.eye(N)

def btree_feats(N):
    """
    Balanced binary tree of depth N ... has 2^N leaves
    """

    F = []
    for n in range(1, N+1):
        F.append(np.repeat(np.eye(2**n), 2**(N-n), axis=1))
    return np.vstack(F)

def btree(N):
    """
    Balanced binary tree of depth N ... has 2^N leaves
    """

    F = btree_feats(N)
    return F.T@F

def circle_feats(N):
    """
    N uniformly placed points on a circle
    """
    theta = np.linspace(0, 2*(N-1)*np.pi/N, N)

    x = np.cos(theta)
    y = np.sin(theta)

    return np.stack([x,y])

def circle(N):
    """
    N uniformly placed points on a circle
    """
    X = circle_feats(N)
    return X.T@X

def grid_feats(N, M=None):
    """
    N x N grid
    """

    if M is None:
        M = N
    gr1 = np.linspace(-1,1,N)
    gr2 = np.linspace(-1,1,M)
    x,y = np.meshgrid(gr1, gr2)
    x = x.flatten()
    y = y.flatten()

    return np.stack([x,y]).T

def grid(N, M=None):

    X = grid_feats(N, M)
    return X@X.T


def randtree_feats(N, bmin=2, bmax=2):
    """
    Random tree with N leaves and out degree between [bmin, bmax]
    """

    nodes = np.arange(N)

    def branch(descendants):
        n = len(descendants) # size of subtree
        b = np.random.choice(range(bmin, bmax+1)) # number of branches
        b = np.min([b, n])
        p = np.ones(b)/b # size probabilities
        s = np.random.multinomial(n-b, p) + 1 # size of each branch
        fid = np.cumsum(s)
        bid = np.flip(np.cumsum(np.flip(s)))

        branches = []
        for i in range(b):
            br = descendants[-bid[i]:fid[i]]
            branches.append(br)
            if s[i] > 1:
                branches += branch(br)

        return branches

    branches = branch(nodes)
    X = np.stack([np.eye(N)[b].sum(0) for b in branches]).T
    return X

def randtree(N, **kwargs):

    X = randtree_feats(N, **kwargs)
    return X@X.T


def sparse_feats(num_feats, num_data, sparsity=0.1):
    """
    Generate continuous but sparse features
    """

    S = np.random.choice([0,1], 
        p=[1-sparsity, sparsity], 
        size=(num_feats, num_data))

    C = np.random.rand(num_feats, num_data)

    return S*C


#############################################
######### LP and QP helpers #################
#############################################

def solve_qp(K, S, reg=1, **solver_args):
    """
    Find a set of convex weights which bring S closest to K
    """


    zero_tol = 1e-6

    N = len(K)
    
    d = util.dot2dist(K)

    # alpha_max = np.max(la.eigvals(np.ones((N,N)), d)).real
    # d = alpha_max*d

    S_kron = util.vec(util.outers(S))

    c = np.abs(S.sum(0)) # sparsity bias

    # A = np.block([[np.ones(len(S_kron)), 0], [S_kron.T, util.vec(d)[:,None]]])

    A = np.block([[S_kron.T, util.vec(d)[:,None]]])

    P = A.T@A
    q = (reg*np.block([c, 0]) - 2*A.sum(0))
    g = np.concatenate([np.ones(len(c)), [0]])

    x = cvx.Variable(len(P))
    cost = cvx.quad_form(x, P) + q@x
    prob = cvx.Problem(cvx.Minimize(cost), constraints=[x>=0, g@x==1])

    prob.solve(**solver_args)

    return x.value[:-1]


def solve_lp(K, S=None, eps=0.0, beta=1, return_full=False):
    """
    Directly solve an LP to find the sparsest cut decomposition of K

    it's exponential in the dimension of K, so can't handle more than ~20
    """

    zero_tol = 1e-6

    N = len(K)
    if N > 10:
        raise ValueError('This is gonna take too long, boss')

    d = util.dot2dist(K)

    if S is None:
        Skron = cut_vertices(N) # don't include 1
    else:
        Skron = util.vec(util.outers(S))

    c = beta*np.min([(Skron>0).sum(1), (Skron<0).sum(1)], axis=0)
    c =  (1-eps)*c + eps*np.random.randn(len(Skron))

    A = np.block([[np.ones(len(Skron)), 0], [Skron.T, util.vec(d)[:,None]]])
    b = np.ones(len(Skron.T)+1)

    prog = lp(c=np.block([c, 0]), A_eq=A, b_eq=b)

    x = prog.x[:-1]

    if return_full:
        return S, x
    else:
        pi = x[x > zero_tol]
        S_out = util.mat(Skron[x > zero_tol])[:,0,:].T

        S_out = S_out[:,np.argsort(-pi)]
        pi = pi[np.argsort(-pi)]

        return S_out, pi


def binary_lp(K, eps=0, S=None):

    N = len(K)

    if S is None:
        S = util.F2(N).T
    i,s = np.nonzero(S)
    c = (S>1e-6).sum(0) + np.random.randn(len(S.T))*eps

    ## Vectorized construction of sparse outer product matrix
    i_kr, s_kr, v_kr = util.spouters(i, s, incl_diag=True)
    
    S_kron = sprs.coo_array((v_kr, (i_kr, s_kr)))

    vecK = util.vec(K)
    vecK = np.append(vecK, np.diag(K))

    ## Solve LP
    prog = lp(c=c, A_eq=S_kron, b_eq=vecK)

    if prog.x is None:
        raise DidntWorkError

    pi = prog.x

    keep = np.isin(s, np.nonzero(pi > 1e-6)[0])

    s = np.unique(s[keep], return_inverse=True)[1]
    i = i[keep]

    pi = pi[pi > 1e-6]
    S = sprs.coo_array((np.ones(len(s)), (i, s)))

    return S, pi


def binary_qp(K, eps=0, S=None, reg=0.1, **solver_args):

    N = len(K)

    if S is None:
        S = util.F2(N)[1:].T
    if S.sum(0).max() < N:
        S = np.hstack([S, np.ones((N,1))])
    i,s = np.nonzero(S)

    i_kr, s_kr, v_kr = util.spouters(i, s, incl_diag=True)
    S_kron = sprs.coo_array((v_kr, (i_kr, s_kr)))

    c = S.sum(0) # sparsity bias

    k = np.append(util.vec(K), np.diag(K))
    P = S_kron.T@S_kron
    q = (reg*c - 2*k@S_kron)

    x = cvx.Variable(P.shape[0])
    cost = cvx.quad_form(x, P, assume_PSD=True) + q@x
    prob = cvx.Problem(cvx.Minimize(cost), constraints=[x>=0])

    prob.solve(**solver_args)

    pi = x.value

    S = S[:, pi>1e-6]
    pi = pi[pi>1e-6]

    return S, pi


def canon2half(A, b):
    """
    Convert LP constraints from canonical to halfspace 
    
    i.e. if 
    P = {x | Ax = b, x >= 0}
    
    returns C, d s.t. 
    
    P = {x | Cx <= d}
    """
    
    n = A.shape[1]
    
    C = np.vstack([A, -A, -np.eye(n)])
    d = np.concatenate([b, -b, np.zeros(n)])

    return C, d


########################################################
################## Optimization ########################
########################################################

def quickpi(K, S):
    """
    Assumes outer products of the columns of S are linearly independent
    """
    S_kron = util.outers(S - S.mean(0,keepdims=True)).reshape((S.shape[1], -1))
    return nnls(S_kron.T, util.center(K).flatten())

def sweep(X, S, pi=None, **kwargs):
    """
    Sweep through the data, refitting one row of S at a time
    """

    for n in tqdm(range(1,len(X))):

        S, pie = reassign(X, S, n, pi=pi, **kwargs)
        # pie = ellpee(S@np.diag(pie)@S.T, S)

        r = np.min([S.sum(0), (1-S).sum(0)], axis=0)
        # dist = util.centered_distance(X@X.T, S@np.diag(pie)@S.T)
        # print('b=%d, L=%.2f, (d=%.2f, r=%.2f)'%(S.shape[1], dist+1e-2*r@pie, dist, r@pie))

        keep = (pie>1e-3)

        S = S[:,keep]
        pie = pie[keep]

        if pi is not None:
            pi = pie

    return S, pie

def reassign(X, S, n, pi=None, **kwargs):
    """
    Remove the nth item and refit the categories
    """

    N = len(X)

    idx = np.arange(N)
    idx[n] = idx[-1]
    idx[-1] = n

    S_ = S[idx]
    X_ = X[idx]

    # Kbad = util.center(util.ker(util.center(K_)))

    if pi is None:
        S_ = np.unique(S_[:-1], axis=1)
        keep = S_.sum(0) > 0
        S_ = S_[:,keep]

        Es, pie = mindistX(X_, allsplit(S_), **kwargs)

    else:
        S_, grp = np.unique(S_[:-1], axis=1, return_inverse=True)
        keep = S_.sum(0) > 0
        pie = util.group_sum(pi, grp)[keep]
        S_ = S_[:,keep]

        p, p0, sig = assign(K_, sprs.csr_array(S_), pie, **kwargs)

        Es, pie = split(S_, sig*pie, p, p0)
        Es = Es.todense()

    return Es[idx], pie

def assign(K, S, pi, incl_pi0=True, beta=1e-5, reg='sparse', eps=0, tol=1e-2):
    """
    K is shape (N,N) dense array
    S is shape (N-1, d) sparse array
    pi is shape (d,)
    """

    N = len(K)

    Kbar = util.center(K,-1)
    Kbar = Kbar/np.sqrt(np.sum(Kbar[:-1,:-1]**2))
    k = Kbar[:-1,-1]
    k0 = Kbar[-1,-1]

    if not sprs.issparse(S):
        S = sprs.coo_array(S)

    s_ = S.mean(0)
    Pi = sprs.diags(pi)

    t = (N-1)/N
    n = N-1
    sqt = np.sqrt(2*t)
    tn = (t**2 - 2*t/n)

    if incl_pi0:
        A1 = [S@Pi, np.zeros((n,1)), -S@Pi@s_[:,None]]
        A2 = [(1-s_)@Pi, [1], [0]]
        ix = -2
    else:
        A1 = [S@Pi, -S@Pi@s_[:,None]]
        A2 = [(1-s_)@Pi, [0]]
        ix = -1

    A1 = sprs.hstack(A1)
    A2 = np.concatenate(A2)
    a = A1.sum(0)

    A = np.vstack([sqt*(A1-a[None,:]/n), -t*(A2-a[None,:]/n)]) 
    b = np.concatenate([sqt*k, [-t*k0]]) 

    ## Construct problem
    x = cvx.Variable(len(s_)+1+incl_pi0) # x = [p_hat, pi_0, sigma]

    qq = np.sqrt(np.sum(util.center((S@Pi@S.T).todense())**2))
    kq = np.sum(((S@Pi@S.T)*Kbar[:-1,:-1]))
    # kk = np.sqrt(np.sum(Kbar[:-1,:-1]**2))
    kk = 1
    
    # cost = cvx.sum_squares(A@x - b) + qq*(x[-1]**2) - 2*kq*x[-1]  
    cost = cvx.sum_squares(A@x - b) + (qq*x[-1] - kq/qq)**2

    if reg == 'sparse': # simple sparsity regularization
        r = np.where(s_ < 0.5, pi, -pi)
        if incl_pi0:
            r = np.append(r, [-1, 0])
        else:
            r = np.append(r, [0])
    elif reg == 'node': # graph-based regularization
        ns = S.sum(0)
        cn = spc.binom(n+1, 2)
        r = ns - (ns**3)/cn 
        if incl_pi0:
            r = np.append(r, [0, 0])
        else:
            r = np.append(r, [0])

    r = ((1-eps)*r + eps*np.random.randn(len(r)))
    regcost = cvx.Minimize(cost + beta*r@x)
    prob = cvx.Problem(regcost, constraints=[x>=0, x[:ix]<=x[-1]])

    Cmin = prob.solve(solver='CLARABEL')
    xstar = x.value

    ## Round within tolerance (dumb rounding)
    sig = xstar[-1]
    p = xstar[:ix]/sig 

    # sort by slack
    slack = np.round(p) - p
    pidx = np.argsort(np.abs(slack))
    piidx = np.argsort(pi)

    # compute marginal loss of rounding each element
    err = A@xstar - b 
    cerr = np.cumsum(A[:,:ix][:,pidx]*slack, axis=1)
    marg = (cerr**2).sum(0)

    # round those below tolerance
    these = pidx[marg/kk < tol]
    p[these] += slack[these]

    if incl_pi0:
        return p, xstar[-2], sig
    else:
        return p, sig


def refine(X, steps=3, pimin=1e-2, S=None, **kwargs):

    S,pi = mindist(X, S=S,**kwargs)

    piout = np.zeros(len(pi))
    keep = []
    for t in range(steps):
        pi = len(X)*pi/np.sum(pi)
        keep = pi>pimin

        Es,pie = mindist(X, S=S[:,keep], **kwargs)
        piout[keep] = pie
        piout[~keep] = 0

    return S, piout


def mindistX(X, S=None, beta=1e-5, eps=0, nonzero=True, 
    tol=1e-4, reg='sparse', branches=1, verbose=False):

    N = len(X)

    if S is None:
        S = util.F2(N-1)[1:]
        S = np.hstack([np.zeros((len(S),1)), S]).T

    # if sprs.issparse(S):
    #     i,j = S.nonzero()
    # else:
    #     i,j = np.nonzero(S)
    # ikr, jkr, vkr = util.spouters(i,j)
    # Skron = sprs.coo_array((vkr, (ikr, skr)))

    ## Coefficients
    # U,s,V = la.svd(X-X.mean(0,keepdims=True), full_matrices=False)
    # l = s**2
    S_ = S - S.mean(0, keepdims=True)
    Q = (S_.T@S_)**2
    # c = l@(U.T@S_)**2
    c = ((S_.T@X)**2).sum(1)
    knrm = np.sum(util.center(X@X.T)**2)

    ## Regularization
    if reg == 'node':
        if S.min() == 0:
            S0 = np.mod(S + S[[0]], 2)
            c2 = spc.binom(S0.sum(0), 2)
        elif S.min() == -1:
            S0 = -S*S[[0]]
            c2 = spc.binom((S0>0).sum(0), 2)
        r = c2 - (c2**2)/spc.binom(N, 2)
    elif reg == 'sparse':
        r = np.min([S.sum(0), (1-S).sum(0)], axis=0)
        r[r==1] = 0
    # elif reg == 'ortho':
    #     D = np.min([])

    pies = []
    if verbose:
        pbar = tqdm(range(branches))
    else:
        pbar = range(branches)
    for br in pbar:
        if br > 1:
            rbr = (1-eps)*r + eps*np.random.randn(len(r))
        else:
            rbr = 1*r

        pi = cvx.Variable(len(S.T))

        cost = (cvx.quad_form(pi, Q, assume_PSD=True)-2*c@pi)/knrm

        prob = cvx.Problem(cvx.Minimize(cost + beta*rbr@pi), constraints=[pi>=0])
        prob.solve(solver='CLARABEL')
        pi = pi.value

        ## round down to tolerance
        idx = np.argsort(-pi, axis=0)
        pisrt = pi[idx]
        Qsrt = Q[idx,:][:,idx]
        csrt = c[idx]
        # compute marginal benefit of keeping each feature
        marg = np.diag(Qsrt)*(pisrt**2) + 2*(np.tril(Qsrt,-1)@pisrt - csrt)*pisrt
        marg = np.cumsum(marg)/knrm + 1
        # round down
        toss = np.argmax(marg <= (1+tol)*marg.min())
        pi[idx[toss+1:]] = 0

        pies.append(pi)

    pi = np.array(pies)

    if nonzero:
        these = pi.max(0) > 1e-12
        S = S[:,these]
        pi = pi[:,these]

    if branches == 1:
        pi = pi[0]

    return S, pi

def mindist(K, S=None, **kwargs):

    l,V = la.eigh(util.center(K))
    X = V[:,l>1e-6]@np.diag(np.sqrt(l[l>1e-6]))

    return mindistX(X, S=S, **kwargs)

# def mindist(K, S=None, beta=1e-3, eps=0, nonzero=True, tol=1e-4, reg='sparse'):

#     N = len(K)

#     if S is None:
#         S = util.F2(N-1)[1:]
#         S = np.hstack([np.zeros((len(S),1)), S]).T

#     # if sprs.issparse(S):
#     #     i,j = S.nonzero()
#     # else:
#     #     i,j = np.nonzero(S)
#     # ikr, jkr, vkr = util.spouters(i,j)
#     # Skron = sprs.coo_array((vkr, (ikr, skr)))

#     ## Coefficients
#     l,V = la.eigh(util.center(K))
#     s_ = S.mean(0)
#     S_ = S - s_[None,:]
#     Q = (S_.T@S_)**2
#     c = l@(V.T@S_)**2

#     ## Regularization
#     if reg == 'node':
#         if S.min() == 0:
#             S0 = np.mod(S + S[[0]], 2)
#             c2 = spc.binom(S0.sum(0), 2)
#         elif S.min() == -1:
#             S0 = -S*S[[0]]
#             c2 = spc.binom((S0>0).sum(0), 2)
#         r = c2 - (c2**2)/spc.binom(N, 2)
#     elif reg == 'sparse':
#         r = np.min([S.sum(0), (1-S).sum(0)], axis=0)
#         r[r==1] = 0

#     r = (1-eps)*r + eps*np.random.randn(len(r))

#     pi = cvx.Variable(len(S.T))

#     cost = (cvx.quad_form(pi, Q, assume_PSD=True)-2*c@pi)/(l@l)

#     prob = cvx.Problem(cvx.Minimize(cost + beta*r@pi), constraints=[pi>=0])
#     prob.solve(solver='CLARABEL')
#     pi = pi.value

#     ## round down to tolerance
#     idx = np.argsort(-pi)
#     pisrt = pi[idx]
#     Qsrt = Q[idx,:][:,idx]
#     csrt = c[idx]
#     # compute marginal benefit of keeping each feature
#     marg = np.diag(Qsrt)*(pisrt**2) + 2*(np.tril(Qsrt,-1)@pisrt - csrt)*pisrt
#     marg = np.cumsum(marg)/(l@l) + 1
#     # round down
#     toss = np.argmax(marg <= (1+tol)*marg.min())
#     pi[idx[toss+1:]] = 0

#     if nonzero:
#         these = pi > 1e-12
#     else:
#         these = pi >= -1
#     return S[:,these], pi[these]

def ellpee(K, S=None, eps=0):
    """
    Restrict S to the linear independent rank-1 matrices
    """

    if S is None:
        S = util.F2(len(K), True)[1:].T

    S_ = S - S.mean(0, keepdims=True)
    S_kron = util.outers(S_).reshape((S.shape[1], -1))

    r = np.min([S.sum(0), (1-S).sum(0)], axis=0)
    r[r==1] = 0
    r = (1-eps)*r + eps*np.random.randn(len(r))

    pistar = lp(r, A_eq=S_kron.T, b_eq=util.center(K).flatten())

    return pistar.x


def suggestions(X, m=1000, unq=False):

    p,r = X.shape
    w = X@np.random.randn(r,m)
    S = np.sign(w)
    if unq:
        S = np.unique(S*S[[0]], axis=1)

    return S

def gradstep(X, K, m=1000):

    N, d = X.shape
    w = np.random.randn(d, m)
    S = np.sign(X@w + 1e-3)
    S = S@np.diag(S[0])
    w = w@np.diag(S[0])

    K_ = util.center(K)
    pred = util.center(S@S.T/m)

    num = np.sum(K_*pred)
    dem = np.sum(pred**2)

    dS = (K_/dem - (num/dem**2)*pred)@(S - S.mean(0, keepdims=True))
    dX = dS@(w*(spc.expit(w)*(1-spc.expit(w)))).T

    # Sunq, iunq, nunq = np.unique(S, axis=1, return_counts=True, return_inverse=True)
    # deez = (Sunq.T@S)==9

    # Es, pie = mindist(K, Sunq>0)
    # pred = util.center(Es@np.diag(pie)@Es.T)
    # dS = (K_ - pred)@(Es - Es.mean(0, keepdims=True))
    # mya = np.diag(pie)@(np.abs((2*Es-1).T@Sunq) == N)@np.diag(1/nunq)

    # W = w@(np.diag(1/nunq)@deez).T
    # WW = W@mya.T
    # dX = dS@(WW*(spc.expit(WW)*(1-spc.expit(WW)))).T

    return  X + dX

########################################################
################## Cut polytope ########################
########################################################


def cut_vertices(N, centered=False):
    """
    Return vertices of cut polytope over N items
    """
    
    s = 2*util.F2(N)[1:-1] - 1
    # if centered:
    #     s -= s.mean(1, keepdims=True)
    v = np.unique(util.vec(util.outers(s.T)), axis=0)

    # v = (s[:,:,None]*s[:,None,:])[:,trind[0], trind[1]]
    
    return v

def schur(S):
    """
    Generate pairwise products of columns (i.e. final indices) of S
    """
    n = S.shape[-1]
    one = np.ones(S.shape[:-1]+(1,))
    pairs = np.array([S[...,i]*S[...,j] for i,j in combinations(range(n), 2)]).T
    return np.concatenate([one, pairs], axis=-1)

def schurcats(N, p, rmax=np.inf):

    i = 2
    S = np.random.choice([1,-1], size=(N,2), p=[p, 1-p])
    while nla.matrix_rank(schur(S)) == (1+spc.binom(i,2)):
        newS = np.random.choice([1,-1], size=(N,1), p=[p, 1-p])
        S = np.hstack([S, newS])
        i += 1
        if i >= rmax:
            break
    S = S[:,:-1]
    S = S[:,np.abs(S.sum(0))<N]

    return S

def noisyembed(S, dim, logsnr, orth=True, scl=0.1):

    N, r = S.shape

    pi_true = r * np.random.dirichlet(np.ones(r)/scl)
    W = np.random.randn(dim,r)/np.sqrt(dim)
    if orth:
        W = la.qr(W, mode='economic')[0]
    W = W@np.diag(np.sqrt(pi_true))

    noise = np.random.randn(dim, N)/np.sqrt(dim)
    a = np.sqrt(np.sum((S@W.T)**2)/np.sum(noise**2))*10**(-logsnr/20)

    return S@W.T + a*noise.T

def extract_rank_one(A, rand=False, eps=0, **solver_args):
    """
    Looking for rank-1 binary components of A, symmetric PSD
    """
    
    N = len(A)
    # psi = oriented_tricycles(N)

    # H = util.center_kernel(np.eye(N))
    # null = v[:,l <= 1e-6]

    l, v = la.eigh(util.center_kernel(A))
    nullA = np.sum(l<=1e-6)
    # kernA = v[:, :nullA]@v[:, :nullA].T

    # find directions which are orthogonal for all scales
    # gl, gv = la.eig(np.ones((N,N)), 1-util.correlify(A))
    # kernA = gv[:, np.abs(gl) > 1e-7]@gv[:, np.abs(gl) > 1e-7].T

    if rand:
        g = np.random.randn(N,1)
        # g = sts.ortho_group.rvs(N)@v[:,[-1]]
        # print(g.T@A@g)
        O = g@g.T
        # g = np.random.laplace(size=(N,1))
        # g = np.random.multivariate_normal(np.zeros(N), A)[:,None]
    else:
        g = v[:,-1] + eps*np.random.randn(N)
        O = np.outer(g,g)
        # print(O)
   
    # active = (np.abs(psi@inv_matrix_map(A) - 1) <= 1e-5) 
   
    X = cvx.Variable((N,N), symmetric='True')
    constraints = [X >> 0]
    constraints += [cvx.diag(X) == 1]
    # constraints += [cvx.trace(X@H) == 1]

    # constraints += [cvx.trace(null@null.T@X) == 0]
    # constraints += [X[:,[i]] + X[[i],:] - X <= 1 for i in range(N)]
    # constraints += [X[:,[i]] + X[[i],:] + X >= -1 for i in range(N)]
    # constraints += [cvx.trace(kernA@X) == 0]
    # if active.sum() > 0:
    #     constraints += [cvx.trace(np.triu(matrix_map(p)-np.eye(N))@X) == 1 for p in psi[active]]
    #constraints += [cvx.trace(C@X)/2 <= 1 for C in matrix_map(psi)]
    
    prob = cvx.Problem(cvx.Maximize(cvx.trace(O@X)), constraints)
    # return prob
    prob.solve(**solver_args)
    
    return X.value


########################################################
################## Graph theoretic #####################
########################################################

class CompGraph:

    def __init__(self, S):
        """
        S is shape (n_item, n_feat)

        Compositional graph of binary embedding S

        It's a DAG whose undirected distances match those of 
        the embedding, but includes some notion of feature 
        hierarchy and reachability
        """

        # self.S = np.hstack([S, np.zeros((len(S),1))])
        self.S = S

        self.N, self.k = S.shape

        self.I = np.eye(self.k)

        self.hidden = []
        self.edges = np.empty((0,2), dtype=int)
        self.e_lab = np.empty(0, dtype=int)

        self.set_origin(0)

        ## recursively add edges
        # visited = []
        # self.add_comp(0, visited) 

        # ## Recenter points
        self.data = ~np.isin(np.arange(self.N), self.hidden)
    
        # ## Remove interfering hidden nodes
        # ## (would be better to do this while generating graph?)

    def feasible(self, s):

        ante = self.issup.T@s == self.numsup
        poss = self.isopp@s == 0

        feas = ante&poss

    def find_center(self):

        d = util.dot2dist(self.S@self.S.T)
        orig = np.argmin(d[self.data].max(0))
        
        return orig

    def remove_nodes(self):

        depth = self.S.sum(1)
        max_depth = np.max(depth[self.data])
        these = np.where(self.data|(depth < max_depth))[0]
        idx = np.repeat(np.arange(len(these)), np.diff(these, append=self.N))

        self.S = self.S[these]
        self.N = len(these)
        self.data = ~np.isin(np.arange(self.N), self.hidden)

        ed = []
        el = []
        for e,l in zip(self.edges, self.e_lab):
            if np.all(np.isin(e, these)):
                ed.append((idx[e[0]], idx[e[1]]))
                el.append(l)

        self.edges = np.array(ed)
        self.e_lab = np.array(el)

    def set_origin(self, node):

        self.S = np.mod(self.S + self.S[[node]], 2)
        self.cats = [set(np.nonzero(s)[0]) for s in self.S]

        self.ovlp = self.S.T@self.S

        ## Superset relation
        self.issup = (self.ovlp == np.diag(self.ovlp)) - self.I
        self.numsup = self.issup.sum(0)

        ## Mutual exclusivity (and no self-connections)
        self.isopp = (self.ovlp == 0) + self.I

        ## flip arrows accordingly
        for i,e in enumerate(self.edges):
            if len(self.cats[e[0]]) > len(self.cats[e[1]]):
                self.edges[i] = (e[1], e[0])

    def add_node(self, s):
        self.ovlp = self.ovlp + s[None,:]*s[:,None]


    def add_comp(self, node, visited, target):
        """
        Recursion rule for adding components
        """

        ## check if any existing edges do the job
        outgoing = (self.edges[:,0] == node)&(self.S[target][self.e_lab]==1)

        ## otherwise look for new feasible edges
        if np.sum(outgoing) > 0:
            feas = self.e_lab[outgoing]
        else:
            ante = self.issup.T@self.S[node] == self.numsup
            poss = self.isopp@self.S[node] == 0
            feas = np.where(ante*poss*self.S[target])[0]

        if len(feas)>0:
        # for s in np.where(feas)[0]:
            score = (self.S*(1-self.S[node])).sum(0)

            s = feas[np.argmax(np.random.permutation(score[feas]))]

            new_node = self.cats[node].union(set([s]))
            if new_node in self.cats:
                idx = self.cats.index(new_node)
                if (node, idx) not in self.edges:
                    self.edges = np.append(self.edges, [[node, idx]], axis=0)
                    self.e_lab = np.append(self.e_lab, [s])

                if new_node not in visited:
                    visited.append(new_node)
                    self.add_comp(idx, visited, target)
            else:
                idx = self.N

                self.S = np.vstack([self.S, self.S[node] + self.I[s]])
                self.cats.append(new_node)
                self.hidden.append(idx)
                self.edges = np.append(self.edges, [[node, idx]], axis=0)
                self.e_lab = np.append(self.e_lab, [s])
                self.N += 1

                visited.append(new_node)
                self.add_comp(idx, visited, target)

## for constructing graph 

def allpaths(S, verbose=False):
    """
    Paths of the graph of S
    """

    N, k = S.shape

    cats = [set(np.nonzero(s)[0]) for s in S]
    ovlp = S.T@S

    ## Superset relation
    issup = (ovlp == np.diag(ovlp)) - np.eye(k)
    numsup = issup.sum(0)

    ## Mutual exclusivity (and no self-connections)
    isopp = (ovlp == 0) + np.eye(k)

    ## Recursion to add components
    def addcomp(node, visited):

        ante = issup.T@H[node] == numsup
        poss = isopp@H[node] == 0

        feas = ante&poss

        for s in np.where(feas)[0]:

            new_node = cats[node].union(set([s]))
            if new_node in cats:
                idx = cats.index(new_node)
                if (node, idx) not in edges:
                    edges.append((node, idx))

                if new_node not in visited:
                    visited.append(new_node)
                    addcomp(idx, visited)
            else:
                idx = len(H)

                H.append(H[node] + np.eye(k)[s])
                cats.append(new_node)
                edges.append((node, idx))

                visited.append(new_node)
                addcomp(idx, visited)

    edges = []
    H = [s for s in S]
    visited = []
    addcomp(0, visited)

    return np.array(edges).T, np.array(H)


def path(S, i, j):

    N, k = S.shape

    S = np.mod(S+S[[i]],2)
    these = S[j]

    cats = [set(np.nonzero(s)[0]) for s in S]
    ovlp = S.T@S

    ## Superset relation
    issup = (ovlp == np.diag(ovlp)) - np.eye(k)
    numsup = issup.sum(0)

    ## Mutual exclusivity (and no self-connections)
    isopp = (ovlp == 0) + np.eye(k)

    ## Recursion to add components
    def addcomp(node, visited):

        ante = issup.T@H[node] == numsup
        poss = isopp@H[node] == 0

        feas = np.where(these*ante*poss)[0]
        if len(feas) > 0:
            s = np.random.choice(feas)

            new_node = cats[node].union(set([s]))
            if new_node in cats:
                idx = cats.index(new_node)
                if (node, idx) not in edges:
                    edges.append((node, idx))

                if new_node not in visited:
                    visited.append(new_node)
                    addcomp(idx, visited)
            else:
                idx = len(H)

                H.append(H[node] + np.eye(k)[s])
                cats.append(new_node)
                edges.append((node, idx))

                visited.append(new_node)
                addcomp(idx, visited)

    edges = []
    H = [s for s in S]
    visited = []
    addcomp(i, visited)

    return np.array(edges).T, np.array(H[N:])


def ancestors(S):
    "Get all common ancestors of rows of S"
    
    aye, jay = np.triu_indices(len(S),1)
    
    ancs = np.unique(S[aye]*S[jay], axis=0)
    das = bindist(ancs, S)
    isnode = das.min(1)==0
    ancs = ancs[~isnode]
    
    return ancs[ancs.sum(1)>0]
    
def hiddens(S):
    "Get all hidden nodes, if S is (N,r), up to 2^r"
    
    nodes = S[S.sum(1)>0]
    ancs = ancestors(nodes)
        
    while len(ancs) > 0:
        nodes = np.vstack([nodes, ancs])
        ancs = ancestors(ancs)
    
    return nodes
    
def bindist(A, B):
    return A.sum(1,keepdims=True) + B.sum(1,keepdims=True).T - 2*(A@B.T)


def catdist(A, B):
    """
    A and B are (N, k) binary matrices
    """
    return np.min([A.T@B, (1-A).T@B, A.T@(1-B), (1-A).T@(1-B)], axis=0)


## 

def oriented_tricycles(N, lower=True):
    """
    Return all signed 3-cycles of N-vertex complete graph
    """
    
    ntri = int(0.5*N*(N-1))

    cyc = np.array([[ inds(i,j) for i,j in combinations(c,2)] for c in combinations(range(N),3)])
    
    if lower:
        signs = np.array([[1,1,-1],
                          [1,-1,1],
                          [-1,1,1],
                          [-1,-1,-1]])
    else:
        signs = np.array([[1,1,-1],
                          [1,-1,1],
                          [-1,1,1]])
        
    return (np.eye(ntri)[:,cyc]@signs.T).reshape((ntri,-1)).T


def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    prog = lp(c, A_eq=A, b_eq=b)
    return prog.success

def find_edges(V):
    """
    Brute force search for edges between vertices V
    """
    
    P = len(V)
    edges = []
    
    for i in range(P):
        for j in range(i+1, P):
            
            rest = ~np.isin(np.arange(P), [i,j])
            mid = 0.5*(V[i] + V[j])
            
            if not in_hull(V[rest], mid):
                edges.append((i,j))
    
    return edges


########################################################
######### Things related to item order #################
########################################################


def item_order(d, sign=1):

    ## Most "explainable" item <=> closest item

    first = np.argmin(sign*np.sum(d, axis=0))

    included = [first]
    remaining = np.setdiff1d(range(len(d)), first).tolist()
    for _ in range(len(d)-1):
        new = remaining[np.argmin(sign*np.sum(d[included,:][:,remaining], axis=0))]
        included.append(new)
        remaining.remove(new)

    return np.array(included)

def ball_order(d, sign=1):

    first = np.argmin(sign*np.sum(d, axis=0))
    return np.argsort(sign*d[first])

def squares_first(d):
    """
    Beware, this is quartic in dimension of d
    """

    squares = []

    for idx in combinations(range(len(d)),4):
        S, pi = enumerate_cuts(d=d[idx,:][:,idx])

        if len(S) == 1:
            squares.append(np.eye(len(d))[idx,:].sum(0))

    if len(squares) > 0:
        in_sq = np.max(squares, axis=0)

        items = np.where(squares[0])[0].tolist()

        included = np.array(squares[0])
        remaining = np.array(squares[1:])

        while np.sum(included) < np.sum(in_sq):

            ovlp = included@remaining.T

            new_sq = np.argmax(ovlp==3)
            new_item = np.argmax(remaining[new_sq] - included)

            items.append(new_item)

            included = 1*(included + remaining[new_sq] > 1e-6)
            remaining = remaining[ovlp<4]
    else:
        items = []

    leftovers = np.setdiff1d(range(len(d)), items).tolist()

    return items + leftovers