
CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
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
from sklearn.manifold import MDS

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

from pypoman import compute_polytope_vertices, compute_polytope_halfspaces
import cvxpy as cvx
import polytope as pc
from hsnf import column_style_hermite_normal_form

# my code
import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as dicplt
import grammars as gram
import dichotomies as dics

#%%

class Quadric:
    
    def __init__(self, Q, p, r):
        """
        Quadric surface, all points x satisfying
        x'Qx + p'x + r = 0
        """
        
        self.Q = Q
        self.p = p
        self.r = r
    
    def __call__(self, x):
        """
        Evaluate the value of surface equation
        """
        return self.qform(x) + self.p@x + self.r
    
    def qform(self, x, y=None):
        """
        Quadratic component of the surface equation
        """
        
        if y is None:
            y = x
        return ((self.Q@x)*y).sum(0)

    def project(self, x, s):
        """
        project x onto the quadric, from perspective point s
        
        i.e., find where the line containing x and s intersects with the surface
        (in general there can be 2 intersection points, this returns the closest)
        """
        
        a = self.qform(s - x)
        b = self.p@(s-x) + 2*self.qform(x, s-x)
        c = self.qform(x) + self.p@x + self.r
        
        disc = b**2 - 4*a*c # impossible that this is negative
        disc = disc*(disc>0)
        
        p1 = x + ((-b + np.sqrt(disc))/(2*a))*(s - x)
        p2 = x + ((-b - np.sqrt(disc))/(2*a))*(s - x)
        
        d1 = np.sum((x - p1)**2, 0)
        d2 = np.sum((x - p2)**2, 0)
        
        return np.where(d1<=d2, p1, p2)


def dirichlet_slice(N, k, size=1):
    """
    Special slice of a dirichlet distribution, where k of the entries
    are constrained to have equal weight.
    """
    
    d1 = np.random.dirichlet(np.ones(N - k + 1), size=size)
    slice_vals = np.repeat(d1[...,[0]], k, axis=-1)/k

    return np.append(slice_vals, d1[...,1:], axis=-1)


def sample_from_surface(y, c, size=1):
    """
    Sample a random point x in the simplex, satisfying
    
    <x, y>/|x||y| = c
    
    assumes y is simplex-valued
    """
    
    N = len(y)
    k = int(np.log2(N+1))
    
    y = y > 0  # reference point
    idx = np.argsort(y)
    
    y_ = np.zeros((N, size)) # waluigi
    y_[y <= 0] = dirichlet_slice(N-np.sum(y), k, size=size).T

    V = la.eigh(util.center_kernel(np.eye(N)))[1][:,1:]    
    A = (c**2)*np.eye(N) - np.outer(y,y)/np.sum(y)
    
    S = Quadric(V.T@A@V, 2*A.mean(0)@V, A.mean())

    x = (dirichlet_slice(N, k, size=size)@V).T
    
    orig = np.where(S(x)<=0, V.T@y_, V.T@y[:,None]/np.sum(y))
    
    return V@S.project(x, orig) + 1/N
    

# def recover_hadamard(H):
    
#     N = len(H)
    
#     epsilon_r = np.random.choice([-1,1], size=N)
    
#     K = np.diag(epsilon_r)@H
    
#     M_plus_dt = (K.T - N*la.inv(K))
    
#     sgn_thresh = np.sign(M_plus_dt)
#     sgn_thresh[np.abs(M_plus_dt) < 0.5] = 0
#     ret = K - sgn_thresh.T
#     ret = np.diag(epsilon_r)@ret
    
#     return ret
    
    
def orthogonal_rank_one(A):
    """
    Random rank-1 cut matrix orthogonal to A
    """
    
    N = len(A)
    # psi = oriented_tricycles(N)
    
    # l, v = la.eigh(A@A.T)
    # null = v[:, l <=1e-6]
    
    # g = np.random.multivariate_normal(np.zeros(N), null@null.T)[:,None]
   
    g = np.random.randn(N,1)
    
    # active = (np.abs(psi@inv_matrix_map(A) - 1) <= 1e-5) 
   
    X = cvx.Variable((N,N), symmetric='True')
    constraints = [X >> 0]
    constraints += [cvx.diag(X) == 1]
    constraints += [X@A == 0]
    # constraints += [X[:,[i]] + X[[i],:] - X <= 1 for i in range(N)]
    # constraints += [X[:,[i]] + X[[i],:] + X >= -1 for i in range(N)]
    
    prob = cvx.Problem(cvx.Maximize(cvx.trace(g@g.T@X)), constraints)
    prob.solve()
    
    return X.value


# def randomard(N):
#     """
#     Random hadamard matrix of order N, where N is a multiple of 4
#     """
    
#     H = np.ones((N,1))
    
#     for _ in tqdm(range(N-1)):
        
#         new_col = np.round(orthogonal_rank_one(H), 6)
#         if np.abs(new_col**2 - 1).max() > 1e-4:
#             it = 0
#             while np.abs(new_col**2 - 1).max() > 1e-4:
#                 new_col = np.round(orthogonal_rank_one(H), 6)
#                 if it > 10:
#                     raise Exception
#                 it += 1
            
#         H = np.hstack([H, np.sign(new_col[:,[0]])])
    
#     return H


# def randomard(N):
#     """
#     Random Hadamard matrix of order N
#     """
    
#     H = np.ones((1,N))
    
#     for n in tqdm(range(1,N)):
        
#         # g = np.random.randn(N)
#         prog = lp(-H.mean(0), A_eq=H, b_eq=H.sum(1)//2, bounds=[0,1], integrality=1)
        
#         H = np.vstack([H, prog.x])
    
#     return 2*H.T - 1


def select_bits(N, rank_x, rank_y, dl=None):
    
    bits = np.log2(N)
    
    # if dl is None:
    #     dl = 
    
    ## generate hadamard matrix
    F = util.F2(bits) # F2
    lex = np.sort(F, axis=1)*(np.argsort(F, axis=1) + 1) 
    idx = np.lexsort(np.hstack([lex, F.sum(1, keepdims=True)]).T)
    F = F[idx] # put it in kinda-lexicographic order
    
    H = np.mod(F@F[1:].T, 2)
    
    ## select distractor classes
    X = H[:,:rank_x]
    
    ## description length of possible target classes
    l = F[1:].sum(1)
    ovlp = F[1:]@F[1:].T 
    is_subset = (ovlp == l[None,:])
    
    entgl = l[rank_x:]   # potential targets
    
    entgl[(np.ceil(entgl/2) < np.min(entgl))] = 2
    
    max_len = np.max(l[:rank_x])
    deez = (l[:rank_x] == max_len)
    for y in range(N - rank_x - 1):
        
        guys = is_subset[y+rank_x][:rank_x]*deez
        
        if np.sum(guys) > 0:
            G = nx.Graph(F[1:rank_x+1][guys]@F[1:rank_x+1][guys].T == 0)
            L = np.max([len(clq) for clq in nx.find_cliques(G)])
            
            entgl[y] -= L*(max_len - 1)
    
    # ent = l[rank_x:]
    # cum = np.zeros((N - rank_x, N))
    # for x, i in enumerate(X[bits:].T):
        
    #     valid = (cum@x == 0)
        
    # ys = H[:,rank_x:]
    
    # entgl = []
    # for y in range(rank_x-1, n):
        
    #     if np.min(l[rank_x:]) > l[y]//2:
    #         entgl.append(1)
    #     else:
            
            
    
    # for this_l in range(1,bits):
    #     A = orth[l==this_l, :][:, l==this_l]
        

    ## select target classes
    # Y = ys[:, np.isin(ent, dl)][:,:rank_y]
    # ys = 

class BalancedOrthogonal:
    
    def __init__(self, num_bits, num_targets, cka, num_distractor=None):
        
        self.num_cond = 2**num_bits
        
        ## Generate hadamard matrix
        F = util.F2(num_bits) # F2
        lex = np.sort(F, axis=1)*(np.argsort(F, axis=1) + 1) 
        idx = np.lexsort(np.hstack([lex, F.sum(1, keepdims=True)]).T)
        F = F[idx] # put it in kinda-lexicographic order
        
        H = np.mod(F@F[1:].T, 2)
        
        ## Choose to be symmetric
        this_l = np.where(spc.binom(num_bits, np.arange(num_bits+1)) >= num_targets)[0][-1]
        these_targs = (F.sum(1) == this_l)
        these_targs *= np.cumsum(these_targs) < num_targets
        
        y = H[:,these_targs]
        
        ## Draw Gram matrix
        pi_x = sample_from_surface(1*these_targs, cka, size=1)
        
        
        
        

#%%
N = 3

V = la.eigh(util.center_kernel(np.eye(N)))[1][:,1:]

dicplt.polytope(V, alpha=0.5, color='k', zorder=1)
dicplt.square_axis()

for c in np.linspace(0,1,7):
    pp = V.T@sample_from_surface(np.array([1,0,0]), c, size=100)
    plt.scatter(pp[0], pp[1], alpha=0.1, zorder=10)

#%%
noise = 0.6
num_bits = 5
samps = 5000

clf = svm.LinearSVC()

perf = []
pr = []
ps = []
for c in np.linspace(0, 1, 100):
    
    ## Generate hadamard matrix
    F = util.F2(num_bits) # F2
    lex = np.sort(F, axis=1)*(np.argsort(F, axis=1) + 1) 
    idx = np.lexsort(np.hstack([lex, F.sum(1, keepdims=True)]).T)
    F = F[idx] # put it in kinda-lexicographic order
    
    H = np.mod(F@F[1:].T, 2)
    
    ## Choose to be symmetric
    these_targs = np.eye(2**num_bits - 1)[:,-1] > 0
    
    y = np.squeeze(H[:,these_targs])
    
    # for _ in tqdm(range(100)):
    #     ## Draw Gram matrix
    #     pi_x = util.sample_aligned(1*these_targs, c, size=1)
        
    #     cnd = np.random.choice(2**num_bits, size=samps)
        
    #     X = (2*H-1)@np.diag(np.sqrt(np.squeeze(np.abs(pi_x)))) 
        
    #     clf.fit(X[cnd] + np.random.randn(samps, 2**num_bits - 1)*noise, y[cnd])
    #     perf.append(clf.score(X[cnd] + np.random.randn(samps, 2**num_bits - 1)*noise, y[cnd]))
    
    #     pr.append(util.participation_ratio(X.T))
    # for _ in tqdm(range(100)):
        ## Draw Gram matrix
    pi_x = util.average_aligned(1*these_targs, c, size=1)
    
    cnd = np.random.choice(2**num_bits, size=samps)
    
    X = (2*H-1)@np.diag(np.sqrt(np.squeeze(np.abs(pi_x)))) 
    
        # clf.fit(X[cnd] + np.random.randn(samps, 2**num_bits - 1)*noise, y[cnd])
        # perf.append(clf.score(X[cnd] + np.random.randn(samps, 2**num_bits - 1)*noise, y[cnd]))
    
    pr.append(util.participation_ratio(X.T))
    ps.append(dics.efficient_parallelism(X.T, y))
    
        



