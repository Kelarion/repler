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


# def make_integer_system(A, new_dist):
#     """
#     Given a binary symbol matrix A, and integer new_dist, construct the inequalities
#     to constrain inference of unknown overlaps.
    
#     Returns:
#         C (*, len(A))
#         d (*, 1)
#         s.t. for any b inside C b <= d, the system Ax = b has a binary solution
#         (that's the hope, at least)
    
#     """
    
#     A_unq, idx = np.unique(A,axis=0, return_inverse=True)
    
#     # # all pairs
#     kern = A_unq@A_unq.T
#     aye, jay = np.triu_indices(len(A_unq), 1)
#     cols = np.zeros(len(aye)*2, int)
#     cols[0::2] = aye.astype(int)
#     cols[1::2] = jay.astype(int)
#     cols = np.repeat(cols, 3)
#     rows = np.tile(np.arange(3),2*len(aye)) + np.repeat(np.arange(len(aye))*3, 6)
    
#     signs = np.array([1, 1, -1, 1, -1, 1])
    
#     P = np.zeros((len(aye)*3, len(A_unq)))
#     P[rows, cols] = np.tile(signs, len(aye))
    
#     k = np.zeros(len(P))
#     k[P.sum(1)==2] = new_dist
#     k += (P*(P>0)*(P.sum(1, keepdims=True)<=0))@np.diag(kern)
#     k += (P.sum(1)-1)*np.repeat(kern[aye,jay],3)
    
#     # all dependent triplets
#     if len(A_unq) > 2:
#         circs = []
#         for i in range(len(A_unq)-1):
#             spans = np.mod(A_unq[[i]] + A_unq[i+1:,:],2)
            
#             matches = np.triu(np.mod(A_unq[i+1:,None,:] - spans[None,:,:], 2).sum(-1) == 0)
            
#             a, b = np.where(matches)
#             circs += [set([i,j+i+1,k+i+1]) for j,k in zip(a,b) ]
        
#         # circs = np.unique(circs)
#         # cols = np.repeat(np.concatenate([list(c) for c in circs]),8)
#         # rows = np.tile(np.arange(8),3*len(circs)) + np.repeat(np.arange(len(circs))*8, 24)
        
#         # signs = 1 - 2*np.mod(np.arange(24)//np.repeat([4,2,1], 8), 2)
        
#         # C = np.zeros((len(circs)*8, len(A_unq)))
#         # C[rows, cols] = np.tile(signs, len(circs))
        
#         # arguably not the most elegant way of implementing this
#         # d_mat = np.concatenate([2*((C.sum(1,keepdims=True) == 3) + -1*(C.sum(1,keepdims=True) == -3)),
#         #                         C*((C.sum(1,keepdims=True) == 1) + (C.sum(1,keepdims=True) == -3)),
#         #                         2*(C.sum(1,keepdims=True) == -3)], axis=1)
        
#         cols = np.repeat(np.concatenate([list(c) for c in circs]),7)
#         rows = np.tile(np.arange(7),3*len(circs)) + np.repeat(np.arange(len(circs))*7, 21)
        
#         signs = 1 - 2*np.mod(np.arange(21)//np.repeat([4,2,1], 7), 2)
        
#         C = np.zeros((len(circs)*7, len(A_unq)))
#         C[rows, cols] = np.tile(signs, len(circs))
        
#         d_mat = np.concatenate([2*((C.sum(1,keepdims=True) == 3) + -1*(C.sum(1,keepdims=True) == -3)),
#                                 C*((C.sum(1,keepdims=True) == 1) + (C.sum(1,keepdims=True) == -3))], axis=1)
#         d = d_mat @ np.vstack([new_dist, A_unq.sum(1,keepdims=True)])
        
#         return np.vstack([P, C])[:,idx], np.concatenate([k.squeeze(), d.squeeze()])
#     else:
#         return P[:,idx], k.squeeze()
    

# def solve_integer_system(symb, cap):
#     """
#     for a new item, assign to it existing symbols such that the overlaps with
#     all the existing items are satisfied.
    
#     i.e. find a binary x which solves Ax = b for an integer-valued b
    
#     solves this as a special type of generalized flow network, algorithm 
#     inspired by Ford-Fulkerson and modified to the structure of this network.
#         (speficially, FF finds a residual path and adds flow along it, until 
#           there are no residual paths. fortunately, the network structure makes
#           it easy to find residual paths even though this isnt a regular network)
    
#     symb (num_item, num_symbol) {0,1}-matrix
#     ovlp (num_item, ) overlaps of new item with each existing item
#     """
    
#     N,S = symb.shape
    
#     flow = np.zeros(S)
#     resid = cap - symb@flow
#     max_cap = np.nanmax(np.where(symb,symb*cap[:,None],np.nan), axis=0)
#     good_syms = max_cap == np.max(cap) # flow should go through max capacity nodes
    
#     warning = False
#     while np.any(resid): # this should always converge
#         min_cap = np.nanmin(np.where(symb,symb*resid[:,None],np.nan), axis=0)
#         best = np.argmax(min_cap*good_syms*(1-flow))
#         if (flow[best] > 0) or (min_cap[best] < 1): # i feel like this shouldn't ever happen but not sure
#             warning = True # this means that the overlaps aren't satisfied
#             break
#         flow[best] = 1
#         resid = cap - symb@flow
    
#     return flow, warning


# #### these functions are for a generic max flow approach, which won't work unfortunately
# # def make_flow_network(symb, new_item):
# #     """
# #     given the known symbols for a set of items, and the overlaps of each of 
# #     those items with a new item, return the relevant flow network 
    
# #     symb (num_item, num_symbol) {0,1}-matrix
# #     new_item (num_item + 1, ) new column to append to kern
# #     """
    
# #     N, S = symb.shape
    
# #     nodes = np.arange(N+S+2)
    
# #     edges = []
# #     caps = []
# #     gains = []
# #     # first from the source to the symbols
# #     edges += [(0,i) for i in range(1,S+1)]
# #     caps += [1]*S
# #     gains += symb.sum(0).tolist()
    
# #     for s in range(S): # then from symbols to items
# #         edges += [(s+1,S+1+i) for i in np.where(symb[:,s])[0]]
# #         caps += [1]*np.sum(symb[:,s])
# #         gains += [1]*np.sum(symb[:,s])
    
# #     # then from items to the sink
# #     edges += [(i,N+S+1) for i in range(S+1,S+1+N)]
# #     caps += new_item[:-1].tolist()
# #     gains += [1]*N
    
# #     return nodes, edges, np.array(caps), np.array(gains)


# # def flow_conservation_matrix(nodes, edges, gains):
# #     """
# #     convert the flow network into a linear equality constraint for an LP
# #     """
    
# #     # find all nodes which aren't the source or sink
# #     eligible = [np.all(np.isin(edges, i).sum(0)>0) for i in nodes]
    
# #     M = []
# #     for i,n in enumerate(nodes):
# #         if eligible[i]:
# #             M.append(gains*(np.isin(edges, i)*[[1,-1]]).sum(1))
    
# #     return np.array(M) 


def gauss_projection(c):
    return 0.5 + (np.arcsin(c) - np.arccos(c))/np.pi


#%%

def revert(S_unq, num_s, steps=1):
    """
    Restore S to previous fit
    """
    
    old_S, which = np.unique(S_unq[:-1], axis=1, return_inverse=True)
    old_num_s = np.array([int(np.round(np.sum(num_s[which==i]))) for i in range(old_S.shape[1])])
    
    return old_S[:,np.argsort(-old_num_s)], old_num_s[np.argsort(-old_num_s)]


def update(S_unq, num_s, x, length=None):
    
    has = x > 1e-12
    split = (num_s - x >= 1)
    
    new_S_unq = np.block([[S_unq, S_unq[:,has&split]], 
                          [has, np.zeros(sum(has&split))]])
    new_num_s = np.concatenate([np.where(has,x,num_s), num_s[has&split] - x[has&split]])
    
    if length is not None:
        resid = int(np.round(length - np.sum(x))) # create new symbols to make up the distance
        if resid > 0:
            new_S_unq = np.hstack([new_S_unq, np.eye(len(new_S_unq))[:,[-1]]])
            new_num_s = np.concatenate([new_num_s, [resid]])
        
    return new_S_unq, new_num_s


def next_item(c, S, counts, K, included, remaining, sparse=True):
    """
    Find best item to fit next
    """    
    
    sgn = ((-1)**sparse)
    
    diffs = []
    mins = []
    succ = []
    # found = False
    for o in range(len(included)):
        A = np.mod(S+S[[o]], 2)*counts[None,:]
        
        K_o = K[o,o]- K[[o],:] - K[:,[o]] + K
        
        bs = K_o[included,:][:,remaining]
        maxlen = K_o[remaining, remaining]
        
        o_diffs = []
        o_mins = []
        
        for i in range(len(remaining)):
            
            prog = lp(c,
                      A_ub=c[None,:], b_ub=maxlen[i],
                      A_eq=A, b_eq=bs[:,i],
                      bounds=[0,1],
                      method='highs')
    
            prog2 = lp(-c,
                        A_ub=c[None,:], b_ub=maxlen[i],
                        A_eq=A, b_eq=bs[:,i],
                        bounds=[0,1],
                        method='highs')
            # prog = lp(-1*sgn*np.ones(len(c)),
            #           A_ub=np.ones((1,len(c))), b_ub=maxlen[i],
            #           A_eq=S_unq, b_eq=bs[:,i],
            #           bounds=[(0,n) for n in num_s],
            #           method='highs')
            
            # succ.append(prog.success)
            
            if prog.success and prog2.success:
                o_diffs.append(np.abs((prog2.x - prog.x)*c).sum())
            else:
                o_diffs.append(np.nan)
            
    
            if prog.success:
                o_mins.append(prog.x@c / maxlen[i])
            else:
                # xs.append( prog.x*num_s )
                o_mins.append(np.nan)
        
        diffs.append(o_diffs)
        mins.append(o_mins)
            
    diffs = np.array(diffs)
    mins = np.array(mins)
    
    valid = ~(np.isnan(diffs)+np.isnan(mins))

    if np.any(valid):
        
        min_diff = diffs == np.nanmin(diffs[valid])
        if sparse:
            these_o, these_i = np.where(min_diff&(mins == np.nanmax(mins[min_diff])))
        else:
            these_o, these_i = np.where(min_diff&(mins == np.nanmin(mins[min_diff])))
            
        id_best = these_i[0]
        orig_best = these_o[0]
            
        this_i = remaining[id_best]
        
        K_o = K[orig_best,orig_best]- K[[orig_best],:] - K[:,[orig_best]] + K
        
        ### new features
        prog = lp(-1*sgn*np.ones(len(counts)),
                  A_ub=np.ones((1,len(counts))), b_ub=K_o[this_i,this_i],
                  A_eq=S, b_eq=K_o[included,this_i],
                  bounds=[(0,n) for n in counts],
                  method='highs',
                  integrality=1)
        
        x_int = np.round(prog.x).astype(int)
        
        ### transform back to reference basis
        x_int = np.abs(S[orig_best]-x_int)
        
        return x_int
        
    else:
        raise Exception('Inconsistency at %d items'%len(included))


def color_points(symbols, edges):
    """
    Given the set of arrow-symbols, color the start and end points accordingly. 
    Does this by finding the coloring of the bipartite graph associated with each
    symbol's arrows. 
    
    """
    inds = util.LexOrder() 
    # N = len(np.unique(inds.inv(edges)))
    pts = np.unique(inds.inv(edges)).astype(int)
    
    # the arrows which share a given symbol cannot form an odd cycle, which 
    # means they form a bipartite graph (i.e. a binary coloring)
    vecs = []
    
    for i in range(symbols.shape[1]):
        e = np.stack(inds.inv(edges[symbols[:,i]>0])).astype(int).T.tolist()
        
        G = nx.Graph()
        G.add_edges_from(e)
        pos, neg = nx.bipartite.sets(G)
        if len(pos)<=len(neg): # x and 1-x are equivvalent, use the smallest
            vecs.append(1*np.isin(pts,list(pos)))
        else:
            vecs.append(1*np.isin(pts,list(neg)))

    return np.array(vecs), pts


def cull(S_unq, num_s, thresh=1e-3):
    
    these = num_s/np.sum(num_s) <= thresh
    H,U = column_style_hermite_normal_form(S_unq)
    
    r = np.sum(np.diag(H)>0)
    
    valid = np.abs(U[:,:r]).max(1) <= 1e-6
    
    return S_unq[:,~(these&valid)], num_s[~(these&valid)]


def feasible(S_unq, num_s, included, remaining):
    
    M = S_unq.shape[1]
    slack = np.block([[S_unq, np.zeros((len(S_unq),M+1))],[np.eye(M),np.eye(M), np.zeros((M,1))],[np.ones((1,M)), np.eye(M+1)[-1]]])
    b = np.vstack([K_o[included,:][:,remaining], np.repeat(num_s[:,None], len(remaining), axis=1), K_o[remaining,remaining]])
    return [fark(slack, this_b)[0] for this_b in b.T]
    

def intBDF(C, thresh=1e-6, in_cut=False, sparse=True, num_samp=5000):
    """
    Binary distance factorization (working name) of K. 
    
    """
    
    N = len(C)
    inds = util.LexOrder()
    sgn = ((-1)**sparse)
    
    ## "Project" into cut polytope, if it isn't already there
    if not in_cut:
        samps = np.sign(np.random.multivariate_normal(np.zeros(N), C, size=num_samp))
        K = samps.T@samps
    else:
        K = C
    
    d = 1 - K/num_samp
    
    orig = np.argmin(sgn*np.sum(d, axis=0)) 
    
    ### center around arbitrary (?) point
    K_o = K[orig,orig]- K[[orig],:] - K[:,[orig]] + K
    
    idx = np.arange(N)
    # i0 = idx[idx != orig][0]
    first = np.setdiff1d(idx, [orig])[np.argmin(sgn*d[orig,:][np.setdiff1d(idx, [orig])])]
    
    S_unq = np.array([[0],[1]])
    num_s = np.array([K_o[first,first]])
    
    included = [orig, first]
    remaining = np.setdiff1d(idx, [orig, first]).tolist()
    
    while len(remaining) > 0:

        ## Most "explainable" item <=> closest item
        this_i = remaining[np.argmin(sgn*np.sum(d[included,:][:,remaining], axis=0))]
        
        ### new features
        prog = lp(-1*sgn*np.ones(len(num_s)),
                  A_ub=np.ones((1,len(num_s))), b_ub=K_o[this_i,this_i],
                  A_eq=S_unq, b_eq=K_o[included,this_i],
                  bounds=[(0,n) for n in num_s],
                  method='highs',
                  integrality=1)
        
        if prog.success:
            x_int = np.round(prog.x).astype(int)
        else:
            raise Exception('Inconsistency at %d items'%len(included))
        

        ### Split clusters to account for fractional membership
        new_S_unq, new_num_s = update(S_unq, num_s, x_int, K_o[this_i,this_i])
        
        S_unq = new_S_unq[:, np.argsort(-new_num_s)]
        num_s = new_num_s[np.argsort(-new_num_s)]
        
        ### Discard clusters (find a good way to do this)
        # S_unq, num_s = cull(S_unq, num_s, thresh=thresh)
        
        included.append(this_i)
        remaining.remove(this_i)
    
    ### fill in remaining difference vectors 
    S_unq = S_unq[np.argsort(included)]
    
    S_full = np.vstack([np.mod(S_unq[(i+1):] + S_unq[i], 2) for i in range(N)])
    
    pt_id = np.sort(included)
    ix = inds(np.concatenate([pt_id[(i+1):] for i in range(N)]), 
              np.concatenate([np.ones(N-i-1)*i for i in range(N)]))
    
    
    ### convert to 'canonical' form
    vecs, pts = color_points(S_full, ix)
    
    return vecs, num_s

#%%
def BDF(K, sparse=True, in_cut=False, num_samp=None, zero_tol=1e-6):
    """
    Binary distance factorization (working name) of K. Using the non-integer 
    formulation.
    """
    
    N = len(K)
    inds = util.LexOrder()
    sgn = (-1)**(sparse + 1) # + if sparse else -
    
    ## "Project" into cut polytope, if it isn't already there
    if not in_cut:
        if num_samp is None:
            C_ = gauss_projection(K)
        else:
            samps = np.sign(np.random.multivariate_normal(np.zeros(N), K, size=num_samp))
            C_ = samps.T@samps/num_samp
    else:
        C_ = K
    
    d = 1 - C_
    
    if sparse:
        alpha = np.sum(d)/np.sum(d**2)   # distance scale which minimizes correlations
        # alpha = (1+np.max(d))/(N/2) # unproven: this is largest alpha which returns sparsest solution?
    else:
        if not np.mod(N, 2):
            alpha = (N**2)/(d.sum()) # project onto balanced dichotomies
        
    C = 1 - alpha*d
    
    orig = np.argmin(sgn*np.sum(d, axis=0)) 
    
    ### center around arbitrary (?) point
    K_o = (C[orig,orig] - C[[orig],:] - C[:,[orig]] + C)/4
    
    idx = np.arange(N)
    first = np.setdiff1d(idx, [orig])[np.argmin(sgn*d[orig,:][np.setdiff1d(idx, [orig])])]
    
    B = np.ones((1,1), dtype=int)
    pi = np.array([K_o[first,first]])
    
    included = [first]
    remaining = np.setdiff1d(idx, [orig, first]).tolist()
    
    while len(remaining) > 0:

        ## Most "explainable" item <=> closest/furthest item
        this_i = remaining[np.argmin(sgn*np.sum(d[included,:][:,remaining], axis=0))]
        
        ### new features
        prog = lp(sgn*np.ones(len(pi)),
                  A_ub=np.ones((1,len(pi))), b_ub=K_o[this_i,this_i],
                  A_eq=B, b_eq=K_o[included,this_i],
                  bounds=[[0,p] for p in pi],
                  method='highs')
        
        if prog.success:
            x = prog.x
        else:
            raise Exception('Inconsistency at %d items'%len(included))
        
        ## Split clusters as necessary 
        has = x > zero_tol
        split = x < pi
        
        new_B = np.block([[B, B[:,has&split]], 
                          [has, np.zeros(sum(has&split))]])
        new_pi = np.concatenate([np.where(has, x, pi), pi[has&split] - x[has&split]])
        
        resid = K_o[this_i, this_i] - np.sum(x) # create new symbols to make up the distance
        if resid > 0:
            new_B = np.hstack([new_B, np.eye(len(new_B))[:,[-1]]])
            new_pi = np.concatenate([new_pi, [resid]])
            
        # new_pi /= np.sum(new_pi)
        
        B = new_B
        pi = new_pi
        
        ### Discard clusters (find a good way to do this)
        # S_unq, num_s = cull(S_unq, num_s, thresh=thresh)
        
        included.append(this_i)
        remaining.remove(this_i)
    
    ## Fix the matrix
    B = np.vstack([np.zeros(len(pi)), B])
    trivial = (B.sum(0) == N) | (B.sum(0) == 0)
    B = B[:,~trivial]
    pi = pi[~trivial]/np.sum(pi[~trivial])
    
    feat_idx = np.argsort(-pi)
    
    ### fill in remaining difference vectors 
    idx = np.append(orig, included)
    idx_order = np.argsort(idx)
    
    B = B[idx_order][:,feat_idx]
    pi = pi[feat_idx]
    
    B_full = np.vstack([np.mod(B[(i+1):] + B[i], 2) for i in range(N)])
    
    pt_id = idx[idx_order]
    ix = inds(np.concatenate([pt_id[(i+1):] for i in range(N)]), 
              np.concatenate([np.ones(N-i-1)*i for i in range(N)]))
    
    ### convert to 'canonical' form
    vecs, pts = color_points(B_full, ix)
    
    return vecs.T, pi


#%%

## this version of BDF does not produce the sparsest solutions

# def BDF(K, thresh=1e-6, in_cut=False, sparse=True, num_samp=5000, zero_tol=1e-6):
#     """
#     Binary distance factorization (working name) of K. Using the non-integer 
#     formulation.
#     """
    
#     N = len(K)
#     inds = util.LexOrder()
#     sgn = (-1)**(sparse)
    
#     ## "Project" into cut polytope, if it isn't already there
#     if not in_cut:
#         # samps = np.sign(np.random.multivariate_normal(np.zeros(N), K, size=num_samp))
#         # C_ = samps.T@samps/num_samp
#         C_ = gauss_projection(K)
#     else:
#         C_ = K
    
#     d = 1 - C_
    
#     if sparse:
#         alpha = np.sum(d)/np.sum(d**2)   # distance scale which minimizes correlations
#         # alpha = (1+np.max(d))/(N/2) # unproven: this is largest alpha which returns sparsest solution?
#     else:
#         if not np.mod(N, 2):
#             alpha = (N**2)/(d.sum()) # project onto balanced dichotomies
        
#     C = 1 - alpha*d
    
#     orig = np.argmin(sgn*np.sum(d, axis=0)) 
    
#     ### center around arbitrary (?) point
#     K_o = (C[orig,orig] - C[[orig],:] - C[:,[orig]] + C)/4
    
#     idx = np.arange(N)
#     first = np.setdiff1d(idx, [orig])[np.argmin(sgn*d[orig,:][np.setdiff1d(idx, [orig])])]
    
#     B = np.array([[1,1],[0,1]])
#     pi = np.array([1-K_o[first,first], K_o[first,first]])
    
#     included = [first]
#     remaining = np.setdiff1d(idx, [orig, first]).tolist()
    
#     for n in tqdm(range(2,N)):

#         ## Most "explainable" item <=> closest item
#         this_i = remaining[np.argmin(sgn*np.sum(d[included,:][:,remaining], axis=0))]
        
#         ### new features
#         prog = lp(sgn*pi,
#                   A_eq=B@np.diag(pi), b_eq=K_o[[this_i] + included,this_i],
#                   bounds=[0,1],
#                   method='highs')
        
#         if prog.success:
#             x = prog.x
#         else:
#             raise Exception('Inconsistency at %d items'%len(included))
        
#         ## Split clusters as necessary 
#         new_pi = []
#         new_b = []
#         for i,x_i in enumerate(x):
            
#             if np.abs(x_i**2 - x_i) <= zero_tol:
#                 new_b.append(int(np.round(x_i)))
#                 new_pi.append(pi[i])
            
#             else:
                
#                 new_b += [0, 1]
#                 new_pi += [pi[i]*(1-x_i), pi[i]*x_i]
        
#         B = np.vstack([np.repeat(B, 1 + (np.abs(x**2 - x) >= zero_tol), axis=1), new_b])
#         pi = np.array(new_pi)
        
#         ### Discard clusters (find a good way to do this)
#         # S_unq, num_s = cull(S_unq, num_s, thresh=thresh)
        
#         included.append(this_i)
#         remaining.remove(this_i)
    
#     ## Fix the matrix
#     B[0] = 0
#     trivial = (B.sum(0) == N) | (B.sum(0) == 0)
#     B = B[:,~trivial]
#     pi = pi[~trivial]/np.sum(pi[~trivial])
    
#     feat_idx = np.argsort(-pi)
    
#     ### fill in remaining difference vectors 
#     idx = np.append(orig, included)
#     idx_order = np.argsort(idx)
    
#     B = B[idx_order][:,feat_idx]
#     pi = pi[feat_idx]
    
#     B_full = np.vstack([np.mod(B[(i+1):] + B[i], 2) for i in range(N)])
    
#     pt_id = idx[idx_order]
#     ix = inds(np.concatenate([pt_id[(i+1):] for i in range(N)]), 
#               np.concatenate([np.ones(N-i-1)*i for i in range(N)]))
    
#     ### convert to 'canonical' form
#     vecs, pts = color_points(B_full, ix)
    
#     return vecs.T, pi


#%%

def SDF(K, zero_tol=1e-10, sparse=True, in_cut=False, thresh=1e-5, num_samp=5000):
    """
    Sign distance factorization
    """
    
    P = len(K)
    sgn = (-1)**(sparse + 1)
    
    ## "Project" into cut polytope, if it isn't already there
    if not in_cut:
        # samps = np.sign(np.random.multivariate_normal(np.zeros(P), K, size=num_samp))
        # C_ = samps.T@samps/num_samp
        C_ = gauss_projection(K)
    else:
        C_ = K
    
    ## Regularize to minimum-norm correlation matrix
    ## this should be re-visited, doesn't work well for e.g. identity matrix
    d = 1 - C_   
    
    if sparse:
        alpha = np.sum(d)/np.sum(d**2)   # distance scale which minimizes correlations
        # alpha = (1+np.max(d))/(N/2) # unproven: this is largest alpha which returns sparsest solution?
    else:
        if not np.mod(P, 2):
            alpha = (P**2)/(d.sum()) # project onto balanced dichotomies
            
    C = 1 - alpha*d
    
    ## Fit features
    S = np.ones((1,1), dtype=int) # signs
    pi = np.ones(1)    # probabilities
    
    first = np.argmin(np.sum(d, axis=0)) 
    
    included = [first]
    remaining = np.setdiff1d(range(P), first).tolist()
    for n in tqdm(range(1,P)):
    # for n in tqdm(range(1,4)):
        
        ## Most "explainable" item <=> closest item
        new = remaining[np.argmin(sgn*np.sum(d[included,:][:,remaining], axis=0))]
        
        if n > 2:
            
            ## Average gradient of the nth moment
            w = np.mean([S[[these]].prod(0) for these in combinations(range(n), n-np.mod(n+1,1))], axis=0)

            prog = lp(sgn*w*pi, A_eq=S@np.diag(pi), b_eq=C[included, new], bounds=[-1,1])
            x = prog.x
            
            ## deal with out-of-bounds solutions
            if x is None:
                
                raise Exception('dagnammit! catastophe at %d items'%len(included))
        
        else:
            x = la.pinv(S@np.diag(pi))@C[included,:][:, new]
        
        ## Split clusters as necessary 
        new_pi = []
        new_s = []
        for i,x_i in enumerate(x):
            
            if 1 - x_i**2 <= zero_tol:
                new_s.append(np.sign(x_i))
                new_pi.append(pi[i])
            
            else:
                pi_pos = 0.5*(x_i + 1)
                
                new_s += [1, -1]
                new_pi += [pi[i]*pi_pos, pi[i]*(1-pi_pos)]
        
        S = np.vstack([np.repeat(S, 1 + (1 - x**2 >= zero_tol), axis=1), new_s])
        pi = np.array(new_pi)
        
        ## Remove likely noise clusters (find a good way to do this)
        chaf = pi <= thresh
        S = S[:,~chaf]
        pi = pi[~chaf]/np.sum(pi[~chaf])
        
        included.append(new)
        remaining.remove(new)
    
    S = S[np.argsort(included),:][:,np.argsort(-pi)]
    pi = pi[np.argsort(-pi)]
    
    ## remove trivial cluster
    mono = np.abs(S.sum(0)) == P
    S = S[:,~mono]
    pi = pi[~mono]/np.sum(pi[~mono])
    
    ## report smallest cluster
    S = S@np.diag(-np.sign(S.sum(0) + 1e-5)).astype(int)
    
    return S, pi


#%%

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
    

def center(K):
    return K - K.mean(-2,keepdims=True) - K.mean(-1,keepdims=True) + K.mean((-1,-2),keepdims=True)

inds = util.LexOrder()

def extract_rank_one(A, rand=False, kern=None):
    """
    Looking for rank-1 binary components of A, symmetric PSD
    """
    
    N = len(A)
    psi = oriented_tricycles(N)
    
    l, v = la.eigh(center(A))
    null = v[:,l <= 1e-6]
    
    if rand:
        # g = np.random.randn(N,1)
        # g = np.random.laplace(size=(N,1))
        g = np.random.multivariate_normal(np.zeros(N), A)[:,None]
    else:
        g = v[:, l >= l.max()-1e-6]
   
    active = (np.abs(psi@inv_matrix_map(A) - 1) <= 1e-5) 
   
    X = cvx.Variable((N,N), symmetric='True')
    constraints = [X >> 0]
    constraints += [cvx.diag(X) == 1]
    constraints += [cvx.trace(null@null.T@X) == 0]
    constraints += [X[:,[i]] + X[[i],:] - X <= 1 for i in range(N)]
    constraints += [X[:,[i]] + X[[i],:] + X >= -1 for i in range(N)]
    if kern is not None:
        constraints += [cvx.trace(k@X) == 0 for k in kern]
    if active.sum() > 0:
        constraints += [cvx.trace(np.triu(matrix_map(p)-np.eye(N))@X) == 1 for p in psi[active]]
    #constraints += [cvx.trace(C@X)/2 <= 1 for C in matrix_map(psi)]
    
    prob = cvx.Problem(cvx.Maximize(cvx.trace(g@g.T@X)), constraints)
    prob.solve()
    
    return X.value


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


def randomard(N):
    """
    Random hadamard matrix of order N, where N is a multiple of 4
    """
    
    H = np.ones((N,1))
    
    while H.shape[1] < N:
        
        new_col = np.round(orthogonal_rank_one(H), 6)
        if np.abs(new_col**2 - 1).max() > 1e-4:
            it = 0
            while np.abs(new_col**2 - 1).max() > 1e-4:
                new_col = np.round(orthogonal_rank_one(H), 6)
                if it > 10:
                    raise Exception
                it += 1
            
        H = np.hstack([H, np.sign(new_col[:,[0]])])
    
    return H




def deflate(A, X):
    
    
    zi = cvx.Variable(1)
    
    constraints = [zi*A + (1-zi)*X >> 0]
    # constraints += [zi*(A[:,[i]] + A[[i],:] - A) + (1-zi)*(X_[:,[i]] + X_[[i],:] - X_) <= 0 for i in range(N)]
    # constraints += [zi*(A[:,[i]] + A[[i],:] + A) + (1-zi)*(X_[:,[i]] + X_[[i],:] + X_) >= 0 for i in range(N)]
    prob = cvx.Problem(cvx.Maximize(zi), constraints)
    prob.solve()
        
    # print(zi.value)
    
    return (zi.value*A + (1-zi.value)*X)


def kurtope(N):
    """
    Linear constraints on kurtosis of sign variables
    """
    
    fours = np.stack(combinations(range(N), 4))
    # twos = combinations(range(N), 2)
    
    
    
def sign_comp_iteration(A):
    
    
    l,V = la.eigh(A)
    # g = V[:,[0]]
    
    U = V[:,l> 1e-7]
    
    N = len(A)
    
    g = np.random.randn(N,1)
    
    X = cvx.Variable((N,N), symmetric='True')

    # A = [np.eye(N)[:,[i]]*np.eye(N)[[i],:] for i in range(N)]
    
    constraints = [X >> 0]
    # constraints += [cvx.trace(A[i]@X) == 1 for i in range(N) ]
    constraints += [cvx.diag(X) == 1]
    constraints += [cvx.trace(U.T@X@U) == N]
    # constraints += [X - X[:,[i]] - X[[i],:] + X[i,i] >= 0 for i in range(N)]
    constraints += [X[:,[i]] + X[[i],:] - X <= 1 for i in range(N)]
    constraints += [X + X[:,[i]] + X[[i],:] >= -1 for i in range(N)]

    prob = cvx.Problem(cvx.Maximize(cvx.trace(g@g.T@X)), constraints)
    prob.solve()
    
    X_ = X.value
    s_i = np.sign(la.eig(X_)[1][:,0])
    
    zi = cvx.Variable(1)
    
    constraints = [zi*A + (1-zi)*X_ >> 0]
    # constraints += [zi*(A[:,[i]] + A[[i],:] - A) + (1-zi)*(X_[:,[i]] + X_[[i],:] - X_) <= 0 for i in range(N)]
    # constraints += [zi*(A[:,[i]] + A[[i],:] + A) + (1-zi)*(X_[:,[i]] + X_[[i],:] + X_) >= 0 for i in range(N)]
    prob = cvx.Problem(cvx.Maximize(zi), constraints)
    prob.solve()
    
    A_ = (zi.value*A + (1-zi.value)*X_)
    
    return A_, s_i
    

def correlify(B):
    
    K = 4*B - 2*np.diag(B)[:,None] - 2*np.diag(B)[None,:] 
    
    return K


def cut_vertices(N):
    """
    Return vertices of cut polytope over N items
    """
    
    trind = inds.inv(np.arange(0.5*N*(N-1)))
    s = 2*tasks.StandardBinary(N)(range(2**(N-1))).numpy()-1
    v = (s[:,:,None]*s[:,None,:])[:,trind[0], trind[1]]
    
    return v


def tricycles(N):
    """
    Return all 3-cycles of N-vertex complete graph
    """

    ntri = int(0.5*N*(N-1))

    cyc = np.array([[ inds(i,j) for i,j in combinations(c,2)] for c in combinations(range(N),3)])
    
    return np.eye(ntri)[:,cyc].sum(-1).T


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


def matrix_map(x):
    
    aye, jay = inds.inv(np.arange(x.shape[-1]))
    
    N = int(aye.max())+1
    
    A = np.zeros((*x.shape[:-1], N, N))
    
    A[...,aye,jay] = x
    A[...,jay,aye] = x
    
    return A + np.expand_dims(np.eye(N), tuple(range(np.ndim(x)-1)))


def inv_matrix_map(A):
    
    # trind = np.triu_indices(A.shape[-1], k=1)
    N = A.shape[-1]
    trind = inds.inv(np.arange(0.5*N*(N-1), dtype=int))
    
    return A[...,trind[0], trind[1]]


#%%
from scipy.special import binom 

class LexOrder:
    """
    Lexicographic order of k-combinations. For a list K of non-negative integers, 
    the index of a list R, containing r <= max(K) non-negative integers, is:

    n(R) = sum_{i=0}^{r-1} binom(R[i], r-i) + sum_{k in K - r} binom(max(R), k)

    If only one K is supplied, this is the standard order on K-combinations, if 
    all K's from 1 to N are supplied, then this is the decimal representation of
    N-bit binary numbers (only for numbers up to 2^N).

    For k=2, this order over pairs (i,j) looks like:
    
    j = __|_0_1_2_3_
    i = 0 | - - - -
        1 | 0 - - -
        2 | 1 2 - -
        3 | 3 4 5 -
        
    """
    def __init__(self, *Ks):
        self.K = Ks
        return 
    
    def __call__(self, *items):
        """
        Send a list of items, (i,j,k,...) to their index in the lex order. 

        Each input is an array of the same length. 

        An even number of repeats of an item cancels out, e.g. (i,j,j,k) -> (i,k)
        """

        # we'll contort ourselves a bit to keep it vectorized

        sorted_items = np.flipud(np.sort(np.stack(items), axis=0))
        reps = run_lengths(sorted_items)

        live = np.mod(reps,2)

        r = live.sum(0, keepdims=True)

        # put -1 at items we don't want to consider
        # otherwise count down from r to 1
        this_k = (r - np.cumsum(live, 0) + 2)*live - 1 

        # awful way of getting K\r
        if len(sorted_items.shape) > 1:
            all_K = np.expand_dims(self.K, *range(1, len(sorted_items.shape)))
        else:
            all_K = np.array(self.K)
        above_r = all_K > r
        below_r = all_K < r
        maxval = (sorted_items*live).max(0, keepdims=True)
        top = (maxval+1)*below_r + maxval*above_r
        bot = (all_K+1)*(below_r + above_r) - 1 # same trick as above

        n = binom(sorted_items, this_k).sum(0) + binom(top, bot).sum(0)

        return np.squeeze(np.where(np.isin(r, self.K), n, -1).astype(int))
    
    # def inv(self, n):
    #     """
    #     Invert the above function -- given an index, return the list of items
    #     """



    #     return items


def run_lengths(A, mask_repeats=True):
    """ 
    A "run" is a series of repeated values. For example, in the sequence

    [0, 2, 2, 1]

    there are 3 runs, of the elements 0, 2, and 1, with lengths 1, 2, and 1
    respectively. The output of this function would be 

    [1, 2, 0, 1]

    indicating the length of each run that an element starts. Now imagine this 
    being done to each column of an array.

    This is like the array analogue of np.unique(..., return_counts=True). 
    In fact you can get the unique elements by doing something like:
    
    R = run_lengths(A)>0
    idx = np.where(R>0)
    
    vals = A[*idx]
    counts = R[*idx]

    """

    n = len(A)
    if len(A.shape) > 1:
        ids = np.expand_dims(np.arange(n), *range(1,len(A.shape)))
    else:
        ids = np.arange(n)

    changes = A[1:] != A[:-1]

    is_stop = np.append(changes, np.ones((1,*A.shape[1:])), axis=0)
    stop_loc = np.where(is_stop, ids, np.inf)
    stop = np.flip(np.minimum.accumulate(np.flip( stop_loc ), axis=0))

    is_start = np.append(np.ones((1,*A.shape[1:])), changes, axis=0)
    start_loc = np.where(is_start, ids, -np.inf)
    start = np.maximum.accumulate(start_loc , axis=0)

    if mask_repeats:
        counts = (stop - start + 1)*is_start
    else:
        counts = (stop - start + 1)

    return counts


def kurtosis_inequalities(N):
    """
    Generate set of inequalities that constrain the kurtosis of sign variables.

    Assembles a sparse {-1,0,1}-valued matrix 
    """

    # K2 = int(binom(N,2))
    K4 = int(binom(N,4))
    
    comb_inds = LexOrder(2, 4)

    rows = []
    cols = []
    vals = []
    b = np.zeros(int(binom(N,4)*(N-2)*42))

    n = 0
    for i,j,k,l in combinations(range(N),4):

        ind_ijkl = comb_inds(i,j,k,l)

        # first constrain the 4th moment <ijkl> using the known 2nd moments
        for this_i, this_j in combinations([i,j,k,l],2):
            this_k, this_l = np.setdiff1d([i,j,k,l], [this_i, this_j])

            for this_m in np.setdiff1d(range(N), [this_k, this_l]):

                # upper and lower bounds
                for this, guy in enumerate([this_m, this_k, this_l]):
                    # "triangle inequality" upper bounds
                    cols.append([ind_ijkl, 
                        comb_inds(this_i, this_j, this_k, this_m), 
                        comb_inds(this_i, this_j, this_m, this_l)])
                    rows.append([n,n,n])
                    vals.append((2*np.eye(3, dtype=int) - 1)[this].tolist())
                    b[n] = 1

                    # "triangle inequality" lower bounds
                    feller, bloke = np.setdiff1d([this_m, this_k, this_l], guy)
                    cols.append([ind_ijkl, 
                        comb_inds(this_i, this_j, this_k, this_m), 
                        comb_inds(this_i, this_j, this_m, this_l),
                        comb_inds(feller, bloke)])
                    rows.append([n+1,n+1,n+1,n+1])
                    vals.append((-(2*np.eye(3, dtype=int) - 1)[this]).tolist() + [-2])
                    b[n+1] = 1

                    n += 2

                cols.append([ind_ijkl, 
                    comb_inds(this_i, this_j, this_k, this_m), 
                    comb_inds(this_i, this_j, this_m, this_l),
                    comb_inds(this_i, this_j)])
                rows.append([n,n,n,n])
                vals.append([1,1,1,-2])
                b[n] = 1

                n += 1
    
    # return vals, cols, rows, b
    data = (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols)))
    return sprs.csr_array(data), b


    
    