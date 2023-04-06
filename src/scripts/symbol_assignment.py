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

# def concepts(distances, sparse=True):
    
#     S, p = edge_symbols(distances, sparse=sparse)
#     vecs, sets = color_points(S, p)
#     V, l = np.unique(vecs, axis=0, return_counts=True)
    
#     return V, l

# def edge_symbols(distances, sparse=True, weights=None):
#     """
#     to each edge drawn between a pair of points, assign a set of symbols which
#     represent the binary components that change between said points. 
    
#     distances (N, N) should be a hamming distance matrix: integral, symmetric,
#                     zero on the diagonal, obeys the triangle inequality.
#     """
    
#     N = len(distances)
    
#     # choose an order in which to add items
#     # idx = np.arange(N) # i don't think the order should matter ... oops it does
#     idx = choose_order(distances, sparse)
#     dist = np.round(distances).astype(int)[idx,:][:,idx] # now we can ignore idx
    
#     if dist[0,1] > 0:
#         S = np.ones((1,dist[0,1])) # this will be size (N*(N-1)/2, S) 
#     else:
#         S = np.zeros((1,1))
#     points = inds(0, 1) # start and end point of edges
    
#     # for i in range(2,N):
#     for i in range(2, 69):
#         # d_ik = dist[i,:i]
#         # choose an arrow from new item to any old item
#         # j = np.argmax(((-1)**sparse)*dist[i,:i])
#         # j = np.argmin(dist[i,:i])
#         j = i - 1
        
#         adj_ovlp, pts = compute_overlaps(dist[:i,:i], dist[i,:i], target=j)
        
#         aye, jay = inds.inv(points) # sorry about the names
#         adj = (aye==j)|(jay==j) # all co-incident edges
#         dis = ~adj
        
#         adj_idx = np.sort(np.where(adj)[0])
#         dis_idx = np.where(dis)[0]
        
#         sym_ij = symbol_program(S[adj_idx], S[dis_idx], dist[i,j], adj_ovlp[np.argsort(pts)], 
#                                 w=weights, sparse=sparse)
        
#         resid = int(dist[i,j] - np.sum(sym_ij)) # create new symbols to make up the distance
#         if resid > 0:
#             sym_ij = np.append(sym_ij, np.ones(resid))
#             S = np.append(S, np.zeros((len(S), resid)), axis=1)
        
#         # assign symbols to the other edges using composition rules (i.e. addition mod 2)
#         sym_ik = np.mod(sym_ij + S[adj].squeeze(), 2)
        
#         # extend arrays and keep account of edge endpoints
#         S = np.vstack([S, sym_ij[None,:], sym_ik]) # sorry again
#         inds_ik = inds(np.where(aye[adj]==j, i, aye[adj]), np.where(jay[adj]==j, i, jay[adj]))
#         points = np.hstack([points, inds(i,j), inds_ik])
        
#         S = S[np.argsort(points)]
#         points = np.sort(points)
    
#     # convert indices back to original
#     p1, p2 = inds.inv(points)
#     old_points = inds(idx[p1.astype(int)], idx[p2.astype(int)])
    
#     return S, old_points

# def compute_overlaps(old_dist, new_dist, target=None):
#     """
#     given a new arrow, compute its overlaps with co-incident arrows
    
#     dist (num_item, num_item) hamming matrix
#     new_dist (num_item, ) hamming distance of new object to old ones
#     target (0 <= int < num_item) which old item is the target (default closest)
#     sparse (bool) do we minimize overlap between non-composing arrows? (default true)
#     """
    
#     if target is None:
#         target = np.argmin(new_dist)
    
#     N = len(old_dist)
    
#     edge = []
#     ovlp = []
#     for s,t in combinations(range(N), 2):
#         if target == s: 
#             ovlp.append(np.max([0, (new_dist[s] + old_dist[s,t] - new_dist[t])/2]))
#         elif target == t:
#             ovlp.append(np.max([0, (new_dist[t] + old_dist[s,t] - new_dist[s])/2]))
#         else:
#             continue
    
#         edge.append(inds(s,t))
    
#     return np.array(ovlp), np.array(edge)


# def symbol_program(A_adj, A_dis, targ_length, ovlp, sparse=False,
#                    w=None, solver='highs', int_tol=1e-6):
#     """
#     Find a binary vector x, corresponding to an unknown edge, which fits the 
#     constraints of the known geometry and previously inferred symbols. Mostly 
#     a wrapper for a linear program solver.
    
#     A_adj $( N-1, S)$ symbols of known edges co-incident with target edge
#     A_dis $( (N-1)(N/2 - 1), S)$ symbols of edges disjoint with target edge
#     ovlp $(N-1, )$ overlap of target edge with co-indincident edges
#     weights $( (N-1)(N/2 - 1), )$ weight on each disjoint edge -- the weighted
#             sum of overlaps with disjoint edges will be maximized
    
#     returns 
    
#     """
    
#     Adj_unq, ids = np.unique(A_adj, axis=0, return_index=True)
    
#     n_adj, n_sym = Adj_unq.shape
#     n_dis = len(A_dis)
    
#     if w is None:
#         if sparse:
#             w = np.ones((1,n_dis))
#         else:
#             w = -np.ones((1,n_dis))
    
#     if n_dis > 0:
#         c = w@A_dis
#     else:
#         c = np.ones((1,n_sym))
    
#     prog = lp(c,
#               A_ub=np.ones((1,n_sym)), b_ub=targ_length, 
#               A_eq=Adj_unq, b_eq=ovlp[ids],
#               bounds=[0,1],
#               method=solver)
    
#     if prog.success:
#         if not np.all(np.abs(np.round(prog.x) - prog.x) <= int_tol):
#             raise Exception('Solution not integer, goddamnit!')
#         else:
#             return np.round(prog.x).astype(int)
#     else:
#         raise Exception('Linear program failed!')


# def combined_program(A_adj, A_dis, targ_length, ovlp, sparse=False,
#                      w=None, solver='highs', int_tol=1e-6):

    
    
#     A_unq, ix, num_s = np.unique(A_adj, axis=1, return_counts=True, return_inverse=True)
    
#     A_int = A_unq*num_s[None,:]
    
#     prog = lp(np.ones((1,len(num_s))), 
#               A_ub=num_s[None,:], b_ub=targ_length, 
#               A_eq=A_int, b_eq=ovlp,
#               bounds=[0,1],
#               method=solver)

    

# def choose_order(distances, sparse):
    
#     i0 = np.random.choice(len(distances))
#     order = [i0]
#     idx = np.zeros(len(distances))
#     idx[i0] = 1
    
#     while np.sum(idx) < len(distances):
        
#         score = (((-1)**sparse)*distances[idx>0,:]).min(0)
#         best = np.argmax(np.where(idx, -np.inf, score))
#         idx[best] = 1
#         order.append(best)
    
#     return np.array(order)

    
# ##### 

# # def compute_overlaps(old_dist, new_dist, target=None, sparse=True):
# #     """
# #     given a new arrow, compute its overlaps with all other arrows
    
# #     dist (num_item, num_item) hamming matrix
# #     new_dist (num_item, ) hamming distance of new object to old ones
# #     target (0 <= int < num_item) which old item is the target (default closest)
# #     sparse (bool) do we minimize overlap between non-composing arrows? (default true)
# #     """
    
# #     if target is None:
# #         target = np.argmin(new_dist)
    
# #     N = len(old_dist)
    
# #     arrow = []
# #     ovlp = []
# #     for s,t in combinations(range(N), 2):
# #         arrow.append(inds(s,t))
# #         if target == s: 
# #             ovlp.append((new_dist[s] + old_dist[s,t] - new_dist[t])/2)
# #         elif target == t:
# #             ovlp.append((new_dist[t] + old_dist[s,t] - new_dist[s])/2)
# #         else: 
# #             # we're computing the minimum overlap between the two arrows
# #             # which still satisfy the rules of composition 
            
# #             # in terms of the known overlaps between arrows: 
# #             # |(ij)*(kl)| = |(ij)*(il)| + |(ij)*(ik)| - 2 |(ij)*(il)*(ik)|
# #             # which i'm re-writing in terms of the distances
            
# #             ov = new_dist[target] + (new_dist[s] + new_dist[t] - old_dist[s,target] - old_dist[t,target])/2
# #             if sparse:
# #                 # upper bound:
# #                 # |(ij)*(il)*(ik)| =  min{(ij)*(il), (ij)*(ik), (il)*(ik)} 
# #                 ov -= np.min([new_dist[target] + new_dist[s] - old_dist[s,target], 
# #                               new_dist[target] + new_dist[t] - old_dist[t,target],
# #                               new_dist[s] + new_dist[t] - old_dist[s,t]])
# #             else:
# #                 # lower bound  
# #                 # |(ij)*(il)*(ik)| =  
# #                 ov -= np.max([0,np.max([new_dist[target] + new_dist[s] - old_dist[s,t] - old_dist[target,t],
# #                                         new_dist[t] + new_dist[s] - old_dist[s,target] - old_dist[target,t],
# #                                         new_dist[target] + new_dist[t] - old_dist[s,target] - old_dist[s,t]])])
            
# #             ovlp.append(ov)
        
# #     return np.array(arrow), np.array(ovlp)

# # def compute_overlaps(symb, pts, new_dist, target=None, sparse=True):
# #     """
# #     given a new arrow, compute its overlaps with all other arrows
    
# #     dist (num_item, num_item) hamming matrix
# #     new_dist (num_item, ) hamming distance of new object to old ones
# #     target (0 <= int < num_item) which old item is the target (default closest)
# #     sparse (bool) do we minimize overlap between non-composing arrows? (default true)
# #     """
    
# #     if target is None:
# #         target = np.argmin(new_dist)
    
# #     old_dist = symb.sum(1)
    
# #     N = len(new_dist)
    
# #     arrow = []
# #     ovlp = []
# #     for s,t in combinations(range(N), 2):
# #         arrow.append(inds(s,t))
# #         if target == s: 
# #             ovlp.append((new_dist[s] + old_dist[inds(s,t)] - new_dist[t])/2)
# #         elif target == t:
# #             ovlp.append((new_dist[t] + old_dist[inds(s,t)] - new_dist[s])/2)
# #         else: 
# #             # we're computing the minimum overlap between the two edges
# #             # which still satisfy the rules of composition 
            
# #             # first find all (known) incident edges to the unknown edge
# #             others = np.array([m for m in range(N) if m != target])
# #             inc = np.isin(pts, inds(target, others))
# #             known_ovlp = symb[inc]@np.squeeze(symb[pts==inds(s,t)]) # overlap of known edge
# #             unk_ovlp = (new_dist[target] + symb[inc].sum(1) - new_dist[others])/2
            
# #             if sparse: 
# #                 ov = np.max(known_ovlp + unk_ovlp - old_dist[inc])
# #             else:
# #                 ov = np.min([new_dist[target] - unk_ovlp + known_ovlp, 
# #                              old_dist[inds(s,t)] - known_ovlp + unk_ovlp])
            
# #             ovlp.append(ov)
                 
# #     return np.array(arrow), np.array(ovlp)

# def edge_symbols2(distances, sparse=True, weights=None):
#     """
#     to each edge drawn between a pair of points, assign a set of symbols which
#     represent the binary components that change between said points. 
    
#     distances (N, N) should be a hamming distance matrix: integral, symmetric,
#                     zero on the diagonal, obeys the triangle inequality.
#     """
    
#     N = len(distances)
    
#     # choose an order in which to add items
#     # idx = np.arange(N) # i don't think the order should matter ... oops it does
#     idx = choose_order(distances, sparse)
#     dist = np.round(distances).astype(int)[idx,:][:,idx] # now we can ignore idx
    
#     S = np.ones((1,dist[0,1])) # this will be size (N*(N-1)/2, S) 
#     points = inds(0, 1) # start and end point of edges
    
#     for i in range(2,N):
#     # for i in idx[2:4]:
#         # d_ik = dist[i,:i]
#         # choose an arrow from new item to any old item
#         # j = np.argmax(((-1)**sparse)*dist[i,:i])
#         j = np.argmin(dist[i,:i])
#         # j = 0
        
#         adj_ovlp, pts = compute_overlaps(dist[:i,:i], dist[i,:i], target=j)
        
#         aye, jay = inds.inv(points) # sorry about the names
#         adj = (aye==j)|(jay==j) # all co-incident edges
#         dis = ~adj
        
#         if np.sum(dis) > 0:
#             C, d = make_integer_system(S, dist[i,j])
            
#             d - C[:,adj]@adj_ovlp[np.argsort(pts)]
            
#             prog = lp(-((-1)**sparse)*np.ones(sum(dis)), 
#                       A_ub=C[:,dis], 
#                       b_ub=d - C[:,adj]@adj_ovlp[np.argsort(pts)],
#                       bounds=[0, dist[i,j]],
#                       method=solver)
            
#             dis_ovlp = prog.x
            
#             sym_ij, warn = solve_integer_system(np.vstack([S[dis], S[adj]]), 
#                                             np.concatenate([dis_ovlp, adj_ovlp]))
            
#         else:
#             dis_ovlp = []
#             sym_ij, warn = solve_integer_system(S, adj_ovlp)
        
        
#         # sym_ij = symbol_program(S[adj_idx], S[dis_idx], dist[i,j], adj_ovlp[np.argsort(pts)], 
#         #                         weights=weights, sparse=sparse)
        
#         resid = int(dist[i,j] - np.sum(sym_ij)) # create new symbols to make up the distance
#         if resid > 0:
#             sym_ij = np.append(sym_ij, np.ones(resid))
#             S = np.append(S, np.zeros((len(S), resid)), axis=1)
        
#         # assign symbols to the other edges using composition rules (i.e. addition mod 2)
#         sym_ik = np.mod(sym_ij + S[adj].squeeze(), 2)
        
#         # extend arrays and keep account of edge endpoints
#         S = np.vstack([S, sym_ij[None,:], sym_ik]) # sorry again
#         inds_ik = inds(np.where(aye[adj]==j, i, aye[adj]), np.where(jay[adj]==j, i, jay[adj]))
#         points = np.hstack([points, inds(i,j), inds_ik])
        
#         S = S[np.argsort(points)]
#         points = np.sort(points)
    
#     # convert indices back to original
#     p1, p2 = inds.inv(points)
#     old_points = inds(idx[p1.astype(int)], idx[p2.astype(int)])
    
#     return S, old_points

# # def infer_overlaps(S, adj, ):
# #     """
# #     given a new arrow, compute its overlaps with co-incident arrows
    
# #     dist (num_item, num_item) hamming matrix
# #     new_dist (num_item, ) hamming distance of new object to old ones
# #     target (0 <= int < num_item) which old item is the target (default closest)
# #     sparse (bool) do we minimize overlap between non-composing arrows? (default true)
# #     """
    
    
    
# #     prog = lp(c, 
# #               A_ub=np.ones((1,n_sym)), b_ub=targ_length, 
# #               A_eq=Adj_unq, b_eq=adj_ovlp[ids],
# #               bounds=[0,1],
# #               method=solver)
    
# #     return np.array(ovlp), np.array(edge)


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
    
# # def solve_integer_system():

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

# # def arrow_symbols(distances, sparse=True):
# #     """
# #     to each arrow drawn between a pair of points, assign a set of symbols which
# #     represent the binary components that change between said points. 
    
# #     distances (N, N) should be a hamming distance matrix: integer-valued, 
# #                     symmetric, with zero on the diagonal
# #     """
    
# #     dist = np.round(distances).astype(int)
# #     N = len(distances)
    
# #     # choose an order in which to add items
# #     idx = np.arange(N) # i don't think the order should matter ... 
# #     S = np.ones((1,dist[idx[0],idx[1]])) # this will be size (N*(N-1)/2, S) 
# #     points = inds(idx[0], idx[1]) # start and end point of arrows
    
# #     for i in idx[2:]:
# #     # for i in idx[2:4]:
# #         d_ik = dist[i,:i]
# #         # choose an arrow from new item to any old item
# #         j = np.argmin(d_ik)
        
# #         # compute overlaps with all other arrows
# #         # pts, ovlp = compute_overlaps(dist[:i,:i], d_ik, target=j, sparse=sparse)
# #         pts, ovlp = compute_overlaps(S, points, d_ik, target=j, sparse=sparse)
        
# #         # assign it symbols using as many existing symbols as possible
# #         sym_ij, warn = assign_symbols(S[np.argsort(points)], ovlp[np.argsort(pts)])
# #         if warn:
# #             print('warning!')
# #             break
# #         resid = int(dist[i,j] - np.sum(sym_ij)) # create new symbols to make up the distance
# #         if resid > 0:
# #             sym_ij = np.append(sym_ij, np.ones(resid))
# #             S = np.append(S, np.zeros((len(S), resid)), axis=1)
        
# #         # assign symbols to the other arrows using composition rules (Z/2)
# #         aye, jay = inds.inv(points) # sorry about the nomenclature
# #         these = (aye==j)|(jay==j)
# #         sym_ik = np.mod(sym_ij + S[these].squeeze(), 2)
        
# #         # extend arrays and keep account of arrow endpoints
# #         S = np.vstack([S, sym_ij[None,:], sym_ik]) # sorry again
# #         inds_ik = inds(np.where(aye[these]==j, i, aye[these]), np.where(jay[these]==j, i, jay[these]))
# #         points = np.hstack([points, inds(i,j), inds_ik])
    
# #     return S, points

    
# def color_points(symbols, edges):
#     """
#     Given the set of arrow-symbols, color the start and end points accordingly. 
#     Does this by finding the coloring of the bipartite graph associated with each
#     symbol's arrows. 
    
#     """
    
#     # ideally we can merge symbols which always occur together, but i'm feeling
#     # lazy right now and it seems like something that should be done carefully
    
#     # N = len(np.unique(inds.inv(edges)))
#     pts = np.unique(inds.inv(edges)).astype(int)
    
#     # the arrows which share a given symbol cannot form an odd cycle, which 
#     # means they form a bipartite graph (i.e. a binary coloring)
#     vecs = []
#     # sets = []
#     for i in range(symbols.shape[1]):
#         e = np.stack(inds.inv(edges[symbols[:,i]>0])).astype(int).T.tolist()
        
#         G = nx.Graph()
#         G.add_edges_from(e)
#         pos, neg = nx.bipartite.sets(G)
#         if len(pos)<=len(neg):
#             # sets.append(pos)
#             vecs.append(1*np.isin(pts,list(pos)))
#         else:
#             # sets.append(neg)
#             vecs.append(1*np.isin(pts,list(neg)))
#         # vecs.append(1*np.isin(np.arange(N),list(neg)))

#     return np.array(vecs), pts
#     # return np.unique(vecs, axis=0), np.unique(sets)

# # def chunk_symbols(symbols):
# #     """
# #     given the symbols associated with each point, group mutually-exclusive
# #     symbols into multi-valued variables, defined potentially only on subsets
# #     of points.
# #     """
    

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


def gf2elim(M):

    m,n = M.shape

    i=0
    j=0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) +i

        # swap rows
        #M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp

        aijn = M[i, j:]

        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

        col[i] = 0 #avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        i += 1
        j +=1

    return M


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
    

# def BDF(K, orig=0, thresh=1e-6, sparse=True):
#     """
#     Binary distance factorization (working name) of K. 
    
#     """
    
#     ### center around arbitrary (?) point
#     K_o = K[orig,orig]- K[[orig],:] - K[:,[orig]] + K
    
#     N = len(K)
#     sgn = ((-1)**sparse)
    
#     idx = np.argsort(sgn*np.diag(K_o))
#     i0 = idx[idx != orig][0]
    
#     S_unq = np.array([[0],[1]])
#     num_s = np.array([K_o[i0,i0]])
    
#     included = [orig, i0]
#     remaining = np.setdiff1d(idx, [orig, i0]).tolist()
    
#     while len(remaining) > 0:

#         ### find next item to fit 
#         diffs = []
#         mins = []
        
#         # found = False
#         for j,o in enumerate(included):
#             A = np.mod(S_unq+S_unq[[j]], 2)*num_s[None,:]
            
#             this_K = K[o,o]- K[[o],:] - K[:,[o]] + K
            
#             bs = this_K[included,:][:,remaining]
#             maxlen = this_K[remaining, remaining]
            
#             o_diffs = []
#             o_mins = []
            
#             for i in range(len(remaining)):
                
#                 prog = lp(-1*sgn*num_s,
#                           A_ub=num_s[None,:], b_ub=maxlen[i],
#                           A_eq=A, b_eq=bs[:,i],
#                           bounds=[0,1],
#                           method='highs')
        
#                 prog2 = lp(sgn*num_s,
#                             A_ub=num_s[None,:], b_ub=maxlen[i],
#                             A_eq=A, b_eq=bs[:,i],
#                             bounds=[0,1],
#                             method='highs')
#                 # prog = lp(-1*sgn*np.ones(len(c)),
#                 #           A_ub=np.ones((1,len(c))), b_ub=maxlen[i],
#                 #           A_eq=S_unq, b_eq=bs[:,i],
#                 #           bounds=[(0,n) for n in num_s],
#                 #           method='highs')
                
#                 # succ.append(prog.success)
                
#                 if prog.success and prog2.success:
#                     o_diffs.append(np.abs((prog2.x - prog.x)*num_s).sum())
#                 else:
#                     o_diffs.append(np.nan)
                
        
#                 if prog.success:
#                     o_mins.append(prog.x@num_s / maxlen[i])
#                 else:
#                     # xs.append( prog.x*num_s )
#                     o_mins.append(np.nan)
            
#             diffs.append(o_diffs)
#             mins.append(o_mins)
                
#         diffs = np.array(diffs)
#         mins = np.array(mins)
        
#         valid = ~(np.isnan(diffs)+np.isnan(mins))
        
#         if np.any(valid):
            
#             min_diff = diffs == np.nanmin(diffs[valid])
#             if sparse:
#                 these_o, these_i = np.where(min_diff&(mins == np.nanmax(mins[min_diff])))
#             else:
#                 these_o, these_i = np.where(min_diff&(mins == np.nanmin(mins[min_diff])))
                
#             id_best = these_i[0]
#             orig_best = these_o[0]
#             o = included[orig_best]
                
#             this_i = remaining[id_best]
            
#             K_o = K[o,o]- K[[o],:] - K[:,[o]] + K
            
#             ### new features
#             prog = lp(-1*sgn*np.ones(len(num_s)),
#                       A_ub=np.ones((1,len(num_s))), b_ub=K_o[this_i,this_i],
#                       A_eq=S_unq, b_eq=K_o[included,this_i],
#                       bounds=[(0,n) for n in num_s],
#                       method='highs',
#                       integrality=1)
            
#             x_int = np.round(prog.x).astype(int)
            
#             # print(x_int)
#             # print(S_unq)
            
#             S_unq = np.mod(S_unq+S_unq[[orig_best]], 2)
            
#             # print(S_unq)
            
            
#             # ### transform back to reference basis
#             # x_int = np.abs(S_unq[orig_best]*num_s - x_int)
            
            
#         else:
#             raise Exception('Inconsistency at %d items'%len(included))
        
        
        
#         # id_best = next_item(-1*sgn*num_s, 
#         #                     S_unq, num_s, 
#         #                     K_o[included,:][:,remaining],
#         #                     K_o[remaining,remaining],
#         #                     sparse=sparse)
#         # x_int = next_item(-1*sgn*num_s, 
#         #                     S_unq, num_s, 
#         #                     K, included, remaining,
#         #                     sparse=sparse)
        
#         # if id_best is None:
#         #     raise Exception('Inconsistency at %d items'%len(included))
            
#         # this_i = remaining[id_best]
        
#         # ### new features
#         # prog = lp(-1*sgn*np.ones(len(num_s)),
#         #           A_ub=np.ones((1,len(num_s))), b_ub=K_o[this_i,this_i],
#         #           A_eq=S_unq, b_eq=K_o[included,this_i],
#         #           bounds=[(0,n) for n in num_s],
#         #           method='highs',
#         #           integrality=1)
        
#         # x_int = np.round(prog.x).astype(int)
        
#         ### Split clusters to account for fractional membership
#         new_S_unq, new_num_s = update(S_unq, num_s, x_int, K_o[this_i,this_i])
        
#         S_unq = new_S_unq[:, np.argsort(-new_num_s)]
#         num_s = new_num_s[np.argsort(-new_num_s)]
        
#         ### Discard clusters (find a good way to do this)
#         # S_unq, num_s = cull(S_unq, num_s, thresh=thresh)
        
#         included.append(this_i)
#         remaining.remove(this_i)
    
#     ### fill in remaining difference vectors 
#     S_unq = S_unq[np.argsort(included)]
#     S_full = np.vstack([S_unq]+[np.mod(S_unq[(i+1):,:]+S_unq[[i],:],2) for i in range(N-1)])
    
#     pt_id = np.append(orig, np.sort(included))
#     ix = inds(np.concatenate([np.ones(N-n-1)*i for n,i in enumerate(pt_id)]), 
#               np.concatenate([np.sort(included)[n:] for n in range(N)]))
    
#     ### convert to 'canonical' form
#     vecs, pts = color_points(S_full, ix)
    
#     return vecs, num_s



def intBDF(C, thresh=1e-6, in_cut=False, sparse=True, num_samp=5000):
    """
    Binary distance factorization (working name) of K. 
    
    """
    
    N = len(C)
    inds = util.LexOrder()
    
    ## "Project" into cut polytope, if it isn't already there
    if not in_cut:
        samps = np.sign(np.random.multivariate_normal(np.zeros(N), C, size=num_samp))
        K = samps.T@samps
    else:
        K = C
    
    
    d = 1 - K/num_samp
    
    orig = np.argmin(np.sum(d, axis=0)) 
    
    ### center around arbitrary (?) point
    K_o = K[orig,orig]- K[[orig],:] - K[:,[orig]] + K
    
    sgn = ((-1)**sparse)
    
    idx = np.arange(N)
    # i0 = idx[idx != orig][0]
    first = np.setdiff1d(idx, [orig])[np.argmin(d[orig,:][np.setdiff1d(idx, [orig])])]
    
    S_unq = np.array([[0],[1]])
    num_s = np.array([K_o[first,first]])
    
    included = [orig, first]
    remaining = np.setdiff1d(idx, [orig, first]).tolist()
    
    while len(remaining) > 0:

        ## Most "explainable" item <=> closest item
        this_i = remaining[np.argmin(np.sum(d[included,:][:,remaining], axis=0))]
        
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
def BDF(K, thresh=1e-6, in_cut=False, sparse=True, num_samp=5000, zero_tol=1e-6):
    """
    Binary distance factorization (working name) of K. Using the non-integer 
    formulation.
    """
    
    N = len(K)
    inds = util.LexOrder()
    
    ## "Project" into cut polytope, if it isn't already there
    if not in_cut:
        samps = np.sign(np.random.multivariate_normal(np.zeros(N), K, size=num_samp))
        C_ = samps.T@samps/num_samp
    else:
        C_ = K
    
    d = 1 - C_
    
    # alpha = np.sum(d)/np.sum(d**2)   # distance scale which minimizes correlations
    alpha = (1+np.max(d))/(N/2) # unproven: this is largest alpha which returns sparsest solution
    C = 1 - alpha*d
    
    orig = np.argmin(np.sum(d, axis=0)) 
    
    ### center around arbitrary (?) point
    K_o = (C[orig,orig]- C[[orig],:] - C[:,[orig]] + C)/4
    
    sgn = (-1)**(sparse + 1)
    
    idx = np.arange(N)
    first = np.setdiff1d(idx, [orig])[np.argmin(d[orig,:][np.setdiff1d(idx, [orig])])]
    
    B = np.array([[1,1],[0,1]])
    pi = np.array([1-K_o[first,first], K_o[first,first]])
    
    included = [first]
    remaining = np.setdiff1d(idx, [orig, first]).tolist()
    
    for n in tqdm(range(2,N)):

        ## Most "explainable" item <=> closest item
        this_i = remaining[np.argmin(np.sum(d[included,:][:,remaining], axis=0))]
        
        ### new features
        prog = lp(sgn*pi,
                  A_eq=B@np.diag(pi), b_eq=K_o[[this_i] + included,this_i],
                  bounds=[0,1],
                  method='highs')
        
        if prog.success:
            x = prog.x
        else:
            raise Exception('Inconsistency at %d items'%len(included))
        
        ## Split clusters as necessary 
        new_pi = []
        new_b = []
        for i,x_i in enumerate(x):
            
            if np.abs(x_i**2 - x_i) <= zero_tol:
                new_b.append(int(np.round(x_i)))
                new_pi.append(pi[i])
            
            else:
                
                new_b += [0, 1]
                new_pi += [pi[i]*(1-x_i), pi[i]*x_i]
        
        B = np.vstack([np.repeat(B, 1 + (np.abs(x**2 - x) >= zero_tol), axis=1), new_b])
        pi = np.array(new_pi)
        
        ### Discard clusters (find a good way to do this)
        # S_unq, num_s = cull(S_unq, num_s, thresh=thresh)
        
        included.append(this_i)
        remaining.remove(this_i)
    
    ## Fix the matrix
    B[0] = 0
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

def SDF(K, zero_tol=1e-10, in_cut=False, thresh=1e-5, num_samp=5000):
    """
    Sign distance factorization
    """
    
    P = len(K)
    
    ## "Project" into cut polytope, if it isn't already there
    if not in_cut:
        samps = np.sign(np.random.multivariate_normal(np.zeros(P), K, size=num_samp))
        C_ = samps.T@samps/num_samp
    else:
        C_ = K
    
    ## Regularize to minimum-norm correlation matrix
    ## this should be re-visited, doesn't work well for e.g. identity matrix
    d = 1 - C_   
    # alpha = np.sum(d)/np.sum(d**2)   # alpha which minimizes correlations
    alpha = (1+np.max(d))/(N/2) # unproven: this is largest alpha which returns sparsest solution
    C = 1 - alpha*d
    
    ## Fit features
    S = np.ones((1,1), dtype=int) # signs
    pi = np.ones(1)    # probabilities
    
    first = np.argmin(np.sum(d, axis=0)) 
    
    included = [first]
    remaining = np.setdiff1d(range(P), first).tolist()
    for n in tqdm(range(1,P)):
        
        ## Most "explainable" item <=> closest item
        new = remaining[np.argmin(np.sum(d[included,:][:,remaining], axis=0))]
        
        if n > 2:
            
            ## Average gradient of the nth moment
            w = np.mean([S[[these]].prod(0) for these in combinations(range(n), n-np.mod(n+1,1))], axis=0)

            prog = lp(w*pi, A_eq=S@np.diag(pi), b_eq=C[included, new], bounds=[-1,1])
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

# all_S = []
# all_included = []
# # for orig in range(N):
# for orig in [0]:
    
#     # i0 = np.argmax(np.diag(clean_K))
    
#     idx = np.argsort(-np.diag(K_o[orig]))
#     i0 = idx[0]
    
#     # S = np.ones((1,dist[orig,i0]))
    
#     # S_frame = [np.ones((1,dist[orig,i])) for i in range(N)]
#     S_unq = np.ones((1,1))
#     num_s = np.array([K_o[orig][i0,i0]])
#     S_int = S_unq*num_s[None,:]
    
#     included = [i0]
#     remaining = np.setdiff1d(idx, [orig, i0]).tolist()
    
#     while len(remaining) > 0:

#         diffs = []
#         mins = []
#         xs = []
#         succ = []
#         # found = False
#         for i in remaining:
            
#             prog = lp(num_s,
#                       A_ub=num_s[None,:], b_ub=K_o[orig][i,i],
#                       A_eq=S_int, b_eq=K_o[orig][included,i],
#                       bounds=[0,1],
#                       method='highs')

#             # prog = lp(np.ones(len(num_s)),
#             #           A_ub=np.ones((1,len(num_s))), b_ub=K_o[orig][i,i],
#             #           A_eq=S_unq, b_eq=K_o[orig][included,i],
#             #           bounds=[(0,n) for n in num_s],
#             #           method='highs',
#             #           integrality=1)

#             prog2 = lp(-num_s,
#                         A_ub=num_s[None,:], b_ub=K_o[orig][i,i],
#                         A_eq=S_int, b_eq=K_o[orig][included,i],
#                         bounds=[0,1],
#                         method='highs')
            
#             succ.append(prog.success)
            
#             if prog.success and prog2.success:
#                 diffs.append(np.abs((prog2.x - prog.x)*num_s).sum())
#             else:
#                 diffs.append(np.nan)
            

#             if prog.success:
#                 xs.append( prog.x*num_s )
#                 mins.append(prog.x@num_s / K_o[orig][i,i])
#             else:
#                 # xs.append( prog.x*num_s )
#                 mins.append(np.nan)
                
#         diffs = np.array(diffs)
#         mins = np.array(mins)
        
#         valid = ~(np.isnan(diffs)+np.isnan(mins))
#         if sum(valid) > 0:
#             print(len(included))
#         else:
#             print('done')

#             break
        
#         min_diff = diffs == np.nanmin(diffs[valid])
#         these_i = np.where(min_diff&(mins == np.nanmax(mins[min_diff])))[0]
#         this_i = remaining[these_i[0]]
        
#         prog = lp(np.ones(len(num_s)),
#                   A_ub=np.ones((1,len(num_s))), b_ub=K_o[orig][this_i,this_i],
#                   A_eq=S_unq, b_eq=K_o[orig][included,this_i],
#                   bounds=[(0,n) for n in num_s],
#                   method='highs',
#                   integrality=1)
        
        
#         x_int = np.round(prog.x).astype(int)
        
#         ### split clusters to account for fractional membership
#         new_S_unq, new_num_s = update(S_unq, num_s, x_int, K_o[orig][this_i,this_i])
        
#         S_unq = new_S_unq[:, np.argsort(-new_num_s)]
#         num_s = new_num_s[np.argsort(-new_num_s)]
        
#         S_unq = S_unq[:,num_s > 5]
#         num_s = num_s[num_s > 5]
        
#         S_int = S_unq*num_s[None,:]
        
#         included.append(this_i)
#         remaining.remove(this_i)


# S_unq = S_unq[np.argsort(included)]
# S_full = np.vstack([S_unq]+[np.mod(S_unq[(i+1):,:]+S_unq[[i],:],2) for i in range(N-1)])

# ix = inds(np.concatenate([np.ones(N-i-1)*i for i in range(N)]), 
#           np.concatenate([np.arange(i+1,N) for i in range(N)]))

# vecs, pts = color_points(S_full, ix)


# S_pred = []
# pts = []
# for i, (items, this_S) in enumerate(zip(all_included, all_S)):
    
#     aye = np.repeat(range(len(items)), len(items))
#     jay = np.tile(range(len(items)), len(items))
    
#     ix = inds(np.array(items)[aye], np.array(items)[jay])
#     _, idxs = np.unique(ix, return_index=True)
#     these = idxs[ix[idxs] > 0]
    
#     pts.append( np.concatenate([inds(i,np.array(items)), ix[these]]) )
#     S_pred.append ( np.vstack([this_S, np.mod(this_S[aye[these]] + this_S[jay[these]], 2)]) )
    
    
    
    