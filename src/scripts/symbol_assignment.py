

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

class LexOrder:
    """
    Lexicographic order of unordered pairs, (i,j) with j < i
    """
    def __init__(self):
        return 
    
    def __call__(self,i,j):
        n = np.where(i>j, i*(i-1)/2 + j, j*(j-1)/2 + i)
        return np.where(i==j, -1, n)
    
    def inv(self, n):
        i = np.floor((1 + np.sqrt(1+8*n))/2)
        j = n - i*(i-1)/2
        return i, j

inds = LexOrder()

def compute_overlaps(dist, new_dist, target=None):
    """
    given a new arrow, compute its overlaps with all other arrows
    
    dist (num_item, num_item) hamming matrix
    new_dist (num_item, ) hamming distance of new object to old ones
    target (0 <= int < num_item) which old item is the target (default closest)
    """
    
    if target is None:
        target = np.argmin(new_dist)
    
    N = len(dist)
    
    arrow = []
    ovlp = []
    for s,t in combinations(range(N), 2):
        arrow.append(inds(s,t))
        if target == s: 
            ovlp.append((new_dist[s] + dist[s,t] - new_dist[t])/2)
        elif target == t:
            ovlp.append((new_dist[t] + dist[s,t] - new_dist[s])/2)
        else: 
            # we're computing the minimum overlap between the two arrows
            # which still satisfy the rules of composition ... 
            # this is after lots of reductions and term cancellations
            
            d_sub = np.append(dist[[s,t,target],:][:,[s,t,target]], new_dist[[s,t,target]][:,None], axis=1)
            d_sub = np.append(d_sub, np.append(new_dist[[s,t,target]],0)[None,:], axis=0)
            
            new_idx = np.argsort(-d_sub[-1,:])[:-1]
            targ_idx = np.argsort(-d_sub[2,:])[:-1]
        
            ovlp.append(new_dist[target] - d_sub[-1,new_idx]@[-1,1,1]/2 - d_sub[2, targ_idx]@[-1,1,1]/2)
        
    return np.array(arrow), np.array(ovlp)

def assign_symbols(symb, cap):
    """
    for a new item, assign to it existing symbols such that the overlaps with
    all the existing items are satisfied.
    
    solves this as a special type of generalized flow network, algorithm 
    inspired by Ford-Fulkerson and modified to the structure of this network.
        (speficially, FF finds a residual path and adds flow along it, until 
         there are no residual paths. fortunately, the network structure makes
         it easy to find residual paths even though this isnt a regular network)
    
    symb (num_item, num_symbol) {0,1}-matrix
    ovlp (num_item, ) overlaps of new item with each existing item
    """
    
    N,S = symb.shape
    
    flow = np.zeros(S)
    resid = cap - symb@flow
    max_cap = np.nanmax(np.where(symb,symb*cap[:,None],np.nan), axis=0)
    good_syms = max_cap == np.max(cap) # flow should go through max capacity nodes
    
    warning = False
    while np.any(resid): # this should always converge
        min_cap = np.nanmin(np.where(symb,symb*resid[:,None],np.nan), axis=0)
        best = np.argmax(min_cap*good_syms*(1-flow))
        if (flow[best] > 0) or (min_cap[best] < 1): # i feel like this shouldn't ever happen but not sure
            warning = True # this means that the overlaps aren't satisfied
            break
        flow[best] = 1
        resid = cap - symb@flow
    
    return flow

def arrow_symbols(distances):
    """
    to each arrow drawn between a pair of points, assign a set of symbols which
    represent the binary components that change between said points. 
    
    distances (N, N) should be a hamming distance matrix: integer-valued, 
                    symmetric, with zero on the diagonal
    """
    
    N = len(distances)
    
    # choose an order in which to add items
    idx = np.arange(N) # i don't think the order should matter ... 
    S = np.ones((1,distances[idx[0],idx[1]])) # this will be size (N*(N-1)/2, S) 
    points = inds(idx[0], idx[1]) # start and end point of arrows
    
    for i in idx[2:]:
        d_ik = distances[i,:i]
        # choose an arrow from new item to any old item
        j = np.argmin(d_ik)
        
        # compute overlaps with all other arrows
        pts, ovlp = compute_overlaps(distances[:i,:i], d_ik, target=j)
        
        # assign it symbols using as many existing symbols as possible
        sym_ij = assign_symbols(S[np.argsort(points)], ovlp[np.argsort(pts)])
        resid = int(distances[i,j] - np.sum(sym_ij)) # create new symbols to make up the distance
        if resid > 0:
            sym_ij = np.append(sym_ij, np.ones(resid))
            S = np.append(S, np.zeros((len(S), resid)), axis=1)
        
        # assign symbols to the other arrows using composition rules (Z/2)
        aye, jay = inds.inv(points) # sorry about the nomenclature
        these = (aye==j)|(jay==j)
        sym_ik = np.mod(sym_ij + S[these].squeeze(), 2)
        
        # extend arrays and keep account of arrow endpoints
        S = np.vstack([S, sym_ij[None,:], sym_ik]) # sorry again
        inds_ik = inds(np.where(aye[these]==j, i, aye[these]), np.where(jay[these]==j, i, jay[these]))
        points = np.hstack([points, inds(i,j), inds_ik])
    
    return S, points

def chunk_symbols(symb):
    
    C = util.cosine_sim(np.round(symb), np.round(symb))
    
    
def color_points(chunks, symb, points):
    """
    given all the arrows that constitute a variable, find a proper coloring of
    the induced graph. these are the values of the variable.
    
    """
    
    print("blab")
    

    
## these functions are for a generic max flow approach, which won't work unfortunately
def make_flow_network(symb, new_item):
    """
    given the known symbols for a set of items, and the overlaps of each of 
    those items with a new item, return the relevant flow network 
    
    symb (num_item, num_symbol) {0,1}-matrix
    new_item (num_item + 1, ) new column to append to kern
    """
    
    N, S = symb.shape
    
    nodes = np.arange(N+S+2)
    
    edges = []
    caps = []
    gains = []
    # first from the source to the symbols
    edges += [(0,i) for i in range(1,S+1)]
    caps += [1]*S
    gains += symb.sum(0).tolist()
    
    for s in range(S): # then from symbols to items
        edges += [(s+1,S+1+i) for i in np.where(symb[:,s])[0]]
        caps += [1]*np.sum(symb[:,s])
        gains += [1]*np.sum(symb[:,s])
    
    # then from items to the sink
    edges += [(i,N+S+1) for i in range(S+1,S+1+N)]
    caps += new_item[:-1].tolist()
    gains += [1]*N
    
    return nodes, edges, np.array(caps), np.array(gains)


def flow_conservation_matrix(nodes, edges, gains):
    """
    convert the flow network into a linear equality constraint for an LP
    """
    
    # find all nodes which aren't the source or sink
    eligible = [np.all(np.isin(edges, i).sum(0)>0) for i in nodes]
    
    M = []
    for i,n in enumerate(nodes):
        if eligible[i]:
            M.append(gains*(np.isin(edges, i)*[[1,-1]]).sum(1))
    
    return np.array(M) 



#%%











