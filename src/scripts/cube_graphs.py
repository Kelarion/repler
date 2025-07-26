CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations, product
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

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import cvxpy as cvx

# my code
import util
import df_util
import pt_util
import bae
import bae_models
import bae_util
import bae_search
import plotting as tpl

#%%

def random_isometric_subgraph(G):
    """
    Find a subset of columns of S 
    """

def is_partial_cube(G):
    """
    Brute force algorithm for checking if G is a partial cube
    """
    
    if not nx.is_bipartite(G) or not nx.is_connected(G):
        return False
    
    S = []
    dw = []
    for e in G.edges:
        edw = []
        for f in G.edges:
            dxu = nx.shortest_path_length(G, e[0], f[0])
            dyv = nx.shortest_path_length(G, e[1], f[1])
            dxv = nx.shortest_path_length(G, e[0], f[1])
            dyu = nx.shortest_path_length(G, e[1], f[0])
            
            edw.append( dxu + dyv != dxv + dyu )
        dw.append(edw)
            
    DW = np.array(dw)
    
    DW2 = DW@DW
    return np.all(DW[DW2>0])
    
def equiv_classes(G):
    
    N = len(G.nodes)
    E = len(G.edges)
    
    # which_class = -1*np.ones(E)
    S = np.zeros((N,E))
    
    k = 0
    for e in G.edges:
        if np.abs(S[e[0]]-S[e[1]]).max() > 0:
            continue
        
        for n in G.nodes:
            dxu = nx.shortest_path_length(G, e[0], n)
            dxv = nx.shortest_path_length(G, e[1], n)
                        
            S[n,k] = 1*(dxu<dxv)
        
        k += 1
    
    return S[:,:k]

def contractions(G):
    """
    All DW contractions 
    """
    if not is_partial_cube(G):
        return []
    
    S = equiv_classes(G)
    E = np.array(G.edges)
    elab = np.argmax(np.abs(S[E[:,0]] - S[E[:,1]]), axis=1)
    
    cons = []
    for i in range(S.shape[1]):
        G_ = G.copy()
        for j in np.where(elab==i)[0]:
            G_ = nx.contracted_edge(G_, E[j], self_loops=False)        
        
        match = False
        for Gref in cons:
            if nx.is_isomorphic(G_, Gref):
                match = True
                break
            
        if not match:
            cons.append(G_)
    
    return cons

def is_isometric(S, G):
    
    if len(S) != len(G.nodes):
        return False
    if not is_partial_cube(G):
        return False
    
    Z = equiv_classes(G)
    if Z.shape[1] != S.shape[1]:
        return False
    
    return df_util.permham(S, Z).sum() == 0

# def is_simplex(G):
    
#     if not is_partial_cube(G):
#         return False
    
#     S = equiv_classes(G)
#     S = (S + S[[0]])%2
    
#     out = True
#     for s in S:        

def fit_J(S):
    
    n, k = S.shape
    aye, jay = np.triu_indices(k)
    
    allS = util.F2(k)
    allS = allS[df_util.minham(allS.T, S.T)>0]
    
    if len(allS) == 0:
        return np.zeros((k,k))
    
    ## vec'd matrices
    allS_ = util.outers(allS.T)[:,aye,jay]
    S_ = util.outers(S.T)[:,aye,jay]
    
    c = np.ones(S_.shape[1])
    bub = -np.ones(len(allS))
    beq = np.zeros(len(S))
    
    sol = lp(c, A_ub=-allS_, b_ub=bub, A_eq=S_, b_eq=beq, bounds=(None,None))
    
    if sol.success:
        J = np.zeros((k,k))
        J[aye,jay] = sol.x
        J += J.T
        
        return J
    else:
        return None

def fixed_points(J):
    """
    Returns a set of fixed points if J is a valid matrix, None otherwise
    """
    
    k = len(J)
    allS = util.F2(k)
    
    val = util.qform(J, allS).squeeze()
    if np.any(val < -1e-6):
        return None 
    
    return allS[val < 1e-6]

#%% Get all partial cubes of dimension n

ns = [2,3,4]

graphs = []
for n in ns:
    S = util.F2(n)
    
    # graphs = []
    # is_iso = []
    # is_cube = []
    for subset in tqdm(util.F2(2**n)[1:]):
        
        deez = S[subset>0]
        
        G = nx.Graph(util.hamming(deez.T)==1)
        if not nx.is_connected(G):
            continue 
        
        match = False
        for Gref in graphs:
            if nx.is_isomorphic(G, Gref):
                match = True
                break
                
        if not match:
            if is_partial_cube(G) and (equiv_classes(G).shape[1] == n):    
                graphs.append(G)
    
            # is_cube.append(is_partial_cube(G))
            # # is_iso.append(is_isometric(deez, G))
            # is_iso.append(equiv_classes(G).shape[1] == n)
            
    # is_iso = np.array(is_iso)
    # is_cube = np.array(is_cube)
    
    # all_n_cubes = [graphs[i] for i in np.where(is_cube*is_iso)[0]]

#%% Contraction digraph

C = []
for i,G in enumerate(graphs):
    
    subs = contractions(G)
    for j,G_ in enumerate(graphs):
        
        if np.any([nx.is_isomorphic(G_, g) for g in subs]):
            C.append((i,j))

iscon = nx.DiGraph()
iscon.add_edges_from(C)

#%% Plot the connected graphs

ng = np.sum(is_cube)

rows = int(np.floor(np.sqrt(ng)))
cols = int(np.ceil(ng/rows))

clr = ['r', 'b']
for i,g in enumerate(np.where(is_cube)[0]):
    plt.subplot(rows, cols, i+1)
    nx.draw(graphs[g], node_color=clr[1*is_iso[g]], node_size=15)

#%%

is_rep = []
is_reg = []
is_flat = []
rep_graphs = []
sup_graphs = []
jays = []
pinvs = []
for G in graphs:
    
    S = equiv_classes(G)
    S = (S+S[[0]])%2
    n = S.shape[1]
    J = fit_J(S)
    
    if J is not None:
        is_rep.append(True)
        # jays.append(J)
        rep_graphs.append(G)
        Jgraph = nx.Graph(J - np.diag(np.diag(J)/2))
        sup_graphs.append(Jgraph)
        jays.append(J)
        
        I = df_util.inc(S)
        k = np.abs(I).sum(0)/2
        diff = np.abs((S-S.mean(0))@I.T - I@(S-S.mean(0)).T).sum()
        is_flat.append((len(np.unique(k)) == 1)*(diff < 1e-6))
        # if len(np.unique(card)) > 1:
            # is_reg.append(False)
        # else:
            # k = card[0]
        # R = np.diag(1/np.sqrt(k))@df_util.inc(S).T@df_util.inc(S)@np.diag(1/np.sqrt(k))
        # R = la.pinv(np.diag(1/np.sqrt(k))@(S-S.mean(0)).T@(S-S.mean(0))@np.diag(1/np.sqrt(k)))
        R = la.pinv((S-S.mean(0)).T@(S-S.mean(0)))
        R = (R > 1e-6) - 1*(R < -1e-6) + np.eye(n)
        r = 1*(R@S.mean(0)>0.5)
        # r = R@S.mean(0)
        # J_ = np.round(R - 2*np.diag(r))
        J_ = R - 2*np.diag(r)
        pinvs.append(J_)
        FP = fixed_points(J_)
        if FP is not None:
            G_ = nx.Graph(util.hamming(FP.T)==1)
            is_reg.append(nx.is_isomorphic(G, G_))
        else:
            is_reg.append(False)
    else:
        is_rep.append(False)

#%% Plot the partial cubes and highlight representable ones

ng = len(graphs)

rows = int(np.floor(np.sqrt(ng)))
cols = int(np.ceil(ng/rows))

clr = ['r', 'b']
sty = ['--', '-']
for i in range(ng):
    plt.subplot(rows, cols, i+1)
    
    S = equiv_classes(graphs[i])
    E = np.array(graphs[i].edges)
    
    elab = np.argmax(np.abs(S[E[:,0]] - S[E[:,1]]), axis=1)
    
    pos = graphviz_layout(graphs[i])
    nx.draw(graphs[i], #style=sty[1*is_rep[i]],
            edge_color=elab, node_size=15,
            node_color='k',
            width=2, alpha=is_rep[i]*0.6 + 0.4)

#%% Plot representable cubes highlighting regular ones

ng = len(sup_graphs)

rows = int(np.floor(np.sqrt(ng)))
cols = int(np.ceil(ng/rows))

clr = ['r', 'b']
sty = ['--', '-']
for i in range(ng):
    plt.subplot(rows, cols, i+1)
    
    S = equiv_classes(rep_graphs[i])
    E = np.array(rep_graphs[i].edges)
    
    elab = np.argmax(np.abs(S[E[:,0]] - S[E[:,1]]), axis=1)
    
    pos = graphviz_layout(rep_graphs[i])
    nx.draw(rep_graphs[i], #style=sty[1*is_rep[i]],
            edge_color=elab, node_size=15,
            node_color='k',
            width=2, alpha=is_reg[i]*0.6 + 0.4)


#%% Find J subgraphs

C = []
lifts = []
for i,thisJ in enumerate(sup_graphs):
    
    for k in thisJ.nodes:
        deez = util.rangediff(len(thisJ.nodes), [k])
        G_ = thisJ.subgraph(deez)
        for j,thatJ in enumerate(sup_graphs):
            if nx.is_isomorphic(G_, thatJ):
                C.append((j,i))

islifting = nx.DiGraph()
islifting.add_edges_from(C)

#%%

ng = len(sup_graphs)

rows = int(np.floor(np.sqrt(ng)))
cols = int(np.ceil(ng/rows))

ecols = ['r', 'b']
for i,jay in enumerate(sup_graphs):
    
    plt.subplot(rows, cols, i+1)
    
    pos = graphviz_layout(jay)
    if len(jay.edges) > 0:
        edges,weights = zip(*nx.get_edge_attributes(jay,'weight').items())
        edge_col = [ecols[int(this)] for this in (np.array(weights)+1)//2]
        
        nx.draw(jay, pos=pos, node_size=15, cmap='bwr', 
                edge_color=edge_col, edgelist=edges, width=2)
    else:
        nx.draw(jay, pos=pos, node_size=15, cmap='bwr')

#%% Draw descendants

root = 3

deez = list(nx.descendants_at_distance(islifting, root, 1))

ng = len(deez)

rows = int(np.floor(np.sqrt(ng)))
cols = int(np.ceil(ng/rows))

estls = ['-', '--']
ecols = ['r', 'b']
for i,i_ in enumerate(deez):
    
    rep = rep_graphs[i_]
    
    plt.subplot(rows, cols, i+1)
    
    pos = graphviz_layout(rep)
    S = equiv_classes(rep)
    E = np.array(rep.edges)
    
    elab = np.argmax(np.abs(S[E[:,0]] - S[E[:,1]]), axis=1)
    
    pos = graphviz_layout(rep)
    nx.draw(rep, #style=sty[1*is_rep[i]],
            edge_color=elab, node_size=15,
            node_color='k',
            width=2)

#%%

plt.figure()

for i,i_ in enumerate(deez):
    
    sup = sup_graphs[i_]
    
    GM = nx.isomorphism.GraphMatcher(sup, sup_graphs[root])
    
    _ = GM.subgraph_is_isomorphic()
    oldnodes = np.isin(list(sup.nodes), list(GM.mapping.keys()))
    diff = np.array(sup.nodes)[~oldnodes]

    plt.subplot(rows, cols, i+1)
    
    pos = graphviz_layout(sup)
    if len(sup.edges) > 0:
        edges,weights = zip(*nx.get_edge_attributes(sup,'weight').items())
        newedge = np.isin(np.array(edges), diff).max(1)
        
        edge_col = [ecols[int(this)] for this in (np.array(weights)+1)//2]
        
        nx.draw(sup, pos=pos, node_size=30, node_color=oldnodes, cmap='viridis',
                style = [estls[1*i] for i in newedge],
                edge_color=edge_col, edgelist=edges, width=2)
    else:
        nx.draw(sup, pos=pos, node_size=30, node_color=oldnodes, cmap='viridis')


#%%

valid = []
invalid = []
which_dec = []
decs = []
for x_ in product([-1,0,1], repeat=3):
    for x0 in [0,2]:
        
        x = np.array(x_)
        newJ = np.block([[sups[root], x[:,None]], [x[None], x0]])

        FP = fixed_points(newJ)
        
        if FP is None:
            invalid.append(1*newJ)
        else:
            G = nx.Graph(util.hamming(FP.T)==1)
            if is_isometric(FP, G):
                valid.append(1*newJ)
                
                match = False
                i = 0
                for Gref in decs:
                    if nx.is_isomorphic(G, Gref):
                        match = True
                        which_dec.append(i)
                    i += 1
                if not match:
                    decs.append(G)
                    which_dec.append(i+1)
            else:
                invalid.append(1*newJ)
                
#%%

for i_,i in enumerate(np.where(is_rep)[0]):
    
    plt.subplot(1, 6, i_+1)
    
    S = equiv_classes(all_n_cubes[i])
    
    h = np.diag(sups[i_])/2
    J = sups[i_] - np.diag(np.diag(sups[i_]))
    
    # G = nx.Graph(J)
    # pos = graphviz_layout(G)
    # if len(G.edges) > 0:
    #     edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    
    #     nx.draw(G, pos=pos, node_size=15, node_color=h, cmap='bwr', 
    #             edge_color=weights, edgelist=edges)
    # else:
    #     nx.draw(G, pos=pos, node_size=15, node_color=h, cmap='bwr')
    E = np.array(all_n_cubes[i].edges)
    
    elab = np.argmax(np.abs(S[E[:,0]] - S[E[:,1]]), axis=1)
    
    # pos = graphviz_layout(all_n_cubes[i])
    nx.draw(all_n_cubes[i], #style=sty[1*is_rep[i]],
            edge_color=elab, node_size=15,
            node_color='k',
            width=2)

#%%
n = 4

S = util.F2(n)

aye, jay = np.triu_indices(n,k=1)
k = int(spc.binom(n,2))

graphs = []
is_iso = []
which_graph = []
vals = []
# for a,b,c,d,e in tqdm(product(np.linspace(0,1,11), repeat=5)):
for x in tqdm(product([-2,-1,0,1,2], repeat=k+n-1)):
# for x in tqdm(product([-1,0,1], repeat=k+n-1)):
    
    # a_ = a + (a-1)*d
    # b_ = b + (b-1)*e
    # c_ = c + (c-1)*(d+e)
    # J = np.array([[0,a_,b_],[a_,0,c_],[b_,c_,0]])
    # h = np.array([0,d,e])
    
    J = np.zeros((n,n))
    J[aye,jay] = x[:k]
    J += J.T
    
    h = np.zeros(n)
    h[1:] = x[k:]
    
    val = util.qform(J, S).squeeze() + 2*S@h
    if np.min(val) < -1e-10:
        continue
    vals.append(x)
    
    eq = val < 1e-10
    
    G = nx.Graph(util.hamming(S[eq].T)==1)
    match = False
    for i,Gref in enumerate(graphs):
        if nx.is_isomorphic(G, Gref):
            which_graph.append(i)
            match = True
            
    if not match:
        graphs.append(G)
        which_graph.append(i+1)
        if is_partial_cube(G):
            is_iso.append(equiv_classes(G).shape[1] == n)
        else:
            is_iso.append(False)

#%%

is_rep = [np.any([nx.is_isomorphic(g,g_) for g in graphs]) for g_ in all_n_cubes]

ng = len(all_n_cubes)

rows = int(np.floor(np.sqrt(ng)))
cols = int(np.ceil(ng/rows))

clr = ['r', 'b']
for i in range(ng):
    plt.subplot(rows, cols, i+1)
    nx.draw(all_n_cubes[i], node_color=clr[1*is_rep[i]], node_size=15)


#%%
is_conn = [nx.is_connected(g) for g in graphs]

# ng = len(graphs)
ng = np.sum(is_conn)

rows = int(np.floor(np.sqrt(ng)))
cols = int(np.ceil(ng/rows))
   
clr = ['r', 'b']
for i,g in enumerate(np.where(is_conn)[0]):
    plt.subplot(rows, cols, i+1)
    nx.draw(graphs[g], node_color=clr[1*is_iso[g]], node_size=15)