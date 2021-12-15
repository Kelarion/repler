
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
from matplotlib import colors as mpc
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

# my code
import students
import assistants
import experiments as exp
import util
import tasks
import plotting as dicplt
import grammars as gram

#%%

num_var = 3
num_val = 2

tree_depth = num_var
c_max = num_val # maximum number of children

child_prob = 0.7
# child_prob = 1 

def add_children(seq, idx, i_max):
    """recursion to generate sequence"""
    
    if idx < i_max:
        n_child = (np.random.binomial(c_max-1, child_prob, len(seq))+1).tolist()
        
        children = []
        for ix in range(len(seq)):
            ins_idx = ix + sum(n_child[:0]) + 1
            c = [[i] for i in np.random.choice(num_val, n_child[ix], replace=False).tolist()]
            seq[ins_idx:ins_idx] = c
            children += c
            
        for child in children:
            add_children(child, idx+1, i_max)


def define_graph(num_vars, N, minimize=True):
    """ num vars is a list of number of variables in each layer """
    L = len(num_vars)
    tot_var = np.cumsum([0,]+num_vars)
    nodes = [np.arange(tot_var[i],tot_var[i]+num_vars[i])+1 for i in range(L)]
    
    # horrible,  awful list comprehensions, to keep things fast
    children = [nodes[0]]
    _ = [children.append(make_children(children[i], nodes[i+1], N, minimize)) for i in range(L-1)]
    edges = [[p,c] for i in range(L-1) for p,c in zip(np.repeat(children[i],N), children[i+1])]
    
    return edges
 

def make_children(L1, L2, N, minimize=True):
    n2 = N*len(L1)
    # reps = (N*len(L1))/(len(L2))//N
    if len(L1) <= len(L2):
        reps = N
    else:
        # reps = N + N*((n2 - len(L2))//N)
        reps = int(np.ceil(n2/len(L2)))
    # reps = N
    # n_s = (reps*len(L2) - N*len(L1))//reps
    # n_s = np.max([n_s, int(np.ceil((reps*len(L2) - n2 + n_s)/reps))])
    # n_dup = n2 - n_s # how many children are duplicated 
    n_dup = int(np.ceil((n2 - len(L2))/(reps-1)))*reps
    n_s = n2 - n_dup
    if minimize: # to minimize recency of common ancestor -- more abstraction
        dups = np.repeat(L2,reps) + np.mod(np.arange(reps*len(L2)),reps)/reps
        c = np.append(dups[:n_dup],L2[len(L2)-n_s:])
    else: # less abstraction
        c = np.tile(L2, reps)[:n2] + (np.arange(n2)//len(L2))/reps
    # p = np.repeat(L1, N)
    # return [[p[i], c[i]] for i in range(n2)]
    return c

#%%
minimize = False
# minimize = True

K = 2
# K = 3

# edges = define_graph([1,2,4,8,16], 2, minimize=minimize)
# edges = define_graph([1,2,3,4,5], 2, minimize=minimize)
# edges = define_graph([1,1,1,1], K, minimize=minimize)
# edges = define_graph([K**0,K**1,K**2,K**3,K**4], K, minimize=minimize)
# edges = define_graph([1,1,1,1,1], K, minimize=minimize)
edges = define_graph([1,K,K,K**2,K**3,K**4,K**5], K, minimize=minimize)


g = nx.DiGraph()
g.add_edges_from(edges)
        
pos = graphviz_layout(g, prog="dot")
nx.draw(g, pos, with_labels=True)

# convert it into a feature-generator
g_ = nx.DiGraph()
_ = [[[g_.add_edge(e[0]+a*1j, e[1]+b*1j) for b in range(K)] for a,e in enumerate(g.out_edges(n))] for n in g.nodes]

roots = [node for node in g_.nodes() if g_.out_degree(node)!=0 and g_.in_degree(node)==0]
leaves = [node for node in g_.nodes() if g_.in_degree(node)!=0 and g_.out_degree(node)==0]

_ = [g_.add_edge(0, n) for n in roots]

data = [np.array(list(nx.algorithms.simple_paths.all_simple_paths(g_, 0, n))).squeeze() for n in leaves]

idx = np.concatenate([d.real.astype(int)[1:] for d in data],-1)
val = np.concatenate([d.imag.astype(int)[1:] for d in data],-1)
var = np.concatenate([np.ones(d.shape[-1]-1, dtype=int)*i for i,d in enumerate(data)],-1)

labels = np.zeros((idx.max(),var.max()+1))*np.nan
labels[idx-1,var] = val
reps = np.concatenate([np.where(np.isnan(l), 0, np.eye(K)[:,np.where(np.isnan(l), 0, l).astype(int)]) for l in labels])

plt.figure()
plt.subplot(131)
pos = graphviz_layout(g, prog="twopi")
nx.draw(g, pos, node_color=np.array(g.nodes).astype(int), cmap='nipy_spectral')
dicplt.square_axis()
plt.subplot(132)
plt.imshow(labels)
plt.subplot(133)
plt.imshow(util.dot_product(reps,reps))


