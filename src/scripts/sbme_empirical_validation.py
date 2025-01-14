CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import numpy.linalg as nla
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
import itertools as itt
from tqdm import tqdm
from time import time

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.special as spc
import scipy.sparse as sprs
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from sklearn.manifold import MDS

import networkx as nx
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

import cvxpy as cvx
# import polytope as pc
# from hsnf import column_style_hermite_normal_form

# my code
import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as dicplt
import dichotomies as dics

import distance_factorization as df
import df_models as mods
import df_util

#%% Random Schur-independent categories

## parameters: (1) dirichlet scale (2) balance

scl = 0.1
p = 0.5
num_draw = 10

beta = 1e-3
eps = 1
num_chain = 5
tol = 1e-2
pimin = 0.1

losses = []
howlong = []
hammean = []
hammed = []
for N in [16,32,64,128,256,512,1024]:

    # r = int((1 + np.sqrt(8*N - 7))//2)
    r = int(np.sqrt(2*N))
    # r = 4
    
    loss = []
    ham = []
    t = []
    for j in tqdm(range(num_draw)):
        i = 0
        S_true = np.random.choice([1,-1], size=(N,r))
        while nla.matrix_rank(df_util.schur(S_true)) < (1+spc.binom(r,2)):
            S_true = np.random.choice([1,-1], size=(N,r), p=[p, 1-p])
            i += 1
            if i > 100:
                print('Dangit!')
                break
            
        pi_true = r * np.random.dirichlet(np.ones(r)/scl)
        K = S_true @ np.diag(pi_true) @ S_true.T
        nrm = np.sum(util.center(K)**2)
        
        t0 = time()
        S, pi = df.cuts(K, branches=num_chain, eps=eps, beta=beta, order='ball', 
                        tol=tol, pimin=pimin)
        t.append(time()-t0)
        if num_chain>1:
            bestS, bestpi = df_util.mindist(K, S.toarray())
        else:
            bestS = S
            bestpi = pi[0]
        loss.append(util.centered_distance(K, bestS@np.diag(bestpi)@bestS.T)/nrm)
        ham.append(1 - (np.abs((2*bestS-1).T@S_true).max(0)/N))
    
    losses.append(loss)
    howlong.append(t)
    hammean.append(np.mean(ham, axis=1))
    hammed.append(np.median(ham, axis=1))
    
#%% Random trees

## parameters: (1) branch number (2) observation density

bmin = 2 # out degree min
bmax = 4 # max

num_draw = 10

beta = 1e-2
eps = 1
num_chain = 2
tol = 1e-2
pimin = 0.5

losses = []
howlong = []
hammean = []
hammed = []
for N in [16,32,64,128,256,512,1024]:

    loss = []
    ham = []
    t = []
    for j in tqdm(range(num_draw)):
        
        Strue = 2*df_util.randtree_feats(N, bmin, bmax)-1
        K = Strue@Strue.T
        nrm = np.sum(util.center(K)**2)
        
        t0 = time()
        S, pi = df.cuts(K, branches=num_chain, eps=eps, beta=beta, order='ball', 
                        tol=tol, pimin=pimin)
        t.append(time()-t0)
        if num_chain>1:
            bestS, bestpi = df_util.mindist(K, S.toarray())
        else:
            bestS = S.toarray()
            bestpi = pi[0]
        loss.append(util.centered_distance(K, bestS@np.diag(bestpi)@bestS.T)/nrm)
        
        ham.append(np.mean(1-(np.abs((2*bestS-1).T@Strue).max(0)/N)))
    
    losses.append(loss)
    howlong.append(t)
    hammean.append(ham)
    # hammed.append(np.median(ham, axis=1))

#%% Random cut matrices

## parameters: (1) dimensionality 


