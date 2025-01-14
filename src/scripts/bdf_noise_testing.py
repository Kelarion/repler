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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
import itertools as itt
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
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

from pypoman import compute_polytope_vertices, compute_polytope_halfspaces
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

#%%

# num_item = 7
# num_feat = 4
# noise_mag = 0.1

num_draw = 100
num_perturb = 10

params = itt.product([10,20,50], [4,8,16], [0.1])

worked = []
recon = []
# num_sols = []
n = []
k = []
for num_item, num_feat, noise_mag in params:
    
    did_it_work = []
    # csim = []
    # n_sol = []
    for i in tqdm(range(num_draw)):
        # x_true = np.random.choice([-1,1], size=(num_feat, num_item))
        # x_noise = x_true + np.random.randn(num_feat, num_item)*noise_mag
        
        # K = x_true.T@x_true/num_feat
        # K = 
        K_noise = x_noise.T@x_noise/num_feat
        K_proj = df.gauss_projection(util.correlify(K_noise))
        
        # sols = []
        # for _ in range(num_perturb):
        #     _,sol = df.solve_lp(K_proj, perturb=1e-3, return_full=True)
        #     sols.append(sol > 1e-6)
        
        # n_sol.append(len(np.unique(sols, axis=0)))
        
        try:
            # idx = df.squares_first(1-K)
            # _,_ = df.BDF(K[idx,:][:,idx], fixed_order=True)
            try:
                _,_ = df.cuts(K, in_cut=True, verbose=False, chains=12)
            except:
                _,_ = df.cuts(K, in_cut=True, verbose=False, chains=100)
            # csim.append(util.centered_kernel_alignment(K, S@np.diag(pi)@S.T))
            did_it_work.append(True)
        except RuntimeError:
            did_it_work.append(False)
        
    
    worked.append(np.mean(did_it_work))
    # recon.append(np.min(csim))
    # num_sols.append(n_sol)
    n.append(num_item)
    k.append(num_feat)



#%%

num_draw = 100

# params = itt.product(range(4, 20, 2), range())

worked = []
n = []
r = []
for num_item in range(4, 20, 2):
    for rank in range(2,num_item):
        did_it_work = []
        # csim = []
        # n_sol = []
        for i in tqdm(range(num_draw)):
            
            K = util.elliptope_sample(num_item).squeeze()
            
            K_proj = df.gauss_projection(K)
            
            try:
                try:
                    _,_ = df.cuts(K_proj, verbose=False, chains=12)
                except:
                    _,_ = df.cuts(K_proj, verbose=False, chains=100)
                did_it_work.append(True)
                
            except df.DidntWorkError:
                did_it_work.append(False)
            
        worked.append(np.mean(did_it_work))
        n.append(num_item)
        r.append(rank)







