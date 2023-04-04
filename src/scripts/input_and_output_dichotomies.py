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
import itertools as itt
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
import super_experiments as sxp
import util
import pt_util
import tasks
import plotting as dicplt
import grammars as gram
import dichotomies as dics

#%%

def centered_kernel_alignment(K1,K2):
    K1_ = K1 - K1.mean(-2,keepdims=True) - K1.mean(-1,keepdims=True) + K1.mean((-1,-2),keepdims=True)
    K2_ = K2 - K2.mean(-2,keepdims=True) - K2.mean(-1,keepdims=True) + K2.mean((-1,-2),keepdims=True)
    denom = np.sqrt((K1_**2).sum((-1,-2))*(K2_**2).sum((-1,-2)))
    return (K1_*K2_).sum((-1,-2))/np.where(denom, denom, 1e-12)

def center(K):
    return K - K.mean(-2,keepdims=True) - K.mean(-1,keepdims=True) + K.mean((-1,-2),keepdims=True)

def face(a,b,c):
    return a

def corner(a,b,c):
    return a*b + a*c + b*c

def snake(a,b,c):
    return a*c + b*(~c)

def net(a,b,c):
    return ~(a*c + a*b + ~(a+b+c))

def xor2(a,b,c):
    return a^b

def xor3(a,b,c):
    return a^b^c

def generate_all(a, b, c):
    """
    brute force generation of all classes of dichotomies
    """
    
    all_dics = []
    which_class = []
    
    for i, logic in enumerate([face, corner, snake, net, xor2, xor3]):
        
        all_colorings = []
        
        for p in permutations(range(3)):
        
            d = np.stack([a, b, c])[p, :]
            colorby = logic(d[0],d[1],d[2])
            
            if not colorby[0]:
            # if True:
                all_colorings.append(colorby)
            
            for cs in itt.chain(combinations(range(3),1),combinations(range(3),2),combinations(range(3),3)):

                d[cs,:] = 1 - d[cs,:]
                
                colorby = logic(d[0],d[1],d[2])
                if not colorby[0]:
                    all_colorings.append(colorby)
        
        class_dics = np.unique(all_colorings, axis=0)
        
        all_dics.append(class_dics)
        which_class.append(np.ones(len(class_dics))*i)
    
    return np.vstack(all_dics), np.concatenate(which_class)


#%%

num_trial = 5000
N = 10
depth = 1
# nonlin = 'ReLU'
nonlin = 'Tanh'
noise = 0.1
dim_inp = 3
num_init = 5

# inps = tasks.StandardBinary(3)
# outs = tasks.RandomDichotomies(d=[(0,1,3,5),(0,2,3,6),(0,1,2,4)]) # 3 corners


# this_exp = sxp.FeedforwardExperiment(inps, outs)
this_exp = exp.LogicTask(tasks.StandardBinary(3).positives, 
                          [(0,1,3,5),(0,2,3,6),(0,1,2,4)], 
                          noise=noise,
                          dim_inp=dim_inp)

# this_exp = exp.LogicTask(tasks.StandardBinary(3).positives, 
#                           [(0,3,5,6)], 
#                           noise=noise,
#                           dim_inp=dim_inp)

nets = this_exp.initialize_network(students.SimpleMLP, 
                                   width=N, 
                                   p_targ=students.Bernoulli, 
                                   depth=depth,
                                   activation=nonlin,
                                   num_init=num_init)

this_exp.train_network(nets, skip_rep_metrics=True, verbose=True, opt_alg=optim.Adam, nepoch=1000)


#%% abstraction metrics

layer = 1 # 1-indexed

clf = svm.LinearSVC()

cube = tasks.StandardBinary(3)(range(8)).numpy().T
a = cube[0] > 0
b = cube[1] > 0
c = cube[2] > 0
D, which_class = generate_all(a, b, c)

test_conds = this_exp.train_conditions
test_inps = this_exp.inputs(test_conds)

all_ps = []
all_ccgp = []
all_dec = []
for this_net in nets:

    test_reps = this_net.enc.network[:2*layer](test_inps).detach().numpy().T
    
    ps = []
    ccgp = []
    dec = []
    # all_dics = dics.Dichotomies(8, [(0,1,2,3),(0,2,5,7),(0,1,5,6)], 100)
    
    for this_dic in tqdm(D):   
        
        cols = this_dic[test_conds]
        
        ps.append(dics.parallelism_score(test_reps, test_conds, cols))
        ccgp.append(np.mean(dics.compute_ccgp(test_reps.T, test_conds, cols, clf)))
        
        clf.fit(test_reps.T + np.random.randn(*test_reps.T.shape)*0.2, cols)
        dec.append(clf.score(test_reps.T + np.random.randn(*test_reps.T.shape)*0.2, cols))
    
    all_ps.append(ps)
    all_ccgp.append(ccgp)
    all_dec.append(dec)

#%%

dicplt.dichotomy_plot(np.mean(all_ps, axis=0), np.mean(all_ccgp, axis=0), np.mean(all_dec, axis=0), 
                      input_dics=[0,1,2],
                      output_dics=[3,4,5], 
                      other_dics=[6],
                      c=which_class, cmap='Set1')

# for i in range(len(all_ps)):
    
#     plt.figure()
#     dicplt.dichotomy_plot(all_ps[i], all_ccgp[i], all_dec[i], 
#                           input_dics=[0,1,2], 
#                           output_dics=[-1],
#                           c=which_class, cmap='Set1')
