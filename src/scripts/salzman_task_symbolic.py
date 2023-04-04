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
import super_experiments as sxp
import util
import pt_util
import tasks
import plotting as dicplt
import grammars as gram
import dichotomies as dics


#%% Make data

I = np.eye(4, dtype=int)
O = np.zeros((4,4))

## response array
##                H R + -
resp = np.array([[1,0,1,0], # A
                 [0,1,1,0], # B
                 [1,0,0,1], # C
                 [0,1,0,1]],# D
                dtype=int)  # context 1
resp2 = np.roll(resp, 1, axis=0) # context 2

s1, s2 = np.where(np.ones((4,4))) # previous stim and current stim identity

SR = np.block([[I, resp], # stim, resp
               [I, resp2]])

SRR = np.block([[I[s1], resp[s1], resp[s2]], # prev_stim, prev_resp, resp 
                [I[s1], resp2[s1], resp2[s2]]]) 

SRS_ = np.block([[I[s1], resp[s1], I[s2]], # prev_stim, prev_resp, stim
                [I[s1], resp2[s1], I[s2]]])

R_ = np.block([[resp[s2]], # resp
               [resp2[s2]]])

SRSR = np.block([[I[s1], resp[s1], I[s2], resp[s2]], # prev_stim, prev_resp, stim, resp
                [I[s1], resp2[s1], I[s2], resp2[s2]]])


#### To do: include pre-stimulus inputs 

SRS_null = np.block([[I[s1], resp[s1], I[s2]], # prev_stim, prev_resp, stim
                     [I,     resp,     O   ],       # pre-stimulus trials
                     [I[s1], resp2[s1], I[s2]],
                     [I,     resp2,    O   ]])

R_null = np.block([[resp[s2], O[s2][:,[0]]], # resp, null
                   [O,        O[:,[0]]+1],
                   [resp2[s2], O[s2][:,[0]]],
                   [O,          O[:,[0]]+1]])

#### To do: transfer task
#### To do: try subsets of prev/current pairs, and subsets of stimuli
#### To do: look at output weights


#%%

num_trial = 5000
N = 100
depth = 2
# nonlin = 'ReLU'
nonlin = 'Tanh'
# noise = 0.1

inps = tasks.BinaryLabels(SRS_null.T)
outs = tasks.BinaryLabels(R_null.T)

this_exp = sxp.FeedforwardExperiment(inps, outs)

nets = this_exp.initialize_network(students.SimpleMLP, 
                                  width=N, 
                                  p_targ=students.Bernoulli, 
                                  depth=depth,
                                  activation=nonlin)

this_exp.train_network(nets, skip_rep_metrics=True, verbose=True)


#%% reps

Z1 = nets[0].enc.network[:2](torch.tensor(SRS_null).float()).detach().numpy()
Z = nets[0].enc.network(torch.tensor(SRS_null).float()).detach().numpy()


#%% abstraction metrics

layer = 1 # 1-indexed

which_stim = np.concatenate([s2, s2+4])

# cntx = np.arange(32) >= 16

test_conds = this_exp.train_conditions
test_inps = inps(test_conds)
test_reps = nets[0].enc.network[:2*layer](test_inps).detach().numpy().T


clf = svm.LinearSVC()

ps = []
ccgp = []
dec = []
all_dics = dics.Dichotomies(8, [(0,1,2,3),(0,2,5,7),(0,1,5,6)], 100)
for d in tqdm(all_dics):
    cols = all_dics.coloring(which_stim[test_conds])
    
    ps.append(dics.parallelism_score(test_reps, which_stim[test_conds], cols))
    ccgp.append(np.mean(dics.compute_ccgp(test_reps.T, which_stim[test_conds], cols, clf)))
    
    clf.fit(test_reps.T + np.random.randn(*test_reps.T.shape)*0.2, cols)
    dec.append(clf.score(test_reps.T + np.random.randn(*test_reps.T.shape)*0.2, cols))


dicplt.dichotomy_plot(ps, ccgp, dec, output_dics=[1,2], other_dics=[0])



