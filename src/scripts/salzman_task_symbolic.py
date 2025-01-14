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
# import pydot 
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

from pypoman import compute_polytope_vertices, compute_polytope_halfspaces
import cvxpy as cvx
# import polytope as pc

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

import distance_factorization as df
import df_util
import bae

#%% Make data

I = np.eye(4, dtype=int)
O = np.zeros((4,4))

## response array
# ##                H R + -
# resp = np.array([[1,0,1,0], # A 
#                  [0,1,1,0], # B
#                  [1,0,0,1], # C
#                  [0,1,0,1]],# D
#                 dtype=int)  # context 1

# resp2 = np.roll(resp, 1, axis=0) # context 2

## Action
A = np.array([[1,0],  # A
              [0,1],  # B
              [1,0],  # C
              [0,1]]) # D

## Value
V = np.array([[1,0,0],# value
              [1,0,0],
              [0,1,0],
              [0,1,0]])

## Incorrect
X = np.ones((4,3))@np.diag([0,0,1])

resp = np.block([A, V])
respx = np.block([1-A, X])

resp2 = np.roll(resp, 1, axis=0) # context 2
respx2 = np.roll(respx, 1, axis=0) # context 2

s1, s2 = np.where(np.ones((4,4))) # previous stim and current stim identity

SR = np.block([[I, resp], # stim, resp
               [I, resp2]])

SRR = np.block([[I[s1], resp[s1], resp[s2]], # prev_stim, prev_resp, resp 
                [I[s1], resp2[s1], resp2[s2]]]) 

SRS_ = np.block([[I[s1], resp[s1], I[s2]], # prev_stim, prev_resp, stim
                [I[s1], resp2[s1], I[s2]]])

SRxS_ = np.block([[I[s1], resp[s1], I[s2]], # prev_stim, prev_resp, stim
                  [I[s1], respx[s1], I[s2]],
                  [I[s1], resp2[s1], I[s2]],
                  [I[s1], respx2[s1], I[s2]]])

R_ = np.block([[resp[s2]], # resp
               [resp2[s2]]])

Rx_ = np.block([[resp[s2]], # resp
                [resp[s2]],
                [resp2[s2]],
                [resp2[s2]]])

SRSR = np.block([[I[s1], resp[s1], I[s2], resp[s2]], # prev_stim, prev_resp, stim, resp
                [I[s1], resp2[s1], I[s2], resp2[s2]]])


#### To do: include pre-stimulus inputs 

# SRS_null = np.block([[I[s1], resp[s1], I[s2]], # prev_stim, prev_resp, stim
#                      [I,     resp,     O   ],       # pre-stimulus trials
#                      [I[s1], resp2[s1], I[s2]],
#                      [I,     resp2,    O   ]])

# R_null = np.block([[resp[s2], O[s2][:,[0]]], # resp, null
#                    [O,        O[:,[0]]+1],
#                    [resp2[s2], O[s2][:,[0]]],
#                    [O,          O[:,[0]]+1]])

#### To do: transfer task
#### To do: try subsets of prev/current pairs, and subsets of stimuli
#### To do: look at output weights


#%%

nepoch = 200
N = 100
depth = 1
# nonlin = 'ReLU'
nonlin = 'Tanh'
# noise = 0.1

# inps = tasks.BinaryLabels(SRS_null.T)
# outs = tasks.BinaryLabels(R_null.T)

inps = tasks.BinaryLabels(SRS_.T)
outs = tasks.BinaryLabels(R_.T[[0]])

# inps = tasks.BinaryLabels(SRxS_.T)
# outs = tasks.BinaryLabels(Rx_.T)

this_exp = sxp.FeedforwardExperiment(inps, outs)

nets = this_exp.initialize_network(students.SimpleMLP,
                                  width=N, 
                                  p_targ=students.Bernoulli,
                                  depth=depth,
                                  activation=nonlin,
                                  num_init=10)

this_exp.train_network(nets, skip_rep_metrics=True, verbose=True, nepoch=nepoch)

#%% reps

# Z1 = nets[0].enc.network[:2](torch.tensor(SRS_).float()).detach().numpy()
Z = nets[0].enc.network(torch.tensor(SRS_).float()).detach().numpy()

X = inps.labels.T

#%% Factorize

baer = bae.BAE(Z, 40, pvar=0.95)
baer.init_optimizer(decay_rate=0.95, period=2, initial=10)

en = []
for t in tqdm(range(400)):
    #r = np.sum(pvar< (0.8 + 0.2*(t//10)/10))
    baer.proj(pvar=0.95)
    #baer.scl = baer.scaleS()
    baer.grad_step()
    en.append(baer.energy())
    
S = baer.S.todense()

Sunq = np.unique(np.mod(S+S[[0]],2), axis=1)

is_dec = util.qform(util.center(SRS_@SRS_.T), Sunq.T).squeeze() > 1e-7

S,pi = df_util.mindistX(Z, Sunq[:,is_dec], beta=1e-5)
S = S[:,np.argsort(-pi)]
pi = pi[np.argsort(-pi)]
    
#%% Circuit

XZ = (2*inps.labels-1)@(2*S-1)
ZY = outs.labels@(2*S-1)

#%% Input Interventions

ortho = True
# ortho = False

if ortho:
    W = df_util.krusty(np.diag(np.sqrt(pi))@S.T, (Z-Z[[0]]).T)
    W = np.diag(np.sqrt(pi))@W
else:
    W = la.pinv(S)@(Z-Z[[0]])

dL = []
for j in range(X.shape[1]):
    row = []
    
    for i in range(len(Z)):
        
        sgn = 2*X[i,j]-1
        # ystar = 2*R_[i,0]-1
        
        newX = 1*X
        newX[i,j] = 1-newX[i,j]
        
        before = W@nets[0].enc(torch.FloatTensor(X[i])).detach().numpy()
        after = W@nets[0].enc(torch.FloatTensor(newX[i])).detach().numpy()
        
        row.append((sgn*(after-before)))
    dL.append(row)

dL = np.squeeze(dL)

#%% Hidden Interventions

ortho = True
# ortho = False

if ortho:
    W = df_util.krusty(np.diag(np.sqrt(pi))@S.T, (Z-Z[[0]]).T)
    W = np.diag(np.sqrt(pi))@W
else:
    W = la.pinv(S)@(Z-Z[[0]])

dL = []
for j in range(len(W)):
    row = []
    for i in range(len(Z)):
        
        sgn = 2*S[i,j]-1
        # ystar = 2*R_[i,0]-1
        
        before = nets[0].dec(torch.FloatTensor(Z[i])).detach().numpy()
        after = nets[0].dec(torch.FloatTensor(Z[i] - sgn*W[j])).detach().numpy()
        
        row.append((sgn*(after-before)))
    dL.append(row)

dL = np.squeeze(dL)


