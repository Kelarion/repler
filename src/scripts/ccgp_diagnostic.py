
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

#%%

num_trial = 5000
N = 100
depth = 1
# nonlin = 'ReLU'
nonlin = 'Tanh'
# noise = 0.1
num_net = 10

num_bits = 3
ks = []

F = util.F2(num_bits)
if len(ks)>0:
    labs = util.addsum(F, *[list(c) for k in ks for c in combinations(range(num_bits), k)])
else:
    labs = F

pi = np.random.dirichlet(np.ones(labs.shape[-1]))

# inps = tasks.LinearExpansion(tasks.Embedding((2*labs - 1)@np.diag(np.sqrt(pi))), 100, noise_var=0)
inps = tasks.BinaryLabels(2*labs.T - 1)
# outs = tasks.BinaryLabels(np.mod(F.sum(1, keepdims=True), 2).T)
outs = tasks.BinaryLabels(np.mod(F[:,[0]]+F[:,[1]], 2).T)

# this_exp = sxp.FeedforwardExperiment(inps, outs)
# this_exp = exp.OODFF(inps, outs, [0,1,2,3,4,5])
this_exp = exp.OODFF(inps, outs, [0,1,6,7])

nets = this_exp.initialize_network(students.SimpleMLP, 
                                  width=N, 
                                  p_targ=students.Bernoulli, 
                                  depth=depth,
                                  activation=nonlin,
                                  init_scale=None,
                                  num_init=num_net)

this_exp.train_network(nets, skip_rep_metrics=True, verbose=True, nepoch=100)


#%%

x_ = inps(np.arange(2**num_bits), 0)
y_ = outs(np.arange(2**num_bits), 0).numpy().squeeze()

Z = nets[0].enc.network(x_).detach().numpy()

z_noise = nets[0].enc.network(this_exp.train_data[0]).detach().numpy()

