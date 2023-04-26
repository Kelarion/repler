
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
import scipy.special as spc
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
import students as stud
import assistants
import experiments as exp
import util
import pt_util
import tasks
import server_utils
import plotting as dicplt
import grammars as gram
import dichotomies as dics


#%%
def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(len(r) for r in M)
    dims = M[0].shape[1:]

    Z = np.zeros((len(M), maxlen, *dims))*np.nan
    for enu, row in enumerate(M):
        Z[enu, :len(row)] = row 
        
    return Z

#%%

exp_prm = {'experiment': exp.RandomOrthogonal,
		   'num_bits':(2,2,3,3,3,4,4,4,4,5,5,5,5,5),
		   'num_targets': (1,2,1,2,3,1,2,3,4,1,2,3,4,5),
		   'signal':[0, 0.5, 1],
		   'seed': None,
		   'use_mean': True,
		   'dim_inp': 100,
		   'input_noise': 0.1,
		   }

net_args = {'model': stud.ShallowNetwork,
			'num_init': 10,
			'width': 128,
			'p_targ': stud.Bernoulli,
			'activation':[pt_util.TanAytch(), pt_util.RayLou()]
			}

opt_args = {'skip_metrics': True,
			'nepoch': 1000,
			'verbose': False,
			'train_outputs': False
			}


#%%

all_exp_args, prm = server_utils.get_all_experiments(exp_prm, net_args, opt_args)

all_metrics = {}
for exp_args in tqdm(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    # this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR+'results/', exp_args['opt_args'])
    
    
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.metrics.items()}
    for k in all_metrics.keys():
        all_metrics[k].append(this_exp.metrics[k])

for k,v in all_metrics.items():
    all_metrics[k] = pad_to_dense(v)




