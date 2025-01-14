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
import students as stud
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

nepoch = 100

exp_prm = {'num_points': 32,
		   'num_targets': 5,
		   'signal': 0,
		   'seed': 0,
		   'scale': 0.0,
		   'dim_inp': 100,
		   'input_noise': 1,
           'max_rank': 5
		   }

net_args = {'model': stud.SimpleMLP,
 			'num_init': 10,
 			'width': 128,
            'depth': 10,
 			'p_targ': stud.Bernoulli,
 			'activation': 'ReLU'
 			}


opt_args = {'skip_metrics': True,
			'nepoch': nepoch,
			'verbose': True,
			'lr': 1e-3
			}

#%%

# this_exp = exp.RandomKernelClassification(**exp_prm)
inps = tasks.BinaryLabels(2*util.F2(5).T - 1)
outs = tasks.BinaryLabels(np.mod(util.F2(5).sum(1,keepdims=True).T,2))
this_exp = sxp.FeedforwardExperiment(inps, outs)

nets = this_exp.initialize_network(**net_args)

this_exp.train_network(nets, **opt_args)


#%%

x = this_exp.inputs(np.arange(this_exp.inputs.num_cond), 0)
y = this_exp.outputs(np.arange(this_exp.inputs.num_cond), 0).detach().numpy()

Kx = (x@x.T).detach().numpy()
Ky = y@y.T

Kn = util.ker(util.center_kernel(Ky+Kx))

czy = []
czx = []
czn = []

czy_n = []
czx_n = []
czn_n = []

layer_K = []
for l in range(net_args['depth']):
    
    kerns = []
    czy_i = []
    czx_i = []
    czn_i = []
    for net in nets:
        z = net.enc.network[:(2*(l+1))](x).detach().numpy()
        
        Kz_i = z@z.T
        czx_i.append(util.centered_kernel_alignment(Kz_i, Kx))
        czy_i.append(util.centered_kernel_alignment(Kz_i, Ky))
        czn_i.append(util.centered_kernel_alignment(Kz_i, Kn)) 
        
        kerns.append(z@z.T)
    
    Kz = np.mean(kerns, axis=0)
    layer_K.append(Kz)
    # Kn = util.ker(Kx+Ky) # kernel (algebra sense) of Kx + Ky
    # Kn = kerXY@kerXY.T

    czx.append(util.centered_kernel_alignment(Kz, Kx))
    czy.append(util.centered_kernel_alignment(Kz, Ky))
    czn.append(util.centered_kernel_alignment(Kz, Kn))
    
    czy_n.append(czy_i)
    czx_n.append(czx_i)
    czn_n.append(czn_i)
    
plt.plot(czx, marker='o', linewidth=2, c='b', zorder=10)
plt.plot(czy, marker='o', linewidth=2, c='g', zorder=10)
plt.plot(czn, marker='o', linewidth=2, color='r', zorder=10)

plt.plot(czx_n, marker='o', alpha=0.3, c='b')
plt.plot(czy_n, marker='o', alpha=0.3, c='g')
plt.plot(czn_n, marker='o', alpha=0.3, color='r')

plt.legend(['input', 'target', 'neither'])
plt.ylim([0,1])

#%%

cos_foo = np.linspace(0,1,1000)
c_xy = this_exp.alignment

# ub = np.sqrt(1-cos_foo**2)

# ub = c_xy*cos_foo + np.sqrt(1-c_xy**2)*np.sqrt(1-cos_foo**2)
# plt.plot((cos_foo-c_xy)/(1-c_xy), (ub-c_xy)/(1-c_xy), 'k--')
# plt.plot(cos_foo, ub)

phi = (np.pi/2 -np.arccos(c_xy))/2
basis = np.array([[np.cos(phi),np.cos(np.pi/2-phi)],[np.sin(phi),np.sin(np.pi/2-phi)]])
ub = basis@np.stack([cos_foo, np.sqrt(1-cos_foo**2)])
plt.plot(ub[0],ub[1], 'k--')
dicplt.square_axis()

plt.plot(czx, czy, marker='o', linewidth=2, c='r', zorder=10)
plt.plot(czx_n, czy_n, marker='o', alpha=0.2, c='r')




