
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
import plotting as tplt
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

# exp_prm = {'experiment': exp.RandomOrthogonal,
#  		   'num_bits':(2,3,3,3,4,4,4,4,5,5,5,5,5),
#  		   'num_targets': (1,1,2,3,1,2,3,4,1,2,3,4,5),
#  		   'signal':[0, 0.5, 1],
#  		   'seed': None,
#  		   'use_mean': True,
#  		   'dim_inp': 100,
#  		   'input_noise': [0.1,1],
#  		   }


# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits':(2,3,3,3,4,4,4,4,5,5,5,5,5),
# 		   'num_targets': (1,1,2,3,1,2,3,4,1,2,3,4,5),
# 		   'signal':[0, 0.5, 1],
# 		   'seed': list(range(6)),
# 		   'scale': 0.5,
# 		   'dim_inp': 100,
# 		   'input_noise': [0.1, 1],
# 		   }


# exp_prm = {'experiment': exp.RandomOrthogonal,
#  		   'num_bits':5,
#  		   'num_targets': [1,2,3,4,5],
#  		   'signal': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#  		   'seed': 0,
#  		   'scale': 0,
#  		   'dim_inp': 100,
#  		   'input_noise': [0.1, 1],
#  		   }


exp_prm = {'experiment': exp.RandomOrthogonal,
 		   'num_bits': (3,3,3,4,4,4,4,5,5,5,5,5),
 		   'num_targets': (1,2,3,1,2,3,4,1,2,3,4,5),
 		   'signal':[0, 0.25, 0.5, 0.75, 1],
 		   'seed': list(range(12)),
 		   'scale': 0.5,
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


#%% Compute input alignment 

all_exp_args, prm = server_utils.get_all_experiments(exp_prm, net_args, opt_args)

all_metrics = {}
# # for exp_args in tqdm(all_exp_args):

#     this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
#     this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
#     # this_exp.initialize_experiment( **exp_args['opt_args'])
#     this_exp.load_experiment(SAVE_DIR+'results/', exp_args['opt_args'])
    
#     if len(all_metrics) == 0:
#         all_metrics = {k:[] for k,v in this_exp.metrics.items()}
#     for k in all_metrics.keys():
#         all_metrics[k].append(this_exp.metrics[k])

PS = []
CCGP = []
inp_align = []
out_align = []
skew = []
PR = []
for exp_args in tqdm(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    # this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR+'results/', exp_args['opt_args'])
    
    n_cond = this_exp.inputs.num_cond
    
    x_ = this_exp.inputs(np.arange(n_cond), 0).T
    y_ = this_exp.outputs(np.arange(n_cond), 0).T
    
    cond = np.random.choice(range(n_cond), 1000)
    x_noise = this_exp.inputs(cond, noise=0.3).T
    # x_noise = this_exp.inputs(cond).T
    y_noise = (this_exp.outputs(cond)).T
    
    Kx = util.dot_product(x_-x_.mean(1,keepdims=True), x_-x_.mean(1,keepdims=True))
    Ky = util.dot_product(y_-y_.mean(1,keepdims=True), y_-y_.mean(1,keepdims=True))
    
    # Kx = util.dot_product(x_noise-x_noise.mean(1,keepdims=True), x_noise-x_noise.mean(1,keepdims=True))
    # Ky = util.dot_product(y_noise-y_noise.mean(1,keepdims=True), y_noise-y_noise.mean(1,keepdims=True))
    
    skew.append( np.sum(Ky*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Ky*Ky)) )
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.metrics.items()}
    for k in all_metrics.keys():
        all_metrics[k].append(this_exp.metrics[k])
        
    
    inp = []
    out = []
    # ccgp = []
    ps = []
    pr = []
    for model in this_exp.models:
        
        z_ = model(x_.T)[1].detach().numpy().T
        z_noise = model(x_noise.T)[1].detach().numpy().T     
        
        Kz = util.dot_product(z_-z_.mean(1,keepdims=True), z_-z_.mean(1,keepdims=True))
        # Kz = util.dot_product(z_noise-z_noise.mean(1,keepdims=True), z_noise-z_noise.mean(1,keepdims=True))
        
        inp.append(np.sum(Kz*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Kz*Kz)))
        out.append(np.sum(Kz*Ky)/np.sqrt(np.sum(Ky*Ky)*np.sum(Kz*Kz)))

        # ps.append(dics.parallelism_score(z_, np.arange(n_cond), y_.T))
        # ps.append(dics.parallelism_score(z_noise, cond, y_noise.T))
        
        ps.append(np.array([dics.efficient_parallelism(z_, y_[i].numpy()) for i in range(len(y_))]))
        # ccgp.append(np.mean(dics.compute_ccgp(z_noise.T, cond, np.squeeze(y_noise), svm.LinearSVC(), twosided=True)))
        pr.append(util.participation_ratio(x_))
        
    PS.append(np.array(ps))
    PR.append(pr)
    # CCGP.append(ccgp)
    inp_align.append(inp)
    out_align.append(out)

inp_align = np.array(inp_align)
out_align = np.array( out_align)
CCGP = np.squeeze(CCGP)
# PS = np.array(PS)
skew = np.array(skew)
PR = np.array(PR)

    #%%

cos_foo = np.linspace(0,1,1000)

ub = np.sqrt(1-cos_foo**2)

phi = (np.pi/2 -np.arccos(skew))/2  # re-align it with the orthogonal case
basis = np.array([[np.cos(phi),np.cos(np.pi/2-phi)],[np.sin(phi),np.sin(np.pi/2-phi)]])
# rot = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]]).transpose((2,0,1))
# correction = np.einsum('...ik,...kj->...ij', rot, np.linalg.inv(basis.transpose((2,0,1))))
correction = np.linalg.inv(basis.transpose((2,0,1)))

# correct_align = (correction)@np.stack([inp_align,out_align])
correct_align = np.einsum('lik,klj->ilj', correction, np.stack([inp_align,out_align]))
# correct_align = np.stack([inp_align,out_align])
correct_bound = np.stack([cos_foo, ub])

#%%

for i in [3,4,5]:
    for j in np.unique(prm['num_targets'][prm['num_bits'] == i]):
        
        
        filt = (prm['activation']=='TanAytch')&(prm['num_targets'] == j)&(prm['num_bits'] == i) # &(prm['input_noise'] == 0.1)
    
        plt.subplot(3, 5, 5*(i-3) + j)
        # plt.scatter(skew[filt], out_align[filt].mean(1), c='r')
        plt.scatter(skew[filt], ps[filt], c='r')
    
        filt = (prm['activation']=='RayLou')&(prm['num_targets'] == j)&(prm['num_bits'] == i) #&(prm['input_noise'] == 0.1)
        # plt.scatter(skew[filt], out_align[filt].mean(1), c='b')
        plt.scatter(skew[filt], ps[filt], c='b')
        

#%% input vs target alignment plots

# cols = cm.viridis(PR.mean(1)/ np.max(PR))

markers = ['o', 'v', 's', '+']

axs = tplt.hierarchical_labels([[0, 0.5, 1]], [[1,2,3,4,5]],    
                                 row_names=['s'], col_names=['k'],
                                 fontsize=13, wmarg=0.3, hmarg=0.1)


for i,t in enumerate([1,2,3,4,5]):
    
    for j,s in enumerate([0, 0.5, 1]):
        
        
        for k,b in enumerate([2,3,4,5]):
            
            # filt = (prm['activation']=='TanAytch')&(prm['num_targets'] == t)&(prm['signal'] == s) #&(prm['input_noise'] == 0.1)
            filt = (prm['activation']=='TanAytch')&(prm['num_targets'] == t)&(prm['signal'] == s)&(prm['num_bits'] == b)
            # plt.subplot(3, 5, (i+1) + j*5)
            # plt.scatter(inp_align[filt].mean(1), out_align[filt].mean(1), marker='o', c=prm['num_bits'][filt])
            axs[j,i].scatter(correct_align[0,filt].mean(1), correct_align[1,filt].mean(1), 
                        marker=markers[k], c='r')
    
    
            # filt = (prm['activation']=='RayLou')&(prm['num_targets'] == t)&(prm['signal'] == s) # &(prm['input_noise'] == 0.1)
            filt = (prm['activation']=='RayLou')&(prm['num_targets'] == t)&(prm['signal'] == s)&(prm['num_bits'] == b)
            # plt.scatter(inp_align[filt].mean(1), out_align[filt].mean(1), marker='v', c=prm['num_bits'][filt])
            axs[j,i].scatter(correct_align[0,filt].mean(1), correct_align[1,filt].mean(1), 
                        marker=markers[k], c='b')
        
        axs[j,i].plot(correct_bound[0,:],correct_bound[1,:], 'k--')
        # tplt.square_axis(axs[j,i])
        
        if j < 2:
            axs[j,i].set_xticks([])
        if i > 0:
            axs[j,i].set_yticks([])


