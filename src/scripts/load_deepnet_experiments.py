
CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/results/'

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

# import umap
from cycler import cycler

# from pypoman import compute_polytope_vertices, compute_polytope_halfspaces
# import cvxpy as cvx
# import polytope as pc
# from hsnf import column_style_hermite_normal_form

# my code
import students as stud
import assistants
import experiments as exp
import util
import pt_util
import tasks
import server_utils as su
import plotting as tplt
import grammars as gram
import dichotomies as dics


#%%

d = su.Set([32]) 								# number of clusters
k = su.Set([1, np.log2(d)]) 					# rank of targets
c = su.Real(num=3) 								# input-target alignment
r = su.Set([np.log2(d) , d - 1 - k])			# input rank
# seed = su.Integer(step=1)						# random seed
# scale = su.Set([0*(seed==0) + 0.5*(seed>0)]) 	# kernel distribution scale

exp_prm = {'experiment': exp.RandomKernelClassification,
			'num_points': d,
			'num_targets': k,
			'signal': 0 << c  << 1,
			'seed': 0,
			'scale': 0,
			'dim_inp': 100,
			'input_noise': 1,
			'max_rank': r
			}


net_args = {'model': stud.SimpleMLP,
			'num_init': 10,
			'width': 128,
			'depth': 1,
			'p_targ': stud.Bernoulli,
			'activation': ['ReLU', 'Tanh']
			}


opt_args = {'skip_metrics': True,
			'nepoch': 1000,
			'verbose': False,
			'lr': 1e-3
			}

#%% Compute input alignment 

# test_noise = None
test_noise = 0.5

all_exp_args, prm = su.get_all_experiments(exp_prm, net_args, opt_args)


all_metrics = {}


clf = svm.LinearSVC()

PS = []
CCGP = []
inp_align = []
out_align = []
inp_align_indiv = []
out_align_indiv = []
# dec = []
skew = []
PR = []
train_loss = []
sanity = []
sanity2 = []
for exp_args in tqdm(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    # this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR, exp_args['opt_args'])
    
    info = this_exp.load_info(SAVE_DIR)
    
    n_cond = this_exp.inputs.num_cond
    
    x_ = this_exp.inputs(np.arange(n_cond), 0).T
    y_ = this_exp.outputs(np.arange(n_cond), 0).T
    
    cond = np.random.choice(range(n_cond), 5000)
    x_noise = this_exp.inputs(cond, noise=test_noise).T
    # x_noise = this_exp.inputs(cond).T
    y_noise = (this_exp.outputs(cond)).T
    
    Kx = info['input_kernel'][0]
    # Kx = util.dot_product(x_-x_.mean(1,keepdims=True), x_-x_.mean(1,keepdims=True))
    Ky = util.dot_product(y_-y_.mean(1,keepdims=True), y_-y_.mean(1,keepdims=True))

    skew.append( np.sum(Ky*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Ky*Ky)) )
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.metrics.items()}
    for k in all_metrics.keys():
        all_metrics[k].append(this_exp.metrics[k])
        
    Kz = (this_exp.metrics['hidden_kernel'][-1])
    
    inp_align.append(util.centered_kernel_alignment(Kz.mean(0),Kx))
    out_align.append(util.centered_kernel_alignment(Kz.mean(0),Ky))
    
    inp_align_indiv.append(util.centered_kernel_alignment(Kz,Kx))
    out_align_indiv.append(util.centered_kernel_alignment(Kz,Ky))
    
 
    # PR.append((np.trace(Kz)**2)/np.trace(Kz@Kz))
    
    # PS.append(np.array([dics.efficient_parallelism(yy.numpy(), K=Kz, aux_func='distsum') for yy in y_]))

    
    train_loss.append(this_exp.metrics['train_loss'][-1])

inp_align = np.array(inp_align)
out_align = np.array(out_align)
CCGP = np.squeeze(CCGP)
# PS = np.array(PS)
skew = np.array(skew)
PR = np.array(PR)

# lindim = np.array(all_metrics['linear_dim'])[:,-1,:]
# t_perf = np.squeeze(np.nanmean(util.pad_to_dense(all_metrics['test_perf'])[:,-1,...], axis=-1))
# for k,v in all_metrics.items():
#     all_metrics[k] = util.pad_to_dense(v)

prlsm = np.array([np.mean(pp) for pp in PS])


#%% input vs target alignment plots


cos_foo = np.linspace(0,1,1000)

plot_params = ['num_targets', 'signal']
color_params = ['max_rank']

markers = ['o', 'v', 's', '+']
cols = ['b','r','g']

targ_vals = np.unique(prm['num_targets'])
# targ_vals = [5]
sig_vals = np.unique(prm['signal'])
inp_vals = np.unique(prm['max_rank'])

axs = tplt.hierarchical_labels([targ_vals], [sig_vals],    
                                 row_names=['target rank'], col_names=['signal'],
                                 fontsize=13, wmarg=0.3, hmarg=0.1)

filt = prm['activation'] == 'Tanh'

for k,c in enumerate(inp_vals):
    for i,t in enumerate(targ_vals):
        for j,s in enumerate(sig_vals):
            
            c_xy = np.sqrt(t*s/(32 - 1))
            
            phi = (np.pi/2 -np.arccos(c_xy))/2  
            basis = np.array([[np.cos(phi),np.cos(np.pi/2-phi)],[np.sin(phi),np.sin(np.pi/2-phi)]])

            ub = basis@np.stack([cos_foo, np.sqrt(1-cos_foo**2)])
            
            axs[i,j].plot(ub[0], ub[1], 'k--')
            
            these = filt*(prm['num_targets']==t)*(prm['signal']==s)*(prm['max_rank']==c)
            
            for idx in np.where(these)[0]:
                
                axs[i,j].plot(inp_align[idx], out_align[idx], 
                              '-', marker=markers[k])
                axs[i,j].plot(inp_align_indiv[idx].T, out_align_indiv[idx].T, 
                              '--', marker=markers[k], alpha=0.6)
                
            tplt.square_axis(axs[i,j])
        








