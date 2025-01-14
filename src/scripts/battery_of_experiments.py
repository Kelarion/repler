
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

d = su.Set([32]) 				# number of clusters
k = su.Set([1, np.log2(d)]) 	# rank of targets
c = su.Real(num=3) 				# input-target alignment
r = su.Set([np.log2(d) + k*(c>1e-6), d - 1 - k*(c<1e-6)]) # input rank
seed = su.Integer(step=1)		# random seed
scale = su.Set([0*(seed==0) + 0.5*(seed>0)]) # kernel distribution scale

exp_prm = {'experiment': exp.RandomKernelClassification,
			'num_points': d,
			'num_targets': k,
			'alignment': 0 << c  << np.sqrt(k/(d - 1)),
			'seed': 0 << seed << 12,
			'scale': scale,
			'dim_inp': 100,
			'input_noise': 1,
			'max_rank': r
			}


net_args = {'model': stud.SimpleMLP,
			'num_init': 10,
			'width': 128,
			'depth': [1,5,10],
			'p_targ': stud.Bernoulli,
			'activation': ['ReLU', 'Tanh']
			}


opt_args = {'skip_metrics': True,
			'nepoch': 1000,
			'verbose': False,
			'lr': 1e-2
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
# dec = []
skew = []
PR = []
train_loss = []
for exp_args in tqdm(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    # this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR, exp_args['opt_args'])
    
    n_cond = this_exp.inputs.num_cond
    
    x_ = this_exp.inputs(np.arange(n_cond), 0).T
    y_ = this_exp.outputs(np.arange(n_cond), 0).T
    
    cond = np.random.choice(range(n_cond), 5000)
    x_noise = this_exp.inputs(cond, noise=test_noise).T
    # x_noise = this_exp.inputs(cond).T
    y_noise = (this_exp.outputs(cond)).T
    
    Kx = util.dot_product(x_-x_.mean(1,keepdims=True), x_-x_.mean(1,keepdims=True))
    Ky = util.dot_product(y_-y_.mean(1,keepdims=True), y_-y_.mean(1,keepdims=True))

    skew.append( np.sum(Ky*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Ky*Ky)) )
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.metrics.items()}
    for k in all_metrics.keys():
        all_metrics[k].append(this_exp.metrics[k])
        
    
    Kz = (this_exp.metrics['hidden_kernel'][-1].mean(0))
    
    inp_align.append(util.centered_kernel_alignment(Kz,Kx))
    out_align.append(util.centered_kernel_alignment(Kz,Ky))
    
    PR.append((np.trace(Kz)**2)/np.trace(Kz@Kz))
    
    PS.append(np.array([dics.efficient_parallelism(yy.numpy(), K=Kz, aux_func='distsum') for yy in y_]))

    # individual models
    # ccgp = []
    # # loss = []
    # for i,model in enumerate(this_exp.models):
        
    #     # loss.append(this_exp.metrics['train_loss'][-1])
        
    #     # z_ = model(x_.T)[1].detach().numpy().T
    #     # z_noise = model(x_noise.T)[1].detach().numpy().T     
        
    #     # this_ccg = [np.mean(dics.efficient_ccgp(yy, svm.LinearSVC(), cond=cond, z=z_noise, num_pairs=1, max_ctx=100)) for yy in y_noise]

    #     # ccgp.append(np.mean(this_ccg))

    # CCGP.append(ccgp)
    
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

#%%

all_filt = prm['input_noise'] == 1

for i,d in enumerate(np.unique(prm['num_bits'])):
    for j,k in enumerate(np.unique(prm['num_targets'][prm['num_bits'] == d])):
        
        plt.subplot(3, 5, 5*(i) + (j) +1)
        # plt.ylim([0.25, 1.05])
        
        filt = all_filt&(prm['activation']=='TanAytch')&(prm['num_targets'] == k)&(prm['num_bits'] == d)
        
        # plt.scatter(skew[filt], out_align[filt], c='r')
        plt.plot(skew[filt], out_align[filt], 'r')
        # plt.scatter(skew[filt], prlsm[filt], c='r')
        # plt.plot(skew[filt], prlsm[filt], c='r')
        # plt.plot(skew[filt], CCGP[filt].mean(1), c='r')
        # plt.scatter(skew[filt], lindim[filt].mean(1), c='r')
        
        # plt.scatter(skew[filt], t_perf[filt].mean(1), c='r')
    
        filt = all_filt&(prm['activation']=='RayLou')&(prm['num_targets'] == k)&(prm['num_bits'] == d) 
        # plt.scatter(skew[filt], out_align[filt], c='b')
        plt.plot(skew[filt], out_align[filt], 'b')
        # # plt.scatter(skew[filt], prlsm[filt], c='b')
        # # plt.plot(skew[filt], prlsm[filt], c='b')
        # plt.plot(skew[filt], CCGP[filt].mean(1), c='b')
        # # plt.scatter(skew[filt], lindim[filt].mean(1), c='b')
        # # plt.scatter(skew[filt], t_perf[filt].mean(1), c='b')
         
        
        # filt = all_filt&(prm['activation']=='RayLou1.0')&(prm['num_targets'] == k)&(prm['num_bits'] == d) 
        # plt.plot(skew[filt], out_align[filt], color=(0.75,0,0.5), linestyle='--')

        # filt = all_filt&(prm['activation']=='RayLou1.5')&(prm['num_targets'] == k)&(prm['num_bits'] == d) 
        # plt.plot(skew[filt], out_align[filt], color=(0.5,0,0.5), linestyle='--')

        # filt = all_filt&(prm['activation']=='RayLou2.0')&(prm['num_targets'] == k)&(prm['num_bits'] == d) 
        # plt.plot(skew[filt], out_align[filt], color=(0.25,0,0.75), linestyle='--')

        
        # plt.plot(skew[filt&(prm['inp_bias_shift'] == 1)], out_align[filt&(prm['inp_bias_shift'] == 1)], 
        #           color=(0.5,0,0.5), linestyle='--')
        
        # plt.plot(skew[filt&(prm['inp_bias_shift'] == 0.5)], out_align[filt&(prm['inp_bias_shift'] == 0.5)], 
        #           color=(0.25,0,0.75), linestyle='--')
        
        # plt.plot(skew[filt&(prm['inp_bias_shift'] == -0.5)], out_align[filt&(prm['inp_bias_shift'] == -0.5)], 
        #           color=(0,0.25,0.75), linestyle='--')
        
        # plt.plot(skew[filt&(prm['inp_bias_shift'] == -1)], out_align[filt&(prm['inp_bias_shift'] == -1)], 
        #           color=(0,0.5,0.5), linestyle='--')


#%% input vs target alignment plots


cos_foo = np.linspace(0,1,1000)

phi = (np.pi/2 -np.arccos(skew))/2  # re-align it with the orthogonal case
basis = np.array([[np.cos(phi),np.cos(np.pi/2-phi)],[np.sin(phi),np.sin(np.pi/2-phi)]])

ub = basis@np.stack([cos_foo, np.sqrt(1-cos_foo**2)])

# ub = np.sqrt(1-cos_foo**2)
# correction = np.linalg.inv(basis.transpose((2,0,1)))
# correct_align = np.einsum('lik,klj->ilj', correction, np.stack([inp_align,out_align]))
# correct_bound = np.stack([cos_foo, ub])

plot_params = ['num_targets', 'alignment']
color_params = ['max_rank']

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


