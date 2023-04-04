CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'

import socket
import os
import sys
import pickle as pkl
import subprocess

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

sys.path.append(CODE_DIR)
import util
import tasks
import students as stud
import experiments as exp
import grammars as gram
import server_utils 
import plotting as tpl

#%%
def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""

    shapes = np.array([m.shape for m in M])

    buff = shapes.max(0, keepdims=True) - shapes
    
    Z = []
    for enu, row in enumerate(M):
        
        Z.append(
            np.pad( 
                row, [(0,s) for s in buff[enu]], mode='constant', constant_values=np.nan 
                )
            )
        
    return np.array(Z)

#%%

# exp_prm = {'experiment': exp.LogicTask,
# 		   'inp_dics': [tasks.StandardBinary(3).positives],
# 		   'out_dics': [ [(0,1,3,5),(0,2,3,6),(0,1,2,4)], [(0,3,5,6)] ],
# 		   'dim_inp': 3,
# 		   'noise': [0, 0.1],
# 		   }

# # exp_prm = {'experiment': exp.FeedforwardExperiment,
# # 		   'inputs': [tasks.RandomPatterns(4, 100), ],
# # 		   'outputs': [tasks.StandardBinary(2), 
# # 		   				tasks.IndependentCategorical(np.eye(4)),
# # 		   				tasks.HierarchicalLabels([1,2])]
# # 		   }

# net_args = {'model': stud.SimpleMLP,
# 			'num_init': 20,
# 			'width': [3, 4, 5, 6, 7, 8, 9, 10, 20, 100],
# 			'depth': 1,
# 			'p_targ': stud.Bernoulli,
# 			'activation':['Tanh', 'ReLU']
# 			}

# opt_args = {'skip_metrics': True,
# 			'nepoch': 1000,
# 			'verbose': False
# 			}


exp_prm = {'experiment': exp.RandomInputMultiClass,
		   'dim_inp': 128,
		   'num_bits': ( 2, 3, 4, 5), 
		   'num_class': ( 2, 3, 4, 5),
		   'input_noise':0.1,
		   'center':[True, False]}

net_args = {'model': stud.NGroupNetwork,
			'n_per_group': 10,
			'num_k': 2,
			'p_targ': stud.Bernoulli,
			'num_init': 10,
			'inp_weight_distr': pt_util.uniform_tensor,
			'inp_bias_distr': torch.ones,
			'inp_bias_var': 0,
			'out_bias_var': 0,
			'activation':[pt_util.TanAytch(), pt_util.RayLou()]}

opt_args = {'train_outputs': False,
			'train_inp_bias': False,
			'train_out_bias':False,
			# 'init_index': list(range(5)),
			'do_rms': (False, True),
			'nepoch': 2000,
			'lr':(1e-1, 1e-2),
			'bsz':200,
			'verbose': False,
			'skip_rep_metrics':True,
			'skip_metrics':False,
			'conv_tol': 1e-10,
			'metric_period':10}


#%%

all_exp_args, prm = server_utils.get_all_experiments(exp_prm, net_args, opt_args, bool_friendly=True)

all_metrics = {}
inp_kern = []
out_kern = []
all_nets = []
for exp_args in tqdm(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    # this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR+'results/', exp_args['opt_args'])
 
    all_nets.append(this_exp.models)
    
    n_cond = this_exp.inputs.num_cond
    
    x_ = this_exp.inputs(range(n_cond), noise=0).detach().T
    y_ = this_exp.outputs(range(n_cond), noise=0).detach().T
    inp_kern.append(util.dot_product(x_, x_))
    out_kern.append(util.dot_product(y_, y_))
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.metrics.items()}
    for k in all_metrics.keys():
        all_metrics[k].append(this_exp.metrics[k])


for k,v in all_metrics.items():
    all_metrics[k] = pad_to_dense(v)


#%%

# xor = np.array([ len(p) == 1 for p in prm['out_dics']])
is_relu = prm['activation'] =='ReLU'

inp_align = all_metrics['input_alignment']
out_align = all_metrics['target_alignment']

# these_guys = (~is_relu)*(~xor)*(prm['noise']>0)
# these_guys = (~is_relu)
# plt.scatter(inp_align[these_guys,-1,:].mean(-1), out_align[these_guys,-1,:].mean(-1))

these_guys = (prm['activation']=='TanAytch')&(prm['num_bits']==5)&(~prm['center'])&(~prm['do_rms'])
# these_guys = (prm['activation']=='RayLou')&(prm['num_bits']==5)&(~prm['center'])&(~prm['do_rms'])


Kz = all_metrics['hidden_kernel'][these_guys].mean(2)

inp_align = util.centered_kernel_alignment(Kz, pad_to_dense(inp_kern)[these_guys][:,None,...])
out_align = util.centered_kernel_alignment(Kz, pad_to_dense(out_kern)[these_guys][:,None,...])

Kx = pad_to_dense(inp_kern)[these_guys].mean(0)
Ky = pad_to_dense(out_kern)[these_guys].mean(0)

c_xy = util.centered_kernel_alignment(Kx, Ky)

cos_foo = np.linspace(c_xy,1,1000)
ub = c_xy*cos_foo + np.sqrt(1-c_xy**2)*np.sqrt(1-cos_foo**2)
plt.plot(cos_foo, ub, 'k--', zorder=0)

col = cm.viridis(np.arange(inp_align.shape[0])/inp_align.shape[0])
for i in range(inp_align.shape[0]):
    plt.plot(inp_align[i,:-1], out_align[i,:-1], color=(0.7,0.7,0.7), zorder=0)
    plt.scatter(inp_align[i,-1], out_align[i,-1], c=col[i], s=100, zorder=2, marker='v')

tplt.square_axis()

#%% abstraction metrics


