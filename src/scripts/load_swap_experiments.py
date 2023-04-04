# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:23:42 2023

@author: mmall
"""


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

    maxlen = max(len(r) for r in M)
    dims = M[0].shape[1:]

    Z = np.zeros((len(M), maxlen, *dims))*np.nan
    for enu, row in enumerate(M):
        Z[enu, :len(row)] = row 
    return Z

#%%

exp_prm = {'experiment':exp.SwapErrors,
		   'T_inp1': 5,
		   'T_inp2': 15,
		   'T_resp': 25,
		   'T_tot': 35,
		   'num_cols': 32,
		   'jitter': 3,
		   'inp_noise': 0.0,
		   'dyn_noise': [0.0, 0.01],
		   'present_len': [1, 3],
		   'test_noise': [0.0, 0.01],
		   'color_func': [util.TrigColors(), util.RFColors()]}

net_args = {'model': stud.MonkeyRNN,
			'dim_hid': 50,
			'p_targ': stud.GausId,
			'p_hid': stud.GausId,
			'nonlinearity':['relu', 'tanh'],
			'fix_encoder':True,
			'beta': [0, 1e-6],
			'fix_decoder': True}

opt_args = {'opt_alg': optim.Adam,
			'init_index': list(range(4)), 
			'skip_metrics': False,
			'nepoch': 10000,
			'verbose': False}


#%%

all_exp_args, prm = server_utils.get_all_experiments(exp_prm, net_args, opt_args, bool_friendly=False)

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
    
    x_ = this_exp.inputs(range(8), noise=0).detach().T
    y_ = this_exp.outputs(range(8), noise=0).detach().T
    inp_kern.append(util.dot_product(x_, x_))
    out_kern.append(util.dot_product(y_, y_))
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.metrics.items()}
    for k in all_metrics.keys():
        all_metrics[k].append(this_exp.metrics[k])

for k,v in all_metrics.items():
    all_metrics[k] = pad_to_dense(v)


#%%

xor = np.array([ len(p) == 1 for p in prm['out_dics']])
is_relu = prm['activation'] =='ReLU'

inp_align = all_metrics['input_alignment']
out_align = all_metrics['target_alignment']

these_guys = (~is_relu)*(~xor)*(prm['noise']>0)
# plt.scatter(inp_align[these_guys,-1,:].mean(-1), out_align[these_guys,-1,:].mean(-1))

Kz = all_metrics['hidden_kernel'][these_guys].mean(2)

inp_align = util.centered_kernel_alignment(Kz, np.array(inp_kern)[these_guys][:,None,...])
out_align = util.centered_kernel_alignment(Kz, np.array(out_kern)[these_guys][:,None,...])

Kx = np.array(inp_kern)[these_guys].mean(0)
Ky = np.array(out_kern)[these_guys].mean(0)

c_xy = util.centered_kernel_alignment(Kx, Ky)

cos_foo = np.linspace(c_xy,1,1000)
ub = c_xy*cos_foo + np.sqrt(1-c_xy**2)*np.sqrt(1-cos_foo**2)
plt.plot(cos_foo, ub, 'k--', zorder=0)

col = cm.viridis(np.arange(inp_align.shape[0])/inp_align.shape[0])
for i in range(inp_align.shape[0]):
    plt.plot(inp_align[i,:-1], out_align[i,:-1], color=(0.7,0.7,0.7), zorder=0)
    plt.scatter(inp_align[i,-1], out_align[i,-1], c=col[i], s=100, zorder=2)

tpl.square_axis()

#%% abstraction metrics


