CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/results/'

import socket
import os
import sys
import pickle as pkl
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import itertools as itt

from matplotlib import pyplot as plt
from matplotlib import cm

sys.path.append(CODE_DIR)
import util
import super_experiments as sxp
import experiments as exp
import server_utils as su
import plotting as tpl

import df_util
import bae_models
import bae_util

#%%

# task_args = {'task': exp.HierarchicalCategories,
#              'samps': 3,
#              'seed': 0,
#              # 'seed': su.Set([0,1,2]),
#              # 'N': N,
#              # 'N': [64, 128, 256, 512],
#              'N': 64,
#              # 'ratio': su.Set([1,5]),
#              'ratio': 10,
#              'bmin':2,
#              'bmax': 4,
#              'snr': 24,
#              # 'snr': 0<<su.Real(12)<<30,
#              # 'snr': 0 << su.Real(7) << 24,
#              'orth': True,
#              # 'nonneg': su.Set([True, False])
#              # 'nonneg': False,
#              'nonneg': True,
#              }

# task_args = {'task': exp.SchurCategories,
#              'samps': 3,
#              # 'samps': 1,
#              'seed': 0,
#              # 'seed': 0,
#              'N': 64,
#              # 'N': 64, 
#              'p':0.5,
#              # 'snr': [0, 10, 24],
#              # 'snr': 0<<su.Real(13)<<24,
#              'snr': 24,
#              # 'ratio': su.Set([1,10]),
#              'ratio': 10,
#              'orth': True,
#              'nonneg': False,
#              # 'nonneg': True,
#              # 'nonneg': su.Set([True, False]),
#              }

task_args = {'task': exp.GridCategories,
             'samps': 9,
             'seed': 0,
             # 'seed': su.Set([0,1]),
             # 'bits': su.Set([2,3,4]),
             'bits': 2,
             # 'values': su.Set([3,4,5]),
             'values': 4,
             # 'snr': 30,
             # 'snr': 0<<su.Real(12)<<30,
             # 'snr': 0 << su.Real(13) << 24,
             'snr': 16,
             # 'ratio': su.Set([1,10]),
             'ratio': 1,
             'orth': True,
             # 'isometric': False,
             'isometric': True,
             # 'isometric': [True, False],
             # 'isometric': su.Set([True, False]),
             # 'nonneg': su.Set([True, False])
             'nonneg': False
             }

mod_args = {'model': exp.KBMF,
            'decay_rate': 0.95,
            'T0': 10,
            # 'T0': 20,
            'max_iter': None,
            # 'tree_reg': (0, 0, 0.1),
            'tree_reg': [0,1e-2, 5e-2, 1e-1, 5e-1],
            # 'tree_reg': 0,
            # 'sparse_reg': (0, 0.1, 0),
            'sparse_reg': 0,
            # 'sparse_reg': [0, 1e-3, 5e-3, 1e-2, 5e-2, 0.1],
            # 'tree_reg': [0, 1],
            'dim_hid': 0.2 << su.Real(13) << 2,
            # 'dim_hid': 2,
            'period': 5,
            }

# mod_args = {'model': exp.SBMF,
#             'ortho': True,
#             # 'ortho': False,
#             # 'nonneg': True,
#             'decay_rate': 0.95,
#             'T0': 10,
#             # 'max_iter': (100, None),
#             # 'sparse_reg': [0, 1e-2, 5e-2, 0.1],
#             # 'tree_reg': su.Set([0, 1e-1]),
#             'tree_reg': [0, 1e-1],
#             'tree_reg': 0,
#             # 'pr_reg': su.Set([0, 1e-2]),
#             # 'pr_reg': [0, 1e-2],
#             # 'pr_reg': [0, 1e-2],
#             'l2_reg': 0,
#             # 'l2_reg': [0, 1e-2],
#             # 'period': 2,
#             'period': 5,
#             # 'dim_hid': None
#             # 'dim_hid': [1, 2],
#             'dim_hid': 0.5 << su.Real(13) << 2,
#             }

# mod_args = {'model': exp.BAE,
#             # 'search': (True, False, False),
#             'search': False,
#             # 'search': [True, False],
#             # 'beta': (1.0, 1.0, 0.0),
#             # 'beta': [0, 0.25, 0.5, 0.75, 1],
#             'beta': 0,
#             # 'decay_rate': (0.9, 1, 1),
#             'decay_rate': 0.9,
#             # 'decay_rate': (0.8, 0.9),
#             # 'T0': (5, 1e-5, 1e-5),
#             # 'T0': (1, 2/3),
#             'T0': 1,
#             # 'max_iter': (None, 1000, 1000),
#             # 'max_iter': 500,
#             # 'period': (10, 1),
#             'period': 20,
#             # 'max_iter': (None, 500),
#             'max_iter': None,
#             # 'pr_reg': su.Set([0, 1e-2]),
#             'pr_reg': 1e-2,
#             # 'pr_reg': [0, 1e-2],
#             # 'sparse_reg': [0, 0.01, 0.05, 0.1],
#             'sparse_reg': 0.01,
#             'tree_reg': 0,
#             # 'batch_size': [1, 256],
#             'batch_size': 256,
#             'dim_hid': 10,
#             }

# mod_args = {'model': exp.NMF2,
#             # 'semi': True,
#             'semi': False,
#             'reg': su.Set([0, 1e-2]),
#             # 'reg': 0,
#             # 'dim_hid': 0.5 << su.Real(13) << 2,
#             'dim_hid': 0.5 << su.Real(13) << 1,
#             }

#%%


all_exp_args, prm = su.get_all_experiments(task_args, mod_args, bool_friendly=True)

all_metrics = {}
for exp_args in tqdm(all_exp_args):
    
    task = exp_args['task_args']
    mod = exp_args['model_args']
    this_exp = sxp.Experiment(task['task'](**task['args']), mod['model'](**mod['args']))
    # this_exp.load_experiment(SAVE_DIR)
    this_exp.run()
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.model.metrics.items()}
    for k in all_metrics.keys():
        if type(this_exp.model.metrics[k][0]) is np.ndarray:
            all_metrics[k].append(su.pad_to_dense(this_exp.model.metrics[k]))
        else:
            all_metrics[k].append(np.array(this_exp.model.metrics[k]))

for k,v in all_metrics.items():
    all_metrics[k] = su.pad_to_dense(v)

# %%

# plot_this = 'time'
# plot_this = 'hamming'
# plot_this = 'norm_hamming'
# plot_this = 'cond_hamming'
# plot_this = 'norm_cond_hamming'
# plot_this = 'loss'
# plot_this = 'binloss'
# plot_this = 'nbs'
plot_this = 'unique_k'

# plot_against = prm['snr']
# plot_against = prm['N']
# plot_against = prm['values']**prm['bits']
# plot_against = 2**prm['bits']
# plot_against = prm['batch_size']
plot_against = prm['dim_hid']
# plot_against = prm['sparse_reg']
# plot_against = prm['tree_reg']
# plot_against = prm['beta']

# splitby = None
# splitby = 'snr'
# splitby = 'dim_hid'
# splitby = 'bits'
# splitby = 'cond'
splitby = 'tree_reg'
# splitby = 'sparse_reg'
# splitby = 'pr_reg'
# splitby = 'batch_size'
# splitby = 'search'
# splitby = 'reg'

# normalize = True
normalize = False

these = np.ones(len(plot_against))>0

# these *= (prm['isometric'])
# these *= ~(prm['nonneg'])
# these *= (prm['reg']==0)
# these *= ~prm['search']
# these *= prm['tree_reg'] == 1
# these *= prm['tree_reg'] == 0.1
# these *= prm['tree_reg'] == 0
these *= prm['tree_reg'] < 5e-1
# these *= prm['tree_reg']  0
# these *= prm['sparse_reg'] == 1e-1
# these *= prm['sparse_reg'] == 0
# these *= prm['pr_reg'] == 1e-2
# these *= prm['pr_reg'] == 0
# these *= prm['l2_reg'] == 0
# these = (prm['ratio'] == 10)
# these = these*(prm['beta']==1)#&(prm['pr_reg']==1e-2)
# these = (prm['ratio'] == 10)&(prm['beta']==0)&(prm['batch_size']==1)&(prm['pr_reg']==0)
# these = (prm['decay_rate'] < 1)&(prm['tree_reg']==1e-2)
# these = (prm['search'])&(prm['pr_reg']>0)&(prm['batch_size']==1)&(prm['beta']>0)
# these *= prm['snr'] == np.unique(prm['snr'])[0]
# these *= prm['beta'] == 0
# these *= prm['reg'] == 0

# these = these*(~prm['isometric'])

# these = these&(prm['values'] == 5)

# these = these&np.isin(prm['snr'], [16,20,24])
# these = these&np.isin(prm['dim_hid'], [1])


style = '-'
# style = '--'
# style = ':'
# style = '-.'

# marker=None
marker = '.'
# marker = 'd'
# marker = '^'

if splitby == 'cond':
    esenar = np.arange(all_metrics[plot_this].shape[-1])
elif splitby is None:
    esenar = [0]
else:
    esenar = np.unique(prm[splitby][these])

cols = cm.viridis(np.linspace(0,1,len(esenar)))
    
for i,snr in enumerate(esenar):
    
    if splitby == 'cond':
        deez = these
        N = np.unique(plot_against[deez])
        vals = np.nanmean(all_metrics[plot_this][...,i], axis=1)[deez]
        errs = np.nanstd(all_metrics[plot_this][...,i], axis=1)[deez]
        line = util.group_mean(vals, plot_against[deez], axis=0)
        ebar = util.group_mean(errs, plot_against[deez], axis=0)
    elif splitby is None:
        deez = these
        N = np.unique(plot_against[deez])
        vals = np.nanmean(all_metrics[plot_this][:,1:], axis=1)[deez]
        errs = np.nanstd(all_metrics[plot_this][:,1:], axis=1)[deez]
        line = util.group_mean(vals, plot_against[deez], axis=0)
        ebar = util.group_mean(errs, plot_against[deez], axis=0)
        
    else:
        deez = these&(prm[splitby] == snr)
    
        N = np.unique(plot_against[deez])
        vals = np.nanmean(all_metrics[plot_this][:,1:], axis=1)[deez]
        errs = np.nanstd(all_metrics[plot_this][:,1:], axis=1)[deez]
        line = util.group_mean(vals, plot_against[deez], axis=0)
        ebar = util.group_mean(errs, plot_against[deez], axis=0)
    if normalize:
        line = line/N
    plt.plot(N, line, style, marker=marker, color=cols[i], linewidth=2, markersize=10)
    plt.errorbar(N, line, yerr=ebar, ecolor=cols[i], 
                 fmt=style, marker=marker, color=cols[i], linewidth=2, markersize=10)

# plt.semilogx()

# def plotmetrics(prm, xaxis, yaxis, ):
    




