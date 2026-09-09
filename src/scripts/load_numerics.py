CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/results/'

import socket
import os
import sys
import pickle as pkl
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
import itertools as itt

from matplotlib import pyplot as plt
from matplotlib import cm

sys.path.append(CODE_DIR)
import util
import super_experiments as sxp
import new_bae_experiments as nbx
import experiments as exp
import server_utils as su
import plotting as tpl

#%%

N = su.Set(2**np.arange(4,12))
task_args = {'task': nbx.StructuredCats,
             # 'spec': 'grid5(d=2)',
             'spec': 'tree5',
             'samps': 3,
             # 'temp': [1e-2, 1e-1, 1],
             'temp': [1e-2, 1],
             # 'samps': 1,
             # 'seed': su.Set([0,1,2]),
             'seed': 0,
             'N': N,
             # 'N': 64, 
             # 'p':0.5,
             # 'snr': 0<<su.Real(7)<<13,
             'snr': 15,
             # 'ratio': 0.5<<su.Integer(num=5)<<2,
             'ratio': 10,
             'orth': True,
             'nonneg': False,
             # 'nonneg': su.Set([True, False]),
             }

mod_args = {'model': nbx.NewBMF,
            'kind': 'BiPCA',
            'decay_rate': 0.88,
            'T0': 10,
            # 'T0': 20,
            'max_iter': None,
            # 'tree_reg': (0, 0, 1e-1),
            # 'sparse_reg': (0, 1, 0),
            'sparse_reg': 1,
            # 'tree_reg': 0,
            'tree_reg': [0, 1, 10], 
            'dim_hid': 0.5 << su.Real(7) << 2,
            # 'dim_hid': 2,
            'period': 10,
            'min_temp': 1,
            'J_lr': [0,1e-2],
            'lr': 1e-1,
            'folds': 10,
            'n_chains': 8,
            }


# all_exp_args, prm = su.get_all_experiments(task_args, mod_args, bool_friendly=True)
all_exp_args, prm = su.load_experiments(task_args, mod_args, SAVE_DIR)

# all_metrics = {}
# for exp_args in tqdm(all_exp_args):
    
#     task = exp_args['task_args']
#     mod = exp_args['model_args']
#     this_exp = sxp.Experiment(task['task'](**task['args']), mod['model'](**mod['args']))
#     this_exp.load_experiment(SAVE_DIR)
    
#     if len(all_metrics) == 0:
#         all_metrics = {k:[] for k,v in this_exp.model.metrics.items()}
#     for k in all_metrics.keys():
#         if type(this_exp.model.metrics[k][0]) is np.ndarray:
#             all_metrics[k].append(su.pad_to_dense(this_exp.model.metrics[k]))
#         else:
#             all_metrics[k].append(np.array(this_exp.model.metrics[k]))

# for k,v in all_metrics.items():
#     all_metrics[k] = su.pad_to_dense(v)

# prm = pd.DataFrame(prm)

all_metrics = {}
for exp_args in tqdm(all_exp_args):
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in exp_args['metrics'].items()}
    for k in all_metrics.keys():
        if type(exp_args['metrics'][k][0]) is np.ndarray:
            all_metrics[k].append(su.pad_to_dense(exp_args['metrics'][k]))
        else:
            all_metrics[k].append(np.array(exp_args['metrics'][k]))

for k,v in all_metrics.items():
    all_metrics[k] = su.pad_to_dense(v)

prm = pd.DataFrame(prm)

#%%

# plot_this = 'time'
# plot_this = 'hamming'
# plot_this = 'norm_hamming'
# plot_this = 'cond_hamming'
# plot_this = 'norm_cond_hamming'
# plot_this = 'losses'
# plot_this = 'nbs'
# plot_this = 'unique_k'
plot_this = 'j_cos'
# plot_this = 'j_ged'
# plot_this = 'test_ll'
# plot_this = 'train_ll'

# plot_against = prm['snr']
plot_against = prm['N']
# plot_against = prm['values']**prm['bits']
# plot_against = 2**prm['bits']
# plot_against = prm['batch_size']
# plot_against = prm['dim_hid']

splitby = ['tree_reg']
# splitby = []
# splitby = ['N']
# splitby = ['temp']
# splitby = ['dim_hid']
# splitby = 'snr'
# splitby = 'dim_hid'
# splitby = 'bits'
# splitby = 'cond'
# splitby = ['snr', 'temp']
# splitby = ['J_lr']

# normalize = True
normalize = False

these = np.ones(len(plot_against))>0

# these *= prm['N'] == 32
these *= prm['J_lr'] > 0
these *= prm['dim_hid'] == 1
these *= prm['temp'] < 1
# these *= prm['folds'] > 0
# these *= (prm['isometric'])
# these *= ~(prm['nonneg'])
# these *= (prm['reg']==0)
# these *= prm['tree_reg'] == 0
# these *= prm['tree_reg'] == 0.1
# these *= prm['tree_reg'] == 0.1
# these *= prm['sparse_reg'] > 0
# these = (prm['ratio'] == 10)
# these = these*(prm['beta']==1)#&(prm['pr_reg']==1e-2)
# these = (prm['ratio'] == 10)&(prm['beta']==0)&(prm['batch_size']==1)&(prm['pr_reg']==0)
# these = (prm['decay_rate'] < 1)&(prm['tree_reg']==1e-2)
# these = (prm['search'])&(prm['pr_reg']>0)&(prm['batch_size']==1)&(prm['beta']>0)
# these *= prm['snr'] == np.unique(prm['snr'])[0]
# these *= prm['snr'] > 11

# these = these*(~prm['isometric'])

# these = these&(prm['values'] == 5)

# these = these&np.isin(prm['snr'], [30])

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
else:
    esenar = np.unique(prm[splitby][these], axis=0)

cols = cm.viridis(np.linspace(0,1,len(esenar)))


for i,snr in enumerate(esenar):
    
    if splitby == 'cond':
        deez = these
        N = np.unique(plot_against[deez])
        vals = np.nanmean(all_metrics[plot_this][...,i], axis=1)[deez]
        errs = np.nanstd(all_metrics[plot_this][...,i], axis=1)[deez]
        line = util.group_mean(vals, plot_against[deez], axis=0)
        ebar = util.group_mean(errs, plot_against[deez], axis=0)
    else:
        deez = these&(prm[splitby] == snr).all(1)
    
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

plt.semilogx()

# def plotmetrics(prm, xaxis, yaxis, ):
    



