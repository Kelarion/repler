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

import distance_factorization as df
import df_util
import df_models

#%%


# N = su.Set(2**np.arange(4,11))
# # task_args = {'task': exp.SchurCategories,
# #              'samps': 6,
# #              'seed': su.Set([0,1,2]),
# #              'N': N,
# #              'p':0.5,
# #              'snr': 30,
# #              'ratio': 10,
# #              'orth': True,
# #              'scl': 1e-3
#              # }
# task_args = {'task': exp.SchurCategories,
#              'samps': 12,
#              'seed': 0,
#              'N': N,
#              'p':0.5,
#              'snr': 30,
#              'ratio': 10,
#              'orth': True,
#              'scl': 1e-3
#              }
task_args = {'task': exp.GridCategories,
             'samps': 6,
             'seed': su.Set([0,1]),
             'bits': su.Set([2,3,4]),
             'values': su.Set([3,4,5]),
             'snr': 30,
             'ratio': 10,
             'orth': True,
             'isometric': su.Set([True, False])
             }

# mod_args = {'model': exp.BAE,
#             'search': (True, True, False),
#             'beta': (1.0, 1.0, 0.0),
#             'decay_rate': (0.9, 1, 1),
#             'T0': (5, 1e-5, 1e-5),
#             'max_iter': (None, 1000, 1000),
#             'pr_reg': su.Set([0, 1e-2]),
#             'tree_reg': 0,
#             'batch_size':su.Set([1, 32, 64, 128]),
#             'epochs': 20,
#             

# mod_args = {'model': exp.BAE,
#             'search': (True, False, False),
#             'beta': (1.0, 1.0, 0.0),
#             'decay_rate': (0.9, 1, 1),
#             'T0': (5, 1e-5, 1e-5),
#             'max_iter': (None, 1000, 1000),
#             'pr_reg': su.Set([0, 1e-2]),
#             'tree_reg': 0,
#             'batch_size':su.Set([1, 64]),
#             'epochs': 10,
#             }

mod_args = {'model': exp.SBMF,
            'ortho': True,
            'decay_rate': 0.95,
            'T0': 5,
            # 'max_iter': (100, None),
            # 'sparse_reg': su.Set([0, 1e-2]),
            'tree_reg': su.Set([0, 1e-2, 1e-1]),
            # 'pr_reg': su.Set([0, 1e-2]),
            'period': 2,
            'dim_hid': None
            }

# mod_args = {'model': exp.KBMF,
#             'dim_hid': None,
#             'decay_rate': 0.95,
#             'T0': 10,
#             'max_iter': None,
#             'tree_reg': su.Set([1e-1, 1]),
#             'period': 2
#             }


all_exp_args, prm = su.get_all_experiments(task_args, mod_args, bool_friendly=True)

all_metrics = {}
for exp_args in tqdm(all_exp_args):
    
    task = exp_args['task_args']
    mod = exp_args['model_args']
    this_exp = sxp.Experiment(task['task'](**task['args']), mod['model'](**mod['args']))
    this_exp.load_experiment(SAVE_DIR)
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.model.metrics.items()}
    for k in all_metrics.keys():
        if type(this_exp.model.metrics[k][0]) is np.ndarray:
            all_metrics[k].append(su.pad_to_dense(this_exp.model.metrics[k]))
        else:
            all_metrics[k].append(np.array(this_exp.model.metrics[k]))

for k,v in all_metrics.items():
    all_metrics[k] = su.pad_to_dense(v)

#%%

# plot_this = 'time'
# plot_this = 'hamming'
plot_this = 'norm_hamming'
# plot_this = 'cond_hamming'
# plot_this = 'norm_cond_hamming'
# plot_this = 'mean_mat_ham'
# plot_this = 'mean_norm_hamming'
# plot_this = 'median_mat_ham'
# plot_this = 'median_hamming'
# plot_this = 'weighted_hamming'
# plot_this = 'loss'
# plot_this = 'nbs'

# plot_against = prm['N']
plot_against = prm['values']**prm['bits']
# plot_against = 2**prm['bits']
# plot_against = prm['batch_size']
# plot_against = prm['dim_hid']

# normalize = True
normalize = False

these = (prm['decay_rate'] < 1)&(prm['tree_reg']==1e-2)
# these = (prm['search'])&(prm['pr_reg']>0)&(prm['batch_size']==1)&(prm['beta']>0)
these = these*(prm['isometric'])

# these = these&(prm['values'] == 5)

# these = these&np.isin(prm['snr'], [30])

style = '-'
# style = '--'
# style = ':'
# style = '-.'

marker = '.'
# marker = 'd'
# marker = '^'

esenar = np.unique(prm['snr'][these])
cols = cm.viridis(np.linspace(0,1,len(esenar)))
for i,snr in enumerate(esenar):
    deez = these&(prm['snr'] == snr)
    N = np.unique(plot_against[these])
    vals = np.nanmean(all_metrics[plot_this], axis=-1)[deez]
    line = util.group_mean(vals, plot_against[deez], axis=0)
    if normalize:
        line = line/N
    plt.plot(N, line, style, marker=marker, color=cols[i], linewidth=2, markersize=10)

plt.semilogx()

# def plotmetrics(prm, xaxis, yaxis, ):
    



