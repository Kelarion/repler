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

# N = su.Set(2**np.arange(4,12))
# task_args = {'task': exp.SchurCategories,
#               'samps': 6,
#               'seed': su.Set([1,2,3]),
#               'N': N,
#               'p':0.5,
#               'snr': 0 << su.Real(num=6) << 30,
#               'dim': su.Set([200,1000]),
#               'orth': True
#               }
# mod_args = {'model': exp.BAE,
#             'eps': 1e-2,
#             'tol': 0.2,
#             'lr': 0.1,
#             'whiten': True,
#             'dim_hid': 1000,
#             'max_iter': 500,
#             'alpha': 0.88,
#             'beta': 5
#             }
# N = su.Set(2**np.arange(4,10))
# # task_args = {'task': exp.HierarchicalCategories,
# #               'samps': 5,
# #               'seed': su.Set([0,1,2,3]),
# #               'N': N,
# #               'bmin':2,
# #               'bmax': 4,
# #               'snr': 0 << su.Real(num=6) << 30,
# #               'dim': 2000,
# #               'orth': True
# #               }
# task_args = {'task': exp.SchurTreeCategories,
#               'samps': 5,
#               'seed': su.Set([0,1,2,3]),
#               'N': N,
#               'p': su.Set([0.1,0.5]),
#               'bmin':2,
#               'bmax': 4,
#               'snr': 0 << su.Real(num=6) << 30,
#               'dim': 2000,
#               'orth': True
#               }

# mod_args = {'model': exp.SBMF,
#             'beta': 1e-5,
#             'eps': 0.5,
#             'tol': 1e-2,
#             'pimin': 0.1,
#             'br': su.Set([1,2]),
#             'order': 'given',
#             'reg': 'sparse'
#             }

# N = su.Set(2**np.arange(4,12))
# task_args = {'task': exp.SchurCategories,
#              'samps': 6,
#              'seed': 1,
#              'N': N,
#              'p':0.5,
#              'snr': 0 << su.Real(num=6) << 30,
#              'dim': 200,
#              'orth': True,
#              'scl': 1e-3
#              }
# N = su.Set(2**np.arange(4,12))
# task_args = {'task': exp.SchurCategories,
#              'samps': 6,
#              'seed': 1,
#              'N': N,
#              'p':0.5,
#              'snr': 0 << su.Real(num=6) << 30,
#              'dim': 200,
#              'orth': True,
#              'scl': 1e-3
#              }

# b = su.Set(np.arange(4,12))
# task_args = {'task': exp.CubeCategories,
#              'samps': 12,
#              'seed': 0,
#              'bits': b,
#              'snr': 0 << su.Real(num=3) << 30,
#              'dim': None,
#              'orth': True,
#              }

# task_args = {'task': exp.GridCategories,
#              'samps': 12,
#              'seed': 0,
#              'bits': su.Set([2,3,4]),
#              'values': su.Set([3,4,5]),
#              'snr': 0 << su.Real(num=3) << 30,
#              'dim': None,
#              'orth': True,
#              'isometric': su.Set([True, False])
#              }


N = su.Set(2**np.arange(4,10))
task_args = {'task': exp.HierarchicalCategories,
             'samps': 6,
             'seed': su.Set([0,1]),
             'N': N,
             'bmin':2,
             'bmax': 4,
             'snr': 0 << su.Real(num=3) << 30,
             'dim': None,
             'orth': True
             }

# mod_args = {'model': exp.SBMF,
#             'ortho': True,
#             'decay_rate': (1, 0.9),
#             'T0': (1e-6, 5),
#             'max_iter': (100, None),
#             'sparse_reg': su.Set([0, 1e-2]),
#             'tree_reg': 0,
#             # 'pr_reg': su.Set([0, 1e-2]),
#             'period': 2
#             }

# mod_args = {'model': exp.KBMF,
#             'dim_hid': su.Set([None, 200]),
#             'decay_rate': 0.9,
#             'T0': 5,
#             'max_iter': None,
#             'tree_reg': su.Set([0, 1e-1]),
#             'period': 2
#             }

mod_args = {'model': exp.KBMF,
            'dim_hid': None,
            'decay_rate': (0.95, 1),
            'T0': (5, 1e-5),
            'max_iter': (None, 10),
            'tree_reg': su.Set([0, 1e-1, 1]),
            'period': 5
            }

# mod_args = {'model': exp.BAE,
#             'search': (True, False),
#             'beta': (1.0, 0.0),
#             'decay_rate': 0.9,
#             'T0': 5,
#             'max_iter':None,
#             'pr_reg': su.Set([0, 1e-2]),
#             'tree_reg': 0,
#             'epochs': 10,
#             }

# mod_args = {'model': exp.BAE,
#             'search': (True, False),
#             'beta': (1.0, 0.0),
#             'decay_rate': 0.9,
#             'T0': 5,
#             'max_iter':None,
#             'pr_reg': su.Set([0, 1e-2]),
#             'tree_reg': su.Set([0, 1]),
#             'epochs': 20,
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

plot_against = prm['N']
# plot_against = prm['values']**prm['bits']
# plot_against = 2**prm['bits']

# normalize = True
normalize = False
# these = (prm['decay_rate']<1)&(prm['tree_reg']==0)&(~prm['isometric'])
these = (prm['decay_rate']<1)&(prm['tree_reg']==1)
# these = (prm['search'])&(prm['decay_rate']<1)&(prm['pr_reg']>0)&(prm['tree_reg']==0)
# these = (prm['beta']==0)
# these = (prm['dim_hid'] == 3000)&(prm['beta']==0)
# these = (prm['dim_hid'] == 3000)
# these = prm['dim'] == 200
# these = (prm['br']==2)
# these = (prm['br']==2)&(prm['p']==0.1)

# these = these&(prm['values'] == 5)

# these = these&np.isin(prm['snr'], [30])

style = '-'
# style = '--'
# style = ':'

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
