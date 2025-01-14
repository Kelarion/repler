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
N = su.Set(2**np.arange(4,12))
task_args = {'task': exp.SchurCategories,
             'samps': 2,
             'seed': su.Set([0,1,2]),
             'N': N,
             'p':0.5,
             'snr': 0 << su.Real(num=6) << 30,
             'dim': 200,
             'orth': True,
             'scl': 1e-3
             }

# mod_args = {'model': exp.BAER,
#             'max_iter': 200,
#             'decay_rate': su.Set([0.8,0.9]),
#             'T0': su.Set([5,10]),
#             'period': su.Set([1,2]),
#             'penalty': 1e-2
#             }

# N = su.Set(2**np.arange(4,11))
# task_args = {'task': exp.HierarchicalCategories,
#              'samps': 1,
#              'seed': su.Set([0,1,2,3,4,5,6]),
#              'N': N,
#              'bmin':2,
#              'bmax': 4,
#              'snr': 0 << su.Real(num=6) << 30,
#              'dim': 2000,
#              'orth': True
#              }
# N = su.Set(2**np.arange(4,11))
# task_args = {'task': exp.HierarchicalCategories,
#              'samps': 2,
#              'seed': su.Set([0,1,2]),
#              'N': N,
#              'bmin':2,
#              'bmax': 4,
#              'snr': 0 << su.Real(num=6) << 30,
#              'dim': 2000,
#              'orth': True
#              }
# task_args = {'task': exp.SchurTreeCategories,
#              'samps': 2,
#              'seed': su.Set([0,1,2]),
#              'N': N,
#              'p': 0.1,
#              'bmin':2,
#              'bmax': 4,
#              'snr': 0 << su.Real(num=6) << 30,
#              'dim': 2000,
#              'orth': True
#              }

# mod_args = {'model': exp.BAER,
#             'max_iter': 250,
#             'decay_rate': 0.9,
#             'T0': 5,
#             'period': 2,
#             'penalty': su.Set([1e-1, 1e-2]),
#             'dim_hid':3000
#             }
# mod_args = {'model': exp.BAER,
#             'max_iter': 200,
#             'decay_rate': 0.9,
#             'T0': 5,
#             'period': 2,
#             'penalty': 1e-2,
#             'dim_hid':3000
#             }
# mod_args = {'model': exp.BAER,
#             'max_iter': 400,
#             'decay_rate': 0.95,
#             'T0': 10,
#             'period': 2,
#             'penalty': 1e-2,
#             'dim_hid':3000
#             }

# mod_args = {'model': exp.BAER,
#             'max_iter': 200,
#             'decay_rate': 0.8,
#             'T0': 5,
#             'period': 2,
#             'penalty': 1e-2,
#             'dim_hid':su.Set([None, 3000])
#             }

mod_args = {'model': exp.BernVAE,
            'dim_hid': 3000,
            'steps': 250,
            'temp': 2/3,
            'alpha': 1,
            'beta': su.Set([0,1]), 
            'period': 10,
            'scale': 0.5
            }
# mod_args = {'model': exp.BernVAE,
#             # 'dim_hid': 3000,
#             'steps': 200,
#             'temp': 2/3,
#             'alpha': 1,
#             'beta': su.Set([0,1]), 
#             'period': 10,
#             'scale': 0.5
            # }
# mod_args = {'model': exp.BernVAE,
#             'dim_hid': su.Set([None, 3000]),
#             'steps': 200,
#             'temp': 2/3,
#             'alpha': 1,
#             'beta': su.Set([0,1]), 
#             'period': 10,
#             'scale': 0.5
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
        all_metrics[k].append(np.array(this_exp.model.metrics[k]))


for k,v in all_metrics.items():
    all_metrics[k] = su.pad_to_dense(v)


#%%

# plot_this = 'time'
plot_this = 'mean_hamming'
# plot_this = 'mean_mat_ham'
# plot_this = 'mean_norm_ham'
# plot_this = 'median_mat_ham'
# plot_this = 'median_hamming'
# plot_this = 'weighted_hamming'
# plot_this = 'loss'
# plot_this = 'nbs'
normalize = True
# normalize = False

# these = (prm['beta']==0)
these = (prm['dim_hid'] == 3000)&(prm['beta']==0)
# these = (prm['dim_hid'] == 3000)
# these = prm['dim'] == 200
# these = (prm['br']==2)
# these = (prm['br']==2)&(prm['p']==0.1)

these = these&np.isin(prm['snr'], [0,18,30])

# style = '-'
# style = '--'
style = ':'

# marker = '.'
# marker = 'd'
marker = '^'

esenar = np.unique(prm['snr'][these])
cols = cm.viridis(np.linspace(0,1,len(esenar)))
for i,snr in enumerate(esenar):
    deez = these&(prm['snr'] == snr)
    N = np.unique(prm['N'])
    line = util.group_mean((all_metrics[plot_this].mean(1))[deez], prm['N'][deez])
    if normalize:
        line = line/N
    plt.plot(N, line, style, marker=marker, color=cols[i], linewidth=2, markersize=10)

plt.semilogx()
