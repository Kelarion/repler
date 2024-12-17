CODE_DIR = '/home/kelarion/github/repler/src/'
SAVE_DIR = '/mnt/c/Users/mmall/OneDrive/Documents/uni/columbia/main/'

import socket
import os
import sys
import pickle as pkl
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
import itertools as itt

sys.path.append(CODE_DIR)
import util
import super_experiments as sxp
import experiments as exp
import server_utils as su

import distance_factorization as df
import df_util
import df_models


####################################

send_remotely = True
# send_remotely = False

###########################################################################
### Set parameters to iterate over ########################################
###########################################################################

# N = su.Set(2**np.arange(4,13))
# task_args = {'task': exp.SchurCategories,
#              'samps': 6,
#              'seed': su.Set([1,2,3]),
#              'N': N,
#              'p':0.5,
#              'snr': 0 << su.Real(num=6) << 30,
#              'dim': su.Set([200,1000]),
#              'orth': True
#              }
# mod_args = {'model': exp.BAE,
#             'eps': 1e-2,
#             'tol': 0.2,
#             'lr': su.Set([0, 0.1]),
#             'whiten': True,
#             'dim_hid': 1000,
#             'max_iter': 500,
#             'alpha': 0.88,
#             'beta': 5
#             }

# N = su.Set(2**np.arange(4,10))
# # task_args = {'task': exp.HierarchicalCategories,
# #              'samps': 5,
# #              'seed': su.Set([0,1,2,3]),
# #              'N': N,
# #              'bmin':2,
# #              'bmax': 4,
# #              'snr': 0 << su.Real(num=6) << 30,
# #              'dim': 2000,
# #              'orth': True
# #              }
# task_args = {'task': exp.SchurTreeCategories,
#              'samps': 5,
#              'seed': su.Set([0,1,2,3]),
#              'N': N,
#              'p': su.Set([0.1, 0.5]),
#              'bmin':2,
#              'bmax': 4,
#              'snr': 0 << su.Real(num=6) << 30,
#              'dim': 2000,
#              'orth': True
#              }

# mod_args = {'model': exp.SBMF,
#             'beta': 1e-5,
#             'eps': 0.5,
#             'tol': 1e-2,
#             'pimin': 0.1,
#             'br': su.Set([1,2]),
#             'order': 'given',
#             'reg': 'sparse'
#             }

N = su.Set(2**np.arange(4,12))
task_args = {'task': exp.SchurCategories,
             'samps': 6,
             'seed': 1,
             'N': N,
             'p':0.5,
             'snr': 0 << su.Real(num=6) << 30,
             'dim': 200,
             'orth': True,
             'scl': 1e-3
             }
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
#             'max_iter': 200,
#             'decay_rate': 0.8,
#             'T0': 5,
#             'period': 2,
#             'penalty': 1e-2,
#             'dim_hid':su.Set([None, 3000])
#             }
# mod_args = {'model': exp.BernVAE,
#             'dim_hid': su.Set([None, 3000]),
#             'steps': 200,
#             'temp': 2/3,
#             'alpha': 1,
#             'beta': su.Set([0,1]), 
#             'period': 10,
#             'scale': 0.5
#             }

# task_args = {'task': exp.SchurCategories,
#              'samps': 6,
#              'seed': 1,
#              'N': 16,
#              'p':0.5,
#              'snr': su.Set([30]),
#              'dim': 200,
#              'orth': True,
#              'scl': 1e-3
#              }

# mod_args = {'model': exp.BAER,
#             'max_iter': 100,
#             'decay_rate': 0.8,
#             'T0': 10,
#             'period': 5,
#             'penalty':1e-2
#             }

mod_args = {'model': exp.AsymBAE,
            'max_iter':(50,100,500),
            'decay_rate':(1, 0.8, 0.95),
            'T0':(1e-6, 5, 5),
            'period':1,
            'penalty': su.Set([0,1]),
            }

### magic
##############################
su.send_to_server(task_args, mod_args, send_remotely, verbose=True)