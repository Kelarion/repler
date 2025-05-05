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

import bae
import bae_util
import bae_models


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

task_args = {'task': exp.GridCategories,
             'samps': 12,
             'seed': 0,
             'bits': su.Set([2,3,4]),
             'values': su.Set([3,4,5]),
             'snr': 0 << su.Real(num=3) << 30,
             'dim': None,
             'orth': True,
             'isometric': su.Set([True, False])
             }

# N = su.Set(2**np.arange(4,10))
# task_args = {'task': exp.HierarchicalCategories,
#              'samps': 6,
#              'seed': su.Set([0,1]),
#              'N': N,
#              'bmin':2,
#              'bmax': 4,
#              'snr': 0 << su.Real(num=3) << 30,
#              'dim': None,
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

mod_args = {'model': exp.KBMF,
            'dim_hid': None,
            'decay_rate': (0.95, 1),
            'T0': (5, 1e-5),
            'max_iter': (None, 10),
            'tree_reg': su.Set([0, 1e-1, 1]),
            'period': 5
            }

# mod_args = {'model': exp.BAE,
#             'search': su.Set([True, False]),
#             'decay_rate': (1, 0.9),
#             'T0': (1e-6, 5),
#             'max_iter':(100, None),
#             'pr_reg': su.Set([0, 1e-2]),
#             'tree_reg': 0,
#             'epochs': 10,
#             }

# mod_args = {'model': exp.BAE,
#             'search': (True, True, False),
#             'beta': (1.0, 1.0, 0.0),
#             'decay_rate': (0.9, 1, 1),
#             'T0': (5, 1e-5, 1e-5),
#             'max_iter': (None, 1000, 1000),
#             'pr_reg': su.Set([0, 1e-2]),
#             'tree_reg': su.Set([0, 1]),
#             'epochs': 10,
#             }

### magic
##############################
su.send_to_server(task_args, mod_args, send_remotely, verbose=True)


