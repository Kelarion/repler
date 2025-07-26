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


# N = su.Set(2**np.arange(4,12))
# task_args = {'task': exp.SchurCategories,
#              'samps': 6,
#              # 'samps': 1,
#              'seed': su.Set([0,1,2]),
#              # 'seed': 0,
#              # 'N': N,
#              'N': 64, 
#              'p':0.5,
#              'snr': 0<<su.Real(12)<<30,
#              # 'snr': 30,
#              # 'ratio': 0.5<<su.Integer(num=5)<<2,
#              'ratio': 10,
#              'orth': True,
#              # 'nonneg': False,
#              'nonneg': su.Set([True, False]),
#              }

task_args = {'task': exp.GridCategories,
             'samps': 9,
             'seed': 0,
             # 'seed': su.Set([0,1]),
             # 'bits': su.Set([2,3,4]),
             'bits': 3,
             # 'values': su.Set([3,4,5]),
             'values': 5,
             # 'snr': 30,
             'snr': 0<<su.Real(12)<<30,
             # 'ratio': su.Set([1,10]),
             'ratio': 1,
             'orth': True,
             'isometric': su.Set([True, False]),
             # 'nonneg': su.Set([True, False])
             'nonneg': False
             }

# N = su.Set(2**np.arange(4,10))
# task_args = {'task': exp.HierarchicalCategories,
#              'samps': 9,
#              'seed': 0,
#              # 'seed': su.Set([0,1,2]),
#              # 'N': N,
#              'N': 64,
#              # 'ratio': su.Set([1,5]),
#              'ratio': 1,
#              'bmin':2,
#              'bmax': 4,
#              # 'snr': 30,
#              'snr': 0<<su.Real(12)<<30,
#              'orth': True,
#              # 'nonneg': su.Set([True, False])
#              'nonneg': False,
#              }


###############################
######### Models ##############
###############################

# mod_args = {'model': exp.SBMF,
#             'ortho': True,
#             'decay_rate': 0.95,
#             'T0': 10,
#             # 'max_iter': (100, None),
#             'sparse_reg': 0,
#             # 'tree_reg': su.Set([0, 1e-1]),
#             'tree_reg': su.Set([1e-1, 1]),
#             # 'tree_reg': 0,
#             # 'pr_reg': su.Set([0, 1e-2]),
#             # 'period': 2,
#             'period': 5,
#             # 'dim_hid': None
#             'dim_hid': 2
#             }

mod_args = {'model': exp.KBMF,
            'decay_rate': 0.95,
            'T0': 10,
            # 'T0': 20,
            'max_iter': None,
            'tree_reg': su.Set([0, 1e-1, 1]),
            # 'tree_reg': 0,
            'dim_hid': 0.5 << su.Real(12) << 2,
            # 'dim_hid': 2,
            'period': 2,
            }

# mod_args = {'model': exp.BAE,
#             # 'search': (True, False, False),
#             'search': False,
#             # 'beta': (1.0, 1.0, 0.0),
#             'beta': su.Set([1, 0]),
#             # 'decay_rate': (0.9, 1, 1),
#             'decay_rate': 1,
#             # 'T0': (5, 1e-5, 1e-5),
#             'T0': 1e-5,
#             # 'max_iter': (None, 1000, 1000),
#             'max_iter': 1000,
#             # 'pr_reg': su.Set([0, 1e-2]),
#             # 'pr_reg': 1e-2,
#             'pr_reg': 0,
#             'sparse_reg': 1e-1,
#             'tree_reg': 0,
#             'batch_size': 256,
#             'epochs': 10,
#             'dim_hid': 2,
#             }

# mod_args = {'model': exp.NMF2,
#             'reg': su.Set([0, 1e-2]),
#             # 'reg': 0,
#             'dim_hid': 1,
#             }


### magic
##############################
su.send_to_server(task_args, mod_args, send_remotely, verbose=True)


