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

import new_bae_experiments as nbx

####################################

send_remotely = True
# send_remotely = False

###########################################################################
### Set parameters to iterate over ########################################
###########################################################################

N = su.Set(2**np.arange(4,12))
# N = 16
task_args = {'task': nbx.StructuredCats,
             # 'spec': 'grid5(d=2)',
             # 'spec': 'tree5',
             'spec': 'cat3x2+tree3',
             'samps': 3,
             'temp': [1e-2,  1],
             # 'samps': 1,
             # 'seed': su.Set([0,1,2]),
             'seed': 0,
             'N': N,
             # 'N': 64, 
             # 'p':0.5,
             # 'snr': 0<<su.Real(7)<<13,
             'snr': 15,
             # 'ratio': 0.5<<su.Integer(num=5)<<2,
             'ratio': 2,
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
            'sparse_reg': 1,
            'tree_reg': [0, 1, 10],
            # 'tree_reg': 0,
            'dim_hid': 0.5 << su.Real(7) << 2,
            # 'dim_hid': 2,
            'period': 50,
            'min_temp': 1,
            # 'J_lr': [0, 1e-2],
            'J_lr': 1e-2,
            'lr': 1e-1,
            'folds': 10,
            'n_chains': 8,
            }

### magic
##############################
su.send_to_server(task_args, mod_args, send_remotely, verbose=True)


