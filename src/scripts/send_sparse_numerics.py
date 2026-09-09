CODE_DIR = '/home/kelarion/github/repler/src/'
SAVE_DIR = '/mnt/c/Users/mmall/OneDrive/Documents/uni/columbia/main/'

import sys
sys.path.append(CODE_DIR)

import numpy as np

import server_utils as su
import new_bae_experiments as nbe   # StructuredCats + NewBMF

####################################

send_remotely = True
# send_remotely = False

###########################################################################
### Set parameters to iterate over ########################################
###########################################################################

task_args = {'task': nbe.StructuredCats,
             'samps': 9,
             'seed': 0,
             # 'spec': su.Set(['cat8', 'tree4', 'cat3x3']),
             'spec': 'cat8',
             'N': 64,
             'temp': 1e-1,
             'slab': True,
             'snr': 0 << su.Real(13) << 24,
             'ratio': 4,
             'orth': True,
             'nonneg': True,
             }

###############################
######### Models ##############
###############################

mod_args = {'model': nbe.NewBMF,
            'kind': 'SemiBMF',
            'dim_hid': None,               # None -> use the true latent dim
            'nonneg': True,
            'tree_reg': su.Set([0.0, 0.1, 1.0]),
            'sparse_reg': 0.0,
            'weight_pr_reg': 0.1,
            'weight_l2_reg': 1e-2,
            'T0': 10,
            'decay_rate': 0.9,
            'period': 8,
            'min_temp': 1.0,
            'max_iter': None,
            }

### magic
##############################
su.send_to_server(task_args, mod_args, send_remotely, verbose=True)
