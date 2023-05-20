CODE_DIR = '/home/kelarion/github/repler/src/'
SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/'

import socket
import os
import sys
import pickle as pkl
import subprocess

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import itertools as itt

sys.path.append(CODE_DIR)
import util
import pt_util
import tasks
import students as stud
import experiments as exp
import grammars as gram
import server_utils as su

###############################################################################

send_remotely = True
# send_remotely = False

### Set parameters to iterate over
##############################
# exp_prm = {'experiment': exp.LogicTask,
# 		   'inp_dics': [tasks.StandardBinary(3).positives],
# 		   'out_dics': [ [(0,1,3,5),(0,2,3,6),(0,1,2,4)], [(0,3,5,6)] ],
# 		   'dim_inp': 3,
# 		   'noise': [0, 0.1],
# 		   }

# exp_prm = {'experiment': exp.FeedforwardExperiment,
# 		   'inputs': [tasks.RandomPatterns(4, 100), ],
# 		   'outputs': [tasks.StandardBinary(2), 
# 		   				tasks.IndependentCategorical(np.eye(4)),
# 		   				tasks.HierarchicalLabels([1,2])]
		   # }


# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits':(2,3,3,3,4,4,4,4,5,5,5,5,5),
# 		   'num_targets': (1,1,2,3,1,2,3,4,1,2,3,4,5),
# 		   'signal':[0, 0.5, 1],
# 		   'seed': None,
# 		   'use_mean': True,
# 		   'dim_inp': 100,
# 		   'input_noise': [0.1, 1],
# 		   }

# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits':(2,3,3,3,4,4,4,4,5,5,5,5,5),
# 		   'num_targets': (1,1,2,3,1,2,3,4,1,2,3,4,5),
# 		   'signal':[0, 0.5, 1],
# 		   'seed': list(range(6)),
# 		   'scale': 0.5,
# 		   'dim_inp': 100,
# 		   'input_noise': [0.1, 1],
# 		   }


# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits':(2,3,4,5),
# 		   'num_targets': (3, 7, 15, 31),
# 		   'signal': [0, 0.25, 0.5, 0.75, 1],
# 		   'seed': list(range(6)),
# 		   'scale': 0.5,
# 		   'dim_inp': 100,
# 		   'input_noise': [0.1, 1],
# 		   }


exp_prm = {'experiment': exp.RandomOrthogonal,
		   'num_bits': 3,
		   'num_targets': [1,2,3],
		   'signal':[0, 0.25, 0.5, 0.75, 1],
		   'seed': list(range(12)),
		   'scale': 0.5,
		   'dim_inp': 100,
		   'input_noise': 0.1,
		   }

# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits':3,
# 		   'num_targets': [1,2,3],
# 		   'signal': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
# 		   'seed': 0,
# 		   'scale': 0,
# 		   'dim_inp': 100,
# 		   'input_noise': [0.1, 1],
# 		   }

# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits':(2,2,3,4,5),
# 		   'num_targets': (2,3,7,15,31),
# 		   'signal':[0, 0.5, 1],
# 		   'seed': list(range(6)),
# 		   'scale': 0.5,
# 		   'dim_inp': 100,
# 		   'input_noise': [0.1, 1],
# 		   }

# net_args = {'model': stud.SimpleMLP,
# 			'num_init': 20,
# 			'width': [3, 4, 5, 6, 7, 8, 9, 10, 20, 100],
# 			'depth': 1,
# 			'p_targ': stud.Bernoulli,
# 			'activation':['Tanh', 'ReLU']
# 			}

net_args = {'model': stud.ShallowNetwork,
			'num_init': 10,
			'width': 128,
			'p_targ': stud.Bernoulli,
			'activation': [pt_util.TanAytch(), pt_util.RayLou()]
			}

opt_args = {'skip_metrics': True,
			'nepoch': 1000,
			'verbose': False,
			'train_outputs': False,
			'train_out_bias': False,
			'lr': 1e-1
			}


### magic
##############################
su.send_to_server(exp_prm, net_args, opt_args, send_remotely)
