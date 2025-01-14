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


# d = su.Set([8, 16, 32])
# k = su.Set([1, np.log2(d)])
# c = su.Real(num=12)

# exp_prm = {'experiment': exp.RandomKernelClassification,
# 		   'num_points': d,
# 		   'num_targets': k,
# 		   'alignment': 0 << c  << np.sqrt(k/(d - 1)),
# 		   'seed': 0,
# 		   'scale': 0.0,
# 		   'dim_inp': 100,
# 		   'input_noise': 1,
# 		   }

# d = su.Set([8, 16, 32])
# k = su.Set([1, np.log2(d)])
# c = su.Real(num=6)

# exp_prm = {'experiment': exp.RandomKernelClassification,
# 		   'num_points': d,
# 		   'num_targets': k,
# 		   'alignment': 0 << c  << np.sqrt(k/(d - 1)),
# 		   'seed': list(range(12)),
# 		   'scale': 0.5,
# 		   'dim_inp': 100,
# 		   'input_noise': 1
# 		   }

# c = su.Real(num=12)

# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits': 2,
# 		   'num_targets': 1,
# 		   'alignment': 0 << c << np.sqrt(1/3),
# 		   'seed': (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
# 		   'scale': (0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
# 		   'dim_inp': 100,
# 		   'input_noise': 1,
# 		   }

# # d = su.Integer(step=1) 
# d = su.Set([2])
# k = su.Set([1, d])
# c = su.Real(num=6)

# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits': d,
# 		   'num_targets': k,
# 		   'alignment': 0 << c  << np.sqrt(k/(2**d - 1)),
# 		   'seed': list(range(12)),
# 		   'scale': 0.5,
# 		   'dim_inp': 100,
# 		   'input_noise': 1,
# 		   }

d = su.Set([3,4,5])
k = su.Set([1, d])
c = su.Real(num=12)

exp_prm = {'experiment': exp.RandomOrthogonal,
		   'num_bits': d,
		   'num_targets': k,
		   'alignment': 0 << c  << np.sqrt(k/(2**d - 1)),
		   'seed': 0,
		   'scale': 0.0,
		   'dim_inp': 100,
		   'input_noise': 1,
		   }

# d = su.Integer(step=1) 
# d = su.Set([3,5])
# k = su.Set([2**d - 1])
# c = su.Real(num=24)

# # exp_prm = {'experiment': exp.RandomOrthogonal,
# # 		   'num_bits': d,
# # 		   'num_targets': k,
# # 		   'alignment': (np.sqrt(d*(2**d - 1))/(2**d - 1)) << c  << 1,
# # 		   'seed': list(range(12)),
# # 		   'scale': 0.5,
# # 		   'dim_inp': 100,
# # 		   'input_noise': 1,
# # 		   }

# exp_prm = {'experiment': exp.RandomOrthogonal,
# 		   'num_bits': d,
# 		   'num_targets': k,
# 		   'alignment': (np.sqrt(d*(2**d - 1))/(2**d - 1)) << c  << 1,
# 		   'seed': 0,
# 		   'scale': 0.0,
# 		   'dim_inp': 100,
# 		   'input_noise': 1,
# 		   }


# net_args = {'model': stud.SimpleMLP,
# 			'num_init': 20,
# 			'width': [3, 4, 5, 6, 7, 8, 9, 10, 20, 100],
# 			'depth': 1,
# 			'p_targ': stud.Bernoulli,
# 			'activation':['Tanh', 'ReLU']
# 			}

# net_args = {'model': stud.ShallowNetwork,
# 			'num_init': 10,
# 			'width': 128,
# 			'p_targ': stud.Bernoulli,
# 			'inp_bias_shift': 0,
# 			'activation': [pt_util.TanAytch(), pt_util.RayLou()]
# 			}


net_args = {'model': stud.ShallowNetwork,
			'num_init': 10,
			'width': 128,
			'p_targ': stud.Bernoulli,
			'activation': [pt_util.RayLouUB(1), pt_util.RayLouUB(1.5), pt_util.RayLouUB(2)]
			}

# net_args = {'model': stud.ShallowNetwork,
# 			'num_init': 10,
# 			'width': 128,
# 			'p_targ': stud.Bernoulli,
# 			'inp_bias_shift': [-1, -0.5, 0, 0.5, 1],
# 			'activation': pt_util.RayLou()
# 			}

# net_args = {'model': stud.ShallowNetwork,
# 			'num_init': 10,
# 			'width': 128,
# 			'p_targ': stud.Bernoulli,
# 			'activation': [pt_util.RayLouShift(1), 
# 						   pt_util.RayLouShift(0.5), 
# 						   pt_util.RayLouShift(-0.5),
# 						   pt_util.RayLouShift(-1)]
# 			}


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
