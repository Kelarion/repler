CODE_DIR = '/home/kelarion/github/repler/src/'
SAVE_DIR = '/mnt/c/Users/mmall/OneDrive/Documents/uni/results/'
	
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

###########################################################################
### Set parameters to iterate over ########################################
###########################################################################
d = su.Set([32]) 								# number of clusters
k = su.Set([1, np.log2(d)])				# rank of targets
s = su.Real(num=3) 								# input-target alignment
r = su.Set([np.log2(d), d-1-k])					# input rank
# seed = su.Integer(step=1)						# random seed
# scale = su.Set([0*(seed==0) + 0.5*(seed>0)]) 	# kernel distribution scale
s_min = (np.log2(d)/k) * (k > np.log2(d))

exp_prm = {'experiment': exp.RandomKernelClassification,
			'num_points': d,
			'num_targets': k,
			'signal': 0 << s  << 1,
			'seed': 0,
			'scale': 0,
			'dim_inp': 100,
			'input_noise': 1,
			'max_rank': r
			}


net_args = {'model': stud.SimpleMLP,
			'num_init': 10,
			'width': 128,
			'depth': [1,5,10],
			'p_targ': stud.Bernoulli,
			'activation': ['ReLU', 'Tanh']
			}


opt_args = {'skip_metrics': True,
			'nepoch': 5000,
			'verbose': False,
			'lr': 1e-3
			}

### magic
##############################
su.send_to_server(exp_prm, net_args, opt_args, send_remotely)
