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
import server_utils 

###############################################################################

send_remotely = True
# send_remotely = False
	
# ### Set parameters to iterate over
# ##############################
# exp_prm = {'experiment': exp.EpsilonSeparableXOR,
# 		   'epsilon': np.linspace(0,1,21).tolist(),
# 		   'input_noise':[0.0, 0.1, 0.2, 0.3], # 0.125 - 8 on log scale
# 		   'dim_inp': 120}

# net_args = {'model': stud.ShallowNetwork,
# 			'width': 120,
# 			'p_targ': stud.Bernoulli,
# 			'num_init': 20,
# 			'inp_weight_distr': pt_util.uniform_tensor,
# 			'out_weight_distr': [pt_util.uniform_tensor, 
# 								 pt_util.sphere_weights,
# 								 pt_util.BalancedBinary(2,1,normalize=True)],
# 			'activation':[pt_util.TanAytch(), pt_util.RayLou()]}

# opt_args = {'train_outputs': [False],
# 			# 'init_index': list(range(5)),
# 			'do_rms': False,
# 			'nepoch': 5000,
# 			'lr':1e-1,
# 			'bsz':200,
# 			'verbose': False,
# 			'skip_metrics':True}

### Set parameters to iterate over
##############################
# exp_prm = {'experiment': exp.WeightDynamics,
# 		   'inp_task': (tasks.RandomPatterns(4, 200, noise_var=1.0), 
# 		   				tasks.NudgedXOR(40, nudge_mag=0.0, noise_var=0.1, sqrt_N_norm=False),
# 		   				tasks.NudgedXOR(40, nudge_mag=0.3, noise_var=0.1, sqrt_N_norm=False),
# 		   				tasks.NudgedXOR(40, nudge_mag=0.5, noise_var=0.1, sqrt_N_norm=False),
# 		   				tasks.NudgedXOR(40, nudge_mag=1.0, noise_var=0.1, sqrt_N_norm=False),
# 		   				tasks.NudgedXOR(40, nudge_mag=0.0, noise_var=1.0),
# 		   				tasks.NudgedXOR(40, nudge_mag=0.3, noise_var=1.0),
# 		   				tasks.NudgedXOR(40, nudge_mag=0.5, noise_var=1.0),
# 		   				tasks.NudgedXOR(40, nudge_mag=1.0, noise_var=1.0),
# 		   				tasks.RandomPatterns(4, 200, noise_var=1.0)),
# 		   'out_task': (tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,3)]),
# 		   				tasks.RandomDichotomies(d=[(0,1), (0,2)]))}
exp_prm = {'experiment': exp.RandomInputMultiClass,
		   'dim_inp': 128,
		   'num_bits': (1, 2, 3, 4, 5), 
		   'num_class': (1, 2, 3, 4, 5),
		   'input_noise':0.1,
		   'center':[True, False]}

net_args = {'model': stud.NGroupNetwork,
			'n_per_group': 10,
			'num_k': 2,
			'p_targ': stud.Bernoulli,
			'num_init': 10,
			'inp_weight_distr': pt_util.uniform_tensor,
			'inp_bias_distr': torch.ones,
			'inp_bias_var': 0,
			'out_bias_var': 0,
			'activation':[pt_util.TanAytch(), pt_util.RayLou()]}

opt_args = {'train_outputs': False,
			'train_inp_bias': False,
			'train_out_bias':False,
			# 'init_index': list(range(5)),
			'do_rms': (False, True),
			'nepoch': 2000,
			'lr':(1e-1, 1e-2),
			'bsz':200,
			'verbose': False,
			'skip_rep_metrics':True,
			'skip_metrics':False,
			'conv_tol': 1e-10,
			'metric_period':10}

### magic
##############################
server_utils.send_to_server(exp_prm, net_args, opt_args, send_remotely)
