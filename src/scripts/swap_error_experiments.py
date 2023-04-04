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
import torch.nn as nn
from tqdm import tqdm
import itertools as itt

sys.path.append(CODE_DIR)
import util
import tasks

import students as stud
import experiments as exp
import grammars as gram
import server_utils 

###############################################################################

send_remotely = True
# send_remotely = False

### Set parameters to iterate over
##############################
exp_prm = {'experiment':exp.SwapErrors,
		   'T_inp1': 5,
		   'T_inp2': 15,
		   'T_resp': 25,
		   'T_tot': 35,
		   'num_cols': 32,
		   'jitter': 3,
		   'inp_noise': 0.0,
		   'dyn_noise': [0.0, 0.01],
		   'present_len': [1, 3],
		   'color_func': [util.TrigColors(), util.RFColors()]}

net_args = {'model': stud.MonkeyRNN,
			'dim_hid': 50,
			'num_init': 4,
			'p_targ': stud.GausId,
			'p_hid': stud.GausId,
			'nonlinearity':['relu', 'tanh'],
			'fix_encoder':True,
			'beta': [0, 1e-6],
			'fix_decoder': True,
			'rnn_type': nn.RNN}

opt_args = {'opt_alg': optim.Adam,
			'skip_metrics': False,
			'nepoch': 10000,
			'verbose': False}

# ### Set parameters to iterate over
# ##############################
# exp_prm = {'experiment':exp.SwapErrors,
# 		   'T_inp1': 5,
# 		   'T_inp2': 15,
# 		   'T_resp': 25,
# 		   'T_tot': 35,
# 		   'num_cols': 32,
# 		   'jitter': 3,
# 		   'inp_noise': 0.0,
# 		   'dyn_noise': 0.0,
# 		   'present_len': 1,
# 		   'test_noise': 0.01,
# 		   'color_func': util.RFColors()}

# net_args = {'model': stud.MonkeyRNN,
# 			'num_init': 1,
# 			'dim_hid': 50,
# 			'p_targ': stud.GausId,
# 			'p_hid': stud.GausId,
# 			'nonlinearity':'relu',
# 			'fix_encoder':True,
# 			'beta': 0,
# 			'fix_decoder': True,
# 			'rnn_type': nn.RNN}

# opt_args = {'opt_alg': optim.Adam,
# 			'skip_metrics': False,
# 			'weight_decay': 0,
# 			'nepoch': 1000,
# 			'verbose': True}


### magic
##############################
server_utils.send_to_server(exp_prm, net_args, opt_args, send_remotely)
