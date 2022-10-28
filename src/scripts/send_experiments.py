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

### Set parameters to iterate over
##############################
exp_prm = {'experiment': exp.HierarchicalClasses,
		   'input_task': tasks.RandomPatterns,
		   'input_noise': [0, 0.1],
		   'dim_inp': 100,
		   'num_vars': [[1,2]],
		   'K': 2,
		   'respect_hierarchy': True}

net_args = {'model': stud.SimpleMLP,
			'width': [2,3,4,5,10,20,30,40,50,100],
			'depth': 1,
			'p_targ': stud.Bernoulli,
			'activation':['Tanh', 'ReLU']}

opt_args = {'opt_alg': optim.SGD,
			'init_index': list(range(5)), 
			'skip_metrics': False,
			'nepoch': 10000,
			'verbose': False}


### magic
##############################
server_utils.send_to_server(exp_prm, net_args, opt_args, send_remotely)
