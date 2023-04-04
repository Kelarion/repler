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
exp_prm = {'experiment': exp.LogicTask,
		   'inp_dics': [tasks.StandardBinary(3).positives],
		   'out_dics': [ [(0,1,3,5),(0,2,3,6),(0,1,2,4)], [(0,3,5,6)] ],
		   'dim_inp': 3,
		   'noise': [0, 0.1],
		   }

# exp_prm = {'experiment': exp.FeedforwardExperiment,
# 		   'inputs': [tasks.RandomPatterns(4, 100), ],
# 		   'outputs': [tasks.StandardBinary(2), 
# 		   				tasks.IndependentCategorical(np.eye(4)),
# 		   				tasks.HierarchicalLabels([1,2])]
		   # }

# net_args = {'model': stud.SimpleMLP,
# 			'num_init': 20,
# 			'width': [3, 4, 5, 6, 7, 8, 9, 10, 20, 100],
# 			'depth': 1,
# 			'p_targ': stud.Bernoulli,
# 			'activation':['Tanh', 'ReLU']
# 			}

net_args = {'model': stud.SimpleMLP,
			'num_init': 20,
			'width': 16,
			'depth': [5, 10, 20],
			'p_targ': stud.Bernoulli,
			'activation':['Tanh', 'ReLU']
			}

opt_args = {'skip_metrics': True,
			'nepoch': 1000,
			'verbose': False
			}


### magic
##############################
server_utils.send_to_server(exp_prm, net_args, opt_args, send_remotely)
