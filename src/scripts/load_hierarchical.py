CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'

import socket
import os
import sys
import pickle as pkl
import subprocess

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

sys.path.append(CODE_DIR)
import util
import tasks
import students as stud
import experiments as exp
import grammars as gram
import server_utils 

#%%

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


#%%

all_exp_args, all_exp_params = server_utils.AllExperiments(exp_prm, net_args, opt_args)

all_metrics = {}
for exp_args in tqdm(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.model = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR+'results/')
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.metrics.items()}
    for k in all_metrics.keys():
        all_metrics[k].append(this_exp.metrics[k])
        
    
for k,v in all_metrics.items():
    all_metrics[k] = np.array(v)




