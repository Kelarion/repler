import socket
import os
import sys

if socket.gethostname() == 'agnello':
    CODE_DIR = '/home/kelarion/github/repler/src/'
    SAVE_DIR = '/mnt/c/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/results/'
    LOAD_DIR = '/mnt/c/Users/mmall/OneDrive/Documents/uni/columbia/main/server_cache/'
else:    
    # CODE_DIR = '/rigel/home/ma3811/repler/'
    # SAVE_DIR = '/rigel/theory/users/ma3811/'  
    CODE_DIR = '/burg/home/ma3811/repler/'
    SAVE_DIR = '/burg/theory/users/ma3811/results/'
    LOAD_DIR = SAVE_DIR
    openmind = False

import pickle as pkl

import numpy as np

sys.path.append(CODE_DIR)
import util
import super_experiments as sxp

from sklearn.exceptions import ConvergenceWarning
import warnings # I hate convergence warnings so much never show them to me
warnings.simplefilter("ignore", category=ConvergenceWarning)


####  Load dataset and parameters  ######
##########################################################
print('Started!')

# get the indices
allargs = sys.argv
idx = int(allargs[1])
ndat = int(allargs[2])

data_idx = int(np.mod(idx,ndat))
param_idx = idx//ndat

# task_dict = pkl.load(open(LOAD_DIR+'task_%d.pkl'%data_idx, 'rb'))
# net_dict = pkl.load(open(LOAD_DIR+'network_%d.pkl'%param_idx, 'rb'))

task_dict = pkl.load(open(LOAD_DIR+'task_%d.pkl'%data_idx, 'rb'))
mod_dict = pkl.load(open(LOAD_DIR+'model_%d.pkl'%param_idx, 'rb'))

print('Loaded data!')

####  Fit model and save #########################
##########################################################

# this_exp = task_dict['experiment'](**task_dict['exp_args'])
task = task_dict['task'](**task_dict['args'])
model = mod_dict['model'](**mod_dict['args'])
this_exp = sxp.Experiment(task, model)

this_exp.run()

this_exp.save_experiment(SAVE_DIR)

print('ALL DONE! THANK YOU VERY MUCH FOR YOUR PATIENCE!!!!!!!')
print(':' + ')'*12)
