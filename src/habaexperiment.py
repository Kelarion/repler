"""
Multi-classification experiments.

Command-line arguments:
    -v [--verbose]  set verbose
    -n val          number of neurons (int)
    -t str          which task, must be (case-sensitive) name of a Task class
    -m              skip computation of metrics
"""

#%%
import socket
import os

if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/matteo/Documents/github/repler/src/'
    SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'
elif socket.gethostname() in ['watson', 'holmes']:    
    CODE_DIR = '/rigel/home/ma3811/repler/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
else: 
    CODE_DIR = '/home/malleman/repler/'
    SAVE_DIR = '/om2/user/malleman/'
    
import getopt, sys
sys.path.append(CODE_DIR)

import math
import pickle
import numpy as np

import experiments # my code packages
import util
import students

#%% parse arguments (need to remake this for each kind of experiment)
allargs = sys.argv

arglist = allargs[1:]

unixOpts = "vmn:i:h:"
gnuOpts = ["verbose"]

opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

num_layer = 1
verbose, skip_metrics, N, init = False, False, None, None # defaults
for op, val in opts:
    if op in ('-v','--verbose'):
        verbose = True
    if op in ('-n'):
        N = int(val)
    if op in ('-i'):
        init = int(val)
    if op in ('-m'):
        skip_metrics = True
    if op in ('-h'):
        num_layer = int(val)

#%% run experiment (put the code you want to run here!)
# task = util.ParityMagnitude()
task = util.ParityMagnitudeEnumerated()
# task = util.Digits()
# task = util.DigitsBitwise()
# task = util.ParityMagnitudeFourunit()
# task = util.RandomDichotomies(2)

# parallelism_conditions = util.ParityMagnitude()
parallelism_conditions = util.DigitsBitwise()
# parallelism_conditions = task

dichotomy_type = 'simple'
# dichotomy_type = 'general'

decay = 1.0

nepoch = 5000

latent_dist = None
# latent_dist = students.GausId

H = 100 # number of hidden units

nonlinearity = 'ReLU'
# nonlinearity = 'Tanh'
# nonlinearity = 'Sigmoid'
# nonlinearity = 'LeakyReLU'

print('- - - - - - - - - - - - - - - - - - - - - - - - - - ')           
exp = experiments.mnist_multiclass(N=N, 
                                   task=task, 
                                   SAVE_DIR=SAVE_DIR, 
                                   H=H,
                                   nonlinearity=nonlinearity,
                                   num_layer=num_layer,
                                   z_prior=latent_dist,
                                   weight_decay=decay,
                                   nepoch=nepoch,
                                   abstracts=parallelism_conditions,
                                   dichotomy_type=dichotomy_type,
                                   init=init,
                                   skip_metrics=skip_metrics)
exp.run_experiment(verbose)
exp.save_experiment(SAVE_DIR)

print('ALL DONE! THANK YOU VERY MUCH FOR YOUR PATIENCE!!!!!!!')
print(':' + ')'*12)


