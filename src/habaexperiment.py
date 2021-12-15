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
import sys

if socket.gethostname() == 'kelarion':
    if sys.platform == 'linux':
        CODE_DIR = '/home/kelarion/github/repler/src'
        SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    else:
        CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
        SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    openmind = False
elif socket.gethostname() == 'openmind7':
    CODE_DIR = '/home/malleman/repler/'
    SAVE_DIR = '/om2/user/malleman/abstraction/'
    openmind = True
else:    
    # CODE_DIR = '/rigel/home/ma3811/repler/'
    # SAVE_DIR = '/rigel/theory/users/ma3811/'
    CODE_DIR = '/burg/home/ma3811/repler/'
    SAVE_DIR = '/burg/theory/users/ma3811/'
    openmind = False

sys.path.append(CODE_DIR)

import getopt

import math
import pickle
import numpy as np

import experiments # my code packages
import util
import tasks
import students

#%% parse arguments (need to remake this for each kind of experiment)
allargs = sys.argv

arglist = allargs[1:]

unixOpts = "vmgn:i:h:c:q:o:f:r:e:t:"
gnuOpts = ["verbose", "mse", "task"]

opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

num_layer = 1
verbose, skip_metrics, N, init = False, False, None, None # defaults
num_class, num_dich, ovlp = 8, 2, 0
this_task = 'mog'
coding_level = None
ols_initialised = False
gaus_obs = False
rot = 0.0
nepoch = 1000
for op, val in opts:
    if op in ('-v','--verbose'):
        verbose = True
    if op in ('-n',):
        N = int(val)
    if op in ('-i',):
        if openmind:
            coding_level = float(val)/10
        else:    
            init = int(val)
    if op in ('-m',):
        skip_metrics = True
    if op in ('-h',):
        num_layer = int(val)
    if op in ('-c',):
        num_class = int(val)
    if op in ('-q',):
        num_dich =  int(val)
    if op in ('-o',):
        ovlp = int(val)
    if op in ('-t','--task'):
        this_task = val
    if op in ('-f',):
        coding_level = float(val)
    if op in ('-r',):
        rot = float(val)
    if op in ('-e',):
        nepoch = int(val)
    if op in ('-g',):
        ols_initialised = True
    if op in ('--mse',):
        gaus_obs = True

#%% run experiment (put the code you want to run here!)
# sample_dichotomies = 4
sample_dichotomies = None

fixed_decoder = True
# fixed_decoder = False

decay = 0.0

readout_weights = None
# readout_weights = students.BinaryReadout
# readout_weights = students.PositiveReadout

latent_dist = None
# latent_dist = students.GausId

H = 100 # number of hidden units

# nonlinearity = 'ReLU'
nonlinearity = 'Tanh'
# nonlinearity = 'Sigmoid'
# nonlinearity = 'LeakyReLU'

print('- - - - - - - - - - - - - - - - - - - - - - - - - - ')        
if this_task == 'mnist': 
    task = tasks.RandomDichotomies(num_class, num_dich, overlap=ovlp, use_mse=gaus_obs)
    exp = experiments.mnist_multiclass(N=N, 
                                      task=task, 
                                      SAVE_DIR=SAVE_DIR, 
                                      H=H,
                                      nonlinearity=nonlinearity,
                                      num_layer=num_layer,
                                      z_prior=latent_dist,
                                      weight_decay=decay,
                                      decoder=readout_weights,
                                      nepoch=nepoch,
                                      sample_dichotomies=sample_dichotomies,
                                      init=init,
                                      skip_metrics=skip_metrics,
                                      good_start=ols_initialised,
                                      init_coding=coding_level,
                                      fix_decoder=fixed_decoder,
                                      rot=rot)
elif this_task == 'mog':
    task = tasks.RandomDichotomies(num_class, num_dich, overlap=ovlp, use_mse=gaus_obs)
    exp = experiments.random_patterns(N=N,
                                      task=task, 
                                      SAVE_DIR=SAVE_DIR,
                                      num_class=num_class,
                                      dim=100,
                                      var_means=1,
                                      H=H,
                                      nonlinearity=nonlinearity,
                                      num_layer=num_layer,
                                      z_prior=latent_dist,
                                      weight_decay=decay,
                                      decoder=readout_weights,
                                      bsz=64,
                                      lr=1e-3,
                                      nepoch=nepoch,\
                                      sample_dichotomies=sample_dichotomies,
                                      init=init,
                                      skip_metrics=skip_metrics,
                                      init_coding=coding_level,
                                      good_start=ols_initialised,
                                      fix_decoder=fixed_decoder,
                                      rot=rot)
elif this_task == 'structured':
    # bits = np.nonzero(1-np.mod(np.arange(num_class)[:,None]//(2**np.arange(np.log2(num_class))[None,:]),2))
    # pos_conds = np.split(bits[0][np.argsort(bits[1])],int(np.log2(num_class)))
    # inp_task = tasks.EmbeddedCube(tasks.StandardBinary(int(np.log2(num_class))))
    # inp_task = tasks.TwistedCube(tasks.StandardBinary(2), 100, f=coding_level, noise_var=0.1)
    inp_task = tasks.NudgedXOR(tasks.StandardBinary(2), 100, nudge_mag=coding_level, noise_var=rot)
    # task = tasks.RandomDichotomies(d=[(0,1,3,5),(0,2,3,6),(0,1,2,4)])
    task = tasks.RandomDichotomies(d=[(0,3)])
    # task = tasks.LogicalFunctions(d=pos_conds, function_class=num_dich)
    exp = experiments.structured_inputs(N=N,
                                      task=task,
                                      input_task=inp_task,
                                      SAVE_DIR=SAVE_DIR,
                                      noise_var=rot,
                                      H=H,
                                      nonlinearity=nonlinearity,
                                      num_layer=num_layer,
                                      z_prior=latent_dist,
                                      weight_decay=decay,
                                      decoder=readout_weights,
                                      bsz=200,
                                      nepoch=nepoch,\
                                      sample_dichotomies=sample_dichotomies,
                                      init=init,
                                      skip_metrics=skip_metrics,
                                      fix_decoder=fixed_decoder)
elif this_task == 'dlog':
    bits = np.nonzero(1-np.mod(np.arange(num_class)[:,None]//(2**np.arange(np.log2(num_class))[None,:]),2))
    pos_conds = np.split(bits[0][np.argsort(bits[1])],int(np.log2(num_class)))
    inp_task = tasks.RandomDichotomies(d=pos_conds) 
    task = tasks.LogicalFunctions(d=pos_conds, function_class=num_dich)
    exp = experiments.delayed_logic(N=N,
                                  task=task, 
                                  input_task=inp_task,
                                  SAVE_DIR=SAVE_DIR,
                                  time_between=ovlp,
                                  skip_metrics=skip_metrics,
                                  weight_decay=decay,
                                  decoder=readout_weights,
                                  bsz=200,
                                  nepoch=nepoch,
                                  init=init,
                                  sample_dichotomies=sample_dichotomies,
                                  nonlinearity=nonlinearity,
                                  num_layer=num_layer,
                                  fix_decoder=fixed_decoder)

exp.run_experiment(verbose)
exp.save_experiment(SAVE_DIR)

print('ALL DONE! THANK YOU VERY MUCH FOR YOUR PATIENCE!!!!!!!')
print(':' + ')'*12)


