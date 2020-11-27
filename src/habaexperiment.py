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
    CODE_DIR = '/rigel/home/ma3811/repler/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
    openmind = False

sys.path.append(CODE_DIR)

import getopt

import math
import pickle
import numpy as np

import experiments # my code packages
import util
import students

#%% parse arguments (need to remake this for each kind of experiment)
allargs = sys.argv

arglist = allargs[1:]

unixOpts = "vmdgn:i:h:c:q:o:f:r:e:"
gnuOpts = ["verbose", "mse"]

opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

num_layer = 1
verbose, skip_metrics, N, init = False, False, None, None # defaults
num_class, num_dich, ovlp = 8, 2, 0
use_digits = False
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
    if op in ('-d',):
        use_digits = True
    if op in ('-f',):
        coding_level = float(val)
    if op in ('-r',):
        rot = int(val)
    if op in ('-e',):
        nepoch = int(val)
    if op in ('-g',):
        ols_initialised = True
    if op in ('--mse',):
        gaus_obs = True

#%% run experiment (put the code you want to run here!)
# task = util.ParityMagnitude()
# task = util.ParityMagnitudeEnumerated()
# task = util.Digits()
# task = util.DigitsBitwise()
# task = util.ParityMagnitudeFourunit()
task = util.RandomDichotomies(num_class, num_dich, overlap=ovlp, use_mse=gaus_obs)
# task = util.RandomDichotomiesCategorical(num_class, num_dich, overlap=ovlp, use_mse=gaus_obs)

sample_dichotomies = 2*num_dich
# sample_dichotomies = None

fixed_decoder = True
# fixed_decoder = False

decay = 0.0

# random_decoder = students.LinearRandomSphere(radius=0.2, eps=0.05, 
#                                               fix_weights=True,
#                                               nonlinearity=task.link)
# random_decoder = students.LinearRandomNormal(var=0.2, 
#                                               fix_weights=True, 
#                                               nonlinearity=task.link)
# random_decoder = students.LinearRandomProportional(scale=0.2, 
#                                                     fix_weights=True, 
#                                                     coef=2,
#                                                     nonlinearity=task.link)
random_decoder = None


latent_dist = None
# latent_dist = students.GausId

H = 100 # number of hidden units

nonlinearity = 'ReLU'
# nonlinearity = 'Tanh'
# nonlinearity = 'Sigmoid'
# nonlinearity = 'LeakyReLU'

print('- - - - - - - - - - - - - - - - - - - - - - - - - - ')        
if use_digits: 
    exp = experiments.mnist_multiclass(N=N, 
                                      task=task, 
                                      SAVE_DIR=SAVE_DIR, 
                                      H=H,
                                      nonlinearity=nonlinearity,
                                      num_layer=num_layer,
                                      z_prior=latent_dist,
                                      weight_decay=decay,
                                      decoder=random_decoder,
                                      nepoch=nepoch,
                                      sample_dichotomies=sample_dichotomies,
                                      init=init,
                                      skip_metrics=skip_metrics,
                                      good_start=ols_initialised,
                                      init_coding=coding_level,
                                      fix_decoder=fixed_decoder,
                                      rot=rot)
else:
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
                                      decoder=random_decoder,
                                      bsz=200,
                                      nepoch=nepoch,\
                                      sample_dichotomies=sample_dichotomies,
                                      init=init,
                                      skip_metrics=skip_metrics,
                                      init_coding=coding_level,
                                      good_start=ols_initialised,
                                      fix_decoder=fixed_decoder,
                                      rot=rot)
exp.run_experiment(verbose)
exp.save_experiment(SAVE_DIR)

print('ALL DONE! THANK YOU VERY MUCH FOR YOUR PATIENCE!!!!!!!')
print(':' + ')'*12)


