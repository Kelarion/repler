"""
Multi-classification experiments.

Command-line arguments:
    -v [--verbose]  set verbose
    -n val          number of neurons (int)
"""

#%%
import socket
import os

if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/matteo/Documents/github/repler/src/'
    SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'
else:    
    CODE_DIR = '/rigel/home/ma3811/repler/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
    
import getopt, sys
sys.path.append(CODE_DIR)

import math
import pickle
import numpy as np
from experiments import *

#%% parse arguments (need to remake this for each kind of experiment)
allargs = sys.argv

arglist = allargs[1:]

unixOpts = "vsn:p:l:q:"
gnuOpts = ["verbose"]

opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

verbose, N, Q = False, None, None # defaults
for op, val in opts:
    if op in ('-v','--verbose'):
        verbose = True
    if op in ('-n'):
        N = int(val)

#%% run experiment (put the code you want to run here!)
        
classifier_function = parity_magnitude
mnist_multiclass(N, classifier_function, SAVE_DIR, verbose)

print(':' + ')'*12)


