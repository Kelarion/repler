CODE_DIR = '/home/matteo/Documents/github/repler/src/'
SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'

import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations

from sklearn import svm, discriminant_analysis, manifold
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

from students import *
from assistants import *
import experiments as exp
import util

#%% Model specification -- for loading purposes
# task = util.ParityMagnitude()
# task = util.RandomDichotomies(2)
task = util.ParityMagnitudeEnumerated()
# task = util.Digits()
# task = util.DigitsBitwise()
# obs_dist = Bernoulli(1)
latent_dist = None
# latent_dist = GausId
nonlinearity = 'ReLU'
# nonlinearity = 'LeakyReLU'

num_layer = 1

decay = 0.0

H = 100
Q = task.num_var
# N_list = None # set to None if you want to automatically discover which N have been tested
# N_list = [2,3,4,5,6,7,8,9,10,11,20,25,50,100]
# N_list = None
# N_list = [2,3,5,10,50,100]
N_list = [100]

# find experiments 
this_exp = exp.mnist_multiclass(task, SAVE_DIR, 
                                z_prior=latent_dist,
                                num_layer=num_layer,
                                weight_decay=decay)
this_folder = SAVE_DIR + this_exp.folder_hierarchy()
if (N_list is None):
    files = os.listdir(this_folder)
    param_files = [f for f in files if 'parameters' in f]
    
    if len(param_files)==0:
        raise ValueError('No experiments in specified folder `^`')
    
    Ns = np.array([re.findall(r"N(\d+)_%s"%nonlinearity,f)[0] \
                    for f in param_files]).astype(int)
    
    N_list = np.unique(Ns)


# load experiments
# loss = np.zeros((len(N_list), 1000))
# test_perf = np.zeros((Q, len(N_list), 1000))
# test_PS = np.zeros((Q, len(N_list), 1000))
# shat = np.zeros((Q, len(N_list), 1000))
nets = [[] for _ in N_list]
all_nets = [[] for _ in N_list]
mets = [[] for _ in N_list]
best_perf = []
for i,n in enumerate(N_list):
    files = os.listdir(this_folder)
    param_files = [f for f in files if ('parameters' in f and '_N%d_%s'%(n,nonlinearity) in f)]
    
    # j = 0
    num = len(param_files)
    all_metrics = {}
    best_net = None
    maxmin = 0
    for j,f in enumerate(param_files):
        rg = re.findall(r"init(\d+)?_N%d_%s"%(n,nonlinearity),f)
        if len(rg)>0:
            init = np.array(rg[0]).astype(int)
        else:
            init = None
            
        this_exp.use_model(N=n, init=init)
        model, metrics, args = this_exp.load_experiment(SAVE_DIR)
        
        if metrics['test_perf'][-1,...].min() > maxmin:    
            maxmin = metrics['test_perf'][-1,...].min()
            best_net = model
        
        for key, val in metrics.items():
            if key not in all_metrics.keys():
                shp = (num,) + val.shape
                all_metrics[key] = np.zeros(shp)*np.nan
            if val.shape[0]==1000:
                continue
            all_metrics[key][j,...] = val
        all_nets[i].append(model)
        
    nets[i] = best_net
    mets[i] = all_metrics
    best_perf.append(maxmin)

test_dat = this_exp.test_data
train_dat = this_exp.train_data

digits = torchvision.datasets.MNIST(SAVE_DIR+'digits/', download=True, 
                                    transform=torchvision.transforms.ToTensor())

#%%
n_compute = 1000
n_batch = 50

indep = []
inp_corr = []
for net in all_nets[0]:
    for _ in range(n_batch):
        idx = np.random.choice(train_dat[0].shape[0], n_compute, replace=False)
        z = net(train_dat[0][idx,:])[2].detach()
        indep.append(np.nanmean(np.abs(np.corrcoef(z.T))[np.eye(z.shape[1])==0]))
        inp_corr.append(np.nanmedian(np.abs(np.corrcoef(train_dat[0][idx,:].T))[np.eye(784)==0]))
    # N = z.shape[1]
    # I = np.zeros((N, N))
    

plt.hist(indep, density=True, alpha=0.5)

