CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
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
from matplotlib import animation as anime
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

import students
import assistants
import experiments as exp
import util

#%% Model specification -- for loading purposes
# mse_loss = True
mse_loss = False
# categorical = True
categorical = False

num_cond = 128
these_Q = [1,2,3,4,5,6,7,8,9,10]
# these_Q = [1,2,3,4,5,6,7]

latent_dist = None
# latent_dist = GausId
nonlinearity = 'ReLU'
# nonlinearity = 'LeakyReLU'

# num_layer = 0
num_layer = 1

# good_start = True
good_start = False
# coding_level = 0.7
coding_level = None

rotation = 0.0

decay = 0.0

H = 100
n = 100

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

# find experiments 

nets = [[] for _ in these_Q]
all_nets = [[] for _ in these_Q]
all_args = [[] for _ in these_Q]
mets = [[] for _ in these_Q]
dicts = [[] for _ in these_Q]
best_perf = []
for i,num_var in enumerate(these_Q):
        
    # use_mnist = True
    use_mnist = False
    
    # task = util.ParityMagnitude()
    if categorical:
        task = util.RandomDichotomiesCategorical(num_cond,num_var,0, mse_loss)
    else:
        task = util.RandomDichotomies(num_cond,num_var,0, mse_loss)
    # task = util.ParityMagnitudeEnumerated()
    # task = util.Digits()
    # task = util.DigitsBitwise()
    
    if use_mnist:
        this_exp = exp.mnist_multiclass(task, SAVE_DIR, 
                                        z_prior=latent_dist,
                                        num_layer=num_layer,
                                        weight_decay=decay,
                                        decoder=random_decoder,
                                        good_start=good_start,
                                        init_coding=coding_level)
    else:
        this_exp = exp.random_patterns(task, SAVE_DIR, 
                                       num_class=num_cond,
                                       dim=100,
                                       var_means=1,
                                       z_prior=latent_dist,
                                       num_layer=num_layer,
                                       weight_decay=decay,
                                       decoder=random_decoder,
                                       good_start=good_start,
                                       init_coding=coding_level,
                                       rot=rotation)
       
    this_folder = SAVE_DIR + this_exp.folder_hierarchy()

    
    # load experiments
    # loss = np.zeros((len(N_list), 1000))
    # test_perf = np.zeros((Q, len(N_list), 1000))
    # test_PS = np.zeros((Q, len(N_list), 1000))
    # shat = np.zeros((Q, len(N_list), 1000))
    
    
    files = os.listdir(this_folder)
    param_files = [f for f in files if ('parameters' in f and '_N%d_%s'%(n,nonlinearity) in f)]
    
    # j = 0
    num = len(param_files)
    all_metrics = {}
    best_net = None
    this_arg = None
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
            this_arg = args
        
        for key, val in metrics.items():
            if key not in all_metrics.keys():
                # shp = (num,) + np.squeeze(np.array(val)).shape
                # all_metrics[key] = np.zeros(shp)*np.nan
                all_metrics[key] = []
            
            # ugh = np.min([all_metrics[key][j,...].shape[0], np.squeeze(val).shape[0]])
            # all_metrics[key][j,:ugh,...] = np.squeeze(val)[:ugh,...]
            all_metrics[key].append(np.squeeze(val))
    
            # if (val.shape[0]==1000) or not len(val):
                # continue
            # all_metrics[key][j,...] = val
        all_nets[i].append(model)
        all_args[i].append(args)
        
    nets[i] = best_net
    mets[i] = all_metrics
    dicts[i] = this_arg
    best_perf.append(maxmin)

#%%
q_id = 0
netid = 0 # which specific experiment to use

params = all_args[q_id][netid]
num_var = these_Q[q_id]

this_exp.load_other_info(params)
this_exp.load_data(SAVE_DIR)

test_dat = this_exp.test_data
train_dat = this_exp.train_data

#%% Compute linear dimension of each representation
n_compute = 2500

pr = []
for i,q in enumerate(these_Q):
    
    pr_net = []
    for j in tqdm(range(len(all_args[i]))):
        this_exp.load_other_info(all_args[i][j])
        this_exp.load_data(SAVE_DIR)

        idx = np.random.choice(this_exp.train_data[0].shape[0], n_compute, replace=False)

        z = all_nets[i][j](this_exp.train_data[0][idx,:])[2].detach().numpy()
        
        _, S, _ = la.svd(z-z.mean(1)[:,None], full_matrices=False)
        eigs = S**2
        pr_net.append((np.sum(eigs)**2)/np.sum(eigs**2))
    
    pr.append(pr_net)

#%% Plot them 

pr_mean = np.array([np.mean(p) for p in pr])/np.log2(num_cond)
pr_err = np.array([np.std(p) for p in pr])/np.log2(num_cond)

normalized_Q = np.array(these_Q)/np.log2(num_cond)

plt.plot(normalized_Q, pr_mean)
plt.fill_between(normalized_Q, pr_mean-pr_err, pr_mean+pr_err, alpha=0.5)

#%% Make the plot tidy
newlims = [np.min([plt.ylim(), plt.xlim()]), np.max([plt.ylim(), plt.xlim()])]

plt.axis('equal')
plt.axis('square')
plt.xlim(newlims)
plt.ylim(newlims)

plt.plot(newlims,newlims,'-.', color=(0.5,0.5,0.5), zorder=0)
plt.plot([1,1],newlims,'-.', color=(0.5,0.5,0.5), zorder=0)
