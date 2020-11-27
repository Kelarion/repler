import socket
import os
import sys

if sys.platform == 'linux':
    CODE_DIR = '/home/kelarion/github/repler/src'
    SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
else:
    CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
    SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
sys.path.append(CODE_DIR)

import re
import pickle
import warnings

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from sklearn import svm, manifold, linear_model
from tqdm import tqdm

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import experiments as exp

#%%
num_class = 8
num_dich = 2
ovlp = 0
N = 100
N_out = 10
N_list = None

input_type = 'task_inp'
# output_type = 'factored'
# output_type = 'rotated1.0'
output_type = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

task = util.RandomDichotomies(num_class, num_dich, overlap=ovlp)

sample_dichotomies = num_dich
# sample_dichotomies = None

this_exp = exp.random_patterns(task, SAVE_DIR, 
                                num_class=num_class,
                                dim=100,
                                var_means=1)

# FOLDERS = 'results/continuous/%d_%d/%s/%s/'%(num_class,num_dich,input_type, output_type)

# if (N_list is None):
#     files = os.listdir(SAVE_DIR+FOLDERS)
#     param_files = [f for f in files if 'parameters' in f]
    
#     if len(param_files)==0:
#         raise ValueError('No experiments in specified folder `^`')
    
#     Ns = np.array([re.findall(r"_(\d+)_(\d+)?",f)[0] \
#                     for f in param_files]).astype(int)
    
#     N_list = np.array(Ns)
#     N_list = np.unique(N_list,axis=0)
    
# load experiments
# loss = np.zeros((len(N_list), 1000))
# test_perf = np.zeros((Q, len(N_list), 1000))
# test_PS = np.zeros((Q, len(N_list), 1000))
# shat = np.zeros((Q, len(N_list), 1000))
# nets = [[] for _ in N_list]
# all_nets = [[] for _ in N_list]
# mets = [[] for _ in N_list]
# dicts = [[] for _ in N_list]
# best_perf = []
nets = [[] for _ in output_type]
all_nets = [[] for _ in output_type]
mets = [[] for _ in output_type]
dicts = [[] for _ in output_type]
best_perf = []
for i,n in enumerate(output_type):
    # files = os.listdir(SAVE_DIR+FOLDERS)
    # param_files = [f for f in files if ('parameters' in f and '_%d_%d'%(n[0],n[1]) in f)]
    # met_files = [f for f in files if ('metrics' in f and '_%d_%d'%(n[0],n[1]) in f)]
    # arg_files = [f for f in files if ('args' in f and '_%d_%d'%(n[0],n[1]) in f)]
    FOLDERS = '/continuous/%d_%d/%s/rotated%.1f/'%(num_class,num_dich,input_type,n)
    # print(FOLDERS)
    files = os.listdir(SAVE_DIR+FOLDERS)
    param_files = [f for f in files if ('parameters' in f and '_%d_%d'%(N,N_out) in f)]
    met_files = [f for f in files if ('metrics' in f and '_%d_%d'%(N,N_out) in f)]
    arg_files = [f for f in files if ('args' in f and '_%d_%d'%(N,N_out) in f)]
    
    # print(param_files)
    # j = 0
    num = len(param_files)
    all_metrics = {}
    best_net = None
    this_arg = None
    maxmin = 0
    for j,f in enumerate(param_files):
        rg = re.findall(r"init(\d+)?",f)
        if len(rg)>0:
            init = np.array(rg[0]).astype(int)
        else:
            init = None
        if init is not None:
            if init>10:
                continue
        else:
            continue
            
        metrics = pickle.load(open(SAVE_DIR+FOLDERS+met_files[j],'rb'))
        args = pickle.load(open(SAVE_DIR+FOLDERS+arg_files[j],'rb'))
        
        net = students.MultiGLM(students.Feedforward([100, N, N], ['ReLU','ReLU']),
                        students.Feedforward([N, N_out], [None]),
                        students.GausId(N_out))
        
        net.load(SAVE_DIR+FOLDERS+f)
        
        # if metrics['test_perf'][-1,...].min() > maxmin:    
        #     maxmin = metrics['test_perf'][-1,...].min()
        #     best_net = model
        #     this_arg = args
        
        for key, val in metrics.items():
            if len(val)==1000:
                continue
            if key not in all_metrics.keys():
                shp = (num,) + np.squeeze(np.array(val)).shape
                all_metrics[key] = np.zeros(shp)*np.nan
            all_metrics[key][j,...] = np.squeeze(val)
    
            # if (val.shape[0]==1000) or not len(val):
            #     continue
            # all_metrics[key][j,...] = val
        all_nets[i].append(net)
        
    # nets[i] = best_net
    mets[i] = all_metrics
    dicts[i] = args
    # best_perf.append(maxmin)

#%%
netid = 10

# show_me = 'train_loss'
# show_me = 'train_perf' 
# show_me = 'test_perf'
show_me = 'PS'
# show_me = 'SD'
# show_me = 'CCGP'
# show_me = 'mean_grad'
# show_me = 'std_grad'
# show_me = 'PR'
# show_me = 'sparsity'

epochs = np.arange(1,mets[netid][show_me].shape[1]+1)

mean = np.nanmean(mets[netid][show_me],0)
error = (np.nanstd(mets[netid][show_me],0))#/np.sqrt(mets[netid][show_me].shape[0]))

if len(mean.shape)>1:
    for dim in range(mean.shape[-1]):
        pls = mean[...,dim]+error[...,dim]
        mns = mean[...,dim]-error[...,dim]
        plt.plot(epochs, mean[...,dim])
        plt.fill_between(epochs, mns, pls, alpha=0.5)
        plt.semilogx()
else:
    plt.plot(epochs, mean)
    plt.fill_between(epochs, mean-error, mean+error, alpha=0.5)
    plt.semilogx()

plt.xlabel('epoch', fontsize=15)
plt.ylabel(show_me, fontsize=15)
plt.title('N=%d'%N)

#%%
# show_me = 'train_loss'
# show_me = 'train_perf' 
# show_me = 'test_perf'
show_me = 'PS'
# show_me = 'SD'
# show_me = 'CCGP'
# show_me = 'mean_grad'
# show_me = 'std_grad'
# show_me = 'PR'
# show_me = 'sparsity'

final_ps = []
final_ps_err = []
for i in range(len(mets)):
    final_ps.append(np.nanmean(mets[i][show_me][:,-1,:],0))
    final_ps_err.append(np.nanstd(mets[i][show_me][:,-1,:],0))

final_ps = np.array(final_ps)
final_ps_err = np.array(final_ps_err)

plt.plot(output_type, final_ps[:,:2].mean(1))
plt.fill_between(output_type, 
                 final_ps[:,:2].mean(1)-final_ps_err[:,:2].mean(1),
                 final_ps[:,:2].mean(1)+final_ps_err[:,:2].mean(1),
                 alpha=0.5)

plt.plot(output_type, final_ps[:,2:].mean(1))
plt.fill_between(output_type, 
                 final_ps[:,2:].mean(1)-final_ps_err[:,2:].mean(1),
                 final_ps[:,2:].mean(1)+final_ps_err[:,2:].mean(1),
                 alpha=0.5)

plt.legend(['Trained Dichotomies','Untrained Dichotomies'])
plt.ylabel('Final ' + show_me)
plt.xlabel('Output "simpliciality"')

#%%
# show_me = 'train_loss'
# show_me = 'train_perf' 
# show_me = 'test_perf'
# show_me = 'PS'
# show_me = 'SD'
# show_me = 'CCGP'
# show_me = 'mean_grad'
# show_me = 'std_grad'
show_me = 'PR'
# show_me = 'sparsity'

final_ps = []
final_ps_err = []
for i in range(len(mets)):
    final_ps.append(np.nanmean(mets[i][show_me][:,-1],0))
    final_ps_err.append(np.nanstd(mets[i][show_me][:,-1],0))

final_ps = np.array(final_ps)
final_ps_err = np.array(final_ps_err)

plt.plot(output_type, final_ps)
plt.fill_between(output_type, 
                 final_ps-final_ps_err,
                 final_ps+final_ps_err,
                 alpha=0.5)

# plt.legend(['Trained Dichotomies','Untrained Dichotomies'])
plt.ylabel('Final ' + show_me)
plt.xlabel('Output "simpliciality"')


#%%
netid = 0

all_PS = []
all_CCGP = []
all_SD = []
for model, args in zip(all_nets[netid], dicts[netid]):
    
    this_exp.load_other_info(args)
    this_exp.load_data(SAVE_DIR)

    z = model(this_exp.train_data[0])[2].detach().numpy()
    # z = this_exp.train_data[0].detach().numpy()
    # z = linreg.predict(this_exp.train_data[0])@W1.T
    n_compute = np.min([5000, z.shape[0]])
    
    idx = np.random.choice(z.shape[0], n_compute, replace=False)
    # idx_tst = idx[::4] # save 1/4 for test set
    # idx_trn = np.setdiff1d(idx, idx_tst)
    
    cond = this_exp.train_conditions[idx]
    # cond = util.decimal(this_exp.train_data[1][idx,...])
    num_cond = len(np.unique(cond))
    
    xor = np.where(~(np.isin(range(8), args['dichotomies'][0])^np.isin(range(8), args['dichotomies'][1])))[0]
    # Loop over dichotomies
    D = assistants.Dichotomies(num_cond, args['dichotomies']+[xor], extra=50)
    clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
    gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
    dclf = assistants.LinearDecoder(N, D.ntot, svm.LinearSVC)
    # clf = LinearDecoder(this_exp.dim_input, 1, MeanClassifier)
    # gclf = LinearDecoder(this_exp.dim_input, 1, svm.LinearSVC)
    # dclf = LinearDecoder(this_exp.dim_input, D.ntot, svm.LinearSVC)
    
    # K = int(num_cond/2) - 1 # use all but one pairing
    K = int(num_cond/4) # use half the pairings
    
    PS = np.zeros(D.ntot)
    CCGP = np.zeros(D.ntot)
    d = np.zeros((n_compute, D.ntot))
    pos_conds = []
    for i, pos in enumerate(D):
        pos_conds.append(pos)
        print('Dichotomy %d...'%i)
        # parallelism
        PS[i] = D.parallelism(z[idx,:], cond, clf)
        
        # CCGP
        CCGP[i] = D.CCGP(z[idx,:], cond, gclf, K)
        
        # shattering
        d[:,i] = D.coloring(cond)
        
    # dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
    dclf.fit(z[idx,:], d, tol=1e-5)
    
    z = model(this_exp.test_data[0])[2].detach().numpy()
    # z = this_exp.test_data[0].detach().numpy()
    # z = linreg.predict(this_exp.test_data[0])@W1.T
    idx = np.random.choice(z.shape[0], n_compute, replace=False)
    
    d_tst = np.array([D.coloring(this_exp.test_conditions[idx]) for _ in D]).T
    SD = dclf.test(z[idx,:], d_tst).squeeze()
    
    all_PS.append(PS)
    all_CCGP.append(CCGP)
    all_SD.append(SD)

PS = np.array(all_PS).mean(0)
CCGP = np.array(all_CCGP).mean(0)
SD = np.array(all_SD).mean(0)

