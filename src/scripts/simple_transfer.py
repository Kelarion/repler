import os, sys, re

if sys.platform == 'linux':
    CODE_DIR = '/home/kelarion/github/repler/src'
    SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
else:
    CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
    SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'

import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
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
from tqdm import tqdm

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

decay = 1.0

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
N = N_list[0]
#%%
# new_task = util.Digits()
# new_task = util.ParityMagnitudeEnumerated()
new_task = util.ParityMagnitude()
bsz = 1
lr = 1e-3
nepoch = 10

n_compute = 1000 # for test error

new_exp = exp.mnist_multiclass(new_task, SAVE_DIR)

# glm = nn.Linear(N, new_task.dim_output)
# glm = nn.Linear(784, new_task.dim_output)
# glm = Feedforward([784, 100, 50, new_task.dim_output], ['ReLU', 'ReLU', None])
# glm = MultiGLM(Feedforward([784, 100, 50]), nn.Linear(50,new_task.dim_output), new_task.obs_distribution)

train = []
test = []
w = []
for m, model in enumerate(all_nets[0]):
    
    z_pretrained = model(new_exp.train_data[0])[2].detach()
    # z_pretrained = train_dat[0]
    targ = new_exp.train_data[1][:,0]
    
    z_test = model(new_exp.test_data[0])[2].detach()
    # z_test = test_dat[0]
    targ_test = new_exp.test_data[1][:,0]
    
    new_dset = torch.utils.data.TensorDataset(z_pretrained, targ)
    dl = torch.utils.data.DataLoader(new_dset, batch_size=bsz, shuffle=True)
    
    
    # glm = nn.Linear(N, new_task.dim_output)
    glm = nn.Linear(N, 1)
    optimizer = new_exp.opt_alg(glm.parameters(), lr=lr)
    
    # print('Model %d'%m)
    # optimize
    train_loss = []
    test_error = []
    with tqdm(range(nepoch), total=nepoch, desc='Model %d'%m, postfix=[{'loss':0,'error':0}]) as looper:
        for epoch in looper:
            
            # running_error = 0
            running_loss = 0
            for i, (x,y) in enumerate(dl):
                with torch.no_grad():
                    idx = np.random.choice(len(targ_test), n_compute, replace=False)
                    pred = glm(z_test[idx,:])[:,0]
                    terr = 1- (new_task.correct(pred, targ_test[idx])/n_compute)
            
                optimizer.zero_grad()
                
                eta = glm(x).squeeze()
                
                # terr = 1- (new_task.correct(eta, y)/x.shape[0])
                loss = -new_task.obs_distribution.distr(eta).log_prob(y).sum()
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                # running_error += terr.item()
                
                train_loss.append(loss.item())
                test_error.append(terr)
            looper.postfix[0]['loss'] = running_loss/(i+1)
            looper.postfix[0]['error'] = terr
            looper.update()
    train.append(train_loss)
    test.append(test_error)
    w.append(glm.weight.data.numpy())
        # print('Epoch %d: loss=%.3f; error=%.3f'%(epoch, running_loss/(i+1), terr))
train = np.stack(train)
test = np.stack(test)#.squeeze()

#%%        
plt.plot(range(1,train.shape[1]+1),test.mean(0))
plt.fill_between(range(1,train.shape[1]+1), 
                 test.mean(0)+test.std(0),
                 test.mean(0)-test.std(0),
                 alpha=0.5)
plt.semilogx()
plt.semilogy()

