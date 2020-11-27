
if sys.platform == 'linux':
    CODE_DIR = '/home/kelarion/github/repler/src'
    SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
else:
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
from itertools import permutations

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler
from tqdm import tqdm

from students import *
from assistants import LinearDecoder
import experiments as exp
import util

#%% Model specification -- for loading purposes
# task = util.ParityMagnitude()
task = util.RandomDichotomies(8,2,0)
# task = util.ParityMagnitudeEnumerated()
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
# this_exp = exp.mnist_multiclass(task, SAVE_DIR, 
#                                 z_prior=latent_dist,
#                                 num_layer=num_layer,
#                                 weight_decay=decay)
this_exp = exp.random_patterns(task, SAVE_DIR, 
                                num_class=8,
                                dim=100,
                                var_means=1,
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
all_args = [[] for _ in N_list]
mets = [[] for _ in N_list]
dicts = [[] for _ in N_list]
best_perf = []
for i,n in enumerate(N_list):
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
                shp = (num,) + np.squeeze(np.array(val)).shape
                all_metrics[key] = np.zeros(shp)*np.nan
                
            all_metrics[key][j,...] = np.squeeze(val)
    
            # if (val.shape[0]==1000) or not len(val):
                # continue
            # all_metrics[key][j,...] = val
        all_nets[i].append(model)
        all_args[i].append(args)
        
    nets[i] = best_net
    mets[i] = all_metrics
    dicts[i] = this_arg
    best_perf.append(maxmin)

test_dat = this_exp.test_data
train_dat = this_exp.train_data

# digits = torchvision.datasets.MNIST(SAVE_DIR+'digits/', download=True, 
#                                     transform=torchvision.transforms.ToTensor())
N = N_list[0]
#%%
# new_task = util.Digits()
# new_task = util.DigitsBitwise()
# new_task = util.ParityMagnitudeEnumerated()
# new_task = util.ParityMagnitude()
new_task = this_exp.task
bsz = 64
lr = 1e-4

n_compute = 5000
n_svm = 30

# lin_clf = svm.LinearSVC
lin_clf = linear_model.LogisticRegression
# lin_clf = linear_model.Perceptron
# lin_clf = linear_model.RidgeClassifier

# new_exp = exp.mnist_multiclass(new_task, SAVE_DIR)
new_exp = this_exp

# glm = nn.Linear(N, new_task.dim_output)
# glm = nn.Linear(784, new_task.dim_output)
# glm = Feedforward([784, 100, 50, new_task.dim_output], ['ReLU', 'ReLU', None])
# glm = MultiGLM(Feedforward([784, 100, 50]), nn.Linear(50,new_task.dim_output), new_task.obs_distribution)

error = np.zeros((len(all_nets[0])*n_svm, new_task.num_var))
w = np.zeros((len(all_nets[0])*n_svm, N, new_task.num_var))
i = 0
for m, model in enumerate(all_nets[0]):
    for _ in range(n_svm):
        print('Model %d'%m)
        z_pretrained = model(new_exp.train_data[0])[2].detach()
        # z_pretrained = train_dat[0]
        targ = new_exp.train_data[1]
        
        idx = np.random.choice(z_pretrained.shape[0], n_compute, replace=False)
        
        z_test = model(new_exp.test_data[0])[2].detach()
        # z_test = test_dat[0]
        targ_test = new_exp.test_data[1]
        
        clf = LinearDecoder(N, new_task.num_var, lin_clf)
        clf.fit(z_pretrained[idx,:], targ[idx,...], max_iter=5000)
        
        w[i,:,:] = clf.coefs.squeeze().T
        error[i,:] = clf.test(z_test.numpy(), targ_test.numpy()).squeeze()
        
        i+=1
        # print('Epoch %d: loss=%.3f; error=%.3f'%(epoch, running_loss/(i+1), terr))

#%%        
plt.figure()
for m in range(w.shape[0]):
    plt.scatter(w[m,:,0],w[m,:,1], alpha=0.4, edgecolors='none')

plt.axis('scaled')
plt.xlim([np.min(plt.xlim()+plt.ylim()),np.max(plt.xlim()+plt.ylim())])
plt.ylim(plt.xlim())
plt.plot(plt.xlim(),[0,0], '--', c=(0.5,0.5,0.5))
plt.plot([0,0], plt.ylim(), '--', c=(0.5,0.5,0.5))
plt.plot(plt.xlim(), plt.xlim(), '-.', c=(0.5,0.5,0.5))

plt.xlabel('Parity classifier weight')
plt.ylabel('Magnitude classifier weight')

#%%
task = util.RandomDichotomies(8,2,0)
# task = util.ParityMagnitude()

# this_exp = exp.mnist_multiclass(task, SAVE_DIR, abstracts=abstract_variables)
this_exp = exp.random_patterns(task, SAVE_DIR, 
                               num_class=8,
                               dim=100,
                               var_means=1)


abstract_conds = util.decimal(this_exp.train_data[1])
dim = 50
num_init = 30

# lin_clf = svm.LinearSVC
lin_clf = linear_model.LogisticRegression
# lin_clf = linear_model.Perceptron
# lin_clf = linear_model.RidgeClassifier

emb = util.ContinuousEmbedding(dim, 0.0)
# z = emb(this_exp.train_data[1])
# _, _, V = la.svd(z-z.mean(0)[None,:],full_matrices=False)
# V = V[:3,:]

vals = np.arange(0.0,1.1,0.1)
pr = []
pcs = []
par = []
ccgp = []
dist = []
sd = []

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
w = []
err = []
for f in vals:
    weights = []
    errors = []
    print(f)
    for i in range(num_init):
        emb = util.ContinuousEmbedding(dim, f)
        z = emb(this_exp.train_data[1])
        
        eps1 = np.random.randn(5000, z.shape[1])*0.2
        eps2 = np.random.randn(5000, z.shape[1])*0.2
        
        clf = LinearDecoder(dim, 2, lin_clf)
        clf.fit(z[:5000,:]+eps1, this_exp.train_data[1][:5000,...], max_iter=5000)
        
        weights.append(clf.coefs.squeeze().T)
        errors.append(clf.test(z[:5000,:].numpy()+eps2, this_exp.train_data[1][:5000,:].numpy()).squeeze())
    w.append(weights)
    err.append(errors)

w = np.array(w)
err = np.array(err)

#%%
r = 10

plt.figure()
for m in range(w.shape[1]):
    plt.scatter(w[r,m,:,0],w[r,m,:,1], alpha=0.4, edgecolors='none')

plt.axis('scaled')
plt.xlim([np.min(plt.xlim()+plt.ylim()),np.max(plt.xlim()+plt.ylim())])
plt.ylim(plt.xlim())
plt.plot(plt.xlim(),[0,0], '--', c=(0.5,0.5,0.5))
plt.plot([0,0], plt.ylim(), '--', c=(0.5,0.5,0.5))
plt.plot(plt.xlim(), plt.xlim(), '-.', c=(0.5,0.5,0.5))

plt.xlabel('Parity classifier weight')
plt.ylabel('Magnitude classifier weight')



