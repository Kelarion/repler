CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
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
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

# my code
import students
import assistants
import experiments as exp
import util

#%% Set up the task
task = util.RandomDichotomies(8,2,0)
# task = util.ParityMagnitude()

# this_exp = exp.mnist_multiclass(task, SAVE_DIR, abstracts=abstract_variables)
this_exp = exp.random_patterns(task, SAVE_DIR, 
                               num_class=8,
                               dim=100,
                               var_means=1)

input_states = this_exp.train_data[0]
output_states = this_exp.train_data[1]

#%%
manual = True
# manual = False
ppp = 1 # 0 is MSE, 1 is cross entropy

correct_mse = False # if True, rescales the MSE targets to be more like the log odds

N = 100

lr = 1e-4

dset = torch.utils.data.TensorDataset(input_states[:5000], output_states[:5000])
dl = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)

# set up network (2 layers)
ba = 1/np.sqrt(N)
W1 = torch.FloatTensor(N,input_states.shape[1]).uniform_(-ba,ba)
b1 = torch.FloatTensor(N,1).uniform_(-ba,ba)
W1.requires_grad_(True)
b1.requires_grad_(True)

ba = 1/np.sqrt(N)
W2 = torch.FloatTensor(N,N).uniform_(-ba,ba)
b2 = torch.FloatTensor(N,1).uniform_(-ba,ba)
W2.requires_grad_(True)
b2.requires_grad_(True)

ba = 1/np.sqrt(output_states.shape[1])
W = torch.FloatTensor(output_states.shape[1],N).uniform_(0,2*ba)
b = torch.FloatTensor(output_states.shape[1],1).uniform_(0,2*ba)

optimizer = optim.Adam([W1, b1, W2, b2], lr=1e-4)


train_loss = [] 
test_perf = []
PS = []
CCGP = []
lindim = [] 
for epoch in tqdm(range(2000)):

    # loss = net.grad_step(dl, optimizer)
    
    running_loss = 0
    
    idx = np.random.choice(this_exp.test_data[0].shape[0], 5000, replace=False)
    z1 = nn.ReLU()(torch.matmul(W1,this_exp.test_data[0][idx,:].T) + b1)
    z = nn.ReLU()(torch.matmul(W2,z1) + b2)
    pred = torch.matmul(W,z) + b
    
    perf = ((pred.T>0) == this_exp.test_data[1][idx,:]).detach().numpy().mean(0)
    test_perf.append(perf)
    
    # this is just the way I compute the abstraction metrics, sorry
    clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
    gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
    D = assistants.Dichotomies(len(np.unique(this_exp.test_conditions)),
                                this_exp.task.positives, extra=0)
    
    ps = []
    ccgp = []
    for _ in D:
        ps.append(D.parallelism(z.T.detach().numpy(), this_exp.test_conditions[idx], clf))
        ccgp.append(D.CCGP(z.T.detach().numpy(), this_exp.test_conditions[idx], gclf))
    PS.append(ps)
    CCGP.append(ccgp)
    
    _, S, _ = la.svd(z.detach()-z.mean(1).detach()[:,None], full_matrices=False)
    eigs = S**2
    lindim.append((np.sum(eigs)**2)/np.sum(eigs**2))
    
    for j, btch in enumerate(dl):
        optimizer.zero_grad()
        
        inps, outs = btch
        z1 = nn.ReLU()(torch.matmul(W1,inps.T) + b1)
        z = nn.ReLU()(torch.matmul(W2,z1) + b2)
        pred = torch.matmul(W,z) + b
        
        # change the scale of the MSE targets, to be more like x-ent
        if (ppp == 0) and correct_mse:
            outs = 1000*(2*outs-1)
        
        # loss = -students.Bernoulli(2).distr(pred).log_prob(outs.T).mean()
        loss = ppp*nn.BCEWithLogitsLoss()(pred.T, outs) + (1-ppp)*nn.MSELoss()(pred.T,outs)
        
        if manual:
            errb = (outs.T - nn.Sigmoid()(pred)) # bernoulli
            errg = (outs.T - pred) # gaussian
            
            err = ppp*errb + (1-ppp)*errg # convex sum, in case you want that
            
            d2 = (W.T@err)*(z>0) # gradient of the currents
            W2.grad = -(d2@z1.T)/inps.shape[0]
            b2.grad = -d2.mean(1, keepdim=True)
            
            d1 = (W2@d2)*(z1>0)
            W1.grad = -(d1@inps)/inps.shape[0]
            b1.gad = -d1.mean(1, keepdim=True)
        
            # W1 += lr*dw
            # b1 += lr*db
        else:
            loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
    # train_loss.append(loss)
    # print('epoch %d: %.3f'%(epoch,running_loss/(j+1)))
    train_loss.append(running_loss/(j+1))
    # print(running_loss/(i+1))
    
    
    
    
    
    
    