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
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment as lsa

# import umap
from cycler import cycler

# my code
import students
import assistants
import experiments as exp
import util
import tasks
import plotting as dicplt
import grammars as gram

#%%

def parallelism_score(Kz, Ky, mask, eps=1e-12):
    
    # dz = (torch.sum(Kz*Ky*(1-mask),0, keepdims=True) + torch.sum(Kz*Ky*(1-mask),1, keepdims=True))
    # dz = torch.sqrt(torch.abs(torch.sum(dz*(1-mask-torch.eye(len(Kz))),0)))
    
    if torch.sum(mask[Ky != 0])>0:
        # dz = torch.sum(dot2dist(Kz)*(1-mask-torch.eye(len(Kz))),0)
        # norm = dz[:,None]*dz[None,:]
        dist = torch.diag(Kz)[:,None] + torch.diag(Kz)[None,:] - 2*Kz
        dz = torch.sqrt(torch.abs(dist[(1-mask-torch.eye(len(Kz)))>0]))
        norm = (dz[:,None]*dz[None,:]).flatten()
        
        # numer = torch.sum((Kz*Ky*mask)[Ky != 0]/norm[Ky != 0])
        numer = torch.sum((Kz*Ky*mask)[Ky != 0]/norm)
        denom = (torch.sum(torch.tril(mask)[Ky != 0])/2) #+ eps
        return numer/denom
    else:
        return 0
    
    # return torch.sum((Kz*Ky*mask)/norm)/(torch.sum(torch.tril(mask))/2)

def dot2dist(K):
    return torch.sqrt(torch.abs(torch.diag(K)[:,None] + torch.diag(K)[None,:] - 2*K))

def make_mask(reps, y, permute=False):
    pos = np.where(y==0)[0]
    neg = np.where(y==1)[0]
    
    if permute:
        # try to deal with non-unique solutions
        sols = []
        for p in permutations(range(len(neg))):
            order = lsa(util.dot_product(reps,reps)[pos,:][:,neg[list(p)]].T, maximize=True)
            sols.append(order[1][np.argsort(p)])
        ix2 = neg[sols[np.random.choice(len(sols))]]
    else:
        order = lsa(util.dot_product(reps,reps)[pos,:][:,neg].T, maximize=True)
        ix2 = neg[order[1]]
    ix1 = pos[order[0]]
    
    mask = 1 - torch.eye(len(y))
    mask[ix1,ix2] = 0
    mask[ix2,ix1] = 0
    
    return mask

# %% Pick data format
K = 2
# respect = False
respect = True

# layers = [K**0,K**1,K**2]
layers = [1, 2, 4]
# layers = [1,1,1]

Data = gram.HierarchicalData(layers, fan_out=K, respect_hierarchy=respect)

ll = Data.labels(Data.terminals)
labs = np.where(np.isnan(ll), np.nanmax(ll)+1, ll)

Ky_all = np.sign((ll[:,:,None]-0.5)*(ll[:,None,:]-0.5))
Ky_all = torch.tensor(np.where(np.isnan(Ky_all), 0, Ky_all))

reps = Data.represent_labels(Data.terminals)

#%%
num_var = Data.num_vars
num_cond = Data.num_data

# N_param = num_cond*(num_cond-1)/2
idx = torch.tril_indices(num_cond, num_cond, offset=-1)
N_param = idx.shape[-1]

Kz_C = torch.FloatTensor(N_param).uniform_(-1,1)
Kz_S = torch.FloatTensor(num_cond).uniform_(0,1)

Kz_C.requires_grad_(True)
Kz_S.requires_grad_(True)
optimizer = optim.Adam([Kz_C, Kz_S], lr=1e-2)

ps = []
for epoch in tqdm(range(int(1e5))):
    optimizer.zero_grad()
    
    D = torch.diag(Kz_S**2)
    C = torch.diag(torch.abs(Kz_S))
    C[idx[0,:],idx[1,:]] = Kz_C
    L = C@torch.diag(1/(torch.abs(Kz_S)+1e-15))
    
    Kz = L@D@L.T
    
    Kz.retain_grad()
    D.retain_grad()
    C.retain_grad()
    L.retain_grad()
    
    # norm.retain_grad()
    # dz.retain_grad()
    # numer.retain_grad()

    loss = 0
    for i in range(num_var):
        loss -= parallelism_score(Kz, Ky_all[i], make_mask(reps, ll[i,:], permute=False))
    ps.append(-loss.item())
    
    loss += (torch.diag(Kz)-3).pow(2).mean()*0.01
    # ps.append(parallelism_score(Kz, Ky[0], make_mask(reps, labs[0,:])).item())
    
    loss.backward()
    
    if torch.any(torch.isnan(Kz_S.grad))|torch.any(torch.isnan(Kz_C.grad)):
        print('doi!')
        break
    
    optimizer.step()




