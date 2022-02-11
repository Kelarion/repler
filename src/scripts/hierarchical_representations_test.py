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

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

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

# def make_mask(reps, y, permute=False):
#     pos = np.where(y==0)[0]
#     neg = np.where(y==1)[0]
    
#     if permute:
#         # try to deal with non-unique solutions
#         sols = []
#         for p in permutations(range(len(neg))):
#             order = lsa(util.dot_product(reps,reps)[pos,:][:,neg[list(p)]].T, maximize=True)
#             sols.append(order[1][np.argsort(p)])
#         ix2 = neg[sols[np.random.choice(len(sols))]]
#     else:
#         order = lsa(util.dot_product(reps,reps)[pos,:][:,neg].T, maximize=True)
#         ix2 = neg[order[1]]
#     ix1 = pos[order[0]]
    
#     mask = 1 - torch.eye(len(y))
#     mask[ix1,ix2] = 0
#     mask[ix2,ix1] = 0
    
#     return mask
# def make_mask(y, p):
#     pos = np.where(y==0)[0]
#     neg = np.where(y==1)[0]
    
#     mask = 1 - torch.eye(len(y))
#     mask[pos,neg[p]] = 0
#     mask[neg[p],pos] = 0
    
#     return mask

# %% Pick data format
K = 2
# respect = False
respect = True

# layers = [K**0,K**1,K**2]
layers = [1, 2, 2]
# layers = [1,1,1]

Data = gram.HierarchicalData(layers, fan_out=K, respect_hierarchy=respect)

ll = Data.labels(Data.terminals)
labs = np.where(np.isnan(ll), np.nanmax(ll)+1, ll)

Ky_all = np.sign((ll[:,:,None]-0.5)*(ll[:,None,:]-0.5))
Ky_all = torch.tensor(np.where(np.isnan(Ky_all), 0, Ky_all))

reps = Data.represent_labels(Data.terminals)
Ky = util.dot_product(reps,reps)

plt.figure()
plt.subplot(131)
pos = graphviz_layout(Data.variable_tree, prog="twopi")
nx.draw(Data.variable_tree, pos, node_color=np.array(Data.variable_tree.nodes).astype(int), cmap='nipy_spectral')
dicplt.square_axis()
plt.subplot(132)
plt.imshow(ll, 'bwr')
plt.subplot(133)
plt.imshow(util.dot_product(reps,reps), 'binary')

#%%
N = 1000
n_samp = 50

noise = 2.0
n_samp_dec = 1000

clf = svm.LinearSVC()

all_PS = []
kernel_align = []
all_ccgp = []
all_decode = []
all_hierarchy = []
for pwr in tqdm(np.linspace(-3,3,21)):
    
    ps_samps = []
    ka_samps = []
    ccg_samps = []
    dec_samps = []
    hier_samps = []
    
    for s in range(n_samp):
        zs = []
        
        pop_N = N*((np.arange(1,4.)**pwr)/np.sum((np.arange(1,4.)**pwr)))
        for i,var in enumerate(np.split(Data.represent_labels(Data.terminals), 2*np.cumsum(layers))[:-1]):
            zs.append( var.T@np.random.randn(var.shape[0], int(pop_N[i]) ))
            # reps.append( var.T@np.random.randn(var.shape[0], N) )
            
        z = np.concatenate(zs,axis=-1)
        
        # Kz = util.dot_product((z-z.mean(0)).T, (z-z.mean(0)).T)/z.shape[1]
        Kz = util.dot_product(z.T,z.T)/z.shape[1]
        
        PS = []
        CCGP = []
        decoding = []
        
        for i,y in enumerate(labs):
            pos = np.where(y==0)[0]
            neg = np.where(y==1)[0]
            
            these = np.where(y!=2)[0]
            idx = np.random.choice(len(these), n_samp_dec)
            z_dec = z[these[idx],:] + np.random.randn(n_samp_dec,z.shape[1])*noise
            
            if len(pos)>1:
                ps = []
                for p in permutations(neg):
                    mask = 1 - torch.eye(len(y))
                    mask[pos,p] = 0
                    mask[p,pos] = 0
                    
                    ps.append(parallelism_score(torch.tensor(Kz), Ky_all[i], mask))
                
                PS.append(np.max(ps))
                CCGP.append(np.mean(util.compute_ccgp(z_dec, idx, y[these[idx]], clf)))
            
            clf.fit(z_dec[:int(0.6*n_samp_dec),:], y[these[idx]][:int(0.6*n_samp_dec)])
            decoding.append(clf.score(z_dec[int(0.6*n_samp_dec):,:], y[these[idx]][int(0.6*n_samp_dec):]))
        
        hierarchy = []
        layer_vars = np.split(Data.represent_labels(Data.terminals), 2*np.cumsum(layers))[:-1]
        for l in range(len(layer_vars)-1):
            y_sup = layer_vars[l].argmax(0)
            y_sub = layer_vars[l+1].argmax(0)
            sigs = [util.decompose_covariance(z[y_sup==s,:].T,y_sub[y_sup==s])[1] for s in np.unique(y_sup)]
            
            dots = np.einsum('ikl,jkl->ij',np.array(sigs),np.array(sigs))
            csim = la.triu(dots,1)/np.sqrt((np.diag(dots)[:,None]*np.diag(dots)[None,:]))
            foo1, foo2 = np.nonzero(np.triu(np.ones(dots.shape),1))
            
            hierarchy.append(np.mean(csim[foo1,foo2]))
            
        ps_samps.append(PS)
        ccg_samps.append(CCGP)
        dec_samps.append(decoding)
        hier_samps.append(hierarchy)
        ka_samps.append(np.sum(Kz*Ky)/np.sqrt(np.sum(Ky*Ky)*np.sum(Kz*Kz)))
    
    all_PS.append(ps_samps)
    kernel_align.append(ka_samps)
    all_ccgp.append(ccg_samps)
    all_decode.append(dec_samps)
    all_hierarchy.append(hier_samps)

#%%
N = 500
n_samp = 100

noise = 0.1
n_samp_dec = 1000

clf = svm.LinearSVC()

layer_vars = np.split(Data.represent_labels(Data.terminals), 2*np.cumsum(layers))[:-1]

all_PS = []
kernel_align = []
all_ccgp = []
all_decode = []
all_hierarchy = []
for pwr in tqdm(np.linspace(0,1,20)):
    
    ps_samps = []
    ka_samps = []
    ccg_samps = []
    dec_samps = []
    hier_samps = []
    
    for s in range(n_samp):
        zs = []
        
        pop_dim = K**(np.arange(len(layers))*pwr + 1) - 1
        for i,var in enumerate(layer_vars):
            V = util.random_basis(N)[:,:var.shape[0]]
            ex = util.part2expo(pop_dim[i], var.shape[0]-1)
            spectr = np.arange(1,var.shape[0]+1, dtype=float)**(-ex)
            
            fake_cov = V@np.diag(spectr)@V.T
            
            means = util.sample_normalize( np.random.randn(N,var.shape[0]), fake_cov)
            zs.append( var.T@means.T )
            # reps.append( var.T@np.random.randn(var.shape[0], N) )
            
        z = np.concatenate(zs,axis=-1)
        
        # Kz = util.dot_product((z-z.mean(0)).T, (z-z.mean(0)).T)/z.shape[1]
        Kz = util.dot_product(z.T,z.T)/z.shape[1]
        
        PS = []
        CCGP = []
        decoding = []
        
        for i,y in enumerate(labs):
            pos = np.where(y==0)[0]
            neg = np.where(y==1)[0]
            
            these = np.where(y!=2)[0]
            idx = np.random.choice(len(these), n_samp_dec)
            z_dec = z[these[idx],:] + np.random.randn(n_samp_dec,z.shape[1])*noise
            
            if len(pos)>1:
                ps = []
                for p in permutations(neg):
                    mask = 1 - torch.eye(len(y))
                    mask[pos,p] = 0
                    mask[p,pos] = 0
                    
                    ps.append(parallelism_score(torch.tensor(Kz), Ky_all[i], mask))
                    
                PS.append(np.max(ps))
                CCGP.append(np.mean(util.compute_ccgp(z_dec, idx, y[these[idx]], clf)))
            
            clf.fit(z_dec[:int(0.6*n_samp_dec),:], y[these[idx]][:int(0.6*n_samp_dec)])
            decoding.append(clf.score(z_dec[int(0.6*n_samp_dec):,:], y[these[idx]][int(0.6*n_samp_dec):]))
        
        hierarchy = []
        for l in range(len(layer_vars)-1):
            y_sup = layer_vars[l].argmax(0)
            y_sub = layer_vars[l+1].argmax(0)
            sigs = [util.decompose_covariance(z[y_sup==s,:].T,y_sub[y_sup==s])[1] for s in np.unique(y_sup)]
            
            dots = np.einsum('ikl,jkl->ij',np.array(sigs),np.array(sigs))
            csim = la.triu(dots,1)/np.sqrt((np.diag(dots)[:,None]*np.diag(dots)[None,:]))
            foo1, foo2 = np.nonzero(np.triu(np.ones(dots.shape),1))
            
            hierarchy.append(np.mean(csim[foo1,foo2]))
            
        ps_samps.append(PS)
        ccg_samps.append(CCGP)
        dec_samps.append(decoding)
        hier_samps.append(hierarchy)
        ka_samps.append(np.sum(Kz*Ky)/np.sqrt(np.sum(Ky*Ky)*np.sum(Kz*Kz)))
    
    all_PS.append(ps_samps)
    kernel_align.append(ka_samps)
    all_ccgp.append(ccg_samps)
    all_decode.append(dec_samps)
    all_hierarchy.append(hier_samps)


#%%


# respect = False
respect = True

# layers = [K**0,K**1,K**2]
layers = [1, 1, 2]
# layers = [1,1,1]

noise = 0.2
n_samp_dec = 1000

Data = gram.HierarchicalData(layers, fan_out=K, respect_hierarchy=respect)

ll = Data.labels(Data.terminals)
labs = np.where(np.isnan(ll), np.nanmax(ll)+1, ll)

Ky_all = np.sign((ll[:,:,None]-0.5)*(ll[:,None,:]-0.5))
Ky_all = torch.tensor(np.where(np.isnan(Ky_all), 0, Ky_all))

z = Data.represent_labels(Data.terminals).T

ps_samps = []
ka_samps = []
ccg_samps = []
dec_samps = []
hier_samps = []

for s in tqdm(range(n_samp)):
    
    PS = []
    CCGP = []
    decoding = []
    
    for i,y in enumerate(labs):
        pos = np.where(y==0)[0]
        neg = np.where(y==1)[0]
        
        these = np.where(y!=2)[0]
        idx = np.random.choice(len(these), n_samp_dec)
        z_dec = z[these[idx],:] + np.random.randn(n_samp_dec,z.shape[1])*noise
        
        Kz = util.dot_product(z.T,z.T)/z.shape[1]
        
        if len(pos)>1:
            ps = []
            for p in permutations(neg):
                mask = 1 - torch.eye(len(y))
                mask[pos,p] = 0
                mask[p,pos] = 0
                
                ps.append(parallelism_score(torch.tensor(Kz), Ky_all[i], mask))
                
            PS.append(np.max(ps))
            CCGP.append(np.mean(util.compute_ccgp(z_dec, idx, y[these[idx]], clf)))
        
        clf.fit(z_dec[:int(0.6*n_samp_dec),:], y[these[idx]][:int(0.6*n_samp_dec)])
        decoding.append(clf.score(z_dec[int(0.6*n_samp_dec):,:], y[these[idx]][int(0.6*n_samp_dec):]))
    
    hierarchy = []
    for l in range(len(layer_vars)-1):
        y_sup = layer_vars[l].argmax(0)
        y_sub = layer_vars[l+1].argmax(0)
        sigs = [util.decompose_covariance(z[y_sup==s,:].T,y_sub[y_sup==s])[1] for s in np.unique(y_sup)]
        
        dots = np.einsum('ikl,jkl->ij',np.array(sigs),np.array(sigs))
        csim = la.triu(dots,1)/np.sqrt((np.diag(dots)[:,None]*np.diag(dots)[None,:]))
        foo1, foo2 = np.nonzero(np.triu(np.ones(dots.shape),1))
        
        hierarchy.append(np.mean(csim[foo1,foo2]))
    
    ps_samps.append(PS)
    ccg_samps.append(CCGP)
    dec_samps.append(decoding)
    hier_samps.append(hierarchy)


plt.figure()
plt.subplot(131)
pos = graphviz_layout(Data.variable_tree, prog="twopi")
nx.draw(Data.variable_tree, pos, node_color=np.array(Data.variable_tree.nodes).astype(int), cmap='nipy_spectral')
dicplt.square_axis()
plt.subplot(132)
plt.imshow(ll, 'bwr')
plt.subplot(133)
plt.bar(range(3), [np.mean(ps_samps), np.mean(ccg_samps), np.mean(hier_samps)])
# plt.imshow(util.dot_product(reps,reps), 'binary')


