import socket
import os
import sys

CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'


sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/Documents/github/continuous_parallelism/')

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
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from sklearn import svm, manifold, linear_model, neighbors
from sklearn import gaussian_process as gp
from tqdm import tqdm
import umap

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import experiments as exp

# jeff's code
import composite_code as cmp

#%%
def weight_func(x, x0, sigma=1):
    alph_ = np.exp(-(x-x0)**2 / (2*sigma)**2)
    return alph_/np.mean(alph_)

#%%
num_var = 2
ndat = 5000
dim = 150

bw = 0.05
n_ps = 200 # subsample size for parallelism
b_ps = 1 # number of 'batches' for parallelism
eps_ps = 0.5 # epsilon for parallelism
n_neigh = 30

n_bin = 10

lin_pwr = 1
nonlin_pwr = 5
n_nonlin = 10

all_PS = []
all_ccgp = []
all_pr = []
for lin_pwr in np.linspace(0,10,30):
    
    inputs = np.random.randn(5000,2)
    
    r = cmp.CombinedCode(dim, lin_pwr, nonlin_pwr, inp_dim=num_var, rotation=True, nonlin_components=n_nonlin)
    
    z = r.stim_resp(inputs, add_noise=True)
    
    U, S, V = la.svd(z-z.mean(0)[None,:],full_matrices=False)
    
    all_pr.append((np.sum(S**2)**2)/np.sum(S**4))
    
    # colorby = la.norm(inputs,axis=1)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.scatter(U[:,0],U[:,1],U[:,2], c=colorby)
    # # for i in np.unique(these_conds):
    # #     c = [int(i), int(np.mod(i+1,U.shape[0]))]
    # #     ax.plot(U[c,0],U[c,1],U[c,2],'k')
    # util.set_axes_equal(ax)
    # plt.title('PCA dimension: %.2f'%((np.sum(S**2)**2)/np.sum(S**4)))
    
    # #%%
    # n_vis = 25
    # which_neur = np.random.choice(dim)
    # # which_neur = 0
    
    # x_vis,y_vis = np.meshgrid(np.linspace(-2,2,25),np.linspace(-2,2,25))
    # inp_vis = np.stack([x_vis.flatten(), y_vis.flatten()])
    
    # z_vis = r.stim_resp(inp_vis.T,add_noise=False)
    
    # plt.imshow(z_vis[:,which_neur].reshape(n_vis,n_vis))
    
    # % Compute
    # x0 = -1 # training context 
    # x_tst = np.linspace(-1,1,n_bin)
    x_tst = np.quantile(inputs[:,0],np.linspace(0.1,0.9,n_bin))
    
    x = inputs[:,1]
    y = inputs[:,0]
    
    trnidx = np.random.choice(ndat,int(0.8*ndat)) # training subset
    tstidx = np.isin(range(ndat), trnidx) # testing subset
    
    # mdl = linear_model.LogisticRegression()
    # mdl_x = linear_model.LinearRegression() # CCGP classifier
    # mdl_y = linear_model.LinearRegression() # CCGP classifier
    mdl_x = linear_model.Ridge() # CCGP classifier
    mdl_y = linear_model.Ridge() # CCGP classifier
    rclf = linear_model.LinearRegression() # Parallelism regressor
    knn = neighbors.NearestNeighbors(n_neighbors=n_neigh)
    
    CCGP = []
    perf = []
    PS = []
    ps_perf = [[] for i in range(num_var)]
    proj_var = []
    for x0  in tqdm(x_tst): 
        w_x = weight_func(x, x0, sigma=bw)
        w_y = weight_func(y, x0, sigma=bw)
        
        ## CCGP
        mdl_y.fit(z[trnidx,:], y[trnidx], w_x[trnidx])
        mdl_x.fit(z[trnidx,:], x[trnidx], w_y[trnidx])
        perf.append([mdl_y.score(z[tstidx,:], y[tstidx], w_x[tstidx]),
                     mdl_x.score(z[tstidx,:], x[tstidx], w_y[tstidx])])
        
        ccgp = []
        # perf = []
        for x_ in x_tst:
            w_x_tst = weight_func(x, x_, sigma=bw)
            w_y_tst = weight_func(y, x_, sigma=bw)
            ccgp.append([mdl_y.score(z[tstidx,:],y[tstidx], w_x_tst[tstidx]),
                         mdl_x.score(z[tstidx,:],x[tstidx], w_y_tst[tstidx])])
        
        CCGP.append(ccgp)
    
        ## Parallelism of the continuous variable 
        psps = [[] for i in range(num_var)]
        # pvpv = [[] for i in range(num_var)]
        for i in range(num_var):
            w_pos = weight_func(inputs[:,i], x0-eps_ps/2, sigma=bw)
            w_neg = weight_func(inputs[:,i], x0+eps_ps/2, sigma=bw)
            # w_pos = (x>x0-eps_ps/2)&(x<x0)
            # w_neg = (x>x0)&(x<x0+eps_ps/2)
            w_tot = (w_pos/w_pos.mean()+w_neg/w_neg.mean())/2
            
            rclf.fit(z[trnidx,:], inputs[trnidx,i], w_tot[trnidx])
            ps_perf[i].append(rclf.score(z[tstidx,:], inputs[tstidx,i], w_tot[tstidx]))
            d_glob = rclf.coef_.T
            d_glob /= la.norm(d_glob)
            
            # d_glob = emb.basis[:,1].numpy()
            
            for b in range(b_ps):
                idx_pos = np.random.choice(range(ndat), n_ps, replace=False, p=w_pos/w_pos.sum())
                # idx_neg = np.random.choice(np.unique(cond)[~coloring], subsamp, replace=False)
                idx_neg = np.random.choice(range(ndat), n_ps, replace=False, p=w_neg/w_neg.sum())
                
                knn.fit(z)
                neigh_idx_pos = knn.kneighbors(z[idx_pos,:])[1]
                z_pos = z[neigh_idx_pos,:].mean(1)
                # knn.fit(z[idx_neg,:])
                neigh_idx_neg = knn.kneighbors(z[idx_neg,:])[1]
                z_neg = z[neigh_idx_neg,:].mean(1)
                
                # C = z.T[:,None,idx_neg] - z.T[:,idx_pos,None]
                C = z_neg.T[...,None] - z_pos.T[...,None,:]
                C /= (la.norm(C,axis=0)+1e-4)
                cost = np.sum(d_glob.squeeze()[:,None,None]*C,0)
                
                # find the best pairing by an approximate cost function
                row, col = opt.linear_sum_assignment(cost, maximize=True)
                # g_pos = idx_pos[row]
                # g_neg = idx_neg[col]
                
                # compute parallelism of that pairing
                # vecs = (z[g_neg,:]-z[g_pos,:])
                vecs = z_neg[row,:] - z_pos[col,:]
                # # pvpv.append((vecs@d_glob).std())
                # mnvec = (z[g_neg,:]-z[g_pos,:]).mean(0)
                # pvpv.append((mnvec@d_glob)/la.norm(mnvec))
                vecs /= (la.norm(vecs,axis=1,keepdims=True)+1e-4)
                csin = np.einsum('ik...,jk...->ij...', vecs, vecs)
                psps[i].append(2*np.triu(csin.T,1).sum()/(len(idx_pos)*(len(idx_pos)-1)))
        
        PS.append(psps)
        # proj_var.append(pvpv)
    
    all_PS.append(np.mean(PS,-1))
    all_ccgp.append(np.array(CCGP))

# CCGP = np.array(CCGP)

# plt.plot(x_tst,np.mean(PS,-1))
# plt.ylabel('Parallelism')
# plt.legend(['Coord%d'%crd for crd in range(num_var)])

# plt.figure()
# plt.plot(x_tst,np.array(ps_perf).T)
# plt.ylabel('Local deocder accuracy')
# plt.legend(['Coord%d'%(crd+1) for crd in range(num_var)])

#%%
which_var = 1

plt.imshow(CCGP[:,:,which_var], extent=[x_tst.min(),x_tst.max(),x_tst.max(),x_tst.min()])
plt.clim([0,1])
plt.colorbar()

# plt.xtick_labels(x_tst)
# plt.ytick_labels(x_tst)
plt.xlabel('Test context')
plt.ylabel('Train context')

