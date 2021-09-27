import socket
import os
import sys

if socket.gethostname() == 'kelarion':
    if sys.platform == 'linux':
        CODE_DIR = '/home/kelarion/github/repler/src'
        SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    else:
        CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
        SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    openmind = False
elif socket.gethostname() == 'openmind7':
    CODE_DIR = '/home/malleman/repler/'
    SAVE_DIR = '/om2/user/malleman/abstraction/'
    openmind = True
else:    
    CODE_DIR = '/rigel/home/ma3811/repler/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
    openmind = False

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
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from itertools import permutations
from sklearn import svm, manifold, linear_model, neighbors
from tqdm import tqdm

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import experiments as exp

#%%
def logistic_deriv(x):
    return spc.expit(x)*(1-spc.expit(x))

#%%
num_cond = 2**10
num_var = 10

task = util.RandomDichotomies(num_cond,num_var,0)
# task = util.ParityMagnitude()t

# this_exp = exp.mnist_multiclass(task, SAVE_DIR, abstracts=abstract_variables)
this_exp = exp.random_patterns(task, SAVE_DIR, 
                               num_class=num_cond,
                               dim=100,
                               var_means=1)

#%% Rotation from square to tetrahedron
abstract_conds = util.decimal(this_exp.train_data[1])
dim = 50
noise = 0.2

ndat = 2000

# lin_clf = svm.LinearSVC
# lin_clf = linear_model.LogisticRegression
# lin_clf = linear_model.Perceptron
# lin_clf = linear_model.RidgeClassifier

emb = util.ContinuousEmbedding(dim, 0.0)
z = emb(this_exp.train_data[1])
_, _, V = la.svd(z-z.mean(0)[None,:],full_matrices=False)
# V = V[:3,:]
cond = this_exp.train_conditions

vals = np.linspace(0,1,10)

pr_ = []
pcs_ = []
par_ = []
ccgp_ = []
dist_res_ = []
dist_proj_ = []
dist_total_ = []
pdcorr = []
sd_ = []
cos_w_ = []
mut_inf_ = []
corr_out_ = []
apprx_cost_ = []
apprx_ps_ = []
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
for f in vals:
    
    z = emb(this_exp.train_data[1], newf=f)
       
    centroids = np.stack([z[cond==i,:].mean(0) for i in np.unique(cond)])

    _, S, _ = la.svd(centroids-centroids.mean(0)[None,:],full_matrices=False)
    pr_.append(((S**2).sum()**2)/(S**4).sum())
    
    # # U = centroids@emb.rotator.numpy()@V.T
    # U = (z[:ndat,:]+eps1)@V.T
    # U = centroids@V.T
    # pcs.append(U)
    
    eps1 = np.random.randn(ndat, dim)*noise
    eps2 = np.random.randn(ndat, dim)*noise
    
    clf = assistants.LinearDecoder(dim, 1, assistants.MeanClassifier)
    gclf = assistants.LinearDecoder(dim, 1, svm.LinearSVC)
    # rclf = svm.LinearSVC()
    rclf = linear_model.LogisticRegression()
    
    D_fake = assistants.Dichotomies(len(np.unique(this_exp.train_conditions)), this_exp.task.positives, extra=50)
    Q = len(this_exp.task.positives)
    mi = np.array([this_exp.task.information(p) for p in D_fake])
    midx = np.append(range(Q),np.flip(np.argsort(mi[Q:]))+Q)
    # these_dics = args['dichotomies'] + [D_fake.combs[i] for i in midx]
    D = assistants.Dichotomies(len(np.unique(cond)), [D_fake.combs[i] for i in midx], extra=0)
    
    PS = []
    DCorr_res = []
    DCorr_proj = []
    DCorr_marg = []
    PDCorr = []
    PDim = []
    CCGP = []
    out_corr = []
    apprx_cost = []
    apprx_ps = []
    for i,pos in tqdm(enumerate(D)):
        # print('Dichotomy %d...'%i)
        # parallelism
        PS.append(D.parallelism(z[:ndat,:], cond[:ndat], clf)[0])
        
        # # independence test
        coloring = np.isin(cond[:ndat],pos)
        # if i in [0,1]:
        #     w = emb.basis[:,i:i+1].numpy()
        # else:
        rclf.fit(z[:ndat,:]+eps1, coloring)
        w = rclf.coef_.T
        w /= la.norm(w)
        
        coloring = np.isin(np.unique(cond),pos) 
        z_err = ((torch.eye(dim) - torch.tensor(w@w.T).float())@centroids.T).numpy()
        z_proj = (centroids@w).T
        marg = (z_proj*(2*coloring-1))
        
        # z_err = ((torch.eye(dim) - torch.tensor(w@w.T).float())@(z[:ndat,:]+eps2).float().T).numpy()
        # z_proj = (((z[:ndat,:]+eps2)@w).T).numpy()
        # marg = (z_proj*(2*coloring-1))
        
        # acc = rclf.score(z[:ndat,:]+eps2, coloring)
        
        R = util.distance_correlation(z_err, coloring.astype(int)[None,:])
        # R = util.distance_correlation(z_err.numpy(), z_proj.numpy())
        DCorr_res.append(R)
        # R = util.distance_correlation(z_proj, coloring.astype(int)[None,:])
        R = util.distance_covariance(z_err,z_err)
        DCorr_proj.append(R)
        
        R = util.distance_correlation(centroids.T, coloring.astype(int)[None,:])
        # R = util.distance_correlation((z[:ndat,:]+eps2).T, coloring.astype(int)[None,:])
        DCorr_marg.append(R)
        
        # R_xyz = util.partial_distance_correlation((z[:ndat,:]+eps2).T,
        #                                           coloring.astype(int)[None,:], 
        #                                           z_proj)
        R_xyz = util.partial_distance_correlation(centroids.T,
                                                  coloring.astype(int)[None,:], 
                                                  z_proj)
        PDCorr.append(R_xyz)
        
        ## Assignment problem formulation
        C = centroids.T[:,coloring,None] - centroids.T[:,None,~coloring]
        C /= (la.norm(C,axis=0)+1e-4)
        cost = np.sum(w.squeeze()[:,None,None]*C,0)
        
        # find the best pairing by an approximate cost function
        row, col = opt.linear_sum_assignment(cost, maximize=True)
        g_pos = np.array(pos)[row]
        g_neg = np.unique(cond)[~coloring][col]
        
        # compute parallelism of that pairing
        vecs = (centroids[g_pos,:]-centroids[g_neg,:])
        vecs /= (la.norm(vecs,axis=1,keepdims=True)+1e-4)
        csin = np.einsum('ik...,jk...->ij...', vecs, vecs)
        psps = 2*np.triu(csin.T,1).sum()/(len(pos)*(len(pos)-1))
        
        apprx_cost.append(cost[row,col].mean())
        # apprx_ps.append(csin.mean())
        apprx_ps.append(psps)
        
        # centroids = np.stack([z_res[:,cond[:ndat]==i].mean(1).T for i in np.unique(cond)])

        # _, S, _ = la.svd(centroids-centroids.mean(0)[None,:],full_matrices=False)
        # PDim.append(((S**2).sum()**2)/(S**4).sum())
        
        # CCGP
        cntxt = D.get_uncorrelated(100)
        out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in this_exp.task.positives]))
        CCGP.append(D.CCGP(z[:ndat,:]+eps2, cond[:ndat], gclf, cntxt, twosided=True, max_iter=500))
    
    par_.append(PS)
    dist_res_.append(DCorr_res)
    dist_proj_.append(DCorr_proj)
    dist_total_.append(DCorr_marg)
    pdcorr.append(PDCorr)
    ccgp_.append(CCGP)
    mut_inf_.append(mi[midx])
    corr_out_.append(out_corr)
    # proj_pr.append(PDim)
    apprx_cost_.append(apprx_cost)
    apprx_ps_.append(apprx_ps)
    
    dclf = assistants.LinearDecoder(z.shape[1], D.ntot, svm.LinearSVC)
    dclf.fit(z[:ndat,:]+eps1, np.array([D.coloring(cond[:ndat]) for _ in D]).T, tol=1e-5)
    sd_.append(dclf.test(z[:ndat,:]+eps2, np.array([D.coloring(cond[:ndat]) for _ in D]).T))
    
    cos_w_.append(np.diag(dclf.coefs[:2,...].squeeze()@emb.basis[:,:2].numpy()))
    

    # if d in vals[[0,-1]]:
    #     ax.scatter(U[:,0],U[:,1],U[:,2],s=100,c=np.unique(cond))
    #     ax.plot(U[[0,1],0],U[[0,1],1],U[[0,1],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[1,3],0],U[[1,3],1],U[[1,3],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[3,2],0],U[[3,2],1],U[[3,2],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[2,0],0],U[[2,0],1],U[[2,0],2],color=(0.5,0.5,0.5))
    # else:
    #     ax.scatter(U[:,0],U[:,1],U[:,2],s=10, c=np.unique(cond))

R = np.repeat(np.array(corr_out_),2,-1)

#%% Visualization purposes
dim = 50
noise = 0.2
this_f = 0.0
# pos = this_exp.task.positives[0]
# pos = (0,1,3,4)
# pos = (0,1,2,3)

emb = util.ContinuousEmbedding(dim, this_f)

z = emb(this_exp.train_data[1])

centroids = np.stack([z[cond==i,:].mean(0) for i in np.unique(cond)])
eps1 = np.random.randn(ndat, dim)*noise

coloring = np.isin(cond[:ndat],pos)
rclf.fit(z[:ndat,:]+eps1, coloring)
w = rclf.coef_.T
w /= la.norm(w)
z_err = ((torch.eye(dim) - torch.tensor(w@w.T).float())@centroids.T).numpy()

z_proj = centroids@w

_, S, V = la.svd(z_err.T-z_err.mean(1)[None,:],full_matrices=False)
# _, _, V = la.svd(z_errz_err-centroids.mean(0)[None,:],full_matrices=False)
U = centroids@V.T

U_err = z_err.T@V.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

coloring = np.isin(np.unique(cond),pos)
# ax.scatter(U[:,0],U[:,1],U[:,2],s=100, c=coloring)
# ax.scatter(U_err[:,0],U_err[:,1],U_err[:,2],s=100, marker='s', c=coloring)
ax.scatter(U[:,0],U[:,1],z_proj,s=100, c=coloring)
ax.scatter(U_err[:,0],U_err[:,1],w.T@z_err,s=100, marker='s', c=coloring)

util.set_axes_equal(ax)
ax.plot([0,0],[0,0],ax.get_zlim(), 'k--')


#%% Plotting
# mask = (R.max(2)==1) # context must be an output variable
# mask = (np.abs(R).sum(2)==0) # context is uncorrelated with either output variable
# mask = (np.abs(R).sum(2)>0) # context is correlated with at least one output variable
mask = ~np.isnan(R).max(2) # context is uncorrelated with the tested variable

mutinfo_cmap = 'copper'

# plot_this = np.squeeze(ccgp_).mean(-1)
# plot_this = util.group_mean(np.squeeze(ccgp_), mask)
# plot_this = np.array(par_)
# plot_this = np.array(par_)[...,0]
# plot_this = np.squeeze(sd_)
# plot_this = np.array(dist_total_)*(np.array(dist_res_))
# plot_this = np.array(dist_res_)
# plot_this = np.array(dist_proj_) - np.array(dist_res_)
# plot_this = np.array(pdcorr)
# plot_this = np.array(apprx_cost_)
plot_this = np.array(apprx_ps_)

for i in range(35):
    if i in range(2):
        plt.plot(vals, plot_this[:,i], linewidth=2, color=cm.get_cmap(mutinfo_cmap)(np.mean(mut_inf_,0)[i]))
    else:
        plt.plot(vals, plot_this[:,i], '--', color=cm.get_cmap(mutinfo_cmap)(np.mean(mut_inf_,0)[i]))

#%%
plt.plot(vals,pr_)

#%% Perturbations from an n-cube
abstract_conds = util.decimal(this_exp.train_data[1])
dim = 1500
noise = 0.05
n_neigh = 0

num_rep = 1

subsamp = 1000

ndat = 10000

# lin_clf = svm.LinearSVC
# lin_clf = linear_model.LogisticRegression
# lin_clf = linear_model.Perceptron
# lin_clf = linear_model.RidgeClassifier

# emb = util.ContinuousEmbedding(dim, 0.0)
# z = emb(this_exp.train_data[1])
basis = torch.tensor(la.qr(np.random.rand(dim, dim))[0]).float()
z = this_exp.train_data[1]@basis[:,:num_var].T

knn = neighbors.NearestNeighbors(n_neighbors=n_neigh)

_, _, V = la.svd(z-z.mean(0)[None,:],full_matrices=False)
# V = V[:3,:]
cond = this_exp.train_conditions

vals = np.linspace(0,1,20)

pr_ = []
pcs_ = []
proj_pr_ = []
par_ = []
ccgp_ = []
dist_res_ = []
dist_proj_ = []
dist_total_ = []
pdcorr = []
sd_ = []
cos_w_ = []
mut_inf_ = []
corr_out_ = []
apprx_cost_ = []
apprx_ps_ = []
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
for d in vals:
    
    # z = emb(this_exp.train_data[1])
    z = this_exp.train_data[1]@basis[:,:num_var].T
    
    pr = []
    pcs = []
    par = []
    ccgp = []
    sd = []
    mut_inf = []
    corr_out = []
    apprx_cost = []
    apprx_ps = []
    
    D = assistants.Dichotomies(len(np.unique(this_exp.train_conditions)), this_exp.task.positives, extra=60)
    # Q = len(this_exp.task.positives)
    # mi = np.array([this_exp.task.information(p) for p in D_fake])
    # midx = np.append(range(Q),np.flip(np.argsort(mi[Q:]))+Q)
    # # these_dics = args['dichotomies'] + [D_fake.combs[i] for i in midx]
    # D = assistants.Dichotomies(len(np.unique(cond)), [D_fake.combs[i] for i in midx], extra=0)
    
    for _ in tqdm(range(num_rep)):
        dz = np.random.randn(len(np.unique(cond)), dim)
        dz /= la.norm(dz,2,1, keepdims=True)
        z += dz[cond,:]*d*np.sqrt(dim)
        
        centroids = np.stack([z[cond==i,:].mean(0) for i in np.unique(cond)])
    
        _, S, _ = la.svd(centroids-centroids.mean(0)[None,:],full_matrices=False)
        pr.append(((S**2).sum()**2)/(S**4).sum())
        
        # # U = centroids@emb.rotator.numpy()@V.T
        # U = (z[:ndat,:]+eps1)@V.T
        # U = centroids@V.T
        # pcs.append(U)
        
        clf = assistants.LinearDecoder(dim, 1, assistants.MeanClassifier)
        gclf = assistants.LinearDecoder(dim, 1, svm.LinearSVC)
        rclf = svm.LinearSVC()
        # rclf = linear_model.LogisticRegression()
        
        eps1 = np.random.randn(ndat, dim)*noise
        eps2 = np.random.randn(ndat, dim)*noise
        
        PS = []
        SD = []
        CCGP = []
        out_corr = []
        apprx_c = []
        apprx_p = []
        for i,pos in enumerate(D):
            # print('Dichotomy %d...'%i)
            # parallelism
            # PS.append(D.parallelism(z[:ndat,:], cond[:ndat], clf))
            
            coloring = np.isin(cond[:ndat],pos)
            # if i in [0,1]:
            #     w = emb.basis[:,i:i+1].numpy()
            # else:
            rclf.fit(z[:ndat,:]+eps1, coloring)
            w = rclf.coef_
            
            SD.append(rclf.score(z[:ndat,:]+eps2, coloring))
            
            if i < num_var:
                ## Assignment problem
                if subsamp < len(pos):
                    these_pos_conds = np.random.choice(pos, subsamp, replace=False)
                    these_neg_conds = np.random.choice(np.setdiff1d(np.arange(num_cond), pos),subsamp,replace=False)
                    
                    z_pos = np.array([z[cond==c,:].mean(0).numpy() for c in these_pos_conds])
                    z_neg = np.array([z[cond==c,:].mean(0).numpy() for c in these_neg_conds])
                else:
                    z_pos = centroids[pos,:]
                    z_neg = centroids[np.setdiff1d(np.arange(num_cond), pos),:]
                
                
                # if n_neigh>0:
                    # idx_pos = np.random.choice(np.arange(ndat)[coloring], subsamp, replace=False)
                    # idx_neg = np.arange(ndat)[~coloring]
                    
                #     knn.fit(z)
                    
                #     neigh_idx_pos = knn.kneighbors(z[idx_pos,:])[1]
                #     z_pos = z[neigh_idx_pos,:].mean(1).numpy()
                    
                #     neigh_idx_neg = knn.kneighbors(z[idx_neg,:])[1]
                #     z_neg = z[neigh_idx_neg,:].mean(1).numpy()
                # else:
                #     z_pos = z[idx_pos,:].numpy()
                #     z_neg = z[idx_neg,:].numpy()
    
                # C = centroids.T[:,idx_pos,None] - centroids.T[:,None,idx_neg]
                C = z_pos.T[...,None] - z_neg.T[...,None,:]
                C /= (la.norm(C,axis=0)+1e-4)
                cost = np.sum(w.squeeze()[:,None,None]*C/la.norm(w),0)
                
                # find the best pairing by an approximate cost function
                row, col = opt.linear_sum_assignment(cost, maximize=True)
                # g_pos = idx_pos[row]
                # g_neg = idx_neg[col]
                
                # compute parallelism of that pairing
                # vecs = (centroids[g_pos,:]-centroids[g_neg,:])
                vecs = z_pos[row,:] - z_neg[col,:]
                vecs /= (la.norm(vecs,axis=1,keepdims=True)+1e-4)
                csin = np.einsum('ik...,jk...->ij...', vecs, vecs)
                psps = 2*np.triu(csin.T,1).sum()/(len(z_pos)*(len(z_pos)-1))
            
                apprx_c.append(cost[row,col].mean())
                apprx_p.append(psps)
                
                # centroids = np.stack([z_res[:,cond[:ndat]==i].mean(1).T for i in np.unique(cond)])
        
                # _, S, _ = la.svd(centroids-centroids.mean(0)[None,:],full_matrices=False)
                # _, S, _ = la.svd(z_err.T-z_err.mean(1)[None,:],full_matrices=False)
                # PDim.append((np.sum(S**2)**2)/np.sum(S**4))
                
                # CCGP
                cntxt = D.get_uncorrelated(50)
                out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in this_exp.task.positives]))
                CCGP.append(D.CCGP(z[:ndat,:]+eps2, cond[:ndat], gclf, cntxt, twosided=True, max_iter=100))
        
        par.append(PS)
        ccgp.append(CCGP)
        # mut_inf.append(mi[midx])
        corr_out.append(out_corr)
        apprx_cost.append(apprx_c)
        apprx_ps.append(apprx_p)
        
        sd.append(SD)
        # dclf = assistants.LinearDecoder(z.shape[1], D.ntot, svm.LinearSVC)
        # dclf.fit(z[:ndat,:]+eps1, np.array([D.coloring(cond[:ndat]) for _ in D]).T, tol=1e-5)
        # sd.append(dclf.test(z[:ndat,:]+eps2, np.array([D.coloring(cond[:ndat]) for _ in D]).T))
        
        # cos_w.append(np.diag(dclf.coefs[:2,...].squeeze()@emb.basis[:,:2].numpy()))
    
    pr_.append(np.mean(pr,0))
    par_.append(np.mean(par,0))
    ccgp_.append(np.mean(ccgp,0))
    sd_.append(sd)
    # mut_inf_.append(np.mean(mut_inf,0))
    corr_out_.append(corr_out)
    apprx_cost_.append(np.mean(apprx_cost,0))
    apprx_ps_.append(np.mean(apprx_ps,0))
    # if d in vals[[0,-1]]:
    #     ax.scatter(U[:,0],U[:,1],U[:,2],s=100,c=np.unique(cond))
    #     ax.plot(U[[0,1],0],U[[0,1],1],U[[0,1],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[1,3],0],U[[1,3],1],U[[1,3],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[3,2],0],U[[3,2],1],U[[3,2],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[2,0],0],U[[2,0],1],U[[2,0],2],color=(0.5,0.5,0.5))
    # else:
    #     ax.scatter(U[:,0],U[:,1],U[:,2],s=10, c=np.unique(cond))

R = np.repeat(np.array(corr_out_).mean(1),2,-1)

 #%% Visualization purposes
dim = 50
noise = 0.2
this_d = 0.01
pos = this_exp.task.positives[0]
# pos = (0,1,3,4)
# pos = (0,1,2,3)

emb = util.ContinuousEmbedding(dim, 0.0)

z = emb(this_exp.train_data[1])

dz = np.random.randn(len(np.unique(cond)), dim)
dz /= la.norm(dz,2,1, keepdims=True)
z += dz[cond,:]*this_d*np.sqrt(dim)

centroids = np.stack([z[cond==i,:].mean(0) for i in np.unique(cond)])
eps1 = np.random.randn(ndat, dim)*noise

coloring = np.isin(cond[:ndat],pos)
rclf.fit(z[:ndat,:]+eps1, coloring)
w = rclf.coef_.T
w /= la.norm(w)
z_err = ((torch.eye(dim) - torch.tensor(w@w.T).float())@centroids.T).numpy()

z_proj = centroids@w

_, S, V = la.svd(z_err.T-z_err.mean(1)[None,:],full_matrices=False)
# _, _, V = la.svd(z_errz_err-centroids.mean(0)[None,:],full_matrices=False)
U = centroids@V.T

U_err = z_err.T@V.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

coloring = np.isin(np.unique(cond),pos)
# ax.scatter(U[:,0],U[:,1],U[:,2],s=100, c=coloring)
# ax.scatter(U_err[:,0],U_err[:,1],U_err[:,2],s=100, marker='s', c=coloring)
ax.scatter(U[:,0],U[:,1],z_proj,s=100, c=coloring)
ax.scatter(U_err[:,0],U_err[:,1],w.T@z_err,s=100, marker='s', c=coloring)

util.set_axes_equal(ax)
ax.plot([0,0],[0,0],ax.get_zlim(), 'k--')


#%% Plotting
# mask = (R.max(2)==1) # context must be an output variable
# mask = (np.abs(R).sum(2)==0) # context is uncorrelated with either output variable
# mask = (np.abs(R).sum(2)>0) # context is correlated with at least one output variable
# mask = ~np.isnan(R).max(2) # context is uncorrelated with the tested variable

mutinfo_cmap = 'copper'

# plot_this = np.squeeze(ccgp_).mean(-1)
# plot_this = util.group_mean(np.squeeze(ccgp_), mask)
# plot_this = np.array(par_)
# plot_this = np.squeeze(sd_).mean(1)
# plot_this = np.mean(np.array(dist_total_)*(1-np.array(dist_res_)),1)
# plot_this = 1-np.abs(np.mean(np.array(dist_res_),1))
# plot_this = np.mean(np.array(dist_total_),1)
# plot_this = np.mean(pdcorr,1)
# plot_this = np.array(apprx_cost_)
plot_this = np.array(apprx_ps_)

for i in range(len(apprx_ps_[0])):
    if i in range(num_var):
        plt.plot(vals, plot_this[:,i], linewidth=2)
    # else:
        # plt.plot(vals, plot_this[:,i], '--', color=cm.get_cmap(mutinfo_cmap)(np.mean(mut_inf_,0)[i]))

#%% Angles
abstract_conds = util.decimal(this_exp.train_data[1])
dim = 50
noise = 0.2
n_neigh = 10

subsamp = 200

ndat = 2000

# lin_clf = svm.LinearSVC
# lin_clf = linear_model.LogisticRegression
# lin_clf = linear_model.Perceptron
# lin_clf = linear_model.RidgeClassifier

# emb = util.ContinuousEmbedding(dim, 0.0)
# z = emb(this_exp.train_data[1])
basis = torch.tensor(la.qr(np.random.rand(dim, dim))[0]).float()
z = this_exp.train_data[1]@basis[:,:num_var].T

knn = neighbors.NearestNeighbors(n_neighbors=n_neigh)

_, _, V = la.svd(z-z.mean(0)[None,:],full_matrices=False)
# V = V[:3,:]
cond = this_exp.train_conditions

vals = np.linspace(0,0.2,10)

pr_ = []
par_ = []
ccgp_ = []
sd_ = []
ovlp = []
mut_inf_ = []
corr_out_ = []
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
for d in vals:
    
    # z = emb(this_exp.train_data[1])
    z = this_exp.train_data[1]@basis[:,:num_var].T
    z -= z.mean(0)
    
    pr = []
    par = []
    ccgp = []
    sd = []
    angles = []
    mut_inf = []
    corr_out = []
    
    D_fake = assistants.Dichotomies(len(np.unique(this_exp.train_conditions)), this_exp.task.positives, extra=50)
    Q = len(this_exp.task.positives)
    mi = np.array([this_exp.task.information(p) for p in D_fake])
    midx = np.append(range(Q),np.flip(np.argsort(mi[Q:]))+Q)
    # these_dics = args['dichotomies'] + [D_fake.combs[i] for i in midx]
    D = assistants.Dichotomies(len(np.unique(cond)), [D_fake.combs[i] for i in midx], extra=0)
    
    for _ in tqdm(range(5)):
        dz = np.random.randn(len(np.unique(cond)), dim)
        dz /= la.norm(dz,2,1, keepdims=True)
        z += dz[cond,:]*d*np.sqrt(dim)
        
        centroids = np.stack([z[cond==i,:].mean(0) for i in np.unique(cond)])
    
        _, S, _ = la.svd(centroids-centroids.mean(0)[None,:],full_matrices=False)
        pr.append(((S**2).sum()**2)/(S**4).sum())
        
        # # U = centroids@emb.rotator.numpy()@V.T
        # U = (z[:ndat,:]+eps1)@V.T
        # U = centroids@V.T
        # pcs.append(U)
        
        clf = assistants.LinearDecoder(dim, 1, assistants.MeanClassifier)
        # gclf = assistants.LinearDecoder(dim, 1, svm.LinearSVC)
        gclf = assistants.LinearDecoder(dim, 1, linear_model.LogisticRegression)
        # rclf = svm.LinearSVC()
        rclf = linear_model.LogisticRegression(fit_intercept=False) # global decoder
        
        eps1 = np.random.randn(ndat, dim)*noise
        eps2 = np.random.randn(ndat, dim)*noise
        
        Z_thry = z.numpy() + np.random.randn(len(z), dim)*noise
        
        PS = []
        CCGP = []
        out_corr = []
        # Thetas = []
        calcs = []
        for i,pos in enumerate(D):
            # print('Dichotomy %d...'%i)
            # parallelism
            PS.append(D.parallelism(z[:ndat,:], cond[:ndat], clf))
            
            # # independence test
            coloring = np.isin(cond[:ndat],pos)
            # if i in [0,1]:
            #     w = emb.basis[:,i:i+1].numpy()
            # else:
            rclf.fit(z[:ndat,:]+eps1, coloring)
            w = rclf.coef_.T
            
            # CCGP
            cntxt = D.get_uncorrelated(50)
            out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in this_exp.task.positives]))
            CCG_things = D.CCGP(z[:ndat,:]+eps2, cond[:ndat], gclf, cntxt, twosided=True, return_weights=True, fit_intercept=False)
            
            delta_w = (rclf.coef_ - np.squeeze(CCG_things[-1])[:,:-1])
            
            idx = np.where(np.max(delta_w@Z_thry.T,0)<0.2)[0]       
            
            # w /= la.norm(w)
            fish_score = (np.isin(cond[idx],pos)[:,None]-spc.expit(Z_thry[idx,...]@w))*Z_thry[idx,...]
            A = ((logistic_deriv(Z_thry[idx,...]@w)*Z_thry[idx,...]).T@Z_thry[idx,...])/len(idx)

            calcs.append(delta_w@opt_grad.mean(0) - np.diag(delta_w@A@delta_w.T))
            
            CCGP.append(CCG_things[0])
        
        par.append(PS)
        angles.append(calcs)
        ccgp.append(CCGP)
        mut_inf.append(mi[midx])
        corr_out.append(out_corr)
        
        dclf = assistants.LinearDecoder(z.shape[1], D.ntot, svm.LinearSVC)
        dclf.fit(z[:ndat,:]+eps1, np.array([D.coloring(cond[:ndat]) for _ in D]).T, tol=1e-5)
        sd.append(dclf.test(z[:ndat,:]+eps2, np.array([D.coloring(cond[:ndat]) for _ in D]).T))
        
        # cos_w.append(np.diag(dclf.coefs[:2,...].squeeze()@emb.basis[:,:2].numpy()))
    
    pr_.append(np.mean(pr,0))
    par_.append(np.mean(par,0))
    ccgp_.append(np.mean(ccgp,0))
    sd_.append(sd)
    ovlp.append(np.mean(angles,0))
    mut_inf_.append(np.mean(mut_inf,0))
    corr_out_.append(corr_out)
    # if d in vals[[0,-1]]:
    #     ax.scatter(U[:,0],U[:,1],U[:,2],s=100,c=np.unique(cond))
    #     ax.plot(U[[0,1],0],U[[0,1],1],U[[0,1],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[1,3],0],U[[1,3],1],U[[1,3],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[3,2],0],U[[3,2],1],U[[3,2],2],color=(0.5,0.5,0.5))
    #     ax.plot(U[[2,0],0],U[[2,0],1],U[[2,0],2],color=(0.5,0.5,0.5))
    # else:
    #     ax.scatter(U[:,0],U[:,1],U[:,2],s=10, c=np.unique(cond))

R = np.repeat(np.array(corr_out_).mean(1),2,-1)

#%% Plotting
# mask = (R.max(2)==1) # context must be an output variable
# mask = (np.abs(R).sum(2)==0) # context is uncorrelated with either output variable
# mask = (np.abs(R).sum(2)>0) # context is correlated with at least one output variable
mask = ~np.isnan(R).max(2) # context is uncorrelated with the tested variable

mutinfo_cmap = 'copper'

# plot_this = np.squeeze(ccgp_).mean(-1)
plot_this = util.group_mean(np.squeeze(ccgp_), mask)
# plot_this = np.array(par_)
# plot_this = np.squeeze(sd_).mean(1)

for i in range(len(par_[0])):
    if i in range(num_var):
        plt.plot(vals, plot_this[:,i], linewidth=2, color=cm.get_cmap(mutinfo_cmap)(np.mean(mut_inf_,0)[i]))
    else:
        plt.plot(vals, plot_this[:,i], '--', color=cm.get_cmap(mutinfo_cmap)(np.mean(mut_inf_,0)[i]))
