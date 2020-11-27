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
from itertools import permutations
from sklearn import svm, manifold, linear_model, neighbors
from sklearn import gaussian_process as gp
from tqdm import tqdm

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import experiments as exp

#%%
def weight_func(x, x0, sigma=1):
    alph_ = np.exp(-(x-x0)**2 / (2*sigma)**2)
    return alph_/np.mean(alph_)

#%% Fake represenatation
dim = 100
ndat = 5000
rot = 0.0

# fake_labels = (np.random.rand(100, 2)>0.5).astype(int)
fake_labels = np.stack([np.random.rand(ndat)>0.5, 2*np.random.rand(ndat)-1]).T

C = np.random.rand(dim, dim)
basis = la.qr(C)[0]

emb = util.ContinuousEmbedding(dim, rot)

#%% Generate data
noise = 0.001

# s = fake_labels[:,1]*0.1
# rot = 1.5

# r = 1-0.5*rot
# iz = (r+fake_labels[:,0])*np.exp(s*1j*np.pi/r)
# z_ = np.stack([np.real(iz),np.imag(iz)])
# z_ -= z_.mean(1,keepdims=True)
# # z_ *= r

# z = (basis[:,:2]@z_).T + np.random.randn(ndat,dim)*0.05

# # z[pos,:] = fake_labels[pos,:]@basis.T 
z = emb(torch.tensor(fake_labels).float()).numpy() + np.random.randn(ndat,dim)*noise

#%% Visualize
x0 = -1 # training context 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

w = weight_func(fake_labels[:,1], x0, sigma=0.5)
# w = w_tot

ax.scatter(z[:,0],z[:,1],z[:,2],c=w, s=1)

util.set_axes_equal(ax)

#%% Compute
# x0 = -1 # training context 
# x_tst = np.linspace(-0.9,0.9,25)
bw = 0.05
n_ps = 300 # subsample size for parallelism
b_ps = 1 # number of 'batches' for parallelism
eps_ps = 0.5 # epsilon for parallelism
n_neigh = 10

dist_weight = 0.0

x = fake_labels[:,1]
y = fake_labels[:,0]

x_tst = np.quantile(x,np.linspace(0,1,int(ndat/n_ps)))

trnidx = np.random.choice(ndat,int(0.8*ndat)) # training subset
tstidx = np.isin(range(ndat), trnidx) # testing subset

rclf = linear_model.LinearRegression() 
knn = neighbors.NearestNeighbors(n_neighbors=n_neigh)

CCGP = []
perf = []
PS = []
ps_perf = []
proj_var = []
for i,x0 in tqdm(enumerate(x_tst[1:-1])): 
    w = weight_func(x, x0, sigma=bw)
   
    ## CCGP
    mdl = linear_model.LogisticRegression(penalty='none')
    mdl.fit(z[trnidx,:], y[trnidx], w[trnidx])
    perf.append(mdl.score(z[tstidx,:], y[tstidx], w[tstidx]))
    
    ccgp = []
    # perf = []
    for x_ in x_tst:
        w_tst = weight_func(x, x_, sigma=bw)
        ccgp.append((mdl.score(z[tstidx,:],y[tstidx], w_tst[tstidx])))
    
    CCGP.append(ccgp)

    ## Parallelism of the continuous variable   
    # w_pos = weight_func(x, x0-eps_ps/2, sigma=bw)
    # w_neg = weight_func(x, x0+eps_ps/2, sigma=bw)
    # w_pos = (x>x0-eps_ps/2)&(x<x0)
    # w_neg = (x>x0)&(x<x0+eps_ps/2)
    w_pos = (x>x_tst[i])&(x<x_tst[i+1])
    w_neg = (x>x_tst[i+1])&(x<x_tst[i+2])
    w_tot = (w_pos/w_pos.mean()+w_neg/w_neg.mean())/2
    
    rclf.fit(z[trnidx,:], x[trnidx], w_tot[trnidx])
    ps_perf.append(rclf.score(z[tstidx,:], x[tstidx], w_tot[tstidx]))
    d_glob = rclf.coef_.T
    d_glob /= la.norm(d_glob)
    
    # d_glob = emb.basis[:,1].numpy()
    
    psps = []
    pvpv = []
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
        dz = la.norm(C, axis=0)
        C /= (la.norm(C,axis=0)+1e-4)
        # dx = 
        cost = np.sum(d_glob.squeeze()[:,None,None]*C,0) + dist_weight*(dz)
        # cost = -la.norm(C, axis=0)
        
        # find the best pairing by an approximate cost function
        row, col = opt.linear_sum_assignment(cost, maximize=True)
        g_pos = idx_pos[row]
        g_neg = idx_neg[col]
        
        # compute parallelism of that pairing
        # vecs = (z[g_neg,:]-z[g_pos,:])
        vecs = z_neg[row,:] - z_pos[col,:]
        # # pvpv.append((vecs@d_glob).std())
        # mnvec = (z[g_neg,:]-z[g_pos,:]).mean(0)
        # pvpv.append((mnvec@d_glob)/la.norm(mnvec))
        vecs /= (la.norm(vecs,axis=1,keepdims=True)+1e-4)
        csin = np.einsum('ik...,jk...->ij...', vecs, vecs)
        psps.append(2*np.triu(csin.T,1).sum()/(len(idx_pos)*(len(idx_pos)-1)))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# z_err = (np.eye(dim)-(d_glob[:,None]*d_glob[None,:]))@z.T
# U, S, _ = la.svd(z_err-z_err.mean(1)[:,None], full_matrices=False)
# pcs = z@U[:,:2]
# # pcs = z@emb.basis[:,:3].numpy()

# pltidx = np.random.choice(np.setdiff1d(trnidx, np.append(g_pos,g_neg)),1000,replace=False)
# ax.scatter(pcs[pltidx,0],pcs[pltidx,1],z[pltidx,:]@d_glob,color=(0.5,0.5,0.5), s=1)
# ax.scatter(pcs[g_pos,0],pcs[g_pos,1],z[g_pos,:]@d_glob, color='b', s=10, zorder=10)
# # ax.scatter(pcs[pltidx,0],pcs[pltidx,1],pcs[pltidx,2],color=(0.5,0.5,0.5), s=1)
# # ax.scatter(pcs[g_pos,0],pcs[g_pos,1],pcs[g_pos,2], color='b', s=10, zorder=10)

# BBB = d_glob@emb.basis[:,:3].numpy()
# # BBB = emb.basis[:,1]@U[:,:2]
# ax.quiver(0,0,0,BBB[0],BBB[1],emb.basis[:,1]@d_glob)

# BBB = (z[g_neg,:]-z[g_pos,:]).mean(0)@U[:,:2]
# ax.quiver(0,0,0,BBB[0],BBB[1],(z[g_neg,:]-z[g_pos,:]).mean(0)@d_glob, length=2)

# v_pcs = (z[g_neg,:]-z[g_pos,:])@U[:,:2]
# ax.quiver(pcs[g_pos,0],pcs[g_pos,1],z[g_pos,:]@d_glob, 
#             v_pcs[:,0], v_pcs[:,1], vecs@d_glob, 
#             normalize=False, length=1, linewidth=1, color='r')
# v_pcs = vecs@emb.basis[:,:3].numpy()
# ax.quiver(pcs[g_pos,0],pcs[g_pos,1],pcs[g_pos,2], 
#             v_pcs[:,0], v_pcs[:,1], v_pcs[:,2], 
#             normalize=False, length=1, linewidth=1, color='r')

# util.set_axes_equal(ax)        

    PS.append(psps)
    proj_var.append(pvpv)


## Plot
# CCGP = np.array(CCGP)
# plt.imshow(CCGP, extent=[x_tst.min(),x_tst.max(),x_tst.min(),x_tst.max()])
# plt.clim([0,1])
# plt.colorbar()

# # plt.xtick_labels(x_tst)
# # plt.ytick_labels(x_tst)
# plt.xlabel('Test context')
# plt.ylabel('Train context')

# # plt.plot(x_tst, ccgp)
# # plt.plot(x_tst, perf)

# # plt.ylim([0,1.1])
# # plt.plot([x0,x0],plt.ylim(),'k--')

# # plt.legend(['CCGP','Local decoder performance','Training context'])
# # plt.xlabel('Test context')
# # plt.ylabel('Accuracy')
# plt.title('CCGP of %.1f-rotated representation (bw=%.2f)'%(rot,bw))

#%% Fake represenatation
dim = 50
num_var = 2
num_nonlin = 3
ndat = 2000
sigma = 1

# fake_labels = (np.random.rand(100, 2)>0.5).astype(int)
# fake_labels = np.stack([np.random.rand(ndat)>0.5, 2*np.random.rand(ndat)-1]).T
fake_labels =  2*np.random.rand(ndat,num_var)-1

basis = la.qr(np.random.rand(dim, dim))[0]

coords = gp.GaussianProcessRegressor(gp.kernels.RBF(sigma))

# ys = np.stack([coords.sample_y(fake_labels[:,i,None], n_samples=1) for i in range(num_var)]).squeeze()
ys = coords.sample_y(fake_labels, n_samples=num_nonlin)
ys -= ys.mean(0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# w = weight_func(fake_labels[:,0], 0, sigma=0.5)
# w = w_tot

# ax.scatter(fake_labels[:,0],fake_labels[:,1],ys[:,0],c=w, s=1)
ax.scatter(ys[:,0],ys[:,1],ys[:,2], s=2)

util.set_axes_equal(ax)

#%% Generate data

z = ys@basis[:,:num_nonlin].T

#%% Compute
# x0 = -1 # training context 
x_tst = np.linspace(-1,1,25)

bw = 0.01
n_ps = 200 # subsample size for parallelism
b_ps = 10 # number of 'batches' for parallelism
eps_ps = 0.3 # epsilon for parallelism
n_neigh = 50

x = fake_labels[:,1]
y = fake_labels[:,0]

trnidx = np.random.choice(ndat,int(0.8*ndat)) # training subset
tstidx = np.isin(range(ndat), trnidx) # testing subset

# mdl = linear_model.LogisticRegression()
mdl_x = linear_model.LinearRegression() # CCGP classifier
mdl_y = linear_model.LinearRegression() # CCGP classifier
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
        w_pos = weight_func(fake_labels[:,i], x0-eps_ps/2, sigma=bw)
        w_neg = weight_func(fake_labels[:,i], x0+eps_ps/2, sigma=bw)
        # w_pos = (x>x0-eps_ps/2)&(x<x0)
        # w_neg = (x>x0)&(x<x0+eps_ps/2)
        w_tot = (w_pos/w_pos.mean()+w_neg/w_neg.mean())/2
        
        rclf.fit(z[trnidx,:], fake_labels[trnidx,i], w_tot[trnidx])
        ps_perf[i].append(rclf.score(z[tstidx,:], fake_labels[tstidx,i], w_tot[tstidx]))
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
    
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# z_err = (np.eye(dim)-(d_glob[:,None]*d_glob[None,:]))@z.T
# U, S, _ = la.svd(z_err-z_err.mean(1)[:,None], full_matrices=False)
# pcs = z@U[:,:2]
# pcs_pos = z_pos@U[:,:2]
# pcs_neg = z_neg@U[:,:2]
# # pcs = z@emb.basis[:,:3].numpy()

# pltidx = np.random.choice(trnidx ,500,replace=False)
# ax.scatter(pcs[pltidx,0],pcs[pltidx,1],z[pltidx,:]@d_glob,color=(0.5,0.5,0.5), s=1)
# ax.scatter(pcs_pos[:,0],pcs_pos[:,1],z_pos@d_glob, color='b', s=10, zorder=10)
# ax.scatter(pcs_neg[:,0],pcs_neg[:,1],z_neg@d_glob, color='r', s=10, zorder=10)
# # ax.scatter(pcs[pltidx,0],pcs[pltidx,1],pcs[pltidx,2],color=(0.5,0.5,0.5), s=1)
# # ax.scatter(pcs[g_pos,0],pcs[g_pos,1],pcs[g_pos,2], color='b', s=10, zorder=10)

# # BBB = d_glob@emb.basis[:,:3].numpy()
# BBB = basis[:,1]@U[:,:2]
# ax.quiver(0,0,0,BBB[0],BBB[1],basis[:,1]@d_glob, length=0.1)

# BBB = (z_neg[row,:] - z_pos[col,:]).mean(0)@U[:,:2]
# ax.quiver(0,0,0,BBB[0],BBB[1],(z_neg[row,:] - z_pos[col,:]).mean(0)@d_glob, length=2)

# v_pcs = (z_neg[row,:] - z_pos[col,:])@U[:,:2]
# ax.quiver(pcs_pos[:,0],pcs_pos[:,1],z_pos@d_glob, 
#             v_pcs[:,0], v_pcs[:,1], (z_neg[row,:] - z_pos[col,:])@d_glob, 
#             normalize=False, length=0.5, linewidth=0.5, color='r', alpha=0.5)
# # v_pcs = vecs@emb.basis[:,:3].numpy()
# # ax.quiver(pcs[g_pos,0],pcs[g_pos,1],pcs[g_pos,2], 
# #             v_pcs[:,0], v_pcs[:,1], v_pcs[:,2], 
# #             normalize=False, length=1, linewidth=1, color='r')

# util.set_axes_equal(ax)        

CCGP = np.array(CCGP)

plt.plot(x_tst,np.mean(PS,-1))
plt.ylabel('Parallelism')
plt.legend(['Coord1','Coord2'])

#%%
which_var = 1

plt.imshow(CCGP[:,:,which_var], extent=[x_tst.min(),x_tst.max(),x_tst.max(),x_tst.min()])
plt.clim([0,1])
plt.colorbar()

# plt.xtick_labels(x_tst)
# plt.ytick_labels(x_tst)
plt.xlabel('Test context')
plt.ylabel('Train context')

# plt.plot(x_tst, ccgp)
# plt.plot(x_tst, perf)

# plt.ylim([0,1.1])
# plt.plot([x0,x0],plt.ylim(),'k--')

# plt.legend(['CCGP','Local decoder performance','Training context'])
# plt.xlabel('Test context')
# plt.ylabel('Accuracy')
# plt.title('CCGP of Coord %d (bw=%.2f)'%(which_var+1,bw))

#%%
which_var = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

this_ps = np.mean(PS,-1)[:,which_var]
cols = this_ps[np.abs(fake_labels[:,which_var,None] - x_tst[None,:]).argmin(1)]

# ax.scatter(fake_labels[:,0],fake_labels[:,1],ys[:,0],c=w, s=1)
ax.scatter(ys[:,0],ys[:,1],ys[:,2],c=cols, s=1)

util.set_axes_equal(ax)

#%% Representation from a simple linear task

task = util.RandomDichotomies(8,1,0)
# task = util.ParityMagnitude()

# this_exp = exp.mnist_multiclass(task, SAVE_DIR, abstracts=abstract_variables)
this_exp = exp.random_patterns(task, SAVE_DIR, 
                               num_class=8,
                               dim=100,
                               var_means=1)

coding_dir = np.random.randn(100)

cont_comp = np.random.randn(this_exp.train_data[0].shape[0],1)*10
input_states = (this_exp.train_data[0].data + cont_comp*coding_dir[None,:]).float()

output_states = torch.cat((this_exp.train_data[1].data, torch.tensor(cont_comp).float()),1)

#%% Train network
N = 100

net = students.Feedforward([input_states.shape[1], N, output_states.shape[1]],['ReLU', None])


optimizer = optim.Adam(net.parameters(), lr=1e-4)
dset = torch.utils.data.TensorDataset(input_states, output_states)
dl = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)

train_loss = []
# train_perf = []
train_PS = []
test_loss = []
min_dist = []
for epoch in range(2000):

    # loss = net.grad_step(dl, optimizer)
    
    running_loss = 0
    
    for i, btch in enumerate(dl):
        optimizer.zero_grad()
        
        inps, outs = btch
        # pred = net(inps[...,:-4],inps[...,-4:])
        pred = net(inps)
        
        # loss = nn.MSELoss()(pred, outs)
        # loss = nn.BCEWithLogitsLoss()(pred[:,0],outs[:,0]) + nn.MSELoss()(pred[:,1], outs[:,1])
        loss = nn.MSELoss()(pred[:,0],outs[:,0]) + nn.MSELoss()(pred[:,1], outs[:,1])

        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        
    # train_loss.append(loss)
    print('epoch %d: %.3f'%(epoch,running_loss/(i+1)))
    train_loss.append(running_loss/(i+1))
    # print(running_loss/(i+1))

#%% 
x_tst = np.linspace(-10,10,50)

bw = 0.1
x = output_states[:5000,1].detach().numpy() # continuous component
y = output_states[:5000,0].detach().numpy() # binary component
z = net.network[:-1](input_states[:5000,:].float()).detach().numpy()

trnidx = np.random.choice(len(x),int(0.8*len(x))) # training subset
tstidx = np.isin(range(len(x)), trnidx) # testing subset

CCGP = []
perf = []
for x0  in x_tst:
    w = weight_func(x, x0, sigma=bw)
    
    mdl = linear_model.LogisticRegression(penalty='none')
    mdl.fit(z[trnidx,:], y[trnidx], w[trnidx])
    perf.append(mdl.score(z[tstidx,:], y[tstidx], w[tstidx]))
    
    ccgp = []
    # perf = []
    for x_ in x_tst:
        w_tst = weight_func(x, x_, sigma=bw)
        ccgp.append((mdl.score(z[tstidx,:],y[tstidx], w_tst[tstidx])))
    
    CCGP.append(ccgp)

CCGP = np.array(CCGP)
plt.imshow(CCGP, extent=[x_tst.min(),x_tst.max(),x_tst.max(),x_tst.min()])
plt.clim([0.85,1])
plt.colorbar()

# plt.xtick_labels(x_tst)
# plt.ytick_labels(x_tst)
plt.xlabel('Test context')
plt.ylabel('Train context')

# plt.plot(x_tst, ccgp)
# plt.plot(x_tst, perf)

# plt.ylim([0,1.1])
# plt.plot([x0,x0],plt.ylim(),'k--')

# plt.legend(['CCGP','Local decoder performance','Training context'])
# plt.xlabel('Test context')
# plt.ylabel('Accuracy')
plt.title('CCGP of learned representation (bw=%.2f)'%bw)

#%% PCA
U, S, _ = la.svd(z-z.mean(1)[:,None], full_matrices=False)
pcs = z@U[:3,:].T

plt.figure()
plt.loglog(np.arange(1,N),(S[:-1]**2)/np.sum(S[:-1]**2))
plt.xlabel('PC')
plt.ylabel('variance explained') 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pos = y==0
ax.scatter(pcs[pos,0],pcs[pos,1],pcs[pos,2], marker='p', c=x[pos], alpha=0.5)
ax.scatter(pcs[~pos,0],pcs[~pos,1],pcs[~pos,2], marker='s', c=x[~pos], alpha=0.5)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')

#%% MDS
n_mds = 2
n_compute = 500

idx = np.random.choice(z.shape[0], n_compute, replace=False)

mds = manifold.MDS(n_components=2)

emb = mds.fit_transform(z[idx,:])

pos = y[idx]==0
scat1 = plt.scatter(emb[pos,0],emb[pos,1], marker='d', c=x[idx][pos])
scat2 = plt.scatter(emb[~pos,0],emb[~pos,1], marker='s', c=x[idx][~pos])
plt.xlabel('MDS1')
plt.ylabel('MDS2')
cb = plt.colorbar(scat1, label='Continuous variable')
cb.set_alpha(1)
cb.draw_all()

plt.legend([scat1, scat2], ['+1','-1'], title='Binary variable')

