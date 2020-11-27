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

# import umap
from cycler import cycler

# my code
import students
import assistants
import experiments as exp
import util

#%% Set up the task
num_cond = 8
num_var = 2

task = util.RandomDichotomies(num_cond,num_var,0)
# task = util.ParityMagnitude()

# this_exp = exp.mnist_multiclass(task, SAVE_DIR, abstracts=abstract_variables)
this_exp = exp.random_patterns(task, SAVE_DIR, 
                               num_class=num_cond,
                               dim=100,
                               var_means=1.0,
                               var_noise=0.3)

#%% (hetero-associative) Autoencoding 
ndat = 10000

p = 2**num_var
allowed_actions = [0,1,2]
# allowed_actions = [0,1,2,4]
# allowed_actions = [0]
p_action = [0.7,0.15,0.15]
# p_action = [0.7, 0.1, 0.1, 0.1]
# p_action = [1.0]

output_states = this_exp.train_data[0][:ndat,:].data

input_states = this_exp.train_data[0][:ndat,:].data

abstract_conds = util.decimal(this_exp.train_data[1])[:ndat]
cond_set = np.unique(abstract_conds)

# draw the "actions" for each data point
actns = torch.tensor(np.random.choice(allowed_actions, ndat, p=p_action)).int()
actions = torch.stack([(actns&(2**i))/2**i for i in range(num_var)]).float().T

# act_rep = assistants.Indicator(p,p)(util.decimal(actions).int())
act_rep = actions.data

# inputs = np.concatenate([input_states,act_rep], axis=1)
# # inputs = np.concatenate([input_states, this_exp.train_data[1]], axis=1)
inputs = input_states.float()

# # sample the successor states, i.e. input + action
successors = np.mod(this_exp.train_data[1][:ndat,:]+actions, 2)

succ_conds = util.decimal(successors)
succ_counts = np.unique(succ_conds, return_counts=True)[1]

# should the targets be sampled from the training set, or another set? 
# train set would be like an autoencoder training, so maybe that's fine
samps = np.concatenate([np.random.choice(np.where(abstract_conds==c)[0],n) \
                        for c,n in zip(cond_set,succ_counts)])

unscramble = np.argsort(np.argsort(succ_conds))
successor_idx = samps[unscramble]
targets = output_states[successor_idx,:]

# targets = output_state

#%% Classification 
ndat = 10000

input_states = this_exp.train_data[0][:ndat,:].data
output_states = this_exp.train_data[1][:ndat,:].data

abstract_conds = util.decimal(this_exp.train_data[1])[:ndat]

inputs = input_states.float()
targets = output_states

#%%
# manual = True
manual = False
ppp = 1 # 0 is MSE, 1 is cross entropy

two_layers = False
# nonneg = True
nonneg = False
# train_out = True
train_out = False

correct_mse = False # if True, rescales the MSE targets to be more like the log odds

N = 100

nepoch = 2000
lr = 1e-4
bsz = 64

n_trn = int(ndat*0.8)
idx_trn = np.random.choice(ndat, n_trn, replace=False)
idx_tst = np.setdiff1d(range(ndat), idx_trn)

dset = torch.utils.data.TensorDataset(inputs[idx_trn], targets[idx_trn])
dl = torch.utils.data.DataLoader(dset, batch_size=bsz, shuffle=True)

# set up network (2 layers)
ba = 1/np.sqrt(N)
W1 = torch.FloatTensor(N,inputs.shape[1]).uniform_(-ba,ba)
b1 = torch.FloatTensor(N,1).uniform_(-ba,ba)
W1.requires_grad_(True)
b1.requires_grad_(True)

if two_layers:
    ba = 1/np.sqrt(N)
    W2 = torch.FloatTensor(N,N).uniform_(-ba,ba)
    b2 = torch.FloatTensor(N,1).uniform_(-ba,ba)
    W2.requires_grad_(True)
    b2.requires_grad_(True)

ba = 1/np.sqrt(targets.shape[1])

if nonneg:
    W = torch.FloatTensor(targets.shape[1],N).uniform_(0,2*ba)
    b = torch.FloatTensor(targets.shape[1],1).uniform_(0,2*ba)
else:
    W = torch.FloatTensor(targets.shape[1],N).uniform_(-ba,ba)
    b = torch.FloatTensor(targets.shape[1],1).uniform_(-ba,ba)

if two_layers:
    optimizer = optim.Adam([W1, b1, W2, b2], lr=lr)
else:
    if train_out:
        optimizer = optim.Adam([W1, b1, W, b], lr=lr)
    else:
        optimizer = optim.Adam([W1, b1], lr=lr)

train_loss = [] 
test_perf = []
PS = []
CCGP = []
lindim = []
gradz_sim = []
gradlin_sim = []
for epoch in tqdm(range(nepoch)):

    # loss = net.grad_step(dl, optimizer)
    
    running_loss = 0
    
    # idx = np.random.choice(n_trn, np.min([5000,ndat]), replace=False)
    if two_layers:
        z1 = nn.ReLU()(torch.matmul(W1,input_states[idx_tst,:].T) + b1)
        z = nn.ReLU()(torch.matmul(W2,z1) + b2)
    else:
        z = nn.ReLU()(torch.matmul(W1,input_states[idx_tst,:].T) + b1)
    pred = torch.matmul(W,z) + b
    
    if ppp == 0:
        perf = np.sum((pred.T-targets[idx_tst,:]).detach().numpy()**2,1).mean(0)
    else:
        perf = ((pred.T>0) == targets[idx_tst,:]).detach().numpy().mean(0)
    test_perf.append(perf)
    
    # this is just the way I compute the abstraction metrics, sorry
    clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
    gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
    D = assistants.Dichotomies(len(np.unique(this_exp.test_conditions)),
                                this_exp.task.positives, extra=0)
    
    ps = []
    ccgp = []
    for _ in D:
        ps.append(D.parallelism(z.T.detach().numpy(), this_exp.train_conditions[:ndat][idx_tst], clf))
        ccgp.append(D.CCGP(z.T.detach().numpy(), this_exp.train_conditions[:ndat][idx_tst], gclf, max_iter=1000))
    PS.append(ps)
    CCGP.append(ccgp)
    
    _, S, _ = la.svd(z.detach()-z.mean(1).detach()[:,None], full_matrices=False)
    eigs = S**2
    lindim.append((np.sum(eigs)**2)/np.sum(eigs**2))
    
    # Gradient similarity
    # if np.mod(epoch,10)==0:
    if epoch in [0,nepoch-1]:
        errb = (targets[idx_tst,:].T - nn.Sigmoid()(pred)) # bernoulli
        errg = (targets[idx_tst,:].T - pred) # gaussian
        
        err = ppp*errb + (1-ppp)*errg # convex sum, in case you want that
        
        d2 = (W.T@err)*(z>0) # gradient of the currents
        if two_layers:
            W2.grad = -(d2@z1.T)/len(idx_tst)
            b2.grad = -d2.mean(1, keepdim=True)
            
            d1 = (W2@d2)*(z1>0)
            W1.grad = -(d1@input_states[idx_tst,:])/len(idx_tst)
            b1.gad = -d1.mean(1, keepdim=True)
        else:
            W1.grad = -(d2@input_states[idx_tst,:])/len(idx_tst)
            b1.gad = -d2.mean(1, keepdim=True)
        
        conds = abstract_conds[idx_tst]
        cond_grad = np.array([d2[:,conds==i].mean(1).detach().numpy() for i in np.unique(conds)])
        gradz_sim.append(util.cosine_sim(cond_grad-cond_grad.mean(0),cond_grad-cond_grad.mean(0)))
        
        cond_grad = np.array([(W.T@err)[:,conds==i].mean(1).detach().numpy() for i in np.unique(conds)])
        gradlin_sim.append(util.cosine_sim(cond_grad-cond_grad.mean(0),cond_grad-cond_grad.mean(0)))
        # cond_grad = np.array([((d2[:,conds==i]@z[:,conds==i].T)/np.sum(conds==i)).mean(1).detach().numpy() \
        #                       for i in np.unique(conds)])
        # gradw_sim.append(util.cosine_sim(cond_grad,cond_grad))
    
    # do learning
    for j, btch in enumerate(dl):
        optimizer.zero_grad()
        
        inps, outs = btch
        if two_layers:
            z1 = nn.ReLU()(torch.matmul(W1,inps.T) + b1)
            z = nn.ReLU()(torch.matmul(W2,z1) + b2)
        else:
            z = nn.ReLU()(torch.matmul(W1,inps.T) + b1)
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
            if two_layers:
                W2.grad = -(d2@z1.T)/inps.shape[0]
                b2.grad = -d2.mean(1, keepdim=True)
                
                d1 = (W2@d2)*(z1>0)
                W1.grad = -(d1@inps)/inps.shape[0]
                b1.gad = -d1.mean(1, keepdim=True)
            else:
                W1.grad = -(d2@inps)/inps.shape[0]
                b1.gad = -d2.mean(1, keepdim=True)
            
            # W1 += lr*dw
            # b1 += lr*db
        else:
            loss.backward()
        
        optimizer.step()
        
        W = W*(W>0)
        b = b*(b>0)
        
        running_loss += loss.item()
        
    # train_loss.append(loss)
    # print('epoch %d: %.3f'%(epoch,running_loss/(j+1)))
    train_loss.append(running_loss/(j+1))
    # print(running_loss/(i+1))

#%%
if two_layers:
    z1 = nn.ReLU()(torch.matmul(W1,input_states.T) + b1).detach().numpy().T
    z = nn.ReLU()(torch.matmul(W2,z1) + b2)
else:
    z = nn.ReLU()(torch.matmul(W1,input_states.T) + b1).detach().numpy().T
pred = torch.matmul(W,torch.tensor(z).T) + b

# z = net.enc.network[:-2](torch.tensor(inputs)).detach().numpy()
N = z.shape[1]
max_dichs = 50 # the maximum number of untrained dichotomies to test

all_PS = []
all_CCGP = []
all_CCGP_ = []
CCGP_out_corr = []
mut_inf = []
all_SD = []
indep = []

indep.append(task.subspace_information())
# z = this_exp.train_data[0].detach().numpy()
# z = linreg.predict(this_exp.train_data[0])@W1.T
n_compute = np.min([5000, z.shape[0]])

idx = np.random.choice(z.shape[0], n_compute, replace=False)
# idx_tst = idx[::4] # save 1/4 for test set
# idx_trn = np.setdiff1d(idx, idx_tst)

cond = this_exp.train_conditions[:ndat][idx]

# cond = util.decimal(this_exp.train_data[1][idx,...])
num_cond = len(np.unique(cond))

# xor = np.where(~(np.isin(range(num_cond), args['dichotomies'][0])^np.isin(range(num_cond), args['dichotomies'][1])))[0]
## Loop over dichotomies
# D = assistants.Dichotomies(num_cond, args['dichotomies']+[xor], extra=50)

# choose dichotomies to have a particular order
Q = num_var
D_fake = assistants.Dichotomies(num_cond, task.positives, extra=7000)
mi = np.array([task.information(p) for p in D_fake])
midx = np.append(range(Q),np.flip(np.argsort(mi[Q:]))+Q)
# these_dics = args['dichotomies'] + [D_fake.combs[i] for i in midx]
D = assistants.Dichotomies(num_cond, [D_fake.combs[i] for i in midx], extra=0)

clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
dclf = assistants.LinearDecoder(N, D.ntot, svm.LinearSVC)
# clf = LinearDecoder(this_exp.dim_input, 1, MeanClassifier)
# gclf = LinearDecoder(this_exp.dim_input, 1, svm.LinearSVC)
# dclf = LinearDecoder(this_exp.dim_input, D.ntot, svm.LinearSVC)

# K = int(num_cond/2) - 1 # use all but one pairing
K = int(num_cond/4) # use half the pairings

PS = np.zeros(D.ntot)
CCGP = [] #np.zeros((D.ntot, 100))
out_corr = []
d = np.zeros((n_compute, D.ntot))
pos_conds = []
for i, pos in tqdm(enumerate(D)):
    pos_conds.append(pos)
    # print('Dichotomy %d...'%i)
    # parallelism
    PS[i] = D.parallelism(z[idx,:], cond, clf)
    
    # CCGP
    cntxt = D.get_uncorrelated(100)
    out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in this_exp.task.positives]))
    
    CCGP.append(D.CCGP(z[idx,:], cond, gclf, cntxt, twosided=True))
    
    # shattering
    d[:,i] = D.coloring(cond)
    
# dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
dclf.fit(z[idx,:], d, tol=1e-5)

if two_layers:
    z1 = nn.ReLU()(torch.matmul(W1,input_states.T) + b1)
    z = nn.ReLU()(torch.matmul(W2,z1) + b2).detach().numpy().T
else:
    z = nn.ReLU()(torch.matmul(W1,input_states.T) + b1).detach().numpy().T
   
# z = this_exp.test_data[0].detach().numpy()
# z = linreg.predict(this_exp.test_data[0])@W1.T
idx = np.random.choice(ndat, n_compute, replace=False)


d_tst = np.array([D.coloring(this_exp.train_conditions[:ndat][idx]) for _ in D]).T
SD = dclf.test(z[idx,:], d_tst).squeeze()


all_PS.append(PS)
all_CCGP.append(CCGP)
CCGP_out_corr.append(out_corr)
all_SD.append(SD)
mut_inf.append(mi[midx])

R = np.repeat(np.array(CCGP_out_corr),2,-1)
basis_dependence = np.array(indep).max(1)
out_MI = np.array(mut_inf)

# %%
# mask = (R.max(2)==1) # context must be an output variable   
# mask = (np.abs(R).sum(2)==0) # context is uncorrelated with either output variable
# mask = (np.abs(R).sum(2)>0) # context is correlated with at least one output variable
mask = ~np.isnan(R).max(2) # context is uncorrelated with the tested variable

dep_mask = basis_dependence==0
# dep_mask = basis_dependence>0

# mask *= dep_mask[:,None,None]

# color_by_info = False
color_by_info = True

mutinfo_cmap = 'copper'
# var_cmap = 'Set1'
var_cmap = 'tab10'

almost_all_CCGP = util.group_mean(np.array(all_CCGP).squeeze(), mask)
# include = (mask.sum(-1)>0)
# np.sum(np.array(all_CCGP)*mask[...,None],2).squeeze()/mask.sum(-1)

# PS = np.array(all_PS).mean(0)
PS = util.group_mean(np.array(all_PS), mask.sum(-1)>0, axis=0)
CCGP = util.group_mean(almost_all_CCGP, mask.sum(-1)>0, axis=0)
SD = util.group_mean(np.array(all_SD), mask.sum(-1)>0, axis=0)
# SD = np.array(all_SD).mean(0)

ndic = len(PS)

# PS_err = np.nanstd(np.array(PS), axis=0)#/np.sqrt(len(all_PS))
# CCGP_err = np.nanstd(CCGP, axis=0)#/np.sqrt(len(all_CCGP))
# SD_err = np.nanstd(np.array(SD), axis=0)#/np.sqrt(len(all_SD))

# be very picky about offsets
offset = np.random.choice(np.linspace(-0.15,0.15,10), 3*ndic)
special_offsets = np.linspace(-0.1,0.1,num_var)
for i in range(num_var):
    offset[i+np.arange(3)*ndic] = special_offsets[i]

xfoo = np.repeat([0,1,2],ndic).astype(int) + offset # np.random.randn(ndic*3)*0.1
yfoo = np.concatenate((PS, CCGP, SD))
# yfoo_err = np.concatenate((PS_err, CCGP_err, SD_err))
cfoo = np.tile(np.flip(np.sort(out_MI).mean(0)),3)

# plt.scatter(xfoo, yfoo, s=12, c=(0.5,0.5,0.5))
# plt.errorbar(xfoo, yfoo, yerr=yfoo_err, linestyle='None', c=(0.5,0.5,0.5), zorder=0)
if color_by_info:
    scat = plt.scatter(xfoo, yfoo, s=31, c=cfoo, zorder=10, cmap=mutinfo_cmap)
else:
    scat = plt.scatter(xfoo, yfoo, s=31, c=(0.5,0.5,0.5), zorder=10)
# plt.errorbar(xfoo, yfoo, yerr=yfoo_err, linestyle='None', linecolor=cfoo)
plt.xticks([0,1,2], labels=['PS', 'CCGP', 'Shattering'])
plt.ylabel('PS or Cross-validated performance')
plt.colorbar(scat, label='Mutual information with output')

# highlight special dichotomies
# par = plt.scatter(xfoo[[20,20+ndic,20+2*ndic]], yfoo[[20,20+ndic,20+2*ndic]], 
#                   marker='o', edgecolors='r', s=60, facecolors='none', linewidths=3)
# mag = plt.scatter(xfoo[[0, ndic, 2*ndic]], yfoo[[0,ndic,2*ndic]], 
#                   marker='o', edgecolors='g', s=60, facecolors='none', linewidths=3)
# other = plt.scatter(xfoo[[9,9+ndic,9+2*ndic]], yfoo[[9,9+ndic,9+2*ndic]], 
#                     marker='o', edgecolors='b', s=60, facecolors='none', linewidths=3)
anns = []
for i,d in enumerate(this_exp.task.positives):
    # n = np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(this_exp.num_class),p))==list(d))\
    #               for p in pos_conds])[0][0]
    anns.append(plt.scatter(xfoo[[i,i+ndic,i+2*ndic]], yfoo[[i,i+ndic,i+2*ndic]], 
               s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

# baba = np.arange(len(xfoo)//ndic)*ndic
# par = plt.scatter(xfoo[baba], yfoo[baba], 
#                   marker='o', edgecolors='r', s=60, facecolors='none', linewidths=3)
# mag = plt.scatter(xfoo[baba+1], yfoo[baba+1], 
#                   marker='o', edgecolors='g', s=60, facecolors='none', linewidths=3)
# xor = plt.scatter(xfoo[baba+2], yfoo[baba+2], 
#                   marker='o', edgecolors='b', s=60, facecolors='none', linewidths=3)

# plt.legend([par,mag, xor], ['Var1', 'Var2', 'XOR'])
plt.legend(anns, ['var %d'%(i+1) for i in range(len(anns))])

#%%

n_mds = 3
n_compute = 500

fake_task = util.RandomDichotomies(num_cond,num_var,0)
fake_task.positives = task.positives

idx = np.random.choice(inputs.shape[0], n_compute, replace=False)

if two_layers:
    z1 = nn.ReLU()(torch.matmul(W1,input_states[idx,:].T) + b1).detach().numpy().T
    z = nn.ReLU()(torch.matmul(W2,z1) + b2)
else:
    z = nn.ReLU()(torch.matmul(W1,input_states[idx,:].T) + b1).detach().numpy().T
   
# ans = this_exp.train_data[1][idx,...]
ans = fake_task(this_exp.train_conditions[:ndat])[idx]

cond = util.decimal(ans)
# cond = this_exp.train_conditions[idx]

# colorby = cond
colorby = this_exp.train_conditions[idx]

mds = manifold.MDS(n_components=n_mds)

emb = mds.fit_transform(z)

if n_mds == 2:
    scat = plt.scatter(emb[:,0],emb[:,1], c=colorby)
    plt.xlabel('MDS1')
    plt.ylabel('MDS2')
elif n_mds == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.margins(0)
    def init():
        U = np.stack([emb[cond==i,:].mean(0) for i in np.unique(cond)])
        
        qq = len(np.unique(cond))
        for ix in combinations(range(qq),2):
            ax.plot(U[ix,0],U[ix,1],U[ix,2],color=(0.5,0.5,0.5))
        # ax.plot(U[[1,3],0],U[[1,3],1],U[[1,3],2],color=(0.5,0.5,0.5))
        # ax.plot(U[[3,2],0],U[[3,2],1],U[[3,2],2],color=(0.5,0.5,0.5))
        # ax.plot(U[[2,0],0],U[[2,0],1],U[[2,0],2],color=(0.5,0.5,0.5))
        # ax.plot(U[[0,3],0],U[[0,3],1],U[[0,3],2],color=(0.5,0.5,0.5))
        # ax.plot(U[[1,2],0],U[[1,2],1],U[[1,2],2],color=(0.5,0.5,0.5))
        
        ax.scatter(U[:,0],U[:,1],U[:,2],s=50, marker='s',c=np.unique(cond))
        scat = ax.scatter(emb[:,0],emb[:,1], emb[:,2], c=colorby)
        
        util.set_axes_equal(ax)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # plt.xticks([])
        # plt.yticks([])
        # plt.zticks([])
        # plt.legend(np.unique(cond), np.unique(cond))
        cb = plt.colorbar(scat,
                          ticks=np.unique(colorby),
                          drawedges=True,
                          values=np.unique(colorby))
        cb.set_ticklabels(np.unique(colorby)+1)
        cb.set_alpha(1)
        cb.draw_all()
        
        return fig,

# def init():
    # ax.view_init(30,0)
    # plt.draw()
    # return ax,

def update(frame):
    ax.view_init(30,frame)
    # plt.draw()
    return fig,

ani = anime.FuncAnimation(fig, update, frames=np.linspace(0, 360, 100),
                          init_func=init, interval=10, blit=True)
# plt.show()
ani.save(SAVE_DIR+'/vidya/tempmovie.mp4', writer=anime.writers['ffmpeg'](fps=30))


