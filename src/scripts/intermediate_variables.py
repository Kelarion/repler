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
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as anime
from itertools import permutations, combinations
from sklearn import svm, manifold, linear_model
from tqdm import tqdm

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import experiments as exp

#%% custom classes to allow for identity gradients
class RayLou(nn.ReLU):
    def __init__(self, linear_grad=False):
        super(RayLou,self).__init__()
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return (x>0).float()

class TanAytch(nn.Tanh):
    def __init__(self, linear_grad=False):
        super(TanAytch,self).__init__()
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return 1-nn.Tanh()(x).pow(2)

#%% data
num_var = 3
dim_inp = 25 # dimension per variable
ndat = 5000 # total
noise = 0.1

# apply_rotation = False
apply_rotation = True

input_task = util.RandomDichotomies(d=[(0,1,2,3),(0,2,4,6),(0,3,4,7)])
# output_task = util.RandomDichotomies(d=[(0,1,6,7), (0,2,5,7)]) # xor of first two
output_task = util.RandomDichotomies(d=[(0,1,6,7)]) # 3d xor
# output_task = util.RandomDichotomies(d=[(0,1,6,7), (0,4,5,6)]) # 3d xor
# output_task = util.RandomDichotomies(d=[(0,1,2,3),(0,2,4,6)])
# output_task = util.RandomDichotomies(d=[(0,1,4,5),(0,2,5,7),(0,1,6,7)]) # 3 incompatible dichotomies
# input_task = util.RandomDichotomies(d=[(0,1),(0,2)])
# output_task = util.RandomDichotomies(d=[(0,1)])

# generate inputs
inp_condition = np.random.choice(2**num_var, ndat)
# var_bit = (np.random.rand(num_var, num_data)>0.5).astype(int)
var_bit = input_task(inp_condition).numpy().T

mns = np.random.randn(dim_inp,num_var,1)*var_bit[None,:,:] \
    + np.random.randn(dim_inp,num_var,1)*(1-var_bit[None,:,:])

clus_mns = np.reshape(mns.transpose((1,0,2)), (num_var*dim_inp,-1)).T

if apply_rotation:
    C = np.random.rand(num_var*dim_inp, num_var*dim_inp)
    clus_mns = clus_mns@la.qr(C)[0][:num_var*dim_inp,:]

inputs = torch.tensor(clus_mns + np.random.randn(ndat, num_var*dim_inp)*noise).float()

# generate outputs
targets = output_task(inp_condition)


#%% Train with manual backprop
# manual = True
manual = False
ppp = 1 # 0 is MSE, 1 is cross entropy

two_layers = False
# two_layers = True
# nonneg = True
nonneg = False
# train_out = True
train_out = False 

linear_grad = False
# linear_grad = True

# nonlinearity = RayLou(linear_grad)
nonlinearity = TanAytch(linear_grad)

correct_mse = False # if True, rescales the MSE targets to be more like the log odds

N = 100

nepoch = 3000
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
inp_PS = []
inp_CCGP = []
crnr_PS = []
crnr_CCGP = []
lindim = []
gradz_sim = []
gradlin_sim = []
for epoch in tqdm(range(nepoch)):

    # loss = net.grad_step(dl, optimizer)
    
    running_loss = 0
    
    # idx = np.random.choice(n_trn, np.min([5000,ndat]), replace=False)
    if two_layers:
        z1 = nonlinearity(torch.matmul(W1,inputs[idx_tst,:].T) + b1)
        z = nonlinearity(torch.matmul(W2,z1) + b2)
    else:
        z = nonlinearity(torch.matmul(W1,inputs[idx_tst,:].T) + b1)
    pred = torch.matmul(W,z) + b
    
    if ppp == 0:
        perf = np.sum((pred.T-targets[idx_tst,:]).detach().numpy()**2,1).mean(0)
    else:
        perf = ((pred.T>0) == targets[idx_tst,:]).detach().numpy().mean(0)
    test_perf.append(perf)
    
    # this is just the way I compute the abstraction metrics, sorry
    clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
    gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
    D = assistants.Dichotomies(len(np.unique(inp_condition)),
                                input_task.positives, extra=0)
    
    ps = []
    ccgp = []
    for _ in D:
        ps.append(D.parallelism(z.T.detach().numpy(), inp_condition[idx_tst], clf))
        ccgp.append(D.CCGP(z.T.detach().numpy(), inp_condition[idx_tst], gclf, max_iter=1000))
    inp_PS.append(ps)
    inp_CCGP.append(ccgp)
    
    D = assistants.Dichotomies(len(np.unique(inp_condition)),
                                [(0,2,3,4),(0,4,6,7),(0,1,3,7),(0,1,2,6)], extra=0)
    ps = []
    ccgp = []
    for _ in D:
        ps.append(D.parallelism(z.T.detach().numpy(), inp_condition[idx_tst], clf))
        ccgp.append(D.CCGP(z.T.detach().numpy(), inp_condition[idx_tst], gclf, max_iter=1000))
    crnr_PS.append(ps)
    crnr_CCGP.append(ccgp)
    
    _, S, _ = la.svd(z.detach()-z.mean(1).detach()[:,None], full_matrices=False)
    eigs = S**2
    lindim.append((np.sum(eigs)**2)/np.sum(eigs**2))
    
    # Gradient similarity
    # if np.mod(epoch,10)==0:
    if epoch in [0,nepoch-1]:
        errb = (targets[idx_tst,:].T - nn.Sigmoid()(pred)) # bernoulli
        errg = (targets[idx_tst,:].T - pred) # gaussian
        
        err = ppp*errb + (1-ppp)*errg # convex sum, in case you want that
        
        d2 = (W.T@err)*nonlinearity.deriv(z) # gradient of the currents
        if two_layers:
            W2.grad = -(d2@z1.T)/len(idx_tst)
            b2.grad = -d2.mean(1, keepdim=True)
            
            d1 = (W2@d2)*nonlinearity.deriv(z1)
            W1.grad = -(d1@inputs[idx_tst,:])/len(idx_tst)
            b1.gad = -d1.mean(1, keepdim=True)
        else:
            W1.grad = -(d2@inputs[idx_tst,:])/len(idx_tst)
            b1.gad = -d2.mean(1, keepdim=True)
        
        conds = inp_condition[idx_tst]
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
            z1 = nonlinearity(torch.matmul(W1,inps.T) + b1)
            z = nonlinearity(torch.matmul(W2,z1) + b2)
        else:
            z = nonlinearity(torch.matmul(W1,inps.T) + b1)
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
            
            d2 = (W.T@err)*nonlinearity.deriv(z) # gradient of the currents
            if two_layers:
                W2.grad = -(d2@z1.T)/inps.shape[0]
                b2.grad = -d2.mean(1, keepdim=True)
                
                d1 = (W2@d2)*nonlinearity.deriv(z1)
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
        
        if nonneg and train_out:
            W = W*(W>0)
            b = b*(b>0)
        
        running_loss += loss.item()
        
    # train_loss.append(loss)
    # print('epoch %d: %.3f'%(epoch,running_loss/(j+1)))
    train_loss.append(running_loss/(j+1))
    # print(running_loss/(i+1))
    
#%% Abstraction metrics
if two_layers:
    z1 = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy()
    z = nonlinearity(torch.matmul(W2,torch.tensor(z1)) + b2).detach().numpy().T
else:
    z = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy().T
pred = torch.matmul(W,torch.tensor(z).T) + b

# z = net.enc.network[:-2](torch.tensor(inputs)).detach().numpy()
N = z.shape[1]
max_dichs = 50 # the maximum number of untrained dichotomies to test

# this_task = input_task
this_task = output_task

all_PS = []
all_CCGP = []
all_CCGP_ = []
CCGP_out_corr = []
mut_inf = []
all_SD = []
indep = []

indep.append(this_task.subspace_information())
# z = this_exp.train_data[0].detach().numpy()
# z = linreg.predict(this_exp.train_data[0])@W1.T
n_compute = np.min([5000, z.shape[0]])

idx = np.random.choice(z.shape[0], n_compute, replace=False)
# idx_tst = idx[::4] # save 1/4 for test set
# idx_trn = np.setdiff1d(idx, idx_tst)

cond = inp_condition[idx]

# cond = util.decimal(this_exp.train_data[1][idx,...])
num_cond = len(np.unique(cond))

# xor = np.where(~(np.isin(range(num_cond), args['dichotomies'][0])^np.isin(range(num_cond), args['dichotomies'][1])))[0]
## Loop over dichotomies
# D = assistants.Dichotomies(num_cond, args['dichotomies']+[xor], extra=50)

# choose dichotomies to have a particular order
Q = num_var
D_fake = assistants.Dichotomies(num_cond, this_task.positives, extra=7000)
mi = np.array([this_task.information(p) for p in D_fake])
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
    out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in this_task.positives]))
    
    CCGP.append(D.CCGP(z[idx,:], cond, gclf, cntxt, twosided=True))
    
    # shattering
    d[:,i] = D.coloring(cond)
    
# dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
dclf.fit(z[idx,:], d, tol=1e-5)

if two_layers:
    z1 = nonlinearity(torch.matmul(W1,inputs.T) + b1)
    z = nonlinearity(torch.matmul(W2,z1) + b2).detach().numpy().T
else:
    z = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy().T
   
# z = this_exp.test_data[0].detach().numpy()
# z = linreg.predict(this_exp.test_data[0])@W1.T
idx = np.random.choice(ndat, n_compute, replace=False)


d_tst = np.array([D.coloring(inp_condition[idx]) for _ in D]).T
SD = dclf.test(z[idx,:], d_tst).squeeze()


all_PS.append(PS)
all_CCGP.append(CCGP)
CCGP_out_corr.append(out_corr)
all_SD.append(SD)
mut_inf.append(mi[midx])

R = np.repeat(np.array(CCGP_out_corr),2,-1)
basis_dependence = np.array(indep).max(1)
out_MI = np.array(mut_inf)

#%%
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
    scat = plt.scatter(xfoo, yfoo, s=10, c=cfoo, zorder=10, cmap=mutinfo_cmap)
else:
    scat = plt.scatter(xfoo, yfoo, s=10, c=(0.5,0.5,0.5), zorder=10)
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
inps = []
for i,d in enumerate(input_task.positives):
    # # n = np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
    # #               for p in pos_conds])[0][0]
    # inps.append(plt.scatter(xfoo[[i,i+ndic,i+2*ndic]], yfoo[[i,i+ndic,i+2*ndic]], 
    #            s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))
    n = np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                  for p in pos_conds])[0][0]
    inps.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='d',
                s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

outs = []
for i,d in enumerate(output_task.positives):
    n = np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                  for p in pos_conds])[0][0]
    outs.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='s',
                s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

# baba = np.arange(len(xfoo)//ndic)*ndic
# par = plt.scatter(xfoo[baba], yfoo[baba], 
#                   marker='o', edgecolors='r', s=60, facecolors='none', linewidths=3)
# mag = plt.scatter(xfoo[baba+1], yfoo[baba+1], 
#                   marker='o', edgecolors='g', s=60, facecolors='none', linewidths=3)
# xor = plt.scatter(xfoo[baba+2], yfoo[baba+2], 
#                   marker='o', edgecolors='b', s=60, facecolors='none', linewidths=3)

# plt.legend([par,mag, xor], ['Var1', 'Var2', 'XOR'])
plt.legend(inps+outs, ['input %d'%(i+1) for i in range(len(inps))] + ['output %d'%(i+1) for i in range(len(outs))])

# plt.legend(outs, ['output %d'%(i+1) for i in range(len(outs))])

#%%
U, S, V = la.svd(inputs.numpy()-inputs.numpy().mean(0)[None,:],full_matrices=False)

# colorby = inp_condition
# colorby = util.decimal(outputs).numpy()
colorby = np.isin(inp_condition, pos_conds[])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(U[:,0],U[:,1],U[:,2],s=100, c=colorby)
# for i in np.unique(these_conds):
#     c = [int(i), int(np.mod(i+1,U.shape[0]))]
#     ax.plot(U[c,0],U[c,1],U[c,2],'k')
util.set_axes_equal(ax)

plt.title('PCA dimension: %.2f'%((np.sum(S**2)**2)/np.sum(S**4)))


