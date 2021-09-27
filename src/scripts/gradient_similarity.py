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
import plotting as dicplt

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
    def __init__(self, linear_grad=False, rand_grad=False):
        super(TanAytch,self).__init__()
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return 1-nn.Tanh()(x).pow(2)

class Iden(nn.Identity):
    def __init__(self, linear_grad=False):
        super(Iden,self).__init__()
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return torch.ones(x.shape)

#%% Pick data format
num_cond = 8
num_var = 3

# which_data = 'assoc'
# which_data = 'class'
which_data = 'struc_class'

ndat = 5000

# Associative task 
if which_data == 'assoc':
    
    p = 2**num_var
    allowed_actions = [0,1,2]
    # allowed_actions = [0,1,2,4]
    # allowed_actions = [0]
    p_action = [0.7,0.15,0.15]
    # p_action = [0.61, 0.13, 0.13, 0.13]
    # p_action = [1.0]
    
    output_states = (this_exp.train_data[0][:ndat,:].data+1)/2
    # output_states = this_exp.train_data[1][:ndat,:].data
    
    input_states = (this_exp.train_data[0][:ndat,:].data+1)/2
    
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

# Classification w/ random inputs
elif which_data == 'class':
    input_states = this_exp.train_data[0][:ndat,:].data
    output_states = this_exp.train_data[1][:ndat,:].data
    
    abstract_conds = util.decimal(this_exp.train_data[1])[:ndat]
    
    inputs = input_states.float()
    targets = output_states
    
    inp_condition = this_exp.train_conditions[:ndat]

# Classification w/ structured inputs
elif which_data == 'struc_class':
    num_var = 2
    dim_inp = 1 # dimension per variable
    noise = 0.0
    
    ndat = 5000
    
    num_cond = 2**num_var
    
    apply_rotation = False
    # apply_rotation = True
    
    # input_task = util.RandomDichotomies(d=[(0,1,2,3),(0,2,4,6),(0,1,4,5)])
    input_task = util.RandomDichotomies(d=[(0,1),(0,2)])
    # task = util.RandomDichotomies(d=[(0,3,5,6)]) # 3d xor
    # task = util.RandomDichotomies(d=[(0,1,6,7)]) # 2d xor
    # task = util.RandomDichotomies(d=[(0,1,3,5),(0,2,3,6),(0,1,2,4)]) # 3 corners
    # task = util.RandomDichotomies(d=[(0,1,3,5)]) # corner dichotomy
    task = util.RandomDichotomies(d=[(0,3)])
    
    # generate inputs
    inp_condition = np.random.choice(2**num_var, ndat)
    # inp_condition = np.arange(ndat)
    # var_bit = (np.random.rand(num_var, num_data)>0.5).astype(int)
    var_bit = input_task(inp_condition).numpy().T
    
    means = np.random.randn(num_var, dim_inp)
    means /= la.norm(means,axis=1, keepdims=True)

    mns = (means[:,None,:]*var_bit[:,:,None]) - (means[:,None,:]*(1-var_bit[:,:,None]))
            
    clus_mns = np.reshape(mns.transpose((0,2,1)), (dim_inp*num_var,-1)).T
        
    if apply_rotation:
        C = np.random.rand(num_var*dim_inp, num_var*dim_inp)
        clus_mns = clus_mns@la.qr(C)[0][:num_var*dim_inp,:]
    
    inputs = torch.tensor(clus_mns + np.random.randn(ndat, num_var*dim_inp)*noise).float()
    
    # generate outputs
    targets = task(inp_condition)
    
    abstract_conds = inp_condition

# %%
manual = True
# manual = False
ppp = 1 # 0 is MSE, 1 is cross entropy

two_layers = False
# two_layers = True
# nonneg = True
nonneg = False
# train_out = True
train_out = False 

linear_grad = False
# linear_grad = True

# average_grad = False
# average_grad = True

# nonlinearity = RayLou(linear_grad)
nonlinearity = TanAytch(linear_grad)
# nonlinearity = Iden()

correct_mse = False # if True, rescales the MSE targets to be more like the log odds

N = 100

nepoch = 2000
lr = 1e-4
bsz = 100

n_trn = int(ndat*0.8)
idx_trn = np.random.choice(ndat, n_trn, replace=False)
idx_tst = np.setdiff1d(range(ndat), idx_trn)

# idx_trn = np.arange(ndat)
# idx_tst = np.arange(ndat)

dset = torch.utils.data.TensorDataset(inputs[idx_trn], targets[idx_trn])
dl = torch.utils.data.DataLoader(dset, batch_size=bsz, shuffle=True)

# set up network (2 layers)
# ba = 1/np.sqrt(N)
ba = 1
W1 = torch.FloatTensor(N,inputs.shape[1]).uniform_(-ba,ba)
# W1 = torch.FloatTensor([[1,1],[1,-1],[-1,1],[-1,-1]]).repeat_interleave(N//4,0).repeat_interleave(dim_inp,1)
# W1 = torch.FloatTensor([[1,-1],[-1,1]]).repeat_interleave(N//2,0)
# b1 = torch.FloatTensor(N,1).uniform_(-ba,ba)
# b1 = torch.FloatTensor(torch.zeros(N,1))
b1 = torch.FloatTensor(torch.ones(N,1)*0.1)
W1.requires_grad_(True)
b1.requires_grad_(True)

if two_layers:
    ba = 1/np.sqrt(N)
    W2 = torch.FloatTensor(N,N).uniform_(-ba,ba)
    b2 = torch.FloatTensor(torch.zeros(N,1))
    W2.requires_grad_(True)
    b2.requires_grad_(True)

ba = 1/np.sqrt(targets.shape[1])

if nonneg:
    W = torch.FloatTensor(targets.shape[1],N).uniform_(0,2*ba)
    b = torch.FloatTensor(targets.shape[1],1).uniform_(0,2*ba)
else:
    # W = torch.FloatTensor(targets.shape[1],N).uniform_(-ba,ba)
    W = torch.FloatTensor([1,-1]).repeat(N//2)[None,:]
    # W *= (W>0)
    # W = torch.FloatTensor(torch.ones(targets.shape[1],N))
    # b = torch.FloatTensor(targets.shape[1],1).uniform_(-ba,ba)
    b = torch.FloatTensor(torch.zeros(targets.shape[1],1))

if two_layers:
    optimizer = optim.Adam([W1, b1, W2, b2], lr=lr)
else:
    if train_out:
        optimizer = optim.Adam([W1, b1, W, b], lr=lr)
    else:
        optimizer = optim.Adam([W1], lr=lr)

train_loss = []
test_perf = []
PS = []
CCGP = []
SD = []
lindim = []
gradz_sim = []
gradlin_sim = []
weights = []
# weights2 = []
biases = []
# grad_mag = []
for epoch in tqdm(range(nepoch)):

    # loss = net.grad_step(dl, optimizer)
    if not np.mod(epoch,10):
        weights.append(1*W1.detach().numpy())
    # if two_layers:
    #     weights2.append(1*W2.detach().numpy())
        biases.append(1*b1.detach().numpy())
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
                                input_task.positives+task.positives, extra=5)
    
    ps = []
    ccgp = [] 
    for _ in D:
        ps.append(D.parallelism(z.T.detach().numpy(), inp_condition[:ndat][idx_tst], clf))
        ccgp.append(D.CCGP(z.T.detach().numpy(), inp_condition[:ndat][idx_tst], gclf, max_iter=1000))
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
        
        d2 = (W.T@err)*nonlinearity.deriv(z) # gradient of the currents
        
        conds = abstract_conds[idx_tst]
        cond_grad = np.array([d2[:,conds==i].mean(1).detach().numpy() for i in np.unique(conds)])
        gradz_sim.append(util.cosine_sim(cond_grad-cond_grad.mean(0),cond_grad-cond_grad.mean(0)))
        
        # cond_grad = np.array([(W.T@err)[:,conds==i].mean(1).detach().numpy() for i in np.unique(conds)])
        cond_grad = np.array([(d2[:,conds==i]@inputs[idx_tst,:][conds==i,:]).detach().numpy().T for i in np.unique(conds)])
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
            curr1 = torch.matmul(W1,inps.T) + b1
            z = nonlinearity(torch.matmul(W2,z1) + b2)
            curr = torch.matmul(W2,z1) + b2
        else:
            z = nonlinearity(torch.matmul(W1,inps.T) + b1)
            curr = torch.matmul(W1,inps.T) + b1
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
            
            d2 = (W.T@err)*nonlinearity.deriv(curr) # gradient of the currents
            if two_layers:
                W2.grad = -(d2@z1.T)/inps.shape[0]
                b2.grad = -d2.mean(1, keepdim=True)
                
                d1 = (W2@d2)*nonlinearity.deriv(curr1)
                W1.grad = -(d1@inps)/inps.shape[0]
                b1.grad = -d1.mean(1, keepdim=True)
            else:
                W1.grad = -(d2@inps)/inps.shape[0]
                b1.grad = -d2.mean(1, keepdim=True)
            
            # W1 += lr*dw
            # b1 += lr*db
        else:
            loss.backward()
        
        if epoch == 0:
            init_grad_w = -(d2@inps)/inps.shape[0]
            init_grad_b = -d2.mean(1, keepdim=True)
        # grad_mag.append(la.norm(W1.grad.numpy(), axis=0))
        
        optimizer.step()

        running_loss += loss.item()
        
    # train_loss.append(loss)
    # print('epoch %d: %.3f'%(epoch,running_loss/(j+1)))
    train_loss.append(running_loss/(j+1))
    # print(running_loss/(i+1))

weights = np.array(weights)
weights2 = np.array(weights2)
biases = np.squeeze(biases)


#%%

# plot_this = np.squeeze(CCGP).mean(-1)
plot_this = np.array(PS)

plt.figure()
epochs = range(1,len(PS)+1)
# plt.plot(range(1,len(inp_PS)+1),out_PS)
# plt.semilogx()

trn = []
for dim in range(task.dim_output):
    thisone = plt.plot(epochs, plot_this[...,dim])[0]
    trn.append(thisone)
    plt.semilogx()

untrn = plt.plot(epochs, plot_this[...,task.dim_output:].mean(1),color=(0.5,0.5,0.5),zorder=0)[0]

plt.legend(trn + [untrn], ['Var %d'%(n+1) for n in range(task.dim_output)] + ['XOR'])

#%%
if two_layers:
    z1 = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy()
    z = nonlinearity(torch.matmul(W2,torch.tensor(z1)) + b2).detach().numpy().T
else:
    z = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy().T
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

cond = inp_condition[idx]

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
    out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in task.positives]))
    
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

# %%
# mask = (R.max(2)==1) # context must be an output variable   
# mask = (np.abs(R).sum(2)==0) # context is uncorrelated with either output variable
# mask = (np.abs(R).sum(2)>0) # context is correlated with at least one output variable
mask = ~np.isnan(R).max(2) # context is uncorrelated with the tested variable

almost_all_CCGP = util.group_mean(np.squeeze(all_CCGP).squeeze(), mask)


PS = util.group_mean(np.squeeze(all_PS), mask.sum(-1)>0, axis=0)
CCGP = util.group_mean(almost_all_CCGP, mask.sum(-1)>0, axis=0)
SD = util.group_mean(np.squeeze(all_SD), mask.sum(-1)>0, axis=0)
# SD = np.array(all_SD).mean(0)

ndic = len(PS)

PS_err = np.nanstd(np.squeeze(all_PS), axis=0)#/np.sqrt(len(all_PS))
CCGP_err = np.nanstd(almost_all_CCGP, axis=0)#/np.sqrt(len(all_CCGP))
SD_err = np.nanstd(np.squeeze(all_SD), axis=0)#/np.sqrt(len(all_SD))

output_dics = []
for d in task.positives:
    output_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                                for p in pos_conds])[0][0])

input_dics = []
for d in input_task.positives:
    input_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
              for p in pos_conds])[0][0])

dicplt.dichotomy_plot(PS, CCGP, SD,
                      input_dics=input_dics, output_dics=output_dics, 
                      other_dics=[pos_conds.index((0,2,5,7))], out_MI=out_MI.mean(0))

#%%
if two_layers:
    z1 = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy()
    z = nonlinearity(torch.matmul(W2,torch.tensor(z1)) + b2).detach().numpy().T
else:
    z = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy().T

x_ = np.stack([inputs[inp_condition==i,:].mean(0).detach().numpy() for i in np.unique(conds)]).T
y_ = np.stack([targets[inp_condition==i,:].mean(0).detach().numpy() for i in np.unique(conds)]).T
z_ = np.stack([z[inp_condition==i,:].mean(0) for i in np.unique(conds)]).T

dx = la.norm(x_[:,:,None] - x_[:,None,:], axis=0)/2
dy = la.norm(y_[:,:,None] - y_[:,None,:], axis=0)
dz = la.norm(z_[:,:,None] - z_[:,None,:], axis=0)

# Kx = np.einsum('i...k,j...k->ij...', x_.T-x_.mean(1,keepdims=True).T, x_.T-x_.mean(1,keepdims=True).T)
# Ky = np.einsum('i...k,j...k->ij...', y_.T-y_.mean(1,keepdims=True).T, y_.T-y_.mean(1,keepdims=True).T)
# Kz = np.einsum('i...k,j...k->ij...', z_.T-z_.mean(1,keepdims=True).T, z_.T-z_.mean(1,keepdims=True).T)

Kx = util.dot_product(x_-x_.mean(1,keepdims=True), x_-x_.mean(1,keepdims=True))
Ky = util.dot_product(y_-y_.mean(1,keepdims=True), y_-y_.mean(1,keepdims=True))
Kz = util.dot_product(z_-z_.mean(1,keepdims=True), z_-z_.mean(1,keepdims=True))

#%%
x_ = np.stack([inputs[inp_condition==i,:].mean(0).detach().numpy() for i in np.unique(conds)]).T
y_ = np.stack([targets[inp_condition==i,:].mean(0).detach().numpy() for i in np.unique(conds)]).T
# x_ = inputs.detach().numpy().T
# y_ = targets.detach().numpy().T

rep = np.einsum('abi,ic->abc',weights,x_)
pred = np.einsum('aib,i->ab',nonlinearity(torch.tensor(rep)+torch.tensor(biases)[:,:,None]),W.squeeze())
f_z = nonlinearity.deriv(torch.tensor(rep+biases[:,:,None]))
err = torch.tensor(y_) - nn.Sigmoid()(torch.tensor(pred))

lin_grad = err[:,:,None,None]*W[None,:,:,None]
nonlin_grad = ((lin_grad.squeeze()*f_z.transpose(1,2)))

dw_lin = lin_grad*x_.T[None,:,None,:]
dw_nonlin = nonlin_grad[...,None]*x_.T[None,:,None,:]

#%% initialization-averaged
# this_nonlin = RayLou()
this_nonlin = TanAytch()

N_grid = 21
this_range = np.abs(weights).max()
# this_range=1

# this_bias = np.random.randn(N_grid**2,1)*0.1
this_bias = np.ones((N_grid**2,1))*0.1

err_avg = y_ - y_.mean()
x_avg = x_ - x_.mean(1,keepdims=True)

wa, wb = np.meshgrid(np.linspace(-this_range,this_range,N_grid),np.linspace(-this_range,this_range,N_grid))

fake_W = np.stack([wa.flatten(),wb.flatten()]).T
fake_fz = this_nonlin.deriv(torch.tensor(fake_W@x_ + this_bias)).numpy()

fake_grads = x_avg@(err_avg*fake_fz).T

plt.quiver(fake_W[:,0],fake_W[:,1],fake_grads[0,:],fake_grads[1,:], color=(0.5,0.5,0.5))

#%%

n_mds = 3
n_compute = 500

fake_task = util.RandomDichotomies(num_cond,num_var,0)
fake_task.positives = task.positives

idx = np.random.choice(inputs.shape[0], n_compute, replace=False)

if two_layers:
    z1 = nonlinearity(torch.matmul(W1,inputs[idx,:].T) + b1).detach().numpy().T
    z = nonlinearity(torch.matmul(W2,z1) + b2)
else:
    z = nonlinearity(torch.matmul(W1,inputs[idx,:].T) + b1).detach().numpy().T
   
# ans = this_exp.train_data[1][idx,...]
ans = fake_task(inp_condition[:ndat])[idx]

cond = util.decimal(ans)
# cond = this_exp.train_conditions[idx]

# colorby = cond
colorby = inp_condition[idx]
# colorby = targets[idx,1]
# colorby = input_task(inp_condition)[idx,0].numpy()

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


