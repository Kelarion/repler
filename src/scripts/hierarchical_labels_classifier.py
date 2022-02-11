

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
import tasks
import plotting as dicplt
import grammars as gram

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

class Poftslus(nn.Softplus):
    def __init__(self, beta=1, linear_grad=False):
        super(Poftslus,self).__init__(beta=beta)
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return (1/(1+torch.exp(-self.beta*x))).float()

class NoisyRayLou(nn.ReLU):
    def __init__(self, beta=1, linear_grad=False):
        super(NoisyRayLou,self).__init__()
        self.linear_grad = linear_grad
        self.beta = beta
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return 0.5*(1+torch.erf(x/(self.beta*np.sqrt(2)))).float()

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

class NoisyTanAytch(nn.Tanh):
    def __init__(self, noise=1, linear_grad=False, rand_grad=False):
        super(NoisyTanAytch,self).__init__()
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
        self.noise = noise
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return torch.exp(-x.pow(2)/(1+(2*self.noise**2)))

class HardTanAytch(nn.Hardtanh):
    def __init__(self, linear_grad=False, rand_grad=False, vmin=-1, vmax=1):
        super(HardTanAytch,self).__init__(vmin, vmax)
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
        self.vmin = vmin
        self.vmax = vmax
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return ((x<self.vmax)&(x>self.vmin)).float()

class Iden(nn.Identity):
    def __init__(self, linear_grad=False):
        super(Iden,self).__init__()
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return torch.ones(x.shape)

# %% Pick data format
ndat = 1000

K = 2
respect = False
# respect = True

layers = [K**0,K**1]
# layers = [1, 1, 2, 4]
# layers = [1,1,1]

child_prob = 0.7
# child_prob = 1 

Data = gram.HierarchicalData(layers, fan_out=K, respect_hierarchy=respect)

#%%
dim_inp = 100
noise = 0.1

num_var = Data.num_vars
num_cond = Data.num_data

pos_conds = [np.nonzero(l)[0] for l in Data.represent_labels(Data.terminals)]

task = tasks.RandomDichotomies(d=pos_conds, c=num_cond)
input_task = tasks.RandomPatterns(num_cond, dim_inp, noise)

inp_condition = np.random.choice(num_cond, ndat)
abstract_conds = inp_condition

inputs = input_task(inp_condition)
targets = task(inp_condition)

x_pos = input_task.means.T

inputs -= inputs.mean(0,keepdims=True)

# %%
# look at effect of output weight magnitude
# other activation functions (neural net? lol)

manual = True
# manual = False
ppp = 0 # 0 is MSE, 1 is cross entropy
# theoretical = True
theoretical = False

use_adam = False
# use_adam = True

do_rms = False
# do_rms = True
rms_beta = 0.99

two_layers = False
# two_layers = True

# nonneg = True
nonneg = False

# train_out = True
train_out = False

linear_grad = False
# linear_grad = True

# train_biases = False
train_biases = True

# nonlinearity = RayLou(linear_grad)
nonlinearity = TanAytch(linear_grad)
# nonlinearity = HardTanAytch(linear_grad, vmin=-0.5, vmax=0.5)
# nonlinearity = Iden()

correct_mse = False # if True, rescales the MSE targets to be more like the log odds

N = 100
# N = 264

nepoch = 10000
lr = 1e-3
bsz = 200

# n_trn = int(ndat*0.8)
# idx_trn = np.random.choice(ndat, n_trn, replace=False)
# idx_tst = np.setdiff1d(range(ndat), idx_trn)

idx_trn = np.arange(ndat)
idx_tst = np.arange(ndat)

dset = torch.utils.data.TensorDataset(inputs[idx_trn], targets[idx_trn], torch.tensor(abstract_conds)[idx_trn])
dl = torch.utils.data.DataLoader(dset, batch_size=bsz, shuffle=True)

# set up network (2 layers) 
ba = 1/np.sqrt(dim_inp)
# ba = 1e-10
W1 = torch.FloatTensor(N,inputs.shape[1]).uniform_(-ba,ba)
# W1 = torch.FloatTensor(N,num_var).uniform_(-ba,ba)@torch.FloatTensor(x_pos.T)
# W1 = students.Feedforward([inputs.shape[1],N], nonlinearity='Tanh').network.layer0.weight.data
# W1 = torch.FloatTensor([[1,1],[1,-1],[-1,1],[-1,-1]]).repeat_interleave(N//4,0).repeat_interleave(dim_inp,1)
# W1 = torch.FloatTensor([[1,-1],[-1,1]]).repeat_interleave(N//2,0)
# W1 = torch.FloatTensor(torch.zeros(N,inputs.shape[1]))
if train_biases:
    b1 = torch.FloatTensor(N,1).uniform_(-ba,ba)
else:
# b1 = students.Feedforward([inputs.shape[1],N], nonlinearity='Tanh').network.layer0.bias.data[:,None]
    b1 = torch.FloatTensor(torch.zeros(N,1))
# b1 = torch.FloatTensor(torch.ones(N,1)*1)
# b1 = torch.FloatTensor(N,1).uniform_(0,2)
# W1.requires_grad_(True)
# b1.requires_grad_(True)

if two_layers:
    ba = 1/np.sqrt(N)
    W2 = torch.FloatTensor(N,N).uniform_(-ba,ba)
    b2 = torch.FloatTensor(torch.zeros(N,1))
    W2.requires_grad_(True)
    b2.requires_grad_(True)

# ba = 1/np.sqrt(targets.shape[1])
ba = 10/N

if nonneg:
    W = torch.FloatTensor(targets.shape[1],N).uniform_(0,2*ba)
    b = torch.FloatTensor(targets.shape[1],1).uniform_(0,2*ba)
else:
    W = torch.FloatTensor(targets.shape[1],N).uniform_(-ba,ba)
    # W = torch.FloatTensor([1,-1]).repeat(N//2)[None,:]
    # W *= (W>0)
    # W = torch.FloatTensor(torch.ones(targets.shape[1],N))
    # W = students.BinaryReadout(N, targets.shape[1], rotated=True).weight.detach()*10/N
    # W = torch.cat([students.BinaryReadout(N//2, targets.shape[1], rotated=False).weight.detach(),
    #                 students.BinaryReadout(N//2, targets.shape[1], rotated=True).weight.detach()], dim=-1)*10/N
    # W_bin = torch.cat([students.BinaryReadout(256, 2**num_var, rotated=False).weight.detach(),
    #                 students.BinaryReadout(8, 2**num_var, rotated=True).weight.detach()], dim=-1)*10/N
    # W = torch.tensor(x_pos).float()@W_bin
    # W = students.Feedforward([N,targets.shape[1]], nonlinearity=[None]).network.layer0.weight.data
    b = torch.FloatTensor(targets.shape[1],1).uniform_(-ba,ba)
    # b = torch.FloatTensor(torch.zeros(targets.shape[1],1))
    # b = students.Feedforward([N,targets.shape[1]], nonlinearity=[None]).network.layer0.bias.data[:,None]

if use_adam:
    if two_layers:
        optimizer = optim.Adam([W1, b1, W2, b2], lr=lr)
    else:
        if train_out:
            W.requires_grad = True
            b.requires_grad = True
            optimizer = optim.Adam([W1, b1, W, b], lr=lr)
        else:
            if train_biases:
                optimizer = optim.Adam([W1,b1], lr=lr)
            else:
                optimizer = optim.Adam([W1], lr=lr)

x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
y_ = task(np.unique(inp_condition)).detach().numpy().T.squeeze()

train_loss = []
test_perf = []
PS = []
CCGP = []
SD = []
lindim = []
gradz_sim = []
gradlin_sim = []
err_m = []
err_v = []
weights = []
weights2 = []
biases = []
adam_m = []
adam_v = []
w_rms = []
b_rms = []
local_ps = []
# grad_mag = []
for epoch in tqdm(range(nepoch)):

    # loss = net.grad_step(dl, optimizer)
    if not np.mod(epoch,10) or epoch<20:
        weights.append(1*W1.detach().numpy())
    # if two_layers:
        weights2.append(1*W.data.detach().numpy())
        biases.append(1*b1.detach().numpy())
        if epoch>0 and use_adam:
            adam_m.append(1*optimizer.state[W1]['exp_avg'].detach().numpy())
            adam_v.append(1*optimizer.state[W1]['exp_avg_sq'].detach().numpy())
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
    
    # reps = nonlinearity(torch.tensor(W1.numpy()@x_)).numpy()
    # pps = np.sum([np.sign((y_[i]-y_.mean())*(y_[j]-y_.mean()))*reps[:,i]*reps[:,j] for i,j in zip([1,0,1,2],[2,3,3,0]) ], axis=0)
    # pnrm = (la.norm(reps,2,axis=-1)**2) # local norm
    # local_ps.append(pps/pnrm)
    
    z_bar = np.array([z[:,inp_condition==c].detach().numpy().mean(1) for c in np.unique(inp_condition)])
    ps = []
    for lab in y_[y_.sum(1)>1,:]:
        ps.append(dic.parallelism_score(z_bar.T, np.arange(num_cond), lab))
    PS.append(ps)
    
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
    # for j in range(ndat//bsz):
        # optimizer.zero_grad()
        
        inps, outs, labs = btch
        # inps = input_task(np.repeat(np.arange(num_cond),bsz))
        # outs = task(np.repeat(np.arange(num_cond),bsz))
        
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
            if theoretical:
                # err = (outs.T - y_.mean())*(np.exp(-epoch/150)+0.01)
                # err = (outs.T - task(np.arange(num_cond)).mean(0,keepdims=True).T)#*(outs.T - nn.Sigmoid()(pred)).abs().mean()
                # err += torch.randn(*err.shape)*0.2
                # fake_z = nonlinearity(torch.matmul(W1,input_task(labs, noise=0).T) + b1)
                # fake_pred = torch.matmul(W,fake_z) + b
                # err = outs.T - nn.Sigmoid()(fake_pred)
                # err += torch.randn(*err.shape)*0.2
                err = (outs.T - nn.Sigmoid()(pred)).numpy()
            else:
                errb = (outs.T - nn.Sigmoid()(pred)) # bernoulli
                errg = (outs.T - pred) # gaussian
                
                err = ppp*errb + (1-ppp)*errg # convex sum, in case you want that
            
            err_m.append([err[0,labs==i].mean() for i in range(4)])
            err_v.append([err[0,labs==i].std() for i in range(4)])
            
            # dir_update = input_task(labs, noise=0)
            
            d2 = (W.T@err)*nonlinearity.deriv(curr) # gradient of the currents
            if two_layers:
                W2.grad = -(d2@z1.T)/inps.shape[0]
                b2.grad = -d2.mean(1, keepdim=True)
                
                d1 = (W2@d2)*nonlinearity.deriv(curr1)
                W1.grad = -(d1@inps)/inps.shape[0]
                b1.grad = -d1.mean(1, keepdim=True)
                
                W2 += lr*W2.grad
                b2 += lr*b2.grad
            else:
                W1.grad = -(d2@inps)/inps.shape[0]
                b1.grad = -d2.mean(1, keepdim=True)
            
            if not use_adam:
                if do_rms:
                    if epoch==0 and j==0:
                        w_rms.append( rms_beta*torch.zeros(W1.grad.shape) + 0.1*W1.grad.pow(2) )
                        b_rms.append( rms_beta*torch.zeros(b1.grad.shape) + 0.1*b1.grad.pow(2) )
                    elif j==0:
                        w_rms.append( rms_beta*w_rms[epoch-1] + (1-rms_beta)*W1.grad.pow(2) )
                        b_rms.append( rms_beta*b_rms[epoch-1] + (1-rms_beta)*b1.grad.pow(2) )
                    else:
                        w_rms[epoch] = rms_beta*w_rms[epoch] + (1-rms_beta)*W1.grad.pow(2)
                        b_rms[epoch] = rms_beta*b_rms[epoch] + (1-rms_beta)*b1.grad.pow(2)
                    w_alr = 1/np.sqrt((w_rms[epoch]/(1-rms_beta))+1e-8)
                    b_alr = 1/np.sqrt((b_rms[epoch]/(1-rms_beta))+1e-8)
                else:
                    w_alr = 1
                    b_alr = 1
                W1 -= lr*w_alr*W1.grad
                if train_biases:
                    b1 -= lr*b_alr*b1.grad
            # W1 += lr*dw
            # b1 += lr*db
        else:
            loss.backward()
        
        # if epoch == 0:
        #     init_grad_w = -(d2@inps)/inps.shape[0]
        #     init_grad_b = -d2.mean(1, keepdim=True)
        # grad_mag.append(la.norm(W1.grad.numpy(), axis=0))
        
        if use_adam:
            optimizer.step()

        running_loss += loss.item()
        
    # train_loss.append(loss)
    # print('epoch %d: %.3f'%(epoch,running_loss/(j+1)))
    train_loss.append(running_loss/(j+1))
    # print(running_loss/(i+1))

weights = np.array(weights)
weights2 = np.array(weights2)
biases = np.squeeze(biases)

adam_m = np.array(adam_m)
adam_v = np.array(adam_v)

weights_proj = np.einsum('ijk,kl->ijl',weights,x_pos/la.norm(x_pos,axis=0, keepdims=True))
W = W.detach().numpy()

#%%

if two_layers:
    z1 = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy()
    z = nonlinearity(torch.matmul(W2,torch.tensor(z1)) + b2).detach().numpy().T
else:
    z = nonlinearity(torch.matmul(W1,inputs.T) + b1).detach().numpy().T

x_ = np.stack([inputs[inp_condition==i,:].mean(0).detach().numpy() for i in np.unique(inp_condition)]).T
y_ = np.stack([targets[inp_condition==i,:].mean(0).detach().numpy() for i in np.unique(inp_condition)]).T
z_ = np.stack([z[inp_condition==i,:].mean(0) for i in np.unique(inp_condition)]).T

dx = la.norm(x_[:,:,None] - x_[:,None,:], axis=0)/2
dy = la.norm(y_[:,:,None] - y_[:,None,:], axis=0)
dz = la.norm(z_[:,:,None] - z_[:,None,:], axis=0)

# Kx = np.einsum('i...k,j...k->ij...', x_.T-x_.mean(1,keepdims=True).T, x_.T-x_.mean(1,keepdims=True).T)
# Ky = np.einsum('i...k,j...k->ij...', y_.T-y_.mean(1,keepdims=True).T, y_.T-y_.mean(1,keepdims=True).T)
# Kz = np.einsum('i...k,j...k->ij...', z_.T-z_.mean(1,keepdims=True).T, z_.T-z_.mean(1,keepdims=True).T)

Kx = util.dot_product(x_-x_.mean(1,keepdims=True), x_-x_.mean(1,keepdims=True))
Ky = util.dot_product(y_-y_.mean(1,keepdims=True), y_-y_.mean(1,keepdims=True))
Kz = util.dot_product(z_-z_.mean(1,keepdims=True), z_-z_.mean(1,keepdims=True))

inp_align = np.sum(Kz*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Kz*Kz))
out_align = np.sum(Kz*Ky)/np.sqrt(np.sum(Ky*Ky)*np.sum(Kz*Kz))

apply_correction = False
# apply_correction = True

c_xy = np.sum(Ky*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Ky*Ky))

if apply_correction:
    cos_foo = np.linspace(0,1,1000)
    
    ub = c_xy*cos_foo + np.sqrt(1-c_xy**2)*np.sqrt(1-cos_foo**2)
    
    phi = (np.pi/2 -np.arccos(c_xy))/2  # re-align it with the orthogonal case
    basis = np.array([[np.cos(phi),np.cos(np.pi/2-phi)],[np.sin(phi),np.sin(np.pi/2-phi)]])
    rot = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    correction = rot@la.inv(basis)
    
    correct_align = (correction)@np.stack([inp_align,out_align])
    correct_bound = (correction)@np.stack([cos_foo, ub])
    
    plt.plot(correct_bound[0,:],correct_bound[1,:], 'k--')
    plt.scatter(correct_align[0,...], correct_align[1,...])
else:
    cos_foo = np.linspace(c_xy,1,1000)
    ub = c_xy*cos_foo + np.sqrt(1-c_xy**2)*np.sqrt(1-cos_foo**2)
    plt.plot(cos_foo, ub, 'k--')
    plt.scatter(inp_align, out_align)


