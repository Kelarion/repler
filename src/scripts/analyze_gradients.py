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
from matplotlib import colors as mpc
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
    def __init__(self, linear_grad=False, rand_grad=False):
        super(HardTanAytch,self).__init__()
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return ((x<1)&(x>-1)).float()

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
# which_data = 'assoc'
# which_data = 'class'
which_data = 'struc_class'

ndat = 1000

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
    num_cond = 4
    num_var = 2
    dim_inp = 100 # dimension per variable
    noise = 0.2
    
    task = tasks.RandomDichotomies(d=[(0,1),(0,2)])
    this_exp = exp.random_patterns(task, SAVE_DIR, 
                                    num_class=num_cond,
                                    dim=dim_inp,
                                    var_noise=noise)
    input_task = tasks.RandomDichotomies(d=[()])
    
    abstract_conds = util.decimal(this_exp.train_data[1])[:ndat]
    inp_condition = this_exp.train_conditions[:ndat]
    
    input_states = this_exp.train_data[0][:ndat,:].data #- this_exp.train_data[0][:ndat,:].data.mean(0,keepdims=True)
    output_states = this_exp.train_data[1][:ndat,:].data
    # input_states = torch.tensor((np.eye(4)[inp_condition,:]))
    
    min_num = np.unique(inp_condition,return_counts=True)[1].min()
    idx = np.sort(np.concatenate([np.random.choice(np.where(inp_condition==i)[0], min_num, replace=False) for i in range(4)]))
    
    inputs = input_states[idx,:].float()
    targets = output_states[idx,:]
    inp_condition = inp_condition[idx]
    ndat = inputs.shape[0]

    x_pos = this_exp.means.T
    # x_pos = np.eye(4) - 0.25
    # x_pos /= la.norm(x_pos,axis=0, keepdims=True)

# Classification w/ structured inputs
elif which_data == 'struc_class':
    num_var = 2
    dim_inp = 100 # dimension per variable
    noise = 0.1
    
    ndat = 1000
    
    num_cond = 2**num_var
    
    apply_rotation = False
    # apply_rotation = True
    
    # input_task = tasks.RandomDichotomies(d=[(0,1,2,3),(0,2,4,6),(0,1,4,5)])
    # input_task = tasks.RandomDichotomies(d=[(0,1),(0,2)])
    # input_task = tasks.TwistedCube(tasks.StandardBinary(2), dim_inp, f=0.2, noise_var=noise)
    # input_task = tasks.EmbeddedCube(tasks.StandardBinary(2), dim_inp, noise_var=noise, rotated=apply_rotation)
    # task = tasks.RandomDichotomies(d=[(0,3,5,6)]) # 3d xor
    # task = tasks.RandomDichotomies(d=[(0,1,6,7)]) # 2d xor
    # task = tasks.RandomDichotomies(d=[(0,1,3,5),(0,2,3,6),(0,1,2,4)]) # 3 corners
    # task = tasks.RandomDichotomies(d=[(0,1,3,5)]) # corner dichotomy
    task = tasks.RandomDichotomies(d=[(1,2)])
    # task = tasks.RandomDichotomies(d=[(0,1)])
    input_task = tasks.NudgedXOR(tasks.StandardBinary(2), dim_inp, nudge_mag=0.0, noise_var=noise, random=False)
    
    # generate inputs
    inp_condition = np.random.choice(2**num_var, ndat)
    # inp_condition = np.arange(ndat)
    # var_bit = (np.random.rand(num_var, num_data)>0.5).astype(int)
    # var_bit = input_task(inp_condition).numpy().T
    
    # means = np.random.randn(num_var, dim_inp)
    # means /= la.norm(means,axis=-1, keepdims=True)

    # mns = (means[:,None,:]*var_bit[:,:,None]) - (means[:,None,:]*(1-var_bit[:,:,None]))
            
    # clus_mns = np.reshape(mns.transpose((0,2,1)), (dim_inp*num_var,-1)).T
    # # clus_mns = np.concatenate([clus_mns, 1*(0.75*(inp_condition[:,None]==3)-0.25)], axis=1)
    # clus_mns = np.concatenate([clus_mns, 0.09*(2*(task(inp_condition)==1)-1)], axis=1)
    
    # if apply_rotation:
    #     C = np.random.rand(num_var*dim_inp+num_append, num_var*dim_inp+num_append)
    #     clus_mns = clus_mns@la.qr(C)[0][:num_var*dim_inp+num_append,:]
    
    # inputs = torch.tensor(clus_mns + np.random.randn(ndat, num_var*dim_inp)*noise).float()
    
    x_pos = la.block_diag(*np.diff(input_task.means[:2],axis=1).squeeze().tolist()).T
    x_pos = np.concatenate([x_pos,input_task.means[2].flatten()[:,None]], axis=1)
    # x_pos /= la.norm(x_pos,axis=0, keepdims=True)
    
    inputs = input_task(inp_condition)
    # inputs -= inputs.mean(0,keepdims=True)
    
    # generate outputs
    targets = task(inp_condition)
    
    abstract_conds = inp_condition
    
#%%

# this_nonlin = RayLou()
# this_nonlin = TanAytch() 
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
# this_nonlin = NoisyRayLou(noise)

# these_vars = [0,2]

# fake_J = [1]

# x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
# # x_ = x_pos
# y_ = task(np.unique(inp_condition)).detach().numpy().T

# inp_coefs = x_pos.T@x_

# # get the basis vectors for looking at the weights
# # labels = input_task.latent_task(np.unique(inp_condition)).numpy()
# # x_pos = (x_[...,None]*labels[None,...]).sum(1)/labels.sum(0) - (x_[...,None]*(1-labels[None,...])).sum(1)/labels.sum(0)
# # x_pos = la.block_diag(*np.diff(input_task.means,axis=1).squeeze().tolist()).T
# # x_pos /= la.norm(x_pos,axis=0, keepdims=True)

# N_grid = 21
# this_range=0.9
# up_range = this_range
# down_range = this_range

# this_bias = np.ones((N_grid**len(these_vars),1))*0

# err_avg = y_ - y_.mean(1,keepdims=True)

# wawa = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*len(these_vars))

# fake_W = np.stack([w.flatten() for w in wawa]).T
# WW = fake_W@(x_pos/la.norm(x_pos,axis=0, keepdims=True))[:,these_vars].T \
#     + x_pos[:,np.setdiff1d(range(x_pos.shape[-1]),these_vars)].T*0
# fake_fz = this_nonlin.deriv(torch.tensor((WW@x_ + this_bias))).numpy()
# # fake_fz = this_nonlin.deriv(torch.tensor((fake_W@(x_pos/la.norm(x_pos,axis=0, keepdims=True))[:,these_vars].T@inputs.numpy().T + this_bias))).numpy()
# # fake_fz = this_nonlin.deriv(torch.tensor((fake_W@x_pos.T@x_ + this_bias))).numpy()

# fake_grads = (x_pos/la.norm(x_pos,axis=0, keepdims=True)).T@(x_@(fake_J@err_avg*fake_fz).T)

#%%

# n_con = len(np.unique(inp_condition))

# plt.figure()
# for i in range(n_con):
#     plt.subplot(1,n_con, i+1)
#     plt.imshow(fake_fz[:,i].reshape((N_grid,N_grid)), extent=[-down_range,up_range,-down_range,up_range], cmap='bwr', alpha=0.5)
#     # plt.scatter(5*prots[these_vars[0],i],5*prots[these_vars[1],i], marker='*', s=100)

# %% 
# this_nonlin = RayLou()
# this_nonlin = TanAytch()
# this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch()
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
this_nonlin = NoisyRayLou(noise)
which_grp = 1

these_vars = [0,2]
basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))@np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])
# basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))@np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])
# basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))

# weights_proj = np.einsum('ijk,kl->ijl',weights,basis)
x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
y_ = task(np.unique(inp_condition)).detach().numpy().T

N_x = x_.shape[0]

x_ = np.concatenate([x_, np.ones((1,x_.shape[-1]))], axis=0)
basis = la.block_diag(basis,1)

inp_coefs = basis.T@(x_/la.norm(x_,axis=0,keepdims=True))

lin_w =np.concatenate([weights_proj[:,grp==which_grp,:],biases[:,grp==which_grp,None]],axis=-1)

N_grid = 101
this_range=3.5
# up_range = this_range
up_range = 0.3
# down_range = this_range
down_range = 0.3

# use_empirical = True
use_empirical = False

# inp_coefs = x_pos.T@x_

this_bias = 0
other_var = 1

fake_J = np.array([[1]])

if use_empirical:
    err_avg = targets.numpy().T - y_.mean(1,keepdims=True)
else:
    err_avg = y_ - y_.mean(1,keepdims=True)
delta = err_avg.squeeze()

wawa = np.meshgrid(*(np.linspace(-down_range,up_range,N_grid),)*len(these_vars))

inds = []
parcels = []
all_fz = []
all_grads = []
for v in np.linspace(-this_range,this_range,101):
    fake_W = np.stack([w.flatten() for w in wawa]).T
    WW = fake_W@basis[:,these_vars].T \
        + basis[:,np.setdiff1d(range(x_pos.shape[-1]),these_vars)[0]].sum(-1).T*v*other_var \
        + basis[:,[-1]].T*v*this_bias
    if use_empirical:
        fake_fz = this_nonlin.deriv(torch.tensor(np.round(WW@inputs.numpy().T, 6))).numpy()
    else:
        fake_fz = this_nonlin.deriv(torch.tensor(np.round(WW@x_, 6))).numpy()
    
    if use_empirical:
        fake_grads = basis.T@(inputs.numpy().T@(fake_J@err_avg*fake_fz).T)
    else:
        fake_grads = basis.T@(x_@(fake_J@err_avg*fake_fz).T)
    
    bofa, deez = np.unique(fake_fz, axis=0,return_inverse=True)
    inds.append(bofa)
    parcels.append(deez.reshape((N_grid,N_grid)))
    all_fz.append(fake_fz)
    all_grads.append(fake_grads)

# lin_w = weights_proj[:,grp==0,:]#@np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])

#%%
deez = parcels[50].flatten()
bofa = inds[50]

ar_scl = 1.8

# cntrs1 = np.array([fake_W[:len(deez)//2][deez[:len(deez)//2]==i,:].mean(0) for i in range(len(bofa))])
# cntrs2 = np.array([fake_W[len(deez)//2:][deez[len(deez)//2:]==i,:].mean(0) for i in range(len(bofa))])
# bofa = np.tile(inds[71].T, 2).T
# cntrs = np.concatenate([cntrs1,cntrs2], axis=0)
cntrs = np.array([fake_W[deez==i,:].mean(0) for i in range(len(bofa))])
dirs = bofa@(inp_coefs*err_avg)[these_vars,:].T
cntrs -= 0.5*ar_scl*dirs

plt.figure()
plt.subplot(1,2,2)
plt.title('Gradients for z=%.1f slice'%other_var)

# the parcels
plt.imshow(deez.reshape((N_grid,N_grid)), extent=[-down_range,up_range,-down_range,up_range], alpha=0.5)

plt.axis('equal')
plt.axis('square')
plt.xlim([-down_range,up_range])
plt.ylim([-down_range,up_range])


# plot the components 
inp_dirs = inp_coefs[these_vars,:]*0.5*this_range
inp_dirs_signed = (inp_coefs*err_avg)[these_vars,:]

cols = np.array(dicplt.color_cycle('brg', 4))
for i,fz in enumerate(bofa):
    if np.sum(fz)>0:
        comps = inp_dirs_signed[:,fz>0]
        idx = np.where(fz>0)[0]
        plt.quiver(cntrs[i,0],cntrs[i,1], comps[0,0], comps[1,0], color=cols[idx[0]],
                   scale_units='xy',
                   scale=1/ar_scl)
        if np.sum(fz)>1:
            for j in range(len(idx)-1):
                plt.quiver(cntrs[i,0]+comps[0,:j+1].sum()*ar_scl,cntrs[i,1]+comps[1,:j+1].sum()*ar_scl, 
                           comps[0,j+1], comps[1,j+1], color=cols[idx[j+1]],
                           scale_units='xy',
                           scale=1/ar_scl)
            plt.quiver(cntrs[i,0],cntrs[i,1],dirs[i,0],dirs[i,1],color='k',
                       scale_units='xy',
                       scale=1/ar_scl)


# the inputs
plt.subplot(1,2,1)
plt.title('Inputs (signed)')

# plt.quiver(fake_W[:,0],fake_W[:,1],
#             fake_grads[these_vars[0],:],fake_grads[these_vars[1],:], color=(0.5,0.5,0.5))
plt.scatter(1.1*inp_dirs[0,delta>0],1.1*inp_dirs[1,delta>0], marker="+", s=100, c=cols[delta>0,:])
plt.scatter(1.1*inp_dirs[0,delta<0],1.1*inp_dirs[1,delta<0], marker="_", s=100, c=cols[delta<0,:])
plt.quiver(0,0,inp_dirs[0,:],inp_dirs[1,:], color=cols, 
          scale_units='xy', scale=1)
# plt.scatter(inp_dirs[0,:],inp_dirs[1,:], marker="*", c=np.arange(4), s=100)

# plt.scatter(inp_dirs[0,delta>0],inp_dirs[1,delta>0], marker="+", s=100)
# plt.scatter(inp_dirs[0,delta<0],inp_dirs[1,delta<0], marker="_", s=100)

plt.axis('equal')
plt.axis('square')
plt.xlim([-down_range,up_range])
plt.ylim([-down_range,up_range])

#%%
# this_nonlin = RayLou()
# this_nonlin = TanAytch()
# this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch()
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
this_nonlin = NoisyRayLou(noise)


these_vars = [0,2]

this_bias = 0
other_var = 0

N_grid = 101
this_range=0.3
up_range = this_range
down_range = this_range

ar_scl = 0.5

fake_J = np.array([[1]])

# use_empirical = True
use_empirical = False

inds = []
parcels = []
all_fz = []
all_grads = []

task = tasks.RandomDichotomies(d=[(0,3)])
for v in np.linspace(0,1,51):
    input_task = tasks.NudgedCube(tasks.StandardBinary(2), task, dim_inp, nudge_mag=v, noise_var=noise)

    x_pos = la.block_diag(*np.diff(input_task.means,axis=1).squeeze().tolist()).T
    x_pos = np.concatenate([x_pos,input_task.nudge_dir.T], axis=1)
    x_pos /= la.norm(x_pos,axis=0, keepdims=True)

    
    basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))@np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])/np.sqrt(2)
    # basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))
    
    x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
    y_ = task(np.unique(inp_condition)).detach().numpy().T
    N_x = x_.shape[0]
    
    x_ = np.concatenate([x_, np.ones((1,x_.shape[-1]))], axis=0)
    basis = la.block_diag(basis,1)
    
    if use_empirical:
        err_avg = targets.numpy().T - y_.mean(1,keepdims=True)
    else:
        err_avg = y_ - y_.mean(1,keepdims=True)
    delta = err_avg.squeeze()
    
    wawa = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*len(these_vars))


    fake_W = np.stack([w.flatten() for w in wawa]).T
    WW = fake_W@basis[:,these_vars].T
    if use_empirical:
        fake_fz = this_nonlin.deriv(torch.tensor(np.round(WW@inputs.numpy().T, 6))).numpy()
    else:
        fake_fz = this_nonlin.deriv(torch.tensor(np.round(WW@x_, 6))).numpy()
    
    if use_empirical:
        fake_grads = basis.T@(inputs.numpy().T@(fake_J@err_avg*fake_fz).T)
    else:
        fake_grads = basis.T@(x_@(fake_J@err_avg*fake_fz).T)
    
    bofa, deez = np.unique(fake_fz, axis=0,return_inverse=True)
    inds.append(bofa)
    parcels.append(deez.reshape((N_grid,N_grid)))
    all_fz.append(fake_fz)
    all_grads.append(fake_grads)


#%%

N_grid = 101
# this_range=1.2
up_range = this_range
down_range = this_range

ar_scl = 0.1


# inp_coefs = x_pos.T@x_

this_bias = np.ones((N_grid**len(these_vars),1))*0

fake_J = np.array([[1]])

err_avg = y_ - y_.mean(1,keepdims=True)
delta = err_avg.squeeze()

# x1, x2 = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*len(these_vars))

x1, = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*1)
x2=0

x_3 = []
for i in range(num_cond):
    x_3.append((inp_coefs[0,i]*x1 + inp_coefs[1,i]*x2)/(-inp_coefs[2,i]))
x_3 = np.stack(x_3)

#%%
# this_nonlin = RayLou()
# this_nonlin = TanAytch()
# this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch()
this_nonlin = Iden()
# this_nonlin = Poftslus(1)
# this_nonlin = NoisyRayLou(1.0)
x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T

bwa = (np.random.rand(200,10000)-0.5)*0.2

nya = this_nonlin.deriv(torch.tensor(bwa.T@x_))
nyanya = nya[:,:,None]*nya[:,None,:]

K = ((nyanya*util.dot_product(x_,x_)[None,:,:]).mean(0))

l,v = la.eig(K)

idx = np.argsort(-l)
vecs = v[:,idx].T@x_.T




