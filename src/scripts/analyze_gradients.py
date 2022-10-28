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
import dichotomies as dics
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
    def accel(self,x):
        if self.linear_grad:
            return torch.zeros(x.shape)
        else:
            return (torch.exp(-x.pow(2)/(2*self.beta**2))/(self.beta*np.sqrt(2*np.pi))).float()

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

# class NoisyTanAytch(nn.Tanh):
#     def __init__(self, noise=1, linear_grad=False, rand_grad=False):
#         super(NoisyTanAytch,self).__init__()
#         self.linear_grad = linear_grad
#         self.rand_grad = rand_grad
#         self.noise = noise
#     def deriv(self, x):
#         if self.linear_grad:
#             if self.rand_grad:
#                 return torch.rand(x.shape)
#             else:
#                 return torch.ones(x.shape)
#         else:
#             return torch.stack([1-nn.Tanh()(x+torch.randn(1)*self.noise).pow(2) for _ in range(5000)]).mean(0)

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


###################
        
def gans_tangent(w_hat, r=1):
    ''' basis for tangent space on the gans disc '''
    N = w_hat.shape[0]
    
    nrm_dsc = la.norm(w_hat,axis=0)**2
    A = (r**2 - nrm_dsc)[None,None,...]
    B = (r**2 + (nrm_dsc/(r**2-nrm_dsc)))
    
    cov = np.einsum('i...,j...->ij...', w_hat, w_hat)
    
    g = (np.tensordot(np.eye(N),np.sqrt(B), axes=0) - cov/np.sqrt(A))/B

    return g

def hemi_tangent(p, r=1):
    
    v1 = gans_tangent(p[:-1])
    
    ww = (p[:-1]**2).sum(0)
    numer = r**4 - ww*r**2
    denom = (r**4 -ww*r**2 + ww)**2
    
    return np.concatenate([v1, numer*p[None,:-1,...]/denom], axis=0)

def plane2disc(w, r=1):
    ''' features along first dimension '''
    return r*w/np.sqrt(r**2 + (w**2).sum(0))

def disc2plane(w_hat, r=1):
    return r*w_hat/np.sqrt(r**2 - (w_hat**2).sum(0))

def plane2hemi(w, r=1):
    z = np.sqrt(r**2 - (w**2).sum(0)/(r**2+(w**2).sum(0)))
    return np.concatenate([plane2disc(w, r), r-z[None,...]],axis=0)

def hemi_expmap(p, v, r=1):
    ''' point p on the hemisphere, tangent vector v '''
    # r = la.norm(p, axis=0)p
    p_cntr = p*1
    p_cntr[-1] -= r
    nrm_v = la.norm(v, axis=0)
    return np.cos(nrm_v/r)*(p_cntr/r) + np.sin(nrm_v/r)*(v/nrm_v)

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
    num_cond = 8
    num_var = 3
    dim_inp = 200 # dimension per variable
    noise = 0.1
    
    # task = tasks.RandomDichotomies(d=[(0,1),(0,2)])
    task = tasks.RandomDichotomies(d=[(0,1,2,3)])
    # task = tasks.StandardBinary(3)
    # this_exp = exp.random_patterns(task, SAVE_DIR, 
    #                                 num_class=num_cond,
    #                                 dim=dim_inp,
    #                                 var_noise=noise)
    input_task = tasks.RandomPatterns(num_cond, dim_inp, noise)
    
    # abstract_conds = util.decimal(this_exp.train_data[1])[:ndat]
    # inp_condition = this_exp.train_conditions[:ndat]
    inp_condition = np.random.choice(num_cond, ndat)
    abstract_conds = inp_condition
    
    inputs = input_task(inp_condition)
    targets = task(inp_condition)

    x_pos = input_task.means.T

    # x_pos = this_exp.means.T
    # x_pos = np.eye(4) - 0.25
    # x_pos /= la.norm(x_pos,axis=0, keepdims=True)

# Classification w/ structured inputs
elif which_data == 'struc_class':
    num_var = 2
    dim_inp = 25 # dimension per variable
    noise = 0.2
    
    ndat = 2000
    
    num_cond = 2**num_var
    
    apply_rotation = False
    # apply_rotation = True
    
    # input_task = tasks.RandomDichotomies(d=[(0,1,2,3),(0,2,4,6),(0,1,4,5)])
    # input_task = tasks.RandomDichotomies(d=[(0,1),(0,2)])
    # input_task = tasks.TwistedCube(tasks.StandardBinary(2), dim_inp, f=0.2, noise_var=noise)
    # input_task = tasks.EmbeddedCube(tasks.StandardBinary(3), dim_inp, noise_var=noise, rotated=False)
    # task = tasks.RandomDichotomies(d=[(0,3,5,6)]) # 3d xor
    # task = tasks.RandomDichotomies(d=[(0,1,6,7)]) # 2d xor
    # task = tasks.RandomDichotomies(d=[(0,1,3,5),(0,2,3,6),(0,1,2,4)]) # 3 corners
    # task = tasks.RandomDichotomies(d=[(0,1,3,5)]) # corner dichotomy
    task = tasks.RandomDichotomies(d=[(1,2)])
    # task = tasks.RandomDichotomies(d=[(0,1)])
    # input_task = tasks.NudgedCube(tasks.StandardBinary(2), task, dim_inp, nudge_mag=0.2, noise_var=noise)
    input_task = tasks.NudgedXOR(tasks.StandardBinary(2), dim_inp, nudge_mag=0.4, noise_var=noise, random=False)
    
    # generate inputs
    inp_condition = np.random.choice(2**num_var, ndat)
    # inp_condition = np.tile(np.arange(num_cond),ndat//num_cond)
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
    
    # x_pos = la.block_diag(*np.diff(input_task.means,axis=1).squeeze().tolist()).T
    
    x_pos /= la.norm(x_pos,axis=0, keepdims=True)
    
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
this_nonlin = TanAytch()
# this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch()
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
# this_nonlin = NoisyRayLou(noise)
which_grp = 1

these_vars = [1,2]
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

# lin_w =np.concatenate([weights_proj[:,grp==which_grp,:],biases[:,grp==which_grp,None]],axis=-1)

N_grid = 21
this_range=3.5
# up_range = this_range
up_range = 5.3
# down_range = this_range
down_range = 5.3

# use_empirical = True
use_empirical = False

# inp_coefs = x_pos.T@x_

this_bias = 0
other_var = 1

fake_J = np.array([[-1]])

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

#%% w.r.t. output weights

# this_nonlin = RayLou()
this_nonlin = TanAytch()
# this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch()
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
# this_nonlin = NoisyRayLou(noise)
which_grp = 1

these_vars = [1,2]



# weights_proj = np.einsum('ijk,kl->ijl',weights,basis)
x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
y_ = task(np.unique(inp_condition)).detach().numpy().T

N_x = x_.shape[0]

x_ = np.concatenate([x_, np.ones((1,x_.shape[-1]))], axis=0)
basis = la.block_diag(basis,1)

inp_coefs = basis.T@(x_/la.norm(x_,axis=0,keepdims=True))

# lin_w =np.concatenate([weights_proj[:,grp==which_grp,:],biases[:,grp==which_grp,None]],axis=-1)

N_grid = 21
this_range=3.5
# up_range = this_range
up_range = 5.3
# down_range = this_range
down_range = 5.3

# use_empirical = True
use_empirical = False

# inp_coefs = x_pos.T@x_

this_bias = 0
other_var = 1

fake_J = np.array([[-1]])

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

#%%

N_grid = 101
this_range=1.2
up_range = this_range
down_range = this_range

ar_scl = 0.1

inp_coefs = basis.T@(x_/la.norm(x_,axis=0,keepdims=True))
# x1, x2 = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*len(these_vars))

x1, = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*1)
x2=0

x_3 = []
for i in range(num_cond):
    x_3.append((inp_coefs[0,i]*x1 + inp_coefs[1,i]*x2)/(-inp_coefs[2,i]))
x_3 = np.stack(x_3)

#%%
noise = 0.0
epsilon = 0.5
N_neur = 41**2
n_epoch = 500
fake_lr = 1e-1
dim_inp = 100

rms_prop = False
# rms_prop = True
rms_beta = 0.99

# this_nonlin = RayLou()
# this_nonlin = TanAytch()
# this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch()
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
this_nonlin = NoisyRayLou(noise)

incl_bias = False
# incl_bias = True

task = tasks.RandomDichotomies(d=[(1,2)])
input_task = tasks.NudgedXOR(tasks.StandardBinary(2), dim_inp, nudge_mag=epsilon, noise_var=noise, random=False)
x_pos = la.block_diag(*np.diff(input_task.means[:2],axis=1).squeeze().tolist()).T
x_pos = np.concatenate([x_pos,input_task.means[2].flatten()[:,None]], axis=1)

basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))@np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])
# basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))

x_ = input_task(np.arange(4),noise=0).detach().numpy().T
if incl_bias:
    x_ = np.concatenate([x_, np.ones((1,x_.shape[-1]))], axis=0)
    basis = np.concatenate([basis, np.zeros((1,3))], axis=0)
    x_pos = np.concatenate([x_pos, np.zeros((1,3))], axis=0)

y_ = task(np.arange(4)).detach().numpy().T

N_x = x_.shape[0]

err_avg = y_ - y_.mean(1,keepdims=True)
# delta = err_avg.squeeze()
delta = np.array([-0.5220,  0.5170,  0.4693, -0.4810])

# init_range = 1/np.sqrt(N_x)
init_range = 0.1

fake_W = (2*np.random.rand(N_neur, N_x)-1)*init_range
# fake_W = np.stack([w.flatten() for w in np.meshgrid(*(np.linspace(-init_range,init_range,int(np.sqrt(N_neur))),)*2)]).T@basis[:,[1,2]].T
fake_J = np.ones((N_neur, 1))

fake_W_init = fake_W*1
if rms_prop:
    w_rms = np.zeros((N_neur, N_x))
    
def grad_func(w):
    fz = this_nonlin.deriv(torch.tensor(np.round(w@x_, 10))).numpy()
    return (x_@(fake_J@err_avg*fz).T).T

ws = []
bs = []
pps = []
pn1 = []
pn2 = []
w_nrm = []
for epoch in tqdm(range(n_epoch)):
    # if not np.mod(epoch,50):
    # ws.append(np.concatenate([fake_W[:,:-1]@basis, fake_W[:,[-1]]], axis=-1))
    # ws.append(fake_W*1)
    ws.append((fake_W@basis))
    if incl_bias:
        bs.append(fake_W[:,-1]*1)
    
    reps = this_nonlin(torch.tensor(fake_W@x_)).numpy()
    pps.append(np.sum([np.sign(delta[i]*delta[j])*reps[:,i]*reps[:,j] for i,j in zip([1,0,1,2],[2,3,3,0]) ], axis=0))
    pn1.append( (reps[:,0] - reps[:,1])**2)
    pn2.append((reps[:,2] - reps[:,3])**2)
    w_nrm.append(la.norm(reps, 2, axis=-1)**2)
    
    fake_fz = this_nonlin.deriv(torch.tensor(np.round(fake_W@x_, 10))).numpy()
    
    if rms_prop:
        w_rms = rms_beta*w_rms + (1-rms_beta)*fake_grads**2
        rms_lr = 1/np.sqrt((w_rms/(1-rms_beta))+1e-8)
    else:
        rms_lr = 1
    
    # RK4
    # rka = grad_func(fake_W)
    # rkb = grad_func(fake_W-(fake_lr/2)*rka)
    # rkc = grad_func(fake_W-(fake_lr/2)*rkb)
    # rkd = grad_func(fake_W-fake_lr*rkc)
    # fake_W -= (fake_lr/6)*(rka + 2*rkb + 2*rkc + rkd)
    
    fake_W -= fake_lr*grad_func(fake_W)*rms_lr

fake_weights_proj_init = fake_W_init[:,:]@basis
fake_weights_proj = fake_W[:,:]@basis

basins = [[0],[1],[2],[3],[0,3],[1,2]]
basin_prototype = np.stack([x_[:,p].sum(1) for p in basins]).T
basin_ovlp = fake_W@basin_prototype

w_coefs = np.array(ws)

plt.figure()
plt.scatter(fake_weights_proj_init[:,1],-fake_weights_proj_init[:,2],c=basin_ovlp.argmax(-1))
dicplt.square_axis()

# eps = np.tan(np.pi/2-np.arctan(input_task.nudge_mag))
eps = np.sqrt(2)/input_task.nudge_mag
plt.plot([-init_range,init_range],[-init_range*eps,init_range*eps],'k--')
plt.plot([-init_range,init_range],[init_range*eps,-init_range*eps],'k--')

#%%
# w_coefs = np.array(ws)

this_range = 4
# plt.scatter(fake_weights_proj_init[:,1],fake_weights_proj_init[:,2],c=basin_ovlp.argmax(-1))
# dicplt.square_axis()

N_grid = 31
# arr_scl = 0.03
these_vars = [1,2]

this_bias = -2

wawa = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*2)

fake_fake_W = np.stack([w.flatten() for w in wawa]).T
# WW = fake_fake_W@basis[:,[1,2]].T 
WW = fake_fake_W@basis[:,these_vars].T 

fake_fz = this_nonlin.deriv(torch.tensor((WW@x_ + this_bias))).numpy()

# fake_az = this_nonlin.accel(torch.tensor((WW@x_ + this_bias))).numpy()*((x_@(-1*err_avg*fake_fz).T).T@x_)

fake_grads = basis[:,these_vars].T@(x_@(-1*err_avg*fake_fz).T)
# fake_acc = basis.T@(x_@(-1*err_avg*fake_az).T)

# fake_hess = 

# nya = plt.quiver(fake_fake_W[:,0],fake_fake_W[:,1],
#             fake_grads[1,:],fake_grads[2,:], 
#             scale_units='xy',scale=1/arr_scl, color=(0.5,0.5,0.5))
# mya = plt.quiver(fake_fake_W[:,0]+fake_grads[1,:]*arr_scl,
#                  fake_fake_W[:,1]+fake_grads[2,:]*arr_scl,
#                  fake_hess[1,:],
#                  fake_hess[2,:], 
#                  scale_units='xy',scale=1/arr_scl,color=(0.2,0.2,0.2))
plt.quiver(fake_fake_W[:,0],fake_fake_W[:,1],
           fake_grads[0,:],fake_grads[1,:], color=(0.5,0.5,0.5))
# nya = plt.quiver(fake_fake_W[:,0],fake_fake_W[:,1],
#             fake_acc[1,:],fake_acc[2,:], color=(0.8,0.8,0.8))

reps = this_nonlin(torch.tensor(WW@x_ + this_bias)).numpy()
pps = np.sum([np.sign(delta[i]*delta[j])*reps[:,i]*reps[:,j] for i,j in zip([1,0,1,2],[2,3,3,0]) ], axis=0)
pn1 = np.abs(reps[:,0] - reps[:,1])
pn2 = np.abs(reps[:,2] - reps[:,3])
pnrm = np.where(pn1>0,pn1,1)*np.where(pn2>0,pn2,1) # local norm

dicplt.square_axis()

plt.plot(w_coefs[:,basin_ovlp.argmax(-1)==4,these_vars[0]],w_coefs[:,basin_ovlp.argmax(-1)==4,these_vars[1]],color='r',linewidth=0.5,alpha=0.5)
plt.plot(w_coefs[:,basin_ovlp.argmax(-1)!=4,these_vars[0]],w_coefs[:,basin_ovlp.argmax(-1)!=4,these_vars[1]],color='b',linewidth=0.5,alpha=0.5)

eps = np.sqrt(2)/input_task.nudge_mag
plt.plot([-this_range,this_range],[-eps*this_range,eps*this_range],'k--')
plt.plot([-this_range,this_range],[eps*this_range,-eps*this_range],'k--')

plt.imshow((pps/(la.norm(reps,2,axis=-1)**2)).reshape((N_grid,N_grid)),'bwr', 
           extent=[-this_range,this_range,-this_range,this_range])
dicplt.diverging_clim(plt.gca())

#%%

r = 8

this_range = 16
# plt.scatter(fake_w4ights_proj_init[:,1],fake_weights_proj_init[:,2],c=basin_ovlp.argmax(-1))
# dicplt.square_axis()

N_grid = 51
# arr_scl = 0.03

wawa = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*2)

fake_fake_W = np.stack([w.flatten() for w in wawa]).T

WW = fake_fake_W@basis[:,[1,2]].T
fake_fz = this_nonlin.deriv(torch.tensor((WW@x_ + this_bias))).numpy()

WW = plane2disc((fake_fake_W@basis[:,[1,2]].T).T, r=r).T
fake_fz = this_nonlin.deriv(torch.tensor(((WW@x_+ this_bias)/np.sqrt(r**2 - (WW**2).sum(axis=1,keepdims=True))))).numpy()


fake_grads = basis[:,[1,2]].T@(x_@(-1*err_avg*fake_fz).T)

fake_fake_W = plane2disc(fake_fake_W.T, r=r).T

Tp = gans_tangent(fake_fake_W.T, r=r)
fake_grads = np.einsum('ijk, jk->ik', Tp, fake_grads)

# w_disc = plane2disc(w_coefs.T, r=r).T

nya = plt.quiver(fake_fake_W[:,0],fake_fake_W[:,1],
            fake_grads[0,:],fake_grads[1,:], color=(0.5,0.5,0.5))

plt.plot(r*np.cos(np.linspace(0,2*np.pi,100)),r*np.sin(np.linspace(0,2*np.pi,100)),'k--')
dicplt.square_axis()

# plt.plot(w_disc[:,basin_ovlp.argmax(-1)==4,1],w_disc[:,basin_ovlp.argmax(-1)==4,2],color='r',linewidth=0.5,alpha=0.5)
# plt.plot(w_disc[:,basin_ovlp.argmax(-1)!=4,1],w_disc[:,basin_ovlp.argmax(-1)!=4,2],color='b',linewidth=0.5,alpha=0.5)

# eps = np.sqrt(2)/input_task.nudge_mag
# plt.plot([-1,1],[-eps,eps],'k--')
# plt.plot([-1,1],[eps,-eps],'k--')


#%%
noise = 0.0
N_grid = 101
n_epoch = 200
fake_lr = 1e-1
dim_inp = 100
 
# this_nonlin = RayLou()
# this_nonlin = TanAytch()
this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch()
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
# this_nonlin = NoisyRayLou(noise)

task = tasks.RandomDichotomies(d=[(1,2)])
y_ = task(np.arange(4)).detach().numpy().T
err_avg = y_ - y_.mean(1,keepdims=True)
delta = err_avg.squeeze()

init_range = 1/np.sqrt(dim_inp)
init_var = (((2*init_range)**2)/12)*(2*dim_inp)

grid_range = 3*init_var

incl_bias = False
# incl_bias = True

# variance of the initial weight overlaps

fake_J = np.ones((N_grid**2, 1))
def grad_func(w, x):
        fz = this_nonlin.deriv(torch.tensor(np.round(w@x, 10))).numpy()
        return (x@(fake_J@err_avg*fz).T).T

inp_basis = np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])

p_abs = []
pps = []
pn1 = []
pn2 = []
w_nrm = []
PS = []
for epsilon in np.linspace(0,1,20):
    
    input_task = tasks.NudgedXOR(tasks.StandardBinary(2), dim_inp, nudge_mag=epsilon, noise_var=noise, random=False)
    x_pos = la.block_diag(*np.diff(input_task.means[:2],axis=1).squeeze().tolist()).T
    x_pos = np.concatenate([x_pos,input_task.means[2].flatten()[:,None]], axis=1)
    
    basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))@inp_basis
    
    x_ = input_task(np.arange(4),noise=0).detach().numpy().T
    if incl_bias:
        x_ = np.concatenate([x_, np.ones((1,x_.shape[-1]))], axis=0)

    N_x = x_.shape[0]
    
    fake_W = (2*np.random.rand(N_grid**2, N_x)-1)*init_range
    # fake_W = np.stack([w.flatten() for w in np.meshgrid(*(np.linspace(-grid_range,grid_range,N_grid),)*2)]).T@basis[:,[1,2]].T
    
    # fake_weights_proj_init = fake_W@basis
    # wp = []
    for epoch in tqdm(range(n_epoch)):
        # RK4
        # wp.append(np.mean((fake_W@x_[:,0]>0)&(fake_W@x_[:,3]>0)))
        rka = grad_func(fake_W, x_)
        rkb = grad_func(fake_W-(fake_lr/2)*rka, x_)
        rkc = grad_func(fake_W-(fake_lr/2)*rkb, x_)
        rkd = grad_func(fake_W-fake_lr*rkc, x_)
        fake_W -= (fake_lr/6)*(rka + 2*rkb + 2*rkc + rkd)

    # fake_weights_proj = fake_W@basis
    reps = this_nonlin(torch.tensor(fake_W@x_)).numpy()
    pps.append(np.sum([np.sign(delta[i]*delta[j])*reps[:,i]*reps[:,j] for i,j in zip([1,0,1,2],[2,3,3,0]) ], axis=0))
    # pn1.append( np.abs(reps[:,0] - reps[:,1]))
    # pn2.append( np.abs(reps[:,2] - reps[:,3]))
    pn1.append( (reps[:,0] - reps[:,1])**2)
    pn2.append( (reps[:,2] - reps[:,3])**2)
    w_nrm.append(la.norm(reps, 2, axis=-1))
    
    abs_dir = x_@delta
    cat_dir = np.diff(x_[:,[0,3]],1).squeeze()
    
    # basin_ovlp = np.abs((fake_W[:,:-1]@np.diff(x_[:-1,[1,2]],1).squeeze())/(fake_W[:,:-1]@x_[:-1,[1,2]].sum(1)))<1
    basin_ovlp = np.abs(fake_W@cat_dir/(la.norm(cat_dir)+1e-8)) < np.abs(fake_W@abs_dir/(la.norm(abs_dir)+1e-8))
    
    # basin_ovlp = (fake_W@x_[:,0]>0)&(fake_W@x_[:,3]>0)
    # prob = sts.multivariate_normal(cov=np.eye(2)*init_var).pdf(fake_weights_proj_init[:,[1,2]])
    
    # p_abs.append(prob[basin_ovlp].sum()/prob.sum())
    p_abs.append(np.mean(basin_ovlp))

pps = np.array(pps)
pn1 = np.array(pn1)
pn2 = np.array(pn2)
w_nrm = np.array(w_nrm)
    
dem = np.where(pn1>0,pn1,1)*np.where(pn2>0,pn2,1)

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

#%%
def task_basis(W, y):
    
    num_cond = y.shape[-1]
    
    grp_weights, grp = np.unique(np.sign(W),axis=1, return_inverse=True)

    # compute relevant directions
    dic = dics.Dichotomies(num_cond)
    all_col = np.stack([2*dic.coloring(range(num_cond))-1 for _ in dic])
    
    corr_dir = W.T@y
    # corr_dir/=la.norm(corr_dir, axis=-1, keepdims=True)
    corr_dir/=np.max(corr_dir, axis=-1, keepdims=True)
    # lin_dir = corr_dir@x.T
    # lin_dir /= la.norm(lin_dir, axis=-1, keepdims=True)
    
    kernels = ((all_col@y.T)@grp_weights==0)
    bad_dirs = np.stack([all_col[k,:].T@all_col[k,:] for k in kernels.T])[grp,:,:]
    anticorr_dir = 1.0*bad_dirs[np.arange(len(grp)),np.argmax(corr_dir>0,axis=1),:]
    # anticorr_dir/=la.norm(anticorr_dir, axis=-1, keepdims=True)
    anticorr_dir/=np.max(anticorr_dir, axis=-1, keepdims=True)
    # bad_dir = anticorr_dir@x.T
    # bad_dir /= la.norm(bad_dir, axis=-1, keepdims=True)
    
    return np.stack([corr_dir,anticorr_dir])

#%%

n_dat = 1000
dt = 1e-3
dim = 100
init_range = 0.1
beta = 100
epsilon = 1.0

task = tasks.RandomDichotomies(d=[(1,2)])
y_ = task(np.arange(4)).detach().numpy().T
err_avg = y_ - y_.mean(1,keepdims=True)
delta = err_avg.squeeze()

# xs = np.random.randn(dim, 4)
input_task = tasks.NudgedXOR(tasks.StandardBinary(2), dim, nudge_mag=epsilon, noise_var=0, random=False)
x_pos = la.block_diag(*np.diff(input_task.means[:2],axis=1).squeeze().tolist()).T
x_pos = np.concatenate([x_pos,input_task.means[2].flatten()[:,None]], axis=1)

xs = input_task(np.arange(4),noise=0).detach().numpy().T

# inp_basis = np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])
# basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))@inp_basis
# basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))@np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])
basis = np.squeeze(task_basis(np.array([[1]]), err_avg))
basis /= la.norm(basis,axis=-1,keepdims=True)


w = np.random.randn(2*dim, n_dat)*init_range

# ws = [w.T@xs]
w_x = w.T@xs
ws = [w_x/np.sqrt(1+la.norm(w_x,axis=1, keepdims=True)**2)]
for t in tqdm(range(10000)):
    idx = np.random.choice(xs.shape[1], n_dat)
    xi = xs[:,idx]
    take_step = np.random.rand(n_dat) < 1/(1+np.exp(-beta*np.sum(w*xi,0)))
    w += dt*delta[idx]*xi*take_step
    # ws.append(w.T@xs)
    # ws.append((w/(1+la.norm(w,axis=0))).T@basis)
    w_x = w.T@xs
    ws.append(w_x/np.sqrt(1+la.norm(w_x,axis=1, keepdims=True)**2))

wa = np.stack(ws)
# www = (xs/la.norm(xs,axis=-1,keepdims=True)).T@basis
# bla = wa@www
bla = wa@basis.T

basins = [[0],[1],[2],[3],[0,3],[1,2]]
# basin_prototype = np.stack([xs[:,p].sum(1) for p in basins]).T
# basin_ovlp = fake_W@basin_prototype
basin_ovlp = np.stack([wa[-1,:,p].sum(0) for p in basins]).T

plt.figure()
plt.plot(bla[:,:,1], bla[:,:,0], alpha=0.1, color=(0.5,0.5,0.5))
# dicplt.square_axis()

# eps = np.tan(np.pi/2-np.arctan(input_task.nudge_mag))
# nya = np.max(np.abs(bla))
eps = np.arctan(np.sqrt(2)/epsilon)
plt.plot([-np.cos(eps),np.cos(eps)],[-np.sin(eps),np.sin(eps)],'k--')
plt.plot([-np.cos(eps),np.cos(eps)],[np.sin(eps),-np.sin(eps)],'k--')
plt.plot(np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),'k--')

x_proj = (xs.T@(xs@basis.T))
x_proj /= la.norm(x_proj, axis=1, keepdims=True)
plt.scatter(x_proj[:,1], x_proj[:,0], c=delta, cmap='bwr')

dicplt.square_axis()
# plt.scatter(bla[0,:,0], bla[0,:,2], c=basin_ovlp.argmax(1), alpha=1, zorder=10)

#%%

init_var = 0.1
eps = 0.1
# dt = 

x = 2*(tasks.StandardBinary(2)(np.arange(4)).numpy() - 0.5)
y = 2*(tasks.RandomDichotomies(d=[(1,2)])(np.arange(4)).numpy()-0.5)
x = np.concatenate([x, eps*y],axis=1)

kern = x@(y*x).T
l,v = la.eig(kern>0)


w_grd = np.

pw = [np.random.randn()]
for t in range(100):
    
    p0 = sts.norm.pdf(-(kern*kern).sum(0)/init_var)
    ci = v
    







