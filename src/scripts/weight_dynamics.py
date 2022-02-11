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
which_data = 'assoc'
# which_data = 'class'
# which_data = 'struc_class'

ndat = 1000

# Associative task 
if which_data == 'assoc':
    num_var = 3
    num_var_task = 3
    dim_inp = 100 # dimension 
    noise = 0.2
    
    # p = 2**num_var
    # allowed_actions = [0,1,2]
    allowed_actions = [0,1,2,4]
    # allowed_actions = [0]
    # p_action = [0.7,0.15,0.15]
    p_action = [0.61, 0.13, 0.13, 0.13]
    # p_action = [1.0]
    
    input_task = tasks.RandomPatterns(2**num_var, dim_inp, noise)
    task = tasks.RandomTransisions(input_task, allowed_actions, p_action, num_var=num_var_task)
    
    inp_condition = np.random.choice(2**num_var, ndat)
    abstract_conds = np.mod(inp_condition, 2**num_var_task)
    
    inputs = input_task(inp_condition)
    targets = task(abstract_conds)

    x_pos = input_task.means.T

    inputs -= inputs.mean(0,keepdims=True)
    
    inp_condition = abstract_conds
    
    # output_states = (this_exp.train_data[0][:ndat,:].data+1)/2
    # # output_states = this_exp.train_data[1][:ndat,:].data
    
    # input_states = (this_exp.train_data[0][:ndat,:].data+1)/2
    
    # abstract_conds = util.decimal(this_exp.train_data[1])[:ndat]
    # cond_set = np.unique(abstract_conds)
    
    # # draw the "actions" for each data point
    # actns = torch.tensor(np.random.choice(allowed_actions, ndat, p=p_action)).int()
    # actions = torch.stack([(actns&(2**i))/2**i for i in range(num_var)]).float().T
    
    # # act_rep = assistants.Indicator(p,p)(util.decimal(actions).int())
    # act_rep = actions.data
    
    # # inputs = np.concatenate([input_states,act_rep], axis=1)
    # # # inputs = np.concatenate([input_states, this_exp.train_data[1]], axis=1)
    # inputs = input_states.float()
    
    # # # sample the successor states, i.e. input + action
    # successors = np.mod(this_exp.train_data[1][:ndat,:]+actions, 2)
    
    # succ_conds = util.decimal(successors)
    # succ_counts = np.unique(succ_conds, return_counts=True)[1]
    
    # # should the targets be sampled from the training set, or another set? 
    # # train set would be like an autoencoder training, so maybe that's fine
    # samps = np.concatenate([np.random.choice(np.where(abstract_conds==c)[0],n) \
    #                         for c,n in zip(cond_set,succ_counts)])
    
    # unscramble = np.argsort(np.argsort(succ_conds))
    # successor_idx = samps[unscramble]
    # targets = output_states[successor_idx,:]
    
    # targets = output_state

# Classification w/ random inputs
elif which_data == 'class':
    num_cond = 4
    num_var = 2
    dim_inp = 200 # dimension per variable
    noise = 0.1
    
    task = tasks.RandomDichotomies(d=[(0,1),(0,2)])
    # task = tasks.RandomDichotomies(d=[(0,1,2,3)])
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

    inputs -= inputs.mean(0,keepdims=True)
    # x_pos = this_exp.means.T
    # x_pos = np.eye(4) - 0.25
    # x_pos /= la.norm(x_pos,axis=0, keepdims=True)

# Classification w/ structured inputs
elif which_data == 'struc_class':
    num_var = 2
    dim_inp = 100 # dimension per variable
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
    input_task = tasks.NudgedXOR(tasks.StandardBinary(2), dim_inp, nudge_mag=0.2, noise_var=noise, random=False)
    
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
    # 
    # x_pos = la.block_diag(*np.diff(input_task.means,axis=1).squeeze().tolist()).T
    
    x_pos /= la.norm(x_pos,axis=0, keepdims=True)
    # x_pos = np.concatenate([x_pos, np.zeros((1,x_pos.shape[1]))], axis=0)
    
    inputs = input_task(inp_condition)
    inputs -= inputs.mean(0,keepdims=True)
    # inputs = torch.cat([inputs, torch.ones((inputs.shape[0],1))], axis=1)
    
    # generate outputs
    targets = task(inp_condition)
    
    abstract_conds = inp_condition

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

train_out = True
# train_out = False

linear_grad = False
# linear_grad = True

# train_biases = False
train_biases = True

nonlinearity = RayLou(linear_grad)
# nonlinearity = TanAytch(linear_grad)
# nonlinearity = HardTanAytch(linear_grad, vmin=-0.5, vmax=0.5)
# nonlinearity = Iden()

correct_mse = False # if True, rescales the MSE targets to be more like the log odds

N = 100
# N = 264

nepoch = 2000
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
    
    # this is just the way I compute the abstraction metrics, sorry
    clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
    gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
    # D = assistants.Dichotomies(len(np.unique(inp_condition)),
                                # input_task.positives+task.positives, extra=0)
    D = assistants.Dichotomies(len(np.unique(inp_condition)),
                                task.positives, extra=0)
    
    ps = []
    ccgp = [] 
    for _ in D:
        ps.append(D.parallelism(z.T.detach().numpy(), inp_condition[:ndat][idx_tst], clf))
        ccgp.append(D.CCGP(z.T.detach().numpy(), inp_condition[:ndat][idx_tst], gclf, max_iter=1000))
    PS.append(ps)
    CCGP.append(ccgp)
    
    # reps = nonlinearity(torch.tensor(W1.numpy()@x_)).numpy()
    # pps = np.sum([np.sign((y_[i]-y_.mean())*(y_[j]-y_.mean()))*reps[:,i]*reps[:,j] for i,j in zip([1,0,1,2],[2,3,3,0]) ], axis=0)
    # pnrm = (la.norm(reps,2,axis=-1)**2) # local norm
    # local_ps.append(pps/pnrm)
        
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

#%%
# x_ = np.stack([inputs[inp_condition==i,:].mean(0).detach().numpy() for i in np.unique(inp_condition)]).T
# y_ = np.stack([targets[inp_condition==i,:].mean(0).detach().numpy() for i in np.unique(inp_condition)]).T
# x_ = inputs.detach().numpy().T
# y_ = targets.detach().numpy().T
x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
# x_ = np.concatenate([x_, np.ones((1,4))], axis=0)
# x_ = x_pos
y_ = task(np.unique(inp_condition)).detach().numpy().T

rep = np.einsum('abi,ic->abc',weights,x_)
pred = np.einsum('aib,ci->acb',nonlinearity(torch.tensor(rep)+torch.tensor(biases)[:,:,None]),W)#+b.detach().numpy()
f_z = nonlinearity.deriv(torch.tensor(rep+biases[:,:,None]))
delta = torch.tensor(y_) - nn.Sigmoid()(torch.tensor(pred))

lin_grad = (delta[:,:,:,None]*W[None,:,None,:]).sum(1)
nonlin_grad = ((lin_grad.squeeze()*f_z.transpose(1,2)))

dw_lin = lin_grad[...,None]*x_.T[None,:,None,:]
dw_nonlin = nonlin_grad[...,None]*x_.T[None,:,None,:]


#%% Hand-picked basis
this_nonlin = nonlinearity
# this_nonlin = RayLou()
# this_nonlin = TanAytch()
# this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch(vmin=-0.5,vmax=0.5)
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
# this_nonlin = NoisyRayLou(noise)

this_group = 1

these_vars = [1,0]

# basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))@np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])
# basis = (x_pos/la.norm(x_pos,axis=0, keepdims=True))

# grp_weights, grp = np.unique(np.sign(W),axis=1, return_inverse=True)
grp_weights, grp = np.unique(np.sign(W_bin),axis=1, return_inverse=True)
grp_weights = x_pos@grp_weights 

fake_J = grp_weights[:,this_group]
# fake_J = grp_weights[:,this_group]
# fake_J = 1

dic = assistants.Dichotomies(num_cond)
all_col = np.stack([2*dic.coloring(range(num_cond))-1 for _ in dic])

# weights_proj = np.einsum('ijk,kl->ijl',weights,basis) #+ biases[:,-1:,None]
x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
# x_ = x_pos
y_ = task(np.unique(inp_condition)).detach().numpy().T

err_avg = y_ - y_.mean(1,keepdims=True)
# err_avg = targets.numpy().T - y_.mean(1,keepdims=True)
# err_avg = delta[0].numpy()
# x_avg = (x_) - (x_).mean(1,keepdims=True)

# good_dir = (grp_weights.T@err_avg@x_.T)[grp,:]
# good_dir /= la.norm(good_dir, axis=-1, keepdims=True)
good_dir = (grp_weights.T@err_avg@x_.T)[this_group,:]
good_dir /= la.norm(good_dir)
bad_dirs = all_col[((all_col@err_avg.T)@fake_J)==0].T@all_col[((all_col@err_avg.T)@fake_J)==0]
bad_dir = x_@bad_dirs[(fake_J@err_avg>0)][0].squeeze()
bad_dir /= la.norm(bad_dir)

# good_dir = (grp_weights.T@err_avg@x_.T)[grp,:]
# good_dir /= la.norm(good_dir, axis=-1, keepdims=True)
# bad_dir = (x_@(grp_weights[:,grp]*err_avg).prod(0))
# bad_dir /= la.norm(bad_dir)

basis = np.stack([good_dir,bad_dir]).T

weights_proj = np.einsum('ijk,kl->ijl',weights,basis)

N_grid = 31
# this_range = np.abs(weights_proj[0]).max() + 0.1
up_range = np.abs((weights_proj[...,these_vars]).max())*1.1 #+ 0.1
down_range = np.abs((weights_proj[...,these_vars]).min())*1.1 #+ 0.1
# this_range=0.7
this_range = np.max([up_range, down_range])
# up_range = this_range
# down_range = this_range
        
# this_bias = np.random.randn(N_grid**2,1)*0.1
this_bias = np.ones((N_grid**len(these_vars),1))*-2#b1.detach().numpy().mean()
# this_bias = b1.detach().numpy()


wawa = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*len(these_vars))

# fake_W = np.stack([w.flatten() for w in wawa]+[np.ones(N_grid**len(these_vars))*0 for _ in np.setdiff1d(range(x_pos.shape[-1]),these_vars)]).T
fake_W = np.stack([w.flatten() for w in wawa]).T
WW = fake_W@basis[:,these_vars].T #+ x_pos[:,np.setdiff1d(range(x_pos.shape[-1]),these_vars)].T*0
fake_fz = this_nonlin.deriv(torch.tensor((WW@x_ + this_bias))).numpy()
# fake_fz = this_nonlin.deriv(torch.tensor((WW@inputs.numpy().T + this_bias))).numpy()
# fake_fz = this_nonlin.deriv(torch.tensor((fake_W@(x_pos/la.norm(x_pos,axis=0, keepdims=True))[:,these_vars].T@inputs.numpy().T + this_bias))).numpy()
# fake_fz = this_nonlin.deriv(torch.tensor((fake_W@x_pos.T@x_ + this_bias))).numpy()

fake_grads = basis.T@(x_@(fake_J@err_avg*fake_fz).T)
# fake_grads = (x_pos/la.norm(x_pos,axis=0, keepdims=True)).T@(inputs.numpy().T@(fake_J@err_avg*fake_fz).T)
# fake_grads = ((fake_J@err_avg*fake_fz)@x_.T)@x_[:,these_vars]
# fake_lin_grads = (x_pos.T@x_)*(err_avg)

plt.figure()
nya = plt.quiver(fake_W[:,0],fake_W[:,1],
            fake_grads[these_vars[0],:],fake_grads[these_vars[1],:], color=(0.5,0.5,0.5))

# plt.quiver(fake_W[:,0],fake_W[:,1],
#            fake_grads[0,:],fake_grads[1,:], color=(0.5,0.5,0.5))

# reps = this_nonlin(torch.tensor(WW@x_ + this_bias)).numpy()
# pps = np.sum([np.sign(err_avg.squeeze()[i]*err_avg.squeeze()[j])*reps[:,i]*reps[:,j] for i,j in zip([1,0,1,2],[2,3,3,0]) ], axis=0)
# pnrm = (la.norm(reps,2,axis=-1)**2) # local norm


# plt.imshow((pps/pnrm).reshape((N_grid,N_grid)),'bwr', 
#            extent=[-this_range,this_range,-this_range,this_range])
# dicplt.diverging_clim(plt.gca())


plt.plot(weights_proj[:,grp==this_group,these_vars[0]],weights_proj[:,grp==this_group,these_vars[1]], 'r', linewidth=1)
# plt.plot(weights_proj[:,np.sign(W.squeeze())<0,these_vars[0]],weights_proj[:,np.sign(W.squeeze())<0,these_vars[1]], 'b', linewidth=1)
# plt.plot(lin_w[:,np.sign(W.squeeze())<0,these_vars[0]],lin_w[:,np.sign(W.squeeze())<0,these_vars[1]], 'b', linewidth=1)


# plt.plot(weights_proj[:,grp==0,these_vars[0]],weights_proj[:,grp==0,these_vars[1]], 'r', linewidth=2)
# plt.plot(weights_proj[:,grp==1,these_vars[0]],weights_proj[:,grp==1,these_vars[1]], 'b', linewidth=2)
# plt.plot(weights_proj[:,grp==2,these_vars[0]],weights_proj[:,grp==2,these_vars[1]], 'g', linewidth=2)
# plt.plot(weights_proj[:,grp==3,these_vars[0]],weights_proj[:,grp==3,these_vars[1]], 'y', linewidth=2)
# plt.plot(weights_proj[:,np.sign(W.squeeze())<0,0],weights_proj[:,np.sign(W.squeeze())<0,1], 'b', linewidth=1)
plt.axis('equal')
plt.axis('square')
plt.xlim([-down_range,up_range])
plt.ylim([-down_range,up_range])

#%% All neurons on one plot

# this_nonlin = RayLou()
# this_nonlin = TanAytch()
# this_nonlin = NoisyTanAytch(noise)
# this_nonlin = HardTanAytch(vmin=-0.5,vmax=0.5)
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
this_nonlin = NoisyRayLou(noise)

N_grid = 21

num_out = (np.sign(W)!=0).sum(0)

# these_neur = num_out>1 # plot all conjunctive-output neurons
these_neur = num_out==1 # plot all abstract-output neurons

x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
y_ = task(np.unique(inp_condition)).detach().numpy().T
err_avg = y_ - y_.mean(1,keepdims=True)

grp_weights, grp = np.unique(np.sign(W),axis=1, return_inverse=True)

# compute relevant directions
dic = assistants.Dichotomies(num_cond)
all_col = np.stack([2*dic.coloring(range(num_cond))-1 for _ in dic])

corr_dir = W.T@err_avg
# corr_dir/=la.norm(corr_dir, axis=-1, keepdims=True)
corr_dir/=np.max(corr_dir, axis=-1, keepdims=True)
lin_dir = corr_dir@x_.T
lin_dir /= la.norm(lin_dir, axis=-1, keepdims=True)

kernels = ((all_col@err_avg.T)@grp_weights==0)
bad_dirs = np.stack([all_col[k,:].T@all_col[k,:] for k in kernels.T])[grp,:,:]
anticorr_dir = 1.0*bad_dirs[np.arange(N),np.argmax(corr_dir>0,axis=1),:]
# anticorr_dir/=la.norm(anticorr_dir, axis=-1, keepdims=True)
anticorr_dir/=np.max(anticorr_dir, axis=-1, keepdims=True)
bad_dir = anticorr_dir@x_.T
bad_dir /= la.norm(bad_dir, axis=-1, keepdims=True)

neur_basis = np.stack([bad_dir,lin_dir])
neur_weights = np.einsum('ilk,jlk->jli', neur_basis, weights)

up_range = np.abs((neur_weights[:,these_neur,:]).max())*1.1 #+ 0.1
down_range = np.abs((neur_weights[:,these_neur,:]).min())*1.1 #+ 0.1
this_range = np.max([up_range, down_range])
# this_range=6.7


plt.figure()


nidx = np.where(these_neur)[0][0] # plot all conjunctive-output neurons

wawa = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*len(these_vars))

fake_W = np.stack([w.flatten() for w in wawa]).T
WW = fake_W@neur_basis[:,nidx,:] 
fake_fz = this_nonlin.deriv(torch.tensor((WW@x_))).numpy()

fake_grads = neur_basis[:,nidx,:]@(x_@(corr_dir[nidx,:]*fake_fz).T)

# corr_grad = fake_fz@((x_.T@x_)*corr_dir[nidx]**2).sum(0)
# anticorr_grad = fake_fz@((x_.T@x_)*(anticorr_dir[nidx]*corr_dir[nidx])).sum(0)
# corr_grad = (corr_dir[nidx,:]*fake_fz)@(x_.T@x_)@corr_dir[nidx]
# anticorr_grad = (corr_dir[nidx,:]*fake_fz)@(x_.T@x_)@anticorr_dir[nidx]

plt.quiver(fake_W[:,0],fake_W[:,1], fake_grads[0,:],fake_grads[1,:], color=(0.5,0.5,0.5))
# plt.quiver(fake_W[:,0],fake_W[:,1], anticorr_grad,corr_grad, color=(0.5,0.5,0.5))

dicplt.square_axis()

cols = cm.viridis(np.unique(grp)/7)
for this_grp in np.unique(grp[these_neur]):
    plt.scatter(neur_weights[0,grp==this_grp,0],neur_weights[0,grp==this_grp,1],marker='o', c=cols[this_grp])
    plt.plot(neur_weights[:,grp==this_grp,0],neur_weights[:,grp==this_grp,1], color=cols[this_grp])

#%%
# basins = list(itt.chain(*([combinations(range(num_cond), i) for i in range(1,3)])))
# basin_prototype = np.stack([x_[:,p].sum(1) for p in basins])
# which_basin = (weights[-1,...]@basin_prototype.T).argmax(1)

inp_coefs = basis.T@x_

fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios':[1,3]})
axs[0].set_xlim([1,nepoch+1])
axs[0].set_ylim([-1,1])
axs[0].semilogx()

axs[1].scatter(inp_coefs[0,:],inp_coefs[1,:], c=1-y_.squeeze()/1, cmap='bwr', marker='o', s=100)
axs[1].set_xlim([-this_range,this_range])
axs[1].set_ylim([-this_range,this_range])
dicplt.square_axis(axs[1])

an1 = dicplt.LineAnime(np.stack([np.array(list(range(20))+list(range(20,nepoch,10))) for _ in range(3)]),
                       np.array(PS)[list(range(20))+list(range(20,nepoch,10)),:].T,
                       ax=axs[0])
an2 = dicplt.LineAnime(weights_proj[:,:,0].T,weights_proj[:,:,1].T, colors=cm.bwr(grp/1), ax=axs[1])

dicplt.AnimeCollection(an1, an2).save(SAVE_DIR+'/vidya/tempmovie.mp4', fps=30)


#%% 
this_grp = 0

ax = dicplt.scatter3d(8*inp_coefs.T, s=200, marker='*', c=cm.viridis(util.decimal(y_.T)/7))
dicplt.LineAnime3D(weights_proj[:,grp==this_grp,0].T,
                   weights_proj[:,grp==this_grp,1].T,
                   weights_proj[:,grp==this_grp,2].T, 
                   view_period=100, ax=ax, 
                   colors=cm.viridis(grp[grp==this_grp]/7),
                   rotation_period=500).save(SAVE_DIR+'vidya/tempmovie_%s.mp4'%grp_weights[:,this_grp])

#%%

p_abs = 0.0

inp_align = []
out_align = []
for p_abs in np.linspace(0,1,100):
    p_conj = 1-p_abs
    
    fake_N = 1000
    
    n_abs = int(fake_N*p_abs)
    n_conj = fake_N - n_abs
    
    abs_labs = 2*tasks.StandardBinary(3)(np.arange(8)).numpy() - 1
    
    abs_weights = np.concatenate([np.eye(3),np.eye(3)*-1])[np.random.choice(6,n_abs)]
    conj_weights = abs_labs[np.random.choice([0,1,2,5,6,7],n_conj)]
    fake_weights = np.concatenate([abs_weights,conj_weights],axis=0)*10
    
    fake_rep = this_nonlin(torch.tensor(fake_weights@abs_labs.T))
    
    Kx = util.dot_product(abs_labs.T, abs_labs.T)
    Ky = util.dot_product(y_-y_.mean(1,keepdims=True), y_-y_.mean(1,keepdims=True))
    Kz = util.dot_product(fake_rep, fake_rep)
    
    inp_align.append(np.sum(Kz*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Kz*Kz)))
    out_align.append(np.sum(Kz*Ky)/np.sqrt(np.sum(Ky*Ky)*np.sum(Kz*Kz)))

c_xy = np.sum(Ky*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Ky*Ky))

cos_foo = np.linspace(c_xy,1,1000)
ub = c_xy*cos_foo + np.sqrt(1-c_xy**2)*np.sqrt(1-cos_foo**2)
plt.plot(cos_foo, ub, 'k--')
plt.scatter(inp_align, out_align, c=np.linspace(0,1,100))
plt.colorbar()

#%%
# this_nonlin = RayLou()
this_nonlin = TanAytch()
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
# this_nonlin = NoisyRayLou(noise)

grp_weights, grp = np.unique(W,axis=1, return_inverse=True)
grp_weights = grp_weights
num_grp = grp_weights.shape[-1]

weights_proj = np.einsum('ijk,kl->ijl',weights,x_pos/la.norm(x_pos,axis=0, keepdims=True)) #+ biases.mean(1)[:,None,None]
x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
# x_ = x_pos
y_ = task(np.unique(inp_condition)).detach().numpy().T

arr_scl = 1e1

N_grid = 21
# this_range = np.abs(weights_proj[0]).max() + 0.1
up_range = np.abs((weights_proj).max())*1.1 #+ 0.1
down_range = np.abs((weights_proj).min())*1.1 #+ 0.1
# this_range=0.3
this_range = np.max([up_range, down_range])
# up_range = this_range
# down_range = this_range

# this_bias = np.random.randn(N_grid**2,1)*0.1
this_bias = np.ones((N_grid**2,1))*0#b1.detach().numpy().mean()
# this_bias = b1.detach().numpy()

err_avg = y_ - y_.mean(1,keepdims=True)
# err_avg = targets.numpy().T - y_.mean(1,keepdims=True)

wawa = np.meshgrid(*(np.linspace(-this_range,up_range,N_grid),)*2)

plt.figure()
for j,these_vars in enumerate(np.split(np.arange(x_pos.shape[-1]),x_pos.shape[-1]//2)):
    
    
    fake_W = np.stack([w.flatten() for w in wawa]).T
    # fake_W = np.stack([w.flatten() for w in wawa]).T
    WW = fake_W@(x_pos/la.norm(x_pos,axis=0, keepdims=True))[:,these_vars].T

    fake_fz = this_nonlin.deriv(torch.tensor((WW@x_ + this_bias))).numpy()
    # fake_fz = this_nonlin.deriv(torch.tensor((fake_W@(x_pos/la.norm(x_pos,axis=0, keepdims=True))[:,these_vars].T@inputs.numpy().T + this_bias))).numpy()
    # fake_fz = this_nonlin.deriv(torch.tensor((fake_W@x_pos.T@x_ + this_bias))).numpy()
    
    # cols = dicplt.color_cycle('brg', num_grp)
    cols = cm.jet(np.arange(num_grp)/num_grp)
    for i in range(num_grp):
        
        plt.subplot(x_pos.shape[-1]//2,num_grp,(i+1)+j*num_grp)
        fake_grads = (x_pos/la.norm(x_pos,axis=0, keepdims=True)).T@(x_@(grp_weights[:,i]@err_avg*fake_fz).T)
        # fake_grads = (x_pos/la.norm(x_pos,axis=0, keepdims=True)).T@(inputs.numpy().T@(grp_weights[:,i]@err_avg*fake_fz).T)
        
        plt.quiver(fake_W[:,0],fake_W[:,1],
                    fake_grads[these_vars[0],:],fake_grads[these_vars[1],:],
                    # color=cols[i],
                    color=(0.5,0.5,0.5),
                    scale_units='xy',
                    scale=arr_scl)
        
        plt.plot(weights_proj[:,grp==i,these_vars[0]],weights_proj[:,grp==i,these_vars[1]], color=cols[i], linewidth=2)
        
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-down_range,up_range])
        plt.ylim([-down_range,up_range])
        
        if i>0:
            plt.yticks([])


#%%

# this_nonlin = RayLou()
this_nonlin = TanAytch() 
# this_nonlin = Iden()
# this_nonlin = Poftslus(1)
# this_nonlin = NoisyRayLou(noise)

which_vars = [[0,1],[0,2]]

grp_weights, grp = np.unique(W,axis=1, return_inverse=True)
num_grp = grp_weights.shape[-1]

weights_proj = np.einsum('ijk,kl->ijl',weights,x_pos/la.norm(x_pos,axis=0, keepdims=True)) #+ biases.mean(1)[:,None,None]
x_ = input_task(np.unique(inp_condition),noise=0).detach().numpy().T
# x_ = x_pos
y_ = task(np.unique(inp_condition)).detach().numpy().T


N_grid = 11
# this_range = np.abs(weights_proj[0]).max() + 0.1
up_range = np.abs((weights_proj).max())*1.1 #+ 0.1
down_range = np.abs((weights_proj).min())*1.1 #+ 0.1
# this_range=0.3
this_range = np.max([up_range, down_range])
# up_range = this_range
# down_range = this_range

# this_bias = np.random.randn(N_grid**2,1)*0.1
this_bias = np.ones((N_grid**2,1))*0.1#b1.detach().numpy().mean()
# this_bias = b1.detach().numpy()

err_avg = y_ - y_.mean(1,keepdims=True)
# err_avg = targets.numpy().T - y_.mean(1,keepdims=True)


wawa = np.meshgrid(*(np.linspace(-this_range,up_range,N_grid),)*2)

##
# fig, axs = plt.subplots(len(which_vars),num_grp+1,gridspec_kw={'width_ratios':[1,]*num_grp + [2,]})
# plt.margins(0)
fig = plt.figure()

ps_ax = plt.subplot(len(which_vars)+1,1,1)
ps_ax.semilogx()

cols = dicplt.color_cycle('brg', num_grp)

axs = [[[] for _ in range(num_grp)] for _ in range(len(which_vars))]
for j,these_vars in enumerate(which_vars):
    
    fake_W = np.stack([w.flatten() for w in wawa]).T
    fake_fz = this_nonlin.deriv(torch.tensor((fake_W@(x_pos/la.norm(x_pos,axis=0, keepdims=True))[:,these_vars].T@x_ + this_bias))).numpy()
    # fake_fz = this_nonlin.deriv(torch.tensor((fake_W@(x_pos/la.norm(x_pos,axis=0, keepdims=True))[:,these_vars].T@inputs.numpy().T + this_bias))).numpy()
    # fake_fz = this_nonlin.deriv(torch.tensor((fake_W@x_pos.T@x_ + this_bias))).numpy()
    
    for i in range(num_grp):
        
        axs[j][i] = plt.subplot(len(which_vars)+1,num_grp,(i+1)+(j+1)*num_grp)
        # idx = (i+1)+j*num_grp
        fake_grads = (x_pos/la.norm(x_pos,axis=0, keepdims=True)).T@(x_@(grp_weights[:,i]@err_avg*fake_fz).T)
        # fake_grads = (x_pos/la.norm(x_pos,axis=0, keepdims=True)).T@(inputs.numpy().T@(grp_weights[:,i]@err_avg*fake_fz).T)
        
        plt.quiver(fake_W[:,0],fake_W[:,1],
                    fake_grads[these_vars[0],:],fake_grads[these_vars[1],:],
                    # color=cols[i],
                    color=(0.5,0.5,0.5),
                    scale_units='xy',
                    scale=1e-2)
        
        # axs[j,i].plot(weights_proj[:,grp==i,these_vars[0]],weights_proj[:,grp==i,these_vars[1]], color=cols[i], linewidth=2)
        
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-down_range,up_range])
        plt.ylim([-down_range,up_range])
        
        if i>0:
            plt.yticks([])

def init():
    ps_plt, = ps_ax.plot([],[])
    plts = [[[] for _ in range(num_grp)] for _ in range(len(which_vars))]
    for j,these_vars in enumerate(which_vars):
            for i in range(num_grp):
                plts[j][i], = axs[j][i].plot(weights_proj[0,grp==i,these_vars[0]],
                                            weights_proj[0,grp==i,these_vars[1]], 
                                            color=cols[i], linewidth=2)

    ps_ax.set_xlim([1,len(weights_proj)+1])
    
    plt.margins(0)
    
    return fig,

# def init():
    # ax.view_init(30,0)
    # plt.draw()
    # return ax,

def update(frame):
    ps_plt.set_data(np.arange(1,frame+1),np.array(PS)[:frame,:])
    for j,these_vars in enumerate(which_vars):
        for i in range(num_grp):
            plts[j][i].set_data(weights_proj[:frame,grp==i,these_vars[0]],weights_proj[:frame,grp==i,these_vars[1]])    

    plt.draw()
    return fig,

ani = anime.FuncAnimation(fig, update, frames=np.arange(1,len(weights_proj)),
                          init_func=init, interval=10, blit=True)
# plt.show()
ani.save(SAVE_DIR+'/vidya/tempmovie.mp4', writer=anime.writers['ffmpeg'](fps=30))


