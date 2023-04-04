

CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'

from sklearn.exceptions import ConvergenceWarning
import warnings # I hate convergence warnings so much never show them to me
warnings.simplefilter("ignore", category=ConvergenceWarning)


import socket
import os
import sys
import pickle as pkl
import subprocess

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

sys.path.append(CODE_DIR)
import util
import pt_util
import tasks
import students as stud
import experiments as exp
import grammars as gram
import server_utils 
import plotting as tplt
import dichotomies as dics

#%%
def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(len(r) for r in M)
    dims = M[0].shape[1:]

    Z = np.zeros((len(M), maxlen, *dims))*np.nan
    for enu, row in enumerate(M):
        Z[enu, :len(row)] = row 
        
    return Z

#%%

exp_prm = {'experiment': exp.RandomInputMultiClass,
		   'dim_inp': 128,
		   'num_bits': ( 2, 3, 4, 5), 
		   'num_class': ( 2, 3, 4, 5),
		   'input_noise':0.1,
		   'center':[True, False]}

net_args = {'model': stud.NGroupNetwork,
			'n_per_group': 10,
			'num_k': 2,
			'p_targ': stud.Bernoulli,
			'num_init': 10,
			'inp_weight_distr': pt_util.uniform_tensor,
			'inp_bias_distr': torch.ones,
			'inp_bias_var': 0,
			'out_bias_var': 0,
			'activation':[pt_util.TanAytch(), pt_util.RayLou()]}

opt_args = {'train_outputs': False,
			'train_inp_bias': False,
			'train_out_bias':False,
			# 'init_index': list(range(5)),
			'do_rms': (False, True),
			'nepoch': 2000,
			'lr':(1e-1, 1e-2),
			'bsz':200,
			'verbose': False,
			'skip_rep_metrics':True,
			'skip_metrics':False,
			'conv_tol': 1e-10,
			'metric_period':10}


#%%

all_exp_args, prm = server_utils.get_all_experiments(exp_prm, net_args, opt_args)

all_metrics = {}
for exp_args in tqdm(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    # this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR+'results/', exp_args['opt_args'])
    
    
    
    if len(all_metrics) == 0:
        all_metrics = {k:[] for k,v in this_exp.metrics.items()}
    for k in all_metrics.keys():
        all_metrics[k].append(this_exp.metrics[k])

for k,v in all_metrics.items():
    all_metrics[k] = pad_to_dense(v)


#%% Compute input alignment 

PS = []
CCGP = []
inp_align = []
out_align = []
skew = []
for exp_args in tqdm(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    # this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR+'results/', exp_args['opt_args'])
    
    n_cond = this_exp.inputs.num_cond
    
    x_ = this_exp.inputs(np.arange(n_cond), 0).T
    y_ = this_exp.outputs(np.arange(n_cond), 0).T
    
    cond = np.random.choice(range(n_cond), 1000)
    x_noise = this_exp.inputs(cond, noise=0.3).T
    # x_noise = this_exp.inputs(cond).T
    y_noise = (this_exp.outputs(cond)).T
    
    Kx = util.dot_product(x_-x_.mean(1,keepdims=True), x_-x_.mean(1,keepdims=True))
    Ky = util.dot_product(y_-y_.mean(1,keepdims=True), y_-y_.mean(1,keepdims=True))
    
    # Kx = util.dot_product(x_noise-x_noise.mean(1,keepdims=True), x_noise-x_noise.mean(1,keepdims=True))
    # Ky = util.dot_product(y_noise-y_noise.mean(1,keepdims=True), y_noise-y_noise.mean(1,keepdims=True))
    
    skew.append( np.sum(Ky*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Ky*Ky)) )
    
    inp = []
    out = []
    # ccgp = []
    # ps = []
    for model in this_exp.models:
        
        z_ = model(x_.T)[1].detach().numpy().T
        z_noise = model(x_noise.T)[1].detach().numpy().T     
        
        Kz = util.dot_product(z_-z_.mean(1,keepdims=True), z_-z_.mean(1,keepdims=True))
        # Kz = util.dot_product(z_noise-z_noise.mean(1,keepdims=True), z_noise-z_noise.mean(1,keepdims=True))
        
        inp.append(np.sum(Kz*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Kz*Kz)))
        out.append(np.sum(Kz*Ky)/np.sqrt(np.sum(Ky*Ky)*np.sum(Kz*Kz)))

        # ps.append(dics.parallelism_score(z_, np.arange(n_cond), y_.T))
        # ps.append(dics.parallelism_score(z_noise, cond, y_noise.T))
                
        # ccgp.append(np.mean(dics.compute_ccgp(z_noise.T, cond, np.squeeze(y_noise), svm.LinearSVC(), twosided=True)))
        
    # PS.append(ps)
    # CCGP.append(ccgp)
    inp_align.append(inp)
    out_align.append(out)

inp_align = np.array(inp_align)
out_align = np.array( out_align)
CCGP = np.squeeze(CCGP)
PS = np.array(PS)
skew = np.array(skew)

#%%

for j in [0.0,0.1,0.2,0.3]:
    filt = (prm['out_weight_distr']=='uniform_tensor')&(prm['activation']=='RayLou')
    mn  = [np.nanmean(all_metrics['parallelism'][(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...], axis=(0,2))[-1,0] for i in np.linspace(0,1,21)]
    err  = [np.nanstd(all_metrics['parallelism'][(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...], axis=(0,2))[-1,0] for i in np.linspace(0,1,21)]
    # mn  = [np.nanmean(all_metrics['ccgp'][(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...], axis=(0,2))[-1,0] for i in np.linspace(0,1,21)]
    # err  = [np.nanstd(all_metrics['ccgp'][(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...], axis=(0,2))[-1,0] for i in np.linspace(0,1,21)]
    # plt.plot(np.linspace(0,1,21), mn)
    plt.errorbar(np.linspace(0,1,21), mn, yerr=err, marker='.')

plt.ylabel('Parallelism (trained)')
plt.xlabel('Linear separability (epsilon)')

#%%

for j in [0.0,0.1,0.2,0.3]:
    filt = (prm['out_weight_distr']=='uniform_tensor')&(prm['activation']=='TanAytch')
    
    # mn  = [np.mean(out_align[(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...]) for i in np.linspace(0,1,21)]
    # err  = [np.mean(out_align[(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...],0).std() for i in np.linspace(0,1,21)]
    # mn  = [np.mean(inp_align[(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...]) for i in np.linspace(0,1,21)]
    # err  = [np.mean(inp_align[(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...],0).std() for i in np.linspace(0,1,21)]
    # mn  = [np.mean(CCGP[(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...]) for i in np.linspace(0,1,21)]
    # err  = [np.mean(CCGP[(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...],0).std() for i in np.linspace(0,1,21)]
    # mn  = [np.mean(PS[(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...]) for i in np.linspace(0,1,21)]
    # err = [np.mean(PS[(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...],0).std() for i in np.linspace(0,1,21)]
    
    mn  = [np.mean(correct_align[1,(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...]) for i in np.linspace(0,1,21)]
    err  = [np.mean(correct_align[1,(prm['input_noise']==j)&(prm['epsilon']==i)&filt, ...],0).std() for i in np.linspace(0,1,21)]
    
    # plt.plot(np.linspace(0,1,21), mn)
    plt.errorbar(np.linspace(0,1,21), mn, yerr=err, marker='.')

# plt.ylabel('Parallelism (trained)')
# plt.ylabel('CCGP')
plt.ylabel('Output alignment')
plt.xlabel('Linear separability (epsilon)')
plt.legend([0.0, 0.1, 0.2, 0.3],title='Train noise')

#%%

cos_foo = np.linspace(0,1,1000)

ub = np.sqrt(1-cos_foo**2)

phi = (np.pi/2 -np.arccos(skew))/2  # re-align it with the orthogonal case
basis = np.array([[np.cos(phi),np.cos(np.pi/2-phi)],[np.sin(phi),np.sin(np.pi/2-phi)]])
# rot = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]]).transpose((2,0,1))
# correction = np.einsum('...ik,...kj->...ij', rot, np.linalg.inv(basis.transpose((2,0,1))))
correction = np.linalg.inv(basis.transpose((2,0,1)))

# correct_align = (correction)@np.stack([inp_align,out_align])
correct_align = np.einsum('lik,klj->ilj', correction, np.stack([inp_align,out_align]))
# correct_align = np.stack([inp_align,out_align])
correct_bound = np.stack([cos_foo, ub])

#%%

# filt = (prm['activation']=='RayLou')
filt = (prm['activation']=='TanAytch')

plt.plot(correct_bound[0,:],correct_bound[1,:], 'k--')

# for j in [0.0,0.1,0.2,0.3]:
# for j in range(2,6):

    
mn  = np.array([np.mean(correct_align[:,(prm['num_bits']==i)&filt], axis=(1,2)) for i in range(2,6)])
err = np.array([np.mean(correct_align[:,(prm['num_bits']==i)&filt], 1).std(1) for i in range(2,6)])

plt.errorbar(mn[:,0],mn[:,1], xerr=err[:,0],yerr=err[:,1], linestyle='', marker='o')

tplt.square_axis()

# plt.scatter(correct_align[0,...], correct_align[1,...])

#%%

exp_prm = {'experiment': exp.WeightDynamics,
		   'inp_task': (tasks.RandomPatterns(4, 200, noise_var=1.0), 
		   				tasks.NudgedXOR(40, nudge_mag=0.00, noise_var=1.0),
		   				tasks.NudgedXOR(40, nudge_mag=0.5, noise_var=1.0),
		   				tasks.NudgedXOR(40, nudge_mag=1.0, noise_var=1.0),
		   				tasks.RandomPatterns(4, 200, noise_var=1.0)),
		   'out_task': (tasks.RandomDichotomies(d=[(0,3)]),
		   				tasks.RandomDichotomies(d=[(0,3)]),
		   				tasks.RandomDichotomies(d=[(0,3)]),
		   				tasks.RandomDichotomies(d=[(0,3)]),
		   				tasks.RandomDichotomies(d=[(0,1), (0,2)]))}

net_args = {'model': stud.ShallowNetwork,
 			'width': 120,
 			'p_targ': stud.Bernoulli,
 			'num_init': 1,
 			'inp_weight_distr': pt_util.uniform_tensor,
 			'out_weight_distr': pt_util.BalancedBinary(2,1,normalize=True),
 			'activation':[pt_util.TanAytch(), pt_util.RayLou()]}

opt_args = {'train_outputs': False,
 			# 'init_index': list(range(5)),
 			'do_rms': False,
 			'nepoch': 5000,
 			'lr':1e-1,
 			'bsz':200,
 			'verbose': False,
 			'skip_rep_metrics':True,
 			'skip_metrics':False,
 			'conv_tol': 1e-10,
 			'metric_period':10}



#%%
def task_basis(W, y):
    
    num_cond = y.shape[-1]
    
    grp_weights, grp = np.unique(np.sign(W),axis=1, return_inverse=True)

    # compute relevant directions
    dic = dics.Dichotomies(num_cond)
    all_col = np.stack([2*dic.coloring(range(num_cond))-1 for _ in dic])
    
    corr_dir = W.T@y
    # corr_dir/=la.norm(corr_dir, axis=-1, keepdims=True)
    corr_dir /= np.max(corr_dir, axis=-1, keepdims=True)
    # lin_dir = corr_dir@x.T
    # lin_dir /= la.norm(lin_dir, axis=-1, keepdims=True)
    
    kernels = ((all_col@y.T)@grp_weights==0)
    bad_dirs = np.stack([all_col[k,:].T@all_col[k,:] for k in kernels.T])[grp,:,:]
    anticorr_dir = 1.0*bad_dirs[np.arange(len(grp)),np.argmax(corr_dir>0,axis=1),:]
    # anticorr_dir/=la.norm(anticorr_dir, axis=-1, keepdims=True)
    anticorr_dir /= np.max(anticorr_dir, axis=-1, keepdims=True)
    # bad_dir = anticorr_dir@x.T
    # bad_dir /= la.norm(bad_dir, axis=-1, keepdims=True)
    
    return np.stack([anticorr_dir,corr_dir])


N_grid = 21


all_exp_args, prm = server_utils.get_all_experiments(exp_prm, net_args, opt_args)

all_metrics = {}
for i,exp_args in enumerate(all_exp_args):

    this_exp = exp_args['exp_prm']['experiment'](**exp_args['exp_prm']['exp_args'])
    this_exp.models = this_exp.initialize_network(exp_args['net_args']['model'], **exp_args['net_args']['model_args'])
    # this_exp.initialize_experiment( **exp_args['opt_args'])
    this_exp.load_experiment(SAVE_DIR+'results/', exp_args['opt_args'])
    
    
    net = this_exp.models[0]
    
    if prm['activation'][i] == 'TanAytch':
        this_nonlin = pt_util.NoisyTanAytch(this_exp.inputs.noise_var)
    else:
        this_nonlin = pt_util.NoisyRayLou(this_exp.inputs.noise_var)
        
    W = net.J.detach().numpy()

    num_out = (np.sign(W)!=0).sum(0)
    
    # these_neur = num_out>1 # plot all conjunctive-output neurons
    # these_neur = num_out==1 # plot all abstract-output neurons
    these_neur = num_out > -np.inf
    
    # these_groups = np.unique(grp[these_neur])

    x_ = this_exp.inputs(range(4),noise=0).detach().numpy().T
    y_ = this_exp.outputs(range(4), noise=0).detach().numpy().T
    err_avg = y_ - y_.mean(1,keepdims=True)
    
    # x_ /= la.norm(x_,axis=0)
    
    grp_weights, grp = np.unique(np.sign(W),axis=1, return_inverse=True)
    this_grp = np.unique(grp)[-1]
    b = np.unique(net.b1.detach().numpy())[0]

    basis = task_basis(W, err_avg)
    corr_dir = basis[1]
    
    neur_basis = basis@x_.T
    # neur_basis = basis / la.norm(basis, axis=-1, keepdims=True)
    # neur_basis = basis/2
    
    # weights = np.squeeze(this_exp.metrics['weights_proj'])@(x_/la.norm(x_, axis=0)).T
    # neur_weights = np.einsum('ilk,jlk->jli', neur_basis, weights)
    neur_weights = np.einsum('thi,jhi->jht', basis, np.squeeze(this_exp.metrics['weights_proj']))
    neur_weights /= (la.norm(neur_basis, axis=-1)+1e-6).T
    neur_basis /= (la.norm(neur_basis, axis=-1, keepdims=True)+1e-6)
    
    up_range = np.abs((neur_weights).max())*1.1 #+ 0.1
    down_range = np.abs((neur_weights).min())*1.1 #+ 0.1
    this_range = np.max([up_range, down_range])
    # this_range=6.7
    
    plt.subplot(2, 5, np.mod(i,2)*5 + i//2 + 1)
    # plt.title(f"{prm['inp_task'][i]}_{prm['out_task'][i]}")
    
    nidx = np.where(grp==this_grp)[0][0] # plot all conjunctive-output neurons
    
    wawa = np.meshgrid(*(np.linspace(-this_range,this_range,N_grid),)*2)
    
    fake_W = np.stack([w.flatten() for w in wawa]).T
    WW = fake_W@neur_basis[:,nidx,:] 
    fake_fz = this_nonlin.deriv(torch.tensor(WW@x_ + b)).numpy()
    
    fake_grads = neur_basis[:,nidx,:]@(x_@(corr_dir[nidx,:]*fake_fz).T)
        
    # corr_grad = fake_fz@((x_.T@x_)*corr_dir[nidx]**2).sum(0)
    # anticorr_grad = fake_fz@((x_.T@x_)*(anticorr_dir[nidx]*corr_dir[nidx])).sum(0)
    # corr_grad = (corr_dir[nidx,:]*fake_fz)@(x_.T@x_)@corr_dir[nidx]
    # anticorr_grad = (corr_dir[nidx,:]*fake_fz)@(x_.T@x_)@anticorr_dir[nidx]
    
    plt.quiver(fake_W[:,0],fake_W[:,1], fake_grads[0,:],fake_grads[1,:], color=(0.5,0.5,0.5))
    # plt.quiver(fake_W[:,0],fake_W[:,1], anticorr_grad,corr_grad, color=(0.5,0.5,0.5))
    
    tplt.square_axis()
    
    # cols = cm.viridis(these_groups/np.max(these_groups))
    
    # for j,this_grp in enumerate(these_groups):
    plt.scatter(neur_weights[0,grp==this_grp,0],neur_weights[0,grp==this_grp,1],marker='o', c='r')
    plt.plot(neur_weights[:,grp==this_grp,0],neur_weights[:,grp==this_grp,1], color='r')
    

#%%


for i in len(models):
    
    net = models[i][0]
    
    

