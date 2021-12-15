CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as anime
from itertools import permutations, combinations
import itertools as itt
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

import students
import assistants
import experiments as exp
import util
import tasks
import plotting as dicplt

#%% Model specification -- for loading purposes
# mse_loss = True
mse_loss = False
# categorical = True
categorical = False

num_cond = 8
num_var = 3

# use_mnist = True
use_mnist = False

# which_task = 'mnist'
# which_task = 'mog'
which_task = 'structured'

# # task = util.ParityMagnitude()
# if categorical:
#     task = util.RandomDichotomiesCategorical(num_cond,num_var,0, mse_loss)
# else:
#     task = util.RandomDichotomies(num_cond,num_var,0, mse_loss)
# # task = util.ParityMagnitudeEnumerated()
# # task = util.Digits()
# # task = util.DigitsBitwise()

latent_dist = None
# latent_dist = GausId
# nonlinearity = 'ReLU'
nonlinearity = 'Tanh'
# nonlinearity = 'LeakyReLU'

num_layer = 0
# num_layer = 10

# good_start = True
good_start = False

# coding_level = 1.0
coding_level = None

rotation = 1.0
# rotation = np.linspace(0.0,2.0,11)

decay = 0.0

H = 100
# N_list = None # set to None if you want to automatically discover which N have been tested
# N_list = [2,3,4,5,6,7,8,9,10,11,20,25,50,100]
# N_list = None
# N_list = [2,3,5,10,50,100]
N_list = [100]

# readout_weights = students.LinearRandomSphere(radius=0.2, eps=0.05, 
#                                               fix_weights=True,
#                                               nonlinearity=task.link)
# readout_weights = students.LinearRandomNormal(var=0.2, 
#                                               fix_weights=True, 
#                                               nonlinearity=task.link)
# readout_weights = students.LinearRandomProportional(scale=0.2, 
#                                                     fix_weights=True, 
#                                                     coef=2,
#                                                     nonlinearity=task.link)

readout_weights = None
# readout_weights = students.BinaryReadout
# readout_weights = students.PositiveReadout

# find experiments 
if which_task == 'mnist': 
    this_exp = exp.mnist_multiclass(task, SAVE_DIR, 
                                    z_prior=latent_dist,
                                    num_layer=num_layer,
                                    weight_decay=decay,
                                    decoder=readout_weights,
                                    nonlinearity=nonlinearity,
                                    good_start=good_start,
                                    init_coding=coding_level)
elif which_task == 'mog':
    task = util.RandomDichotomies(num_cond,num_var)
    this_exp = exp.random_patterns(task, SAVE_DIR, 
                                    num_class=num_cond,
                                    dim=100,
                                    var_means=1,
                                    z_prior=latent_dist,
                                    num_layer=num_layer,
                                    weight_decay=decay,
                                    decoder=readout_weights,
                                    nonlinearity=nonlinearity,
                                    good_start=good_start,
                                    init_coding=coding_level,
                                    rot=rotation)
elif which_task == 'structured':
    inp_task = tasks.EmbeddedCube(tasks.StandardBinary(int(np.log2(num_cond))),100,noise_var=0.1)
    # inp_task = tasks.TwistedCube(tasks.StandardBinary(2), 100, f=rotation, noise_var=0.1)
    # inp_task = tasks.NudgedXOR(tasks.StandardBinary(2), 100, nudge_mag=rotation, noise_var=0.1, random=True)
    # task = tasks.LogicalFunctions(d=decs, function_class=num_var)
    task = tasks.RandomDichotomies(d=[(0,1,3,5),(0,2,3,6),(0,1,2,4)])
    # task = tasks.RandomDichotomies(d=[(0,3)])
    this_exp = exp.structured_inputs(task, input_task=inp_task,
                                      SAVE_DIR=SAVE_DIR,
                                      noise_var=0.1,
                                      num_layer=num_layer,
                                      z_prior=latent_dist,
                                      weight_decay=decay,
                                      decoder=readout_weights,
                                      nonlinearity=nonlinearity)
    num_var = task.num_var + inp_task.num_var


# for rot in rotation:
    # this_exp.param = rot
    
this_folder = SAVE_DIR + this_exp.folder_hierarchy()
    
all_nets, mets, all_args = this_exp.aggregate_nets(SAVE_DIR, N_list)

#%%

n_id = 0
netid = 0 # which specific experiment to use

model = all_nets[netid]
params = all_args[netid]
N = N_list[netid]

this_exp.load_other_info(params[n_id])
this_exp.load_data(SAVE_DIR)

test_dat = this_exp.test_data
train_dat = this_exp.train_data

#%% General metrics (always computed)
# show_me = 'train_loss'
# show_me = 'train_perf' 
show_me = 'test_perf'
# show_me = 'mean_grad'
# show_me = 'std_grad'
show_me = 'linear_dim'
# show_me = 'sparsity'

has_val = np.array([m for m in mets[n_id][show_me] if len(m)>0])
# shp = np.max([np.max(h.shape) ])

epochs = np.arange(1,has_val.shape[1]+1)

mean = np.nanmean(has_val,0)
error = np.nanstd(has_val,0)#/np.sqrt(mets[netid][show_me].shape[0]))

if len(mean.shape)>1:
    for dim in range(mean.shape[-1]):
        pls = mean[...,dim]+error[...,dim]
        mns = mean[...,dim]-error[...,dim]
        plt.plot(epochs, mean[...,dim])
        plt.fill_between(epochs, mns, pls, alpha=0.5)
        plt.semilogx()
else:
    plt.plot(epochs, mean)
    plt.fill_between(epochs, mean-error, mean+error, alpha=0.5)
    plt.semilogx()

plt.xlabel('epoch', fontsize=15)
plt.ylabel(show_me, fontsize=15)
plt.title('N=%d'%N)

# %% Abstraction metrics
show_me = 'test_PS'
# show_me = 'shattering'
# show_me = 'test_cacgp'

has_val = np.array([m for m in mets[n_id][show_me] if len(m)>0])
# shp = np.max([np.max(h.shape) ])

epochs = np.arange(1,has_val.shape[1]+1)

mean = np.nanmean(has_val,0)
error = np.nanstd(has_val,0)#/np.sqrt(mets[netid][show_me].shape[0]))

trn = []
for dim in range(num_var):
    pls = mean[...,dim]+error[...,dim]
    mns = mean[...,dim]-error[...,dim]
    thisone = plt.plot(epochs, mean[...,dim])[0]
    trn.append(thisone)
    plt.fill_between(epochs, mns, pls, alpha=0.5)
    plt.semilogx()

# remainder
if num_var < mean.shape[-1]:
    pls = mean[...,num_var:].mean(1)+error[...,num_var:].mean(1)
    mns = mean[...,num_var:].mean(1)-error[...,num_var:].mean(1)
    untrn = plt.plot(epochs, mean[...,num_var:].mean(1))[0]
    plt.fill_between(epochs, mns, pls, alpha=0.5)
    plt.semilogx()

plt.xlabel('epoch', fontsize=15)
plt.ylabel(show_me, fontsize=15)
plt.title('N=%d'%N)
# plt.legend(trn + [untrn], ['Var %d'%(n+1) for n in range(num_var)] + (num_var < mean.shape[-1])*['Non-output dichotomies'])


#%% Shattering dimension
# idx = []
# shat = np.zeros(mets[netid]['shattering'].shape)
# baba = np.arange(35)[None,None,:]*np.ones((30,1000,1))
real_shat = np.array([m for m in mets[n_id]['shattering'] if len(m)>0])
shat_args = [a for m,a in zip(mets[n_id]['shattering'], all_args[n_id]) if len(m)>0]
shat = np.zeros(real_shat.shape)

for i,arg in enumerate(shat_args):
    xor = np.where(~(np.isin(range(8), arg['dichotomies'][0])^np.isin(range(8), arg['dichotomies'][1])))[0]
    
    ba = []
    for d in arg['dichotomies']+[xor]:
        ba.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(8),p))==list(d))\
                      for p in assistants.Dichotomies(8,arg['dichotomies'],extra=50)])[0][0])
    idx = np.concatenate([ba, np.setdiff1d(range(35),ba)])
    shat[i,:,:] = real_shat[i,:,idx].T
    # shat[i,:,:] = baba[i,:,idx].T
# idx = np.array(idx)

mean = np.nanmean(shat,0)
err = np.nanstd(shat,0)

plt.plot(epochs,mean[:,:2].mean(1))
plt.plot(epochs,mean[:,3:4].mean(1))
plt.plot(epochs,mean[:,4:].mean(1))

plt.fill_between(epochs,mean[:,:2].mean(1)-mean[:,:2].std(1),mean[:,:2].mean(1)+mean[:,:2].std(1),
                 alpha=0.5)
plt.fill_between(epochs,mean[:,3:4].mean(1)-mean[:,3:4].std(1),mean[:,3:4].mean(1)+mean[:,3:4].std(1),
                 alpha=0.5)
plt.fill_between(epochs,mean[:,4:].mean(1)-mean[:,4:].std(1),
                 mean[:,4:].mean(1)+mean[:,4:].std(1),
                 alpha=0.5)

plt.semilogx()
plt.ylabel('Shattering dimension')
plt.legend(['Trained','XOR','Untrained'])

#%% compute dichotomy metrics
# skip_dichs = False
skip_dichs = True

all_PS = []
all_CCGP = []
all_spec = []
CCGP_out_corr = []
mut_inf = []
all_SD = []
indep = []
coding_vectors = []
dic_pos = []
kern = []
distances = []
for model, args in zip(all_nets[0], all_args[0]):
    # rep = model.enc.network
    # rep = model.enc.network[:2]
    
    this_exp.load_other_info(args)
    this_exp.load_data(SAVE_DIR)
    
    fake_task = tasks.RandomDichotomies(d=this_exp.task.positives)

    indep.append(fake_task.subspace_information())
    
    layer_PS = []
    layer_CCGP = []
    layer_SD = []
    layer_spec = []
    layer_kern = []
    layer_dist = []
    for i in range(num_layer+1):
        
        rep = model.enc.network[:2*(i+1)]
        
        z = rep(this_exp.train_data[0]).detach().numpy()
        # z = this_exp.train_data[0].detach().numpy()
        # z = linreg.predict(this_exp.train_data[0])@W1.T
        n_compute = np.min([5000, z.shape[0]])
        
        idx = np.random.choice(z.shape[0], n_compute, replace=False)
        # idx_tst = idx[::4] # save 1/4 for test set
        # idx_trn = np.setdiff1d(idx, idx_tst)
        
        cond = this_exp.train_conditions[idx]
        # cond = util.decimal(this_exp.train_data[1][idx,...])
        num_cond = len(np.unique(cond))
        
        z_ = np.stack([z[this_exp.train_conditions==i,:].mean(0) for i in np.unique(cond)]).T

        layer_kern.append(util.dot_product(z_-z_.mean(1,keepdims=True), z_-z_.mean(1,keepdims=True)))
        layer_dist.append(la.norm(z_[:,:,None] - z_[:,None,:], axis=0))
        
        # xor = np.where(~(np.isin(range(num_cond), args['dichotomies'][0])^np.isin(range(num_cond), args['dichotomies'][1])))[0]
        ## Loop over dichotomies
        # D = assistants.Dichotomies(num_cond, args['dichotomies']+[xor], extra=50)
        
        if not skip_dichs:
            # choose dichotomies to have a particular order
            task_dics = []
            for d in task.positives:
                if 0 in d:
                    task_dics.append(d)
                else:
                    task_dics.append(list(np.setdiff1d(range(num_cond),d)))
            # if num_cond>8:
            Q = len(task_dics)
            D_fake = assistants.Dichotomies(num_cond, task_dics, extra=50)
            mi = np.array([fake_task.information(p) for p in D_fake])
            midx = np.append(range(Q),np.flip(np.argsort(mi[Q:]))+Q)
            # these_dics = args['dichotomies'] + [D_fake.combs[i] for i in midx]
            D = assistants.Dichotomies(num_cond, [D_fake.combs[i] for i in midx], extra=0)
            # else:
                # Q = len(args['dichotomies'])
                # D = assistants.Dichotomies(num_cond)
                # mi = np.array([fake_task.information(p) for p in D])
                # midx = np.arange(D.ntot, dtype=int)
            
            clf = assistants.LinearDecoder(z.shape[1], 1, assistants.MeanClassifier)
            gclf = assistants.LinearDecoder(z.shape[1], 1, svm.LinearSVC)
            dclf = assistants.LinearDecoder(z.shape[1], D.ntot, svm.LinearSVC)
            # clf = LinearDecoder(this_exp.dim_input, 1, MeanClassifier)
            # gclf = LinearDecoder(this_exp.dim_input, 1, svm.LinearSVC)
            # dclf = LinearDecoder(this_exp.dim_input, D.ntot, svm.LinearSVC)
            
            # K = int(num_cond/2) - 1 # use all but one pairing
            K = int(num_cond/4) # use half the pairings
            
            PS = np.zeros(D.ntot)
            CCGP = [] #np.zeros((D.ntot, 100))
            out_corr = []
            spec = []
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
                
                CCGP.append(D.CCGP(z[idx,:], cond, gclf, cntxt, twosided=True, max_iter=200))
                
                spec.append(util.projected_variance(z[idx,:].T, D.coloring(cond), only_var=True))
                
                # shattering
                d[:,i] = D.coloring(cond)
                
            # dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
            dclf.fit(z[idx,:], d, max_iter=500)
            
            z = rep(this_exp.test_data[0]).detach().numpy()
            # z = this_exp.test_data[0].detach().numpy()
            # z = linreg.predict(this_exp.test_data[0])@W1.T
            idx = np.random.choice(z.shape[0], np.min([5000, z.shape[0]]), replace=False)
            
            d_tst = np.array([D.coloring(this_exp.test_conditions[idx]) for _ in D]).T
            SD = dclf.test(z[idx,:], d_tst).squeeze()
        
            layer_PS.append(PS)
            layer_CCGP.append(CCGP)
            layer_SD.append(SD)
            layer_spec.append(spec)
    
    all_PS.append(layer_PS)
    all_CCGP.append(layer_CCGP)
    all_spec.append(layer_spec)
    all_SD.append(layer_SD)
    kern.append(layer_kern)
    distances.append(layer_dist)
    if not skip_dichs:
        CCGP_out_corr.append(out_corr)
        mut_inf.append(mi[midx])
        coding_vectors.append(dclf.coefs[midx,...])
        dic_pos.append(pos_conds)

if not skip_dichs:
    R = np.repeat(np.array(CCGP_out_corr),2,-1)
    basis_dependence = np.array(indep).max(1)
    out_MI = np.array(mut_inf)

#%% plot PS and CCGP
# take_mean = False
take_mean = True

# mask = (R.max(2)==1) # context must be an output variable   
# mask = (np.abs(R).sum(2)==0) # context is uncorrelated with either output variable
# mask = (np.abs(R).sum(2)>0) # context is correlated with at least one output variable
mask = ~np.isnan(R).max(2) # context is uncorrelated with the tested variable

almost_all_CCGP = util.group_mean(np.squeeze(all_CCGP), mask)

if take_mean:
    
    # include = (mask.sum(-1)>0)
    # np.sum(np.array(all_CCGP)*mask[...,None],2).squeeze()/mask.sum(-1)
    
    # PS = np.array(all_PS).mean(0)
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
    if 'bits' in dir(task):
        input_dics = []
        for d in task.bits:
            input_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                      for p in pos_conds])[0][0])
    else:
        input_dics = None

    dicplt.dichotomy_plot(PS, CCGP, SD, PS_err=PS_err, CCGP_err=CCGP_err, SD_err=SD_err,
                          input_dics=input_dics, output_dics=output_dics, 
                           out_MI=out_MI.mean(0))

else:
    nrow = int(np.sqrt(len(all_PS)))
    ncol = len(all_PS)//nrow
    
    plt.figure()
    for i, (PS, CCGP, SD) in enumerate(zip(all_PS, almost_all_CCGP, all_SD)):
        output_dics = []
        for d in task.positives:
            output_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                                        for p in dic_pos[i]])[0][0])
        
        if 'bits' in dir(task):
            input_dics = []
            for d in task.bits:
                input_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                          for p in dic_pos[i]])[0][0])
        else:
            input_dics = None
        
        plt.subplot(nrow, ncol, i+1)
        dicplt.dichotomy_plot(PS, CCGP, SD,
                              input_dics=input_dics,
                              output_dics=output_dics,
                              out_MI=out_MI[i], 
                              include_legend=(i==0), 
                              include_cbar=False,
                              s=10)
        if np.mod(i,ncol)>0:
            plt.yticks([])
        plt.ylabel('')
        if (i+1)//nrow < nrow:
            plt.xlabel('')
            plt.xticks([])

#%% Kernel alignment
args = all_args[0][0]

this_exp.load_other_info(args)
this_exp.load_data(SAVE_DIR)

x_ = np.stack([this_exp.train_data[0][this_exp.train_conditions==i,:].mean(0).detach().numpy() for i in np.unique(this_exp.train_conditions)]).T
x_latent = this_exp.input_task.latent_task(np.arange(4)).T
y_ = np.stack([this_exp.train_data[1][this_exp.train_conditions==i,:].mean(0).detach().numpy() for i in np.unique(this_exp.train_conditions)]).T

dx = la.norm(x_[:,:,None] - x_[:,None,:], axis=0)/2
dy = la.norm(y_[:,:,None] - y_[:,None,:], axis=0)

Kx = util.dot_product(x_-x_.mean(1,keepdims=True), x_-x_.mean(1,keepdims=True))
# Kx = util.dot_product(x_latent-x_latent.mean(1,keepdims=True), x_latent-x_latent.mean(1,keepdims=True))
Ky = util.dot_product(y_-y_.mean(1,keepdims=True), y_-y_.mean(1,keepdims=True))
Kz = np.array(kern)

inp_align = (np.einsum('kij,ij->k',Kz.mean(0),Kx)/la.norm(Kz.mean(0),'fro',axis=(-2,-1))/np.sqrt(np.sum(Kx*Kx)))
out_align = (np.einsum('kij,ij->k',Kz.mean(0),Ky)/la.norm(Kz.mean(0),'fro',axis=(-2,-1))/np.sqrt(np.sum(Ky*Ky)))
# inp_align = (np.einsum('lkij,ij->lk',Kz,Kx)/la.norm(Kz,'fro',axis=(-2,-1))/np.sqrt(np.sum(Kx*Kx)))
# out_align = (np.einsum('lkij,ij->lk',Kz,Ky)/la.norm(Kz,'fro',axis=(-2,-1))/np.sqrt(np.sum(Ky*Ky)))


inp_dcorr = util.distance_correlation(dist_x=np.array(distances).mean(0), dist_y=dx[None,...])
out_dcorr = util.distance_correlation(dist_x=np.array(distances).mean(0), dist_y=dy[None,...])
# inp_dcorr = util.distance_correlation(dist_x=np.array(distances), dist_y=dx[None,...])
# out_dcorr = util.distance_correlation(dist_x=np.array(distances), dist_y=dy[None,...])

#%%

apply_correction = False
# apply_correction = True

c_xy = np.sum(Ky*Kx)/np.sqrt(np.sum(Kx*Kx)*np.sum(Ky*Ky))

if apply_correction:
    cos_foo = np.linspace(c_xy,1,1000)
    
    ub = c_xy*cos_foo - np.sqrt(1-c_xy**2)*np.sqrt(1-cos_foo**2)
    
    phi = (np.pi/2 -np.arccos(c_xy))/2  # re-align it with the orthogonal case
    basis = np.array([[np.cos(phi),np.cos(np.pi/2-phi)],[np.sin(phi),np.sin(np.pi/2-phi)]])
    rot = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    correction = rot@la.inv(basis)
    
    correct_align = (correction)@np.stack([inp_align,out_align])
    correct_bound = (correction)@np.stack([cos_foo, ub])
    
    plt.plot(correct_bound[0,:],correct_bound[1,:], 'k--')
    plt.scatter(correct_align[0,...], correct_align[1,...], c=np.arange(num_layer+1))
else:
    cos_foo = np.linspace(c_xy,1,1000)
    ub = c_xy*cos_foo + np.sqrt(1-c_xy**2)*np.sqrt(1-cos_foo**2)
    plt.plot((cos_foo-c_xy)/(1-c_xy), (ub-c_xy)/(1-c_xy), 'k--')
    # plt.scatter(inp_align, out_align, c=np.arange(num_layer+1))
    plt.scatter((inp_align-c_xy)/(1-c_xy), (out_align-c_xy)/(1-c_xy))

plt.axis('equal')
plt.axis('square')

#%% ALL dichotomies, even the unbalanced ones
# measure_individuals = True
measure_individuals = False

unq_conds = np.unique(this_exp.train_conditions)
k = len(unq_conds)//2

colorings = []
for pos_conds in itt.chain(*([combinations(unq_conds, i) for i in range(1,k)] + [assistants.Dichotomies(k*2)])):
    
    colorings.append(np.isin(unq_conds, pos_conds))

all_pos = np.nonzero(colorings)
all_pos = np.split(all_pos[1],np.unique(all_pos[0], return_index=True)[1])[1:]

# dclf = assistants.LinearDecoder(N, len(colorings), svm.LinearSVC)
nclf = assistants.LinearDecoder(1, len(colorings), linear_model.LogisticRegression) # neuron classifier

SD_unbalanced = []
clf_weights = []
single_neurs = []
factorization = []
specialization = []
diagon = []
for model, args in tqdm(zip(all_nets[0], all_args[0])):
    
    this_exp.load_other_info(args)
    this_exp.load_data(SAVE_DIR)
    
    fake_task = util.RandomDichotomies(d=this_exp.task.positives)

    z = model(this_exp.train_data[0])[2].detach().numpy()
    # z = this_exp.train_data[0].detach().numpy()
    # z = linreg.predict(this_exp.train_data[0])@W1.T
    n_compute = np.min([5000, z.shape[0]])
    
    idx = np.random.choice(z.shape[0], n_compute, replace=False)
    neur_idx = (np.arange(z.shape[1])[None,:]*np.ones((len(idx),1))).astype(int)
    
    dclf = assistants.LinearDecoder(z.shape[1], len(colorings), linear_model.LogisticRegression)
    
    # idx_tst = idx[::4] # save 1/4 for test set
    # idx_trn = np.setdiff1d(idx, idx_tst)
    
    cond = this_exp.train_conditions[idx]
    # cond = util.decimal(this_exp.train_data[1][idx,...])
    # num_cond = len(np.unique(cond))
    
    # CCGP = []
    # for i,c in tqdm(enumerate(colorings)):
    #     pos_conds = np.nonzero(c)[0]
    #     neg_conds = np.nonzero(~c)[0]
        
        # trn_set_size = len(pos_conds)//2
        
        # CCGP
        # ccg = []
        # for trn_cond in combinations(neg_conds, trn_set_size):
        #     is_trn = np.isin(cond, trn_cond)
    
    color = np.array([np.isin(cond, p) for p in all_pos])
    
    factorization.append([util.projected_variance(z[idx,:].T, X) for X in color])
    specialization.append([util.projected_variance(z[idx,:].T, X, only_var=True) for X in color])
    diagon.append([util.diagonality(z[idx,:].T, X, cutoff=0.9) for X in color])
    
    dclf.fit(z[idx,:], color.T, max_iter=500, class_weight='balanced')
    if measure_individuals:
        nclf.fit(z[idx,:,None], np.repeat(color.T[:,None,:],N,axis=1), 
                 t_=neur_idx, max_iter=500, class_weight='balanced')
    
    z = model(this_exp.test_data[0])[2].detach().numpy()
    # z = this_exp.test_data[0].detach().numpy()
    # z = linreg.predict(this_exp.test_data[0])@W1.T
    idx = np.random.choice(z.shape[0], np.min([5000, z.shape[0]]), replace=False)
    neur_idx = (np.arange(z.shape[1])[None,:]*np.ones((len(idx),1))).astype(int)
    
    color = np.array([np.isin(this_exp.test_conditions[idx], p) for p in all_pos])
    SD_unbalanced.append(dclf.test(z[idx,:], color.T).squeeze())
    if measure_individuals:
        single_neurs.append(nclf.test(z[idx,:,None], np.repeat(color.T[:,None,:],N,axis=1), t_=neur_idx))
    
    clf_weights.append(dclf.coefs.squeeze())

clf_weights = np.array(clf_weights)
single_neurs = np.array(single_neurs)

#%%
# take_mean = False
take_mean = True

if take_mean:
    
    # include = (mask.sum(-1)>0)
    # np.sum(np.array(all_CCGP)*mask[...,None],2).squeeze()/mask.sum(-1)
    
    # PS = np.array(all_PS).mean(0)
    fac = np.array(factorization).mean(0)
    spec = np.array(specialization).mean(0)
    SD = np.array(SD_unbalanced).mean(0)
    # SD = np.array(all_SD).mean(0)
    
    ndic = len(fac)
    
    fac_err = np.nanstd(np.array(factorization), axis=0)#/np.sqrt(len(all_PS))
    spec_err = np.nanstd(specialization, axis=0)#/np.sqrt(len(all_CCGP))
    SD_err = np.nanstd(np.array(SD_unbalanced), axis=0)#/np.sqrt(len(all_SD))
    
    output_dics = []
    for d in task.positives:
        output_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                                    for p in all_pos])[0][0])
    if 'bits' in dir(task):
        input_dics = []
        for d in task.bits:
            input_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                      for p in all_pos])[0][0])
    
    dicplt.dichotomy_plot(fac, spec, SD, PS_err=fac_err, CCGP_err=spec_err, SD_err=SD_err,
                          input_dics=input_dics, output_dics=output_dics)
    
else:
    nrow = int(np.sqrt(len(factorization)))
    ncol = len(factorization)//nrow
    
    plt.figure()
    for i, (fac, spec, SD) in enumerate(zip(factorization, specialization, SD_unbalanced)):
        output_dics = []
        for d in task.positives:
            output_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                                        for p in all_pos])[0][0])
        
        if 'bits' in dir(task):
            input_dics = []
            for d in task.bits:
                input_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                          for p in all_pos])[0][0])
        else:
            input_dics = None
        
        plt.subplot(nrow, ncol, i+1)
        dicplt.dichotomy_plot(fac, spec, SD,
                              input_dics=input_dics,
                              output_dics=output_dics,
                              include_legend=(i==0), 
                              include_cbar=False,
                              s=10)
        if np.mod(i,ncol)>0:
            plt.yticks([])
        plt.ylabel('')
        if (i+1)//nrow < nrow:
            plt.xlabel('')
            plt.xticks([])


#%% Plot coefficients against each other
these_pos = np.nonzero(np.squeeze(SD_unbalanced).mean(0)>0.9)[0]
num_pos = len(these_pos)
 
# plot_this = single_neurs
plot_this = clf_weights

lims = [plot_this[:,these_pos,:].min()-0.1, plot_this[:,these_pos,:].max()+0.1]

centroids = np.array([this_exp.train_data[0][this_exp.train_conditions==c].mean(0).numpy() \
                      for c in np.unique(this_exp.train_conditions)])

fig = plt.figure()
for i,p1 in enumerate(these_pos):
    for j,p2 in enumerate(these_pos):
        if i < j:
            continue
        
        if i == j:
            clr = np.isin(np.unique(this_exp.train_conditions),all_pos[these_pos[i]]).squeeze()
            ax = fig.add_subplot(num_pos, num_pos, (i+num_pos*i)+1, projection='3d')
            dicplt.pca3d(ax, centroids.T, clr, s=100, cmap='bwr')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
        else:
            ax = plt.subplot(num_pos, num_pos, (j+num_pos*i)+1)
            plt.scatter(plot_this[:,p1,:].flatten(),plot_this[:,p2,:].flatten(), 
                        c=(np.arange(len(SD_unbalanced))[:,None]*np.ones((1,N))).flatten(), s=10, alpha=0.1)
            
            # newlims = [np.min([plt.ylim(), plt.xlim()]), np.max([plt.ylim(), plt.xlim()])]
    
            plt.axis('equal')
            plt.axis('square')
            # plt.xlim(newlims)
            # plt.ylim(newlims)
            plt.xlim(lims)
            plt.ylim(lims)
            
            plt.plot(plt.xlim(),plt.xlim(),'--',c=(0.5,0.5,0.5))
            plt.plot(plt.xlim(),[0,0],'--',c=(0.5,0.5,0.5))
            plt.plot([0,0],plt.xlim(),'--',c=(0.5,0.5,0.5))
            # plt.plot(plt.xlim(),[0.5,0.5],'--',c=(0.5,0.5,0.5))
            # plt.plot([0.5,0.5],plt.xlim(),'--',c=(0.5,0.5,0.5))
    
    
            if i<(num_pos-1):
                plt.xticks([])
            else:
                plt.xticks([0,1])
            if j>0:
                plt.yticks([])
            else:
                plt.yticks([0,1])


#%% Projecting onto specific dichotmy subspaces
these_dichotomies = (0,9)
this_network = -1

model = all_nets[0][this_network]
args = all_args[0][this_network]
vecs = coding_vectors[this_network]

this_exp.load_other_info(args)
this_exp.load_data(SAVE_DIR)

z = model(this_exp.train_data[0])[2].detach().numpy()

fake_task = util.RandomDichotomies(d=this_exp.task.positives)
colorby = util.decimal(fake_task(this_exp.train_conditions)[:5000])

# vals = util.cosine_sim()

orth_vecs = la.orth(vecs[these_dichotomies,...].squeeze().T) # basis for subspace
Proj = orth_vecs@orth_vecs.T

z_proj = Proj@z.T

U, S, _ = la.svd(z_proj-z_proj.mean(1)[:,None], full_matrices=False)

pcs = z_proj.T[:5000,:]@U[:3,:].T

proj_perf = spc.expit(model.dec(torch.tensor(z_proj).float().T).detach())[fake_task(this_exp.train_conditions)==1].mean()

# plt.figure()
plt.loglog(np.arange(1,N),(S[:-1]**2)/np.sum(S[:-1]**2))
plt.plot([num_var,num_var],plt.ylim(), 'k--')
plt.xlabel('PC')
plt.ylabel('variance explained')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(pcs[:,0],pcs[:,1],pcs[:,2], c=colorby, alpha=0.1, s=500)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')
ax.set_title('perf='+str(proj_perf.item()))

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()
util.set_axes_equal(ax)

#%% PCA
model = all_nets[0][0]
args = all_args[0][0]

this_exp.load_other_info(args)
this_exp.load_data(SAVE_DIR)

fake_task = util.RandomDichotomies(d=this_exp.task.positives)

z = model(this_exp.train_data[0])[2].detach().numpy().T
# this_exp = exp.mnist_multiclass(N, task, SAVE_DIR, abstracts=abstract_variables)
# this_exp = exp.mnist_multiclass(n, class_func, SAVE_DIR)
ans = fake_task(this_exp.train_conditions)
cond = util.decimal(ans)[:5000]

# cmap_name = 'nipy_spectral'
# colorby = util.decimal(ans)
colorby = cond
# colorby = this_exp.train_conditions[:5000]

U, S, _ = la.svd(z-z.mean(1)[:,None], full_matrices=False)
pcs = z.T[:5000,:]@U[:4,:].T

# plt.figure()
plt.loglog(np.arange(1,N),(S[:-1]**2)/np.sum(S[:-1]**2))
plt.plot([num_var,num_var],plt.ylim(), 'k--')
plt.xlabel('PC')
plt.ylabel('variance explained')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(pcs[:,0],pcs[:,1],pcs[:,2], c=colorby, alpha=0.1, s=500)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()
util.set_axes_equal(ax)

#%% MDS
n_mds = 3
n_compute = 500
layer = 0

average_model = False
# average_model = True

if average_model:
    cond = np.arange(num_cond)
    
    mds = manifold.MDS(n_components=n_mds, dissimilarity='precomputed')
    emb = mds.fit_transform(np.array(distances).mean(0)[layer])
else:
    model = all_nets[0][0]
    args = all_args[0][0]
    
    this_exp.load_other_info(args)
    this_exp.load_data(SAVE_DIR)
    
    fake_task = tasks.RandomDichotomies(num_cond,num_var,0)
    fake_task.positives = this_exp.task.positives
    
    idx = np.random.choice(this_exp.train_data[0].shape[0], n_compute, replace=False)
    
    rep = model.enc.network[:2*(layer+1)]
    z = rep(this_exp.train_data[0][idx,...]).detach().numpy()
    
    # ans = this_exp.train_data[1][idx,...]
    ans = fake_task(this_exp.train_conditions)[idx]
    
    cond = util.decimal(ans)
    # cond = this_exp.train_conditions[idx]
    # cond = np.isin(this_exp.train_conditions[idx], (0,3,5,7))

    # colorby = this_exp.train_conditions[idx]
    # colorby = np.isin(this_exp.train_conditions[idx], (0,2,3,6))
    
    mds = manifold.MDS(n_components=n_mds)
    emb = mds.fit_transform(z+np.random.randn(*z.shape)*1e-4)

colorby = cond

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

# plt.show()

#%% See if the learned representation makes another task easier 
use_xor = False
# use_xor = True

model = all_nets[0][1]
args = all_args[0][1]

this_exp.load_other_info(args)
this_exp.load_data(SAVE_DIR)

new_task = util.RandomDichotomies(num_cond,1,0)
if use_xor:
    v1 = np.isin(range(num_cond), this_exp.task.positives[0])
    v2 = np.isin(range(num_cond), this_exp.task.positives[1])
    xor = np.where(~(v1^v2))[0]
    new_task.positives = [xor]
else:
    new_task.positives = [this_exp.task.positives[0]]

bsz = 1
lr = 1e-4
nepoch = 10

num_inits = 30
n_compute = 2500
num_data = 5000

# optimize
train_loss = []
test_error = []
for it in range(num_inits):
    glm = nn.Linear(this_exp.dim_input, new_task.dim_output)
    
    # there's just way too much data
    idx_trn = np.random.choice(len(this_exp.train_conditions), num_data, replace=False)
    idx_tst = np.random.choice(len(this_exp.test_conditions), n_compute, replace=False)
    
    glm = nn.Linear(this_exp.dim_input, new_task.dim_output)
    z_pretrained = model(this_exp.train_data[0][idx_trn,:])[2].detach()
    
    # z_pretrained = new_exp.train_data[0]
    targ = new_task(this_exp.train_conditions[idx_trn])
    
    z_test = model(this_exp.test_data[0][idx_tst,:])[2].detach()
    # z_test = this_exp.test_data[0]
    targ_test = new_task(this_exp.test_conditions[idx_tst])
    
    new_dset = torch.utils.data.TensorDataset(z_pretrained, targ)
    dl = torch.utils.data.DataLoader(new_dset, batch_size=bsz, shuffle=True)
    
    optimizer = this_exp.opt_alg(glm.parameters(), lr=lr)
        
    trn_lss = []
    tst_err = []
    for epoch in tqdm(range(nepoch)):
        
        # running_error = 0
        running_loss = 0
        for i, (x,y) in enumerate(dl):
        
            pred = glm(z_test)
            tst_err.append(1-(new_task.correct(pred, targ_test)))
        
            optimizer.zero_grad()
            
            eta = glm(x)
            
            # terr = 1- (new_task.correct(eta, y)/x.shape[0])
            loss = -new_task.obs_distribution.distr(eta).log_prob(y).mean()
            
            loss.backward()
            optimizer.step()
            
            trn_lss.append(loss.item())
        
    train_loss.append(trn_lss)
    test_error.append(tst_err)
    

mean_err = np.mean(test_error,0).squeeze()
std_err = np.std(test_error,0).squeeze()
    
# plt.figure()
# plt.plot(np.arange(1,nepoch+1),train_loss)
# plt.semilogx()
# plt.xlabel('epoch')
# plt.ylabel('training loss')

# plt.figure()
# plt.loglog(np.arange(1,nepoch*num_data+1), (mean_err+1e-6)*100)
plt.plot(np.arange(1,nepoch*num_data+1), mean_err)
plt.fill_between(np.arange(1,nepoch*num_data+1), 
                 (mean_err-std_err)+1e-6,
                 (mean_err+std_err)+1e-6,
                 alpha=0.5)
plt.semilogx()
plt.ylim([-0.05,0.7])
plt.plot([num_data,num_data],plt.ylim(),'--',color=(0.5,0.5,0.5))
plt.xlabel('epoch')
plt.ylabel('test error')

#%% Tolerance to deletion
# new_task = this_exp.task
new_task = util.RandomDichotomies(8,2,0)
new_task.positives = this_exp.task.positives
bsz = 64
lr = 1e-4
# sparsity =  np.exp(np.arange(-10,1,1))
sparsity = np.linspace(0,1,10)

n_compute = 5000
n_svm = 50

# lin_clf = svm.LinearSVC
lin_clf = linear_model.LogisticRegression
# lin_clf = linear_model.Perceptron
# lin_clf = linear_model.RidgeClassifier

# new_exp = exp.mnist_multiclass(new_task, SAVE_DIR)
new_exp = this_exp

# glm = nn.Linear(N, new_task.dim_output)
# glm = nn.Linear(784, new_task.dim_output)
# glm = Feedforward([784, 100, 50, new_task.dim_output], ['ReLU', 'ReLU', None])
# glm = MultiGLM(Feedforward([784, 100, 50]), nn.Linear(50,new_task.dim_output), new_task.obs_distribution)

error = np.zeros((len(sparsity),n_svm, new_task.num_var))
w = np.zeros((len(sparsity),n_svm, N, new_task.num_var))
i = 0
for i, p in enumerate(sparsity):
    for j in tqdm(range(n_svm), desc='Sparsity %.1f'%p):
        # print('Sparsity %d'%p)
        z_pretrained = model(new_exp.train_data[0])[2].detach()
        # z_pretrained = train_dat[0]
        targ = new_task(new_exp.train_conditions)
        
        idx = np.random.choice(z_pretrained.shape[0], n_compute, replace=False)
        
        z_test = model(new_exp.test_data[0])[2].detach()
        # z_test = test_dat[0]
        targ_test = new_task(new_exp.test_conditions).detach()
        
        clf = LinearDecoder(N, new_task.num_var, lin_clf)
        clf.fit(z_pretrained[idx,:], targ[idx,...], max_iter=5000)
        
        mask = np.random.rand(z_test.shape[0],z_test.shape[1])>p
        w[i,j,:,:] = clf.coefs.squeeze().T
        error[i,j,:] = clf.test(z_test.numpy()*mask, targ_test.numpy()).squeeze()
        
        # i+=1
        # print('Epoch %d: loss=%.3f; error=%.3f'%(epoch, running_loss/(i+1), terr))
        
    
