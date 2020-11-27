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

#%% Model specification -- for loading purposes
# mse_loss = True
mse_loss = False
# categorical = True
categorical = False

num_cond = 8
num_var = 2

# use_mnist = True
use_mnist = False

# task = util.ParityMagnitude()
if categorical:
    task = util.RandomDichotomiesCategorical(num_cond,num_var,0, mse_loss)
else:
    task = util.RandomDichotomies(num_cond,num_var,0, mse_loss)
# task = util.ParityMagnitudeEnumerated()
# task = util.Digits()
# task = util.DigitsBitwise()

latent_dist = None
# latent_dist = GausId
nonlinearity = 'ReLU'
# nonlinearity = 'LeakyReLU'

# num_layer = 0
num_layer = 1

good_start = True
# good_start = False
# coding_level = 0.7
coding_level = None

rotation = 0.0

decay = 0.0

H = 100
# N_list = None # set to None if you want to automatically discover which N have been tested
# N_list = [2,3,4,5,6,7,8,9,10,11,20,25,50,100]
# N_list = None
# N_list = [2,3,5,10,50,100]
N_list = [50]

# random_decoder = students.LinearRandomSphere(radius=0.2, eps=0.05, 
#                                               fix_weights=True,
#                                               nonlinearity=task.link)
# random_decoder = students.LinearRandomNormal(var=0.2, 
#                                               fix_weights=True, 
#                                               nonlinearity=task.link)
# random_decoder = students.LinearRandomProportional(scale=0.2, 
#                                                     fix_weights=True, 
#                                                     coef=2,
#                                                     nonlinearity=task.link)
random_decoder = None

# find experiments 
if use_mnist:
    this_exp = exp.mnist_multiclass(task, SAVE_DIR, 
                                    z_prior=latent_dist,
                                    num_layer=num_layer,
                                    weight_decay=decay,
                                    decoder=random_decoder,
                                    good_start=good_start,
                                    init_coding=coding_level)
else:
    this_exp = exp.random_patterns(task, SAVE_DIR, 
                                    num_class=num_cond,
                                    dim=100,
                                    var_means=1,
                                    z_prior=latent_dist,
                                    num_layer=num_layer,
                                    weight_decay=decay,
                                    decoder=random_decoder,
                                    good_start=good_start,
                                    init_coding=coding_level,
                                    rot=rotation)

this_folder = SAVE_DIR + this_exp.folder_hierarchy()
if (N_list is None):
    files = os.listdir(this_folder)
    param_files = [f for f in files if 'parameters' in f]
    
    if len(param_files)==0:
        raise ValueError('No experiments in specified folder `^`')
    
    Ns = np.array([re.findall(r"N(\d+)_%s"%nonlinearity,f)[0] \
                    for f in param_files]).astype(int)
    
    N_list = np.unique(Ns)

# load experiments
# loss = np.zeros((len(N_list), 1000))
# test_perf = np.zeros((Q, len(N_list), 1000))
# test_PS = np.zeros((Q, len(N_list), 1000))
# shat = np.zeros((Q, len(N_list), 1000))
nets = [[] for _ in N_list]
all_nets = [[] for _ in N_list]
all_args = [[] for _ in N_list]
mets = [[] for _ in N_list]
dicts = [[] for _ in N_list]
best_perf = []
for i,n in enumerate(N_list):
    files = os.listdir(this_folder)
    param_files = [f for f in files if ('parameters' in f and '_N%d_%s'%(n,nonlinearity) in f)]
    
    # j = 0
    num = len(param_files)
    all_metrics = {}
    best_net = None
    this_arg = None
    maxmin = 0
    for j,f in enumerate(param_files):
        rg = re.findall(r"init(\d+)?_N%d_%s"%(n,nonlinearity),f)
        if len(rg)>0:
            init = np.array(rg[0]).astype(int)
        else:
            init = None
            
        this_exp.use_model(N=n, init=init)
        model, metrics, args = this_exp.load_experiment(SAVE_DIR)
        
        if metrics['test_perf'][-1,...].min() > maxmin:    
            maxmin = metrics['test_perf'][-1,...].min()
            best_net = model
            this_arg = args
        
        for key, val in metrics.items():
            if key not in all_metrics.keys():
                # shp = (num,) + np.squeeze(np.array(val)).shape
                # all_metrics[key] = np.zeros(shp)*np.nan
                all_metrics[key] = []
            
            # ugh = np.min([all_metrics[key][j,...].shape[0], np.squeeze(val).shape[0]])
            # all_metrics[key][j,:ugh,...] = np.squeeze(val)[:ugh,...]
            all_metrics[key].append(np.squeeze(val))
    
            # if (val.shape[0]==1000) or not len(val):
                # continue
            # all_metrics[key][j,...] = val
        all_nets[i].append(model)
        all_args[i].append(args)
        
    nets[i] = best_net
    mets[i] = all_metrics
    dicts[i] = this_arg
    best_perf.append(maxmin)

#%%
n_id = 0
netid = 0 # which specific experiment to use

model = nets[netid]
params = dicts[netid]
N = N_list[netid]

this_exp.load_other_info(params)
this_exp.load_data(SAVE_DIR)

test_dat = this_exp.test_data
train_dat = this_exp.train_data

#%% General metrics (always computed)
# show_me = 'train_loss'
# show_me = 'train_perf' 
# show_me = 'test_perf'
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

#%% Abstraction metrics
# show_me = 'test_PS'
# show_me = 'shattering'
show_me = 'test_ccgp'

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
pls = mean[...,num_var:].mean(1)+error[...,num_var:].mean(1)
mns = mean[...,num_var:].mean(1)-error[...,num_var:].mean(1)
untrn = plt.plot(epochs, mean[...,num_var:].mean(1))[0]
plt.fill_between(epochs, mns, pls, alpha=0.5)
plt.semilogx()

plt.xlabel('epoch', fontsize=15)
plt.ylabel(show_me, fontsize=15)
plt.title('N=%d'%N)
plt.legend(trn + [untrn], ['Var %d'%(n+1) for n in range(num_var)] + ['Non-output dichotomies'])


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
all_PS = []
all_CCGP = []
all_CCGP_ = []
CCGP_out_corr = []
mut_inf = []
all_SD = []
indep = []
for model, args in zip(all_nets[netid], all_args[netid]):
    
    this_exp.load_other_info(args)
    this_exp.load_data(SAVE_DIR)
    
    fake_task = util.RandomDichotomies(num_cond,num_var,0)
    fake_task.positives = this_exp.task.positives
    
    indep.append(fake_task.subspace_information())
    z = model(this_exp.train_data[0])[2].detach().numpy()
    # z = this_exp.train_data[0].detach().numpy()
    # z = linreg.predict(this_exp.train_data[0])@W1.T
    n_compute = np.min([5000, z.shape[0]])
    
    idx = np.random.choice(z.shape[0], n_compute, replace=False)
    # idx_tst = idx[::4] # save 1/4 for test set
    # idx_trn = np.setdiff1d(idx, idx_tst)
    
    cond = this_exp.train_conditions[idx]
    # cond = util.decimal(this_exp.train_data[1][idx,...])
    num_cond = len(np.unique(cond))
    
    # xor = np.where(~(np.isin(range(num_cond), args['dichotomies'][0])^np.isin(range(num_cond), args['dichotomies'][1])))[0]
    ## Loop over dichotomies
    # D = assistants.Dichotomies(num_cond, args['dichotomies']+[xor], extra=50)
    
    # choose dichotomies to have a particular order
    Q = len(args['dichotomies'])
    D_fake = assistants.Dichotomies(num_cond, args['dichotomies'], extra=7000)
    mi = np.array([fake_task.information(p) for p in D_fake])
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
        
        CCGP.append(D.CCGP(z[idx,:], cond, gclf, cntxt, twosided=True)[0])
        
        # shattering
        d[:,i] = D.coloring(cond)
        
    # dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
    dclf.fit(z[idx,:], d, tol=1e-5)
    
    z = model(this_exp.test_data[0])[2].detach().numpy()
    # z = this_exp.test_data[0].detach().numpy()
    # z = linreg.predict(this_exp.test_data[0])@W1.T
    idx = np.random.choice(z.shape[0], n_compute, replace=False)
    
    d_tst = np.array([D.coloring(this_exp.test_conditions[idx]) for _ in D]).T
    SD = dclf.test(z[idx,:], d_tst).squeeze()
    
    all_PS.append(PS)
    all_CCGP.append(CCGP)
    CCGP_out_corr.append(out_corr)
    all_SD.append(SD)
    mut_inf.append(mi[midx])

R = np.repeat(np.array(CCGP_out_corr),2,-1)
basis_dependence = np.array(indep).max(1)
out_MI = np.array(mut_inf)

#%% plot PS and CCGP
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

PS_err = np.nanstd(np.array(all_PS), axis=0)#/np.sqrt(len(all_PS))
CCGP_err = np.nanstd(almost_all_CCGP, axis=0)#/np.sqrt(len(all_CCGP))
SD_err = np.nanstd(np.array(all_SD), axis=0)#/np.sqrt(len(all_SD))

# be very picky about offsets
offset = np.random.choice(np.linspace(-0.15,0.15,10), 3*ndic)
special_offsets = np.linspace(-0.1,0.1,num_var)
for i in range(num_var):
    offset[i+np.arange(3)*ndic] = special_offsets[i]

xfoo = np.repeat([0,1,2],ndic).astype(int) + offset # np.random.randn(ndic*3)*0.1
yfoo = np.concatenate((PS, CCGP, SD))
yfoo_err = np.concatenate((PS_err, CCGP_err, SD_err))
cfoo = np.tile(np.flip(np.sort(out_MI.mean(0))),3)

# plt.scatter(xfoo, yfoo, s=12, c=(0.5,0.5,0.5))
plt.errorbar(xfoo, yfoo, yerr=yfoo_err, linestyle='None', c=(0.5,0.5,0.5), zorder=0)
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


#%% PCA
model = all_nets[0][1]
args = all_args[0][1]

this_exp.load_other_info(args)
this_exp.load_data(SAVE_DIR)

fake_task = util.RandomDichotomies(num_cond,num_var,0)
fake_task.positives = this_exp.task.positives

z = model(this_exp.train_data[0])[2].detach().numpy()
# this_exp = exp.mnist_multiclass(N, task, SAVE_DIR, abstracts=abstract_variables)
# this_exp = exp.mnist_multiclass(n, class_func, SAVE_DIR)
ans = fake_task(this_exp.train_conditions)
cond = util.decimal(ans)[:5000]

# cmap_name = 'nipy_spectral'
# colorby = util.decimal(ans)
colorby = cond
# colorby = this_exp.train_conditions[:5000]

U, S, _ = la.svd(z-z.mean(1)[:,None], full_matrices=False)
pcs = z[:5000,:]@U[:3,:].T

# plt.figure()
plt.loglog(np.arange(1,N),(S[:-1]**2)/np.sum(S[:-1]**2))
plt.plot([num_var,num_var],plt.ylim(), 'k--')
plt.xlabel('PC')
plt.ylabel('variance explained')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(pcs[:,0],pcs[:,1],pcs[:,2], c=colorby, alpha=0.1)
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

#%% MDS
n_mds = 3
n_compute = 500

model = all_nets[0][0]
args = all_args[0][0]

this_exp.load_other_info(args)
this_exp.load_data(SAVE_DIR)

fake_task = util.RandomDichotomies(num_cond,num_var,0)
fake_task.positives = this_exp.task.positives

idx = np.random.choice(this_exp.train_data[0].shape[0], n_compute, replace=False)

z = model(this_exp.train_data[0][idx,...])[2].detach().numpy()

# ans = this_exp.train_data[1][idx,...]
ans = fake_task(this_exp.train_conditions)[idx]

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
        
    
