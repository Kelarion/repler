CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

import students
from assistants import *
import experiments as exp
import util

#%% Model specification -- for loading purposes
# task = util.ParityMagnitude()
task = util.RandomDichotomies(8,2,0)
# task = util.ParityMagnitudeEnumerated()
# task = util.Digits()
# task = util.DigitsBitwise()
# obs_dist = Bernoulli(1)
latent_dist = None
# latent_dist = GausId
nonlinearity = 'ReLU'
# nonlinearity = 'LeakyReLU'

num_layer = 0
# num_layer = 1

good_start = True
# good_start = False
coding_level = 0.8
# coding_level = None

decay = 0.0

H = 100
# N_list = None # set to None if you want to automatically discover which N have been tested
# N_list = [2,3,4,5,6,7,8,9,10,11,20,25,50,100]
# N_list = None
# N_list = [2,3,5,10,50,100]
N_list = [100]

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
# this_exp = exp.mnist_multiclass(task, SAVE_DIR, 
#                                 z_prior=latent_dist,
#                                 num_layer=num_layer,
#                                 weight_decay=decay,
#                                 decoder=random_decoder,
#                                 good_start=good_start,
#                                 init_coding=coding_level)

this_exp = exp.random_patterns(task, SAVE_DIR, 
                                num_class=8,
                                dim=100,
                                var_means=1,
                                z_prior=latent_dist,
                                num_layer=num_layer,
                                weight_decay=decay,
                                decoder=random_decoder,
                                good_start=good_start,
                                init_coding=coding_level)

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
                shp = (num,) + np.squeeze(np.array(val)).shape
                all_metrics[key] = np.zeros(shp)*np.nan
            all_metrics[key][j,...] = np.squeeze(val)
    
    nets[i] = best_net
    mets[i] = all_metrics
    dicts[i] = this_arg
    best_perf.append(maxmin)

#%%
netid = 0 # which specific experiment to use

model = nets[netid]
params = dicts[netid]
N = N_list[netid]

this_exp.load_other_info(params)
this_exp.load_data(SAVE_DIR)

test_dat = this_exp.test_data
train_dat = this_exp.train_data

#%%
# show_me = 'train_loss'
# show_me = 'train_perf' 
# show_me = 'test_perf'
# show_me = 'test_PS'
# show_me = 'shattering'
# show_me = 'test_ccgp'
# show_me = 'mean_grad'
# show_me = 'std_grad'
# show_me = 'linear_dim'
show_me = 'sparsity'

epochs = np.arange(1,mets[netid]['train_loss'].shape[-1]+1)

mean = np.nanmean(mets[netid][show_me],0)
error = (np.nanstd(mets[netid][show_me],0)/np.sqrt(mets[netid][show_me].shape[0]))

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

#%%
ba = []
for d in this_exp.task.positives:
    ba.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(8),p))==list(d))\
                  for p in pos_conds])[0][0])
        
mean = np.nanmean(mets[netid]['shattering'],0)
plt.plot(epochs,mean[:,ba].mean(1))
plt.plot(epochs,mean[:,np.setdiff1d(range(35),ba)].mean(1))
plt.fill_between(epochs,mean[:,ba].mean(1)-mean[:,ba].std(1),mean[:,ba].mean(1)+mean[:,ba].std(1),
                 alpha=0.5)
plt.fill_between(epochs,mean[:,np.setdiff1d(range(35),ba)].mean(1)-mean[:,np.setdiff1d(range(35),ba)].std(1),
                 mean[:,np.setdiff1d(range(35),ba)].mean(1)+mean[:,np.setdiff1d(range(35),ba)].std(1),
                 alpha=0.5)

plt.semilogx()
plt.ylabel('Shattering dimension')
plt.legend(['Trained','Untrained'])
#%%
best = mets[netid]['test_perf'][:,-1,:].max(-1).argmin(0)
plt.plot(epochs,mets[netid][show_me][best,...], 'r--', alpha=0.6)

worst = mets[netid]['test_perf'][:,-1,:].min(-1).argmax(0)
plt.plot(epochs,mets[netid][show_me][worst,...], 'b--', alpha=0.6)


#%% do the linear readout
clsfr = svm.LinearSVC # the classifier to use
cfargs = {'tol': 1e-5, 'max_iter':5000}

# train linear decoders
z = model(test_dat[0])[2].detach().numpy()
ans = test_dat[1].detach().numpy()

clf = LinearDecoder(N, Q, clsfr)
clf.fit(z, ans, **cfargs)

coefs = clf.coefs
thrs = clf.thrs

# test
z = model(train_dat[0])[2].detach().numpy()
ans = train_dat[1].detach().numpy()
cond = util.decimal(ans)

perf = clf.test(z, ans)
marg = clf.margin(z, ans)
inner = np.einsum('ik...,jk...->ij...', coefs, coefs)
proj = clf.project(z)

plt.figure()
scat = plt.scatter(proj[0,:], proj[1,:], c=cond)
plt.xlabel('Parity classifier')
plt.ylabel('Magnitude classifier')
cb = plt.colorbar(scat, 
                  ticks=np.unique(cond),
                  drawedges=True,
                  values=np.unique(cond))
cb.set_ticklabels(np.unique(cond))
cb.set_alpha(1)
cb.draw_all()

plt.figure()
plt.scatter(clf.coefs[0,:,0],clf.coefs[1,:,0])
lims = [min([plt.xlim()[0],plt.ylim()[0]]), max([plt.xlim()[1],plt.ylim()[1]])]
plt.gca().set_xlim(lims)
plt.gca().set_ylim(lims)
plt.gca().set_aspect('equal')
plt.plot(lims, lims, 'k--', alpha=0.2)
plt.plot(lims,[0,0], 'k-.', alpha=0.2)
plt.plot([0,0],lims, 'k-.', alpha=0.2)
plt.xlabel('Parity weight')
plt.ylabel('Magnitude weight')

#%% MDS
n_mds = 2
n_compute = 500

idx = np.random.choice(train_dat[0].shape[0], n_compute, replace=False)

z = model(train_dat[0][idx,...])[2].detach().numpy()

ans = this_exp.train_data[1][idx,...]
cond = util.decimal(ans)

mds = manifold.MDS(n_components=2)

emb = mds.fit_transform(z)

scat = plt.scatter(emb[:,0],emb[:,1], c=cond)
plt.xlabel('MDS1')
plt.ylabel('MDS2')
cb = plt.colorbar(scat,
                  ticks=np.unique(cond),
                  drawedges=True,
                  values=np.unique(cond))
cb.set_ticklabels(np.unique(cond)+1)
cb.set_alpha(1)
cb.draw_all()

#%% compute dichotomy metrics
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

# Loop over dichotomies
D = Dichotomies(num_cond)
clf = LinearDecoder(N, 1, MeanClassifier)
gclf = LinearDecoder(N, 1, svm.LinearSVC)
dclf = LinearDecoder(N, D.ntot, svm.LinearSVC)
# clf = LinearDecoder(this_exp.dim_input, 1, MeanClassifier)
# gclf = LinearDecoder(this_exp.dim_input, 1, svm.LinearSVC)
# dclf = LinearDecoder(this_exp.dim_input, D.ntot, svm.LinearSVC)

K = int(num_cond/2) - 1 # use all but one pairing
# K = int(num_cond/4) # use half the pairings

PS = np.zeros(D.ntot)
CCGP = np.zeros(D.ntot)
d = np.zeros((n_compute, D.ntot))
pos_conds = []
for i, pos in enumerate(D):
    pos_conds.append(pos)
    print('Dichotomy %d...'%i)
    # parallelism
    PS[i] = D.parallelism(z[idx,:], cond, clf)
    
    # CCGP
    CCGP[i] = D.CCGP(z[idx,:], cond, gclf, K)
    
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

#%% plot PS and CCGP
ndic = len(PS)

xfoo = np.repeat([0,1,2],ndic).astype(int) + np.random.randn(ndic*3)*0.1
yfoo = np.concatenate((PS, CCGP, SD))

plt.scatter(xfoo, yfoo, s=12, c=(0.5,0.5,0.5))
plt.xticks([0,1,2], labels=['PS', 'CCGP', 'Shattering'])
plt.ylabel('PS or Cross-validated performance')

# highlight special dichotomies
# par = plt.scatter(xfoo[[20,20+ndic,20+2*ndic]], yfoo[[20,20+ndic,20+2*ndic]], 
#                   marker='o', edgecolors='r', s=60, facecolors='none', linewidths=3)
# mag = plt.scatter(xfoo[[0, ndic, 2*ndic]], yfoo[[0,ndic,2*ndic]], 
#                   marker='o', edgecolors='g', s=60, facecolors='none', linewidths=3)
# other = plt.scatter(xfoo[[9,9+ndic,9+2*ndic]], yfoo[[9,9+ndic,9+2*ndic]], 
#                     marker='o', edgecolors='b', s=60, facecolors='none', linewidths=3)
anns = []
for d in this_exp.task.positives:
    n = np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(this_exp.num_class),p))==list(d))\
                  for p in pos_conds])[0][0]
    anns.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], 
                marker='o', s=60, linewidths=3))

# par = plt.scatter(xfoo[[9,9+ndic,9+2*ndic]], yfoo[[9,9+ndic,9+2*ndic]], 
#                   marker='o', edgecolors='r', s=60, facecolors='none', linewidths=3)
# mag = plt.scatter(xfoo[[27, 27+ndic, 27+2*ndic]], yfoo[[27,27+ndic,27+2*ndic]], 
#                   marker='o', edgecolors='g', s=60, facecolors='none', linewidths=3)

plt.legend(anns, ['var %d'%(i+1) for i in range(len(anns))])
# plt.legend([par,mag], ['{3,4,7,8}', '{2,3,6,7}'])


#%%
scat = plt.scatter(z[:,0],z[:,1], c=cond)
    
cb = plt.colorbar(scat, 
                  ticks=np.unique(cond),
                  drawedges=True,
                  values=np.unique(cond))
cb.set_ticklabels(np.unique(cond)+1)
cb.set_alpha(1)
cb.draw_all()


#%% PCA
# abstract_variables = util.DigitsBitwise()
# abstract_variables = util.ParityMagnitude()

z = model(this_exp.train_data[0])[2].detach().numpy()
# this_exp = exp.mnist_multiclass(N, task, SAVE_DIR, abstracts=abstract_variables)
# this_exp = exp.mnist_multiclass(n, class_func, SAVE_DIR)
ans = this_exp.train_data[1]
cond = util.decimal(ans)

# cmap_name = 'nipy_spectral'
colorby = util.decimal(ans)

U, S, _ = la.svd(z-z.mean(1)[:,None], full_matrices=False)
pcs = z@U[:3,:].T

plt.figure()
plt.loglog(np.arange(1,N),(S[:-1]**2)/np.sum(S[:-1]**2))
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

#%%
# clf = LinearDecoder(N, 1, MeanClassifier)
clf = LinearDecoder(N, 1, svm.LinearSVC)

pos = np.unique(D.cond[coloring])

ps = [] 
for neg in permutations(np.unique(D.cond[~coloring])):
    # for a given pairing of positive and negative conditions, I need to
    # generate labels for a classifier.
    # labels = np.array([np.where((D.cond==pos[n])|(D.cond==neg[n]), 
    #                             D.cond==pos[n], np.nan) \
    #                    for n, _ in enumerate(neg)]).T
    
    whichone = np.array([(D.cond==pos[n])|(D.cond==neg[n]) \
                         for n, _ in enumerate(neg)]).argmax(0)
    lbs = np.isin(D.cond, pos)
        
    clf.fit(z, lbs[:,None], t_=whichone)
    clf.coefs = clf.coefs.transpose(2,1,0)
    ps.append(clf.avg_dot()[0])
    
    plt.figure()
    plt.scatter(z[:,0],z[:,1], c = coloring)
    
    xy = z[lbs&(whichone==0),:].mean(0)
    plt.quiver(xy[0],xy[1], -clf.coefs[0,0,0], -clf.coefs[0,1,0], scale_units='xy', scale=1)
    
    xy = z[lbs&(whichone==1),:].mean(0)
    plt.quiver(xy[0],xy[1], -clf.coefs[1,0,0], -clf.coefs[1,1,0], scale_units='xy', scale=1)


#%% 
wa = np.meshgrid(sts.norm.ppf(np.linspace(0.01,0.99,20)),sts.norm.ppf(np.linspace(0.01,0.99,20)))
z_q = np.append(wa[0].flatten()[:,None], wa[1].flatten()[:,None],axis=1)

#%% See if the learned representation makes another task easier
new_task = util.Digits()
# new_task = util.ParityMagnitudeEnumerated()
# new_task = this_exp.task

bsz = 64
lr = 1e-4
nepoch = 300

n_compute = 5000

# new_exp = exp.mnist_multiclass(new_task, SAVE_DIR)
new_exp = this_exp

# glm = nn.Linear(N, new_task.dim_output)
glm = nn.Linear(new_exp.dim_input, new_task.dim_output)
# glm = Feedforward([784, 100, 50, new_task.dim_output], ['ReLU', 'ReLU', None])
# glm = MultiGLM(Feedforward([784, 100, 50]), nn.Linear(50,new_task.dim_output), new_task.obs_distribution)

# z_pretrained = model(new_exp.train_data[0])[2].detach()
z_pretrained = new_exp.train_data[0]
targ = new_exp.train_data[1]

# z_test = model(new_exp.test_data[0])[2].detach()
z_test = new_exp.test_data[0]
targ_test = new_exp.test_data[1]

new_dset = torch.utils.data.TensorDataset(z_pretrained, targ)
dl = torch.utils.data.DataLoader(new_dset, batch_size=bsz, shuffle=True)

optimizer = new_exp.opt_alg(glm.parameters(), lr=lr)

# optimize
train_loss = np.zeros(nepoch)
test_error = np.zeros((nepoch, new_task.dim_output)).squeeze()
for epoch in range(nepoch):
    
    idx = np.random.choice(len(targ_test), n_compute, replace=False)
    
    pred = glm(z_test[idx,:])
    terr = 1- (new_task.correct(pred, targ_test[idx])/n_compute)
    # running_error = 0
    running_loss = 0
    for i, (x,y) in enumerate(dl):
        optimizer.zero_grad()
        
        eta = glm(x)
        
        # terr = 1- (new_task.correct(eta, y)/x.shape[0])
        loss = -new_task.obs_distribution.distr(eta).log_prob(y).sum()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # running_error += terr.item()
        
    train_loss[epoch] = running_loss/(i+1)
    test_error[epoch] = terr
    print('Epoch %d: loss=%.3f; error=%.3f'%(epoch, running_loss/(i+1), terr.mean()))

# plt.figure()
# plt.plot(np.arange(1,nepoch+1),train_loss)
# plt.semilogx()
# plt.xlabel('epoch')
# plt.ylabel('training loss')



plt.figure()
plt.loglog(np.arange(1,nepoch+1), test_error*100)
plt.xlabel('epoch')
plt.ylabel('test error')


