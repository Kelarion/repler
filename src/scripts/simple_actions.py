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

#%%
num_cond = 8
num_var = 2

task = util.RandomDichotomies(num_cond,num_var,0)
# task = util.ParityMagnitude()

# this_exp = exp.mnist_multiclass(task, SAVE_DIR, abstracts=abstract_variables)
this_exp = exp.random_patterns(task, SAVE_DIR, 
                               num_class=8,
                               dim=100,
                               var_means=1,
                               var_noise=0.1)

#%% set up the task
p = 2**num_var
allowed_actions = [0,1,2]
# allowed_actions = [0]
p_action = [0.8,0.1,0.1]
# p_action = [1.0]

# output_states = this_exp.train_data[1].data
# output_states = util.decimal(this_exp.train_data[1])
output_states = this_exp.train_data[0].data
# output_states = ContinuousEmbedding(N_, 1.0)(this_exp.train_data[1])

# output_states = assistants.Indicator(p,p)(util.decimal(this_exp.train_data[1]).int())

input_states = this_exp.train_data[0].data
# input_states = 1*this_exp.train_data[1].data
# input_states = assistants.Indicator(p,p)(util.decimal(this_exp.train_data[1]).int())@W2.T+np.random.randn(56000,N_)*0.2
# input_states = this_exp.train_data[1]@W1.T + np.random.randn(56000,N_)*0.2

abstract_conds = util.decimal(this_exp.train_data[1])
cond_set = np.unique(abstract_conds)

# draw the "actions" for each data point
actns = torch.tensor(np.random.choice(allowed_actions, this_exp.train_data[0].shape[0], p=p_action)).int()
actions = torch.stack([(actns&(2**i))/2**i for i in range(num_var)]).float().T

# act_rep = assistants.Indicator(p,p)(util.decimal(actions).int())
act_rep = actions.data

# inputs = np.concatenate([input_states,act_rep], axis=1)
# # inputs = np.concatenate([input_states, this_exp.train_data[1]], axis=1)
inputs = input_states.float().detach().numpy()

# # sample the successor states, i.e. input + action
successors = np.mod(this_exp.train_data[1]+actions, 2)

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


#%%
N = 100

# net = students.Feedforward([inputs.shape[1],100,2],['ReLU',None])
net = students.MultiGLM(students.Feedforward([inputs.shape[1], N], ['ReLU']),
                        students.Feedforward([N, targets.shape[1]], [None]),
                        students.GausId(targets.shape[1]))
# net = students.MultiGLM(students.Feedforward([inputs.shape[1],N], ['ReLU']),
#                         students.Feedforward([N,targets.shape[1]], [None]),
#                         students.Bernoulli(targets.shape[1]))
# net = students.MultiGLM(students.Feedforward([inputs.shape[1],N], ['ReLU']),
#                         students.Feedforward([N, p], [None]),
#                         students.Categorical(p))

n_trn = int(0.5*targets.shape[0])   
trn = np.random.choice(targets.shape[0],n_trn,replace=False)
tst = np.random.choice(np.setdiff1d(range(targets.shape[0]),trn), int(0.5*n_trn), replace=False)

optimizer = optim.Adam(net.parameters(), lr=1e-4)
dset = torch.utils.data.TensorDataset(torch.tensor(inputs[trn,:]).float(),
                                      targets[trn,:].float())
dl = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)

train_loss = []
# train_perf = []
train_PS = []
test_loss = []
min_dist = []
for epoch in range(5000):
    # check whether outputs are in the correct class centroid
    idx2 = np.random.choice(tst, 5000, replace=False)
    zee, _, z = net(torch.tensor(inputs[idx2,:]).float())
    # tst_ls = -net.obs.distr(zee).log_prob(targets[idx2,:].float()).mean()
    tst_ls = ((zee-targets[idx2,:].float())**2).sum(1).mean()
    test_loss.append(tst_ls)
    
    idx = np.random.choice(trn, 5000, replace=False)
    zee, _, z = net(torch.tensor(inputs[idx,:]).float())
    # why = successors[idx,...].detach().numpy()
    
    # centroids = np.stack([z[this_exp.train_conditions[idx]==i,:].detach().mean(0) \
    #                       for i in np.unique(this_exp.train_conditions[idx])])
    # dist_to_class = np.sum((zee[:,:,None].detach().numpy() - centroids.T)**2,1)
    # nearest = dist_to_class.argmin(1)
    # labs = this_exp.task(torch.tensor(nearest)).detach().numpy()
    # perf = np.mean(util.decimal(labs) == util.decimal(why))
    # train_perf.append(perf)
    # train_perf.append(la.norm(centroids.T[:,:,None]-centroids.T[:,None,:],2,0))
    
    clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
    D = assistants.Dichotomies(len(np.unique(this_exp.train_conditions)),
                                this_exp.task.positives, extra=0)
    PS = [D.parallelism(z.detach().numpy(), this_exp.train_conditions[idx], clf) for _ in D]
    train_PS.append(PS)
    
    loss = net.grad_step(dl, optimizer)
    
    # running_loss = 0
    
    # for i, btch in enumerate(dl):
    #     optimizer.zero_grad()
        
    #     inps, outs = btch
    #     # pred = net(inps[...,:-4],inps[...,-4:])
    #     pred = net(inps)
        
    #     # loss = nn.MSELoss()(pred, outs)
    #     loss = net.

    #     loss.backward()
    #     optimizer.step()
    
    #     running_loss += loss.item()
    train_loss.append(loss)
    print('epoch %d: %.3f'%(epoch,loss))
    # train_loss.append(running_loss/(i+1))
    # print(running_loss/(i+1))

#%%
z = net(torch.tensor(inputs))[2].detach().numpy()
# these_conds = abstract_conds
# these_conds = succ_conds
these_conds = this_exp.train_conditions
# these_conds = util.decimal(np.concatenate([this_exp.train_data[1], inputs[:,-2:]], 1))

centroids = np.stack([z[these_conds==i,:].mean(0) for i in np.unique(these_conds)])

U, S, V = la.svd(centroids-centroids.mean(0)[None,:],full_matrices=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(U[:,0],U[:,1],U[:,2],s=100,c=np.unique(these_conds))
# for i in np.unique(these_conds):
#     c = [int(i), int(np.mod(i+1,U.shape[0]))]
#     ax.plot(U[c,0],U[c,1],U[c,2],'k')

plt.title('PCA dimension: %.2f'%((np.sum(S**2)**2)/np.sum(S**4)))

#%%
z = net(torch.tensor(inputs))[2].detach().numpy()
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

cond = this_exp.train_conditions[idx]
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
    out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in this_exp.task.positives]))
    
    CCGP.append(D.CCGP(z[idx,:], cond, gclf, cntxt, twosided=True))
    
    # shattering
    d[:,i] = D.coloring(cond)
    
# dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
dclf.fit(z[idx,:], d, tol=1e-5)

z = net(torch.tensor(inputs))[2].detach().numpy()
# z = this_exp.test_data[0].detach().numpy()
# z = linreg.predict(this_exp.test_data[0])@W1.T
idx = np.random.choice(z.shape[0], n_compute, replace=False)

d_tst = np.array([D.coloring(this_exp.train_conditions[idx]) for _ in D]).T
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

#%%
n_mds = 3
n_compute = 500

fake_task = util.RandomDichotomies(num_cond,num_var,0)
fake_task.positives = task.positives

idx = np.random.choice(inputs.shape[0], n_compute, replace=False)

# z = net(torch.tensor(inputs[idx,...]))[2].detach().numpy()
z = net.enc.network[:-2](torch.tensor(inputs[idx,...])).detach().numpy()

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

#%%
N = 50
num_data = 5000
num_init = 150
num_var = 2
max_dichs = 50
init_type = 'pytorch'
# init_type = 'normal'
bias = True
# act = nn.ReLU()

y = (np.random.rand(num_data,num_var)>0.5).astype(int)

cond = util.decimal(y)
num_cond = len(np.unique(cond))
pos_conds = [np.unique(cond[y[:,i]==0]) for i in range(num_var)]

# K = int(num_cond/2) - 1 # use all but one pairing
K = int(num_cond/4) # use half the pairings

D = assistants.Dichotomies(num_cond, special=pos_conds, extra=max_dichs)
clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
dclf = assistants.LinearDecoder(N, D.ntot, svm.LinearSVC)

PS = [] # paralellism
CCGP = [] 
SD = [] # shattering dimension
PR = [] # pca dimension
CL = [] # coding level
for i in tqdm(range(num_init)):
    if init_type == 'pytorch':
        bnd = 1/np.sqrt(N)
        # bnd = 1
        W = (2*bnd*np.random.rand(num_var,N)-bnd)
        if bias:
            b = (2*bnd*np.random.rand(1,N)-bnd)
        else:
            b = 0
    elif init_type == 'normal':
        W = np.random.randn(num_var, N)
        if bias:
            b = np.random.randn(1, N)
        else:
            b = 0
    z = np.tanh(y@W+b)
    # z *= (z>0)
    
    CL.append(np.mean(z>0))
    
    U, S, _ = la.svd(z-z.mean(0)[None,:],full_matrices=False)
    PR.append(((S**2).sum()**2)/(S**4).sum())
    
    ps = []
    ccgp = []
    d = np.zeros((num_data, D.ntot))
    for i, _ in enumerate(D):
        # parallelism
        ps.append(D.parallelism(z, cond, clf))
        
        # CCGP
        ccgp.append(D.CCGP(z, cond, gclf, K))
        
        # shattering
        d[:,i] = D.coloring(cond)
    
    idx = np.random.rand(num_data)>0.5
    dclf.fit(z[idx,:], d[idx,:], tol=1e-5)
    SD.append(dclf.test(z[~idx,:], d[~idx,:]).squeeze())
    PS.append(ps)
    CCGP.append(ccgp)


# C = np.random.rand(N, N)
# W1 = la.qr(C)[0][:,:this_exp.dim_output]





