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
import plotting as dicplt

#%% data
dim_inp = 32 # dimension per variable
num_data = 5000 # total
noise = 0.05

switch_fraction = 0.1

input_task = util.StandardBinary(3)
# output_task = util.RandomDichotomies(d=[(0,1,6,7), (0,2,5,7)]) # xor of first two
output_task = util.RandomDichotomies(d=[(0,3,5,6)]) # 3d xor

inp_condition = np.random.choice(2**3, num_data)
var_bit = input_task(inp_condition).numpy().T
action_outcome = var_bit[:2,:]
context = var_bit[2,:]

stimulus = util.decimal(action_outcome.T).astype(int)
stimulus[context==1] = np.mod(stimulus[context==1]+1,4) # effect of context

means_pos = np.random.randn(2, dim_inp)
means_neg = np.random.randn(2, dim_inp)
stim_pattern = np.random.randn(4, dim_inp)

mns = (means_pos[:,None,:]*action_outcome[:,:,None]) + (means_neg[:,None,:]*(1-action_outcome[:,:,None]))

ao_t = np.reshape(mns.transpose((0,2,1)), (dim_inp*2,-1)).T
s_t = stim_pattern[stimulus,:]

prev_trial = np.arange(num_data)
next_trial = np.arange(num_data)
next_trial[context==1] = np.random.permutation(next_trial[context==1])
next_trial[context==0] = np.random.permutation(next_trial[context==0])

cat_inp = np.concatenate([s_t[prev_trial,:], ao_t[prev_trial,:], s_t[next_trial,:]], axis=1)

inputs = torch.tensor(cat_inp + np.random.randn(num_data, 4*dim_inp)*noise).float()

# generate outputs
outputs = torch.tensor(action_outcome[:,next_trial].T)


#%%
U, S, V = la.svd(inputs.numpy()-inputs.numpy().mean(0)[None,:],full_matrices=False)

# colorby = inp_condition
colorby = stimulus
# colorby = np.isin(inp_condition[:-1], input_task.positives[2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(U[:,0],U[:,1],U[:,2],s=100, c=colorby)
# for i in np.unique(these_conds):
#     c = [int(i), int(np.mod(i+1,U.shape[0]))]
#     ax.plot(U[c,0],U[c,1],U[c,2],'k')
util.set_axes_equal(ax)

# plt.title('PCA dimension: %.2f'%((np.sum(S**2)**2)/np.sum(S**4)))

#%%
N = 200

# net = students.Feedforward([inputs.shape[1],100,2],['ReLU',None])
# net = students.MultiGLM(students.Feedforward([inputs.shape[1], N], ['ReLU']),
#                         students.Feedforward([N, targets.shape[1]], [None]),
#                         students.GausId(targets.shape[1]))
net = students.MultiGLM(students.Feedforward([dim_inp*4,N,N], ['ReLU','ReLU']),
                        students.Feedforward([N,2], [None]),
                        students.Bernoulli(2))
# net = students.MultiGLM(students.Feedforward([inputs.shape[1],N], ['ReLU']),
#                         students.Feedforward([N, p], [None]),
#                         students.Categorical(p))

n_trn = int(0.8*num_data)
trn = np.random.choice(num_data,n_trn,replace=False)
tst = np.setdiff1d(range(num_data),trn)

optimizer = optim.Adam(net.enc.parameters(), lr=1e-4)
dset = torch.utils.data.TensorDataset(inputs[trn,:].float(),
                                      outputs[trn,:].float())
dl = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)

n_compute = np.min([len(tst),1000])

train_loss = []
# train_perf = []
inp_PS = []
out_PS = []
test_loss = []
lin_dim = []
min_dist = []
for epoch in tqdm(range(10000)):
    # check whether outputs are in the correct class centroid
    idx2 = np.random.choice(tst, n_compute, replace=False)
    zee, _, z = net(inputs[idx2,:].float())
    tst_ls = -net.obs.distr(zee).log_prob(outputs[idx2,:].float()).mean()
    # tst_ls = ((zee-outputs[idx2,:].float())**2).sum(1).mean()
    test_loss.append(tst_ls)
    
    idx = np.random.choice(trn, n_compute, replace=False)
    zee, _, z = net(inputs[idx,:].float())
    # why = successors[idx,...].detach().numpy()
    
    _, S, _ = la.svd(z.detach().numpy()-z.detach().numpy().mean(0)[None,:],full_matrices=False)
    lin_dim.append((np.sum(S**2)**2)/np.sum(S**4))
    
    # centroids = np.stack([z[this_exp.train_conditions[idx]==i,:].detach().mean(0) \
    #                       for i in np.unique(this_exp.train_conditions[idx])])
    # dist_to_class = np.sum((zee[:,:,None].detach().numpy() - centroids.T)**2,1)
    # nearest = dist_to_class.argmin(1)
    # labs = this_exp.task(torch.tensor(nearest)).detach().numpy()
    # perf = np.mean(util.decimal(labs) == util.decimal(why))
    # train_perf.append(perf)
    # train_perf.append(la.norm(centroids.T[:,:,None]-centroids.T[:,None,:],2,0))
    
    clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
    D = assistants.Dichotomies(len(np.unique(inp_condition)),
                                output_task.positives, extra=2)
    PS = [D.parallelism(z.detach().numpy(), inp_condition[idx], clf) for _ in D]
    out_PS.append(PS)
    
    # D = assistants.Dichotomies(len(np.unique(inp_condition)),
    #                             input_task.positives, extra=2)
    # PS = [D.parallelism(z.detach().numpy(), inp_condition[idx], clf) for _ in D]
    # inp_PS.append(PS)
    
    D = assistants.Dichotomies(len(np.unique(inp_condition)),
                                input_task.positives, extra=0)
    PS = [D.parallelism(z.detach().numpy(), inp_condition[idx], clf) for _ in D]
    inp_PS.append(PS)
    
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
    # print('epoch %d: %.3f'%(epoch,loss))
    # train_loss.append(running_loss/(i+1))
    # print(running_loss/(i+1))
    
#%%
plt.figure()
epochs = range(1,len(inp_PS)+1)
# plt.plot(range(1,len(inp_PS)+1),out_PS)
# plt.semilogx()

trn = []
for dim in range(output_task.dim_output):
    thisone = plt.plot(epochs, np.array(out_PS)[...,dim])[0]
    trn.append(thisone)
    plt.semilogx()

# remainder
untrn = plt.plot(epochs, np.array(out_PS)[...,output_task.dim_output:].mean(1),zorder=0)[0]
plt.semilogx()

plt.xlabel('epoch', fontsize=15)
plt.title('Output PS')
plt.legend(trn + [untrn], ['Var %d'%(n+1) for n in range(output_task.dim_output)] + ['Non-output dichotomies'])

plt.figure()
epochs = range(1,len(inp_PS)+1)
# plt.plot(range(1,len(inp_PS)+1),out_PS)
# plt.semilogx()

trn = []
for dim in range(input_task.dim_output):
    thisone = plt.plot(epochs, np.array(inp_PS)[...,dim])[0]
    trn.append(thisone)
    plt.semilogx()

# remainder
untrn = plt.plot(epochs, np.array(inp_PS)[...,input_task.dim_output:].mean(1),zorder=0)[0]
plt.semilogx()

plt.xlabel('epoch', fontsize=15)
plt.title('Input PS')
plt.legend(trn + [untrn], ['Var %d'%(n+1) for n in range(input_task.dim_output)] + ['Non-input dichotomies'])

# plt.figure()
# plt.plot(range(1,len(inp_PS)+1),train_loss)
# # plt.plot(range(1,len(inp_PS)+1),test_loss)
# plt.semilogx()

# %%
rep = net.enc.network # penultimate layer
# rep = net.enc.network[:-2] # antepenultimate layer
# rep = net.enc.network[:-4] # you get the picture

# z = inputs.numpy()
z = rep(inputs.float()).detach().numpy()
# z = net.enc.network[:-2](inputs.float()).detach().numpy()
# z = net.enc.network[:-4](inputs.float()).detach().numpy()
N = z.shape[1]
max_dichs = 50 # the maximum number of untrained dichotomies to test 

this_task = input_task
# this_task = output_task

all_PS = []
all_CCGP = []
all_CCGP_ = []
CCGP_out_corr = []
mut_inf = []
all_SD = []
indep = []

indep.append(this_task.subspace_information())
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
Q = 3
D_fake = assistants.Dichotomies(num_cond, this_task.positives, extra=7000)
mi = np.array([this_task.information(p) for p in D_fake])
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
# K = int(num_cond/4) # use half the pairings

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
    out_corr.append(np.array([[(2*np.isin(p,c)-1).mean() for c in cntxt] for p in this_task.positives]))
    
    CCGP.append(D.CCGP(z[idx,:], cond, gclf, cntxt, twosided=True))
    
    # shattering
    d[:,i] = D.coloring(cond)
    
# dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
dclf.fit(z[idx,:], d, tol=1e-5)

# z = inputs.numpy()
z = rep(inputs.float()).detach().numpy()
# z = this_exp.test_data[0].detach().numpy()
# z = linreg.predict(this_exp.test_data[0])@W1.T
idx = np.random.choice(z.shape[0], n_compute, replace=False)

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
    scat = plt.scatter(xfoo, yfoo, s=10, c=cfoo, zorder=10, cmap=mutinfo_cmap)
else:
    scat = plt.scatter(xfoo, yfoo, s=10, c=(0.5,0.5,0.5), zorder=10)
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
inps = []
for i,d in enumerate(input_task.positives):
    # # n = np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
    # #               for p in pos_conds])[0][0]
    # inps.append(plt.scatter(xfoo[[i,i+ndic,i+2*ndic]], yfoo[[i,i+ndic,i+2*ndic]], 
    #            s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))
    n = np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                  for p in pos_conds])[0][0]
    inps.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='d',
                s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

outs = []
for i,d in enumerate(output_task.positives):
    n = np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                  for p in pos_conds])[0][0]
    outs.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], marker='s',
                s=70, linewidths=3, facecolors='none', edgecolors=cm.get_cmap(var_cmap)(i)))

# baba = np.arange(len(xfoo)//ndic)*ndic
# par = plt.scatter(xfoo[baba], yfoo[baba], 
#                   marker='o', edgecolors='r', s=60, facecolors='none', linewidths=3)
# mag = plt.scatter(xfoo[baba+1], yfoo[baba+1], 
#                   marker='o', edgecolors='g', s=60, facecolors='none', linewidths=3)
# xor = plt.scatter(xfoo[baba+2], yfoo[baba+2], 
#                   marker='o', edgecolors='b', s=60, facecolors='none', linewidths=3)

# plt.legend([par,mag, xor], ['Var1', 'Var2', 'XOR'])
plt.legend(inps+outs, ['input %d'%(i+1) for i in range(len(inps))] + ['output %d'%(i+1) for i in range(len(outs))])

# plt.legend(outs, ['output %d'%(i+1) for i in range(len(outs))])
#%%
# mask = (R.max(2)==1) # context must be an output variable   
# mask = (np.abs(R).sum(2)==0) # context is uncorrelated with either output variable
# mask = (np.abs(R).sum(2)>0) # context is correlated with at least one output variable
mask = ~np.isnan(R).max(2) # context is uncorrelated with the tested variable

almost_all_CCGP = util.group_mean(np.array(all_CCGP).squeeze(), mask)
# include = (mask.sum(-1)>0)
# np.sum(np.array(all_CCGP)*mask[...,None],2).squeeze()/mask.sum(-1)

# PS = np.array(all_PS).mean(0)
PS = util.group_mean(np.array(all_PS), mask.sum(-1)>0, axis=0)
CCGP = util.group_mean(almost_all_CCGP, mask.sum(-1)>0, axis=0)
SD = util.group_mean(np.array(all_SD), mask.sum(-1)>0, axis=0)


output_dics = []
for d in output_task.positives:
    output_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
                                for p in pos_conds])[0][0])
input_dics = []
for d in input_task.positives:
    input_dics.append(np.where([(list(p) == list(d)) or (list(np.setdiff1d(range(num_cond),p))==list(d))\
              for p in pos_conds])[0][0])

dicplt.dichotomy_plot(PS, CCGP, SD, input_dics=input_dics, output_dics=output_dics, 
                      other_dics=[pos_conds.index((0,2,5,7))], out_MI=out_MI.mean(0))

#%%
U, S, V = la.svd(inputs.numpy()-inputs.numpy().mean(0)[None,:],full_matrices=False)

centroids = np.stack([U[inp_condition==c,:].mean(0) for c in np.unique(inp_condition)])

# colorby = inp_condition
# colorby = util.decimal(outputs).numpy()
colorby = np.isin(np.unique(inp_condition), pos_conds[17])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],'o',s=1000, c=colorby)
# for i in np.unique(these_conds):
#     c = [int(i), int(np.mod(i+1,U.shape[0]))]
#     ax.plot(U[c,0],U[c,1],U[c,2],'k')
util.set_axes_equal(ax)

plt.title('PCA dimension: %.2f'%((np.sum(S**2)**2)/np.sum(S**4)))


#%%
n_mds = 3
n_compute = 500

model = net.enc.network # penultimate layer
# model = net.enc.network[:-2] # antepenultimate layer
# model = net.enc.network[:-4] # you get the picture

fake_task = util.RandomDichotomies(num_cond,1,0)
# fake_task.positives = output_task.positives
fake_task.positives = [pos_conds[25]]
# fake_task.positives = [(0,1,2,6)]

idx = np.random.choice(inputs.shape[0], n_compute, replace=False)

z = model(inputs[idx,...].float()).detach().numpy()

# ans = this_exp.train_data[1][idx,...]
# ans = output_task(this_exp.train_conditions)[idx]

# cond = util.decimal(ans)
# cond = this_exp.train_conditions[idx]
cond = inp_condition[idx]

# colorby = cond
# colorby = this_exp.train_conditions[idx]
point_col = util.decimal(fake_task(cond))

centr_col = util.decimal(fake_task(np.unique(cond)))


mds = manifold.MDS(n_components=n_mds)

# emb = mds.fit_transform(la.norm(z.T[:,:,None] - z.T[:,None,:],axis=0))
# emb = mds.fit(z)
emb = mds.fit_transform(np.round(z,2))

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
        
        ax.scatter(U[:,0],U[:,1],U[:,2],s=50, marker='s',c=centr_col)
        scat = ax.scatter(emb[:,0],emb[:,1], emb[:,2], c=point_col)
        
        util.set_axes_equal(ax)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # plt.xticks([])
        # plt.yticks([])
        # plt.zticks([])
        # plt.legend(np.unique(cond), np.unique(cond))
        cb = plt.colorbar(scat,
                          ticks=np.unique(point_col),
                          drawedges=True,
                          values=np.unique(point_col))
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

