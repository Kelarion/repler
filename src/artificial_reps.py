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
import scipy.special as spc

# import umap
from cycler import cycler

import students
from assistants import *
import experiments as exp
import util

#%%
abstract_variables = util.DigitsBitwise()
# abstract_variables = util.ParityMagnitude()
# abstract_variables = util.RandomDichotomies(2)

# task = util.RandomDichotomies(8,2,0)
task = util.ParityMagnitude()

Q = abstract_variables.num_var

dic_type = 'general'
# dic_type = 'simple'

# this_exp = exp.mnist_multiclass(task, SAVE_DIR, abstracts=abstract_variables)
this_exp = exp.random_patterns(task, SAVE_DIR, 
                                num_class=8, 
                                dim=100, 
                                var_means=1,
                                abstracts=abstract_variables)

#%% make factorized representation
N = 100

C = np.random.rand(N, N)
W1 = la.qr(C)[0][:,:this_exp.dim_output]

targ = ((2*this_exp.train_data[1].numpy()-1)*10)@W1.T
b = 0.8*targ.min()

linreg = linear_model.LinearRegression()
# linreg.fit(this_exp.train_data[0],this_exp.train_data[1].numpy()@W1.T)
linreg.fit(this_exp.train_data[0], targ-b)

# W = W1@linreg.coef_
# b = (W1@linreg.intercept_)[:,None]

# targ = this_exp.train_data[1]@W1.T
# targ -= 0.8*targ.min()
# targ *= targ>=0

# # lin = nn.Linear(this_exp.dim_input, N)
# # lin = students.Feedforward([this_exp.dim_input, N, N],['ReLU','ReLU'])
# lin = students.Feedforward([this_exp.dim_input, N],['ReLU'])

# new_dset = torch.utils.data.TensorDataset(this_exp.train_data[0], targ)
# dl = torch.utils.data.DataLoader(new_dset, batch_size=64, shuffle=True)

# optimizer = this_exp.opt_alg(lin.parameters(), lr=1e-3)

# train_loss = []
# for epoch in range(10):
#     running_loss = 0
#     for i, (x,y) in enumerate(dl):
#         optimizer.zero_grad()
        
#         pred = lin(x)
#         loss = nn.L1Loss()(y, pred)
        
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         # running_error += terr.item()
        
#     train_loss.append(running_loss/(i+1))
#     print('Epoch %d: loss=%.3f'%(epoch, running_loss/(i+1)))

#%%
n_mds = 2
n_compute = 500

idx = np.random.choice(this_exp.train_data[0].shape[0], n_compute, replace=False)

z = (W@(this_exp.train_data[0][idx,...].detach().numpy().T) + b).T
# z = lin(this_exp.train_data[0])[idx,...].detach().numpy()
# z = targ[idx,...]
# ans = this_exp.train_conditions[idx,...]
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

z = (W@(this_exp.train_data[0].detach().numpy().T) + b).T
# z = lin(this_exp.train_data[0]).detach().numpy()
# z = targ
# z = this_exp.train_data[0]
n_compute = np.min([5000, z.shape[0]])

idx = np.random.choice(z.shape[0], n_compute, replace=False)
idx_tst = idx[::4] # save 1/4 for test set
idx_trn = np.setdiff1d(idx, idx_tst)

ans = this_exp.train_conditions
cond = util.decimal(ans)

# Loop over dichotomies
D = Dichotomies(ans, dic_type)
clf = LinearDecoder(z.shape[1], 1, MeanClassifier)
gclf = LinearDecoder(z.shape[1], 1, svm.LinearSVC)
dclf = LinearDecoder(z.shape[1], D.ntot, svm.LinearSVC)
# clf = LinearDecoder(this_exp.dim_input, 1, MeanClassifier)
# gclf = LinearDecoder(this_exp.dim_input, 1, svm.LinearSVC)
# dclf = LinearDecoder(this_exp.dim_input, D.ntot, svm.LinearSVC)

K = 2**(this_exp.num_cond-1) - 1 # use all but one pairing
# K = 2**(this_exp.num_cond-2) # use half the pairings

PS = np.zeros(D.ntot)
CCGP = np.zeros(D.ntot)
d = np.zeros((n_compute, D.ntot))
pos_conds = []
for i, coloring in enumerate(D):
    pos = np.unique(D.cond[coloring]).astype(int)
    neg = np.unique(D.cond[~coloring])
    pos_conds.append(pos)
    print('Dichotomy %d...'%i)
    # parallelism
    PS[i] = D.parallelism(z, clf)
    
    # CCGP
    CCGP[i] = D.CCGP(z, gclf, K)
    
    # shattering
    d[:,i] = coloring[idx]
    
# dclf.fit(z[idx_trn,:], d[np.isin(idx, idx_trn),:], tol=1e-5, max_iter=5000)
dclf.fit(z[idx,:], d, tol=1e-5, max_iter=5000)

z = (W@(this_exp.test_data[0].detach().numpy().T) + b).T
# z = lin(this_exp.test_data[0]).detach().numpy()
# z = this_exp.test_data[1]@W1.T
# z -= 0.8*z.min()
# z *= z>=0
# z = this_exp.test_data[0]
ans = this_exp.test_conditions
d_tst = np.array([c for c in Dichotomies(ans, dic_type)]).T

SD = dclf.test(z, d_tst).squeeze()

#%% plot PS and CCGP
ndic = len(PS)

xfoo = np.repeat([0,1,2],ndic).astype(int) + np.random.randn(ndic*3)*0.1
yfoo = np.concatenate((PS, CCGP, SD))

plt.scatter(xfoo, yfoo, s=12, c=(0.5,0.5,0.5))
plt.xticks([0,1,2], labels=['PS', 'CCGP', 'Shattering'])
plt.ylabel('PS or Cross-validated performance')

# highlight special dichotomies
par = plt.scatter(xfoo[[20,20+ndic,20+2*ndic]], yfoo[[20,20+ndic,20+2*ndic]], 
                  marker='o', edgecolors='r', s=60, facecolors='none', linewidths=3)
mag = plt.scatter(xfoo[[0, ndic, 2*ndic]], yfoo[[0,ndic,2*ndic]], 
                  marker='o', edgecolors='g', s=60, facecolors='none', linewidths=3)
# other = plt.scatter(xfoo[[9,9+ndic,9+2*ndic]], yfoo[[9,9+ndic,9+2*ndic]], 
#                     marker='o', edgecolors='b', s=60, facecolors='none', linewidths=3)
# anns = []
# for d in this_exp.task.positives:
#     n = np.where([(list(p) == list(d-1)) or (list(np.setdiff1d(range(8),p))==list(d-1))\
#                   for p in pos_conds])[0][0]
#     anns.append(plt.scatter(xfoo[[n,n+ndic,n+2*ndic]], yfoo[[n,n+ndic,n+2*ndic]], 
#                 marker='o', s=60, linewidths=3))

# par = plt.scatter(xfoo[[9,9+ndic,9+2*ndic]], yfoo[[9,9+ndic,9+2*ndic]], 
#                   marker='o', edgecolors='r', s=60, facecolors='none', linewidths=3)
# mag = plt.scatter(xfoo[[27, 27+ndic, 27+2*ndic]], yfoo[[27,27+ndic,27+2*ndic]], 
#                   marker='o', edgecolors='g', s=60, facecolors='none', linewidths=3)

# plt.legend(anns, ['var %d'%(i+1) for i in range(len(anns))])
# plt.legend([par,mag], ['{3,4,7,8}', '{2,3,6,7}'])
