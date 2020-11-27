"""
Multi-classification experiments.

Command-line arguments:
    -v [--verbose]  set verbose
    -n val          number of neurons (int)
    -t str          which task, must be (case-sensitive) name of a Task class
    -m              skip computation of metrics
"""

#%%
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

import getopt

import math
import pickle
import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from sklearn import svm, manifold, linear_model

import experiments # my code packages
import util
import students
import assistants

#%% parse arguments (need to remake this for each kind of experiment)
allargs = sys.argv

arglist = allargs[1:]

unixOpts = "vmdn:i:h:c:q:o:f:g:r:"
gnuOpts = ["verbose"]

opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

n_out = 100
verbose, skip_metrics, N, init = False, False, None, None # defaults
num_class, num_dich, ovlp = 8, 2, 0
input_type = 'task_inp'
output_type = 'rotated'
rot = 0.0
for op, val in opts:
    if op in ('-v','--verbose'):
        verbose = True
    if op in ('-n'):
        N = int(val)
    if op in ('-i'):
        init = int(val)
    if op in ('-m'):
        skip_metrics = True
    if op in ('-h'):
        n_out = int(val)
    if op in ('-c'):
        num_class = int(val)
    if op in ('-q'):
        num_dich =  int(val)
    if op in ('-o'):
        ovlp = int(val)
    if op in ('-f'):
        input_type = val
    if op in ('-g'):
        output_type = val
    if op in ('-r'):
        rot = float(val)

#%% run experiment (put the code you want to run here!)
# task = util.ParityMagnitude()
# task = util.ParityMagnitudeEnumerated()
# task = util.Digits()
# task = util.DigitsBitwise()
# task = util.ParityMagnitudeFourunit()
task = util.RandomDichotomies(num_class, num_dich, overlap=ovlp)

sample_dichotomies = num_dich
# sample_dichotomies = None

nepoch = 1000

this_exp = experiments.random_patterns(task, SAVE_DIR, 
                                       num_class=num_class,
                                       dim=100,
                                       var_means=1)

nonlinearity = 'ReLU'
# nonlinearity = 'Tanh'
# nonlinearity = 'Sigmoid'
# nonlinearity = 'LeakyReLU'

print('- - - - - - - - - - - - - - - - - - - - - - - - - - ')
#%%    

num_var = task.dim_output
p = 2**num_var
num_dat = len(this_exp.train_conditions)

eps1 = torch.tensor(np.random.randn(num_dat,n_out)*0.1)
# eps2 = torch.tensor(np.random.randn(num_dat,n_out)*0.1)


# data = {'task_inp': this_exp.train_data[0],
#         'task_out': this_exp.train_data[1],
#         'factored': this_exp.train_data[1]) +eps1}
# if (n_out>=p):
#     onehot = assistants.Indicator(p,p)(torch.tensor(util.decimal(this_exp.train_data[1].numpy())).int())
#     data['flat'] = onehot@W2.T+eps2

# inputs = data[input_type]
# targets = data[output_type]

embedding = util.ContinuousEmbedding(n_out, rot)

output_type = 'rotated%.1f'%rot
inputs = this_exp.train_data[0]
targets = embedding(this_exp.train_data[1]) #+eps1

all_args = {'inputs': inputs,
            'outputs': targets,
            'W1': embedding.basis,
            'dichotomies': this_exp.task.positives}

#%%
net = students.MultiGLM(students.Feedforward([inputs.shape[1],N,N], ['ReLU','ReLU']),
                        students.Feedforward([N,targets.shape[1]], [None]),
                        students.GausId(targets.shape[1]))
# net = students.MultiGLM(students.Feedforward([inputs.shape[1],N,N], ['ReLU','ReLU']),
#                         students.Feedforward([N,targets.shape[1]], [None]),
#                         students.Bernoulli(targets.shape[1]))

optimizer = optim.Adam(net.parameters(), lr=1e-3)
dset = torch.utils.data.TensorDataset(inputs.float(),targets.float())
dl = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)

metrics = {'train_loss': [],
           'distances': [],
           'PS': [],
           'SD': [],
           'CCGP': [],
           'PR': [],
           'sparsity': []}
for epoch in range(nepoch):
    # check whether outputs are in the correct class centroid
    idx = np.random.choice(inputs.shape[0], 5000, replace=False)
    z = net(inputs[idx,:].float())[2].detach().numpy()
    
    centroids = np.stack([z[this_exp.train_conditions[idx]==i,:].mean(0) \
                          for i in np.unique(this_exp.train_conditions[idx])])
    # dist_to_class = np.sum((zee[:,:,None].detach().numpy() - centroids.T)**2,1)
    # nearest = dist_to_class.argmin(1)
    # labs = this_exp.task(torch.tensor(nearest)).detach().numpy()
    # perf = np.mean(util.decimal(labs) == util.decimal(why))
    # train_perf.append(perf)
    metrics['distances'].append(la.norm(centroids.T[:,:,None]-centroids.T[:,None,:],2,0))
    
    U, S, _ = la.svd(z-z.mean(0)[None,:],full_matrices=False)
    metrics['PR'].append(((S**2).sum()**2)/(S**4).sum())
    
    metrics['sparsity'].append(np.mean(z>0))
    
    K = int(num_class/4) # use half the pairings

    D = assistants.Dichotomies(len(np.unique(this_exp.train_conditions)),
                               this_exp.task.positives, extra=sample_dichotomies)
    
    clf = assistants.LinearDecoder(N, 1, assistants.MeanClassifier)
    gclf = assistants.LinearDecoder(N, 1, svm.LinearSVC)
    dclf = assistants.LinearDecoder(N, D.ntot, svm.LinearSVC)
    
    cond = this_exp.train_conditions[idx]
    ps = []
    ccgp = []
    d = np.zeros((z.shape[0], D.ntot))
    for i, _ in enumerate(D):
        # parallelism
        ps.append(D.parallelism(z, cond, clf))
        
        # CCGP
        ccgp.append(D.CCGP(z, cond, gclf, K))
        
        # shattering
        d[:,i] = D.coloring(cond)
    
    idx = np.random.rand(z.shape[0])>0.5
    dclf.fit(z[idx,:], d[idx,:], tol=1e-5)
    metrics['SD'].append(dclf.test(z[~idx,:], d[~idx,:]).squeeze())
    metrics['PS'].append(ps)
    metrics['CCGP'].append(ccgp)
    
    loss = net.grad_step(dl, optimizer)
    if verbose:
        print('Epoch %d: %.2f'%(epoch, loss))
    
    metrics['train_loss'].append(loss)


#%%
FOLDERS = 'continuous/%d_%d/%s/%s/'%(num_class,num_dich, input_type, output_type)
if init is not None:
    exp_inf = '_%d_%d_init%d'%(N, n_out, init)
else:
    exp_inf = '_%d_%d'%(N, n_out)

if not os.path.isdir(SAVE_DIR+FOLDERS):
    os.makedirs(SAVE_DIR+FOLDERS)
    
net.save(SAVE_DIR+FOLDERS+'parameters'+exp_inf+'.pt')
with open(SAVE_DIR+FOLDERS+'metrics'+exp_inf+'.pkl', 'wb') as f:
    pickle.dump(metrics, f, -1)
with open(SAVE_DIR+FOLDERS+'args'+exp_inf+'.pkl', 'wb') as f:
    pickle.dump(all_args, f, -1)

#%%
print('ALL DONE! THANK YOU VERY MUCH FOR YOUR PATIENCE!!!!!!!')
print(':' + ')'*12)


