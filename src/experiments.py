"""
Functions which implement experiments. They are what's called in the habanero
experiment scripts.
"""

# CODE_DIR = '/home/matteo/Documents/github/repler/src/'
# SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'

import os
import pickle
# sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
# import scipy

from students import *
from assistants import *

#%% MNIST classification
def mnist_multiclass(N, class_func, SAVE_DIR, verbose=False):
    """
    Train a feedforward network to do multiple classifications on MNIST.
    
    Make sure that Q matches the dimension of class_func output.
    """
    # parameters
    H = 500
    Q = class_func()
    
    noise = True 
    nonlinearity = 'ReLU'
    include_kl = 'no'
    
    bsz = 64 
    nepoch = 10000
    lr = 1e-4
    
    opt_alg = optim.Adam
    # opt_alg = optim.Adagrad
    
    expinf = '_N%d_H%d_%s_%s-kl'%(N,H,nonlinearity,include_kl)
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('Running %s ...'%expinf)
    # -------------------------------------------------------------------------
    # Model/Encoder specification
    latent_dist = GausId(N)
    obs_dist = Bernoulli(Q)
    
    enc = Feedforward([784, H, N], [nonlinearity, None])
    dec = Feedforward([N, Q], ['Sigmoid'])
    vae = VAE(enc, dec, latent_dist, obs_dist)
    
    # data
    digits = torchvision.datasets.MNIST(SAVE_DIR+'digits/',download=False, 
                                        transform=torchvision.transforms.ToTensor())
    classes = class_func(digits)
    digits = torch.utils.data.TensorDataset(digits.data.float(), classes)
    
    stigid = torchvision.datasets.MNIST(SAVE_DIR+'digits/',download=False, train=False,
                                        transform=torchvision.transforms.ToTensor())
    classes = class_func(stigid)
    stigid = torch.utils.data.TensorDataset(stigid.data.float(), classes)
    
    dl = torch.utils.data.DataLoader(digits, batch_size=bsz, shuffle=True)
    
    # inference
    metrics = {'train_loss': np.zeros(0),
               'test_err': np.zeros((0,Q))} # put all training metrics here
    print('Starting inference')
    optimizer = opt_alg(vae.parameters(), lr=lr)
    for epoch in range(nepoch):
        if include_kl == 'anneal':
            if epoch>50:
                beta = np.exp((epoch-300)/30)/(1+np.exp((epoch-300)/30))
        elif include_kl == 'always':
            beta = 1
        else:
            beta = 0
        
        loss = vae.grad_step(dl, optimizer, beta)
        
        metrics['train_loss'] = np.append(metrics['train_loss'], loss)
        
        # test error
        idx = np.random.choice(10000, 1000, replace=False)
        
        pred = vae(stigid.tensors[0][idx,:,:].reshape(-1,784).float()/252)[0]
        terr = (stigid.tensors[1][idx,:] == (pred>=0.5)).sum(0).float()/1000
        metrics['test_err'] = np.append(metrics['test_err'], terr[None,:], axis=0)
        
        if verbose:
            print('Epoch %d: ELBO=%.3f'%(epoch, -loss))
    
    # -------------------------------------------------------------------------
    # save
    FOLDERS = folder_hierarchy(class_func, obs_dist, latent_dist)
    
    if not os.path.isdir(SAVE_DIR+FOLDERS):
        os.makedirs(SAVE_DIR+FOLDERS)
    
    # save all hyperparameters, for posterity
    all_args = {'enc_specs': str(enc),
                'dec_specs': str(dec),
                'noise': noise,
                'batch_size': bsz,
                'nepoch': nepoch,
                'learning_rate': lr,
                'optimizer': str(type(opt_alg))}
    
    params_fname = 'parameters'+expinf+'.pt'
    metrics_fname = 'metrics'+expinf+'.pkl'
    args_fname = 'arguments'+expinf+'.npy'
    
    vae.save(SAVE_DIR+FOLDERS+params_fname)
    with open(SAVE_DIR+FOLDERS+metrics_fname, 'wb') as f:
        pickle.dump(metrics, f, -1)
    with open(SAVE_DIR+FOLDERS+args_fname, 'wb') as f:
        np.save(f, all_args)

    print('ALL DONE! THANK YOU VERY MUCH FOR YOUR PATIENCE!!!!!!!')

# The `class functions' go here: functions which take in `digits' and output 
# binary vectors indicating Q classes. They should also have an option to take
# no argument, and return the number of classes (Q). 
def parity_magnitude(digits=None):
    """Compute the parity and magnitude of digits"""
    if digits is None:
        return 2
    else:
        parity = np.mod(digits.targets, 2).float()
        magnitude = (digits.targets>=5).float()
        return torch.cat((parity[:,None], magnitude[:,None]), dim=1)

# file I/O functions
def folder_hierarchy(class_func, obs_dist, latent_dist):
    FOLDERS = 'results/'
    FOLDERS += class_func.__name__ + '/'
    FOLDERS += obs_dist.name() + '/'   
    FOLDERS += latent_dist.name() + '/'   
    
    return FOLDERS
    
# def expinfo(N, H):
    
    
    
    
    
    
    
    
    
    