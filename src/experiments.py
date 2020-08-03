"""
Classes which implement experiments. They are what's called in the habanero
experiment scripts. They standardise my experiments.
"""

# CODE_DIR = '/home/matteo/Documents/github/repler/src/'
# SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'

import os
import pickle
import warnings
# sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la

from students import *
from assistants import *
from itertools import permutations
from sklearn import svm

#%% MNIST classification
class mnist_multiclass():
    """
    Train a feedforward network to do multiple classifications on MNIST.
    
    class_func must output an integer Q when no argument is provided.
    """
    def __init__(self, task, SAVE_DIR, N=None, H=100,
                 nonlinearity='ReLU', num_layer=1, z_prior=None,
                 bsz=64, nepoch=1000, lr=1e-4, opt_alg=optim.Adam, weight_decay=0,
                 abstracts=None, init=None, skip_dichotomies=False,
                 skip_metrics=False, dichotomy_type='general'):
        """
        Everything required to fully specify an experiment.
        
        Failure to supply the N argument will create the class in 'task only mode',
        which means that will not have a model. Call the `self.use_model` method
        to later equip it with a particular model.
        """
        
        if abstracts is None:
            abstracts = task
        # -------------------------------------------------------------------------
        # Parameters
        
        # these values index particular models (show up in the file names)
        self.dim_latent = N
        self.nonlinearity = nonlinearity
        self.init = init # optionally specify an initialisation index -- for randomising
        self.H = H # doesn't actually show up in the name -- don't change
        
        # these values specify model classes (show up in folder names)
        self.num_layer = num_layer
        self.task = task
        
        # output and abstraction dimensions
        self.dim_output = task.dim_output
        self.num_cond = abstracts.num_var
        self.dichotomy_type = dichotomy_type
        
        # optimization hyperparameters
        self.bsz = bsz
        self.nepoch = nepoch
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.opt_alg = opt_alg
        
        # flags
        self.skip_metrics = skip_metrics
        if N is None:
            # if the class is being called just to call the task
            self.task_only_mode = True
        else:
            self.task_only_mode = False
        
        # -------------------------------------------------------------------------
        # Model specification (changing this will make past experiments incompatible)
        if z_prior is None:
            latent_dist = PointMass()
        else:
            latent_dist = z_prior
        obs_dist = task.obs_distribution
        
        if self.task_only_mode:
            enc = None
            dec = None
        else:
            enc = Feedforward([784]+[self.H for _ in range(num_layer)]+[N], self.nonlinearity)
            dec = Feedforward([N, self.dim_output], [task.link])
            
        self.model = MultiGLM(enc, dec, obs_dist, latent_dist)
        
        # -------------------------------------------------------------------------
        # Import data, create the train and test sets, and the dataloader for optimisation
        digits = torchvision.datasets.MNIST(SAVE_DIR+'digits/', download=True, 
                                            transform=torchvision.transforms.ToTensor())
        valid = (digits.targets <= 8) & (digits.targets>=1)
        self.train_data = (digits.data[valid,...].reshape(-1,784).float()/252, 
                           task(digits)[valid,...])
        self.train_conditions = abstracts(digits)[valid,...]
        self.ntrain = int(valid.sum())
        
        stigid = torchvision.datasets.MNIST(SAVE_DIR+'digits/',download=True, train=False,
                                            transform=torchvision.transforms.ToTensor())
        valid = (stigid.targets<=8) & (stigid.targets>=1)
        self.test_data = (stigid.data[valid,...].reshape(-1,784).float()/252, 
                          task(stigid)[valid,...])
        self.test_conditions = abstracts(stigid)[valid,...]
        self.ntest = int(valid.sum())
    
    def use_model(self, N, init=None, H=None):
        """If the object is in task_only_mode, it can be equipped later with
        a model by specifying the particular parameters"""
        
        self.task_only_mode = (self.task_only_mode and False)
        
        # specify model
        self.dim_latent = N
        self.init = init
        enc = Feedforward([784]+[self.H for _ in range(self.num_layer)]+[N], self.nonlinearity)
        dec = Feedforward([N, self.dim_output], [self.task.link])
        
        new_model = MultiGLM(enc, dec, self.model.obs, self.model.latent)
        self.model = new_model
    
    def run_experiment(self, verbose=False):
        
        expinf = self.file_suffix()
        print('Running %s ...'%expinf)
        
        dset = torch.utils.data.TensorDataset(self.train_data[0], self.train_data[1])
        dl = torch.utils.data.DataLoader(dset, batch_size=self.bsz, shuffle=True) 
        
        optimizer = self.opt_alg(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # inference
        n_compute = 5000
        # n_dichotomy = int(spc.binom(2**self.num_cond, 2**(self.num_cond-1))/2)
        n_dichotomy = Dichotomies(self.train_conditions, self.dichotomy_type).ntot
        
        metrics = {'train_loss': np.zeros(self.nepoch),
                   'train_perf': np.zeros((self.nepoch, self.task.num_var)),
                   'train_PS': np.zeros((self.nepoch, n_dichotomy)),
                   'test_perf': np.zeros((self.nepoch, self.task.num_var)),
                   'test_PS': np.zeros((self.nepoch, n_dichotomy)),
                   'shattering': np.zeros((self.nepoch, n_dichotomy)),
                   'mean_grad': np.zeros((self.nepoch, 3)),
                   'std_grad': np.zeros((self.nepoch, 3)),
                   'train_ccgp': np.zeros((self.nepoch, n_dichotomy)),
                   'test_ccgp': np.zeros((self.nepoch, n_dichotomy)),
                   'linear_dim': np.zeros(self.nepoch)} # put all training metrics here
        
        for epoch in range(self.nepoch):
            # check each quantity before optimisation
            
            if not self.skip_metrics:
                with torch.no_grad():
                    # train error #############################################
                    idx_trn = np.random.choice(self.ntrain, n_compute, replace=False)
                    
                    pred, _, z_train = self.model(self.train_data[0][idx_trn,...])
                    # terr = (self.train_data[1][idx_trn,...] == (pred>=0.5)).sum(0).float()/n_compute
                    terr = self.task.correct(pred, self.train_data[1][idx_trn,...])/n_compute
                    # terr = terr.expand_as(torch.empty((1,self.task.num_var)))
                    # metrics['train_perf'] = np.append(metrics['train_perf'], terr, axis=0)
                    metrics['train_perf'][epoch,:] = terr
                    
                    # test error ##############################################
                    idx_tst = np.random.choice(self.ntest, n_compute, replace=False)
                    
                    pred, _, z_test = self.model(self.test_data[0][idx_tst,...])
                    # terr = (self.test_data[1][idx_tst,...] == (pred>=0.5)).sum(0).float()/n_compute
                    terr = self.task.correct(pred, self.test_data[1][idx_tst,...])/n_compute
                    # terr = terr.expand_as(torch.empty((1,self.task.num_var)))
                    # metrics['test_perf'] = np.append(metrics['test_perf'], terr, axis=0)
                    metrics['test_perf'][epoch,:] = terr
                    
                    # Dimensionality ##########################################
                    _, S, _ = la.svd(z_train.detach()-z_train.mean(1).detach()[:,None], full_matrices=False)
                    eigs = S**2
                    pr = (np.sum(eigs)**2)/np.sum(eigs**2)
                    # metrics['linear_dim'] = np.append(metrics['linear_dim'], pr)
                    metrics['linear_dim'][epoch] = pr      
                    
                    # various dichotomy-based metrics #########################
                    clf = LinearDecoder(self.dim_latent, 1, MeanClassifier)
                    dclf = LinearDecoder(self.dim_latent, n_dichotomy, svm.LinearSVC)
                    gclf = LinearDecoder(self.dim_latent, 1, svm.LinearSVC)
                    K = 2**(self.num_cond-1)-1
                    
                    # train set
                    D = Dichotomies(self.train_conditions[idx_trn,...].detach().numpy(), self.dichotomy_type) 
                    
                    # PS = np.zeros(n_dichotomy)
                    # CCGP = np.zeros(n_dichotomy)
                    d = np.zeros((n_compute, n_dichotomy))
                    for i, dic in enumerate(D):
                    #     PS[i] = D.parallelism(z_train.detach().numpy(), clf)
                    #     CCGP[i] = D.CCGP(z_train.detach().numpy(), gclf, K)
                        d[:,i] = dic
                    dclf.fit(z_train.detach().numpy(), d)
                    
                    # metrics['train_PS'] = np.append(metrics['train_PS'], PS[None,:], axis=0)
                    # metrics['train_ccgp'] = np.append(metrics['train_ccgp'], CCGP[None,:], axis=0)
                    
                    #test set
                    D = Dichotomies(self.test_conditions[idx_tst,...].detach().numpy(), self.dichotomy_type) 
                    
                    PS = np.zeros(n_dichotomy)
                    d = np.zeros((n_compute, n_dichotomy))
                    CCGP = np.zeros(n_dichotomy)
                    for i, dic in enumerate(D):
                        PS[i] = D.parallelism(z_test.detach().numpy(), clf)
                        CCGP[i] = D.CCGP(z_test.detach().numpy(), gclf, K)
                        d[:,i] = dic
                    SD = dclf.test(z_test.detach().numpy(), d).T
                    
                    # metrics['test_PS'] = np.append(metrics['test_PS'], PS[None,:], axis=0)
                    metrics['test_PS'][epoch,:] = PS
                    # metrics['test_ccgp'] = np.append(metrics['test_ccgp'], CCGP[None,:], axis=0)
                    metrics['test_ccgp'][epoch,:] = CCGP
                    # metrics['shattering'] = np.append(metrics['shattering'], SD, axis=0)
                    metrics['shattering'][epoch,:] = SD
                    
            # Actually update model #######################################
            loss = self.model.grad_step(dl, optimizer)
            
            # metrics['train_loss'] = np.append(metrics['train_loss'], loss)
            metrics['train_loss'][epoch] = loss
            
            # gradient SNR
            # means = np.zeros(3)
            # std = np.zeros(3)
            # i=0
            # for k,v in zip(self.model.state_dict().keys(), self.model.parameters()):
            #     if 'weight' in k:
            #         means[i] = (v.grad.data.mean(1)/v.data.norm(2,1)).norm(2,0).numpy()
            #         std[i] = (v.grad.data/v.data.norm(2,1,keepdim=True)).std().numpy()
            #         i+=1
            # metrics['mean_grad'] = np.append(metrics['mean_grad'], means[None,:], axis=0)
            # metrics['std_grad'] = np.append(metrics['std_grad'], std[None,:], axis=0)
            
            # print updates ############################################
            if verbose:
                print('Epoch %d: Loss=%.3f'%(epoch, -loss))
            
        self.metrics = metrics
                
    def save_experiment(self, SAVE_DIR):
        """
        Save the experimental information: model parameters, learning metrics,
        and hyperparameters. 
        """
        
        FOLDERS = self.folder_hierarchy()
        expinf = self.file_suffix()
        
        if not os.path.isdir(SAVE_DIR+FOLDERS):
            os.makedirs(SAVE_DIR+FOLDERS)
        
        # save all hyperparameters, for posterity
        all_args = {'model': str(self.model),
                    'batch_size': self.bsz,
                    'nepoch': self.nepoch,
                    'learning_rate': self.lr,
                    'optimizer': str(self.opt_alg.__name__)}
        
        if self.task.__name__ == 'RandomDichotomies':
            all_args['dichotomies'] = self.task.positives
        
        params_fname = 'parameters'+expinf+'.pt'
        metrics_fname = 'metrics'+expinf+'.pkl'
        args_fname = 'arguments'+expinf+'.npy'
        
        self.model.save(SAVE_DIR+FOLDERS+params_fname)
        with open(SAVE_DIR+FOLDERS+metrics_fname, 'wb') as f:
            pickle.dump(self.metrics, f, -1)
        with open(SAVE_DIR+FOLDERS+args_fname, 'wb') as f:
            np.save(f, all_args)
            
    def load_experiment(self, SAVE_DIR):
        """
        Loads model parameters from the files that ought to exist, if they were
        saved with the save_experiments method.
        """
        
        FOLDERS = self.folder_hierarchy()
        expinf = self.file_suffix()
        
        params_fname = 'parameters'+expinf+'.pt'
        metrics_fname = 'metrics'+expinf+'.pkl'
        args_fname = 'arguments'+expinf+'.npy'
        
        self.model.load(SAVE_DIR+FOLDERS+params_fname)
        with open(SAVE_DIR+FOLDERS+metrics_fname,'rb') as f:
            metrics = pickle.load(f)
        args = np.load(SAVE_DIR+FOLDERS+args_fname, allow_pickle=True).item()
        
        return self.model, metrics, args
    
    # file I/O functions for this experiment
    # I do this in order to standardise everything within this experiment
    def folder_hierarchy(self):
        FOLDERS = 'results/mnist/'
        if self.num_layer != 1:
            FOLDERS += str(self.num_layer+1)+'layer/'
        FOLDERS += self.task.__name__ + '/'
        FOLDERS += self.model.obs.name() + '/'
        if self.model.latent is not None:
            FOLDERS += self.model.latent.name() + '/'   
        if self.weight_decay > 0:
            FOLDERS += 'L2reg/'
        
        return FOLDERS
        
    def file_suffix(self):
        if self.init is None:
            return '_N%d_%s'%(self.dim_latent, self.nonlinearity)
        else:
            return '_init%d_N%d_%s'%(self.init, self.dim_latent, self.nonlinearity)
    
    
    
    
    
    
    
    
    