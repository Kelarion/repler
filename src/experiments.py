"""
Classes which implement experiments. They are what's called in the habanero
experiment scripts. They standardise my experiments with a Byzantine web of 
class inheritance and exchangeable modules. Not for human consumption.
"""

import os
import pickle
import warnings

import torch
import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc

# this is my code base, this assumes that you can access it
from students import *
from assistants import *
from itertools import permutations
from sklearn import svm, linear_model

#%% Multi-classification tasks
class MultiClassification():
    """
    Basic class for multi-classification experiments. To make an instance of such an experiment,
    make a child class and define the `load_data` method. This contains all the methods
    to run the experiment, and save and load it.
    """
    def __init__(self, task, SAVE_DIR, N=None, H=100,
                 nonlinearity='ReLU', num_layer=1, z_prior=None,
                 bsz=64, nepoch=1000, lr=1e-4, opt_alg=optim.Adam, weight_decay=0,
                 init=None, skip_metrics=False, sample_dichotomies=0,
                 decoder=None, init_from_saved=False, good_start=False, init_coding=None):
        """
        Everything required to fully specify an experiment.
        
        Failure to supply the N argument will create the class in 'task only mode',
        which means that will not have a model. Call the `self.use_model` method
        to later equip it with a particular model.
        """
        
        # if abstracts is None:
        #     self.abstracts = task
        # else:
        #     self.abstracts = abstracts
        # -------------------------------------------------------------------------
        # Parameters
        
        # these values index particular models (show up in the file names)
        self.dim_latent = N
        self.nonlinearity = nonlinearity
        self.init = init # optionally specify an initialisation index -- for randomising
        self.H = H # doesn't actually show up in the name -- don't change
        
        # these values specify model/task classes (show up in folder names)
        self.base_dir = 'results/' # append at will
        self.num_layer = num_layer
        self.task = task
        self.fixed_decoder = decoder
        self.good_start = good_start
        self.init_coding = init_coding
        
        # output and abstraction dimensions
        self.dim_output = task.dim_output
        self.sample_dichotomies = sample_dichotomies
        # self.num_cond = self.abstracts.num_var
        # self.dichotomy_type = dichotomy_type
        
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
            enc = Feedforward([self.dim_input]+[self.H for _ in range(num_layer)]+[N], self.nonlinearity)
            if decoder is not None:
                dec = decoder
            else:
                dec = Feedforward([N, self.dim_output], [task.link])
            
        self.model = MultiGLM(enc, dec, obs_dist, latent_dist)

         # an option to create the experiment instance from a previously saved experiment
        if (not self.task_only_mode) and init_from_saved:
            self.load_experiment(SAVE_DIR) 

        # Import data, create the train and test sets, and the dataloader for optimisation
        self.load_data(SAVE_DIR)

    def load_data(self, SAVE_DIR):
        """
        This function must create the following attributes:
            self.train_data: tuple of tensors with (inputs, outputs) of shape
                ((N_train, N_feature), (N_train, N_label))
            self.train_conditions: tensor of condition labels (for abstraction metrics)
                (N_train, N_conditions)
            self.ntrain: int, number of training points
        
        and the same for test data:
            self.test_data: 
                ((N_test, N_feature), (N_test, N_label))
            self.test_conditions: 
                (N_test, N_conditions)
            self.ntest: int, number of test points
        """
        raise NotImplementedError
    
    def use_model(self, N, init=None, H=None):
        """If the object is in task_only_mode, it can be equipped later with
        a model by specifying the particular parameters"""
        
        self.task_only_mode = (self.task_only_mode and False) # for some reason I can't just set to False
        
        # specify model
        self.dim_latent = N
        self.init = init
        enc = Feedforward([self.dim_input]+[self.H for _ in range(self.num_layer)]+[N], self.nonlinearity)
        if self.fixed_decoder is not None:
            dec = self.fixed_decoder
        else:
            dec = Feedforward([N, self.dim_output], [self.task.link])
        
        new_model = MultiGLM(enc, dec, self.model.obs, self.model.latent)
        self.model = new_model
    
    def run_experiment(self, verbose=False):
        
        expinf = self.file_suffix()
        print('Running %s ...'%expinf)

        if self.good_start:
            if self.init_coding is None:
                self.init_coding = 0.8
            C = np.random.rand(self.dim_latent, self.dim_latent)
            W1 = la.qr(C)[0][:,:self.dim_output]

            # ideal representation: a random ortho-linear pre-image of the logits
            # print(self.train_data[1])
            fake_rep = ((2*self.train_data[1].numpy()-1)*10)@W1.T
            offset = np.quantile(fake_rep, 1-self.init_coding)
            # offset = 0.8*fake_rep.min()
            # print(self.train_data[1][:20])
            # print(self.train_conditions[:20])
            ols = linear_model.LinearRegression()
            ols.fit(self.train_data[0], fake_rep - offset)

            self.model.enc.network.layer0.weight.data = torch.tensor(ols.coef_).float()
            self.model.enc.network.layer0.bias.data = torch.tensor(ols.intercept_).float()

            self.model.dec.network.layer0.weight.data = torch.tensor(W1).float().T
            self.model.dec.network.layer0.bias.data = torch.tensor((offset*W1).sum(0)).float()
            # self.model.dec.network.layer0.weight.data = torch.tensor(W1, requires_grad=False).float().T
            # self.model.dec.network.layer0.bias.data = torch.tensor((offset*W1).sum(0),requires_grad=False).float()
        elif self.init_coding is not None:
            init_rep = self.model.enc.network.layer0(self.train_data[0])
            offset = -1*np.quantile(init_rep.detach().numpy(), 1-self.init_coding)

            self.model.enc.network.layer0.bias.data = (torch.ones(self.dim_latent)*offset).float()

        dset = torch.utils.data.TensorDataset(self.train_data[0], self.train_data[1])
        dl = torch.utils.data.DataLoader(dset, batch_size=self.bsz, shuffle=True) 
        
        optimizer = self.opt_alg(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # inference
        n_compute = 5000
        # n_dichotomy = int(spc.binom(2**self.num_cond, 2**(self.num_cond-1))/2)
        # n_dichotomy = Dichotomies(self.train_conditions, self.dichotomy_type).ntot
        num_cond = len(np.unique([self.train_conditions,self.train_conditions]))
        # num_dic_max = int(spc.binom(num_cond, int(num_cond/2))/2)

        if self.sample_dichotomies is not None:
            these_dics = [tuple(p.tolist()) for p in self.task.positives]
            dics = Dichotomies(num_cond, these_dics, self.sample_dichotomies)
            # dic_shat = Dichotomies(num_cond, these_dics, self.sample_dichotomies)
        else:
            dics = Dichotomies(num_cond)
        dic_shat = Dichotomies(num_cond)

        metrics = {'train_loss': np.zeros(0),
                   'train_perf': np.zeros((0, self.task.num_var)),
                   'train_PS': np.zeros((0, dics.ntot)),
                   'test_perf': np.zeros((0, self.task.num_var)),
                   'test_PS': np.zeros((0, dics.ntot)),
                   'shattering': np.zeros((0, dic_shat.ntot)),
                   'mean_grad': np.zeros((0, 3)),
                   'std_grad': np.zeros((0, 3)),
                   'train_ccgp': np.zeros((0, dics.ntot)),
                   'test_ccgp': np.zeros((0, dics.ntot)),
                   'linear_dim': np.zeros(0),
                   'sparsity': np.zeros(0)} # put all training metrics here
        
        for epoch in range(self.nepoch):
            # check each quantity before optimisation
            
            with torch.no_grad():
                # train error ##############################################
                idx_trn = np.random.choice(self.ntrain, n_compute, replace=False)
                
                pred, _, z_train = self.model(self.train_data[0][idx_trn,...])
                # terr = (self.train_data[1][idx_trn,...] == (pred>=0.5)).sum(0).float()/n_compute
                terr = self.task.correct(pred, self.train_data[1][idx_trn,...])/n_compute
                terr = terr.expand_as(torch.empty((1,self.task.num_var)))
                metrics['train_perf'] = np.append(metrics['train_perf'], terr, axis=0)
                
                # test error ##############################################
                idx_tst = np.random.choice(self.ntest, n_compute, replace=False)
                
                pred, _, z_test = self.model(self.test_data[0][idx_tst,...])
                # terr = (self.test_data[1][idx_tst,...] == (pred>=0.5)).sum(0).float()/n_compute
                terr = self.task.correct(pred, self.test_data[1][idx_tst,...])/n_compute
                terr = terr.expand_as(torch.empty((1,self.task.num_var)))
                metrics['test_perf'] = np.append(metrics['test_perf'], terr, axis=0)

                # representation sparsity
                metrics['sparsity'] = np.append(metrics['sparsity'], np.mean(z_test.detach().numpy()>0))

                # things that take up time! ###################################
                if not self.skip_metrics:
                    # Dimensionality #########################################
                    _, S, _ = la.svd(z_train.detach()-z_train.mean(1).detach()[:,None], full_matrices=False)
                    eigs = S**2
                    pr = (np.sum(eigs)**2)/np.sum(eigs**2)
                    metrics['linear_dim'] = np.append(metrics['linear_dim'], pr)
                    
                    # shattering dimension #####################################
                    dclf = LinearDecoder(self.dim_latent, dic_shat.ntot, svm.LinearSVC)

                    trn_conds_all = np.array([dic_shat.coloring(self.train_conditions[idx_trn]) \
                        for _ in dic_shat])
                    dclf.fit(z_train.detach().numpy(), trn_conds_all.T)

                    tst_conds_all = np.array([dic_shat.coloring(self.test_conditions[idx_tst]) \
                        for _ in dic_shat])
                    SD = dclf.test(z_test.detach().numpy(), tst_conds_all.T).T

                    metrics['shattering'] = np.append(metrics['shattering'], SD, axis=0)  

                    # various dichotomy-based metrics ########################
                    clf = LinearDecoder(self.dim_latent, 1, MeanClassifier)
                    gclf = LinearDecoder(self.dim_latent, 1, svm.LinearSVC)
                    K = int(num_cond/2)-1
                    # K = int(num_cond/4)

                    PS = np.zeros(dics.ntot)
                    CCGP = np.zeros(dics.ntot)
                    for i, _ in enumerate(dics):
                        PS[i] = dics.parallelism(z_test.detach().numpy(),
                            self.test_conditions[idx_tst], clf)
                        CCGP[i] = dics.CCGP(z_test.detach().numpy(), 
                            self.test_conditions[idx_tst], gclf, K)

                    metrics['test_PS'] = np.append(metrics['test_PS'], PS[None,:], axis=0)
                    metrics['test_ccgp'] = np.append(metrics['test_ccgp'], CCGP[None,:], axis=0)
                
            # Actually update model #######################################
            loss = self.model.grad_step(dl, optimizer) # this does a pass through the data
            
            metrics['train_loss'] = np.append(metrics['train_loss'], loss)
            
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
        
        if 'RandomDichotomies' in self.task.__name__:
            all_args['dichotomies'] = self.task.positives
        all_args = self.save_other_info(all_args) # subroutine specific to experiment class
        
        params_fname = 'parameters'+expinf+'.pt'
        metrics_fname = 'metrics'+expinf+'.pkl'
        args_fname = 'arguments'+expinf+'.npy'
        
        self.model.save(SAVE_DIR+FOLDERS+params_fname)
        with open(SAVE_DIR+FOLDERS+metrics_fname, 'wb') as f:
            pickle.dump(self.metrics, f, -1)
        with open(SAVE_DIR+FOLDERS+args_fname, 'wb') as f:
            np.save(f, all_args)
    
    def save_other_info(self, arg_dict):
        """ 
        If there is other information that must be saved when saving an experiment,
        store it using this method. Add the information as keys in the args dict
        """
        return arg_dict

    def load_experiment(self, SAVE_DIR):
        """
        Only works when the task AND the model are completely specified, i.e.
        it won't work if task_only_mode == True.

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

        self.load_other_info(args)
        
        return self.model, metrics, args

    def load_other_info(self, arg_dict):
        """
        If you need to load other saved information from the arg_dict, use this 
        method to do so by adding that information as an attribute
        """
        if 'RandomDichotomies' in self.task.__name__:
            self.task.positives = arg_dict['dichotomies']
    
    # file I/O functions for this experiment
    # I do this in order to standardise everything within this experiment
    def folder_hierarchy(self):
        FOLDERS = self.base_dir
        if self.num_layer != 1:
            FOLDERS += str(self.num_layer+1)+'layer/'
        FOLDERS += self.task.__name__ + '/'
        FOLDERS += self.model.obs.name() + '/'
        if self.fixed_decoder is not None:
            FOLDERS += self.fixed_decoder.__name__ + '/'
        if self.good_start:
            FOLDERS += 'ols_%.1f_init/'%self.init_coding
        elif self.init_coding is not None:
            FOLDERS += '%.1f_init/'%self.init_coding
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

#%% particular experiment types   
class mnist_multiclass(MultiClassification):
    """
    Train a feedforward network to do multiple classifications on MNIST.
    """
    def __init__(self, task, SAVE_DIR, **expargs):
        """
        Everything required to fully specify an experiment.
        
        Failure to supply the N argument will create the class in 'task only mode',
        which means that will not have a model. Call the `self.use_model` method
        to later equip it with a particular model.
        """
        self.dim_input = 784
        self.num_class = 8
        super(mnist_multiclass, self).__init__(task, SAVE_DIR, **expargs)
        self.base_dir = 'results/mnist/' # append at will
    
    def load_data(self, SAVE_DIR):
        # -------------------------------------------------------------------------
        # Import data, create the train and test sets
        digits = torchvision.datasets.MNIST(SAVE_DIR+'digits/', download=True, 
                                            transform=torchvision.transforms.ToTensor())
        valid = (digits.targets <= 8) & (digits.targets>=1)
        self.train_data = (digits.data[valid,...].reshape(-1,784).float()/252, 
                           self.task(digits.targets-1)[valid,...])
        # self.train_conditions = self.abstracts(digits.targets)[valid,...]
        self.train_conditions = digits.targets[valid,...].detach().numpy()-1
        self.ntrain = int(valid.sum())
        
        stigid = torchvision.datasets.MNIST(SAVE_DIR+'digits/',download=True, train=False,
                                            transform=torchvision.transforms.ToTensor())
        valid = (stigid.targets<=8) & (stigid.targets>=1)
        self.test_data = (stigid.data[valid,...].reshape(-1,784).float()/252, 
                          self.task(stigid.targets-1)[valid,...])
        # self.test_conditions = self.abstracts(stigid.targets)[valid,...]
        self.test_conditions = stigid.targets[valid,...].detach().numpy()-1
        self.ntest = int(valid.sum())

class random_patterns(MultiClassification):
    """
    Random uncorrelated patterns, with random labels.
    Draws from a gaussian mixture model
    """
    def __init__(self, task, SAVE_DIR, num_class, dim=700, var_means=5, var_noise=1, **expargs):
        """
        Generates num_cond
        """
        self.num_class = num_class
        self.var_means = var_means
        self.var_noise = var_noise
        self.dim_input = dim
        self.means = None

        super(random_patterns, self).__init__(task, SAVE_DIR, **expargs)
        self.base_dir = 'results/mog-%d-%d-%.1f/'%(dim, num_class, var_means/var_noise) # append at will

    def load_data(self, SAVE_DIR):
        # -------------------------------------------------------------------------
        # Import data, create the train and test sets
        n_total = 70000 # hardcode the number of datapoints
        n_per_class = int(n_total/self.num_class) 

        if self.means is None:
            means = np.random.randn(self.num_class, self.dim_input)*self.var_means
            self.means = means

        means = self.means
        noise = np.random.randn(n_per_class, self.num_class, self.dim_input)*self.var_noise
        X_ = means+noise
        Y_ = np.tile(np.arange(self.num_class, dtype=int),n_per_class)

        shf = np.random.permutation(self.num_class*n_per_class)

        X = torch.tensor(X_.reshape((-1, self.dim_input))[shf,:]).float()
        Y = torch.tensor(Y_[shf])

        trn = int(np.floor(0.8*n_per_class*self.num_class))

        self.train_data = (X[:trn,:], self.task(Y[:trn]))
        # self.train_conditions = self.abstracts(Y[:trn])
        self.train_conditions = Y[:trn].detach().numpy()
        self.ntrain = trn
        
        self.test_data = (X[trn:,:], 
                          self.task(Y[trn:]))
        # self.test_conditions = self.abstracts(Y[trn:])
        self.test_conditions = Y[trn:].detach().numpy()
        self.ntest = n_per_class*self.num_class - trn

    def save_other_info(self, arg_dict):
        arg_dict['class_means'] = self.means
        return arg_dict

    def load_other_info(self, arg_dict):
        self.means = arg_dict['class_means']
        if 'RandomDichotomies' in self.task.__name__:
            self.task.positives = arg_dict['dichotomies']