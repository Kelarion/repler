"""
Classes which implement experiments. They are what's called in the habanero
experiment scripts. They standardise my experiments with a Byzantine web of 
class inheritance and exchangeable modules. Not for human consumption.
"""

import os
import pickle
import warnings
import re

import torch
import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc

# this is my code base, this assumes that you can access it
from students import *
# from recurrent import *
from assistants import *
import util
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
                 nonlinearity='ReLU', num_layer=1, z_prior=None, bias=True,
                 bsz=64, nepoch=1000, lr=1e-4, opt_alg=optim.Adam, weight_decay=0,
                 init=None, skip_metrics=False, sample_dichotomies=0, fix_decoder=False,
                 decoder=None, init_from_saved=False, good_start=False, init_coding=None, rot=0.0):
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
        self.init_rot = rot
        
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
        self.fix_decoder = fix_decoder
        
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
            dec = Feedforward([N, self.dim_output], [task.link], layer_type=decoder)
            
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
        dec = Feedforward([N, self.dim_output], [self.task.link], layer_type=self.fixed_decoder)
        
        new_model = MultiGLM(enc, dec, self.model.obs, self.model.latent)
        self.model = new_model
    
    def run_experiment(self, verbose=False):
        
        expinf = self.file_suffix()
        print('Running %s ...'%expinf)

        if self.good_start:
            if self.init_coding is None:
                self.init_coding = 0.9
            # C = np.random.rand(self.dim_latent, self.dim_latent)
            # W1 = la.qr(C)[0][:,:self.dim_output]

            # ideal representation: a random ortho-linear pre-image of the logits
            # fake_rep = ((2*self.train_data[1].numpy()-1)*10)@W1.T
            emb = util.ContinuousEmbedding(self.dim_latent, self.init_rot)
            W1 = 20*emb.rotation_mat(-self.init_rot/2)@(emb.basis[:,:2])
            fake_rep = emb(self.train_data[1])
            offset = np.quantile(fake_rep, 1-self.init_coding)

            ols = linear_model.LinearRegression()
            if self.num_layer>0:
                regressor = self.model.enc.network[:-2](self.train_data[0]).detach().numpy()
            else:
                regressor = self.train_data[0]
            ols.fit(regressor, fake_rep - offset)

            penult = getattr(self.model.enc.network, 'layer%d'%(self.num_layer))
            penult.weight.data = torch.tensor(ols.coef_).float()
            penult.bias.data = torch.tensor(ols.intercept_).float()

            self.model.dec.network.layer0.weight.data = W1.float().T
            self.model.dec.network.layer0.bias.data = (offset*W1).sum(0).float()
        elif self.init_coding is not None:
            init_rep = self.model.enc.network.layer0(self.train_data[0])
            offset = -1*np.quantile(init_rep.detach().numpy(), 1-self.init_coding)

            self.model.enc.network.layer0.bias.data = (torch.ones(self.dim_latent)*offset).float()

        if self.fix_decoder:
            for p in self.model.dec.network.parameters():
                p.requires_grad = False

        dset = torch.utils.data.TensorDataset(self.train_data[0], self.train_data[1])
        dl = torch.utils.data.DataLoader(dset, batch_size=self.bsz, shuffle=True) 
        
        optimizer = self.opt_alg(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # inference
        n_compute = np.min([5000,self.ntest])
        num_cond = len(np.unique([self.train_conditions,self.train_conditions]))
        # num_dic_max = int(spc.binom(num_cond, int(num_cond/2))/2)

        if not self.skip_metrics:
            if self.sample_dichotomies is not None:
                these_dics = [tuple(p.tolist()) for p in self.task.positives]
                dics = Dichotomies(num_cond, these_dics, self.sample_dichotomies)
                dic_shat = Dichotomies(num_cond, these_dics, self.sample_dichotomies)
            else:
                dics = Dichotomies(num_cond)
                dic_shat = Dichotomies(num_cond)
            # x_mean = np.stack([self.train_data[0][self.train_conditions==i,:].mean(0).detach().numpy() \
            #     for i in np.unique(self.train_conditions)]).T
            # y_mean = np.stack([self.train_data[1][self.train_conditions==i,:].mean(0).detach().numpy() \
            #     for i in np.unique(self.train_conditions)]).T

            # Kern_x = util.dot_product(x_mean-x_mean.mean(1,keepdims=True), x_mean-x_mean.mean(1,keepdims=True))
            # Kern_y = util.dot_product(y_mean-y_mean.mean(1,keepdims=True), y_mean-y_mean.mean(1,keepdims=True))
        else:
            dics = Dichotomies(0)
            dic_shat = Dichotomies(0)

        metrics = {'train_loss': np.zeros(0),
                   'train_perf': np.zeros((0, self.task.num_var)),
                   'train_PS': np.zeros((0, dics.ntot)),
                   'test_perf': np.zeros((0, self.task.num_var)),
                   'test_PS': np.zeros((0, dics.ntot)),
                   'shattering': np.zeros((0, dic_shat.ntot)),
                   'mean_grad': [],
                   'std_grad': [],
                   'train_ccgp': np.zeros((0, dics.ntot)),
                   'test_ccgp': np.zeros((0, dics.ntot)),
                   'linear_dim': np.zeros(0),
                   'sparsity': np.zeros(0),
                   'input_alignment': [],
                   'output_alignment': []} # put all training metrics here

        for epoch in range(self.nepoch):
            # check each quantity before optimisation
            
            with torch.no_grad():
                # train error ##############################################
                idx_trn = np.random.choice(self.ntrain, n_compute, replace=False)

                pred, _, z_train = self.model(self.train_data[0][idx_trn,...])
                # terr = (self.train_data[1][idx_trn,...] == (pred>=0.5)).sum(0).float()/n_compute
                # print(pred.shape)
                # print(self.train_data[1][idx_trn,...].shape)
                terr = self.task.correct(pred, self.train_data[1][idx_trn,...])
                metrics['train_perf'] = np.append(metrics['train_perf'], terr, axis=0)
                
                # test error ##############################################
                idx_tst = np.random.choice(self.ntest, np.min([self.ntest,n_compute]), replace=False)
                
                pred, _, z_test = self.model(self.test_data[0][idx_tst,...])
                # terr = (self.test_data[1][idx_tst,...] == (pred>=0.5)).sum(0).float()/n_compute
                terr = self.task.correct(pred, self.test_data[1][idx_tst,...])
                metrics['test_perf'] = np.append(metrics['test_perf'], terr, axis=0)

                # representation sparsity
                metrics['sparsity'] = np.append(metrics['sparsity'], np.mean(z_test.detach().numpy()>0))

                # Dimensionality #########################################
                _, S, _ = la.svd(z_train.detach()-z_train.mean(1).detach()[:,None], full_matrices=False)
                eigs = S**2
                pr = (np.sum(eigs)**2)/np.sum(eigs**2)
                metrics['linear_dim'] = np.append(metrics['linear_dim'], pr)

                # things that take up time! ###################################
                if not self.skip_metrics:
                    # distance correlations
                    # didx = np.random.choice(n_compute,np.min([n_compute, 2000]),replace=False)
                    # Z = z_train[didx,...].T
                    # X = self.train_data[0][idx_trn,...][didx,...].T
                    # Y = self.train_data[1][idx_trn,...][didx,...].T
                    # metrics['dcorr_input'].append(util.distance_correlation(Z, X))
                    # metrics['dcorr_output'].append(util.distance_correlation(Z, Y))

                    # z_mean = np.stack([z_train[self.train_conditions[idx_trn]==i,:].mean(0).detach().numpy() \
                    #     for i in np.unique(self.train_conditions[idx_trn])]).T
                    # Kern_z = util.dot_product(z_mean-z_mean.mean(1,keepdims=True), z_mean-z_mean.mean(1,keepdims=True))
                    # inp_align = (np.einsum('kij,ij->k',Kz.mean(0),Kx)/la.norm(Kz.mean(0),'fro',axis=(-2,-1))/np.sqrt(np.sum(Kx*Kx)))
                    # metrics['input_alignment'].append()

                    # shattering dimension #####################################
                    dclf = LinearDecoder(self.dim_latent, dic_shat.ntot, svm.LinearSVC)

                    trn_conds_all = np.array([dic_shat.coloring(self.train_conditions[idx_trn]) \
                        for _ in dic_shat])
                    dclf.fit(z_train.detach().numpy(), trn_conds_all.T, max_iter=200)

                    tst_conds_all = np.array([dic_shat.coloring(self.test_conditions[idx_tst]) \
                        for _ in dic_shat])
                    SD = dclf.test(z_test.detach().numpy(), tst_conds_all.T).T

                    metrics['shattering'] = np.append(metrics['shattering'], SD, axis=0)   
                    
                    # various abstraction metrics #########################
                    clf = LinearDecoder(self.dim_latent, 1, MeanClassifier)
                    gclf = LinearDecoder(self.dim_latent, 1, svm.LinearSVC)

                    # K = int(num_cond/2)-1
                    K = int(num_cond/4)

                    PS = np.zeros(dics.ntot)
                    CCGP = np.zeros(dics.ntot)
                    for i, _ in enumerate(dics):
                        PS[i] = dics.parallelism(z_test.detach().numpy(),
                            self.test_conditions[idx_tst], clf)
                        CCGP[i] = np.mean(dics.CCGP(z_test.detach().numpy(), 
                            self.test_conditions[idx_tst], gclf, max_iter=200))

                    metrics['test_PS'] = np.append(metrics['test_PS'], PS[None,:], axis=0)
                    metrics['test_ccgp'] = np.append(metrics['test_ccgp'], CCGP[None,:], axis=0)
 
            # Actually update model #######################################
            loss = self.model.grad_step(dl, optimizer) # this does a pass through the data
            
            metrics['train_loss'] = np.append(metrics['train_loss'], loss)
            
            # gradient SNR
            if not np.mod(epoch,10):
                means = []
                std = []
                # i=0
                for k,v in zip(self.model.state_dict().keys(), self.model.parameters()):
                    if 'weight' in k and v.requires_grad:
                        means.append((v.grad.data.mean(1)/v.data.norm(2,1)).norm(2,0).numpy())
                        std.append((v.grad.data/v.data.norm(2,1,keepdim=True)).std().numpy())
                        # print(means)
                        # i+=1
                metrics['mean_grad'].append(np.array(means)[None,:])
                metrics['std_grad'].append(np.array(std)[None,:])
            
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
        
        # if 'RandomDichotomies' in self.task.__name__ or 'LogicalFunctions' in self.task.__name__:
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
        # if 'RandomDichotomies' in self.task.__name__ or 'LogicalFunctions' in self.task.__name__:
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
            FOLDERS += '%.1ffactored_init/'%self.init_rot
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

    def aggregate_nets(self, SAVE_DIR, N_list):

        this_folder = SAVE_DIR + self.folder_hierarchy()

        all_nets = [[] for _ in N_list]
        all_args = [[] for _ in N_list]
        mets = [[] for _ in N_list]
        for i,n in enumerate(N_list):
            files = os.listdir(this_folder)
            param_files = [f for f in files if ('parameters' in f and '_N%d_%s'%(n,self.nonlinearity) in f)]
            # j = 0
            num = len(param_files)
            all_metrics = {}
            best_net = None
            this_arg = None
            maxmin = 0
            for j,f in enumerate(param_files):
                rg = re.findall(r"init(\d+)?_N%d_%s"%(n,self.nonlinearity),f)
                if len(rg)>0:
                    init = np.array(rg[0]).astype(int)
                else:
                    init = None
                    
                self.use_model(N=n, init=init)
                model, metrics, args = self.load_experiment(SAVE_DIR)
                
                # if metrics['test_perf'][-1,...].min() > maxmin:    
                #     maxmin = metrics['test_perf'][-1,...].min()
                #     best_net = model
                #     this_arg = args
                
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
                
            mets[i] = all_metrics

        return all_nets, mets, all_args

# particular experiment types   
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
    def __init__(self, task, SAVE_DIR, num_class, dim=100, var_means=1, var_noise=1, **expargs):
        """
        Generates num_cond
        """
        self.num_class = num_class
        self.var_means = var_means
        self.var_noise = var_noise
        self.dim_input = dim
        self.means = None

        super(random_patterns, self).__init__(task, SAVE_DIR, **expargs)
        self.base_dir = 'results/mog/%d-%d-%.1f/'%(dim, num_class, var_noise/var_means) # append at will

    def load_data(self, SAVE_DIR):
        # -------------------------------------------------------------------------
        # Import data, create the train and test sets
        # hardcode the number of datapoints
        n_per_class = 2000
        n_total = n_per_class*self.num_class

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

class structured_inputs(MultiClassification):
    """
    Random uncorrelated patterns, with random labels.
    Draws from a gaussian mixture model
    """
    def __init__(self, task, input_task, SAVE_DIR, noise_var=0.1, mixing=False, **expargs):

        self.num_inp = input_task.num_var # how many inputs 
        self.noise_var = noise_var 
        self.dim_input = input_task.dim_output
        # self.dim_factor = dim_inputs
        self.mixing = mixing
        self.input_task = input_task
        self.means = input_task.means
        # self.means = None

        super(structured_inputs, self).__init__(task, SAVE_DIR, **expargs)
        self.base_dir = 'results/structured/%d-%d-%.1f-%s/'%(self.num_inp, self.dim_input, noise_var, ['mixed' if mixing else 'unmixed'][0]) # append at will

    def load_data(self, SAVE_DIR):
        # -------------------------------------------------------------------------
        # Import data, create the train and test sets
        n_total = 1000*(2**self.num_inp) # hardcode the number of datapoints

        inp_condition = np.random.choice(2**self.num_inp, n_total)

        X = self.input_task(inp_condition)

        # generate outputs
        Y = torch.tensor(inp_condition)

        trn = int(np.floor(0.8*n_total))

        self.train_data = (X[:trn,:], self.task(Y[:trn]))
        # self.train_conditions = self.abstracts(Y[:trn])
        self.train_conditions = Y[:trn].detach().numpy()
        self.ntrain = trn
        
        self.test_data = (X[trn:,:], 
                          self.task(Y[trn:]))
        # self.test_conditions = self.abstracts(Y[trn:])
        self.test_conditions = Y[trn:].detach().numpy()
        self.ntest = n_total - trn

    def save_other_info(self, arg_dict):
        arg_dict['class_means'] = self.means
        arg_dict['input_dichotomies'] = self.input_task.positives
        return arg_dict

    def load_other_info(self, arg_dict):
        self.input_task.define_basis(arg_dict['class_means'])
        # if 'RandomDichotomies' in self.task.__name__:
        self.task.positives = arg_dict['dichotomies']
        self.input_task.positives = arg_dict['input_dichotomies']

    def folder_hierarchy(self):
        FOLDERS = self.base_dir
        if self.num_layer != 1:
            FOLDERS += str(self.num_layer+1)+'layer/'
        FOLDERS += self.task.__name__ + '/'
        FOLDERS += self.input_task.__name__ + '/'
        FOLDERS += self.model.obs.name() + '/'
        if self.fixed_decoder is not None:
            FOLDERS += self.fixed_decoder.__name__ + '/'
        if self.model.latent is not None:
            FOLDERS += self.model.latent.name() + '/'
        if self.weight_decay > 0:
            FOLDERS += 'L2reg/'
        
        return FOLDERS

class random_bits(MultiClassification):
    """
    Random uncorrelated binary (+/-1) vectors, with random labels
    """
    def __init__(self, task, SAVE_DIR, num_class, dim=100, p_means=0.5, p_noise=0.1, **expargs):
        """
        Generates num_cond
        """
        self.num_class = num_class
        self.p_means = p_means
        self.p_noise = p_noise
        self.dim_input = dim
        self.means = None

        super(random_bits, self).__init__(task, SAVE_DIR, **expargs)
        self.base_dir = 'results/bits/%d-%d-%.1f/'%(dim, num_class, p_noise) # append at will

    def load_data(self, SAVE_DIR):
        # -------------------------------------------------------------------------
        # Import data, create the train and test sets
        # n_total = 10000 # hardcode the number of datapoints
        n_per_class = 5000
        n_total = n_per_class*self.num_class 

        if self.means is None:
            means = 2*(np.random.rand(self.num_class, self.dim_input)>self.p_means).astype(int) - 1
            self.means = means

        means = self.means
        noise = 2*(np.random.rand(n_per_class, self.num_class, self.dim_input)>self.p_noise)-1
        X_ = means[None,...]*noise
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


#%% Sequential classification
class SequentialClassification():
    """
    Basic class for multi-classification experiments. To make an instance of such an experiment,
    make a child class and define the `load_data` method. This contains all the methods
    to run the experiment, and save and load it.
    """
    def __init__(self, task, SAVE_DIR, N=None,
                 nonlinearity='ReLU', num_layer=1,
                 bsz=64, nepoch=2000, lr=1e-4, opt_alg=optim.Adam, weight_decay=0,
                 init=None, skip_metrics=False, sample_dichotomies=0, fix_decoder=True,
                 decoder=None, init_from_saved=False):

        self.dim_latent = N
        self.nonlinearity = nonlinearity
        self.init = init # optionally specify an initialisation index -- for randomising
   
        # these values specify model/task classes (show up in folder names)
        self.base_dir = 'results/' # append at will
        self.num_layer = num_layer
        self.task = task
        self.fixed_decoder = decoder
        
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
        self.fix_decoder = fix_decoder
        
        self.opt_alg = opt_alg

        self.save_directory = SAVE_DIR
        
        # flags
        self.skip_metrics = skip_metrics
        if N is None:
            # if the class is being called just to call the task
            self.task_only_mode = True
        else:
            self.task_only_mode = False
        
        # -------------------------------------------------------------------------
        # Model specification (changing this will make past experiments incompatible)
        obs_dist = task.obs_distribution
        
        if decoder is None:
            dec = None
        else:
            dec = self.fixed_decoder()

        if self.task_only_mode:
            self.model = GenericRNN(1, 1, obs_dist, 
                rnn_type=nonlinearity, decoder=dec, nlayers=num_layer)
        else:
            self.model = GenericRNN(self.dim_input, self.dim_latent, obs_dist, 
                rnn_type=nonlinearity, decoder=dec, nlayers=num_layer)

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

        if self.fixed_decoder is None:
            dec = None
        else:
            dec = self.fixed_decoder(self.dim_latent, self.dim_output)
        new_model = GenericRNN(self.dim_input, self.dim_latent, self.model.obs, 
            rnn_type=self.model.rnn_type, decoder=dec, nlayers=self.model.nlayers)
        self.model = new_model
    
    def run_experiment(self, verbose=False):
        
        expinf = self.file_suffix()
        print('Running %s ...'%expinf)

        if self.fix_decoder:
            for p in self.model.decoder.parameters():
                p.requires_grad = False

        dset = torch.utils.data.TensorDataset(self.train_data[0], self.train_data[1])
        dl = torch.utils.data.DataLoader(dset, batch_size=self.bsz, shuffle=True) 
        
        optimizer = self.opt_alg(self.model.rnn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # print(list(self.model.parameters()))
        # inference
        n_compute = np.min([5000,self.ntest])
        num_cond = len(np.unique([self.train_conditions,self.train_conditions]))
        # num_dic_max = int(spc.binom(num_cond, int(num_cond/2))/2)

        if not self.skip_metrics:
            these_dics = [tuple(p.tolist()) for p in self.task.positives]
            if self.sample_dichotomies is not None:
                dics = Dichotomies(num_cond, these_dics, self.sample_dichotomies)
                # dic_shat = Dichotomies(num_cond, these_dics, self.sample_dichotomies)
            else:
                dics = Dichotomies(num_cond, these_dics, 50)
            dic_shat = Dichotomies(num_cond, these_dics, 50)
        else:
            dics = Dichotomies(0)
            dic_shat = Dichotomies(0)

        metrics = {'train_loss': np.zeros(0),
                   'train_perf': np.zeros((0, self.task.num_var)),
                   'PS': np.zeros((0, dics.ntot)),
                   'test_perf': np.zeros((0, self.task.num_var)),
                   'shattering': np.zeros((0, dic_shat.ntot)),
                   'CCGP': np.zeros((0, dics.ntot)),
                   'linear_dim': np.zeros(0),
                   'pos_conds': dics.combs} # put all training metrics here

        for epoch in range(self.nepoch):
            # check each quantity before optimisation
            
            with torch.no_grad():
                # train error ##############################################
                idx_trn = np.random.choice(self.ntrain, n_compute, replace=False)
                # print(self.model.decoder)
                pred, z_train = self.model(self.train_data[0][idx_trn,...].transpose(0,1))
                z_train = z_train.transpose(0,1).reshape((self.dim_latent,-1)).T
                # terr = (self.train_data[1][idx_trn,...] == (pred>=0.5)).sum(0).float()/n_compute
                # print(pred.shape)
                # print(self.train_data[1][idx_trn,...].shape)
                terr = self.task.correct(pred, self.train_data[1][idx_trn,...])
                # print(pred.shape)
                metrics['train_perf'] = np.append(metrics['train_perf'], terr, axis=0)
                
                # test error ##############################################
                idx_tst = np.random.choice(self.ntest, np.min([self.ntest,n_compute]), replace=False)
                
                pred, z_test = self.model(self.test_data[0][idx_tst,...].transpose(0,1))
                # terr = (self.test_data[1][idx_tst,...] == (pred>=0.5)).sum(0).float()/n_compute
                terr = self.task.correct(pred, self.test_data[1][idx_tst,...])
                metrics['test_perf'] = np.append(metrics['test_perf'], terr, axis=0)

                # representation sparsity
                # metrics['sparsity'] = np.append(metrics['sparsity'], np.mean(z_test.detach().numpy()>0))

                # Dimensionality #########################################
                _, S, _ = la.svd(z_train.detach()-z_train.mean(1).detach()[:,None], full_matrices=False)
                eigs = S**2
                pr = (np.sum(eigs)**2)/np.sum(eigs**2)
                metrics['linear_dim'] = np.append(metrics['linear_dim'], pr)
 
            # Actually update model #######################################
            loss = self.model.grad_step(dl, optimizer) # this does a pass through the data
            
            metrics['train_loss'] = np.append(metrics['train_loss'], loss)
            
            # gradient SNR
            # if not np.mod(epoch,10):
            #     means = []
            #     std = []
            #     # i=0
            #     for k,v in zip(self.model.state_dict().keys(), self.model.parameters()):
            #         if 'weight' in k and v.requires_grad:
            #             means.append((v.grad.data.mean(1)/v.data.norm(2,1)).norm(2,0).numpy())
            #             std.append((v.grad.data/v.data.norm(2,1,keepdim=True)).std().numpy())
            #             # print(means)
            #             # i+=1
            #     metrics['mean_grad'].append(np.array(means)[None,:])
            #     metrics['std_grad'].append(np.array(std)[None,:])
            
            # print updates ############################################
            if verbose:
                print('Epoch %d: Loss=%.3f'%(epoch, -loss))
        
        # things that take up time! ###################################
        if not self.skip_metrics:
            if verbose:
                print('Training over, computing metrics ...')
            self.load_data(self.save_directory, jitter_override=False)
            test_inps = self.train_data[0]
            z_ = self.model.transparent_forward(test_inps.transpose(0,1))[1].detach().numpy()

            which_time = np.repeat(range(test_inps.shape[1]), self.ntrain)

            n_compute = np.min([5000,self.ntrain])

            all_PS = []
            all_CCGP = []
            all_SD = []
            all_pr = []
            for t in range(test_inps.shape[1]):
                
                these_times = (which_time==t)
                
                z_test = z_.transpose((1,0,2)).reshape((self.dim_latent,-1))[:,these_times].T 

                # Dimensionality #########################################
                _, S, _ = la.svd(z_test-z_test.mean(1)[:,None], full_matrices=False)
                eigs = S**2
                pr = (np.sum(eigs)**2)/np.sum(eigs**2)
                all_pr.append(pr)

                # Abstraction ############################################
                stim_cond = np.tile(self.train_conditions, test_inps.shape[1])[these_times]
                
                idx = np.random.choice(z_test.shape[0], n_compute, replace=False)
     
                cond = stim_cond[idx]
                # print(cond.shape)
                # print(z_test.shape)
                # print()
                # choose dichotomies to have a particular order
                dclf = LinearDecoder(self.dim_latent, dic_shat.ntot, svm.LinearSVC)

                trn_conds_all = np.array([dic_shat.coloring(stim_cond[idx]) \
                    for _ in dic_shat])
                dclf.fit(z_test[idx,...], trn_conds_all.T, max_iter=200)

                idx2 = np.random.choice(z_test.shape[0], n_compute, replace=False)
                tst_conds_all = np.array([dic_shat.coloring(stim_cond[idx2]) \
                    for _ in dic_shat])
                SD = dclf.test(z_test[idx2,...], tst_conds_all.T).T 
                
                # various abstraction metrics #########################
                clf = LinearDecoder(self.dim_latent, 1, MeanClassifier)
                gclf = LinearDecoder(self.dim_latent, 1, svm.LinearSVC)

                # K = int(num_cond/2)-1
                K = int(self.task.num_cond/4)

                PS = np.zeros(dics.ntot)
                CCGP = np.zeros(dics.ntot)
                for i, _ in enumerate(dics):
                    PS[i] = dics.parallelism(z_test[idx,...], stim_cond[idx], clf)
                    CCGP[i] = np.mean(dics.CCGP(z_test[idx,...], stim_cond[idx], gclf, max_iter=200))
                
                all_PS.append(PS)
                all_CCGP.append(CCGP)
                all_SD.append(SD)

            metrics['shattering'] = np.array(all_SD)
            metrics['PS'] = np.array(all_PS)
            metrics['CCGP'] = np.array(all_CCGP)        
            metrics['linear_dim'] = np.array(all_pr) 

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
        
        # if 'RandomDichotomies' in self.task.__name__ or 'LogicalFunctions' in self.task.__name__:
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
        # if 'RandomDichotomies' in self.task.__name__ or 'LogicalFunctions' in self.task.__name__:
        self.task.positives = arg_dict['dichotomies']

    def folder_hierarchy(self):
        FOLDERS = self.base_dir
        if self.num_layer != 1:
            FOLDERS += str(self.num_layer+1)+'layer/'
        FOLDERS += self.task.__name__ + '/'
        FOLDERS += self.model.obs.name() + '/'
        if self.fixed_decoder is not None:
            FOLDERS += self.fixed_decoder.__name__ + '/'
        if self.weight_decay > 0:
            FOLDERS += 'L2reg/'
        
        return FOLDERS
        
    def file_suffix(self):
        if self.init is None:
            return '_N%d_%s'%(self.dim_latent, self.nonlinearity)
        else:
            return '_init%d_N%d_%s'%(self.init, self.dim_latent, self.nonlinearity)


class delayed_logic(SequentialClassification):
    def __init__(self, task, input_task, SAVE_DIR, time_between=20, input_channels=1, jitter=True, **expargs):
        """
        Generates num_cond
        """
        self.num_inp = input_task.dim_output # how many inputs
        self.dim_input = len(np.unique(input_channels))
        self.input_task = input_task
        self.input_channels = input_channels
        self.time_between = time_between
        self.jitter = jitter

        super(delayed_logic, self).__init__(task, SAVE_DIR, **expargs)
        if input_channels == 1:
            self.base_dir = 'results/dlog/%d-%d-%d/'%(self.num_inp, time_between, input_channels) # append at will
        else:
            inp_chn = ("%d"*self.num_inp)%tuple(input_channels)
            self.base_dir = 'results/dlog/%d-%d-%s/'%(self.num_inp, time_between, inp_chn) # append at will

    def load_data(self, SAVE_DIR, jitter_override=None):

        if jitter_override is None:
            jitter = self.jitter
        else:
            jitter = jitter_override

        # -------------------------------------------------------------------------
        # Import data, create the train and test sets
        n_total = 1000*(2**self.num_inp) # hardcode the number of datapoints
        # total_time = self.time_between*(self.num_inp-1) + 1
        total_time = self.time_between*self.num_inp 

        inp_condition = np.random.choice(2**self.num_inp, n_total)

        inputs = torch.zeros(n_total, total_time, self.dim_input)
        dt = self.time_between//2

        input_times = np.zeros((n_total, self.num_inp))
        input_times[:,0] = 0 
        input_times[:,1] = self.time_between + jitter*np.random.randint(-dt,dt, size=n_total)
        input_times[:,2] = 2*self.time_between + jitter*np.random.randint(-dt,int(1.5*dt), size=n_total)

        vals = 2*self.input_task(inp_condition).flatten()-1
        seq_idx = np.repeat(range(n_total),self.num_inp)
        if self.input_channels == 1:
            inp_idx = np.tile([0,0,0],n_total)
        else: # assume input channel is a list
            inp_idx = np.tile(self.input_channels, n_total)
        inputs[seq_idx,input_times.flatten(),inp_idx] = vals

        Y = torch.tensor(inp_condition)

        trn = int(np.floor(0.8*n_total))

        self.train_data = (inputs[:trn,:].float(), self.task(Y[:trn]))
        # self.train_conditions = self.abstracts(Y[:trn])
        self.train_conditions = Y[:trn].detach().numpy()
        self.ntrain = trn
        
        self.test_data = (inputs[trn:,:], 
                          self.task(Y[trn:]))
        # self.test_conditions = self.abstracts(Y[trn:])
        self.test_conditions = Y[trn:].detach().numpy()
        self.ntest = n_total - trn

    def save_other_info(self, arg_dict):
        arg_dict['input_dichotomies'] = self.input_task.positives
        return arg_dict

    def load_other_info(self, arg_dict):
        self.task.positives = arg_dict['dichotomies']
        self.input_task.positives = arg_dict['input_dichotomies']