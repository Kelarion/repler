"""
Classes which implement experiments. They are what's called in the habanero
experiment scripts. They standardise my experiments with a Byzantine web of 
class inheritance and exchangeable modules. Not for human consumption.
"""

import os
import pickle
import warnings
import re
from dataclasses import dataclass, fields, field

import torch
# import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc
from itertools import permutations
from sklearn import svm, linear_model

# this is my code base, this assumes that you can access it
import util
import pt_util
import students
import assistants as ta
import dichotomies as dic
import server_utils as su

from sklearn.exceptions import ConvergenceWarning
import warnings # I hate convergence warnings so much never show them to me
warnings.simplefilter("ignore", category=ConvergenceWarning)


#%%%%%%%%%%%%

@dataclass
class Task:

    def __post_init__(self):
        """
        Store the arguments that were used to create object
        """

        fs = fields(self)
        # for base in type(self).__bases__:
        #     basefs = set(fields(base))
        #     fs = [f for f in fs if f not in basefs]
        self.args = {f.name: getattr(self, f.name) for f in fs}

    def sample(self):
        """
        Should return a single data point, which a Model can accept,
        and be indexed bt `i' in some way
        """
        return NotImplementedError

@dataclass
class Model:
    """
    Abstract class enforcing the core functions

    A Model should contain the code for fitting the parameters given data, and
    optionally for saving and loading the parameters. Any metrics computed 
    during fitting (like training loss) are stored in a dict in the Model. 
    """

    # if there's multiple model instances, need to handle case-by-case

    def __post_init__(self):
        """
        Store the arguments that were used to create object
        """

        fs = fields(self)
        # for base in type(self).__bases__:
        #     basefs = set(fields(base))
        #     fs = [f for f in fs if f not in basefs]
        self.args = {f.name: getattr(self, f.name) for f in fs}

    def fit(self):
        """
        A function which takes in data and changes parameters
        Should also populate the metrics dictionary
        """
        return NotImplementedError

    def save(self, fname):
        """
        To save the model parameters -- NOT the metrics
        """
        pass

    def load(self, fname):
        """
        To load the model parameters -- NOT the metrics
        """
        pass

@dataclass
class PTModel(Model):
    """
    If the model is an instance of a 'NeuralNet' claass
    """

    opt_args: dict = field(default_factory=dict)
    batch_size: int = 64
    epochs: int = 100

    def fit(self, **data):
        
        self.init_metrics()
        self.net = self.init_network(**data)
        self.net.init_optimizer(**self.opt_args)

        dl = pt_util.batch_data(*[d for d in data.values()],  batch_size=self.batch_size)

        ls = []
        for epoch in range(self.epochs):
            self.loop(**data)
            self.metrics['train_loss'].append(self.net.grad_step(dl))

    def init_network(self, **data):
        """
        Should output an initialized NeuralNet object
        """
        return NotImplementedError

    def loop(self, **data):
        """
        All the code that should be run during a training loop

        could include, for example, a tqdm iterator
        """
        pass

    def init_metrics(self):
        self.metrics = {'train_loss': []}

    def save(self, fname):
        self.net.save(fname)

    def load(self, fname):
        self.net.load(fname)

#%% Base experiment
@dataclass
class Experiment:

    task: Task
    model: Model

    def run(self):
        """
        Should take in args
        """

        data = self.task.sample()
        self.model.fit(**data)

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
        # all_args = {'task': self.task, 'model': self.model}
        
        params_fname = 'parameters_'+expinf  # extension may vary 
        metrics_fname = 'metrics_'+expinf+'.pkl'
        # args_fname = 'arguments_'+expinf+'.pkl'
        
        self.model.save(f'{SAVE_DIR}{FOLDERS}{params_fname}')

        with open(SAVE_DIR+FOLDERS+metrics_fname, 'wb') as f:
            pickle.dump(self.model.metrics, f, -1)

        # with open(SAVE_DIR+FOLDERS+args_fname, 'wb') as f:
        #     pickle.dump(all_args, f, -1)

    def load_experiment(self, SAVE_DIR):
        """
        Requires that the model is specified
        """

        FOLDERS = self.folder_hierarchy()
        expinf = self.file_suffix()
        
        params_fname = 'parameters_'+expinf
        metrics_fname = 'metrics_'+expinf+'.pkl'
        
        self.model.load(f'{SAVE_DIR}{FOLDERS}{params_fname}')

        path2file = os.path.normpath(SAVE_DIR+FOLDERS+metrics_fname)
        with open(path2file , 'rb') as f:
            self.model.metrics = pickle.load(f)

    ####### file I/O functions for this experiment
    # I do this in order to standardise everything within this experiment
    # def retrive(self, DIR):
    #     """
    #     Get a list of all experiments stored in DIR
    #     """

    def folder_hierarchy(self):
        """
        This should return a string FOLDERS which child classes append to
        """

        FOLDERS = '/%s/'%self.task.__class__.__name__
        for key, val in self.task.args.items():
            FOLDERS += '/%s_%s/'%(key, su.stringify(val))

        FOLDERS += '/%s/'%self.model.__class__.__name__
        for key, val in self.model.args.items():
            FOLDERS += '/%s_%s/'%(key, su.stringify(val))

        return FOLDERS

    def file_suffix(self):
        return ''


#%% Base class
class NetworkExperiment:

    def __init__(self):
        """
        Needs to define: test_data_args, dim_inp, dim_out
        """
        return NotImplementedError


    def run(self, args):

        model = self.initialize_network(args['model'], **args['model_args'])
        self.train_network(model, **args['opt_args'])

    def folder_hierarchy(self):

        FOLDERS = super().folder_hierarchy()

        FOLDERS += self.inputs.__name__ + '/'
        FOLDERS += self.outputs.__name__ + '/'
        FOLDERS += self.exp_folder_hierarchy()

        return FOLDERS

    def exp_folder_hierarchy(self):
        return ''

    def file_suffix(self):
        return self.models[0].__name__

    def init_metrics(self):

        super().init_metrics()

        self.metrics = self.metrics | {'train_loss': []}

    def compute_metrics(self):
        pass

    def draw_data(self):
        """
        Needs to return a tuple of conditions and data, where
        the data is itself a tuple of inputs and outputs
        """

        return NotImplementedError

    def initialize_network(self, model, num_init=None, **net_args):
        """ 
        a method which allows you to tailor the model to the experiment
        """

        self.net_args = net_args

        if num_init is None:
            self.num_init = 1
        else:
            self.num_init = num_init

        nets = []
        for i in range(self.num_init):
            nets.append( model(dim_inp=self.dim_inp, 
                                dim_out=self.dim_out, **net_args) )

        return nets

    def train_network(self, models, verbose=False, skip_rep_metrics=True, skip_metrics=False, 
        conv_tol=1e-5,bsz=64, nepoch=1000, n_train_dat=5000, n_test_dat=1000, metric_period=10,
        **opt_args):

        ### book-keeping
        # self.opt_args = {k:v for k,v in locals().items() \
        #     if k not in ['model', 'verbose', 'skip_metrics', 'self']}
        self.opt_args = opt_args
        self.models = models
        # self.init = init_index

        ### generate data
        self.train_conditions, self.train_data = self.draw_data(n_train_dat)
        self.test_conditions, self.test_data = self.draw_data(**self.test_data_args)

        dset = torch.utils.data.TensorDataset(self.train_data[0], self.train_data[1])
        dl = torch.utils.data.DataLoader(dset, batch_size=bsz, shuffle=True) 
        
        ### train network
        # self.optimizer = opt_alg(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        for model in self.models:
            model.init_optimizer(**opt_args)

        expinf = self.file_suffix()
        print('Running %s ...'%expinf)

        self.init_metrics()
        for epoch in range(nepoch):

            # Actually update model #######################################
            losses = []
            prm_change = []
            for model in self.models:
                prm_init = [1*p.data for p in model.parameters()]
                losses.append(model.grad_step(dl)) # this does a pass through the data
                prm_change.append(max([torch.max(torch.abs(p1.data - p0)) for p1,p0 in zip(prm_init, model.parameters())]))

            # Measure things ##############################################
            if not np.mod(epoch, metric_period):
                for k in self.metrics.keys():
                    self.metrics[k].append([])

                self.metrics['train_loss'][-1].append( losses)

                with torch.no_grad():
                    for model in self.models:
                        self.model = model

                        if not skip_metrics:
                            self.compute_metrics()

            # print updates ############################################
            if verbose:
                print('Epoch %d: Loss=%.3f'%(epoch, -np.mean(losses)))

            # print(max(prm_change))
            if max(prm_change) <= conv_tol:
                if verbose:
                    print('Converged')
                break
            
        #### package computed metrics
        for k,v in self.metrics.items():
            self.metrics[k] = np.array(v)

#%% Multi-classification tasks

class FeedforwardExperiment(NetworkExperiment):
    """
    Basic class for multi-classification experiments. To make an instance of such an experiment,
    make a child class and define the `load_data` method. This contains all the methods
    to run the experiment, and save and load it.
    """
    def __init__(self, inputs, outputs):
        """
        Everything required to fully specify an experiment.
        
        Failure to supply the N argument will create the class in 'task only mode',
        which means that will not have a model. Call the `self.use_model` method
        to later equip it with a particular model.
        """
        
        self.inputs = inputs
        self.outputs = outputs

        self.dim_inp = inputs.dim_output
        self.dim_out = outputs.dim_output

        self.test_data_args = {'num_dat': 1000}

        self.__name__ = self.__class__.__name__

    def draw_data(self, num_dat):

        condition = np.random.choice(self.inputs.num_cond, num_dat, replace=True)
        
        inps = self.inputs(condition)
        outs = self.outputs(condition)

        return condition, (inps, outs)

    def init_metrics(self):
        self.metrics = {'train_loss': [],
                       'test_perf': [],
                       'hidden_kernel': [],
                       'linear_dim': [],
                       'sparsity': []} # put all training metrics here

    def compute_metrics(self):

        # pred, z_test = self.model(self.test_data[0])[:2]
        # _, z_train = self.model(self.train_data[0])[:2]
        pred = self.model(self.test_data[0])
        z_test = self.model.hidden(self.test_data[0])
        z_train = self.model.hidden(self.train_data[0])
        N = z_test.shape[-1]

        # print(pred.shape)
        # print(z_test.shape)

        z_test = z_test.detach().numpy().T
        z_train = z_train.detach().numpy().T

        # terr = (self.test_data[1][idx_tst,...] == (pred>=0.5)).sum(0).float()/n_compute
        terr = self.outputs.correct(pred, self.test_data[1])
        # print(terr)
        self.metrics['test_perf'][-1].append(terr)

        # representation sparsity
        self.metrics['sparsity'][-1].append(np.mean(z_test>0))

        # Dimensionality #########################################
        pr = util.participation_ratio(z_test)
        self.metrics['linear_dim'][-1].append( pr)

        # # Input and output alignment ###########################

        z_test = self.model.hidden(self.inputs(range(self.inputs.num_cond), noise=0))
        Kz_mean = util.batch_dot(z_test.detach(), z_test.detach(), transpose=True)
        self.metrics['hidden_kernel'][-1].append(Kz_mean)

    def folder_hierarchy(self):

        FOLDERS = '/%s/'%self.__name__

        FOLDERS += self.inputs.__name__ + '/'
        FOLDERS += self.outputs.__name__ + '/'
        FOLDERS += self.exp_folder_hierarchy()

        return FOLDERS


#%% Sequential classification
class RNNExperiment(NetworkExperiment):
    """
    Basic class for multi-classification experiments. To make an instance of such an experiment,
    make a child class and define the `load_data` method. This contains all the methods
    to run the experiment, and save and load it.
    """
    def __init__(self, task):
        self.task = task
        self.dim_inp = task.dim_in
        self.dim_out = task.dim_out

        self.__name__ = self.__class__.__name__

    def draw_data(self, num_dat):
        """
        needs to produce a list of conditoins, and a tuple of (inputs, outputs) or 
        (inputs, outputs, initial_hidden) with shapes (num_seq, len_seq, *),
        where * is dim_inp, dim_out, or dim_hidden
        """

        raise NotImplementedError

    def init_metrics(self):
        self.metrics = {'train_loss': [],
                        'train_perf':[],
                        'test_perf': [],
                        'decoding': [],
                        'linear_dim': []} # put all training metrics here

    def compute_representation_metrics(self, skip_metrics=True):

        pred, z = self.model(self.test_data[0].transpose(0,1))[:2]
        trn_pred = self.model(self.test_data[0].transpose(0,1))[0]
        N = z.shape[2]
        T = z.shape[1]
        nseq = z.shape[0]

        z = z_test.detach().transpse(1,2).numpy() # shape (nseq, N, T)

        terr = self.task.correct(trn_pred, self.train_data[1])
        self.metrics['train_perf'].append(terr)

        # terr = (self.test_data[1][idx_tst,...] == (pred>=0.5)).sum(0).float()/n_compute
        terr = self.task.correct(pred, self.test_data[1])
        self.metrics['test_perf'].append(terr)

        # Dimensionality #########################################
        pr = util.participation_ratio(z)
        self.metrics['linear_dim'] = np.append(self.metrics['linear_dim'], pr)

        # # things that take up time! ###################################
        # if not skip_metrics:

        #     which_time = np.repeat(range(T), self.ntrain)

        #     dclf = ta.LinearDecoder(N, self.decoded_vars.shape[-1], svm.LinearSVC)

        #     dclf.fit(z[:nseq//2,...], self.decoded_vars[:nseq//2,...], 
        #         t_=which_time, max_iter=200)

        #     dec = dclf.test(z[nseq//2:,...],  self.decoded_vars[nseq//2:,...])
        #     self.metrics['decoding'].append(dec)

    def folder_hierarchy(self):

        FOLDERS = '/%s/'%self.__name__

        FOLDERS += self.task.__name__ + '/'

        # additional folders for deviations from default
        FOLDERS += (
                    f"{'/{opt_alg.__name__}/' if self.opt_args['opt_alg'].__name__!='Adam' else ''}"
                    f"{'/l2_{weight_decay}/' if self.opt_args['weight_decay']>0 else ''}"
                    f"{'/bs_{bsz}/' if self.opt_args['bsz']!=64 else ''}"
                    ).format(**self.opt_args)
        
        return FOLDERS
        


# class delayed_logic(SequentialClassification):
#     def __init__(self, task, input_task, SAVE_DIR, time_between=20, input_channels=1, jitter=True, **expargs):
#         """
#         Generates num_cond
#         """
#         self.num_inp = input_task.dim_output # how many inputs
#         self.dim_input = len(np.unique(input_channels))
#         self.input_task = input_task
#         self.input_channels = input_channels
#         self.time_between = time_between
#         self.jitter = jitter

#         super(delayed_logic, self).__init__(task, SAVE_DIR, **expargs)
#         if input_channels == 1:
#             self.base_dir = 'results/dlog/%d-%d-%d/'%(self.num_inp, time_between, input_channels) # append at will
#         else:
#             inp_chn = ("%d"*self.num_inp)%tuple(input_channels)
#             self.base_dir = 'results/dlog/%d-%d-%s/'%(self.num_inp, time_between, inp_chn) # append at will

#     def load_data(self, SAVE_DIR, jitter_override=None):

#         if jitter_override is None:
#             jitter = self.jitter
#         else:
#             jitter = jitter_override

#         # -------------------------------------------------------------------------
#         # Import data, create the train and test sets
#         n_total = 1000*(2**self.num_inp) # hardcode the number of datapoints
#         # total_time = self.time_between*(self.num_inp-1) + 1
#         total_time = self.time_between*self.num_inp 

#         inp_condition = np.random.choice(2**self.num_inp, n_total)

#         inputs = torch.zeros(n_total, total_time, self.dim_input)
#         dt = self.time_between//2

#         input_times = np.zeros((n_total, self.num_inp))
#         input_times[:,0] = 0 
#         input_times[:,1] = self.time_between + jitter*np.random.randint(-dt,dt, size=n_total)
#         input_times[:,2] = 2*self.time_between + jitter*np.random.randint(-dt,int(1.5*dt), size=n_total)

#         vals = 2*self.input_task(inp_condition).flatten()-1
#         seq_idx = np.repeat(range(n_total),self.num_inp)
#         if self.input_channels == 1:
#             inp_idx = np.tile([0,0,0],n_total)
#         else: # assume input channel is a list
#             inp_idx = np.tile(self.input_channels, n_total)
#         inputs[seq_idx,input_times.flatten(),inp_idx] = vals

#         Y = torch.tensor(inp_condition)

#         trn = int(np.floor(0.8*n_total))

#         self.train_data = (inputs[:trn,:].float(), self.task(Y[:trn]))
#         # self.train_conditions = self.abstracts(Y[:trn])
#         self.train_conditions = Y[:trn].detach().numpy()
#         self.ntrain = trn
        
#         self.test_data = (inputs[trn:,:], 
#                           self.task(Y[trn:]))
#         # self.test_conditions = self.abstracts(Y[trn:])
#         self.test_conditions = Y[trn:].detach().numpy()
#         self.ntest = n_total - trn

#     def save_other_info(self, arg_dict):
#         arg_dict['input_dichotomies'] = self.input_task.positives
#         return arg_dict

#     def load_other_info(self, arg_dict):
#         self.task.positives = arg_dict['dichotomies']
#         self.input_task.positives = arg_dict['input_dichotomies']