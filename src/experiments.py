import os
import pickle
import warnings
import re

import torch
import torchvision
import torchvision.transforms as transforms
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
import tasks
import recurrent_tasks as rectasks
import assistants
import dichotomies as dic
import super_experiments as exp
import grammars as gram


######################################################################
"""
Each class should include an __init__ method, and initialize_network method, 
and usually compute_metrics and exp_folder_hierarchy methods
"""
######################################################################

class HierarchicalClasses(exp.FeedforwardExperiment):

    def __init__(self, input_task, dim_inp, num_vars,
                 input_noise=0.1, K=2, respect_hierarchy=True):

        self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

        self.DGP = gram.HierarchicalData(num_vars, 
            fan_out=K, respect_hierarchy=respect_hierarchy)

        out_task = tasks.BinaryLabels(self.DGP.labels(self.DGP.terminals))

        inp_task = input_task(self.DGP.num_data, dim_inp, input_noise)

        self.dim_inp = dim_inp
        self.num_cond = self.DGP.num_data
        self.num_var = out_task.num_var

        super(HierarchicalClasses, self).__init__(inp_task, out_task)

class LogicTask(exp.FeedforwardExperiment):

    def __init__(self, inp_dics, out_dics, noise=0.1, dim_inp=100):

        self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

        self.dim_inp = dim_inp

        inps = tasks.LinearExpansion(tasks.RandomDichotomies(d=inp_dics), 
            noise_var=noise,
            dim_pattern=dim_inp)
        outs = tasks.RandomDichotomies(d=out_dics)

        super(LogicTask, self).__init__(inps, outs)


class OODFF(exp.FeedforwardExperiment):

    def __init__(self, inputs, outputs, train_split):

        self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

        self.train_split = train_split

        super(OODFF, self).__init__(inputs, outputs)

        self.test_data_args = {'num_dat':1000, 'train': False}


    def draw_data(self, num_dat, train=True):

        if train:
            samps = self.train_split
        else:
            samps = np.setdiff1d(np.arange(self.inputs.num_cond), self.train_split)

        condition = np.random.choice(samps, num_dat, replace=True)
        
        inps = self.inputs(condition)
        outs = self.outputs(condition)

        return condition, (inps, outs)



# class LogicTask(exp.FeedforwardExperiment):

#     def __init__(self, inp_dics, out_dics, noise=0.1, dim_inp=100):

#         self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

#         self.dim_inp = dim_inp

#         inps = tasks.LinearExpansion(tasks.RandomDichotomies(d=inp_dics), 
#             noise_var=noise,
#             dim_pattern=dim_inp)
#         outs = tasks.RandomDichotomies(d=out_dics)

#         super(LogicTask, self).__init__(inps, outs)
    

class Cifar10(exp.NetworkExperiment):

    def __init__(self):

        self.transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dim_inp = 3
        self.dim_out = 10

        self.test_data_args = {'train': False}

    def init_metrics(self):

        self.metrics = {'train_loss':[]} 

    def draw_data(self, train=True):

        dset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                        download=True, transform=self.transform)

        cond = torch.tensor(dset.targets)

        inps = torch.tensor(dset.data.transpose(0,3,1,2)).float()

        return cond, (inps, cond)


class RandomOrthogonal(exp.FeedforwardExperiment):

    def __init__(self, num_bits, num_targets, dim_inp, signal=None, alignment=None, 
        input_noise=0.1, seed=None, scale=1, rand_comps=True):

        self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

        ## Parse parameters
        self.num_cond = 2**num_bits
        self.num_var = num_bits
        if alignment is not None:
            self.alignment = alignment
            self.signal = (self.num_cond - 1)*(alignment**2)/num_targets
            if signal is not None:
                warnings.warn('Both signal and alignment provided, using the alignment')
        elif signal is not None:
            self.signal = signal
            self.alignment = np.sqrt(num_targets*signal/(self.num_cond - 1))
        else:
            raise ValueError('Must provide either signal or alignment')
        
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        self.scale = scale

        ## Generate hadamard matrix
        F = util.F2(num_bits) # F2
        lex = np.sort(F, axis=1)*(np.argsort(F, axis=1) + 1) 
        idx = np.lexsort(np.hstack([lex, F.sum(1, keepdims=True)]).T)
        # F = F[idx] # put it in kinda-lexicographic order
        
        H = np.mod(F@F[idx[1:]].T, 2)
        
        ## Choose targets
        # heuristic to make them equally difficult
        # num_guys = spc.binom(num_bits, np.arange(num_bits+1))
        # if np.max(num_guys) < num_targets:
        #     this_l = np.where(np.cumsum(num_guys) >= num_targets)[0][-1]
        # else:
        #     this_l = np.where(num_guys >= num_targets)[0][-1]
        # these_targs = (F[1:].sum(1) >= this_l)
        # these_targs *= np.cumsum(these_targs) <= num_targets
        these_targs = np.arange(self.num_cond - 1) < num_targets
        # print(these_targs)

        ## Draw inputs 
        pi_x = util.sample_aligned(1*these_targs, self.alignment, 
                                    size=1, scale=np.max([scale, 1e-12]))
        pi_x = np.squeeze(np.abs(pi_x))

        ## Define tasks
        mns = tasks.Embedding((2*H-1)@np.diag(np.sqrt(pi_x)))
        inps = tasks.LinearExpansion(mns, dim_inp, noise_var=input_noise)
        outs = tasks.BinaryLabels(H[:,these_targs].T)

        super(RandomOrthogonal, self).__init__(inps, outs)

    def exp_folder_hierarchy(self):

        FOLDERS = (f"/{self.signal}_signal/"
                   f"/seed_{'None' if self.seed is None else self.seed}/"
                   f"/scale_{self.scale}/")

        return FOLDERS


class RandomKernelClassification(exp.FeedforwardExperiment):

    def __init__(self, num_points, num_targets, dim_inp, signal=None, alignment=None, 
        input_noise=0.1, seed=None, scale=1):

        self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

        ## Parse parameters
        self.num_cond = num_points
        self.num_var = num_points
        if alignment is not None:
            self.alignment = alignment
            self.signal = (self.num_cond - 1)*(alignment**2)/num_targets
            if signal is not None:
                warnings.warn('Both signal and alignment provided, using the alignment')
        elif signal is not None:
            self.signal = signal
            self.alignment = np.sqrt(num_targets*signal/(self.num_cond - 1))
        else:
            raise ValueError('Must provide either signal or alignment')
        
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        self.scale = scale

        ## Generate hadamard matrix
        num_bits = int(np.ceil(np.log2(num_points)))
        F = util.F2(num_bits) # F2
        lex = np.sort(F, axis=1)*(np.argsort(F, axis=1) + 1) 
        idx = np.lexsort(np.hstack([lex, F.sum(1, keepdims=True)]).T) 
        H = np.mod(F@F[idx[1:]].T, 2)
        
        ## Choose targets
        these_targs = np.arange(self.num_cond-1) < num_targets
        Y = H[:,these_targs]

        ## Draw inputs 
        Ky = (2*Y-1)@(2*Y-1).T
        Kx = util.random_psd(Ky, self.alignment,
                            size=1, scale=np.max([scale, 1e-12]))
        lx, vx = la.eigh(np.squeeze(Kx))

        ## Define tasks
        mns = tasks.Embedding(vx@np.diag(np.sqrt(np.abs(lx))))
        inps = tasks.LinearExpansion(mns, dim_inp, noise_var=input_noise)
        outs = tasks.BinaryLabels(H[:,these_targs].T)

        super(RandomKernelClassification, self).__init__(inps, outs)

    def exp_folder_hierarchy(self):

        FOLDERS = (f"/{self.signal}_signal/"
                   f"/seed_{'None' if self.seed is None else self.seed}/"
                   f"/scale_{self.scale}/")

        return FOLDERS



class EpsilonSeparableXOR(exp.FeedforwardExperiment):

    def __init__(self, dim_inp, epsilon=0.0, input_noise=0.1):

        self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

        out_task = tasks.RandomDichotomies(d=[(1,2)])

        inp_task = tasks.NudgedXOR(dim_inp//3, nudge_mag=epsilon, noise_var=input_noise, 
                                    random=False, rotated=False)

        self.dim_inp = dim_inp
        self.num_cond = 4
        self.num_var = 1

        super(EpsilonSeparableXOR, self).__init__(inp_task, out_task)

    def init_metrics(self):
        self.metrics = {'train_loss': [],
                       'test_perf': [],
                       'parallelism': [],
                       'decoding': [],
                       'mean_grad': [],
                       'std_grad': [],
                       'ccgp': [],
                       'hidden_kernel': [],
                       'deriv_kernel':[],
                       'linear_dim': [],
                       'sparsity': []} # put all training metrics here

    def compute_metrics(self):

        x_ = self.inputs(np.arange(4), noise=0)

        c = torch.matmul(self.model.W, x_.T) + self.model.b1
        z = self.model.activation(c)
        deriv = self.model.activation.deriv(c).numpy()

        K_deriv = util.dot_product(deriv-deriv.mean(1,keepdims=True), deriv-deriv.mean(1,keepdims=True))
        K_z = util.dot_product(z-z.mean(1,keepdims=True), z-z.mean(1,keepdims=True))

        self.metrics['deriv_kernel'][-1].append(K_deriv)
        self.metrics['hidden_kernel'][-1].append(K_z)

    def exp_folder_hierarchy(self):

        FOLDERS = (f"/{'J_trained' if self.opt_args['train_outputs'] else 'J_fixed'}/"
                    f"/{'rms_prop' if self.opt_args['do_rms'] else ''}/"
                    f"/{self.net_args['out_weight_distr'].__name__}/")

        return FOLDERS


# class RandomInputMultiClass(exp.FeedforwardExperiment):

#     def __init__(self, dim_inp, num_bits, num_class, class_imbalance=0, input_noise=0.1,
#         center=True):
#         """
#         Number of items is 2**num_bits

#         Only works for imbalance=0
#         """

#         self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

#         labs = (1+la.hadamard(2**num_bits)[1:(num_class+1)])/2
#         out_task = tasks.BinaryLabels(labs.astype(int))

#         inp_task = tasks.RandomPatterns(2**num_bits, dim_inp, 
#             noise_var=input_noise, center=center, random=False)

#         super(RandomInputMultiClass, self).__init__(inp_task, out_task)

    # def init_metrics(self):
    #     self.metrics = {'train_loss': [],
    #                    'test_perf': [],
    #                    'parallelism': [],
    #                    'decoding': [],
    #                    'mean_grad': [],
    #                    'std_grad': [],
    #                    'ccgp': [],
    #                    'hidden_kernel': [],
    #                    'deriv_kernel':[],
    #                    'linear_dim': [],
    #                    'sparsity': []} # put all training metrics here

    # def compute_metrics(self):

    #     x_ = self.inputs(np.arange(4), noise=0)

    #     c = torch.matmul(self.model.W, x_.T) + self.model.b1
    #     z = self.model.activation(c)
    #     deriv = self.model.activation.deriv(c).numpy()

    #     K_deriv = util.dot_product(deriv-deriv.mean(1,keepdims=True), deriv-deriv.mean(1,keepdims=True))
    #     K_z = util.dot_product(z-z.mean(1,keepdims=True), z-z.mean(1,keepdims=True))

    #     self.metrics['deriv_kernel'][-1].append(K_deriv)
    #     self.metrics['hidden_kernel'][-1].append(K_z)

    # def compute_metrics(self):

    #     x_ = self.inputs(np.arange(4), noise=0)

    #     self.metrics['weights_proj'][-1].append(torch.matmul(self.model.W, x_.T).numpy() )


    def exp_folder_hierarchy(self):

        FOLDERS = (f"/{'J_trained' if self.opt_args['train_outputs'] else 'J_fixed'}/"
                    f"/{'rms_prop' if self.opt_args['do_rms'] else ''}/"
                    f"/{self.net_args['num_k']}/")

        return FOLDERS


class WeightDynamics(exp.FeedforwardExperiment):

    def __init__(self, inp_task, out_task):

        self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

        super(WeightDynamics, self).__init__(inp_task, out_task)

    def init_metrics(self):
        self.metrics = {'train_loss': [],
                       'weights_proj': []} # put all training metrics here

    def compute_representation_metrics(self, skip_metrics):
        return

    def compute_metrics(self):

        x_ = self.inputs(np.arange(4), noise=0)

        self.metrics['weights_proj'][-1].append(torch.matmul(self.model.W, x_.T).numpy() )

    def exp_folder_hierarchy(self):

        FOLDERS = (f"/{'rms_prop' if self.opt_args['do_rms'] else ''}/"
                    f"/{self.net_args['out_weight_distr'].__name__}/")

        return FOLDERS

##############################################################################

class SwapErrors(exp.RNNExperiment):

    def __init__(self, T_inp1, T_inp2, T_resp, T_tot, num_cols=32,
        jitter=3, inp_noise=0.0, dyn_noise=0.0, present_len=1, go_cue=True,
        report_cue=True, report_uncue_color=True, color_func=util.TrigColors(), 
        sustain_go_cue=False):

        self.exp_prm = {k:v for k,v in locals().items() if k not in ['self', '__class__']}

        self.test_data_args = {'num_dat':2000,  
                               'jitter':0,
                               'retro_only':True}

        task = rectasks.TwoColorSelection(**self.exp_prm)

        super(SwapErrors, self).__init__(task)

    def draw_data(self, num_dat, *args, **kwargs):

        inp, out, upc, downc, cue = self.task.generate_data(num_dat, 
            net_size=self.models[0].nhid, *args, **kwargs)

        # self.train_data = (torch.tensor(inp).float(), torch.tensor(out).float())
        # self.train_conditions = (upc, downc, cue)

        return (upc, downc, cue), (torch.tensor(inp).float(), torch.tensor(out).float())

    # def package_data(self):

    #     trn = self.draw_data(self.opt_args['n_train_dat'], net_size=self.model.nhid)

    #     self.train_data = (torch.tensor(trn[1][0]).float(), torch.tensor(trn[1][1]).float())
    #     self.train_conditions = trn[0]

    #     self.test_data = []
    #     self.test_conditions = []
    #     for sig in self.test_noise:
    #         tst = self.draw_data(self.opt_args['n_test_dat'], net_size=self.model.nhid, 
    #                          jitter=0, retro_only=True, dyn_noise=sig)

    #         self.test_data.append((torch.tensor(tst[1][0]).float(), 
    #             torch.tensor(tst[1][1]).float()))
    #         self.test_conditions.append( tst[0])

    #     self.decoded_vars = [] # decode cue during time

    #     dset = torch.utils.data.TensorDataset(self.train_data[0], self.train_data[1])
    #     return torch.utils.data.DataLoader(dset, batch_size=self.opt_args['bsz'], shuffle=True) 

    def folder_hierarchy(self):

        FOLDERS = '/%s/'%self.__name__

        FOLDERS += self.task.__name__ + '/'

        FOLDERS += (
                    f"{'/{T_inp1}_{T_inp2}_{T_resp}/'}"
                    f"{'/report_cue_{report_cue}/' if not self.exp_prm['report_cue'] else ''}"
                    f"{'/report_uncue_{report_uncue_color}/' if not self.exp_prm['report_uncue_color'] else ''}"
                    f"{'/present_len_{present_len}/'}"
                    f"{'/jitter_{jitter}/'}"
                    f"{'/train_noise_{dyn_noise:.2f}/'}"
                    ).format(**self.exp_prm)

        # additional folders for deviations from default
        FOLDERS += (
                    f"{'/{opt_alg.__name__}/' if self.opt_args['opt_alg'].__name__!='Adam' else ''}"
                    f"{'/l2_{weight_decay}/' if self.opt_args['weight_decay']>0 else ''}"
                    ).format(**self.opt_args)
        
        return FOLDERS

    def init_metrics(self):
        self.metrics = {'train_loss': [],
                        'train_perf':[],
                        'train_swaps':[],
                        'test_perf': [],
                        'test_swaps':[],
                        'decoding': [],
                        'linear_dim': []} # put all training metrics here

    def compute_representation_metrics(self, skip_metrics=True):

        trn_pred, z = self.model(self.train_data[0].transpose(0,1))[:2]
        N = z.shape[2]
        T = z.shape[1]
        nseq = z.shape[0]

        z = z.detach().transpose(1,2).numpy() # shape (nseq, N, T)

        cuecol, uncuecol = self.task(*self.train_conditions)
        terr = self.task.correct(trn_pred, cuecol)
        self.metrics['train_perf'].append(terr)
        terr = self.task.correct(trn_pred, uncuecol)
        self.metrics['train_swaps'].append(terr)

        pred = self.model(self.test_data[0].transpose(0,1))[0]

        cuecol, uncuecol = self.task(*self.test_conditions)

        terr = self.task.correct(pred, cuecol)
        tswp = self.task.correct(pred, uncuecol)

        self.metrics['test_perf'].append(terr)
        self.metrics['test_swaps'].append(tswp)

        # Dimensionality #########################################
        pr = util.participation_ratio(z)
        self.metrics['linear_dim'].append(pr)

        # things that take up time! ###################################
        # if not skip_metrics:

        #     which_time = np.repeat(range(T), self.ntrain)

        #     dclf = ta.LinearDecoder(N, self.decoded_vars.shape[-1], svm.LinearSVC)

        #     dclf.fit(z[:nseq//2,...], self.decoded_vars[:nseq//2,...], 
        #         t_=which_time, max_iter=200)

        #     dec = dclf.test(z[nseq//2:,...],  self.decoded_vars[nseq//2:,...])
        #     self.metrics['decoding'].append(dec)