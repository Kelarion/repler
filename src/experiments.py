import os
import pickle
import warnings
import re
from time import time
from dataclasses import dataclass

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import numpy.linalg as nla
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc
import scipy.stats as sts
from itertools import permutations
from sklearn import svm, linear_model
from sklearn.decomposition import NMF
from sklearn.cluster import k_means

import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from functools import partial

# this is my code base, this assumes that you can access it
import util
import pt_util
import tasks
import recurrent_tasks as rectasks
import assistants
import students
import dichotomies as dic
import super_experiments as exp
import grammars as gram
import distance_factorization as df
import df_util
import sparse_autoencoder as spae
import bae
import bae_models
import bae_util

#############################################################################
################## Models for server interface ##############################
#############################################################################

# @dataclass
# class SBMF(exp.Model):

#     beta: float 
#     eps: float
#     br: int
#     tol: float
#     pimin: float
#     reg: str = 'sparse'
#     order: str = 'ball'

#     def fit(self, K, Strue):

#         self.metrics = {'loss':[],
#                         'mean_hamming':[],
#                         'median_hamming':[],
#                         'weighted_hamming':[],
#                         'time':[]}

#         for it in range(len(K)):

#             t0 = time()
#             S, pi = df.cuts(K[it], 
#                             branches=self.br, eps=self.eps, beta=self.beta, 
#                             order=self.order, tol=self.tol, pimin=self.pimin,
#                             reg = self.reg, rmax=2*len(K[it]))
#             self.metrics['time'].append(time()-t0)

#             S = S.toarray()
#             bestS, bestpi = df_util.mindist(K[it], S, beta=1e-5)

#             nrm = np.sum(util.center(K[it])**2)
#             ls = util.centered_kernel_alignment(K[it], bestS@np.diag(bestpi)@bestS.T)
#             ham = len(bestS) - (np.abs((2*S-1).T@Strue[it]).max(0))
#             w = np.min([(Strue[it]>0).sum(0), (Strue[it]<0).sum(0)], axis=0)

#             self.metrics['loss'].append(ls)
#             self.metrics['mean_hamming'].append(np.mean(ham))
#             self.metrics['median_hamming'].append(np.median(ham))
#             self.metrics['weighted_hamming'].append(ham@w/np.sum(w))

@dataclass
class BAE(exp.Model):

    dim_hid: int = None
    search: bool = True     # whether to include discrete search
    tree_reg: float = 0
    sparse_reg: float = 1e-2
    beta: float = 1.0
    pr_reg: float = 1e-2
    max_iter: int = None
    epochs: int = 10
    decay_rate: float = 0.88
    T0: float = 5
    batch_size: int = 1

    def fit(self, X, Strue):

        self.metrics = {'loss':[],
                        'nbs':[],
                        'hamming':[],
                        'norm_hamming': [],
                        'cond_hamming': [],
                        'norm_cond_hamming': [],
                        'time':[]}

        for it in range(len(X)):

            K = X[it]@X[it].T

            if self.dim_hid is None:
                h = len(Strue[it].T)
            else:
                h = self.dim_hid

            if self.search:
                mod = bae_models.BinaryAutoencoder(
                            dim_inp=X[it].shape[1], 
                            dim_hid=h,
                            tree_reg=self.tree_reg, 
                            sparse_reg=self.sparse_reg, 
                            weight_reg=self.pr_reg,
                            beta=self.beta
                            )
            else:
                mod = bae_models.BernVAE(
                            dim_inp=X[it].shape[1], 
                            dim_hid=h,
                            weight_reg=self.pr_reg,
                            sparse_reg=self.sparse_reg,
                            beta=self.beta
                            )

            neal = bae_util.Neal(decay_rate=self.decay_rate, 
                period=self.epochs, 
                initial=self.T0)

            dl = pt_util.batch_data(torch.FloatTensor(X[it]), 
                batch_size=self.batch_size)

            t0 = time()
            en = neal.fit(mod, dl, max_iter=self.max_iter, T_min=1e-6)
            self.metrics['time'].append(time()-t0)

            S = mod.hidden(torch.FloatTensor(X[it])).detach().numpy()
            Xhat = mod(torch.FloatTensor(X[it])).detach().numpy()

            ls = np.mean((Xhat - X[it])**2)
            mat_ham = df_util.permham(Strue[it], S)
            norm_ham = df_util.permham(Strue[it], S, norm=True)
            nbs = util.nbs(X[it], S)

            depth = df_util.porder(Strue[it])
            minham = df_util.minham(Strue[it], S, sym=True)
            cham = util.group_mean(minham, depth)
            ncham = util.group_mean(minham/Strue[it].sum(0), depth)

            self.metrics['norm_hamming'].append(np.mean(norm_ham))
            self.metrics['hamming'].append(np.mean(mat_ham))
            self.metrics['cond_hamming'].append(cham)
            self.metrics['norm_cond_hamming'].append(ncham)
            self.metrics['loss'].append(en)
            self.metrics['nbs'].append(nbs)

@dataclass
class KBMF(exp.Model):

    dim_hid: int = None
    steps: int = 1
    decay_rate: float = 0.8
    T0: float = 5
    period: int = 2
    tree_reg: float = 1e-2
    max_iter: int = None

    def fit(self, X, Strue):

        self.metrics = {'loss':[],
                        'nbs':[],
                        'hamming':[],
                        'norm_hamming': [],
                        'cond_hamming': [],
                        'norm_cond_hamming': [],
                        'time':[]}

        for it in range(len(X)):

            K = X[it]@X[it].T

            if self.dim_hid is None:
                h = len(Strue[it].T)
            else:
                h = self.dim_hid

            mod = bae_models.KernelBMF(h, tree_reg=self.tree_reg,
                scale_lr=0.9)
            neal = bae_util.Neal(decay_rate=self.decay_rate, 
                period=self.period, 
                initial=self.T0)

            # mod.init_optimizer(decay_rate=self.decay_rate,
            #     period=self.period,
            #     initial=self.T0)

            t0 = time()
            en = neal.fit(mod, X[it], max_iter=self.max_iter)
            # for i in range(self.max_iter):
            #     mod.proj()
            #     mod.grad_step()

            self.metrics['time'].append(time()-t0)

            # S = mod.S.todense()
            S = mod.S

            # bestS, bestpi = df_util.mindistX(X[it], S)

            nrm = np.sum(util.center(K)**2)
            # ls = util.centered_kernel_alignment(K, bestS@np.diag(bestpi)@bestS.T)
            # ham = len(K) - (np.abs((2*S-1).T@Strue[it]).max(0))
            cka = util.cka(Strue[it]@Strue[it].T, S@S.T)
            mat_ham = df_util.permham(Strue[it], S)
            norm_ham = df_util.permham(Strue[it], S, norm=True)
            nbs = util.nbs(X[it], S)

            depth = df_util.porder(Strue[it])
            minham = df_util.minham(Strue[it], S, sym=True)
            cham = util.group_mean(minham, depth)
            ncham = util.group_mean(minham/Strue[it].sum(0), depth)

            self.metrics['norm_hamming'].append(np.mean(norm_ham))
            self.metrics['hamming'].append(np.mean(mat_ham))
            self.metrics['cond_hamming'].append(cham)
            self.metrics['norm_cond_hamming'].append(ncham)
            self.metrics['loss'].append(cka)
            self.metrics['nbs'].append(nbs)

@dataclass
class NMF2(exp.Model):

    dim_hid: int = None
    l1_ratio: float = 1
    reg: float = 0

    def fit(self, X, Strue):

        self.metrics = {'loss':[],
                        'nbs':[],
                        'hamming':[],
                        'norm_hamming': [],
                        'cond_hamming': [],
                        'norm_cond_hamming': [],
                        'time':[]}

        for it in range(len(X)):

            if self.dim_hid is None:
                h = len(Strue[it].T)
            else:
                h = self.dim_hid

            nmf = NMF(h, l1_ratio=self.l1_ratio, alpha_W=self.reg)

            t0 = time()
            Z = nmf.fit_transform(X[it])
            S = []
            for i in range(Z.shape[1]):
                S.append(k_means(Z[:,[i]], 2)[1])
            self.metrics['time'].append(time()-t0)
            S = np.array(S.T)

            cka = util.cka(Strue[it]@Strue[it].T, S@S.T)
            mat_ham = df_util.permham(Strue[it], S)
            norm_ham = df_util.permham(Strue[it], S, norm=True)
            nbs = util.nbs(X[it], S)

            depth = df_util.porder(Strue[it])
            minham = df_util.minham(Strue[it], S, sym=True)
            cham = util.group_mean(minham, depth)
            ncham = util.group_mean(minham/Strue[it].sum(0), depth)

            self.metrics['norm_hamming'].append(np.mean(norm_ham))
            self.metrics['hamming'].append(np.mean(mat_ham))
            self.metrics['cond_hamming'].append(cham)
            self.metrics['norm_cond_hamming'].append(ncham)
            self.metrics['loss'].append(cka)
            self.metrics['nbs'].append(nbs)

@dataclass
class SBMF(exp.Model):

    dim_hid: int = None
    ortho: bool = True
    decay_rate: float = 0.8
    T0: float = 5
    period: int = 2
    sparse_reg: float = 1e-2
    tree_reg: float = 0
    pr_reg: float = 1e-2
    max_iter: int = None

    def fit(self, X, Strue):

        self.metrics = {'loss':[],
                        'nbs':[],
                        'hamming':[],
                        'norm_hamming': [],
                        'cond_hamming': [],
                        'norm_cond_hamming': [],
                        'time':[]}

        for it in range(len(X)):

            K = X[it]@X[it].T

            if self.dim_hid is None:
                h = len(Strue[it].T)
            else:
                h = self.dim_hid

            if self.ortho:
                mod = bae_models.BiPCA(h, 
                        sparse_reg=self.sparse_reg, 
                        tree_reg=self.tree_reg)
            else:
                mod = bae_models.SemiBMF(h, 
                        tree_reg=self.tree_reg,
                        weight_reg=self.pr_reg)

            neal = bae_util.Neal(decay_rate=self.decay_rate, 
                period=self.period, 
                initial=self.T0)

            # mod.init_optimizer(decay_rate=self.decay_rate,
            #     period=self.period,
            #     initial=self.T0)

            t0 = time()
            en = neal.fit(mod, X[it], max_iter=self.max_iter)

            self.metrics['time'].append(time()-t0)

            S = mod.S

            cka = util.cka(Strue[it]@Strue[it].T, S@S.T)
            mat_ham = df_util.permham(Strue[it], S)
            norm_ham = df_util.permham(Strue[it], S, norm=True)
            nbs = util.nbs(X[it], S)

            depth = df_util.porder(Strue[it])
            minham = df_util.minham(Strue[it], S, sym=True)
            cham = util.group_mean(minham, depth)
            ncham = util.group_mean(minham/Strue[it].sum(0), depth)

            self.metrics['norm_hamming'].append(np.mean(norm_ham))
            self.metrics['hamming'].append(np.mean(mat_ham))
            self.metrics['cond_hamming'].append(cham)
            self.metrics['norm_cond_hamming'].append(ncham)
            self.metrics['loss'].append(cka)
            self.metrics['nbs'].append(nbs)

@dataclass
class SAE(exp.Model):
    
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False
    neuron_resample_window: Optional[int] = None
    neuron_resample_scale: float = 0.2

    batch_size: int = 1024
    steps: int = 10_000
    log_freq: int = 100
    lr: float = 1e-3
    lr_scale: Callable[[int, int], float] = spae.constant_lr

    def fit(self, X, Strue, Wtrue=None, verbose=False):
        """
        X is shape (n_instance, n_item, dim_x)
        S is shape (n_instance, n_item, dim_latent)
        W is shape (n_instance, dim_latent, dim_x)
        """

        self.metrics = {'loss':[],
                        'mean_hamming':[],
                        'median_hamming':[],
                        'cosine':[],
                        'time':[],
                        'frac':[]}

        cfg = spae.AutoEncoderConfig(n_instances=len(X), 
                                n_input_ae=len(X[0].T), 
                                n_hidden_ae=self.n_hidden_ae,
                                l1_coeff=self.l1_coeff,
                                tied_weights=self.tied_weights,
                                neuron_resample_window=self.neuron_resample_window,
                                neuron_resample_scale=self.neuron_resample_scale,
                                )

        sae = spae.SparseAutoencoder(cfg)
        sae.init_optimizer(lr=self.lr)
        
        ptX = torch.FloatTensor(X)
        dl = pt_util.batch_data(ptX.transpose(0,1), 
                                batch_size=self.batch_size)
        
        t0 = time()
        for step in range(self.steps):            

            # Update learning rate
            step_lr = self.lr * self.lr_scale(step, self.steps)
            for group in sae.optimizer.param_groups:
                group['lr'] = step_lr
            
            ## Optimize
            self.metrics['loss'].append(sae.grad_step(dl))

            ## Active fraction
            Z = sae.hidden(ptX).detach().numpy()
            self.metrics['frac'].append((Z>1e-6).mean(-2))

        self.metrics['time'].append(time()-t0)

        ## Feature recovery
        if Wtrue is not None:
            West = sae.W_dec.detach().numpy()
            Wtrue = np.array(Wtrue)
            nrmtrue = Wtrue/la.norm(Wtrue, axis=-1, keepdims=True)
            nrmest = West/la.norm(West, axis=-1, keepdims=True)
            dot = np.einsum('ijl,ikl->ijk',nrmtrue,nrmest)
            self.metrics['cosine'].append(dot.max(-1).mean(-1))
            
        ## Binarize the latents
        Z = sae.hidden(torch.FloatTensor(X))
        S = 2*(Z.detach().numpy() > 1e-8) - 1  # binarize into 'active' and not
        
        ## Assess ground truth recovery
        ham = len(X[0]) - (np.abs(S.transpose((0,2,1))@Strue).max(1))

        self.metrics['mean_hamming'].append(np.mean(ham, axis=-1))
        self.metrics['median_hamming'].append(np.median(ham, axis=-1))

@dataclass
class BernVAE(exp.Model):
    
    dim_hid: int = None
    steps: int = 500
    temp: float = 2/3
    alpha: float = 1
    beta: float = 1 
    period: float = 10
    scale: float = 0.5
    
    def fit(self, X, Strue):
        
        self.metrics = {'loss':[],
                        'mean_hamming':[],
                        'median_hamming':[],
                        'mean_mat_ham':[],
                        'median_mat_ham':[],
                        'mean_norm_ham': [],
                        'time':[]}
        
        for it in range(len(X)):

            if self.dim_hid is None:
                dh = len(Strue[it].T)
            else:
                dh = self.dim_hid
            
            mod = students.BVAE(len(X[it].T), 
                                dh, 
                                temp=self.temp,
                                beta=self.beta,
                                scale=self.scale)
            
            ptX = torch.FloatTensor(X[it] - X[it].mean(0))
            
            dl = pt_util.batch_data(ptX)
            
            ## Fit 
            t0 = time()
            for step in range(self.steps):
                ## Exponential annealing schedule
                T = self.temp*(self.alpha**(step//self.period))
                mod.temp = T
                
                ls = mod.grad_step(dl)
                
            self.metrics['time'].append(time()-t0)
            # self.metrics['loss'].append(ls)
            
            S = np.sign(mod.q(ptX).detach().numpy())
            
            ## Assess ground truth recovery
            ham = len(X[it]) - (np.abs(S.T@Strue[it]).max(0))
            cka = util.centered_kernel_alignment(X[it]@X[it].T, S@S.T)
            mat_ham = df_util.permham(Strue[it], S)
            norm_ham = mat_ham/np.min([(Strue[it]>0).sum(0), (Strue[it]<=0).sum(0)],0)

            self.metrics['mean_norm_ham'].append(np.mean(norm_ham))
            self.metrics['mean_mat_ham'].append(np.mean(mat_ham))
            self.metrics['median_mat_ham'].append(np.median(mat_ham))
            self.metrics['mean_hamming'].append(np.mean(ham))
            self.metrics['median_hamming'].append(np.median(ham))
            self.metrics['loss'].append(cka)
            


#############################################################################
################## Tasks for server interface ###############################
#############################################################################

@dataclass
class CatTask(exp.Task):

    samps: int
    snr: float
    ratio: int = 1
    orth: bool = True
    seed: int = 0 
    nonneg: bool = False

    def latents(self):
        return NotImplementedError

    def sample(self):

        np.random.seed(self.seed)

        Xs = []
        Strues = []
        for it in range(self.samps):

            S = self.latents()

            dim = self.ratio*S.shape[1]

            X = df_util.noisyembed(S, dim, 
                snr=self.snr, orth=self.orth, 
                nonneg=self.nonneg, scl=1e-4)

            Xs.append(X)
            Strues.append(S) 

        return {'X': Xs, 'Strue': Strues}



@dataclass
class SparseFeatures(exp.Task):

    samps: int
    num_points: int 
    num_feats: int
    dim_feats: int 
    sparsity: float = 0.1
    spaced: bool = True
    enforce_nz: bool = False
    seed: int = 0

    def sample(self):

        np.random.seed(self.seed)

        Xs = []
        Strues = []
        Wtrues = []
        for j in range(self.samps):

            ## Draw random features, sparse mask, but at least one is active
            S = np.random.rand(self.num_points, self.num_feats)
            M = np.random.choice([0,1], 
                size=(self.num_points, self.num_feats),
                p=[1-self.sparsity, self.sparsity])
            if self.enforce_nz:
                foo = np.eye(self.num_feats)[np.random.choice(np.arange(self.num_feats), self.num_points)]
                M[M.sum(1)==0] = foo[M.sum(1)==0]
            S = S*M

            if self.spaced:
                W = util.chapultapec(self.num_feats, self.dim_feats)
            else:
                W = np.random.randn(self.num_feats, self.dim_feats)/np.sqrt(self.dim_feats)
            X = S@W

            Xs.append(X)
            Strues.append(S)
            Wtrues.append(W)

        return {'X': Xs, 'Strue': Strues, 'Wtrue': Wtrues}

@dataclass(kw_only=True)
class SchurCategories(CatTask):

    N: int
    p: float = 0.5
    r: int = None

    def latents(self):

        if self.r is None:
            r = int(np.sqrt(2*self.N))
        else:
            r = self.r 

        S = df_util.schurcats(self.N, self.p, r)
        it = 0
        while len(S.T) < r:
            S = df_util.schurcats(self.N, self.p, r)
            it += 1
            if it > 100:
                break

        return (1+S)//2


@dataclass
class HierarchicalCategories(exp.Task):

    N: int
    bmin: int 
    bmax: int 

    def latents(self):
        return df_util.randtree_feats(self.N, self.bmin, self.bmax)


@dataclass
class CubeCategories(exp.Task):

    samps: int
    bits: int 
    snr: float
    seed: int = 0
    ratio: int = 1
    orth: bool = True
    perturb: bool = False

    def sample(self):

        np.random.seed(self.seed)

        N = 2**self.bits

        S = util.F2(self.bits)
        if self.perturb:
            S = np.hstack([S, np.eye(2**self.bits)])

        dim = self.ratio*S.shape[1]

        Xs = []
        Strue = []
        for it in range(self.samps):

            Xs.append(df_util.noisyembed(S, dim, self.snr, self.orth, scl=1e-3))
            Strue.append(S)

        return {'X': Xs, 'Strue': Strue}

@dataclass
class GridCategories(exp.Task):

    bits: int 
    values: int
    isometric: bool = True

    def latents(self):

        S = df_util.gridcats(self.values, self.bits)
        if self.isometric:
            Srep = df_util.gridcats(self.values, self.bits)
        else:
            Srep = df_util.grid_feats(self.values, self.bits)

        return Srep

    def sample(self):

        np.random.seed(self.seed)

        N = self.values**self.bits

        S = df_util.gridcats(self.values, self.bits)
        if self.isometric:
            Srep = df_util.gridcats(self.values, self.bits)
        else:
            Srep = df_util.grid_feats(self.values, self.bits)

        dim = self.ratio*Srep.shape[1]

        Xs = []
        Strue = []
        for it in range(self.samps):

            Xs.append(df_util.noisyembed(Srep, dim, self.snr, self.orth, scl=1e-3))
            Strue.append(S)

        return {'X': Xs, 'Strue': Strue}


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

    def __init__(self, labels=None):
        """
        labels is a (10, num_label) tensor
        """

        self.transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if labels is None:
            self.dim_out = 10
            self.targ = torch.eye(10)
        else:
            self.dim_out = labels.shape[1]
            self.targ = labels

        self.dim_inp = 3    # RGB channels

        self.test_data_args = {'train': False}

    def init_metrics(self):

        self.metrics = {'train_loss':[] }

    def draw_data(self, train=True):

        dset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                        download=True, transform=self.transform)

        cond = torch.tensor(dset.targets)

        inps = torch.tensor(dset.data.transpose(0,3,1,2)).float()

        return cond, (inps, self.targ[cond])


class MNIST(exp.NetworkExperiment):

    def __init__(self, labels=None):
        """
        labels is a (10, num_label) tensor
        """

        self.transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if labels is None:
            self.dim_out = 10
            self.targ = torch.eye(10)
        else:
            self.dim_out = labels.shape[1]
            self.targ = labels

        self.dim_inp = 1    # RGB channels

        self.test_data_args = {'train': False}

    def init_metrics(self):

        self.metrics = {'train_loss':[] }

    def draw_data(self, train=True):

        dset = torchvision.datasets.MNIST(root='./data', train=train,
                            download=True, transform=self.transform)

        cond = torch.tensor(dset.targets)

        inps = torch.tensor(dset.data[:,None,...]).float()

        return cond, (inps, self.targ[cond])

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
        input_noise=0.1, seed=None, scale=1, max_rank=None):

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
        
        if max_rank is None:
            self.max_rank = num_points - 1
        elif self.alignment < 1e-6:
            self.max_rank = max_rank 
        else:
            self.max_rank = max_rank + num_targets

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
                            size=1, scale=np.max([scale, 1e-12]),
                            max_rank=self.max_rank)
        lx, vx = la.eigh(np.squeeze(Kx))

        ## Define tasks
        mns = tasks.Embedding(vx@np.diag(np.sqrt(np.abs(lx))))
        inps = tasks.LinearExpansion(mns, dim_inp, noise_var=input_noise)
        outs = tasks.BinaryLabels(H[:,these_targs].T)

        ## Save information
        self.info = {'input_kernel': Kx, 'target_kernel': Ky}

        super(RandomKernelClassification, self).__init__(inps, outs)

    def exp_folder_hierarchy(self):

        FOLDERS = (f"/{self.signal}_signal/"
                   f"/rank_{self.max_rank}/"
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