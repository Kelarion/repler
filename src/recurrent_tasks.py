CODE_DIR = '/home/matteo/Documents/github/repler/src/'

import os, sys
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
import scipy
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations
import itertools as itt

# my code
import students
import assistants
import util
import grammars as gram


class ContextDependent(object):
    def __init__(self, *Ys, min_length=None, max_length=None, ctx_per_trial=1):
        """
        Each label vector Y should be (num_stim, dim_label)
        """

        if len(Ys) > 0:
            self.num_ctx = len(Ys)
        else:
            raise ValueError('Must have at least one context')

        if np.any([y.shape != Ys[0].shape for y in Ys]):
            raise ValueError('All labels must have the same dimension')

        self.num_stim = len(Ys[0])
        self.dim_out = len(Ys[0].T)


class TwoColorSelection(object):
    
    def __init__(self, num_cols, T_inp1, T_inp2, T_resp, T_tot, go_cue=True,
        jitter=3, inp_noise=0.0, dyn_noise=0.0, present_len=1, sustain_go_cue=False,
        report_cue=True, report_uncue_color=True, color_func=util.TrigColors()):
        
        self.num_col = num_cols
        
        self.T_inp1 = T_inp1
        self.T_inp2 = T_inp2
        self.T_resp = T_resp
        self.T_tot = T_tot
        
        self.go_cue = go_cue

        self.color_func = color_func

        self.jitter = jitter
        self.inp_noise = inp_noise
        self.dyn_noise = dyn_noise
        self.present_len = present_len
        self.report_cue = report_cue 
        self.report_uncue_color = report_uncue_color
        self.sustain_go = sustain_go_cue

        self.dim_in = self.color_func.dim_out*2 + 1 + 1*go_cue
        self.dim_out = self.color_func.dim_out*(1+report_uncue_color) + report_cue

        self.__name__ = f'TwoColors_{self.color_func.__name__}_{num_cols}_{T_tot}'
        
    def correct(self, pred, targ):

        # theta = np.arctan2(pred[:,1].detach(),pred[:,0].detach()).numpy()
        # diff = np.arctan2(np.exp(targ*1j - theta*1j).imag, np.exp(targ*1j - theta*1j).real)
        # trg = np.stack([np.cos(targ), np.sin(targ)]).T

        # return np.mean(np.abs(diff))
        trg = self.color_func(targ)
        prd = pred[:,:self.color_func.dim_out].detach().numpy()
        prd = prd/la.norm(prd, axis=1, keepdims=True)

        return np.mean(np.sum(prd*trg, 1))
        # return [np.mean(np.sum(prd*trg, 1)), np.mean(np.abs(diff))]

    def __call__(self, upcol, downcol, cue):
        cuecol = np.where(cue>0, upcol, downcol)
        uncuecol = np.where(cue>0, downcol, upcol)

        return cuecol, uncuecol

    def generate_data(self, n_seq, *seq_args, **seq_kwargs):
        
        upcol, downcol, cue = self.generate_colors(n_seq)
        
        inps, outs = self.generate_sequences(upcol, downcol, cue, *seq_args, **seq_kwargs)
        
        return inps, outs, upcol, downcol, cue
    
    def generate_colors(self, n_seq):
        
        upcol = np.random.choice(np.linspace(0,2*np.pi, self.num_col), n_seq)
        downcol = np.random.choice(np.linspace(0,2*np.pi, self.num_col), n_seq)
            
        cue = np.random.choice([-1.,1.], n_seq) 
        
        return upcol, downcol, cue
        
    def generate_sequences(self, upcol, downcol, cue, new_T=None, retro_only=False,
                           pro_only=False, net_size=None, jitter=None, 
                           inp_noise=None, dyn_noise=None, present_len=None):

        if inp_noise is None:
            inp_noise = self.inp_noise
        if dyn_noise is None:
            dyn_noise = self.dyn_noise
        if present_len is None:
            present_len = self.present_len
        if jitter is None:
            jitter = self.jitter

        if new_T is None:
            T = self.T_tot
        else:
            T = new_T
        n_seq = len(upcol)
        ndat = n_seq

        if net_size is None and dyn_noise > 0:
            raise IOError('cannot do dynamics noise without providing the net '
                          'size')
        elif net_size is not None:
            net_inp = net_size
        else:
            net_inp = 0
            
        col_inp = self.color_func(upcol, downcol)
        col_inp = col_inp + np.random.randn(*col_inp.shape)*inp_noise
        col_size = col_inp.shape[1]
        
        cuecol = np.where(cue>0, upcol, downcol)
        uncuecol = np.where(cue>0, downcol, upcol)
        
        cue += np.random.randn(n_seq)*inp_noise
        inps = np.zeros((n_seq, T, col_size + net_inp +
                         1 + 1*self.go_cue))
        
        t_stim1 = np.random.choice(range(self.T_inp1 - jitter, self.T_inp1 + jitter + 1),
                                   (ndat, 1))
        t_stim2 = np.random.choice(range(self.T_inp2 - jitter, self.T_inp2 + jitter + 1),
                                   (ndat, 1))
        t_targ = np.random.choice(range(self.T_resp - jitter, self.T_resp + jitter + 1),
                                  (ndat, 1))
        
        
        t_stim1 = t_stim1 + np.arange(present_len).reshape((1, -1))
        t_stim2 = t_stim2 + np.arange(present_len).reshape((1, -1))

        comb_cue_t = np.concatenate((t_stim2[:n_seq//2], t_stim1[n_seq//2:]))
        comb_col_t = np.concatenate((t_stim1[:n_seq//2], t_stim2[n_seq//2:]))

        t_stim1 = np.concatenate(t_stim1.T)
        t_stim2 = np.concatenate(t_stim2.T)
        comb_cue_t = np.concatenate(comb_cue_t.T)
        comb_col_t = np.concatenate(comb_col_t.T)
        
        retro_cue = t_stim2
        retro_col = t_stim1
        pro_cue = t_stim1
        pro_col = t_stim2
        
        seq_inds = np.tile(np.arange(n_seq), present_len)
        col_inp = np.tile(col_inp, (present_len, 1))
        cue_rep = np.tile(cue, present_len)
        if retro_only:
            inps[seq_inds, retro_col, :col_size] = col_inp # retro
            inps[seq_inds, retro_cue, col_size] = cue_rep
        elif pro_only:
            inps[seq_inds, pro_cue, col_size] = cue_rep # pro
            inps[seq_inds, pro_col, :col_size] = col_inp
        else:
            inps[seq_inds, comb_cue_t, col_size] = cue_rep
            inps[seq_inds, comb_col_t, :col_size] = col_inp

        report_list = self.color_func(cuecol)
        if self.report_uncue_color:
            report_list = np.concatenate((report_list, self.color_func(uncuecol)),
                                         axis=1)
        if self.report_cue:
            report_list = np.concatenate((report_list, np.expand_dims(cue, 1)),
                                         axis=1)
        outs = report_list.T
        
        outputs = np.zeros((n_seq, T, outs.shape[0]))
        
        if self.go_cue:
            for i, targ_i in enumerate(np.squeeze(t_targ)):
                if self.sustain_go:
                    inps[i, targ_i:, col_size+1] = 1
                else:
                    inps[i, targ_i, col_size+1] = 1
                outputs[i, targ_i:, :] = outs[:, i]

        inps[:,:,col_size+1+1*self.go_cue:] = np.random.randn(n_seq, T, net_inp)*dyn_noise
        
        return inps, outputs