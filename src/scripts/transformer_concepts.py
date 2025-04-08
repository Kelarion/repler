CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import socket
import os
import sys
import pickle as pkl
import subprocess
from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import torch
import torch.optim as optim
import itertools as itt
from tqdm import tqdm

sys.path.append(CODE_DIR)
import util
import pt_util
import students 
import super_experiments as sxp
import experiments as exp
import server_utils as su
import grammars as gram

# import umap
from cycler import cycler
import linecache

import transformer_lens as tfl
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

device = utils.get_device()

#%%

@dataclass
class TransformerPTB(sxp.Task):
    
    layers: list[int]
    model: str
    dep_parse: bool = False
    min_length: int = 8
    max_length: int = 100
    max_size: int = 5e4 # 50k pairs maximum
    adjacent_only: bool = False
    
    def sample(self):
        
        ## Parse trees
        if self.dep_parse:
            file = SAVE_DIR + '/PTB/dependency_train_bracketed'
        else:
            file = SAVE_DIR + '/PTB/train_bracketed.txt'
        lines = linecache.getlines(file)
        
        ## transformer features
        model = HookedTransformer.from_pretrained(self.model, device=device)

        ignore = ["``","''", ".", ",",":","-LRB-","-RRB","-NONE-","$","#"]

        V1 = []
        V2 = []
        dT = []
        n_pairs = 0
        for this_line in tqdm(lines):
            
            bs = gram.ParsedSequence(this_line)
            
            if (bs.ntok < self.min_length) or (bs.ntok > self.max_length):
                continue
            
            ## Need to keep track of which token belongs to which word
            tokstr = ['<|endoftext|>']
            which_word = []
            ispunc = []
            for i,w in enumerate(bs.words):
                word_toks = model.to_str_tokens(w)[1:]
                tokstr += word_toks
                which_word += [i]*len(word_toks)
                ispunc.append(bs.word_tags[i] in ignore)
            ispunc = np.array(ispunc)
            which_word = np.array(which_word)
            
            # won't be exactly the same as the normal tokenisation
            toks = model.to_tokens(tokstr,prepend_bos=False).T 
            
            # assert len(toks) == len(tokstr)
            # assert torch.all(toks == model.to_tokens(tokstr,prepend_bos=False).T)
            
            logits, cache = model.run_with_cache(toks, remove_batch_dim=True)
            
            if self.adjacent_only:
                aye = np.arange(len(which_word)-1)
                jay = np.arange(1,len(which_word))
            else:
                aye, jay = np.triu_indices(len(which_word),1)
                
            valid = ~(ispunc[which_word[aye]]|ispunc[which_word[jay]])
            
            Z = cache.accumulated_resid()[self.layers][:,1:]
            V1.append(Z[:,aye[valid]])
            V2.append(Z[:,jay[valid]])
            
            dists = []
            for i,j in zip(aye[valid], jay[valid]):
                dists.append(bs.tree_dist(which_word[i],which_word[j]))
            dT.append(torch.FloatTensor(dists))
            
            n_pairs += len(dists)
            if n_pairs >= self.max_size:
                break
        
        V1 = torch.cat(V1, dim=1)
        V2 = torch.cat(V2, dim=1)
        dT = torch.cat(dT)
        
        return {'V1': V1, 'V2': V2, 'dT': dT}

@dataclass
class SimulatedPTB(sxp.Task):
    
    signal_dim: int
    noise_dim: int 
    noise_std: float = 1.0
    dep_parse: bool = False
    min_length: int = 8
    max_length: int = 100
    max_size: int = 5e4 # 50k pairs maximum
    adjacent_only: bool = False
    
    def sample(self):
        
        model = HookedTransformer.from_pretrained('gpt2-small', device=device)
        
        ## Parse trees
        if self.dep_parse:
            file = SAVE_DIR + '/PTB/dependency_train_bracketed'
        else:
            file = SAVE_DIR + '/PTB/train_bracketed.txt'
        lines = linecache.getlines(file)
        
        ignore = ["``","''", ".", ",",":","-LRB-","-RRB","-NONE-","$","#"]

        V1 = []
        V2 = []
        dT = []
        n_pairs = 0
        for this_line in tqdm(lines):
            
            bs = gram.ParsedSequence(this_line)
            
            if (bs.ntok < self.min_length) or (bs.ntok > self.max_length):
                continue
            
            ## Need to keep track of which token belongs to which word
            which_word = []
            ispunc = []
            for i,w in enumerate(bs.words):
                word_toks = model.to_str_tokens(w)[1:]
                which_word += [i]*len(word_toks)
                ispunc.append(bs.word_tags[i] in ignore)
            ispunc = np.array(ispunc)
            which_word = np.array(which_word)
            
            ## Silly simulated representation
            B = bs.binary()[which_word]
            U,s,V = la.svd(B)
            N = np.random.randn(len(which_word), self.noise_dim)*self.noise_std
            
            r = np.min([len(s), self.signal_dim])
            Z = np.zeros((len(which_word), self.signal_dim+self.noise_dim))
            Z[:,:r] = U[:,:r]@np.diag(s[:r])
            Z[:,self.signal_dim:] = N
            
            if self.adjacent_only:
                aye = np.arange(len(which_word)-1)
                jay = np.arange(1,len(which_word))
            else:
                aye, jay = np.triu_indices(len(which_word),1)
                
            valid = ~(ispunc[which_word[aye]]|ispunc[which_word[jay]])
            
            V1.append(torch.FloatTensor(Z[aye][valid]))
            V2.append(torch.FloatTensor(Z[jay][valid]))
            
            dists = []
            for i,j in zip(aye[valid], jay[valid]):
                dists.append(bs.tree_dist(which_word[i],which_word[j]))
            dT.append(torch.FloatTensor(dists))
            
            n_pairs += len(dists)
            if n_pairs >= self.max_size:
                break
        
        V1 = torch.cat(V1, dim=0)
        V2 = torch.cat(V2, dim=0)
        dT = torch.cat(dT)
        
        return {'V1': V1, 'V2': V2, 'dT': dT}

@dataclass
class SynProbe(sxp.Model):
    
    dim: int
    max_iter: int = 100
    batch_size: int = 64
    poisson_loss: bool = False
    
    def fit(self, V1, V2, dT):
        
        self.metrics = {'loss': [], 
                        'rank_corr': [],
                        'nbs': []}
        
        dl = pt_util.batch_data(V1, V2, dT, batch_size=self.batch_size)
        
        mod = HewMan(V1.shape[-1], self.dim, use_pois=self.poisson_loss)
        mod.init_optimizer(lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(mod.optimizer, 
                                                         mode='min', 
                                                         factor=0.1, 
                                                         patience=0)
        
        for t in tqdm(range(self.max_iter)):
            ls = mod.grad_step(dl)
            self.metrics['loss'].append(ls)
            scheduler.step(ls)
        
        x = dT.numpy()
        y = mod(V1, V2).detach().numpy()
        self.metrics['rank_corr'].append(sts.spearmanr(x,y).statistic)
        
        
        
        # sts.spearmanr(dT, ).statistic

@dataclass(eq=False)
class HewMan(students.NeuralNet):
    
    dim_inp: int
    dim_hid: int
    use_pois: bool = False 
    
    def __post_init__(self):
        super().__init__()
        
        self.B = nn.Linear(self.dim_inp, self.dim_hid, bias=False)
    
    def forward(self, W1, W2):
        """
        W1 is shape (...,T1,d)
        W2 is shape (...,T2,d)
        
        output is shape (...,T1,T2)
        """
        
        Z1 = self.B(W1)
        Z2 = self.B(W2)
        
        return ((Z1 - Z2)**2).sum(-1)
        
    def loss(self, batch):
        """
        batch is (V1, V2, dT)
        
        V1 and V2 are shape (...,num_pairs, d)
        dT is shape (..., num_pairs)
        """
        
        D = self(batch[0], batch[1])
        
        if self.use_pois:
            return nn.PoissonNLLLoss(log_input=False)(D, batch[2])
        else:
            return nn.L1Loss()(D, batch[2])
        
# def tok2word(words, toks):
#     """
#     Takes a list of words, and a list of tokens, and assigns tokens to words
#     """
    
#     t2w = []
#     for i,t in enumerate(toks):
#         found = False
#         for j,w in enumerate(words[:i]):
#             if t in w:
#                 t2w.append(i)
#                 found = True
#         if not found:
#             t2w.append(-1)
    
#     return t2w
    

#%% Syntax distance probes

task = TransformerPTB(list(range(13)), 'gpt2-small', 
                      max_size=5e4, 
                      adjacent_only=True)
data = task.sample()

loss = []
rcorr = []
control_loss = []
control_corr = []
for layer in task.layers:
    
    print('Layer %d ... '%layer)
    
    print(' ... fitting')
    mod = SynProbe(100, max_iter=100)
    mod.fit(data['V1'][layer], data['V2'][layer], data['dT'])
    
    loss.append(mod.metrics['loss'][-1])
    rcorr.append(mod.metrics['rank_corr'])

    print('... control')
    permdT = data['dT'][torch.randperm(len(data['dT']))]
    ctrmod = SynProbe(100, max_iter=100)
    ctrmod.fit(data['V1'][layer], data['V2'][layer], permdT)

    control_loss.append(ctrmod.metrics['loss'][-1])
    control_corr.append(ctrmod.metrics['rank_corr'])


# layer = list(range(13))
# task_args = {'task': TransformerPTB,
#              'layer':layer,
#              'model': 'gpt2-small'
#              }

# mod_args = {'model': SynProbe,
#             'dim': su.Set([100, 768]),
#             }


# #%%

# su.send_to_server(task_args, mod_args, send_remotely, verbose=True)

