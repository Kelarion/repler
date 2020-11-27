CODE_DIR = '/home/matteo/Documents/github/repler/src/'
SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'

import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations

from sklearn import svm, discriminant_analysis, manifold
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

from students import MultiGLM, Feedforward
from assistants import Indicator
import experiments as exp
import util
from recurrent import RNNModel
# from needful_functions import *

#%%
class RNNClassifier(nn.Module):
    """Fairly thin wrapper for nn.RNN with a classifier readout"""
    def __init__(self, hidden_size, input_size, num_classes=2, encoder=None, embedding=None, **rnnargs):
        super(RNNClassifier, self).__init__()
        
        if encoder is not None:
            self.encoder = encoder
            self.use_encoder = True
        elif embedding is not None:
            self.encoder = Indicator(input_size, input_size)
            self.use_encoder = True
        else:
            self.use_encoder = False
        
        self.rnn = nn.RNN(input_size, hidden_size, **rnnargs)
        self.decoder = nn.Linear(hidden_size, num_classes)
        # self.soxt
        
        self.init_weights()
        
        self.nhid = hidden_size
        self.ninp = input_size
        self.nout = num_classes
        
    def init_weights(self):
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_()
    
    def init_hidden(self, bsz):
        return torch.zeros(1, bsz, self.nhid)
        
    def forward(self, inp, hid):
        if self.use_encoder:
            inp = self.encoder(inp[0],inp[1])
        output, hidden = self.rnn(inp, hid)
        
        decoded = self.decoder(output)
        
        return decoded, hidden
    
    def save(self, to_path):
        """
        save model parameters to path
        """
        with open(to_path, 'wb') as f:
            torch.save(self.state_dict(), f)
    
    def load(self, from_path):
        """
        load parameters into model
        """
        with open(from_path, 'rb') as f:
            self.load_state_dict(torch.load(f))

class RotatedOnehot(object):
    def __init__(self, dim, num):
        super(RotatedOnehot, self).__init__()
        self.basis = sts.ortho_group.rvs(dim,1)[:,:num-1]
        self.dim = dim
        self.num = num
    
    def __call__(self, indices, binary_seq):
        onehot = np.eye(self.num+1)[indices,:][...,1:-1]
        expansion = np.zeros(indices.shape + (self.dim+1,))
        active = (binary_seq==0)|(binary_seq==1)
        expansion[active,:-1] = np.matmul(onehot, self.basis.T)[active,:]
        expansion[binary_seq==2,-1] = 1
        expansion[binary_seq==-1,:] = 0
        
        return torch.tensor(expansion, requires_grad=False).float()

class FeedforwardContextual(nn.Module):
    def __init__(self, *ff_args):
        super(FeedforwardContextual, self).__init__()
        self.ffn = Feedforward(*ff_args)
    
    def forward(self, inp, binary_seq):
        # inp_ = inp.clone()
        # inp_[(binary_seq==-1)|(binary_seq==2),:] = 0
        # print(inp_)
        main = self.ffn(inp)
        main[(binary_seq==-1)|(binary_seq==2),:] = 0
        final = torch.cat([main, torch.tensor(binary_seq==2).float().unsqueeze(-1)],dim=-1)
        return final

#%%

def AnBn(nseq, nT, L, eps=0.5, cue=True, align=False, atfront=True):
    """
    Generate nseq sequences according to the A^n B^n grammar
    Sequences are padded with -1, with tokens occuring at random times
    eps sets the proportion of sequences which are ungrammatical
    
    the ungrammatical ('noise') sequences are of random length and A/B proportion
    """
    
    p_gram = (1-eps)
    p_nois = eps
    # here's one way to generate the sequences, 
    # going to create an empty array, fill it with the valid sequences first
    seqs = -1*np.ones((nseq, nT+1*cue))
    
    n = int(p_gram*nseq/len(L))
    N = 0
    for l in L:
        
        valid_seqs = np.apply_along_axis(np.repeat, 1, np.repeat([[0,1]],n,0), [l, l])
        
        if cue:
            valid_seqs = np.append(valid_seqs, np.ones(n)[:,None]*2, axis=1)
        if align:
            idx = np.arange(0,nT-np.mod(nT,2*l),np.floor(nT/(2*l)))
            idx = np.ones(n,nT)*idx[None,:]
        else:
            idx = np.random.rand(n,nT+1*cue).argsort(1)[:,:(2*l)+1*cue]
            idx = np.sort(idx,1)
        np.put_along_axis(seqs[N:N+n,:], idx, valid_seqs, axis=1)
        N+=n
    
    # now I want to add noise sequences, i.e. random number of A and B tokens
    # but I want to make sure that the sparseness of the sequences isn't
    # too different from the grammatical ones -- so I set that manually
    
    thr = sts.norm.ppf(2*np.mean(L)/nT)
    noise_seqs = ((np.ones(nseq-N)[:,None]*np.arange(nT+1*cue) - np.random.choice(nT-5,(nseq-N,1)))>0).astype(int)
    noise_seqs[np.random.randn(nseq-N,nT+1*cue)>thr] = -1
    if cue:
        noise_seqs[np.arange(nseq-N), -np.argmax(np.fliplr(noise_seqs>=0),1)-1] = 2
        
    seqs[N:,:] = noise_seqs
    labels = (seqs == 0).sum(1) == (seqs==1).sum(1)
    
    # if cue:
    #     seqs = np.append(seqs, np.ones(nseq)[:,None]*2, axis=1)
    if atfront:
        # push to the front
        seqs = np.where(seqs==-1, np.nan, seqs)
        seqs = np.sort(seqs,1)
        seqs = np.where(np.isnan(seqs),-1,seqs)
    
    shf = np.random.choice(nseq,nseq,replace=False)
    seqs = seqs[shf,:]
    labels = labels[shf]
    
    return seqs, labels

def sample_images(binary_seq, digits):
    
    pm = util.ParityMagnitude()(digits)
    valid = (digits.targets <= 8) & (digits.targets>=1)
    parity = pm[valid,0].detach().numpy()
    
    drawings = digits.data[valid,:,:].reshape((-1,784)).float()/255
    
    total_max = np.prod(binary_seq.shape) # at most how many inputs do we need?
    n_perm = int(np.ceil(total_max/np.min([np.sum(parity), np.sum(parity==0)])))
    
    pos_idx = np.concatenate([np.random.permutation(np.argwhere(parity==1).squeeze()) for _ in range(n_perm)])
    neg_idx = np.concatenate([np.random.permutation(np.argwhere(parity==0).squeeze()) for _ in range(n_perm)])
    
    all_pos = pos_idx[:total_max].reshape(binary_seq.shape)
    all_neg = neg_idx[:total_max].reshape(binary_seq.shape)
    
    samples = np.ones(binary_seq.shape, dtype=int)*-1
    samples[binary_seq==0] = all_neg[binary_seq==0]
    samples[binary_seq==1] = all_pos[binary_seq==1]
    
    numbers = digits.targets[valid][samples]
    numbers[binary_seq==-1] = 0
    numbers[binary_seq==2] = digits.targets[valid].max()+1
    
    images = np.zeros(binary_seq.shape+(784,))
    images[(binary_seq==0)|(binary_seq==1),:] = drawings[samples[(binary_seq==0)|(binary_seq==1)],:]
    images = torch.tensor(images).float()
    
    return images, numbers


def represent(images, encoder, binary_seq):
    
    pretrained_rep = encoder(images).detach()
    reps = np.zeros(binary_seq.shape+(encoder.ndim[-1]+1*cued,))
    if cued:
        reps[(binary_seq==0)|(binary_seq==1),:-1] = pretrained_rep[(binary_seq==0)|(binary_seq==1),:]
        reps[(binary_seq==0)|(binary_seq==1),-1] = 0
        reps[binary_seq==2, -1] = pretrained_rep.mean()
    else:
        reps[(binary_seq==0)|(binary_seq==1),:] = pretrained_rep[(binary_seq==0)|(binary_seq==1),:]
        
    reps = torch.tensor(reps, requires_grad=False).float()
    
    return reps

#%%

fold = 'results/sequential/'

# rnn_bern = RNNClassifier(25, bern_inp.shape[-1], rnn_type=rnn_type)
# # rnn_indic = RNNClassifier(N_rnn, numbers.max(), nonlinearity=rnn_type, embedding=True)
# if rotate_onehot:
#     rnn_indic = RNNClassifier(N_rnn, indic_inp.shape[-1], rnn_type=rnn_type)
# else:
#     rnn_indic = RNNClassifier(N_rnn, numbers.max()+1, rnn_type=rnn_type)
# rnn_cat = RNNClassifier(N_rnn, cat_inp.shape[-1], rnn_type=rnn_type)
# rnn_regcat = RNNClassifier(N_rnn, regcat_inp.shape[-1], rnn_type=rnn_type)

bern_loss = np.zeros((30, 2000))*np.nan
cat_loss = np.zeros((30, 2000))*np.nan
indic_loss = np.zeros((30, 2000))*np.nan
regcat_loss = np.zeros((30, 2000))*np.nan
test = np.zeros((30,2000))*np.nan
train = np.zeros((30,2000))*np.nan

bern_gen = np.zeros((30, 12))*np.nan
cat_gen = np.zeros((30, 12))*np.nan
indic_gen = np.zeros((30, 12))*np.nan
regcat_gen = np.zeros((30, 12))*np.nan
for init in range(30,60):
    # rnn_bern.save(SAVE_DIR+fold+'/bern_rnn_params_%d.pt'%init)
    # rnn_cat.save(SAVE_DIR+fold+'/cat_rnn_params_%d.pt'%init)
    # rnn_indic.save(SAVE_DIR+fold+'/indic_rnn_params_%d.pt'%init)
    # rnn_regcat.save(SAVE_DIR+fold+'/l2cat_rnn_params_%d.pt'%init)
    # rnn_e2e.save(SAVE_DIR+fold+'/e2e_rnn_params_%d.pt'%init)
    
    try:
        b_loss = np.load(SAVE_DIR+fold+'/bern_train_loss_%d.npy'%init)
        btrn_err = np.load(SAVE_DIR+fold+'/bern_train_error_%d.npy'%init)
        btst_err = np.load(SAVE_DIR+fold+'/bern_test_error_%d.npy'%init)
        i_loss = np.load(SAVE_DIR+fold+'/indic_train_loss_%d.npy'%init)
        c_loss = np.load(SAVE_DIR+fold+'/cat_train_loss_%d.npy'%init)
        r_loss = np.load(SAVE_DIR+fold+'/regcat_train_loss_%d.npy'%init)
        
        b_gen = np.load(SAVE_DIR+fold+'/bern_generalization_%d.npy'%init)
        i_gen = np.load(SAVE_DIR+fold+'/indic_generalization_%d.npy'%init)
        c_gen = np.load(SAVE_DIR+fold+'/cat_generalization_%d.npy'%init)
        r_gen = np.load(SAVE_DIR+fold+'/regcat_generalization_%d.npy'%init)
    except:
        continue
    # np.save(SAVE_DIR+fold+'/_%d.npy'%init)
    
    bern_loss[init-30,:] = b_loss
    indic_loss[init-30,:] = i_loss
    cat_loss[init-30,:] = c_loss
    regcat_loss[init-30,:] = r_loss
    
    bern_gen[init-30,:] = b_gen
    indic_gen[init-30,:] = i_gen
    cat_gen[init-30,:] = c_gen
    regcat_gen[init-30,:] = r_gen
    
    train[init-30,:] = btrn_err
    test[init-30,:] = btst_err

#%%

plt.figure()
plt.plot(range(1,2001), np.nanmean(bern_loss,0))
plt.fill_between(range(1,2001), 
                 np.nanmean(bern_loss,0)+np.nanstd(bern_loss,0),
                 np.nanmean(bern_loss,0)-np.nanstd(bern_loss,0),
                 alpha=0.5)
plt.plot(range(1,2001), np.nanmean(cat_loss,0))
plt.fill_between(range(1,2001), 
                 np.nanmean(cat_loss,0)+np.nanstd(cat_loss,0),
                 np.nanmean(cat_loss,0)-np.nanstd(cat_loss,0),
                 alpha=0.5)
plt.plot(range(1,2001), np.nanmean(regcat_loss,0))
plt.fill_between(range(1,2001), 
                 np.nanmean(regcat_loss,0)+np.nanstd(regcat_loss,0),
                 np.nanmean(regcat_loss,0)-np.nanstd(regcat_loss,0),
                 alpha=0.5)
plt.plot(range(1,2001), np.nanmean(indic_loss,0))
plt.fill_between(range(1,2001), 
                 np.nanmean(indic_loss,0)+np.nanstd(indic_loss,0),
                 np.nanmean(indic_loss,0)-np.nanstd(indic_loss,0),
                 alpha=0.5)
plt.semilogx()
plt.legend(['Bernoulli', 'Categorical', 'Regularized categorical', 'One-hot'])

#%%
plt.figure()
plt.plot(range(1,13), np.nanmean(bern_gen,0), marker='o')
plt.fill_between(range(1,13), 
                 np.nanmean(bern_gen,0)+np.nanstd(bern_gen,0),
                 np.nanmean(bern_gen,0)-np.nanstd(bern_gen,0),
                 alpha=0.5)
plt.plot(range(1,13), np.nanmean(cat_gen,0), marker='o')
plt.fill_between(range(1,13), 
                 np.nanmean(cat_gen,0)+np.nanstd(cat_gen,0),
                 np.nanmean(cat_gen,0)-np.nanstd(cat_gen,0),
                 alpha=0.5)
plt.plot(range(1,13), np.nanmean(regcat_gen,0), marker='o')
plt.fill_between(range(1,13), 
                 np.nanmean(regcat_gen,0)+np.nanstd(regcat_gen,0),
                 np.nanmean(regcat_gen,0)-np.nanstd(regcat_gen,0),
                 alpha=0.5)
plt.plot(range(1,13), np.nanmean(indic_gen,0), marker='o')
plt.fill_between(range(1,13), 
                 np.nanmean(indic_gen,0)+np.nanstd(indic_gen,0),
                 np.nanmean(indic_gen,0)-np.nanstd(indic_gen,0),
                 alpha=0.5)
plt.ylim([0,1.1])
plt.plot([1,13],[0.5,0.5],'--', c=(0.5,0.5,0.5))
plt.plot([[3,5,7,12],[3,5,7,12]],plt.ylim(),'-.', c=(0.5,0.5,0.5))

plt.legend(['Bernoulli', 'Categorical','Regularized categorical','One-hot','chance','training set'])

plt.ylabel('Test accuracy')
plt.xlabel('n of testing set')


