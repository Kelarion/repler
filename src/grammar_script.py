import socket
import os

if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/matteo/Documents/github/repler/src/'
    SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'
elif socket.gethostname() in ['watson', 'holmes']:    
    CODE_DIR = '/rigel/home/ma3811/repler/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
else: 
    CODE_DIR = '/home/malleman/repler/'
    SAVE_DIR = '/om2/user/malleman/'
    
import sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from sklearn import svm, calibration, linear_model, discriminant_analysis, manifold
import scipy.stats as sts
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.linalg as la
import scipy.stats as sts
# import umap
from cycler import cycler

# from tqdm import tqdm

from students import MultiGLM, Feedforward
from assistants import Indicator
import experiments as exp
import util
# from recurrent import RNNModel
# from needful_functions import *

#%%
allargs = sys.argv

arglist = allargs[1:]

init = int(arglist[0])

fold = 'results/sequential/'
if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)
    
#%%
class RNNClassifier(nn.Module):
    """Fairly thin wrapper for nn.RNN with a classifier readout"""
    def __init__(self, hidden_size, input_size, num_classes=2, 
                 encoder=None, embedding=None, rnn_type='tanh',
                 **rnnargs):
        super(RNNClassifier, self).__init__()
        
        if encoder is not None:
            self.encoder = encoder
            self.use_encoder = True
        elif embedding is not None:
            self.encoder = Indicator(input_size, input_size)
            self.use_encoder = True
        else:
            self.use_encoder = False
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, **rnnargs)
        else:
            try:
                self.rnn = nn.RNN(input_size, hidden_size, nonlinearity=rnn_type, **rnnargs)
            except:
                raise ValueError("Invalid rnn_type: give from {'LSTM', 'GRU', 'tanh', 'relu'}")
        # self.rnn = nn.RNN(input_size, hidden_size, **rnnargs)
        self.rnn_type = rnn_type
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
    
    pretrained_rep = encoder(images).detach().numpy()
    reps = np.zeros(binary_seq.shape+(encoder.ndim[-1]+1*cued,))
    if cued:
        reps[(binary_seq==0)|(binary_seq==1),:-1] = pretrained_rep[(binary_seq==0)|(binary_seq==1),:]
        reps[(binary_seq==0)|(binary_seq==1),-1] = 0
        reps[binary_seq==2, -1] = pretrained_rep.mean()
    else:
        reps[(binary_seq==0)|(binary_seq==1),:] = pretrained_rep[(binary_seq==0)|(binary_seq==1),:]
        
    reps = torch.tensor(reps, requires_grad=False).float()
    
    return reps

#%% Load the digits data 
digits = torchvision.datasets.MNIST(SAVE_DIR+'digits/', download=True, 
                                    transform=torchvision.transforms.ToTensor())

# drawings = digits.data.reshape((-1,784)).float()/255

#%% generate underlying sequences
nseq = 3000
nT = 24
# dense = True # is the sequence dense? (no time without input)
dense = False
L_train = [3,5,7,12]
L_test = [2,4,6,8]
cued = True

# train set -- manually append the query token
seqs, labels = AnBn(nseq, nT, L_train, atfront=dense, cue=cued)

# test set
tseqs, tlabels = AnBn(nseq, nT, L_train, atfront=dense, cue=cued)

images, numbers = sample_images(seqs, digits)
test_images, test_numbers = sample_images(tseqs, digits)

#%% Parameters of each representation
N = 100 # size of representation
rotate_onehot = True

# Pretrained Bernoulli representation
# this_exp = exp.mnist_multiclass(util.DigitsBitwise(), SAVE_DIR, 
#                                 N=N,
#                                 init=1,
#                                 z_prior=None,
#                                 num_layer=1,
#                                 weight_decay=0)
this_exp = exp.mnist_multiclass(util.ParityMagnitude(), SAVE_DIR, 
                                N=N,
                                init=1,
                                z_prior=None,
                                num_layer=1,
                                weight_decay=0)
network, metrics, args = this_exp.load_experiment(SAVE_DIR)
bern_encoder = network.enc
bern_inp = represent(images, network.enc, seqs)
test_bern_inp = represent(test_images, network.enc, tseqs)

# Pretrained categorical representation
# this_exp = exp.mnist_multiclass(util.Digits(), SAVE_DIR, 
#                                 N=N,
#                                 init=1,
#                                 z_prior=None,
#                                 num_layer=1,
#                                 weight_decay=0)
this_exp = exp.mnist_multiclass(util.ParityMagnitudeEnumerated(), SAVE_DIR, 
                                N=N,
                                init=1,
                                z_prior=None,
                                num_layer=1,
                                weight_decay=0)
network, metrics, args = this_exp.load_experiment(SAVE_DIR)
cat_encoder = network.enc
cat_inp = represent(images, network.enc, seqs)
test_cat_inp = represent(test_images, network.enc, tseqs)

# Regularised categorical representattion
# this_exp = exp.mnist_multiclass(util.Digits(), SAVE_DIR, 
#                                 N=N,
#                                 init=None,
#                                 z_prior=None,
#                                 num_layer=1,
#                                 weight_decay=1.0)
this_exp = exp.mnist_multiclass(util.ParityMagnitudeEnumerated(), SAVE_DIR, 
                                N=N,
                                init=1,
                                z_prior=None,
                                num_layer=1,
                                weight_decay=1.0)
network, metrics, args = this_exp.load_experiment(SAVE_DIR)
regcat_encoder = network.enc
regcat_inp = represent(images, network.enc, seqs)
test_regcat_inp = represent(test_images, network.enc, tseqs)

# Random rotation of one-hot encoding to match dimension
onehot_encoder = RotatedOnehot(N, numbers.max())
indic_inp = onehot_encoder(numbers, seqs)

#%% Train and test each representation
N_rnn = 25
# rnn_type = 'relu'
# rnn_type = 'tanh'
rnn_type = 'GRU'
# rnn_type = 'LSTM'

nepoch = 2000
alg = optim.Adam
dlargs = {'num_workers': 2, 
          'batch_size': 64,
          'shuffle': True}  # dataloader arguments
optargs = {'lr': 1e-3, 'weight_decay':1e-2}
criterion = torch.nn.CrossEntropyLoss()

# define networks
rnn_bern = RNNClassifier(N_rnn, bern_inp.shape[-1], rnn_type=rnn_type)
# rnn_indic = RNNClassifier(N_rnn, numbers.max(), nonlinearity=rnn_type, embedding=True)
if rotate_onehot:
    rnn_indic = RNNClassifier(N_rnn, indic_inp.shape[-1], rnn_type=rnn_type)
else:
    rnn_indic = RNNClassifier(N_rnn, numbers.max()+1, rnn_type=rnn_type)
rnn_cat = RNNClassifier(N_rnn, cat_inp.shape[-1], rnn_type=rnn_type)
rnn_regcat = RNNClassifier(N_rnn, regcat_inp.shape[-1], rnn_type=rnn_type)
# rnn_e2e = RNNClassifier(N_rnn, N+1*cued, nonlinearity=rnn_type, encoder=e2e_encoder)

# make dataloaders
inp_idx = torch.arange(bern_inp.shape[0])
dset = torch.utils.data.TensorDataset(inp_idx, 
                                      torch.tensor(labels, requires_grad=False).type(torch.LongTensor))
trainloader = torch.utils.data.DataLoader(dset, **dlargs)

# optimizer
optimizer_bern = alg(rnn_bern.parameters(), **optargs)
optimizer_indic = alg(rnn_indic.parameters(), **optargs)
optimizer_cat = alg(rnn_cat.parameters(), **optargs)
optimizer_regcat = alg(rnn_regcat.parameters(), **optargs)
# optimizer_e2e = alg(rnn_e2e.parameters(), **optargs)

train_loss_bern = np.zeros(nepoch)
train_loss_indic = np.zeros(nepoch)
train_loss_cat = np.zeros(nepoch)
train_loss_regcat = np.zeros(nepoch)
train_loss_e2e = np.zeros(nepoch)
train_error = np.zeros(nepoch)
test_error = np.zeros(nepoch)
# with tqdm(range(nepoch), total=nepoch, postfix=[dict(bern_loss=0, cat_loss=0, indic_loss=0, l2cat_loss=0)]) as looper:
for epoch in range(nepoch):
    running_loss_bern = 0.0
    running_loss_indic = 0.0
    running_loss_cat = 0.0
    running_loss_regcat = 0.0
    # running_loss_e2e = 0.0
    
    idx = np.random.choice(test_bern_inp.shape[0], 500, replace=False)
    hid = rnn_bern.init_hidden(500)
    t_final = -(np.fliplr(tseqs[idx,:]==-1).argmin(1)+1)
    out, hid = rnn_bern(test_bern_inp[idx,:,:].transpose(1,0), hid)
    output = out[t_final, np.arange(500),:].detach().numpy()
    test_error[epoch] = 1-np.sum(output.argmax(1)==tlabels[idx])/500
    
    idx = np.random.choice(bern_inp.shape[0], 500, replace=False)
    hid = rnn_bern.init_hidden(500)
    t_final = -(np.fliplr(seqs[idx,:]==-1).argmin(1)+1)
    out, hid = rnn_bern(bern_inp[idx,:,:].transpose(1,0), hid)
    output = out[t_final, np.arange(500),:].detach().numpy()
    train_error[epoch] = 1-np.sum(output.argmax(1)==labels[idx])/500
    
    for i, (inp, lab) in enumerate(trainloader):
        # Bernoulli
        inp_seqs = bern_inp[inp,:,:].transpose(1,0)
        optimizer_bern.zero_grad()
        hid = rnn_bern.init_hidden(len(inp))
        # need to get the cue time -- the last non-padding input
        t_final = -(np.fliplr(seqs[inp,:]==-1).argmin(1)+1)
        out, _ = rnn_bern(inp_seqs, hid)
        output = out[t_final, np.arange(len(inp)),:]
        loss_bern = criterion(output, lab)
        loss_bern.backward()
        
        # Indicator
        inp_seqs = indic_inp[inp,:,:].transpose(1,0)
        optimizer_indic.zero_grad()
        hid = rnn_indic.init_hidden(len(inp))
        # need to get the cue time -- the last non-padding input
        t_final = -(np.fliplr(seqs[inp,:]==-1).argmin(1)+1)
        out, _ = rnn_indic(inp_seqs, hid)
        output = out[t_final, np.arange(len(inp)),:]
        loss_indic = criterion(output, lab)
        loss_indic.backward()
        
        # Cetegorical
        inp_seqs = cat_inp[inp,:,:].transpose(1,0)
        optimizer_cat.zero_grad()
        hid = rnn_cat.init_hidden(len(inp))
        # need to get the cue time -- the last non-padding input
        t_final = -(np.fliplr(seqs[inp,:]==-1).argmin(1)+1)
        out, _ = rnn_cat(inp_seqs, hid)
        output = out[t_final, np.arange(len(inp)),:]
        loss_cat = criterion(output, lab)
        loss_cat.backward()
        
        # Regularised categorical
        inp_seqs = regcat_inp[inp,:,:].transpose(1,0)
        optimizer_regcat.zero_grad()
        hid = rnn_regcat.init_hidden(len(inp))
        # need to get the cue time -- the last non-padding input
        t_final = -(np.fliplr(seqs[inp,:]==-1).argmin(1)+1)
        out, _ = rnn_regcat(inp_seqs, hid)
        output = out[t_final, np.arange(len(inp)),:]
        loss_regcat = criterion(output, lab)
        loss_regcat.backward()
        
        # End-to-end training
        # inp_seqs = (images[inp,:,:].transpose(1,0), seqs[inp,:].T)
        # optimizer_e2e.zero_grad()
        # hid = rnn_e2e.init_hidden(len(inp))
        # # need to get the cue time -- the last non-padding input
        # t_final = -(np.fliplr(seqs[inp,:]==-1).argmin(1)+1)
        # out, _ = rnn_e2e(inp_seqs, hid)
        # output = out[t_final, np.arange(len(inp)),:]
        # loss_e2e = criterion(output, lab)
        # loss_e2e.backward()
        
        optimizer_bern.step()
        optimizer_indic.step()
        optimizer_cat.step()
        optimizer_regcat.step()
        # optimizer_e2e.step()
        
        running_loss_bern += loss_bern.item()
        running_loss_indic += loss_indic.item()
        running_loss_cat += loss_cat.item()
        running_loss_regcat += loss_regcat.item()
        # running_loss_e2e += loss_e2e.item()
        
    train_loss_bern[epoch] = running_loss_bern/(i+1)
    train_loss_indic[epoch] = running_loss_indic/(i+1)
    train_loss_cat[epoch] = running_loss_cat/(i+1)
    train_loss_regcat[epoch] = running_loss_regcat/(i+1)
    # train_loss_e2e[epoch] = running_loss_e2e/(i+1)        
    

print('Done training!')
rnn_bern.save(SAVE_DIR+fold+'/bern_rnn_params_%d.pt'%init)
rnn_cat.save(SAVE_DIR+fold+'/cat_rnn_params_%d.pt'%init)
rnn_indic.save(SAVE_DIR+fold+'/indic_rnn_params_%d.pt'%init)
rnn_regcat.save(SAVE_DIR+fold+'/l2cat_rnn_params_%d.pt'%init)
# rnn_e2e.save(SAVE_DIR+fold+'/e2e_rnn_params_%d.pt'%init)

np.save(open(SAVE_DIR+fold+'/bern_train_loss_%d.npy'%init,'wb'), train_loss_bern)
np.save(open(SAVE_DIR+fold+'/bern_train_error_%d.npy'%init,'wb'), train_error)
np.save(open(SAVE_DIR+fold+'/bern_test_error_%d.npy'%init,'wb'), test_error)
np.save(open(SAVE_DIR+fold+'/indic_train_loss_%d.npy'%init,'wb'), train_loss_indic)
np.save(open(SAVE_DIR+fold+'/cat_train_loss_%d.npy'%init,'wb'), train_loss_cat)
np.save(open(SAVE_DIR+fold+'/regcat_train_loss_%d.npy'%init,'wb'), train_loss_regcat)
# np.save(SAVE_DIR+fold+'/_%d.npy'%init)

np.save(open(SAVE_DIR+fold+'/indic_basis_%d.npy'%init, 'wb'), onehot_encoder.basis)

#%%
ntest = 500
L_testing = list(range(1,max([max(L_train),max(L_test)])+1))

perf_bern = np.zeros(len(L_testing))
perf_indic = np.zeros(len(L_testing))
perf_regcat = np.zeros(len(L_testing))
perf_cat = np.zeros(len(L_testing))
for i,l in enumerate(L_testing):
    new_seq, new_labs = AnBn(ntest, nT, [l], cue=cued, atfront=dense)
    timg, tnum = sample_images(new_seq, digits)
    
    tinp = represent(timg, bern_encoder, new_seq)
    hid = rnn_bern.init_hidden(ntest)
    outp, _ = rnn_bern(tinp.transpose(1,0), hid)
    test_tfinal = -(np.fliplr(new_seq==-1).argmin(1)+1)
    outp = outp[test_tfinal, np.arange(new_seq.shape[0]), :].detach().numpy()
    perf_bern[i] = np.sum(outp.argmax(1)==new_labs)/ntest
    
    tinp = onehot_encoder(tnum, new_seq)
    # tinp = e2e_encoder(timg, new_seq)
    hid = rnn_indic.init_hidden(ntest)
    outp, _ = rnn_indic(tinp.transpose(1,0), hid)
    test_tfinal = -(np.fliplr(new_seq==-1).argmin(1)+1)
    outp = outp[test_tfinal, np.arange(new_seq.shape[0]), :].detach().numpy()
    perf_indic[i] = np.sum(outp.argmax(1)==new_labs)/ntest
    
    tinp = represent(timg, regcat_encoder, new_seq)
    hid = rnn_regcat.init_hidden(ntest)
    outp, _ = rnn_regcat(tinp.transpose(1,0), hid)
    test_tfinal = -(np.fliplr(new_seq==-1).argmin(1)+1)
    outp = outp[test_tfinal, np.arange(new_seq.shape[0]), :].detach().numpy()
    perf_regcat[i] = np.sum(outp.argmax(1)==new_labs)/ntest
    
    tinp = represent(timg, cat_encoder, new_seq)
    hid = rnn_cat.init_hidden(ntest)
    outp, _ = rnn_cat(tinp.transpose(1,0), hid)
    test_tfinal = -(np.fliplr(new_seq==-1).argmin(1)+1)
    outp = outp[test_tfinal, np.arange(new_seq.shape[0]), :].detach().numpy()
    perf_cat[i] = np.sum(outp.argmax(1)==new_labs)/ntest    

np.save(open(SAVE_DIR+fold+'/bern_generalization_%d.npy'%init,'wb'), perf_bern)
np.save(open(SAVE_DIR+fold+'/indic_generalization_%d.npy'%init,'wb'), perf_indic)
np.save(open(SAVE_DIR+fold+'/regcat_generalization_%d.npy'%init,'wb'), perf_regcat)
np.save(open(SAVE_DIR+fold+'/cat_generalization_%d.npy'%init,'wb'), perf_cat)

print('Done testing!')


