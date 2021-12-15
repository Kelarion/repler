CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
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
from matplotlib import animation as anime
from matplotlib import colors as mpc
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

# my code
import students
import assistants
import experiments as exp
import util
import tasks
import plotting as dicplt
import grammars as gram

#%%

K = 2
respect = False
# respect = True

# layers = [K**0,K**1,K**2]
layers = [1, 2, 3]
# layers = [1,1,1]

child_prob = 0.7
# child_prob = 1 

Data = gram.HierarchicalData(layers, fan_out=K, respect_hierarchy=respect)
            
#%% generate data

dim_inp = 50
noise = 1.0
num_seq = 5000

x_ = np.random.randn(dim_inp, Data.num_data)

seq_labs = []
seq_length = []
seqs = []
seq_toks = []
bs = []
i=0
while i<num_seq:
    
    sent = Data.random_sequence(child_prob)
    if sent.ntok<=int(0.3*Data.num_data):
        continue
    i += 1
    # labels = np.array([np.array(sent.node_tags)[sent.path_to_root([],i)[:-1]].astype(int) for i in sent.words])
    labels = Data.represent_labels([complex(sent.node_tags[i]) for i in sent.words])
    
    seq_labs.append(torch.tensor(labels).T)
    bs.append(sent.bracketed_string)
    seq_length.append(sent.ntok)

    toks = np.array([Data.terminals.index(complex(sent.node_tags[i])) for i in sent.words])
    seqs.append( torch.tensor(x_[:,toks]).T ) #+ np.random.randn(dim_inp, )
    seq_toks.append(torch.tensor(toks))
    
inputs = nn.utils.rnn.pad_sequence(seqs).float()
input_tokens = nn.utils.rnn.pad_sequence(seq_toks)
input_labels = nn.utils.rnn.pad_sequence(seq_labs)

pad_mask = torch.arange(max(seq_length)).expand(len(seq_length), max(seq_length)) < torch.tensor(seq_length).unsqueeze(1)

# positional encoding
p = torch.arange(inputs.shape[0])[:,None,None]*torch.ones(1,1,inputs.shape[-1])
d = torch.arange(inputs.shape[-1])[None,None,:]*torch.ones(inputs.shape[0],1,1)

args = p/(1000**(2*(d//2)/dim_inp))

pos_enc = torch.where(pad_mask.T[...,None], torch.where(np.mod(d,2)>0, np.cos(args), np.sin(args)), torch.tensor(0.0))

# input masking
n_mask = int(.15*max(seq_length))
which_inps = np.array([np.random.choice(s, n_mask, replace=False) for s in seq_length])
inp_mask = torch.tensor((np.arange(max(seq_length))[:,None,None] - which_inps[None,...])==0).sum(-1)>0

masked_inputs = torch.where((inp_mask*(torch.rand(inp_mask.shape)>0.15))[...,None], torch.zeros(dim_inp) , inputs)
# masked_inputs = torch.where(inp_mask[...,None], torch.zeros(dim_inp) , inputs)


#%% network (tiny transformer)

N = 100
num_layer_mlp = 2
num_layer_attn = 1
n_head = 1
nonlin = 'ReLU'

# mlp = students.Feedforward([dim_inp] + [N,]*num_layer_mlp, nonlin)
# # attn1 = students.AttentionLayer(N, N_qk=N//2, N_v=N//2)
# # attn2 = students.AttentionLayer(N, N_qk=N//2, N_v=N//2)
# attn = students.AttentionLayer(N, N_qk=N, N_v=N)
trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(N, n_head, N), num_layer_attn)

tiny_trans = students.TinyTransformer([dim_inp] + [N,]*(num_layer_mlp-1) + [dim_inp], num_layer_attn, n_head,
                                      N_qk=dim_inp//n_head, N_v=dim_inp//n_head, linear_link=False, resnorm=True)

dec = nn.Linear(dim_inp, Data.num_data)


#%% training
nepoch = 1000

dset = torch.utils.data.TensorDataset(masked_inputs.transpose(0,1),
                                      input_tokens.transpose(0,1), 
                                      inp_mask.transpose(0,1),
                                      pos_enc.transpose(0,1), 
                                      pad_mask)

dl = torch.utils.data.DataLoader(dset, batch_size=200, shuffle=True)

# optimizer = optim.Adam(list(mlp.parameters())+list(attn.parameters())+list(dec.parameters()), lr=1e-3)

# optimizer = optim.Adam(list(mlp.parameters())+list(attn1.parameters())+list(attn2.parameters())+list(dec.parameters()), lr=1e-3)

# optimizer = optim.Adam(list(trans.parameters())+list(mlp.parameters())+list(dec.parameters()), lr=1e-3)

optimizer = optim.Adam(list(tiny_trans.parameters()) + list(dec.parameters()), lr=1e-3)

train_loss = []
train_perf = []
for epoch in tqdm(range(nepoch)):
    
    # y = dec(attn(mlp(masked_inputs+pos_enc).transpose(0,1), pad_mask))
    # train_perf.append((nn.Softmax(-1)(y).argmax(-1)[inp_mask.T] == input_tokens[inp_mask].T).numpy().mean())
    
    epoch_lss = []
    epoch_perf = []
    for batch in dl:
        seqs, toks, inp_msk, pos, msk = batch
        
        optimizer.zero_grad()
        
        # z = torch.cat([attn1(mlp(seqs+pos), msk), attn2(mlp(seqs+pos), msk)], dim=-1)
        # out = dec(z)
        
        # z = attn(mlp(seqs+pos) + (seqs+pos), msk)
        # out = dec(z)
        
        # z = trans(mlp(seqs+pos).transpose(0,1), src_key_padding_mask=~msk)
        # out = dec(z.transpose(0,1))
        
        out = dec(tiny_trans((seqs+pos).transpose(0,1), msk).transpose(0,1))
        
        epoch_perf.append((out.argmax(-1)[inp_msk] == toks[inp_msk]).numpy().mean())
        
        loss = (nn.CrossEntropyLoss(reduction='none')(out.transpose(-1,-2),toks.long())*inp_msk).sum(-1).mean()
        loss.backward()
        epoch_lss.append(loss.item())
        
        optimizer.step()
    
    train_loss.append(np.mean(epoch_lss))
    train_perf.append(np.mean(epoch_perf))




