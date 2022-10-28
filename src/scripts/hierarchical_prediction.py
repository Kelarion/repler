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
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.manifold import MDS

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

# my code
import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as dicplt
import grammars as gram

#%% custom classes to allow for identity gradients
class RayLou(nn.ReLU):
    def __init__(self, linear_grad=False):
        super(RayLou,self).__init__()
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return (x>0).float()

class Poftslus(nn.Softplus):
    def __init__(self, beta=1, linear_grad=False):
        super(Poftslus,self).__init__(beta=beta)
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return (1/(1+torch.exp(-self.beta*x))).float()

class NoisyRayLou(nn.ReLU):
    def __init__(self, beta=1, linear_grad=False):
        super(NoisyRayLou,self).__init__()
        self.linear_grad = linear_grad
        self.beta = beta
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return 0.5*(1+torch.erf(x/(self.beta*np.sqrt(2)))).float()

class TanAytch(nn.Tanh):
    def __init__(self, linear_grad=False, rand_grad=False):
        super(TanAytch,self).__init__()
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return 1-nn.Tanh()(x).pow(2)

class NoisyTanAytch(nn.Tanh):
    def __init__(self, noise=1, linear_grad=False, rand_grad=False):
        super(NoisyTanAytch,self).__init__()
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
        self.noise = noise
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return torch.exp(-x.pow(2)/(1+(2*self.noise**2)))

class HardTanAytch(nn.Hardtanh):
    def __init__(self, linear_grad=False, rand_grad=False, vmin=-1, vmax=1):
        super(HardTanAytch,self).__init__(vmin, vmax)
        self.linear_grad = linear_grad
        self.rand_grad = rand_grad
        self.vmin = vmin
        self.vmax = vmax
    def deriv(self, x):
        if self.linear_grad:
            if self.rand_grad:
                return torch.rand(x.shape)
            else:
                return torch.ones(x.shape)
        else:
            return ((x<self.vmax)&(x>self.vmin)).float()

class Iden(nn.Identity):
    def __init__(self, linear_grad=False):
        super(Iden,self).__init__()
        self.linear_grad = linear_grad
    def deriv(self, x):
        if self.linear_grad:
            return torch.ones(x.shape)
        else:
            return torch.ones(x.shape)

# class SharedWeights(nn.Module):
    


#%%

# labs = [set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3])] # overcomplete
labs = [set([0,2]), set([0,3]), set([1,2]), set([1,3])] # disentangled 
# labs = [set([0,2]), set([0,3]), set([1,4]), set([1,5])] # hierarchical
# labs = [set(c) for c in combinations(range(6),2)] # overcomplete hierarchical
# labs = [set([0,2]), set([0,3]), set([1,2]), set([1,4])] # asymmetric
# labs = [set([0,2]), set([0,3]), set([1,4]), set([1,5]), set([0,4]), set([0,5]), set([1,2]), set([1,3])]


Data = gram.LabelledItems(labels=labs)

# Data = gram.RegularTree([1,1,1], fan_out=2, respect_hierarchy=True)

#%%

F = Data.similar_representation(only_leaves=False, similarity='dca', tol=1e-10)
feats = util.pca_reduce(F.T, num_comp=3)
# feats = MDS(3).fit_transform(F)

n = list(Data.similarity_graph)
node_xyz = np.array([feats[n.index(v)] for v in sorted(Data.similarity_graph)])
# edge_xyz = np.array([(feats[n.index(u)], feats[n.index(v)]) for u, v in Data.similarity_graph.edges()])
uv = np.argsort(Data.similarity_graph)[np.array(Data.similarity_graph.edges)]
edge_u = feats[uv[:,0],:]
edge_v =  feats[uv[:,1],:] - edge_u

leaves = np.isin(sorted(Data.similarity_graph),Data.items)
observed = np.isin(sorted(Data.similarity_graph), [])

ax = plt.subplot(projection='3d')
ax.scatter(*node_xyz[leaves,:].T, s=300, ec="w", c='r', zorder=3)
ax.scatter(*node_xyz[(~leaves)&observed,:].T, s=100, ec="w", c='k', zorder=3)
ax.scatter(*node_xyz[(~leaves)&(~observed),:].T, s=100, ec="k", c='w', zorder=3)

ax.quiver(edge_u[:,0],edge_u[:,1],edge_u[:,2], 
          edge_v[:,0],edge_v[:,1],edge_v[:,2],
          color=(0.5,0.5,0.5), zorder=-1)

# ax.quiver(feats[uv[:,0],0],feats[uv[:,0],1],feats[uv[:,0],2], feats[uv[:,1],0],feats[uv[:,1],1],feats[uv[:,1],2])

# for vizedge in edge_xyz:
    # ax.plot(*vizedge.T, color="tab:gray")
    
# for i in sorted(Data.similarity_graph):
#     ax.text(node_xyz[i,0],node_xyz[i,1],node_xyz[i,2], s=Data.similarity_graph.nodes('category')[i]) 

ax.axis('off')

#%%
# dim_inp = 50
dim_inp = 4
max_children = 2
child_prob = 0.7

num_sent = 5000

seqs = []
left_word = []
right_word = []
which_seq = []
for s in range(num_sent):
    sent = Data.random_sequence(child_prob, replace=max_children>Data.fan_out, max_child=max_children)
    
    which_word = np.array([np.isin(Data.items, complex(sent.node_tags[x])) for x in sent.words])
    
    left_word.append(which_word[:-1,:].argmax(1))
    right_word.append(which_word[1:,:].argmax(1))
    which_seq.append(np.ones(sent.ntok-1)*s)
    
    seqs.append(sent)

left_word = np.concatenate(left_word)
right_word = np.concatenate(right_word)    
which_seq = np.concatenate(which_seq)

cooc = np.array([[np.mean(right_word[left_word==l]==r) for r in range(Data.num_data)] for l in range(Data.num_data)])

# inp_task = tasks.RandomPatterns(Data.num_data, dim_inp)
inp_task = tasks.BinaryLabels(np.eye(Data.num_data))
# out_task = tasks.RandomPatterns(Data.num_data, dim_inp)
out_task = inp_task

inputs = inp_task(left_word).float()  
outputs = out_task(right_word).int()

plt.figure()
plt.imshow(cooc, 'binary')

#%%
N_hid = 96
n_epoch = 10000

# net = students.Feedforward([dim_inp, N_hid, dim_inp], nonlinearity=['ReLU',None], bias=False)
# net = students.Feedforward([dim_inp, N_hid, dim_inp], nonlinearity=[None,None], bias=False)
net = students.Feedforward([dim_inp, N_hid, dim_inp], nonlinearity=['Tanh',None], bias=True)

with torch.no_grad():
    net.network.layer1.weight.requires_grad = False
    which_one = np.random.choice(range(4), 100)
    # net.network.layer1.weight.copy_( x_[which_one,:].T)
    net.network.layer1.weight.copy_(students.BinaryReadout(N_hid, 4).weight.data)/(10*N_hid)


optimizer = optim.Adam(net.parameters(), lr=1e-3)

dl = pt_util.batch_data(inputs, outputs, batch_size=200, shuffle=True)

train_loss = []
kernel_align = []
for epoch in tqdm(range(n_epoch)):
    
    z = net.network[:2](inp_task(np.arange(Data.num_data),noise=0)).detach().numpy()
    Kz = z@z.T
    
    kernel_align.append(np.sum(Kz*cooc)/np.sqrt(np.sum(cooc*cooc)*np.sum(Kz*Kz)))
    
    running_loss = 0
    for ibtc, (inp, out) in enumerate(dl):
        optimizer.zero_grad()
        
        # loss = nn.MSELoss()(net(inp), out)
        # loss = ((net(inp) - out)**2).sum(1).mean()
        loss = nn.CrossEntropyLoss()(net(inp), out.argmax(1))
        running_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
        
        # train_loss.append(loss.item())
    
    train_loss.append(running_loss/(ibtc+1))
    

#%%

N_hid = 48
n_epoch = 10000

nonlin = RayLou()
# nonlin = TanAytch()

ba = 1/np.sqrt(2*N_hid)
W_pos = torch.FloatTensor(N_hid,Data.num_data).uniform_(-ba,ba) + ba
W_neg = torch.FloatTensor(N_hid,Data.num_data).uniform_(-ba,ba) + ba

W_pos.requires_grad = True
W_neg.requires_grad = True

optimizer = optim.Adam([W_pos, W_neg], lr=1e-3)

dl = pt_util.batch_data(inputs, outputs, batch_size=200, shuffle=True)

train_loss = []
kernel_pos = []
kernel_neg = []
neg_w = []
pos_w = []
for epoch in tqdm(range(n_epoch)):
    
    # z = net.network[:2](inp_task(np.arange(Data.num_data),noise=0)).detach().numpy()
    z_pos = nonlin(W_pos@inp_task(np.arange(Data.num_data),noise=0))
    z_neg = nonlin(W_neg@inp_task(np.arange(Data.num_data),noise=0))
    z = torch.cat([z_pos,z_neg]).detach().numpy().T
    Kz = z@z.T
    
    kernel_pos.append((z_pos.T@z_pos).detach())
    kernel_neg.append((z_neg.T@z_neg).detach())
    
    neg_w.append(W_neg.data.numpy()*1)
    pos_w.append(W_pos.data.numpy()*1)
    
    # kernel_align.append(np.sum(Kz*cooc)/np.sqrt(np.sum(cooc*cooc)*np.sum(Kz*Kz)))
    
    running_loss = 0
    for ibtc, (inp, out) in enumerate(dl):
        optimizer.zero_grad()
        
        # pred = W_pos.T@nonlin(W_pos@inp.T) - W_neg.T@nonlin(W_neg@inp.T)
        pred = - W_neg.T@nonlin(W_neg@inp.T)
        
        # loss = nn.MSELoss()(net(inp), out)
        # loss = ((net(inp) - out)**2).sum(1).mean()
        loss = nn.CrossEntropyLoss()(pred.T, out.argmax(1))
        running_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
        
        W_neg.data.clamp_(0)
        W_pos.data.clamp_(0)
        # train_loss.append(loss.item())
    
    train_loss.append(running_loss/(ibtc+1))

    



