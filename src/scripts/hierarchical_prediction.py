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

#%%

def parallelism_score(Kz, Ky, mask, eps=1e-12):
    
    # dz = (torch.sum(Kz*Ky*(1-mask),0, keepdims=True) + torch.sum(Kz*Ky*(1-mask),1, keepdims=True))
    # dz = torch.sqrt(torch.abs(torch.sum(dz*(1-mask-torch.eye(len(Kz))),0)))
    
    if torch.sum(mask[Ky != 0])>0:
        # dz = torch.sum(dot2dist(Kz)*(1-mask-torch.eye(len(Kz))),0)
        # norm = dz[:,None]*dz[None,:]
        dist = torch.diag(Kz)[:,None] + torch.diag(Kz)[None,:] - 2*Kz
        dz = torch.sqrt(torch.abs(dist[(1-mask-torch.eye(len(Kz)))>0]))
        norm = (dz[:,None]*dz[None,:]).flatten()
        
        # numer = torch.sum((Kz*Ky*mask)[Ky != 0]/norm[Ky != 0])
        numer = torch.sum((Kz*Ky*mask)[Ky != 0]/norm)
        denom = (torch.sum(torch.tril(mask)[Ky != 0])/2) #+ eps
        return numer/denom
    else:
        return 0
    
    # return torch.sum((Kz*Ky*mask)/norm)/(torch.sum(torch.tril(mask))/2)

def dot2dist(K):
    return torch.sqrt(torch.abs(torch.diag(K)[:,None] + torch.diag(K)[None,:] - 2*K))


# %% Pick data format
K = 2
respect = False
# respect = True

# layers = [K**0,K**1,K**2]
layers = [1, 2,2]
# layers = [1,1,1]

Data = gram.HierarchicalData(layers, fan_out=K, respect_hierarchy=respect, graph_rule='minimal')

ll = Data.labels(Data.terminals)
labs = np.where(np.isnan(ll), np.nanmax(ll)+1, ll)

Ky_all = np.sign((ll[:,:,None]-0.5)*(ll[:,None,:]-0.5))
Ky_all = torch.tensor(np.where(np.isnan(Ky_all), 0, Ky_all))

reps = Data.represent_labels(Data.terminals)
Ky = util.dot_product(reps,reps)

plt.figure()
plt.subplot(131)
pos = graphviz_layout(Data.value_tree, prog="twopi")
nx.draw(Data.value_tree, pos, node_color=np.array(Data.value_tree.nodes(data='var'))[:,1], cmap='nipy_spectral')
dicplt.square_axis()
plt.subplot(132)
plt.imshow(ll, 'bwr')
plt.subplot(133)
plt.imshow(util.dot_product(reps,reps), 'binary')

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

leaves = np.isin(sorted(Data.similarity_graph),Data.terminals)
observed = np.isin(sorted(Data.similarity_graph), Data.value_tree.nodes)

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
dim_inp = 50
max_children = 2
child_prob = 0.7

num_sent = 5000

seqs = []
left_word = []
right_word = []
which_seq = []
for s in range(num_sent):
    sent = Data.random_sequence(child_prob, replace=max_children>K, max_child=max_children)
    
    which_word = np.array([np.isin(Data.terminals, complex(sent.node_tags[x])) for x in sent.words])
    
    left_word.append(which_word[:-1,:].argmax(1))
    right_word.append(which_word[1:,:].argmax(1))
    which_seq.append(np.ones(sent.ntok-1)*s)
    
    seqs.append(sent)

left_word = np.concatenate(left_word)
right_word = np.concatenate(right_word)    
which_seq = np.concatenate(which_seq)

cooc = np.array([[np.mean(right_word[left_word==l]==r) for r in range(Data.num_data)] for l in range(Data.num_data)])


inp_task = tasks.RandomPatterns(Data.num_data, dim_inp)
out_task = tasks.RandomPatterns(Data.num_data, dim_inp)

inputs = inp_task(left_word)    
outputs = out_task(right_word)

plt.figure()
plt.imshow(cooc, 'binary')

#%%
N_hid = 100
n_epoch = 10000

net = students.Feedforward([dim_inp, N_hid, dim_inp], nonlinearity=['Tanh',None])

optimizer = optim.Adam(net.parameters(), lr=1e-3)

dl = pt_util.batch_data(inputs, outputs, batch_size=200, shuffle=True)

train_loss = []
kernel_align = []
for epoch in tqdm(range(n_epoch)):
    z=net.network[:2](inp_task(np.arange(8),noise=0)).detach().numpy()
    Kz = z@z.T
    
    kernel_align.append(np.sum(Kz*cooc)/np.sqrt(np.sum(cooc*cooc)*np.sum(Kz*Kz)))
    
    running_loss = 0
    for ibtc, (inp, out) in enumerate(dl):
        optimizer.zero_grad()
        
        loss = nn.MSELoss()(net(inp), out)
        running_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
    
    train_loss.append(running_loss/(ibtc+1))
    



    



