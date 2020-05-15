"""
A lot of the methods and functions here were directly lifted from Nickel and 
Kiela's code, on https://github.com/facebookresearch/poincare-embeddings 
because of some very pernicious numerical troubles. If there is any code 
involving in-place pytorch operations (i.e. tensor.func_()), or the function
'clamp', assume it was written by them.
"""
SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/'
CODE_DIR = '/home/matteo/Documents/github/'

import sys
sys.path.append(CODE_DIR+'repler/src/')

from tqdm import tqdm, trange
from time import sleep

import numpy as np
import pandas
import torch 
import torch.nn as nn

import scipy.special as spc
import scipy.linalg as la
import scipy.sparse as sprs
import matplotlib.pyplot as plt

from torch.autograd import Function

from hyperbole.hyperbolic_utils import Hyperboloid
from hyperbole.dataset_utils import SparseGraphDataset # my code

from nltk.corpus import wordnet as wn # natural language toolkit


#%% helper functions

def load_edge_list(path, symmetrize=False):
    df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights

def torch_where(cond, x1, x2):
    cond = cond.float()
    return (cond * x1) + ((1-cond) * x2)    

#%%
dim = 2 # dimension of poincare embedding

n_neg = 10

bsz = 64
nepoch = 1000
eta = 0.3

burnin = 20
c_bi = 10

idx, obj, weights = load_edge_list('/home/matteo/Documents/github/repler/src/hyperbole/mammal_closure.csv')
# idx, obj, weights = load_edge_list('/home/matteo/Documents/github/repler/src/hyperbole/noun_closure.csv')

D = SparseGraphDataset(idx, weights, obj, bsz, n_neg=n_neg)
U = Hyperboloid(len(obj), dim+1, init_range=1e-3)

train_loss = np.zeros(nepoch)
for epoch in range(nepoch):
    
    lr = eta
    if epoch <= burnin:
        lr /= c_bi
        
    running_loss = 0
    with tqdm(D, total=D.num_batches, desc='Epoch %d'%epoch, postfix=[dict(loss_=0)]) as looper:
        for i, (n,t) in enumerate(looper):
            
            U.zero_grad()
            
            e = U(n)
        
            u_jk = e.narrow(-2, 1, e.size(-2)-1) # the neighbour set around u_i
            u_i = e.narrow(-2,0,1).expand_as(u_jk) # the point in question
            
            dists = U.dist(u_i, u_jk)
            loss = nn.CrossEntropyLoss()(-dists, t)
            
            loss.backward()
            
            dL = U.weight.grad.data # euclidean gradient
            assert torch.all(dL==dL), "Contains NaN"
            
            gradL = U.rgrad(U.weight.data, dL) # project onto tangent space
            gradL = U.proj(U.weight.data, gradL)
            
            u_ = U.expmap(U.weight, -lr*gradL) # exponential map
            assert torch.all(u_==u_), "Contains NaN"
            u_ = U.normalize(u_)
            U.weight.data.copy_(u_) # update
            
            running_loss += loss.item()
            
            looper.postfix[0]['loss_'] = np.round(running_loss/(i+1), 3)
            looper.update()
            
    # print('Epoch %d, loss=%.3f'%(epoch, running_loss/(i+1)))    
    train_loss[epoch] = running_loss/(i+1)

fname = '/home/matteo/Documents/uni/columbia/bleilearning/results/hyperbole/noun_reproduction.pt'
torch.save(U.state_dict(), open(fname, 'wb'))

#%%
num_obj = 600
num_label = 6

must_include = ['mammal.n.01', 'carnivore.n.01', 'ungulate.n.01',\
                'primate.n.02', 'aquatic_mammal.n.01','rodent.n.01', 'pug.n.01']

leftovers = [i for i in obj if (i not in must_include)]
random_entries = list(np.random.choice(leftovers, num_obj, False))
drawn_obj = must_include + random_entries
labelled_obj = must_include + list(np.random.choice(drawn_obj, num_label, False))

# get hypernymy graph

these_ind = []
hypernyms = []
for name in drawn_obj:
    these_ind.append(obj.index(name))
    
    this = wn.synset(name)
    
    closure = this.closure(lambda s:s.hypernyms())
    parents = [(n, obj.index(n.name())) for n in closure if n.name() in obj]
    
    hypernyms.append(parents)

poinc = U.weight.data[:,1:]/(1+U.weight.data[:,:1])

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
edge_color = 'm'

plt.figure()

already_done = []
drawn_edge = []
for ind,(n, i) in enumerate(zip(drawn_obj, these_ind)):
    
    if (n not in already_done) and (n in labelled_obj):
        plt.text(poinc[i,0],poinc[i,1], n, verticalalignment='bottom', bbox=props)
        already_done.append(n)
    
    k = i # keep track of previous index
    for (m, j) in hypernyms[ind]:
        
        if ([k,j] not in drawn_edge):
            plt.plot(poinc[[k,j],0], poinc[[k,j],1], color=edge_color, alpha=0.7,\
                     linewidth=0.5)
            drawn_edge.append([k,j])
        
        if (m.name() not in already_done) and (m.name() in labelled_obj):
            plt.text(poinc[j,0],poinc[j,1], m.name(), verticalalignment='bottom', bbox=props)
            already_done.append(m.name())
        
        k = j
        
plt.scatter(poinc[:,0].detach(), poinc[:,1].detach(), s=1, c='k')

plt.axis('equal')
plt.axis('off')
