"""
An attempt to make a parametric encoder onto hyperbolic space, using the 
psuedo-polar coordinates of Gulchere et al. (2019). 

Uses a lot of code adapted from Nickel and Kiela (2018), and their pytorch 
implementation of a non-parametric hyperboloid embedding.
"""

# SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/'
CODE_DIR = 'C:/Users/mmall/Documents/github/'

import sys
from tqdm import tqdm, trange
from time import sleep

import numpy as np
import pandas
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import scipy.special as spc
import scipy.linalg as la
import scipy.sparse as sprs
import matplotlib.pyplot as plt

from torch.autograd import Function
from collections import OrderedDict

from nltk.corpus import wordnet as wn # natural language toolkit

# my code
sys.path.append(CODE_DIR+'repler/src/')
from hyperbole.hyperbolic_utils import Hyperboloid, PseudoPolar, TangentSpace, CartesianHyperboloid, GeodesicCoordinates, RelaxedGeodesics
from hyperbole.dataset_utils import DenseDataset, SparseGraphDataset
from students import Feedforward 
from assistants import Indicator

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
nepoch = 2000
# nepoch = 700
# eta = 1e-2
eta = 0.3
burnin = 50
c_bi = 20

anneal = 500

idx, obj, weights = load_edge_list(CODE_DIR + '/repler/src/hyperbole/mammal_closure.csv')
# idx, obj, weights = load_edge_list('/home/matteo/Documents/github/repler/src/hyperbole/noun_closure.csv')

D = SparseGraphDataset(idx, weights, obj, bsz, n_neg=n_neg)

enc = Feedforward([len(obj), dim], [None],
                  encoder=Indicator(len(obj), len(obj)), bias=False)
# hype = TangentSpace(enc)
# hype = CartesianHyperboloid(enc)
# hype = PseudoPolar(enc)
# hype = GeodesicCoordinates(enc, max_norm=1e2, norm_type=2)
hype = RelaxedGeodesics(enc, max_norm=1e4, norm_type=np.inf)
hype.init_weights(torch.tensor(np.arange(1180)).long())

# optimizer = optim.Adam(hype.parameters(), lr=eta)
optimizer = optim.SGD(hype.parameters(), lr=eta)

prev_weights = hype.state_dict()

train_loss = np.zeros(nepoch)
# viz = np.zeros((1180,2,nepoch))
# grads = np.zeros(nepoch)
for epoch in range(nepoch):
    if epoch<=burnin:
        optimizer.param_groups[0]['lr'] = eta/c_bi
    else:
        optimizer.param_groups[0]['lr'] = eta
    
    # if epoch<=anneal:
    #     D.n_neg = int(np.ceil((n_neg-5)*epoch/anneal + 5))
    # else:
    #     D.n_neg = n_neg
    
    # wa = hype(torch.arange(0,len(obj)).long())
    # poinc = (wa[:,1:]/(1+wa[:,:1])).detach().numpy()
    # viz[:,:,epoch] = poinc
    running_loss = 0
    with tqdm(D, total=D.num_batches, desc='Epoch %d'%epoch, postfix=[dict(loss_=0)]) as looper:
        for i, (n,t) in enumerate(looper):
            optimizer.zero_grad()
            
            emb = hype(n)
            
            u_jk = emb.narrow(-2, 1, emb.size(-2)-1) # the neighbour set around u_i
            u_i = emb.narrow(-2,0,1).expand_as(u_jk) # the point in question
            
            d = hype.dist(u_i, u_jk)
            # logprob = -d
            # logprob = -hype.distances(n)
            # logprob = hype.dec(-hype.distances(n).unsqueeze(2)).squeeze(2)
            if not torch.all(d==d): # Contains NaN
                hype.load_state_dict(prev_weights)
                raise ValueError('Oops, NaNs')
            else:
                prev_weights = hype.state_dict()
                
            loss = nn.CrossEntropyLoss()(-d, t)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()#/D.n_neg
            
            looper.postfix[0]['loss_'] = np.round(running_loss/(i+1), 3)
            looper.update()
            
    # print('Epoch %d, loss=%.3f'%(epoch, running_loss/(i+1)))    
    train_loss[epoch] = running_loss/(i+1)

fname = 'C:/Users/mmall/Documents/uni/columbia/bleilearning/results/hyperbole/mammal_parametric.pt'
torch.save(hype.state_dict(), open(fname, 'wb'))

    #%%
num_obj = 600
num_label = 6

must_include = ['mammal.n.01', 'carnivore.n.01', 'ungulate.n.01',
                'primate.n.02', 'aquatic_mammal.n.01','rodent.n.01', 'pug.n.01',
                'homo_sapiens_sapiens.n.01']

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

wa = hype(torch.arange(0,len(obj)).long())
poinc = (wa[:,1:]/(1+wa[:,:1])).detach().numpy()

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
        
plt.scatter(poinc[:,0], poinc[:,1], s=1, c='k')

plt.axis('equal')
plt.axis('off')
