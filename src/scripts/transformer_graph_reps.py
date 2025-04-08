CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import socket
import os
import sys
import pickle as pkl
import subprocess
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
import itertools as itt
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from sklearn.manifold import MDS

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

# import gensim as gs
# from gensim import models as gmod
# from gensim import downloader as gdl

from nltk.corpus import wordnet as wn # natural language toolkit

sys.path.append(CODE_DIR)
import util
import pt_util
import df_util
import students 
import super_experiments as sxp
import experiments as exp
import server_utils as su
import grammars as gram
import plotting as tpl

# import umap
from cycler import cycler
import linecache

import nnsight

# import transformer_lens as tfl
# import transformer_lens.utils as tfl_utils
# from transformer_lens.hook_points import (
#     HookPoint,
# )  # Hooking utilities
# from transformer_lens import HookedTransformer, FactoredMatrix

# device = tfl_utils.get_device()
torch.set_grad_enabled(False)

#%%

@dataclass
class ICLR(sxp.Task):
    """
    Random walks on a graph
    """
    
    graph: nx.graph.Graph
    samps: int
    context: int = 1400 
    
    def sample(self):
        
        X = [] # list of contexts
        
        for i in range(self.samps):
            
            x = [np.random.choice(list(self.graph.nodes))]
            for t in range(self.context):
                E = list(self.graph.edges(x[t]))
                e = np.random.choice(range(len(E)))
                x.append(list(set(E[e]) - set([x[t]]))[0])
            
            X.append(x)
        
        return {'X': np.array(X)}
    
    
@dataclass
class ICLAR(sxp.Task):
    """
    Random 'actions' on a graph
    
    expects an integer 'action' attribute for each edge in the graph
    """
    
    graph: nx.graph.Graph
    samps: int
    heldout: int = 1 # number of edges held out during 'training'
    context: int = 1400 
    
    def sample(self):
        
        X = [] # list of (x1, a1, x2, a2, ..., xn) sequences
        
        for i in range(self.samps):
            
            train_edges, test_edges = self.remove_edges(self.heldout)
            
            ## final node is incident to a held out edge
            en = test_edges[np.random.choice(range(len(test_edges)))]
            an = self.graph[en[0]][en[1]]['action']
            
            x = [en[0], an, en[1]]
            for t in range(self.context):
                Ex = list(train_edges(x[t]))
                e = Ex[np.random.choice(range(len(Ex)))]
                x.append(self.graph[e[0]][e[1]]['action'])
                x.append(list(set(e) - set([x[t]]))[0])
            
            X.append(np.flip(x))
        
        return {'X': np.array(X)}
    
    def remove_edges(self, n):
        """
        Remove as many edges as possible while remaining connected
        """
        
        E_test = set()
        
        G = nx.Graph()
        G.add_edges_from(self.graph.edges)
        
        while len(E_test) < n:
            
            allowed = list(set(G.edges) - set(nx.bridges(G)))
            if len(allowed) > 0:
                e = np.random.choice(range(len(allowed)))
                G.remove_edge(allowed[e][0], allowed[e][1])
                E_test.add(allowed[e])
            else:
                break
        
        return G.edges, list(E_test)
        

@dataclass
class Bernardi(sxp.Task):

    num_trials: int
    trials_per_context: int
    samps: int

    def sample(self):

        num_ctx = self.num_trials//self.trials_per_context

        A1 = np.array([1,0,1,0]) # action
        R1 = np.array([1,1,0,0]) # reward

        A = np.concatenate([A1, np.roll(A1,1)])
        R = np.concatenate([R1, np.roll(R1,1)])

        X = []
        Y = []
        for i in range(self.samps):

            s = np.random.choice(range(4), self.num_trials)
            c = np.tile(np.repeat([0,1], self.trials_per_context), num_ctx//2)

            a = A[s+4*c] + 8
            r = R[s+4*c] + 10

            stim = np.array([s,a,r]).T.flatten()
            X.append(stim)
            Y.append([c,a,r])
        
        return {'X':X, 'Y':Y}

#%%

from nnsight import CONFIG

CONFIG.API.APIKEY = input("Enter your API key: ")
# clear_output()

#%%

from nnsight import LanguageModel

# don't worry, this won't load locally!
llm = LanguageModel("meta-llama/Meta-Llama-3.1-70B", device_map="auto")

print(llm)

#%%

with llm.trace("The Eiffel Tower is in the city of", remote=True):

    # user-defined code to access internal model components
    hidden_states = llm.model.layers[-1].output[0].save()
    output = llm.output.save()


#%% 

# # model = HookedTransformer.from_pretrained('meta-llama/Llama-3.2-1B', device=device)
# model = HookedTransformer.from_pretrained('gpt2-medium', device=device)

# vocab = []
# tokens = []
# lines = linecache.getlines(SAVE_DIR+'top-1000-nouns.txt')
# for line in lines:
#     word = line.split('\n')[0]
#     toks = model.to_str_tokens(word, prepend_bos=False)
#     if len(toks) == 1:
#         vocab.append(toks[0])
#         tokens.append(model.to_single_token(toks[0]))

# tokens = np.array(tokens)

#%%

num_ctx = 90
ctx_len = 500
prepend_bos = False

# B = df_util.cyclecats(4)
B = df_util.gridcats(4)
A = df_util.adj(B)
G = nx.Graph(df_util.adj(B))
T = len(G.nodes)

task = ICLR(G, samps=num_ctx, context=ctx_len)
data = task.sample()

task_toks = np.random.permutation(tokens)

if prepend_bos:
    bos = model.to_tokens('')[0][0].cpu().item()
    inp = torch.LongTensor(np.hstack([np.ones((num_ctx,1))*bos,task_toks[data['X']]]))
else:
    inp = torch.LongTensor(task_toks[data['X']])

#%%

L = []
Z = []
for i in tqdm(range(num_ctx)):
    logits, cache = model.run_with_cache(inp[i], remove_batch_dim=True)
    
    Z.append(cache.accumulated_resid(pos_slice=-1).cpu().numpy())
    L.append(logits.cpu().detach().numpy()[0][:,task_toks[:T]])

L = np.array(L)
Z = np.array(Z)

#%%

probs = np.exp(L)/np.exp(L).sum(-1, keepdims=True)
n,t,x = np.where(A[data['X']]) # adjacent tokens for each time and sequence
acc = util.group_sum(probs[n,t,x], t)/num_ctx
plt.plot(acc)

#%%

for layer in range(1,13):

    Zfin = Z[:,layer]
    Zrep = util.group_mean(Zfin, data['X'][:,-1], axis=0)
    
    U,s,V = la.svd(Zrep - Zrep.mean(0), full_matrices=False)
    
    PC = U[:,:2]@np.diag(s[:2])
    
    plt.subplot(3, 4, layer)
    plt.scatter(PC[:,0], PC[:,1])
    
    