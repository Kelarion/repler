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
import bae_util
import bae_models

import students 
import super_experiments as sxp
import experiments as exp
import server_utils as su
import grammars as gram
import plotting as tpl

# import umap
from cycler import cycler
import linecache

# import nnsight

import transformer_lens as tfl
import transformer_lens.utils as tfl_utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

device = tfl_utils.get_device()
# torch.set_grad_enabled(False)

#%%

# from nnsight import CONFIG

# CONFIG.API.APIKEY = input("Enter your API key: ")
# # clear_output()

#%%

# from nnsight import LanguageModel

# # don't worry, this won't load locally!
# llm = LanguageModel("meta-llama/Meta-Llama-3.2-70B", device_map="auto")

# print(llm)

# #%%

# with llm.trace("The Eiffel Tower is in the city of", remote=True):

#     # user-defined code to access internal model components
#     hidden_states = llm.model.layers[-1].output[0].save()
#     output = llm.output.save()


#%% 

# model = HookedTransformer.from_pretrained('meta-llama/Llama-3.2-1B', device='cpu')
model = HookedTransformer.from_pretrained('gpt2-medium', device='cpu')

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

gamma = model.W_U.T.detach()
w, d = gamma.shape
gam_ = gamma.mean(0)
Covg = (gamma-gam_).T@(gamma-gam_) / w
# Covg = (gamma.T @ gamma - len(gamma)*torch.outer(gam_,gam_)) / W
l, V = torch.linalg.eigh(Covg)
inv_sqrt_Cov = V @ torch.diag(1/torch.sqrt(l + 1e-6)) @ V.T
whgamma = gamma @ inv_sqrt_Cov # the "causal" representation

#%%

def getword(wnword):
    i = wnword.index('.n')
    return wnword[:i]

def randleaf(root):
    childs = root.hyponyms()
    if len(childs)>0:
        return randleaf(np.random.choice(childs))
    else:
        return root

def path2root(path, word):
    """Recursion to fill `path` with the ancestors of `word`"""

    path.append(getword(word))
    ancs = wn.synset(word).hypernyms()
    if len(ancs)>0:

        parent = np.random.choice(ancs)
        path = path2root(path, parent.name())

    return path

#%%

maxwords = 10000
# root = 'abstraction.n.06'
root = 'organism.n.01'
# root = 'person.n.01'
# root = 'entity.n.01'

words = [getword(root)]
toks = [model.to_single_token(' ' + words[0])]
isleaf = [False]
leaves = [h.name() for h in wn.synset(root).hyponyms()]

it = 1
while (len(words) < maxwords) and (len(leaves) > 0):

    print('%d words and %d leaves at level %d'%(len(words), len(leaves), it))

    newleaves = []
    for leaf in leaves:
        nextleaves = [h.name() for h in wn.synset(leaf).hyponyms()]
        newleaves += nextleaves

        thisword = getword(leaf)
        thistok = model.to_str_tokens(' ' + thisword, prepend_bos=False)
        if (len(thistok) == 1) and (thisword not in words):
            words.append(thisword)
            toks.append(model.to_single_token(' ' + thisword))
            isleaf.append(len(nextleaves) == 0)

    leaves = newleaves
    it += 1

isleaf = np.array(isleaf)
vecs = whgamma[np.array(toks)].numpy()

#%%

neal = bae_util.Neal(decay_rate=0.9, period=2, initial=10)

# mod = bae_models.BinaryAutoencoder(600, vecs.shape[1], 
#                                    tree_reg=1e-1, 
#                                    sparse_reg=1e-3,
#                                    weight_reg=1e-2)
# # dl = pt_util.batch_data(vecs.cuda(), batch_size=512)
# # mod.cuda()
# dl = pt_util.batch_data(vecs, batch_size=512)

# en = neal.fit(mod, dl, T_min=1e-6)

mod = bae_models.KernelBMF(vecs.shape[1], tree_reg=0.2, scale_lr=1)
# mod = bae_models.BiPCA(vecs.shape[1], sparse_reg=0, tree_reg=1)
en = neal.fit(mod, vecs, T_min=1e-5)

S = mod.S
# W = mod.W*mod.scl
pi = np.ones(mod.S.shape[1])

# mod = bae_models.KernelBMF(600, tree_reg=1e-1, scale_lr=0.9)
# en = neal.fit(mod, vecs.numpy(), pvar=0.95)

# mod3 = bae_models.BernVAE(600, vecs.shape[1], weight_reg=1e-2)
# dl = pt_util.batch_data(vecs.cuda(), batch_size=512)
# mod3.cuda()
# # dl = pt_util.batch_data(vecs, batch_size=512)
# en = neal.fit(mod3, dl, T_min=1e-6)

#%%
# ls = [mod.grad_step(dl) for _ in range(100)] # train a bit at zero temperature
S = mod.hidden(vecs).detach().numpy()
W = mod.p.weight.data.cpu().numpy()
pi = np.diag(W.T@W)

#%%
# kqmw = [words.index('king'),
#         words.index('queen'),
#         words.index('man'),
#         words.index('woman'),
#         words.index('uncle'),
#         words.index('aunt')]
kqmw = [words.index('king'),
        words.index('queen'),
        words.index('man'),
        words.index('woman')]


#%%

S_cntr = np.mod(S+S[kqmw[0]],2)
Z,grp = np.unique(S_cntr[kqmw], axis=1, return_inverse=True)

piw = pi/(util.group_sum(pi,grp)[grp])
P = util.group_sum(S_cntr@np.diag(piw), grp, axis=1)

#%%

cl = 2 # the "class" dimension
gn = 3 # the "gender" dimension

plt.plot([0,0],[0,1], 'k--')
plt.plot([0,1],[1,1], 'k--')
plt.plot([1,1],[1,0], 'k--')
plt.plot([1,0],[0,0], 'k--')

plt.scatter(P[kqmw,cl], P[kqmw,gn], c=[0,1,2,3], s=100, marker='*', cmap='Set2', zorder=10)

tpl.hovertext(P[:,cl], P[:,gn], labels=words)

tpl.square_axis()
plt.axis(False)

#%%



#%%

these_words = ['monarch', 
               'male', 
               'female', 
               'person', 
               'guy', 
               'pope', 
               'princess', 
               'girl',
               'organism']
deez = [words.index(n) for n in these_words]

plt.plot([0,0],[0,1], 'k--')
plt.plot([0,1],[1,1], 'k--')
plt.plot([1,1],[1,0], 'k--')
plt.plot([1,0],[0,0], 'k--')

plt.scatter(P[kqmw,cl], P[kqmw,gn], c=[0,1,2,3], s=100, marker='*', cmap='Set2', zorder=10)
plt.scatter(P[:,cl], P[:,gn], c=(0.7,0.7,0.7), alpha=0.5)

tpl.scatterlabel(P[deez,cl], P[deez,gn], these_words)


tpl.square_axis()
plt.axis(False)

#%%

thiscl = 533
thisgn = 60

XW = (mod.data - mod.b)@mod.W/mod.scl

plt.scatter(XW[:,thiscl], XW[:,thisgn], c=(0.7,0.7,0.7), alpha=0.5)
tpl.scatterlabel(XW[kqmw,thiscl], XW[kqmw,thisgn], ['king', 'queen', 'man', 'woman'])

tpl.hovertext(XW[:,thiscl], XW[:,thisgn], s=1, labels=words)

