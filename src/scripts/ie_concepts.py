CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/iecor/cldf/'
 
import os, sys, re
import pickle
from time import time
sys.path.append(CODE_DIR)

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
import pandas as pd

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from sklearn.manifold import MDS

import networkx as nx
# import pydot
from networkx.drawing.nx_pydot import graphviz_layout

import gensim as gs
from gensim import models as gmod
from gensim import downloader as gdl

from nltk.corpus import wordnet as wn # natural language toolkit

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
import dichotomies as dics

import distance_factorization as df
import df_util
import df_models as mods
import bae
import bae_models
import bae_util

#%%

form = pd.read_csv(SAVE_DIR+'forms.csv')
cogs = pd.read_csv(SAVE_DIR+'cognates.csv')
cset = pd.read_csv(SAVE_DIR+'cognatesets.csv')
lang = pd.read_csv(SAVE_DIR+'languages.csv')

# cg = cogs['Form_ID'].to_numpy()
# cid = cset.ID.to_numpy()
# lid = lang.ID.to_numpy()
# cid = cogs.ID.to_numpy()
clad, grp = np.unique(lang.clade_name, return_inverse=True)
cunq, cid = np.unique(cogs.Cognateset_ID, return_inverse=True)
lunq, lid = np.unique(form.Language_ID, return_inverse=True)
fid = form.ID.to_numpy()

#%%

cogmat = np.zeros((len(lunq), len(cunq)))
# for f,l in tqdm(zip(frms, lids[ids])):
for f,l in tqdm(zip(fid, lid)):
    
    c = cid[cogs['Form_ID'].tolist().index(f)]
    cogmat[l,c] += 1

#%%

mod = bae_models.KernelBMF(300, tree_reg=1, scale_lr=0.9)

neal = bae_util.Neal(decay_rate=0.95, period=2, initial=5)

en = neal.fit(mod, cogmat)

# neal = bae_util.Neal(decay_rate=0.95, period=2)

# mod = bae_models.BinaryAutoencoder(600, vecs.shape[1], 
#                                    tree_reg=1e-1, 
#                                    sparse_reg=1e-3,
#                                    weight_reg=1e-2)
# # dl = pt_util.batch_data(vecs.cuda(), batch_size=512)
# # mod.cuda()
# dl = pt_util.batch_data(vecs, batch_size=512)

# en = neal.fit(mod, dl, T_min=1e-6)

# cv = []
# neal = bae_util.Neal(decay_rate=0.98, initial=1, period=2)
# for k in tqdm(range(2, 29)):
#     mod = bae_models.GeBAE(k, tree_reg=0.1, weight_reg=1e-1)
#     ba = neal.cv_fit(mod, zZ, draws=10, folds=10)
#     cv.append(np.mean(ba[0]))
#     eses.append(ba[1])

#%%

E,H = df_util.allpaths(mod.S)

G = nx.Graph()
G.add_edges_from(E)

# G = nx.maximum_spanning_tree(G)

pos = graphviz_layout(G, prog='twopi')

#%%

cmap = cm.tab10

fig,ax = plt.subplots()
plt_these = np.isin(G.nodes,range(160))
nx.draw(G, pos=pos, node_size=10*plt_these, 
        node_color=cmap(grp[np.where(plt_these, G.nodes, 0)]))

p_cntr = np.mean([p for p in pos.values()],0)
X = np.array([pos[i] for i in range(160)])

names = lang['Glottolog_Name'].tolist()

dicplt.hovertext(X[:,0], X[:,1], labels=names, c=grp)

dicplt.square_axis()

#%%

these_langs = ['Latin',
               'Old Breton',
               'Tregorrois',
               'Italian',
               'Gothic',
               'Elfdalian',
               'Danish',
               'Dutch',
               'Central Alemannic',
               'Irish',
               'Bakhtiari',
               'Southern Kurdish',
               'Sogdian',
               'Parthian',
               'Albanian',
               'Achaean Greek',
               'Modern Greek',
               'Oscan',
               'Urdu',
               # 'Hindi',
               'Kashmiri',
               'Bengali',
               'Vedic Sanskrit',
               'Latvian',
               'Macedonian',
               'Polish',
               'Slovenian',
               'Slovak',
               'Ukrainian',
               'Walloon',
               'Barbaricino',
               'Romanian',
               'Catalan']

# these_langs = ['Latin',
#                # 'Brazilian Portuguese',
#                # 'Portuguese',
#                'Hindi',
#                'Urdu',
#                'Italian',
#                'English',
#                'Sinhala',
#                'Hittite',
#                'Gothic',
#                'Kamviri',
#                'Transalpine Gaulish',
#                'Polish',
#                'Bengali',
#                'Eastern Pahari',
#                'Scottish Gaelic',
#                'Northern Welsh',
#                'Barbaricino',
#                'Slovenian',
#                # 'Slovak',
#                # 'Umbrian',
#                'Western Farsi',
#                'Modern Greek',
#                'Early Irish',
#                'Icelandic',
#                'Catalan',
#                'Southern Kurdish',
#                'Kumzari',
#                'Elfdalian',
#                'Danish',
#                'Western Flemish',
#                # 'German',
#                'Takestani',
#                # 'Hawraman-I Taxt',
#                'Tokharian A',
#                'Macedonian',
#                'Bakhtiari',
#                'Romanian',
#                'Central Alemannic',
#                'Francoprovencalic',
#                'Oscan',
#                'Latvian',
#                # 'Dalmatian',
#                'Khwarezmian',
#                'Ukrainian',
#                'Eastern Armenian']

plt.figure()
plt_these = np.isin(G.nodes,range(160))
nx.draw(G, pos=pos, node_size=10*plt_these, 
        node_color=cmap(grp[np.where(plt_these, G.nodes, 0)]))

p_cntr = np.mean([p for p in pos.values()],0)

for this_lang in these_langs:
    this = lang['Glottolog_Name'].tolist().index(this_lang)
    
    if pos[this][0] > p_cntr[0]:
        dx = 5
        ha = 'left'
    elif pos[this][0] < p_cntr[0]:
        dx = -5
        ha = 'right'
    else:
        dx = 1e-2
        ha = 'left'
    if pos[this][1] > p_cntr[1]:
        dy = 5
        va = 'bottom'
        # ha = 'right'
    if pos[this][1] < p_cntr[1]:
        dy = -5
        va = 'top'
        # ha = 'right'
    else:
        dy = 5e-2
        va = 'bottom'
    plt.scatter(pos[this][0], pos[this][1], color=cmap(grp[this]), zorder=10)
    plt.text(pos[this][0]+dx,pos[this][1]+dy, this_lang,
             horizontalalignment = ha,
             verticalalignment = va,
             color=cmap(grp[this]),
             bbox={'facecolor':'white', 
                   'edgecolor': cmap(grp[this]), 
                   'alpha': 0.8,
                   'boxstyle': 'round'})
    
dicplt.square_axis()