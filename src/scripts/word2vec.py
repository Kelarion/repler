CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
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

w2v = gdl.load('word2vec-google-news-300')

#%% net2word2vec

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
# root = 'group.n.01'
root = 'person.n.01'

words = [getword(root)]
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
        if (thisword in w2v.index_to_key) and (thisword not in words):
            words.append(thisword)
            isleaf.append(len(nextleaves) == 0)
            
    leaves = newleaves
    it += 1

isleaf = np.array(isleaf)
vecs = w2v[words]

#%%

kqmw = [words.index('king'), 
        words.index('queen'), 
        words.index('man'), 
        words.index('woman')]

#%%
# steps = 200

# # bae = util.BAE(vecs, 5000)
# baer = bae.BAE(vecs, 1000, pvar=0.95, alpha=1, beta=5, penalty=1e-1, max_ctx=None)

# # foo = bae.fit(500, verbose=True)

# # baer.init_optimizer(decay_rate=0.8, period=1, initial=1)
# baer.init_optimizer(decay_rate=0.95, period=1, initial=1)
# # en = []
# cka = []
# nbs = []
# for t in tqdm(range(steps)):
#     #r = np.sum(pvar< (0.8 + 0.2*(t//10)/10))
#     baer.proj()
#     baer.scl = baer.scaleS()
#     baer.grad_step()
#     # en.append(baer.energy())
#     # cka.append(util.centered_kernel_alignment(vecs@vecs.T, (baer.S@baer.S.T).todense()))
#     nbs.append(util.nbs(vecs, baer.S))
    
# S = baer.S.todense()

# # S = np.sign(bae.current())
# # S = np.unique(S*S[[0]], axis=1)
# # S = S[:,np.abs(S.sum(0))<len(vecs)]

# S = []
# for _ in range(20):
#     mod2 = bae_models.BiPCA(300, center=False, sparse_reg=1e-2)
#     # mod = bae_models.KernelBMF(300, penalty=1e-2, scale_lr=1)
    
#     neal = bae_util.Neal(decay_rate=0.95, period=5)
#     en = neal.fit(mod2, vecs)
    
#     S.append(np.unique(np.mod(mod2.S+(mod2.S.mean(0)>0.5),2), axis=1))
#     # S = np.unique(np.mod(mod2.S+(mod2.S.mean(0)>0.5),2), axis=1)
# S, pi = df_util.mindistX(vecs, S, beta=1e-6, nonzero=False)

mod = bae_models.BinaryAutoencoder(600, 300, 
                                   tree_reg=0, 
                                   sparse_reg=1e-2,
                                   weight_reg=1e-2)
dl = pt_util.batch_data(torch.FloatTensor(vecs), batch_size=512)
neal = bae_util.Neal(decay_rate=0.95, period=5)

en = neal.fit(mod, dl, T_min=1e-6)
ls = [mod.grad_step(dl) for _ in range(100)] # train a bit at zero temperature

S = mod.hidden(torch.FloatTensor(vecs)).detach().numpy()
W = mod.p.weight.data.numpy()
pi = np.diag(W.T@W)

#%% Local Structure

S_cntr = np.mod(S+S[kqmw[0]],2)
Z,grp = np.unique(S_cntr[kqmw], axis=1, return_inverse=True)

piw = pi/(util.group_sum(pi,grp)[grp])
P = util.group_sum(S_cntr@np.diag(piw), grp, axis=1)

#%%
cl = 3 # the "class" dimension
gn = 5 # the "gender" dimension

pos3 = [words.index('worker')]
pos5 = [words.index('girl'), words.index('princess'), words.index('redhead')]
neg3 = [words.index('courtier'), words.index('vassal')]
neg5 = [words.index('boy'), words.index('dunce'), words.index('prince')]
cntr = [words.index('lord'), words.index('eunuch')]
alleg = np.concatenate([pos3, pos5, neg3, neg5, cntr])

# plt.plot([0,0,1,1,0],[0,1,1,0,0], 'k--')
plt.plot([0,0],[0,1], 'k--')
plt.plot([0,1],[1,1], 'k--')
plt.plot([1,1],[1,0], 'k--')
plt.plot([1,0],[0,0], 'k--')

plt.scatter(P[:,cl], P[:,gn], color=(0.7,0.7,0.7), alpha=0.4, s=5)
plt.scatter(P[kqmw,cl], P[kqmw,gn], c=[0,1,2,3], s=100, marker='*', cmap='Set2', zorder=10)

plt.scatter(P[alleg,cl], P[alleg,gn], c='k')
for this in alleg:
    if P[this,cl] > 0.8:
        dx = -3e-2
        ha = 'right'
    elif P[this,cl] < 0.4:
        dx = -3e-2
        ha = 'right'
    else:
        dx = 1e-2 
        ha = 'left'
    if P[this,1] > 0.8:
        dy = -3e-2
        va = 'top'
        ha = 'right'
    elif P[this,gn] < 0.4:
        dy = -3e-2
        va = 'top'
        # ha = 'right'
    else:
        dy = 3e-2
        va = 'bottom'
    plt.text(P[this,cl]+dx,P[this,gn]+dy,words[this],
             horizontalalignment = ha,
             verticalalignment = va,
             bbox={'facecolor':'white', 
                   'edgecolor': 'black', 
                   'alpha': 0.8,
                   'boxstyle': 'round'})

dicplt.square_axis()
plt.axis(False)

#%% Global structure

Scorr = np.abs((2*Es-1).T@(2*Es-1))

aye,jay = np.triu_indices(Es.shape[1],1)

indep = np.argsort(Scorr[aye,jay]) # independence score

#%%

scores = []
balance = []
plsm = []

# for this in tqdm(np.where(Scorr[aye,jay]==1)[0]):
for this in tqdm(range(len(aye))):
    
    c1 = aye[this]
    c2 = jay[this]
    
    _, grp = np.unique(Es[:,[c1,c2]], axis=0, return_inverse=True)
    
    centroids = util.group_mean(vecs, grp, axis=0)
    
    if len(centroids) < 4:
        plsm.append([0,0])
        continue
    
    # alldist = []
    # for j in range(4):
    #     alldist.append(la.norm(vecs - centroids[np.mod(grp+j, 4)], axis=1))
    # alldist = np.array(alldist)
    
    # score = alldist[0]**2 - np.mean(alldist[1:], axis=0)**2
    
    # scores.append(util.group_sum(score, grp))
    # balance.append(Es.sum(0)[c1])
    
    d01 = (centroids[0]-centroids[1])
    d02 = (centroids[0]-centroids[2])
    d13 = (centroids[1]-centroids[3])
    d23 = (centroids[2]-centroids[3])
    
    p1 = d01@d23/np.sqrt((d01@d01)*(d23@d23))
    p2 = d02@d13/np.sqrt((d02@d02)*(d13@d13))
    plsm.append([p1, p2])

# dist = alldist[0]

# print(np.sqrt(util.group_mean(dist**2, grp)))

# top10 = []

# for j in range(4):
#     print([words[i] for i in np.argsort(score[grp==j])[:5]])

#%% 

# this = np.where(Scorr[aye,jay]==1)[0][781]
# this = np.where(Scorr[aye,jay]==1)[0][670]
# this = 39526
# this = 17400
# this = 5743
# this = 11807
# this = 
this = 30190
# this = indep[0]

c1 = aye[this]
c2 = jay[this]

_, grp = np.unique(Es[:,[c1,c2]], axis=0, return_inverse=True)
grp_ids = [np.where(grp==j)[0] for j in range(4)]

centroids = util.group_mean(vecs, grp, axis=0)

alldist = []
for j in range(4):
    alldist.append(la.norm(vecs - centroids[np.mod(grp+j, 4)], axis=1))
alldist = np.array(alldist)

score = alldist[0]**2 - np.mean(alldist[1:], axis=0)**2
# score = alldist[0]

d01 = (centroids[0]-centroids[1])
d02 = (centroids[0]-centroids[2])
d13 = (centroids[1]-centroids[3])
d23 = (centroids[2]-centroids[3])

p1 = d01@d23/np.sqrt((d01@d01)*(d23@d23))
p2 = d02@d13/np.sqrt((d02@d02)*(d13@d13))

# D01 = np.abs((2*Es[grp==0][:,util.rangediff(len(pie), [c2])]-1)@(2*Es[grp==1][:,util.rangediff(len(pie), [c2])]-1).T)
# D02 = np.abs((2*Es[grp==0][:,util.rangediff(len(pie), [c1])]-1)@(2*Es[grp==2][:,util.rangediff(len(pie), [c1])]-1).T)
# D13 = np.abs((2*Es[grp==1][:,util.rangediff(len(pie), [c1])]-1)@(2*Es[grp==3][:,util.rangediff(len(pie), [c1])]-1).T)
# D23 = np.abs((2*Es[grp==2][:,util.rangediff(len(pie), [c2])]-1)@(2*Es[grp==3][:,util.rangediff(len(pie), [c2])]-1).T)

top50 = []
for j in range(4):
    best = np.argsort(score[grp==j])[:50]
    top50.append([words[i] for i in grp_ids[j][best]])

tbl = pd.DataFrame({'00': top50[0][:20], 
                    '01': top50[1][:20], 
                    '10': top50[2][:20], 
                    '11': top50[3][:20]})

#%%
cmap = cm.Dark2

_, ax = plt.subplots()
ax.axis('off')
tabl = ax.table(cellText=tbl.values, colLabels=tbl.keys(), loc='center', 
         colColours=cmap(range(4)), alpha=0.5)
tabl.auto_set_font_size(False)
tabl.set_fontsize(11)
# pd.plotting.table(ax, tbl, loc='center', fontsize=50, colColours=cmap(range(4)))

#%%
I = 10

dom1 = 0
dom2 = 2

dom3 = 1
dom4 = 3

# val1 = 

# dv = centroids[dom2] - centroids[dom1]
# dv = W[c1]
# nrm1 = (vecs[grp==dom1]**2).sum(1,keepdims=True)
# nrm2 = (vecs[grp==dom2]**2).sum(1,keepdims=True)
# D = nrm1 + nrm2.T - 2*vecs[grp==dom1]@vecs[grp==dom2].T
# D = util.yuke(Es[grp==dom1],Es[grp==dom2])

# aln = ((vecs[grp==dom2]@dv)[None,:] - (vecs[grp==dom1]@dv)[:,None])/np.sqrt(D*(dv@dv))
# wa = ((vecs[grp==dom1] + dv01[None,:])@vecs[grp==dom2].T)
# wa = wa / (la.norm(vecs[grp==dom1],axis=1,keepdims=True)*la.norm(vecs[grp==dom2],axis=1,keepdims=True).T)
aln = -util.yuke(Es[grp==dom1],Es[grp==dom2])


# analog = aln.argmax(1)
# base = np.arange(len(analog))
base, analog = util.unbalanced_assignment(-aln, one_sided=True)

dom1ids = grp_ids[dom1][base]
dom2ids = grp_ids[dom2][analog]

# idx = np.argsort(1*score[dom1ids] + 1*score[dom2ids])
# idx = np.argsort(-aln[base,analog])

# print([(words[i], words[j]) for i,j in zip(dom1ids[idx[:10]], dom2ids[idx[:10]])])


# dv2 = centroids[dom4] - centroids[dom3]
# # dv = W[c1]
# nrm3 = (vecs[grp==dom3]**2).sum(1,keepdims=True)
# nrm4 = (vecs[grp==dom4]**2).sum(1,keepdims=True)
# D = nrm3 + nrm4.T - 2*vecs[grp==dom3]@vecs[grp==dom4].T
# D = util.yuke(Es[grp==dom1],Es[grp==dom2])

# aln2 = ((vecs[grp==dom4]@dv)[None,:] - (vecs[grp==dom3]@dv)[:,None])/np.sqrt(D*(dv2@dv2))
# wa = ((vecs[grp==dom1] + dv01[None,:])@vecs[grp==dom2].T)
# wa = wa / (la.norm(vecs[grp==dom1],axis=1,keepdims=True)*la.norm(vecs[grp==dom2],axis=1,keepdims=True).T)
aln2 = -util.yuke(Es[grp==dom3],Es[grp==dom4])

# analog = aln.argmax(1)
# base = np.arange(len(analog))
base2, analog2 = util.unbalanced_assignment(-aln2, one_sided=True)

dom3ids = grp_ids[dom3][base2]
dom4ids = grp_ids[dom4][analog2]

## Higher-order alignment

diff12 = (vecs[dom2ids] - vecs[dom1ids])
diff34 = (vecs[dom4ids] - vecs[dom3ids])
nrm12 = la.norm(diff12,axis=1, keepdims=True)
nrm34 = la.norm(diff34,axis=1, keepdims=True)

ovlp = (diff12@diff34.T)/(nrm12*nrm34.T)

pair = ovlp.argmax(1)
idx = np.argsort(-ovlp[np.arange(len(pair)),pair])

print([f'{words[dom1ids[i]]}:{words[dom2ids[i]]}::{words[dom3ids[j]]}:{words[dom4ids[j]]}' for i,j in zip(idx[:I],pair[idx[:I]])])



# idx = np.argsort(1*score[dom1ids] + 1*score[dom2ids])
# idx = np.argsort(-aln2[base2,analog2])

# #%%
# dom1 = 0
# dom2 = 2

# dv = centroids[dom2] - centroids[dom1]
# # dv = W[c1]
# nrm1 = (vecs[grp==dom1]**2).sum(1,keepdims=True)
# nrm2 = (vecs[grp==dom2]**2).sum(1,keepdims=True)
# D = nrm1 + nrm2.T - 2*vecs[grp==dom1]@vecs[grp==dom2].T

# aln = ((vecs[grp==dom2]@dv)[None,:] - (vecs[grp==dom1]@dv)[:,None])/np.sqrt(D*(dv@dv))
# # wa = ((vecs[grp==dom1] + dv01[None,:])@vecs[grp==dom2].T)
# # wa = wa / (la.norm(vecs[grp==dom1],axis=1,keepdims=True)*la.norm(vecs[grp==dom2],axis=1,keepdims=True).T)

# # analog = aln.argmax(1)
# # base = np.arange(len(analog))
# base, analog = util.unbalanced_assignment(-aln, one_sided=True)

# dom1ids = grp_ids[dom1][base]
# dom2ids = grp_ids[dom2][analog]

# # idx = np.argsort(1*score[dom1ids] + 1*score[dom2ids])
# idx = np.argsort(-aln[base,analog])

# print([(words[i], words[j]) for i,j in zip(dom1ids[idx[:10]], dom2ids[idx[:10]])])

#%%

# P = (vecs-vecs[[0]])@(W[[c1,c2]].T@np.diag(1/(pie[[c1,c2]])))
# P = (vecs-vecs.mean(0))@(W[[c1,c2]].T@np.diag(1/(pie[[c1,c2]])))
P = (vecs-vecs.mean(0))@(W[[c1,c2]].T)@np.diag(1/(pie[[c1,c2]]))
# P = V_@U.T
# P = V_@V[:2].T

plt.scatter(P[:,0], P[:,1], s=10, c=cmap(grp), 
            zorder=0, alpha=0.5, edgecolors='none')

# plt.scatter(centroids@W[c1]/pie[c1], centroids@W[c2]/pie[c2], s=400, marker='*', 
#             c=cmap(np.arange(4)),edgecolors='white', zorder=1, linewidths=2)

egs00 = [words.index('niece'), words.index('retiree')]
egs01 = [words.index('girl'), words.index('pensioner')]
egs10 = [words.index('uncle'), words.index('roommate')]
egs11 = [words.index('boy'), words.index('flatmate')]

# egs00 = [words.index('nephew'), words.index('physicist')]
# egs01 = [words.index('niece'), words.index('radiologist')]
# egs10 = [words.index('boy'), words.index('chemist')]
# egs11 = [words.index('girl'), words.index('pharmacist')]

# egs00 = [words.index('trekker'), words.index('angler'), words.index('climber')]
# egs01 = [words.index('canadian'), words.index('hausa'), words.index('hindu')]
# egs10 = [words.index('linguist'), words.index('neuroscientist'), words.index('tinsmith')]
# egs11 = [words.index('neoliberal'), words.index('king'), words.index('communist')]
# egs00 = [i for i in  np.where(grp==0)[0][np.argsort(score[grp==0])[:4]]]
# egs01 = [i for i in  np.where(grp==1)[0][np.argsort(score[grp==1])[:4]]]
# egs10 = [i for i in  np.where(grp==2)[0][np.argsort(score[grp==2])[:4]]]
# egs11 = [i for i in  np.where(grp==3)[0][np.argsort(score[grp==3])[:4]]]

alleg = np.concatenate([egs00, egs01, egs10, egs11])

plt.scatter(P[alleg,0], P[alleg,1], c=cmap(grp[alleg]), s=20, edgecolor='k')
for this in alleg:
    if P[this,0] > 0:
        dx = 5e-2
        ha = 'left'
    elif P[this,0] < 0:
        dx = -5e-2
        ha = 'right'
    else:
        dx = 1e-2 
        ha = 'left'
    if (P[this,1] > 0) or (words[this] in ['uncle']):
        dy = 5e-2
        va = 'bottom'
        # ha = 'right'
    if (P[this,1] < 0)*(words[this] not in ['uncle']) or (words[this] in ['boy','girl']):
        dy = -5e-2
        va = 'top'
        # ha = 'right'
    else:
        dy = 5e-2
        va = 'bottom'
    plt.text(P[this,0]+dx,P[this,1]+dy,words[this],
             horizontalalignment = ha,
             verticalalignment = va,
             color=cmap(grp[this]),
             bbox={'facecolor':'white', 
                   'edgecolor': cmap(grp[this]), 
                   'alpha': 0.8,
                   'boxstyle': 'round'})

dicplt.square_axis()
plt.axis(False)

