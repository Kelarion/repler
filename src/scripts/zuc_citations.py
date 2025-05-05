CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
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
import scipy.io as spio
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

stefano = spio.loadmat(SAVE_DIR + 'joint_author.mat')

#%%

W = stefano['joint_papers'].astype(int)
names = np.array([l.item() for l in stefano['label'][0]])

n = W.sum(1)
these = n > 0
N = np.sum(these)
names = names[these]

A = 1*(W>0)[these,:][:,these]
B = (W + np.diag(n))[these,:][:,these]
C = np.diag(1/n[these])@B@np.diag(1/n[these])
# P = np.where(C>1, np.log2(C), 0)
L = np.diag(n[these]) - W[these,:][:,these]

# np.diag()

#%%
# dothis = A
dothis = B
# dothis = C
# dothis = L

l,V = la.eigh(util.center(dothis))
X = V[:,l>1e-6]@np.diag(np.sqrt(l[l>1e-6]))

#%%

# neal = bae_util.Neal(1, initial=1e-5)
# max_iter = 5

neal = bae_util.Neal(0.95)
max_iter = None

allS = []
for k in tqdm(range(1, 2*N)):

    mod = bae_models.KernelBMF(k, tree_reg=1, scale_lr=1)
    en = neal.fit(mod, X, max_iter=max_iter, verbose=False)
    
    allS.append(mod.S)

#%%

plt.plot(range(1,2*N), [util.cka(S@S.T, X@X.T) for S in allS])

