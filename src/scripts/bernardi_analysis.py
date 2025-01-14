CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle
from time import time
import math
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

import matplotlib.pyplot as plt

import networkx as nx
import cvxpy as cvx

from numba import njit

# my code
import util
import df_util
import bae
import plotting as tpl
import anime

#%%

bern_hpc = pkl.load(open(SAVE_DIR+'HPC_dump.pck','rb'))
bern_pfc = pkl.load(open(SAVE_DIR+'DLPFC_dump.pck','rb'))
bern_acc = pkl.load(open(SAVE_DIR+'ACC_dump.pck','rb'))

Z_hpc = np.stack([np.mean(z,axis=0) for z in bern_hpc.values()])
Z_pfc = np.stack([np.median(z,axis=0) for z in bern_pfc.values()])
Z_acc = np.stack([np.median(z,axis=0) for z in bern_acc.values()])

#%% HPC

# baer = bae.BAE(Z_hpc, 100, pvar=0.95, penalty=0.1)
baer = bae.BAE(Z_hpc, 100, pvar=1, penalty=0.1)
baer.init_optimizer(decay_rate=0.98, period=2)

en = []
for t in tqdm(range(1000)):
    #r = np.sum(pvar< (0.8 + 0.2*(t//10)/10))
    # baer.proj()
    # baer.scl = baer.scaleS()
    baer.grad_step()
    en.append(baer.energy())
    
Shpc = baer.S.todense()

#%% PFC

baer = bae.KernelBAE(Z_pfc, 100, pvar=0.95, penalty=1e-1)
baer.init_optimizer(decay_rate=0.98, period=2, initial=5)

en = []
for t in tqdm(range(1000)):
    #r = np.sum(pvar< (0.8 + 0.2*(t//10)/10))
    # baer.proj(pvar=0.9)
    
    baer.grad_step()
    en.append(baer.energy())
    
Spfc = baer.S.todense()

#%% ACC

baer = bae.KernelBAE(Z_acc, 100, pvar=0.95, penalty=1)
baer.init_optimizer(decay_rate=0.98, period=2, initial=5)

en = []
for t in tqdm(range(1000)):
    #r = np.sum(pvar< (0.8 + 0.2*(t//10)/10))
    # baer.proj(pvar=0.9)
    baer.scl = baer.scaleS()
    baer.grad_step()
    en.append(baer.energy())
    
Sacc = baer.S.todense()

