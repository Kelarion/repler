CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/cut_figs/vids/'
 
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

p = 32

# Strue = df_util.randtree_feats(p, 2, 4)
Strue = df_util.schurcats(p, 0.5)

X = df_util.noisyembed(Strue, 200, 30, scl=1e-3)

#%%

these_periods = [1,50,70,100]
samples = 100
epochs = 1500

baer = bae.BAE(X, 4*Strue.shape[1], pvar=0.95, penalty=1e-2, steps=2)
baer.init_optimizer(decay_rate=0.98, period=10, initial=5)

en = []
S_hat = []
for t in tqdm(range(epochs)):
    #r = np.sum(pvar< (0.8 + 0.2*(t//10)/10))
    # baer.proj(pvar=1)
    
    baer.grad_step()
    en.append(baer.energy())
    
    if t/baer.schedule.period in these_periods:
        for _ in range(samples):
            baer.grad_step(pause_schedule=True)
            S_hat.append(baer.S.todense())
            
    
S = baer.S.todense()

S_hat = np.array(S_hat)

#%%

which_period = 3

# idx = np.lexsort([S_hat[-1].argmax(0), -S_hat[-1].sum(0)])
idx = np.argsort(np.abs((2*S-1).T@Strue).argmax(1))

t1 = which_period*samples
t2 = (which_period+1)*samples

# scramble = np.random.permutation(np.arange(t1,t2))

matanime = anime.MatrixAnime(S_hat[t1:t2][...,idx], grid=False, cmap='binary')

fname = 'annealing_p%d_t%d_schur.mp4'%(p, which_period)

matanime.save(SAVE_DIR+fname)

# for this_S in S_hat:
    
    



