CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
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

import networkx as nx
import cvxpy as cvx

from numba import njit

# my code
import util
import df_util
import bae

#%%


#%% Numerical verification of equations
n = 10
m = 16 # this should be small enough to loop over {0,1}^m

scl = 1.3

t = (n-1)/n

S = np.random.choice([0,1], size=(n,m))
K = util.center(S@S.T,-1)

s_ = S[:-1].mean(0)
stild = 2*s_-1
Sbar = S[:-1] - s_

k = K[:-1,-1]
k0 = K[-1,-1]

J = 2*(scl**2)*Sbar.T@Sbar + t*(scl**2)*np.outer(stild, stild)
h = J@s_ + t*(scl**2)*((1-s_)@s_)*stild - t*k0*scl*stild + 2*scl*Sbar.T@k


F = util.F2(m)

opt = util.qform(J, F).squeeze() - 2*F@h
opt2 = []
for f in F:
    Q = np.vstack([S[:-1], f])@np.vstack([S[:-1], f]).T
    opt2.append(util.centered_distance(K, scl*Q))


