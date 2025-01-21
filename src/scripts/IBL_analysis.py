
CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle as pkl
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

X = pkl.load(open('C:/Users/mmall/Downloads/X_forMatteo.pck','rb'))
trl = pkl.load(open('C:/Users/mmall/Downloads/trials_forMatteo.pck','rb'))

neurs = np.hstack([x for x in X.values()])
area = np.concatenate([i*np.ones(len(x.T)) for i,x in enumerate(X.values())])

neurz = (neurs-neurs.mean(0,keepdims=True))/(neurs.std(0,keepdims=True)+1e-12)

#%%




