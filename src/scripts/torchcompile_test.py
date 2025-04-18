CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'

import os, sys, re
import pickle
from time import time
import copy
from dataclasses import dataclass
from typing import Optional

sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.distributions as dis
import torch.linalg as tla
import numpy as np
from itertools import permutations, combinations
# from tqdm import tqdm

import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import numpy.linalg as nla
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

import matplotlib.pyplot as plt

from numba import njit
import math

# my code
import util
import pt_util
import bae_util
import bae_models
import students

#%%

import torch._dynamo
torch._dynamo.config.suppress_errors = True


#%%

Strue = torch.FloatTensor(util.F2(4)).cuda()
W = torch.FloatTensor(sts.ortho_group.rvs(4)).cuda()
X = (Strue-Strue.mean(0))@W.T
b = -Strue.mean(0)@W.T

perturb = torch.FloatTensor(np.random.choice([0,1], Strue.shape, p=[0.9,0.1])).cuda()
S = (Strue + perturb)%2 

#%%

Sest = bmf(X-b, 1*S, W, S.T@S, N=len(X), temp=1e-6)

