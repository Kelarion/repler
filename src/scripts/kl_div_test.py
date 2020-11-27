dCODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model, neighbors
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

import students
import assistants
import experiments as exp
import util

#%%
mu1 = 0 # P
s1 = 2
mu2 = 0 # Q (always standard)
s2 = 1

d = 2

k = 4

n = 2500

x = np.random.randn(n,d) + mu1
y = np.random.randn(n,d) + mu2

knn =  neighbors.NearestNeighbors(n_neighbors=k)

rk = knn.fit(x).kneighbors()[0].max(1) # nn of x_i in X
sk = knn.fit(y).kneighbors(x)[0].max(1) # nn of x_i in Y

dkl = (d/n)*(np.sum(np.log(rk) - np.log(sk)) + np.log(n/(n-1)))

dkl_actual = 0.5*(la.norm(np.ones(d)*mu1)**2 + d*s1 - d - np.log(s1))
