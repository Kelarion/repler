
CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from sklearn.manifold import MDS

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

from pypoman import compute_polytope_vertices, compute_polytope_halfspaces
import cvxpy as cvx
import polytope as pc
from hsnf import column_style_hermite_normal_form

# my code
import students as stud
import assistants
import experiments as exp
import util
import pt_util
import tasks
import server_utils as su
import plotting as tplt
import grammars as gram
import dichotomies as dics


#%%

nbin = 20
n_samp = 10000
std = 1

wa = np.linspace(0,2*np.pi, nbin)

phi, theta = np.meshgrid(wa, wa)

phi = phi.flatten()
theta = theta.flatten()

T = util.flat_torus(phi, theta)
K = T.T@T / 2

col = np.random.choice(range(nbin**2), n_samp)
guess = np.argmax(K@np.eye(nbin**2)[:,col] + np.random.randn(nbin**2, len(col))*std, axis = 0)

theta_err = util.circ_distance(theta[col], theta[guess])
theta_swap = util.circ_distance(phi[col], theta[guess])

phi_err = util.circ_distance(phi[col], phi[guess])
phi_swap = util.circ_distance(theta[col], phi[guess])

theta_hist = np.vstack([np.histogram(theta_err[phi[col]==p], range=[-np.pi, np.pi], density=True)[0] for p in wa])
phi_hist = np.vstack([np.histogram(phi_err[theta[col]==p], range=[-np.pi, np.pi], density=True)[0] for p in wa])


