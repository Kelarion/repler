
CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

from itertools import permutations, combinations
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp

from sklearn import svm, discriminant_analysis, manifold, linear_model
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

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.infer import SVI, Trace_ELBO

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

bins = 16

cols = np.linspace(0, 2*np.pi, bins)
phi, theta = np.meshgrid(cols, cols)
vals = torch.tensor(np.stack([phi.flatten(), theta.flatten()]))

def cosine_kernel(phi, theta, upper, lower):
    return torch.cos(phi - upper) + torch.cos(theta - lower)


@config_enumerate
def POMR(upper_col, lower_col, cue, resp=None):
    """
    Partially observable multinomial regression
    """
    
    T = len(upper_col)
    
    # alpha = pyro.param("alpha", dist.Beta(1,1)) 
    # beta = pyro.param("beta", dist.LogNormal(0,1))
    # sigma = pyro.param("sigma", dist.LogNormal(0, 2))
    
    alpha = pyro.param("alpha", torch.tensor(0.5), constraint=dist.constraints.unit_interval) 
    beta = pyro.param("beta", torch.tensor(1.0), constraint=dist.constraints.positive)
    # sigma = pyro.param("sigma", torch.tensor(1.0), constraint=dist.constraints.positive)
    
    ortho = cosine_kernel(vals[0][None,:], vals[1][None,:], upper_col[:,None], lower_col[:,None])
    swapped = cosine_kernel(vals[0][None,:], vals[1][None,:], lower_col[:,None], upper_col[:,None])
    logits = beta*(ortho + alpha*swapped)
    
    with pyro.plate("data", T):
        mem = pyro.sample("mem", dist.Categorical(logits=logits))
        mean = cue*vals[0, mem] + (1-cue)*vals[1, mem]
        pyro.sample("resp", dist.VonMises(mean, 1))


def train(model, guide, lr=0.005, n_steps=201, **data):
    pyro.clear_param_store()
    adam_params = {"lr": lr}
    adam = pyro.optim.Adam(adam_params)
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for step in range(n_steps):
        loss = svi.step(**data)
        if step % 50 == 0:
            print('[iter {}]  loss: {:.4f}'.format(step, loss))
            
            
#%%

nbin = 16
n_samp = 5000
std = 1

angle = np.pi/2
# angle = 0

wa = np.linspace(0, 2*np.pi, nbin)

upc, lowc = np.meshgrid(wa, wa)

upc = upc.flatten()
lowc = lowc.flatten()

A,B = util.random_canonical_bases(4, 2, np.ones(2)*angle)

# T = util.flat_torus(upc, lowc)
T = A@util.flat_torus(upc) + B@util.flat_torus(lowc)
K = T.T@T 

#%%
col = np.random.choice(range(nbin**2), n_samp)
mem = np.argmax(K[:,col] + np.random.randn(nbin**2, len(col))*std, axis = 0)

cue = np.random.choice([0,1], n_samp)
resp = cue*upc[mem] + (1-cue)*lowc[mem]

#%%

def guide(**data):
    pass

train(POMR, guide, n_steps=1000, lr=0.01,
      upper_col=torch.tensor(upc[col]), 
      lower_col=torch.tensor(lowc[col]),
      cue=torch.tensor(cue),
      resp=torch.tensor(resp))

#%%
kernel = NUTS(POMR)
mcmc = MCMC(kernel, num_samples=250, warmup_steps=50)

mcmc.run(upper_col=torch.tensor(upc[col]), 
          lower_col=torch.tensor(lowc[col]), 
          cue=torch.tensor(cue), 
          resp=torch.tensor(resp))

posterior_samples = mcmc.get_samples()


#%%

# theta_err = util.circ_distance(theta[col], theta[guess])
# theta_swap = util.circ_distance(phi[col], theta[guess])

# phi_err = util.circ_distance(phi[col], phi[guess])
# phi_swap = util.circ_distance(theta[col], phi[guess])

# theta_hist = np.vstack([np.histogram(theta_err[phi[col]==p], range=[-np.pi, np.pi], density=True)[0] for p in wa])
# phi_hist = np.vstack([np.histogram(phi_err[theta[col]==p], range=[-np.pi, np.pi], density=True)[0] for p in wa])


