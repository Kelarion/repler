
CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import permutations, combinations
import itertools as itt
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

import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as dicplt
import dichotomies as dics

import distance_factorization as df

#%%

class FAN:
    """
    Feature analogy network
    """
    
    def __init__(self, dim_inp, dim_out, activation=nn.ReLU(), tau_a=1):
        
        self.Nx = dim_inp
        self.Ny = dim_out
        
        self.init_plastic()

        # hyperparameters
        self.sigma = activation
        self.tau_a = tau_a        
    
    def init_plastic(self):
        
        # recurrent weights
        self.A = torch.zeros(1,self.Nx, self.Nx)
        self.R = torch.eye(1,self.Nx, self.Nx)

    def forward(self, X, initialize=False):
        """
        Advance one forward pass of the network
        
        X and Y are shape (batch, neur)
        """
        
        if initialize:
            self.init_plastic()
        
        X_ = X[...,None]
        
        # forward pass
        X_rec = torch.sign((self.R + self.A)@X_)
        
        return X_rec
    
    def backward(self, X, X_samp):
        """
        Learning rule for the "recurrent" weights
        """
        
        self.A = self.A + self.tau_a*(X - self.A@X_samp)@X_samp.T
        

class Ratt:
    """
    Recursive attention from plastic weights
    """
    
    def __init__(self, dim_inp, dim_out, num_head):
        
        self.nx = dim_inp
        self.ny = dim_out
        self.nh = num_head
        
        # fixed weights
        self.R = torch.randn(self.nh, self.nx, self.nx)
        
        # plastic weights
        self.init_plastic()
    
    def init_plastic(self):
        
        self.W = torch.zeros(1, self.nh, self.ny, self.nx) # input-output
        self.V = torch.zeros(1, self.nh, 1, self.nx) # normalization
    
    def forward(self, X, initialize=False):
        """
        Advance one timestep
        
        X is shape (batch, nx)
        """
        
        if initialize:
            self.init_plastic()
            
        btch = X.shape[:-1]
        
        # 'forward'
        A = self.R@X # key-query transformed inputs
        Y = self.W@A/self.V@A
        
        # 'backward'
        
        eps = torch.randn(btch, self.ny, 1)
        self.W += eps*X.swapaxes(-1,-2) # store in weights
        
        # self.V += 
        





