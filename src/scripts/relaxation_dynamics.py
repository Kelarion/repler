CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
# import pickle
# from dataclasses import dataclass
# import itertools

sys.path.append(CODE_DIR)
from dataclasses import dataclass, fields, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import scipy.stats as sts
import scipy.linalg as la 
import scipy.special as spc

# import util
import plotting as tpl
import students
import super_experiments as sxp
import experiments as exp
import util

#%%

class RadialBoyfriend:
    """
    My special normalised (von) mises kernel
    
    a mises function which has zero integral and k(0) = 1
    """
    
    def __init__(self, width, center=True):
        
        self.kap = 0.5/width
        ## Set the scale so that the average variance of each neuron is 1
        ## and shift so the mean population response across stimuli is 0
        self.scale = 1/(np.exp(self.kap) - spc.i0(self.kap))
        if center:
            self.shift = spc.i0(self.kap)
        else:
            self.shift = 0
        
    def __call__(self, error, quantile=1e-4):
        """
        compute k(x,y) = k(x-y) ... so input x-y
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return (np.exp(self.kap*np.cos(error)) - self.shift)/denom

    def curv(self, x):
        """
        Second derivative
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return self.kap*np.exp(self.kap*np.cos(x))*(self.kap*np.sin(x)**2 - np.cos(x))/denom

    def deriv(self, x):
        """
        Derivative
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return -self.kap*np.sin(x)*np.exp(self.kap*np.cos(x))/denom

    def perturb(self, x, y):
        """
        The 'perturbation kernel', i.e. (f(x)-f(0))*(f(y)-f(0))
        """
        numer = self(x-y) - self(x) - self(y) + 1
        denom = 2*np.sqrt((1-self(x))*(1-self(y)))
        return numer/denom

    def sample(self, colors, size=1):
        """
        Sample activity in response to colors
        """
        K = self(colors[None] - colors[:,None])
        mu = np.zeros(len(colors))
        return np.random.multivariate_normal(mu, K, size=size).T

#%%

