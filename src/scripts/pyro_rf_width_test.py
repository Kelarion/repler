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
from sklearn.neighbors import KNeighborsRegressor as knr
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.integrate import quad as qint
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from sklearn.manifold import MDS

import numpyro as npr
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrng

import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as dicplt
import dichotomies as dics

#%%

class MisesLogNormal:
    """
    Mises RF model (no 'von' because we are anti-aristocracy) with uniformly
    distributed means and log-normally distributed concentration
    """
    
    def __init__(self, mean, std):
        """
        Create a infinite population of neurons, whose RF widths are distributed
        with specified mean and standard deviation
        
        note: these are the mean and std of the widths, not the log-widths
        """
        
        tol = 1e-10
        
        ## compute mean and sigma of the log-variable
        ## NB: concentration is 1/width, which is also log-normal but with -mean
        self.mu = -np.log((mean**2)/(np.sqrt(mean**2 + std**2) + tol) + tol)
        self.sig = np.sqrt(np.log(1 + (std**2)/(mean**2 + tol)))
        
        self.mu = np.min([self.mu, 100])
        self.sig = np.max([self.sig, 1e-5])
        
    def __call__(self, error, quantile=1e-4):
        """
        compute k(x,y) = k(x-y) ... so input x-y
        """
        
        k_min = sts.lognorm.ppf(quantile, self.sig, scale=np.exp(self.mu))
        k_max = sts.lognorm.ppf(1-quantile, self.sig, scale=np.exp(self.mu))
        
        return np.array([qint(self.integrand, k_min, k_max, args=(c,))[0] for c in error])

    def sample(self, colors, size=1):
        """
        Sample activity in response to colors
        """
        
        # because numpy broadcasting ...... :(((
        colshp = np.shape(colors)
        col_dims = (1,)*len(colshp)
        if len(np.shape(size)) == 0:
            size = (size,)
        exp_dims = tuple([len(colshp)+i for i in range(len(size))])
        cols = np.expand_dims(colors, exp_dims)
        
        mean = np.pi*(2*np.random.rand(*col_dims, *size)-1)
        kap = np.exp(self.sig*np.random.randn(*col_dims, *size) + self.mu)
        
        # clamp so that it's not too big or too small
        kap = np.min([kap, (1e2)*np.ones(col_dims+size)], axis=0)
        kap = np.max([kap, (1e-3)*np.ones(col_dims+size)], axis=0)
        
        numer = np.exp(kap*np.cos(cols - mean))
        denom = 2*np.pi*spc.i0(kap)
        
        return numer/denom

    # def subpop_kernel(self, error, kappa):
    #     """
    #     kernel of subpopulation, conditional on specific concentration parameter
    #     """
        
    #     kap = kappa*(kappa < 100) + 100*(kappa >= 100)
        
    #     numer = spc.i0(2*kap*np.cos(error/2))
    #     return numer/(2*np.pi*spc.i0(kap)**2)
    def subpop_kernel(self, error, kappa):
        """
        kernel of subpopulation, conditional on specific concentration parameter
        """
        
        kap = kappa*(kappa < 100) + 100*(kappa >= 100)
        
        numer = spc.i0(2*kap*np.cos(error/2))
        return numer/spc.i0(2*kap)

    def p_k(self, k):
        """
        log-normal pdf of inverse width
        """
        
        denom = k*self.sig*np.sqrt(2*np.pi)
        return np.exp(-(np.log(k) - self.mu)**2/(2*self.sig**2))/denom

    def integrand(self, k, err):
        
        return self.subpop_kernel(err, k)*self.p_k(k)
    
def mises_simil(error, kappa):
    """
    kernel of subpopulation, conditional on specific concentration parameter
    """
    
    kap = kappa*(kappa < 100) + 100*(kappa >= 100)
    
    numer = spc.i0(2*kap*np.cos(error/2))
    return numer/(spc.i0(2*kap))

def model(bins=10, beta=1, mu_prior=0, sig_prior=1, err=None):
    """
    generative model
    """
    
    # concentration parameter
    mu = npr.sample('mu', dist.Normal(0,1))  
    sig = npr.sample('sig', dist.InverseGamma(1,1))
    w = npr.sample('w', dist.Normal(mu, sig))
    k = jnp.exp(w) # the von-mises concentration parameter
    
    # sample error
    bins = jnp.linspace(-np.pi, np.pi, bins)
    kern = jsp.special.i0(2*k*jnp.cos(bins/2))/jsp.special.i0(2*k)
    
    return npr.sample('err', dist.CategoricalLogits(beta*kern), obs=err)

    
#%%

this_sig = 1
this_mu = 1
num_bin = 10
num_samp = 5000
beta = 3

colors = np.linspace(-np.pi, np.pi, num_bin)

pop = MisesLogNormal(this_mu, this_sig)

kernel = pop(colors, quantile=1e-3)
K_perturb = kernel[:,None] + np.random.gumbel(scale=1/beta, size=(num_bin,num_samp))

tcc_pred = K_perturb.argmax(0)
tcc_err = util.circ_distance(np.zeros(num_samp), colors[tcc_pred])


#%%

nuts_kern = npr.infer.NUTS(model)
mcmc = npr.infer.MCMC(nuts_kern, num_warmup=500, num_samples=1000)

rng = jrng.PRNGKey(0)
mcmc.run(rng, bins=10, beta=3, err=tcc_pred)






