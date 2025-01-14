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

import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as dicplt
import dichotomies as dics

# import distance_factorization as df

#%%

# class GaussRF(nn.Module):
    
#     def __init__(self, n_neur, sigma=1):
        
#         self.mu = np.random.rand(n_neur)
#         self.sigma = sigma
        
#     def __call__(self, theta):
        
#         return np.exp((-(theta - self.mu)**2)/self.sigma**2)

#     # @staticmethod
#     def kernel(self, theta, phi):
        
#         c = np.sqrt(np.pi/4)
#         s2 = 2*self.sigma
#         erfs = (spc.erf((phi+theta)/s2)+spc.erf((2-phi-theta)/s2))
        
#         return c*self.sigma*np.exp((-1/s2**2)*(theta-phi)**2)*erfs

# class MisesRF(nn.Module):
    
#     def __init__(self, n_neur, kappa=1):
        
#         self.mu = np.pi*(2*np.random.rand(n_neur)-1)
#         self.kappa = np.array(kappa)
        
#     def __call__(self, theta):
        
#         # final_shape = self.kappa.shape + theta.shape + self.mu.shape
#         kapsz = len(self.kappa.shape)
#         thsz = len(theta.shape)
#         musz = len(self.mu.shape)
#         sz = kapsz + thsz + musz
        
#         # need to do funky things for broadcasting purposes
#         denom = 2*np.pi*spc.iv(0, self.kappa)
#         arg = np.einsum('i...,...jk->...ijk', denom, np.cos(theta-self.mu))

#         return np.exp(arg)/denom

#     # @staticmethod
#     def kernel(self, theta, phi):
        
#         numer = spc.iv(0, 2*self.kappa*np.cos((theta-phi)/2))
#         return numer/spc.iv(0,self.kappa)

# def GRF_kernel()
 

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

    def apprk(self, error):
        
        c = np.cos(error/2)
        
        # denom =

    def subpop_kernel(self, error, kappa):
        """
        kernel of subpopulation, conditional on specific concentration parameter
        """
        
        kap = kappa*(kappa < 100) + 100*(kappa >= 100)
        
        numer = spc.i0(2*kap*np.cos(error/2))
        return numer/(2*np.pi*spc.i0(kap)**2)

    def p_k(self, k):
        """
        log-normal pdf of inverse width
        """
        
        denom = k*self.sig*np.sqrt(2*np.pi)
        return np.exp(-(np.log(k) - self.mu)**2/(2*self.sig**2))/denom

    def integrand(self, k, err):
        
        return self.subpop_kernel(err, k)*self.p_k(k)
    

def ln_cdf(k, mu, sig):
   return 0.5*(1 + spc.erf((np.log(k)-mu)/(sig*np.sqrt(2))))
      
def ln_pdf(self, k):
    """
    log-normal pdf of inverse width
    """
    
    denom = k*self.sig*np.sqrt(2*np.pi)
    return np.exp(-(np.log(k) - self.mu)**2/(2*self.sig**2))/denom

    
def mises_simil(error, kappa):
    """
    kernel of subpopulation, conditional on specific concentration parameter
    """
    
    kap = kappa*(kappa < 100) + 100*(kappa >= 100)
    
    numer = spc.i0(2*kap*np.cos(error/2))
    return numer/(spc.i0(2*kap))

# def tcc_sample(k, x):

#%%

n_col = 3000
n_err = 100
n_neur = 500

samp_every = 3

n_pop = 1

beta = 5

dx = (2*np.pi/n_err)

colors = np.linspace(-np.pi, np.pi, n_col) # for the samples
errors = np.linspace(-np.pi, np.pi, n_err) # for the theory

clf = knr(n_neighbors=1)

mus = np.linspace(0, np.pi/2, 100)
sigs = np.linspace(0, 2, 5)

err_std = []
samp_err_std = []
tcc_err_std = []
err_kur = []
tcc_err_kur = []
samp_err_kur = []
for i,this_mu in enumerate(mus):
    stds = []
    samp_stds = []
    tcc_stds = []
    # tst = []
    kur = []
    tcc_kur = []
    samp_kur = []
    for this_sig in sigs:
        
        pop = MisesLogNormal(this_mu, this_sig)

        ## theory
        kernel = pop(errors, quantile=1e-3) # numerically integrate
        # kernel /= np.sqrt(np.sum(dx*kernel**2))
        kernel /= np.max(kernel)
        apprx = np.exp(beta*kernel)/np.sum(np.exp(beta*kernel)*dx)
        var = np.sum(dx*apprx*errors**2)
        stds.append(var)
        kur.append(np.sum(dx*apprx*errors**4)/var**2 - 3)
        
        if np.mod(i, samp_every) <=0:
            ## decoder
            X = pop.sample(colors, size=n_neur)
            X /= la.norm(X, axis=1,keepdims=True) # normalize response across neurons
            
            K = X@X.T
            K_perturb = K + np.random.gumbel(scale=1/beta, size=K.shape)
            tcc_pred = K_perturb.argmax(1)
            tcc_err = util.circ_distance(colors, colors[tcc_pred])
            tcc_stds.append(np.var(tcc_err))
            tcc_kur.append(sts.kurtosis(tcc_err))
            
            # X_tst = X + np.random.randn(n_col, n_neur)/np.sqrt(2*beta)
            X_tst = X + np.random.randn(n_col, n_neur)*np.sqrt(1/6)*(np.pi/beta)
            
            clf.fit(X, colors)
            pred = clf.predict(X_tst)
            err = np.arctan2(np.sin(colors-pred), np.cos(colors-pred))
            # err = util.circ_distance(pred, colors)
            
            samp_stds.append(np.var(err))
            samp_kur.append(sts.kurtosis(err))
        
    err_std.append(stds)
        # apprx *= (n_col/2*np.pi)
    err_kur.append(kur)
    
    if np.mod(i, samp_every) <=0:
        samp_err_std.append(samp_stds)
        tcc_err_std.append(tcc_stds)
        tcc_err_kur.append(tcc_kur)
        samp_err_kur.append(samp_kur)

err_std = np.array(err_std)
samp_err_std = np.array(samp_err_std)
tcc_err_std = np.array(tcc_err_std)

err_kur = np.array(err_kur)
tcc_err_kur = np.array(tcc_err_kur)
samp_err_kur = np.array(samp_err_kur)

#%%

# samps = samp_err_std
# samps = tcc_err_std 
# thry = err_std
thry = err_kur
samps = tcc_err_kur
# samps = samp_err_kur

cols = cm.copper(np.linspace(0,1,len(sigs)))
for i in range(len(sigs)):
    plt.plot(mus, thry[:,i]/(2*np.pi), color=cols[i], linestyle='--')
    plt.plot(mus[::samp_every], samps[:,i]/(2*np.pi), color=cols[i])


