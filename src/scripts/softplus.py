
import os, sys, re
import pickle
from dataclasses import dataclass
import itertools

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

import numpy as np
import scipy as sp
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
from scipy.optimize import root_scalar
from sklearn.manifold import MDS


#%%

class RadialBoyfriend:
    """
    Gaussian process receptive fields with RBF kernel
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
        return np.random.multivariate_normal(mu, K, size=size).T/np.sqrt(size)

#%% Plot a specific choice of kappa

n_samp = 500
n_noise = 10000
dim = 5000

kappa = 1
noise_std = 4

th = np.linspace(-np.pi, np.pi, n_samp+1)
th0 = int(n_samp//2)

pop = RadialBoyfriend(0.5/kappa)

crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
thresh = np.sqrt(2*(1-pop(np.pi)))/2

X = pop.sample(th, size=dim)

U,s,V = la.svd(X-X.mean(0), full_matrices=False)
k = np.argmax(np.cumsum(s**2)/np.sum(s**2) > 0.99) + 1
pr = ((s**2).sum()**2)/(s**4).sum()

noise = np.random.randn(n_noise, dim)*noise_std

# noise_proj = noise@V[:k].T
x_pert = X[th0] + noise

xhat = th[(x_pert@X.T).argmax(1)]

diffs = X - X[th0]
diffs = diffs/(la.norm(diffs, axis=1, keepdims=True)+1e-6)

x_proj = noise@diffs.T #/la.norm(x_pert-X[th0], axis=1, keepdims=True) 

ovlp = diffs@noise.T/la.norm(noise, axis=1)

legit = np.abs(th[ovlp.argmax(0)]) > th[th0+1]

#%%

plt.subplot(1,2,1)
# plt.scatter(th[ovlp.argmax(0)][legit], 
#             x_proj[np.arange(n_noise),ovlp.argmax(0)][legit]**2, 
#             c=np.abs(xhat[legit])>crit, alpha=0.2)
plt.scatter(th[ovlp.argmax(0)], 
            x_proj[np.arange(n_noise),ovlp.argmax(0)]**2, 
            c=np.abs(xhat)>crit, alpha=0.2)

ylims = plt.ylim()

plt.plot([crit, crit], ylims)
plt.plot([-crit, -crit], ylims)
plt.plot(plt.xlim(), [thresh**2, thresh**2])
plt.ylim(ylims)

wa = np.mean(x_proj[np.arange(n_noise),ovlp.argmax(0)][legit]**2)
alf = wa/(2*noise_std**2)

plt.subplot(1,2,2)
# dist = sts.gamma(a=k/10, scale=2*noise_std**2)
dist = sts.gamma(a=alf, scale=2*noise_std**2)
plt.hist(x_proj[np.arange(n_noise),ovlp.argmax(0)][legit]**2, 
         bins=25, density=True, orientation='horizontal')
plt.plot(dist.pdf(np.linspace(0,ylims[1],100)), 
         np.linspace(0,ylims[1],100), 'k-')

plt.ylim(ylims)

#%%


n_samp = 500
n_noise = 10000
dim = 5000

i = 0
for kappa in [0.1, 1, 10]:
    for noise_std in [0.5, 1, 2]:

        th = np.linspace(-np.pi, np.pi, n_samp+1)
        th0 = int(n_samp//2)
        
        pop = RadialBoyfriend(0.5/kappa)
        
        crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
        thresh = np.sqrt(2*(1-pop(np.pi)))/2
        
        X = pop.sample(th, size=dim)
        
        U,s,V = la.svd(X-X.mean(0), full_matrices=False)
        k = np.argmax(np.cumsum(s**2)/np.sum(s**2) > 0.99) + 1
        pr = ((s**2).sum()**2)/(s**4).sum()
        
        noise = np.random.randn(n_noise, dim)*noise_std
        
        # noise_proj = noise@V[:k].T
        x_pert = X[th0] + noise
        
        xhat = th[(x_pert@X.T).argmax(1)]
        
        plt.subplot(3,3,i+1)
        cnts, values, bars = plt.hist(xhat, bins=25, density=True)
        bin_ctr = (values[:-1] + values[1:])/2
        
        cols = ['royalblue', 'darkorange']
        for value, bar in zip(bin_ctr, bars):
            bar.set_facecolor(cols[int(np.abs(value) >= crit)])
                
        plt.ylabel('density')
        plt.xlabel('error')
        if i == 4:
            plt.legend([bars[0], bars[int(len(bars)//2)]], ['guess', 'correct'])
        
        i += 1

#%% Sweep over many kappa

n_samp = 500
n_noise = 10000
dim = 5000

# noise_std = 2
# kaps = np.linspace(0.1, 200, 100)
kaps = 2**np.linspace(-8,8,100)

th = np.linspace(-np.pi, np.pi, n_samp+1)
th0 = int(n_samp//2)

wa = []
for noise_std in [0.5, 1, 2]:
    alf = []
    alf2 = []
    pl = []
    pr = []
    pg = []
    pg_ = []
    for kap in tqdm(kaps):
        
        pop = RadialBoyfriend(0.5/kap)
    
        crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
        thresh = np.sqrt(2*(1-pop(np.pi)))/2
    
        X = pop.sample(th, size=dim)
    
        noise = np.random.randn(n_noise, dim)*noise_std
        x_pert = X[th0] + noise
        
        diffs = X - X[th0]
        diffs = diffs/(la.norm(diffs, axis=1, keepdims=True)+1e-6)
    
        U,s,V = la.svd(diffs, full_matrices=False)
        pr.append(((s**2).sum()**2)/(s**4).sum())
    
        x_proj = noise@diffs.T #/la.norm(x_pert-X[th0], axis=1, keepdims=True) 
    
        ovlp = diffs@noise.T/la.norm(noise, axis=1)
        
        legit = np.abs(th[ovlp.argmax(0)]) > th[th0+1]
        
        max_proj = x_proj[np.arange(n_noise),ovlp.argmax(0)]
        average = np.mean(max_proj[legit]**2)
        alf.append(average/(2*noise_std**2)) # fact about gamma distribution
        pl.append(np.mean(legit))
        
        xhat = th[(x_pert@X.T).argmax(1)]
        
        distr = sts.gamma(a=alf[-1], scale=2*noise_std**2)
        
        pg.append(np.mean(np.abs(xhat) > crit))
        pg_.append(pl[-1]*(1-distr.cdf(thresh))*(1-crit/np.pi))
    
    alf = np.array(alf)
    pg = np.array(pg)
    pg_ = np.array(pg_)
    pl = np.array(pl)
    pr = np.array(pr)
    
    wa.append(pg)
    

#%%

thresh = np.sqrt(2*(1 - (np.exp(-kaps) - spc.i0(kaps))/(np.exp(kaps) - spc.i0(kaps))))/2
crit = 2*np.arccos((np.sqrt(1 + 4*kaps**2) - 1)/(2*kaps))

plt.subplot(1,3,1)
alf_approx = 2 + np.log(1+kaps)/2.5
plt.plot(np.log(kaps), 2*alf)
plt.plot(np.log(kaps), alf_approx)
plt.title('Effective degrees of freedom')

plt.subplot(1,3,2)
pl_approx = spc.expit(np.log(1+kaps*9)/3.5)
plt.plot(np.log(kaps), pl)
plt.plot(np.log(kaps), pl_approx)
plt.title('Probability of potent perturbation')

plt.subplot(1,3,3)
plt.plot(np.log(kaps), pl_approx*(1-sts.gamma(a=alf_approx/2, scale=2*noise_std**2).cdf(thresh))*(1-crit/np.pi))
plt.plot(np.log(kaps), pg)
plt.title('Guess probability')

