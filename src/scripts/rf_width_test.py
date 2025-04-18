CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)
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

import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as tpl
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

class RadialBoyfriend:
    """
    Gaussian process receptive fields with RBF kernel
    """
    
    def __init__(self, width, center=True):
        
        # self.width = width
        self.kap = 0.5/width
        ## Set the scale so that the average variance of each neuron is 1
        ## and shift so the mean population response across stimuli is 0
        # self.scale = 1/(1-np.exp(-0.5/width)*spc.i0(0.5/width))
        # self.shift = 0
        self.scale = 1/(np.exp(0.5/width) - spc.i0(0.5/width))
        if center:
            self.shift = spc.i0(self.kap)
        else:
            self.shift = 0
        
    def __call__(self, error, quantile=1e-4):
        """
        compute k(x,y) = k(x-y) ... so input x-y
        """
        # return self.scale*np.exp((np.cos(error)-1)/(2*self.width))
        # return self.scale*(np.exp(0.5*np.cos(error)/self.width) - self.shift)
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return (np.exp(self.kap*np.cos(error)) - self.shift)/denom

    def sample(self, colors, size=1):
        """
        Sample activity in response to colors
        """
        # kern = util.CircularRBF(sigma=self.width, scale=self.scale)
        K = self(colors[None] - colors[:,None])
        mu = np.zeros(len(colors))
        return np.random.multivariate_normal(mu, K, size=size).T/np.sqrt(size)


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
    

class VMM:
    ## fix the local minima
    
    def __init__(self, k=0, pic=0.5, kmax=100):
        
        self.k = k
        self.pic = pic
        self.pig = 1-pic
        self.kmax = kmax
    
    def fit(self, err, iters=10):
        
        lik = []
        for i in range(iters):
            ## E step
            pe_c = self.pcorr(err)
            pc = (self.pic*pe_c/(self.pig/(2*np.pi) + self.pic*pe_c))
            
            ## M step
            R = np.cos(err)@pc/np.sum(pc)
            if (R > 0) and (self.ratio(self.kmax, R)>0):
                sol = root_scalar(self.ratio, args=(R,), bracket=(0,self.kmax))
                self.k = sol.root
            elif R <=0:
                self.k = 0
            else:
                self.k = self.kmax
            
            self.pic = np.mean(pc)
            self.pig = 1 - self.pic
            
            lik.append(np.mean(np.log(self.p(err))))
        
        return lik
        
    def sample(self, n):
        c = np.random.choice([0,1], size=n, p=(self.pig, self.pic))
        guess = np.pi*(2*np.random.rand(n)-1)
        corr = sts.vonmises(loc=0, kappa=np.max([self.k, 1e-6])).rvs(n)
        return np.where(c>0, corr, guess)
    
    def hess(self, n_samp=5000):
        """
        Monte carlo estimate of the likelihood Hessian 
        """
        
        H = np.zeros((2,2))
        for th in self.sample(n_samp):
            p = self.pcorr(th)
            foo = (np.cos(th) - spc.i1(self.k))
            Hij = p*foo
            Hjj = self.pic*p*(foo**2 + 0.5*(spc.i0(self.k) + spc.iv(2,self.k)))
            
            H += np.array([[0,Hij],[Hij,Hjj]])/n_samp
        
        return H
    
    def pcorr(self, err):
        return np.exp(self.k*np.cos(err))/(2*np.pi*spc.i0(self.k))
    
    def p(self, err):
        return self.pic*self.pcorr(err) + self.pig/(2*np.pi)
    
    def ratio(self, x, R=1):
        return spc.i1(x)/spc.i0(x) - R

## function for fitting the kernel model


def vmpdf(err, k):
    return np.exp(k*np.cos(err))/(2*np.pi*spc.i0(k))

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

def kuiper_two(data1, data2):
    """Compute the Kuiper statistic to compare two samples.

    Parameters
    ----------
    data1 : array-like
        The first set of data values.
    data2 : array-like
        The second set of data values.
    
    Returns
    -------
    D : float
        The raw test statistic.
    fpp : float
        The probability of obtaining two samples this different from
        the same distribution.

    Notes
    -----
    Warning: the fpp is quite approximate, especially for small samples.

    """
    data1, data2 = np.sort(data1), np.sort(data2)

    if len(data2)<len(data1):
        data1, data2 = data2, data1

    cdfv1 = np.searchsorted(data2, data1)/float(len(data2)) # this could be more efficient
    cdfv2 = np.searchsorted(data1, data2)/float(len(data1)) # this could be more efficient
    D = (np.amax(cdfv1-np.arange(len(data1))/float(len(data1))) + 
            np.amax(cdfv2-np.arange(len(data2))/float(len(data2))))

    Ne = len(data1)*len(data2)/float(len(data1)+len(data2))
    return D, kuiper_FPP(D, Ne)    

def kuiper_FPP(D,N):
    """Compute the false positive probability for the Kuiper statistic.

    Uses the set of four formulas described in Paltani 2004; they report 
    the resulting function never underestimates the false positive probability 
    but can be a bit high in the N=40..50 range. (They quote a factor 1.5 at 
    the 1e-7 level.

    Parameters
    ----------
    D : float
        The Kuiper test score.
    N : float
        The effective sample size.

    Returns
    -------
    fpp : float
        The probability of a score this large arising from the null hypothesis.

    Reference
    ---------
    Paltani, S., "Searching for periods in X-ray observations using 
    Kuiper's test. Application to the ROSAT PSPC archive", Astronomy and
    Astrophysics, v.240, p.789-790, 2004.

    """
    if D<0. or D>2.:
        raise ValueError("Must have 0<=D<=2 by definition of the Kuiper test")

    if D<2./N:
        return 1. - sp.factorial(N)*(D-1./N)**(N-1)
    elif D<3./N:
        k = -(N*D-1.)/2.
        r = np.sqrt(k**2 - (N*D-2.)/2.)
        a, b = -k+r, -k-r
        return 1. - sp.factorial(N-1)*(b**(N-1.)*(1.-a)-a**(N-1.)*(1.-b))/float(N)**(N-2)*(b-a)
    elif (D>0.5 and N%2==0) or (D>(N-1.)/(2.*N) and N%2==1):
        def T(t):
            y = D+t/float(N)
            return y**(t-3)*(y**3*N-y**2*t*(3.-2./N)/N-t*(t-1)*(t-2)/float(N)**2)
        s = 0.
        # NOTE: the upper limit of this sum is taken from Stephens 1965
        for t in range(int(np.floor(N*(1-D)))+1):
            term = T(t)*spc.comb(N,t)*(1-D-t/float(N))**(N-t-1)
            s += term
        return s
    else:
        z = D*np.sqrt(N) 
        S1 = 0.
        term_eps = 1e-12
        abs_eps = 1e-100
        for m in itertools.count(1):
            T1 = 2.*(4.*m**2*z**2-1.)*np.exp(-2.*m**2*z**2)
            so = S1
            S1 += T1
            if abs(S1-so)/(abs(S1)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        S2 = 0.
        for m in itertools.count(1):
            T2 = m**2*(4.*m**2*z**2-3.)*np.exp(-2*m**2*z**2)
            so = S2
            S2 += T2
            if abs(S2-so)/(abs(S2)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        return S1 - 8*D/(3.*np.sqrt(N))*S2

#%%

# n_neur = 100
# n_col = 1000
# draws = 50

# betas = np.linspace(0.01, 2.0, 20)
# # betas = np.array([0.1, 0.5, 1.0, 2.0])
# kaps = 1/np.linspace(0.01, 5, 10)
# # kaps = 1/np.linspace(0.01, 5, 40)
# # kaps = np.array([0.1, 1, 10, 100])

# cols = (2*np.random.rand(n_col)-1)*np.pi

# K = np.zeros((len(kaps), len(betas)))
# Pi = np.zeros((len(kaps), len(betas)))
# # logL = np.zeros((len(kaps), len(betas)))
# KS = np.zeros((len(kaps), len(betas)))
# FP = np.zeros((len(kaps), len(betas)))


# for i,kap in tqdm(enumerate(kaps)):
#     pop = RadialBoyfriend(1/kap)
    
#     for j,beta in enumerate(betas):
        
#         # pop = VMM(k=kap, pic=)
        
#         k = []
#         p = []
#         l = []
#         f = []
#         for it in range(draws):
            
#             X = pop.sample(cols, n_neur)
            
#             noise = np.random.randn(*X.shape)*beta
#             idx = np.argmax((X+noise)@X.T, axis=1)
#             err = util.circ_err(cols,cols[idx])
            
#             vmm = VMM(k=1,pic=0.1, kmax=300)
#             lik = vmm.fit(err, iters=50)
#             samps = vmm.sample(n_col)
            
#             k.append(vmm.k)
#             p.append(vmm.pic)
#             # l.append(lik[-1])
#             kt = kuiper_two(err, samps)
#             l.append(kt[0])
#             f.append(kt[1])
        
#         K[i,j] = np.mean(k)
#         Pi[i,j] = np.mean(p)
#         KS[i,j] = np.mean(l)
#         FP[i,j] = np.mean(f)

#%%

n_col = 1000
draws = 50

# betas = np.array([0.1, 0.5, 1.0, 2.0])
# kaps = 1/np.linspace(0.01, 5, 10)
# kaps = 1/np.linspace(0.01, 5, 40)
# kaps = np.array([0.1, 1, 10, 100])
# kaps = np.linspace(0, 10, 10)
kaps = np.array([0, 1, 5])
# betas = np.array([1])
betas = np.array([0.1, 0.5, 1])

# cols = (2*np.random.rand(n_col)-1)*np.pi
cols = np.linspace(-np.pi, np.pi, 100)

K = np.zeros((draws,len(kaps), len(betas)))
Pi = np.zeros((draws,len(kaps), len(betas)))
# logL = np.zeros((len(kaps), len(betas)))
# KS = np.zeros((len(kaps), len(betas)))
# FP = np.zeros((len(kaps), len(betas)))


for i,kap in tqdm(enumerate(kaps)):
    for j, beta in enumerate(betas):
        pop = VMM(k=kap, pic=beta )
        
        for it in range(draws):
            
            err = pop.sample(n_col)
            
            vmm = VMM(k=1,pic=0.1, kmax=300)
            lik = vmm.fit(err, iters=50)
            samps = vmm.sample(n_col)
            
            K[it,i,j] = vmm.k
            Pi[it,i,j] = vmm.pic

cols = cm.Dark2(np.arange(len(kaps)))
for i, kap in enumerate(kaps):
    for j,beta in enumerate(betas):
        plt.scatter(K[:,i,j], Pi[:,i,j], c=cols[i])
        plt.scatter(kap, beta, s=100, marker='*', c=cols[i])

#%%

plt.subplot(1,2,1)
for j,beta in enumerate(betas):
    plt.scatter(kaps, K[:,j])
plt.plot(plt.xlim(), plt.ylim(), 'k--')
tpl.square_axis()

plt.subplot(1,2,2)
for j,beta in enumerate(betas):
    plt.scatter(kaps, Pi[:,j])
    plt.plot([kaps.min(), kaps.max()], [beta, beta], '--')


#%%
c = np.linspace(-np.pi, np.pi, 100)

pop = RadialBoyfriend(5)

cols = (2*np.random.rand(1000)-1)*np.pi

X = pop.sample(cols, 1000)

ovlp = X@(X+np.random.randn(*X.shape)*5).T

idx = np.argmax(ovlp, axis=0)

err = util.circ_err(cols, cols[idx])

plt.hist(err, density=True, bins=20, histtype='step', linewidth=2)


#%%

n_col = 300
n_err = 100
n_neur = 5000

samp_every = 3

n_pop = 1

beta = 5

dx = (2*np.pi/n_err)

colors = np.linspace(-np.pi, np.pi, n_col) # for the samples
errors = np.linspace(-np.pi, np.pi, n_err) # for the theory

clf = knr(n_neighbors=1)

# mus = np.linspace(0, np.pi/2, 100)
# mus = [1]
betas = np.linspace(1,10,5)
this_mu = 0
sigs = np.linspace(0.5, 2, 10)

err_std = []
samp_err_std = []
tcc_err_std = []
err_kur = []
tcc_err_kur = []
samp_err_kur = []
# for i,this_mu in enumerate(mus):
for i,this_beta in enumerate(betas):
    stds = []
    samp_stds = []
    tcc_stds = []
    # tst = []
    kur = []
    tcc_kur = []
    samp_kur = []
    for this_sig in sigs:
        
        # pop = MisesLogNormal(this_mu, this_sig)
        pop = RadialBoyfriend(this_sig)
        
        ## theory
        kernel = pop(errors, quantile=1e-3) # numerically integrate
        # kernel /= np.sqrt(np.sum(dx*kernel**2))
        # kernel /= np.max(kernel)
        # kernel -= kernel.mean()
        apprx = np.exp(this_beta*kernel)/np.sum(np.exp(this_beta*kernel)*dx)
        var = np.sum(dx*apprx*errors**2)
        stds.append(var)
        kur.append(np.sum(dx*apprx*errors**4)/var**2 - 3)
        
        # if np.mod(i, samp_every) <=0:
        ## decoder
        X = pop.sample(colors, size=n_neur)
        # X /= la.norm(X, axis=1,keepdims=True) # normalize response across neurons
        
        # K = util.center(X@X.T)
        K = X@X.T/X.shape[1]
        K_perturb = K + np.random.gumbel(scale=1/this_beta, size=K.shape)
        tcc_pred = K_perturb.argmax(1)
        # tcc_err = util.circ_distance(colors, colors[tcc_pred])
        tcc_err = np.arctan2(np.sin(colors-colors[tcc_pred]), np.cos(colors-colors[tcc_pred])) 
        tcc_stds.append(np.var(tcc_err))
        tcc_kur.append(sts.kurtosis(tcc_err))
        
        # X_tst = X + np.random.randn(n_col, n_neur)/np.sqrt(2*beta)
        X_tst = X + np.random.randn(n_col, n_neur)*np.sqrt(1/6)*(np.pi/this_beta)
        
        clf.fit(X, colors)
        pred = clf.predict(X_tst)
        err = np.arctan2(np.sin(colors-pred), np.cos(colors-pred))
        # err = util.circ_distance(pred, colors)
        
        samp_stds.append(np.var(err))
        samp_kur.append(sts.kurtosis(err))
    
    err_std.append(stds)
        # apprx *= (n_col/2*np.pi)
    err_kur.append(kur)
    
    # if np.mod(i, samp_every) <=0:
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
samps = tcc_err_std 
thry = err_std
# thry = err_kur
# samps = tcc_err_kur
# samps = samp_err_kur

cols = cm.copper(np.linspace(0,1,len(betas)))
# for i in range(len(sigs)):
#     plt.plot(mus, thry[:,i]/(2*np.pi), color=cols[i], linestyle='--')
#     plt.plot(mus[::samp_every], samps[:,i]/(2*np.pi), color=cols[i])
for i in range(len(betas)):
    plt.plot(sigs, thry[i]/(2*np.pi), color=cols[i], linestyle='--')
    plt.scatter(sigs, samps[i]/(2*np.pi), color=cols[i])

