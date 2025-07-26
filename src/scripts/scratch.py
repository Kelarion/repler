CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/concepts/sbmf/')

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
import cvxpy as cvx

# my code
import util
import df_util
import pt_util
import bae
import bae_models
import bae_search
import plotting as tpl

#%%
# from sparse_factorization import *
from factorization import *

Ssp = sprs.csr_array(df_util.randtree_feats(2**6, 2, 4))
S = Ssp.todense()
X_ = util.pca_reduce(Ssp.todense().T, thrs=1)
X = df_util.noisyembed(X_, 2*X_.shape[1], 30, nonneg=False)
# X = df_util.noisyembed(S, 2*S.shape[1], 100, nonneg=True, orth=True)

mod = KernelBMF(S.shape[1], sparse_reg=0, tree_reg=0)
mod.initialize(X)
mod.temp = 1e-7

inds, ptr = mod.S.to_csr()
Sest = sprs.csr_array((np.ones(len(inds)), inds, ptr), 
                      shape=(mod.n, mod.dim_hid)).todense()

mod2 = bae_models.KernelBMF(S.shape[1], sparse_reg=0, tree_reg=0)
# mod2 = bae_models.SemiBMF(S.shape[1], nonneg=True, weight_reg=1e-2, tree_reg=1e-2)
mod2.initialize(X, S0=Sest)
# mod2.temp = 1e-7

# mod = SparseKernelBMF(S.shape[1], tree_reg=1e-2)
# mod = KernelBMF(S.shape[1], tree_reg=1e-2)
# mod2 = bae_models.KernelBMF(S.shape[1], tree_reg=1e-2)
#%%

import search

Ssp = sprs.csr_array(df_util.randtree_feats(2**6, 2, 4))
X_ = util.pca_reduce(Ssp.todense().T, thrs=1)
X = df_util.noisyembed(X_, 2*X_.shape[1], 30, nonneg=False)
X = X-X.mean(0)

n,m = Ssp.shape
StS = (Ssp.T@Ssp).todense()
StX = Ssp.T@X

S = search.BiMat(Ssp.indices, Ssp.indptr, m)
sampler = search.KernelSampler(StS/n, StX/n, 1/n)

C1 = sampler.sample(S, X, 1, 1e-7, 0, 0)
C2 = bae_search.kerbmf(X, Ssp.todense(), 
                       StX=1*StX, StS=1*StS, scl=1, N=n, 
                       beta=0, alpha=0, temp=1e-7)
# C3 = bae_search.oldkerbmf(X, Ssp.todense(), 
#                        StX=S.T@X, StS=S.T@S, scl=mod2.scl, N=mod2.n, 
#                        beta=mod2.tree_reg, temp=mod2.temp)
# C2 = mod2.EStep()
plt.scatter(C1.flatten(), C2.flatten(), c=np.arange(np.prod(C1.shape)))

#%%
from spbae import KernelSampler, BiMat

bs = 2**9
lr = 0.0

Ssp = sprs.csr_array(df_util.randtree_feats(2**9, 2, 4))
inds = Ssp.indices
ptr = Ssp.indptr

X = util.pca_reduce(Ssp.todense().T, thrs=1)
StS = (Ssp.T@Ssp).todense() / Ssp.shape[0]
StX = Ssp.T@(X) / Ssp.shape[0]

Sbin = BiMat(Ssp.indices, Ssp.indptr, Ssp.shape[1])
sampler = KernelSampler(StS, StX, bs, lr)

#%%

S = df_util.allpaths(df_util.randtree_feats(8,2,4))[1]
# S = df_util.cyclecats(7)
# S = df_util.gridcats(4, 3)

k = S.shape[1]
Sall = util.F2(k)
is_attr = df_util.minham(S.T, Sall.T) == 0

I = 1.0*df_util.inc(S)

R = -I.T@I
R /= (-R[0,0]/2)

r = -R@S.mean(0)

A = R+np.eye(k)
b = 2*r-1

dA = []
for s in Sall[is_attr]:
    # samps = df_util.walk((R+np.eye(k)), 2*r-1, s)
    # samps = df_util.walk(R, -r, s)
    samps = df_util.walk(A*(1-np.eye(k)), (b+np.diag(A))/2, s, temp=1e-12)
    
    dH = k - (2*samps-1)@(2*S-1).T
    
    dA.append(dH.min(1))
    
    plt.plot(dA[-1])
    plt.xticks(ticks=np.arange(len(samps)-1,step=k),labels=np.arange(len(samps)//k))


#%%

deez = (2**np.linspace(9, 14, 20)).astype(int)
n = 10

tdense = []
tsparse = [] 
tnump = []
ndense = []
nsparse = []

for d in tqdm(deez):
    
    td = []
    ts = []
    tn = []
    for i in range(n):
        J = 1.0*np.random.choice([-1,0,1], size=(d,d), p=[0.1,0.8,0.1])
        J = np.triu(J,1) + np.triu(J,1).T
        h = -10*np.ones(d)
        
        S = np.random.choice([0,1], size=d, p=[0.95, 0.05])
        Slist = FiniteSet(np.where(S)[0].astype(int), d)
        # Slist = set(list(np.where(S)[0].astype(int)))
        
        t0 = time()
        wa = greedy(J, h, S, 1e-3)
        td.append(time() - t0)
        
        t0 = time()
        ba = spgreedy(J, h, Slist, 1e-3)
        ts.append(time() - t0)
        
        Jspr = sprs.csr_array(J)
        t0 = time()
        ba = Jspr@S + h
        tn.append(time() - t0)
    
    tdense.append(np.mean(td))
    tsparse.append(np.mean(ts))
    tnump.append(np.mean(tn))

#%%

S = df_util.randtree_feats(2**9, 2, 4)
X = util.pca_reduce(S.T, thrs=1)

StS = S.T@S/len(S)
StX = S.T@X/len(S)

Ssp = sprs.csr_array(S)
Sbin = BinaryMatrix(Ssp.indices, Ssp.indptr, S.shape[1])

#%%
t0=time()
ba = bae_search.kerbae(X, 1*S, StX, StS, 1, 1e-4)
print(time()-t0)

t0=time()
wa = spkerbmf(, X, StX, StS, 1, 1e-4)
print(time()-t0)


#%%

N = 2**np.arange(3, 9)
betas = [1e-1, 1]
draws = 20

neal = bae_util.Neal(0.95, period=4)
ham = []
tree = []
nbs = []
for n in N:
    disham = []
    distree = []
    disnbs = []
    for beta in tqdm(betas):
        
        wa = []
        ba = []
        ga = []
        for draw in range(draws):
        
            S = df_util.randtree_feats(n, 2,4)
            X = df_util.noisyembed(S, S.shape[1], 30, scl=1e-4)
            
            mod = bae_models.KernelBMF(2*S.shape[1], tree_reg=beta)
            # mod = bae_models.BiPCA(S.shape[1], tree_reg=beta, sparse_reg=0)
            en = neal.fit(mod, X, verbose=False, T_min=1e-3)
            
            wa.append(df_util.permham(S, mod.S, norm=True).mean())
            ba.append(df_util.treecorr(mod.S).mean())
            ga.append(util.nbs(mod.S, X))
        
        disham.append(np.mean(wa, 0))
        distree.append(np.mean(ba, 0))
        disnbs.append(np.mean(ga, axis=0))
    
    ham.append(disham)
    tree.append(distree)
    nbs.append(disnbs)

#%%

nepoch = 400
draws = 12

loss = []
nbs = []
ham = []
pham = []
tree = []

for _ in tqdm(range(draws)):
    
    # S = df_util.randtree_feats(64, 2, 4)
    S = df_util.schurcats(64, 0.5)
    depth = df_util.porder(S)
    W = sts.ortho_group(S.shape[1]).rvs()[:,:S.shape[1]]    
    X = S@W.T
    
    # mod = bae_models.BiPCA(S.shape[1], tree_reg=0)
    mod = bae_models.KernelBMF(S.shape[1], tree_reg=0)
    
    # dl = pt_util.batch_data(torch.FloatTensor(X), batch_size=len(S))
        
    # mod = bae_models.BernVAE(S.shape[1], X.shape[1], beta=0, weight_reg=1e-2)
    # mod.initialize(dl)
    
    ls = []
    br = []
    hm = []
    ph = []
    tr = []
    # for _ in tqdm(range(nepoch)):
    #     ls.append(mod.grad_step(dl))
    #     Sest = mod.hidden(dl.dataset.tensors[0]).detach().numpy()
    #     br.append(util.nbs(Sest, X))
    #     hm.append(df_util.permham(S, Sest).mean())
    mod.initialize(X)
    for it in (range(nepoch)):
        br.append(util.nbs(mod.S, X))
        tr.append(df_util.treecorr(mod.S).mean())
        
        minham = df_util.minham(S, mod.S, sym=True)
        hm.append(util.group_mean(minham/S.sum(0), depth))
        ph.append(df_util.permham(S, mod.S, norm=False).mean())
        
        T = 5*(0.95**(it//4))
        mod.temp = T
        # mod.temp = 1e-4
        ls.append(mod.grad_step(X))
            
    loss.append(ls)
    nbs.append(br)
    ham.append(np.array(hm))
    tree.append(tr)
    pham.append(ph)

allham = util.pad_to_dense(ham)[...,:-1]

#%%
# cmap = cm.viridis
cmap = cm.spring

num_ham = allham.shape[-1]
norm = plt.Normalize(1, num_ham)

# cols = cm.viridis(np.arange(num_ham))
for i, thisham in enumerate(np.nanmean(allham,0).T):
    # plt.plot(thisham, c=cols[i])
    plt.plot(thisham, c=cmap(norm(i)))

#%%

S = df_util.randtree_feats(16, 2,4)

R = df_util.dH(S)[0]
aye,jay = np.where(np.triu(R) == 1)

depth = df_util.porder(S)

#%%

# dim_emb = np.arange(16, 220, 20)
dim_emb = [200]
# batch_size = np.append(1, np.arange(8,136,8)).astype(int)
batch_size = [128]
draws = 5

# S = df_util.schurcats(2**7, 0.5)

# neal = bae_util.Neal(0.9, period=20)
ham = []
nbs = []
for h in tqdm(dim_emb):
    disham = []
    distree = []
    disnbs = []
    for bsz in batch_size:
        
        neal = bae_util.Neal(0.9, period=int(10 + 50*bsz/128))
        
        wa = []
        ga = []
        for draw in range(draws):
        
            W = sts.ortho_group(h).rvs()[:,:16]    
            X = S@W.T
            
            if bsz == 1:
                dl = pt_util.batch_data(torch.FloatTensor(X))
            else:
                dl = pt_util.batch_data(torch.FloatTensor(X), batch_size=bsz)
                
            mod = bae_models.BinaryAutoencoder(S.shape[1], X.shape[1], weight_reg=0.1)
            en = neal.fit(mod, dl, T_min=1e-4, verbose=False)
            
            Sest = mod.hidden(dl.dataset.tensors[0]).detach().numpy()
            
            wa.append(df_util.permham(S, Sest, norm=True).mean())
            ga.append(util.nbs(Sest, X))
        
        disham.append(wa)
        disnbs.append(ga)
    
    ham.append(disham)
    nbs.append(disnbs)

#%%

nodes = np.arange(len(S))
P, npath = df_util.isometrichull(G, nodes )

npath = npath[:,None]

ij = [(aye==i)|(jay==i) for i in range(M)]

A = P[:,aye==jay]

removed = []
while np.min(npath) > 0:
    
    thisone = np.argmin(A.sum(0))
    
    if np.min(npath-A[:,[thisone]]) > 0:
        removed.append(thisone)
        npath -= A[:,[thisone]]
        A -= P[:,ij[thisone]]
    else:
        break

#%%

dims = list(range(10,110,10))
draws = 50
gap = 1.5   # max = gap*(top eigenvalue)

algo = util.imf
# algo = util.hopcut

err = np.zeros((draws, len(dims)))
ham = np.zeros((draws, len(dims)))
runtime = np.zeros((draws, len(dims)))
monotone = np.zeros((draws, len(dims)))
for j,dim in enumerate(dims):
    for i in tqdm(range(draws)):
        
        smax = np.random.choice([0,1], dim)
        
        ## Scipy won't let you sample from a Wishart with singular scale
        sst = util.center(np.outer(smax-smax.mean(), smax-smax.mean()))
        Q = np.eye(dim) - 1/dim - sst/np.sqrt(np.sum(sst**2))
        X = np.random.multivariate_normal(np.zeros(dim), Q, size=dim)
        K = X.T@X/dim
       
        ## Set the desired vector to be top eigenvector
        l,V = la.eigh(K)
        f_max = gap*l.max()
        K += f_max*sst/np.sqrt(np.sum(sst**2))
        
        t0 = time()
        s, ls = algo(-K)
        runtime[i,j] = time()-t0
        
        err[i,j] = (s@K@s)/(smax@K@smax)
        ham[i,j] = dim - np.abs((2*s,s-1)@(2*smax-1))
    
        monotone[i,j] = np.max()
    
    
#%%

A = -K
n = len(A)

this_A = 1*A
this_A[np.arange(n), np.arange(n)] = 0

this_b = np.diag(A)

Am = this_A*(this_A<0)  # negative component
Ap = -this_A*(this_A>0)  # positive component

s = 1*s_max

sig = np.random.permutation(np.where(s)[0])
sig = np.concatenate([sig, np.random.permutation(np.where(1-s)[0])])

Aps = Ap[sig,:][:,sig]

v = np.diag(Aps) + np.triu(Aps, 1).sum(0) + np.tril(Aps, -1).sum(1)
v = v[np.argsort(sig)]


#%%

@dataclass
class Bernardi(sxp.Task):

    num_trials: int
    trials_per_context: int
    samps: int

    def sample(self):

        num_ctx = self.num_trials//self.trials_per_context

        A1 = np.array([1,0,1,0]) # action
        R1 = np.array([1,1,0,0]) # reward

        A = np.concatenate([A1, np.roll(A1,1)])
        R = np.concatenate([R1, np.roll(R1,1)])

        X = []
        Y = []
        for i in range(self.samps):

            s = np.random.choice(range(4), self.num_trials)
            c = np.tile(np.repeat([0,1], self.trials_per_context), num_ctx//2)

            a = A[s+4*c] + 8
            r = R[s+4*c] + 10

            stim = np.array([s,a,r]).T.flatten()
            X.append(stim)
            Y.append([c,a,r])
        
        return {'X':X, 'Y':Y}


#%%

neal = bae_util.Neal(decay_rate=0.98, period=2, initial=1)

cntr_fit = []
aff_fit = []
cntr_nbs = []
aff_nbs = []
for _ in tqdm(range(100)):
    
    Strue = df_util.randtree_feats(16, 2, 4) 
    X = df_util.noisyembed(Strue, 100, 30, scl=1e-3)

    mod = bae_models.BiPCA(Strue.shape[1], center=False, tree_reg=1)
    en = neal.fit(mod, X, verbose=False)
    aff_fit.append(df_util.permham(Strue, mod.S))
    aff_nbs.append(util.nbs(Strue, mod.S))
    
    mod = bae_models.BiPCA(Strue.shape[1], center=True, tree_reg=1)
    en = neal.fit(mod, X, verbose=False)
    cntr_fit.append(df_util.permham(Strue, mod.S))
    cntr_nbs.append(util.nbs(Strue, mod.S))
    
#%%

samps = 5000

theta = np.pi/6
phi = np.pi/2

delta = np.random.randn(samps)*0.4
x = np.array([np.cos(theta), np.sin(theta)])

eps = x[:,None] + delta*np.array([[np.cos(phi)], [np.sin(phi)]])
xhat = eps/np.sqrt(np.sum(eps**2, axis=0))
thhat = np.arctan2(xhat[1], xhat[0])

deez = np.random.choice(range(samps), np.min([samps, 500]), replace=False)

circ = np.linspace(-np.pi, np.pi, 100)
plt.plot(np.sin(circ), np.cos(circ))
plt.scatter(xhat[0][deez], xhat[1][deez])
plt.scatter(x[0], x[1],  s=500, marker='*')
plt.scatter(np.cos(phi), np.sin(phi))
tpl.square_axis()

#%%

theta = 0
# noise_angle = np.pi/2
noise_angle = np.pi/4

kappa = 0

x = np.array([np.cos(theta),np.sin(theta)])
eps = np.array([np.cos(noise_angle),np.sin(noise_angle)])

these_rho = np.linspace(-10,10,100)

delts = []
xhats = []
err = []
for rho in these_rho:
    delt = x+rho*eps
    xhat = delt/la.norm(delt)
    thhat = np.arctan2(xhat[1], xhat[0])
    # err.append(np.arccos(np.cos(thhat-theta))**2)
    err.append(thhat**2)
    xhats.append(xhat)
    delts.append(delt)
xhats = np.array(xhats)
delts = np.array(delts)
err = np.array(err)


plt.subplot(1,2,1)
circ = np.linspace(-np.pi, np.pi, 100)
plt.plot(np.sin(circ), np.cos(circ))
plt.scatter(x[0], x[1], s=500, marker='*', zorder=10)
plt.scatter(xhats[:,0],xhats[:,1])
plt.plot(x[0] + np.linspace(-2,2,100)*eps[0], x[1]+ np.linspace(-2,2,100)*eps[1], 'k--')

plt.subplot(1,2,2)
plt.plot(these_rho, err)
plt.plot(these_rho, np.arctan(these_rho*np.sin(noise_angle)/(1+these_rho*np.cos(noise_angle)))**2)

#%%
from scipy.optimize import root_scalar

def Phi(x):
    return 0.5*(1+spc.erf(x/np.sqrt(2)))

def phi(x):
    return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)

samps = 5000
std = 0.5

theta = 0

x = np.array([np.cos(theta), np.sin(theta)])

eps = x[:,None] + np.random.randn(2,samps)*std
xhat = eps/np.sqrt(np.sum(eps**2, axis=0))
thhat = np.arctan2(xhat[1], xhat[0])

deez = np.random.choice(range(samps), np.min([samps, 50]), replace=False)

circ = np.linspace(-np.pi, np.pi, 100)
plt.scatter(xhat[0][deez], xhat[1][deez], s=50, marker='o', color=(0.5,0.5,0.5))
plt.scatter(eps[0][deez], eps[1][deez], s=50, marker='.', color=(0.5,0.5,0.5))
plt.plot([eps[0][deez], xhat[0][deez]], [eps[1][deez], xhat[1][deez]], color=(0.5,0.5,0.5))
plt.plot(np.cos(circ), np.sin(circ), 'k-')
plt.scatter(x[0], x[1], s=200, marker='*', color='k', zorder=100)

plt.plot(np.cos(circ)*std + x[0], np.sin(circ)*std + x[1], 'k-')

tpl.square_axis()


#%%
plt.hist(thhat, bins=25, density=True, color=(0.5,0.5,0.5))

kap = 1/std
xi = (kap**2)/4
rho = np.sqrt(np.pi*xi/2)*np.exp(-xi)*(spc.i0(xi) + spc.i1(xi))
guess = np.exp(-0.5*kap**2)/(2*np.pi) 
corr = kap*np.cos(circ)*Phi(kap*np.cos(circ))/phi(kap*np.cos(circ))
# corr = kap*np.cos(circ)*Phi(kap*np.cos(circ))*np.exp(-0.5*(kap*np.sin(circ))**2)/np.sqrt(2*np.pi)

plt.plot(circ, guess*(1+corr), 'k', linewidth=2)

ratio = lambda x,r=1: spc.i1(x)/spc.i0(x) - r
sol = root_scalar(ratio, args=(rho,), bracket=(0,100))
vmkap = sol.root

plt.plot(circ, np.exp(vmkap*np.cos(circ))/(2*np.pi*spc.i0(vmkap)), 'k--', linewidth=2)

plt.legend(['Actual', 'Von Mises approximation'])

#%%

n = 20
m = 10

S = np.random.choice([0,1], (n,m))

St1 = S.sum(0)
StS = S.T@S

ABCD = np.stack([StS,
                 St1[:,None] - StS,
                 St1[None,:] - StS,
                 n - St1[None,:] - St1[:,None] + StS,
                 ]).transpose((1,2,0))

deez = np.zeros((n,m,m))
doze = np.zeros((n,m,m))
truth = np.zeros((n,m))
troof = np.zeros((n,m))
for i in range(n):
    d = np.array([S[i], 1-S[i], -S[i], -(1-S[i])]).T
    
    R,r = df_util.dH(S[util.rangediff(n, [i])])
    
    for j in range(m):
        # diff = 0
        for k in range(m):
            
            ## non-stacked version
            A = StS[j,k]/n - (S[i,j]*S[i,k])/n
            B = StS[j,j] - A - S[i,j] 
            C = StS[k,k] - A
            D = 1 - A - B - C

            if A < min(B,C-1,D):
                deez[i,j,k] += S[i,k]
            if B < min(A,C,D-1):
                deez[i,j,k] += (1 - S[i,k] )
            if C <= min(A,B,D):
                deez[i,j,k] -= S[i,k]
            if D <= min(A,B,C):
                deez[i,j,k] -= (1 - S[i,k])
            
            ## stacked version
            doze[i,j,k] += np.min(ABCD[j,k] + d[k]*(1-S[i,j])) 
            doze[i,j,k] -= np.min(ABCD[j,k] - d[k]*S[i,j]) 

        troof[i,j] = R[j]@S[i] + r[j]

        ## ground truth
        thisS = 1*S
        thisS[i,j] = 1
        truth[i,j] += df_util.treecorr(thisS).sum()/2
        thisS[i,j] = 0
        truth[i,j] -= df_util.treecorr(thisS).sum()/2
    
# plt.scatter(deez, doze)

#%%

bits = 6
beta = 1e-3

# S = util.F2(bits)
Strue = df_util.btree_feats(bits).T
W = sts.ortho_group.rvs(Strue.shape[1])
X = (Strue-Strue.mean(0))@W.T
b = -Strue.mean(0)@W.T

S = (Strue + np.random.choice([0,1], Strue.shape, p=[0.9,0.1]))%2
N = len(S)

# ba = bae_models.bmf(X-b, 1.0*S, W, 1.0*(S.T@S), N=len(S), temp=1e-6, beta=beta)
# ba = bae_models.update_concepts_asym((X-b)@W, 1.0*S, scl=1, beta=beta, 
#                                      temp=1e-6, STS=S.T@S, N=len(S))
# wa = bae_search.sbmf(XW=(X-b)@W, S=1.0*S, WtW=W.T@W, 
#                      StS=1.0*(S.T@S), N=len(S), temp=1e-6, beta=beta)
ba = bae_models.update_concepts_kernel(X=X-X.mean(0), S=1.0*S, scl=1.0,
                                       beta=beta, temp=1e-6)
# wa = bae_search.bpca(XW=(X-b)@W, S=1.0*S, scl=1,
#                      StS=1.0*(S.T@S), N=len(S), temp=1e-6, beta=beta)
wa = bae_search.kerbmf(X=X-X.mean(0), S=1.0*S, scl=1.0, StX=1.0*(S.T@(X-X.mean(0))),
                       StS=1.0*(S.T@S), N=len(S), beta=beta, temp=1e-6)

t0 = time()
# wa = bae_models.bmf(X-b, 1.0*S, W, 1.0*(S.T@S), N=len(S), temp=1e-6, beta=beta)
# wa = bae_search.sbmf(XW=(X-b)@W, S=1.0*S, WtW=W.T@W, 
#                      StS=1.0*(S.T@S), N=len(S), temp=1e-6, beta=beta)
# ba = bae_models.update_concepts_asym((X-b)@W, 1.0*S, scl=1, beta=beta, 
#                                      temp=1e-6, STS=S.T@S, N=len(S))
# wa = bae_search.bpca(XW=(X-b)@W, S=1.0*S, scl=1,
#                      StS=1.0*(S.T@S), N=len(S), temp=1e-6, beta=beta)
# wa = bae_search.bpca(XW=(X-b)@W, S=1.0*S, scl=1, temp=1e-6, beta=beta)
wa = bae_search.kerbmf(X=X-X.mean(0), S=1.0*S, scl=1.0, StX=1.0*(S.T@(X-X.mean(0))),
                       StS=1.0*(S.T@S), N=len(S), beta=beta, temp=1e-6)
# ba = bae_models.update_concepts_kernel(X=X-X.mean(0), S=1.0*S, scl=1.0,
#                                        beta=beta, temp=1e-6)
# C = 2*(X-b)@W - 1
# newS = 1*(np.random.rand(*S.shape) > spc.expit(C/1e-6))
print(time() - t0)

#%%

n,m = S.shape
StS = S.T@S

deez = np.zeros((n,m))
truth = np.zeros((n,m))
troof = np.zeros((n,m))
for i in tqdm(range(n)):
    d = np.array([S[i], 1-S[i], -S[i], -(1-S[i])]).T
    
    R,r = df_util.dH(S[util.rangediff(n, [i])])
    
    for j in range(m):
        # diff = 0
        inhib = 0
        for k in range(m):
            
            ## non-stacked version
            A = StS[j,k] - S[i,j]*S[i,k]
            B = StS[j,j] - A - S[i,j] 
            C = StS[k,k] - A
            D = n - A - B - C
            
            if A < min(B,C-1,D):
                inhib += S[i,k]
            if B < min(A,C,D-1):
                inhib += (1 - S[i,k])
            if C <= min(A,B,D):
                inhib -= S[i,k]
            if D <= min(A,B,C):
                inhib -= (1 - S[i,k])
                
        deez[i,j] = inhib
            
        troof[i,j] = R[j]@S[i] + r[j]

        ## ground truth
        thisS = 1*S
        thisS[i,j] = 1
        truth[i,j] += df_util.treecorr(thisS).sum()/2
        thisS[i,j] = 0
        truth[i,j] -= df_util.treecorr(thisS).sum()/2

#%%

n,m = S.shape
StS = S.T@S

deez = np.zeros((n,m))
truth = np.zeros((n,m))
troof = np.zeros((n,m))
for i in tqdm(range(n)):
    d = np.array([S[i], 1-S[i], -S[i], -(1-S[i])]).T
    
    R,r = df_util.dH(S[util.rangediff(n, [i])], sym=False)
    
    for j in range(m):
        # diff = 0
        inhib = 0
        for k in range(m):
            
            ## non-stacked version
            A = StS[j,k] - S[i,j]*S[i,k]
            B = StS[j,j] - A - S[i,j] 
            C = StS[k,k] - A
            
            if A < min(B,C-1):
                inhib += S[i,k]
            if B < min(A,C):
                inhib += (1 - S[i,k])
            if C <= min(A,B):
                inhib -= S[i,k]
                
        deez[i,j] = inhib
            
        troof[i,j] = R[j]@S[i] + r[j]

        ## ground truth
        thisS = 1*S
        thisS[i,j] = 1
        truth[i,j] += df_util.treecorr(thisS, sym=False).sum()/2
        thisS[i,j] = 0
        truth[i,j] -= df_util.treecorr(thisS, sym=False).sum()/2


#%%

n = 20
m = 10

F = np.random.choice([0,1], (n,m))
Fsum = F.sum(0)
# F = np.mod(F+(Fsum>n/2),2)
# Fsum = F.sum(0) #, keepdims=True)

# follows = (2*F.T@F > Fsum)

# P1 = follows|follows.T
# P2 = (2*F.T@F == Fsum)|(2*F.T@F == Fsum.T)

# Q = (P1)*(1-np.eye(m))
# Q = (1-2*P1)*(1-P2)
# Q = (2*P2 + 1*P1 - 1)

FF = F.T@F

# I1 = np.sign(2*FF - Fsum)
# I2 = np.sign(2*FF - Fsum.T)
# I3 = np.sign(Fsum - Fsum.T)
# I4 = np.sign(Fsum + Fsum.T - n)

# minab = np.min([Fsum*np.ones((m,1)), Fsum.T*np.ones((1,m))],0)

# Q = (1*(2*F.T@F < minab) - 1*(2*F.T@F > Fsum)*(Fsum < Fsum.T) - 1*(2*F.T@F > Fsum.T)*(Fsum > Fsum.T))
# Q = Q*(Fsum + Fsum.T < n)*(1-np.eye(m))
# # Q = (1*(2*F.T@F < minab) - 1*(2*F.T@F > minab))*(1-np.eye(m))

# # p = (2*F.T@F > minab).sum(1)
# # p = 2*(2*F.T@F > Fsum.T).sum(1)
# p = 2*((2*F.T@F > Fsum.T)*(Fsum > Fsum.T)).sum(1)

D1 = FF
D2 = Fsum[None,:] - FF
D3 = Fsum[:,None] - FF
D4 = n - Fsum[None,:] - Fsum[:,None] + FF

deez = np.array([D1,D2,D3,D4])
best = 1*(deez == deez.min(0))
best *= (best.sum(0)==1)

# Q = best[0] - best[1] - best[2] + best[3]
# p = 2*(best[1].sum(0) - best[3].sum(0))

R,r = df_util.dH(F)
Q = R
p = 2*r

# I1 = np.sign(2*FF - Fsum[None,:]) # si'sj - (1-si)'sj
# I2 = np.sign(2*FF - Fsum[:,None]) # si'sj - si'(1-sj)
# I3 = np.sign(Fsum[None,:] - Fsum[:,None]) # (1-si)'sj - si'(1-sj)
# I4 = np.sign(Fsum[None,:] + Fsum[:,None] - n) # si'sj - (1-si)'(1-sj)

# Q = (1*(I1<0)*(I2<0) - 1*(I1>0)*(I3<0) - 1*(I2>0)*(I3>0))*(I4<0)
# p = ((I2>0)*(I3>0)*(I4<0)).sum(1)

oldD = deez.min(0)

deez = []
doze = []
for f in util.F2(m):
    
    newF = np.vstack([F, f])
    
    doze.append(f@Q@f + f@p)
    
    # newD = np.min([newF.T@newF, (1-newF).T@newF, newF.T@(1-newF), (1-newF).T@(1-newF)],0)
    # deez.append(newD.sum() - oldD.sum())
    deez.append(df_util.treecorr(F).sum() - df_util.treecorr(newF).sum())
    
plt.scatter(deez, doze)

#%%
import pypoman as ppm
from tqdm import tqdm
from itertools import combinations

d = 4

F2 = util.F2(d)
sols = []
for n in range(5,6):
    for idx in tqdm(combinations(range(2**d),n)):
        
        ids = np.array(idx)
        K = util.center(F2[ids]@F2[ids].T)
        
        k = 2**(n-1)-1
        
        all_cats = util.F2(n, True)[1:]
        cut = util.outers(util.H(n)@all_cats.T).reshape((k,-1))
        
        A = np.vstack([cut.T, -cut.T, -np.eye(k)])
        b = np.concatenate([K.flatten(), -K.flatten(), np.zeros(k)])
        
        verts = ppm.compute_polytope_vertices(A, b)

        sols.append([np.diag(v[v>1e-6])@all_cats[v>1e-6] for v in verts])
        
#%%

def plot_graph(S):

    E,H = df_util.allpaths(S)
    
    G = nx.Graph()
    G.add_edges_from(E.T)
    
    nx.draw(G, node_size=100*(np.isin(G.nodes,range(5))))
    
#%% 

def stiefelSGD(X, S, lr=1e-2, s=2):
    
    W = sts.ortho_group.rvs(len(X.T))[:,:len(S.T)]
    scl = 1 
    b = np.zeros(len(X.T))
    
    ## Project gradient onto tangent space
    dW = -2*scl*(X-b).T@S
    Z = dW@W.T - 0.5*W@W.T@dW@W.T
    Z = Z - Z.T
    dW = Z@W
    
    ## Cayley transform
    Y = W + lr*dW
    for i in range(s):
        Y = W + (lr/2)*Z@(X + Y)
    
    return Y
    



    