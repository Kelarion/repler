CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)

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

import networkx as nx
import cvxpy as cvx

# my code
import util
import df_util
import bae
import bae_models
import bae_util
import bae_search
import plotting as tpl

#%%

# S = df_util.allpaths(df_util.randtree_feats(8,2,4))[1]
# S = df_util.cyclecats(7)
S = df_util.gridcats(4, 3)

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
for s in Sall[~is_attr]:
    # samps = df_util.walk((R+np.eye(k)), 2*r-1, s)
    # samps = df_util.walk(R, -r, s)
    samps = df_util.walk(A*(1-np.eye(k)), (b+np.diag(A))/2, s, temp=1e-12)
    
    dH = k - (2*samps-1)@(2*S-1).T
    
    dA.append(dH.min(1))
    
    plt.plot(dA[-1])
    plt.xticks(ticks=np.arange(len(samps)-1,step=k),labels=np.arange(len(samps)//k))
    

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
noise_angle = -np.pi/4

x = np.array([np.cos(theta),np.sin(theta)])
eps = np.array([np.cos(noise_angle),np.sin(noise_angle)])

these_rho = np.linspace(-10,10,100)

xhats = []
err = []
for rho in these_rho:
    xhat = (x+rho*eps)/la.norm(x+rho*eps)
    thhat = np.arctan2(xhat[1], xhat[0])
    # err.append(np.arccos(np.cos(thhat-theta))**2)
    err.append(thhat**2)
    xhats.append(xhat)
xhats = np.array(xhats)
err = np.array(err)

plt.subplot(1,2,1)
circ = np.linspace(-np.pi, np.pi, 100)
plt.plot(np.sin(circ), np.cos(circ))
plt.scatter(x[0], x[1], s=500, marker='*', zorder=10)
plt.scatter(xhats[:,0],xhats[:,1])
plt.plot(x[0] + np.linspace(-2,2,100)*eps[0], x[1]+ np.linspace(-2,2,100)*eps[1], 'k--')

plt.subplot(1,2,2)
plt.plot(these_rho,err)
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
            A = StS[j,k] - S[i,j]*S[i,k]
            B = StS[j,j] - A - S[i,j] 
            C = StS[k,k] - A
            D = n - A - B - C

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

for f,l in tqdm(zip(frms, lids[ids])):
    
    c = cgid[cogs['Form_ID'].tolist().index(f)]
    cogmat[l,c] = 1

#%%

these_langs = ['Latin',
               'Brazilian Portuguese',
               'Hindi',
               'Urdu',
               'Italian',
               'English',
               'Sinhala',
               'Hittite',
               'Gothic',
               'Kamviri',
               'Transalpine Gaulish',
               'Polish',
               'Bengali',
               'Eastern Pahari',
               'Scottish Gaelic',
               'Northern Welsh',
               # 'Slovenian',
               'Slovak',
               'Western Farsi',
               'Modern Greek',
               'Early Irish',
               'Icelandic',
               'Catalan',
               'Southern Kurdish',
               'Kumzari',
               'Elfdalian',
               # 'Danish',
               'Western Flemish',
               'German',
               'Takestani',
               # 'Hawraman-I Taxt',
               'Macedonian',
               'Bakhtiari',
               'Romanian',
               'Francoprovencalic',
               'Khwarezmian',
               'Ukrainian',
               'Eastern Armenian']


plt_these = np.isin(Gtree.nodes,range(160))
nx.draw(Gtree,pos=pos, node_size=10*plt_these, 
        node_color=cmap(grp[np.where(plt_these, Gtree.nodes, 0)]))


cmap = cm.tab10

for this_lang in these_langs:
    this = lan['Glottolog_Name'].tolist().index(this_lang)
    
    if pos[this][0] > p_cntr[0]:
        dx = 5
        ha = 'left'
    elif pos[this][0] < p_cntr[0]:
        dx = -5
        ha = 'right'
    else:
        dx = 1e-2
        ha = 'left'
    if pos[this][1] > p_cntr[1]:
        dy = 5
        va = 'bottom'
        # ha = 'right'
    if pos[this][1] < p_cntr[1]:
        dy = -5
        va = 'top'
        # ha = 'right'
    else:
        dy = 5e-2
        va = 'bottom'
    plt.scatter(pos[this][0], pos[this][1], color=cmap(grp[this]), zorder=10)
    plt.text(pos[this][0]+dx,pos[this][1]+dy, this_lang,
             horizontalalignment = ha,
             verticalalignment = va,
             color=cmap(grp[this]),
             bbox={'facecolor':'white', 
                   'edgecolor': cmap(grp[this]), 
                   'alpha': 0.8,
                   'boxstyle': 'round'})

dicplt.square_axis()

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
    



    