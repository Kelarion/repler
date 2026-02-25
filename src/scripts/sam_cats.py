
CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
from dataclasses import dataclass, fields, field
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn

from tqdm import tqdm
from itertools import permutations, combinations

import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt

import networkx as nx 
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.algorithms import isomorphism

import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.special as spc
from sklearn.svm import LinearSVC, SVC

from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from scipy.optimize import nnls

# my code
import students
import experiments as exp
import super_experiments as sxp
import util
import pt_util
import tasks
import plotting as tpl

import df_util
import bae
import bae_models
import bae_util

#%%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

#%%

def sensitivity(kernel, item, others):
    """
    How does the kernel model predicition for `item` depend on the label
    assigned to all other items in `others`?
    """
    pass



#%%

## to do: SVM generalization
##        ReLU generalization
##        COCO curation 

n = 3

## get list of symmetries (permutations and flips)
F = util.F2(n) # cube
syms = [] 
for perm in permutations(range(n)): # permutations
    for flip in F:                  # reflections
        bits = (F[:,perm]+flip)%2
        syms.append(bits@(2**np.arange(n)))
syms = np.array(syms)

H = 1 - 2*((F@F.T)%2) # hadamard matrix

## look over all possible functions combinations
punqs = []
funqs = []
fclass = []
for comb in tqdm(combinations(range(2**n), 2**(n-1))):
    
    f = np.zeros(2**n)
    f[list(comb)] = 1

    if len(funqs) == 0:
        funqs = f[None]

    ## find global equivalence classes (there should be 6 for n=3)
    d = df_util.minham(funqs.T, f[syms].T, sym=True)
    ix = np.argmin(d)
    if d[ix] == 0:
        fclass.append(ix)
    else:
        fclass.append(len(funqs))
        funqs = np.vstack([funqs, f])
        
fclass = np.array(fclass)

fhat = (2*funqs-1)@H    
order = F.sum(1)
punqs = ((fhat**2)@np.eye(n+1)[order] / (2**n))[:,1:]

idx = np.lexsort(np.fliplr(-punqs).T)

funqs = funqs[idx]
punqs = punqs[idx]

#%%

nums = ['I', 'II', 'III', 'IV', 'V', 'VI']
for i in range(len(funqs)):
    ax = plt.subplot(1, len(funqs), i+1, projection='3d')
    
    tpl.scatter3d(F, ax=ax, c=funqs[i], s=100, cmap='bwr')
    plt.title(nums[i])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

#%%

clf = SVC(kernel='precomputed')
# clf = KernelRidge(kernel='precomputed', alpha=1e-6)

# cols = cols = cm.ocean(np.arange(64))

works = []
sups = []
for i in range(len(funqs)):
    
    doo = []
    yhat = np.zeros((len(ki), 2))
    for this, kern in tqdm(enumerate(ki)):
        
        K = kern[D]
        
        # A = K[~trainset[f],:][:,trainset[f]]@la.inv(K[trainset[f],:][:,trainset[f]])
        
        clf.fit(K, funqs[i])
        
        doo.append((2**np.arange(8))[clf.support_].sum())
    
    sups.append(doo)
        
    ax = plt.subplot(1, len(funqs), i+1, projection='3d')
    
    scat = tpl.pca3d(s.T, c=doo, ax=ax)
    plt.legend(scat.legend_elements()[0], np.unique(doo))
    
    
    plt.title(nums[i])

#%%

nums = ['I', 'II', 'III', 'IV', 'V', 'VI']
for i in range(len(funqs)):
    ax = plt.subplot(1, len(funqs), i+1, projection='3d')
    
    sup_labs = np.mod(126//(2**np.arange(int(8))),2)
    
    tpl.scatter3d(F, ax=ax, c=sup_labs, s=100, cmap='bwr')
    plt.title(nums[i])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


#%%

train_funqs = [] 
for f in tqdm(funqs):
    for i in np.where(f==1)[0]:
        for j in np.where(f==0)[0]:

            f_ = 1*f
            f_[[i, j]] = 0.5 # a hack
            
            if len(train_funqs) == 0:
                train_funqs = f_[None]
            
            d = np.min([util.yuke(train_funqs, f_[syms]).min(1),
                        util.yuke(1-train_funqs, f_[syms]).min(1),
                        util.yuke(train_funqs, 1-f_[syms]).min(1),
                        util.yuke(1-train_funqs, 1-f_[syms]).min(1)], axis=0)
            
            if np.min(d) > 0:
                train_funqs = np.vstack([train_funqs, f_])

trainset = np.abs(train_funqs**2 - train_funqs) == 0

adj = [np.abs(np.diff(F[~trainset[i]], axis=0)).sum() for i in range(len(trainset))]
idx = np.argsort(adj)
train_funqs = train_funqs[idx]
trainset = trainset[idx]
train_group = np.array(adj)[idx]

#%% sufficient statistics for each split

D = util.hamming(F.T)

T = []
for f, ts in zip(train_funqs, trainset):
    T.append(np.concatenate([util.group_sum(2*f-1, d)[1:] for d in D[~ts]]))

T = np.array(T)
T1 = T[:,:3]
T2 = T[:,3:]

tees, ss_grp = np.unique(T, axis=0, return_inverse=True)

#%%

gs, cnt = np.unique(train_group, return_counts=True)
nrow = len(gs)
ncol = np.max(cnt)

for i,grp in enumerate(gs):
    deez = np.where(train_group==grp)[0]
    
    for j,f in enumerate(deez):
        k = i*ncol + j
        
        ax = plt.subplot(nrow, ncol, k+1, projection='3d')
        
        tpl.scatter3d(F[trainset[f]], ax=ax, c=train_funqs[f][trainset[f]], s=100, cmap='bwr')
        plt.title('group: ' + str(ss_grp[f]))
        
        j1, j2 = np.where(~trainset[f])[0]
        ax.text(F[j1,0], F[j1,1], F[j1,2], '1')
        ax.text(F[j2,0], F[j2,1], F[j2,2], '2')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # plt.axis(False)

#%% generate different possible geometries

s = np.random.dirichlet(np.ones(4), size=1000)
A = np.array([[1,3,3,1],
              [1,2,1,0],
              [1,1,0,0],
              [1,0,0,0]]).T
B = np.array([[1,6,12,8],
              [1,4,4,0],
              [1,2,0,0],
              [1,0,0,0]]).T
K = np.array([[1,1,1,1],
              [3,1,-1,-3],
              [3,-1,-1,3],
              [1,-1,1,-1]])

ki = s@A
li = ki@K
# li = np.random.dirichlet(np.ones(4), size=1000)
gi = (1/li)@K / 8

R = (gi[:,1] - gi[:,3])/(gi[:,2] - gi[:,3])
qd = gi[:,-1] / gi[:,0]

#%% combine the two

denom = gi[:,-1,None]**2 - gi[:,0,None]**2
y1 = (gi[:,0,None]*gi[:,1:]@T1.T - gi[:,-1,None]*gi[:,1:]@T2.T) / denom
y2 = (gi[:,0,None]*gi[:,1:]@T2.T - gi[:,-1,None]*gi[:,1:]@T1.T) / denom

#%%

gs, cnt = np.unique(train_group, return_counts=True)
nrow = len(gs)
ncol = np.max(cnt)

for i,grp in enumerate(gs):
    deez = np.where(train_group==grp)[0]
    
    for j,f in enumerate(deez):
        k = i*ncol + j
        
        ax = plt.subplot(nrow, ncol, k+1)
        
        plt.scatter(y1[:,f], y2[:,f])
        
        # plt.xlim([-1, 1700])
        # plt.ylim([-3, 1700])
        
        plt.plot(plt.xlim(), [0,0], 'k--')
        plt.plot([0,0], plt.ylim(), 'k--')
        
        plt.title('group: ' + str(ss_grp[f]))
        
#%%

clf = SVC(kernel='precomputed')
# clf = KernelRidge(kernel='precomputed', alpha=1e-6)

gs, cnt = np.unique(train_group, return_counts=True)
nrow = len(gs)
ncol = np.max(cnt)

cols = cols = cm.ocean(np.arange(64))

works = []
sups = []
for i,grp in enumerate(gs):
    deez = np.where(train_group==grp)[0]
    
    for j,f in enumerate(deez):
        k = i*ncol + j
        
        ba = []
        doo = []
        yhat = np.zeros((len(ki), 2))
        for this, kern in tqdm(enumerate(ki)):
            
            K = kern[D]
            
            # A = K[~trainset[f],:][:,trainset[f]]@la.inv(K[trainset[f],:][:,trainset[f]])
            
            clf.fit(K[trainset[f],:][:,trainset[f]], train_funqs[f][trainset[f]])
            
            doo.append((2**np.arange(6))[clf.support_].sum())
            # alpha = clf.dual_coef_
            alpha = np.zeros(6)
            alpha[clf.support_] = clf.dual_coef_
            yhat[this] = (alpha@K[trainset[f],:][:,~trainset[f]]) + clf.intercept_
            # yhat[this] = clf.predict(K[~trainset[f],:][:,trainset[f]])
            # yhat[this] = A@(2*train_funqs[f][trainset[f]]-1)
            
            foo = alpha@K[trainset[f],:][:,trainset[f]] + clf.intercept_
            ba.append(np.sign(foo)@clf.predict(K[trainset[f],:][:,trainset[f]])/6)
        
        sups.append(doo)
        works.append(np.mean(ba))
            
        ax = plt.subplot(nrow, ncol, k+1, projection='3d')
        
        scat = tpl.pca3d(s.T, c=doo, ax=ax)
        # bawa = plt.scatter(yhat[:,0], yhat[:,1], c=doo)
        plt.legend(scat.legend_elements()[0], np.unique(doo))
        
        # plt.plot(plt.xlim(), [0,0], 'k--')
        # plt.plot([0,0], plt.ylim(), 'k--')
        
        plt.title('group: ' + str(ss_grp[f]))

#%%

gs, cnt = np.unique(train_group, return_counts=True)
nrow = len(gs)
ncol = np.max(cnt)


for i,grp in enumerate(gs):
    deez = np.where(train_group==grp)[0]
    
    for j,f in enumerate(deez):
        k = i*ncol + j
        
        sup_labs = np.mod(30//(2**np.arange(int(6))),2)
        
        ax = plt.subplot(nrow, ncol, k+1, projection='3d')
        
        tpl.scatter3d(F[trainset[f]], ax=ax, c=sup_labs, s=100, cmap='bwr')
        plt.title('group: ' + str(ss_grp[f]))
        
        j1, j2 = np.where(~trainset[f])[0]
        ax.text(F[j1,0], F[j1,1], F[j1,2], '1')
        ax.text(F[j2,0], F[j2,1], F[j2,2], '2')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # plt.axis(False)

#%%

epochs = 50

N = 500
L = 1
activation = 'ReLU'
init_scale = 10

gs, cnt = np.unique(train_group, return_counts=True)
nrow = len(gs)
ncol = np.max(cnt)

H_pt = torch.FloatTensor(H).to(device)
li_pt = torch.FloatTensor(li).to(device)
train_funqs_pt = torch.FloatTensor(2*train_funqs-1).to(device)

mod = students.FFN(H_pt.shape[-1], N, 1, L,
                   loss_func=nn.MSELoss(),
                   nonlinearity=activation)
mod.to(device)

worked = []
for i,grp in enumerate(gs):
    deez = np.where(train_group==grp)[0]
    
    for j,f in enumerate(deez):
        k = i*ncol + j
        
        ba = []
        yhat = torch.zeros((len(ki), 2))
        for this, lambs in tqdm(enumerate(li_pt)):
            
            lam = lambs[F.sum(1)]
            reps = H_pt*lam[None]
            
            X_trn = reps[trainset[f]]
            X_tst = reps[~trainset[f]]
            Y = train_funqs_pt[f][trainset[f]]
            
            dl = pt_util.batch_data(X_trn, Y)
            
            mod.init_weights(init_scale)
            
            ls = []
            for it in range(epochs):
                 ls.append(mod.grad_step(dl))
            
            yhat[this] = mod(X_tst).squeeze()
            ba.append(ls[-1])
    
        worked.append(np.mean(ba))
            
        ax = plt.subplot(nrow, ncol, k+1)
        
        yhat = yhat.detach().numpy()
        
        plt.scatter(yhat[:,0], yhat[:,1])
        
        plt.plot(plt.xlim(), [0,0], 'k--')
        plt.plot([0,0], plt.ylim(), 'k--')
        
        plt.title('group: ' + str(ss_grp[f]))
        
        
#%%

schema = 0

X_full = torch.FloatTensor(F)
X = torch.FloatTensor(F[trainset[schema]])
Y = torch.FloatTensor(train_funqs[schema][trainset[schema]])

#%%

N = 100
L = 1
activation = 'ReLU'
init_scale = 1e-2

# mod = students.NewFeedforward(X.shape[-1], *([N]*L), Y.shape[-1], 
                              # nonlinearity=a.\ctivation)

# mod = FFN(dim_hidden=N, depth=L, activation=activation, init_scale=init_scale)

#%%

epochs = 100

n_schema = len(train_funqs)

ls = np.zeros((n_schema, epochs))
ft = np.zeros((n_schema, epochs, 2**n))
gen = np.zeros((n_schema, epochs, 2))

for i in range(n_schema):
    
    X_full = torch.FloatTensor(F)
    X = torch.FloatTensor(F[trainset[i]])
    Y = torch.FloatTensor(train_funqs[i][trainset[i]])

    mod = students.FFN(X.shape[-1], N, 1, L,
                       loss_func=nn.MSELoss(),
                       nonlinearity=activation)

    dl = pt_util.batch_data(X, Y)
    
    for it in tqdm(range(epochs)):
        
        ls[i,it] = mod.grad_step(dl)
        
        f_mod = mod(X_full)
        
        gen[i,it] = f_mod.detach().numpy()[~trainset[i]].squeeze()
        ft[i,it] = H@(2*f_mod.detach().numpy()-1).squeeze()
    
order = F.sum(1)
power = ((ft**2)@np.eye(n+1)[order])[...,1:] / (2**n)

#%%

simp = np.array([[-1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]])

P_sch = (punqs/punqs.sum(-1,keepdims=True))@simp.T
P_mod = (power[:,-1]/power[:,-1].sum(-1, keepdims=True))@simp.T

# tpl.scatter3d(P_sch, marker='*', c=np.arange(6), s=500)
# tpl.scatter3d(P_mod, c=np.arange(13), s=100, cmap='Set1')

plt.scatter(P_sch[:,0], P_sch[:,1], marker='*', c=np.arange(6), s=500)
plt.scatter(P_mod[:,0], P_mod[:,1], c=np.arange(13), s=100, cmap='Set1')

ax = plt.gca()

for i in range(len(P_mod)):
    ax.text(P_mod[i,0]+0.01, P_mod[i,1]+0.01, str(i), fontsize=15)

nums = ['I', 'II', 'III', 'IV', 'V', 'VI']
for i in range(len(P_sch)):
    ax.text(P_sch[i,0]-0.01, P_sch[i,1]-0.01, nums[i], fontsize=15,
            ha='right', va='top', )

tpl.square_axis()

#%%

def random_isometric_subgraph(G):
    """
    Find a subset of columns of S 
    """

def is_partial_cube(G):
    """
    Brute force algorithm for checking if G is a partial cube
    """
    
    if not nx.is_bipartite(G) or not nx.is_connected(G):
        return False
    
    S = []
    dw = []
    for e in G.edges:
        edw = []
        for f in G.edges:
            dxu = nx.shortest_path_length(G, e[0], f[0])
            dyv = nx.shortest_path_length(G, e[1], f[1])
            dxv = nx.shortest_path_length(G, e[0], f[1])
            dyu = nx.shortest_path_length(G, e[1], f[0])
            
            edw.append( dxu + dyv != dxv + dyu )
        dw.append(edw)
            
    DW = np.array(dw)
    
    DW2 = DW@DW
    return np.all(DW[DW2>0])
    
def equiv_classes(G):
    
    N = len(G.nodes)
    E = len(G.edges)
    
    # which_class = -1*np.ones(E)
    S = np.zeros((N,E))
    
    k = 0
    for e in G.edges:
        if np.abs(S[e[0]]-S[e[1]]).max() > 0:
            continue
        
        for n in G.nodes:
            dxu = nx.shortest_path_length(G, e[0], n)
            dxv = nx.shortest_path_length(G, e[1], n)
                        
            S[n,k] = 1*(dxu<dxv)
        
        k += 1
    
    return S[:,:k]

def contractions(G):
    """
    All DW contractions 
    """
    if not is_partial_cube(G):
        return []
    
    S = equiv_classes(G)
    E = np.array(G.edges)
    elab = np.argmax(np.abs(S[E[:,0]] - S[E[:,1]]), axis=1)
    
    cons = []
    for i in range(S.shape[1]):
        G_ = G.copy()
        for j in np.where(elab==i)[0]:
            G_ = nx.contracted_edge(G_, E[j], self_loops=False)        
        
        match = False
        for Gref in cons:
            if nx.is_isomorphic(G_, Gref):
                match = True
                break
            
        if not match:
            cons.append(G_)
    
    return cons

def is_isometric(S, G):
    
    if len(S) != len(G.nodes):
        return False
    if not is_partial_cube(G):
        return False
    
    Z = equiv_classes(G)
    if Z.shape[1] != S.shape[1]:
        return False
    
    return df_util.permham(S, Z).sum() == 0

# def is_simplex(G):
    
#     if not is_partial_cube(G):
#         return False
    
#     S = equiv_classes(G)
#     S = (S + S[[0]])%2
    
#     out = True
#     for s in S:        

def fit_J(S, strict=True):
    
    n, k = S.shape
    aye, jay = np.triu_indices(k)
    
    allS = util.F2(k)
    allS = allS[df_util.minham(allS.T, S.T)>0]
    
    if len(allS) == 0:
        return np.zeros((k,k))
    
    ## vec'd matrices
    allS_ = util.outers(allS.T)[:,aye,jay]
    S_ = util.outers(S.T)[:,aye,jay]
    
    
    if strict:
        c = np.ones(S_.shape[1])
        bub = -np.ones(len(allS))
    else:
        # c = -allS_.sum(0)
        c = -(1/(allS_.sum(1)+1e-12))@allS_
        bub = np.zeros(len(allS))
    beq = np.zeros(len(S))
    
    sol = lp(c, A_ub=-allS_, b_ub=bub, A_eq=S_, b_eq=beq, bounds=(-1,1))
    
    if sol.success:
        J = np.zeros((k,k))
        J[aye,jay] = sol.x
        J += J.T
        
        return J
    else:
        return None

def fixed_points(J):
    """
    Returns a set of fixed points if J is a valid matrix, None otherwise
    """
    
    k = len(J)
    allS = util.F2(k)
    
    val = util.qform(J, allS).squeeze()
    if np.any(val < -1e-6):
        return None 
    
    return allS[val < 1e-6]

#%%

ns = [2,3,4]

graphs = []
for n in ns:
    S = util.F2(n)
    
    for subset in tqdm(util.F2(2**n)[1:]):
        
        deez = S[subset>0]
        
        G = nx.Graph(util.hamming(deez.T)==1)
        if not nx.is_connected(G):
            continue 
        
        match = False
        for Gref in graphs:
            if nx.is_isomorphic(G, Gref):
                match = True
                break
                
        if not match:
            if is_partial_cube(G) and (equiv_classes(G).shape[1] == n):    
                graphs.append(G)

#%%

is_rep = []
is_reg = []
is_flat = []
rep_graphs = []
sup_graphs = []
jays = []
pinvs = []
for G in graphs:
    
    S = equiv_classes(G)
    S = (S+S[[0]])%2
    n = S.shape[1]
    J = fit_J(S)
    
    if J is not None:
        is_rep.append(True)
        # jays.append(J)
        rep_graphs.append(G)
        Jgraph = nx.Graph(J - np.diag(np.diag(J)/2))
        sup_graphs.append(Jgraph)
        jays.append(J)
        
        I = df_util.inc(S)
        k = np.abs(I).sum(0)/2
        diff = np.abs((S-S.mean(0))@I.T - I@(S-S.mean(0)).T).sum()
        is_flat.append((len(np.unique(k)) == 1)*(diff < 1e-6))
        # if len(np.unique(card)) > 1:
            # is_reg.append(False)
        # else:
            # k = card[0]
        # R = np.diag(1/np.sqrt(k))@df_util.inc(S).T@df_util.inc(S)@np.diag(1/np.sqrt(k))
        # R = la.pinv(np.diag(1/np.sqrt(k))@(S-S.mean(0)).T@(S-S.mean(0))@np.diag(1/np.sqrt(k)))
        R = la.pinv((S-S.mean(0)).T@(S-S.mean(0)))
        R = (R > 1e-6) - 1*(R < -1e-6) + np.eye(n)
        r = 1*(R@S.mean(0)>0.5)
        # r = R@S.mean(0)
        # J_ = np.round(R - 2*np.diag(r))
        J_ = R - 2*np.diag(r)
        pinvs.append(J_)
        FP = fixed_points(J_)
        if FP is not None:
            G_ = nx.Graph(util.hamming(FP.T)==1)
            is_reg.append(nx.is_isomorphic(G, G_))
        else:
            is_reg.append(False)
    else:
        is_rep.append(False)

#%%

ng = len(sup_graphs)

rows = int(np.floor(np.sqrt(ng)))
cols = int(np.ceil(ng/rows))

clr = ['r', 'b']
sty = ['--', '-']
for i in range(ng):
    plt.subplot(rows, cols, i+1)
    
    S = equiv_classes(rep_graphs[i])
    E = np.array(rep_graphs[i].edges)
    
    elab = np.argmax(np.abs(S[E[:,0]] - S[E[:,1]]), axis=1)
    
    pos = graphviz_layout(rep_graphs[i])
    nx.draw(rep_graphs[i], #style=sty[1*is_rep[i]],
            edge_color=elab, node_size=15,
            node_color='k',
            width=2, alpha=is_reg[i]*0.6 + 0.4)

#%%

# n = 3
i = 7
n = len(jays[i])

J = jays[i]
G = rep_graphs[i]

## get list of symmetries (permutations and flips)
# F = util.F2(n) # cube
F = fixed_points(J)
autos = isomorphism.GraphMatcher(G, G).match()
syms = np.array([list(auto.keys()) for auto in autos])

## look over all possible functions combinations
funqs = []
fclass = []
for f in util.F2(len(F)):
    
    if len(funqs) == 0:
        funqs = f[None]

    ## find global equivalence classes (there should be 6 for n=3)
    d = df_util.minham(funqs.T, f[syms].T, sym=True)
    ix = np.argmin(d)
    if d[ix] == 0:
        fclass.append(ix)
    else:
        fclass.append(len(funqs))
        funqs = np.vstack([funqs, f])
        
fclass = np.array(fclass)
funqs = (funqs + 1*(funqs.mean(1,keepdims=True)>0.5))%2

# funqs = funqs[idx]

#%%

rows = int(np.floor(np.sqrt(len(funqs))))
cols = int(np.ceil(len(funqs)/rows))

idx = np.argsort(funqs.sum(1))

pos = graphviz_layout(G)
for j in range(len(funqs)):
    plt.subplot(rows, cols, j+1)
    # nx.draw(rep_graphs[i], node_color=funqs[idx[j]], cmap='bwr')
    # pos = graphviz_layout(rep_graphs[i])
    nx.draw(G, #style=sty[1*is_rep[i]],
            # pos=pos,
            # node_size=30,
            cmap='bwr',
            node_color=funqs[idx[j]],
            )

#%%

# train_funqs = [] 
# for f in tqdm(funqs):
#     for i in range(len(f)):

#         f_ = 1.0*f
#         f_[i] = 0.5 # a hack
        
#         if len(train_funqs) == 0:
#             train_funqs = f_[None]
        
#         d = np.min([util.yuke(train_funqs, f_[syms]).min(1),
#                     util.yuke(1-train_funqs, f_[syms]).min(1),
#                     util.yuke(train_funqs, 1-f_[syms]).min(1),
#                     util.yuke(1-train_funqs, 1-f_[syms]).min(1)], axis=0)
        
#         if np.min(d) > 0:
#             train_funqs = np.vstack([train_funqs, f_])

# trainset = np.abs(train_funqs**2 - train_funqs) == 0

# adj = [np.abs(np.diff(F[~trainset[i]], axis=0)).sum() for i in range(len(trainset))]
# idx = np.argsort(adj)
# train_funqs = train_funqs[idx]
# trainset = trainset[idx]
# train_group = np.array(adj)[idx]

#%%

balanced = ((train_funqs*trainset).sum(1) == 4)

nb = np.sum(balanced)

rows = int(np.floor(np.sqrt(nb)))
cols = int(np.ceil(nb/rows))

# idx = np.argsort(funqs.sum(1))

pos = graphviz_layout(G)
for j in range(nb):
    plt.subplot(rows, cols, j+1)
    # nx.draw(rep_graphs[i], node_color=funqs[idx[j]], cmap='bwr')
    # pos = graphviz_layout(rep_graphs[i])
    nx.draw(G, #style=sty[1*is_rep[i]],
            pos=pos,
            # node_size=30,
            cmap='bwr',
            node_color=train_funqs[balanced][j],
            )

#%% Sample random kernels from the saliency simplex

S = equiv_classes(G)

s = np.random.dirichlet(np.ones(S.shape[1])+1, size=1000)
A = np.flipud(la.pascal(np.ones(S.shape[1])+1, kind="lower")).T

ki = s@A

d = util.hamming(S.T).astype(int)

kerns = ki[:,d]

#%%

trainsets = funqs[funqs.sum(1)==2] == 0

train = trainsets[0]

Ktrn = kerns[:,train][...,train]
Ktst = kerns[:,~train][...,train]

weights = (Ktst @ nla.inv(Ktrn))

#%%

ynoti = util.F2(np.sum(train))

ynoti = (2*ynoti-1).T

yhat = np.sign(weights @ ynoti)

flips = 1 - 2*np.eye(np.sum(train))

yflip = np.sign(np.einsum('ijk,lkm->ijlm', weights, (flips[:,:,None] * ynoti[:,None,:])))

influence = (yflip*yhat[:,:,None] < 0).mean(-1)






