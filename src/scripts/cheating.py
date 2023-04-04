CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation as anime
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations, combinations
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.manifold import MDS

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# import umap
from cycler import cycler

from pypoman import compute_polytope_vertices, compute_polytope_halfspaces
import cvxpy as cvx
import polytope as pc

# my code
import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as dicplt
import grammars as gram
import dichotomies as dics


#%%

# def dot2dist(K):
#     return torch.sqrt(torch.abs(torch.diag(K)[:,None] + torch.diag(K)[None,:] - 2*K))
def dot2dist(K):
    return np.sqrt(np.abs(np.diag(K)[:,None] + np.diag(K)[None,:] - 2*K))

def centered_kernel_alignment(K1,K2):
    K1_ = K1 - K1.mean(-2,keepdims=True) - K1.mean(-1,keepdims=True) + K1.mean((-1,-2),keepdims=True)
    K2_ = K2 - K2.mean(-2,keepdims=True) - K2.mean(-1,keepdims=True) + K2.mean((-1,-2),keepdims=True)
    denom = np.sqrt((K1_**2).sum((-1,-2))*(K2_**2).sum((-1,-2)))
    return (K1_*K2_).sum((-1,-2))/np.where(denom, denom, 1e-12)

def semicentered_kernel_alignment(K1,K2):
    K1_ = K1 - K1.mean(-2,keepdims=True) - K1.mean(-1,keepdims=True) + K1.mean((-1,-2),keepdims=True)
    # K2_ = K2 - K2.mean(-2,keepdims=True) - K2.mean(-1,keepdims=True) + K2.mean((-1,-2),keepdims=True)
    denom = np.sqrt((K1_**2).sum((-1,-2))*(K2**2).sum((-1,-2)))
    return (K1_*K2).sum((-1,-2))/np.where(denom, denom, 1e-12)

def hsic(K1,K2):
    K1_ = K1 - K1.mean(-2,keepdims=True) - K1.mean(-1,keepdims=True) + K1.mean((-1,-2),keepdims=True)
    K2_ = K2 - K2.mean(-2,keepdims=True) - K2.mean(-1,keepdims=True) + K2.mean((-1,-2),keepdims=True)
    return (K1_*K2_).mean((-1,-2))

def center(K):
    return K - K.mean(-2,keepdims=True) - K.mean(-1,keepdims=True) + K.mean((-1,-2),keepdims=True)

def get_depths(clus):

    ovlp = (1*(clus>0))@(clus>0).T
    subs = (ovlp == np.diag(ovlp)) - np.eye(len(ovlp))
    # subs = ((clus>0)@clus.T==1) - np.eye(len(ovlp))
    
    G = nx.from_numpy_array((subs@subs==0)*subs, create_using=nx.DiGraph)
    depth = np.zeros(len(clus))
    for i in nx.topological_sort(G):
        anc = list(nx.ancestors(G,i))
        if len(anc)>0:
            depth[i] = np.max(depth[anc])+1

    return depth


def clus2dot(clus):
    
    depth = get_depths(clus)
    
    return (np.max((clus[:,None,:]*clus[:,:,None]>0)*depth[:,None,None], axis=0))


def ellipsoid_step(C, b, zero_tol=1e-10, mult_tol=1e-6):
    """
    Inputs: 
        C (K,N): inequality weights 
        b (K,1): inequality thresholds
        zero_tol: (default 1e-6) tolerance around 0
        mult_tol: (default 1e-6) multiplicity tolerance
    Outputs:
        x_max: maximum-norm point(s) on the maximum-volume ellipse
    """
    
    N = C.shape[1]
    
    # find maximum-volume inscribed ellipse
    Q = cvx.Variable((N,N), symmetric='True')
    r = cvx.Variable((N,1))
    constr = [Q >> 0]
    constr += [(cvx.norm(Q@c) + c@r <= b_) for c,b_ in zip(C,b)]
    prob = cvx.Problem(cvx.Maximize(cvx.log_det(Q)), constr)
    prob.solve()
    
        # find point on ellipse with maximum norm 
    P = la.inv(Q.value)@la.inv(Q.value)
    rr = r.value
    
    rrP = rr.T@P
    
    if la.norm(rr) > zero_tol: # if the center is non-zero, solve convex program
        gam = cvx.Variable((1,1))
        lam = cvx.Variable((1,1))
        cstr = [lam>=0]
        cstr += [cvx.bmat([[(-np.eye(N) + lam*P), -lam*(rrP.T)], [-lam*(rrP), lam*(rrP@rr) - lam - gam]]) >> 0]
        prob = cvx.Problem(cvx.Maximize(gam), cstr)
        prob.solve()
        
        x_max = la.pinv(-np.eye(N) + lam.value*P)@(lam.value*P@rr)
        
    else: # otherwise, the answer is just the top singular vector
        _,l,v = la.svd(Q.value)
        tops = (l >= np.max(l) - mult_tol)
        
        x_max = v[:,tops]*np.max(l)
    
    new_c = -(x_max.T@P - rrP)
    new_b = (new_c.T*x_max).sum(0,keepdims=True)
        
    return x_max, new_c, new_b



def max_norm_point(C, b, max_steps=5, conv_tol=1e-5, touch_tol=1e-4, **iter_args):
    """
    Can return one of two things:
        a single vertex which achieves the maximum norm
        a set of maximum-norm vertices which share a facet
    """
    
    # first step is special because it's an svd
    x_max, new_c, new_b = ellipsoid_step(C, b, **iter_args)
    
    if x_max.shape[1] > 1: 
        # need to figure out something better to do here
        # this feels like a place where tricks are possible
        C_ = np.append(C, new_c[[0],:], axis=0)
        b_ = np.append(b, new_b[:,[0]], axis=0)
        x_max = x_max[:,[0]]
    
    else:
        C_ = np.append(C, new_c, axis=0)
        b_ = np.append(b, new_b, axis=0)
    
    nrm = 0
    i = 0
    flag = False
    while x_max.T@x_max - nrm >= conv_tol:
        
        if i > max_steps:
            break
        
        nrm = x_max.T@x_max
        
        x_max, new_c, new_b = ellipsoid_step(C_, b_, **iter_args)
        
        # check whether the new maximum touches any faces 
        touching = C@x_max - b
        if np.any(touching >= -touch_tol):
            faces = np.where(touching >= -touch_tol)[0]
            
            # Ax > b <=> -Ax < - b
            this_C = np.append(C, -C[faces,:], axis=0)
            this_b = np.append(b, -b[faces], axis=0)
            
            x_max = np.stack(compute_polytope_vertices(this_C, this_b))
            flag = True
            break
        
        C_ = np.append(C_, new_c, axis=0)
        b_ = np.append(b_, new_b, axis=0)
        
        i += 1

    return x_max, flag

def mvee(points, tol = 0.001):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.concatenate((points, np.ones((N,1))), axis=1).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q@np.diag(u)@Q.T
        M = np.diag(Q.T@la.inv(X)@Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u    
    A = la.inv(points.T@np.diag(u)@points)/d
    return A


def mutual_information(binary):
    """
    Input: binary matrix (num_item, num_cluster)
    Output: mutual information (num_cluster, num_cluster)
    """
    
    a = np.stack([binary.T, 1-binary.T])
    
    # p(a)
    p_a = a.sum(-1)/binary.shape[0]
    Ha = np.sum(-p_a*np.log2(p_a, where=p_a>0), 0)
    
    # p(a | b=1)
    p_ab = a@binary / binary.sum(0)
    Hab = np.sum(-p_ab*np.log2(p_ab, where=p_ab>0), 0) # entropy
    
    # p(a | b=0)
    p_ab_ = a@(1-binary) / (1-binary).sum(0)
    Hab_ = np.sum(-p_ab_*np.log2(p_ab_, where=p_ab_>0), 0) # entropy
    
    return Ha[:,None] - binary.mean(0)[None,:]*Hab - (1-binary).mean(0)[None,:]*Hab_


def intersect_vars(b1, b2):
    return np.stack([b1*b2, b1*(1-b2), (1-b1)*b2, (1-b1)*(1-b2)]).T

def aaT(A):
    return np.einsum('...ij,...kj->...ik', A, A)

def proj_coef(A, B):
    return np.sum(A*center(B))/(1e-6+np.sum(center(A)**2))


# def is_subset(B):
#     ovlp = np.stack([B.T@B/B.sum(0), B.T@(1-B)/(1-B).sum(0), (1-B).T@B/B.sum(0), (1-B).T@(1-B)/(1-B).sum(0)])
#     return ovlp


#%% debugging

# GT = gram.RegularTree([1,1,1], fan_out=2, respect_hierarchy=False)
# GT = gram.RegularTree([1,1,4], fan_out=2, respect_hierarchy=False)
GT = gram.RegularTree([1,2], fan_out=2, respect_hierarchy=True)
# GT = gram.RegularTree([1,1,2,4,8,16], fan_out=2, respect_hierarchy=False)
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,4])])
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3])])
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3]), set([0,1,2])])
# GT = gram.LabelledItems(labels=[set([0]),set([1]),set([2]), set([3])])
# GT = gram.LabelledItems(labels=[set([0]),set([1]),set([2])])
# GT = gram.LabelledItems(labels=[set([0,1]),set([2,3,4]),set([3,4]),set([0]),set([1]),set([2]), set([3]), set([4])])
F = GT.represent_labels(GT.items).T
# # F += np.random.randn(*F.shape)*0.1
# # F = (F - F.mean(0))#/np.sqrt(F.shape[-1])
# # F += np.random.randn(1,F.shape[-1])

K_ = F@F.T
# # K = (F-F.mean(0))@(F-F.mean(0)).T
# # K = 1 - 0.5*dot2dist(F@F.T)**2

d = dot2dist(F@F.T)**2

# GT = gram.RegularTree([1,1], fan_out=2, respect_hierarchy=False)
# # K_ = GT.deepest_common_ancestor(only_leaves=False)[:,[5,6,7,8,0,1,2,3]][[5,6,7,8,0,1,2,3],:]
# # K_ = GT.deepest_common_ancestor(only_leaves=False)[:,[0,1,2,3,5,6,7,8]][[0,1,2,3,5,6,7,8],:]
# K_ = GT.deepest_common_ancestor(only_leaves=False)

# K = -0.5*dot2dist(K_)**2
# d = dot2dist(K_)**2

# d_ = d - d.sum(1,keepdims=True)/(len(d)-2) - d.sum(0,keepdims=True)/(len(d)-2) + d.sum()/((len(d)-2)*(len(d)-1))
# d_ *= (1-np.eye(len(d)))
# d_ = d

# K = K/np.sqrt(np.diag(K)[None,:]*np.diag(K)[:,None])

N = len(K_)

# levels = (np.unique(np.round(K[np.triu_indices(N)], 5)))


clusters = []
clus_mat = np.zeros((0,N))
clus_depth = []
corr = []
delta = []
out_group_loss = []
in_group_loss = []
test = []
alignment = []
# in_group_gain = []]

y = np.ones((N,1))
cka_score = [centered_kernel_alignment(y@y.T, K_)]

actual = [nx.descendants(GT.similarity_graph, n).intersection(set(GT.items)) for n in GT.similarity_graph]

#%%
X = cvx.Variable((N,N), symmetric='True')

H = np.eye(N) - np.ones((N,N))/N
A = [np.eye(N)[:,[i]]*np.eye(N)[[i],:] for i in range(N)]

constraints = [X >> 0]
constraints += [cvx.trace(A[i]@X) == 1 for i in range(N) ]
constraints += [cvx.trace(X@H) == 1]

# enforce non-obtuse triangles
constraints += [X - X[:,[i]] - X[[i],:] + X[i,i] >= 0 for i in range(N)]

scl = (np.sum(center(K_)*(y@y.T))/(1e-12 + np.sum(center(y@y.T)**2)))
prob = cvx.Problem(cvx.Maximize(cvx.trace(center(K_ - scl*(y@y.T))@X)), constraints)
prob.solve()

l,v = la.eigh(center(X.value))

eigs = v[:,l>l.max()-1e-5]

if eigs.shape[1] == 1:
    e_max = eigs
    
else: 
    C = np.append(eigs, -eigs, axis=0)
    C = C[(C**2).sum(1) > 1e-5]
    
    x_max, flag = max_norm_point(C, np.ones((len(C),1)))
    e_max = eigs@x_max
    
zeros = np.squeeze(np.abs(e_max) <= 1e-3)
k = np.sum(zeros) # hopefully this is small! otherwise I'll cry! 
if k > 0:
    # can't think of anything better than enumerating the equidistant vertices
    test = np.repeat(np.sign(e_max), 2**k, axis=1).T
    test[:,zeros] = 2*tasks.StandardBinary(k)(np.arange(2**k)).numpy() - 1
    
    score = centered_kernel_alignment(y@y.T + test[:,None,:]*test[:,:,None], K_)
    
    y = np.append(y, np.sign(test[[np.argmax(score)],:]).T, axis=1)
else:
    y = np.append(y, np.sign(e_max), axis=1)


cka_score.append(centered_kernel_alignment(y@y.T, K_))
print(cka_score[-1])

# for i in np.argsort(-l):
#     best = centered_kernel_alignment(K_, y@y.T)
#     new_best = centered_kernel_alignment(K_, y@y.T + np.sign(v[:,[i]]*v[:,[i]].T))
#     if new_best - best > 1e-6:
#         y = np.append(y, np.sign(v[:,[i]]), axis=1)
#     else:
#         break

# y = np.unique(np.append(y, np.sign(v[:,:np.argmax(np.cumsum(l)/np.sum(l) >=1)+1]), axis=1), axis=1)
# y = np.unique(np.append(y, (v[:,:np.argmax(np.cumsum(l)/np.sum(l) >=1)+1]), axis=1), axis=1)

#%%

def beig(K, brute_max=20, zero_tol=1e-5, max_depth=5, depth=0):
    
    N = len(K)
    
    if N <= brute_max: # few enough for brute force
        # Y = 2*np.mod(np.arange(1,2**(N-1))[:,None]//(2**np.arange(N)[None,:]),2)  - 1
            Y = np.mod(np.arange(0,2**N)[:,None]//(2**np.arange(N)[None,:]),2)
            score = centered_kernel_alignment( Y[:,None,:]*Y[:,:,None], K)
            
            # sort secondarily by cluster size
            maxscore = score>score.max()-zero_tol
            sidx = np.argmin(np.abs(Y[maxscore,:].sum(1)))
            idx = [np.where(maxscore)[0][sidx]]
            
            best = np.max(score)
            B = Y[idx,:].T
            # B = np.append(Y[idx,:].T, 1- Y[idx,:].T, axis=1)
            # nidx = np.setdiff1d(np.arange(len(Y)), np.append(idx, 2**N+np.bitwise_not(idx)))
            nidx = np.arange(len(Y))
            
            while np.max(score) < 1:
                scl = (np.sum(center(K)*(B@B.T))/(1e-12 + np.sum(center(B@B.T)**2)))
                score = centered_kernel_alignment(B@B.T + Y[nidx,None,:]*Y[nidx,:,None], K)
                
                if np.max(score) > best:
                    # sort secondarily by cluster size
                    maxscore = score>score.max()-zero_tol
                    sidx = np.argmin(np.abs(Y[nidx][maxscore,:].sum(1)))
                    idx.append(nidx[np.where(maxscore)[0][sidx]])
                    # nidx = np.setdiff1d(np.arange(len(Y)), np.append(idx, 2**N+np.bitwise_not(idx)))
                    
                    B = Y[idx,:].T
                    # B = np.append(Y[idx,:].T, 1- Y[idx,:].T, axis=1)
                    best = np.max(score)
                else:
                    break
            
        B = np.where(B.sum(0)>N/2, 1-B, B) # all clusters are at most half the data
        
        B_unq, unq, num = np.unique(B, axis=1, return_counts=True, return_index=True)
        
        ### so now, we want to group beigenvectors into "variables", which are 
        ### a set of mutually exclusive clusters that cover the data ... 
        ### if a group doesn't fully cover the data, we'll add another cluster
        ### which is made up of the remaining data
        
        groups = []
        nidx = np.arange(B.shape[1])
        while len(nidx) > 0:
            cliques = list(nx.find_cliques(nx.from_numpy_array(B[:,nidx].T@B[:,nidx] == 0 + np.eye(len(nidx)))))
            this = cliques[np.argmax([len(c) for c in cliques])]
            groups.append(nidx[this])
            nidx = np.setdiff1d(np.arange(B.shape[1]), np.concatenate(groups))
        
        B_grouped = []
        for g in groups:
            if np.all(B[:,g].sum(1)):
                B_grouped.append(B[:,g])
            else:
                B_grouped.append(np.append(B[:,g], 1-B[:,g].sum(1,keepdims=True), axis=1))
        B_g = np.concatenate(B_grouped, axis=1)
        
        
        ### try to deal with duplicate clusters
        
        B_unq, unq, num = np.unique(B, axis=1, return_counts=True, return_index=True)
        vals = np.sum(center(aaT(B_unq.T[:,:,None]))*center(B@B.T), axis=(-1,-2))
        idx = np.argsort(-np.unique(np.round(vals, 2)))
        
        
        # for v, i in zip(np.unique(vals, return_index=True)):
        
        B_flat = np.ones((N,0))
        jdx = list(range(len(unq)))
        for i in np.repeat(idx, num[idx]-1):
            # test_B = B_unq.T[jdx,None,:]*B_unq.T[jdx,:,None]
            # score = centered_kernel_alignment(B_flat@B_flat.T + B_unq[:,[i]]@B_unq[:,[i]].T*test_B, K)
            test_vars = intersect_vars(B_unq[:,[i]], B_unq[:,jdx])
            score = centered_kernel_alignment(B_flat@B_flat.T + aaT(test_vars), K)
            
            
            j = np.argmax(score)
            # if j == i: 
            #     B_flat = np.append(B_flat, 1-B_unq[:,[i]], axis=1)
            # elif num[j] == 1:
            #     B_flat = np.append(B_flat, (B_unq[:,[i]])*(B_unq[:,[j]]), axis=1)
            #     B_flat = np.append(B_flat, (1-B_unq[:,[i]])*(B_unq[:,[j]]), axis=1)
            #     B_flat = np.append(B_flat, (B_unq[:,[i]])*(1-B_unq[:,[j]]), axis=1)
            #     B_flat = np.append(B_flat, (1-B_unq[:,[i]])*(1-B_unq[:,[j]]), axis=1)

            B_flat = np.unique(np.append(B_flat, test_vars[j], axis=1), axis=1)
            if np.sum(center(B_flat@B_flat.T)*center(aaT(B_unq[:,[jdx[j]]]))) >= vals[jdx[j]]-zero_tol:
                jdx.pop(j)
            
            if centered_kernel_alignment(B_flat@B_flat.T,K) == best:
                break
        
        # # get all intersections of the terminal clusters
        # inter =  np.mod(np.arange(0,2**B.shape[1])[:,None]//(2**np.arange(B.shape[1])[None,:]),2)
        
        
        # score = centered_kernel_alignment(B@B.T + B[:,[0]]@B[:,[0]].T + Y[:,None,:]*Y[:,:,None], K)
        
        # # sort secondarily by cluster size
        # maxscore = score>score.max()-zero_tol
        # sidx = np.argmin(np.abs(Y[maxscore,:].sum(1)))
        # idx.append(np.where(maxscore)[0][sidx])
        
        # best = np.max(score)
        # B = Y[idx,:].T
        # # B = np.append(Y[idx,:].T, 1- Y[idx,:].T, axis=1)
        # # nidx = np.setdiff1d(np.arange(len(Y)), np.append(idx, 2**N+np.bitwise_not(idx)))
        # nidx = np.arange(len(Y))
        
        # while np.max(score) < 1:
        #     scl = (np.sum(center(K)*(B@B.T))/(1e-12 + np.sum(center(B@B.T)**2)))
        #     score = centered_kernel_alignment(B@B.T + B[:,[0]]@B[:,[0]].T + Y[nidx,None,:]*Y[nidx,:,None], K)
            
        #     if np.max(score) > best:
        #         # sort secondarily by cluster size
        #         maxscore = score>score.max()-zero_tol
        #         sidx = np.argmin(np.abs(Y[nidx][maxscore,:].sum(1)))
        #         idx.append(nidx[np.where(maxscore)[0][sidx]])
        #         # nidx = np.setdiff1d(np.arange(len(Y)), np.append(idx, 2**N+np.bitwise_not(idx)))
        
        #         B = Y[idx,:].T
        #         # B = np.append(Y[idx,:].T, 1- Y[idx,:].T, axis=1)
        #         best = np.max(score)
        #     else:
        #         break
        
        # B_unq, num = np.unique(B, axis=1, return_counts=True)
        
        # Bee = B[:,[0]]
        # centered_kernel_alignment(Bee@Bee.T + B[:,[0]]@B[:,[0]].T*(B.T[:,None,:]*B.T[:,:,None]), K)
        # Bee = np.append(Bee, (B[:,[0]])*(B[:,[1]]), axis=1)
    
    else:
        
        B, L = slightly_less_brute_force(K, zero_tol)
        
        # score = centered_kernel_alignment(B@L@B.T, K)
        best = centered_kernel_alignment(B@B.T, K)
        
        while best < 1:
            
            scl = (np.sum(center(K)*(B@B.T))/(1e-12 + np.sum(center(B@B.T)**2)))
            new_B, new_L = slightly_less_brute_force(K - scl*(B@B.T))
            # new_B, new_L = slightly_less_brute_force(K - B@L@B.T)
            
            
            # score = centered_kernel_alignment(B@L@B.T + new_B@new_L@new_B.T, K)
            score = centered_kernel_alignment(B@B.T + new_B@new_B.T, K)
            if score > best:
                B = np.append(B, new_B, axis=1)
                L = np.diag(np.append(np.diag(L),np.diag(new_L)))
                best = score*1
            else:
                break
    
    return B


def binary_matrix_factorization(D):

    R = la.qr(D, mode='economic')[1]
    inds = (np.abs(np.diag(R))>1e-5)
    r = np.sum(inds)
    B = 2*np.mod(np.arange(0,2**r)[:,None]//(2**np.arange(r)[None,:]),2)-1
    Z = D[:,inds]@la.inv(D[:,inds][inds,:])
    T = Z@B.T
    T_bin = T[:,la.norm(T**2 - 1, axis=0)<=1e-4]
    R_t = la.qr(T_bin.T, mode='economic')[1]
    

def generate_candidates(D, zero_tol=1e-5):
    """
    Inspired by https://arxiv.org/pdf/1401.6024.pdf , who had a method for
    finding an optimal factorization D=TA, where T in {0,1}^N and A in R^N.
    
    This instead just returns candidate vectors in {-1,1}^N. If D has rank r, 
    then this will return around 2^(r-1) vectors. So r had better be small.
    """
    
    U,s,V = la.svd(D)
    r = np.sum(s>1e-3)
    
    inds = np.arange(len(s))<r
    Z = U[:,inds]@la.inv(U[:,inds][inds,:])
    B = 2*np.mod(np.arange(0,2**(r-1))[:,None]//(2**np.arange(r)[None,:]),2)-1
    # B = np.mod(np.arange(0,2**(r))[:,None]//(2**np.arange(r)[None,:]),2)
    T = Z@B.T
    T_bin = np.where(np.abs(T)>1e-4, np.sign(T), 0)
    # T_bin = 1*(T<0.5)
    # remove the vector of all 1s or -1s
    T_bin = T_bin[:,np.abs(np.sum(T_bin,axis=0)) < len(T_bin)]
    # T_bin = T_bin[:,np.sum(T_bin,axis=0) > 0]
    
    # zeros = np.squeeze(np.abs(T_bin) <= zero_tol)
    # k = np.sum(zeros,axis=0) # hopefully this is small! otherwise I'll cry! 
    # if k > 0:
    #     # can't think of anything better than enumerating the equidistant vertices
    #     test = np.repeat(np.sign(e_max)[:,:,None], 2**k, axis=-1)
    #     test[zeros,] = 2*np.mod(np.arange(2**k)[:,None]//(2**np.arange(k)[None,:]),2) - 1
    
    return T_bin

def slightly_less_brute_force(K, zero_tol=1e-5, brute_max=20):
    
    N = len(K)
    
    X = cvx.Variable((N,N), symmetric='True')
    
    H = np.eye(N) - np.ones((N,N))/N
    A = [np.eye(N)[:,[i]]*np.eye(N)[[i],:] for i in range(N)]
    
    constraints = [X >> 0]
    constraints += [cvx.trace(A[i]@X) == 1 for i in range(N) ]
    constraints += [cvx.trace(X@H) == 1]
    
    prob = cvx.Problem(cvx.Maximize(cvx.trace(center(K)@X)), constraints)
    prob.solve()
    
    l,v = la.eigh(center(X.value))
    
    if np.sum(l>l.max()-1e-5) > 1:
    
        Y = generate_candidates(X.value)
        score = centered_kernel_alignment(K, Y.T[:,None,:]*Y.T[:,:,None])
        
        # sort secondarily by cluster size
        maxscore = score>score.max()-zero_tol
        sidx = np.argmax(np.abs(Y[:,maxscore].sum(0)))
        idx = [np.where(maxscore)[0][sidx]]
        
        best = np.max(score)
        B = Y[:,idx]
        # L = np.diag([np.sum(center(K)*(Y[:,idx]@Y[:,idx].T))/N**2])
        L = np.diag((B*(center(K)@B)).sum(0)/(center(B.T[:,:,None]*B.T[:,None,:])**2).sum((-1,-2)))
        while np.max(score)<1:
            # score = centered_kernel_alignment(K, B@L@B.T + Y.T[:,None,:]*Y.T[:,:,None])
            score = centered_kernel_alignment(K, B@B.T + Y.T[:,None,:]*Y.T[:,:,None])
            
            if np.max(score) > best:
                # sort secondarily by cluster size
                maxscore = score>score.max()-zero_tol
                sidx = np.argmax(np.abs(Y[:,maxscore].sum(0)))
                idx.append(np.where(maxscore)[0][sidx])
                
                B = Y[:,idx]
                # L = np.diag(np.sum(center(K)*(Y.T[idx,:,None]*Y.T[idx,None,:]),axis=(-1,-2))/N**2)
                L = np.diag((B*(center(K)@B)).sum(0)/(center(B.T[:,:,None]*B.T[:,None,:])**2).sum((-1,-2)))
                # best = centered_kernel_alignment(K, B@L@B.T)
                best = centered_kernel_alignment(K, B@B.T)
            else:
                break
                
    else:
        B = np.sign(v[:,[-1]])
        L = B.T@center(K)@B/np.sum(center(B@B.T)**2)
        
    return B, L

def projected_brute_force(K, zero_tol=1e-5, brute_max=20):
    
    N = len(K)
    
    X = cvx.Variable((N,N), symmetric='True')
    
    H = np.eye(N) - np.ones((N,N))/N
    A = [np.eye(N)[:,[i]]*np.eye(N)[[i],:] for i in range(N)]
    
    constraints = [X >> 0]
    constraints += [cvx.trace(A[i]@X) == 1 for i in range(N) ]
    constraints += [cvx.trace(X@H) == 1]
    
    prob = cvx.Problem(cvx.Maximize(cvx.trace(center(K)@X)), constraints)
    prob.solve()
    
    l,v = la.eigh(center(X.value))
    
    eigs = v[:,l>=l.max()-zero_tol]    
    if eigs.shape[1] > brute_max:
        print("Oh God, I'm so sorry, I just can't do this. I'm such a failure I'm so sorry.")
        print("(Dimensonality is too high)")
        return
    
    if eigs.shape[1] == 1:
        e_max = eigs
        
    else: 
        verts = eigs@np.stack(compute_polytope_vertices(np.append(eigs,-eigs, axis=0), np.ones((2*N,1)))).T
        nrm = la.norm(verts,axis=0)
        e_max = verts[:,nrm >= nrm.max()-zero_tol]
    
    zeros = np.squeeze(np.abs(e_max) <= zero_tol)
    k = np.sum(zeros) # hopefully this is small! otherwise I'll cry! 
    if k > 0:
        # can't think of anything better than enumerating the equidistant vertices
        test = np.repeat(np.sign(e_max)[:,:,None], 2**k, axis=-1)
        test[zeros,] = 2*np.mod(np.arange(2**k)[:,None]//(2**np.arange(k)[None,:]),2) - 1
        
    else:
        test = np.sign(e_max).T
    
    
    sub_score = centered_kernel_alignment(test[:,None,:]*test[:,:,None], X.value)

    best = np.max(sub_score)
    idx = np.argmax(sub_score)
    Y = test[[idx],:].T
    while np.max(sub_score) < 1:
        
        sub_score = centered_kernel_alignment(Y@Y.T + test[:,None,:]*test[:,:,None], X.value)
        
        if np.max(sub_score) > best:
            Y = np.append(Y, test[[np.argmax(sub_score)],:].T, axis=1)
            best = np.max(sub_score)
        else:
            break
    
    return Y


#%%
N = 100
k = 5 # dimension of eigenbasis

vecs = np.random.randn(N,k)
eigs = la.qr(vecs - vecs.mean(0), mode='economic')[0]

C_proj = np.append(eigs,-eigs, axis=0)

# project simplex onto constraint space
simp_a, simp_b = compute_polytope_halfspaces(C_proj)
simp_verts = np.stack(compute_polytope_vertices(simp_a, simp_b))

# find the largest contained sphere centered at zero
face_nrm = np.abs(simp_b)/la.norm(simp_a, axis=1)
xTx = 1/np.min(face_nrm**2)

# points at which that sphere touches the faces
tangents = simp_a[face_nrm**2 <= 1/xTx + 1e-4]
tangents /= np.sqrt(xTx)*la.norm(tangents, axis=1, keepdims=True)

# 
v_max = eigs@(xTx*tangents.T)

# Q = cvx.Variable((k,k), symmetric='True')
# # r = cvx.Variable((k,1))
# constr = [Q >> 0]
# # constr += [cvx.norm(Q@c) + c@r <= 1 for c in C_proj]  
# constr += [cvx.norm(Q@c) <= 1 for c in C_proj]
# prob = cvx.Problem(cvx.Maximize(cvx.log_det(Q)), constr)
# prob.solve()

#%%

Y = tasks.StandardBinary(3)(np.arange(8)).numpy()
Y_ = 2*Y-1

vecs = np.random.randn(3,3)
eigs = la.qr(vecs - vecs.mean(0), mode='economic')[0]

# normal = np.random.randn(3,1)
# normal = normal/la.norm(normal)
normal = eigs[:,[0]]
perp = eigs[:,1:]
P = normal@normal.T
y_proj = Y_.T - P@Y_.T

C = np.concatenate([np.eye(3), -np.eye(3), normal.T, -normal.T], axis=0)
b = np.concatenate([np.ones(6), np.zeros(2)])
verts = np.stack(compute_polytope_vertices(C, b))

plt.figure()
ax = dicplt.PCA3D(Y_.T)
ax.overlay(y_proj, s=100)
ax.overlay(verts.T)
# dicplt.scatter3d(Y_.T)
# dicplt.scatter3d(y_proj, s=100)
# dicplt.scatter3d(verts.T)

dicplt.set_axes_equal(plt.gca())

C_proj = np.append(perp,-perp, axis=0)

Q = cvx.Variable((2,2), symmetric='True')
r = cvx.Variable((2,1))
constr = [Q >> 0]
constr += [cvx.norm(Q@c) + c@r <= 1 for c in C_proj]  
prob = cvx.Problem(cvx.Maximize(cvx.log_det(Q)), constr)
prob.solve()

P = la.inv(Q.value)@la.inv(Q.value)
rr = r.value
gam = cvx.Variable((1,1))
lam = cvx.Variable((1,1))
cstr = [lam>=0]
cstr += [cvx.bmat([[(-np.eye(2) + lam*P), -lam*(P@rr)], [-lam*(rr.T@P), lam*((rr.T@P)@rr) - lam - gam]]) >> 0]
prob = cvx.Problem(cvx.Maximize(gam), cstr)
prob.solve()
    
u = np.stack([np.sin(np.linspace(-np.pi,np.pi,100)),np.cos(np.linspace(-np.pi,np.pi,100))])
E = perp@(Q.value@u)

x1 = np.linspace(-2,2,10)
x = np.stack([x1[None,:]*np.ones((6,10)), (1-C_proj[:,[0]]*x1[None,:])/C_proj[:,[1]]])

for i in range(6):
    plt.plot((perp@x[:,i,:])[0,:],(perp@x[:,i,:])[1,:],(perp@x[:,i,:])[2,:])

plt.plot(E[0,:],E[1,:],E[2,:])
_,l,v = la.svd(Q.value)

new_c = - (P@(v[:,0]*l[0]) - 2*rr.T@P)

vmax = perp@v[:,0]*l[0]

dicplt.scatter3d(vmax, marker='*', s=100)

#%%
N = 3
k = 2 # dimension of eigenbasis
num_samp = 5000

Y = tasks.StandardBinary(N)(np.arange(2**N)).numpy()
Y_ = 2*Y-1

vecs = np.random.randn(N,N-1)
eigs = la.qr(vecs - vecs.mean(0), mode='economic')[0]

# normal = np.random.randn(3,1)
# normal = normal/la.norm(normal)
# normal = eigs[:,[0]]
normal = np.append(np.ones((N,1))/np.sqrt(N), eigs[:,k:], axis=1)


P = normal@normal.T
y_proj = Y_.T - P@Y_.T

C = np.concatenate([np.eye(N), -np.eye(N), normal.T, -normal.T], axis=0)
b = np.concatenate([np.ones(2*N), np.zeros(2*(N-k))])
verts = np.stack(compute_polytope_vertices(C, b))

nrm_max = np.max((verts**2).sum(1))

C_proj = np.append(eigs[:,:k],-eigs[:,:k], axis=0)

v,_,_ = la.svd(np.eye(2*N) - np.ones((2*N, 2*N))/(2*N))
pert = np.random.randn(2*N - 1,num_samp) 
pert = pert * np.sqrt(1/(2*nrm_max))/la.norm((C_proj@C_proj.T@(v[:,:-1]@pert + np.ones((2*N,1))/(2*N)))/2, axis=0)

#%%

Y = tasks.StandardBinary(3)(np.arange(8)).numpy()
Y_ = 2*Y-1

vecs = np.random.randn(3,3)
eigs = la.qr(vecs, mode='economic')[0]
normal = eigs[:,[0]]
perp = eigs[:,1:]

# vecs = np.random.randn(3,2)
# eigs = la.qr(vecs - vecs.mean(0), mode='economic')[0]
# normal = np.ones((3,1))/np.sqrt(3)
# perp = eigs

P = normal@normal.T
y_proj = Y_.T - P@Y_.T

C = np.concatenate([np.eye(3), -np.eye(3), normal.T, -normal.T], axis=0)
b = np.concatenate([np.ones(6), np.zeros(2)])
verts = np.stack(compute_polytope_vertices(C, b))

plt.figure()
ax = dicplt.PCA3D(Y_.T)
# ax.overlay(y_proj, s=100)
ax.overlay(verts.T)
# dicplt.scatter3d(Y_.T)
# dicplt.scatter3d(y_proj, s=100)
# dicplt.scatter3d(verts.T)

dicplt.set_axes_equal(plt.gca())

C_proj = np.append(perp,-perp, axis=0)

Q = cvx.Variable((2,2), symmetric='True')
r = cvx.Variable((2,1))
constr = [Q >> 0]
constr += [cvx.norm(Q@c) + c@r <= 1 for c in C_proj]  
prob = cvx.Problem(cvx.Maximize(cvx.log_det(Q)), constr)
prob.solve()

P = la.inv(Q.value)@la.inv(Q.value)
rr = r.value
gam = cvx.Variable((1,1))
lam = cvx.Variable((1,1))
cstr = [lam>=0]
cstr += [cvx.bmat([[(-np.eye(2) + lam*P), -lam*(P@rr)], [-lam*(rr.T@P), lam*((rr.T@P)@rr) - lam - gam]]) >> 0]
prob = cvx.Problem(cvx.Maximize(gam), cstr)
prob.solve()
    
u = np.stack([np.sin(np.linspace(-np.pi,np.pi,100)),np.cos(np.linspace(-np.pi,np.pi,100))])
E = perp@(Q.value@u)

x1 = np.linspace(-2,2,10)
x = np.stack([x1[None,:]*np.ones((6,10)), (1-C_proj[:,[0]]*x1[None,:])/C_proj[:,[1]]])

for i in range(6):
    plt.plot((perp@x[:,i,:])[0,:],(perp@x[:,i,:])[1,:],(perp@x[:,i,:])[2,:], c='r')

plt.plot(E[0,:],E[1,:],E[2,:], c='b')
_,l,v = la.svd(Q.value)

new_c = - (P@(v[:,0]*l[0]) - 2*rr.T@P)

vmax = perp@v[:,0]*l[0]

dicplt.scatter3d(vmax, marker='*', s=100)

new_c = (-(P@(v[:,0]*l[0]) - rr.T@P))
newb = new_c@(v[:,0]*l[0])

new_C = new_c
new_b = newb[None,:]

y = np.stack([x1[None,:]*np.ones((1,10)), (1-(new_c[0,0]/newb)*x1[None,:])/(new_c[0,1]/newb)])
plt.plot((perp@y[:,0,:])[0,:],(perp@y[:,0,:])[1,:],(perp@y[:,0,:])[2,:], c='r')

Q = cvx.Variable((2,2), symmetric='True')
r = cvx.Variable((2,1))
constr = [Q >> 0]
constr += [cvx.norm(Q@c) + c@r <= 1 for c in C_proj]  
constr += [cvx.norm(Q@c) + c@r <= b for (c,b) in zip(new_C, new_b)]
prob = cvx.Problem(cvx.Maximize(cvx.log_det(Q)), constr)
prob.solve()

P = la.inv(Q.value)@la.inv(Q.value)
rr = r.value
gam = cvx.Variable((1,1))
lam = cvx.Variable((1,1))
cstr = [lam>=0]
cstr += [cvx.bmat([[(-np.eye(2) + lam*P), -lam*(P@rr)], [-lam*(rr.T@P), lam*((rr.T@P)@rr) - lam - gam]]) >> 0]
prob = cvx.Problem(cvx.Maximize(gam), cstr)
prob.solve()

x_max = la.pinv(-np.eye(2) +lam.value*P)@(lam.value*P@rr)
   
u = np.stack([np.sin(np.linspace(-np.pi,np.pi,100)),np.cos(np.linspace(-np.pi,np.pi,100))])
E = perp@(Q.value@u + r.value)

plt.plot(E[0,:],E[1,:],E[2,:], c='b')

vmax = (perp@x_max).T

dicplt.scatter3d(vmax, marker='*', s=100)

new_c = -(x_max.T@P - rr.T@P)
newb = new_c@x_max

new_C = np.append(new_C, new_c, axis=0)
new_b = np.append(new_b, new_c@x_max, axis=0)

y = np.stack([x1[None,:]*np.ones((1,10)), (1-(new_c[0,0]/newb)*x1[None,:])/(new_c[0,1]/newb)])
plt.plot((perp@y[:,0,:])[0,:],(perp@y[:,0,:])[1,:],(perp@y[:,0,:])[2,:], c='r')

Q = cvx.Variable((2,2), symmetric='True')
r = cvx.Variable((2,1))
constr = [Q >> 0]
constr += [cvx.norm(Q@c) + c@r <= 1 for c in C_proj]  
constr += [cvx.norm(Q@c) + c@r <= b for (c,b) in zip(new_C, new_b)]
prob = cvx.Problem(cvx.Maximize(cvx.log_det(Q)), constr)
prob.solve()

P = la.inv(Q.value)@la.inv(Q.value)
rr = r.value
gam = cvx.Variable((1,1))
lam = cvx.Variable((1,1))
cstr = [lam>=0]
cstr += [cvx.bmat([[(-np.eye(2) + lam*P), -lam*(P@rr)], [-lam*(rr.T@P), lam*((rr.T@P)@rr) - lam - gam]]) >> 0]
prob = cvx.Problem(cvx.Maximize(gam), cstr)
prob.solve()

x_max = la.pinv(-np.eye(2) +lam.value*P)@(lam.value*P@rr)
   
u = np.stack([np.sin(np.linspace(-np.pi,np.pi,100)),np.cos(np.linspace(-np.pi,np.pi,100))])
E = perp@(Q.value@u + r.value)

plt.plot(E[0,:],E[1,:],E[2,:], c='b')

vmax = (perp@x_max).T

dicplt.scatter3d(vmax, marker='*', s=100)

new_c = -(x_max.T@P - rr.T@P)
newb = new_c@x_max

new_C = np.append(new_C, new_c, axis=0)
new_b = np.append(new_b, new_c@x_max, axis=0)

y = np.stack([x1[None,:]*np.ones((1,10)), (1-(new_c[0,0]/newb)*x1[None,:])/(new_c[0,1]/newb)])
plt.plot((perp@y[:,0,:])[0,:],(perp@y[:,0,:])[1,:],(perp@y[:,0,:])[2,:], c='r')


#%%



