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


# %% Pick data format
K = 2
respect = False
# respect = True

# layers = [K**0,K**1,K**2]
layers = [1, 2, 2]
# layers = [1,1,1]

Data = gram.HierarchicalData(layers, fan_out=K, respect_hierarchy=respect, graph_rule='minimal')

ll = Data.labels(Data.terminals)
labs = np.where(np.isnan(ll), np.nanmax(ll)+1, ll)

Ky_all = np.sign((ll[:,:,None]-0.5)*(ll[:,None,:]-0.5))
Ky_all = torch.tensor(np.where(np.isnan(Ky_all), 0, Ky_all))

reps = Data.represent_labels(Data.terminals)
Ky = util.dot_product(reps,reps)

plt.figure()
plt.subplot(131)
pos = graphviz_layout(Data.value_tree, prog="twopi")
nx.draw(Data.value_tree, pos, node_color=np.array(Data.value_tree.nodes(data='var'))[:,1], cmap='nipy_spectral')
dicplt.square_axis()
plt.subplot(132)
plt.imshow(ll, 'bwr')
plt.subplot(133)
plt.imshow(util.dot_product(reps,reps), 'binary')

#%%

F = Data.similar_representation(only_leaves=False, similarity='dca', tol=1e-10)
feats = util.pca_reduce(F.T, num_comp=3)
# feats = MDS(3).fit_transform(F)

n = list(Data.similarity_graph)
node_xyz = np.array([feats[n.index(v)] for v in sorted(Data.similarity_graph)])
# edge_xyz = np.array([(feats[n.index(u)], feats[n.index(v)]) for u, v in Data.similarity_graph.edges()])
uv = np.argsort(Data.similarity_graph)[np.array(Data.similarity_graph.edges)]
edge_u = feats[uv[:,0],:]
edge_v =  feats[uv[:,1],:] - edge_u

leaves = np.isin(sorted(Data.similarity_graph), Data.items)
# observed = np.isin(sorted(Data.similarity_graph), Data.value_tree.nodes)
observed = np.isin(sorted(Data.similarity_graph), [])

ax = plt.subplot(projection='3d')
ax.scatter(*node_xyz[leaves,:].T, s=300, ec="w", c='r', zorder=3)
ax.scatter(*node_xyz[(~leaves)&observed,:].T, s=100, ec="w", c='k', zorder=3)
ax.scatter(*node_xyz[(~leaves)&(~observed),:].T, s=100, ec="k", c='w', zorder=3)

ax.quiver(edge_u[:,0],edge_u[:,1],edge_u[:,2], 
          edge_v[:,0],edge_v[:,1],edge_v[:,2],
          color=(0.5,0.5,0.5), zorder=-1)

# ax.quiver(feats[uv[:,0],0],feats[uv[:,0],1],feats[uv[:,0],2], feats[uv[:,1],0],feats[uv[:,1],1],feats[uv[:,1],2])

# for vizedge in edge_xyz:
    # ax.plot(*vizedge.T, color="tab:gray")
    
# for i in sorted(Data.similarity_graph):
#     ax.text(node_xyz[i,0],node_xyz[i,1],node_xyz[i,2], s=Data.similarity_graph.nodes('category')[i]) 

ax.axis('off')

#%%

# # exclusions = [(0,1), (2,3)]
# exclusions = [(0,1), (2,3,4), (0,4), (1,2)]
# # exclusions = [(0,1), (2,3,4,5), (0,4,5), (1,2,3)]

# valid_combo = lambda x:np.isin(exclusions,x).sum(1).max()==1

# num_lab = np.max(exclusions)

noise = 0.1
n_samp_dec = 1000

clf = svm.LinearSVC()


# labs = [set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3])] # overcomplete
# labs = [set([0,2]), set([0,3]), set([1,2]), set([1,3])] # disentangled 
# labs = [set([0,2]), set([0,3]), set([1,4]), set([1,5])] # hierarchical
# labs = [set(c) for c in combinations(range(6),2)] # overcomplete hierarchical
labs = [set([0,2]), set([0,3]), set([1,2]), set([1,4])] # asymmetric
# labs = [set([0,2]), set([0,3]), set([1,4]), set([1,5]), set([0,4]), set([0,5]), set([1,2]), set([1,3])]


Data = gram.LabelledItems(labels=labs)


z = Data.similar_representation(only_leaves=True, similarity='dca', tol=1e-10)
z /= np.mean(la.norm(z, axis=1))


PS = []
CCGP = []
decoding = []
for var in range(Data.num_vars):
    
    y = np.array([(var in l) for l in Data.cats])
    
    pos = np.where(y)[0]
    neg = np.where(y)[0]
    
    idx = np.random.choice(Data.num_data, n_samp_dec)
    z_dec = z[idx,:] + np.random.randn(n_samp_dec,z.shape[1])*noise
    
    Kz = util.dot_product(z.T,z.T)#/z.shape[1]
    Ky = y[None,:]*y[:,None]
    
    if len(pos)>1:

        PS.append(dics.parallelism_score(z.T, np.arange(Data.num_data), y, average=False))
        CCGP.append(np.mean(dics.compute_ccgp(z_dec, idx, y[idx], clf)))
    
    clf.fit(z_dec[:int(0.6*n_samp_dec),:], y[idx][:int(0.6*n_samp_dec)])
    decoding.append(clf.score(z_dec[int(0.6*n_samp_dec):,:], y[idx][int(0.6*n_samp_dec):]))

# hierarchy = []
# for y_sup in np.setdiff1d(range(Data.num_vars), ):
    
#     y_sup = layer_vars[l].argmax(0)
#     y_sub = layer_vars[l+1].argmax(0)
#     sigs = [util.decompose_covariance(z[y_sup==s,:].T,y_sub[y_sup==s])[1] for s in np.unique(y_sup)]
    
#     dots = np.einsum('ikl,jkl->ij',np.array(sigs),np.array(sigs))
#     csim = la.triu(dots,1)/np.sqrt((np.diag(dots)[:,None]*np.diag(dots)[None,:]))
#     foo1, foo2 = np.nonzero(np.triu(np.ones(dots.shape),1))
    
#     hierarchy.append(np.mean(csim[foo1,foo2]))

#%%

# GT = gram.RegularTree([1,1,1,1,1], fan_out=2, respect_hierarchy=False)
# GT = gram.RegularTree([1,2,4,8,16], fan_out=2, respect_hierarchy=False)
# GT = gram.RegularTree([1,2,2,4,8], fan_out=2, respect_hierarchy=True)
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,4])])
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3])])
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3]), set([0,1,2])])
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3]), 
#                                 set([0,1,2]), set([1,2,3]),set([0,2,3]),set([0,1,3])])
F = GT.represent_labels(GT.items).T 
# F += np.random.randn(*F.shape)*0.1
# F = (F - F.mean(0))#/np.sqrt(F.shape[-1])
# F += np.random.randn(1,F.shape[-1])

K = F@F.T
# K = (F-F.mean(0))@(F-F.mean(0)).T
K = 1 - 0.5*dot2dist(F@F.T)**2
d = dot2dist(F@F.T)**2

# d_ = d - d.sum(1,keepdims=True)/(len(d)-2) - d.sum(0,keepdims=True)/(len(d)-2) + d.sum()/((len(d)-2)*(len(d)-1))
# d_ *= (1-np.eye(len(d)))
# d_ = d

# K = K/np.sqrt(np.diag(K)[None,:]*np.diag(K)[:,None])

levels = (np.unique(np.round(K[np.triu_indices(GT.num_data)], 5)))


clusters = []
clus_mat = np.zeros((0,GT.num_data))
corr = []
delta = []
out_group_loss = []
in_group_loss = []
test = []
alignment = []
# in_group_gain = []
for i,eps in tqdm(enumerate(levels)):
    
    nbrs = (K>=eps-1e-4)
    # pairs = np.nonzero(np.triu(nbrs,1))
    
    # if len(pairs[0]) > 0:
    # wawa.append(np.mean((nbrs[:,pairs[0]]*nbrs[:,pairs[1]]).sum(0)/(nbrs[:,pairs[0]]+nbrs[:,pairs[1]]).sum(0)))
    # d[:,pairs[0]]
    # wawa.append(np.sum([d_[:,i]@d_[:,j]/len(d) for i,j in zip(*pairs)])/len(pairs[0]))
    # wawa.append(np.corrcoef(K[:,pairs[0]].flatten(),K[:,pairs[1]].flatten())[0,1])
    G = nx.from_numpy_array(nbrs)
    # G_ = nx.from_numpy_array((K>=levels[i+1]-1e-5).numpy())
    
    # clq = list(nx.find_cliques(G))
    CLQ = np.array([1*np.isin(np.arange(GT.num_data), w) for w in nx.find_cliques(G)])
    not_CLQ = 1-CLQ
    
    CLQ = CLQ / CLQ.sum(-1,keepdims=True)
    not_CLQ = not_CLQ / not_CLQ.sum(-1,keepdims=True)
    
    # hoods = spt.cKDTree(CLQ)
    
    print('got cliques')
    
    ## CLQ and not_CLQ sizes are (num_cliques, num_items)
    
    clq_mean = CLQ@K
    K_cntr = K[None,...] - clq_mean[:,:,None] - clq_mean[:,None,:] + (CLQ*clq_mean).sum(1)[:,None,None]
    
    # K_cond_cntr = K[None,None,...] - clq_cond_mean[...,:,None] - clq_cond_mean[...,None,:] + (CLQ_cond*clq_cond_mean).sum(-1)[...,None,None]
    
    
    # K_cntr size is (num_cliques, num_items, num_items)
    
    # be a little clever to avoid (K^2) complexity
    # K_old = (1*(clus_mat>0)).T@(1*(clus_mat>0))
    # K_prop = 1*(CLQ[:,None,:]*CLQ[:,:,None]>0) + K_old
    # align = (K_prop*K).sum((-1,-2))/np.sqrt((K**2).sum()*(K_prop**2).sum((-1,-2)))
    # out_loss = - (K_prop*K).sum((-1,-2))/np.sqrt((K**2).sum()*(K_prop**2).sum((-1,-2)))
    
    H_c = (clus_mat>0)*np.nansum(clus_mat*np.log(clus_mat),1,keepdims=True) # entropy of clusters
    if len(clusters)>0:
        d_old = np.nanmin(np.where((clus_mat[:,None,:]*clus_mat[:,:,None] > 0),np.sqrt(H_c[:,None,:]*H_c[:,:,None]), np.nan), axis=0)
    else:
        d_old = 0
    
    H_clq = (CLQ>0)*np.nansum(CLQ*np.log(CLQ),1,keepdims=True)
    d_prop =  np.where(CLQ[:,None,:]*CLQ[:,:,None]>0,np.sqrt(H_clq[:,None,:]*H_clq[:,:,None]),d_old)
    
    # out_loss = -(d_prop*d).sum((-1,-2))/np.sqrt((d**2).sum()*(d_prop**2).sum((-1,-2)))
    # out_loss = - centered_kernel_alignment(d_prop,d)
    out_loss= -util.distance_correlation(dist_x=d_prop, dist_y=d)

    # out_loss = (np.einsum('cij,cj->ci',K_cntr**2, CLQ)*not_CLQ).sum(1) 
    # in_loss = (np.einsum('cij,cj->ci',K_cntr**2, CLQ)*CLQ).sum(1)
    pr_denom = (np.einsum('cij,cj->ci',K_cntr**2, CLQ)*CLQ).sum(1)
    in_loss = (np.einsum('cjj,cj->c',K_cntr, CLQ)**2)/np.where(pr_denom, pr_denom, 1e-12)
    # in_loss = (np.einsum('cjj,cj->c',K_cntr, CLQ)**2)
    # in_gain = ((CLQ@(K**2))*CLQ).sum(1)
    lala = (np.einsum('cjj,cj->c',K_cntr, CLQ)**2)
    
    idx = np.argsort(out_loss)

    nrm = (CLQ*CLQ).sum(1)
    
    
    if len(clusters)>0:
        excluded = []
        optima = []
        while np.mean(np.isin(idx, excluded)+np.isin(idx,optima))<1:
        # for ix in idx:
            
            util.partial_distance_correlation(dist_x=d_prop, dist_y=d, dist_z=d_prop[ix])
            
            ix = idx[np.where(~(np.isin(idx, excluded)+np.isin(idx,optima)))[0]][0]
            
            # nbrs = np.array(hoods.query_ball_point(CLQ[ix], np.sqrt(nrm[ix])-1e-6))
            alts = np.where((CLQ[ix]@CLQ.T)/nrm > 0.5)[0]
            
            ls = out_loss[alts]
            # best = np.argmin(ls)
            
            if np.min(out_loss[alts]) == out_loss[ix]:
                optima.append(ix)
            
            excluded += alts[nrm[alts]<=nrm[ix]][ls>ls.min()].tolist()
    
        # optima = [o for o in optima \
        #           if np.min(np.array(in_group_loss)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) == 1]) > in_loss[o]]
        optima = [o for o in optima \
                  if np.min(np.array(in_group_loss)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > in_loss[o]
                  and np.min(np.array(test)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > lala[o]]
        # optima = [o for o in optima \
        #           if np.min(np.array(test)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > lala[o]]
        # optima = [o for o in optima \
        #           if np.max(np.array(in_group_gain)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1))== 1]) < in_gain[o]]
    else:
        optima = [0]

    accepted = CLQ[optima,:]
    out_group_loss += out_loss[optima].tolist()
    in_group_loss += in_loss[optima].tolist()
    # in_group_gain += in_gain[optima].tolist()
    test += lala[optima].tolist()
    # alignment += align[optima].tolist()
    # alignment.append(centered_kernel_alignment(K_old, K))
    
    clus_mat = np.append(clus_mat, accepted, axis=0)
    clusters += [set(np.where(w)[0]) for w in accepted]
    
    # foo = (clus_mat>0)/(clus_mat**2).sum(1,keepdims=True)
    foo = (clus_mat>0)*np.nansum(clus_mat*np.log(clus_mat),1,keepdims=True)
    dasgupta = np.nanmin(np.where((clus_mat[:,None,:]*clus_mat[:,:,None] > 0),np.sqrt(foo[:,None,:]*foo[:,:,None]), np.nan), axis=0)
    alignment.append((dasgupta*d).sum()/np.sqrt((d**2).sum()*(dasgupta**2).sum()))
    
    # ipr = ((CLQ@(K**2))*CLQ).sum(1)/((CLQ@np.diag(K))**2) # participation ratio
    
    # ipr = (np.einsum('cij,cj->ci',K_cntr**2, CLQ)*CLQ).sum(1)/(np.einsum('cii,ci->c',K_cntr,CLQ)**2)
    
    # clus = (((K - np.diag(np.diag(K)))@CLQ.T)*CLQ.T).sum(0)/(CLQ@np.diag(K))
    # ovlp = CLQ@CLQ.T
    # expl = ovlp > np.diag(ovlp)/2
    
    # accepted = CLQ[np.diag(expl*clus) == np.max(expl*clus),:]
    # pntwise = CLQ*ipr[:,None]
    
    # accepted = CLQ[((CLQ*ipr[:,None]).max(0) == CLQ*ipr[:,None]).sum(-1) == CLQ.sum(-1), :]
    
    # wawa += [set(np.where(w)[0]) for w in accepted]
    
    # wa = [c for c in clq if np.unique([i for j in nx.find_cliques(G_.subgraph(c)) for i in j], return_counts=True)[1].var() == 0]
    # clus = np.array([(K[w,:][:,w].sum()/K[w,w].sum())-1 for w in clq])
    # expl = [ clus[np.where([len(np.intersect1d(c, w))>len(c)*0.5 for w in clq])[0]].max() for c in clq]
     
    # accepted = [c for i,c in enumerate(clq) if clus[i]>=expl[i]]
    
    # wawa += accepted
    
    # corr.append(np.mean([np.isin(accepted,i).sum() for i in np.arange(GT.num_data)]))
    # if len(accepted) == 1:
        # corr.append(1)
    # else:
    # corr.append(np.mean([len(set(a).intersection(set(b)))/np.sqrt(len(a)*len(b)) for i,b in enumerate(accepted) for a in accepted[i+1:]]))
    corr.append(util.cosine_sim(accepted.T,accepted.T)[np.triu_indices(len(accepted),1)].mean())
    
    delta.append(eps/K.max())


plt.plot(delta,corr,marker='.')
# K_clus = np.array([ [ sum([(i in w)*(j in w) for w in wawa if len(w)>1])-1 for i in range(GT.num_data)] for j in range(GT.num_data)])
# K_clus = K_clus - K_clus.mean(1, keepdims=True) - K_clus.mean(0,keepdims=True) + K_clus.mean()


# match = np.sum(K_clus*K)/np.sqrt(np.sum(K**2)*np.sum(K_clus**2))
actual = [nx.descendants(GT.similarity_graph, n).intersection(set(GT.items)) for n in GT.similarity_graph]

recovery = np.mean([set([n+1 for n in w]) in actual for w in clusters if len(w)>1])
completeness = np.mean([a in ([set([n+1 for n in w]) for w in clusters]) for a in actual if len(a)>0])
print(f'{recovery}, {completeness}')

#%% debugging

# GT = gram.RegularTree([1,1], fan_out=2, respect_hierarchy=False)
# GT = gram.RegularTree([1,2,4], fan_out=2, respect_hierarchy=False)
# GT = gram.RegularTree([1,1,2,4,8,16], fan_out=2, respect_hierarchy=False)
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,4])])
# GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3])])
GT = gram.LabelledItems(labels=[set([0,2]), set([0,3]), set([1,2]), set([1,3]), set([0,1]), set([2,3]), set([0,1,2])])
# GT = gram.LabelledItems(labels=[set([0]),set([1]),set([2])])
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

#%%
X = cvx.Variable((N,N), symmetric='True')

H = np.eye(N) - np.ones((N,N))/N
A = [np.eye(N)[:,[i]]*np.eye(N)[[i],:] for i in range(N)]

constraints = [X >> 0]
constraints += [cvx.trace(A[i]@X) == 1 for i in range(N) ]
constraints += [cvx.trace(X@H) == 1]

scl = (np.sum(center(K_)*(y@y.T))/(1e-12 + np.sum(center(y@y.T)**2)))
prob = cvx.Problem(cvx.Maximize(cvx.trace(center(K_ - scl*(y@y.T))@X)), constraints)
prob.solve()

l,v = la.eigh(center(X.value))


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

Y = tasks.StandardBinary(3)(np.arange(8)).numpy()
Y_ = 2*Y-1

vecs = np.random.randn(3,2)
eigs = la.qr(vecs - vecs.mean(0), mode='economic')[0]

# normal = np.random.randn(3,1)
# normal = normal/la.norm(normal)
normal = eigs[:,[0]]
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

C_proj = np.append(eigs[:,1:],-eigs[:,1:], axis=0)

Q = cvx.Variable((2,2), symmetric='True')
constr = [Q >> 0]
constr += [cvx.norm(Q@c) <= 1 for c in C_proj]  
prob = cvx.Problem(cvx.Maximize(cvx.log_det(Q)), constr)
prob.solve()

u = np.stack([np.sin(np.linspace(-np.pi,np.pi,100)),np.cos(np.linspace(-np.pi,np.pi,100))])
E = eigs[:,1:]@(Q.value@u)

x1 = np.linspace(-2,2,10)
x = np.stack([x1[None,:]*np.ones((6,10)), (1-C_proj[:,[0]]*x1[None,:])/C_proj[:,[1]]])

for i in range(6):
    plt.plot((eigs[:,1:]@x[:,i,:])[0,:],(eigs[:,1:]@x[:,i,:])[1,:],(eigs[:,1:]@x[:,i,:])[2,:])

plt.plot(E[0,:],E[1,:],E[2,:])
_,l,v = la.svd(Q.value)

vmax = eigs[:,1:]@v[:,0]*l[0]

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

levels = np.flip(np.unique(np.round(d[np.triu_indices(N)], 5)))

eps = levels[1]
# for eps in tqdm(levels):
for eps in tqdm(levels[[0]]):

    nbrs = (d<=eps+1e-4)
    # pairs = np.nonzero(np.triu(nbrs,1))
    
    # if len(pairs[0]) > 0:
    # wawa.append(np.mean((nbrs[:,pairs[0]]*nbrs[:pairs[1]]).sum(0)/(nbrs[:,pairs[0]]+nbrs[:,pairs[1]]).sum(0)))
    # d[:,pairs[0]]
    # wawa.append(np.sum([d_[:,i]@d_[:,j]/len(d) for i,j in zip(*pairs)])/len(pairs[0]))
    # wawa.append(np.corrcoef(K[:,pairs[0]].flatten(),K[:,pairs[1]].flatten())[0,1])
    G = nx.from_numpy_array(nbrs)
    # G_ = nx.from_numpy_array((K>=levels[i+1]-1e-5).numpy())
    
    # clq = list(nx.find_cliques(G))
    CLQ = np.array([1*np.isin(np.arange(N), w) for w in nx.find_cliques(G)])
    dups = ((clus_mat>0)@CLQ.T == (clus_mat>0).sum(1, keepdims=True)).sum(0) # remove cliques which have already been accepted
    CLQ = CLQ[dups==0,:]
    not_CLQ = 1-CLQ
    
    # CLQ = CLQ / CLQ.sum(-1,keepdims=True)
    # not_CLQ = not_CLQ / not_CLQ.sum(-1,keepdims=True)
    
    # hoods = spt.cKDTree(CLQ)
    
    # print('got cliques')
    
    ## CLQ and not_CLQ sizes are (num_cliques, num_items)
    CLQ_conv = CLQ / CLQ.sum(-1,keepdims=True)
    not_CLQ_conv = not_CLQ / not_CLQ.sum(-1,keepdims=True)
    
    clq_mean = CLQ_conv@K_
    K_cntr = K_[None,...] - clq_mean[:,:,None] - clq_mean[:,None,:] + (CLQ_conv*clq_mean).sum(1)[:,None,None]
    
    pr_denom = (np.einsum('cij,cj->ci',K_cntr**2, CLQ_conv)*CLQ_conv).sum(1)
    in_loss = (np.einsum('cjj,cj->c',K_cntr, CLQ_conv)**2)/np.where(pr_denom, pr_denom, 1e-12)
    lala = (np.einsum('cjj,cj->c',K_cntr, CLQ_conv)**2)
    
    if len(clusters)>0:
        
        clq_depth = np.max(get_depths(clus_mat)*(CLQ@(clus_mat>0).T==1), axis=1)+1
     
        # k_prop = np.where( (CLQ[:,None,:]*CLQ[:,:,None]>0), clq_depth[:,None,None], k_old)
        k_prop = clus_mat.T@clus_mat + CLQ[:,None,:]*CLQ[:,:,None]
        depth_prop = (k_prop*np.eye(N)[None,...]).max(1)
        # # d_prop = depth_prop[:,None,:] + depth_prop[:,:,None] - 2*k_prop
        d_prop = np.sqrt(depth_prop[:,None,:] + depth_prop[:,:,None] - 2*k_prop)
        
        # d_prop_cntr = util.dependence_statistics(dist_x=d_prop)
        # d_prop_var = util.distance_covariance(dist_x=d_prop_cntr, dist_y=d_prop_cntr)
        # d_cntr = util.dependence_statistics(dist_x=d)
        # d_var = util.distance_covariance(dist_x=d_cntr,dist_y=d_cntr)
        
        # # align = (d_prop*d).sum((-1,-2))/np.sqrt((d**2).sum()*(d_prop**2).sum((-1,-2)))
        # # align = centered_kernel_alignment(d_prop,d)
        # align = util.distance_correlation(dist_x=d_prop, dist_y=d)
        
        # d_prop_cntr = util.dependence_statistics(dist_x=np.sqrt(d_prop))
        # d_prop_var = (d_prop_cntr**2).sum((-1,-2))/(N*(N-3))
        # d_cntr = util.dependence_statistics(dist_x=np.sqrt(d))
        # d_var = (d_cntr**2).sum((-1,-2))/(N*(N-3))
        
        # align = (d_prop*d).sum((-1,-2))/np.sqrt((d**2).sum()*(d_prop**2).sum((-1,-2)))
        # align = centered_kernel_alignment(d_prop,d)
        # align = util.distance_correlation(dist_x=d_prop, dist_y=d)
        # align = (d_prop_cntr*d_cntr).sum((-1,-2))/(np.sqrt(d_var*d_prop_var))/(N*(N-3))
        # align = hsic(K_, k_prop)
        align = centered_kernel_alignment(K_, k_prop)
        # align = util.normalized_kernel_alignment(k_prop, K_)
        
        nrm = (CLQ*CLQ).sum(1)
    
        out_loss = -align
        
        # be a little clever to avoid (K^2) complexity
    
        idx = np.argsort(out_loss)
    
        # excluded = []
        # optima = []
        # while np.mean(np.isin(idx, excluded)+np.isin(idx,optima))<1:
        # # for ix in idx:
            
        #     ix = idx[np.where(~(np.isin(idx, excluded)+np.isin(idx,optima)))[0]][0]
            
        #     # nbrs = np.array(hoods.query_ball_point(CLQ[ix], np.sqrt(nrm[ix])-1e-6))
        #     alts = np.where((CLQ[ix]@CLQ.T)/(nrm[ix]*nrm) > 0.5/nrm[ix])[0]
            
        #     ls = out_loss[alts]
        #     # best = np.argmin(ls)
            
        #     if np.min(out_loss[alts]) == out_loss[ix]:
        #         optima.append(ix)
            
        #     # pdcorr = util.partial_distance_correlation(dist_x=d_prop[alts[alts!=ix]], 
        #     #                                             dist_y=d, dist_z=d_prop[ix])
        #     V_xz = (d_prop_cntr[alts[alts!=ix]]*d_prop_cntr[ix]).sum((-1,-2))/(N*(N-3))
            
        #     R_xz = V_xz/np.sqrt(d_prop_var[alts[alts!=ix]]*d_prop_var[ix])
        #     pdcorr = (align[alts[alts!=ix]] - R_xz*align[ix])/np.sqrt((1-R_xz**2)*(1-align[ix]**2))
        #     # excluded += alts[alts!=ix][pdcorr<=current_best].tolist()
        #     excluded += alts[nrm[alts]<=nrm[ix]][ls>ls.min()].tolist()
        
        # optima = [o for o in optima \
        #           if np.min(np.array(in_group_loss)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > in_loss[o]
        #           and np.min(np.array(test)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > lala[o]]
        
        excluded = []
        optima = []
        # while np.mean(np.isin(idx, excluded)+np.isin(idx,optima))<1:
        for ix in idx:
            
            # nbrs = np.array(hoods.query_ball_point(CLQ[ix], np.sqrt(nrm[ix])-1e-6))
            
            if ix in optima:
                continue
            
            # if len(optima)>0:
            #     # resid = util.partial_distance_correlation(dist_x=d_prop[ix], dist_y=d, dist_z=np.min(d_prop[optima],axis=0))
            #     resid = util.partial_distance_correlation(dist_x=d_prop[ix], 
            #                                               dist_y=d, 
            #                                               dist_z=dot2dist(clus2dot(np.append(clus_mat, CLQ[optima,:], axis=0))))
            #     # resid = util.partial_kernel_alignment(k_prop[ix],
            #     #                                       K_,
            #     #                                       (clus2dot(np.append(clus_mat, CLQ[optima,:], axis=0))))
            # else:
            #     resid = align[ix]
            # best = util.distance_correlation(dist_x=d, dist_y=dot2dist(clus2dot(np.append(clus_mat, CLQ[optima,:], axis=0))))
            # new_best = util.distance_correlation(dist_x=d, 
            #                                       dist_y=dot2dist(clus2dot(np.append(clus_mat, CLQ[optima+[ix],:], axis=0))))
            # best = centered_kernel_alignment(K_, (clus2dot(np.append(clus_mat, CLQ[optima,:], axis=0))))
            # new_best = centered_kernel_alignment(K_, (clus2dot(np.append(clus_mat, CLQ[optima+[ix],:], axis=0))))
            best = centered_kernel_alignment(K_, clus_mat.T@clus_mat + CLQ[optima,:].T@CLQ[optima,:])
            new_best = centered_kernel_alignment(K_, clus_mat.T@clus_mat + CLQ[optima+[ix],:].T@CLQ[optima+[ix],:])
            resid = new_best - best
            # hsic(K_, clus2dot(np.append(clus_mat, CLQ[optima,:], axis=0)))
    
            if resid<=1e-6:
                break
            # if np.all(~(nbrs^(CLQ[optima,:].T@CLQ[optima,:]>0))):
            #     break
                
            pr_crit = np.min(np.array(in_group_loss)[clus_mat@CLQ[ix,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > in_loss[ix]
            clus_crit = np.min(np.array(test)[clus_mat@CLQ[ix,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > lala[ix]
            
            # if pr_crit and clus_crit:
            optima.append(ix)
            complements = not_CLQ[ix]@CLQ.T/np.sqrt(np.sum(not_CLQ[ix]**2)*nrm)
            if np.any(complements == 1):
                optima.append(np.where(complements==1)[0].item())

    else:
        optima = [0]
    
        out_loss = np.array([0])
    
    accepted = CLQ[optima,:]
    out_group_loss += out_loss[optima].tolist()
    in_group_loss += in_loss[optima].tolist()
    # in_group_gain += in_gain[optima].tolist()
    test += lala[optima].tolist()
    # alignment += align[optima].tolist()
    
    clus_mat = np.append(clus_mat, accepted, axis=0)
    clusters += [set(np.where(w)[0]+1) for w in accepted]
    
    # clus_depth += get_depths(CLQ[optima]).tolist()
    # k_old = clus2dot(clus_mat)
    k_old = (clus_mat.T@clus_mat)
    
    # foo = (clus_mat>0)*np.nansum(clus_mat*np.log(clus_mat),1,keepdims=True) # entropy of clusters
    # dasgupta = np.nanmin(np.where((clus_mat[:,None,:]*clus_mat[:,:,None] > 0),np.sqrt(foo[:,None,:]*foo[:,:,None]), np.nan), axis=0)
    # alignment.append((dasgupta*d).sum()/np.sqrt((d**2).sum()*(dasgupta**2).sum()))
    # alignment.append(centered_kernel_alignment(dasgupta,d))
    # alignment.append(util.distance_correlation(dist_x=dasgupta, dist_y=d))
    # alignment.append(util.distance_correlation(dist_x=dot2dist(k_old)**2, dist_y=d))
    # alignment.append(util.distance_correlation(dist_x=dot2dist(k_old), dist_y=d))
    alignment.append(util.normalized_kernel_alignment(k_old, K_))

#%%

# prune = [util.distance_correlation(dist_x=d, dist_y=dot2dist(clus2dot(clus_mat[np.setdiff1d(range(len(clus_mat)), j)])))>=alignment[-1] for j in range(len(clus_mat))]
actual = [nx.descendants(GT.similarity_graph, n).union(set([n])).intersection(set(GT.items)) for n in GT.similarity_graph]

recovery = np.mean([set([n for n in w]) in actual for w in clusters])
completeness = np.mean([a in ([set([n for n in w]) for w in clusters]) for a in actual])
print(f'\nBefore pruning: {recovery}, {completeness}')

useful = np.where([util.distance_correlation(dist_x=d, dist_y=dot2dist(clus2dot(clus_mat[np.setdiff1d(range(len(clus_mat)), j)])))<alignment[-1] for j in range(len(clus_mat))])[0]
# useful = np.where([util.normalized_kernel_alignment(K_, (clus2dot(clus_mat[np.setdiff1d(range(len(clus_mat)), j)])))<alignment[-1] for j in range(len(clus_mat))])[0]

clus_mat_ = clus_mat[useful]
clusters_ = [clusters[c] for c in useful]

recovery = np.mean([set([n for n in w]) in actual for w in clusters_])
completeness = np.mean([a in ([set([n for n in w]) for w in clusters_]) for a in actual])
print(f'After pruning: {recovery}, {completeness}')


#%%
eps = levels[1]


nbrs = (K>=eps-1e-4)
# pairs = np.nonzero(np.triu(nbrs,1))

# if len(pairs[0]) > 0:
# wawa.append(np.mean((nbrs[:,pairs[0]]*nbrs[:,pairs[1]]).sum(0)/(nbrs[:,pairs[0]]+nbrs[:,pairs[1]]).sum(0)))
# d[:,pairs[0]]
# wawa.append(np.sum([d_[:,i]@d_[:,j]/len(d) for i,j in zip(*pairs)])/len(pairs[0]))
# wawa.append(np.corrcoef(K[:,pairs[0]].flatten(),K[:,pairs[1]].flatten())[0,1])
G = nx.from_numpy_array(nbrs)
# G_ = nx.from_numpy_array((K>=levels[i+1]-1e-5).numpy())

# clq = list(nx.find_cliques(G))
CLQ = np.array([1*np.isin(np.arange(N), w) for w in nx.find_cliques(G)])
not_CLQ = 1-CLQ

CLQ = CLQ / CLQ.sum(-1,keepdims=True)
not_CLQ = not_CLQ / not_CLQ.sum(-1,keepdims=True)

# hoods = spt.cKDTree(CLQ)

print('got cliques')

## CLQ and not_CLQ sizes are (num_cliques, num_items)

clq_mean = CLQ@K
K_cntr = K[None,...] - clq_mean[:,:,None] - clq_mean[:,None,:] + (CLQ*clq_mean).sum(1)[:,None,None]

# filt = (1-np.eye(len(K))>0)
# CLQ_cond = (CLQ[:,None,:]*filt[None,:,:])
# CLQ_cond /= CLQ_cond.sum(-1,keepdims=True)
# not_CLQ_cond = 1-CLQ_cond
# not_CLQ_cond /= not_CLQ_cond.sum(-1,keepdims=True)
# clq_cond_mean = CLQ_cond@K
# K_cond_cntr = K[None,None,...] - clq_cond_mean[...,:,None] - clq_cond_mean[...,None,:] + (CLQ_cond*clq_cond_mean).sum(-1)[...,None,None]


# clq_min = (K*(CLQ[:,None,:]>0)).min(-1)
# clq_min += (clq_min*(CLQ>0)).max(1,keepdims=True)/2
# d_cntr = K[None,...] - 0.5*(clq_min[:,:,None] + clq_min[:,None,:])

# clq_max = (d*(CLQ[:,None,:]>0)).max(-1)
# clq_max -= (clq_max*(CLQ>0)).max(1,keepdims=True)/2
# d_cntr = d[None,...] - 0.5*(clq_max[:,:,None] + clq_max[:,None,:])

# K_cntr size is (num_cliques, num_items, num_items)

# H_c = (clus_mat>0)*np.nansum(clus_mat*np.log(clus_mat),1,keepdims=True) # entropy of clusters
# if len(clusters)>0:
#     d_old = np.nanmin(np.where((clus_mat[:,None,:]*clus_mat[:,:,None] > 0),np.sqrt(H_c[:,None,:]*H_c[:,:,None]), np.nan), axis=0)
#     current_best = util.distance_correlation(d_old, d)
# else:
#     d_old = 0
#     current_best = -1

# H_clq = (CLQ>0)*np.nansum(CLQ*np.log(CLQ),1,keepdims=True)
# d_prop =  np.where(CLQ[:,None,:]*CLQ[:,:,None]>0,np.sqrt(H_clq[:,None,:]*H_clq[:,:,None]),d_old)
# d_prop *= (1-np.eye(N))

if len(clusters)>0:
    d_old = 
    current_best = util.distance_correlation(d_old, d)
else:
    d_old = 0
    current_best = -1

H_clq = (CLQ>0)*np.nansum(CLQ*np.log(CLQ),1,keepdims=True)
d_prop =  np.where(CLQ[:,None,:]*CLQ[:,:,None]>0,np.sqrt(H_clq[:,None,:]*H_clq[:,:,None]),d_old)
d_prop *= (1-np.eye(N))

d_prop_cntr = util.dependence_statistics(dist_x=d_prop)
d_prop_var = util.distance_covariance(dist_x=d_prop_cntr, dist_y=d_prop_cntr)
d_cntr = util.dependence_statistics(dist_x=d)
d_var = util.distance_covariance(dist_x=d_cntr,dist_y=d_cntr)

# # align = (d_prop*d).sum((-1,-2))/np.sqrt((d**2).sum()*(d_prop**2).sum((-1,-2)))
# # align = centered_kernel_alignment(d_prop,d)
# align = util.distance_correlation(dist_x=d_prop, dist_y=d)

# d_prop_cntr = util.dependence_statistics(dist_x=np.sqrt(d_prop))
# d_prop_var = (d_prop_cntr**2).sum((-1,-2))/(N*(N-3))
# d_cntr = util.dependence_statistics(dist_x=np.sqrt(d))
# d_var = (d_cntr**2).sum((-1,-2))/(N*(N-3))

# align = (d_prop*d).sum((-1,-2))/np.sqrt((d**2).sum()*(d_prop**2).sum((-1,-2)))
# align = centered_kernel_alignment(d_prop,d)
# align = util.distance_correlation(dist_x=d_prop, dist_y=d)
align = (d_prop_cntr*d_cntr).sum((-1,-2))/(np.sqrt(d_var*d_prop_var))/(N*(N-3))

# K_old = (1*(clus_mat>0)).T@(1*(clus_mat>0))
# K_prop = 1*(CLQ[:,None,:]*CLQ[:,:,None]>0) + K_old
# K_prop = K_prop - K_prop.mean(1,keepdims=True) - K_prop.mean(2,keepdims=True) + K_prop.mean((1,2), keepdims=True)
# # K_prop = K_prop - (K_prop*CLQ[:,:,None]).sum(1,keepdims=True) - (K_prop*CLQ[:,None,:]).sum(2,keepdims=True) + (K_prop*CLQ[:,:,None]*CLQ[:,None,:]).sum((1,2), keepdims=True)
# # align = (K_prop*K).sum((-1,-2))/np.sqrt((K**2).sum()*(K_prop**2).sum((-1,-2)))
# # out_loss = -(K_prop*K).sum((-1,-2))/np.sqrt((K**2).sum()*(K_prop**2).sum((-1,-2)))
# align = (K_prop*d).sum((-1,-2))/np.sqrt((d**2).sum()*(K_prop**2).sum((-1,-2)))

out_loss = -align
# out_loss = -util.distance_correlation(dist_x=d_prop, dist_y=d)
# out_loss = 

# be a little clever to avoid (K^2) complexity
# out_loss = (np.einsum('cij,cj->ci',K_cntr**2, CLQ)*not_CLQ).sum(1)
# in_loss = (np.einsum('cij,cj->ci',K_cntr**2, CLQ)*CLQ).sum(1)
pr_denom = (np.einsum('cij,cj->ci',K_cntr**2, CLQ)*CLQ).sum(1)
in_loss = (np.einsum('cjj,cj->c',K_cntr, CLQ)**2)/np.where(pr_denom, pr_denom, 1e-12)
# in_loss = (np.einsum('cjj,cj->c',K_cntr, CLQ)**2)
# in_gain = ((CLQ@(K**2))*CLQ).sum(1)
lala = (np.einsum('cjj,cj->c',K_cntr, CLQ)**2)

idx = np.argsort(out_loss)

nrm = (CLQ*CLQ).sum(1)

if len(clusters)>0:
    excluded = []
    optima = []
    while np.mean(np.isin(idx, excluded)+np.isin(idx,optima))<1:
    # for ix in idx:
        
        ix = idx[np.where(~(np.isin(idx, excluded)+np.isin(idx,optima)))[0]][0]
        
        # nbrs = np.array(hoods.query_ball_point(CLQ[ix], np.sqrt(nrm[ix])-1e-6))
        alts = np.where((CLQ[ix]@CLQ.T)/(nrm[ix]*nrm) > 0.5/nrm[ix])[0]
        
        ls = out_loss[alts]
        # best = np.argmin(ls)
        
        if np.min(out_loss[alts]) == out_loss[ix]:
            optima.append(ix)
        
        # pdcorr = util.partial_distance_correlation(dist_x=d_prop[alts[alts!=ix]], 
        #                                             dist_y=d, dist_z=d_prop[ix])
        V_xz = (d_prop_cntr[alts[alts!=ix]]*d_prop_cntr[ix]).sum((-1,-2))/(N*(N-3))
        
        R_xz = V_xz/np.sqrt(d_prop_var[alts[alts!=ix]]*d_prop_var[ix])
        pdcorr = (align[alts[alts!=ix]] - R_xz*align[ix])/np.sqrt((1-R_xz**2)*(1-align[ix]**2))
        # excluded += alts[alts!=ix][pdcorr<=current_best].tolist()
        excluded += alts[nrm[alts]<=nrm[ix]][ls>ls.min()].tolist()

    # optima = [o for o in optima \
    #           if np.min(np.array(in_group_loss)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) == 1]) > in_loss[o]]
    # optima = [o for o in optima \
    #           if np.min(np.array(test)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > lala[o]]
    optima = [o for o in optima \
              if np.min(np.array(in_group_loss)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > in_loss[o]
              and np.min(np.array(test)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1)) >= 1-1e-12]) > lala[o]]
    # optima = [o for o in optima \
    #           if np.max(np.array(in_group_gain)[clus_mat@CLQ[o,:].T/((clus_mat**2).sum(1))== 1]) < in_gain[o]]
    
    
    # optima = []
    # while 
    
else:
    optima = [0]

accepted = CLQ[optima,:]
out_group_loss += out_loss[optima].tolist()
in_group_loss += in_loss[optima].tolist()
# in_group_gain += in_gain[optima].tolist()
test += lala[optima].tolist()
# alignment += align[optima].tolist()

clus_mat = np.append(clus_mat, accepted, axis=0)
clusters += [set(np.where(w)[0]) for w in accepted]

foo = (clus_mat>0)*np.nansum(clus_mat*np.log(clus_mat),1,keepdims=True) # entropy of clusters
dasgupta = np.nanmin(np.where((clus_mat[:,None,:]*clus_mat[:,:,None] > 0),np.sqrt(foo[:,None,:]*foo[:,:,None]), np.nan), axis=0)
# alignment.append((dasgupta*d).sum()/np.sqrt((d**2).sum()*(dasgupta**2).sum()))
# alignment.append(centered_kernel_alignment(dasgupta,d))
alignment.append(util.distance_correlation(dist_x=dasgupta, dist_y=d))

#%%

# compute depth of 

