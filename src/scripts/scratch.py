import numpy as np
import scipy.sparse as sprs
import scipy.special as spc
import util
import df_util
import distance_factorization as df
import cvxpy as cvx
import torch
import torch.nn as nn
import torch.optim as optim

#%%



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

Q = best[0] - best[1] - best[2] + best[3]
p = 2*(best[1].sum(0) - best[3].sum(0))

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
    
    newD = np.min([newF.T@newF, (1-newF).T@newF, newF.T@(1-newF), (1-newF).T@(1-newF)],0)
    deez.append(newD.sum() - oldD.sum())
    
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
    



    