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

def inc(S):
    """
    Given the binary embedding of a partial cube, return the incidence 
    """
    
    dif = S[None] - S[:,None]
    dif *= (np.abs(dif).sum(-1,keepdims=True)==1)
    
    return dif.sum(1)

def sadj(S):
    """
    Given the binary embedding of a partial cube, find the adjacency 
    structure of the set elements
    """
    
    n,k = S.shape
    
    I = 2*inc(S)
    S_ = np.hstack([S, np.ones((n,1))])
    
    A = la.pinv(2*S_-1)@I
    adj = (A[:-1]+np.eye(k))
    
    return 2*adj, A[-1] - adj.sum(0) 

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
    



    