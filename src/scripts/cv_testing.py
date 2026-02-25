CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import itertools as itt
from tqdm import tqdm


import util
import plotting as tpl

import bae_models
import bae_util
import bae_search
import df_util
import symbolic as sym

#%%

N = 64
d = 100
n_samp = 50

logsnr = 0

# kernel_mod = True
kernel_mod = False

mod_args = {'sparse_reg': 0, 'tree_reg': 0}
opt_args = {'decay_rate': 0.88, 'initial_temp': 10}

Strue = df_util.schurcats(N, 0.5)
Wtrue, noise = df_util.noisyembed(Strue, d, logsnr)

X = Strue@Wtrue.T + noise
X = (X-X.mean(0))/np.sqrt(np.mean((X-X.mean(0))**2))

ktrue = Strue.shape[1]

ntunq = [N//2, 3*(N//4), N-1]
nfunq = [d//2, 3*(d//4), d-1]
kunq = np.arange(2, 2*ktrue+1)

owen_loss = []
fu_loss = []
imp_loss = []
unq = []
ham = []
train_loss = []
kay = []
nt = []
nf = []
for k in tqdm(kunq):
    
    if kernel_mod:
        mod = bae_models.KernelBMF(dim_hid=k, kernel_input=True, **mod_args)
        en = mod.fit(X@X.T, verbose=False, **opt_args)
    else:
        mod = bae_models.BiPCA(dim_hid=k, **mod_args)
        en = mod.fit(X, verbose=False, **opt_args)
        
    ham.append(df_util.minham(mod.S, Strue, sym=True).mean())
    unq.append(np.unique((mod.S+mod.S[[0]])%2, axis=1).shape[1])
    train_loss.append(en[-1])
    
    for ntrain_items in ntunq:
        for ntrain_feats in nfunq:
            
            p_tst = 1 - (ntrain_feats*ntrain_items)/(N*d)
            owen = []
            fu = []
            imp = []
            for _ in range(n_samp):
                
                train_items = np.isin(range(N), np.random.choice(range(N), ntrain_items, replace=False))
                train_feats = np.isin(range(d), np.random.choice(range(d), ntrain_feats, replace=False))
                
                if kernel_mod:
                    imp.append(bae_util.kerimpcv(mod, X, folds=1/p_tst, diag=False, **opt_args))
                
                else:
                    owen.append(bae_util.bicv(mod, X, train_items, train_feats, n_iter=5, **opt_args))
                    fu.append(bae_util.fpcv(mod, X, train_items, train_feats, **opt_args))
                    imp.append(bae_util.impcv(mod, X, folds=1/p_tst, **opt_args))
                    
            owen_loss.append(owen)
            fu_loss.append(fu)
            imp_loss.append(imp)
            
            kay.append(k)
            nf.append(ntrain_feats)
            nt.append(ntrain_items)

nf = np.array(nf)
nt = np.array(nt)
kay = np.array(kay)

fu_loss = np.array(fu_loss)
owen_loss = np.array(owen_loss)
imp_loss = np.array(imp_loss)

ham = np.array(ham)
unq = np.array(unq)

#%%

i = 0
j = 0
deez = (nf==nfunq[i])*(nt==ntunq[j])

# plt.plot(kay[deez], np.mean(imp_loss[...,1], axis=1)[deez])
# plt.plot(kay[deez], np.mean(imp_loss[...,0], axis=1)[deez], '--')
plt.plot(kay[deez], np.mean(owen_loss[...,1], axis=1)[deez])
plt.plot(kay[deez], np.mean(owen_loss[...,0], axis=1)[deez], '--')
# plt.plot(kay[deez], ham[deez], '--')
plt.plot(kunq, train_loss, '--')

plt.plot([ktrue, ktrue], plt.ylim(), 'k--')

#%%

N = 64
d = 100
n_samp = 50

opt_args = {'decay_rate': 0.88, 'initial_temp': 10}

Strue = df_util.schurcats(N, 0.5)
Wtrue, noise = df_util.noisyembed(Strue, d, 10)

X = Strue@Wtrue.T + noise
X = (X-X.mean(0))/np.sqrt(np.mean((X-X.mean(0))**2))

K = X@X.T

ktrue = Strue.shape[1]

S = np.random.choice([0,1], Strue.shape)

train_items = np.eye(N)[np.random.choice(range(N), N//2, replace=False)].sum(0) == 1

#%%

t = (N-1)/N
gam = (N-2)/N
M = (~np.outer(~train_items, ~train_items))
m = np.sum(train_items )

totdot = M*util.center(S@S.T)*K
totnrm = M*util.center(S@S.T)**2

nrm0 = []
nrm1 = []
nrm2 = []
nrm3 = []
dot0 = []
dot1 = []
dot2 = []
dot3 = []
for i in range(N):
    
    noti = (np.eye(N)[i] == 0)
    
    olddot = M[noti][:,noti]*util.center(S[noti]@S[noti].T)*util.center(X[noti]@X[noti].T)
    oldnrm = M[noti][:,noti]*util.center(S[noti]@S[noti].T)**2
    
    # olddot = totdot[noti][:,noti].sum()
    # oldnrm = totnrm[noti][:,noti].sum()
    # Mi = M[noti][:,noti]
    # olddot = np.sum(Mi*util.center(S[noti]@S[noti].T)*util.center(X[noti]@X[noti].T))
    # oldnrm = np.sum(Mi*util.center(S[noti]@S[noti].T)**2)
    
    dot0.append(totdot.sum() - olddot.sum())
    nrm0.append(totnrm.sum() - oldnrm.sum())
    # dot0.append(totdot[i].sum())
    # nrm0.append(totnrm[i].sum())
    
    savg = S[noti].mean(0)
    
    ni = np.sum(M[i]*noti)
    mii = 1*M[i,i]
    strn = S[noti*M[i]].mean(0)
    # stst = (strn - t*savg)*(ni/N)
    
    # Ci = S[noti*M[i]].T@S[noti*M[i]]
    # foo = np.outer(savg, np.diag(Ci))
    # Ji = Ci - t*foo - t*foo.T  + ni*(t**2)*np.outer(savg, savg) + mii*(t**2)*np.outer(S[i]-savg, S[i]-savg)
    # # Ji = (S[noti*M[i]]-savg).T@(S[noti*M[i]] - savg) + mii*(t**2)*np.outer(S[i]-savg, S[i]-savg)
    # Ji += np.outer(-stst, S[i]) + np.outer(S[i], -stst) + ni*np.outer(S[i],S[i])/(N**2)
    
    Ji = (S[noti*M[i]] - t*savg - S[i]/N).T@(S[noti*M[i]] - t*savg - S[i]/N) + mii*(t**2)*np.outer(S[i] - savg, S[i] - savg)
    Wi = (S[noti*M[i]] - t*savg - S[i]/N).T@X[noti*M[i]] + mii*t*np.outer(S[i] - savg, X[i])
    
    nrm2.append((t**2)*(S[i] - savg)@Ji@(S[i] - savg))
    dot2.append(t*(S[i] - savg)@Wi@X[i])

    if train_items[i]:
        
        C = S[noti].T@S[noti]/(N-1)
        Csx = S[noti].T@X[noti]/(N-1)
        # Jtst = (N-1)*(C - (N**2 - N + 1)*np.outer(savg, savg)/(N**2) 
        #               - (N - 1)*(np.outer(savg, S[i]) + np.outer(S[i], savg))/(N**2)
        #               +  np.outer(S[i],S[i])/N)
        Jtst = (N-1)*(C - t*np.outer(savg, savg) 
                      + (np.outer(S[i],S[i]) - np.outer(S[i],savg) - np.outer(savg, S[i]))/N )
        
        Wtst = (N-1)*(Csx + np.outer(S[i], X[i])/(N-1))
        
        # Jtst = (S[noti]- t*savg - S[i]/N).T@(S[noti] - t*savg -S[i]/N) + (t**2)*np.outer(S[i]-savg, S[i]-savg)
        
    else:
        C = S[noti*M[i]].T@S[noti*M[i]]
        Csx = S[noti*M[i]].T@X[noti*M[i]]/m
        xtrn = X[noti*M[i]].mean(0)
        
        Jtst = (C - t*m*np.outer(savg, strn) - t*m*np.outer(strn, savg) 
                + (t**2)*m*np.outer(savg, savg) 
                + (m*t/N)*np.outer(S[i], savg) + (m*t/N)*np.outer(savg, S[i]) 
                - (m/N)*np.outer(strn,S[i]) - (m/N)*np.outer(S[i], strn) 
                + (m/N**2)*np.outer(S[i], S[i]))
    
        Wtst = m*(Csx - t*np.outer(savg, xtrn) - np.outer(S[i], xtrn)/N)
    
    nrm3.append((t**2)*(S[i] - savg)@Jtst@(S[i] - savg))
    dot3.append(t*(S[i] - savg)@Wtst@X[i])
    
    strn = S[noti*train_items].mean(0)
    
    if train_items[i]:
        
        C = S[noti].T@S[noti]/(N-1)
        Csx = S[noti].T@X[noti]/(N-1)

        J = (C - (N-4)*np.outer(savg,savg)/N 
             - 2*(savg[None] + savg[:,None])/N
             + 1/N 
             - t*np.outer(2*savg-1, 2*savg-1)/(2*N) ) 
        h = (C@savg - t*(savg@savg)*savg - (savg@savg)*(1-savg)/N
             - t*(savg@savg)*(2*savg-1)/(2*N) )
        
        hinp = Csx@X[i] + (X[i]@X[i])*(1-savg)/(N-1)
        
        const = savg@C@savg - t*(savg@savg)**2  - t*((savg@savg)**2 )/(2*N)
        binp = savg@Csx@X[i]
        
        nrm1.append((N-1)*(t**2)*(S[i]@J@S[i] - 2*h@S[i] + const))
        dot1.append((N-1)*t*(hinp@S[i] - binp))
        
    else:
        C = S[noti*train_items].T@S[noti*train_items] / m
        Csx = S[noti*M[i]].T@X[noti*M[i]]/m
        
        J = (C + (gam**2)*np.outer(savg, savg) 
             - gam*np.outer(savg, strn) - gam*np.outer(strn, savg)
             + gam*savg[None]/N + gam*savg[:,None]/N 
             - strn[None]/N - strn[:,None]/N + 1/(N**2))
    
        h = (C@savg + gam*t*(savg@savg)*savg - t*(savg@savg)*strn - gam*(savg@strn)*savg 
             + t*(savg@savg)/N - (savg@strn)/N)    
        hinp = Csx@X[i] - xtrn@X[i]*(t*savg + (1-savg)/N)
        
        const = savg@C@savg + (t**2)*(savg@savg)**2 - 2*t*(savg@savg)*(savg@strn)
        binp = savg@Csx@X[i] - t*(savg@savg)*(xtrn@X[i])
    
        nrm1.append(m*(t**2)*(S[i]@J@S[i] - 2*h@S[i] + const))
        dot1.append(m*t*(hinp@S[i] - binp))

nrm0 = np.array(nrm0)
nrm1 = np.array(nrm1)
nrm2 = np.array(nrm2)    
nrm3 = np.array(nrm3)
dot0 = np.array(dot0)
dot1 = np.array(dot1)
dot2 = np.array(dot2)
dot3 = np.array(dot3)

#%%

def canon_order(S, train_items, i):
    
    noti = util.rangediff(len(S), [i])
    idx = np.argsort(train_items[noti])
    
    return np.vstack([S[noti][idx], S[i]])

#%%

M = (~np.outer(~train_items, ~train_items))

i = 0 
    
noti = (np.eye(N)[i] == 0)

savg = S[noti].mean(0)
strn = S[noti*train_items].mean(0)
xtrn = X[noti*train_items].mean(0)
m = np.sum(train_items)
t = (N-1)/N
gam = (N-2)/N

these_i = util.F2(ktrue)

totdot = M*util.center(S@S.T)*K
totnrm = M*util.center(S@S.T)**2

mi = np.sum(~train_items*noti)

## Symbolic functions
f1 = sym.Contraction(sym.new_nrm - sym.old_nrm)
f2 = sym.Contraction(sym.new_nrm-sym.old_nrm, k_bounds=(1,'m'), l_bounds=(1,'m'))
f3 = sym.Contraction((sym.new_dot-sym.old_dot))
f4 = sym.Contraction(sym.new_dot-sym.old_dot, k_bounds=(1,'m'), l_bounds=(1,'m'))

f5 = sym.Contraction(sym.new_nrm, k_bounds=(1,'n-1'), l_bounds=('n','n'))
f6 = sym.Contraction(sym.new_nrm, k_bounds=(1,'m'), l_bounds=('n','n'))
f7 = sym.Contraction(sym.new_nrm, k_bounds=('n','n'), l_bounds=('n','n'))
f8 = sym.Contraction(sym.new_nrm, k_bounds=('m+1','n-1'), l_bounds=('n','n'))

f1.set_bounds(n=N)
f2.set_bounds(n=N, m=mi)
f3.set_bounds(n=N)
f4.set_bounds(n=N, m=mi)

f5.set_bounds(n=N)
f6.set_bounds(n=N, m=mi)
f7.set_bounds(n=N)
f8.set_bounds(n=N, m=mi)

Ssort = canon_order(S, train_items, i)
Xsort = canon_order(X, train_items, i)
Xsort = Xsort - Xsort[:-1].mean(0)

J1, h1, b1 = f1(S=Ssort.T)
J2, h2, b2 = f2(S=Ssort.T)
J3, h3, b3 = f3(S=Ssort.T, X=Xsort.T, x=Xsort[-1])
J4, h4, b4 = f4(S=Ssort.T, X=Xsort.T, x=Xsort[-1])

J5, h5, b5 = f5(S=Ssort.T)
J6, h6, b6 = f6(S=Ssort.T)
J7, h7, b7 = f7(S=Ssort.T)
J8, h8, b8 = f8(S=Ssort.T)

# Jnrm_local = (N-1)*J5 - mi*J6
# hnrm_local = (N-1)*h5 - mi*h6
# bnrm_local = (N-1)*b5 - mi*b6
Jnrm_local = (N-1-mi)*J8
hnrm_local = (N-1-mi)*h8
bnrm_local = (N-1-mi)*b8

Jnrm = ((N-1)**2)*J1 - (mi**2)*J2
hnrm = ((N-1)**2)*h1 - (mi**2)*h2
bnrm = ((N-1)**2)*b1 - (mi**2)*b2

# Jdot = ((N-1)**2)*J3 - (mi**2)*J4
hdot = ((N-1)**2)*h3 - (mi**2)*h4
bdot = ((N-1)**2)*b3 - (mi**2)*b4

#%%
dnrm_local = np.zeros(len(these_i))
dnrm_nonlocal = np.zeros(len(these_i))
ddot_local = np.zeros(len(these_i))
ddot_nonlocal = np.zeros(len(these_i))
true_nrm = []
true_dot = []
foo_nrm = []
dnrm_A = np.zeros(len(these_i))
dnrm_B = np.zeros(len(these_i))
dnrm_C = np.zeros(len(these_i))
dnrm_pred = np.zeros(len(these_i))
ddot_pred = np.zeros(len(these_i))
dnrm_pred_local = np.zeros(len(these_i))

nrm = []
dot = []
for this, si in enumerate(these_i):
    
    S_ = 1*S
    S_[i] = si
    
    Qn = M*((S_ - savg)@((S_ - savg)).T)
    
    # olddot = totdot[noti][:,noti].sum()
    # oldnrm = totnrm[noti][:,noti].sum()
    olddot = util.center(S[noti]@S[noti].T)*util.center(X[noti]@X[noti].T)
    oldnrm = util.center(S[noti]@S[noti].T)**2
    
    newdot = util.center(S_@S_.T)*util.center(X@X.T)
    newnrm = util.center(S_@S_.T)**2
    
    dnrm_local[this] = (2*(M[i][noti]*newnrm[i][noti]).sum() + M[i][i]*newnrm[i,i])
    dnrm_nonlocal[this] = ((M[noti][:,noti]*newnrm[noti][:,noti]).sum() - (M[noti][:,noti]*oldnrm).sum())
    
    ddot_local[this] = (2*(M[i][noti]*newdot[i][noti]).sum() + M[i][i]*newdot[i,i])
    ddot_nonlocal[this] = ((M[noti][:,noti]*newdot[noti][:,noti]).sum() - (M[noti][:,noti]*olddot).sum())
    
    # dnrm_A[this] = (newnrm[train_items*noti][:,train_items*noti].sum() - oldnrm[train_items[noti]][:,train_items[noti]].sum())
    # dnrm_B[this] = (newnrm[~train_items*noti][:,train_items*noti].sum() - oldnrm[~train_items[noti]][:,train_items[noti]].sum())
    # dnrm_C[this] = (newnrm[~train_items*noti][:,~train_items*noti].sum() - oldnrm[~train_items[noti]][:,~train_items[noti]].sum())
    
    # foo_nrm.append(2*t*np.sum(Qn[i][noti]**2) + (t**2)*Qn[i,i]**2 )
    dnrm_pred[this] = si@Jnrm@si + 2*si@hnrm + bnrm
    ddot_pred[this] = 2*si@hdot + bdot
    
    dnrm_pred_local[this] = si@Jnrm_local@si + 2*si@hnrm_local + bnrm_local
    
    ## predict dnrm_test
    Ctst = S[~train_items*noti].T@S[~train_items*noti]/np.sum(~train_items*noti)
    stst = np.diag(Ctst)
    stilde = t*savg - stst
    ab = savg@stst
    aa = savg@savg
    bb = stst@stst
    
    # Jtst = (Ctst - np.outer(stst, stst) + 2*np.outer(stilde, stilde) 
    #           + (stilde[None] + stilde[:,None])/N + 1/(2*N**2))/(N**2)
    # htst = (Ctst@stilde + stilde@stilde*(t*savg + 1/(2*N)) - t*savg@stilde*stst)/N
    # const = (2*stst@Ctst@savg - (2*N-1)*(savg@Ctst@savg + aa*bb + ab**2) 
    #          + (3*N**2 - 3*N + 1)*aa*ab/(N**2) - (2*N-1)*(2*N**2 - 2*N + 1)*aa**2 / (N**3) ) / N
    
    # dnrm_pred.append(2*(si@Jtst@si + 2*htst@si))
    
    true_nrm.append((M*newnrm).sum() - (M[noti][:,noti]*oldnrm).sum())
    true_dot.append((M*newdot).sum() - (M[noti][:,noti]*olddot).sum())
    
    if train_items[i]:
        
        C = S[noti].T@S[noti]/(N-1)
        Csx = S[noti].T@X[noti]/(N-1)
    
        J = (C - (N-4)*np.outer(savg,savg)/N 
             - 2*(savg[None] + savg[:,None])/N
             + 1/N 
             - t*np.outer(2*savg-1, 2*savg-1)/(2*N) ) 
        h = (C@savg - t*(savg@savg)*savg - (savg@savg)*(1-savg)/N
             - t*(savg@savg)*(2*savg-1)/(2*N) )
        
        hinp = Csx@X[i] + (X[i]@X[i])*(1-savg)/(N-1)
        
        const = savg@C@savg - t*(savg@savg)**2  - t*((savg@savg)**2 )/(2*N)
        
        binp = savg@Csx@X[i]
        
        nrm.append((N-1)*(t**2)*(si@J@si - 2*h@si + const))
        dot.append((N-1)*t*(hinp@si - binp))
        
    else:
        C = S[noti*train_items].T@S[noti*train_items] / m
        Csx = S[noti*train_items].T@X[noti*train_items]/m
        
        J = (C + (gam**2)*np.outer(savg, savg) 
             - gam*np.outer(savg, strn) - gam*np.outer(strn, savg)
             + gam*savg[None]/N + gam*savg[:,None]/N 
             - strn[None]/N - strn[:,None]/N + 1/(N**2))
    
        h = (C@savg + gam*t*(savg@savg)*savg - t*(savg@savg)*strn - gam*(savg@strn)*savg 
             + t*(savg@savg)/N - (savg@strn)/N)    
        hinp = Csx@X[i] - xtrn@X[i]*(t*savg + (1-savg)/N)
        
        const = savg@C@savg + (t**2)*(savg@savg)**2 - 2*t*(savg@savg)*(savg@strn)
        binp = savg@Csx@X[i] - t*(savg@savg)*(xtrn@X[i])

        nrm.append(m*(t**2)*(si@J@si - 2*h@si + const))
        dot.append(m*t*(hinp@si - binp))

nrm = np.array(nrm)
true_nrm = np.array(true_nrm)
foo_nrm = np.array(foo_nrm)
dot = np.array(dot)
true_dot = np.array(true_dot)
foo_nrm = np.array(foo_nrm)
