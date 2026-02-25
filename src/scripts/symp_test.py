CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import numpy as np
import sympy as sp

import util
import plotting as tpl

import symbolic as sym

#%%

n = sp.symbols('n')

#%% Define the summands

f1 = sym.Contraction(sym.new_nrm-sym.old_nrm, k_bounds=(1,'n-1'), l_bounds=(1,'n-1'))
f2 = sym.Contraction(sym.new_nrm-sym.old_nrm, k_bounds=(1,'m'), l_bounds=(1,'m'))
f3 = sym.Contraction(sym.new_nrm, k_bounds=(1,'n-1'), l_bounds=('n','n'))
f4 = sym.Contraction(sym.new_nrm, k_bounds=(1,'m'), l_bounds=('n','n'))
f5 = sym.Contraction(sym.new_nrm, k_bounds=('n','n'), l_bounds=('n','n'))
f6 = sym.Contraction(sym.new_nrm, k_bounds=('m+1','n-1'), l_bounds=('n','n'))
f7 = sym.Contraction(sym.new_nrm-sym.old_nrm, k_bounds=(1,'m'), l_bounds=('m+1','n-1'))

#%% Equations for a training item

Q_nonlocal = sp.expand(f1.k_sum*f1.l_sum*f1.quad - f2.k_sum*f1.l_sum*f2.quad)
h_nonlocal = sp.expand(f1.k_sum*f1.l_sum*f1.lin - f2.k_sum*f1.l_sum*f2.lin)

Q_local = sp.expand(2*f3.k_sum*f3.l_sum*f3.quad + f5.k_sum*f5.l_sum*f5.quad)
h_local = sp.expand(2*f3.k_sum*f3.l_sum*f3.lin + f5.k_sum*f5.l_sum*f5.lin)

quad = sp.factor(Q_nonlocal + Q_local)

#%% Equations for a test item

Q_nonlocal = sp.expand(f1.k_sum*f1.l_sum*f1.quad - f2.k_sum*f1.l_sum*f2.quad)
h_nonlocal = sp.expand(f1.k_sum*f1.l_sum*f1.lin - f2.k_sum*f1.l_sum*f2.lin)

Q_local = sp.expand(2*(f3.k_sum*f3.l_sum*f3.quad - f4.k_sum*f4.l_sum*f4.quad))
h_local = sp.expand(2*f4.k_sum*f4.l_sum*f4.lin)

#%%

# these_subs = {s_[i]: (n-m-1)*s_trn[i]/(n-1) - m*s_tst[i]/(n-1), 
#               s_[j]: (n-m-1)*s_trn[j]/(n-1) - m*s_tst[j]/(n-1), 
#               C[i,j]: (n-m-1)*C[i,j]/(n-1) - m*Ctst[i,j]/(n-1)}

# these_subs = {s_tst[i]: (n-1)*s_[i]/m - (n-m-1)*s_trn[i]/m, 
#               s_tst[j]: (n-1)*s_[j]/m - (n-m-1)*s_trn[j]/m, 
#               Ctst[i,j]: (n-1)*C[i,j]/m - (n-m-1)*Ctrn[i,j]/m}

these_subs = {s_[i]: s_trn[i]/2 - s_tst[i]/2, 
              s_[j]: s_trn[j]/2 - s_tst[j]/2, 
              C[i,j]: C[i,j]/2 - Ctst[i,j]/2}

# coef_subs = {(n-1)/n: t, (m-n+1)/n: -p, (m-1)/m: th, 1/m: 1-th, 1/n:1-t}
# coef_subs_inv = {p: (n-m-1)/n, t:(n-1)/n, th: (m-1)/m}

coef_subs = {m: 1/(1-th), n:1/(1-t)}
coef_subs_inv = {t:(n-1)/n, th: (m-1)/m}
p_sub = {t*th - 2*t + 1: p*(th-1)}
p_sub_inv = {p: (t*th - 2*t + 1)/(th-1)}

# these_subs = {s_trn[i]: (n-1)*s_[i]/(n-m-1) - m*s_tst[i]/(n-m-1), 
#               s_trn[j]: (n-1)*s_[j]/(n-m-1) - m*s_tst[j]/(n-m-1), 
#               Ctrn[i,j]: (n-1)*C[i,j]/(n-m-1) - m*Ctst[i,j]/(n-m-1)}

foo2 = sp.expand(f7.k_sum*f7.l_sum*f7.quad).subs(these_subs)
ba2 = sp.expand(foo2 + 2*f6.k_sum*f6.l_sum*f6.quad)

#%%

quad = (Q_nonlocal + 2*f6.k_sum*f6.l_sum*f6.quad).subs(these_subs)

bawa = sym.get_monomials(sp.expand(quad/n), these_vars, repl=coef_subs)
poodoo = sum([a*x for a,x in bawa.items()])
poodoo = sp.factor(poodoo, these_vars)#/(p*t)

#%%
sgn_ = sp.IndexedBase('sgn_')
sgn_trn = sp.IndexedBase('sgn_trn')
sgn_tst = sp.IndexedBase('sgn_tst')
ds_trn = sp.IndexedBase('ds_trn')
ds_tst = sp.IndexedBase('ds_tst')

abbrv = {sgn_[i]: 2*s_[i]-1, 
         sgn_[j]: 2*s_[j]-1,
         sgn_trn[i]: 2*s_trn[i]-1,
         sgn_trn[j]: 2*s_trn[j]-1,
         sgn_tst[i]: 2*s_tst[i]-1,
         sgn_tst[j]: 2*s_tst[j]-1,
         d_[i]: 
         }

test_term = (1-t)*((2*s_trn[i]-1)*(2*s_trn[j]-1) - (t**2)*(2*s_[i]-1)*(2*s_[j]-1))
test_term += 2*t*th*(s_[i]-s_trn[i])*(s_[j]-s_trn[j])
test_term += 2*(Ctrn[i,j] - s_trn[i]*s_trn[j])

newdoo = sp.factor(sp.expand(poodoo - p*t*test_term, these_vars))
newdoo.subs(coef_subs_inv).expand().subs(coef_subs)

#%%

# these_subs = {s_tst[i]: (n-1)*s_[i]/m - (n-m-1)*s_trn[i]/m, 
#               s_tst[j]: (n-1)*s_[j]/m - (n-m-1)*s_trn[j]/m, 
#               Ctst[i,j]: (n-1)*C[i,j]/m - (n-m-1)*Ctrn[i,j]/m}

# these_subs = {s_tst[i]: (n-1)*s_[i]/m - (n-m-1)*s_trn[i]/m, 
#               s_tst[j]: (n-1)*s_[j]/m - (n-m-1)*s_trn[j]/m, 
#               Ctrn[i,j]: (n-1)*C[i,j]/(n-m-1) - m*Ctst[i,j]/(n-m-1)}

# these_subs = {s_trn[i]: (n-1)*s_[i]/(n-m-1) - m*s_tst[i]/(n-m-1), 
#               s_trn[j]: (n-1)*s_[j]/(n-m-1) - m*s_tst[j]/(n-m-1), 
#               Ctrn[i,j]: (n-1)*C[i,j]/(n-m-1) - m*Ctst[i,j]/(n-m-1)}

these_subs = {s_[i]: m*s_tst[i]/(n-1) + (n-m-1)*s_trn[i]/(n-1), 
              s_[j]: m*s_tst[j]/(n-1) + (n-m-1)*s_trn[j]/(n-1), 
              Ctrn[i,j]: (n-1)*C[i,j]/(n-m-1) - m*Ctst[i,j]/(n-m-1)}

s_nrm = sp.symbols('sTs')
tst_nrm = sp.symbols('yTy')
trn_nrm = sp.symbols('xTx')
trn_dot_tst = sp.symbols('xTy')
s_dot_trn = sp.symbols('sTx')
s_dot_tst = sp.symbols('sTy')
s_sum = sp.symbols('sT1')

abbrv = {s_[j]**2: s_nrm, 
         s_trn[j]**2: trn_nrm, 
         s_[j]*s_trn[j]: s_dot_trn,
         s_[j]*s_tst[j]: s_dot_tst,
         s_tst[j]**2: tst_nrm, 
         s_tst[j]*s_trn[j]: trn_dot_tst}

abbrv_inv = {s_nrm: s_[j]**2, 
             trn_nrm: s_trn[j]**2, 
             tst_nrm: s_tst[j]**2,
             s_dot_trn: s_[j]*s_trn[j],
             s_dot_tst: s_[j]*s_tst[j],
             trn_dot_tst: s_trn[j]*s_tst[j],
             s_sum: s_[j]
             }


z_ = sp.IndexedBase('z_')
z_trn = sp.IndexedBase('z_trn')
z_tst = sp.IndexedBase('z_tst')

emp_subs = {s_[i]: z_[i]/(n-1),
            s_trn[i]: z_trn[i]/(n-m-1),
            s_tst[i]: z_tst[i]/m,
            s_[j]: z_[j]/(n-1),
            s_trn[j]: z_trn[j]/(n-m-1),
            s_tst[j]: z_tst[j]/m
            }

emp_subs_inv = {z_[i]: s_[i]*(n-1),
                z_trn[i]: s_trn[i]*(n-m-1),
                z_tst[i]: s_tst[i]*m,
                z_[j]: s_[j]*(n-1),
                z_trn[j]: s_trn[j]*(n-m-1),
                z_tst[j]: s_tst[j]*m,
                }

emp_vars = [Ctst[i,j], Ctrn[i,j], C[i,j], z_[i], z_[j], z_trn[i], z_trn[j], z_tst[i], z_tst[j]]

#%%

constraints = [(n-1)*s_[i] - m*s_tst[i] - (n-m-1)*s_trn[i],
               (n-1)*s_[j] - m*s_tst[j] - (n-m-1)*s_trn[j]]

#%%
lin = sp.expand((h_nonlocal + 2*f6.k_sum*f6.l_sum*f6.lin).subs(these_subs))

Qs = (quad*s_[j]).expand().subs(these_subs)
# Qs = (2*(n-1)*(n-m-1)*Ctrn[i,j]/n).subs(these_subs)*s_[j]
# Qs = 0

# bawa = sym.get_monomials(sp.expand((-lin - Qs)/n), these_vars, repl=coef_subs)
bawa = sym.get_monomials(sp.expand((-lin - Qs)), these_vars)
# bawa = sym.get_monomials(sp.expand((-lin)*n/((n-1)*(n-m-1))), these_vars)
bawa = {a:sp.factor(x.subs(coef_subs)) for a,x in bawa.items()}
poodoo = sum([a*x for a,x in bawa.items()])
poodoo = sp.factor(poodoo, these_vars)#/(p*t)

#%%

test_term = 2*m*t*(s_[j]-s_tst[j])*(Ctst[i,j] - s_tst[i]*s_tst[j])
test_term += 2*m*t*(s_[j]-s_tst[j])**2 * (s_[i] - s_tst[i])
test_term += 2*m*t*s_[j]*(1-s_[j])*(s_[i] - s_tst[i])/n
# test_term = -2*m*t*(s_[j]*(s_[j]-s_tst[j]) + s_[j]*(1-s_[j])/n )*s_tst[i]
# test_term += 2*m*t*((s_[j]-s_tst[j])**2 + s_[j]*(1-s_[j])/n)*s_[i]
test_term += (t*(t+1)*(t-m/n)*s_[j]*(1-s_[j])/n - m*t*(s_[j]-s_tst[j])**2/n)*(2*s_[i]-1)

newdoo = sp.expand(poodoo.subs(coef_subs_inv) - test_term.subs(coef_subs_inv))

#%%
