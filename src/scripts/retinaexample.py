
CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
from time import time
sys.path.append(CODE_DIR)

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import colour as clr

import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as tpl
import dichotomies as dics

#%%

def srgb(vals):
    return (vals/vals.max(1,keepdims=True))*(vals>0)

n_clr = 321
n_comb = 5000
 
wl = np.linspace(400, 700, n_clr)
xyz = clr.wavelength_to_XYZ(wl)
xy = clr.XYZ_to_xy(xyz)
lms = clr.models.Yrg_to_LMS(clr.XYZ_to_Yrg(xyz))
lms = np.fliplr(lms).T

rgb = clr.convert(wl, 'Wavelength', 'RGB')
rgb = (rgb/rgb.max(1,keepdims=True))*(rgb>0) # clip into gamut

comb = np.random.dirichlet(1e-2*np.ones(n_clr), n_comb)
comb_xy = comb@xy
comb_rgb = clr.convert(comb_xy, 'cie XY', 'sRGB')
comb_rgb = (comb_rgb/comb_rgb.max(1,keepdims=True))*(comb_rgb>0)

comb_lms = lms@comb.T

#%% Retina model

s_sig = 600
m_sig = 2000
l_sig = 2000

s = np.exp(-((wl - 440)**2)/(2*s_sig))  
m = np.exp(-((wl - 530)**2)/(2*m_sig))
l = np.exp(-((wl - 560)**2)/(2*l_sig))

sml = np.stack([s,m,l])
comb_sml = sml@comb.T

tpl.scatter3d(sml.T, c=rgb)
tpl.scatter3d(comb_sml.T, c=comb_rgb)

#%% LGN model
n_lgn = 100

by = np.random.dirichlet(np.array([2,1,1])*5, n_lgn)
by = by*np.outer(np.random.choice([1,-1], n_lgn, p=[0.9,0.1]),np.array([1,-1,-1]))

rg = np.random.dirichlet(np.array([1e-10,1,1])*10, n_lgn)
rg = rg*np.outer(np.random.choice([1,-1], n_lgn),np.array([1,1,-1]))

plt.scatter(rg[:,1],rg[:,2])
plt.scatter(by[:,1],by[:,2])

#%%
# sml = np.flipud(lms)

by = np.array([-1,-1,2])
rg = np.array([-1,1,0])

sml_nrm = lms/la.norm(lms,axis=0)
comb_sml_nrm = comb_lms/la.norm(comb_lms,axis=0)

LGN = np.vstack([by@sml_nrm, rg@sml_nrm])
comb_LGN = np.vstack([by@comb_sml_nrm, rg@comb_sml_nrm])

tpl.pca(LGN, c=rgb)
# tpl.pca(LGN@comb.T, c=comb@rgb)
tpl.pca(LGN@comb.T, c=comb_rgb)
# tpl.pca(comb_LGN, c=comb_rgb)

#%%
plt.scatter(s-0.5*m-0.5*l, l-m, c=rgb)
plt.scatter(comb_sml[:,0]-0.5*comb_sml[:,1]-0.5*comb_sml[:,2],comb_sml[:,2]-comb_sml[:,1], c=comb_rgb)

#%%

tpl.scatter3d(lms.T, c=np.where(rgb>1, 1, rgb)*(rgb>0))
# tpl.scatter3d(comb@lms.T, c=comb_rgb)


#%%

plt.plot(wl, s, c='b')
plt.plot(wl, m, c='g')
plt.plot(wl, l, c='r')

plt.plot(wl, lms[0], '--', c='b')
plt.plot(wl, lms[1], '--', c='g')
plt.plot(wl, lms[2], '--', c='r')

#%%

oppo = np.diag(1/np.sqrt([2,3,2]))@np.array([[0,-1,1],[1,-1,-1],[0,1,1]])



