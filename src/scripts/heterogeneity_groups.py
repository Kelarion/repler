import socket
import os
import sys

if socket.gethostname() == 'kelarion':
    if sys.platform == 'linux':
        CODE_DIR = '/home/kelarion/github/repler/src'
        SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    else:
        CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
        SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    openmind = False
elif socket.gethostname() == 'openmind7':
    CODE_DIR = '/home/malleman/repler/'
    SAVE_DIR = '/om2/user/malleman/abstraction/'
    openmind = True
else:    
    CODE_DIR = '/rigel/home/ma3811/repler/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
    openmind = False

sys.path.append(CODE_DIR)

import re
import pickle
import warnings

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as anime
from itertools import permutations, combinations
from sklearn import svm, manifold, linear_model
from tqdm import tqdm

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import experiments as exp
#%%
dim_inp = 100
dim_grp = 25
n_samp = 10000
num_resamp = 25

inp_noise = 0.3

nonlin = 'relu'
# nonlin = 'tanh'
# nonlin = 'binary'

noise = lambda x: x*np.random.randn(dim_grp,2) # gaussian noise
# noise = lambda x: x*(2*np.random.rand(dim_grp,2) - 1) # uniform noise

n_tst = 15

# v_win = 0.1
# anti_mag = 0.0

tst_pnts = np.meshgrid(np.linspace(0,1,n_tst),np.linspace(0,1,n_tst))

all_PS = np.zeros((n_tst**2,3))
all_CCGP = np.zeros((n_tst**2,3))
all_SD = np.zeros((n_tst**2,3))
for i, (v_win, anti_mag) in tqdm(enumerate(zip(tst_pnts[0].flatten(),tst_pnts[1].flatten()))):

    these_ps = []
    these_ccgp = []
    these_sd = []
    for k in range(num_resamp):
        class_means = 2*(np.random.rand(dim_inp, 4)>0.5).astype(int) - 1
        which_class = np.random.choice(4, n_samp)
        
        inps = (class_means[:, which_class]*(2*(np.random.rand(dim_inp, n_samp)>inp_noise)-1))

        W1 = class_means[:,0] - anti_mag*class_means[:,3] \
            + noise(v_win)@(class_means[:,[1,2]].T)
        W2 = class_means[:,1] - anti_mag*class_means[:,2] \
            + noise(v_win)@(class_means[:,[0,3]].T)
        W3 = class_means[:,2] - anti_mag*class_means[:,1] \
            + noise(v_win)@(class_means[:,[0,3]].T)
        W4 = class_means[:,3] - anti_mag*class_means[:,0] \
            + noise(v_win)@(class_means[:,[1,2]].T)
        
        z1 = W1@inps
        z2 = W2@inps
        z3 = W3@inps
        z4 = W4@inps
        
        z = np.concatenate([z1,z2,z3,z4], axis=0)
        
        if nonlin == 'relu':
            z *= (z>=0)
        elif nonlin == 'tanh':
            z = np.tanh(z)
        elif nonlin == 'binary':
            z = (z>=0).astype(int)
        
        D = assistants.Dichotomies(4)
        
        clf = assistants.LinearDecoder(dim_grp*4, 1, assistants.MeanClassifier)
        gclf = assistants.LinearDecoder(dim_grp*4, 1, svm.LinearSVC)
        dclf = assistants.LinearDecoder(dim_grp*4, D.ntot, svm.LinearSVC)
        # clf = LinearDecoder(this_exp.dim_input, 1, MeanClassifier)
        # gclf = LinearDecoder(this_exp.dim_input, 1, svm.LinearSVC)
        # dclf = LinearDecoder(this_exp.dim_input, D.ntot, svm.LinearSVC)
        
        # K = int(num_cond/2) - 1 # use all but one pairing
        # K = int(num_cond/4) # use half the pairings
        
        PS = np.zeros(D.ntot)
        CCGP = [] #np.zeros((D.ntot, 100))
        out_corr = []
        d = np.zeros((n_samp, D.ntot))
        pos_conds = []
        for j, pos in enumerate(D):
            pos_conds.append(pos)
            # print('Dichotomy %d...'%i)
            # parallelism
            PS[j] = D.parallelism(z.T, which_class, clf)
            
            # CCGP
            cntxt = [p for p in [(0,1), (0,2)] if p!=pos]
            CCGP.append(np.mean(D.CCGP(z.T, which_class, gclf, these_vars=cntxt,
                                       twosided=True, max_iter=1000)))
            
            # shattering
            d[:,j] = D.coloring(which_class)
        
        dclf.fit(z.T, d, tol=1e-3)
        
        SD = dclf.test(z.T, d).squeeze()
        
        these_ps.append(PS)
        these_ccgp.append(CCGP)
        these_sd.append(SD)

    all_PS[i,:] = np.mean(these_ps,0)
    all_CCGP[i,:] = np.mean(these_ccgp, 0)
    all_SD[i,:] = np.mean(these_sd, 0)

#%%
# columns = v_win
# rows = anti_mag

ps_lim = np.max(np.abs([all_PS.min(), all_PS.max()]))
for i in range(3):
    plt.subplot(3,3,i+1)
    plt.imshow(all_PS[:,i].reshape(n_tst,n_tst), extent=[0,1,1,0], cmap='bwr')
    plt.clim([-ps_lim, ps_lim])
    plt.title(['par','mag','XOR'][i])
    if i == 0:
        plt.ylabel('anti str')
        plt.yticks([0,1])
    else:
        plt.yticks([])
    plt.xticks([])
plt.colorbar()

ccgp_lim = np.max(np.abs([0.5-all_CCGP.min(), all_CCGP.max()-0.5]))
for i in range(3):
    plt.subplot(3,3,i+4)
    plt.imshow(all_CCGP[:,i].reshape(n_tst,n_tst), extent=[0,1,1,0], cmap='bwr')
    plt.clim([0.5-ccgp_lim,0.5+ccgp_lim])
    if i == 0:
        plt.ylabel('anti str')
        plt.yticks([0,1])
    else:
        plt.yticks([])
    plt.xticks([])
plt.colorbar()

for i in range(3):
    plt.subplot(3,3,i+7)
    plt.imshow(all_SD[:,i].reshape(n_tst,n_tst), extent=[0,1,1,0], cmap='binary')
    plt.clim([0.5,all_SD.max()])
    if i == 0:
        plt.ylabel('anti str')
        plt.yticks([0,1])
    else:
        plt.yticks([])
    plt.xlabel('noise str')
    plt.xticks([0,1])
plt.colorbar()

    
