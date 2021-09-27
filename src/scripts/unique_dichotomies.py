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
import itertools as itt
from sklearn import svm, manifold, linear_model
from tqdm import tqdm

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import experiments as exp

#%%
def face(a,b,c):
    return a

def corner(a,b,c):
    return a*b + a*c + b*c

def snake(a,b,c):
    return a*c + b*(~c)

def net(a,b,c):
    return ~(a*c + a*b + ~(a+b+c))

def xor2(a,b,c):
    return a^b

def xor3(a,b,c):
    return a^b^c

#%%

# dics = util.RandomDichotomies(8,1)

X = np.mod(np.arange(8)[:,None]//(2**np.arange(3)[None,:]),2)

fig = plt.figure()
for i,d in enumerate(assistants.Dichotomies(8)):
    # colorby = inp_condition
    # colorby = util.decimal(outputs).numpy()
    colorby = np.isin(np.arange(8), d)
    
    # whichone = int('57%d'%(i+1))
    ax = fig.add_subplot(5,7,i+1, projection='3d')
    
    ax.scatter(X[:,0],X[:,1],X[:,2],s=500, c=colorby)
    # for i in np.unique(these_conds):
    #     c = [int(i), int(np.mod(i+1,U.shape[0]))]
    #     ax.plot(U[c,0],U[c,1],U[c,2],'k')
    ax.set_title(d)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    util.set_axes_equal(ax)

#%%

X = np.mod(np.arange(8)[:,None]//(2**np.arange(3)[None,:]),2).astype(bool)

# logic = face
logic = corner
# logic = snake
# logic = net
# logic = xor2
# logic = xor3

all_colorings = []
perm = []

fig = plt.figure()
i = 0
for p in permutations(range(3)):
    # colorby = inp_condition
    # colorby = util.decimal(outputs).numpy()
    # colorby = np.isin(np.arange(8), d)
    # d = X[:,p]

    d = X[:,p]
    colorby = logic(d[:,0],d[:,1],d[:,2])
    if not colorby[0]:
    # if True:
        all_colorings.append(colorby)
        perm.append(np.array(p)+1)
    
    # whichone = int('57%d'%(i+1))
    ax = fig.add_subplot(6,8,i+1, projection='3d')
    
    ax.scatter(X[:,0],X[:,1],X[:,2],s=200, c=colorby)
    # for i in np.unique(these_conds):
    #     c = [int(i), int(np.mod(i+1,U.shape[0]))]
    #     ax.plot(U[c,0],U[c,1],U[c,2],'k')
    # ax.set_title(d)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    util.set_axes_equal(ax)
    i += 1
    
    for c in itt.chain(combinations(range(3),1),combinations(range(3),2),combinations(range(3),3)):
        d = X[:,p]
        d[:,c] = 1 - d[:,c]
        
        colorby = logic(d[:,0],d[:,1],d[:,2])
        if not colorby[0]:
        # if True:
            all_colorings.append(colorby)
            foo = np.array(p)+1
            foo[np.array(c)] *= -1
            perm.append(foo)
        
        # whichone = int('57%d'%(i+1))
        ax = fig.add_subplot(6,8,i+1, projection='3d')
        
        ax.scatter(X[:,0],X[:,1],X[:,2],s=200, c=colorby)
        # for i in np.unique(these_conds):
        #     c = [int(i), int(np.mod(i+1,U.shape[0]))]
        #     ax.plot(U[c,0],U[c,1],U[c,2],'k')
        # ax.set_title(d)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        util.set_axes_equal(ax)
        i += 1


cols = np.array(all_colorings)
perms = np.array(perm)

