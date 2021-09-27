CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as anime
from itertools import permutations, combinations
import itertools as itt
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la

# import umap
from cycler import cycler

import students
import assistants
import experiments as exp
import util
import tasks
import plotting as dicplt

#%%

num_cond = 5
num_var = 1

# which_task = 'mnist'
# which_task = 'mog'
which_task = 'structured'


nonlinearity = 'ReLU'
# nonlinearity = 'Tanh'
# nonlinearity = 'LeakyReLU'

num_layer = 0

# nudges = np.concatenate([np.arange(0,0.21,0.01), np.arange(0.2, 1.1, 0.1)])
# nudges = np.arange(0,0.21,0.01)
nudges = np.arange(0,1.1,0.1)

H = 100

N_list = [100]

# basins = list(itt.chain(*([combinations(range(4), i) for i in range(1,3)])))
basins = [[0],[1],[2],[3],[0,3],[1,2]]

agg_nets = []
agg_mets = []
agg_args = []
pred_prob = []
actual_prob = []
which_basin = []
lin_dim = []
ps = []
weights_proj = []
for r in nudges:
    # inp_task = tasks.StandardBinary(np.log2(num_cond))
    # inp_task = tasks.TwistedCube(tasks.StandardBinary(2), 100, f=rotation, noise_var=0.1)
    inp_task = tasks.NudgedXOR(tasks.StandardBinary(2), 100, nudge_mag=np.round(r,2), noise_var=0.1, random=False)
    # task = tasks.LogicalFunctions(d=decs, function_class=num_var)
    # task = tasks.RandomDichotomies(d=[(0,1,3,5),(0,2,3,6),(0,1,2,4)])
    task = tasks.RandomDichotomies(d=[(0,3)])
    this_exp = exp.structured_inputs(task, input_task=inp_task,
                                      SAVE_DIR=SAVE_DIR,
                                      noise_var=0.1,
                                      num_layer=num_layer,
                                      weight_decay=0,
                                      nonlinearity=nonlinearity)
    num_var = task.num_var + inp_task.num_var

    this_folder = SAVE_DIR + this_exp.folder_hierarchy()
    
    all_nets, mets, all_args = this_exp.aggregate_nets(SAVE_DIR, N_list)
    
    wba = []
    ppro = []
    apro = []
    wp = []
    for args, met, net in zip(all_args[0], mets[0], all_nets[0]):
        this_exp.load_other_info(args)
        
        x_pos = la.block_diag(*np.diff(args['class_means'][:2],axis=1).squeeze().tolist()).T
        x_pos = np.concatenate([x_pos,args['class_means'][2].flatten()[:,None]], axis=1)
        x_pos /= la.norm(x_pos,axis=0, keepdims=True)
        
        x_ = this_exp.input_task(np.arange(4), noise=0).T
        
        basin_prototype = np.stack([x_[:,p].sum(1) for p in basins]).T
        apro.append([np.sum((net.enc.network.layer0.weight.detach().numpy()@basin_prototype).argmax(1)==4),
                    np.sum((net.enc.network.layer0.weight.detach().numpy()@basin_prototype).argmax(1)==5)])
        
        wba.append((net.enc.network.layer0.weight.detach().numpy()@basin_prototype).argmax(1))
        
        ppro.append([np.arccos(-util.cosine_sim(x_[:,p],x_[:,p])[0,1])/(np.pi*2) for p in [[0,3],[1,2]]])

        wp.append(net.enc.network.layer0.weight.detach().numpy()@x_pos)

    which_basin.append(wba)
    pred_prob.append(ppro)
    actual_prob.append(apro)
    weights_proj.append(wp)
    
    agg_nets.append(all_nets)
    agg_mets.append(mets)
    agg_args.append(all_args)
    

#%%






