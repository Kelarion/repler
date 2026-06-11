CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/zhang_data/' 

import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/sca/')

import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
 
from sklearn import svm, discriminant_analysis, manifold, linear_model
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
import scipy.io as scio

import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
# import cvxpy as cvx

# my code
import util
import df_util
import pt_util
import bae
import bae_models
import bae_search
import bae_util
import plotting as tpl

#%%

## The data doesn't label event information, so I had to figure it out by 
## inspecting and referencing against the wiki online
col_names = {0:"row_number", 
             1:"trial_start",
             2:"trial_end",
             3:"task",              ## 5 is category task, 6 is saccade task
             4:"cue_ID",          
             5:"target_1_ID",       ## if task is 6, then it's the same as the cue
             6:"target_1_pos",      ## location index from 1 to 20
             7:"target_1_x",        ## x coordinate
             8:"target_1_y",        ## y coordinate
             9:"target_2_ID",       ## if task is 6, then this is 9999
             10:"target_2_pos",
             11:"target_2_x",
             12:"target_2_y",
             13:"resp_ID",          ## final fixation, if task is 6 its the same as cue
             14:"resp_pos",
             15:"resp_x",
             16:"resp_y",
             17:"fix_start",
             18:"????",             ## no idea what this number is ...
             19:"cue_start",
             20:"cue_end",
             21:"delay_start",
             22:"delay_end",
             23:"search_start",
             24:"num_fix",          ## number of fixations
             25:"array_gone",       ## time when array disappears
             26:"reward_start",     ## reward time (correct_trials)
             27:"fix_break",        ## break time (error trials)
             28:"is_correct",       ## was the final fixation in the right category
             29:"type",             ## no idea
             }

cols = {v:k for k,v in col_names.items()}


#%%

import pandas as pd

def load_cell_data(file_path):
    parsed_data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines. Also cleanly handles any weird prefixes 
            # by slicing the string to start exactly at 'start:'
            if not line or 'start:' not in line:
                continue
            line = line[line.find('start:'):]
            
            parts = line.split(':')
            data_dict = {}
            
            i = 0
            while i < len(parts):
                key = parts[i]
                
                if key == 'start':
                    data_dict['unit_id'] = int(parts[i+1])
                    i += 1
                elif key == 'search':
                    data_dict['name_search'] = parts[i+1]
                    i += 1
                elif key == 'mapRF':
                    data_dict['name_mapRF'] = parts[i+1]
                    i += 1
                elif key == 'site':
                    # Site values contain an internal colon (e.g., +3-2:0320)
                    data_dict['atlas_coordinates'] = f"{parts[i+1]}:{parts[i+2]}"
                    i += 2
                elif key == 'V4':
                    # V4 is a standalone flag for the brain region
                    data_dict['brain_region'] = 'V4'
                elif key == 'sample':
                    data_dict['responsiveness_cue'] = parts[i+1]
                    i += 1
                elif key == 'array':
                    data_dict['responsiveness_search'] = parts[i+1]
                    i += 1
                elif key == 'prefer':
                    data_dict['category_selectivity'] = parts[i+1]
                    i += 1
                elif key == 'SI':
                    data_dict['selectivity_index'] = parts[i+1]
                    i += 1
                elif key == 'RFin':
                    data_dict['rf_foveal_peripheral'] = parts[i+1]
                    i += 1
                elif key == 'RFout':
                    data_dict['rf_peripheral_only'] = parts[i+1]
                    i += 1
                elif key == 'end':
                    break
                
                i += 1
                
            if data_dict:
                parsed_data.append(data_dict)
                
    # Convert list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(parsed_data)
    
    # Clean up data types: convert numeric columns from strings to floats
    # 'coerce' safely turns string 'NaN' into actual np.nan float values
    numeric_cols = ['responsiveness_cue', 'responsiveness_search', 'selectivity_index']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Replace literal 'NaN' strings in object columns with actual None/NaN
    df = df.replace('NaN', np.nan)
            
    return df

#%%

# start = "cue_start"
# start = "delay_end"
start = "delay_start"
# end = "delay_end"
# start = "reward_start"

# labs = ["resp_pos", "resp_ID"]
lab = 'cue_ID'

dt = 0.25

monkey = '6'

area = []
condition = []
neurons = []
session = []

for region in ['ofc', 'te', 'teo', 'v4', 'vpa']:

    df = load_cell_data(LOAD_DIR + f"RM00{monkey}/cell list/cell list {region}.txt")
    
    for neuron in tqdm(df['name_search']):
        
        sess, neu = re.findall(f"m0{monkey}cat(\d+)spk(\w+)", neuron)[0]
        
        try:
            neurdf = scio.loadmat(LOAD_DIR + f"RM00{monkey}/neurons/{neuron}.mat")
        except:
            print("oops")
            continue
        
        trl = neurdf['TrlInfoMatrix']
        
        deez = (trl[:,cols['task']]==5)         # category task
        deez *= (trl[:,cols['is_correct']]==1)  # correct trials 
        
        t0 = trl[:,cols[start]][deez]
        # t1 = trl[:,cols[end]][deez]
        
        bins = np.array([t0, t0 + dt]).T.flatten()
        labs = trl[:,cols[lab]][deez]
        
        unqs, ba = np.unique(labs, return_counts=True)
        if len(unqs) < 80: # or (np.min(ba) < 15):
            print('bad')
            continue
        
        trl_spikes = np.histogram(neurdf['neuron'][0], bins)[0][::2] / dt
        
        for this_lab in unqs:
            neurons.append(trl_spikes[labs == this_lab])
            condition.append(this_lab)
            area.append(region)
            session.append(sess)

neurons = np.array(neurons, dtype=object)
condition = np.array(condition)
area = np.array(area)
session = np.array(session)

# Xs = {k:np.array(v) for k,v in all_neurs.items()}

# all_neurs = {k1:{k2:np.array(v2).T for k2,v2 in v1.items()} for k1,v1 in all_neurs.items()}


# labels = {} 
# all_neurs = {}
# for neuron in tqdm(df['name_search']):
    
#     sess, neu = re.findall(f"m0{monkey}cat(\d+)spk(\w+)", neuron)[0]
    
#     try:
#         neurdf = scio.loadmat(LOAD_DIR + f"RM00{monkey}/neurons/{neuron}.mat")
#     except:
#         print("oops")
#         continue
    
#     trl = neurdf['TrlInfoMatrix']
    
#     deez = (trl[:,cols['task']]==5)         # category task
#     deez *= (trl[:,cols['is_correct']]==1)  # correct trials 
    
#     t0 = trl[:,cols[start]][deez]
#     # t1 = trl[:,cols[end]][deez]
    
#     bins = np.array([t0, t0 + dt]).T.flatten()
#     labels = trl[:,cols[lab]][deez]
    
#     if len(np.unique(labels)) < 80:
#         continue
    
#     if sess not in all_neurs.keys():
#         all_neurs[sess] = {}
#         # labels[sess] = [trl[:,cols[lab]][deez] for lab in labs]
    
#     trl_spikes = np.histogram(neurdf['neuron'][0], bins)[0][::2] / dt
    
#     for this_lab in np.unique(labels):
#         if int(this_lab) not in all_neurs[sess].keys():
#             all_neurs[sess][int(this_lab)] = []
#         all_neurs[sess][int(this_lab)].append(trl_spikes[labels == this_lab])

# # Xs = {k:np.array(v) for k,v in all_neurs.items()}

# all_neurs = {k1:{k2:np.array(v2).T for k2,v2 in v1.items()} for k1,v1 in all_neurs.items()}

#%%

pp = util.ppop([np.median(x,axis=0, keepdims=True)[:,None] for x in neurons], 
               condition, 
               session,
               subsets=area,
               independent=True, 
               K=1)

X = pp['data'][:,pp['neur_labels'] == 'vpa']

#%%

# X_ = X
X_ = X - X.mean(0) 
# X_ = (X / X.std(0)) - (X / X.std(0)).mean(0)

# mod = bae_models.SemiBMF(4,
#                          nonneg=True, 
#                          sparse_reg=1,
#                          weight_pr_reg=1, 
#                          tree_reg=1, 
#                          weight_l2_reg=1,
#                          )

# mod = bae_models.SpikeNMF(5,
#                          nonneg=True, 
#                          sparse_reg=1e-1, 
#                          weight_pr_reg=1, 
#                          tree_reg=1e-1, 
#                          weight_l2_reg=1,
#                          )

mod = bae_models.KernelBMF2(4,
                           sparse_reg=1,
                           tree_reg=1,
                           uniform_scale=False,
                           # l1_reg=1e-3,
                           )


en = mod.fit(X_ / X_.var(), 
             initial_temp=100,
             decay_rate=0.9,
             min_temp=1, 
             period=50, 
             scl_lr=1e-3,
             # scl_lr=1e-2,
             )

plt.figure()
samps = mod.sample(X_ / X_.var(), n_samp=1000)
# samps = mod.sample(X / X.std(), n_samp=1000, slab=False)

samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

# plt.imshow(samps.mean(0))
plt.imshow(samps.mean(0)[:,np.argsort(mod.scl)])


#%%

all_list_cells = [[s for s in
    load_cell_data(LOAD_DIR + f"RM00{monkey}/cell list old/cell list {region}.txt")['name_search'] 
    if 'cat' in s]
    for region in ['v4', 'te', 'teo', 'ofc',  'lpfc anterior', 'lpfc posterior']
                ]

# all_list_cells = np.unique(all_list_cells)

#%%

for X in Xs:    
    mavg = np.apply_along_axis(np.convolve, 1, X, np.ones(50)/50)
    deez = mavg[50:-50].min(0) > 0
    
#%%

# frac = 0.8 
frac = 1.0

ntrls = []
nneur = []
for this_sess in all_neurs.values():
    ntrls.append( [foo.shape[0] for foo in this_sess.values()] )
    nneur.append( [foo.shape[-1] for foo in this_sess.values()] )

ntrls = np.array(ntrls)
nneur = np.array(nneur)

## how many trials to sample from each condition?
# ntrnsamp = np.round(frac*ntrls.mean(0)).astype(int)
# ntstsamp = np.round((1-frac)*ntrls.mean(0)).astype(int)
ntrnsamp = int(np.round(frac*ntrls.mean())) * np.ones(ntrls.shape[1], dtype=int)
ntstsamp = int(np.round((1-frac)*ntrls.mean())) * np.ones(ntrls.shape[1], dtype=int)
# n_samp = ntrls.min(0)

ntrn = np.floor(frac*ntrls).astype(int)

pp_trn = {}
pp_tst = {}
for i, this_sess in enumerate(all_neurs.values()):
    for j, (lab, rep) in enumerate(this_sess.items()):
        
        trnset = np.random.choice(range(ntrls[i,j]), ntrn[i,j], replace=False)
        tstset = np.setdiff1d(range(ntrls[i,j]), trnset)
        
        ix_trn = np.random.choice(trnset, ntrnsamp[j])
        X_trn = rep[ix_trn]
    
        ix_tst = np.random.choice(tstset, ntstsamp[j])
        X_tst = rep[ix_tst]
        
        if lab in pp_trn.keys():
            pp_trn[lab] = np.concatenate([pp_trn[lab], X_trn], axis=1)
            pp_tst[lab] = np.concatenate([pp_tst[lab], X_tst], axis=1)
        else:
            pp_trn[lab] = X_trn
            pp_tst[lab] = X_tst

Xtrn = np.vstack([foo for foo in pp_trn.values()])
Xtst = np.vstack([foo for foo in pp_tst.values()])
Xtrn_grp = np.stack([foo.mean(0) for foo in pp_trn.values()])
Xtst_grp = np.stack([foo.mean(0) for foo in pp_tst.values()])
idx = np.concatenate([np.ones(len(wa), dtype=int)*i for i,wa in enumerate(pp_trn.values())])

labels = np.array(list(pp_trn.keys()))
