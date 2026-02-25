CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/dandisets/' 

import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)

import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

import torch

from sklearn import svm, discriminant_analysis, manifold, linear_model
from sklearn.cluster import KMeans
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc

import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
# import cvxpy as cvx

import h5py

from dandi.download import download as dandi_download
from pynwb import NWBHDF5IO
from pathlib import Path

# my code
import util
import df_util
import pt_util
import bae
import bae_models
import bae_search
import bae_util
import plotting as tpl

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

def _str_to_bool(s: str) -> bool:
    """Convert a string to a boolean."""
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError(f"Cannot convert string {s} to boolean")

def _filter_trials_and_units(active, min_trl, min_units):
    """Get trials and units to remove.

    Args:
        presence: Boolean numpy array of shape (n_trials, n_units)
            indicating whether each unit was present in each trial.

    Returns:
        units_to_remove: Numpy array of shape (n_units,) indicating
            whether each unit should be removed.
        trials_to_remove: Numpy array of shape (n_trials,) indicating
            whether each trial should be
    """
    trials_per_unit = np.sum(active, axis=0)
    units_per_trial = np.sum(active, axis=1)
    units_to_remove = trials_per_unit < min_trl
    trials_to_remove = units_per_trial < min_units
    delta = np.sum(active[:, units_to_remove]) + np.sum(
        active[trials_to_remove, :]
    )
    if delta == 0:
        return units_to_remove, trials_to_remove
    else:
        active[:, units_to_remove] = False
        active[trials_to_remove, :] = False
        return _filter_trials_and_units(active, min_trl, min_units)

def sillouette(x, c):
    
    same = c[None] == c[:,None]
    diff = ~same
    
    d = np.abs(x[None] - x[:,None])
    a = (d*same).sum(1)/same.sum(1)
    b = (d*diff).sum(1)/diff.sum(1)
    
    return (b - a)/np.max([a,b], axis=0)

#%%

# pseudopop 

dset = '000004'
# dset = 
lbin = -0.25
rbin = 0.25

X_resp = {}
X_stim = {}

k = 0
trials = []
names = []
subses = []
for i,subj in tqdm(enumerate(os.listdir(f"{LOAD_DIR}/{dset}"))):
    
    sub_id = re.findall('sub-(.*)', subj)
    
    if len(sub_id) == 0:
        continue
    else:
        sub_id = sub_id[0]
    
    # os.makedirs(f"{SAVE_DIR}/{dset}/{sub_id}/")
    
    cats = []
    for j,sess in enumerate(os.listdir(f"{LOAD_DIR}/{dset}/{subj}/")):
        pathtoNWBFile = f"{LOAD_DIR}/{dset}/{subj}/{sess}"
        nwbBasePath = Path(pathtoNWBFile)
        
        sess_id = re.findall('ses-(\d+)', sess)[0]
        
        io = NWBHDF5IO(str(nwbBasePath), mode='r')
        nwb = io.read()
        
        stim_on = np.array(nwb.trials.stim_on_time.data)
        resp = np.array(nwb.trials.response_time.data)
        
        stim = np.array(nwb.trials.external_image_file.data)
        stim_cat = np.array(nwb.trials.stimCategory.data)
        # resp_val = np.array(nwb.trials.response_value.data)
        recog = np.array(nwb.trials.new_old_labels_recog.data)
        recog = np.unique(recog, return_inverse=True)[1]
        phase = np.array(nwb.trials.stim_phase.data)
        
        seen = ~np.isin(stim, stim[phase==b'learn'])
        # recog = np.array(nwb.trials.new_old_labels_recog.data)
        # recog = np.unique(recog, return_inverse=True)[1]

        trial_type = np.stack([stim_cat, recog, seen])
        
        trials.append(trial_type)
        names.append(np.array(nwb.trials.category_name.data))
        subses.append(np.ones(len(seen))*k)
        
        k += 1
        # unqs, idx = np.unique(trial_type, axis=1, return_inverse=True)
        
        # n_neur = len(nwb.units.origClusterID)
        
        # ## response-centered
        # bins = np.array([resp+lbin,resp+rbin]).T.flatten()

        # X = []
        # for neur in range(n_neur):
        #     X.append(np.histogram(nwb.units.get_unit_spike_times(neur), bins)[0][::2])

        
        # # X_resp.append(np.array(X).T)
        # # fname = f"{SAVE_DIR}/{dset}/{sub_id}/{sess_id}_resp_binned_{lbin+rbin}.npy"
        # # np.save(open(fname, 'wb'), np.array(X).T)         
                
        # ## stim-on
        # bins = np.array([stim_on+lbin,stim_on+rbin]).T.flatten()

        # X = []
        # for neur in range(n_neur):
        #     X.append(np.histogram(nwb.units.get_unit_spike_times(neur), bins)[0][::2])
        # X = np.array(X).T

        # # X_stim.append(np.array(X).T)
        # # fname = f"{SAVE_DIR}/{dset}/{sub_id}/{sess_id}_stim_on_binned_{lbin+rbin}.npy"
        # # np.save(open(fname, 'wb'), np.array(X).T)         
        
        # for i,trl in enumerate(unqs):
        #     if trl in X_stim.keys():
        #         X_stim[trl] 
            
        # cats.append(np.array(nwb.trials.stimCategory.data))
        
#%%

# dset = '000628'
# dset = '000575'
dset = '000620'
# dset = '001357'
# dset = 
# lbin = -0.25
# rbin = 0.25

modes = {'neur': 'spikesorting',
         'behavior': 'behavior+task'}

X = {}

subj = 'sub-Perle'

# bin_time = 'phase_cue_time'
bin_time = 'phase_delay_time'
# bin_time = 'phase_cue_time'
# t0 = -0.5
# t1 = 0
t0 = 0
t1 = 1

filter_window = 100

neurs = {}
is_pfc = []
# for i,subj in tqdm(enumerate(os.listdir(f"{LOAD_DIR}/{dset}"))):
    
    # sub_id = re.findall('sub-(.*)', subj)
    
    # if len(sub_id) == 0:
    #     continue
    # else:
    #     sub_id = sub_id[0]
    
    # os.makedirs(f"{SAVE_DIR}/{dset}/{sub_id}/")

these_sess = np.unique([re.findall('ses-(.*)_', sess)[0] for sess in os.listdir(f"{LOAD_DIR}/{dset}/{subj}/")])

for j,sess in tqdm(enumerate(these_sess)):
    
    ## behavior 
    
    pathtoNWBFile = f"{LOAD_DIR}/{dset}/{subj}/{subj}_ses-{sess}_{modes['behavior']}.nwb"
    nwbBasePath = Path(pathtoNWBFile)
    
    io = NWBHDF5IO(str(nwbBasePath), mode='r')
    nwb_beh = io.read()
    
    beh_df = nwb_beh.trials.to_dataframe()
    
    broke = beh_df.broke_fixation
    stim_pos = beh_df.stimulus_object_positions[~broke]
    stim_pos = stim_pos.apply(
            lambda x: eval(x)
        )
    stim_id = beh_df.stimulus_object_identities[~broke]
    stim_id = stim_id.apply(
            lambda x: [eval(s.strip()) for s in x[1:-1].split(",")]
        )
    stim_targ = beh_df.stimulus_object_target[~broke]
    stim_targ = stim_targ.apply(
            lambda x: [_str_to_bool(s.strip()) for s in x[1:-1].split(",")]
        )
    num_stim = stim_pos.apply(len).to_numpy()
    
    ## Triangle positions
    exes = np.cos(np.linspace(0,2*np.pi,4)[:3])*0.35 + 0.5
    whys = np.sin(np.linspace(0,2*np.pi,4)[:3])*0.35 + 0.5
    tris = np.stack([exes,whys]).T
    
    idx = stim_id.apply(lambda x: [ord(o) - ord('a') for o in x])
    sort_pos = np.zeros((len(idx),6))
    names = ['a', 'b', 'c']
    labels = np.chararray((len(idx),3))
    labels[:] = '-'
    for trl, (pos, ids) in enumerate(zip(stim_pos, idx)):
        for p,i in zip(pos,ids):
            sort_pos[trl, i*2:(i+1)*2] = p
            dtri = np.abs(tris - p).sum(1)
            if np.min(dtri) < 1e-5:
                labels[trl, np.argmin(dtri)] = names[i]
                
    labels = labels[:,0] + labels[:,1] + labels[:,2]
    labels = labels.astype(str)
    ontri = labels != '---'
    
    if len(np.unique(labels[ontri])) < 33:
        continue
    
    # get target info
    targ_id = []
    targ_pos = []
    for pos, targ, ids in zip(stim_pos[ontri], stim_targ[ontri], stim_id[ontri]):
        targ_id.append(ids[np.argmax(targ)])
        targ_pos.append(pos[np.argmax(targ)])
    
    bin_cntr = np.array(nwb_beh.trials[bin_time].data)[~broke]
    
    ## neural
    
    pathtoNWBFile = f"{LOAD_DIR}/{dset}/{subj}/{subj}_ses-{sess}_{modes['neur']}.nwb"
    nwbBasePath = Path(pathtoNWBFile)
    
    io = NWBHDF5IO(str(nwbBasePath), mode='r')
    nwb_neu = io.read()
    units = nwb_neu.processing["ecephys"]["units"]
    units_df = units.to_dataframe()
    
    n_neur = len(units.spike_times_index)
    
    bins = np.array([bin_cntr+t0, bin_cntr+t1]).T.flatten()
    
    X = []
    for neur in range(n_neur):
        X.append(np.histogram(units.get_unit_spike_times(neur), bins)[0][::2])
    X = np.array(X).T

    ## Filter neurons
    badneu, badtrl = _filter_trials_and_units(X>0, 200, 5)
    
    mavg = np.apply_along_axis(np.convolve, 0, X, np.ones(filter_window)/filter_window)
    deez = mavg[filter_window:-filter_window].min(0) > 0
    
    keep = np.isin(units_df.quality, ['good', 'mua'])*deez*(~badneu)
    
    X = X[~badtrl][:,keep]
    labels = labels[~badtrl]
    ontri = ontri[~badtrl]
    
    is_pfc.append((units_df['electrodes_group'] == 's0').to_numpy()[keep])
    
    # X = X / np.sqrt(np.mean((X-X.mean(0))**2))
    
    ## Prepare for pseudopop
    neurs[sess] = {}
    for lab in np.unique(labels[ontri]):
        neurs[sess][lab] = X[labels==lab]

is_pfc = np.concatenate(is_pfc)

#%%

# frac = 0.8 
frac = 1.0

ntrls = []
nneur = []
for this_sess in neurs.values():
    ntrls.append( [len(foo) for foo in this_sess.values()] )
    nneur.append( [len(foo.T) for foo in this_sess.values()] )

ntrls = np.array(ntrls)
nneur = np.array(nneur)

## how many trials to sample from each condition?
ntrnsamp = np.round(frac*ntrls.mean(0)).astype(int)
ntstsamp = np.round((1-frac)*ntrls.mean(0)).astype(int)
# n_samp = ntrls.min(0)

ntrn = np.floor(frac*ntrls).astype(int)

pp_trn = {}
pp_tst = {}
for i, this_sess in enumerate(neurs.values()):
    for j, (lab, rep) in enumerate(this_sess.items()):
        
        trnset = np.random.choice(range(ntrls[i,j]), ntrn[i,j], replace=False)
        tstset = np.setdiff1d(range(ntrls[i,j]), trnset)
        
        ix_trn = np.random.choice(trnset, ntrnsamp[j])
        X_trn = rep[ix_trn]
        
        ix_tst = np.random.choice(tstset, ntstsamp[j])
        X_tst = rep[ix_tst]
        
        if lab in pp_trn.keys():
            pp_trn[lab] = np.hstack([pp_trn[lab], X_trn])
            pp_tst[lab] = np.hstack([pp_tst[lab], X_tst])
        else:
            pp_trn[lab] = X_trn
            pp_tst[lab] = X_tst

Xtrn = np.vstack([foo for foo in pp_trn.values()])
Xtst = np.vstack([foo for foo in pp_tst.values()])
Xtrn_grp = np.stack([foo.mean(0) for foo in pp_trn.values()])
Xtst_grp = np.stack([foo.mean(0) for foo in pp_tst.values()])
idx = np.concatenate([np.ones(len(wa))*i for i,wa in enumerate(pp_trn.values())])

Xtrn_grp = Xtrn_grp / np.sqrt(np.mean((Xtrn_grp - Xtrn_grp.mean(0))**2))
Xtst_grp = Xtst_grp / np.sqrt(np.mean((Xtst_grp - Xtst_grp.mean(0))**2))

labels = np.char.array(list(pp_trn.keys()))

#%%

kays = np.arange(2,15)

args = {'nonneg':True,
        'sparse_reg': 2,
        'tree_reg': 1e-2,
        # 'fit_intercept': False,
        }

opt_args = {'initial_temp': 10,
            'decay_rate': 0.9,
            'period': 50}

n_run = 10

trn = []
tst = []
for _ in range(n_run):
    tr = []
    ts = []
    for k in tqdm(kays):
        
        # mod = bae_models.BiPCA(k, sparse_reg=1e-4)
        mod = bae_models.SemiBMF(k, **args)
        
        # wa,ba = bae_util.impcv(mod, X, verbose=True, **opt_args)
        wa,ba = bae_util.impcv(mod, Xtrn_grp[:,is_pfc], verbose=False, **opt_args)
        
        tr.append(1*wa)
        ts.append(1*ba)
    trn.append(tr)
    tst.append(ts)

plt.plot(kays, np.mean(trn,axis=0))
plt.plot(kays, np.mean(tst, axis=0))

#%%

n_chain = 20
# k = 50
k = 10

args = {'nonneg':True,
        'sparse_reg': 2,
        'tree_reg': 1e-2,
        'weight_pr_reg': 1e-1,
        # 'fit_intercept': False,
        }

opt_args = {'initial_temp': 10,
            'decay_rate': 0.9,
            'period': 50}

mods = [bae_models.SemiBMF(k, **args) for i in range(n_chain)]
ens = [m.fit(Xtrn_grp[:,is_pfc], **opt_args) for m in mods]

allW = np.hstack([m.W for m in mods])
allS = np.hstack([m.S for m in mods])

#%%
K = k

kmn = KMeans(K)
kmn.fit(allW.T)
# kmn.fit(allS.T)

avg = util.group_mean(allS, kmn.labels_)
quality = np.argsort(np.log2(avg+1e-6).mean(0))

plt.imshow(avg[:,quality], cmap='binary')


#%%

# G = 10 # number of groups
# N = 50 # elements per group

# A = sprs.eye(G**2).tocsr()[np.eye(G).flatten()==0]
# B = util.bipartite_incidence(np.ones((N,N)))[0]

# incs = sprs.kron(A, B, 'csr')

# ## Use "special" indices to facilitate construction of constraints
# idx = np.tile(np.arange(N**2).reshape((N,N)), (G,G))
# idx += (N**2)*np.kron(np.arange(G**2).reshape((G,G)), np.ones((N,N), dtype=int))
# xdi = np.argsort(idx.flatten())

# ## Full bipartite incidence
# Inc, edges = util.bipartite_incidence(np.ones((N*G,N*G)))
# Inc = Inc.tocsr()[:,xdi]

# ## Form cursed LP
# A_eq = sprs.bmat([[Inc], [incs]])
# b_eq = np.concatenate([(G-1)*np.ones(2*N*G), np.ones(2*N*G*(G-1))])
# c = cost.flatten()[xdi]
