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
import old_bae
import old_bae_models
import old_bae_search
import bae_util
import plotting as tpl


#%%

# dset = '000232'
# dset = '000628'
dset = '000019'
# dset = 
t0 = 0
t1 = 0.5

filter_window = 100

phase = 'stimOnset_times'

subj = os.listdir(f"{LOAD_DIR}/{dset}")[1]

neurs = {}
 

these_sess = np.unique([re.findall('ses-([^_.]+)', sess)[0] for sess in os.listdir(f"{LOAD_DIR}/{dset}/{subj}/")])

for j,sess in enumerate(these_sess):
    
    # pathtoNWBFile = f"{LOAD_DIR}/{dset}/{subj}/{subj}_ses-{sess}_behavior.nwb"
    # pathtoNWBFile = f"{LOAD_DIR}/{dset}/{subj}/{subj}_ses-{sess}_image.nwb"
    pathtoNWBFile = f"{LOAD_DIR}/{dset}/{subj}/{subj}_ses-{sess}.nwb"
    
    nwbBasePath = Path(pathtoNWBFile)
    
    sess_id = re.findall('ses-(\d+)', sess)[0]
    
    io = NWBHDF5IO(str(nwbBasePath), mode='r')
    nwb = io.read()
    
    beh_df = nwb.trials.to_dataframe()
    # resp = beh_df['response']
    # legit = resp > 0
    block = np.unique(beh_df['blockType'], return_inverse=True)[1]
    trial = np.unique(beh_df['trialType'], return_inverse=True)[1]
    resp = np.unique(beh_df['response'], return_inverse=True)[1]
    
    
    unit_df = nwb.units.to_dataframe()
    
    n_neur = len(nwb.units.id)
    
    bin_cntr = beh_df[phase]
    
    bins = np.array([bin_cntr+t0, bin_cntr+t1]).T.flatten()
    
    X = []
    for neur in range(n_neur):
        X.append(np.histogram(nwb.units.get_unit_spike_times(neur), bins)[0][::2])
    X = np.array(X).T
    
    neurs[sess_id] = {}
    for lab in np.unique(labels[ontri]):
        neurs[sess_id][lab] = X[labels==lab]

        
        