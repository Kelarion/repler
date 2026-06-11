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
import torch.nn as nn
import torch.optim as optim
 
from sklearn import svm, discriminant_analysis, manifold, linear_model
from sklearn.cluster import KMeans, AgglomerativeClustering
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

#%%

dset = '000239'
# dset = '000628'
# dset = '001187' 

phase = 'start_time'

t0 = 0
t1 = 1

subj = os.listdir(f"{LOAD_DIR}/{dset}")[2]

# these_sess = np.unique([re.findall('ses-([^_.]+)', sess)[0] for sess in os.listdir(f"{LOAD_DIR}/{dset}/{subj}/")])

for j,sess in enumerate(os.listdir(f"{LOAD_DIR}/{dset}/{subj}/")):
    
    if 'behavior' not in sess:
        continue
    
    pathtoNWBFile = f"{LOAD_DIR}/{dset}/{subj}/{sess}"
    
    nwbBasePath = Path(pathtoNWBFile)

    io = NWBHDF5IO(str(nwbBasePath), mode='r')
    nwb = io.read()
    
    beh_df = nwb.trials.to_dataframe()
    unit_df = nwb.units.to_dataframe()
    
    n_neur = len(nwb.units.id)
    
    bin_cntr = nwb.trials[phase]

    bins = np.array([bin_cntr+t0, bin_cntr+t1]).T.flatten()
    
    X = []
    for neur in range(n_neur):
        X.append(np.histogram(unit_df.spike_times[neur], bins)[0][::2])
    X = np.array(X).T
    
    neurs[sess_id] = {}
    for lab in np.unique(labels[ontri]):
        neurs[sess_id][lab] = X[labels==lab]

