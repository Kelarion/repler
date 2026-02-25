CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/'
 
import os, sys, re
import pickle as pkl
from time import time
import math
sys.path.append(CODE_DIR)
sys.path.append('C:/Users/mmall/OneDrive/Documents/github/nsd_access/')

import numpy as np
import numpy.linalg as nla
from itertools import permutations, combinations
from tqdm import tqdm
from dataclasses import dataclass
import pickle as pkl

import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import ToTensor
from PIL import Image
import transformers 

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc

import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
# import cvxpy as cvx

import nibabel as nib
import h5py

from nsd_access import NSDAccess

# my code
import util
import df_util
import pt_util
import bae
import bae_models
import bae_search
import bae_util
import plotting as tpl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

nsda = NSDAccess(SAVE_DIR+'nsd')

#%% Load image categories and convert to a sparse matrix

cats = nsda.read_image_coco_category(range(73_000))

#%%

# weights = ResNet50_Weights.DEFAULT
# pp = weights.transforms()

# mod = resnet50(weights=weights)

# mod.eval()

# # layers = ['maxpool','layer1', 'layer2', 'layer3', 'layer4', 'avgpool']


#%%

clip = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

mod = clip.vision_model
mod.to(device)

mod.eval()

#%%

# layers = ['embeddings','encoder', 'post_layernorm']

layers = ['post_layernorm']

activations = {}
def get_activation(name):
    def hook(model, input, output):
        if hasattr(output, 'last_hidden_state'):
            activations[name] = output.last_hidden_state.detach()
        else:
            activations[name] = output.detach()
    return hook

handles = {}
for layer in layers:
    handles[layer] = getattr(mod, layer).register_forward_hook(get_activation(layer))

#%%

X = []
for i in tqdm(range(73_000)):
    
    im = nsda.read_images([i])

    ba = Image.fromarray(im[0])
    wa = processor(images=ba, return_tensors='pt')['pixel_values']

    with torch.no_grad():
        output = mod(wa.to(device))
        
    X.append(activations['post_layernorm'].squeeze())

for handle in handles.values():
    handle.remove()

#%%



#%% buncha bullshit

fold = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/nsd/nsddata/ppdata/'
save_dir = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'

this_roi = 5

# idx = []
# neurs = []
for subject in range(1,9):
    
    fname = fold + 'subj0%d/func1pt8mm/roi/streams.nii.gz'%subject
    streams = nib.load(fname).get_fdata()
    
    # ix = []
    ns = []
    for session in tqdm(range(1,41)):
        
        try:
            # beh = nsda.read_behavior('subj0%d'%subject, session)
            
            betas = nsda.read_betas('subj0%d'%subject, session)
            
            ns.append(betas[streams==this_roi])
            
            # del(betas)

        except: 
            print('no session')
        # foo = nsda.read_betas('subj01', i)
        
        # ix.append(beh['73KID'].to_numpy())

    np.save(open(save_dir+'subj0%d_roi%d.npy'%(subject, this_roi), 'wb'),np.hstack(ns))
    # np.save(open(save_dir+'subj0%d_trials.npy'%subject, 'wb'),np.concatenate(ix))
    
    # neurs.append(np.hstack(ns))

    # idx.append(np.unique(np.concatenate(ix)))
# 
# shared = np.array(list(set.intersection(*[set(ix) for ix in idx])))


#%%


def impcv(model, X, mask='random', folds=10, max_iter=20, verbose=False):
    """
    Imputation-based cross validation 

    the `fold` of the CV is the fraction of the data masked

    optimizer should already be initialized
    """

    if verbose:
        pbar = tqdm(range(max_iter))

    n,d = X.shape
    idx = np.arange(n*d).reshape((n,d))
    ntest = int(n*d/folds)

    Z = 1*X

    these_idx = np.random.choice(idx.flatten(), ntest, replace=False)
    M = np.isin(idx, these_idx)

    ## Initialize masked values from `empirical distribution`
    Z[M] = np.random.choice(Z[~M], M.sum())

    for it in range(max_iter):
        feats = model.fit_transform(X)
        Z[M] = (feats@model.components_)[M]
        
        if verbose:
            pbar.update()

    pred = feats @ model.components_
    test = np.mean((X[M] - pred[M])**2)/np.mean(X[M]**2)
    train = np.mean((X[~M] - pred[~M])**2)/np.mean(X[~M]**2)

    return train, test

#%%

kays = [2,5,10,50,100,150,200,250]
# kays = [100,200,300,400]
# kays = [5,10,50,100,200]

trn = []
tst = []
for k in kays:
    
    # mod = SparsePCA(k, max_iter=5)
    mod = NMF(k, l1_ratio=1, alpha_W=1e-3, max_iter=1)
    
    wa,ba = impcv(mod, X, max_iter=200, verbose=True, folds=10)
    
    trn.append(1*wa)
    tst.append(1*ba)
    
#%%

kays = [2,5,10,50,100,150,200,250]
# kays = [5,10,50,100,200,300,400]
# kays = []
# kays = np.arange(25,125,5)

trn = []
tst = []
for k in kays:
    
    # mod = bae_models.BiPCA(k, sparse_reg=1e-4)
    mod = bae_models.SemiBMF(k, nonneg=True, tree_reg=1e-2, weight_pr_reg=0.1)
    
    wa,ba = bae_util.impcv(mod, X, decay_rate=0.95, initial_temp=10, verbose=True)
    
    trn.append(1*wa)
    tst.append(1*ba)
    
#%%


