CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'

import sys

from sklearn import gaussian_process as gp
from sklearn import svm
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from tqdm import tqdm


#%%

dim = 50
num_var = 2
num_nonlin = 3
ndat = 3000
num_test = 50

clf = svm.LinearSVC()

fake_labels =  2*np.random.rand(ndat,num_var)-1   
basis = la.qr(np.random.rand(dim, dim))[0]

CCG = []
CV = []
for sigma in tqdm(np.logspace(-5,2,100)):

    # fake_labels = (np.random.rand(100, 2)>0.5).astype(int)
    # fake_labels = np.stack([np.random.rand(ndat)>0.5, 2*np.random.rand(ndat)-1]).T
    
    coords = gp.GaussianProcessRegressor(gp.kernels.RBF(1/sigma))
    
    # ys = np.stack([coords.sample_y(fake_labels[:,i,None], n_samples=1) for i in range(num_var)]).squeeze()
    ys = coords.sample_y(fake_labels, n_samples=dim)
    ys -= ys.mean(0)
    
    rep = fake_labels@basis[:2,:] + ys
    
    ccg = []
    cv = []
    for i in range(num_test):
        part_dir = np.random.randn(num_var,1)
        part_dir /= la.norm(part_dir)
        
        ctx_dir = np.random.randn(num_var,1)
        ctx_dir -= (ctx_dir.T@part_dir)*part_dir
        ctx_dir /= la.norm(ctx_dir)
        
        labs = np.squeeze((fake_labels-fake_labels.mean(0))@part_dir > 0)
        
        trn_set = np.squeeze((fake_labels-fake_labels.mean(0))@ctx_dir > 0)
        tst_set = 1-trn_set
        
        clf.fit(rep[trn_set,:], labs[trn_set])
        ccg.append(clf.score(rep[tst_set,:],labs[tst_set]))
        
        trn_set_cv = np.random.permutation(trn_set)
        tst_set_cv = 1- trn_set_cv
        clf.fit(rep[trn_set_cv,:], labs[trn_set_cv])
        cv.append(clf.score(rep[tst_set_cv,:],labs[tst_set_cv]))
    
    CCG.append(ccg)
    CV.append(cv)

    