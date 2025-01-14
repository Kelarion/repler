CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
sys.path.append(CODE_DIR)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import permutations, combinations
import itertools as itt
from tqdm import tqdm

from sklearn import svm, discriminant_analysis, manifold, linear_model
import scipy.stats as sts
import scipy.linalg as la
import scipy.spatial as spt
import scipy.sparse as sprs
import scipy.special as spc
from scipy.optimize import linear_sum_assignment as lsa
from scipy.optimize import linprog as lp
from sklearn.manifold import MDS

import students
import assistants
import experiments as exp
import util
import pt_util
import tasks
import plotting as dicplt
import dichotomies as dics

import distance_factorization as df

#%%

class HebbFB(students.NeuralNet):
    
    def __init__(self, n_inp, n_hid, n_out, tau_a=0.5, tau_f=0.5, eta_a=1, eta_f=1):
        
        super(HebbFB,self).__init__()
        
        self.taua = tau_a
        self.tauf = tau_f
        
        self.etaa = eta_a
        self.etaf = eta_f
        
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        
        # fixed weights
        self.W = nn.Parameter(torch.randn(1, n_hid, n_inp)/np.sqrt(n_hid))
        self.J = nn.Parameter(torch.randn(1, n_out, n_hid)/np.sqrt(n_out))
        
        # plastic weights
        self.init_plastic()
        
        # self.f = nn.ReLU()
        # self.f = nn.Tanh()
        self.f = nn.Identity()
        
        # self.error = nn.BCEWithLogitsLoss(reduction='none')
        self.error = nn.MSELoss(reduction='none')
        
    def init_plastic(self):
        
        # self.A = torch.zeros(1, self.n_hid, self.n_inp, requires_grad=False)
        self.F = torch.zeros(1, self.n_hid, 2*self.n_out, requires_grad=False)
        self.A = torch.randn(1, self.n_hid, self.n_inp, requires_grad=False)
        
    def forward(self, X, Y, tau_a=None, tau_f=None, eta_a=None, eta_f=None):
        """
        X and Y are (time, batch, dim)
        """
        
        if tau_a is None:
            tau_a = self.taua
        if tau_f is None:
            tau_f = self.tauf
        if eta_a is None:
            eta_a = self.etaa
        if eta_f is None:
            eta_f = self.etaf
            
        T, P, _ = Y.shape
        
        self.init_plastic()
        
        X_ = X[...,None] # add extra dimension for batching
        # Y_ = 2*Y[...,None]-1
        Y_ = torch.cat([Y, 1-Y], dim=-1)[...,None]
        
        Y_hat = torch.zeros(T, P, self.n_out)
            
        for t in range(T):
            
            c = self.W@X_[t]
            z = self.f((self.W + self.A)@X_[t])
            y_hat = self.J@z
            z_hat = self.F@Y_[t]
            
            with torch.no_grad(): # don't pass gradients through plasticity
                eligible = 1*(y_hat*(2*Y[t][...,None]-1) > 0)
                # err = (nn.Sigmoid()(y_hat) - Y[t][...,None])**2
                err = self.error(nn.Sigmoid()(y_hat), Y[t][...,None])
                # print(err.max())
                
                # print(z_hat.shape)
                zxT = torch.einsum('ki...,kj...->kij', z_hat-c, X_[t])
                self.A = self.A + err*(eta_a*zxT - (1-tau_a)*self.A)
                # self.A = self.A + (eta_a*zxT - (1-tau_a)*self.A)
                # self.A = tau_a*self.A + (1-eligible)*eta_a*zxT
                # self.A /= (1+self.A.norm(dim=(1,2), keepdim=True))
                
                zyT = torch.einsum('ki...,kj...->kij', z, Y_[t])
                # zyT = torch.einsum('ki...,kj...->kij', z-z_hat, Y_[t])
                # self.F = tau_f*self.F + eligible*eta_f*zyT
                self.F = self.F + eligible*(eta_f*zyT - (1-tau_f)*self.F)
                # self.F /= (1+self.F.norm(dim=(1,2), keepdim=True))

            Y_hat[t] = y_hat[...,0]
        
        return Y_hat
    
    def hidden(self, X, Y, tau_a=None, tau_f=None, eta_a=None, eta_f=None):
        """
        X and Y are (time, batch, dim)
        """
        
        if tau_a is None:
            tau_a = self.taua
        if tau_f is None:
            tau_f = self.tauf
        if eta_a is None:
            eta_a = self.etaa
        if eta_f is None:
            eta_f = self.etaf
            
        T, P, _ = Y.shape
        
        self.init_plastic()
        
        X_ = X[...,None] # add extra dimension for batching
        # Y_ = 2*Y[...,None]-1
        Y_ = torch.cat([Y, 1-Y], dim=-1)[...,None]
        
        Y_hat = torch.zeros(T, P, self.n_out)
        Z = torch.zeros(T, P, self.n_hid)
        Z_hat = torch.zeros(T, P, self.n_hid)
        A = torch.zeros(T, P, self.n_hid, self.n_inp)
        F = torch.zeros(T, P, self.n_hid, 2*self.n_out)
        
        for t in range(T):
            
            z = self.f((self.W + self.A)@X_[t])
            y_hat = self.J@z
            z_hat = self.F@Y_[t]
            
            with torch.no_grad(): # don't pass gradients through plasticity
                # eligible = nn.Sigmoid()(y_hat*(2*Y[t][...,None]-1))
                eligible = 1*(y_hat*(2*Y[t][...,None]-1) > 0)
                err = (nn.Sigmoid()(y_hat) - Y[t][...,None])**2
                
                # print(z_hat.shape)
                zxT = torch.einsum('ki...,kj...->kij', z_hat-z, X_[t])
                # self.A = self.A + (1-eligible)*(eta_a*zxT - (1-tau_a)*self.A)
                self.A = self.A + err*(eta_a*zxT - (1-tau_a)*self.A)
                # self.A /= (1+self.A.norm(dim=(1,2), keepdim=True))
            
                zyT = torch.einsum('ki...,kj...->kij', z, Y_[t])
                # zyT = torch.einsum('ki...,kj...->kij', z-z_hat, Y_[t])
                # self.F = tau_f*self.F + eligible*eta_f*zyT
                self.F = self.F + eligible*(eta_f*zyT - (1-tau_f)*self.F)
            
            Y_hat[t] = y_hat[...,0]
            Z[t] = z[...,0]
            Z_hat[t] = z_hat[...,0]
            A[t] = 1*self.A
            F[t] = 1*self.F
        
        return Z, Z_hat, Y_hat, A, F
    
    def grad_step(self, dl, **fwd_args):
        
        if not self.initialized:
            self.init_optimizer()
        
        running_loss = 0
        
        for i, batch in enumerate(dl):
            
            self.optimizer.zero_grad()
            
            X = batch[0].transpose(0,1)
            Y = batch[1].transpose(0,1)
            
            # reshape
            out = self(X, Y, **fwd_args)
            
            loss = nn.BCEWithLogitsLoss()(out, Y)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss/(i+1)
        
    
#%% transitive inference task like sam's

n_item = 3
n_ctx = 2

M = n_item*n_ctx
T = 40
n_trial = 1000

# left, right = np.where(np.ones((n_item,n_item))-np.eye(n_item))
# num_cond = len(left)

# X = np.hstack([np.eye(M)[left], np.eye(M)[right]])  # dim_X x num_cond
# Y = np.triu(np.ones((M,M)), 1)[left, right][:,None] # dim_Y x num_cond

# cond = np.random.choice(range(num_cond), size=(T, n_trial))

# X_cond = torch.tensor(X[cond]).float()
# Y_cond = torch.tensor(Y[cond]).float()

left, right = np.where(np.ones((M,M))-np.eye(M))
adj = np.abs(left - right) < 2 # train on adjacent items
is_trn = (left < n_item)*(right < n_item)*adj
is_tst = (left >= n_item)*(right >= n_item)*adj

num_cond = len(left)

X = np.hstack([np.eye(M)[left], np.eye(M)[right]])  # dim_X x num_cond
Y = np.triu(np.ones((M,M)), 1)[left, right][:,None] # dim_Y x num_cond

cond = np.random.choice(np.arange(num_cond)[is_trn], size=(T, n_trial))

# cond2_a = np.random.choice(np.arange(num_cond)[is_trn], size=(T//2, n_trial))
# cond2_b = np.random.choice(np.arange(num_cond)[is_tst], size=(1, n_trial))
# cond2 = np.vstack([cond2_a, np.repeat(cond2_b, T-T//2, axis=0)])
cond2_a = np.random.choice(np.arange(num_cond)[is_trn], size=(T//2, n_trial))
cond2_b = np.random.choice(np.arange(num_cond)[is_tst], size=(T-T//2, n_trial))
cond2 = np.vstack([cond2_a, cond2_b])

X_cond = torch.tensor(X[cond]).float()
Y_cond = torch.tensor(Y[cond]).float()

X2_cond = torch.tensor(X[cond2]).float()
Y2_cond = torch.tensor(Y[cond2]).float()

num_pres = (np.eye(num_cond)[cond2].cumsum(0)*np.eye(num_cond)[cond2]).max(-1)

#%% define network

net = HebbFB(2*M, 100, 1, eta_a=1, eta_f=1, tau_f=1, tau_a=1)

net.init_optimizer()

#%% train network

n_epoch = 50

dl = pt_util.batch_data(X_cond.transpose(0,1), Y_cond.transpose(0,1), 
                        batch_size=32, shuffle=True)

crit = nn.BCEWithLogitsLoss(reduction='none')

loss = []
test_loss = []
test_loss_fx = []
test_perf = []
for epoch in tqdm(range(n_epoch)):
    
    # ls = net.grad_step(dl, eta_a=0, eta_f=0)
    ls = net.grad_step(dl)
    loss.append(ls)
    
    outs = net(X2_cond, Y2_cond)
    tst = crit(outs[T//2:,:,0], Y2_cond[T//2:,:,0]).detach().numpy()
    perf = 1*(outs[T//2:,:,0]*(2*Y2_cond[T//2:,:,0] - 1) > 0).detach().numpy()
    tst_n = []
    prf_n = []
    for n in range(1,5):
        tst_n.append(tst[num_pres[T//2:]==n].mean())
        prf_n.append(perf[num_pres[T//2:]==n].mean())
    test_loss.append(tst_n)
    test_perf.append(prf_n)
    
    outs_fx = net(X2_cond, Y2_cond, eta_a=0, eta_f=0)
    tst_fx = nn.BCEWithLogitsLoss()(outs_fx[-1], Y2_cond[-1])
    test_loss_fx.append(tst_fx.item())

# plt.plot(test_loss)
plt.plot(test_perf)


