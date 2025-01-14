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

# class HebbFB(students.NeuralNet):
    
#     def __init__(self, n_inp, n_hid, n_out, activation=pt_util.Iden(),
#                   tau_a=0.5, tau_f=0.5, tau_j=0, 
#                   eta_a=1, eta_f=1, eta_j=0):
        
#         super(HebbFB,self).__init__()
        
#         self.taua = tau_a
#         self.tauf = tau_f
        
#         self.etaa = eta_a
#         self.etaf = eta_f
#         self.etaj = eta_j
        
#         self.n_inp = n_inp
#         self.n_hid = n_hid
#         self.n_out = n_out
        
#         # plastic weights
#         self.init_plastic()
        
#         # self.f = nn.ReLU()
#         # self.f = nn.Tanh()
#         self.f = activation
        
#         # self.error = nn.BCEWithLogitsLoss(reduction='none')
#         self.error = nn.MSELoss(reduction='none')
        
#     def init_plastic(self):
        
#         self.A = torch.zeros(1, self.n_hid, self.n_inp, requires_grad=False)
#         self.F = torch.zeros(1, self.n_hid, 2*self.n_out)
#         # self.A = torch.randn(1, self.n_hid, self.n_inp)/np.sqrt(self.n_hid)
#         self.J = torch.randn(1, self.n_out, self.n_hid)/np.sqrt(self.n_out)
        
#     def forward(self, X, Y, initialize=True, tau_a=None, tau_f=None, eta_a=None, eta_f=None):
#         """
#         X and Y are (time, batch, dim)
#         """
        
#         if tau_a is None:
#             tau_a = self.taua
#         if tau_f is None:
#             tau_f = self.tauf
#         if eta_a is None:
#             eta_a = self.etaa
#         if eta_f is None:
#             eta_f = self.etaf
        
#         if initialize:
#             self.init_plastic()
        
#         T, P, _ = Y.shape
        
#         X_ = X[...,None] # add extra dimension for batching
#         # Y_ = 2*Y[...,None]-1
#         Y_ = torch.cat([Y, 1-Y], dim=-1)[...,None]
        
#         Y_hat = torch.zeros(T, P, self.n_out)
            
#         for t in range(T):
            
            
#             c = self.A@X_[t]
#             e = torch.randn(P, self.n_hid, 1)
            
#             z = self.f(e + c)
#             f_1 = self.f.deriv(e + c)
            
#             # z = self.f((self.W + self.A)@X_[t])
#             y_hat = self.J@z
#             z_hat = self.f(self.F@Y_[t])
#             f_2 = self.f.deriv(z_hat)
            
#             with torch.no_grad(): # don't pass gradients through plasticity
#                 eligible = 1*(y_hat*(2*Y[t][...,None]-1) > 0)
#                 # err = (nn.Sigmoid()(y_hat) - Y[t][...,None])**2
#                 # err = self.error(nn.Sigmoid()(y_hat), Y[t][...,None])
#                 err = (Y[t][...,None]-nn.Sigmoid()(y_hat))
                
#                 # train output weights like GLM
#                 ecT = torch.einsum('ki...,kj...->kij', err, c)
#                 self.J = self.J + self.etaj*ecT
                
#                 # print(z_hat.shape)
#                 # zxT = torch.einsum('ki...,kj...->kij', z_hat-c, X_[t])
#                 zxT = torch.einsum('ki...,kj...->kij', z_hat, X_[t])
#                 self.A = self.A + (err**2)*(eta_a*zxT - (1-tau_a)*self.A)
#                 # self.A = self.A + (eta_a*zxT - (1-tau_a)*self.A)
#                 # self.A = tau_a*self.A + (1-eligible)*eta_a*zxT
#                 # self.A /= (1+self.A.norm(dim=(1,2), keepdim=True))
                
#                 zyT = torch.einsum('ki...,kj...->kij', (z-c), Y_[t])
#                 # zyT = torch.einsum('ki...,kj...->kij', z-z_hat, Y_[t])
#                 # self.F = tau_f*self.F + eligible*eta_f*zyT
#                 self.F = self.F + eligible*(eta_f*zyT - (1-tau_f)*self.F)
#                 # self.F /= (1+self.F.norm(dim=(1,2), keepdim=True))

#             Y_hat[t] = y_hat[...,0]
        
#         return Y_hat
    
#     def hidden(self, X, Y, **args):
#         """
#         X and Y are (time, batch, dim)
#         """
        
#         T, P, _ = Y.shape
        
#         Y_hat = torch.zeros(T, P, self.n_out)
#         A = torch.zeros(T, P, *self.A.shape[1:])
#         F = torch.zeros(T, P, *self.F.shape[1:])
#         J = torch.zeros(T, P, *self.J.shape[1:])
        
#         self.init_plastic()
        
#         for t in range(T):
            
#             yhat = self(X[[t]], Y[[t]], initialize=False, **args)
            
#             Y_hat[t] = yhat[0,...]
#             A[t] = 1*self.A
#             F[t] = 1*self.F
#             J[t] = 1*self.J
        
#         return Y_hat, A, F, J
    
#     def grad_step(self, dl, **fwd_args):
        
#         if not self.initialized:
#             self.init_optimizer()
        
#         running_loss = 0
        
#         for i, batch in enumerate(dl):
            
#             self.optimizer.zero_grad()
            
#             X = batch[0].transpose(0,1)
#             Y = batch[1].transpose(0,1)
            
#             # reshape
#             out = self(X, Y, **fwd_args)
            
#             loss = nn.BCEWithLogitsLoss()(out, Y)
            
#             loss.backward()
#             self.optimizer.step()
            
#             running_loss += loss.item()
        
#         return running_loss/(i+1)

# class HebbFB(students.NeuralNet):
    
#     def __init__(self, n_inp, n_hid, n_out, activation=pt_util.Iden(),
#                   tau_a=0.5, tau_f=0.5, tau_j=0, 
#                   eta_a=1, eta_f=1, eta_j=0):
        
#         super(HebbFB,self).__init__()
        
#         self.taua = tau_a
#         self.tauf = tau_f
        
#         self.etaa = eta_a
#         self.etaf = eta_f
#         self.etaj = eta_j
        
#         self.n_inp = n_inp
#         self.n_hid = n_hid
#         self.n_out = n_out
        
#         # plastic weights
#         self.init_plastic()
        
#         # self.f = nn.ReLU()
#         # self.f = nn.Tanh()
#         self.f = activation
        
#         # self.error = nn.BCEWithLogitsLoss(reduction='none')
#         self.error = nn.MSELoss(reduction='none')
        
#     def init_plastic(self):
        
#         self.W = torch.randn(1, self.n_hid, self.n_inp)/np.sqrt(self.n_hid)
#         self.F1 = torch.zeros(1, self.n_hid, self.n_inp, requires_grad=False)
#         self.F2 = torch.randn(1, self.n_out, self.n_hid)/np.sqrt(self.n_out)
#         self.B = torch.zeros(1, self.n_hid, 2*self.n_out)
        
#     def forward(self, X, Y, initialize=True, tau_a=None, tau_f=None, eta_a=None, eta_f=None):
#         """
#         X and Y are (time, batch, dim)
#         """
        
#         if tau_a is None:
#             tau_a = self.taua
#         if tau_f is None:
#             tau_f = self.tauf
#         if eta_a is None:
#             eta_a = self.etaa
#         if eta_f is None:
#             eta_f = self.etaf
        
#         if initialize:
#             self.init_plastic()
        
#         T, P, _ = Y.shape
        
#         X_ = X[...,None] # add extra dimension for batching
#         # Y_ = 2*Y[...,None]-1
#         Y_ = torch.cat([Y, 1-Y], dim=-1)[...,None]
        
#         Y_hat = torch.zeros(T, P, self.n_out)
            
#         for t in range(T):
            
            
#             Fx = self.F1@X_[t]
#             # e = torch.randn(P, self.n_hid, 1)
#             Wx = self.W@X_[t]
            
#             z = self.f(Wx + Fx)
#             f_1 = self.f.deriv(Wx + Fx)
            
#             # z = self.f((self.W + self.A)@X_[t])
#             y_hat = self.F2@z
#             z_hat = self.f(self.B@Y_[t])
#             f_2 = self.f.deriv(z_hat)
            
#             with torch.no_grad(): # don't pass gradients through plasticity
#                 eligible = 1*(y_hat*(2*Y[t][...,None]-1) > 0)
#                 # err = (nn.Sigmoid()(y_hat) - Y[t][...,None])**2
#                 # err = self.error(nn.Sigmoid()(y_hat), Y[t][...,None])
#                 err = (Y[t][...,None]-nn.Sigmoid()(y_hat))
                
#                 # train output weights like GLM
#                 ecT = torch.einsum('ki...,kj...->kij', err, z)
#                 self.F2 = self.F2 + self.etaj*ecT
                
#                 # print(z_hat.shape)
#                 # zxT = torch.einsum('ki...,kj...->kij', z_hat-c, X_[t])
#                 zxT = torch.einsum('ki...,kj...->kij', z_hat, X_[t]) 
#                 # zF = torch.einsum('ki...,kij...->kij', z_hat**2, self.F1)
#                 self.F1 = self.F1 + (err**2)*(eta_a*zxT - (1-tau_a)*self.F1)
#                 # self.F1 = self.F1 + (err**2)*eta_a*(zxT - zF) # oja's
#                 # self.A = self.A + (eta_a*zxT - (1-tau_a)*self.A)
#                 # self.A = tau_a*self.A + (1-eligible)*eta_a*zxT
#                 # self.A /= (1+self.A.norm(dim=(1,2), keepdim=True))
                
#                 zyT = torch.einsum('ki...,kj...->kij', z, Y_[t])
#                 # zyT = torch.einsum('ki...,kj...->kij', z-z_hat, Y_[t])
#                 # self.F = tau_f*self.F + eligible*eta_f*zyT
#                 self.B = self.B + eligible*(eta_f*zyT - (1-tau_f)*self.B)
#                 # self.F /= (1+self.F.norm(dim=(1,2), keepdim=True))

#             Y_hat[t] = y_hat[...,0]
        
#         return Y_hat
    
#     def hidden(self, X, Y, **args):
#         """
#         X and Y are (time, batch, dim)
#         """
        
#         T, P, _ = Y.shape
        
#         Y_hat = torch.zeros(T, P, self.n_out)
#         F1 = torch.zeros(T, P, *self.F1.shape[1:])
#         B = torch.zeros(T, P, *self.B.shape[1:])
#         F2 = torch.zeros(T, P, *self.F2.shape[1:])
        
#         self.init_plastic()
        
#         for t in range(T):
            
#             yhat = self(X[[t]], Y[[t]], initialize=False, **args)
            
#             Y_hat[t] = yhat[0,...]
#             F1[t] = 1*self.F1
#             B[t] = 1*self.B
#             F2[t] = 1*self.F2
        
#         return Y_hat, F1, B, F2
    
#     def grad_step(self, dl, **fwd_args):
        
#         if not self.initialized:
#             self.init_optimizer()
        
#         running_loss = 0
        
#         for i, batch in enumerate(dl):
            
#             self.optimizer.zero_grad()
            
#             X = batch[0].transpose(0,1)
#             Y = batch[1].transpose(0,1)
            
#             # reshape
#             out = self(X, Y, **fwd_args)
            
#             loss = nn.BCEWithLogitsLoss()(out, Y)
            
#             loss.backward()
#             self.optimizer.step()
            
#             running_loss += loss.item()
        
#         return running_loss/(i+1)


class HebbFB(students.NeuralNet):
    
    def __init__(self, n_inp, n_hid, n_out, activation=pt_util.Iden(),
                  tau_a=0.5, tau_f=0.5, tau_j=0, 
                  eta_a=1, eta_f=1, eta_j=0):
        
        super(HebbFB,self).__init__()
        
        self.taua = tau_a
        self.tauf = tau_f
        
        self.etaa = eta_a
        self.etaf = eta_f
        self.etaj = eta_j
        
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        
        # plastic weights
        self.init_plastic()
        
        # self.f = nn.ReLU()
        # self.f = nn.Tanh()
        self.f = activation
        
        # self.error = nn.BCEWithLogitsLoss(reduction='none')
        self.error = nn.MSELoss(reduction='none')
        
    def init_plastic(self):
        
        self.W = torch.randn(1, self.n_hid, self.n_inp)/np.sqrt(self.n_hid)
        self.F1 = torch.zeros(1, self.n_hid, self.n_inp, requires_grad=False)
        self.F2 = torch.randn(1, self.n_out, self.n_hid)/np.sqrt(self.n_out)
        self.B = torch.zeros(1, self.n_hid, self.n_out)
        
    def forward(self, X, Y, initialize=True, tau_a=None, tau_f=None, eta_a=None, eta_f=None):
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
        
        if initialize:
            self.init_plastic()
        
        T, P, _ = Y.shape
        
        X_ = X[...,None] # add extra dimension for batching
        # Y_ = 2*Y[...,None]-1
        Y_ = torch.cat([Y, 1-Y], dim=-1)[...,None]
        
        Y_hat = torch.zeros(T, P, self.n_out)
            
        for t in range(T):
            
            Fx = self.F1@X_[t]
            # e = torch.randn(P, self.n_hid, 1)
            Wx = self.W@X_[t]
            
            z = self.f(Wx + Fx)
            f_1 = self.f.deriv(Wx + Fx)
            
            # z = self.f((self.W + self.A)@X_[t])
            y_hat = self.F2@z
            
            err = (Y[t][...,None]-nn.Sigmoid()(y_hat))
            z_hat = self.f(self.B@err)
            f_2 = self.f.deriv(z_hat)
            
            with torch.no_grad(): # don't pass gradients through plasticity
                eligible = 1*(y_hat*(2*Y[t][...,None]-1) > 0)
                # err = (nn.Sigmoid()(y_hat) - Y[t][...,None])**2
                # err = self.error(nn.Sigmoid()(y_hat), Y[t][...,None])
                
                # train output weights like GLM
                ecT = torch.einsum('ki...,kj...->kij', err, z)
                self.F2 = self.F2 + self.etaj*ecT
                
                # print(z_hat.shape)
                zxT = torch.einsum('ki...,kj...->kij', z_hat, X_[t])
                # zxT = torch.einsum('ki...,kj...->kij', z_hat, X_[t]) 
                # zF = torch.einsum('ki...,kij...->kij', z_hat**2, self.F1)
                self.F1 = self.F1 + (eta_a*zxT - (1-tau_a)*self.F1)
                # self.F1 = self.F1 + (err**2)*eta_a*(zxT - zF) # oja's
                # self.A = self.A + (eta_a*zxT - (1-tau_a)*self.A)
                # self.A = tau_a*self.A + (1-eligible)*eta_a*zxT
                # self.A /= (1+self.A.norm(dim=(1,2), keepdim=True))
                
                zyT = torch.einsum('ki...,kj...->kij', z, err)
                # zyT = torch.einsum('ki...,kj...->kij', z-z_hat, Y_[t])
                # self.F = tau_f*self.F + eligible*eta_f*zyT
                self.B = self.B + eligible*(eta_f*zyT - (1-tau_f)*self.B)
                # self.F /= (1+self.F.norm(dim=(1,2), keepdim=True))

            Y_hat[t] = y_hat[...,0]
        
        return Y_hat
    
    def hidden(self, X, Y, **args):
        """
        X and Y are (time, batch, dim)
        """
        
        T, P, _ = Y.shape
        
        Y_hat = torch.zeros(T, P, self.n_out)
        F1 = torch.zeros(T, P, *self.F1.shape[1:])
        B = torch.zeros(T, P, *self.B.shape[1:])
        F2 = torch.zeros(T, P, *self.F2.shape[1:])
        
        self.init_plastic()
        
        for t in range(T):
            
            yhat = self(X[[t]], Y[[t]], initialize=False, **args)
            
            Y_hat[t] = yhat[0,...]
            F1[t] = 1*self.F1
            B[t] = 1*self.B
            F2[t] = 1*self.F2
        
        return Y_hat, F1, B, F2
    
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


def lr_loss(y, yhat):
    mu = spc.expit(yhat)
    return y*np.log(mu+ 1e-16) + (1-y)*np.log(1-mu+ 1e-16)
    
#%% define task

ntrial = 100
nsess = 1000

# X = np.eye(2)
# Y = np.array([0,1])
# X = np.hstack([2*util.F2(2)-1, np.eye(4)])
# X = np.eye(4)
X = 2*util.F2(2)-1
Y = np.array([0,1,1,0])

cond1 = np.random.choice(range(2), (ntrial,nsess))
cond2 = np.random.choice(range(2,4), (ntrial,nsess))
cond = np.vstack([cond1, cond2, cond1, cond2]) 

Xcond = torch.tensor(X[cond]).float()
Ycond = torch.tensor(Y[cond,None]).float()

#%% define network

net = HebbFB(X.shape[1], 100, 1,
             eta_a=0.7, eta_f=0.4, tau_f=1, tau_a=0.8, eta_j=0.1)
# net = HebbFB(X.shape[1], 100, 1, activation=pt_util.RayLou(),
#               eta_a=0.3, eta_f=0.7,
#               tau_f=1, tau_a=0.8, eta_j=1)


wa = np.squeeze(net(Xcond, Ycond).detach())
plt.plot(nn.Sigmoid()(wa*(2*Y[cond]-1)).mean(-1))


#%% logistic regression
    
lr = 1

ls = []
perf = []
beta = np.zeros((nsess, X.shape[1]))

for i in cond:
    
    yhat = np.sum(beta*X[i], axis=1)
    mu = spc.expit(yhat)
    
    grad = (Y[i] - mu)[:,None]*X[i]
    beta += lr*grad
    
    ls.append(Y[i]*np.log(mu) + (1-Y[i])*np.log(1-mu))
    perf.append(spc.expit(yhat*(2*Y[i]-1)))

plt.plot(np.mean(perf, axis=1))
  

#%% simple transitive inference task like sams

n_item = 3
n_ctx = 2

ntrial = 200
nsess = 1000

M = n_item*n_ctx


left, right = np.where(np.ones((M,M))-np.eye(M))
adj = np.abs(left - right) < 2 # train on adjacent items
grp1 = (left < n_item)*(right < n_item)
grp2 = (left >= n_item)*(right >= n_item)

num_cond = len(left)
allcond = np.arange(num_cond)

X = np.hstack([np.eye(M)[left], np.eye(M)[right]])  # dim_X x num_cond
Y = np.triu(np.ones((M,M)), 1)[left, right] # dim_Y x num_cond

cond1 = np.random.choice(allcond[grp1*adj], (ntrial,nsess))
cond2 = np.random.choice(allcond[grp2*adj], (ntrial,nsess))
cond = np.vstack([cond1, cond2, cond1, cond2]) 

Xcond = torch.tensor(X[cond]).float()
Ycond = torch.tensor(Y[cond,None]).float()

#%% define network

# net = HebbFB(X.shape[1], 100, 1, eta_a=0.4, eta_f=1, tau_f=1, tau_a=0.9, eta_j=1)
net = HebbFB(X.shape[1], 100, 1, activation=pt_util.RayLou(),
             eta_a=0.4, eta_f=1, tau_f=1, tau_a=0.9, eta_j=1)

Yhat = np.squeeze(net(Xcond, Ycond).detach())

plt.plot(spc.expit(Yhat*(2*Y[cond]-1)).mean(-1))

# Yhat, A, F, J = net.hidden(Xcond, Ycond)
# Yhat = np.squeeze(Yhat.detach().numpy())

# plt.plot(spc.expit(Yhat*(2*Y[cond]-1)).mean(-1))

# test = np.squeeze((J@A)@X[grp2*(~adj)].T)
# plt.plot(spc.expit(test*(2*Y[grp2*(~adj)]-1)).mean((-1,-2)))



#%% logistic regression

lr = 1

ls = []
perf = []
tst_perf = []
beta = np.zeros((nsess, X.shape[1]))

for i in cond:
    
    yhat = np.sum(beta*X[i], axis=1)
    mu = spc.expit(yhat)
    
    grad = (Y[i] - mu)[:,None]*X[i]
    beta += lr*grad
    
    ls.append(Y[i]*np.log(mu) + (1-Y[i])*np.log(1-mu))
    perf.append(spc.expit(yhat*(2*Y[i]-1)))
    
    yhat_tst = beta@X[grp2*(~adj)].T
    y_tst = Y[grp2*(~adj)]
    tst_perf.append(spc.expit(yhat_tst*(2*y_tst-1)))
    
plt.plot(np.mean(perf, axis=1))
    
#%% transverse patterning

n_item = 3
n_ctx = 2

ntrial = 200
nsess = 1000

M = n_item*n_ctx

left, right = np.where(la.block_diag(*([np.ones((n_item,n_item))-np.eye(n_item)]*n_ctx)))
adj = np.abs(left - right) < 2 # train on adjacent items
grp1 = (left < n_item)*(right < n_item)
grp2 = (left >= n_item)*(right >= n_item)

num_cond = len(left)
allcond = np.arange(num_cond)

X = np.hstack([np.eye(M)[left], np.eye(M)[right]])  # dim_X x num_cond
Y = la.block_diag(*([np.roll(np.eye(3),1,axis=1)]*n_ctx))[left, right] # dim_Y x num_cond

cond1 = np.random.choice(allcond[grp1], (ntrial,nsess))
cond2 = np.random.choice(allcond[grp2*adj], (ntrial,nsess))
cond = np.vstack([cond1, cond2, cond1, cond2]) 

Xcond = torch.tensor(X[cond]).float()
Ycond = torch.tensor(Y[cond,None]).float()

#%% train and plot

net = HebbFB(X.shape[1], 100, 1, activation = pt_util.RayLou(),
             eta_a=0.4, eta_f=1, 
             tau_f=1, tau_a=0.4, eta_j=1)

Yhat = np.squeeze(net(Xcond, Ycond).detach())

plt.plot(spc.expit(Yhat*(2*Y[cond]-1)).mean(-1))

# Yhat, A, F, J = net.hidden(Xcond, Ycond)
# Yhat = np.squeeze(Yhat.detach().numpy())

# plt.plot(spc.expit(Yhat*(2*Y[cond]-1)).mean(-1))

# test = np.squeeze((J@A)@X[grp2*(~adj)].T)
# plt.plot(spc.expit(test*(2*Y[grp2*(~adj)]-1)).mean((-1,-2)))

#%%

dl_trn = pt_util.batch_data(torch.Tensor(X[grp1]), torch.Tensor(Y[grp1,None]))
dl_tst = pt_util.batch_data(torch.Tensor(X[grp2*adj]), torch.Tensor(Y[grp2*adj,None]))

test_net = students.ShallowNetwork(X.shape[1], 100, 1, 
                                   pt_util.RayLou(), students.Bernoulli)

W_lr = 1
J_lr = 0

num_trn = 200
num_tst = 200

ls = []
perf = []
tst_perf = []
# beta = np.zeros((nsess, X.shape[1]))

for _ in range(num_trn):
    
    loss = test_net.grad_step(dl_trn, W_lr=W_lr, J_lr=J_lr)
    yhat = np.squeeze(test_net(torch.Tensor(X[grp1]))[0].detach().numpy())
    
    ls.append(loss)
    perf.append(spc.expit(yhat*(2*Y[grp1]-1)).mean())
    
    tst_perf.append(0)

for _ in range(num_tst):
    
    loss = test_net.grad_step(dl_tst, W_lr=W_lr, J_lr=J_lr)
    yhat = np.squeeze(test_net(torch.Tensor(X[grp2*adj]))[0].detach())
    
    ls.append(loss)
    perf.append(spc.expit(yhat*(2*Y[grp2*adj]-1)).mean(0)) 
    
    yhat_tst = np.squeeze(test_net(torch.Tensor(X[grp2*(~adj)]))[0].detach())
    tst_perf.append(spc.expit(yhat_tst*(2*Y[grp2*(~adj)]-1)).mean())
