

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:31:38 2020

@author: matteo
"""

CODE_DIR = '/home/matteo/Documents/github/'

import sys

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(CODE_DIR+'repler/src/')
from students import Feedforward 
from assistants import Indicator
from hyperbole.hyperbolic_utils import Acosh
from hyperbole.hyperbolic_utils import LorentzDot
from hyperbole.hyperbolic_utils import GeodesicCoordinates

#%%
def expmap(w, v):
    """Exponential map on the hyperboloid"""
    # w_norm = LorentzDot.apply(w, w).sqrt()
    v_norm = LorentzDot.apply(v, v).sqrt()
    return torch.cosh(v_norm)*w + np.sin(v_norm)*(v/v_norm)

# def geochart(u, v):
#     """Geodesic coordinates of the 2-sphere, p.a.l."""
#     cv = torch.cos(2*np.pi*v)
#     cu = torch.cos(np.pi*u - np.pi/2)
#     sv = torch.sin(2*np.pi*v)
#     su = torch.sin(np.pi*u - np.pi/2)
#     return torch.stack((cv*cu, sv*cu, su)).T
def geochart(u, v):
    """Geodesic coordinates of the hyperboloid"""
    cv = torch.cosh(v)
    cu = torch.cosh(u)
    sv = torch.sinh(v)
    su = torch.sinh(u)
    return torch.stack((cu*cv, cv*su, sv)).T
# def geochart(u, v):
#     """Geodesic coordinates of the hyperboloid"""
#     cv = torch.cosh(v)
#     cu = torch.cosh(u)
#     sv = torch.sinh(v)
#     su = torch.sinh(u)
#     return torch.stack((su*cv, su*sv, cu)).T


def invgeochart(w):
    """Geodesic coordinates of the 2-sphere, p.a.l."""
    # u = torch.asin(w[...,2])
    v = np.arcsinh(w[...,2])
    # v = torch.acos(w[...,0]/torch.cos(u))
    u = np.arctanh(w[...,1]/w[...,0])
    return np.stack((u,v))

def ginverse(p, x):
    """Inverse metric at p, applied to x"""
    x_ = x.data
    coshv = torch.cosh(p.narrow(-1,1,1)).pow(2)
    x_.narrow(-1,0,1).mul_(coshv.pow(-1))
    return x_
    
def dist_hyp(w1, w2, eps=1e-5):
    """Distance between w1 and w2 on the 2-sphere"""
    theta = LorentzDot.apply(w1, w2)
    return Acosh.apply(-theta, eps)


#%%
dim = 3
k = 1

# generate a random convex loss function
C = np.random.randn(dim,dim)
C = C@C.T # random positive definite matrix

l, v = la.eig(C)
w_min = k*v[:,np.real(l).argmin()] # the unit vector the minimises the quadratic form
w_max = k*v[:,np.real(l).argmax()] # the unit vector the minimises the quadratic form

w_test = np.random.randn(dim, 1000)
w_test/=la.norm(w_test,2,0)
L = np.sum(w_test*(C@w_test),0)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# scat = ax.scatter(w_test[0,:],w_test[1,:],w_test[2,:], c=L)

#%%
nplot = 50
x = np.linspace(-2,2,nplot) # u
# x = 1
y = np.linspace(-2,2,nplot) # v
# y = 1
X,Y = np.meshgrid(x, y) # grid of point

u_test = torch.tensor(np.vstack((X.flatten(),Y.flatten())))

w_map = geochart(u_test[0,:], u_test[1,:]).numpy().T

w0 = torch.tensor(np.random.randn(dim-1))
w0 = geochart(w0[0], w0[1])
# w0 /= torch.sqrt(-LorentzDot.apply(w0,w0))
L = torch.sin(2*dist_hyp(w0, torch.tensor(w_map).T))
# L = np.sum(w_map*(C@w_map),0)
# L = loss(torch.tensor(w_map).float())



fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
scat = ax1.scatter(w_map[0,:],w_map[1,:],w_map[2,:], c=L)
ax1.set_title('On the sphere')

ax2 = fig.add_subplot(122)
ax2.pcolormesh(X,Y, L.reshape((nplot,nplot)))
ax2.set_title('Geodesic coordinates')

#%%
nstep = 1000
dt = 0.01

init = torch.tensor(np.random.randn(dim-1))

u = Feedforward([1,2], [None],
                encoder=Indicator(1, 1), bias=False)
u.network.layer0.weight.data.copy_(init.unsqueeze(1))

u_met = Feedforward([1,2], [None],
                encoder=Indicator(1, 1), bias=False)
u_met.network.layer0.weight.data.copy_(init.unsqueeze(1))

hype = GeodesicCoordinates(u_met)

# U = nn.Embedding(1, 3)
emb = u(torch.zeros(1).long())
w = geochart(init[0], init[1]).unsqueeze(1).float()
U = torch.tensor(w.T, requires_grad=True)

optimizer = optim.SGD(u.parameters(), lr=dt)
optimizer_met = optim.SGD(u_met.parameters(), lr=dt)
optimizer_met = optim.SGD(hype.parameters(), lr=dt)

ws = np.zeros((dim, nstep))
Ws = np.zeros((dim, nstep))
ws_met = np.zeros((dim, nstep))
us = np.zeros((dim-1, nstep))
Us = np.zeros((dim-1, nstep))
us_met = np.zeros((dim-1, nstep))
gradu = np.zeros((dim-1, nstep))
gradw = np.zeros((dim, nstep))
gradW = np.zeros((dim, nstep))
grad_met = np.zeros((dim-1, nstep))
for t in range(nstep):
    # SGD on the coordinates
    optimizer.zero_grad()
    emb = u(torch.zeros(1).long())
    emb.retain_grad()
    w = geochart(emb[:,0].T, emb[:,1].T).T
    w.retain_grad()
    
    ws[:,t] = w.detach().numpy().squeeze()
    us[:,t] = emb.data.numpy().squeeze()
    
    loss = torch.sin(2*dist_hyp(w.T, w0))
    loss.backward()
    
    gradu[:,t] = emb.grad.data.numpy().squeeze()
    gradw[:,t] = w.grad.data.numpy().squeeze()
    # grads[:,t] = u.weight.grad.data.numpy().squeeze()
    optimizer.step()
    
    # SGD on the coordinates with Riemannian gradients
    optimizer_met.zero_grad()
    # emb = u_met(torch.zeros(1).long())
    # emb.retain_grad()
    # w = geochart(emb[:,0].T, emb[:,1].T).T
    # w.retain_grad()
    
    # ws_met[:,t] = w.detach().numpy().squeeze()
    # us_met[:,t] = emb.data.numpy().squeeze()
    
    # loss = torch.sin(2*dist_hyp(w.T, w0))
    # loss.backward()
    
    # rgrad = ginverse(u_met.network.layer0.weight.data.T, 
    #                   u_met.network.layer0.weight.grad.T)
    # u_met.network.layer0.weight.grad.data.copy_(rgrad.T)
    
    w = hype(torch.zeros(1).long())
    
    ws_met[:,t] = w.detach().numpy().squeeze()
    us_met[:,t] = hype.invchart(w).data.numpy().squeeze()
    
    loss = torch.sin(2*dist_hyp(w, w0))
    loss.backward()
    
    # grad_met[:,t] = rgrad.numpy().squeeze()emb
    # gradw[:,t] = w.grad.data.numpy().squeeze()
    # grads[:,t] = u.weight.grad.data.numpy().squeeze()
    optimizer_met.step()
    
    # Riemannian SGD
    # w2 = U(torch.zeros(1).long())
    # w2.retain_grad()
    loss2 = torch.sin(2*dist_hyp(U, w0))
    loss2.backward()
    
    Ws[:,t] = U.data.numpy().squeeze()
    Us[:,t] = invgeochart(U.data.numpy()).squeeze()
    
    dL = U.grad.data
    dL.narrow(-1,0,1).mul_(-1)
    # gradL = U@A - (U@A@U.data.T)*U.data
    gradL = dL + LorentzDot.apply(dL, U.data)*U.data
    gradW[:,t] = gradL.data.numpy().squeeze()
    new = expmap(U.data, -dt*gradL)
    U.data.copy_(torch.tensor(new))
    U.grad.zero_()
    

#%%
nplot = 50
path_cmap = 'cool'
man_cmap = 'spring'
riem_cmap = 'hot'
loss_cmap = 'Blues'

# x = np.linspace(0,1,nplot)
# y = np.linspace(0,1,nplot)
x = np.linspace(-2,2,nplot)
y = np.linspace(-2,2,nplot)
X,Y = np.meshgrid(x, y) # grid of point

u_test = torch.tensor(np.vstack((X.flatten(),Y.flatten())))

w_map = geochart(u_test[0,:], u_test[1,:]).T.numpy()
# L = np.sum(w_map*(C@w_map),0)
L = torch.sin(dist_hyp(w0, torch.tensor(w_map).T)*2)
col = getattr(cm,loss_cmap)(L.reshape((nplot,nplot))/L.max())

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(w_map[1,:].reshape((nplot,nplot)),
                 w_map[2,:].reshape((nplot,nplot)),
                 w_map[0,:].reshape((nplot,nplot)), 
                 facecolors=col, linewidth=0, zorder=0)
geo = ax1.scatter(0.9*ws[1,:], 0.9*ws[2,:], 0.9*ws[0,:], s=20, 
                  c=np.log(np.arange(1,nstep+1)), 
                  cmap=path_cmap, zorder=10)
man = ax1.scatter(0.9*Ws[1,:], 0.9*Ws[2,:], 0.9*Ws[0,:], s=20, 
                  c=np.log(np.arange(1,nstep+1)), 
                  cmap=man_cmap, zorder=10)
ax1.set_title('On the hyperboloid')

ax2 = fig.add_subplot(122)
ax2.pcolormesh(X,Y, L.reshape((nplot,nplot)), cmap=loss_cmap)
coords = ax2.scatter(us[0,:], us[1,:], s=20, c=np.log(np.arange(1,nstep+1)), cmap=path_cmap)
ax2.scatter(us[0,0], us[1,0], s=30, marker='o')
# ax2.quiver(us[0,::10],us[1,::10],-gradu[0,::10], -gradu[1,::10], scale=10, scale_units='xy', color='r')
# ax2.scatter(Us[0,:], Us[1,:], s=20, c=np.log(np.arange(1,nstep+1)), cmap=man_cmap)
# ax2.scatter(Us[0,0], Us[1,0], s=30, marker='o')
riem = ax2.scatter(us_met[0,:], us_met[1,:], s=20, c=np.log(np.arange(1,nstep+1)), cmap=riem_cmap)
# ax2.quiver(us_met[0,::10],us_met[1,::10],
#            -grad_met[0,::10], -grad_met[1,::10], scale=10, scale_units='xy', color='b')
man = ax2.plot(Us[0,:], Us[1,:], 'k--')
# ax2.scatter(locmax[:,0], locmax[:,1], s=30, marker='x')
# ax2.scatter(locmin[:,0], locmin[:,1], s=30, marker='*')
ax2.legend([coords, riem, man[0]],['Euclidean gradient', 'Riemannian gradient', 'Manifold path'])

ax2.set_title('`Geodesic` coordinates')
# ax2.axis('square')


#%%
nstep = 1000
dt = 0.1

w0 = torch.tensor(np.random.randn(dim,1))
w0 /= w0.norm(2,0, keepdim=True)

u = Feedforward([1,2], 'Sigmoid',
                encoder=Indicator(1, 1), bias=False)
# u = nn.Embedding(1, 2)
# u.weight.datacopy_()

optimizer = optim.SGD(u.parameters(), lr=dt)
A = torch.tensor(C).float()

ws = np.zeros((dim, nstep))
us = np.zeros((dim-1, nstep))
grads = np.zeros((dim-1, nstep))
for t in range(nstep):
    optimizer.zero_grad()
    # emb = nn.Sigmoid()(u(torch.zeros(1).long()))
    emb = u(torch.zeros(1).long())
    w = geochart(emb[:,0:1].T, emb[:,1:].T)
    
    ws[:,t] = w.detach().numpy().squeeze()
    us[:,t] = emb.data.numpy().squeeze()
    
    loss = dist_sph(w0.T, w.T)
    loss.backward()
    
    grads[:,t] = u.network.layer0.weight.grad.data.numpy().squeeze()
    # grads[:,t] = u.weight.grad.data.numpy().squeeze()
    optimizer.step()

#%%
nplot = 50
path_cmap = 'autumn'
loss_cmap = 'Blues'

x = np.linspace(0,1,nplot)
y = np.linspace(0,1,nplot)
X,Y = np.meshgrid(x, y) # grid of point

u_test = torch.tensor(np.vstack((X.flatten(),Y.flatten())))

w_map = geochart(u_test[0:1,:], u_test[1:,:]).numpy()
d = dist_sph(w0.T, torch.tensor(w_map).T)
col = getattr(cm,loss_cmap)(d.reshape((nplot,nplot))/d.max())

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(w_map[0,:].reshape((nplot,nplot)),
                 w_map[1,:].reshape((nplot,nplot)),
                 w_map[2,:].reshape((nplot,nplot)), 
                 facecolors=col, linewidth=0, zorder=0)
ax1.scatter(1.1*ws[0,:], 1.1*ws[1,:], 1.1*ws[2,:], s=20, 
            c=np.log(np.arange(1,nstep+1)), 
            cmap=path_cmap, zorder=10)
ax1.set_title('On the sphere')

ax2 = fig.add_subplot(122)
ax2.pcolormesh(X,Y, d.reshape((nplot,nplot)), cmap=loss_cmap)
ax2.scatter(us[0,:], us[1,:], s=20, c=np.log(np.arange(1,nstep+1)), cmap=path_cmap)
ax2.scatter(us[0,0], us[1,0], s=30, marker='o')
ax2.set_title('Geodesic coordinates')
ax2.axis('square')

#%%

