CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
# import pickle
# from dataclasses import dataclass
# import itertools

sys.path.append(CODE_DIR)
from dataclasses import dataclass, fields, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import scipy.stats as sts
import scipy.linalg as la 
import scipy.special as spc

# import util
import plotting as tpl
import students
import super_experiments as sxp
import experiments as exp
import util

#%%

class RadialBoyfriend:
    """
    My special normalised (von) mises kernel
    
    a mises function which has zero integral and k(0) = 1
    """
    
    def __init__(self, width, center=True):
        
        self.kap = 0.5/width
        ## Set the scale so that the average variance of each neuron is 1
        ## and shift so the mean population response across stimuli is 0
        self.scale = 1/(np.exp(self.kap) - spc.i0(self.kap))
        if center:
            self.shift = spc.i0(self.kap)
        else:
            self.shift = 0
        
    def __call__(self, error, quantile=1e-4):
        """
        compute k(x,y) = k(x-y) ... so input x-y
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return (np.exp(self.kap*np.cos(error)) - self.shift)/denom

    def curv(self, x):
        """
        Second derivative
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return self.kap*np.exp(self.kap*np.cos(x))*(self.kap*np.sin(x)**2 - np.cos(x))/denom

    def deriv(self, x):
        """
        Derivative
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return -self.kap*np.sin(x)*np.exp(self.kap*np.cos(x))/denom

    def perturb(self, x, y):
        """
        The 'perturbation kernel', i.e. (f(x)-f(0))*(f(y)-f(0))
        """
        numer = self(x-y) - self(x) - self(y) + 1
        denom = 2*np.sqrt((1-self(x))*(1-self(y)))
        return numer/denom

    def sample(self, colors, size=1):
        """
        Sample activity in response to colors
        """
        K = self(colors[None] - colors[:,None])
        mu = np.zeros(len(colors))
        return np.random.multivariate_normal(mu, K, size=size).T


#%%

class softplus:
    
    def __init__(self, beta=2, axis=None):
        self.beta = beta
        self.axis = axis # normalization axis
    
    def __call__(self,x):
        phi = np.log(1 + np.exp(self.beta*x))/self.beta
        if self.axis is not None:
            phi /= phi.mean(self.axis, keepdims=True)
        return phi 
    
    def inv(self,x):
        return np.log(np.exp(self.beta*x)-1)/self.beta
    
class erf:
    
    def __init__(self, beta=2):
        self.beta = beta
    
    def __call__(self,x):
        return 1 + spc.erf(self.beta*np.sqrt(np.pi)*x/2)
    
    def inv(self,x):
        return spc.erfinv(x-1)*2/(np.sqrt(np.pi)*self.beta)

def rnn(J, x0, T, dt, phi=softplus, c=0):
    """
    Simulate an rnn up to time T with interval dt
    """
    
    x = 1*x0
    xs = [x0]
    for _ in range(T):
        p = phi(x)
        x = (1-dt)*x + dt*(J@p - c*(p.mean(0) - 1))
        
        xs.append(1*x)
        # t += dt
    
    return np.array(xs)

#%%

def classic_ring(N, width):
    """
    Construct a ring attractor network with cirulant weight matrix
    """
    
    kern = RadialBoyfriend(width)
    theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
    
    return kern(theta[None,:] - theta[:,None])


def clark_ring(Phi, X, lam=1e-7):
    """
    David's construction of a heterogeneously-tuned approximate ring attractor
    """
    
    N_theta, N = Phi.shape    

    K = Phi@Phi.T/N + lam*N_theta*np.eye(N_theta)/N
    Omg = la.pinv(Phi.T@Phi/N_theta + lam*np.eye(N))
    
    Junc = X.T@la.pinv(K)@Phi / N
    
    return Junc - np.diag(np.diag(Junc)/np.diag(Omg))*Omg

#%%

N = 1000
N_theta = 500
width = 1
act = 'soft'
# act = 'erf'
classic = False

theta = np.linspace(-np.pi, np.pi, N_theta, endpoint=False)
diff = theta[None] - theta[:,None]

kern = RadialBoyfriend(width)
X = kern.sample(theta, size=N) # / np.sqrt(N)

if classic:
    
    Phi = np.cos(diff)/2 + 1
    
else:
    if act == 'soft':
        phi = softplus(2)
        Phi = phi(X)
        Phi = Phi / Phi.mean(1, keepdims=True)
        c = 10
        
    else:
        phi = erf(2)
        Phi = 0.9*phi(X) + 0.05
        c = 0

X = phi.inv(Phi)

J = np.sqrt(6)*np.cos(diff)/N_theta
# J = clark_ring(Phi, X, lam=1e-6)
# J = X.T@la.pinv(Phi.T)

# x0 = np.random.randn(len(J), 200) + X.mean(0)[:,None]
x0 = X[::10].T + np.random.randn(*X[::10].T.shape)*5

X_t = rnn(J, x0, 1000, 0.05, phi=phi, c=c)

#%%

U, S, _ = la.svd(Phi.T-Phi.T.mean(1)[:,None], full_matrices=False)
pcs = Phi.T.T@U[:,:3]@np.diag(S[:3]/np.sum(S[:3]))
# pcs = phi(-X).T.T@U[:,:3]@np.diag(S[:3]/np.sum(S[:3]))

plt.figure()
ax = plt.subplot(111, projection='3d')
scat = ax.scatter3D(pcs[:,0],pcs[:,1],pcs[:,2], c='k', s=1)

# pcs = phi(X_t[0]).T@U[:,:3]
pcs = phi(X_t[-1]).T@U[:,:3]@np.diag(S[:3]/np.sum(S[:3]))
scat = ax.scatter3D(pcs[:,0],pcs[:,1],pcs[:,2])

tpl.set_axes_equal(ax)

#%%

dot = np.einsum('ijk,lj->ilk', phi(X_t), Phi)
dist = (Phi**2).sum(1,keepdims=True) + (phi(X_t)**2).sum(1,keepdims=True) - 2*dot

plt.plot(dist.min(1))

#%%

# diff = theta[dist[-1].argmin(0)][None] - theta[dist[-1].argmin(0)][:,None]
diff = util.circ_err(theta[dist[-1].argmin(0)][None], theta[dist[-1].argmin(0)][:,None])
# diff = np.mod(diff, np.pi)
# C = (X_t[-1].T)@(X_t[-1])/N
# C = phi(X_t[-1].T)@phi(X_t[-1])/N
C = util.center(phi(X_t[-1].T)@phi(X_t[-1])/N)
C /= np.mean(np.diag(C))

plt.plot(np.unique(diff), util.group_mean(C.flatten(), diff.flatten()))
plt.plot(np.unique(diff), kern(np.unique(diff)))

# C_ = Phi@Phi.T / N
# plt.plot(np.unique(diff), util.group_mean(C_.flatten(), diff.flatten()))

#%%

# g = 1

# chi = g*np.random.randn(*J.shape)/np.sqrt(N)
# # x0 = phi.inv(Phi[[N_theta//2]]).T + np.random.randn(N, 100)*1
# # x0 = phi.inv(Phi).T

# X_t = rnn(J + chi, x0, 1000, 0.05, phi=phi, c=c)

g = 1

J = 3 * np.cos(diff) / N_theta
chi = g*np.random.randn(*J.shape)/np.sqrt(N_theta)

# x0 = np.random.randn(N_theta,1)
x0 = np.cos(diff[:,::5]) #+ np.random.randn(N_theta, 100)

X_t = rnn(J + chi, x0, 1000, 0.01, phi=erf(2), c=0)

# plt.plot(theta[Phi.argmax(0)][X_t.argmax(1).squeeze()])
# plt.imshow(phi(X_t[...,-1]).T)
plt.plot(phi(X_t[:,::10,0]))
# plt.plot(X_t.argmax(1).squeeze(), 'k--')

#%%

U, S, _ = la.svd(Phi.T-Phi.T.mean(1)[:,None], full_matrices=False)
pcs = Phi.T.T@U[:,:3]@np.diag(S[:3]/S.sum())

plt.figure()
ax = plt.subplot(111, projection='3d')
# scat = ax.scatter3D(pcs[:,0],pcs[:,1],pcs[:,2], c='k', s=1)
tpl.plot3d(pcs, ax=ax, color='k')

# pcs = phi(X_t[...,0])@U[:,:3]@np.diag(S[:3]/S.sum())
pcs = phi(X_t.swapaxes(-1,-2))@U[:,:3]@np.diag(S[:3]/S.sum())
tpl.plot3d(pcs, ax=ax, linewidth=2, color=(0.5,0.5,0.5,0.5))

# pcs = phi(X_t[-1].T)@U[:,:3]@np.diag(S[:3]/S.sum())
scat = ax.scatter3D(pcs[-1,:,0],pcs[-1,:,1],pcs[-1,:,2], marker='*', s=100)
scat = ax.scatter3D(pcs[0,:,0],pcs[0,:,1],pcs[0,:,2], marker='.', s=50)

tpl.set_axes_equal(ax)

#%%

dot = np.einsum('ijk,lj->ilk', phi(X_t), Phi)
dist = (Phi**2).sum(1,keepdims=True) + (phi(X_t)**2).sum(1,keepdims=True) - 2*dot

# plt.plot(theta[dist.argmax(1).squeeze()])
plt.plot(dist.min(1))



