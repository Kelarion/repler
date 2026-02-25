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
from scipy.optimize import root_scalar 

import torch
import torch.nn as nn
import torch.optim as optim

# import util
import plotting as tpl
import students
import super_experiments as sxp
import experiments as exp
import util

#%%


## Simulations
### check lower norm on guess trials

#%%

class RadialBoyfriend:
    """
    Gaussian process receptive fields with RBF kernel
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


class VMM:
    ## fix the local minima
    
    def __init__(self, k=0, pic=0.5, kmax=100):
        
        self.k = k
        self.pic = pic
        self.pig = 1-pic
        self.kmax = kmax
        
    def EStep(self, err):
        
        pe_c = self.pcorr(err)
        pc = self.pic*pe_c/(self.pig/(2*np.pi) + self.pic*pe_c)
        
        return pc
    
    def MStep(self, R):
        
        if (R > 0) and (self.ratio(self.kmax, R)>0):
            sol = root_scalar(self.ratio, args=(R,), bracket=(0,self.kmax))
            return sol.root
        elif R <=0:
            return 0
        else:
            return self.kmax
    
    def fit(self, err, iters=10):
        
        lik = []
        for i in range(iters):
            ## E step
            pc = self.EStep(err)
            self.pic = np.mean(pc)
            self.pig = 1 - self.pic
            
            ## M step
            self.k = self.MStep(np.cos(err)@pc/np.sum(pc))

            lik.append(np.mean(np.log(self.p(err))))
        
        return lik
        
    def sample(self, n):
        c = np.random.choice([0,1], size=n, p=(self.pig, self.pic))
        guess = np.pi*(2*np.random.rand(n)-1)
        corr = sts.vonmises(loc=0, kappa=np.max([self.k, 1e-6])).rvs(n)
        return np.where(c>0, corr, guess)
    
    def hess(self, n_samp=5000):
        """
        Monte carlo estimate of the likelihood Hessian 
        """
        
        H = np.zeros((2,2))
        for th in self.sample(n_samp):
            p = self.pcorr(th)
            foo = (np.cos(th) - spc.i1(self.k))
            Hij = p*foo
            Hjj = self.pic*p*(foo**2 + 0.5*(spc.i0(self.k) + spc.iv(2,self.k)))
            
            H += np.array([[0,Hij],[Hij,Hjj]])/n_samp
        
        return H
    
    def pcorr(self, err):
        return np.exp(self.k*np.cos(err))/(2*np.pi*spc.i0(self.k))
    
    def p(self, err):
        return self.pic*self.pcorr(err) + self.pig/(2*np.pi)
    
    def ratio(self, x, R=1):
        return spc.i1(x)/spc.i0(x) - R

#%%

## RNN model
### Add noise to different places (inputs, dynamics)
### noise could be additive or multiplicative
### look for attractor dynamics, or argmax at the decoder

class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.weight_ih_l0 = self.input2h.weight
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.
        
        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)
        
        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden
    

class RNNRegressor(students.NeuralNet):
    
    def __init__(self, dim_inp, dim_out, dim_hid, noise=0, rnn_type=nn.RNN, **rnn_args):
        
        super().__init__()
        
        self.dim_inp = dim_inp
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        
        self.noise = noise
        
        self.dec = nn.Linear(dim_hid, dim_out)
        self.rnn = rnn_type(dim_inp+dim_hid, dim_hid, **rnn_args, batch_first=True)
        
        with torch.no_grad():
            train_mask = np.zeros(dim_inp+dim_hid, dtype=bool)
            train_mask[dim_inp:] = True
            weight_mask = torch.ones_like(self.rnn.weight_ih_l0)
            weight_mask[:, train_mask] = 0
            ident_weights = torch.tensor(np.identity(dim_hid), dtype=torch.float)
            self.rnn.weight_ih_l0[:, train_mask] = ident_weights
            self.rnn.weight_ih_l0.register_hook(lambda grad: grad.mul_(weight_mask))
            
    def forward(self, X, H0=None, new_noise=None):
        if new_noise is None:
            new_noise = self.noise
        eps = torch.randn(*X.shape[:-1], self.dim_hid)*new_noise
        return self.dec(self.rnn(torch.cat([X, eps], dim=-1), H0)[0])
    
    def hidden(self, X, H0=None, new_noise=None):
        
        if new_noise is None:
            new_noise = self.noise
        eps = torch.randn(*X.shape[:-1], self.dim_hid)*new_noise
        return self.rnn(torch.cat([X, eps], dim=-1), H0)[0]
    
    def loss(self, batch):
        y_hat = self(batch[0])#.swapaxes(1,-1) 
        mask = batch[1] != -2
        L = nn.MSELoss(reduction='none') 
        return L(y_hat, batch[1])[mask].mean()

@dataclass
class EasyPeasy(sxp.Task):
    
    samps: int
    dim_inp: int
    min_delay: int = 5
    max_delay: int = 5
    response_window: int = 1
    silent_delay: bool = True
    input_width: float = 1   # curvature of the input representation
    noise: float = 0.1
    fixation: bool = True
    seed: int = 0
    
    def sample(self, new_noise=None):
        
        if new_noise is None:
            new_noise = self.noise
        
        # pad = nn.utils.rnn.pad_sequence
        np.random.seed(self.seed)
        
        t = np.random.choice(range(self.min_delay, self.max_delay+1), self.samps)
        T = np.max(t) + self.response_window

        cols = 2*np.pi*np.random.rand(self.samps) - np.pi
        cntr = np.linspace(-np.pi, np.pi, self.dim_inp)
        damp = np.exp(np.cos(cols[:,None] - cntr[None]) / self.input_width)
        # finput = RadialBoyfriend(self.input_width)
        # inps = finput.sample(cols, size=self.dim_inp)
        inps = np.hstack([np.sin(cols[:,None] - cntr[None])*damp , np.cos(cols[:,None]-cntr[None])*damp])
        outs = np.stack([np.sin(cols), np.cos(cols)]).T
        
        X = np.zeros((self.samps, T, 2*self.dim_inp + 2))
        X[:,0,:-2] = inps + np.random.randn(self.samps, 2*self.dim_inp)*new_noise
        X[np.arange(self.samps),t,-2] = 1
        
        if self.silent_delay:
            Y = np.zeros((self.samps, T, 2))
        else:
            Y = -2*np.ones((self.samps, T, 2))
            
        # Y[np.arange(self.samps),t] = outs
        for i in range(self.samps):
            Y[i, t[i]:t[i]+self.response_window] = outs[i]
            Y[i, t[i]+self.response_window:] = -2
            X[i, :t[i], -1] += self.fixation*1
        
        return {'X': torch.FloatTensor(X), 'Y': torch.FloatTensor(Y)}

@dataclass(kw_only=True)
class WMRNN(sxp.PTModel):
    
    dim_hid: int
    rnn_type: object
    rnn_noise: float = 0.0
    rnn_args: dict = field(default_factory=dict)
    
    def init_network(self, X, Y):
        
        self.metrics['kern'] = []
        
        self.pbar = None
        return RNNRegressor(dim_inp=X.shape[-1], dim_out=Y.shape[-1], 
                             dim_hid=self.dim_hid, noise=self.rnn_noise,
                             rnn_type=self.rnn_type, **self.rnn_args)

    def loop(self, X, Y):
        
        if self.pbar is None:
            self.pbar = tqdm(range(self.epochs))
        self.pbar.update(1)
        
        # Z = self.model.net.hidden(X, new_noise=0.0)
        
        
#%%

task = EasyPeasy(5000, 100, min_delay=5, max_delay=20,
                 response_window=5, input_width=100, noise=0.1, 
                 silent_delay=True, fixation=True)
net = WMRNN(dim_hid=100, 
            # rnn_type=nn.RNN, rnn_args={'nonlinearity': 'relu'},
            rnn_type=CTRNN, rnn_args={'dt': 10},
            epochs=500, rnn_noise=0.01,
            batch_size=64,
            opt_args={'lr':1e-3, 'weight_decay':0})

this_exp = sxp.Experiment(task, net)

this_exp.run()

plt.plot(this_exp.model.metrics['train_loss'])

#%%

data = task.sample(new_noise=0.1)

out = this_exp.model.net(data['X'], new_noise=0.1)

if task.silent_delay:
    t_resp = (data['Y'] !=-2).sum(1)[:,0] - task.response_window
else:
    _,t_resp = np.where(data['Y'][...,0] !=-2)

y = data['Y'][np.arange(len(out)), t_resp].detach().numpy()
yhat = out[np.arange(len(out)), t_resp].detach().numpy()

that = yhat/np.sqrt((yhat**2).sum(1, keepdims=True))
# plt.scatter(np.arctan2(y[:,0], y[:,1]), np.arctan2(yhat[:,0], yhat[:,1]))

theta = np.arctan2(y[:,0], y[:,1])
thetahat = np.arctan2(that[:,0], that[:,1])

plt.hist(util.circ_err(theta, thetahat), bins=25)

#%%

data = task.sample(new_noise=0.1)

Z = this_exp.model.net.hidden(data['X'], new_noise=0.1).detach().numpy()
Zfin = Z[np.arange(len(Z)), t_resp]

tpl.pca3d(Zfin.T, c=theta)

#%% Plot a specific choice of kappa

n_samp = 500
n_noise = 10000
dim = 5000

kappa = 100
noise_std = 10

th = np.linspace(-np.pi, np.pi, n_samp+1)
th0 = int(n_samp//2)

pop = RadialBoyfriend(0.5/kappa)

crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
thresh = np.sqrt(2*(1-pop(np.pi)))/2

# X = pop.sample(th, size=dim)/np.sqrt(dim)
k = pop(th)

# U,s,V = la.svd(X-X.mean(0), full_matrices=False)

# noise = np.random.randn(n_noise, dim)*noise_std

# x_pert = X[th0] + noise

e = noise_std*pop.sample(th, size=n_noise)

# ovlp = diffs@noise.T/la.norm(noise, axis=1)
x_proj = e - e[th0]
ovlp = e + k[:,None]

# ihat = (x_pert@X.T).argmax(1)
# xhat = th[ihat]
xhat = th[ovlp.argmax(0)]

# diffs = X - X[th0]
# diffs = diffs/(la.norm(diffs, axis=1, keepdims=True)+1e-6)
# 
# x_proj = noise@diffs.T #/la.norm(x_pert-X[th0], axis=1, keepdims=True) 
# eps = X@noise.T

# ovlp = diffs@noise.T/la.norm(noise, axis=1)

legit = np.abs(th[ovlp.argmax(0)]) > th[th0+1]

# these = (x_proj[:,np.abs(th)>crit].max(1)<x_proj[:,np.abs(th)<crit].max(1))
guess = np.abs(xhat)>crit

#%%
inmax = ovlp[np.abs(th)<crit].max(0)
outmax = ovlp[np.abs(th)>crit].max(0)
# plt.scatter(inmax, outmax, c=guess, s=1, cmap='bwr')
# tpl.square_axis()

#%%

kaps = [0.1, 0.5, 1, 2, 5, 10, 100]

i = 1
for kappa in kaps:
    
    pop = RadialBoyfriend(0.5/kappa)
    
    foo = pop.sample(th, n_noise)
    # foo = pop(th)[:,None] + 0.1*pop.sample(th, n_noise)
    
    crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
    
    plt.subplot(1,len(kaps),i)
    inmax = foo[np.abs(th)<crit].max(0)
    outmax = foo[np.abs(th)>crit].max(0)
    plt.scatter(inmax, outmax, s=0.5, cmap='bwr')
    tpl.square_axis()
    
    plt.plot(plt.xlim(), plt.xlim(), 'k--')
    plt.title(np.round(crit/np.pi,3))
    
    i += 1 

# %%

plt.figure()

plt.scatter(th[e.argmax(0)][(~these)*(guess)], xhat[(~these)*(guess)], s=10, alpha=0.5, c='r')
plt.scatter(th[e.argmax(0)][(these)*(guess)], xhat[(these)*(guess)], s=10, alpha=0.5, c='b')
plt.scatter(th[e.argmax(0)][(these)*(~guess)], xhat[(these)*(~guess)], s=10, alpha=0.5, c='c')
plt.scatter(th[e.argmax(0)][(~these)*(~guess)], xhat[(~these)*(~guess)], s=10, alpha=0.5, c='k')
# plt.scatter(th[ovlp.argmax(0)][(~these)*(guess)], xhat[(~these)*(guess)], s=10, alpha=0.5, c='r')
# plt.scatter(th[ovlp.argmax(0)][(these)*(guess)], xhat[(these)*(guess)], s=10, alpha=0.5, c='b')
# plt.scatter(th[ovlp.argmax(0)][(these)*(~guess)], xhat[(these)*(~guess)], s=10, alpha=0.5, c='c')
# plt.scatter(th[ovlp.argmax(0)][(~these)*(~guess)], xhat[(~these)*(~guess)], s=10, alpha=0.5, c='k')


plt.plot([crit, crit], [-np.pi, np.pi], 'k--')
plt.plot([-crit, -crit], [-np.pi, np.pi], 'k--')
plt.plot([-np.pi, np.pi], [crit, crit], 'k--')
plt.plot([-np.pi, np.pi], [-crit, -crit], 'k--')

#%%

plt.scatter(th[e.argmax(0)][guess], xhat[guess], s=10, alpha=0.5, c='k')

plt.scatter(th[e.argmax(0)][~guess], xhat[~guess], s=10, alpha=0.5, c='c')


tpl.square_axis()


plt.plot([crit, crit], [-np.pi, np.pi], 'k--')
plt.plot([-crit, -crit], [-np.pi, np.pi], 'k--')
plt.plot([-np.pi, np.pi], [crit, crit], 'k--')
plt.plot([-np.pi, np.pi], [-crit, -crit], 'k--')

#%%
# this_noise = 6
# these_noise = [1,4,5]
these_noise = np.where(guess)[0][:100]

delta = np.linspace(0,4,100)

for this_noise in these_noise:
    
    thhat = th[np.argmax(k + delta[:,None]*e[:,this_noise], axis=1)]
    plt.scatter(delta, thhat/thhat[-1])

#%%

ovlp = X@noise.T
ovlp = ovlp + k[:,None]

plt.subplot(1,2,1)
# plt.scatter(th[ovlp.argmax(0)][legit], 
#             x_proj[np.arange(n_noise),ovlp.argmax(0)][legit]**2, 
#             c=np.abs(xhat[legit])>crit, alpha=0.2)
plt.scatter(th[ovlp.argmax(0)],
            x_proj[np.arange(n_noise),ovlp.argmax(0)]**2, 
            s=10,
            c=np.abs(xhat)>crit, alpha=0.2, cmap='bwr')

ylims = plt.ylim()

plt.plot([crit, crit], ylims)
plt.plot([-crit, -crit], ylims)
plt.plot(plt.xlim(), [thresh**2, thresh**2])
plt.ylim(ylims)

# wa = np.mean(x_proj[np.arange(n_noise),ovlp.argmax(0)][legit]**2)
wa = np.mean(ovlp[np.abs(th)<crit].max(0)**2)
alf = wa/(2*noise_std**2)

plt.subplot(1,2,2)
# dist = sts.gamma(a=k/10, scale=2*noise_std**2)
dist = sts.gamma(a=alf, scale=2*noise_std**2)
# plt.hist(x_proj[np.arange(n_noise),ovlp.argmax(0)][legit]**2, 
#          bins=25, density=True, orientation='horizontal')
plt.hist(ovlp[np.abs(th)<crit].max(0)**2, 
         bins=25, density=True, orientation='horizontal')
plt.plot(dist.pdf(np.linspace(0,ylims[1],100)), 
         np.linspace(0,ylims[1],100), 'k-')

plt.ylim(ylims)

#%%


n_samp = 500
n_noise = 20_000
dim = 5000

i = 0
for kappa in [0.1, 1, 10]:
    for noise_std in [0.5, 2, 8]:

        th = np.linspace(-np.pi, np.pi, n_samp+1)
        th0 = int(n_samp//2)
        
        pop = RadialBoyfriend(0.5/kappa)
        
        crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
        thresh = np.sqrt(2*(1-pop(np.pi)))/2
        
        X = pop.sample(th, size=dim)/np.sqrt(dim)
        
        U,s,V = la.svd(X-X.mean(0), full_matrices=False)
        k = np.argmax(np.cumsum(s**2)/np.sum(s**2) > 0.99) + 1
        pr = ((s**2).sum()**2)/(s**4).sum()
        
        noise = np.random.randn(n_noise, dim)*noise_std
        
        # noise_proj = noise@V[:k].T
        x_pert = X[th0] + noise
        
        xhat = th[(x_pert@X.T).argmax(1)]
        
        plt.subplot(3,3,i+1)
        cnts, values, bars = plt.hist(xhat, bins=25, density=True)
        bin_ctr = (values[:-1] + values[1:])/2
        
        cols = ['royalblue', 'darkorange']
        for value, bar in zip(bin_ctr, bars):
            bar.set_facecolor(cols[int(np.abs(value) >= crit)])
                
        plt.ylabel('density')
        plt.xlabel('error')
        if i == 4:
            plt.legend([bars[0], bars[int(len(bars)//2)]], ['guess', 'correct'])
        
        i += 1

#%% Sweep over many kappa

n_samp = 500
n_noise = 10000
dim = 5000

noise_std = 1
# kaps = np.linspace(0.1, 200, 100)
kaps = 2**np.linspace(-8,8,10)
noises = np.linspace(0.1,2,50)

th = np.linspace(-np.pi, np.pi, n_samp+1)
th0 = int(n_samp//2)

# wa = []
# for noise_std in [0.5, 1, 2]:
alf = []
alf2 = []
pl = []
pr = []
pg = []
pg_ = []
for noise_std in tqdm(noises):
    thisalf = []
    thisalf2 = []
    thispl = []
    thispr = []
    thispg = []
    thispg_ = []
    
    for kap in kaps:
        
        pop = RadialBoyfriend(0.5/kap)
    
        k = pop(th)
    
        crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
        thresh = np.sqrt(2*(1-pop(np.pi)))/2
    
        # X = pop.sample(th, size=dim)/np.sqrt(dim)
    
        # noise = np.random.randn(n_noise, dim)*noise_std
        # x_pert = X[th0] + noise
        
        # diffs = X - X[th0]
        # diffs = diffs/(la.norm(diffs, axis=1, keepdims=True)+1e-6)
    
        # U,s,V = la.svd(diffs, full_matrices=False)
        # thispr.append(((s**2).sum()**2)/(s**4).sum())
    
        # x_proj = noise@diffs.T #/la.norm(x_pert-X[th0], axis=1, keepdims=True) 
    
        e = noise_std*pop.sample(th, size=n_noise)
    
        # ovlp = diffs@noise.T/la.norm(noise, axis=1)
        x_proj = e - e[th0]
        ovlp = e + k[:,None]
        
        legit = np.abs(th[ovlp.argmax(0)]) > th[th0+1]
        
        # max_proj = x_proj[e.argmax(0), np.arange(n_noise)]
        # average = np.mean(max_proj[legit]**2)
        # average = np.mean(ovlp[np.abs(th)>crit].max(0)**2)
        # average = np.mean(e.max(0)**2) 
        # average = np.mean(max_proj**2)
        average = np.mean((e + pop(np.pi))[e.argmax(0), np.arange(n_noise)]**2)
        thisalf.append(average/(2*noise_std**2)) # fact about gamma distribution
        thispl.append(np.mean(legit))
        
        thisalf2.append(np.mean(ovlp[np.abs(th)<crit].max(0)**2)/(2*noise_std**2))
        
        xhat = th[ovlp.argmax(0)]
        
        # distr = sts.gamma(a=thisalf[-1], scale=2*noise_std**2)
        distr = sts.betaprime(a=thisalf2[-1], b=thisalf[-1])
        
        thispg.append(np.mean(np.abs(xhat) > crit))
        # thispg_.append(thispl[-1]*(1-distr.cdf(thresh))*(1-crit/np.pi))
        thispg_.append(thispl[-1]*distr.cdf(1)*(1-crit/np.pi))
    
    
    alf.append(thisalf)
    alf2.append(thisalf2)
    pg.append(thispg)
    pg_.append(thispg_)
    pl.append(thispl)
    pr.append(thispr)

alf = np.array(alf)
alf2 = np.array(alf2)
pg = np.array(pg)
pg_ = np.array(pg_)
pl = np.array(pl)
pr = np.array(pr)

#%%

# thresh = np.sqrt(2*(1 - (np.exp(-kaps) - spc.i0(kaps))/(np.exp(kaps) - spc.i0(kaps))))/2
# crit = 2*np.arccos((np.sqrt(1 + 4*kaps**2) - 1)/(2*kaps))

# plt.subplot(1,3,1)
# alf_approx = 2 + np.log(1+kaps)/2.5
# plt.plot(np.log(kaps), 2*alf)
# plt.plot(np.log(kaps), alf_approx)
# plt.title('Effective degrees of freedom')

# plt.subplot(1,3,2)
# pl_approx = spc.expit(np.log(1+kaps*9)/3.5)
# plt.plot(np.log(kaps), pl)
# plt.plot(np.log(kaps), pl_approx)
# plt.title('Probability of potent perturbation')

# plt.subplot(1,3,3)
# plt.plot(np.log(kaps), pl_approx*(1-sts.gamma(a=alf_approx/2, scale=2*noise_std**2).cdf(thresh))*(1-crit/np.pi))
# plt.plot(np.log(kaps), pg)
# plt.title('Guess probability')

thresh = np.sqrt(2*(1 - (np.exp(-kaps) - spc.i0(kaps))/(np.exp(kaps) - spc.i0(kaps))))/2
crit = 2*np.arccos((np.sqrt(1 + 4*kaps**2) - 1)/(2*kaps))

# plt.subplot(1,3,1)
plt.subplot(1,2,1)
alf_approx = 0.5*(1 + np.log(1 + kaps)/2)
# alf_approx = 2 + np.log(1+kaps)/2.5
plt.plot(np.log(kaps), alf.T)
plt.plot(np.log(kaps), alf_approx, 'k--', linewidth=2)
plt.title('Effective degrees of freedom')

# plt.subplot(1,3,2)
pl_approx = spc.expit(np.log(1+kaps*9)/3.5)
# plt.plot(np.log(kaps), pl.T)
# plt.plot(np.log(kaps), pl_approx, 'k--', linewidth=2)
# plt.title('Probability of potent perturbation')

# plt.subplot(1,3,3)
plt.subplot(1,2,2)
pg_approx = (1-sts.gamma(a=alf_approx[None], scale=2*noises[:,None]**2).cdf(thresh[None]))
# pg_approx = pl_approx[None]*(1-sts.gamma(a=alf_approx[None], scale=2*noises[:,None]**2).cdf(thresh[None]))
plt.plot(np.log(noises), (pg))
plt.plot(np.log(noises), (pg_approx*(1-crit[None]/np.pi)), '--')
plt.title('Guess probability')

#%%

n_samp = 500
n_trial = 50000

noise_std = 4
# kaps = np.linspace(0.1, 200, 100)
kaps = 2**np.linspace(-8,8,100)

th = np.linspace(-np.pi, np.pi, n_samp+1)
th0 = int(n_samp//2)

# wa = []
# for noise_std in [0.5, 1, 2]:
alf = []
alf2 = []
pt = []
pg = []
for kap in tqdm(kaps):
    
    pop = RadialBoyfriend(0.5/kap)

    crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
    thresh = np.sqrt(2*(1-pop(np.pi)))/2

    eps = noise_std*pop.sample(th, size=n_trial).T

    k = pop(th) + eps 
    xhat = th[k.argmax(1)]
    
    guess = np.abs(xhat) > crit
    loc = np.abs(th[eps.argmax(1)]) > crit
    reg = np.abs(th) > crit
    
    pg.append(np.mean(guess))
    pt.append(loc[guess].sum()/loc.sum())

    average = np.mean(k[:,~reg].max(1)) # non-guess region
    alf.append(average/(noise_std))    
    
    average2 = np.mean(k[:,reg].max(1))
    alf2.append(average2/(noise_std))
    
pg = np.array(pg)
pt = np.array(pt)
alf = np.array(alf)
alf2 = np.array(alf2)

#%%

kap = 1
noise_std = 0.5

pop = RadialBoyfriend(0.5/kap)

crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
thresh = np.sqrt(2*(1-pop(np.pi)))/2

eps = noise_std*pop.sample(th, size=n_trial).T

k = pop(th) + eps 
xhat = th[k.argmax(1)]

# guess = np.abs(xhat) > crit
# loc = np.abs(th[eps.argmax(1)]) > crit
loc = np.abs(th) > crit

vals = np.linspace(0,k.max()**2,100)

# plt.hist(k[:,loc].max(1) - k[:,~loc].max(1), bins=25, density=True)
plt.hist(-k[:,~loc].max(1), bins=25, density=True)

# plt.figure()
# plt.hist(k[:,loc].max(1), bins=25, density=True, alpha=0.5)
# alf = np.mean((k[:,loc].max(1))/(2*noise_std**2))
# # distr = sts.gamma(a=alf, scale=2*noise_std**2)
# # plt.plot(vals, distr.pdf(vals))

# plt.hist(k[:,~loc].max(1), bins=25, density=True, alpha=0.5)
# alf = np.mean((k[:,~loc].max(1))/(2*noise_std**2))
# # distr = sts.gamma(a=alf, scale=2*noise_std**2)
# # plt.plot(vals, distr.pdf(vals))

# plt.legend(['outside', 'inside'])
#%%

nsamp = 100000
n_col = 500

kaps = 2**np.linspace(-8,8,100)

p_guess = []
p_guess_perm = []
mean_diff = []
# for noise_std in [0.25,0.5,1,2,4,8,16]:
for noise_std in [1]:
    pg = []
    pgp = []
    mn = []
    for kap in tqdm(kaps):
        
        idx = np.random.permutation(nsamp)
        
        pop = RadialBoyfriend(0.5/kap)
        th = np.linspace(-np.pi, np.pi, n_col+1)
    
        e = noise_std*pop.sample(th, size=nsamp)
        crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
        
        k = (pop(th)[:,None] + e)
        guess = np.abs(th[k.argmax(0)]) > crit
        # guess = np.abs(th[k.argmax(0)]) > 1
    
        # gmn = np.mean(k[np.abs(th)>crit].max(0))
        guess_max = k[np.abs(th)>crit].max(0)
        corr_max = k[np.abs(th)<crit].max(0)
        
        pg.append(np.mean(guess))
        # pgp.append(np.mean(k[np.abs(th)<crit].max(0) <= k[np.abs(th)>crit].max(0)[idx]))
        # pg.append(np.mean(guess_max>corr_max))
        pgp.append(np.mean(0.5*(1+spc.erf(k[np.abs(th)<crit].max(0)/np.sqrt(2)))))
        mn.append(np.mean(guess_max-corr_max))
    
    p_guess.append(pg)
    p_guess_perm.append(pgp)
    mean_diff.append(mn)

p_guess = np.array(p_guess)
p_guess_perm = np.array(p_guess_perm)
mean_diff = np.array(mean_diff)

#%%



#%%
logits = spc.logit(p_guess)
crit = 2*np.arccos((np.sqrt(1 + 4*kaps**2) - 1)/(2*kaps))
# a = 0.5
a = 1

plt.figure()
# plt.plot(np.log(kaps), logits.T)
# plt.plot(np.log(kaps), logits.T-logits[:,50].T)
plt.plot(np.log(kaps), np.log(p_guess).T)
# plt.plot(np.log(kaps), np.log(kaps) - 0.5*np.log(1 + a*kaps), 'k--')
# plt.plot(np.log(kaps), np.log(kaps) - np.log(1 + kaps), 'k--')
plt.plot(np.log(kaps), np.log(1 - crit/np.pi), 'k--')

#%%

# Y = torch.FloatTensor(np.log(p_guess.T+1e-10))
# mask = Y > np.log(1e-9)
# X = torch.FloatTensor(np.log(kaps)[:,None])
# # X = torch.FloatTensor(kaps[:,None])
# A = nn.Linear(1, len(p_guess))
# # B = nn.Linear(1, len(p_guess))

# optimizer = optim.Adam(list(A.parameters()) + list(C.parameters()), lr=1e-2)

# ls = []
# for _ in range(3000):
#     optimizer.zero_grad()
    
#     # xhat = A(X)
#     Yhat = C(A(X) - nn.Softplus()(A(X)))
#     # Yhat = A(torch.log(X)) - torch.log(X)
    
#     loss = nn.MSELoss()(Yhat[mask], Y[mask])
#     loss.backward()
    
#     optimizer.step()
    
#     ls.append(loss.item())


#%%

kap = 1
noise_std = 1

nsamp = 5000
n_col = 500

pop = RadialBoyfriend(0.5/kap)
th = np.linspace(-np.pi, np.pi, n_col+1)

e = noise_std*pop.sample(th, size=nsamp)

X,Y = np.meshgrid(th,th)
l,V = la.eigh(pop(X-Y))
pr = (np.sum(l)**2)/np.sum(l**2)
crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))

k = (pop(th)[:,None] + e)
guess = np.abs(th[k.argmax(0)]) > crit

# gmn = np.mean(k[np.abs(th)>crit].max(0))
guess_max = k[np.abs(th)>crit].max(0)
corr_max = k[np.abs(th)<crit].max(0)

mse = []
err = []
mn_in = []
mn_out = []
ks_guess = []
ks_corr = []
for r in range(1, 30):
    
    # ctr = np.linspace(0, 2*np.pi, k+1)[:-1]
    # B = pop(th[:,None] - ctr[None])
    B = V[:,-r:]#@np.diag(l[-r:])
    # w = la.pinv(B)@e
    w = np.diag(np.sqrt(l[-r:]))@np.random.randn(r, nsamp)
    ehat = noise_std*B@w
    
    khat = (pop(th)[:,None] + ehat)
    guesshat = np.abs(th[khat.argmax(0)]) > crit
    
    mn_in.append(np.mean(khat[np.abs(th)>crit].max(0)))
    mn_out.append(np.mean(khat[np.abs(th)<crit].max(0)))
    
    ks_guess.append(sts.ks_2samp(guess_max, khat[np.abs(th)>crit].max(0)).statistic)
    ks_corr.append(sts.ks_2samp(corr_max, khat[np.abs(th)<crit].max(0)).statistic)
    
    err.append(np.mean(guess==guesshat))
    mse.append(1 - np.mean((e - ehat)**2)/np.mean(e**2))

# plt.plot(range(1, 30), mse, marker='.')
# plt.plot([pr, pr], plt.ylim(), 'k--')
# plt.plot(range(1,30), err, marker='.')

plt.plot(range(1,30), ks_guess, marker='.')
plt.plot(range(1,30), ks_corr, marker='.')

# plt.plot(range(1,30), mn_in)
# plt.plot([1,30], [np.mean(guess_max), np.mean(guess_max)], 'k--')
# plt.plot(range(1,30), mn_out)
# plt.plot([1,30], [np.mean(corr_max), np.mean(corr_max)], 'k--')

#%%

def fitvar(R, kmax=500):
    
    def ratio(x, R=1):
        return spc.i1(x)/spc.i0(x) - R
    
    if (R > 0) and (ratio(kmax, R)>0):
        sol = root_scalar(ratio, args=(R,), bracket=(0,kmax))
        return sol.root
    elif R <=0:
        return 0
    else:
        return kmax

#%%
pt = []
pg = []
pgpred = []
kest = []
kpred = []
ppi = []
cv = []
cvlocal = []
cvthresh = []
corvar = []
ll = []
llest = []
mse = []

nsamp = 100_000
n_col = 500

kaps = 2**np.linspace(-8,8,10)
# kaps = np.array([0.1, 10, 100])
# noises = np.array([0.1, 0.5, 1, 2])
# noises = np.linspace(0.1,2,50)
noises = 2**np.linspace(-3,1,50)

pt_cond = np.zeros((len(noises), len(kaps)))

th = np.linspace(-np.pi, np.pi, n_col+1)

for i, noise_std in tqdm(enumerate(noises)):
    thispt = []
    thispg = []
    thispgpred = []
    thiskest = []
    thiskpred = []
    thisppi = []
    thiscv = []
    thiscvlocal = []
    thiscvthresh = []
    
    thisll = []
    thisllest = []
    this_mse = []
    for j,kap in enumerate(kaps):
        
        pop = RadialBoyfriend(0.5/kap)
    
        e = noise_std*pop.sample(th, size=nsamp)
        crit = 2*np.arccos((np.sqrt(1 + 4*pop.kap**2) - 1)/(2*pop.kap))
        
        FI = -pop.curv(0)/(noise_std**2)
        varpred = np.sqrt(1/(1 + 1/FI))
        kappred = fitvar(varpred)
        thiskpred.append(kappred)
        
        k = (pop(th)[:,None] + e)
        thhat = th[k.argmax(0)]
        guess = np.abs(thhat) > crit
        this_mse.append(np.mean(thhat**2))
        
        thispt.append(np.mean(guess))
        thiscv.append([np.mean(np.cos(thhat)), np.mean(np.sin(thhat))])
        thiscvlocal.append([np.mean(np.cos(thhat[~guess])), 
                            np.mean(np.sin(thhat[~guess]))])
        thiscvthresh.append([np.mean(np.cos(thhat[guess])),
                             np.mean(np.sin(thhat[guess]))])

        # vmm = VMM(k=1, pic=0.1, kmax=300)
        # lik = vmm.fit(thhat, iters=50)
        
        # thispg.append(1-vmm.pic)
        # thiskest.append(vmm.k*1)
        # thisppi.append(vmm.pcorr(np.pi))
        
        # thisll.append(np.mean(np.log(vmm.p(thhat))))
        
        pt_cond[i,j] = np.mean(guess[np.abs(th[e.argmax(0)])> crit])
        
        # pcorr = np.exp(-kappred)/(2*np.pi*spc.i0(kappred))
        # pgest = (np.mean(guess)/(2*(np.pi-crit)) - pcorr)/(1/(2*np.pi) - pcorr)
        Phi = sts.vonmises(kappred).cdf(-crit)
        pgest = (np.mean(guess) - 2*Phi)/((1-crit/np.pi) - 2*Phi)
        pest = pgest/(2*np.pi) + (1-pgest)*np.exp(kappred*np.cos(thhat))/(2*np.pi*spc.i0(kappred))
        
        thispgpred.append(pgest)
        thisllest.append(np.mean(np.log(pest)))
        
    pt.append(thispt)
    pg.append(thispg)
    pgpred.append(thispgpred)
    kest.append(thiskest)
    kpred.append(thiskpred)
    ppi.append(thisppi)
    cv.append(thiscv)
    cvlocal.append(thiscvlocal)
    cvthresh.append(thiscvthresh)
    ll.append(thisll)
    llest.append(thisllest)
    mse.append(this_mse)

mse = np.array(mse)
pt = np.array(pt)
pg = np.array(pg)
pgpred = np.array(pgpred)
kest = np.array(kest)
kpred = np.array(kpred)
ppi = np.array(ppi)
cv = np.array(cv)
cvlocal = np.array(cvlocal)
cvthresh = np.array(cvthresh)
ll = np.array(ll)
llest = np.array(llest)

#%%

# pncv = np.pi/(8*noises**2)*np.exp(-1/(2*noises**2))*(spc.i0(1/(4*noises**2)) + spc.i1(1/(4*noises**2)))**2
beta = 1 / (4*noises**2)
pncv = np.sqrt(2*np.pi*beta)*np.exp(-beta)*(spc.i0(beta) + spc.i1(beta)) / 2

#%% "theory"

thresh = np.sqrt(2*(1 - (np.exp(-kaps) - spc.i0(kaps))/(np.exp(kaps) - spc.i0(kaps))))/2
crit = 2*np.arccos((np.sqrt(1 + 4*kaps**2) - 1)/(2*kaps))
alf_approx = 2 + np.log(1+kaps)/2.5
# alf_approx = 0.5*(1 + np.log(1 + kaps)/2)
pl_approx = spc.expit(np.log(1+kaps*9)/3.5)
pg_approx = (1-sts.gamma(a=alf_approx[None]/2, scale=2*noises[:,None]**2).cdf(thresh[None]))
pt_approx = (pl_approx[None]*pg_approx*(1-crit[None]/np.pi))
# pt_approx = (pg_approx*(1-crit[None]/np.pi))

pcorr = np.exp(-kpred)/(2*np.pi*spc.i0(kpred))
pgest = (pt_approx/(2*(np.pi-crit)) - pcorr)/(1/(2*np.pi) - pcorr)
# pgest = (pt/(2*(np.pi-crit)) - pcorr)/(1/(2*np.pi) - pcorr)
# Phi = sts.vonmises(kpred).cdf(-crit)
# pgest = (pt_approx - 2*Phi)/((1-crit/np.pi) - 2*Phi)

curv = kaps*np.exp(kaps)/(np.exp(kaps)-spc.i0(kaps))
FI = curv[None]/(noises**2)[:,None]

cmap = cm.spring
for i in range(len(kaps)):
    plt.plot(-np.log(noises), -spc.logit((1-pgest[:,i])/np.sqrt(1+1/FI[:,i])), c=cmap(i/len(kaps)))

for i in range(len(kaps)):
    plt.scatter(-np.log(noises[::4]), -spc.logit(cv[::4,i,0]), c=cmap(i/len(kaps)))
    
# for i in range(len(kaps)):
#     # plt.plot(-np.log(noises), spc.logit(pt[:,i]/(1 - crit[i]/np.pi)), c=cmap(i/len(kaps)))
#     # plt.plot(-np.log(noises), spc.logit(pt_cond[:,i]), c=cmap(i/len(kaps)))
#     plt.plot(-np.log(noises), spc.logit(pt[:,i]), c=cmap(i/len(kaps)))
#     # plt.plot(-np.log(noises), spc.logit(pt_cond[:,i]*(1 - crit[i]/np.pi)),'--', c=cmap(i/len(kaps)))


#%%

cmap = cm.spring
for i in range(len(kaps)):
    plt.plot(-np.log(noises), -spc.logit(cv[:,i,0]), c=cmap(i/len(kaps)))
    # plt.plot(np.log(noises), 1 - 1/(1+1/FI[:,i]),  c=cmap(i/len(kaps)))
    # plt.plot(np.log(noises), pt[:,i], c=cmap(i/len(kaps)))
    # plt.plot(np.log(noises), pg_approx[:,i], c=cmap(i/len(kaps)))
    # plt.plot(np.log(noises), spc.erfc(thresh[i]/(np.sqrt(2)*noises))/2, c=cmap(i/len(kaps)))
    # plt.plot(np.log(noises), sts.norm(0,1).cdf(-thresh[i]/noises), c=cmap(i/len(kaps)))
    # plt.plot(np.log(noises), pg_approx[:,i], c=cmap(i/len(kaps)))
    # plt.plot(np.log(noises), pt_approx[:,i], '--', c=cmap(i/len(kaps)))

plt.legend(np.round(np.log(kaps),2), title='log k')

# for i in range(len(kaps)):
#     plt.plot(-np.log(noises), -spc.logit(jeff_mse(1/noises, 1/kaps[i])), '-.', c=cmap(i/len(kaps)))

plt.plot(-np.log(noises), -spc.logit(pncv), 'k--')
# plt.xlabel('log std')
plt.xlabel('log snr')
# plt.ylabel('p(threshold)')
# plt.ylabel('p(guess)')
# plt.ylabel('p(e > d/2)')
plt.ylabel('logit MCE')
# plt.semilogy()

#%%

def pnsamp(sigma, size, T=np.pi):
    
    x = [[1],[0]] + np.random.randn(2,size)*sigma
    xproj = x / np.sqrt((x**2).sum(0))

    that = T*np.arctan2(xproj[1], xproj[0])/np.pi
    
    return that    

#%%
import sklearn.gaussian_process as skgp
import sklearn.neighbors as skn

class VMKernel(skgp.kernels.Kernel):
    def __init__(self, k=1.0, normalized=True):
        self.k = k
        
        if normalized:
            self.denom = (np.exp(self.k) - spc.i0(self.k))
        else:
            self.denom = np.sqrt(spc.i0(2 * self.k) - spc.i0(self.k)**2)
        
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        
        dx = X - Y.T
        num = np.exp(self.k * np.cos(dx)) - spc.i0(self.k)

        return num / self.denom
    
    def diag(self, X):
        num = np.exp(self.k) - spc.i0(self.k)
        return np.ones(X.shape[0])*num/self.denom
    
    def is_stationary(self):
        return True

class JeffKernelPopulation:
    def __init__(
        self,
        pwr,
        wid,
        n_units=1000,
        stim_pts=101,
        stim_range=(-np.pi, np.pi),
        **kwargs,
    ):
        self.stim_range = stim_range
        self.stim = np.expand_dims(np.linspace(*stim_range, stim_pts), 1)
        self.rng = np.random.default_rng()
        self.reps = self.make_pop(self.stim, pwr, wid, n_units)
        self.decoder = skn.KNeighborsRegressor(**kwargs)
        self.decoder.fit(self.reps, self.stim)
        self.wid = wid
        self.n_units = n_units

    def sample_reps(self, n_samps=1000, single_stim=None, add_noise=False, noise=1):
        if single_stim is not None:
            ind = np.argmin(np.sum((self.stim - single_stim) ** 2, axis=1))
            inds = np.array((ind,) * n_samps)
        else:
            inds = self.rng.choice(len(self.stim), size=n_samps)
        use_stim = self.stim[inds]
        use_reps = self.reps[inds]
        if add_noise:
            use_reps = use_reps + self.rng.normal(0, noise, size=use_reps.shape)
        return inds, use_stim, use_reps

    def empirical_kernel(self):
        kernel = self.reps @ self.reps.T
        diffs = np.expand_dims(self.stim, 1) - np.expand_dims(self.stim, 0)

        return diffs.flatten(), kernel.flatten()

    def simulate_decoding(self, **kwargs):
        inds, true_stim, use_reps = self.sample_reps(add_noise=True, **kwargs)
        dec_stim = self.decoder.predict(use_reps)
        return true_stim, dec_stim

    def sample_dec_gp(self, **kwargs):
        inds, true_stim, use_reps = self.sample_reps(add_noise=True, **kwargs)
        dec_samps = use_reps @ self.reps.T
        return np.squeeze(self.stim), dec_samps

class JeffGPKernelPopulation(JeffKernelPopulation):
    def make_pop(self, stim, pwr, wid, n_units):
        p = pwr / n_units
        kernel = skgp.kernels.ConstantKernel(p) * skgp.kernels.RBF(length_scale=wid)
        self.kernel = kernel
        gp = skgp.GaussianProcessRegressor(kernel=kernel)
        y = gp.sample_y(stim, n_samples=n_units, random_state=None)
        return y

    def theoretical_kernel(self, n_bins=100):
        s_diffs = np.expand_dims(np.linspace(*self.stim_range, n_bins), 1)
        k = self.n_units * self.kernel(s_diffs, np.zeros((1, 1)))
        return s_diffs, np.squeeze(k)

    def sample_dec_gp(self, n_samps=1000, noise_sigma=1, **kwargs):
        s_diffs, mu = self.theoretical_kernel(**kwargs)
        cov = self.n_units * noise_sigma**2 * self.kernel(s_diffs, s_diffs)
        pre = self.kernel([[0]], [[0]]) + noise_sigma**2
        post = (
            self.kernel(s_diffs, s_diffs)
            + self.kernel(s_diffs, np.zeros((1, 1)))
            * self.kernel(s_diffs, np.zeros((1, 1))).T
        )
        cov = self.n_units * pre * post
        gp = sts.multivariate_normal(mu, cov, allow_singular=True)
        dec_samps = gp.rvs(n_samps)
        return s_diffs, dec_samps

class MaGPKernelPopulation(JeffKernelPopulation):
    def make_pop(self, stim, pwr, wid, n_units):
        p = pwr / n_units
        self.kernel = skgp.kernels.ConstantKernel(p) * VMKernel(1/wid, normalized=True)
        gp = skgp.GaussianProcessRegressor(kernel=self.kernel)
        y = gp.sample_y(stim, n_samples=n_units, random_state=None)
        return y

    def theoretical_kernel(self, n_bins=100):
        s_diffs = np.expand_dims(np.linspace(*self.stim_range, n_bins), 1)
        k = self.n_units * self.kernel(s_diffs, np.zeros((1, 1)))
        return s_diffs, np.squeeze(k)

    def sample_dec_gp(self, n_samps=1000, noise_sigma=1, **kwargs):
        s_diffs, mu = self.theoretical_kernel(**kwargs)
        cov = self.n_units * noise_sigma**2 * self.kernel(s_diffs, s_diffs)
        pre = self.kernel([[0]], [[0]]) + noise_sigma**2
        post = (
            self.kernel(s_diffs, s_diffs)
            + self.kernel(s_diffs, np.zeros((1, 1)))
            * self.kernel(s_diffs, np.zeros((1, 1))).T
        )
        cov = self.n_units * pre * post
        gp = sts.multivariate_normal(mu, cov, allow_singular=True)
        dec_samps = gp.rvs(n_samps)
        return s_diffs, dec_samps

def vm_kernel(x, wid):
    k = 1 / wid
    num = np.exp(k * np.cos(x)) - spc.i0(k)
    denom = np.sqrt(spc.i0(2 * k) - spc.i0(k)**2)
    return num / denom

def jeff_threshold_err(snr, wid):
    return np.ones_like(snr) * np.ones_like(wid) * (1 / 3) * np.pi**2

def ma_threshold_err(snr, wid):
    return np.ones_like(snr) * np.ones_like(wid) * 0

def jeff_local_err(snr, wid):
    orig = (wid / snr) ** 2

    comb = np.min([orig, np.ones_like(snr) * wid ** 2, jeff_threshold_err(snr, wid)], axis=0)
    return comb

def jeff_local_err_vm(snr, wid):
    curv = (1/wid)*np.exp(1/wid)/(np.exp(1/wid)-spc.i0(1/wid))
    orig = 1/(curv*(snr**2))

    comb = np.min([orig, np.ones_like(snr) * wid ** 2, jeff_threshold_err(snr, wid)], axis=0)
    return comb

def ma_local_err_norm(snr, wid):
    
    curv = (1/wid)*np.exp(1/wid)/(spc.i0(2/wid)-spc.i0(1/wid)**2)
    FI = curv*(snr**2)

    return 1/np.sqrt(1+1/FI)

def ma_local_err(snr, wid):
    
    curv = (1/wid)*np.exp(1/wid)/(np.exp(1/wid)-spc.i0(1/wid))
    FI = curv*(snr**2)

    return 1/np.sqrt(1+1/FI)


def ma_local_err_gauss(snr, wid):
    
    FI = (snr / wid) ** 2

    return 1/np.sqrt(1+1/FI)

def jeff_mse(snr, wid):
    # p = jeff_threshold_prob(snr, wid)
    p = ma_threshold_prob(snr, wid)
    # t_err = jeff_threshold_err(snr, wid)
    t_err = ma_threshold_err(snr, wid)
    # l_err = jeff_local_err(snr, wid)
    l_err = ma_local_err(snr, wid)
    # l_err = ma_local_err_norm(snr, wid)
    # l_err = ma_local_err_gauss(snr, wid)
    return (1 - p) * l_err + p * t_err

def jeff_threshold_prob(snr, wid):
    n = np.pi / wid
    dist = sts.norm(0, 1).cdf(-np.sqrt(2) * snr / 2)

    prod = n * dist
    return np.min((prod, np.ones_like(prod)), axis=0)

def ma_threshold_prob(snr, wid):
    
    crit = 2*np.arccos((np.sqrt(1 + 4/(wid**2)) - 1)/(2/wid))
    n = np.pi / (crit**2)
    
    thresh = np.sqrt(2*(1 - (np.exp(-1/wid) - spc.i0(1/wid))/(np.exp(1/wid) - spc.i0(1/wid))))/2
    dist = sts.norm(0, 1).cdf(-thresh * snr)

    prod = n * dist
    return np.min((prod, np.ones_like(prod)), axis=0)


#%%

bounds = (0.5, 6)
snrs = np.linspace(*bounds, 1000)
wid = 100

snrs_fit = np.linspace(*bounds, 19)
samp_mse_circ = np.zeros(len(snrs_fit))
samp_mce = np.zeros(len(snrs_fit))
for i, snr in enumerate(snrs_fit):
    # m = JeffKernelTheory(snr, wid)
    # m = JeffGPKernelPopulation(snr**2, wid)
    m = MaGPKernelPopulation(snr**2, wid)
    ts, rs = m.simulate_decoding(n_samps = 10000)
    
    samp_mse_circ[i] = np.mean(np.arctan2(np.sin(ts-rs), np.cos(ts-rs))**2)
    samp_mce[i] = np.mean(1 - np.cos(ts-rs))
    
# plt.plot(snrs_fit, samp_mse_circ, 'g')
plt.plot(np.log(snrs_fit), samp_mce)
plt.plot(np.log(snrs), 1-jeff_mse(snrs, wid), '--')
plt.legend(['MCE', 'theory'])
plt.xlabel('log snr')
plt.ylabel('mse')


#%%

f, ax = plt.subplots(1, 1)

ax.plot(np.mean(sig, axis=1), np.mean(guess_rates, axis=1), "-o")
errs = np.sqrt(jeff_local_err(snrs, wid))
gs = jeff_threshold_prob(snrs, wid)
ax.plot(errs, gs)

ax.set_xlabel(r"local errors $\sigma$") 
ax.set_ylabel("guess rate") 