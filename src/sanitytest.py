
CODE_DIR = '/home/matteo/Documents/github/repler/src/'
svdir = '/home/matteo/Documents/uni/columbia/bleilearning/'

import sys, os, re
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import scipy

from students import *

#%% model 
z_dim = 2
supervise = False
selfsupervise = False
angle = 90

if supervise:
    enc = Feedforward([784, 400, 400, 2*z_dim], ['ReLU','ReLU', None])
    dec = Feedforward([z_dim, 10], ['Softmax'])
    vae = VAE(enc, dec, GausDiag(z_dim), Categorical(10))
else:
    enc = Feedforward([784, 500, 2*z_dim], ['ReLU', None])
    dec = Feedforward([z_dim, 500, 784], ['ReLU', 'Sigmoid'])
    vae = VAE(enc, dec, GausDiag(z_dim), Bernoulli(784))

#%% data
    
digits = torchvision.datasets.MNIST(svdir+'digits/',download=True, 
                                    transform=torchvision.transforms.ToTensor())
stigid = torchvision.datasets.MNIST(svdir+'digits/',download=True, train=False,
                                    transform=torchvision.transforms.ToTensor())
dl = torch.utils.data.DataLoader(digits, batch_size=bsz, shuffle=True)

#%% inference
nepoch = 500
bsz = 100
lr = 1e-4
include_kl = 'always' # or 'always' or 'no'

optimizer = optim.Adam(vae.parameters(), lr=lr)
# optimizer = optim.Adagrad(vae.parameters(), lr=lr)

elbo = np.zeros(0)
test_err = np.zeros(0)
# z_samples = np.zeros((2, 0))
for epoch in range(nepoch):
    running_loss = 0
    
    if include_kl == 'anneal':
        if epoch>50:
            beta = np.exp((epoch-300)/30)/(1+np.exp((epoch-300)/30))
    elif include_kl == 'always':
        beta = 1
    else:
        beta = 0
    for i, batch in enumerate(dl):
        nums, labels = batch
        if selfsupervise:
            rot = scipy.ndimage.rotate(nums.detach().numpy(), angle, axes=(1,2), reshape=False)
            rot = torch.tensor(rot).squeeze(1).reshape((-1, 784))
        nums = nums.squeeze(1).reshape((-1, 784))
        
        optimizer.zero_grad()
        
        # forward
        px_params, qz_params, z = vae(nums)
        if supervise:
            loss = -free_energy(vae, nums, px_params, qz_params, regularise=beta, y=labels)
            
            idx = np.random.choice(10000, 1000, replace=False)
            pred = vae(stigid.data.reshape(-1,784).float()/252)[0]
            terr = (stigid.targets == pred.argmax(1)).sum().float()/10000
            test_err = np.append(test_err, terr)
        elif selfsupervise:
            loss = -free_energy(vae, nums, px_params, qz_params, regularise=beta, 
                                xtrans=rot)
        else:
            loss = -free_energy(vae, nums, px_params, qz_params, regularise=beta)
        
        # optimise
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        elbo = np.append(elbo, loss.item())
        # z_samples = np.append(z_samples, z.detach().numpy(), axis=1)
    
    print('Epoch %d: ELBO=%.3f'%(epoch, -running_loss/(i+1)))


#%%
recon, _, _ = vae(digits.data[:10,:,:].reshape(-1,784).float()/252)
recon = recon.reshape((-1,28,28))

z = vae(digits.data.reshape(-1,784).float()/252)[2].detach().numpy()

#%%
idx = 1

plt.subplot(1,2,1)
plt.imshow(digits.data[idx,...].detach().numpy())

plt.subplot(1,2,2)
plt.imshow(recon[idx,...].detach().numpy())


#%% 
wa = np.meshgrid(sts.norm.ppf(np.linspace(0.01,0.99,20)),sts.norm.ppf(np.linspace(0.01,0.99,20)))
z_q = np.append(wa[0].flatten()[:,None], wa[1].flatten()[:,None],axis=1)

