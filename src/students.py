"""
Objects that learn things, to be imported in any scripts I run.

Current classes:
    - Basic VAE
"""

#%%
CODE_DIR = '/home/matteo/Documents/github/repler/'
svdir = '/home/matteo/Documents/uni/columbia/bleilearning/'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np

#%% Neural networks !!!
class Feedforward(nn.Module):
    """
    Generic feedforward module, can be, e.g., the encoder or decoder of VAE
    """
    def __init__(self, dim_layers, nonlinearity='relu'):
        super(Feedforward, self).__init__()
        
        onion = OrderedDict()
        self.ndim = dim_layers
        
        if type(nonlinearity) is str:
            nonlinearity = [nonlinearity for _ in dim_layers[1:]]
        
        for l in range(len(dim_layers)-1):
            onion['layer%d'%l] = nn.Linear(dim_layers[l], dim_layers[l+1])
            if nonlinearity[l] is not None:
                onion['link%d'%l] = getattr(nn, nonlinearity[l])()
        
        self.network = nn.Sequential(onion)
        
    def forward(self, x):
        h = self.network(x)
        # h = self.activation(self.layers[0](x))
        # for layer in self.layers[1:]:
        #     h = self.activation(layer(h))
        return h

#%% Latent variable distribution families !!!
class GausDiag(nn.Module):
    """
    A family of distributions for deep generative models:
    Gaussian with diagonal covariance
    
    This module relates the output of a neural net to the parameters of a 
    gaussian distribution, assuming first N are the mean, and second N are
    the log-variances of each dimension.
    """
    
    def __init__(self, dim_z, prior_params=None):
        
        super(GausDiag, self).__init__()
        
        self.ndim = dim_z
        
        if prior_params is None:
            prior_params = {'loc': torch.zeros(dim_z),
                         'covariance_matrix': torch.eye(dim_z)}
        
        self.prior = D.multivariate_normal.MultivariateNormal(**prior_params)
        
    def distr(self, theta):
        """Return instance(s) of distribution, with parameters theta"""
        mu, logvar = theta.chunk(2, dim=1)
        var = torch.exp(logvar)
        sigma = var[...,None]*torch.eye(2)[None,...]
        
        d = D.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        return d
        
    def sample(self, theta):
        """
        Sample from posterior, given parameters theta, using reparameterisation
        """
        mu, logvar = theta.chunk(2, dim=1) # decompose into mean and variance
        std = torch.exp(0.5*logvar)
        
        eps = torch.randn_like(mu) 
        z = mu + std*eps
        
        return z

class Bernoulli(nn.Module):
    """
    A family of distributions for deep generative models:
    Bernouli
    """
    
    def __init__(self, dim_z, prior_params=None):
        
        super(Bernoulli, self).__init__()
        
        self.ndim = dim_z
        
        if prior_params is None:
            prior_params = {'probs': 0.5*torch.ones(dim_z)}
        
        self.prior = D.bernoulli.Bernoulli(**prior_params)
        
    def distr(self, theta):
        """Return instance(s) of distribution, with parameters theta"""
        d = D.bernoulli.Bernoulli(probs=theta)
        return d
        
    def sample(self, theta):
        """
        Sample from variable, given parameters theta, using reparameterisation
        """
        z = torch.bernoulli(theta)
        return z

#%% Models !!!
class VAE(nn.Module):
    """Abstract basic VAE class"""
    def __init__(self, encoder, decoder, latent, obs):
        super(VAE,self).__init__()
        
        self.enc = encoder
        self.dec = decoder
        
        self.latent = latent
        self.obs = obs
        
    def forward(self, x):
        """
        Outputs the parameters of p_x, so that the likelihood can be evaluated
        """
        qz_params = self.enc(x) # encoding
        z = self.latent.sample(qz_params) # stochastic part
        
        px_params = self.dec(z) # decoding
        # recon_x = self.p_x(px_params) # draw outputs
        
        return px_params, qz_params, z
    
#%%
def free_energy(model, x, px_params, qz_params):
    """Evidence lower bound
    ToDo: add support for >1 MC sample in the cross-entropy estimation
    """
    btch_size = x.shape[0]
    # z_post_params = model.enc(x)
    
    # z_samples = model.latent.sample(z_post_params)
    # px_params = model.dec(z_samples)
    
    # reconstruction error (i.e. cross-entropy)
    xent = model.obs.distr(px_params).log_prob(x).sum()
    
    # regularisation (i.e. KL-to-prior)
    prior = model.latent.prior.expand([btch_size])
    apprx = model.latent.distr(qz_params)
    
    dkl = D.kl.kl_divergence(apprx, prior).sum()
    
    return xent-dkl

#%%
nepoch = 200
bsz = 64
lr = 1e-3

enc = Feedforward([784, 400, 4], ['ReLU', None])
dec = Feedforward([2, 400, 784], ['ReLU', 'Sigmoid'])
vae = VAE(enc, dec, GausDiag(2), Bernoulli(784))

optimizer = optim.Adam(vae.parameters(), lr=lr)

digits = torchvision.datasets.MNIST(svdir+'digits/',download=True, 
                                    transform=torchvision.transforms.ToTensor())
dl = torch.utils.data.DataLoader(digits, batch_size=bsz, shuffle=True)

elbo = np.zeros(0)
# z_samples = np.zeros((2, 0))
for epoch in range(nepoch):
    running_loss = 0
    for i, batch in enumerate(dl):
        nums, labels = batch
        nums = nums.squeeze(1).reshape((-1, 784))
        
        optimizer.zero_grad()
        
        # forward
        px_params, qz_params, z = vae(nums)
        loss = -free_energy(vae, nums, px_params, qz_params)
        
        # optimise
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        elbo = np.append(elbo, loss.item())
        # z_samples = np.append(z_samples, z.detach().numpy(), axis=1)
    
    print('Epoch %d: ELBO=%.3f'%(epoch, -running_loss/(i+1)))







