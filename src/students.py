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
import numpy as np

#%%
class Feedforward(nn.Module):
    """
    Generic feedforward module, can be, e.g., the encoder or decoder of VAE
    """
    def __init__(self, dim_layers, nonlinearity='relu'):
        
        super(Feedforward, self).__init__()
        
        self.layers = []
        self.ndim = dim_layers
        
        for l in range(len(dim_layers)-1):
            self.layers.append(nn.Linear(dim_layers[l], dim_layers[l+1]))
        
        self.activation = getattr(F, nonlinearity)
        
    def forward(self, x):
        
        h = self.activation(self.layers[0](x))
        for layer in self.layers[1:]:
            h = self.activation(layer(h))
        
        return h

class GausDiag(nn.Module):
    """
    Gaussian with diagonal covariance
    
    This module relates the output of a neural net to the parameters of a 
    gaussian distribution, assuming first N are the mean, and second N are
    the variances of each dimension.
    """
    
    def __init__(self, dim_z, prior_params=None):
        
        super(GausDiag, self).__init__()
        
        self.ndim = dim_z
        
        if prior_params is None:
            prior_params = {'loc': torch.zeros(dim_z),
                         'covariance_matrix': torch.eye(dim_z)}
        
        # self.posterior = D.multivariate_normal.MultivariateNormal
        self.prior = D.multivariate_normal.MultivariateNormal(**prior_params)
        
    def log_prob(self, x, theta):
        """Compute log p(x|theta(z)) """
        mu, sigma = theta.chunk(2, dim=1)
        
        a = torch.log(mu.prod(1)*(2*np.pi)**self.ndim)
        logpos = -0.5*(a + (x-mu).dot((x-mu)))
        
    def sample(self, xi):
        """
        Sample from posterior, given parameters xi, using reparameterisation
        """
        loc_z, scale_z = xi.chunk(2, dim=1) # decompose into mean and variance
        
        eps = torch.randn_like(loc_z) 
        z = loc_z + scale_z*eps
        
        return z

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
        pz_params = self.enc(x) # encoding
        z = self.latent.sample(pz_params) # stochastic part
        
        px_params = self.dec(z) # decoding
        # recon_x = self.p_x(px_params) # draw outputs
        
        return px_params
    
    def encode(self, x):
        """Just the encoding step"""
        pz_params = self.enc(x) # encoding
        z = self.p_z(pz_params) # stochastic part
        return z
    
    # def decode(self, z):
    #     """Draw a sample reconstruction """
    #     px_params = self.dec(z)
        

#%%
def elbo(model, x):
    """Evidence lower bound"""
    
    






