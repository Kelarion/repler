"""
Objects that learn things, to be imported in any scripts I run.

Current classes:
    - Basic VAE
"""

#%%
CODE_DIR = '/home/matteo/Documents/github/repler/src/'
svdir = '/home/matteo/Documents/uni/columbia/bleilearning/'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import scipy

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
class DeepDistribution(nn.Module):
    """
    Abstract class for distributions that I want to use. Designed to play with
    neural networks of the NeuralNet class (below).
    """
    def __init__(self):
        super(DeepDistribution, self).__init__()
        
    def name(self):
        return self.__class__.__name__
    
    def distr(self):
        raise NotImplementedError
        
    def sample(self):
        raise NotImplementedError
    
class GausDiag(DeepDistribution):
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
        std = torch.exp(0.5*logvar)
        sigma = std[...,None]*torch.eye(self.ndim)[None,...]
        
        d = D.multivariate_normal.MultivariateNormal(loc=mu, scale_tril=sigma)
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

class GausId(DeepDistribution):
    """
    A family of distributions for deep generative models:
    Gaussian with identity covariance
    
    This module relates the output of a neural net to the parameters of a 
    gaussian distribution, assuming first N are the mean, and second N are
    the log-variances of each dimension.
    """
    
    def __init__(self, dim_z, prior_params=None):
        
        super(GausId, self).__init__()
        
        self.ndim = dim_z
        
        if prior_params is None:
            prior_params = {'loc': torch.zeros(dim_z),
                         'covariance_matrix': torch.eye(dim_z)}
        
        self.prior = D.multivariate_normal.MultivariateNormal(**prior_params)
        
    def distr(self, theta):
        """Return instance(s) of distribution, with parameters theta"""
        mu = theta
        sigma = torch.ones(theta.shape + (1,))*torch.eye(self.ndim)[None,...]
        
        d = D.multivariate_normal.MultivariateNormal(loc=mu, scale_tril=sigma)
        return d
        
    def sample(self, theta):
        """
        Sample from posterior, given parameters theta, using reparameterisation
        """
        mu = theta # decompose into mean and variance
        # std = 1
        
        eps = torch.randn_like(mu) 
        z = mu + eps
        
        return z

# class PointMass(DeepDistribution):
    
#     def __init__(self):
        
#         super(PointMass, self).__init__()
        
#         self.ndim = dim_z
        
#         self.prior = D.multivariate_normal.MultivariateNormal(**prior_params)
        
#     def distr(self, theta):
#         """Return instance(s) of distribution, with parameters theta"""
#         mu = theta
#         sigma = torch.ones(theta.shape + (1,))*torch.eye(self.ndim)[None,...]
        
#         d = D.multivariate_normal.MultivariateNormal(loc=mu, scale_tril=sigma)
#         return d
        
#     def sample(self, theta):
#         """
#         Sample from posterior, given parameters theta, using reparameterisation
#         """
#         mu = theta # decompose into mean and variance
#         # std = 1
        
#         eps = torch.randn_like(mu) 
#         z = mu + eps
        
        return z

class Bernoulli(DeepDistribution):
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

class Categorical(DeepDistribution):
    """
    A family of distributions for deep generative models:
    A categorical distribution
    """
    
    def __init__(self, dim_z, prior_params=None):
        
        super(Categorical, self).__init__()
        
        self.ndim = dim_z
        
        if prior_params is None:
            prior_params = {'probs': torch.ones(dim_z)/dim_z}
        
        self.prior = D.categorical.Categorical(**prior_params)
        
    def distr(self, theta):
        """Return instance(s) of distribution, with parameters theta"""
        d = D.categorical.Categorical(probs=theta)
        return d
        
    def sample(self, theta):
        """
        Sample from variable, given parameters theta, using reparameterisation
        """
        z = torch.multinomial(theta)
        return z

#%% Models !!!
class NeuralNet(nn.Module):
    """Skeleton of all pytorch models"""
    def __init__(self):
        super(NeuralNet,self).__init__()
    
    def forward(self):
        raise NotImplementedError
        
    def grad_step(self):
        raise NotImplementedError
    
    def save(self, to_path):
        """ save model parameters to path """
        with open(to_path, 'wb') as f:
            torch.save(self.state_dict(), f)
    
    def load(self, from_path):
        """ load parameters into model """
        with open(from_path, 'rb') as f:
            self.load_state_dict(torch.load(f))
        
class VAE(NeuralNet):
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
    
    def grad_step(self, data, optimizer, beta=1.0):
        """ Single step of the AEVB algorithm on the VAE generator-posterior pair """

        running_loss = 0
        
        for i, batch in enumerate(data):
            nums, labels = batch
            nums = nums.squeeze(1).reshape((-1, 784))
            
            optimizer.zero_grad()
            
            # forward
            px_params, qz_params, z = self(nums)
            
            loss = -free_energy(self, nums, px_params, qz_params, regularise=beta, y=labels)
            
            # optimise
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                
        return running_loss/(i+1)
            
#%% custom loss functions
def free_energy(model, x, px_params, qz_params, regularise=True, y=None, xtrans=None):
    """Computes free energy, or evidence lower bound
    If y is supplied, does a cheeky thing that isn't really the free energy
    ToDo: add support for >1 MC sample in the cross-entropy estimation
    """
    btch_size = x.shape[0]
    # z_post_params = model.enc(x)
    
    # z_samples = model.latent.sample(z_post_params)
    # px_params = model.dec(z_samples)
    
    # reconstruction error (i.e. cross-entropy)
    if y is not None:
        xent = model.obs.distr(px_params).log_prob(y).sum()
    else:
        if xtrans is not None:
            xent = model.obs.distr(px_params).log_prob(xtrans).sum()
        else:
            xent = model.obs.distr(px_params).log_prob(x).sum()
    
    # regularisation (i.e. KL-to-prior)
    prior = model.latent.prior.expand([btch_size])
    apprx = model.latent.distr(qz_params)
    
    dkl = regularise*(D.kl.kl_divergence(apprx, prior).sum())
    
    return xent-dkl


#%% helpers
def decimal(binary):
    """ convert binary vector to dedimal number (i.e. enumerate) """
    d = (binary*(2**np.arange(binary.shape[1]))[None,:]).sum(1)
    return d



