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
    def __init__(self, dim_layers, nonlinearity='ReLU', encoder=None, bias=True):
        super(Feedforward, self).__init__()
        
        onion = OrderedDict()
        self.ndim = dim_layers
        
        if type(nonlinearity) is str:
            nonlinearity = [nonlinearity for _ in dim_layers[1:]]
        
        if encoder is not None:
            # optionally include a pre-network encoder, e.g. if inputs are indices
            onion['embedding'] = encoder
        
        for l in range(len(dim_layers)-1):
            # onions have layers
            onion['layer%d'%l] = nn.Linear(dim_layers[l], dim_layers[l+1], bias=bias)
            if nonlinearity[l] is not None:
                if 'softmax' in nonlinearity[l].lower():
                    onion['link%d'%l] = getattr(nn, nonlinearity[l])(dim=-1)
                else:
                    onion['link%d'%l] = getattr(nn, nonlinearity[l])()
        
        self.network = nn.Sequential(onion)
        
    def forward(self, x):
        h = self.network(x)
        # h = self.activation(self.layers[0](x))
        # for layer in self.layers[1:]:
        #     h = self.activation(layer(h))
        return h

# Layers with random weights
class LinearRandom(object):
    """
    Abstract class for linear layers with random weights
    """
    def __init__(self, fix_weights=False, nonlinearity=None):
        if nonlinearity is not None:
            self.link = getattr(nn, nonlinearity)()
        else:
            self.link = None

        self.fixed = fix_weights
        if fix_weights:
            self.called = False

        self.__name__ = self.__class__.__name__
        #     self.w = self.draw_weights()

    def draw_weights(self, num_weights):
        raise NotImplementedError

    def __call__(self, inp):
        if self.fixed and self.called:
            W = self.weights
        else:
            W = self.draw_weights(inp.shape[-1])
            if self.fixed:
                self.weights = W
        self.called = True
        if self.link is not None:
            return self.link(torch.matmul(inp, W.T))
        else:
            return torch.matmul(inp, W.T)

class LinearRandomSphere(LinearRandom):
    """
    Weights drawn one p-norm sphere with orthogonal gaussian noise
    only works for a curve right now!!
    and the parametrization is a hack, should do better
    """
    def __init__(self, dim=2, p=1, radius=1, eps=0.1, 
                 fix_weights=False, bias=False, nonlinearity=None):
        super(LinearRandomSphere, self).__init__(fix_weights,
                                                 nonlinearity)
        self.dim = dim
        self.p = p
        # self.num_weights = num_weights
        self.radius = radius
        self.eps = eps  # noise scale relative to radius

    def draw_weights(self, num_weights):
        theta = np.random.rand(num_weights)*2*np.pi 
        orth_noise = np.random.randn(num_weights)*self.eps*self.radius

        coords = np.array([np.cos(theta), np.sin(theta)])
        scl = np.sum(np.abs(coords)**self.p,0)**(1/self.p) # the p-normalizing factor
        coords /= scl/self.radius
        normal = np.sign(coords)*(np.abs(coords)/scl)**(self.p-1)

        return torch.tensor(coords + orth_noise*normal, requires_grad=False).float()

class LinearRandomNormal(LinearRandom):
    def __init__(self, dim=2, var=1, fix_weights=False, nonlinearity=None):
        super(LinearRandomNormal, self).__init__(fix_weights,
                                                 nonlinearity)
        self.dim = dim
        self.var = var

    def draw_weights(self, num_weights):
        return torch.tensor(np.random.randn(self.dim, num_weights)*self.var, requires_grad=False).float()

class LinearRandomProportional(LinearRandom):
    """
    Create that very strange assymetric cross-shaped distribution
    """
    def __init__(self, dim=2, scale=1, coef=1,
                 fix_weights=False, nonlinearity=None):
        super(LinearRandomProportional, self).__init__(fix_weights,
                                                 nonlinearity)
        self.dim = dim
        self.scale = scale
        self.coef = coef

    def draw_weights(self, num_weights):
        param = np.random.rand(num_weights)*2*self.scale - self.scale
        coords = np.ones((self.dim,num_weights))*param
        coords *= np.sign(np.array([np.random.randn(num_weights), 
                                    np.ones(num_weights)]))
        coords += np.random.randn(self.dim,num_weights)*0.05*self.scale
        coords[0,np.all(coords>0, axis=0)] *= self.coef
        coords[0,np.all(coords<0, axis=0)] /= self.coef
        return torch.tensor(coords).float()

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

class PointMass(DeepDistribution):
    def __init__(self, dim_z=None):
        super(PointMass, self).__init__()
        
    def distr(self, theta=None):
        """Return instance(s) of distribution, with parameters theta"""
        return None
        
    def sample(self, theta):
        """
        Sample from posterior, given parameters theta, using reparameterisation
        """
        return theta

class Bernoulli(DeepDistribution):
    """
    A family of distributions for deep generative models:
    Bernouli
    """
    
    def __init__(self, dim_z, prior_params=None):
        
        super(Bernoulli, self).__init__()
        
        self.ndim = dim_z
        
        if prior_params is None:
            prior_params = {'logits': torch.zeros(dim_z)}
        
        self.prior = D.bernoulli.Bernoulli(**prior_params)
        
    def distr(self, theta):
        """Return instance(s) of distribution, with parameters theta"""
        d = D.bernoulli.Bernoulli(logits=theta)
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
    A categorical distribution, parameterised by log-probabilities
    """
    
    def __init__(self, dim_z, prior_params=None):
        
        super(Categorical, self).__init__()
        
        self.ndim = dim_z
        
        if prior_params is None:
            prior_params = {'logits': torch.log(torch.ones(dim_z)/dim_z)}
        
        self.prior = D.categorical.Categorical(**prior_params)
        
    def distr(self, theta):
        """Return instance(s) of distribution, with parameters theta"""
        # d = D.categorical.Categorical(probs=theta.exp())
        d =  D.categorical.Categorical(logits=theta)
        return d
        
    def sample(self, theta):
        """
        Sample from variable, given parameters theta, using reparameterisation
        """
        z = torch.multinomial(theta)
        return z

#%% Models !!!
class NeuralNet(nn.Module):
    """Abstract class for all pytorch models, to enforce some regularity"""
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
    """Basic VAE class"""
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

class MultiGLM(NeuralNet):
    """A deep GLM model with multiple outputs"""
    def __init__(self, encoder, decoder, p_targ, p_latent=None, p_data=None):
        """
        Parameters
        ----------
        encoder : Pytorch Module
            Mapping from data (x) to code (z), a feedforward network.
        decoder : Pytorch Module
            Mapping from code (z) to the natural parameters of p_targ.
            Usually just a linear-nonlinear layer, e.g. linear-sigmoid for 
            logistic regression.
        p_targ : DeepDistribution
            Distributions of the targets, ideally from the exponential family.
        p_latent : DeepDistribution, optional
            Distribution of the latent code. The default (None) is a point
            mass (i.e. deterministic).
        p_data : DeepDistribution, optional
            Distribution of the data, to model noise in the inputs. The 
            default (None) is also deterministic.
        """
        
        super(MultiGLM,self).__init__()
        
        self.enc = encoder
        self.dec = decoder
        
        if p_latent is not None:
            if p_latent.name() == 'PointMass':
                p_latent = None
        self.latent = p_latent
        self.data = p_data
        
        self.obs = p_targ
        
    def forward(self, x):
        """
        Outputs the parameters of p_x, so that the likelihood can be evaluated
        """
        qz_params = self.enc(x) # encoding
        if self.latent is None:
            z = qz_params
        else:
            z = self.latent.sample(qz_params) # stochastic part
        
        py_params = self.dec(z) # decoding
        # recon_x = self.p_x(px_params) # draw outputs
        
        return py_params, qz_params, z
    
    def grad_step(self, data, optimizer):
        """ Single step of maximum likelihood over the data """

        running_loss = 0
        
        for i, batch in enumerate(data):
            optimizer.zero_grad()
            
            nums, labels = batch
            # nums = nums.squeeze(1).reshape((-1, 784))
            
            # # forward
            px_params, qz_params, z = self(nums)
            
            loss = -self.obs.distr(px_params).log_prob(labels).sum()
            if self.latent is not None:
                loss -= self.latent.distr(qz_params).log_prob(z).sum()
            
            # loss = -loglihood(self, nums, labels)
            # foo = self(nums)[0]
            # loss = nn.NLLLoss(reduction='sum')(foo, labels)

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

# def loglihood(model, x, y):
#     """
#     Log-likelihood of data (x,y) under model. 
#     """
    
#     py_params, qz_params, z = model(x)
    
#     # data likelihood 
#     # ll = model.obs.distr(py_params).log_prob(y).mean()
#     ll = model.obs.distr(py_params).log_prob(y).sum()
#     # regularisation (if distributions exist)
#     if model.latent is not None:
#         ll += model.latent.distr(qz_params).log_prob(z).sum()
        
#     # if model.data is not None:
#     #     p_x = model.data.distr()
    
#     return ll




