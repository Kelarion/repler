"""
Objects that learn things, to be imported in any scripts I run.

Current classes:
    - Basic VAE
"""

#%%

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
    def __init__(self, dim_layers, nonlinearity='ReLU', encoder=None, bias=True, 
        layer_type=None):
        super(Feedforward, self).__init__()
        
        onion = OrderedDict()
        self.ndim = dim_layers
        
        if type(nonlinearity) is str:
            nonlinearity = [nonlinearity for _ in dim_layers[1:]]

        if layer_type is None:
            self.layer_type = nn.Linear
        else:
            self.layer_type = layer_type
        
        if encoder is not None:
            # optionally include a pre-network encoder, e.g. if inputs are indices
            onion['embedding'] = encoder
        
        for l in range(len(dim_layers)-1):
            # onions have layers
            onion['layer%d'%l] = self.layer_type(dim_layers[l], dim_layers[l+1], bias=bias)
            if nonlinearity[l] is not None:
                if 'softmax' in nonlinearity[l].lower():
                    onion['link%d'%l] = getattr(nn, nonlinearity[l])(dim=-1)
                else:
                    onion['link%d'%l] = getattr(nn, nonlinearity[l])()
        
        self.network = nn.Sequential(onion)

        
    def forward(self, x):
        h = self.network(x)
        return h

class AttentionLayer(nn.Module):
    def __init__(self, N_in, n_head=1, N_qk=None, N_v=None, queries=False):
        """ Scaled dot-product attention, optional key, query, and value maps """
        super(AttentionLayer, self).__init__()

        self.h = n_head
        self.n_qk = N_qk
        self.n_v = N_v
        self.dim = N_in

        if N_qk is None: ## implement key matrix
            self.K = nn.Identity(N_in, N_in*n_head)
            self.Q = nn.Identity(N_in, N_in*n_head)
        else:
            self.K = nn.Linear(N_in, N_qk*n_head)
            if queries:
                self.Q = nn.Linear(N_in, N_qk*n_head)
            else:
                self.Q = self.K
        if N_v is None:
            self.V = nn.Identity(N_in, N_in*n_head)
        else:
            self.V = nn.Linear(N_in, N_v*n_head)

    def weights(self, x, mask):
        """ 
        x is shape (num_tok, *, dim_inp) 
        expects padded inputs to be nans!
        """

        # mask = x.mask
        # x_msk = torch.where(mask, torch.tensor(0.0), x)

        keys = self.K(x).view(*x.shape[:-1], self.h, self.n_qk)
        queries = self.Q(x).view(*x.shape[:-1], self.h, self.n_qk)

        kern = torch.einsum('i...j,k...j->...ik',keys,queries)/np.sqrt(self.dim)
        attn_mask = mask[...,None,None,:]

        kern = torch.where(attn_mask,kern,torch.tensor(-np.inf))
        return nn.Softmax(-1)(kern)

    def forward(self, x, mask):
        """ x is shape (num_tok, *, dim_inp) """
        # x = inps[0]
        # mask = inps[1]
        # print(len(inps))

        A = self.weights(x, mask)
        values = self.V(x).view(*x.shape[:-1], self.h, self.n_v)

        out = torch.einsum('...ij,j...k->i...k', A, values).reshape(*x.shape[:-1], self.h*self.n_v)
        return out

class ResNorm(nn.Module):
    """ a simple wrapper of another module with residual connection and layer norm """
    def __init__(self, module):
        super(ResNorm, self).__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        fx = x + self.module(x, *args, **kwargs)
        out = (fx - fx.mean(-1, keepdims=True))/(fx.std(-1, keepdims=True) + 1e-6)
        return out

class TinyTransformer(nn.Module):
    def __init__(self, dim_ff, num_layer, N_head, N_qk=None, N_v=None, queries=False, 
        linear_link=False, resnorm=True, **mlp_args):
        """ Separate linear maps for the keys, queries, and values are optional """

        super(TinyTransformer, self).__init__()

        if N_v is None:
            dim_out = dim_ff[-1]*N_head
        else:
            dim_out = N_v*N_head
        dim_in = dim_ff[0]

        self.linear_link = linear_link
        self.num_layer = num_layer
        self.h = N_head
        self.dim_ff = dim_ff

        # onion = OrderedDict()
        # for l in range(num_layer):
        #     if resnorm:
        #         onion['MLP%d'%l] = ResNorm(Feedforward(dim_ff, **mlp_args))
        #         onion['Attention%d'%l] = ResNorm(AttentionLayer(dim_ff[-1], 
        #             n_head=N_head, N_qk=N_qk, N_v=N_v, queries=queries))
        #     else:
        #         onion['MLP%d'%l] = Feedforward(dim_ff, **mlp_args)
        #         onion['Attention%d'%l] = AttentionLayer(dim_ff[-1], 
        #             n_head=N_head, N_qk=N_qk, N_v=N_v, queries=queries)

        #     if self.linear_link:
        #         onion['Linear%d'%l] = nn.Linear(dim_out, dim_in)

        # self.network = nn.Sequential(onion)
        self.mlps = []
        self.attn = []
        self.lins = []
        for l in range(num_layer):
            if resnorm:
                self.mlps.append(ResNorm(Feedforward(dim_ff, **mlp_args)))
                self.attn.append( ResNorm(AttentionLayer(dim_ff[-1], 
                    n_head=N_head, N_qk=N_qk, N_v=N_v, queries=queries)))
            else:
                self.mlps.append( Feedforward(dim_ff, **mlp_args))
                self.attn.append( AttentionLayer(dim_ff[-1], 
                    n_head=N_head, N_qk=N_qk, N_v=N_v, queries=queries))

            if self.linear_link:
                self.lins.append( nn.Linear(dim_out, dim_in))

    # def apply_mask(self, x, mask):
    #     return torch.where(mask.unsqueeze(-1), x, torch.tensor(np.nan))

    def forward(self, x, mask=None):
        """ the mask tells you which inputs are considered padding """
        
        for l in range(self.num_layer):
            z = self.mlps[l](x)
            x = self.attn[l](z, mask)

            if self.linear_link:
                x = self.lins[l](x)

        return x


class ClusteredConnections(nn.Module):
    """
    Single layer, but input dimensions have exclusive connections with specific output dimensions
    """
    def __init__(self, dim_inp, dim_out, nonlinearity='ReLU', embedding=None, bias=True):
        """
        dim_inp and dim_out are lists of the same length, where dim_inp[i] connects to dim_out[i]

        the full layer will be a mapping of dimension sum(dim_inp) -> sum(dim_out)
        """
        super(ClusteredConnections, self).__init__()
        
        self.n_inp = sum(dim_inp)
        self.n_out = sum(dim_out)
        
        if type(nonlinearity) is str:
            nonlinearity = [nonlinearity for _ in dim_layers[1:]]

        if layer_type is None:
            self.layer_type = nn.Linear
        else:
            self.layer_type = layer_type
        
        if embedding is not None:
            # optionally include a pre-network encoder, e.g. if inputs are indices
            self.embedding = embedding
        
        self.weight_list = []
        for d_i, d_o in zip(dim_inp, dim_out):
            self.weight_list.append(self.layer_type(d_i, d_o, bias=True))
        self.activation = getattr(nn, nonlinearity[l])()
        # for l in range(len(dim_layers)-1):
        #     # onions have layers
        #     onion['layer%d'%l] = self.layer_type(dim_layers[l], dim_layers[l+1], bias=bias)
        #     if nonlinearity[l] is not None:
        #         if 'softmax' in nonlinearity[l].lower():
        #             onion['link%d'%l] = getattr(nn, nonlinearity[l])(dim=-1)
        #         else:
        #             onion['link%d'%l] = getattr(nn, nonlinearity[l])()
        
        self.network = nn.Sequential(onion)
        
    def forward(self, x):
        
        return h

#%% Custom GRU, originally by Miguel but substantially changed
class CustomGRU(nn.Module):
    """
    A GRU class which gives access to the gate activations during a forward pass

    Supposed to mimic the organisation of torch.nn.GRU -- same parameter names
    """
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity=torch.tanh):
        """Mimics the nn.GRU module. Currently num_layers is not supported"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input weights
        self.weight_ih_l0 = Parameter(torch.Tensor(3*hidden_size, input_size))

        # hidden weights
        self.weight_hh_l0 = Parameter(torch.Tensor(3*hidden_size, hidden_size))

        # bias
        self.bias_ih_l0 = Parameter(torch.Tensor(3*hidden_size)) # input
        self.bias_hh_l0 = Parameter(torch.Tensor(3*hidden_size)) # hidden

        self.f = nonlinearity

        self.init_weights()

    def __repr__(self):
        return "CustomGRU(%d,%d)"%(self.input_size,self.hidden_size)

    def init_weights(self):
        for p in self.parameters():
            k = np.sqrt(self.hidden_size)
            nn.init.uniform_(p.data, -k, k)

    def forward(self, x, init_state, give_gates=False):
        """Assumes x is of shape (len_seq, batch, input_size)"""
        seq_sz, bs, _ = x.size()

        update_gates = torch.empty(seq_sz, bs, self.hidden_size)
        reset_gates = torch.empty(seq_sz, bs, self.hidden_size)
        hidden_states = torch.empty(seq_sz, bs, self.hidden_size)

        h_t = init_state

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[t, :, :]

            gi = F.linear(x_t, self.weight_ih_l0, self.bias_ih_l0) # do the matmul all together
            gh = F.linear(h_t, self.weight_hh_l0, self.bias_hh_l0)

            i_r, i_z, i_n = gi.chunk(3,1) # input currents
            h_r, h_z, h_n = gh.chunk(3,2) # hidden currents

            r_t = torch.sigmoid(i_r + h_r)
            z_t = torch.sigmoid(i_z + h_z)
            n = self.f(i_n + r_t*h_n)
            h_t = n + z_t*(h_t - n)

            update_gates[t,:,:] = z_t
            reset_gates[t,:,:] = r_t
            hidden_states[t,:,:] = h_t

        output = hidden_states

        if give_gates:
            return output, h_t, (update_gates, reset_gates)
        else:
            return output, h_t

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

class BinaryWeights(nn.Linear):
    """
    Abstract class for linear layers with binary weights
    """
    def __init__(self, *args, **kwargs):
        super(BinaryWeights, self).__init__(*args, bias=False, **kwargs)
        self.__name__ = self.__class__.__name__
        #     self.w = self.draw_weights()

class PositiveReadout(BinaryWeights):
    def __init__(self, N_in, N_out, bias=None):

        super(PositiveReadout, self).__init__(N_in, N_out)

        self.weight = Parameter(torch.ones(N_out, N_in).float())

class BinaryReadout(BinaryWeights):
    def __init__(self, N_in, N_out, shuffle=False, bias=None, rotated=False):

        super(BinaryReadout, self).__init__(N_in, N_out)

        if rotated:
            bits = np.concatenate([np.eye(N_out)*i for i in [1,-1]])
        else:
            bits = 2*(np.mod(np.arange(2**N_out)[:,None]//(2**np.arange(N_out)[None,:]),2)) - 1

        num_pop = len(bits)
        num_per_pop = N_in//num_pop

        which_pop = np.arange(num_per_pop*num_pop)//num_per_pop
        leftovers = np.random.choice(num_pop, N_in - num_per_pop*num_pop, replace=False)
        if shuffle:
            which_pop = np.random.permutation(np.append(which_pop, leftovers))
        else:
            which_pop = np.append(which_pop, leftovers)

        self.weight = Parameter(torch.tensor(bits[which_pop,:].T).float())


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

class GausIdMixture(DeepDistribution):
    """
    A family of distributions for deep generative models:
    Mixture of Gaussians with identity covariance
    
    This module relates the output of a neural net to the parameters of a 
    gaussian distribution, assuming first N are the mean, and second N are
    the log-variances of each dimension.
    """
    
    def __init__(self, dim_z, gaus_prior_params=None, cat_prior_params=None):
        
        super(GausIdMixture, self).__init__()
        
        self.ndim = dim_z
        
        if prior_params is None:
            prior_params = {'loc': torch.zeros(dim_z),
                         'covariance_matrix': torch.eye(dim_z)}
        if cat_prior_params is None:
            prior_params = {'logits': torch.zeros(dim_z)}

        comp = D.categorical.Categorical(**cat_prior_params)
        mix = D.multivariate_normal.MultivariateNormal(**gaus_prior_params)
        
        self.prior = D.mixture_same_family.MixtureSameFamily(**prior_params)
        
    def distr(self, theta):
        """Return instance(s) of distribution, with parameters theta"""
        mu1, mu2 = theta.chunk(2, dim=1)
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
            loss = -self.obs.distr(px_params).log_prob(labels).mean()
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

class GenericRNN(NeuralNet):
    def __init__(self, ninp, nhid, out_dist, nlayers=1, rnn_type='relu',
        recoder=None, decoder=None, fix_decoder=True, z_dist=None):
        super(GenericRNN,self).__init__()

        self.recoder = recoder
        nout = out_dist.ndim
        self.obs = out_dist
        if z_dist is not None: # implement activity regularization
            self.hidden_dist = z_dist
        else:
            self.hidden_dist = None
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        elif rnn_type in ['tanh-GRU', 'relu-GRU']:
            nlin = getattr(torch, rnn_type.split('-GRU')[0])
            self.rnn = CustomGRU(ninp, nhid, nlayers, nonlinearity=nlin) # defined below
        else:
            try:
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=rnn_type.lower())
            except:
                raise ValueError("Invalid rnn_type: give from {'LSTM', 'GRU', 'tanh', 'relu'}")

        if decoder is None:
            self.decoder = nn.Linear(nhid, nout)
        else:
            self.decoder = decoder
        if fix_decoder:
            self.decoder.requires_grad = False
        # self.softmax = nn.LogSoftmax(dim=2)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_()
        self.rnn.weight_hh_l0.data.normal_(0,1.0/np.sqrt(self.nhid))
        self.rnn.bias_hh_l0.data.zero_()
        self.rnn.weight_ih_l0.data.normal_(0,1.0/np.sqrt(self.nhid))
        self.rnn.bias_ih_l0.data.zero_()

    def forward(self, emb, hidden=None, give_gates=False, debug=False, only_final=True):
        """
        Run the RNN forward. Expects input to be (lseq,nseq,...)
        Only set give_gates=True if it's the custom GRU!!!
        use `debug` argument to return also the embedding the input
        """

        # if self.recoder is None:
        #     emb = inp
        # else:
        #     emb = self.recoder(inp)

        if hidden is None:
            hidden = self.init_hidden(emb.shape[1])
        # if emb.dim()<3:
        #     emb = emb.unsqueeze(0)

        if give_gates:
            output, hidden, extras = self.rnn(emb, hidden, give_gates)
        else:
            output, hidden = self.rnn(emb, hidden)
        # print(output.shape)

        # decoded = self.softmax(self.decoder(output))
        decoded = self.decoder(output)
        if only_final:
            decoded = decoded[-1,...] # assume only final timestep matters

        if give_gates:
            return decoded, hidden, extras
        else:
            return decoded, hidden

    def transparent_forward(self, inp, hidden=None, give_gates=False, debug=False):
        """
        Run the RNNs forward function, but returning hidden activity throughout the sequence

        it's slower than regular forward, but often necessary
        """

        lseq = inp.shape[0]
        nseq = inp.shape[1]
        # ispad = (input == self.padding)

        if hidden is None:
            hidden = self.init_hidden(nseq)

        H = torch.zeros(lseq, self.nhid, nseq)
        if give_gates:
            Z = torch.zeros(lseq, self.nhid, nseq)
            R = torch.zeros(lseq, self.nhid, nseq)
        
        # because pytorch only returns hidden activity in the last time step,
        # we need to unroll it manually. 
        O = torch.zeros(lseq, nseq, self.decoder.out_features)
        if self.recoder is None:
            emb = inp
        else:
            emb = self.recoder(inp)
        for t in range(lseq):
            if give_gates:
                out, hidden, ZR = self.rnn(emb[t:t+1,...], hidden, give_gates=True)
                Z[t,:,:] = ZR[0].squeeze(0).T
                R[t,:,:] = ZR[1].squeeze(0).T
            else:
                out, hidden = self.rnn(emb[t:t+1,...], hidden)
            dec = self.decoder(out)
            # naan = torch.ones(hidden.squeeze(0).shape)*np.nan
            # H[t,:,:] = torch.where(~ispad[t:t+1,:].T, hidden.squeeze(0), naan).T
            H[t,:,:] = hidden.squeeze(0).T
            O[t,:,:] = dec.squeeze(0)

        if give_gates:
            if debug:
                return O, H, Z, R, emb
            else:
                return O, H, Z, R
        else:
            if debug:
                return O, H, emb
            else:
                return O, H

    def init_hidden(self, bsz):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(1, bsz, self.nhid),
                    torch.zeros(1, bsz, self.nhid))
        else:
            return torch.zeros(1, bsz, self.nhid)

    def grad_step(self, data, optimizer, init_state=False, only_final=True):
        """ Single step of maximum likelihood over the data """

        running_loss = 0
        
        for i, batch in enumerate(data):
            optimizer.zero_grad()
            
            if init_state:
                nums, labels, hidden = batch
                hidden = hidden[None,...]
            else:
                nums, labels = batch
                hidden = self.init_hidden(nums.size(0))

            if not only_final:
                labels = labels.transpose(0,1)

            nums = nums.transpose(0,1)

            # # forward
            if self.hidden_dist is None:
                out, _ = self(nums, hidden, only_final=only_final)
                loss = -self.obs.distr(out).log_prob(labels).mean()
            else:
                out, hid = self.transparent_forward(nums, hidden)
                loss = -self.obs.distr(out).log_prob(labels).mean() \
                - self.hidden_dist.prior.log_prob(hid.transpose(-1,-2)).mean()

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




