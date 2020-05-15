"""
A whole lot of classes that I find make it easier to do hypberbolic embeddings.
"""

import numpy as np
import pandas
import torch 
import torch.nn as nn
import torch.nn.functional as F

import scipy.special as spc
import scipy.linalg as la
import scipy.sparse as sprs

from torch.autograd import Function

#%% Non-parametric embedding
class Hyperboloid(nn.Embedding):
    """An embedding on the (upper two-sheet) hyperboloid, from Nickel and Kiela"""
    def __init__(self, n_obj, embedding_dim, init_range=1e-3, 
                 max_norm=1e6, norm_clip=1, padding=False):

        # super(Hyperboloid,self).__init__(n_obj, embedding_dim)
        if padding:
            super(Hyperboloid,self).__init__(n_obj+1, embedding_dim, padding_idx=n_obj)
        else:
            super(Hyperboloid,self).__init__(n_obj, embedding_dim)

        self.dim = embedding_dim
        
        self.weight.data.uniform_(-init_range, init_range)
        self.normalize(self.weight.data)
        
        self.max_norm = max_norm # to avoid float precision errors
        self.norm_clip = norm_clip

        self.dec = nn.Linear(1,1) # for implementing attention

        # raise NotImplementedError('Still got to normalize!!')
        
    def distances(self, inp):
        """
        Apply self.dist() to inp in a particular way
        
        inp is a tensor of indices, with shape (bsz, 2+N_neg). 
        
        inp[:,2:] are N_neg samples from N(i,j)
        inp[:,0:2] are [i,j]
        """
        e = self(inp)
        
        u_jk = e.narrow(-2, 1, e.size(-2)-1) # the neighbour set around u_i
        u_i = e.narrow(-2,0,1).expand_as(u_jk) # the point in question
        
        d = self.dist(u_i, u_jk)
        return d
    
    @staticmethod
    def inner(x, y, keepdim=False):
        """Lorenzian scalar product
        Assumes that vector indices are along last dimension"""
        sp = x*y
        sp.narrow(-1,0,1).mul_(-1)
        return torch.sum(sp, dim=-1, keepdim=keepdim)

    def normalize(self, x):
        """Ensure that x lies on manifold"""
        d = x.size(-1) - 1
        narrowed = x.narrow(-1, 1, d)
        if self.max_norm:
            narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)

        tmp = 1 + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
        tmp.sqrt_()
        x.narrow(-1, 0, 1).copy_(tmp)
        return x
    
    def dist(self,x,y,eps=1e-5):
        """Distance function on hyperboloid"""
        sp = -LorentzDot.apply(x,y)
        sp.data.clamp_(min=1)
        return Acosh.apply(sp, eps)
        
    def metric_inverse(self,x):
        """Inverse of Riemannian metric at point x in M"""
        g = torch.eye(self.dim)
        g[0,0] = -1
        return g
    
    def proj(self,x,u):
        """Project vector u onto the tangent space at point x"""
        p = torch.addcmul(u, self.inner(x,u, keepdim=True).expand_as(x), x)
        return p
    
    def rgrad(self, x, u):
        """Riemannian gradient of hyperboloid, combines metric_invers and proj"""
        u_ = u
        u_.narrow(-1,0,1).mul_(-1)
        u_.addcmul_(self.inner(x, u_, keepdim=True).expand_as(x), x)
        return u_

    def expmap(self,x,u):
        """Exponential map, exp_{x}(u)"""
        # is it safe to clamp positive? it shouldn't be negative anyway...
        u_norm = (self.inner(u, u, keepdim=True)).clamp_(min=0).sqrt()
        nrm = torch.clamp(u_norm, max=self.norm_clip)
        u_norm.clamp_(min=1e-10)
        emap = torch.cosh(nrm)*x + torch.sinh(nrm)*(u/u_norm)
        return emap

#%% Parametric encoders
class EuclideanEncoder(nn.Module):
    """
    Abstract class for endowing an encoder with `pythagorean` pseudo-distance. 

    """
    def __init__(self, enc, max_norm=1e8):
        super(EuclideanEncoder, self).__init__()
        self.enc = enc
        self.max_norm = max_norm

    def init_weights(self, test_inp):
        raise NotImplementedError
    
    def forward(self, inp):
        """Take the input to the network (e.g. tokens in indicator format)
        and return an embedding which lies on the hyperboloid"""
        
        u = self.enc(inp)
        if self.max_norm: # norm clipping
            u = u.renorm(p=2, dim=-1, maxnorm=self.max_norm)
        return u

    def invchart(self, inp):
        return inp

    # generic methods of the Lorentz model
    @staticmethod
    def inner(x, y, keepdim=False):
        """Lorenzian scalar product
        Assumes that vector indices are along last dimension"""
        sp = x*y
        return torch.sum(sp, dim=-1, keepdim=keepdim)
    
    @staticmethod
    def dist(x,y,eps=1e-5, p=2):
        """Distance function on hyperboloid"""
        d = torch.norm(x-y, p=p, dim=-1).pow(p)
        return d

class HyperboloidEncoder(nn.Module):
    """
    Abstract class for endowing an encoder with hyperbolic distance. 
    Reparameterizes the output of encoder as the domain of some mapping 
    onto the upper component of the two-sheet hyperboloid.
    """
    def __init__(self):
        super(HyperboloidEncoder, self).__init__()

    def init_weights(self, test_inp):
        raise NotImplementedError
    
    def forward(self, inp):
        """Take the input to the network (e.g. tokens in indicator format)
        and return an embedding which lies on the hyperboloid"""
        
        u = ManifoldFunction.apply(self.enc(inp))
        if self.max_norm: # norm clipping
            u = u.renorm(p=2, dim=-1, maxnorm=self.max_norm)
        z = self.chart(u)
        # z = self.chart(u)
        return z
    
    # def backward(self, grad_):

    def chart(self, inp):
        """
        Apply the `chart' from pseudo-polar coordinates into hyperboloid
        Assumes the vectors are along the last dimension of inp
        """
        raise NotImplementedError

    def normalize(self, x):
        """Ensure that x lies on manifold"""
        d = x.size(-1) - 1
        narrowed = x.narrow(-1, 1, d)
        if self.max_norm:
            narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)

        tmp = 1 + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
        tmp.sqrt_()
        x.narrow(-1, 0, 1).copy_(tmp)
        return x

    # generic methods of the Lorentz model
    @staticmethod
    def inner(x, y, keepdim=False):
        """Lorenzian scalar product
        Assumes that vector indices are along last dimension"""
        sp = x*y
        sp.narrow(-1,0,1).mul_(-1)
        return torch.sum(sp, dim=-1, keepdim=keepdim)
    
    @staticmethod
    def dist(x,y,eps=1e-5):
        """Distance function on hyperboloid"""
        sp = -LorentzDot.apply(x,y)
        sp.data.clamp_(min=1)
        return Acosh.apply(sp, eps)

    @staticmethod
    def to_poincare(inp):
        """Take points on hyperboloid and map to poincare disc"""
        return inp.narrow(-1,1,inp.shape[-1]-1)/(1+inp.narrow(-1,0,1))

    def expmap(self,p,u):
        """Exponential map, exp_{x}(u)"""
        # is it safe to clamp positive? it shouldn't be negative anyway...
        u_norm = (self.inner(u, u, keepdim=True)).clamp_(min=0).sqrt()
        nrm = torch.clamp(u_norm, max=self.norm_clip)
        u_norm.clamp_(min=1e-10)
        emap = torch.cosh(nrm)*p + torch.sinh(nrm)*(u/u_norm)
        return emap

    def logmap(self,p,z,eps=1e-5):
        """ Logarithmic map"""
        alph = -self.inner(p, z, keepdim=True)
        scale = Acosh.apply(alph, eps)/torch.sqrt((alph+eps).pow(2)-1)
        u = scale*(z-alph*p)
        return u

    def distances(self, inp):
        """
        Apply self.dist() to inp in a particular way
        
        inp is a tensor of indices, with shape (bsz, 2+N_neg). 
        
        inp[:,2:] are N_neg samples from N(i,j)
        inp[:,0:2] are [i,j]
        """
        e = self(inp)
        
        u_jk = e.narrow(-2, 1, e.size(-2)-1) # the neighbour set around u_i
        u_i = e.narrow(-2,0,1).expand_as(u_jk) # the point in question
        
        d = self.dist(u_i, u_jk)
        
        return d

class PseudoPolar(HyperboloidEncoder):
    """
    Interprets the encoder output as pseudo-polar coordinates, with a 
    'radius' being x0 and angle specifying the rest. 

    This is lifted from Gulchere et al. (2019) Hyperbolic Attention Networks
    """
    def __init__(self, encoder, max_norm=1e6, dec=None):
        """encoder should output points in euclidean space
        by default implements very severe norm clipping to avoid infs
        """
        super(PseudoPolar, self).__init__()
        
        self.enc = encoder
        self.max_norm = max_norm
        
        self.dec = dec
        
    def init_weights(self, test_inp=None, these_weights=None):
        """
        Do a hacky initialisation to begin with a centred output
        Assumes that the encoder has """
        
        # final_bias = list(self.enc.network.children())[-2].bias
        lyrs = [module for name,module in self.enc.network.named_modules() if 'layer' in name]
        final_bias = lyrs[-1].bias
        final_w = lyrs[-1].weight

        if test_inp is not None:
            test_out = self.enc(test_inp)
            if final_bias is not None:
                final_bias.data = final_bias.data*0 #- test_out.mean(0)
            final_w.data = final_w.data/10
        elif these_weights is not None:
            final_w.data.copy_(these_weights)
    
    def chart(self, inp):
        """
        Apply the `chart' from pseudo-polar coordinates into hyperboloid
        Assumes the vectors are along the last dimension of inp
        """
        k = inp.shape[-1]
        z0 = torch.cosh(inp.narrow(-1,0,1))
        
        d = F.normalize(inp.narrow(-1,1,k-1), p=2, dim=-1)
        # d_norm = torch.norm(inp.narrow(-1,1,k-1), p=2, dim=-1, keepdim=True)
        
        ztwiddle = torch.sinh(inp.narrow(-1,0,1)).mul(d)
        
        return torch.cat((z0, ztwiddle), -1)

class TangentSpace(HyperboloidEncoder):
    """
    Interprets the encoder output as points on the tangent space of H *at the
    point (1,0,0,...)*. 

    This is from Nagano, alia, Koyama (2019) on the Hyperbolic Normal distribution.
    """
    def __init__(self, encoder, max_norm=1e8, dec=None):
        """encoder should output points in euclidean space
        by default implements very severe norm clipping to avoid infs
        """
        super(TangentSpace, self).__init__()

        self.enc = encoder
        self.max_norm = max_norm
        
        self.dec = dec
        
    def init_weights(self, test_inp=None, these_weights=None):
        """
        Do a hacky initialisation to begin with a centred output
        Assumes that the encoder has """
        
        # final_bias = list(self.enc.network.children())[-2].bias
        lyrs = [module for name,module in self.enc.network.named_modules() if 'layer' in name]
        final_bias = lyrs[-1].bias
        final_w = lyrs[-1].weight

        if test_inp is not None:
            test_out = self.enc(test_inp)
            if final_bias is not None:
                final_bias.data = final_bias.data*0 #- test_out.mean(0)
            final_w.data = final_w.data/10
        elif these_weights is not None:
            final_w.data.copy_(these_weights)

    def chart(self, inp):
        """
        Apply the exponential map, defined at the `origin', to inp
        This function assumes inp is in the tangent space of the origin, 
        and so that the first coordinate is 0. So, inp just contains the 
        subsequent coordinates.
        """
        k = inp.shape[-1]

        # implement the exponential map at the origin
        inp_norm = torch.norm(inp, p=2, dim=-1, keepdim=True)
        d = F.normalize(inp, p=2, dim=-1)

        h0 = torch.cosh(inp_norm)
        h_ = torch.sinh(inp_norm).mul(d)
        
        return torch.cat((h0, h_), dim=-1)
        # return d
    
    def invchart(self, emb, eps=1e-5):
        """
        Apply logarithmic map to get the tangent vector which produces an embedding
        """
        k = emb.shape[-1]
        alph = emb.narrow(-1,0,1)
        scale = Acosh.apply(alph, eps)/torch.sqrt((alph+eps).pow(2)-1)
        z = scale.mul(emb.narrow(-1,1,k-1))
        return z

class CartesianHyperboloid(HyperboloidEncoder):
    """
    Interprets points as the pre-image of the global chart onto the hyperboloid.
    """
    def __init__(self, encoder, max_norm=1e8, dec=None):
        """encoder should output points in euclidean space
        by default implements very severe norm clipping to avoid infs
        """
        super(CartesianHyperboloid, self).__init__()

        self.enc = encoder
        self.max_norm = max_norm
        
        self.dec = dec
        
    def init_weights(self, test_inp=None, these_weights=None):
        """
        Do a hacky initialisation to begin with a centred output
        Assumes that the encoder has """
        
        # final_bias = list(self.enc.network.children())[-2].bias
        lyrs = [module for name,module in self.enc.network.named_modules() if 'layer' in name]
        final_bias = lyrs[-1].bias
        final_w = lyrs[-1].weight

        if test_inp is not None:
            test_out = self.enc(test_inp)
            if final_bias is not None:
                final_bias.data = final_bias.data*0 #- test_out.mean(0)
            final_w.data = final_w.data/10
        elif these_weights is not None:
            final_w.data.copy_(these_weights)
    
    def chart(self, inp):
        """
        Map inp onto the hyperboloid using the global chart
        """
        k = inp.shape[-1]

        inp_norm = torch.norm(inp, p=2, dim=-1, keepdim=True)
        # d = F.normalize(inp, p=2, dim=-1)

        h0 = torch.sqrt(1+inp_norm.pow(2))
        # h_ = inp
        
        return torch.cat((h0, inp), dim=-1)
        # return d
    
    def invchart(self, emb, eps=1e-5):
        """
        Get the preimage of emb under the chart.
        """
        return emb.narrow(-1,1,emb.shape[-1]-1)

class GeodesicCoordinates(HyperboloidEncoder):
    """
    ONLY WORKS FOR 2 DIMENSIONS.
    """
    def __init__(self, encoder, max_norm=1e8, dec=None):
        """encoder should output points in euclidean space
        by default implements very severe norm clipping to avoid infs
        """
        super(GeodesicCoordinates, self).__init__()

        self.enc = encoder
        self.max_norm = max_norm
        
        self.dec = dec
        
    def init_weights(self, test_inp=None, these_weights=None):
        """
        Do a hacky initialisation to begin with a centred output
        Assumes that the encoder has """
        
        # final_bias = list(self.enc.network.children())[-2].bias
        lyrs = [module for name,module in self.enc.network.named_modules() if 'layer' in name]
        final_bias = lyrs[-1].bias
        final_w = lyrs[-1].weight

        if test_inp is not None:
            test_out = self.enc(test_inp)
            if final_bias is not None:
                final_bias.data = final_bias.data*0 #- test_out.mean(0)
            final_w.data = final_w.data/10
        elif these_weights is not None:
            final_w.data.copy_(these_weights)
    
    def chart(self, inp):
        """
        Map inp onto the hyperboloid using the global chart
        """
        k = inp.shape[-1]

        # inp_norm = torch.norm(inp, p=2, dim=-1, keepdim=True)
        # d = F.normalize(inp, p=2, dim=-1)

        cv = torch.cosh(inp.narrow(-1,1,1)).squeeze()
        cu = torch.cosh(inp.narrow(-1,0,1)).squeeze()
        sv = torch.sinh(inp.narrow(-1,1,1)).squeeze()
        su = torch.sinh(inp.narrow(-1,0,1)).squeeze()
        # h_ = inp
        
        return torch.stack((cu*cv, cv*su, sv), dim=-1)
        # return d
    
    def invchart(self, emb, eps=1e-5):
        """
        Get the preimage of emb under the chart.
        """
        nump = emb.detach().numpy()
        v = np.arcsinh(nump[...,2])
        # v = torch.acos(w[...,0]/torch.cos(u))
        u = np.arctanh(nump[...,1]/nump[...,0])
        return torch.tensor(np.stack((u,v)))

#%% Helper classes 
class ManifoldFunction(Function):
    """
    To implement Riemannian gradients during backward pass
    Applying this function says 'the input is on the hyperboloid'
    """
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, g):
        inp, = ctx.saved_tensors
        coshv = torch.cosh(inp.narrow(-1,1,1)).pow(2) + 1e-5
        g.narrow(-1,0,1).mul_(coshv.pow(-1))
        return g

# these are taken from Nickel & Kiela's code
class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        z = g / z
        return z, None

class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return Hyperboloid.inner(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u