CODE_DIR = '/home/matteo/Documents/github/repler/src/'
SAVE_DIR = '/home/matteo/Documents/uni/columbia/bleilearning/'

import os, sys
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from itertools import permutations

import students
import assistants

#%% Tasks
class IndependentBinary(object):
    """Abstract class encompassing all classifications of multiple binary variables"""
    def __init__(self):
        super(IndependentBinary,self).__init__()
        self.__name__ = self.__class__.__name__
        
    def __call__(self):
        raise NotImplementedError
    
    def correct(self, pred, targets):
        n = (targets.detach().numpy() == (pred.detach().numpy()>=0.5)).sum(0, keepdims=True)
        return n.astype(float)

class Classification(object):
    def __init__(self):
        super(Classification,self).__init__()
        self.__name__ = self.__class__.__name__
        
    def __call__(self):
        raise NotImplementedError
    
    def correct(self, pred, targets):
        n = (targets.detach().numpy() == pred.detach().numpy().argmax(-1)).sum(0, keepdims=True)
        return n.astype(float)

class ParityMagnitude(IndependentBinary):
    def __init__(self):
        super(ParityMagnitude,self).__init__()
        self.num_var = 2
        self.dim_output = 2
        
        self.obs_distribution = students.Bernoulli(2)
        self.link = None
    
    def __call__(self,digits):
        parity = np.mod(digits.targets, 2).float()
        magnitude = (digits.targets>=5).float()
        return torch.cat((parity[:,None], magnitude[:,None]), dim=1)\

class ParityMagnitudeFourunit(IndependentBinary):
    def __init__(self):
        super(ParityMagnitudeFourunit,self).__init__()
        self.num_var = 4
        self.dim_output = 4
        
        self.obs_distribution = students.Bernoulli(4)
        self.link = None
    
    def __call__(self, digits):
        """Compute the parity and magnitude of digits"""
        parity = np.mod(digits.targets, 2).float()>0
        magnitude = (digits.targets>=5)
        return torch.cat((parity[:,None], ~parity[:,None], 
                          magnitude[:,None], ~magnitude[:,None]), dim=1).float()

class ParityMagnitudeEnumerated(Classification):
    def __init__(self):
        super(ParityMagnitudeEnumerated,self).__init__()
        self.num_var = 1
        self.dim_output = 4
        
        self.obs_distribution = students.Categorical(4)
        # self.link = 'LogSoftmax'
        self.link = None
    
    def __call__(self,digits):
        """Compute the parity and magnitude"""
        parity = np.mod(digits.targets, 2).float()
        magnitude = (digits.targets>=5).float()
        parmag = 1*parity + 2*magnitude
        return  (1*parity + 2*magnitude)

class DigitsBitwise(IndependentBinary):
    """Digits represented as n-bit binary variables"""
    def __init__(self, n=3):
        super(DigitsBitwise,self).__init__()
        self.num_var = n
        self.dim_output = n
        self.obs_distribution = students.Bernoulli(n)
        # self.link = 'Sigmoid'
        self.link = None
    
    def __call__(self,digits):
        targ = digits.targets-1
        bits = torch.stack([(targ&(2**i))/2**i for i in range(self.num_var)]).float().t()
        return bits

class Digits(Classification):
    def __init__(self, start=1, stop=8, noise=None):
        super(Digits,self).__init__()
        n = stop-start+1
        self.start = start
        self.num_var = 1
        self.dim_output = n
        if noise is None:
            self.obs_distribution = students.Categorical(n)
        else:
            self.obs_distribution = noise
        # self.link = 'LogSoftmax'
        self.link = None
    
    def __call__(self, digits):
        return digits.targets - self.start

class RandomDichotomies(IndependentBinary):
    def __init__(self, n=3):
        super(RandomDichotomies,self).__init__()
        self.num_var = n
        self.dim_output = n
        
        conds = np.arange(2**n)
        pos = []
        for d in range(n):
            these_pos = np.sort(np.random.permutation(conds)[:2**(n-1)])
            if not np.any([np.all(np.isin(these_pos,p)) for p in pos]):
                pos.append(these_pos)
        self.positives = pos

        self.obs_distribution = students.Bernoulli(n)
        # self.link = 'Sigmoid'
        self.link = None
    
    def __call__(self, digits):
        these = torch.tensor([np.isin(digits.targets, p) for p in self.positives]).float()
        return these.t()

#%%
def dependence_statistics(x):
    """Assume x is (..., n_sample)"""
    
    x_kl = np.abs(x[...,None] - x[...,None,:]) 
    x_k = x_kl.mean(-2, keepdims=True)
    x_l = x_kl.mean(-1, keepdims=True)
    x_ = x_kl.mean((-2,-1), keepdims=True)
    
    return x_kl - x_k - x_l + x_

# def distance_covariance(x, y):
#     A = dependence_statistics(x)
#     B = dependence_statistics(y)
#     return np.max([0, np.mean(A*B)]) # this should be non-negative anyway ...

def distance_covariance(X, joint=True):
    """
    concatenation of variables into (n_var, n_sample)
    save time by setting joint=False, if you only want diagonal
    """
    D = dependence_statistics(X)
    if joint:
        return (D[None,...]*D[:,None,...]).mean((-2,-1)) 
    else:
        return (D*D).mean((-2,-1))

# def distance_correlation(x, y):
#     V_xy = distance_covariance(x, y)
#     V_x = distance_covariance(x, x)
#     V_y = distance_covariance(y, y)
#     # print([V_x, V_y, V_xy])
#     if 0 in [V_x, V_y]:
#         return 0
#     else:
#         return np.sqrt(V_xy/np.sqrt(V_x*V_y))

def distance_correlation(X):
    V = distance_covariance(X)
    V_x = np.diag(V)
    normlzr = V_x[None,:]*V_x[:,None]
    R = np.zeros(V.shape)
    R[normlzr>0] = np.sqrt(V[normlzr>0]/np.sqrt(normlzr[normlzr>0]))
    return R


def rbf_kernel(X, sigma=1, p=2):
    """X is (n_sample, n_dim)"""
    # pairwise_dists = squareform(pdist(X, 'minkowski', p))
    K = np.exp((np.abs(X[...,None] - X[...,None,:])**p)/(2*sigma**p))
    


#%% miscellaneous
    
def decimal(binary):
    """ convert binary vector to dedimal number (i.e. enumerate) """
    d = (binary*(2**np.arange(binary.shape[1]))[None,:]).sum(1)
    return d

