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
import scipy.linalg as la
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
        return (targets == (pred>=0.5)).sum(0).float()

class Classification(object):
    def __init__(self):
        super(Classification,self).__init__()
        self.__name__ = self.__class__.__name__
        
    def __call__(self):
        raise NotImplementedError
    
    def correct(self,pred, targets):
        return (targets == pred.argmax(-1)).sum(0).float()

# MNIST tasks
class ParityMagnitude(IndependentBinary):
    def __init__(self):
        super(ParityMagnitude,self).__init__()
        self.num_var = 2
        self.dim_output = 2
        
        self.obs_distribution = students.Bernoulli(2)
        self.link = None

        self.positives = [np.array([0,2,4,6]),np.array([0,1,2,3])]
    
    def __call__(self,labels):
        parity = (np.mod(labels, 2)==0).float()
        magnitude = (labels<4).float()
        return torch.cat((parity[:,None], magnitude[:,None]), dim=1)\

class ParityMagnitudeFourunit(IndependentBinary):
    def __init__(self):
        super(ParityMagnitudeFourunit,self).__init__()
        self.num_var = 4
        self.dim_output = 4
        
        self.obs_distribution = students.Bernoulli(4)
        self.link = None
    
    def __call__(self, labels):
        """Compute the parity and magnitude of digits"""
        parity = np.mod(labels, 2).float()>0
        magnitude = (labels>=5)
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
    
    def __call__(self, labels):
        """Compute the parity and magnitude"""
        parity = np.mod(labels, 2).float()
        magnitude = (labels<4).float()
        return  (1*parity + 2*magnitude)

class DigitsBitwise(IndependentBinary):
    """Digits represented as n-bit binary variables"""
    def __init__(self, n=3):
        super(DigitsBitwise,self).__init__()
        self.num_var = n
        self.dim_output = n
        self.obs_distribution = students.Bernoulli(n)
        self.link = None
    
    def __call__(self,labels):
        targ = labels-1
        bits = torch.stack([(targ&(2**i))/2**i for i in range(self.num_var)]).float().T
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
    
    def __call__(self, labels):
        return labels - self.start

class RandomDichotomies(IndependentBinary):
    def __init__(self, c, n, overlap=0):
        """overlap is given as the log2 of the dot product on their +/-1 representation"""
        super(RandomDichotomies,self).__init__()
        self.__name__ = 'RandomDichotomies_%d-%d-%d'%(c, n, overlap)
        self.num_var = n
        self.dim_output = n
        self.num_cond = c
        if n>c:
            raise ValueError('Cannot have more dichotomies than conditions!!')

        if overlap == 0:
            # generate uncorrelated dichotomies, only works for powers of 2
            H = la.hadamard(c)[:,1:]
            pos = np.nonzero(H[:,np.random.choice(c-1,n,replace=False)]>0)
            self.positives = [pos[0][pos[1]==d] for d in range(n)]
        elif overlap == 1:
            prot = 2*(np.random.permutation(c)>=(c/2))-1
            pos = np.where(prot>0)[0]
            neg = np.where(prot<0)[0]
            idx = np.random.choice((c//2)**2, n-1, replace=False)
            # print(idx)
            swtch = np.stack((pos[idx%(c//2)],neg[idx//(c//2)])).T
            # print(swtch)
            ps = np.ones((n-1,1))*prot
            ps[np.arange(n-1), swtch[:,0]] *= -1
            ps[np.arange(n-1), swtch[:,1]] *= -1
            pos = [np.nonzero(p>0)[0] for p in ps]
            pos.append(np.nonzero(prot>0)[0])
            self.positives = pos

        self.obs_distribution = students.Bernoulli(n)
        self.link = None
    
    def __call__(self, labels):
        these = torch.tensor([np.isin(labels, p) for p in self.positives]).float()
        return these.T

# Random pattern tasks
# class GaussGaussBern(IndependentBinary):
#     def __init__(self, n):
#         super(GaussGaussBern,self).__init__()
#         self.num_var = n
#         self.dim_output = n
        
#         self.obs_distribution = students.Bernoulli(n)
#         self.link = None

#     def __call__(self, ):

#%% miscellaneous functions
def decimal(binary):
    """ convert binary vector to dedimal number (i.e. enumerate) """
    d = (binary*(2**np.arange(binary.shape[1]))[None,:]).sum(1)
    return d

