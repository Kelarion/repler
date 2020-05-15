"""
Classes that standardise how I handle datasets, when fitting hyperbolic
embeddings. 
"""

import numpy as np
import pandas
import torch 
import torch.nn as nn

import os
import linecache

import scipy.special as spc
import scipy.linalg as la
import scipy.sparse as sprs

#%% Different data iterators
class SparseGraphDataset(object):
    """Represent dataset as an iterator which, can loop through the data 
    in batches"""
    
    def __init__(self, idx, weights, obj, bsz, n_neg=10):
        super(SparseGraphDataset, self).__init__()
        
        self.idx = torch.tensor(idx).long()
        self.weights = sprs.csr_matrix((weights, (idx[:,0], idx[:,1])),
                                        shape=(len(obj),len(obj)))
        self.weights_iter = torch.tensor(weights)
        self.names = obj
        
        self.batch_size = bsz
        self.n_neg = n_neg
        
        self.nobj = np.unique(idx).shape[0]
        self.nedge = idx.shape[0]
        
        self.num_batches = int(np.ceil(self.nedge/self.batch_size))
        
        self.__iter__()
    
    def __iter__(self):
        """ """
        self.current = 0
        self.perm = np.random.permutation(self.idx.shape[0])
        return self
    
    def __next__(self):
        if self.current < self.nedge:
            fin = np.min([self.current+self.batch_size, self.nedge])
            batch_idx = self.perm[self.current:fin]
            batch = self.idx[batch_idx,:]
            self.current += batch.shape[0]
            
            N = self.sample_negatives(batch, n_max=self.n_neg)
            targ = torch.zeros(batch.shape[0]).long()
            # targ = self.weights_iter[batch_idx]
            
            item = (N, targ)
            return item
        else:
            raise StopIteration

    def sample_negatives(self, btch, n_max=10):
        """
        Return the negative examples of the batch
        
        btch is a subsample of all edges [i,j]
        For each edge in btch, will produce a list of n_max nodes that are
        not j-neighbours of i (i.e. N(i,j))
        
        Output is size (n_btch, 2+n_max)
        
        This is based on what I could decipher from Nickel & Kiela's code 
        """
        
        def getnegs(ij, n_max, max_try=200):
            """Rejection sampler, again inspired Nickel & Kiela"""
            
            negs = np.random.randint(0, self.nobj, n_max+2) # initialise randomly
            negs[0:2] = ij
            
            # pos = torch.unique(torch_where(self.idx==ij[0], torch.flip(self.idx, [1]), -1)[self.idx==ij[0]])
            
            xx = 0
            n = 2
            while (n<n_max+2) and (xx<max_try):
                k = np.random.randint(0,self.nobj)
                if (self.weights[ij[0],k] < self.weights[ij[0],ij[1]]) and (k not in negs[:n]):
                    negs[n] = k
                    n = n + 1
                xx = xx + 1
                
            # if (n>2) & (n<n_max+2):
            #     negs[n:] = np.random.choice(negs[2:n], n_max+2-n)
            
            return negs
        
        N = np.array([getnegs(ij, n_max) for ij in btch])
        
        return torch.tensor(N).long()
    
class DenseDataset(object):
    """Represent dataset as an iterator which, can loop through the data 
    in batches"""
    
    def __init__(self, weights, obj_names, bsz, n_neg=10, padding=False):
        super(DenseDataset, self).__init__()
        
        self.weights = weights
        self.names = obj_names
        
        self.batch_size = bsz
        self.n_neg = n_neg
        
        self.nobj = weights.shape[0]
        self.nedge = np.prod(weights.shape)
        if padding:
            self.padding_idx = self.nobj
        else:
            self.padding_idx = None
        
        self.num_batches = int(np.ceil((self.nedge-self.nobj)/self.batch_size))
        
        self.__iter__()
    
    def __iter__(self):
        """ """
        self.current = 0
        off_diag = np.setdiff1d(np.arange(self.nedge), (1+self.nobj)*np.arange(self.nobj))
        self.perm = np.random.permutation(off_diag) # dont include diagonal
        return self
    
    def __next__(self):
        if self.current < (self.nedge-self.nobj):
            fin = np.min([self.current+self.batch_size, self.nedge])
            batch_idx = self.perm[self.current:fin]
            batch = torch.tensor([np.floor(batch_idx/self.nobj), \
                                  np.mod(batch_idx,self.nobj)]).long().T
            self.current += batch.shape[0]
            
            N = self.sample_negatives(batch, n_max=self.n_neg)
            
            item = (N, torch.zeros(batch.shape[0]).long())
            return item
        else:
            raise StopIteration
        
    def sample_negatives(self, btch, n_max=10):
        """
        Return the negative examples of the batch
        
        btch is a subset of idx, i.e. a subsample of all edges [i,j]
        For each edge in btch, will produce a list of n_max nodes that are
        not j-neighbours of i (i.e. N(i,j))
        
        Output is size (n_btch, 2+n_max)
        
        This is based on what I could decipher from Nickel & Kiela's code 
        """
        
        def getnegs(ij, n_max, max_try=100):
            """Rejection sampler, again inspired Nickel & Kiela"""
            
            if self.padding_idx is None:
                negs = torch.randint(self.nobj, (n_max+2,)) # initialise randomly
            else:
                negs = torch.ones(n_max+2).long()*self.padding_idx
            negs[0:2] = ij
            
            xx = 0 # number of rejections
            n = 2
            while (n<n_max+2) and (xx<max_try):
                k = torch.randint(self.nobj, (1,1))
                if (self.weights[ij[0],k]<self.weights[ij[0],ij[1]]) and (k not in negs[:n]):
                    negs[n] = k
                    n = n + 1
                xx = xx + 1
            
            return negs
         
        N = torch.stack([getnegs(ij, n_max) for ij in btch])
        
        return N
    

#%%
class FeatureVectorDataset(object):
    """Represent dataset as an iterator which, can loop through the data 
    in batches"""
    
    def __init__(self, folder, bsz, bracketed, line_filter=None):
        """
        Iterator over BERT features located in folder. Assumes that there is a separate
        folder for each line, and that line indices match the lines in bracketed.

        bracketed should be a path to the `train.txt` file.

        line_filter is a function that rejects certain lines. e.g. lambda x:len(x)>=10 
        """
        super(FeatureVectorDataset, self).__init__()
        
        if line_filter is None:
            line_filter = lambda x:True
        self.filter = line_filter

        self.folder = folder
        self.bracketed = bracketed

        self.batch_size = bsz
        
        self.num_batches = int(np.ceil((self.nedge-self.nobj)/self.batch_size))
        
        self.__iter__()
    
    def __iter__(self):
        """ """
        self.current = 0
        off_diag = np.setdiff1d(np.arange(self.nedge), (1+self.nobj)*np.arange(self.nobj))
        self.perm = np.random.permutation(off_diag) # dont include diagonal
        return self
    
    def __next__(self):
        if self.current < (self.nedge-self.nobj):
            fin = np.min([self.current+self.batch_size, self.nedge])
            batch_idx = self.perm[self.current:fin]
            batch = torch.tensor([np.floor(batch_idx/self.nobj), \
                                  np.mod(batch_idx,self.nobj)]).long().T
            self.current += batch.shape[0]
            
            N = self.sample_negatives(batch, n_max=self.n_neg)
            
            item = (N, torch.zeros(batch.shape[0]).long())
            return item
        else:
            raise StopIteration
        
    def tree_distances(self, btch, n_max=10):
        """
        For the current batch of sentences, btch, return the parse distance between
        all (or at most n_max) unique pairs of words in each sentence. 
        """
        
        def getnegs(ij, n_max, max_try=100):
            """Rejection sampler, again inspired Nickel & Kiela"""
            
            if self.padding_idx is None:
                negs = torch.randint(self.nobj, (n_max+2,)) # initialise randomly
            else:
                negs = torch.ones(n_max+2).long()*self.padding_idx
            negs[0:2] = ij
            
            xx = 0 # number of rejections
            n = 2
            while (n<n_max+2) and (xx<max_try):
                k = torch.randint(self.nobj, (1,1))
                if (self.weights[ij[0],k]<self.weights[ij[0],ij[1]]) and (k not in negs[:n]):
                    negs[n] = k
                    n = n + 1
                xx = xx + 1
            
            return negs
         
        N = torch.stack([getnegs(ij, n_max) for ij in btch])
        
        return N
