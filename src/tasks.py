CODE_DIR = '/home/matteo/Documents/github/repler/src/'

import os, sys
import pickle
sys.path.append(CODE_DIR)

import torch
import torchvision
import torch.optim as optim
import numpy as np
import scipy
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations
import itertools as itt

# my code
import students
import assistants
import util
import grammars as gram

## Abstract classes
class IndependentBinary(object):
    """Abstract class encompassing all classifications of multiple binary variables"""
    def __init__(self):
        super(IndependentBinary,self).__init__()
        self.__name__ = self.__class__.__name__
        
    def __call__(self):
        raise NotImplementedError
    
    def correct(self, pred, targets):
        msk = np.isnan(targets.detach().numpy())
        n = targets.detach().numpy() == (pred.detach().numpy()>=0.5)
        return np.nanmean(np.where(msk,np.nan,n), 0, keepdims=True).astype(float)

    def information(self, test_var, normalize=False):
        """
        Computes the mutual information between the output of the task, and another variable
        """
        # given b is positive
        ab = util.decimal(self(test_var)).numpy()
        # pab = np.unique(ab, return_counts=True)[1]/np.min([self.num_cond,(2**self.num_var)])
        pab = np.unique(ab, return_counts=True)[1]/len(test_var)

        # given b is negative
        b_ = np.setdiff1d(range(self.num_cond), test_var)
        ab_ = util.decimal(self(b_)).numpy()
        # pab_ = np.unique(ab_, return_counts=True)[1]/np.min([self.num_cond,(2**self.num_var)])
        pab_ = np.unique(ab_, return_counts=True)[1]/len(test_var)

        # entropy of outputs (in case they are degenerate)
        a = util.decimal(self(np.arange(self.num_cond)))
        pa = np.unique(a, return_counts=True)[1]/self.num_cond
        Ha = -np.sum(pa*np.log2(pa))

        # I(a,b) = H(a) - H(a | b)
        MI = Ha + 0.5*(np.sum(pab*np.log2(pab))+np.sum(pab_*np.log2(pab_)))

        return MI

    def subspace_information(self):
        """
        Computes the mutual information between each output, and the rest
        """
        MI = []
        outputs = self(range(self.num_cond)).numpy()
        # print(outputs)
        for i, pos in enumerate(self.positives):
            mi = []
            for j, bos in enumerate(self.positives):

                a = outputs[:,j]

                ab = a[outputs[:,i]>0]
                pab = np.unique(ab, return_counts=True)[1]/np.sum(outputs[:,i]>0)
                hab = -np.mean(outputs[:,i]>0)*np.sum(pab*np.log2(pab))

                ab_ = a[outputs[:,i]==0]
                pab_ = np.unique(ab_, return_counts=True)[1]/np.sum(outputs[:,i]==0)
                hab_ = -np.mean(outputs[:,i]==0)*np.sum(pab_*np.log2(pab_))

                pa = np.unique(a, return_counts=True)[1]/self.num_cond  
                ha = -np.sum(pa*np.log2(pa))

                # I(a,b) = H(a) - H(a|b) = H(a) - p(b=0)H(a|b=0) - p(b=1)H(a|b=1)
                mi.append(ha - hab - hab_)
            MI.append(mi)
        return MI

# class LookupTable(object):


class IndependentCategorical(object):
    def __init__(self, labels):
        super(IndependentCategorical, self).__init__()
        self.__name__ = self.__class__.__name__
        self.labels = labels
        self.num_val = int(np.nanmax(labels)) + 1
        self.num_var = labels.shape[0]

    def correct(self, pred, targets):
        n = targets.detach().numpy() == pred.detach().numpy().argmax(-1)
        return np.nanmean(n, 0, keepdims=True)[None,:].astype(float)

    def __call__(self, idx, noise=None):
        return torch.tensor(self.labels[:,idx]).float().T

    def subspace_information(self):
        """
        Computes the mutual information between each variable, and the rest
        """
        MI = []
        pa = np.array([np.mean(self.labels==i, axis=1) for i in range(self.num_val)]).T
        ha = -np.sum(pa*np.log2(np.where(pa, pa, 1)), axis=1)
        for b in self.labels:
            pab = np.array([[np.mean(self.labels[:,b==j]==i, axis=1) for i in range(self.num_val)] \
                for j in np.unique(b)])
            
            pb = np.array([np.mean(b==i, keepdims=True) for i in np.unique(b)])

            hab = -np.sum(pb*np.sum(pab*np.log2(np.where(pab, pab, 1)), axis=1), axis=0)

            MI.append(ha - hab)
        return MI

class Classification(object):
    def __init__(self):
        super(Classification,self).__init__()
        self.__name__ = self.__class__.__name__
        
    def __call__(self):
        raise NotImplementedError
    
    def correct(self, pred, targets):
        n = (targets.detach().numpy() == pred.detach().numpy().argmax(-1)).mean(0, keepdims=True)[None,:]
        return n.astype(float)

class Regression(object):
    def __init__(self):
        super(Regression,self).__init__()
        self.__name__ = self.__class__.__name__
        
    def __call__(self):
        raise NotImplementedError
    
    def correct(self, pred, targets):
        n = (targets.detach().numpy() - pred.detach().numpy()).pow(2).mean(0, keepdims=True)
        return n.astype(float)


## Generic discrete tasks
class BinaryLabels(IndependentBinary):
    def __init__(self, labels):
        """
        labels is shape (num_var, num_item)
        """
        super(BinaryLabels, self).__init__()

        self.labels = labels

        self.num_var = labels.shape[0]
        self.num_cond = labels.shape[1]
        self.dim_output = self.num_var
        self.positives = None

        self.__name__ = 'BinaryLabels_%d_%d'%(self.num_cond, self.num_var)

    def __call__(self, idx, noise=None):
        return torch.tensor(self.labels[:,idx]).float().T

class HierarchicalLabels(BinaryLabels):
    def __init__(self, num_vars, respect=True, K=2):

        self.DGP = gram.HierarchicalData(num_vars, fan_out=K, respect_hierarchy=respect)
        super(HierarchicalLabels, self).__init__(self.DGP.labels(self.DGP.terminals))

        self.__name__ = 'HierarchicalLabels_%d_%s_%s'%(K, num_vars, respect)


class RandomDichotomies(IndependentBinary):
    def __init__(self, c=None, n=None, overlap=0, d=None, use_mse=False):
        """overlap is given as the log2 of the dot product on their +/-1 representation"""
        super(RandomDichotomies,self).__init__()
        
        if d is None:
            if c is None:
                raise ValueError('Must supply either (c,n), or d')
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
        else:
            self.positives = d
            n = len(self.positives)
            if c is None:
                c = 2*len(self.positives[0])

        self.__name__ = 'RandomDichotomies_%d-%d-%d'%(c, n, overlap)
        self.num_var = n
        self.dim_output = n
        self.num_cond = c

        if use_mse:
            self.obs_distribution = students.GausId(n)
        else:
            self.obs_distribution = students.Bernoulli(n)
        self.link = None
    
    def __call__(self, labels, noise=None):
        these = torch.tensor([np.isin(labels, p) for p in self.positives]).float()
        return these.T

# class 

class RandomDichotomiesCategorical(Classification):
    def __init__(self, c, n, overlap=0, use_mse=False):
        """overlap is given as the log2 of the dot product on their +/-1 representation"""
        super(RandomDichotomiesCategorical,self).__init__()
        self.__name__ = 'RandomDichotomiesCat_%d-%d-%d'%(c, n, overlap)
        self.num_var = 1
        self.dim_output = 2**n
        self.num_cond = c
        self.use_mse = use_mse

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

        if use_mse:
            self.obs_distribution = students.GausId(self.dim_output)
        else:
            self.obs_distribution = students.Categorical(self.dim_output)
        self.link = None
    
    def __call__(self, labels, noise=None):
        these = np.array([np.isin(labels, p) for p in self.positives]).astype(float)
        if self.use_mse:
            return assistants.Indicator(self.dim_output, self.dim_output)(util.decimal(these.T).astype(int)).float()
        else:
            return torch.tensor(util.decimal(these.T)).int()

    def correct(self, pred, targets):
        if self.use_mse:
            return (targets.detach().numpy().argmax(-1)==pred.detach().numpy().argmax(-1)).mean(0, keepdims=True)[None,:]
        else:
            return super(RandomDichotomiesCategorical,self).correct(pred, targets)

class StandardBinary(IndependentBinary):
    def __init__(self, n, q=None, use_mse=False):
        """overlap is given as the log2 of the dot product on their +/-1 representation"""
        super(StandardBinary,self).__init__()
        
        if q is None:
            q = n

        bits = np.nonzero(1-np.mod(np.arange(2**n)[:,None]//(2**np.arange(n)[None,:]),2))
        pos_conds = np.split(bits[0][np.argsort(bits[1])],n)[:q]

        self.positives = pos_conds
        self.__name__ = 'StandardBinary%d-%d'%(n, q)
        self.num_var = q
        self.dim_output = q
        self.num_cond = 2**n

        if use_mse:
            self.obs_distribution = students.GausId(n)
        else:
            self.obs_distribution = students.Bernoulli(n)
        self.link = None
    
    def __call__(self, labels):
        these = torch.tensor([np.isin(labels, p) for p in self.positives]).float()
        return these.T

class LogicalFunctions(IndependentBinary):
    '''Balanced dichotomies expressed as logical functions of balanced dichotomies'''
    def __init__(self, d, function_class, use_mse=False):
        """overlap is given as the log2 of the dot product on their +/-1 representation"""
        super(LogicalFunctions,self).__init__()
        
        self.__name__ = 'LogicalFunctions_%dbit-%d'%(len(d), function_class)
        self.num_var = 1
        self.dim_output = 1
        self.num_cond = 2**len(d)

        self.bits = d
        self.function_class = function_class

        self.positives = [np.nonzero(self(np.arange(self.num_cond).squeeze()).numpy())[0]]
        # print(self(np.arange(self.num_cond)))
        # print(self.positives)

        if use_mse:
            self.obs_distribution = students.GausId(1)
        else:
            self.obs_distribution = students.Bernoulli(1)
        self.link = None

    def __call__(self, labels):
        these = torch.tensor([np.isin(labels, p) for p in self.bits]).bool().T
        # print(these)
        if self.function_class == 0:
            return self.face(these[:,0],these[:,1],these[:,2])[:,None].float()
        elif self.function_class == 1:
            return self.corner(these[:,0],these[:,1],these[:,2])[:,None].float()
        elif self.function_class == 2:
            return self.snake(these[:,0],these[:,1],these[:,2])[:,None].float()
        elif self.function_class == 3:
            return self.net(these[:,0],these[:,1],these[:,2])[:,None].float()
        elif self.function_class == 4:
            return self.xor2(these[:,0],these[:,1],these[:,2])[:,None].float()
        elif self.function_class == 5:
            return self.xor3(these[:,0],these[:,1],these[:,2])[:,None].float()

    @staticmethod
    def face(a,b,c): # equiv = 0
        return a

    @staticmethod
    def corner(a,b,c): # equiv = 1
        return a*b + a*c + b*c

    @staticmethod
    def snake(a,b,c): # equiv = 2
        return a*c + b*(~c)

    @staticmethod
    def net(a,b,c): # equiv = 3
        return a*c + a*b + ~(a+b+c)

    @staticmethod
    def xor2(a,b,c): # equiv = 4
        return ~(a^b)

    @staticmethod
    def xor3(a,b,c): # equiv = 5
        return a^b^c

# class HierarchicalLabels(IndependentBinary):

## Tasks which produce continuous representations of the binary tasks

class LinearExpansion(object):

    def __init__(self, task, dim_pattern, noise_var=0.1, 
        centered=True, random=False, sqrt_N_norm=True):

        self.latent = task

        self.dim_output = dim_pattern
        self.noise_var = noise_var
        self.num_cond = task.num_cond
        self.num_var = task.num_var

        self.centered = centered

        if not random:
            np.random.seed(0)
        C = np.random.rand(dim_pattern, dim_pattern)
        self.expansion = la.qr(C)[0][:task.num_var,:]
        if sqrt_N_norm:
            self.expansion *= np.sqrt(self.dim_output)

        self.__name__ = f'Linear_{dim_pattern}D_{noise_var:.2f}snr_' + self.latent.__name__ 

    def __call__(self, labels, noise=None):

        if noise is None:
            noise = self.noise_var

        if not self.centered:
            L = 2*self.latent(labels) - 1
        else:
            L = self.latent(labels)
        means = L @ self.expansion
        return torch.tensor(means + np.random.randn(len(labels), self.dim_output)*noise).float()


class Embedding(object):

    def __init__(self, vals, noise=0):
        """
        Dimensions are weighted binary classes
        """

        self.num_cond = len(vals)
        self.num_var = vals.shape[1]
        self.dim_output = vals.shape[1]
        self.noise = noise

        self.means = vals

        self.__name__ = "Embedding_%d_%d"%(self.num_cond, self.dim_output)

    def __call__(self, labels):

        return torch.tensor(self.means[labels]).float()


class RandomPatterns(object):
    def __init__(self, num_cond, dim_pattern, noise_var=0.1, center=True, random=False):

        self.dim_output = dim_pattern
        self.noise_var = noise_var
        self.num_cond = num_cond

        if not random:
            np.random.seed(0)

        means = np.random.randn(self.num_cond, self.dim_output)
        self.means = means
        if center:
            self.means -= self.means.mean(0, keepdims=True)

        self.__name__ = f'RandomPatterns_{num_cond}P_{dim_pattern}D_{noise_var:.2f}snr'

    def __call__(self, labels, noise=None):

        if noise is None:
            noise = self.noise_var

        means = self.means[labels,...]
        return torch.tensor(means + np.random.randn(len(labels), self.dim_output)*noise).float()

class RandomTransitions(object):
    def __init__(self, rep_task, actions, p_action, num_var=2, num_val=2):

        self.rep_task = rep_task
        self.actions = actions
        self.p_action = p_action
        self.num_var = num_var
        self.num_val = num_val
        self.num_cond = num_val**num_var

        x = np.arange(self.num_cond)
        self.positives = [tuple(np.where(1-np.mod(x,num_val**(i+1))//(num_val**i))[0].tolist()) for i in range(num_var)]

    def __call__(self, labels, **kwargs):

        actions = np.random.choice(self.actions, len(labels), p=self.p_action)
        # actions = torch.stack([(actns&(self.num_val**i))/self.num_val**i for i in range(self.num_var)]).float().T

        successors = np.mod(labels+actions, self.num_val**self.num_var)
        # succ_conds = util.decimal(successors)
        return self.rep_task(successors, **kwargs)

class EmbeddedCube(object):
    def __init__(self, latent_task, dim_factor, noise_var=0.1, rotated=False):
        super(EmbeddedCube,self).__init__()

        self.latent_task = latent_task
        self.rotated = rotated
        self.dim_output = latent_task.num_var*dim_factor
        self.noise_var = noise_var

        self.__name__ = 'EmbeddedCube_%d-%d-%.1f'%(latent_task.num_var, self.dim_output, noise_var)

        self.positives = self.latent_task.positives
        self.num_var = latent_task.num_var
        self.num_cond = latent_task.num_cond

        means = np.random.randn(latent_task.num_var, 2, dim_factor)
        means = means/la.norm(means, axis=-1, keepdims=True)
        self.means = means

        if self.rotated:
            C = np.random.rand(self.dim_output, self.dim_output)
            self.rot_mat = la.qr(C)[0][:self.dim_output,:]

    def __call__(self, labels, noise=None):

        if noise is None:
            noise = self.noise_var

        var_bit = self.latent_task(labels).numpy()
        var_idx = np.arange(self.num_var,dtype=int)[None,:]*np.ones((len(labels),1),dtype=int)

        # mns = (self.means[:,None,:]*var_bit[:,:,None]) - (self.means[:,None,:]*(1-var_bit[:,:,None]))
        clus_mns = self.means[var_idx,var_bit.astype(int),:].reshape((len(labels),-1))

        # clus_mns = np.reshape(mns.transpose((2,0,1)), (self.dim_output,-1)).T
        if self.rotated:
            clus_mns = clus_mns@self.rot_mat
        clus_mns -= clus_mns.mean(0,keepdims=True)

        return torch.tensor(clus_mns + np.random.randn(len(labels), self.dim_output)*noise).float()

class NudgedCube(object):
    def __init__(self, latent_task, nudge_task, dim_factor, noise_var=0.1, nudge_mag=1, rotated=False):
        super(NudgedCube,self).__init__()

        self.latent_task = latent_task
        self.nudge_task = nudge_task
        self.rotated = rotated
        self.dim_output = latent_task.num_var*dim_factor
        self.noise_var = noise_var
        self.nudge_mag = nudge_mag

        self.__name__ = 'NudgedCube%d-%d-%.2f'%(latent_task.num_var, self.dim_output, noise_var)

        self.positives = self.latent_task.positives
        self.num_var = latent_task.num_var
        self.num_cond = latent_task.num_cond

        means = np.random.randn(latent_task.num_var, 2, dim_factor)
        means = means/la.norm(means, axis=-1, keepdims=True)
        means[:,1,:] = - means[:,0,:]
        self.means = means

        nudge_dir = np.random.randn(1, dim_factor*latent_task.num_var)
        nudge_dir /= la.norm(nudge_dir)
        nudge_dir[:,:dim_factor] -= (nudge_dir[:,:dim_factor]@means[0,0,:])*means[0,0,:]
        nudge_dir[:,dim_factor:] -= (nudge_dir[:,dim_factor:]@means[1,0,:])*means[1,0,:]
        self.nudge_dir = nudge_dir

        if self.rotated:
            C = np.random.rand(self.dim_output, self.dim_output)
            self.rot_mat = la.qr(C)[0][:self.dim_output,:]

    def __call__(self, labels, noise=None):

        if noise is None:
            noise = self.noise_var

        var_bit = self.latent_task(labels).numpy()
        var_idx = np.arange(self.num_var,dtype=int)[None,:]*np.ones((len(labels),1),dtype=int)
        nudge_bit = self.nudge_task(labels).numpy()

        # mns = (self.means[:,None,:]*var_bit[:,:,None]) - (self.means[:,None,:]*(1-var_bit[:,:,None]))
        clus_mns = self.means[var_idx,var_bit.astype(int),:].reshape((len(labels),-1)) 
        clus_mns += 2*(nudge_bit - 0.5)*self.nudge_mag*self.nudge_dir

        # clus_mns = np.reshape(mns.transpose((2,0,1)), (self.dim_output,-1)).T
        if self.rotated:
            clus_mns = clus_mns@self.rot_mat
        # clus_mns -= clus_mns.mean(0,keepdims=True)

        return torch.tensor(clus_mns + np.random.randn(len(labels), self.dim_output)*noise).float()

class NudgedXOR(object): # version specific to the xor
    def __init__(self, dim_factor, num_var=2, noise_var=0.1, nudge_mag=1, 
        rotated=False, random=False, sqrt_N_norm=True):
        super(NudgedXOR,self).__init__()

        self.rotated = rotated
        self.dim_output = (num_var+1)*dim_factor
        self.noise_var = noise_var
        self.nudge_mag = nudge_mag

        if len(str(nudge_mag))-2 > 1:
            self.__name__ = 'NudgedXOR%d-%d-%.1f-%.2f'%(num_var, self.dim_output, noise_var, nudge_mag)
        else:
            self.__name__ = 'NudgedXOR%d-%d-%.1f-%.1f'%(num_var, self.dim_output, noise_var, nudge_mag)
        if random:
            self.__name__ += '_rand'
        if rotated:
            self.__name__ += '_rot'

        self.bits = StandardBinary(num_var)
        self.positives = self.bits.positives
        self.num_var = num_var
        self.num_cond = 2**num_var

        if not random:
            np.random.seed(0)
        # self.means = la.qr(np.random.randn(self.dim_output, self.num_cond+1), mode='economic')[0]
        means = np.abs(np.random.randn(self.dim_output, self.num_var+1))
        means *= np.repeat(np.eye(num_var+1), dim_factor, axis=0)
        means /= la.norm(means, axis=0, keepdims=True)
        if sqrt_N_norm:
            means *= np.sqrt(self.dim_output/3)
        self.means = means

        if self.rotated:
            C = np.random.rand(self.dim_output, self.dim_output)
            self.rot_mat = la.qr(C)[0]

    def __call__(self, labels, noise=None):

        if noise is None:
            noise = self.noise_var

        var_bit = self.bits(labels).numpy()
        nudge_bit = 2*(var_bit.sum(1, keepdims=True)%2)-1  # the nd xor

        clus_mns = np.concatenate([2*var_bit-1, nudge_bit*self.nudge_mag], axis=1)@self.means.T

        # # mns = (self.means[:,None,:]*var_bit[:,:,None]) - (self.means[:,None,:]*(1-var_bit[:,:,None]))
        # clus_mns = self.means[var_idx,var_bit.astype(int),:].reshape((len(labels),-1)) 
        # clus_mns += 2*(nudge_bit[:,None] - 0.5)*self.nudge_mag*self.means[2,...].flatten()
        # clus_mns 

        # clus_mns = np.reshape(mns.transpose((2,0,1)), (self.dim_output,-1)).T
        if self.rotated:
            clus_mns = clus_mns@self.rot_mat
        # clus_mns -= clus_mns.mean(0,keepdims=True)

        return torch.tensor(clus_mns + np.random.randn(len(labels), self.dim_output)*noise).float()

    def define_basis(self, means, newf=None):

        self.means = means

class TwistedCube(object):
    def __init__(self, latent_task, dim_emb, f, noise_var=0.1):
        """f in [0,1]"""
        super(TwistedCube,self).__init__()

        C = np.random.rand(dim_emb, dim_emb)
        # self.means = torch.tensor(la.qr(C)[0]).float()

        self.latent_task = latent_task
        
        self.dim_output = dim_emb
        self.noise_var = noise_var

        self.positives = self.latent_task.positives
        self.num_var = latent_task.num_var
        self.num_cond = latent_task.num_cond

        self.param = f

        self.define_basis(torch.tensor(la.qr(C)[0]).float())

        # self.rotator = self.rotation_mat(f)
        # self.offset = np.min([f, 2-f])*(1-np.sqrt(1-2*(0.5**2)))

        self.__name__ = 'TwistedCube_%d-%d-%.1f'%(latent_task.num_var, self.dim_output, self.param)

    def rotation_mat(self, f):
        """returns a matrix which rotates by angle theta in the x2-x3 plane"""

        theta = f*np.pi/2
        rot = torch.eye(self.dim_output).float()
        rot[1:3,1:3] = torch.tensor([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]).float()

        return (self.basis@rot)@(self.basis.T)

    def define_basis(self, means, newf=None):

        theta = self.param*np.pi/2
        rot = torch.eye(self.dim_output).float()
        rot[1:3,1:3] = torch.tensor([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]).float()

        self.basis = means
        self.means = means[:,:2]
        self.rotator = (means@rot)@(means.T)
        self.offset = np.min([self.param, 2-self.param])*(1-np.sqrt(1-2*(0.5**2)))

    def __call__(self, labels, noise=None, newf=None):
        ''' labels of shape (..., n_variable) '''
        if newf is not None:
            self.rotator = self.rotation_mat(newf)
            self.offset = np.min([newf, 2-newf])*(1-np.sqrt(1-2*(0.5**2)))
        if noise is None:
            noise = self.noise_var
        latent = self.latent_task(labels)
        
        output = latent@(self.means.T)
        output -= output.mean(0)
        output[latent[:,0]==0,:] = output[latent[:,0]==0,:]@self.rotator
        output[latent[:,0]==0,:] += self.offset*self.means[:,0]
        output += np.random.randn(len(latent), self.dim_output)*noise
    
        return output.float()


## MNIST tasks (old shit)
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