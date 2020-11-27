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
        n = (targets.detach().numpy() == (pred.detach().numpy()>=0.5)).mean(0, keepdims=True)
        return n.astype(float)

    def information(self, test_var, normalize=False):
        """
        Computes the mutual information between the output of the task, and another variable
        """
        # given b is positive
        ab = decimal(self(test_var)).numpy()
        # pab = np.unique(ab, return_counts=True)[1]/np.min([self.num_cond,(2**self.num_var)])
        pab = np.unique(ab, return_counts=True)[1]/len(test_var)

        # given b is negative
        b_ = np.setdiff1d(range(self.num_cond), test_var)
        ab_ = decimal(self(b_)).numpy()
        # pab_ = np.unique(ab_, return_counts=True)[1]/np.min([self.num_cond,(2**self.num_var)])
        pab_ = np.unique(ab_, return_counts=True)[1]/len(test_var)

        # entropy of outputs (in case they are degenerate)
        a = decimal(self(np.arange(self.num_cond)))
        pa = np.unique(a, return_counts=True)[1]/self.num_cond
        Ha = -np.sum(pa*np.log2(pa))

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
            # rest = [self.positives[p] for p in range(self.num_var) if p!=i]
            # ab = decimal(np.array([np.isin(pos, p) for p in rest]).T)
            not_i = ~np.isin(range(self.num_var), i)
            if not_i.sum()==1:
                a = decimal(outputs[:,not_i][:,None])
            else:
                a = decimal(outputs[:,not_i])
            # print(a)

            ab = a[outputs[:,i]>0]
            # pab = np.array([np.mean(ab==c) for c in range(2**(self.num_var-1))])
            # pab = np.unique(ab, return_counts=True)[1]/(2**(self.num_var-1))
            pab = np.unique(ab, return_counts=True)[1]/np.sum(outputs[:,i])

            ab_ = a[outputs[:,i]==0]
            # print(ab_)
            # pab_ = np.array([np.mean(ab_==c) for c in range(2**(self.num_var-1))])
            # pab_ = np.unique(ab_, return_counts=True)[1]/(2**(self.num_var-1))
            pab_ = np.unique(ab_, return_counts=True)[1]/np.sum(outputs[:,i])

            pa = np.unique(a, return_counts=True)[1]/self.num_cond  
            ha = -np.sum(pa*np.log2(pa))

            # MI.append(ha + 0.5*(pab*np.ma.log2(pab)+(1-pab)*np.ma.log2(1-pab)).sum())
            MI.append(ha + 0.5*(np.sum(pab*np.log2(pab))+np.sum(pab_*np.log2(pab_))))
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
    
    def __call__(self, labels):
        these = torch.tensor([np.isin(labels, p) for p in self.positives]).float()
        return these.T

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
    
    def __call__(self, labels):
        these = np.array([np.isin(labels, p) for p in self.positives]).astype(float)
        if self.use_mse:
            return assistants.Indicator(self.dim_output, self.dim_output)(decimal(these.T).astype(int)).float()
        else:
            return torch.tensor(decimal(these.T)).int()

    def correct(self, pred, targets):
        if self.use_mse:
            return (targets.detach().numpy().argmax(-1)==pred.detach().numpy().argmax(-1)).mean(0, keepdims=True)[None,:]
        else:
            return super(RandomDichotomiesCategorical,self).correct(pred, targets)
#%%
def discrete_metric(x,y):
    return np.abs(x - y).sum(0)

def dependence_statistics(x, dist_x=None):
    """Assume x is (n_feat, ..., n_sample)"""
    
    n = x.shape[-1]
    if dist_x is None:
        x_kl = la.norm(x[...,None] - x[...,None,:],2,axis=0)
    else:
        x_kl = dist_x(x[...,None], x[...,None,:])
    x_k = x_kl.sum(-2, keepdims=True)/(n-2)
    x_l = x_kl.sum(-1, keepdims=True)/(n-2)
    x_ = x_kl.sum((-2,-1), keepdims=True)/((n-2)*(n-1))
    
    D = x_kl - x_k - x_l + x_
    D *= 1-np.eye(n)

    return D

def distance_covariance(x, y, dist_x=None, dist_y=None):    
    A = dependence_statistics(x, dist_x)
    B = dependence_statistics(y, dist_y)
    n = x.shape[-1]
    dCov = np.sum(A*B)/(n*(n-3))
    # return np.max([0, dCov]) # this should be non-negative anyway ...
    return dCov

# def distance_covariance(X, joint=True):
#     """
#     concatenation of variables into (n_var, n_sample)
#     save time by setting joint=False, if you only want diagonal
#     """
#     D = dependence_statistics(X)
#     if joint:
#         return (D[None,...]*D[:,None,...]).mean((-2,-1)) 
#     else:
#         return (D*D).mean((-2,-1))

def distance_correlation(x, y, dist_x=None, dist_y=None):
    """Assume x is (n_feat, ..., n_sample)"""
    V_xy = distance_covariance(x, y, dist_x, dist_y)
    V_x = distance_covariance(x, x, dist_x, dist_x)
    V_y = distance_covariance(y, y, dist_y, dist_y)
    # print([V_x, V_y, V_xy])
    if 0 in [V_x, V_y]:
        return 0
    else:
        # return np.sqrt(V_xy/np.sqrt(V_x*V_y))
        return V_xy/np.sqrt(V_x*V_y)

def partial_distance_correlation(x, y, z, dist_x=None, dist_y=None, dist_z=None):
    R_xy = distance_correlation(x, y, dist_x=dist_x, dist_y=dist_y)
    R_xz = distance_correlation(x, z, dist_x=dist_x, dist_y=dist_z)
    R_yz = distance_correlation(y, z, dist_x=dist_y, dist_y=dist_z)

    if ((R_xz**2) > 1-1e-4) or ((R_yz**2) > 1-1e-4):
        return 0
    else:
        return (R_xy - R_xz*R_yz)/np.sqrt((1-R_xz**2)*(1-R_yz**2))

# def distance_correlation(X):
#     V = distance_covariance(X)
#     V_x = np.diag(V)
#     normlzr = V_x[None,:]*V_x[:,None]
#     R = np.zeros(V.shape)
#     R[normlzr>0] = np.sqrt(V[normlzr>0]/np.sqrt(normlzr[normlzr>0]))
#     return R

def rbf_kernel(X, sigma=1, p=2):
    """X is (n_sample, n_dim)"""
    # pairwise_dists = squareform(pdist(X, 'minkowski', p))
    K = np.exp((np.abs(X[...,None] - X[...,None,:])**p)/(2*sigma**p))

# def rbf_kernel(x1, x2, variance = 1):
#     return np.exp(-1 * ((x1-x2) ** 2) / (2*variance))

# def gram_matrix(xs, var=1):
    # return [[rbf_kernel(x1,x2,var) for x2 in xs] for x1 in xs]

def gauss_process(xs, var=1):
    mean = [0 for x in xs]
    gram = rbf_kernel(xs, sigma=var)

    ys = np.array([np.random.multivariate_normal(mean, gram) for _ in xs])
    return ys

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

def group_mean(X, mask, axis=-1, **mean_args):
    """Take the mean of X along axis, but exluding particular elements"""
    exclude = np.ones(mask.shape)
    exclude[~mask] = np.nan
    return np.nanmean(X*exclude, axis=axis, **mean_args)

def group_std(X, mask, axis=-1, **mean_args):
    """Take the mean of X along axis, but exluding particular elements"""
    exclude = np.ones(mask.shape)
    exclude[~mask] = np.nan
    return np.nanstd(X*exclude, axis=axis, **mean_args)

def cosine_sim(x,y):
    """Assume features are in last axis"""

    x_ = x/(la.norm(x,axis=-1,keepdims=True)+1e-5)
    y_ = y/(la.norm(y,axis=-1,keepdims=True)+1e-5)
    return np.einsum('ik...,jk...->ij...', x_, y_)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    From karlo on stack exchange

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def diverging_clim(ax):
    for im in ax.get_images():
        cmax = np.max(np.abs(im.get_clim()))
        im.set_clim(-cmax,cmax)

class ContinuousEmbedding(object):
    def __init__(self, dim, f):
        """f in [0,1]"""
        C = np.random.rand(dim, dim)
        self.basis = torch.tensor(la.qr(C)[0]).float()
        
        self.emb_dim = dim
        self.rotator = self.rotation_mat(f)
        self.offset = f*(1-np.sqrt(1-2*(0.5**2)))

    def rotation_mat(self, f):
        """returns a matrix which rotates by angle theta in the x2-x3 plane"""

        theta = f*np.pi/2
        rot = torch.eye(self.emb_dim).float()
        rot[1:3,1:3] = torch.tensor([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]).float()

        return (self.basis@rot)@(self.basis.T)
        
    def __call__(self, labels, newf=None):
        if newf is not None:
            self.rotator = self.rotation_mat(newf)
            self.offset = newf*(1-np.sqrt(1-2*(0.5**2)))
                
        output = labels@(self.basis[:,:2].T)
        output -= output.mean(0)
        output[labels[:,0]==0,:] = output[labels[:,0]==0,:]@self.rotator
        output[labels[:,0]==0,:] += self.offset*self.basis[:,0]
    
        return output