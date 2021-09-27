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
from itertools import permutations, combinations
import itertools as itt

import students
import assistants

#%%
def pca(X,**kwargs):
    '''assume X is (n_feat, n_sample)'''
    U,S,_ = la.svd((X-X.mean(1, keepdims=True)), full_matrices=False, **kwargs)
    return U, S**2

def pca_reduce(X, thrs, **kwargs):

    U,S = pca(X, **kwargs)
    these_comps = np.cumsum(S)/np.sum(S) <=  thrs
    return X.T@U[:,these_comps]

#%%
# def 

#%%
def discrete_metric(x,y):
    return np.abs(x - y).sum(0)

#%% Factorization
def decompose_covariance(Z, X, compute_signal=True, only_var=False):
    """
    Compute the signal covariance and noise covariance matrices of Z w.r.t. X

    Z is size N_feat x N_sample
    X is length N_sample 
    """

    # values of X for conditioning
    unq, unq_idx = np.unique(X, return_inverse=True)
    Z_given_x = np.array([Z[:,X==x].mean(1) for x in unq])[unq_idx,:].T    

    if only_var:
        noise_cov = np.var(Z - Z_given_x, axis=1) 
    else:
        # equivalent to mean([cov(Z[:,X==x]) for x in unq])
        noise_cov = np.cov(Z - Z_given_x)
    if compute_signal:
        if only_var:
            sig_cov = np.var(Z, axis=1) - noise_cov + np.ones(len(Z))*1e-3
        else:
            sig_cov = np.cov(Z) - noise_cov + np.eye(len(Z))*1e-5
            # sig_cov = np.cov(Z_given_x)
        return noise_cov, sig_cov
    else:
        return noise_cov

def projected_variance(Z, X, Y=None, cutoff=None, only_var=False):
    """
    Project Z onto noise covariance of X, and compare

    optionally project onto the signal covariance of Y
    """

    if Y is None:
        noise_cov, sig_cov = decompose_covariance(Z,X,only_var=only_var)
        # noise_var = np.trace(noise_cov)
    else: # compare the noise covariances of X and Y
        _, sig_cov = decompose_covariance(Z,X, only_var=only_var)
        _, noise_cov = decompose_covariance(Z,Y, only_var=only_var)
        # sig_cov, _ = decompose_covariance(Z,X)
        # noise_cov, _ = decompose_covariance(Z,Y)

    if cutoff is not None:
        vals, eigs = la.eigh(noise_cov)
        these_vals = (np.cumsum(np.flip(np.sort(vals)))/np.sum(vals))<cutoff
        # return vals
        # print(eigs.shape)
        noise_cov = eigs[:,these_vals]@eigs[:,these_vals].T

    if only_var:
        return 1-np.sum(noise_cov*sig_cov)/(la.norm(noise_cov)*la.norm(sig_cov))
    else:
        return 1- np.trace(noise_cov.T@sig_cov)/(la.norm(noise_cov,'fro')*la.norm(sig_cov,'fro'))
    # noise = noise_cov@Z/(np.mean(np.diag(noise_cov)))
    # noise = noise_cov@Z/(np.sqrt(np.trace(noise_cov)))
    # noise = noise_cov@Z
    # _, proj_sig_cov = decompose_covariance(noise, X, only_var=True)
    # print(np.sum(proj_sig_cov))

    # return 1 - np.sum(proj_sig_cov)/(sig_var)
    # return 1 - np.sum(proj_sig_cov)/(sig_var*np.trace(noise_cov))

def diagonality(Z, X, cutoff=0.9):
    """ 
    Ratio of top n signal-variance neurons, to the top n signal variance components
    """

    _, sig_cov = decompose_covariance(Z,X)

    vals = la.eigvalsh(sig_cov)
    # print(np.flip(np.sort(vals))/np.sum(vals))
    # print((np.cumsum(np.flip(np.sort(vals)))/np.sum(vals)))
    # print(np.sum(vals))
    n = np.argmax((np.cumsum(np.flip(np.sort(vals)))/np.sum(vals))<cutoff) + 1
    # print(n)

    comp_sum = np.sum(((np.flip(np.sort(vals))))[:n])
    # print(comp_sum)
    neur_sum = np.sum(((np.flip(np.sort(np.diag(sig_cov)))))[:n])


    return neur_sum/comp_sum

#%% Distance correlation functions
def dependence_statistics(x=None, dist_x=None):
    """
    Assume x is (n_feat, ..., n_sample), or dist_x is (...,n_sample, n_sample)
    """
    
    if x is not None:
        n = x.shape[-1]
        x_kl = la.norm(x[...,None] - x[...,None,:],2,axis=0)
    elif dist_x is not None:
        n = dist_x.shape[-1]
        x_kl = dist_x
    else:
        raise ValueError('Gotta supply something!!!')
    x_k = x_kl.sum(-2, keepdims=True)/(n-2)
    x_l = x_kl.sum(-1, keepdims=True)/(n-2)
    x_ = x_kl.sum((-2,-1), keepdims=True)/((n-2)*(n-1))
    
    D = x_kl - x_k - x_l + x_
    D *= 1-np.eye(n)

    return D

def distance_covariance(x=None, y=None, dist_x=None, dist_y=None):    
    A = dependence_statistics(x, dist_x)
    B = dependence_statistics(y, dist_y)
    if x is None:
        n = dist_x.shape[-1]
    else:
        n = x.shape[-1]
    dCov = np.sum(A*B, axis=(-2,-1))/(n*(n-3))
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

def distance_correlation(x=None, y=None, dist_x=None, dist_y=None):
    """
    Assume x is (n_feat, ..., n_sample), or dist_x is (...,n_sample, n_sample)
    """
    V_xy = distance_covariance(x, y, dist_x, dist_y)
    V_x = distance_covariance(x, x, dist_x, dist_x)
    V_y = distance_covariance(y, y, dist_y, dist_y)
    # print([V_x, V_y, V_xy])
    sing = V_x*V_y > 0
    return sing*V_xy/(np.sqrt(V_x*V_y)+1e-5)
    # if 0 in [V_x, V_y]:
        # return 0
    # else:
        # return np.sqrt(V_xy/np.sqrt(V_x*V_y))
        # return V_xy/np.sqrt(V_x*V_y)

def partial_distance_correlation(x=None, y=None, z=None, dist_x=None, dist_y=None, dist_z=None):
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

#%% Mobius strip (maybe this belongs as a class?)
def noose_curve(x, l=2):
    """the curve h(x) from Sabitov (2007) section 8 ... """
    # For specific choices of g1 and g2

    # if np.abs(x)>(l/2):
    #     return np.array([l/2 - np.abs(x), 0])
    # else:
    g1 = l/4 - (x**2)/l
    # g1 = 2*x - 2*np.log(1+np.exp(2*x)) - (2 - 2*np.log(1+np.exp(l)))
    g2 = 2*l*x*np.exp(-2*(l**2)*(x**2))

    extrm = np.abs(x)>(l/2)

    h1 = np.where(np.abs(x)>(l/2), (l/2)-np.abs(x), g1)
    h2 = np.where(np.abs(x)>(l/2), 0, g2)

    return np.stack([h1,h2])

        # return np.array([g1, g2])

def open_curve(t, l=2):

    # odd, bounded by +/- (l/2)*sin(pi/6)
    f1 = (l/3)*np.sin(np.pi/6)*np.tanh(t)

    # even, monotonic in |t|
    f2 = 2*np.log(1+np.exp(t)) - t
    
    return np.stack([f1,f2])

def flat_mobius_strip(s, t, l=2, A=0.5):
    """The insanely convoluted computation of a flat mobius strip"""

    A = (4+A)*l*np.sqrt(3)/12
    # print(A)

    costh = np.cos(np.pi/6)
    sinth = np.sin(np.pi/6)

    r32 = np.sqrt(3)/2 # this gets used a lot

    f = open_curve(t, l)
    
    # this is X1(s, -t), so I change f1 and f2 appropriately
    X1_12 = noose_curve(s*costh - f[0,:]*sinth) # f1 is an odd function
    X1_3 = -s*sinth - f[0,:]*costh
    X1_4 = f[1,:] # f2 is an even function
    X1 = np.concatenate([X1_12, X1_3[None,:], X1_4[None,:]], axis=0)

    # X2(2A+s, t)
    h_x2 = noose_curve(r32*(2*A+s) + f[0,:]/2)
    X2_1 = -0.5*h_x2[0,:] + r32*(2*A+s)/2 - r32*f[0,:]/2 + r32*l/2 - np.sqrt(3)*A
    X2_2 = h_x2[1,:]
    X2_3 = r32*h_x2[0,:] + (2*A+s)/4 - r32*f[0,:]/2 - r32*l/2 + A
    X2_4 = f[1,:]
    X2 = np.stack([X2_1, X2_2, X2_3, X2_4])

    # X3(-2*A+3, t)
    h_x3 = noose_curve(r32*(-2*A+3) + f[0,:]/2)
    h1_x3 = noose_curve(r32 + f[0,:]/2)[0,:]
    X3_1 = -0.5*h1_x3 - r32*(-2*A+3)/2 + r32*f[0,:]/2 + r32*l/2 - np.sqrt(3)*A
    X3_2 = h_x3[1,:]
    X3_3 = -r32*h_x3[0,:] + (-2*A+3)/4 - r32*f[0,:]/2 + r32*l/2 - A
    X3_4 = f[1,:]
    X3 = np.stack([X3_1, X3_2, X3_3, X3_4])

    use_x2 = (s < -A).astype(int)
    use_x3 = (s > A).astype(int)
    use_x1 = ((use_x2==0)*(use_x3==0)).astype (int)

    # print(np.sum(use_x2))
    # print(np.sum(use_x3))
    # print(use_x1)

    return use_x1*X1 + use_x2*X2 + use_x3*X3

#%% A better moebius strip
def little_h(rho):
    return np.sqrt(4+rho**2)/16

def big_h(rho):
    return 1/(2*np.sqrt(4-rho**2) + 1e-3) + np.sqrt(4-rho**2)/8

def isom_t(rho):
    return (7/8)*rho + (1/8)*np.log((2-rho)/(2+rho+1e-3))

def flat_moebius(rho, u):

    x1 = rho*np.cos(u/2 + little_h(rho))
    x2 = rho*np.sin(u/2 + little_h(rho))
    x3 = np.sqrt(4-rho**2)*np.cos(u + big_h(rho))/2
    x4 = np.sqrt(4-rho**2)*np.sin(u + big_h(rho))/2

    return np.stack([x1,x2,x3,x4])

#%% Yet another moebius strip
def big_f(rho, R=2):
    return np.log(R**2 - rho**2) + (R**2)/(R**2 - rho**2 )

def blanusa_moebius(rho, u, R=2):

    F = big_f(rho, R)
    x1 = rho*np.cos(u/2 + F/2 - 2/(R**2 - rho**2 ))
    x2 = rho*np.sin(u/2 + F/2 - 2/(R**2 - rho**2 ))
    x3 = np.sqrt(4-rho**2)*np.cos(u + F)/2
    x4 = np.sqrt(4-rho**2)*np.sin(u + F)/2

    return np.stack([x1,x2,x3,x4])

#%% miscellaneous functions
def decimal(binary):
    """ 
    convert binary vector to dedimal number (i.e. enumerate) 
    assumes second axis is the bits
    """
    d = (binary*(2**np.arange(binary.shape[1]))[None,:,None]).sum(1)
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
    """Assume features are in first axis"""

    x_ = x/(la.norm(x,axis=0,keepdims=True)+1e-5)
    y_ = y/(la.norm(y,axis=0,keepdims=True)+1e-5)
    return np.einsum('k...i,k...j->...ij', x_, y_)

def dot_product(x,y):
    """Assume features are in first axis"""
    return np.einsum('k...i,k...j->...ij', x, y)

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
        ''' labels of shape (..., n_variable) '''
        if newf is not None:
            self.rotator = self.rotation_mat(newf)
            self.offset = newf*(1-np.sqrt(1-2*(0.5**2)))
                
        output = labels@(self.basis[:,:2].T)
        output -= output.mean(0)
        output[labels[:,0]==0,:] = output[labels[:,0]==0,:]@self.rotator
        output[labels[:,0]==0,:] += self.offset*self.basis[:,0]
    
        return output