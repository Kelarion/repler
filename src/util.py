
import os, sys
import pickle

import numpy as np
import numpy.linalg as nla
import scipy
import scipy.linalg as la
import scipy.special as spc
import scipy.sparse as sprs
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment 
from scipy.optimize import linprog as lp
from itertools import permutations, combinations
import itertools as itt
import mpmath as mpm


##########################
#### All by Matteo #######
##########################


############################
######### Indexing #########
############################

class LexOrder:
    """
    Lexicographic order of symmetric pairs, (i,j), j < i
    
    j = __|_0_1_2_3_
    i = 0 | - - - -
        1 | 0 - - -
        2 | 1 2 - -
        3 | 3 4 5 -
        
    """
    def __init__(self):
        return 
    
    def __call__(self,i,j):
        """
        (i,j) -> n
        n = i*(i-1)/2 + j
        """
        n = np.where(i>j, i*(i-1)/2 + j, j*(j-1)/2 + i)
        return np.where(i==j, -1, n).astype(int)
    
    def inv(self, n):
        """
        n -> (i,j)
        i = floor( (1 + sqrt(1 + 8n))/2 )
        j = n - (i*(i-1)/2)
        """
        i = np.floor((1 + np.sqrt(1+8*n))/2).astype(int)
        j = (n - i*(i-1)//2).astype(int)
        return i, j


class LexOrderK:
    """
    Lexicographic order of k-combinations. For a list K of non-negative integers, 
    the index of a list R, containing r <= max(K) non-negative integers, is:

    n(R) = sum_{i=0}^{r-1} binom(R[i], r-i) + sum_{k in K - r} binom(max(R), k)

    If only one K is supplied, this is the standard order on K-combinations, if 
    all K's from 1 to N are supplied, then this is the decimal representation of
    N-bit binary numbers (only for numbers up to 2^N).

    For k=2, this order over pairs (i,j) looks like:
    
    j = __|_0_1_2_3_
    i = 0 | - - - -
        1 | 0 - - -
        2 | 1 2 - -
        3 | 3 4 5 -
        
    """
    def __init__(self, *Ks):
        self.K = Ks
        return 
    
    def __call__(self, *items):
        """
        Send a list of items, (i,j,k,...) to their index in the lex order. 

        Each input is an array of the same length. 

        An even number of repeats of an item cancels out, e.g. (i,j,j,k) -> (i,k)
        """

        # we'll contort ourselves a bit to keep it vectorized

        sorted_items = np.flipud(np.sort(np.stack(items), axis=0))
        reps = run_lengths(sorted_items)

        live = np.mod(reps,2)

        r = live.sum(0, keepdims=True)

        # put -1 at items we don't want to consider
        # otherwise count down from r to 1
        this_k = (r - np.cumsum(live, 0) + 2)*live - 1 

        # awful way of getting K\r
        if len(sorted_items.shape) > 1:
            all_K = np.expand_dims(self.K, *range(1, len(sorted_items.shape)))
        else:
            all_K = np.array(self.K)
        above_r = all_K > r
        below_r = all_K < r
        maxval = (sorted_items*live).max(0, keepdims=True)
        top = (maxval+1)*below_r + maxval*above_r
        bot = (all_K+1)*(below_r + above_r) - 1 # same trick as above

        n = binom(sorted_items, this_k).sum(0) + binom(top, bot).sum(0)

        return np.squeeze(np.where(np.isin(r, self.K), n, -1).astype(int))
    
    # def inv(self, n):
    #     """
    #     Invert the above function -- given an index, return the list of items
    #     """



    #     return items


def run_lengths(A, mask_repeats=True):
    """ 
    A is shape (N, ...)

    A "run" is a series of repeated values. For example, in the sequence

    [0, 2, 2, 1]

    there are 3 runs, of the elements 0, 2, and 1, with lengths 1, 2, and 1
    respectively. The output of this function would be 

    [1, 2, 0, 1]

    indicating the length of each run that an element starts. Now imagine this 
    being done to each column of an array. The sum of along the first axis will
    all be the same (equal to its dimension).

    This is like the array analogue of np.unique(..., return_counts=True). 
    In fact you can get the unique elements by doing something like:
    
    R = run_lengths(A)>0
    idx = np.where(R>0)
    
    vals = A[*idx]
    counts = R[*idx]

    """

    n = len(A)
    if len(A.shape) > 1:
        ids = np.expand_dims(np.arange(n), *range(1,len(A.shape)))
    else:
        ids = np.arange(n)

    changes = A[1:] != A[:-1]

    is_stop = np.append(changes, np.ones((1,*A.shape[1:])), axis=0)
    stop_loc = np.where(is_stop, ids, np.inf)
    stop = np.flip(np.minimum.accumulate(np.flip( stop_loc ), axis=0))

    is_start = np.append(np.ones((1,*A.shape[1:])), changes, axis=0)
    start_loc = np.where(is_start, ids, -np.inf)
    start = np.maximum.accumulate(start_loc , axis=0)

    if mask_repeats:
        counts = (stop - start + 1)*is_start
    else:
        counts = (stop - start + 1)

    return counts


#########################
######### PCA ###########
#########################

def pca(X,**kwargs):
    '''assume X is (..., n_feat, n_sample)'''
    U,S,_ = nla.svd((X-X.mean(-1, keepdims=True)), full_matrices=False, **kwargs)
    return U, S**2

def pca_reduce(X, thrs=None, num_comp=None, **kwargs):
    """
    takes in (n_feat, n_sample) returns (n_sample, n_comp)
    """

    U,S = pca(X, **kwargs)
    if thrs is not None:
        these_comps = np.cumsum(S)/np.sum(S) <=  thrs
    elif num_comp is not None:
        these_comps = np.arange(len(S)) < num_comp
    return X.T@U[:,these_comps]

def participation_ratio(X):
    """ (..., n_feat, n_sample) """
    _, evs = pca(X)
    return (np.sum(evs, -1)**2)/np.sum(evs**2, -1)

def sample_normalize(data, sigma, eps=1e-10):
    """ 
    data is (num_feat, num_sample)
    sigma is (num_feat, num_feat)

    Generates random data with a specified covariance eigenvalue spectrum, spec

    This ensures that the *sample* covariance has a certain spectrum, not the population
    """

    data_cntr = data - data.mean(1, keepdims=True)

    L = la.cholesky(np.cov(data) + np.eye(data.shape[0])*eps)
    return la.cholesky(sigma+ np.eye(data.shape[0])*eps).T@la.inv(L).T@data_cntr

def expo2part(p, N):
    """ 
    Predict the participation ratio of an N-neuron code with exponentially 
    decaying spectrum 
    """

    numer = float((mpm.zeta(p) - mpm.zeta(p,N+1))**2)
    denom = float((mpm.zeta(2*p) - mpm.zeta(2*p,N+1)))
    return numer/denom

def part2expo(pr,N):
    """ Brute force inversion of expo2part """

    exps = np.linspace(0,5,100)
    parts = np.array([expo2part(p, N) for p in exps])
    return exps[np.argmin(np.abs(parts-pr))]

#############################
####### Color wheel #########
#############################

def _cos_sin_col(col):
    return np.cos(col), np.sin(col)

def _rf_decomp(col, n_units=10, wid=2):
    cents = np.linspace(0, 2*np.pi - (1/n_units)*2*np.pi, n_units)
    cents = np.expand_dims(cents, 0)
    col = np.expand_dims(col, 1)
    r = np.exp(wid*np.cos(col - cents))/np.exp(wid)
    return list(r[:, i] for i in range(n_units))

def decompose_colors(*cols, decomp_func=_cos_sin_col, **kwargs):
    all_cols = []
    for col in cols:
        all_cols.extend(decomp_func(col, **kwargs))
    return np.stack(all_cols, axis=1)

class RFColors():
    def __init__(self, n_units=10, wid=2):

        self.dim_out = n_units
        self.wid = wid
        self.__name__ = 'RFColors'

    def __call__(self, *cols):
        return decompose_colors(*cols, decomp_func=_rf_decomp, 
            n_units=self.dim_out, wid=self.wid)

class TrigColors():
    def __init__(self):

        self.dim_out = 2
        self.__name__ = 'TrigColors'

    def __call__(self, *cols):
        return decompose_colors(*cols, decomp_func=_cos_sin_col)

def von_mises_mle(x, newt_steps=2):
    """ x is shape """
    
    rho = np.stack([np.sin(x),np.cos(x)])

    R_ = la.norm(rho.mean(1))
    rho_bar = rho.mean(1)/R_
    mu_mle = np.arctan2(rho_bar[0], rho_bar[1])

    kap = R_*(2-R_**2)/(1-R_**2)
    for s in range(newt_steps):
        kap -= (A2(kap) - R_)/(1 - A2(kap)**2 - (1/kap)*A2(kap))

    return mu_mle, kap

def A2(kappa):
    return spc.i1(kappa)/spc.i0(kappa)


#################################
##### Covariance alignment ######
#################################

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

############################################
######### Independence statsitics ##########
############################################

def dependence_statistics(x=None, dist_x=None):
    """
    Assume x is (n_feat, ..., n_sample), or dist_x is (...,n_sample, n_sample)

    U-centers the distance matrices, for unbiased estimators
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

    return V_xy/np.where(V_x*V_y>0, np.sqrt(V_x*V_y), 1e-12)
    # print([V_x, V_y, V_xy])
    # sing = V_x*V_y > 0
    # return sing*V_xy/(np.sqrt(V_x*V_y)+1e-30)
    # if 0 in [V_x, V_y]:
        # return 0
    # else:
        # return np.sqrt(V_xy/np.sqrt(V_x*V_y))
        # return V_xy/np.sqrt(V_x*V_y)

def partial_distance_correlation(x=None, y=None, z=None, dist_x=None, dist_y=None, dist_z=None):
    R_xy = distance_correlation(x, y, dist_x=dist_x, dist_y=dist_y)
    R_xz = distance_correlation(x, z, dist_x=dist_x, dist_y=dist_z)
    R_yz = distance_correlation(y, z, dist_x=dist_y, dist_y=dist_z)

    # print(R_xy)
    # print(R_xz)
    # print(R_yz)

    sing = (((R_xz**2) > 1-1e-12)+((R_yz**2) > 1-1e-12))
    return np.where(sing, 0, (R_xy - R_xz*R_yz)/np.sqrt((1-R_xz**2)*(1-R_yz**2)))

    # if ((R_xz**2) > 1-1e-4) or ((R_yz**2) > 1-1e-4):
    #     return 0
    # else:
    #     return (R_xy - R_xz*R_yz)/np.sqrt((1-R_xz**2)*(1-R_yz**2))

# def distance_correlation(X):
#     V = distance_covariance(X)
#     V_x = np.diag(V)
#     normlzr = V_x[None,:]*V_x[:,None]
#     R = np.zeros(V.shape)
#     R[normlzr>0] = np.sqrt(V[normlzr>0]/np.sqrt(normlzr[normlzr>0]))
#     return R

def centered_kernel_alignment(K1,K2):

    K1_ = K1 - np.nanmean(K1,-2,keepdims=True) - np.nanmean(K1,-1,keepdims=True) + np.nanmean(K1,(-1,-2),keepdims=True)
    K2_ = K2 - np.nanmean(K2,-2,keepdims=True) - np.nanmean(K2,-1,keepdims=True) + np.nanmean(K2,(-1,-2),keepdims=True)
    denom = np.sqrt(np.nansum((K1_**2),(-1,-2))*np.nansum((K2_**2),(-1,-2)))

    return np.nansum((K1_*K2_),(-1,-2))/np.where(denom, denom, 1e-12)


def kernel_alignment(k, l):
    """
    Unbiased estimator of the Hilbert-Schmidt Independence Criterion, which 
    I'm calling the kernel alignment because it is an objectively better name 
    """

    n = k.shape[-1]
    k_ = (1-np.eye(n))*k
    l_ = (1-np.eye(n))*l

    a = np.sum(k_*l_, axis=(-1,-2))
    b = np.sum(k_, axis=(-1,-2))*np.sum(l_, axis=(-1,-2))/((n-1)*(n-2))
    c = 2*np.sum(np.einsum('...ik,...kj->...ij',k_,l_), axis=(-1,-2))/(n-2)

    return (a + b - c)/(n*(n-3))

def normalized_kernel_alignment(k,l):
    return kernel_alignment(k,l)/np.sqrt(kernel_alignment(k,k)*kernel_alignment(l,l))

def partial_kernel_alignment(k1, k2, k3):
    h11 = kernel_alignment(k1, k1)
    h12 = kernel_alignment(k1, k2)
    h13 = kernel_alignment(k1, k3)
    h22 = kernel_alignment(k2, k2)
    h23 = kernel_alignment(k2, k3)
    h33 = kernel_alignment(k3, k3)
    
    r12 = h12/np.sqrt(h11*h22) 
    r13 = h13/np.sqrt(h11*h33)
    r23 = h23/np.sqrt(h22*h33)
    
    denom = np.sqrt((1-r13**2)*(1-r23**2))

    return (r12 - r23*r13)/np.where(denom>0, denom, 1e-12)
    # return np.where(denom>0, (r12 - r23*r13)/denom, 0)

#####################################
######### Gaussian process ##########
#####################################

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


###################################
######### Mobius strips ###########
###################################

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

####################################
######### Flow/assignment ##########
####################################

def flow_conservation_matrix(nodes, edges, capacities):
    """
    convert the flow network into a linear equality constraint for an LP
    """
    
    # find all nodes which aren't the source or sink
    eligible = [np.all(np.isin(edges, i).sum(0)>0) for i in nodes]
    
    M = []
    for i,n in enumerate(nodes):
        if eligible[i]:
            M.append((np.isin(edges, i)*[[1,-1]]).sum(1))
    
    return np.array(M)


def unbalanced_assignment(cost, adj=None):
    """
    cost is shape (P, N) with P <= N
    adj is the same shape and binary (default all ones)

    returns row_indices, col_indices

    Solves the assignment problem, generalizing the classic
    problem to the case of unbalanced sets. Unlike standard
    modifications or reductions, which find a one-sided 
    perfect matching (bidirectionally one-to-one), this ensures
    that every item participates (bidirectionally onto). So, an 
    item in the smaller set can connect with multiple in the larger,
    but an item in the larger can still only connect with one.

    """

    inds = LexOrder()

    ## left set is at most smaller than right set
    P, N = cost.shape

    if N < P: 
        transpose = True
        C = cost.T
        P, N = C.shape

    elif P == N: # scipy is certainly faster than me
        return linear_sum_assignment(cost)

    else:
        transpose = False
        C = cost

    ## bipartite incidence matrix
    left = np.arange(P) 
    right = np.arange(P, P+N)
    if adj is None:
        aye, jay = np.where(np.ones((P,N)))
    else:
        aye, jay = np.where(adj)
    
    ne = len(aye)
    edges = inds(left[aye], right[jay])

    cols = np.repeat(np.arange(len(edges)), 2)
    rows = np.empty((2*ne,), dtype=int)
    rows[0::2] = left[aye]
    rows[1::2] = right[jay]

    Inc = sprs.coo_matrix((np.ones(2*ne), (rows, cols)))

    ## modify for slack variables
    I = sprs.coo_matrix( (-np.ones(P), (np.arange(P), np.arange(P)) ), shape=(P+N,P))
    A = sprs.bmat([[Inc, I]], format='csr')
    b = np.ones(P+N)

    ## solve LP
    c = np.append(C[aye,jay], np.zeros(P))
    prog = lp(c, A_eq=A, b_eq=b)
    
    r, l = inds.inv(edges[prog.x[:ne] > 1e-6])

    if transpose:
        return r-P, l
    else:
        return l, r-P



##############################################
############# Binary variables ###############
##############################################

def mutual_information(binary):
    """
    Input: binary matrix (num_item, num_cluster)
    Output: mutual information (num_cluster, num_cluster)
    """
    
    a = np.stack([binary.T, 1-binary.T])
    
    # p(a)
    p_a = a.sum(-1)/binary.shape[0]
    Ha = np.sum(-p_a*np.log2(p_a, where=p_a>0), 0)
    
    # p(a | b=1)
    p_ab = a@binary / binary.sum(0)
    Hab = np.sum(-p_ab*np.log2(p_ab, where=p_ab>0), 0) # entropy
    
    # p(a | b=0)
    p_ab_ = a@(1-binary) / (1-binary).sum(0)
    Hab_ = np.sum(-p_ab_*np.log2(p_ab_, where=p_ab_>0), 0) # entropy
    
    return Ha[:,None] - binary.mean(0)[None,:]*Hab - (1-binary).mean(0)[None,:]*Hab_


def gf2elim(M):
    """
    Gaussian elimination over Z/2 (a.k.a. GF(2))

    From someone's github ... 
    """

    m,n = M.shape

    i=0
    j=0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) +i

        # swap rows
        #M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp

        aijn = M[i, j:]

        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

        col[i] = 0 #avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        i += 1
        j +=1

    return M


def get_depths(clus):
    """
    Not super sure what this does, I think it takes a binary matrix
    whose rows are clusters (a "hypergraph" incidence matrix) and 
    assigns to each one a depth.

    Dependency on networkx package
    """

    ovlp = (1*(clus>0))@(clus>0).T
    subs = (ovlp == np.diag(ovlp)) - np.eye(len(ovlp))
    # subs = ((clus>0)@clus.T==1) - np.eye(len(ovlp))
    
    G = nx.from_numpy_array((subs@subs==0)*subs, create_using=nx.DiGraph)
    depth = np.zeros(len(clus))
    for i in nx.topological_sort(G):
        anc = list(nx.ancestors(G,i))
        if len(anc)>0:
            depth[i] = np.max(depth[anc])+1

    return depth


#####################################
####### Metrics/similarities ########
#####################################

def cosine_sim(x,y):
    """Assume features are in first axis"""

    x_ = x/(la.norm(x,axis=0,keepdims=True)+1e-5)
    y_ = y/(la.norm(y,axis=0,keepdims=True)+1e-5)
    return np.einsum('k...i,k...j->...ij', x_, y_)

def dot_product(x,y):
    """Assume features are in first axis"""
    return np.einsum('k...i,k...j->...ij', x, y)

def discrete_metric(x,y):
    return np.abs(x - y).sum(0)

def norms(x,y,p=2):
    """Assume features are in first axis"""
    return la.norm(x[...,None] - y[...,None,:], p, axis=0)

def dot2dist(K):
    """
    Dot product to squared distances
    """
    return (np.abs(np.diag(K)[:,None] + np.diag(K)[None,:] - 2*K))

#########################################
######## Miscelaneous (for now) #########
#########################################

def random_basis(dim):
    C = np.random.randn(dim, dim)
    return la.qr(C)[0][:dim,:]

def decimal(binary, base=2):
    """ 
    convert binary vector to dedimal number (i.e. enumerate) 
    assumes second axis is the bits
    """
    d = (binary*(base**np.arange(binary.shape[1]))[None,:]).sum(1)
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


def significant_figures(nums, sig_figs):
    '''Because for some reason numpy doesnt do this ?????????'''

    digis = np.log10(np.abs(nums))
    sgn = np.sign(digis)
    mag = np.floor(np.abs(digis))

    pwr = 10**(sgn*mag + sgn*(sgn<0))
    return np.round(nums/pwr, sig_figs-1)*pwr

def unroll_dict(this_dict):
    """ 
    Takes a dict, where some values are lists, and creates a list of 
    dicts where those values are each entry of the corresponding list
    """

    variable_prms = {k:v for k,v in this_dict.items() if type(v) is list}
    fixed_prms = {k:v for k,v in this_dict.items() if type(v) is not list}

    if len(variable_prms)>0:
        all_dicts = []
        var_k, var_v = zip(*variable_prms.items())

        for vals in list(itt.product(*var_v)):
            all_dicts.append(dict(zip(var_k, vals), **fixed_prms))
    else:
        all_dicts = this_dict

    return all_dicts

