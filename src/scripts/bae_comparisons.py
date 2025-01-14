CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
# SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
 
import os, sys, re
import pickle
from time import time
sys.path.append(CODE_DIR)

import os
import pickle
import warnings
import re
from time import time

import torch as t
from torch import nn, Tensor
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from functools import partial
from dataclasses import dataclass
import tqdm

device = t.device("cuda" if t.cuda.is_available() else "cpu")

from collections import OrderedDict
import numpy as np
import scipy
import scipy.linalg as la
import scipy.special as spc
import scipy.stats as sts
import scipy.sparse as sprs
from scipy.optimize import nnls

from sklearn.exceptions import ConvergenceWarning
import warnings # I hate convergence warnings so much never show them to me
warnings.simplefilter("ignore", category=ConvergenceWarning)


import students
import super_experiments as exp
import util
import pt_util


#%% NMF + column clustering
import logging
import logging.config

class PyMFBase:
    """ PyMF base class used in (almost) all matrix factorization methods
    PyMF Base Class. Does nothing useful apart from providing some basic methods. """
    # some small value
    _EPS = 1e-6

    def __init__(self, data, num_bases=4, **kwargs):
        """
        """

        def setup_logging():
            # create logger
            self._logger = logging.getLogger("pymf")

            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()

        # set variables
        self.data = data
        self._num_bases = num_bases

        # initialize H and W to random values
        self._data_dimension, self._num_samples = self.data.shape

    def residual(self):
        """ Returns the residual in % of the total amount of data
            Returns: residual (float)
        """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0 * res / np.sum(np.abs(self.data))
        return total

    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank approximation given by WH.
            Minimizing the Fnorm ist the most common optimization criterion for matrix factorization methods.
            Returns: frobenius norm: F = ||data - WH||
        """
        # check if W and H exist
        if hasattr(self, 'H') and hasattr(self, 'W'):
            if sprs.issparse(self.data):
                tmp = self.data[:, :] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt(np.sum((self.data[:, :] - np.dot(self.W, self.H)) ** 2))
        else:
            err = None

        return err

    def _init_w(self):
        """ Initalize W to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as
        # they have difficulties recovering from zero.
        self.W = np.random.random((self._data_dimension, self._num_bases)) + 10 ** -4

    def _init_h(self):
        """ Initalize H to random values [0,1].
        """
        self.H = np.random.random((self._num_bases, self._num_samples)) + 0.2

    def _update_h(self):
        """ Overwrite for updating H.
        """
        pass

    def _update_w(self):
        """ Overwrite for updating W.
        """
        pass

    def _converged(self, i):
        """
        If the optimization of the approximation is below the machine precision,
        return True.

        Parameters
        ----------
            i   : index of the update step

        Returns
        -------
            converged : boolean
        """
        derr = np.abs(self.ferr[i] - self.ferr[i - 1]) / self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100, show_progress=False,
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data

        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].

        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

            # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self, 'W') and compute_w:
            self._init_w()

        if not hasattr(self, 'H') and compute_h:
            self._init_h()

            # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)

        for i in range(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()

            if compute_err:
                self.ferr[i] = self.frobenius_norm()
                self._logger.info('FN: %s (%s/%s)' % (self.ferr[i], i + 1, niter))
            else:
                self._logger.info('Iteration: (%s/%s)' % (i + 1, niter))

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


class SNMF(PyMFBase):
    """
    SNMF(data, num_bases=4)

    Semi Non-negative Matrix Factorization. Factorize a data matrix into two
    matrices s.t. F = | data - W*H | is minimal. For Semi-NMF only H is
    constrained to non-negativity.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())

    The result is a set of coefficients snmf_mdl.H, s.t. data = W * snmf_mdl.H.
    """

    def _update_w(self):
        W1 = np.dot(self.data[:, :], self.H.T)
        W2 = np.dot(self.H, self.H.T)
        self.W = np.dot(W1, np.linalg.pinv(W2))

    def _update_h(self):
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        XW = np.dot(self.data[:, :].T, self.W)

        WW = np.dot(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)

        XW_pos = separate_positive(XW)
        H1 = (XW_pos + np.dot(self.H.T, WW_neg)).T

        XW_neg = separate_negative(XW)
        H2 = (XW_neg + np.dot(self.H.T, WW_pos)).T + 10 ** -9

        self.H *= np.sqrt(H1 / H2)



#%% Alternating maximum likelihood

# def AML(X, r, S0=None, maxiter=100, decay=0.95, period=1, T0=1, alpha=0.1):
    
#     N, d = X.shape
    
#     ## Process X
#     X_ = X - X.mean(0)
#     # X_ = X
#     Ux,sx,Vx = la.svd(X_, full_matrices=False)
#     trXX = np.sum(sx**2)
#     P = Ux[:,sx>1e-6]
    
#     ## Initialise S
#     if S0 is None:
#         S = 1*(X_ @ np.random.randn(d,r) > 0)
#     else:
#         S = 1*S0
#     S_ = S - S.mean(0)
#     trSS = np.trace(S_.T@S_)
    
#     ## Initialise W
#     U,s,V = la.svd(X_.T@S_, full_matrices=False)
#     W = U@V
#     scl = np.sum(s)/trSS
    
#     ## Initialise b
#     b = np.zeros(d)
    
#     ls = []
#     for it in tqdm(range(maxiter)):
        
#         T = T0*decay**(it//period)
        
#         ## update S
#         C = (P@(P.T@S) + S.mean(0) - 0.5)
#         S = np.random.binomial(1, spc.expit((2*X_@W - 2*b@W - scl)/T))
#         # S = 1*(P@(P.T@S) + S.mean(0) > 0.5)
        
#         S_ = S - S.mean(0)
#         trSS = np.trace(S_.T@S_)
        
#         ## update W and scl
#         U,s,V = la.svd((X).T@S, full_matrices=False)
#         W = (1-alpha)*W + alpha*U@V
#         scl = (1-alpha)*scl + alpha*(np.sum(s)/trSS)
        
#         ## update b
        
        
#         ls.append(np.sum(s)/np.sqrt(trXX*trSS))
#         # ls.append(np.sum((scl*S_@W.T - X_)**2))
    
#     return (S, W, b), ls

def IQT(X, r, S0=None, maxiter=100, decay=0.95, period=1, T0=1, alpha=0.1, beta=1):
    """
    Invert affine factorization 
    
    X = SW' + b
    
    with MSE loss
    """
    
    N, d = X.shape
    
    ## Process X
    X_ = X - X.mean(0)
    Ux,sx,Vx = la.svd(X_, full_matrices=False)
    trXX = np.sum(sx**2)
    P = Ux[:,sx>1e-6]
    
    ## Initialise S
    if S0 is None:
        S = 1*(X_ @ np.random.randn(d,r) > 0)
    else:
        S = 1*S0
    trSS = np.trace(S.T@S)
    
    ## Initialise b
    b = np.zeros(d) # -X.mean(0)
    
    ## Initialise W
    U,s,V = la.svd((X_-b).T@S, full_matrices=False)
    W = U@V
    scl = np.sum(s)/trSS
    
    ls = []
    for it in tqdm(range(maxiter)):
        
        T = T0*decay**(it//period)
        
        ## update S
        C = (P@(P.T@S) + S.mean(0) - 0.5)
        S = np.random.binomial(1, spc.expit((beta*C + 2*(X_ - b)@W - scl)/T))
        # S = 1*(P@(P.T@S) + S.mean(0) > 0.5)
        
        # S_ = S - S.mean(0)
        trSS = np.trace(S.T@S)
        
        ## update W and scl
        U,s,V = la.svd((X_ - b).T@S, full_matrices=False)
        W = U@V
        scl = (np.sum(s))/trSS
        
        ## update b
        b = (X_ - scl*S@W.T).mean(0)
        
        ls.append(np.sum(s)/np.sqrt((trXX + N*b@b)*trSS))
        # ls.append(np.sum((scl*S_@W.T - X_)**2))
    
    return (scl*S, W, b), ls


