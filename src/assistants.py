"""
Classes that support analysis of the model outputs. More general.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import scipy.linalg as la
import scipy.special as spc
from itertools import combinations

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class LinearDecoder(object):
    """
    For getting the probability that each token is currently represented
    Right now all decoders are separate 
    """

    def __init__(self, N_feature, P, clsfr, padding=-1):
        super(LinearDecoder,self).__init__()
        
        # ntoken = rnn.encoder.ntoken

        # self.ntoken = ntoken
        self.tokens = list(range(P))
        self.nhid = N_feature
        self.padding = padding

        self.clsfr = clsfr

    def __repr__(self):
        return "LinearDecoder(ntoken=%d, classifier=%s)"%(len(self.tokens),self.clsfr)

    def fit(self, H, labels, t_=None, **cfargs):
        """
        Trains classifiers for each time bin
        optionally supply `t_` manually, e.g. if you want to train a single
        classifier over all time.

        H is shape (N_seq, N_feat, N_time or N_whatever)
        labels is shape (N_seq, N_whatever, P)
        """

        if t_ is None:
            t_ = np.zeros((H.shape[0], H.shape[2]), dtype=int)

        t_lbs = np.unique(t_[t_>=0])

        # L = int(np.ceil(len(t_lbs)/2))

        # # make targets
        # labels = np.zeros(I.shape + (len(self.tokens),)) # lenseq x nseq x ntoken
        # for p in self.tokens:
        #     labels[:,:,p] =np.apply_along_axis(self.is_memory_active, 0, I, p)

        # define and train classifiers
        clf = [[self.clsfr(**cfargs) for _ in t_lbs] for _ in self.tokens]

        # H_flat = H.transpose(1,0,2).reshape((self.nhid,-1))
        for t in t_lbs:
            idx = np.nonzero(t_==t)
            for p in self.tokens:
                clf[p][t].fit(H[idx[0],:,idx[1]], labels[t_==t,p])
            # clf[-1][t].fit(H_flat.T, t_.flatten()>=L) # last decoder is always context

        # package
        coefs = np.zeros((len(self.tokens), self.nhid, len(t_lbs)))*np.nan
        thrs = np.zeros((len(self.tokens), len(t_lbs)))*np.nan
        for p in self.tokens:
            for t in t_lbs:
                if ~np.all(np.isnan(clf[p][t].coef_)):
                    coefs[p,:,t] = clf[p][t].coef_/la.norm(clf[p][t].coef_)
                    thrs[p,t] = -clf[p][t].intercept_/la.norm(clf[p][t].coef_)

        self.clf = clf
        self.nclfr = len(self.tokens)
        self.time_bins = t_lbs
        self.coefs = coefs
        self.thrs = thrs

    def test(self, H, labels, t_=None):
        """
        Compute performance of the classifiers on dataset (H, I)
        H is shape (N_seq, N_feat, N_time or N_whatever)
        """

        # this is a bit tricky, because, in the most general setup, the model we use
        # at a given time & token depends on the particular sequence ... 

        if t_ is None:
            t_ = np.zeros((H.shape[0], H.shape[2]), dtype=int)

        if not (t_.max()<=self.time_bins.max() and t_.min()>=self.time_bins.min()):
            raise ValueError('The time bins of testing data are not '
                'the same as training data!\n'
                'Was trained on: %s \n'
                'Was given: %s'%(str(self.time_bins),str(np.unique(t_[t_>=0]))))

        # compute appropriate predictions
        proj = self.project(H, t_)

        # evaluate
        correct = labels.transpose((2,0,1)) == (proj>=0)
        perf = np.array([correct[:,t_==i].mean(1) for i in np.unique(t_)]).T

        return perf

    def margin(self, H, labels, t_=None):
        """ 
        Compute classification margin of each classifier, defined as the minimum distance between
        any point and the classification boundary.

        assumes class labels are 0 or 1
        """

        if t_ is None:
            t_ = np.zeros((H.shape[0], H.shape[2]), dtype=int)

        # compute predictions
        proj = self.project(H, t_)
        dist = proj*(2*labels.transpose((2,0,1))-1) # distance to boundary

        marg = np.array([[dist[p, t_==t].min() for t in np.unique(t_)] \
            for p in self.tokens])

        return marg

    def project(self, H, t_clf=None):
        """
        returns H projected onto classifiers, where `t_clf` gives the classifier index
        (in time bins) of the classifier to use at each time in each sequence.
        """
        if t_clf is None:
            t_clf = np.zeros((H.shape[0], H.shape[2]), dtype=int)

        C = (self.coefs[:,:,t_clf].transpose((0,2,1,3))*H[None,:,:,:]).sum(2)
        proj = C - self.thrs[:,t_clf]

        return proj

    def orthogonality(self, which_clf=None):
        """
        Computes the average dot product.
        """
        
        if which_clf is None:
            # which_clf = np.ones(self.nclfr)>0
            which_clf = ~np.any(np.isnan(self.coefs),axis=(1,2))

        C = self.coefs[which_clf, ...]
        # C = np.where(np.isnan(C), 0, C)
        csin = (np.einsum('ik...,jk...->ij...', C, C))
        PS = np.triu(csin.transpose(2,1,0),1).sum((1,2))
        PS /= sum(which_clf)*(sum(which_clf)-1)/2
        
        return PS

class MeanClassifier(object):
    """
    A class which just computes the vector between the mean of two classes. Is used for
    computing the parallelism score, for a particular choice of dichotomy.
    """
    def __init__(self):
        super(MeanClassifier,self).__init__()

    def fit(self, X, Y):
        """
        X is of shape (N_sample x N_dim), Y is (N_sample,) binary labels
        """

        V = np.nanmean(X[Y>0,:],0)-np.nanmean(X[Y<=0,:],0)

        self.coef_ = V
        if np.all(np.isnan(V)):
            self.intercept_ = np.nan
        else:
            self.intercept_ = la.norm(V)/2 # arbitrary intercept

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# helper classes

class Dichotomies:
    """An iterator for looping over all dichotomies when computing parallelism"""

    def __init__(self, labels, dictype='simple'):
        """
        Takes an n-bit binary vector for each datapoint, and produces an iterator over the possible
        dichotomies of those classes. Each 'dichotomy' of the iterator is a re-labeling of the data
        into 2^(n-1) binary classifications.

        More specifically, 'labels' should be a binary vector for each datapoint you want to compute
        parallelism with. e.g. each timepoint in a sequence. 
        
        labels should be shape (..., n) where n is the number of binary variables.

        TODO: support various types of dichotomy -- right now only considers dichotomies where the
        same binary variable is flipped in every class (calling it a 'simple' dichotomy).
        """

        self.labels = labels
        if dictype == 'simple':
            self.ntot = labels.shape[-1] # number of binary variables

            # 'cond' is the lexicographic enumeration of each binary condition
            # i.e. converting each ntot-bit binary vector into decimal number
            self.cond = np.einsum('...i,i',self.labels,2**np.arange(self.ntot))

        elif dictype == 'general':
            p = labels.shape[-1]
            self.ntot = int(spc.binom(2**p, 2**(p-1))/2)

            if p > 5:
                raise ValueError("I can't do %d dichotomies ..."%self.ntot)

            self.cond = np.einsum('...i,i',self.labels,2**np.arange(p))
            self.combs = combinations(range(2**p), 2**p-1)
    
        else:
            raise ValueError('Value for "dictype" is not valid: ' + dictype)

        self.dictype = dictype
        self.curr = 0

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr < self.ntot:
            if self.dictype == 'simple':
                p = self.curr
                # we want g_neg, the set of points in which condition p is zero
                bit_is_zero = np.arange(2**self.ntot)&np.array(2**p) == 0
                g_neg = np.arange(2**self.ntot)[bit_is_zero]

                L = np.array([np.where((self.cond==n)|(self.cond==n+(2**p)),self.cond==n,np.nan)\
                    for n in g_neg]).transpose(1,2,0)

            if self.dictype == 'general':
                pos = next(self.combs)

                L = np.isin(self.cond, pos)

            self.curr +=1

            return L

        else:
            raise StopIteration
