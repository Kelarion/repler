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
from itertools import combinations, permutations

# silence that horrible, probably useless warning
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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
        return "LinearDecoder(ntoken=%d, classifier=%s)"%(len(self.tokens),self.clsfr.__name__)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, H, labels, t_=None, **cfargs):
        """
        Trains classifiers for each time bin
        optionally supply `t_` manually, e.g. if you want to train a single
        classifier over all time.

        H is shape (N_seq, ..., N_feat)
        labels is shape (N_seq, ..., P)
        """

        if t_ is None:
            # t_ = np.zeros(H.shape[:1]+H.shape[2:], dtype=int)
            t_ = np.zeros(H.shape[:-1], dtype=int)

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
            for p in self.tokens:
                clf[p][t].fit(H[t_==t,:], labels[t_==t,p])
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
        H is shape (N_seq, ..., N_feat)
        """

        # this is a bit tricky, because, in the most general setup, the model we use
        # at a given time & token depends on the particular sequence ... 

        if t_ is None:
            t_ = np.zeros(H.shape[:1]+H.shape[2:], dtype=int)

        if not (t_.max()<=self.time_bins.max() and t_.min()>=self.time_bins.min()):
            raise ValueError('The time bins of testing data are not '
                'the same as training data!\n'
                'Was trained on: %s \n'
                'Was given: %s'%(str(self.time_bins),str(np.unique(t_[t_>=0]))))

        # compute appropriate predictions
        proj = self.project(H, t_)

        # evaluate
        # correct = labels.transpose((2,0,1)) == (proj>=0)
        correct = np.moveaxis(labels,-1,0) == (proj>=0)
        perf = np.array([correct[:,t_==i].mean(1) for i in np.unique(t_)]).T

        return perf

    def margin(self, H, labels, t_=None):
        """ 
        Compute classification margin of each classifier, defined as the minimum distance between
        any point and the classification boundary.

        assumes class labels are 0 or 1
        """

        if t_ is None:
            t_ = np.zeros(H.shape[:1]+H.shape[2:], dtype=int)

        # compute predictions
        proj = self.project(H, t_)
        dist = proj*(2*np.moveaxis(labels,-1,0)-1) # distance to boundary

        marg = np.array([[dist[p, t_==t].min() for t in np.unique(t_)] \
            for p in self.tokens])

        return marg

    def project(self, H, t_=None):
        """
        returns H projected onto classifiers, where `t_clf` gives the classifier index
        (in time bins) of the classifier to use at each time in each sequence.

        H is shape (..., N_feat)
        """
        if t_ is None:
            t_ = np.zeros(H.shape[:1]+H.shape[2:], dtype=int)

        self.coefs[:,:,t_].shape
        # C = (self.coefs[:,:,t_].transpose((0,2,1,3))*H[None,...]).sum(2)
        C = np.einsum('ij...,...j->i...', self.coefs[:,:,t_], H)
        proj = C - self.thrs[:,t_]

        return proj

    def avg_dot(self, which_clf=None):
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
    """An iterator for looping over all dichotomies when computing parallelism, which 
    also includes a method for computing said parallelism if you want to."""

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
            # i.e. converting each p-bit binary vector into decimal number
            self.cond = np.einsum('...i,i',self.labels,2**np.arange(self.ntot))

        elif dictype == 'general':
            p = labels.shape[-1]
            if p > 5:
                raise ValueError("Sorry, n=%d variables have too many dichotomies ..."%p)

            self.ntot = int(spc.binom(2**p, 2**(p-1))/2)
            self.cond = np.einsum('...i,i',self.labels,2**np.arange(p))
            self.combs = combinations(range(2**p), 2**(p-1))
    
        else:
            raise ValueError('Value for "dictype" is not valid: ' + dictype)

        self.dictype = dictype
        self.num_var = labels.shape[-1]

        self.__iter__()

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

                # L = 
                # L = np.array([np.where((self.cond==n)|(self.cond==n+(2**p)),self.cond==n,np.nan)\
                #     for n in g_neg]).transpose(1,2,0)
                L = ~np.isin(self.cond, g_neg)

            if self.dictype == 'general':
                pos = next(self.combs)

                L = np.isin(self.cond, pos)

            self.curr +=1
            self.coloring = L

            return L

        else:
            raise StopIteration

    def parallelism(self, H, clf):
        """
        Computes the parallelism of H under the current coloring. It's admittedly weird 
        to include this as a method on the iterator, but here we are
        
        H is shape (N, ..., N_feat)
        clf is an instance of a LinearDecoder, with a method clf.orthogonality()
        """

        warnings.filterwarnings('ignore',message='invalid value encountered in')
        # get which conditions are on each side of the coloring
        pos = np.unique(self.cond[self.coloring])

        ps = []
        for neg in permutations(np.unique(self.cond[~self.coloring])):
            # for a given pairing of positive and negative conditions, I need to
            # generate labels for a classifier.
            whichone = np.array([(self.cond==pos[n])|(self.cond==neg[n]) \
                         for n, _ in enumerate(neg)]).argmax(0)
            lbs = np.isin(self.cond, pos)
            
            clf.fit(H, lbs[:,None], t_=whichone)
            clf.coefs = clf.coefs.transpose(2,1,0)
            ps.append(clf.avg_dot()[0])

        PS = np.max(ps)
        warnings.filterwarnings('default')

        return PS

    def CCGP(self, H, clf, K, **cfargs):
        """
        Cross-condition generalisation performance, computed on representation H using
        classifier clf, with K conditions (on each side) in the training set.
        """
        pos = np.unique(self.cond[self.coloring])
        neg = np.unique(self.cond[~self.coloring])
        
        ntot = spc.binom(2**(self.num_var-1), K)**2
        ccgp = 0
        for p in combinations(pos,K):
            for n in combinations(neg,K):
                train_set = np.append(p,n)
                is_trn = np.isin(self.cond, train_set)
                
                clf.fit(H[is_trn,...], self.coloring[is_trn][:,None], **cfargs)
                perf = clf.test(H[~is_trn,...], self.coloring[~is_trn][:,None])[0]
                ccgp += perf/ntot
        return ccgp

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class Indicator(nn.Module):
    """
    class to implement indicator function (i.e. one-hot encoding)
    it's a particular type of embedding, which isn't trainable

    by default, `-1` maps to all zeros. change this by setting the `padding` argument
    """
    def __init__(self, ntoken, ninp, padding_idx=-1):
        super().__init__()
        self.ntoken = ntoken
        self.ninp = ntoken

        self.padding = padding_idx

    def __repr__(self):
        return "Indicator(ntoken=%d, ninp=%d)"%(self.ntoken,self.ntoken)

    def forward(self, x):
        """
        Convert list of sequences x to an indicator (i.e. one-hot) representation
        extra dimensions are added at the end
        """

        def all_idx(idx, axis):
            """ from stack exchange"""
            grid = np.ogrid[tuple(map(slice, idx.shape))]
            grid.insert(axis, idx)
            return tuple(grid)
        
        ignore = np.repeat(np.expand_dims(x==self.padding, -1),self.ntoken,-1)

        out = np.zeros(x.shape + (self.ntoken,), dtype = int)
        out[all_idx(x, axis=2)] = 1
        out = torch.tensor(out).type(torch.FloatTensor)
        out[torch.tensor(ignore)] = 0
        return out

class ContextIndicator(nn.Module):
    """
    class to implement indicator function (i.e. one-hot encoding)
    with an additional dimension to indicate context (so, it's actually one-or-two-hot)
    ntoken should still be the number of tokens + 1 !!

    Caveat: this needs the FULL SEQUENCE, not a single time point. That's probably a bad
    thing, and I should fix it. TODO: make this work on single time points.
    """
    def __init__(self, ntoken, ninp, padding_idx=-1):
        super().__init__()
        self.ntoken = ntoken
        self.padding = padding_idx

    def __repr__(self):
        return "ContextIndicator(ntoken=%d, ninp=%d)"%(self.ntoken,self.ntoken)

    def forward(self, x):
        """
        Convert list of indices x to an indicator (i.e. one-hot) representation
        x is shape (lseq, ...)
        """

        def all_idx(idx, axis):
            """ from stack exchange"""
            grid = np.ogrid[tuple(map(slice, idx.shape))]
            grid.insert(axis, idx)
            return tuple(grid)

        # determine the context
        y = self.determine_context(x.detach().numpy())

        ignore = np.repeat(np.expand_dims(x==self.padding, -1),self.ntoken,-1)

        out = np.zeros(x.shape + (self.ntoken,), dtype = int)
        out[all_idx(x, axis=2)] = 1
        out[:,:,-1] += y
        out = torch.tensor(out).type(torch.FloatTensor)
        out[torch.tensor(ignore)] = 0
        return out

    def determine_context(self,x):
        """return the times at which a number is being repeated"""
        def find_reps(seq,mem):
            o = np.zeros((1,)+seq.shape[1:])
            r = np.diff(np.append(o,np.cumsum(seq==mem, axis=0).astype(int) % 2, axis=0), axis=0)<0
            return r
        # rep = [np.diff(np.cumsum(x==t, axis=1).astype(int) % 2, prepend=0)<0 for t in range(self.ntoken)]
        rep = [find_reps(x,t) for t in range(self.ntoken)]
        return np.any(rep, axis=0).astype(int)
