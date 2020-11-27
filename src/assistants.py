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
from itertools import combinations, permutations, islice, filterfalse, chain
from sklearn.utils._testing import ignore_warnings
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
                    coefs[p,:,t] = clf[p][t].coef_/(la.norm(clf[p][t].coef_)+1e-3)
                    thrs[p,t] = -clf[p][t].intercept_/(la.norm(clf[p][t].coef_)+1e-3)

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
    """An iterator for looping over all dichotomies when computing abstraction metrics, which 
    also includes methods for computing said metrics if you want to."""

    def __init__(self, num_cond, special=None, extra=0):
        """
        """
        
        self.num_cond = num_cond
        # self.cond = cond_labels

        if special is None:
            if num_cond > 32:
                raise ValueError("Sorry, %d conditions have too many dichotomies ..."%num_cond)

            self.ntot = int(spc.binom(num_cond, int(num_cond)/2)/2)
            # self.combs = list(combinations(range(num_cond), int(num_cond/2)))
            self.combs = list(islice(combinations(range(num_cond), int(num_cond/2)),self.ntot))

        else:
            # if num_cond > 32:
            #     raise ValueError("Sorry, %d conditions have too many dichotomies ..."%num_cond)

            combs = [tuple(np.sort(p).tolist()) for p in special]

            nmax = int(spc.binom(num_cond, int(num_cond)/2)/2)
            self.ntot = np.min([len(special)+extra, nmax])
            # print(self.ntot)
            # get all non-special dichotomies; this is convoluted but memory-efficient?
            if self.ntot>0.5*nmax: # direct
                # print('direct')
                remain = list(filterfalse(lambda x: x in combs,
                    islice(combinations(range(num_cond),int(num_cond/2)),nmax)))
                combs += remain
            else: # rejection sample
                # print('rejection sampling')
                brk = 0
                while len(combs)<self.ntot:
                    if brk > 2*self.ntot:
                        break
                    tst = np.sort((np.random.choice(num_cond-1, int(num_cond/2)-1, replace=False)+1))
                    tst = tuple(np.append(0,tst).tolist())
                    if tst not in combs:
                        # print(remain)
                        combs.append(tst)
                    else:
                        brk += 1
            # print(remain)
            # self.combs = special + np.random.permutation(remain)[:extra].tolist()
            # self.combs = chain(special, remain)
            self.combs = combs

        self.__iter__()

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr < self.ntot:
            self.pos = self.combs[self.curr]
            # self.pos = next(self.combs)
            # L = np.isin(self.cond, self.pos)

            self.curr +=1
            # self.coloring = L

            return self.pos

        else:
            self.__iter__()
            raise StopIteration

    def coloring(self, cond):
        return np.isin(cond, self.pos)

    def correlation(self, dics):
        """Get 'correlation' of current dichotomy with dics"""
        return np.array([(2*np.isin(self.pos,d)-1).mean() for d in dics])

    def get_uncorrelated(self, num_max=None):
        """
        Returns up to num_max dichotomies which are uncorrelated with the current coloring
        
        If num_max is less than the highest possible number of such dichotomies, it will use 
        a rejection sampling based approach which will be very slow if num_max is still close to
        the theoretical limit.
        """
        
        K = int(self.num_cond/4)
        n_tot = 0.5*spc.binom(int(self.num_cond/2), K)**2
        neg = np.setdiff1d(range(self.num_cond),self.pos)
        if (num_max is None) or (num_max>=n_tot):
            x = np.concatenate([[np.sort(np.concatenate([p,n])) for n in combinations(neg,K)] \
                for p in combinations(self.pos,K)])
            x = x[:int(len(x)/2)]
        else:
            x = []
            for ix in Dichotomies(len(neg)-1, [], num_max):
                trn = np.array(self.pos)[np.append(0,np.array(ix)+1)]
                x.append(np.sort(np.append(trn, np.random.choice(neg, K, replace=False))))
        return x

    # def parallelism(self, H, cond, clf):
    #     """
    #     Computes the parallelism of H under the current coloring. It's admittedly weird 
    #     to include this as a method on the iterator, but here we are
        
    #     H is shape (N, ..., N_feat)
    #     clf is an instance of a LinearDecoder, with a method clf.orthogonality()
    #     """

    #     coloring = np.isin(cond, self.pos)

    #     ps = []
    #     for neg in permutations(np.unique(cond[~coloring])):
    #         # for a given pairing of positive and negative conditions, I need to
    #         # generate labels for a classifier.
    #         whichone = np.array([(cond==self.pos[n])|(cond==neg[n]) \
    #                      for n, _ in enumerate(neg)]).argmax(0)

    #         clf.fit(H, coloring[:,None], t_=whichone)
    #         clf.coefs = clf.coefs.transpose(2,1,0)
    #         ps.append(clf.avg_dot()[0])

    #     PS = np.max(ps)

    #     return PS

    def parallelism(self, H, cond, clf, debug=False):
        """
        Computes the parallelism of H under the current coloring. It's admittedly weird 
        to include this as a method on the iterator, but here we are
        
        H is shape (N, ..., N_feat)
        clf is an instance of a LinearDecoder, with a method clf.orthogonality()
        """

        coloring = np.isin(cond, self.pos)

        ps = []
        negs = []
        for neg in permutations(np.unique(cond[~coloring])):
            # for a given pairing of positive and negative conditions, I need to
            # generate labels for a classifier.
            whichone = np.array([(cond==self.pos[n])|(cond==neg[n]) \
                         for n, _ in enumerate(neg)]).argmax(0)

            clf.fit(H, coloring[:,None], t_=whichone)
            clf.coefs = clf.coefs.transpose(2,1,0)
            ps.append(clf.avg_dot()[0])
            negs.append(neg)

        PS = np.max(ps)
        if debug:
            return PS, negs[np.argmax(PS)]
        else:
            return PS

    # def CCGP(self, H, cond, clf, K, **cfargs):
    #     """
    #     Cross-condition generalisation performance, computed on representation H using
    #     classifier clf, with K conditions (on each side) in the training set.
    #     """
    #     coloring = np.isin(cond, self.pos)
    #     # pos = np.unique(self.cond[self.coloring])
    #     # pos = self.pos
    #     neg = np.unique(cond[~coloring])
        
    #     ntot = spc.binom(int(self.num_cond/2), K)**2
    #     ccgp = 0
    #     for p in combinations(self.pos,K):
    #         for n in combinations(neg,K):
    #             train_set = np.append(p,n)
    #             is_trn = np.isin(cond, train_set)
                
    #             clf.fit(H[is_trn,...], coloring[is_trn][:,None], **cfargs)
    #             perf = clf.test(H[~is_trn,...], coloring[~is_trn][:,None])[0]
    #             ccgp += perf/ntot
    #     return ccgp

    def CCGP(self, H, cond, clf, these_vars=None, twosided=False, debug=False, **cfargs):
        """
        Cross-condition generalisation performance, computed on representation H using
        classifier clf, with K conditions (on each side) in the training set.
        """
        coloring = np.isin(cond, self.pos)
        # pos = np.unique(self.cond[self.coloring])
        # pos = self.pos

        if these_vars is None:
            x = self.get_uncorrelated()
        else:
            x = these_vars
        # ntot = spc.binom(int(self.num_cond/2), K)**2
        ccgp = []
        for train_set in x:
            is_trn = np.isin(cond, train_set)

            clf.fit(H[is_trn,...], coloring[is_trn][:,None], **cfargs)
            perf = clf.test(H[~is_trn,...], coloring[~is_trn][:,None])[0]
            ccgp.append(perf)
            if twosided:
                clf.fit(H[~is_trn,...], coloring[~is_trn][:,None], **cfargs)
                perf = clf.test(H[is_trn,...], coloring[is_trn][:,None])[0]
                ccgp.append(perf)
        if debug:
            return ccgp, x
        else:
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
