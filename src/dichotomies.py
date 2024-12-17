import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import scipy.linalg as la
import scipy.special as spc
import scipy.optimize as opt
from itertools import combinations, permutations, islice, filterfalse, chain
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import util

class Dichotomies:
    """An iterator for looping over all dichotomies when computing abstraction metrics"""

    def __init__(self, num_cond, special=None, extra=0, unbalanced=False):
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

            if unbalanced and num_cond>16:
                raise ValueError("Sorry, can't do unbalanced dichotomies of %d conditions ..."%num_cond)
            elif unbalanced:
                dcs = [j for i in range(1,int(num_cond/2)) for j in combinations(range(num_cond), i)]
                self.ntot += len(dcs)
                self.combs += dcs

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


def uncorrelated_dichotomies(num_item, pos_items, num_max=None):

    K = int(num_item/4)
    n_tot = 0.5*spc.binom(int(num_item/2), K)**2
    neg = np.setdiff1d(range(num_item),pos_items)
    if (num_max is None) or (num_max>=n_tot):
        x = np.concatenate([[np.sort(np.concatenate([p,n])) for n in combinations(neg,K)] \
            for p in combinations(pos_items,K)])
        x = x[:int(len(x)/2)]
    else:
        x = []
        for ix in Dichotomies(len(neg)-1, [], num_max):
            trn = np.array(pos_items)[np.append(0,np.array(ix)+1)]
            x.append(np.sort(np.append(trn, np.random.choice(neg, K, replace=False))))

    return x


def compute_ccgp(z, cond, coloring, clf, these_vars=None, twosided=False, 
        debug=False, return_weights=False, num_max=None):
    """
    z is shape (num_item, num_feat)
    cond and coloring are shape (num_item, )
    """

    num_cond = len(np.unique(cond))
    pos_conds = np.unique(cond[coloring==0])

    if these_vars is None:
        x = uncorrelated_dichotomies(num_cond, pos_conds, num_max=num_max)
    else:
        x = these_vars

    # ntot = spc.binom(int(self.num_cond/2), K)**2
    ccgp = []
    ws = []
    for train_set in x:
        is_trn = np.isin(cond, train_set)

        clf.fit(z[is_trn,...], coloring[is_trn])
        # print(clf.thrs)
        perf = clf.score(z[~is_trn,...], coloring[~is_trn])
        ccgp.append(perf)
        if return_weights:
            ws.append(np.append(clf.coef_, clf.intercept_))
        if twosided:
            clf.fit(z[~is_trn,...], coloring[~is_trn])
            perf = clf.score(z[is_trn,...], coloring[is_trn])
            ccgp.append(perf)
            if return_weights:
                ws.append(np.append(clf.coef_, clf.intercept_))
    outs = [ccgp]
    if debug:
        outs.append(x)
    if return_weights:
        outs.append(ws)

    return outs


def efficient_ccgp(coloring, clf, z=None, K=None, cond=None, num_pairs=None, max_ctx=None, parallel=True,
    return_weights=False, return_pairs=False):
    """
    A more efficient way of computing CCGP, not sure how equivalent to the actual quantity
    """

    tol = 1e-6

    if cond is not None:
        N = len(np.unique(cond))
    else:
        N = len(coloring)
        cond = np.arange(N)
    pos_conds = np.unique(cond[coloring>0])
    neg_conds = np.unique(cond[coloring<=0])

    if num_pairs is None:
        num_pairs = N//2 - 1
    # if max_ctx is None:
    #     max_ctx = 

    if z is not None:
        dual = False
    elif K is not None:
        dual = True
    else:
        raise ValueError('Must supply features or kernel')

    if parallel:
        ## set the "positives" to be the smaller set

        y_ = np.array([coloring[cond==c].mean() for c in np.unique(cond)])
        ids = np.argsort(1-y_)
        y_ = y_[ids] > tol # convert to binary if it isn't

        if dual:
            Kz = np.array([[K[cond==c,:][:,cond==c_].mean()  for c in np.unique(cond)] for c_ in np.unique(cond)])
            Kz = Kz[ids,:][:,ids]
        else:
            z_ = np.hstack([z[:,cond==c].mean(1, keepdims=True) for c in np.unique(cond)])
            z_ = z_[:, ids]
            Kz = z_.T@z_

        pos = np.arange(np.sum(y_), dtype=int)
        neg = np.arange(np.sum(y_), N, dtype=int)

        ## compute (squared) distances
        norms = Kz[range(N), range(N)]
        d = norms[:,None] + norms[None,:] - 2*Kz

        ## find approximate optimal pairing
        aye, jay = util.unbalanced_assignment(d[y_,:][:,~y_])

        pairs = {pos_conds[i]: neg_conds[j] for i,j in zip(aye,jay)}

    ## enumerate over contexts

    if max_ctx is not None and spc.binom(N//2, num_pairs) > max_ctx:
        idx = np.random.choice(int(spc.binom(N//2, num_pairs)), max_ctx)
        these = [pos_conds[util.kcomblexorder(N//2, num_pairs, ii)>0] for ii in idx]
        # print(these)
    else:
        these = combinations(pos_conds, num_pairs)

    ccgp = []
    ws = []
    tset = []
    for train_pos in these:

        if parallel:
            ctx = [[pairs[i] for i in train_pos]]
        else:
            if spc.binom(N//2, num_pairs) > max_ctx:
                idx = np.random.choice(int(spc.binom(N//2, num_pairs)), max_ctx)
                ctx = [neg_conds[util.kcomblexorder(N//2, num_pairs, ii)>0] for ii in idx]
                # print(these)
            else:
                ctx = combinations(neg_conds, num_pairs)

        for train_neg in ctx:
            train_set = np.concatenate([train_pos, train_neg])

            is_trn = np.isin(cond, train_set)

            if dual:
                clf.fit(K[is_trn,:][:,is_trn], coloring[is_trn])
                # pred = clf.predict(K[~is_trn,:][:,is_trn], coloring[~is_trn])
                perf = clf.score(K[~is_trn,:][:,is_trn], coloring[~is_trn])
            else:
                clf.fit(z[...,is_trn].T, coloring[is_trn])
                perf = clf.score(z[..., ~is_trn].T, coloring[~is_trn])

            ccgp.append(perf)
            if return_weights:
                ws.append(np.append(clf.coef_, clf.intercept_))
            if return_pairs:
                tset.append(train_set)

    outs = [ccgp]
    # if debug:
    #     outs.append(x)
    if return_weights:
        outs.append(ws)
    if return_pairs:
        outs.append(tset)

    return outs

def parallelism_score(z, cond, coloring, eps=1e-12, debug=False, average=True):
    """ 
    Computes parallelism of coloring

    z is shape (num_feat, ..., num_items)
    cond and coloring are shape (num_items,)

    Instead of directly computing the difference vectors etc., uses a different but equivalent
    method based on kernels
    """

    if average:
        Z = np.array([z[...,cond==c].mean(-1) for c in np.unique(cond)]).T
        Y = np.array([coloring[cond==c].mean() for c in np.unique(cond)])
        conds = np.unique(cond)
    else:
        Z = z
        Y = coloring
        conds = cond

    # compute kernels
    z_cntr = Z - Z.mean(-1, keepdims=True)
    y_cntr = np.sign(Y - Y.mean())
    Kz = np.einsum('k...i,k...j->...ij', z_cntr, z_cntr)
    Ky = y_cntr[:,None]*y_cntr[None,:]
    
    pos = np.unique(conds[y_cntr>0])
    neg = np.unique(conds[y_cntr<0])

    ps = []
    pairs = []
    for n in permutations(neg, len(pos)):
        pairs.append(n)

        mask = 1 - np.eye(len(conds))
        mask[pos,n] = 0
        mask[n,pos] = 0

        incl = np.isin(conds, np.concatenate([pos,n]))
        y_mask = incl[:,None]*incl[None,:]

        if np.sum(mask)>0:
            # dz = torch.sum(dot2dist(Kz)*(1-mask-torch.eye(len(Kz))),0)
            # norm = dz[:,None]*dz[None,:]
            dist = np.diag(Kz)[:,None] + np.diag(Kz)[None,:] - 2*Kz
            dz = np.sqrt(np.abs(dist[(1-mask-np.eye(len(Kz)))>0]))
            norm = (dz[:,None]*dz[None,:]).flatten()

            # numer = torch.sum((Kz*Ky*mask)[Ky != 0]/norm[Ky != 0])
            numer = np.sum((Kz*Ky*mask)[y_mask]/norm)
            denom = (np.sum(np.tril(mask)[y_mask])/2) #+ eps
            ps.append(numer/denom)
        else:
            ps.append(0)

    if debug:
        return ps, pairs
    else:
        return np.max(ps)



def parallelism(coloring, z=None, K=None, 
        tol=1e-6, norm=True, aux_func='distsum', one_sided=False):
    """
    z is shape (features, items)
    coloring is shape (items,)

    Efficient computation of parallelism score, which generalizes to 
    unbalanced colorings. 
    """

    N = len(coloring)

    ## set the "positives" to be the smaller set
    if np.sum(coloring > tol) > N//2:
        y = 1 - coloring
    else:
        y = coloring

    ids = np.argsort(1-y)
    y_ = y[ids] > tol # convert to binary if it isn't
    k = N - np.sum(y_)

    pos = np.arange(np.sum(y_), dtype=int)
    neg = np.arange(np.sum(y_), N, dtype=int)

    ## compute (squared) distances
    if K is None:
        Kz = z[:,ids].T@z[:,ids]
    else:
        Kz = K[ids,:][:,ids]
    d = util.dot2dist(Kz)
    
    ## find approximate optimal pairing
    if aux_func == 'average':
        # Rather than maximize the average pairwise alignment, 
        # maximise the alignment of each pair to the average  

        yy = 2*y_ - 1

        C = np.sum((d[:,:,None] - d[:,None,:])*yy[:,None,None], 0)
        C /= np.sqrt(np.where(d<1e-5, 1e-5, d))
        aye, jay = util.unbalanced_assignment(-C[y_,:][:,~y_])

    elif aux_func == 'distsum':
        aye, jay = util.unbalanced_assignment(d[y_,:][:,~y_], one_sided=one_sided)

    # elif aux_func == 'none':
        # Solve the actual QAP, which takes a long time

    # print(aye)
    # print(jay)

    ## account for multiple pairing (imbalance)
    order = np.argsort(aye)
    if one_sided:
        
        # subset = np.isin(np.arange(N), np.append(pos[aye], neg[jay]))
        subset = np.concatenate([pos[aye], neg[jay]])
        k = len(aye)

        y_copy = y_[subset]
        d_copy = d[subset,:][:,subset]

        mask = np.ones((2*k, 2*k), dtype=bool)
        mask[range(k), range(k, 2*k)] = 0
        mask[range(k, 2*k), range(k)] = 0

        # print(d_copy)
        # print(y_copy)
        # print(mask)

    else:
        _, pos_reps = np.unique(aye, return_counts=True)
        reps = np.concatenate([pos_reps, np.ones(len(neg), dtype=int)], dtype=int)
        extra = np.sum(pos_reps - 1)

        i_id = np.arange(np.sum(pos_reps))
        j_id = jay[order]

        y_copy = np.repeat(y_, reps)
        d_copy = np.repeat(np.repeat(d, reps, axis=0), reps, axis=1)

        ## create matrices 
        mask = np.ones((N+extra,N+extra), dtype=bool)
        mask[i_id, neg[j_id]+extra] = 0
        mask[neg[j_id]+extra, i_id] = 0

    numer = mask*d_copy*np.outer(-(2*y_copy-1),(2*y_copy-1))
    denom = np.sqrt(np.outer(d_copy[~mask],d_copy[~mask]))

    if norm:
        total = np.sum(numer/np.where(denom <= tol, tol, denom))
    else:
        total = np.sum(numer)

    return total/(2*k*(k-1))


def hierarchical_parallelism(z, y_super, y_sub):

    sigs = [util.decompose_covariance(z[y_sup==s,:].T,y_sub[y_sup==s])[1] for s in np.unique(y_sup)]
            
    dots = np.einsum('ikl,jkl->ij',np.array(sigs),np.array(sigs))
    csim = la.triu(dots,1)/np.sqrt((np.diag(dots)[:,None]*np.diag(dots)[None,:]))
    foo1, foo2 = np.nonzero(np.triu(np.ones(dots.shape),1))

    return np.mean(csim[foo1,foo2])

