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


class Dichotomies:
    """An iterator for looping over all dichotomies when computing abstraction metrics, which 
    also includes methods for computing said metrics if you want to."""

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
            trn = np.array(pos_conds)[np.append(0,np.array(ix)+1)]
            x.append(np.sort(np.append(trn, np.random.choice(neg, K, replace=False))))

    return x

def compute_ccgp(z, cond, coloring, clf, these_vars=None, twosided=False, 
        debug=False, return_weights=False,num_max=None):

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
            clf.fit(z[~is_trn,...], coloring[~is_trn][:,None])
            perf = clf.score(z[is_trn,...], coloring[is_trn][:,None])
            ccgp.append(perf)
            if return_weights:
                ws.append(np.append(clf.coef_, clf.intercept_))
    outs = [ccgp]
    if debug:
        outs.append(x)
    if return_weights:
        outs.append(ws)

    return outs


def parallelism_score(z, cond, coloring, eps=1e-12, debug=False):
    """ 
    Computes parallelism of coloring

    z is shape (num_feat, ..., num_items)
    cond and coloring are shape (num_items,)

    Instead of directly computing the difference vectors etc., uses a different but equivalent
    method based on kernels
    """

    # compute kernels
    z_cntr = z - z.mean(-1, keepdims=True)
    y_cntr = np.sign(coloring - coloring.mean())
    Kz = np.einsum('k...i,k...j->...ij', z_cntr, z_cntr)
    Ky = y_cntr[:,None]*y_cntr[None,:]
    
    pos = np.unique(cond[y_cntr>0])
    neg = np.unique(cond[y_cntr<0])

    ps = []
    pairs = []
    for n in permutations(neg, len(pos)):
        pairs.append(n)

        mask = 1 - np.eye(len(cond))
        mask[pos,n] = 0
        mask[n,pos] = 0

        incl = np.isin(cond, np.concatenate([pos,n]))
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

