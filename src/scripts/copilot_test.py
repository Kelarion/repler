import numpy as np
from itertools import combinations
from scipy.special import binom


class LexOrder:
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
    
    def inv(self, n):
        """
        Inverts the function above -- given an index, return the list of items
        """

        K_sorted = np.flip(np.sort(self.K))

        # find the largest item first
        r_max = 0:
        while True:
            if n < binom(r, K_sorted[0]) + binom(r+1, K_sorted[1:]).sum()
            r_max += 1 


        n_new = n - binom(r_max, K_sorted[0])




def run_lengths(A, mask_repeats=True):
    """ 
    A "run" is a series of repeated values. For example, in the sequence

    [0, 2, 2, 1]

    there are 3 runs, of the elements 0, 2, and 1, with lengths 1, 2, and 1
    respectively. The output of this function would be 

    [1, 2, 0, 1]

    indicating the length of each run that an element starts. Now imagine this 
    being done to each column of an array.

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

inds = LexOrder(2, 4)

def kurtosis_inequalities(N):
    """
    Generate set of inequalities that constrain the kurtosis of sign variables.

    Assembles a sparse {-1,0,1}-valued matrix 
    """

    # K2 = int(binom(N,2))
    K4 = int(binom(N,4))

    rows = []
    cols = []
    vals = []
    b = np.zeros(int(binom(N,4)*(N-2)*42))

    n = 0
    for i,j,k,l in combinations(range(N),4):

        ind_ijkl = inds(i,j,k,l)

        # first constrain the 4th moment <ijkl> using the known 2nd moments
        for this_i, this_j in combinations([i,j,k,l],2):
            this_k, this_l = np.setdiff1d([i,j,k,l], [this_i, this_j])

            for this_m in np.setdiff1d(range(N), [this_k, this_l]):

                # upper and lower bounds
                for this, guy in enumerate([this_m, this_k, this_l]):
                    # "triangle inequality" upper bounds
                    cols.append([ind_ijkl, 
                        inds(this_i, this_j, this_k, this_m), 
                        inds(this_i, this_j, this_m, this_l)])
                    rows.append([n,n,n])
                    vals.append((2*np.eye(3, dtype=int) - 1)[this].tolist())
                    b[i] = 1

                    # "triangle inequality" lower bounds
                    feller, bloke = np.setdiff1d([this_m, this_k, this_l], guy)
                    cols.append([ind_ijkl, 
                        inds(this_i, this_j, this_k, this_m), 
                        inds(this_i, this_j, this_m, this_l),
                        inds(feller, bloke)])
                    rows.append([n+1,n+1,n+1,n+1])
                    vals.append(-(2*np.eye(3, dtype=int) - 1)[guy].tolist() + [-2])
                    b[n+1] = 1

                    n += 2

                cols.append([ind_ijkl, 
                    inds(this_i, this_j, this_k, this_m), 
                    inds(this_i, this_j, this_m, this_l),
                    inds(this_i, this_j)])
                rows.append([n,n,n,n])
                vals.append([1,1,1,-2])
                b[n] = 1

                n += 1


