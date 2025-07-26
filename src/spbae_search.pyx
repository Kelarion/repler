cimport numpy as np
import numpy as np
from libc.math cimport exp, ceil, lround
from libc.stdlib cimport malloc, free
from cython.parallel import prange

ctypedef np.int32_t int32
ctypedef np.npy_uint32 UINT32_t
cdef inline UINT32_t DEFAULT_SEED = 1
cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647

np.import_array()

######################################################################
#### Utils, because it's really annoying to import across files ######
######################################################################

cdef class FiniteSet:
    """
    Custom unordered set with O(1) add/remove methods
    which takes advantage of the fixed maximum set size

    Many thanks to Jerome Richard on stackoverflow for the suggestion
    """
    cdef:
        int* index     # Index of element in value array
        int* value     # Array of contained element
        int size       # Current number of elements

    def __cinit__(self, int[:] indices, int maximum):
        """C-level initialization"""
        self.size = len(indices)
        
        # Allocate arrays
        self.index = <int*>malloc(maximum * sizeof(int))
        self.value = <int*>malloc(maximum * sizeof(int))
        
        if not self.index or not self.value:
            raise MemoryError("Could not allocate memory")
        
        # Initialize index array to -1 (invalid)
        cdef int i
        for i in range(maximum):
            self.index[i] = -1
            
        # Set up initial values
        for i in range(self.size):
            self.value[i] = indices[i]
            self.index[indices[i]] = i 

    def __dealloc__(self):
        """Cleanup C memory"""
        if self.index:
            free(self.index)
        if self.value:
            free(self.value)

    cdef inline bint contains(self, int i) nogil:
        return self.index[i] >= 0 

    cdef inline void add(self, int i) nogil:
        """
        Add element to set
        """
        # if not self.contains(i):
        if self.index[i] < 0:
            self.value[self.size] = i
            self.index[i] = self.size
            ## Increase
            self.size += 1

    cdef inline void remove(self, int i) nogil:
        """
        Remove element from set
        """
        # if self.contains(i):
        if self.index[i] >= 0:
            self.value[self.index[i]] = self.value[self.size - 1]
            self.index[self.value[self.size - 1]] = self.index[i]
            self.index[i] = -1
            self.size -= 1

cdef class BiMat:
    """
    A format for sparse binary matrices, initialized from csr format,
    which you could call an "uncompressed sparse row" format

    Designed for O(1) bit flipping and O(nnz) iteration over rows, but
    uses the same memory (more, actually) as a dense array. Just a 2d
    extension of the FiniteSet format suggested by Jerome.
    """
    ## Examples: 
    ## To get a list of non-zero columns for row i, do
    ## elements = []
    ## for j in range(size[i]):
    ##     elements.append(value[i][j])
    ## 
    ## To check if element j is in set i, do
    ## isin = (index[i][j] > 0)
    cdef:
        int **index    # Index of element in value array
        int **value    # Array of contained elements
        int *size      # Current number of elements per row
        int nrow
        int ncol

    def __cinit__(self, int[:] indices, int[:] indptr, int dimension):
        """C-level initialization"""
        
        self.nrow = len(indptr) - 1
        self.ncol = dimension
        cdef int i, j

        self.index = <int **>malloc(self.nrow * sizeof(int *))
        self.value = <int **>malloc(self.nrow * sizeof(int *))
        self.size = <int *>malloc(self.nrow * sizeof(int))

        # print('mallocd rows')
        if not self.index or not self.value or not self.size:
            raise MemoryError("Could not allocate memory")
        
        for i in range(self.nrow): # each row's pointer
            self.index[i] = <int *>malloc(dimension * sizeof(int))
            self.value[i] = <int *>malloc(dimension * sizeof(int))

            if not self.index[i] or not self.value[i]:
                raise MemoryError("Could not allocate memory")

        # print('malloced cols')
        # Initialize index array to -1 (invalid) and size to 0
        for i in range(self.nrow):
            self.size[i] = 0
            for j in range(dimension):
                self.index[i][j] = -1
            
        # Set up initial values
        for i in range(self.nrow):
            for j in range(indptr[i], indptr[i+1]):
                self.value[i][self.size[i]] = indices[j]
                self.index[i][indices[j]] = self.size[i]
                self.size[i] += 1 

    def __dealloc__(self):
        """Cleanup C memory"""
        cdef int i
        if self.index:
            for i in range(self.nrow):
                if self.index[i]:
                    free(self.index[i])
            free(self.index)
        if self.value:
            for i in range(self.nrow):
                if self.value[i]:
                    free(self.value[i])
            free(self.value)
        if self.size:
            free(self.size)

    cdef inline int add(self, int i, int j) nogil:
        """
        Add element to set
        """
        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            return -1

        if self.index[i][j] < 0:
            self.value[i][self.size[i]] = j
            self.index[i][j] = self.size[i]
            self.size[i] += 1

        return 1

    cdef inline int remove(self, int i, int j) nogil:
        """
        Remove element from set
        """

        if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
            return -1

        cdef int remove_idx, last_idx

        if self.index[i][j] >= 0:
            remove_idx = self.index[i][j]
            last_idx = self.size[i] - 1

            if remove_idx < last_idx: # don't swap if last element
                self.value[i][remove_idx] = self.value[i][last_idx]
                self.index[i][self.value[i][last_idx]] = remove_idx

            self.index[i][j] = -1
            self.size[i] -= 1

        return 1 

    cdef inline double[:,:] _dot(self, double[:,:] X, double[:,:] Z) nogil:
        """
        Matrix multiplication on the left, Z = SX
        Must provide the output array Z of appropriate shape
        """
        cdef nout = X.shape[1]
        cdef int i, j

        for i in prange(self.nrow, nogil=True):
            for j in range(nout):
                Z[i][j] = 0
                for k in range(self.size[i]):
                    Z[i][j] += X[j][self.value[i][k]]
        return Z

    cdef inline double[:,:] kernel(self, double[:,:] K) nogil:
        """
        Return the kernel matrix K = SS'    
        """
        cdef int i,j,k

        for i in prange(self.nrow, nogil=True):
            for j in range(self.nrow):
                K[i][j] = 0
                if self.size[i] <= self.size[j]: # only need to check the smallest
                    for k in range(self.size[i]):
                        if self.self.index[j][k] > 0: # k in set j
                            K[i][j] += 1.0 
                else:
                    for k in range(self.size[j]): 
                        if self.self.index[i][k] > 0: # k in set i
                            K[i][j] += 1.0 
        return K

    def pyadd(self, int i, int j):
        return self.add(i,j)

    def pyremove(self, int i, int j):
        return self.remove(i,j)

    def dot(self, double[:,:] X):
        cdef int M = X.shape[1]
        cdef np.ndarray[double, ndim=2] out = np.zeros((self.nrow, M))
        return self._dot(X, out)

    def ker(self):
        cdef np.ndarray[double, ndim=2] K = np.zeros((self.nrow, self.nrow))
        return self.kernel(K)

    def to_csr(self):
        """
        Output indices and indptr arrays
        """
        cdef np.ndarray[int, ndim=1] indptr = np.zeros(self.nrow+1, dtype=int)
        cdef np.ndarray[int, ndim=1] indices 
        cdef list ind_list = []
        cdef int i, j, k

        for i in range(self.nrow):
            indptr[i+1] = self.size[i] + indptr[i]
            for j in range(self.size[i]):
                ind_list.append(self.value[i][j])
        indices = np.array(ind_list, dtype=int)

        return indices, indptr


cdef class XORRNG:
    """
    Custom XORRNG sampler that I copied from the scikit-learn source code
    """

    cdef UINT32_t state

    def __cinit__(self, UINT32_t seed=DEFAULT_SEED):
        if (seed == 0):
            seed = DEFAULT_SEED
        self.state = seed

    cdef inline double sample(self) nogil:
        """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
        self.state ^= <UINT32_t>(self.state << 13)
        self.state ^= <UINT32_t>(self.state >> 17)
        self.state ^= <UINT32_t>(self.state << 5)

        # Use the modulo to make sure that we don't return a values greater than the
        # maximum representable value for signed 32bit integers (i.e. 2^31 - 1).
        # Note that the parenthesis are needed to avoid overflow: here
        # RAND_R_MAX is cast to UINT32_t before 1 is added.
        return <double>(self.state % ((<UINT32_t>RAND_R_MAX) + 1))/RAND_R_MAX

cdef inline double cymin(double a, double b, double c) nogil:
    cdef double out = a
    if b < out:
        out = b
    if c < out:
        out = c
    return out

cdef inline double sigmoid(double curr) nogil:
    if curr < -100:
        return 0.0
    elif curr > 100:
        return 1.0
    else:
        return 1.0 / (1.0 + exp(-curr))


######################################################################
############# Classes for sampling from p(s|x) #######################
######################################################################

cdef XORRNG prng = XORRNG()

cdef class KernelSampler:
    """
    This stores the statistics of S in order to sample from
    the p(s|x) under the kernel MSE likelihood:
        p(x|s) ~ exp(s'Cov(s,s)s - 2 s'Cov(s,x)x) / Z
    """
    
    cdef double **J             # recurrent weights (Cov s,s)
    cdef double **W             # input weights (Cov s,x)
    cdef double **dJ            # accumulated change in J
    cdef double **dW            # ditto for W
    cdef double *h              # offset/intercept
    cdef double *r              # inhibition offset
    cdef double lr              # learning rate
    cdef double th              # inhib threshold
    cdef int dimx               # input dimension
    cdef int dims               # latent dimension   
    cdef int bsz                # batch size

    def __cinit__(self, double[:,:] Jinit, double[:,:] Winit, 
                        int batch_size, double lr):
    
        cdef int i, j

        self.dims = Winit.shape[0]
        self.dimx = Winit.shape[1]
        self.bsz = batch_size
        self.lr = lr
        self.th = lr / batch_size

        # Allocate arrays
        self.J = <double **>malloc(self.dims * sizeof(double *))
        self.W = <double **>malloc(self.dims * sizeof(double *))
        self.dJ = <double **>malloc(self.dims * sizeof(double *))
        self.dW = <double **>malloc(self.dims * sizeof(double *))
        self.h = <double *>malloc(self.dims * sizeof(double))
        self.r = <double *>malloc(self.dims * sizeof(double))

        if not self.J or not self.W or not self.h or not self.r:
            raise MemoryError("Could not allocate memory")
        
        for i in range(self.dims): 
            self.J[i] = <double *>malloc(self.dims * sizeof(double))
            self.W[i] = <double *>malloc(self.dimx * sizeof(double))
            self.dJ[i] = <double *>malloc(self.dims * sizeof(double))
            self.dW[i] = <double *>malloc(self.dimx * sizeof(double))

            if not self.J[i] or not self.W[i] or not self.dJ[i] or not self.dW[i]:
                raise MemoryError("Could not allocate memory")

        ## Initialize arrays
        for i in range(self.dims):
            self.h[i] = 0.0
            self.r[i] = 0.0
            for j in range(self.dims):
                self.J[i][j] = Jinit[i][j]
                self.h[i] += (Jinit[i][j] - Jinit[i][i] * Jinit[j][j]) * Jinit[j][j]
                self.r[i] += self.R0(i,j)
                self.dJ[i][j] = 0.0

            for j in range(self.dimx):
                self.W[i][j] = Winit[i][j]
                self.dW[i][j] = 0.0

    def __dealloc__(self):
        """Cleanup C memory"""
        cdef int i
        if self.J:
            for i in range(self.dims):
                if self.J[i]:
                    free(self.J[i])
            free(self.J)
        if self.W:
            for i in range(self.dims):
                if self.W[i]:
                    free(self.W[i])
            free(self.W)
        if self.dJ:
            for i in range(self.dims):
                if self.dJ[i]:
                    free(self.dJ[i])
            free(self.dJ)
        if self.dW:
            for i in range(self.dims):
                if self.dW[i]:
                    free(self.dW[i])
            free(self.dW)
        if self.h:
            free(self.h)
        if self.r:
            free(self.r)

    cdef inline void learn(self) nogil:
        """
        Update shared parameters
        """
        cdef int i, j

        for i in prange(self.dims, nogil=True):
            self.h[i] = 0.0
            self.r[i] = 0.0
            for j in range(self.dims):
                self.J[i][j] += self.lr*(self.dJ[i][j] - self.J[i][j])
                self.h[i] += (self.J[i][j] - self.J[i][i] * self.J[j][j]) * self.J[j][j]
                self.r[i] += self.R0(i,j)
                self.dJ[i][j] = 0.0

            for j in range(self.dimx):
                self.W[i][j] += self.lr*(self.dW[i][j] - self.W[i][j])
                self.dW[i][j] = 0.0

    cdef inline double R(self, int i, int j) nogil:
        """
        Compute value of R matrix at index (i,j)
        """

        cdef double A, B, C, D

        A = (1 - self.th)*self.J[i][j] 
        B = (1 - self.th)*self.J[i][i] - A
        C = self.J[j][j] - A
        D = 1 - A - B - C

        if A - cymin(B,C-self.th,D) < self.th:
            return 1.0
        if B - cymin(A,C,D-self.th) < self.th:
            return -1.0
        if C - cymin(A,B,D) < self.th:
            return -1.0
        if D - cymin(A,B,C) < self.th:
            return 1.0
        return 0.0

    cdef inline double R0(self, int i, int j) nogil:
        """
        Compute constant value of R matrix at index (i,j)
        """

        cdef double A, B, C, D

        A = (1 - self.th)*self.J[i][j] 
        B = (1 - self.th)*self.J[i][i] - A
        C = self.J[j][j] - A
        D = 1 - A - B - C

        if B - cymin(A,C,D-self.th) < self.th:
            return 1.0
        if D - cymin(A,B,C) < self.th:
            return -1.0
        return 0.0

    cdef inline int recur(self, 
                          BiMat S, 
                          double[:] x, 
                          int i,
                          double scl,
                          double temp,
                          double beta) nogil:

        cdef bint regularize = (beta > 1e-6)
        cdef int check
        cdef double inp, dot, inhib, curr, prob
        cdef int j, k, l

        for j in range(self.dims):

            inp = 0.0
            for k in range(self.dimx):
                inp += 2 * self.W[j][k] * x[k]

            check = S.remove(i,j)

            dot = self.J[j][j] * (1 - self.J[j][j]) - 2 * self.h[j]
            inhib = beta * self.r[j]
            for l in range(S.size[i]):
                k = S.value[i][l]
                dot += 2 * (self.J[j][k] - self.J[j][j] * self.J[k][k])
                if regularize:
                    inhib += beta * self.R(j, k)

            curr = (inp - scl * dot - inhib) / temp
            prob = sigmoid(curr)

            if (prng.sample() < prob):
                check = S.add(i,j)

        return check

    def sample(self, BiMat S, 
                     double[:,:] X, 
                     double scl,
                     double temp, 
                     double beta):

        cdef int n = S.nrow
        cdef int n_batch = lround(ceil(n/self.bsz))

        cdef int check
        cdef bint regularize = (beta > 1e-6)
        
        cdef int b, i0, i, ii, j, k

        # cdef np.ndarray[double, ndim=2] curr = np.zeros((n, self.dims))

        for b in range(n_batch):

            i0 = b*self.bsz

            for i in prange(self.bsz, nogil=True):
            # for i in range(self.bsz):

                ii = i0 + i

                ## roughly O(nnz)
                check = self.recur(S, X[ii], ii, scl, temp, beta)
                # curr[ii] = self.recur(S, X[ii], ii, scl, temp, beta)

                for j in range(S.size[ii]):
                    for k in range(S.size[ii]):
                        self.dJ[S.value[ii][j]][S.value[ii][k]] += 1.0/self.bsz
                    for k in range(self.dimx):
                        self.dW[S.value[ii][j]][k] += X[ii][k]/self.bsz

            if check < 0:
                print("index error")

            self.learn()

        return S
        # return curr

    def get_weights(self, str name):

        cdef np.ndarray[double, ndim=2] weight

        if name == "J":
            weight = np.zeros((self.dims, self.dims))
            for i in range(self.dims):
                for j in range(self.dims):
                    weight[i][j] = self.J[i][j]

        elif name == "W":
            weight = np.zeros((self.dims, self.dimx))
            for i in range(self.dims):
                for j in range(self.dimx):
                    weight[i][j] = self.W[i][j]
        
        elif name == "dJ":
            weight = np.zeros((self.dims, self.dims))
            for i in range(self.dims):
                for j in range(self.dims):
                    weight[i][j] = self.dJ[i][j]
        
        elif name == "dW":
            weight = np.zeros((self.dims, self.dimx))
            for i in range(self.dims):
                for j in range(self.dimx):
                    weight[i][j] = self.dW[i][j]

        elif name == "R":
            weight = np.zeros((self.dims, self.dims))
            for i in range(self.dims):
                for j in range(self.dims):
                    weight[i][j] = self.R(i,j)

        else:
            raise ValueError("%s not a parameter"%name)

        return weight
    
    def get_offsets(self, str name):

        cdef np.ndarray[double] offset

        if name == 'h':
            offset = np.zeros(self.dims)
            for i in range(self.dims):
                offset[i] = self.h[i]

        elif name == 'r':
            offset = np.zeros(self.dims)
            for i in range(self.dims):
                offset[i] = self.r[i]
        
        else:
            raise ValueError("%s not a parameter"%name)

        return offset


###############################################################

cdef class GaussianSampler:
    """
    This stores the statistics of S in order to sample from
    the p(s|x,W) under the gaussian likelihood:
        p(x|s,W) ~ exp(-|x - Ws|^2)
    """

    cdef double **J             # Cov(s,s)
    cdef double **dJ            # accumulated update for J
    cdef double *r              # inhibition offset
    cdef double lr              # learning rate
    cdef double th              # inhib threshold
    cdef int dims               # latent dimension   
    cdef int bsz                # batch size

    def __cinit__(self, double[:,:] Jinit, int batch_size, double lr):
    
        cdef int i, j

        self.dims = Jinit.shape[0]
        self.bsz = batch_size
        self.lr = lr
        self.th = lr / batch_size

        # Allocate arrays
        self.J = <double **>malloc(self.dims * sizeof(double *))
        self.dJ = <double **>malloc(self.dims * sizeof(double *))
        self.r = <double *>malloc(self.dims * sizeof(double))

        if not self.J or not self.r or not self.dJ:
            raise MemoryError("Could not allocate memory")
        
        for i in range(self.dims): 
            self.J[i] = <double *>malloc(self.dims * sizeof(double))
            self.dJ[i] = <double *>malloc(self.dims * sizeof(double))

            if not self.J[i] or not self.dJ[i]:
                raise MemoryError("Could not allocate memory")

        ## Initialize arrays
        for i in range(self.dims):
            self.r[i] = 0.0
            for j in range(self.dims):
                self.J[i][j] = Jinit[i][j]
                self.r[i] += self.R0(i,j)
                self.dJ[i][j] = 0.0

    def __dealloc__(self):
        """Cleanup C memory"""
        cdef int i
        if self.J:
            for i in range(self.dims):
                if self.J[i]:
                    free(self.J[i])
            free(self.J)
        if self.dJ:
            for i in range(self.dims):
                if self.dJ[i]:
                    free(self.dJ[i])
            free(self.dJ)
        if self.r:
            free(self.r)

    cdef inline void learn(self) nogil:
        """
        Update shared parameters
        """
        cdef int i, j

        for i in prange(self.dims, nogil=True):
            self.r[i] = 0.0
            for j in range(self.dims):
                self.J[i][j] += self.lr*(self.dJ[i][j] - self.J[i][j])
                self.r[i] += self.R0(i,j)
                self.dJ[i][j] = 0.0

    cdef inline double R(self, int i, int j) nogil:
        """
        Compute value of R matrix at index (i,j)
        """

        cdef double A, B, C, D

        A = (1 - self.th)*self.J[i][j] 
        B = (1 - self.th)*self.J[i][i] - A
        C = self.J[j][j] - A
        D = 1 - A - B - C

        if A - cymin(B,C-self.th,D) < self.th:
            return 1.0
        if B - cymin(A,C,D-self.th) < self.th:
            return -1.0
        if C - cymin(A,B,D) < self.th:
            return -1.0
        if D - cymin(A,B,C) < self.th:
            return 1.0

    cdef inline double R0(self, int i, int j) nogil:
        """
        Compute constant value of R matrix at index (i,j)
        """

        cdef double A, B, C, D

        A = (1 - self.th)*self.J[i][j] 
        B = (1 - self.th)*self.J[i][i] - A
        C = self.J[j][j] - A
        D = 1 - A - B - C

        if B - cymin(A,C,D-self.th) < self.th:
            return 1.0
        if D - cymin(A,B,C) < self.th:
            return -1.0
        return 0.0

    cdef inline int recur(self, 
                          BiMat S, 
                          double[:] Wx, 
                          double[:,:] WtW, 
                          int i,
                          double temp,
                          double beta) nogil:

        cdef bint regularize = (beta > 1e-6)
        cdef int check
        cdef double dot, inhib, curr, prob
        cdef int j, k, l

        for j in range(self.dims):

            check = S.remove(i,j)

            dot = 0.5*WtW[j][j]
            inhib = beta * self.r[j]
            for l in range(S.size[i]):
                k = S.value[i][l]
                dot += WtW[j][k]

                if regularize:
                    inhib += beta * self.R(j, k)

            curr = (Wx[j] - dot - inhib) / temp
            prob = sigmoid(curr)

            if (prng.sample() < prob):
                check = S.add(i,j)

        return check

    def sample(self, BiMat S, 
                     double[:,:] WX,
                     double[:,:] WtW,
                     double temp, 
                     double beta):

        cdef int n = S.nrow
        cdef int n_batch = lround(ceil(n/self.bsz))

        cdef int check
        cdef bint regularize = (beta > 1e-6)
        
        cdef int b, i0, i, ii, j, k

        for b in range(n_batch):

            i0 = b*self.bsz

            for i in prange(self.bsz, nogil=True):
            # for i in range(self.bsz):

                ii = i0 + i

                ## roughly O(nnz)
                check = self.recur(S, WX[ii], WtW, ii, temp, beta)

                for j in range(S.size[ii]):
                    for k in range(S.size[ii]):
                        self.dJ[S.value[ii][j]][S.value[ii][k]] += 1.0/self.bsz

            if check < 0:
                print("index error")

            self.learn()

        return S

    def get_weights(self, str name):

        cdef np.ndarray[double, ndim=2] weight

        if name == "J":
            weight = np.zeros((self.dims, self.dims))
            for i in range(self.dims):
                for j in range(self.dims):
                    weight[i][j] = self.J[i][j]
        
        elif name == "dJ":
            weight = np.zeros((self.dims, self.dims))
            for i in range(self.dims):
                for j in range(self.dims):
                    weight[i][j] = self.dJ[i][j]
        
        elif name == "R":
            weight = np.zeros((self.dims, self.dims))
            for i in range(self.dims):
                for j in range(self.dims):
                    weight[i][j] = self.R(i,j)

        else:
            raise ValueError("%s not a parameter"%name)

        return weight
    
    def get_offsets(self, str name):

        cdef np.ndarray[double] offset

        if name == 'r':
            offset = np.zeros(self.dims)
            for i in range(self.dims):
                offset[i] = self.r[i]
        
        else:
            raise ValueError("%s not a parameter"%name)

        return offset
