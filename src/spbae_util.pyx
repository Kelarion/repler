cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport malloc, free

np.import_array()

ctypedef np.int32_t int32
cdef inline UINT32_t DEFAULT_SEED = 1
cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647

##################################################################

cdef class FiniteSet:
    """
    Custom unordered set with O(1) add/remove methods
    which takes advantage of the fixed maximum set size

    Many thanks to Jerome Richard on stackoverflow for the suggestion
    """

    def __cinit__(self, int[:] indices, int maximum):
        """C-level initialization - fastest"""
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
    A format for sparse binary matrices, initialized from csr format

    Designed for O(1) bit flipping and O(nnz) iteration over rows, but
    uses the same memory (more, actually) as a dense array. Just a 2d
    extension of the FiniteSet format suggested by Jerome.
    """

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

    cdef inline void add(self, int i, int j) nogil:
        """
        Add element to set
        """
        if self.index[i][j] < 0:
            self.value[i][self.size[i]] = j
            self.index[i][j] = self.size[i]
            self.size[i] += 1

    cdef inline void remove(self, int i, int j) nogil:
        """
        Remove element from set
        """
        cdef int remove_idx, last_idx

        if self.index[i][j] >= 0:
            remove_idx = self.index[i][j]
            last_idx = self.size[i] - 1

            if remove_idx < last_idx: # don't swap if last element
                self.value[i][remove_idx] = self.value[i][last_idx]
                self.index[i][self.value[i][last_idx]] = remove_idx

            self.index[i][j] = -1
            self.size[i] -= 1


cdef class XORRNG:
    """
    Custom XORRNG sampler that I copied from the scikit-learn source code
    """

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

###############################################################
