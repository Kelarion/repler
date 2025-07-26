cimport numpy as np

np.import_array()

ctypedef np.npy_uint32 UINT32_t

##################################################################

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
        
    cdef inline bint contains(self, int i) nogil

    cdef inline void add(self, int i) nogil

    cdef inline void remove(self, int i) nogil
   
cdef class BiMat:
    """
    A format for sparse binary matrices, initialized from csr format

    Designed for O(1) bit flipping and O(nnz) iteration over rows, but
    uses the same memory (more, actually) as a dense array. Just a 2d
    extension of the FiniteSet format suggested by Jerome.
    """

    cdef:
        int **index    # Index of element in value array
        int **value    # Array of contained elements
        int *size      # Current number of elements per row
        int nrow
        int ncol

    cdef inline void add(self, int i, int j) nogil

    cdef inline void remove(self, int i, int j) nogil


cdef class XORRNG:
    """
    Custom XORRNG sampler that I copied from the scikit-learn source code
    """

    cdef UINT32_t state

    cdef inline double sample(self) nogil


cdef inline double cymin(double a, double b, double c) nogil


cdef inline double sigmoid(double curr) nogil

#############################################