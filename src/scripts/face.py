import numpy as np

#%%

def vec(A, incl_diag=False):
    return A[..., *np.triu_indices(A.shape[-1], 1-incl_diag)]

def outers(V):
    return np.einsum('...ik,...jk->...kij', V, V)

def cube(n):
    return np.mod(np.arange(2**int(n))[:,None]//(2**np.arange(int(n))[None,:]),2)

def yuke(X, Y=None):

    if Y is None:
        Y = X

    Xnrm = (X**2).sum(-1,keepdims=True)
    Ynrm = (Y**2).sum(-1,keepdims=True).swapaxes(-1,-2)
    XYdot = np.einsum('...ik,...jk->...ij',X,Y)

    return Xnrm + Ynrm - 2*XYdot

#%% A non-staircase face for the boolean quadric

## A well-graded family, whose graph happens to be a tree
tree = np.array([[0,0,0,0,0,0],
                 [1,0,0,0,0,0],
                 [1,1,0,0,0,0],
                 [1,0,1,0,0,0],
                 [0,0,0,1,0,0],
                 [0,0,0,0,1,0],
                 [0,0,0,0,0,1]])

## Quadratic form
J = np.array([[ 0.,  1.,  1., -1., -1., -1.],
              [ 1., -2., -1.,  0.,  0.,  0.],
              [ 1., -1., -2.,  0.,  0.,  0.],
              [-1.,  0.,  0.,  0., -1., -1.],
              [-1.,  0.,  0., -1.,  0., -1.],
              [-1.,  0.,  0., -1., -1.,  0.]])

## vectorized version, diagonal counts for half
vecJ = vec(J - np.diag(np.diag(J))/ 2, True)

## Boolean quadric vertices
BQ6 = vec(outers(cube(6).T), True)

## Check that this is a valid inequality, with given roots
isroot = (yuke(tree, cube(6)).min(0) == 0)

assert np.all(BQ6[~isroot]@vecJ < 0)
assert np.all(BQ6[isroot]@vecJ == 0)

#%% Check that it is also a face of the cut polytope

CUT7 = np.hstack([vec(outers(2*cube(6).T - 1)), 2*cube(6) - 1])

## This is one way to make a valid inequality, which isn't pure but
## nevertheless the face may still be pure-exposable
vecJ = np.concatenate([vec(J), J.sum(0)])
j0 = - (J.sum()/2 + np.diag(J).sum()/2)

assert np.all(CUT7[~isroot]@vecJ < j0)
assert np.all(CUT7[isroot]@vecJ == j0)