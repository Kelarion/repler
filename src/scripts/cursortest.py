from numba import njit
import numpy as np


@njit
def newbmf(X: np.ndarray, 
           S: np.ndarray, 
           W: np.ndarray, 
           StS: np.ndarray, 
           N: int, 
           temp: float,
           beta: float = 0.0,
           alpha: float = 0.0):
    
    n,m = S.shape
    WtW = np.dot(W.T, W)
    XW = np.dot(X, W)
    St1 = np.diag(StS)
    
    regularize = (beta > 1e-6)

    # Pre-allocate arrays if beta > 1e-6 to avoid repeated allocation
    if regularize:
        R = np.empty_like(StS)
        r = np.empty(m, dtype=StS.dtype)

    # for i in np.random.permutation(np.arange(n)):
    for i in range(n):
        if regularize:
            St1 -= S[i]
            StS -= np.outer(S[i],S[i])
            
            # Direct element-wise computation - Numba will optimize this
            for j1 in range(m):
                r[j1] = 0
                for j2 in range(m):
                    D1 = StS[j1,j2] 
                    D2 = St1[j2] - StS[j1,j2] 
                    D3 = St1[j1] - StS[j1,j2]
                    D4 = (N-1) - St1[j1] - St1[j2] + StS[j1,j2]
                    
                    # Simple conditional assignment
                    if D1 < min(D2, D3, D4):
                        R[j1,j2] = 1
                    elif D2 < min(D1,D2,D3):
                        R[j1,j2] = -1
                        r[j1] += 1
                    elif D3 < min(D1,D2,D4):
                        R[j1,j2] = -1
                        r[j1] -= 1
                    elif D4 < min(D1,D2,D3):
                        R[j1,j2] = 1

        # for j in np.random.permutation(np.arange(m)):
        for j in range(m):
            dot = 0
            for k in range(m):
                dot += WtW[j,k] * S[i,k]
            dot += (0.5 - S[i,j])*WtW[j,j]
            
            if beta > 1e-6:
                inhib = r[j]
                for k in range(m):
                    inhib += R[j,k] * S[i,k]

            curr = (XW[i,j] - beta*inhib - dot - alpha)/temp

            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            ds = (np.random.rand() < prob) - S[i,j]
            
            if regularize:
                St1[j] += ds
                StS[j] += S[i]*ds

            S[i,j] += ds
        
    return S