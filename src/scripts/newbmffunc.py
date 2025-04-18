from numba import njit
import numpy as np
import math


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
    
    regularize = (beta > 1e-6)

    # for i in np.random.permutation(np.arange(n)):
    for i in range(n):

        # for j in np.random.permutation(np.arange(m)):
        for j in range(m):
            
            Sij = S[i,j]
            
            dot = (0.5 - Sij)*WtW[j,j]
            inhib = 0
            for k in range(m):
                Sik = S[i,k]
                
                dot += WtW[j,k] * Sik
                
                if regularize:
                    A = StS[j,k] - Sij*Sik
                    B = StS[k,k] - A - Sij
                    C = StS[j,j] - A
                    D = n - A - B - C
                    
                    # Simple conditional assignment
                    if (A < min(B, C, D)) or (D < min(A,B,C)):
                        inhib += Sik
                    elif (B < min(A,B,D)):
                        inhib += 1 - Sik
                    elif C < min(A,B,D):
                        inhib -= 1 + Sik
                    
            curr = (XW[i,j] - beta*inhib - dot - alpha)/temp

            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))

            ds = (np.random.rand() < prob) - Sij

            if regularize:
                for k in range(m):
                    StS[j,k] += S[i,k]*ds

            S[i,j] += ds
        
    return S

