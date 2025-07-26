
@njit
def kerbae(X: np.ndarray,
           S: np.ndarray, 
           StX: np.ndarray, 
           StS: np.ndarray,
           scl: float,  
           temp: float,
           beta: float = 0.0):
    """
    Kernel matching search function for binary autoencoders
    """

    n, m = S.shape
    n2, d = X.shape

    regularize = (beta > 1e-6)

    assert n == n2

    for i in np.random.permutation(np.arange(n)):
    # for i in np.arange(n):

        for j in np.random.permutation(np.arange(m)):
        # for j in range(m): # concept
            Sij = S[i,j]
            S_j = StS[j,j]

            ## Inputs
            inp = 0    
            for k in range(d):
                inp += 2*StX[j,k]*X[i,k]
            
            ## Recurrence
            dot = S_j*(1-S_j)*(1 - 2*Sij)
            inhib = 0.0
            for k in range(m):                        
                Sik = S[i,k] 
                S_k = StS[k,k]
                
                dot += 2*(StS[j,k] - S_j*S_k)*(Sik - S_k)

                if regularize:
                    A = StS[j,k] 
                    B = S_j - A
                    C = S_k - A
                    D = 1 - A - B - C
                    
                    # Simple conditional assignment
                    if A < min(B,C,D):
                        inhib += Sik
                    if B < min(A,C,D):
                        inhib += (1 - Sik) 
                    if C < min(A,B,D):
                        inhib -= Sik
                    if D < min(A,B,C):
                        inhib -= (1 - Sik)

            ## Compute currents
            curr = ((inp - scl*dot) - beta*inhib)/temp

            ## Apply sigmoid (overflow robust)
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + math.exp(-curr))
            
            ## Update outputs
            S[i,j] = 1*(np.random.rand() < prob)
    
    return S