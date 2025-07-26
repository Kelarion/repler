def spkerbmf(X: np.ndarray,
             S: list,                # list of index lists
             StX: np.ndarray, 
             StS: np.ndarray,
             scl: float,  
             temp: float,
             beta: float = 0.0):
    """
    Kernel matching search function for binary autoencoders
    """

    m = len(StS)
    n, d = X.shape

    regularize = (beta > 1e-6)

    h = np.dot(StS, np.diag(StS)) - np.diag(StS)*np.sum(np.diag(StS)**2)

    # for i in np.random.permutation(np.arange(n)):
    for i in np.arange(n):

        # for j in np.random.permutation(np.arange(m)):
        for j in range(m): # concept
            S_j = StS[j,j]

            ## Inputs
            inp = 0    
            for k in range(d):
                inp += 2*StX[j,k]*X[i,k]

            if j in S[i]:
                S[i].remove(j)

            ## Recurrence
            dot = S_j*(1-S_j) - h[j]

            inhib = 0.0
            for k in S[i]:
                # Sik = S[i,k] 
                S_k = StS[k,k]

                dot += 2*(StS[j,k] - S_j*S_k)

                if regularize:
                    A = StS[j,k] 
                    B = S_j - A
                    C = S_k - A
                    
                    # Simple conditional assignment
                    if A < min(B,C):
                        inhib += Sik
                    if B < min(A,C):
                        inhib -= Sik
                    if C < min(A,B):
                        inhib -= Sik

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
            if np.random.rand() < prob:
                S[i].append(j)
    
    return S