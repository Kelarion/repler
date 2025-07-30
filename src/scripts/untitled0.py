
Ssp = sprs.csr_array(df_util.randtree_feats(2**6, 2, 4))
S_ = Ssp.todense()
X_ = util.pca_reduce(Ssp.todense().T, thrs=1)
X = df_util.noisyembed(X_, 2*X_.shape[1], 30, nonneg=False)
X = X - X.mean(0)

# X *= np.sqrt(np.prod(X.shape)/np.sum(X**2))

S0 = (S_ + np.random.choice([0,1], p=[0.99,0.01], size=S_.shape))%2

#%%

S = 1*S0
StX = S.T@X
StS = S.T@S
N = len(S)
scl = 1  
temp = 1e-5
alpha = 0.0
# beta = 1e-2
beta = 0

n, m = S.shape
n2, d = X.shape

regularize = (beta > 1e-6)

C1 = np.zeros(S.shape)
sign = 2*(np.diag(StS) < N//2)-1 # which concepts are flipped

t = (N-1)/N

#%%

# for i in np.random.permutation(np.arange(n)):
for i in np.arange(n):

    # for j in np.random.permutation(np.arange(m)):
    for j in range(m): # concept
        Sij = S[i,j]
        S_j = (StS[j,j] - Sij)/(N-1)

        ## Inputs
        inp = 0    
        # for k in range(d):
        #     inp += (2*StX[j,k]*X[i,k] + (1-2*Sij)*X[i,k]**2)/t

        ## Recurrence
        # dot = 0
        dot = t*(2*(N-2)*S_j*(1-S_j) + 1)*(1/2 - Sij)
        inhib = 0.0
        for k in range(m):                        
            Sik = S[i,k] 
            S_k = (StS[k,k]-Sik)/(N-1)
            
            dot += 2*(StS[j,k] - Sij*Sik + t*(1/2 - S_j - S_k - (N-2)*S_j*S_k))*(Sik - S_k)
            dot -= t*(1-S_k)*S_k*(2*S_j-1)

            if regularize:
                A = StS[j,k] - Sij*Sik
                B = StS[j,j] - A - Sij
                C = StS[k,k] - A
                D = N - A - B - C
                
                # Simple conditional assignment
                if A < min(B,C-1,D):
                    inhib += Sik
                if B < min(A,C,D-1):
                    inhib += (1 - Sik) 
                if C <= min(A,B,D):
                    inhib -= Sik
                if D <= min(A,B,C):
                    inhib -= (1 - Sik)

        ## Compute currents
        # curr = (scl*inp - (scl**2)*dot - beta*inhib)/temp
        # curr = (scl*(inp - scl*dot)/N - beta*inhib)/temp
        curr = ((inp - scl*dot)/(N-1) - beta*inhib)/temp
        # curr = ((inp/scl - dot)/N - beta*inhib)/temp

        C1[i,j] = temp*curr

        # ## Apply sigmoid (overflow robust)
        # if curr < -100:
        #     prob = 0.0
        # elif curr > 100:
        #     prob = 1.0
        # else:
        #     prob = 1.0 / (1.0 + math.exp(-curr))
        
        # ## Update outputs
        # ds = (np.random.rand() < prob) - Sij
        # S[i,j] += ds

        # StS[j,j] += ds
        # for k in range(m):
        #     if k != j:
        #         StS[j,k] += S[i,k]*ds
        #         StS[k,j] += S[i,k]*ds

        # for k in range(d):
        #     StX[j,k] += X[i,k]*ds

#%%

S = 1*S0
sign = np.random.choice([-1,1], size=len(S.T))
S[:,sign<0] = 1-S[:,sign<0]
StX = S.T@X
StS = S.T@S

n, m = S.shape
n2, d = X.shape

regularize = (beta > 1e-6)

C2 = np.zeros(S.shape)
sign = 2*(np.diag(StS) < N//2)-1 # which concepts are flipped

t = (N-1)/N

#%%

# for i in np.random.permutation(np.arange(n)):
for i in np.arange(n):

    # for j in np.random.permutation(np.arange(m)):
    for j in range(m): 
        Sij = S[i,j]
        S_j = (StS[j,j] - Sij)/(N-1)

        if sign[j] < 0: # take care of sign flips
            S_j = 1 - S_j  # only flip the mean

        ## Inputs
        inp = 0 
        # for k in range(d):
        #     inp += sign[j]*(2*StX[j,k]*X[i,k] + (1 - 2*Sij)*X[i,k]**2)/(t*(N-1))

        ## Recurrence
        dot = 0
        dot = sign[j]*((N-2)*S_j*(1-S_j) + 1/2)*(1 - 2*Sij) / N
        inhib = 0.0
        for k in range(m):                    
            Sik = 1*S[i,k]
            S_k = (StS[k,k] - Sik)/(N-1)
            SjSk = (StS[j,k] - Sij*Sik)/(N-1) # raw second moment

            if sign[j] < 0: # the order matters here
                SjSk = S_k - SjSk  
            if sign[k] < 0:
                Sik = 1 - Sik # we can safely flip this
                S_k = 1 - S_k
                SjSk = S_j - SjSk 

            dot += 2*(SjSk + (1/2 - S_j - S_k - (N-2)*S_j*S_k)/N)*(Sik - S_k)
            dot -= (2*S_j-1)*(1-S_k)*S_k / N

            if regularize:
                A = SjSk
                B = S_j - SjSk
                C = S_k + Sik/(N-1) - SjSk
                
                if A < min(B,C - 1/(N-1)):
                    inhib += Sik
                if B < min(A,C):
                    inhib += (1 - Sik) 
                if C <= min(A,B):
                    inhib -= Sik

        ## Compute currents
        # curr = (scl*inp - (scl**2)*dot - beta*inhib)/temp
        # curr = (scl*(inp - scl*dot)/N - beta*inhib)/temp
        # curr = ((inp - scl*dot)/N - beta*inhib - alpha)/temp
        curr = sign[j]*(inp - scl*dot - beta*inhib - alpha)/temp
        # curr = ((inp/scl - dot)/N - beta*inhib)/temp

        C2[i,j] = sign[j]*temp*curr

        # ## Apply sigmoid (overflow robust)
        # if curr < -100:
        #     prob = 0.0
        # elif curr > 100:
        #     prob = 1.0
        # else:
        #     prob = 1.0 / (1.0 + math.exp(-curr))
        
        ## Update outputs
        # ds = (np.random.rand() < prob) - Sij 
        # S[i,j] += ds    # this is why we couldn't flip Sij

        # ## Update covariance matrices
        # StS[j,j] += ds
        # for k in range(m):
        #     if k != j:
        #         StS[j,k] += S[i,k]*ds
        #         StS[k,j] += S[i,k]*ds

        # for k in range(d):
        #     StX[j,k] += X[i,k]*ds
            
        # sign[j] = 2*(StS[j,j] < N//2) - 1
            
        
        