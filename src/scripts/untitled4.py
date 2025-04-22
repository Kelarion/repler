
bits = 4
beta = 1e-3

# S = util.F2(bits)
Strue = df_util.btree_feats(bits).T
W = sts.ortho_group.rvs(Strue.shape[1])
X = (Strue-Strue.mean(0))@W.T
b = -Strue.mean(0)@W.T

S = (Strue + np.random.choice([0,1], Strue.shape, p=[0.9,0.1]))%2
N = len(S)

StS = S.T@S
StX = S.T@X
scl = 1
temp = 1e-6
steps = 1

n, m = S.shape


St1 = 1*np.diag(StS)

en = np.zeros((3,n,m))
# for i in np.random.permutation(np.arange(n)):
for i in np.arange(n):

    ## Pick current item
    t = (N-1)/N

    x = X[i]
    s = S[i]

    ## Subtract current item
    St1 -= s
    StS -= np.outer(s,s)
    StX -= np.outer(s,x)

    # Compute the rank-one terms
    s_ = St1/(N-1)
    u = 2*s_ - 1

    s_sc_ = s_*(1-s_)

    ## Organize states
    xtx = np.sum(x**2)
    Sk = (np.dot(StX, x) + s_*xtx)/t
    k0 = xtx/(t**2)

    if beta > 1e-6:
        ## Regularization (more verbose because of numba reasons)
        D1 = StS
        D2 = St1[None,:] - StS
        D3 = St1[:,None] - StS
        D4 = (N-1) - St1[None,:] - St1[:,None] + StS

        best1 = 1*(D1<D2)*(D1<D3)*(D1<D4)
        best2 = 1*(D2<D1)*(D2<D3)*(D2<D4)
        best3 = 1*(D3<D2)*(D3<D1)*(D3<D4)
        best4 = 1*(D4<D2)*(D4<D3)*(D4<D1)

        R = (best1 - best2 - best3 + best4)*1.0
        r = (best2.sum(0) - best4.sum(0))*1.0
                        

    ## Constants
    # sx = Sx.sum()
    sx = np.dot(s_, s - s_)
    ux = 2*sx - s.sum() + s_.sum()
    
    # Form the threshold 
    h = t*((scl**2)*s_sc_.sum() - scl*k0)*u + 2*scl*Sk 
    
    # Need to subtract the diagonal and add it back in
    Jii = 2*(N-1)*s_sc_ + t*u**2

    ## Hopfield update of s
    news = 1*s
    for step in range(steps):
        # for j in np.random.permutation(np.arange(m)):
        for j in range(m): # concept

            en[0,i,j] = 2*Sk[j] - t*k0*u[j]
            # rows = ridx[rptr[j]:rptr[j+1]]

            # Compute sparse dot product
            dot = 2*np.dot(StS[j], news - s_)
            dot -= 2*(N-1)*s_[j]*sx
            dot += t*u[j]*ux
            dot -= Jii[j]*news[j]
            if beta > 1e-6:
                dot += beta*(np.dot(R[j], news) + r[j])
                en[1,i,j] = np.dot(R[j], news) + r[j]

            thisS = 1*S
            thisS[i,j] = 1
            en[2,i,j] += df_util.treecorr(thisS).sum()/2
            thisS[i,j] = 0 
            en[2,i,j] -= df_util.treecorr(thisS).sum()/2

            ## Compute currents
            curr = (h[j] - (scl**2)*Jii[j]/2 - (scl**2)*dot)/temp

            # en[2,i,j] = curr

            ## Apply sigmoid (overflow robust)
            if curr < -100:
                prob = 0.0
            elif curr > 100:
                prob = 1.0
            else:
                prob = 1.0 / (1.0 + np.exp(-curr))
            
            ## Update outputs
            sj = 1*(np.random.rand() < prob)
            ds = sj - news[j]
            news[j] = sj
            
            ## Update dot products
            if np.abs(ds) > 0:
                sx += ds*s_[j]
                ux += ds*u[j]

    ## Update 
    S[i] = news
    St1 += news
    StS += np.outer(news, news)
    StX += np.outer(news, x)
