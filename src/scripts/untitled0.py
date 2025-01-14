import numpy as np
import scipy.linalg as la

#%%

def expmap(x, v, t):
    """
    Exponential map on the sphere; at point x, with velocity v, at time t
    """
    vnrm = la.norm(v, axis=0)
    return np.cos(t*vnrm)*x + np.sin(t*vnrm)*v/vnrm


#%%
d = 256
n = 512
steps = 200
temp = 1
lr = 1e-2

x = np.random.randn(d,n)/np.sqrt(d)
x = x/la.norm(x,axis=0,keepdims=True)

margin = []
for it in tqdm(range(steps)):
    
    margin.append(np.max((x.T@x - 2*np.eye(n)),1))
    
    ## get gradient (i.e. nearest neighbor)
    idx = np.argmax(x.T@x - 2*np.eye(n), axis=1)
    dX = -x[:,idx]
    
    ## Project onto tangent plane 
    projX = dX - (dX*x).sum(0)*x
    
    ## Exponential map
    x = expmap(x,projX,lr)
    
