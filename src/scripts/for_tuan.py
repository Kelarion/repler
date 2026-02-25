import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg as la 
import scipy.special as spc

#%% My kernel

class RadialBoyfriend:
    """
    My special normalised (von) mises kernel
    
    a mises function which has zero integral and k(0) = 1
    """
    
    def __init__(self, width, center=True):
        
        self.kap = 0.5/width
        ## Set the scale so that the average variance of each neuron is 1
        ## and shift so the mean population response across stimuli is 0
        self.scale = 1/(np.exp(self.kap) - spc.i0(self.kap))
        if center:
            self.shift = spc.i0(self.kap)
        else:
            self.shift = 0
        
    def __call__(self, error, quantile=1e-4):
        """
        compute k(x,y) = k(x-y) ... so input x-y
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return (np.exp(self.kap*np.cos(error)) - self.shift)/denom

    def curv(self, x):
        """
        Second derivative
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return self.kap*np.exp(self.kap*np.cos(x))*(self.kap*np.sin(x)**2 - np.cos(x))/denom

    def deriv(self, x):
        """
        Derivative
        """
        denom = (np.exp(self.kap) - spc.i0(self.kap))
        return -self.kap*np.sin(x)*np.exp(self.kap*np.cos(x))/denom

    def perturb(self, x, y):
        """
        The 'perturbation kernel', i.e. (f(x)-f(0))*(f(y)-f(0))
        """
        numer = self(x-y) - self(x) - self(y) + 1
        denom = 2*np.sqrt((1-self(x))*(1-self(y)))
        return numer/denom

    def sample(self, colors, size=1):
        """
        Sample activity in response to colors
        """
        K = self(colors[None] - colors[:,None])
        mu = np.zeros(len(colors))
        return np.random.multivariate_normal(mu, K, size=size).T

#%% Activation functions

class softplus:
    
    def __init__(self, beta=2, axis=None):
        self.beta = beta
    
    def __call__(self,x):
        return np.log(1 + np.exp(self.beta*x))/self.beta
    
    def inv(self,x):
        return np.log(np.exp(self.beta*x)-1)/self.beta
    
class erf:
    
    def __init__(self, beta=2):
        self.beta = beta
    
    def __call__(self,x):
        return 1 + spc.erf(self.beta*np.sqrt(np.pi)*x/2)
    
    def inv(self,x):
        return spc.erfinv(x-1)*2/(np.sqrt(np.pi)*self.beta)

#%% 

def clark_ring(Phi, X, lam=1e-5):
    """
    David's construction of a heterogeneously-tuned approximate ring attractor
    """    
    
    N_theta, N = Phi.shape    

    K = Phi@Phi.T/N + lam*N_theta*np.eye(N_theta)/N
    Omg = la.pinv(Phi.T@Phi/N_theta + lam*np.eye(N))
    
    Junc = X.T@la.pinv(K)@Phi / N
    
    return Junc - np.diag(np.diag(Junc)/np.diag(Omg))*Omg


def rnn(J, x0, T, dt, phi, c=0):
    """
    Simulate an rnn up to time T with interval dt
    
    uses david's stabilization method, assumes that the average of phi(x)
    over all the neurons is 1
    """
    
    x = 1*x0
    xs = [x0]
    for _ in range(T):
        p = phi(x)
        x = (1-dt)*x + dt*(J@p - c*(p.mean(0) - 1))
        
        xs.append(1*x)
    
    return np.array(xs)

#%%

N = 1000
N_theta = 100
width = 0.1
act = 'soft' # 'erf'

theta = np.linspace(-np.pi, np.pi, N_theta, endpoint=False)
diff = theta[None] - theta[:,None]

kern = RadialBoyfriend(width)
X = kern.sample(theta, size=N)

if act == 'soft':
    phi = softplus(2)
    Phi = phi(X)
    Phi = Phi / Phi.mean(1, keepdims=True)
    c = 1
    
else:
    phi = erf(2.76)
    Phi = phi(X)
    c = 0

X = phi.inv(Phi)

# J = np.sqrt(6)*np.cos(diff)/N_theta
J = clark_ring(Phi, X, lam=1e-6)

x0 = np.random.randn(len(J), 200) + X.mean(0)[:,None]
# x0 = X.T + np.random.randn(*X.T.shape)

X_t = rnn(J, x0, 1000, 0.05, phi=phi, c=c)

#%% PCA 

U, S, _ = la.svd(Phi.T-Phi.T.mean(1)[:,None], full_matrices=False)
pcs = Phi.T.T@U[:,:3]

plt.figure()
ax = plt.subplot(111, projection='3d')
scat = ax.scatter3D(pcs[:,0],pcs[:,1],pcs[:,2], c='k')

pcs = phi(X_t[-1]).T@U[:,:3]
scat = ax.scatter3D(pcs[:,0],pcs[:,1],pcs[:,2])

#%%

plt.figure()
dot = np.einsum('ijk,lj->ilk', phi(X_t), Phi)
dist = (Phi**2).sum(1,keepdims=True) + (phi(X_t)**2).sum(1,keepdims=True) - 2*dot

plt.plot(dist.min(1))
