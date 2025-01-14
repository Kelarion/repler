

class NicePolytope:
    """
    Polytope defined by underdetermined linear equalities., i.e.
    
    {x | Ax = b, x >= 0}
    
    Allows for easily moving between adjacent vertices.
    
    """
    
    def __init__(self, A_eq, b_eq):
        
        self.A = A_eq
        self.b = b_eq
        
        self.n_con, self.n_dim = self.A.shape
        
        self.zero_tol = 1e-12
    
    def adj(self, x, c=None):
        """
        Find a vertex adjacent to x. If a linear objective c is provided, will
        return a vertex which most increases or least decreases c. Otherwise,
        just uses x itself as the objective.
        """
        
        if c is None:
            c = x/la.norm(x)
        
        B_ids = np.where(x > self.zero_tol)[0].tolist() # basic variables
        N_ids = np.where(x <= self.zero_tol)[0].tolist() # non-basic varaibles
        
        B = self.A[:,B_ids] 
        N = self.A[:,N_ids]
        
        exes = np.repeat(x[:,None], len(N_ids), axis=1) 
        
        ### Edge difference vectors
        d = -la.pinv(B)@N
        diffs = np.zeros((self.n_dim, self.n_con))
        diffs[B_ids,:] = d
        diffs += np.eye(self.n_dim)[:,N_ids]
        
        ### Get step sizes
        ratios = x[B_ids,None]/np.where(-d >= 1e-6, -d, np.nan)
        step = np.nanmin(ratios, axis=0)[None,:]
        
        ### Check feasibility
        new_exes = exes + step*diffs
        feas = (np.abs(self.A@new_exes - self.b[:,None]).max(0) <= self.zero_tol) & (new_exes.min(0) >= -self.zero_tol)
        
        ### Pick a feasible solution
        which_one = np.argmax(c@new_exes)
        
        return new_exes[:,which_one]
        
        
        # ### Find 'entering' variable
        # z = c[B_ids]@la.pinv(B)@N - c[N_ids] # projection of c onto edges
        # newguy = N_ids[np.argmax(z)]
        
        # ### Find 'leaving' variable
        # n = self.A[:,newguy]
        # Q = la.inv(B)@n
        # ratio = x[B_ids]/np.where(Q >= self.zero_tol, Q, np.nan)
        # oldguy = B_ids[np.nanargmin(ratio)]
        
        # ### Swap
        # new_B_ids = list((set(B_ids) - set([oldguy])).union(set([newguy])))
        # new_N_ids = list((set(N_ids) - set([newguy])).union(set([oldguy])))
        
        # ### Compute new vertex 
        # x_B = la.inv(self.A[:,new_B_ids])@self.b
        
        # ### Reshuffle to match original variable ordering
        # new_x = np.concatenate([x_B, np.zeros(len(N_ids))]) 
        # perm = np.argsort(np.concatenate([new_B_ids, new_N_ids])) 
        
        # return new_x[perm]
        
