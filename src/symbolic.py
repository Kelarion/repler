import sympy as sp
import numpy as np
import string
from sympy import IndexedBase, symbols, expand, collect, Sum, factor_terms, Poly, Idx, Wild
from sympy.printing.pycode import pycode 

#########

wi, wj, wk = symbols('i j k', cls=Wild)
i,j,k,l = symbols('i j k l', integer=True, positive=True)
n, m = symbols('n m')
S = IndexedBase('S')  
X = IndexedBase('X')  
s = sp.IndexedBase('s')
x = sp.IndexedBase('x')
s_ = sp.IndexedBase('s_')
x_ = sp.IndexedBase('x_')

##########

def free_idx(idx_expr):
    """
    Rename an implicit summation to a particular moment variable
    """

    free = sp.get_indices(idx_expr)[0]
    these_idx = list(free - set([wk]))

    return these_idx

#########

## Give names to each empirical moment, and its associated indexed formula
names = {sp.IndexedBase('s_'): S[wi,wk],
         sp.IndexedBase('x_'): X[wi,wk],
         sp.IndexedBase('C'): S[wi,wk]*S[wj,wk], 
         sp.IndexedBase('W'): S[wi,wk]*X[wj,wk]}

bnds = {'': (1,n-1), 
        'tst': (1,m),
        'trn': (m+1, n-1)}

subs_dict = {}
eval_dict = {}
for name, expr in names.items():
    varname = name[*free_idx(expr)]
    for subname, lims in bnds.items():
        sum_expr = sp.Sum(expr, (wk,)+lims)
        size = sp.Sum(1, (wk,)+lims).doit()
        this_name = varname.subs(str(name), str(name)+subname) 

        subs_dict[sum_expr] = size*this_name
        eval_dict[name.subs(str(name), str(name)+subname)] = (expr, lims)



## Defining the possible summations that I want to use
new_nrm = ( (S[i,k] - s_[i] - (s[i]-s_[i])/n)*(S[i,l] - s_[i] - (s[i]-s_[i])/n)
            *(S[j,k] - s_[j] - (s[j]-s_[j])/n)*(S[j,l] - s_[j] - (s[j]-s_[j])/n) )
old_nrm = ( (S[i,k] - s_[i])*(S[i,l] - s_[i])
            *(S[j,k] - s_[j])*(S[j,l] - s_[j]) )

# new_dot = ( (S[i,k] - s_[i] - (s[i]-s_[i])/n)*(S[i,l] - s_[i] - (s[i]-s_[i])/n)
#             *(X[j,k] - x_[j] - (x[j]-x_[j])/n)*(X[j,l] - x_[j] - (x[j]-x_[j])/n) )
# old_dot = ( (S[i,k] - s_[i])*(S[i,l] - s_[i])
#             *(X[j,k] - x_[j])*(X[j,l] - x_[j]) )
new_dot = ( (S[i,k] - s_[i] - (s[i]-s_[i])/n)*(S[i,l] - s_[i] - (s[i]-s_[i])/n)
            *X[j,k]*X[j,l] )
old_dot = ( (S[i,k] - s_[i])*(S[i,l] - s_[i])
            *X[j,k]*X[j,l] )


class Contraction:
    """
    The point of this class is to take a summation over pairs of items and
    convert it to a summation over pairs of features. e.g. if you have:
    ____
    \\
     \\   A_{ik}*B_{il}*A_{jk}*B_{jl} 
     //   
    //__{ijkl}

    then I want to convert this to a sum over just `i` and `j`, usually by 
    gathering certain sums over `k or `l` into pre-defined moments. It can
    be a big pain when there are enough terms floating around.

    I feel bad about the things I had to do when writing this class, I hope
    you can forgive me.
    """
    
    def __init__(self, summand, k_bounds=(1,'n-1'), l_bounds=(1,'n-1'), verbose=True):
        
        self.k_bounds = k_bounds
        self.l_bounds = l_bounds
        
        n = symbols('n')      # the only bound that all equations share
        
        ## Remember the ranges of each dummy variable, 
        ## and store functions for computing them numerically
        k_range = sp.Sum(1, (wk,)+k_bounds)
        l_range = sp.Sum(1, (wk,)+l_bounds)

        self.sizes = list(set([n]).union(k_range.free_symbols).union(l_range.free_symbols))

        self.k_sum = k_range.doit()
        self.l_sum = l_range.doit()

        ## Simplify symbolic expression
        if verbose:
            print('Rearranging sums')

        ## Normalize to remove bounding variables
        denom = self.k_sum*self.l_sum

        ## Rearrange the summation
        expr = expand(summand/denom)
        
        summed = sp.Sum(expr, (k,)+self.k_bounds)
        
        summed = expand(factor_terms(expand(summed))).doit()
        full_sum = expand(factor_terms(expand(sp.Sum(summed, (l,)+self.l_bounds)))).doit()

        result = full_sum
        
        ## Make substitutions according to the predefined dictionary
        for pre,post in subs_dict.items():
            result = result.replace(pre, post)

        ## Apply identities 
        result = result.replace(S[wi,n], s[wi])
        result = result.replace(X[wi,n], x[wi])
        result = result.replace(s[wi]**2, s[wi])

        self.simplified = result

        self.vars = [term for term in result.free_symbols if isinstance(term, sp.Indexed)]
        self.terms = set([str(term.base) for term in self.vars])

        ## Gather terms
        # Quadratic terms
        self.qf = get_monomials(result.coeff(s[i]*s[j]), self.vars)
        
        # Linear terms (this should be multiplied by 2)
        self.lf = get_monomials(result.subs(s[j],0).coeff(s[i]), self.vars)

        # Constant
        self.cf = get_monomials(result.subs({s[i]:0, s[j]:0}), self.vars)
        
        self.quad  = sum([a*x for a,x in self.qf.items()])
        self.lin = sum([a*x for a,x in self.lf.items()])
        self.const = sum([a*x for a,x in self.cf.items()])

    def __call__(self, **kwargs):
        """
        Compute coefficient arrays given matrices
        """
        coeffs = self.compute_vars(**kwargs)
        args = list(kwargs.keys())

        if self.quad != 0:
            qf = self.quad.subs(self.size_vals)
            J = einify(qf, include_args=args)(**coeffs, **kwargs)
        else:
            J = 0
        if self.lin != 0:
            lf = self.lin.subs(self.size_vals)
            h = einify(lf, include_args=args)(**coeffs, **kwargs)
        else:
            h = 0 
        if self.const != 0:
            cf = self.const.subs(self.size_vals)
            b = einify(cf, include_args=args)(**coeffs, **kwargs)
        else:
            b = 0

        return J, h, b

    def __add__(self, func):
        
        if not isinstance(func, Contraction):
            return ValueError('Must be another instance of this class')
        
        comb_vars = list(set(self.vars).union(set(func.vars)))
        
        a1 = self.k_sum*self.l_sum
        a2 = func.k_sum*func.l_sum

        qf_out = get_monomials(expand(a1*self.quad + a2*func.quad), comb_vars)
        lf_out = get_monomials(expand(a1*self.lin + a2*func.lin), comb_vars)
        const_out = get_monomials(expand(a1*self.const + a2*func.const), comb_vars)
        
        return qf_out, lf_out, const_out

    def __sub__(self, func):
        
        if not isinstance(func, Contraction):
            return ValueError('Must be another instance of this class')
        
        comb_vars = list(set(self.vars).union(set(func.vars)))
        
        a1 = self.k_sum*self.l_sum
        a2 = func.k_sum*func.l_sum

        qf_out = get_monomials(expand(a1*self.quad - a2*func.quad), comb_vars)
        lf_out = get_monomials(expand(a1*self.lin - a2*func.lin), comb_vars)
        const_out = get_monomials(expand(a1*self.const - a2*func.const), comb_vars)
        
        return qf_out, lf_out, const_out

    def replace(self, repl):

        self.simplified = self.simplified.subs(repl)

        self.vars = [term for term in self.simplified.free_symbols if isinstance(term, sp.Indexed)]
        self.terms = set([str(term.base) for term in self.vars])

        ## Gather terms
        # Quadratic terms
        self.qf = get_monomials(self.simplified.coeff(s[i]*s[j]), self.vars)
        
        # Linear terms (this should be multiplied by 2)
        self.lf = get_monomials(self.simplified.subs(s[j],0).coeff(s[i]), self.vars)

        # Constant
        self.cf = get_monomials(self.simplified.subs({s[i]:0, s[j]:0}), self.vars)
        

    def set_bounds(self, **kwargs):
        """
        Set specific values for each bounding variable in the `sizes` list
        """

        size_vals = sp.lambdify(self.sizes, self.sizes)(**kwargs)
        self.size_vals = {str(name):val for name,val in zip(self.sizes, size_vals)}

    def compute_vars(self, **kwargs):

        n_val = self.size_vals['n']
        args = list(kwargs.keys())
        # var_funcs = {}
        var_vals = {}

        ## I've got to do some uncouth stuff for the sake of generality
        for name, (expr, lims) in eval_dict.items():
            if str(name) in self.terms:
                size_expr = sp.Sum(1, (wk,)+lims).doit()
                size = sp.lambdify(self.sizes, size_expr)(**self.size_vals)
                lims_ = sp.lambdify(self.sizes, lims)(**self.size_vals)
                lims_ = list(lims_)
                lims_[0] -= 1

                varfunc = einify(expr/size, [wk], {wk: lims_}, include_args=args)
                var_vals[str(name)] = varfunc(**kwargs)

        return var_vals

    
#####################################################################
########## Helpers ##################################################
#####################################################################

def get_monomials(expr, these_vars, repl={}):
    
    poly = Poly(expr, *these_vars)
    out_dict = {}
    for powers, cfs in poly.as_dict().items():
        term = sp.prod([g**e for g,e in zip(poly.gens, powers)])
        # coef = sp.factor(sp.expand(sp.factor(cfs).subs(repl)))
        out_dict[term] = sp.factor(cfs)
    
    return out_dict

def einify(idx_expr, dummies=[], bounds={}, include_args=[]):
    """
    Convert indexed expression into a sum of einsums for numerical evaluation

    Can mannually specify indices to sum over even if they only appear once
    """

    ## Give indices new names to ensure they are single letters
    alphabet = string.ascii_lowercase
    
    ## Get indices and limits for each indexed variable
    these_vars = []     # storing indexed variables
    var_idx = []        # indices of each variable
    var_lims = {}       # limits of each index
    ijk = {ix:alphabet[i] for i,ix in enumerate(bounds.keys())}
    nidx = len(bounds)
    for x in idx_expr.free_symbols:
        if isinstance(x, sp.Indexed):
            these_vars.append(x) 
            indices = ''
            lims = []
            for ix in x.indices:
                if ix in bounds.keys():
                    indices += ijk[ix]
                    lims.append(f"{bounds[ix][0]}:{bounds[ix][1]}")
                else:
                    ijk[ix] = alphabet[nidx]
                    indices += ijk[ix]
                    bounds[ix] = ('','')
                    lims.append(f"{bounds[ix][0]}:{bounds[ix][1]}")
                    nidx += 1

            var_idx.append(indices)
            var_lims[x.base] = ','.join(lims)

    poly = Poly(idx_expr, *these_vars)
    terms = poly.terms()
    
    ## Enforce consistent output shape across terms
    out_idx = set([])  # the indices which appear in the output
    term_args = []
    term_idx = []
    term_coefs = []
    fun_args = set([])
    for powers, cfs in terms:

        ## Deal with any coefficients
        coef_inps = cfs.free_symbols
        term_coefs.append(pycode(cfs))
        fun_args.update(coef_inps)

        ## Figure out the einsum shapes
        arg_idx = []
        args = []
        for v, p in enumerate(powers):
            args += [these_vars[v].base]*p
            arg_idx += [var_idx[v]]*p
        term_args.append(args)
        term_idx.append(arg_idx)

        for ix, ab in ijk.items():
            if (''.join(arg_idx).count(ab) == 1) and ix not in dummies:
                out_idx.update(set([ab]))
    out_str = ''.join(out_idx)

    ## Best I can think of is to write a function and define it with exec
    fun_str = "\tout = 0"
    for args, arg_idx, coef in zip(term_args, term_idx, term_coefs):
        
        fun_str += f"\n\n\ta = {coef}"

        ## Make the einsum
        if len(args) > 0: # if it's not a scalar

            ## Add any missing indices and axes
            arg_strings = []
            for this_arg in range(len(args)):
                lims = var_lims[args[this_arg]]
                for guy in out_idx-set(arg_idx[this_arg]):
                    arg_idx[this_arg] += guy
                    lims += ',None'
                arg_strings.append(f"{str(args[this_arg])}[{lims}]")
            arg_str = ','.join(arg_strings)

            ## Put it all into a string
            fun_args.update(set(args))
            idx_str = ','.join(arg_idx)

            fun_str += f"\n\tout += a*np.einsum('{idx_str}->{out_str}',{arg_str})"
        
        else: # this will broadcast as a scalar
            fun_str += "\n\tout += a"
    
    fun_args = set([str(x) for x in fun_args])
    fun_args = set(include_args).union(fun_args)
    sorted_args = sorted(fun_args, key=str)
    def_str = f"def meinsums({','.join(str(x) for x in sorted_args)}):"
    doc_str = f"\t\"\"\"\n\tComputes:\n{fun_str}\n\t \"\"\""
    func_str = '\n'.join([def_str, doc_str, fun_str, "\treturn out"]) 
    
    ## god help me 
    local_namespace = {}
    exec(func_str, globals(), local_namespace)
    
    out_fun = local_namespace["meinsums"]

    return out_fun

