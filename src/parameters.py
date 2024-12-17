import socket
import os
import sys
import pickle as pkl
import subprocess
import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import itertools as itt

import util

class Parameter(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Define an parameter and its range using symbolic expressions. Can be used to 
    define dependencies between iterators by abusing python's native syntax. 

    For example, to iterate k over integers from 1 to 5, and to make j go from 
    1 to 2^k, you'd write:

    1 << k << 5
    1 << j << 2**k
    """

    def __init__(self):

        self.funcs = []
        self.value = None
        self.root_instance = True

        self.checks = [] # logical functions of the parameter

    # def generate_values(self):
    #     """
    #     This function needs to be implemented in descendent classes, 
    #     it should return an interable
    #     """

    #     raise NotImplementedError

    def new_value(self):
        raise NotImplementedError

    def __next__(self):
        """
        Advance the value by one element
        """

        while not self: 
            self.new_value()

        return self.value

    def __iter__(self):
        """
        Initialize to the first value
        """

        return self

    def __bool__(self):

        return bool(np.all([c() for c in self.checks]))

    def __eq__(self, other):

        if isinstance(other, Parameter):
            return id(self) == id(other)
        else:
            return False

    def __hash__(self):

        return id(self)

    def __deepcopy__(self, memo):
        """
        Custom behavior for deepcopy, so root refers to original instance
        """

        clss = self.__class__
        result = clss.__new__(clss)
        memo[id(self)] = result
        for k,v in self.__dict__.items():
            if k in ['roots', 'checks']:
                setattr(result, k, v)
            elif k in ['funcs']:
                setattr(result, k, copy.copy(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        result.root_instance = False

        return result

    def __call__(self, *vals):

        if len(vals) == len(self.roots):
            for r,v in zip(self.roots, vals):
                r.value = v

        if self.root_instance:
            return self.value

        for f, args, kwargs in self.funcs:
            arrgs = []
            for arg in args:
                if isinstance(arg, Parameter):
                    arrgs.append(arg())
                else:
                    arrgs.append(arg)
            ret = f(*tuple(arrgs), **kwargs)

        return ret 

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Keep track of functions, which will be applied to value
        """

        if method == '__call__':


            clone = copy.deepcopy(self)

            ## Union the roots of Parameter inputs
            for this_inp in inputs:
                if isinstance(this_inp, Parameter):
                    if this_inp.root_instance:
                        clone.roots = clone.roots | {this_inp}
                    else:
                        clone.roots = clone.roots | this_inp.roots

            clone.funcs += [(ufunc, inputs, kwargs)]

            return clone

        else:
            return NotImplemented

    def __or__(self, other):
        # print('or')

        if isinstance(other, tuple):
            self.checks.append(*other)
        else:
            self.checks.append(other)

        return self

    def __rshift__(self, other):

        if self.root_instance:

            other.ub = self
            # return ParamSet(self) >> other

        elif isinstance(other, Parameter):
            if other.root_instance:
                return ParamSet(other) << self
            else:
                return ParamSet(self) >> other
        elif isinstance(other,ParamSet):
            return other >> self
        else:
            return ParamSet(self) >> other

    def __lshift__(self, other):

        if self.root_instance:
            return ParamSet(self) << other

        elif isinstance(other, Parameter):
            if other.root_instance:
                return ParamSet(other) >> self
            else:
                return ParamSet(self) << other
        elif isinstance(other,ParamSet):
            return other >> self
        else:
            return ParamSet(self) << other

    def __rrshift__(self, other):

        if self.root_instance:
            return ParamSet(self) << other
        elif isinstance(other, Parameter):
            if other.root_instance:
                return ParamSet(other) >> self
            else:
                return ParamSet(self) << other
        elif isinstance(other,ParamSet):
            return other >> self
        else:
            return ParamSet(self) << other

    def __rlshift__(self, other):

        if self.root_instance:
            return ParamSet(self) >> other
        elif isinstance(other, Parameter):
            if other.root_instance:
                return ParamSet(other) << self
            else:
                return ParamSet(self) >> other
        elif isinstance(other,ParamSet):
            return other >> self
        else:
            return ParamSet(self) >> other

    # def __rshift__(self, other):
    #     # print('rshift')
    #     if self.root_instance:
    #         self.lb = other

    #     return self

    # def __lshift__(self, other):
    #     # print('lshift')
    #     self.ub = other

    #     return self

    # def __rrshift__(self, other):
    #     # print('rrshift')
    #     self.ub = other

    #     return self

    # def __rlshift__(self, other):
    #     # print('rlshift')
    #     self.lb = other

    #     return self


## Clothing
class Set(Parameter):
    """
    A parameter which takes values from a pre-defined set
    """

    def __init__(self, values):

        super(Set, self).__init__()

        self.values = values

        roots = set()
        for val in values:
            if isinstance(val, Parameter):
                roots = roots | val.roots

        if len(roots) > 0:
            self.roots = roots
        else:
            self.roots = {self}

    def generate_values(self):

        return [v() if isinstance(v, Parameter) else v for v in self.values]


class Integer(Parameter):

    def __init__(self, num=None, step=None):

        super(Integer, self).__init__()

        self.num = num
        self.step = step

        self.roots = {self} # base parameter, without any functions applied

    def generate_values(self, lb, ub):

        if self.step is not None:
            step = self.step
        elif self.num is not None:
            step = int((ub - lb) // self.num)
        else:
            raise ValueError('Need to define either step size or number')

        return np.arange(lb, ub+1, step)


class Real(Parameter):

    def __init__(self, num=None, step=None):

        super(Real, self).__init__()

        self.num = num
        self.step = step

        self.roots = {self} # base parameter, without any functions applied

    def __next__(self):

        newval = self.value + self.step


    def generate_values(self, lb, ub):

        if self.num is not None:
            num = self.num
        elif self.step is not None:
            num = int((ub - lb) // self.step)
        else:
            raise ValueError('Need to define either step size or number')

        return np.linspace(lb, ub, num=num) 


## utility classes that should be hidden to user
class ParamSet:
    """
    A set of values of a parameter
    """

    def __init__(self, param):

        self.param = param

        self.lb = None 
        self.ub = None 

    def __iter__(self):

        ## Disallow iteration with parametric bounds
        if isinstance(self.lb, Parameter) or isinstance(self.ub, Parameter):
            raise TypeError('Not iterable: range depends on other parameters')

        return ParamIter(self).__iter__()

    def init_values(self):
        """
        Return the set of values which satisfy constraints
        """

        # if self.param.root_instance:
        if isinstance(self.param, Set):
            values = self.param.generate_values()

        else:
            if isinstance(self.lb, Parameter):
                lb = self.lb()
            else:
                lb = self.lb

            if isinstance(self.ub, Parameter):
                ub = self.ub()
            else:
                ub = self.ub

            values = self.param.generate_values(lb, ub)

        self.param.value = values[0]

        if self.param.root_instance:
            return filter(self.valid , values) 
        else:
            return values.__iter__()


    def dependencies(self):
        """
        Return a set of all parameters which constrain this set
        """

        deps = self.param.roots - {self.param}
        for c in self.param.checks:
            deps = deps | c.roots - {self.param}

        if isinstance(self.lb, Parameter):
            deps = deps | self.lb.roots
            
        if isinstance(self.ub, Parameter):
            deps = deps | self.ub.roots

        return deps 

    def valid(self, val):

        # if self.param not in self.param.roots:
        #     raise Exception('Can only check constraints on root parameters')
        # else:
        self.param.value = val

        return bool(np.all([c() for c in self.param.checks]))

    def __bool__(self):
        return bool(np.all([c() for c in self.param.checks]))

    def __rshift__(self, other):
        # print('rshift')
        self.lb = other

        return self

    def __lshift__(self, other):
        # print('lshift')
        self.ub = other

        return self

    def __rrshift__(self, other):
        # print('rrshift')
        self.ub = other

        return self

    def __rlshift__(self, other):
        # print('rlshift')
        self.lb = other

        return self

    def __mod__(self, other):

        if isinstance(other, ParamIter):
            return ParamIter(self, *other.sets)
        elif isinstance(other, ParamSet):
            return ParamIter(self, other)
        elif isinstance(other, Parameter):
            return ParamIter(self, ParamSet(other))
        else:
            raise Exception('Multiplication not defined for %s'%type(other))

    def __rmod__(self, other):

        if isinstance(other, ParamIter):
            return ParamIter(self, *other.sets)
        elif isinstance(other, ParamSet):
            return ParamIter(self, other)
        elif isinstance(other, Parameter):
            return ParamIter(self, ParamSet(other))
        else:
            raise Exception('Multiplication not defined for %s'%type(other))

    def __or__(self, other):
        # print('or')

        if isinstance(other, tuple):
            self.param.checks.append(*other)
        else:
            self.param.checks.append(other)

        return self

class ParamIter:
    """
    Iterator over the product of parameter sets
    """

    def __init__(self, *sets):

        self.sets = sets

    def __iter__(self):

        params = [s.param for s in self.sets]

        ## Construct a partial order of parameter dependencies
        self.couplings = {i: set() for i in range(len(self.sets))}
        for i,s in enumerate(self.sets):

            for dep in s.dependencies():
                if dep in params:
                    j = params.index(dep)
                    self.couplings[j].add(i)
                else:
                    raise Exception('Constraining parameters not included!')

        ## coupling graph needs to be acyclic
        if util.is_cyclic(self.couplings):
            raise Exception("Parameters have cyclic dependencies, that's bad (`^`)")

        self.order = util.recursive_topological_sort(self.couplings)
        self.inv_ord = np.argsort(self.order)

        ## Initialize iterators following the partial order
        self.ittrs = [self.sets[i].init_values() for i in self.order] 
        self.current = None

        return self 

    def __next__(self):

        ## Basically implement nested for loops
        if self.current is None:
            self.current = [next(itr) for itr in self.ittrs]
        else:
            max_loop = self.loop_next(-1)
        
        return tuple(self.current[i] for i in self.inv_ord)

    def loop_next(self, it):
        """
        Recursion to advance nested for loops
        """

        try:
            # print(it)
            self.current[it] = next(self.ittrs[it])

        except StopIteration:
            # if -it > len(self.ittrs):
            #     raise StopIteration

            self.loop_next(it-1)

            self.ittrs[it] = self.sets[self.order[it]].init_values()
            self.current[it] = next(self.ittrs[it])

        except IndexError:
            raise StopIteration

    def __mod__(self, other):

        ## Support products between paramter sets, parameters, lists, and tuples

        if isinstance(other, ParamIter):
            return ParamIter(*self.sets, *other.sets)
        elif isinstance(other, ParamSet):
            return ParamIter(*self.sets, other)
        elif isinstance(other, Parameter):
            return ParamIter(*self.sets, ParamSet(other))
        else:
            raise Exception('Multiplication not defined for %s'%type(other))

    def __rmod__(self, other):

        ## Support products between paramter sets, parameters, lists, and tuples

        if isinstance(other, ParamIter):
            return ParamIter(*self.sets, *other.sets)
        elif isinstance(other, ParamSet):
            return ParamIter(*self.sets, other)
        elif isinstance(other, Parameter):
            return ParamIter(*self.sets, ParamSet(other))
        else:
            raise Exception('Multiplication not defined for %s'%type(other))

