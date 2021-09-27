import numpy as np
import sympy as simp
import sympy.logic.boolalg as boo
import re 

#%%
def nor_circuit(vertices, edges, expr):
    """
    Recursion which converts CNF expression to NOR circuit graph
    """
    
    expr.split()