import numpy as np
import networkx as nx
import util

def isometrichull(G, nodes):
    """
    Remove as as many nodes from `G` as we cal while maintaining 
    shortest path length between all pairs in `nodes`

    Brute force approach
    """

    hidden = util.rangediff(len(G.nodes), nodes)
    N = len(nodes)
    M = len(hidden)

    ## Create an (M x M x N x N) array I such that I[i,j,k,l] is 
    ## the number of times hidden nodes i and j are both part of 
    ## a shorted path between nodes k and l in G
    ## But it should be sparse for efficiency
    for k in range(N):
        for l in range(k+1,N):
            paths = nx.all_shortest_paths(G, nodes[k], nodes[l])
            


    return path_incidence[:,hidden], num_paths