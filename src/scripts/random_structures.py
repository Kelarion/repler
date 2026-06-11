import numpy as np
import scipy.stats as sts
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from collections import defaultdict


CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
import sys
sys.path.append(CODE_DIR)
import util
import plotting as tpl
import df_util

#%%

def generate_parametric_no_chain_tree(N, rho=0.5, alpha=1.0):
    """
    N: Number of nodes
    rho: (0, 1] Control for number of internal nodes M.
    alpha: (>0) Control for branching concentration. 
           High alpha = uniform branching; Low alpha = hub-heavy.
           
    Modified from Gemini
    """
    if N < 3:
        # A tree with N < 3 cannot avoid degree 1 unless it's just a single node.
        return nx.path_graph(N, create_using=nx.DiGraph())

    # 1. Determine M (Number of internal nodes)
    # We use a Binomial draw so there is still some variance, centered on rho
    M_max = (N - 1) // 2
    M = np.random.binomial(M_max - 1, rho) + 1 # Ensure at least 1 hub
    
    # 2. Determine Degrees using Dirichlet-Multinomial
    # Surplus edges beyond the minimum 2 per internal node
    surplus_count = (N - 1) - (2 * M)
    
    # Generate weights for the M nodes
    weights = np.random.dirichlet([alpha] * M)
    
    # Distribute surplus edges based on weights
    surplus_dist = np.random.multinomial(surplus_count, weights)
    internal_degrees = 2 + surplus_dist
    
    # 3. Assemble the full degree sequence
    degrees = np.zeros(N, dtype=int)
    degrees[:M] = internal_degrees
    np.random.shuffle(degrees)
    
    # 4. Standard Cyclic Lemma + BFS Construction
    steps = degrees - 1
    prefix_sums = np.cumsum(steps)
    shift = (np.argmin(prefix_sums) + 1) % N
    valid_degrees = np.roll(degrees, -shift)
    
    nodes = np.random.permutation(N)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    queue = [nodes[0]]
    child_idx = 1
    for d in valid_degrees:
        if not queue: break
        parent = queue.pop(0)
        for _ in range(d):
            if child_idx < N:
                child = nodes[child_idx]
                G.add_edge(parent, child)
                queue.append(child)
                child_idx += 1
                
    return G

#%%

N = 16

balanced_tree = generate_parametric_no_chain_tree(N, rho=0.9, alpha=1e3)

hub_tree = generate_parametric_no_chain_tree(N, rho=1, alpha=1e-1)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("alpha tiny")
nx.draw(balanced_tree, node_size=20, alpha=0.7, pos=graphviz_layout(balanced_tree))

plt.subplot(122)
plt.title("alpha huge")
nx.draw(hub_tree, node_size=20, alpha=0.7, pos=graphviz_layout(hub_tree))
plt.show()

#%%

def generate_signed_clique_matrix(G, alpha=0, min_size=2):
    """
    Converts a directed tree G into the adjacency matrix of a signed graph.
    - Each node i in G becomes a clique of size max(1, out_degree(i)).
    - Internal clique edges are -1.
    - Directed edges (i -> j) become +1 edges from one node in C_i to all in C_j.
    
    Modified from Gemini
    """
    # 1. Calculate clique sizes and start indices
    nodes = list(G.nodes())
    out_degrees = {u: G.out_degree(u) for u in nodes}
    
    clique_sizes = {u: max(min_size, out_degrees[u]+np.random.poisson(alpha)) for u in nodes}
    
    # Map tree node ID to the start index in the adjacency matrix
    start_indices = {}
    current_idx = 0
    for u in nodes:
        start_indices[u] = current_idx
        current_idx += clique_sizes[u]
    
    N_total = current_idx
    adj = np.zeros((N_total, N_total), dtype=int)
    
    # 2. Fill cliques with negative edges
    for u in nodes:
        size = clique_sizes[u]
        start = start_indices[u]
        if size > 1:
            # Create a clique of -1s
            for i in range(size):
                for j in range(i + 1, size):
                    adj[start + i, start + j] = -1
                    adj[start + j, start + i] = -1
                    
    # 3. Fill directed edges with positive connections
    for u in nodes:
        start_u = start_indices[u]
        # Get successors (children)
        successors = list(G.successors(u))
        
        for k, v in enumerate(successors):
            # Anchor node is the k-th node in clique C_u
            anchor_node_idx = start_u + k
            
            # Target is the entire clique C_v
            start_v = start_indices[v]
            size_v = clique_sizes[v]
            
            for m in range(size_v):
                target_node_idx = start_v + m
                adj[anchor_node_idx, target_node_idx] = 1
                adj[target_node_idx, anchor_node_idx] = 1
                
    return adj, {k: np.arange(start_indices[k], start_indices[k]+clique_sizes[k]) for k in nodes}

#%%

N = 16

balanced_tree = generate_parametric_no_chain_tree(N, rho=1, alpha=1e3)

hub_tree = generate_parametric_no_chain_tree(N, rho=0.8, alpha=1e-1)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("alpha tiny")
nx.draw(balanced_tree, node_size=20, alpha=0.7, pos=graphviz_layout(balanced_tree))
    
plt.subplot(122)
plt.title("alpha huge")
nx.draw(hub_tree, node_size=20, alpha=0.7, pos=graphviz_layout(hub_tree))
plt.show()


#%%

def generate_blown_up_signed_matrix(G, lam, alpha=0, min_size=2):
    """
    G: Directed tree.
    lam: Poisson rate for the size of each independent set replacing a sign-graph node.
    """
    nodes = list(G.nodes())
    # S_i is the number of 'slots' or 'independent sets' for tree node i
    clique_slots = {u: max(min_size, G.out_degree(u) + np.random.poisson(alpha)) for u in nodes}
    
    # 1. Sample sizes for every independent set
    # map: tree_node -> list of set_sizes
    set_sizes = {}
    for u in nodes:
        num_slots = clique_slots[u]
        # Sample Poisson for each slot. We use max(1, ...) to avoid 
        # empty sets which would effectively delete nodes from the hierarchy.
        sizes = [max(1, np.random.poisson(lam)) for _ in range(num_slots)]
        set_sizes[u] = sizes

    # 2. Map slots to global indices in the final adjacency matrix
    slot_indices = {} # (tree_node, slot_index) -> (start, end)
    current_idx = 0
    for u in nodes:
        for k, size in enumerate(set_sizes[u]):
            slot_indices[(u, k)] = (current_idx, current_idx + size)
            current_idx += size
            
    N_total = current_idx
    adj = np.zeros((N_total, N_total), dtype=int)
    
    # 3. Fill the Multipartite Negative Structures
    for u in nodes:
        num_slots = clique_slots[u]
        if num_slots > 1:
            # All sets within the same tree-node 'clique' connect with -1
            for k in range(num_slots):
                start_k, end_k = slot_indices[(u, k)]
                for l in range(k + 1, num_slots):
                    start_l, end_l = slot_indices[(u, l)]
                    # Complete bipartite -1 connection between sets k and l
                    adj[start_k:end_k, start_l:end_l] = -1
                    adj[start_l:end_l, start_k:end_k] = -1
                    
    # 4. Fill the Positive Bridges
    for u in nodes:
        successors = list(G.successors(u))
        for k, v in enumerate(successors):
            # The anchor is the k-th independent set of tree-node u
            start_anchor, end_anchor = slot_indices[(u, k)]
            
            # The target is ALL independent sets of tree-node v
            for m in range(clique_slots[v]):
                start_target, end_target = slot_indices[(v, m)]
                
                # Complete bipartite +1 connection
                adj[start_anchor:end_anchor, start_target:end_target] = 1
                adj[start_target:end_target, start_anchor:end_anchor] = 1
                
    return adj
