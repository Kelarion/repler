import sys

import pickle as pkl

import numpy as np
import scipy.linalg as la
import networkx as nx
from networkx.algorithms.simple_paths import all_simple_paths
from networkx.linalg.graphmatrix import incidence_matrix
from networkx.algorithms.lowest_common_ancestors import all_pairs_lowest_common_ancestor
from networkx.algorithms.shortest_paths.weighted import dijkstra_path_length as dpl
import torch

import os
import re
import random
from time import time
from itertools import combinations, product

#%% 
class ParsedSequence(object):
    """
    A class for interacting with a parsed sequence. Takes in strings of the form:

    (s (a1 (b1 word1 ) (b2 (c1 word2) (c2 word3) ) (a2 word4) )

    which concisely, but illegibly, represent a parse tree. Converts this to a more legible 
    list of edges and nodes, lets you compute various nice things, and also implements some 
    simple perturbations of the sequence (like swapping phrases, or pairs of words).

    __init__ inputs:
        bs: the bracketed sentence string
        dep_tree (Bool, default False): is this a dependency parse?
        op_brak (default '['): what kind of bracket starts a phrase
        cl_brak (default ']'): what kind of bracket closes a phrase
        no_words (default True): are the words in addition to phrase tags?

    Attributes (most of them, at least):
        edges: list of tuples, (parent, child), representing the edges of the tree
        nodes: list of strings, 'POS [word]', of all tokens (terminal and non-terminal)
        words: list of strings, 'word', which are the words of the sentence (i.e. terminals)
        brackets: array of {+1, -1, 0}, for {opening, closing, terminal tokens}
        bracketed_string: string, the input used to create the object
        parents: array, the parent of each token (is -1 for the root)
        node_span: array, for each token, the index of the token right after its constituent ends
        subtree_order: array, ints, the `order' of each token

    Note that attributes which return indices, they index the tokens in the order
    they appear in the bracketed sequence -- for a dependency parse, this doesn't 
    generally match the order in the actual sentence. It also doesn't match the 
    indices of word in the sentence (which go from 0, ..., num_terminal_tokens).
    To convert between these three indexing coordinates, use the `term2node`, `node2word`, 
    `word2node` etc. attributes, where a2b contains indices of a in b coordinates.

    Methods (external use):
        tree_dist(i, j, term=True):
            Distance between tokens i and j on the parse tree, returns an integer
            `term` toggles whether i and j index the words, or all nodes
        bracket_dist(i, j):
            Number of (open or closed) brackets between i and j, not including 
            brackets associated with terminal tokens. Related to tree distance.
            Only works on terminal tokens
        is_relative(i, j, order=1):
            Are tokens i and j in the same phrase? 
            `order` is the maximum depth of the subtree that is considered a phrase 
        
    """
    def __init__(self, bs, dep_tree=False, op_brak='[', cl_brak=']', no_words=True):
        """
        Take a bracketed sentence, bs, and compute various quantities defined 
        on it and/or the corresponding parse tree. Provides methods to ask
        questions about the tokens and subtrees.
        
        If the brackets represent a dependency parse, set `dep_tree` to be True 
        """
        
        super(ParsedSequence,self).__init__()
        
        self.dep_parse = dep_tree

        op = np.array([i for i,j in enumerate(bs) if j==op_brak])
        cl = np.array([i for i,j in enumerate(bs) if j==cl_brak])
        braks = np.sort(np.append(op,cl))
        
        N_node = len(op)
        
        # First: find children
        iscl = np.isin(braks,cl) # label brackets
        
        term_op = braks[np.diff(iscl.astype(int),append=0)==1]
        term_cl = braks[np.diff(iscl.astype(int),append=0)==-1]
        
        assert np.all(np.isin(term_op,op)), "Unequal open and closed brackets!"
        
        leaves = np.where(np.isin(op,term_op))[0]
        # parse depth of each token is the total number of unclosed brackets
        depth = np.cumsum(~iscl*2-1)[np.isin(braks,op)]-1 
        
        # algorithm for finding all edges
        # takes advantage of the fact that nodes only have one parent in a tree
        nodes = np.arange(N_node)
        drawn = list(leaves)
        edges = []
        labels = ['' for _ in range(N_node)]
        i = 0
        while ~np.all(np.isin(nodes,drawn[:i])):
            # find edge
            if drawn[i]>0: # start token has no parent
                parent = np.where((op<op[drawn[i]]) & (depth==(depth[drawn[i]]-1)))[0][-1]
                edges.append((parent, drawn[i]))
            if parent not in drawn:
                drawn.append(parent)
                
            # get token string
            tok_end = braks[np.argmax(braks>op[drawn[i]])]
            tok = bs[op[drawn[i]]:tok_end]
            labels[drawn[i]] = tok.replace(op_brak,'')#.replace(' ','')
            
            i += 1
            
        self.edges = edges
        self.nodes = labels
        self.term2word = leaves # indices of terminal tokens in the sentence
        
        parents = [edges[np.where(np.array(edges)[:,1]==i)[0][0]][0] \
                   for i in range(1,len(nodes))]
        self.parents = np.append(-1,parents) # seq-indexed
        
        self.depth = depth
        
        # Find the order of each subtree 
        depf = np.append(self.depth,0)
        # first, for each node find the node right after its constituent ends
        span = [(depf[i+1:]<=depf[i]).argmax()+(i+1) for i in range(len(self.nodes))]
        # then find the maxmimum distance between each node and its descendants
        max_depth = [self.depth[i:span[i]].max()-self.depth[i] for i in range(len(self.nodes))]
        self.node_span = np.array(span) # each token's corresponding closing
        self.subtree_order = np.array(max_depth) # each node's order (=0 for terminals)
        
        # Represent the bracketed sentence as +/- 1 string
        # brah = np.sort(np.append(op,cl))
        # brah[np.isin(braks,op)] = 1
        # brah[np.isin(braks,cl)] = -1
        # brah[np.diff(iscl.astype(int),append=0)==1] = 0
        # # brah[np.diff(iscl.astype(int),append=0)==-1] = 0
        brah = np.array([1 if b in op[~np.isin(op,term_op)] \
                         else -1 if b in cl[~np.isin(cl,term_cl)] \
                             else 0 if b in term_op \
                                 else np.nan for b in braks])
        brah = brah[~np.isnan(brah)]
        self.brackets = brah.astype(int)
        self.term2brak = np.where(brah==0)[0]
        
        # # old code, not sure what exactly it computes
        # is_trough = np.flip(np.diff(np.flip(~iscl*2-1),prepend=1)) == -2
        # trough_depths = np.cumsum(~iscl*2-1)[is_trough]-1
        # is_trough[-1] = False
        # subtree_depths = trough_depths[np.cumsum(is_trough)][np.isin(braks,op)]
        # self.subtree_depth = subtree_depths
        
        # 
        if no_words:
            self.words = leaves
            self.ntok = len(leaves)
        else:
            if dep_tree:
                line = [n.split(' ')[1] for n in labels]
            else:
                words = np.array(labels)[leaves]
                line = [w.split(' ' )[1] for w in words]
            self.words = line
            self.ntok = len(line)
        
        self.bracketed_string = bs
        
        # when dealing with dep trees, the index in the actual sentence
        # is no the same as the index in the bracketed sentence 
        if dep_tree:
            node_names = [int(n.split(' ')[2]) for n in labels]
            order = list(np.argsort(node_names))
            self.words = [self.words[i] for i in order]
            word_names = [node_names.index(i) for i in range(self.ntok)]
        else:
            node_names = list(range(len(labels)))
            word_names = list(range(len(labels)))
            
        self.node2word = np.array(node_names) # indices of each node in the sequence
        self.word2node = np.array(word_names) # indices of each word in the node list
        self.node2term = np.array([leaves.tolist().index(i) if i in leaves else -1 \
                                   for i in node_names])
        
        self.node_tags = [w.split(' ')[0] for w in self.nodes]
        self.pos_tags = list(np.array(self.node_tags)[leaves])
        
    def __repr__(self):
        return self.bracketed_string
    
    def parse_depth(self, i, term=True):
        """Distance to root"""
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
        else:
            i = self.word2node[i]
        
        return self.depth[i]
    
    def path_to_root(self, path, tok):
        """Recursion to fill `path` with the ancestors of `tok`"""
        path.append(tok)
        whichedge = np.where(np.array(self.edges)[:,1]==tok)[0]
        if len(whichedge)>0:
            parent = self.edges[whichedge[0]][0]
            path = self.path_to_root(path, parent)
        return path
    
    def tree_dist(self, i, j, term=None):
        """
        Compute d = depth(i) + depth(j) - 2*depth(nearest common ancestor)
        """
        if term is None:
            term = not self.dep_parse
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
            j = self.word2node[self.term2word[j]]
        else:
            i = self.word2node[i]
            j = self.word2node[j]
            
        # take care of pathological cases
        if i>j:
            i_ = j
            j_ = i
        elif i<j:
            i_ = i
            j_ = j
        else:
            return 0
        if i==0:
            return self.depth[j_]
        
        # get ancestors of both
        anci = np.array(self.path_to_root([], i_))  # [i, parent(i), ..., 0]
        ancj = np.array(self.path_to_root([], j_))  # [j, parent(j), ..., 0]
        
        # get nearest common ancestor
        nearest = np.array(anci)[np.isin(anci,ancj)][0]
        
        return self.depth[i_] + self.depth[j_] - 2*self.depth[nearest]
    
    def ancestor_tags(self, i, n=2):
        i_ = self.word2node[self.term2word[i]]
        anc = np.array(self.path_to_root([], i_)) # all ancestors
        return self.node_tags[anc[np.min([n, len(anc)-1])]] # choose the nth one
    
    def phrases(self, order=1, min_length=2, strict=False):
        """
        Phrases of a given order are subtrees whose deepest member is at 
        most `order' away from the subtree root
        
        Returns a list of arrays, which contain the indices of all phrases of 
        specified order (if strict) or at least specified order (if not strict)
        
        Note that if strict=False, some indices might appear twice, as phrases 
        of lower order are nested in phrases of higher order.
        """
        if order == 0:
            is_fake = True
            order = 1
        else:
            is_fake = False
        if strict:
            phrs = np.where((self.subtree_order!=0)&(self.subtree_order==order))[0]
        else:
            phrs = np.where((self.subtree_order!=0)&(self.subtree_order<=order))[0]
        
        # if is_fake:
        #     phrs = np.sort(np.random.choice(len(self.node2term),6,replace=False))

        phrases = [np.array(range(i+1,self.node_span[i])) for i in phrs]
        # print(phrases)

        chunks = [self.node2term[p[np.isin(p, self.term2word)]] for p in phrases]
        chunks = [c for c in chunks if len(c)>=min_length]

        if is_fake:
            c0 = np.array([c[0]+len(c)//2 for c in chunks])
            len_phrs = np.array([len(c) for c in chunks])
            ovlp = np.array([c0[i] - (c0[i-1]+len_phrs[i-1]) for i in range(1,len(c0))] + [0])
            c0 += ovlp*(ovlp<0)
            chunks = [np.arange(c0[i], c0[i]+len(chunks[i])) for i in range(len(chunks))]
        
        return [c for c in chunks if len(c)>=min_length and max(c)<=self.ntok]
    
    def bracket_dist(self,i,j):
        """Number of (non-terminal) brackets between i and j. Only guaranteed 
        to be meaningful for adjacent terminal tokens."""
        i = self.node2term[self.word2node[self.term2word[i]]]
        j = self.node2term[self.word2node[self.term2word[j]]]
            
        return np.abs(self.brackets)[self.term2brak[i]:self.term2brak[j]+1].sum()
        
    def is_relative(self, i, j, order=1, term=None):
        """
        Bool, are tokens i and j part of the same n-th order subtree?
        Equivalently: is the nearest common ancestor of i and j of the specified
        order?
        """
        if term is None:
            term = not self.dep_parse
        
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
            j = self.word2node[self.term2word[j]]
        else:
            i = self.word2node[i]
            j = self.word2node[j]
        
        if order==1 and (self.depth[i]!=self.depth[j]):
            return False # a necessary condition
        
        anci = np.array(self.path_to_root([], i)) # these are node indices
        ancj = np.array(self.path_to_root([], j)) # these are word indices
        nearest = np.array(anci)[np.isin(anci,ancj)][0]
        
        return self.subtree_order[nearest]<=order

    def ngram_shuffle(self, n):
        """ shuffle sentence in units of n """
        if self.ntok<n:
            raise ValueError
        n_pad = int(n-np.mod(self.ntok,n))
        # padded = np.insert(swap_idx.astype(float), 
        #                    ntok-n_pad-1, 
        #                    np.ones(n_pad)*np.nan)
        padded = np.append(np.arange(self.ntok, dtype=float), np.ones(n_pad)*np.nan)
        while 1:
            shuf_idx = np.random.permutation(padded.reshape((-1,n))).flatten()
            shuf_idx = shuf_idx[~np.isnan(shuf_idx)].astype(int)
            if np.any(shuf_idx != np.arange(self.ntok)):
                break

        return shuf_idx

    def phrase_swap(self, order=1):
        """ swap two phrases of a given order """
        phr = self.phrases(order=order, strict=True)

        if len(phr)<2:
            return []

        these_phr = np.sort(np.random.choice(len(phr),2,replace=False))
        phra = [phr[i] for i in these_phr]

        splt_idx = np.concatenate([(s[0],s[-1]+1) for s in phra])
        chunked = np.split(np.arange(self.ntok),splt_idx)
        swap_idx = np.concatenate(np.array(chunked)[[0,3,2,1,4]])
        
        return swap_idx

    def adjacent_swap(self, t):
        """ Swap adjacent words that are distance t apart """

        crossings = np.diff(np.abs(self.brackets).cumsum()[self.term2brak])

        if not np.any(np.isin(crossings, t-2)):
            return []

        i = int(np.random.choice(np.where(crossings==t-2)[0]))

        swap_idx = np.array(range(self.ntok))
        swap_idx[i] = i+1
        swap_idx[i+1] = i

        return swap_idx



class HierarchicalData(object):

    def __init__(self):
        raise NotImplementedError

    def labels(self):
        raise NotImplementedError

    # def represent_labels(self, these_leaves, rep=None):
    #     """ 
    #     'rep' is any function which maps an integer index to some representation, 
    #     as long as the representation is the same dimension for each index.

    #     'these_leaves' is a lists of leaves to be represented
    #     """

    #     labs = self.labels(these_leaves)

    #     fix_nan = lambda x, l : np.where(np.isnan(l), 0 ,x)
    #     if rep is None:
    #         rep = lambda l : np.eye(self.fan_out)[:,l]

    #     reps = np.concatenate([fix_nan(rep(fix_nan(l,l).astype(int)), l) for l in labs])
    #     return reps
    def represent_labels(self, these_leaves):
        """ finds the label of each of 'these_leaves' """

        idx = np.array([ i for j in these_leaves for \
            i in self.similarity_graph.nodes('category')[j] ] )
        var = np.repeat(np.arange(len(these_leaves)), 
            [len(self.similarity_graph.nodes('category')[j]) for j in these_leaves])

        labels = np.zeros((self.num_cat,len(these_leaves)))
        labels[idx,var] = 1

        return labels

    def similar_representation(self, dim=500, similarity='laplacian',
        only_leaves=True, sigma=1, tol=1e-10):
        """ 
        Makes a random representation whose kernel matches the inverse Laplacian 
        or the depth of deepest common ancestor
        """

        if similarity == 'laplacian':
            L = self.graph_laplacian()
            leaves = np.isin(self.similarity_graph,self.items)

            C = la.inv(L+np.diag(leaves)*(1/sigma))
        elif similarity == 'dca':
            C = self.deepest_common_ancestor(only_leaves=False)

        if only_leaves:
            these = np.isin(self.similarity_graph,self.items)
        else:
            these = np.ones(len(C))>0

        eps = np.random.randn(len(C),500) # random gaussian vectors 
        eps -= eps.mean(1, keepdims=True)

        std = la.cholesky(np.cov(eps) + np.eye(eps.shape[0])*tol)
        rep = la.cholesky(C + np.eye(eps.shape[0])*tol).T@la.inv(std).T@eps
        return rep[these,:]

    def graph_laplacian(self):
        """ Laplacian of the similarity graph """
        N = incidence_matrix(self.similarity_graph, oriented=True).toarray()

        return N@N.T # laplacian

    def deepest_common_ancestor(self, only_leaves=True):

        G = self.similarity_graph
        if only_leaves:
            n = self.items.tolist()
        else:
            n = list(G)

        # lca = list(all_pairs_lowest_common_ancestor(G, product(n, n)))
        lca = []
        ix = []
        for i,j in product(n, n):
            dec_i = nx.descendants(G.reverse(), i).union(set([i]))
            dec_j = nx.descendants(G.reverse(), j).union(set([j]))
            all_anc = dec_i.intersection(dec_j)
            anc_depths = [len(G.nodes('category')[a]) for a in all_anc]
            lca.append(max(anc_depths))
            ix.append((n.index(i),n.index(j)))

        ix = np.array(ix)
        lca = np.array(lca)

        # ijv = np.array([(n.index(a[0][0]),n.index(a[0][1]),dpl(G.reverse(), a[1], 0)) for a in lca])
        
        LCA = np.ones((len(n), len(n)))
        # LCA[ijv[:,0],ijv[:,1]] = ijv[:,2]
        LCA[ix[:,0],ix[:,1]] = lca

        return LCA


    def random_sequence(self, child_prob=0.7, **recurse_args):
        """ 'child_prob' is the probability of a parent node producing a child node """

        self.child_prob = child_prob

        seq = [0]
        self.make_children(seq, 0, **recurse_args)

        sent = ParsedSequence(''.join(str(seq).split(',')))

        return sent

    def make_children(self, seq, idx, replace=False, max_child=None):
        """recursion to generate random sequence"""
        
        if max_child is None:
            max_child = self.fan_out

        if idx < self.depth:
            n_child = (np.random.binomial(max_child-1, self.child_prob, len(seq))+1).tolist()
            
            children = []
            for ix in range(len(seq)):
                c_all = [e[1] for e in self.similarity_graph.out_edges(seq[ix])]

                ins_idx = ix + sum(n_child[:0]) + 1
                c = [[i] for i in np.random.choice(c_all, n_child[ix], replace=replace).tolist()]
                seq[ins_idx:ins_idx] = c
                children += c
                
            for child in children:
                self.make_children(child, idx+1, replace=replace, max_child=max_child)

    ## functions for making the similarity graph
    def fill_similarity_graph(self, max_depth=None, path_rule='minimal'):
        """
        Constructs a directed acyclic graph such that, for any pair of leaf nodes, 
        the depth of their nearest common ancestor matches the number of features 
        they have in common. This is good for constructing an embedding in which 
        dot products match feature similarity, for example.
 
        This is done in two steps:
            First infer abstract hidden nodes based on shared labels. If two 
            leaf nodes, i and j, share all but 1 label, then we create a new node 
            whose labels are the intersection of i and j. After all leaf nodes, 
            this is repeated on the newly-created hidden nodes, etc. Then we do 
            another pass but this time for all pairs sharing all but 2 labels, 
            etc. until we reach `max_depth`. 

            Then add the edges by constructing paths from the root to each leaf
            node. This will create new hidden nodes when they are necessary, 
            where "necessary" is determined by the `path_rule` (see path_to_node 
            method).

        """

        G = nx.DiGraph()
        G.add_nodes_from(self.items)
        nx.set_node_attributes(G, 
            values={i:self.cats[i-1] for i in self.items}, 
            name='category')

        G.add_node(0, category=set())

        if max_depth is None:
            max_depth = max([len(i) for i in self.cats])

        # self.connect_layer(G, self.items, self.cats)

        ### Infer hidden nodes based on pairwise label intersections
        for n in range(1,max_depth):
            # connect pairs which are the same in all but n labels

            these_nodes = 1*self.items 
            while len(these_nodes)>0:
                # find common sources for all pairs at each depth 
                new_nodes = []
                for i in these_nodes:
                    for j in these_nodes[these_nodes>i]:

                        G.nodes('category')
                        L_ij = G.nodes('category')[i].intersection(G.nodes('category')[j])

                        if len(G.nodes('category')[i])-len(L_ij) != n:
                            continue
                        
                        # print(f"{(i,j)}, {G.nodes('category')[i]}V{G.nodes('category')[j]}={L_ij}")

                        matching_nodes = np.isin(G.nodes('category'), L_ij)

                        if np.any(matching_nodes):
                            new_source = np.where(matching_nodes)[0].item()
                        else:
                            new_source = max(G) + 1
                            G.add_node(new_source, category=L_ij)
                            new_nodes.append(new_source)

                        # G.add_edge(new_source, i)
                        # G.add_edge(new_source, j)

                these_nodes = np.array(new_nodes)

        ### Connect each leaf node to the source 
        for i in (self.items):
            self.path_to_node(G, 0, i, rule=path_rule) # recursion!

        return G

    def path_to_node(self, G, source, target, rule='minimal'):
        """ 
        Lay down path on G from source to target. To get from i to j, first find all 
        nodes k with one label more than i, and which share all their labels with j. 
        Then add an edge from i to k. Furthermore, if k's labels can be expressed as 
        a union of i's labels an another node, l's labels, add and edge from l to k.

        Possible rules for finding "parents" of k:
            'minimal' (default)
                only adds a new step if there exists a node at the same depth as source,
                such that labels(source) V labels(node) = labels(step)
            'matching' 
                creates all possible new steps, but only uses existing valid parents
            'maximal'
                creates all possible new steps, and all possible valid parents
    
        """

        source_cat = G.nodes('category')[source]
        targ_cat = G.nodes('category')[target]

        ds = len(source_cat)
        # dt = len(targ_cat) 

        steps = product(targ_cat-source_cat, [source_cat])

        for s in steps:
            # we start by specifying the next step's category
            new_cat = set([s[0]]).union(s[1])

            matching_nodes = [G.nodes('category')[i] == new_cat for i in sorted(G)]

            if np.any(matching_nodes):
                new_step = sorted(G)[np.where(matching_nodes)[0].item()]
            else:
                new_step = max(G) + 1

            # print('%s->%s'%(source_cat,new_cat))
            # find a spouse if it exists

            # bachelors = nx.descendants_at_distance(G.reverse(), target, dt-ds)
            bachelors = [i for (i,j) in G.nodes('category') if len(j) == ds]
            matching_nodes = [G.nodes('category')[i].union(source_cat) == new_cat for i in bachelors]

            if np.any(matching_nodes):
                # add spouses if they exist
                if new_step not in list(G):
                    G.add_node(new_step, category=new_cat)
                G.add_edge(source, new_step)

                for this_spouse in np.where(matching_nodes)[0]: # there are multiple parents
                    G.add_edge(list(bachelors)[this_spouse], new_step)
                    # print('%s+%s=%s'%(G.nodes('category')[list(bachelors)[this_spouse]],source_cat,new_cat))
            
            elif rule == 'minimal':
                # don't make the child if there is no parent
                if new_step in list(G):
                    G.add_edge(source, new_step) # lay a stone, move on
                else: 
                    continue # otherwise this fork is over
            elif rule == 'maximal':
                # invent all possible parents
                if new_step not in list(G):
                    G.add_node(new_step, category=new_cat)

                for sps in product(new_cat-source_cat, [source_cat]):
                    new_spouse = max(G) + 1
                    G.add_node(new_spouse, category=set([sps[0]]).union(sps[1]))
                    G.add_edge(new_spouse, new_step)

            # print('%s+%s=%s'%(G.nodes('category')[s],G.nodes('category')[source],new_cat))

            if new_step != target:
                self.path_to_node(G, new_step, target, rule=rule)

class RegularTree(HierarchicalData):

    def __init__(self, num_vars, fan_out, respect_hierarchy=True, 
                    graph_rule='minimal', max_depth=None):
        """ 
        'num vars' is a list of number of variables in each layer,
        while 'fan_out' is how many children each parent has 
        e.g. num_vars=[1,2,4,8] with fan_out=2 is a normal tree
        while num_vars=[1,1,1,1] with fan_out=2 is a hypercube

        Possible rules for connecting the graph:
            'minimal' (default)
                only adds a new step if there exists a node at the same depth as source,
                such that labels(source) V labels(node) = labels(step)
            'matching' 
                creates all possible new steps, but only uses existing valid parents
            'maximal'
                creates all possible new steps, and all possible valid parents
        """

        self.fan_out = fan_out
        self.respect_hierarchy = respect_hierarchy

        self.depth = len(num_vars)
        self.num_vars = sum(num_vars)

        ######### Make the variable tree
        tot_var = np.cumsum([0,]+num_vars)
        nodes = [np.arange(tot_var[i],tot_var[i]+num_vars[i])+1 for i in range(self.depth)]

        # horrible,  awful list comprehensions, to keep things fast (?)
        var_labels = [nodes[0]]
        _ = [var_labels.append(self.label_nodes(var_labels[i], nodes[i+1])) \
            for i in range(self.depth-1)]
        var_labels = np.concatenate(var_labels)
        # print(children)
        # edges = [[p,c] for i in range(self.depth-1) for p,c in zip(np.repeat(children[i], fan_out), children[i+1])]
        
        var_num = fan_out**np.arange(self.depth)
        node_idx = np.split(np.arange(np.sum(var_num)), var_num- 1)[1:]
        edges = [[p,c] for i in range(self.depth-1) \
            for p,c in zip(np.repeat(node_idx[i],fan_out), node_idx[i+1])]

        # match nodes with var label
        nds = [(np.arange(np.sum(var_num))[i], {'var':int(np.floor(var_labels[i]))}) \
            for i in range(np.sum(var_num))]

        # turn into a networkx graph which can easily generate data
        var_graph = nx.DiGraph()
        var_graph.add_nodes_from(nds)
        var_graph.add_edges_from(edges)
        
        ######### Make the value tree
        # convert it into a feature-generator
        val_graph = nx.DiGraph()
        val_num = fan_out**np.arange(self.depth+1)
        node_idx = np.split(np.arange(np.sum(val_num)+1), val_num-1)[1:]
        val_edges = [[p,c] for i in range(self.depth) \
            for p,c in zip(np.repeat(node_idx[i],fan_out), node_idx[i+1])]

        # _ = [[[val_graph.add_edge(e[0]+a*1j, e[1]+b*1j) for b in range(self.fan_out)] \
        #     for a,e in enumerate(var_graph.out_edges(n))] for n in var_graph.nodes]
        val_labels = [{'var':n[1], 'val':b} for n in var_graph.nodes(data='var') \
        for b in range(self.fan_out) ]

        val_graph.add_nodes_from([(i,j) for i,j in zip(np.arange(1,np.sum(val_num)+1), val_labels)])
        val_graph.add_edges_from(val_edges)

        # roots = [node for node in val_graph.nodes() \
        #     if val_graph.out_degree(node)!=0 and val_graph.in_degree(node)==0]
        leaves = [node for node in val_graph.nodes() \
            if val_graph.in_degree(node)!=0 and val_graph.out_degree(node)==0]

        val_graph.add_node(0, var=0,val=0)
        # _ = [val_graph.add_edge(0, n) for n in roots]

        self.variable_tree = var_graph # tree of variables
        self.value_tree = val_graph # tree of variable labels 

        self.terminals = leaves

        # unique label combinations into categories
        varval = [(self.value_tree.nodes(data='var')[i],self.value_tree.nodes(data='val')[i]) \
            for i in sorted(self.value_tree.nodes)[1:]]
        cats = np.unique(varval, axis=0, return_inverse=True)[1]
        paths = [np.array(list(all_simple_paths(self.value_tree, 0, n))).squeeze() \
            for n in self.terminals]
        self.cats = [set([cats[i] for i in p[1:]-1]) for p in paths]
        self.num_cat = max([max(i) for i in self.cats])+1

        ###### Convert to similarity graph
        # if graph_rule == 'generic':
        self.items = np.arange(len(leaves))+1
        self.similarity_graph = self.fill_similarity_graph()
        # else:
        #     self.similarity_graph = self.fill_similarity_graph_from_tree(graph_rule)
        #     self.items = self.terminals 

        self.num_data = len(leaves)

    def labels(self, these_leaves):
        """ finds the label of each of 'these_leaves' """

        data = [np.array(list(all_simple_paths(self.value_tree, 0, n))).squeeze() for n in these_leaves]

        idx = np.array([self.value_tree.nodes(data='var')[n]-1 for d in data for n in d[1:]])
        val = np.array([self.value_tree.nodes(data='val')[n] for d in data for n in d[1:]])
        var = np.concatenate([np.ones(len(d)-1, dtype=int)*i for i,d in enumerate(data)],-1)

        labels = np.zeros((self.num_vars,var.max()+1))*np.nan
        labels[idx,var] = val

        return labels

    def fill_similarity_graph_from_tree(self, rule='minimal'):
        """ 
        DEPRECATED

        Make a graph so that distance between nodes matches distance in the labels
        
        Basic algorithm is:
            see if any two nodes share a label

        """

        # Make similarity graph
        G = nx.DiGraph()
        G.add_edges_from(self.value_tree.edges)
        # turn variable/value combinations into unique categories
        varval = [(self.value_tree.nodes(data='var')[i],self.value_tree.nodes(data='val')[i]) \
            for i in self.value_tree.nodes]
        cats = np.unique(varval, axis=0, return_inverse=True)[1]
        cats = {i:set([j]) for i,j in zip(self.value_tree.nodes, cats)}
        cats[0] = set()
        nx.set_node_attributes(G, values=cats, name='category')

        # first label all nodes according to ancestry
        node_cats = [] # vertex labels
        depth = []
        for node in sorted(G):
            if node==0:
                depth.append(0)
                node_cats.append(set())
                continue
            par_cats = set.union(*[(G.nodes('category')[j]) for j in G.predecessors(node)])
            new_cat = G.nodes('category')[node].union(par_cats)
            G.nodes[node]['category'] = new_cat
            node_cats.append(new_cat)
            depth.append(len(new_cat))
        depth = np.array(depth)

        # iterate over pairs of the same depth
        all_nodes = np.arange(len(G))
        for i in all_nodes: # shallower nodes first
            for j in all_nodes[(depth==depth[i])&(all_nodes>i)]:
                L_ij = node_cats[i].intersection(node_cats[j])

                if len(L_ij)<1: # check if they're related
                    continue

                # find or create nearest common ancestor

                all_cats = [G.nodes('category')[n] for n in sorted(G)]
                matching_nodes = np.isin(all_cats, L_ij)

                all_anc = nx.descendants(G.reverse(), i).intersection(nx.descendants(G.reverse(), j))
                anc_depths = [len(G.nodes('category')[a]) for a in all_anc]
                anc = list(all_anc)[np.argmax(anc_depths).item()]

                if np.any(matching_nodes): # there is an existing node
                    new_anc = sorted(G)[np.where(matching_nodes)[0].item()]
                    if np.any(np.isin(list(all_anc), new_anc)): # it's already connected
                        continue

                else:
                    new_anc = max(G)+1
                    G.add_node(new_anc, category=L_ij)

                # sanity checks
                if len(L_ij)-len(G.nodes('category')[anc]) == 1: # pretty sure it's always ==1
                    G.add_edge(anc, new_anc)
                elif len(L_ij) == len(G.nodes('category')[anc]):
                    print('OOSP!')
                    raise ValueError
                else: 
                    print('DOUBLE OOPS!') # but can't prove it
                    raise ValueError
                    print('%s, %s, %s'%(node_cats[i], node_cats[j], L_ij))
                    print([all_cats[anc]])
                # find/make a path from new NCA to each node
                self.path_to_node(G, new_anc, i, rule)
                self.path_to_node(G, new_anc, j, rule)

        return G

    def label_nodes(self, parents, children): 
        """ 
        connects parent nodes to children nodes 
        
        this function should be subject to the most change, I think
        """
        N = self.fan_out

        L1 = len(parents)
        L2 = len(children)

        n2 = N*L1
        # reps = (N*len(L1))/(len(L2))//N
        if L1 <= L2:
            reps = N
        else:
            # reps = N + N*((n2 - len(L2))//N)
            reps = int(np.ceil(n2/L2))
        # reps = N
        # n_s = (reps*len(L2) - N*len(L1))//reps
        # n_s = np.max([n_s, int(np.ceil((reps*len(L2) - n2 + n_s)/reps))])
        # n_dup = n2 - n_s # how many children are duplicated 
        n_dup = int(np.ceil((n2 - L2)/(reps-1)))*reps
        n_s = n2 - n_dup
        if self.respect_hierarchy:
            dups = np.repeat(children,reps) + np.mod(np.arange(reps*L2),reps)/reps
            c = np.append(dups[:n_dup],children[L2-n_s:])
        else: 
            c = np.tile(children, reps)[:n2] + (np.arange(n2)//L2)/reps
        # p = np.repeat(L1, N)
        # return [[p[i], c[i]] for i in range(n2)]
        return c


class LabelledItems(HierarchicalData):

    def __init__(self, num_item=None, num_lab=None, labels=None):
        """
        Either supply the number of conditions and labels, or just supply 
        a list of sets containing each item's labels

        If both are supplied, labels are used
        """

        if labels is not None:
            self.cats = 1*labels
            self.items = np.arange(len(labels)) + 1
            self.num_vars = max([max(i) for i in labels if len(i)>0]) + 1
            self.num_data = len(labels)
            self.depth = max([len(l) for l in labels])

            # assumes that each item has the same number of labels!!!
            self.fan_out = int(len(labels)**(1/len(labels[0])))

        elif num_item is not None:
            # not implemented yet, need to think a lil bit
            self.num_data = num_items
            # self.

        self.similarity_graph = self.fill_similarity_graph()

    def represent_labels(self, these_leaves):
        """ finds the label of each of 'these_leaves' """

        idx = np.array([ i for j in these_leaves for \
            i in self.similarity_graph.nodes('category')[j] ] )
        var = np.repeat(np.arange(len(these_leaves)), 
            [len(self.similarity_graph.nodes('category')[j]) for j in these_leaves])

        labels = np.zeros((self.num_vars,len(these_leaves)))
        labels[idx,var] = 1

        return labels


