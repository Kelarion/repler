import numpy as np
import scipy.stats as sts
import scipy.linalg as la

def seqify(X, Y, append=True):
    """
    Turn separate sequences of X and Y into on sequence of interleaved Xs and Ys
    X is shape (num_seq, len_seq, dimX)
    Y is shape (num_seq, len_seq, dimY)
    if append is true, puts X and Y into disjoint dimensions 
        (should be true if dimX != dimY)
    
    the output is shape (num_seq, 2*len_seq, dimX+dimy if separate else dimX)
    """

    N, T, dX = X.shape
    _, _, dY = Y.shape
    
    if append:
        y_fill = np.hstack([np.zeros((N, dX)), Y])
        x_fill = np.hstack([X, np.zeros((N, dY))])
        dSeq = dX + dY
    else:
        y_fill = Y
        x_fill = X
        dSeq = dX
        
    X_seq = np.empty((N, 2*T, dSeq))
    X_seq[:,0::2,...] = x_fill
    X_seq[:,1::2, ...] = y_fill
    
    return X_seq

class TransferTask:

    def __init__(self, X, Y, k, dim_embed=None, ordered=False, 
                 dom_per_seq=None, len_seq=None,
                 p_dom_trn=None, p_dom_tst=None,
                 p_inp_trn=None, p_inp_tst=None, 
                 autoregressive=False, base_dom=None):
        """
        Given a prototypical set of input-output pairs, generate many instances
        in a sequence of orthogonal subspaces.
        
        Generates k mutually orthogonal embedding matrices, A^{1}, ..., A^{k}
        Domain d consists of len_seq examples [A^{d}x_1, ..., A^{d}x_l]
        A sequence consists of dom_per_seq domains, [D_1, ..., D_n]
        
        X: shape (num_inputs, dim_X)
        Y: shape (num_inputs, dim_Y)
        k: integer number of domains
        
        optional params:
        dim_embed: dimension of embedding matrices .
            if None, just uses dim(X) and dim(Y), no rotation
        ordered: preserve the order of the inputs? if False, randomly samples
        dom_per_seq: number of domains per sequence (default 1)
        len_seq: number of inputs per domain (default num_inputs)
        p_dom_trn/tst: distribution over domains, must sum to 1 (default uniform)
        p_inp_trn/tst: distribution over inputs, must sum to 1 (default uniform)
        autoregressive: should the output interleave X and Y into one sequence?
        base_dom: a domain to always include at the beginning of test sequences
        """
        
        # parse args        
        if dom_per_seq is None:
            self.dom_per_seq = 1
        else:
            self.dom_per_seq = dom_per_seq
        
        if p_dom_trn is None:
            self.p_dom_trn = np.ones(k)/k
        else:
            self.p_dom_trn = p_dom_trn
        if p_dom_tst is None:
            self.p_dom_tst = np.ones(k)/k
        else:
            self.p_dom_tst = p_dom_tst
        if p_inp_trn is None:
            self.p_inp_trn = np.ones(len(X))/(len(X))
        else:
            self.p_inp_trn = p_inp_trn
        if p_inp_tst is None:
            self.p_inp_tst = np.ones(len(X))/(len(X))
        else:
            self.p_inp_tst = p_inp_tst
            
        if len_seq is None:
            self.len_seq = len(X)
        
        self.X = X  # prototypical X
        if len(Y.shape) == 1:
            self.Y = Y[:,None]
        else:
            self.Y = Y  # prototypical Y
        
        self.ordered = ordered
        self.autoreg = autoregressive
        self.base_dom = base_dom
        self.k = k
        self.nX, self.dX = X.shape
        self.dY = self.Y.shape[-1]
        
        self.nCond = self.nX*k
        
        X_dom = la.block_diag(*((self.X,)*k))
        Y_dom = np.tile(self.Y, [k,1])

        if dim_embed is not None:
            
            basis = sts.ortho_group.rvs(dim_embed, size=k)[...,:self.dX]
            basis = basis.swapaxes(-1,-2).reshape((self.dX*k,-1))
            X_dom = X_dom@basis
            
            self.dX = dim_embed
        
        self.X_dom = X_dom
        self.Y_dom = Y_dom

    def sample(self, split):
            
        T = self.len_seq*self.dom_per_seq
        
        ## Training set
        if split == 'train':
            # draw context
            dom = np.random.choice(np.arange(self.k), 
                                   size=self.dom_per_seq, 
                                   p=self.p_dom_trn)
            dom = np.repeat(dom, self.len_seq, axis=1)
    
            # draw condition
            if self.ordered:
                trn_inp = np.arange(self.nX)[self.p_inp_trn > 0]
                
                cond = np.repeat(trn_inp[:self.len_seq], self.dom_per_seq)
                # cond = np.repeat(cond[None,:], 1, axis=0)
            else:
                
                cond = np.random.choice(np.arange(self.nX), 
                                        size=T,
                                        p=self.p_inp_trn)
            cond = cond + self.nX*dom
            
        ## Test set
        elif split == 'test':
            dom = np.random.choice(np.arange(self.k), 
                                   size=self.dom_per_seq,
                                   p=self.p_dom_tst)
            
            if self.base_dom is not None:
                dom = np.append(self.base_dom, dom)
            
            dom = np.repeat(dom, self.len_seq, axis=1)
            
            # draw condition
            if self.ordered:
                tst_inp = np.arange(self.nX)[self.p_inp_tst > 0]
                
                cond = np.repeat(tst_inp[:self.len_seq], self.dom_per_seq)
            else:
                cond = np.random.choice(np.arange(self.nX), 
                                        size=T,
                                        p=self.p_inp_tst)
            cond = cond + self.nX*dom
        
        ## Make inputs
        X = self.X_dom[cond]
        Y = self.Y_dom[cond]
        
        if self.autoreg:
            X_seq = seqify(X, Y)
            Y_seq = np.roll(X_seq, -1, axis=1)
            
        else:
            X_seq = X
            Y_seq = Y
        
        samp = {'input': X_seq, 'target':Y_seq}
            
        return samp


