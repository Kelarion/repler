"""
Classes used in the Remember-Forget experiments.
 
Includes:
    - RNNModel: class that currently supports {'LSTM', 'GRU', 'tanh', 'relu','tanh-GRU', 'relu-GRU'}
    recurrent neural networks. includes a save, load, and train method.
    - stateDecoder: does the memory decoding. includes a train method.

"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import scipy.linalg as la
import scipy.special as spc
from itertools import combinations

from assistants import Indicator, ContextIndicator

#%%
class RNNModel(nn.Module):
    """ 
    Skeleton for a couple RNNs 
    Namely: rnn_type = {'LSTM', 'GRU', 'tanh', 'relu', 'tanh-GRU', 'relu-GRU'}
    The final two (tanh-GRU and relu-GRU) use the custom GRU class.
    """
    
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, nout=None,
        embed=False, persistent=False, padding=-1):
        super(RNNModel,self).__init__()

        if nout is None:
            nout = ntoken
        if embed:
            self.encoder = nn.Embedding(ntoken, ninp, padding_idx=padding)
        else:
            if persistent:
                self.encoder = ContextIndicator(ntoken, ninp, padding_idx=padding) # defined below
            else:
                self.encoder = Indicator(ntoken, ninp, padding_idx=padding) # defined below
            
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        elif rnn_type in ['tanh-GRU', 'relu-GRU']:
            nlin = getattr(torch, rnn_type.split('-GRU')[0])
            self.rnn = CustomGRU(ninp, nhid, nlayers, nonlinearity=nlin) # defined below
        else:
            try:
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=rnn_type)
            except:
                raise ValueError("Invalid rnn_type: give from {'LSTM', 'GRU', 'tanh', 'relu'}")

        self.decoder = nn.Linear(nhid, nout)
        # self.softmax = nn.LogSoftmax(dim=2)
        
        self.embed = embed
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.padding = padding

    def init_weights(self):
        if self.embed:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_()

    def forward(self, input, hidden, give_gates=False, debug=False):
        """
        Run the RNN forward. Expects input to be (lseq,nseq,...)
        Only set give_gates=True if it's the custom GRU!!!
        use `debug` argument to return also the embedding the input
        """

        emb = self.encoder(input)
        if emb.dim()<3:
            emb = emb.unsqueeze(0)

        if give_gates:
            output, hidden, extras = self.rnn(emb, hidden, give_gates)
        else:
            output, hidden = self.rnn(emb, hidden)

        # decoded = self.softmax(self.decoder(output))
        decoded = self.decoder(output)

        if give_gates:
            if debug:
                return decoded, hidden, extras, emb
            else:
                return decoded, hidden, extras
        else:
            if debug:
                return decoded, hidden, emb
            else:
                return decoded, hidden

    def transparent_forward(self, input, hidden, give_gates=False, debug=False):
        """
        Run the RNNs forward function, but returning hidden activity throughout the sequence

        it's slower than regular forward, but often necessary
        """

        lseq, nseq = input.shape
        ispad = (input == self.padding)

        H = torch.zeros(lseq, self.nhid, nseq)
        if give_gates:
            Z = torch.zeros(lseq, self.nhid, nseq)
            R = torch.zeros(lseq, self.nhid, nseq)
        
        # because pytorch only returns hidden activity in the last time step,
        # we need to unroll it manually. 
        O = torch.zeros(lseq, nseq, self.decoder.out_features)
        emb = self.encoder(input)
        for t in range(lseq):
            if give_gates:
                out, hidden, ZR = self.rnn(emb[t:t+1,...], hidden, give_gates=True)
                Z[t,:,:] = ZR[0].squeeze(0).T
                R[t,:,:] = ZR[1].squeeze(0).T
            else:
                out, hidden = self.rnn(emb[t:t+1,...], hidden)
            dec = self.decoder(out)
            # naan = torch.ones(hidden.squeeze(0).shape)*np.nan
            # H[t,:,:] = torch.where(~ispad[t:t+1,:].T, hidden.squeeze(0), naan).T
            H[t,:,:] = hidden.squeeze(0).T
            O[t,:,:] = dec.squeeze(0)

        if give_gates:
            if debug:
                return O, H, Z, R, emb
            else:
                return O, H, Z, R
        else:
            if debug:
                return O, H, emb
            else:
                return O, H

    def test_inputs(self, seqs, padding=-1):
        """ 
        for debugging purposes: convert seqs to appropriate type and
        shape to give into rnn.forward(), and also run init_hidden

        seqs should be (nseq, lseq), i.e. output of make_dset or draw_seqs
        """
        inp = torch.tensor(seqs.T).type(torch.LongTensor)
        inp[inp==padding] = 0

        hid = self.init_hidden(seqs.shape[0])

        return inp, hid

    def init_hidden(self, bsz):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(1, bsz, self.nhid),
                    torch.zeros(1, bsz, self.nhid))
        else:
            return torch.zeros(1, bsz, self.nhid)

    def save(self, to_path):
        """
        save model parameters to path
        """
        with open(to_path, 'wb') as f:
            torch.save(self.state_dict(), f)
    
    def load(self, from_path):
        """
        load parameters into model
        """
        with open(from_path, 'rb') as f:
            self.load_state_dict(torch.load(f))

    ### specific for our task (??)
    def train(self, X, Y, optparams, dlparams, algo=optim.SGD,
        criterion=nn.CrossEntropyLoss(), nepoch=1000, do_print=True,
        epsilon=0, test_data=None, save_params=False):
        """
        Train rnn on data X and labels Y (both torch tensors).
        X, Y need to have samples as the FIRST dimension

        supply test data as (X,Y), optionally
        """

        if type(criterion) is torch.nn.modules.loss.BCEWithLogitsLoss:
            self.q_ = int(torch.sum(criterion.weight>0))
        else:
            self.q_ = self.decoder.out_features
        
        self.optimizer = algo(self.parameters(), **optparams)
        padding = self.padding

        dset = torch.utils.data.TensorDataset(X, Y)
        trainloader = torch.utils.data.DataLoader(dset, **dlparams)

        self.init_metrics()

        # loss_ = np.zeros(0)
        # test_loss_ = np.zeros(0)
        prev_loss = 0
        for epoch in range(nepoch):
            running_loss = 0.0
            
            for i, batch in enumerate(trainloader, 0):
                # unpack
                btch, labs = batch
                btch = btch.transpose(1,0) # (lenseq, nseq, ninp)
                
                # initialise
                self.optimizer.zero_grad()
                hidden = self.init_hidden(btch.size(1))
                
                # forward -> backward -> optimize

                # we're going to train on the final non-padding time point
                t_final = -(np.flipud(btch!=self.padding).argmax(0)+1) # index of final time point
                # btch[btch==self.padding] = 0 # set to some arbitrary value

                out, hidden = self(btch, hidden)
                output = out[t_final, np.arange(btch.size(1)), :]

                loss = criterion(output.squeeze(0),labs.squeeze()) # | || || |_
                loss.backward()

                self.optimizer.step()

                # update loss
                running_loss += loss.item() 
                # self.metrics['train_loss'] = np.append(self.metrics['train_loss'], loss.item())
            
            # compute the metrics at each epoch of learning
            idx = np.random.choice(X.size(0),np.min([1500, X.size(0)]), replace=False)
            self.compute_metrics((X[idx,:].T, Y[idx]), test_data, criterion)
            self.metrics['train_loss'] = np.append(self.metrics['train_loss'], running_loss/(i+1))

            if save_params:
                thisdir = '/home/matteo/Documents/github/rememberforget/results/justremember/'
                self.save(thisdir+'params_epoch%d.pt'%epoch)

            #print loss
            if (epsilon>0) and (np.abs(running_loss-prev_loss) <= epsilon):
                print('~'*5)
                print('[%d] Converged at loss = %0.3f'%(epoch+1, running_loss/(i+1)))
                return
                # return loss_, metrics
            # print to screen
            if do_print:
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss / i))
                running_loss = 0.0
                prev_loss = running_loss

        print('[%d] Finished at loss = %0.3f'%(epoch+1, running_loss/(i+1)))
        print('~'*5)
        # return loss_

    # for computing metrics
    def init_metrics(self):
        """
        Initialise the various metrics to be computed during learning
        """

        self.metrics = {}

        self.metrics['train_loss'] = np.zeros(0)
        self.metrics['test_loss'] = np.zeros(0)

        # self.orth_clf = LinearDecoder(self, self.q_, MeanClassifier)
        # self.metrics['train_orthogonality'] = np.zeros(0)
        # self.metrics['test_orthogonality'] = np.zeros(0)

        self.metrics['train_parallelism'] = np.zeros((0,self.q_)) 
        self.metrics['test_parallelism'] = np.zeros((0,self.q_))

    def compute_metrics(self, train_data, test_data, criterion):
        """
        Compute the various metrics on the train and test data. Can be done at any point in training.
        """
        m = self.metrics
        warnings.filterwarnings('ignore','Mean of empty slice')

        ## load data
        trn, trn_labs = train_data
        tst, tst_labs = test_data

        # trn = trn.transpose(1,0)
        tst = tst.transpose(1,0)

        t_final = -(np.flipud(trn!=self.padding).argmax(0)+1)
        test_tfinal = -(np.flipud(tst!=self.padding).argmax(0)+1)

        ntest = tst.size(1)
        P = self.decoder.out_features

        ## training data ###########################################################
        # hidden = self.init_hidden(trn.size(1))
        # out, hidden = self.transparent_forward(trn, hidden)
        # # output = out[t_final, np.arange(trn.size(1)), :]
        # output = out.squeeze()
        # # compute orthogonality
        # mem_act = np.array([np.cumsum(trn==p,axis=0).int().detach().numpy() % 2 \
        #     for p in range(self.q_)]).transpose((1,2,0))

        # ps_clf = LinearDecoder(self, 2**(self.q_-1), MeanClassifier)
        # ps = []
        # for d in Dichotomies(mem_act, 'simple'):
        #     np.warnings.filterwarnings('ignore',message='invalid value encountered in')
        #     ps_clf.fit(hidden.detach().numpy(), d)
        #     new_ps = ps_clf.orthogonality()
        #     ps.append(new_ps)
        #     # if new_ps > ps:
        #     #     ps = new_ps
        # m['train_parallelism'] = np.append(m['train_parallelism'], np.array(ps).T, axis=0)

        # # print(mem_act.shape)
        # # print(hidden.shape)
        # # self.orth_clf.fit(hidden.detach().numpy(), mem_act)
        # # orth_score = self.orth_clf.orthogonality()
        # # m['train_orthogonality'] = np.append(m['train_orthogonality'], orth_score)

        ## test data ##############################################################
        hidden = self.init_hidden(tst.size(1))
        out, hidden = self.transparent_forward(tst, hidden)
        # output = out.squeeze()
        # print(hidden.shape)
        # print(out.shape)
        # print(test_tfinal)
        output = out[test_tfinal, np.arange(tst.size(1)), :]
        # raise Exception

        # compute loss
        test_loss = criterion(output.squeeze(0),tst_labs.squeeze())

        m['test_loss'] = np.append(m['test_loss'], test_loss.item())

        # compute orthogonality
        # mem_act = np.array([np.cumsum(tst==p,axis=0).int().detach().numpy() % 2 \
        #     for p in range(self.q_)]).transpose((1,2,0))

        # # self.orth_clf.fit(hidden.detach().numpy(), mem_act)
        # # orth_score = self.orth_clf.orthogonality()
        # # m['test_orthogonality'] = np.append(m['test_orthogonality'], orth_score)

        # # compute parallelism
        # ps_clf = LinearDecoder(self, 2**(self.q_-1), MeanClassifier)
        # ps = []
        # for d in Dichotomies(mem_act, 'simple'):
        #     np.warnings.filterwarnings('ignore',message='invalid value encountered in')
        #     ps_clf.fit(hidden.detach().numpy(), d)
        #     new_ps = ps_clf.orthogonality()
        #     ps.append(new_ps)
        #     # if new_ps > ps:
        #     #     ps = new_ps
        # m['test_parallelism'] = np.append(m['test_parallelism'], np.array(ps).T, axis=0)

        ## package #################################################################
        self.metrics = m
        warnings.filterwarnings('default')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Custom GRU, originally by Miguel but substantially changed
class CustomGRU(nn.Module):
    """
    A GRU class which gives access to the gate activations during a forward pass

    Supposed to mimic the organisation of torch.nn.GRU -- same parameter names
    """
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity=torch.tanh):
        """Mimics the nn.GRU module. Currently num_layers is not supported"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input weights
        self.weight_ih_l0 = Parameter(torch.Tensor(3*hidden_size, input_size))

        # hidden weights
        self.weight_hh_l0 = Parameter(torch.Tensor(3*hidden_size, hidden_size))

        # bias
        self.bias_ih_l0 = Parameter(torch.Tensor(3*hidden_size)) # input
        self.bias_hh_l0 = Parameter(torch.Tensor(3*hidden_size)) # hidden

        self.f = nonlinearity

        self.init_weights()

    def __repr__(self):
        return "CustomGRU(%d,%d)"%(self.input_size,self.hidden_size)

    def init_weights(self):
        for p in self.parameters():
            k = np.sqrt(self.hidden_size)
            nn.init.uniform_(p.data, -k, k)

    def forward(self, x, init_state, give_gates=False):
        """Assumes x is of shape (len_seq, batch, input_size)"""
        seq_sz, bs, _ = x.size()

        update_gates = torch.empty(seq_sz, bs, self.hidden_size)
        reset_gates = torch.empty(seq_sz, bs, self.hidden_size)
        hidden_states = torch.empty(seq_sz, bs, self.hidden_size)

        h_t = init_state

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[t, :, :]

            gi = F.linear(x_t, self.weight_ih_l0, self.bias_ih_l0) # do the matmul all together
            gh = F.linear(h_t, self.weight_hh_l0, self.bias_hh_l0)

            i_r, i_z, i_n = gi.chunk(3,1) # input currents
            h_r, h_z, h_n = gh.chunk(3,2) # hidden currents

            r_t = torch.sigmoid(i_r + h_r)
            z_t = torch.sigmoid(i_z + h_z)
            n = self.f(i_n + r_t*h_n)
            h_t = n + z_t*(h_t - n)

            update_gates[t,:,:] = z_t
            reset_gates[t,:,:] = r_t
            hidden_states[t,:,:] = h_t

        output = hidden_states

        if give_gates:
            return output, h_t, (update_gates, reset_gates)
        else:
            return output, h_t
