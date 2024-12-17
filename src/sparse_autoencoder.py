import os, sys, re
import pickle
from time import time

import os
import pickle
import warnings
import re
from time import time

import torch as t
from torch import nn, Tensor
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from functools import partial
from dataclasses import dataclass

device = t.device("cuda" if t.cuda.is_available() else "cpu")

from collections import OrderedDict
import numpy as np
import scipy
import scipy.linalg as la
import scipy.special as spc
import scipy.stats as sts

from sklearn.exceptions import ConvergenceWarning
import warnings # I hate convergence warnings so much never show them to me
warnings.simplefilter("ignore", category=ConvergenceWarning)


import students
import super_experiments as exp
import util
import pt_util


#%% Sparse Autoencoder

def constant_lr(*_):
    return 1.0

@dataclass
class AutoEncoderConfig:
    n_instances: int
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False
    
    neuron_resample_window: Optional[int] = None
    neuron_resample_scale: float = 0.2

class SparseAutoencoder(students.NeuralNet):
    """
    Sparse Autoencoder
    
    Modified from a tutorial by Jack Lindsey at CCN (sorry I didn't help more, Jack!)
    """
    
    W_enc: Float[Tensor, "n_instances n_input_ae n_hidden_ae"]
    W_dec: Float[Tensor, "n_instances n_hidden_ae n_input_ae"]
    b_enc: Float[Tensor, "n_instances n_hidden_ae"]
    b_dec: Float[Tensor, "n_instances n_input_ae"]

    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        
        self.cfg = cfg
        self.W_enc = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))))
        if not(cfg.tied_weights):
            self.W_dec = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))))
        self.b_enc = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_hidden_ae))
        self.b_dec = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_input_ae))

        ## For neuron resampling
        self.resamp = (cfg.neuron_resample_window is not None)
        self.window = [] # this will be filled to size neuron_resample_window 
        self.resamp_step = 0 # keep track of this separately for 

        self.to(device)

    def forward(self, h: Float[Tensor, "batch_size n_instances n_hidden"]):
        
        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = einops.einsum(
            acts, (self.W_enc.transpose(-1, -2) if self.cfg.tied_weights else self.W_dec),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + self.b_dec
        
        return h_reconstructed
        
    def hidden(self, h: Float[Tensor, "batch_size n_instances n_hidden"]):
        
        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)
        
        return acts
    
    def loss(self, batch):


        h = batch[0]

        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = einops.einsum(
            acts, (self.W_enc.transpose(-1, -2) if self.cfg.tied_weights else self.W_dec),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + self.b_dec
        
        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).mean(-1) # shape [batch_size n_instances]
        l1_loss = acts.abs().sum(-1) # shape [batch_size n_instances]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum() # scalar

        return loss, acts

    @t.no_grad()
    def normalize_decoder(self) -> None:
        '''
        Normalizes the decoder weights to have unit norm. If using tied weights, we we assume W_enc is used for both.
        '''
        if self.cfg.tied_weights:
            self.W_enc.data = self.W_enc.data / self.W_enc.data.norm(dim=1, keepdim=True)
        else:
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=2, keepdim=True)

    @t.no_grad()
    def resample_neurons(
        self,
        frac_active_in_window: Float[Tensor, "window n_instances n_hidden_ae"],
        neuron_resample_scale: float = 1,
    ) -> None:
        '''
        Resamples neurons that have been dead for 'dead_neuron_window' steps, according to `frac_active`.
        '''
        # Get a tensor of dead neurons
        dead_features_mask = frac_active_in_window.sum(0) < 1e-8 # shape [instances hidden_ae]
        n_dead = dead_features_mask.int().sum().item()
    
        # Get our random replacement values
        replacement_values = t.randn((n_dead, self.cfg.n_input_ae), device=self.W_enc.device) # shape [n_dead n_input_ae]
        replacement_values_normalized = neuron_resample_scale * replacement_values / (replacement_values.norm(dim=-1, keepdim=True) + 1e-8)
    
        # Change the corresponding values in W_enc, W_dec, and b_enc (note we transpose W_enc to return a view with correct shape)
        self.W_enc.data.transpose(-1, -2)[dead_features_mask] = replacement_values_normalized
        self.W_dec.data[dead_features_mask] = replacement_values_normalized
        self.b_enc.data[dead_features_mask] = 0.0

    def grad_step(self, dl, **opt_args):
        """ Needs init_optimizer """

        if not self.initialized:
            self.init_optimizer(**opt_args)

        running_loss = 0
        for i, batch in enumerate(dl):
            self.optimizer.zero_grad()
            
            # Normalize the decoder weights before each optimization step
            self.normalize_decoder()
            
            # Resample dead neurons
            if self.resamp and (len(self.window) >= self.cfg.neuron_resample_window):
                # Get the fraction of neurons active in the previous window
                frac_window = t.stack(self.window, dim=0)
                
                # Resample
                self.resample_neurons(frac_window, self.cfg.neuron_resample_scale)
                
                self.window = []
        
            loss, acts = self.loss(batch)

            # optimise
            loss.backward()
            self.optimizer.step()
            
            # append active fraction
            frac_act = (einops.reduce((acts.abs() > 1e-8).float(), 
                                      "batch_size instances hidden_ae -> instances hidden_ae", "mean"))
            self.window.append(frac_act)
            
            running_loss += loss.item()
            
        return running_loss/(i+1)
