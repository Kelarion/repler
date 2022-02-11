import os, sys
import pickle

import torch
import torchvision
import torch.optim as optim
import numpy as np
import scipy
import scipy.linalg as la
import scipy.special as spc
import numpy.linalg as nla
from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations
import itertools as itt

def batch_data(*data, **dl_args):
	""" 
	wrapper for weird data batching in pytorch
	data supplied as positional arguments, must have same size in first dimension 
	"""
	dset = torch.utils.data.TensorDataset(*data)
	return torch.utils.data.DataLoader(dset, **dl_args)
