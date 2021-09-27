
import socket
import os
import sys

if socket.gethostname() == 'kelarion':
    if sys.platform == 'linux':
        CODE_DIR = '/home/kelarion/github/repler/src'
        SAVE_DIR = '/mnt/c/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    else:
        CODE_DIR = 'C:/Users/mmall/Documents/github/repler/src/'
        SAVE_DIR = 'C:/Users/mmall/Documents/uni/columbia/multiclassification/saves/'
    openmind = False
elif socket.gethostname() == 'openmind7':
    CODE_DIR = '/home/malleman/repler/'
    SAVE_DIR = '/om2/user/malleman/abstraction/'
    openmind = True
else:    
    CODE_DIR = '/rigel/home/ma3811/repler/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
    openmind = False

sys.path.append(CODE_DIR)

import re
import pickle
import warnings

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as anime
from itertools import permutations, combinations
from sklearn import svm, manifold, linear_model
from tqdm import tqdm

# this is my code base, this assumes that you can access it
import students
import assistants
import util
import recurrent
import experiments as exp
import plotting as dicplt

#%%
class sanityRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out):
        super(sanityRNN,self).__init__()
        
        self.rnn = nn.RNN(n_inp, n_hid, nonlinearity='relu')
        self.decoder = nn.Linear(n_hid,n_out)
        self.n_hid = n_hid
        
    def forward(self, inps):
        
        _, out = self.rnn(inps, torch.zeros(1, inps.shape[1], self.n_hid))
        
        return self.decoder(out)
    
#%%

input_task = util.RandomDichotomies(d=[(0,1,2,3),(0,2,4,6),(0,1,4,5)])
output_task = util.RandomDichotomies(d=[(0,3,5,6)]) # 3d xor

this_exp = exp.delayed_logic(input_channels=1,
                            task=output_task, 
                            input_task=input_task,
                            SAVE_DIR=SAVE_DIR,
                            time_between=20)

num_data = this_exp.ntrain

inputs = this_exp.train_data[0]

outputs = this_exp.train_data[1]

which_inp = inputs.abs().cumsum(1).detach().numpy()

#%%
net = sanityRNN(1, 50, 1)

optimizer = optim.Adam(net.rnn.parameters())
dset = torch.utils.data.TensorDataset(inputs.float(), outputs.float())
dl = torch.utils.data.DataLoader(dset, batch_size=200, shuffle=True)


ls = []
for epoch in tqdm(range(2000)):
    losses = []
    for nums, labels in dl:
        optimizer.zero_grad()
        out = net(nums.transpose(0,1))
        loss = nn.BCEWithLogitsLoss()(out.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    ls.append(np.mean(losses))


#%%












