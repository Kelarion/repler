import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

#%%
class Encoder(nn.Module):
    
    def __init__(self, dim_z, dim_x, dim_hidden):
        
        super.__init__()
        
        