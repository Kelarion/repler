import os
import pickle
import warnings

import torch
import torchvision
import torch.optim as optim
import numpy as np
import scipy.special as spc
import scipy.linalg as la
import scipy.special as spc

# this is my code base, this assumes that you can access it
import students
import assistants
import util
from itertools import permutations
from sklearn import svm, linear_model

