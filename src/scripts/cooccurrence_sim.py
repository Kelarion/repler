import numpy as np
import scipy.sparse as sprs

n_group = 4
n_trial = 10000
n_obs_in_trial = 50
n_item_in_group = 100
overlap = 2


which_group = np.random.choice(range(n_group), n_trial)
foo = np.random.choice(range(overlap*n_item_in_group), (n_trial,n_obs_in_trial))
which_group_item = np.mod(which_group[:,None]*n_item_in_group + foo, n_item_in_group*n_group)
which_item = np.where(np.random.rand(n_trial,n_obs_in_trial)>0.0, 
                      which_group_item, 
                      np.random.choice(range(n_group*n_item_in_group), (n_trial, n_obs_in_trial)))

items = sprs.coo_matrix((np.ones(np.prod(which_item.shape)), 
                     [np.repeat(range(n_trial), n_obs_in_trial), which_item.flatten()] ))
items = items.tocsr() > 0

K = ((1*items).T@items).toarray()
