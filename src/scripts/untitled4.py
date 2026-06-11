CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/jeff/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/swap_errors/data/pickles/'

import os, sys, re
import pickle as pkl
from time import time
import math
import matplotlib.pyplot as plt

sys.path.append(CODE_DIR)
sys.path.append(CODE_DIR + 'swap_errors')


#%%

import numpy as np
from sklearn.preprocessing import StandardScaler

def _norm_periodic(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def _bin_kernel(D, c_row, c_col, row_mask, col_mask, bins):
    d = D[row_mask][:, col_mask].ravel()
    dc = _norm_periodic(c_col[col_mask][None, :] - c_row[row_mask][:, None]).ravel()
    valid = ~np.isnan(d)
    num, _ = np.histogram(dc[valid], bins, weights=d[valid])
    cnt, _ = np.histogram(dc[valid], bins)
    return np.where(cnt > 0, num / cnt, np.nan)

def compute_kernel_tc(spks, uc, lc, cues, ps, rc=None, p_thr=0.3, n_bins=5, t_ref=None):
    """
    spks : (trials, dims, time_points)
    uc, lc, cues, rc : (trials,)
    ps   : (trials, n_cats)  — col 0=correct, col 2=guess
    Returns k_c, k_g, k_rc — each (n_bins, n_time_points) — and bin_cents
    """
    bins = np.linspace(-np.pi, np.pi, n_bins + 1) 
    bin_cents = (bins[:-1] + bins[1:]) / 2
    n_times = spks.shape[2]
    k_c, k_g, k_rc = (np.zeros((n_bins, n_times)) for _ in range(3))
    row_corr = ps[:, 0] > p_thr
    
    if t_ref is not None:
        r_ref = spks[...,t_ref]
    
    for t in range(n_times):
        r = StandardScaler().fit_transform(spks[:, :, t])
        if t_ref is None:
            r_ref = r
        else:
            r_ref = StandardScaler().fit_transform(spks[:, :, t_ref])
        D = r_ref @ r.T
        np.fill_diagonal(D, np.nan)

        for k_arr, col_ind, c_col_override, same_cue in [
            (k_c,  0, None, False),
            (k_g,  2, None, False),
            (k_rc, 2, rc,   True ),
        ]:
            halves = []
            for cue_val, color in ((1, uc), (0, lc)):
                col_mask = (ps[:, col_ind] > p_thr) & (cues == cue_val)
                row_mask = row_corr & (cues == cue_val) if same_cue else row_corr
                c_col = color if c_col_override is None else c_col_override
                halves.append(_bin_kernel(D, color, c_col, row_mask, col_mask, bins))
            k_arr[:, t] = np.nanmean(halves, axis=0)

    return k_c, k_g, k_rc, bin_cents

#%%

task = "retro"

# epoch = 'pre-cue'
epoch = "post-cue"
# epoch = "color"
# epoch = "wheel"

monkey = "elmo"
# monkey = "waldorf"

n_bins = 10

# t_ref = 18
t_ref = 12
# t_ref = None

monk2sess = {"elmo": list(range(13)),
             "waldorf": list(range(13,23)),
             }

kerns = []
for i,sess in enumerate(monk2sess[monkey]):
    
    ## get cues first
    fname = f"lmtc_{task}_all_post-cue-presentation_{sess}.pkl"
    dat = pkl.load(open(LOAD_DIR + fname, 'rb'))
    cues = dat['cues']
    
    ## Actual data 
    fname = f"lmtc_{task}_all_{epoch}-presentation_{sess}.pkl"
    dat = pkl.load(open(LOAD_DIR + fname, 'rb'))
    
    k_c, k_g, k_rc, bs = compute_kernel_tc(dat['spks'], 
                                           dat['uc'], 
                                           dat['lc'], 
                                           cues, 
                                           dat['ps'], 
                                           rc=dat['rc'],
                                           n_bins=n_bins,
                                           t_ref=t_ref,
                                           )
    
    ind0 = np.argmin(np.abs(bs))
    ind1 = np.argmax(np.abs(bs))
    
    kerns.append( np.stack([k_c[ind0] - k_c[ind1], 
                             k_g[ind0] - k_g[ind1], 
                             k_rc[ind0] - k_rc[ind1]]))
    
    
#%%

k = np.mean(kerns, axis=0)

cols = ['r', 'b', 'g']

for i in range(3):
    # plt.plot(dat['other']['xs'], k[i], color=cols[i])
    plt.plot(dat['other']['xs'], k[i], '--', color=cols[i])
    # plt.plot(dat['other']['xs'], k[i], '-.', color=cols[i])


