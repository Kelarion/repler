CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/shah_data/data/'

import os, sys, glob, itertools
sys.path.append(CODE_DIR)

import numpy as np
import pickle as pkl

import scipy.io as sio
import scipy.ndimage
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

import util
import df_util
import bae_util
import old_bae_models
import bae_models
import plotting as tpl


#%%
# ----------------------------------------------------------------------------
# Shah et al. (2024) -- "Pseudo-linear Summation explains Neural Geometry of
# Multi-finger Movements in Human Premotor Cortex". Single human participant
# (T5), Utah arrays in (pre)motor cortex; neural feature = binned threshold-
# crossing spike counts per electrode.
#
# This rebuilds the *neural representation arrays* that go into Figure 3:
#   3B -- 38 natural gestures           (data/natural_gestures.mat)
#   3C -- 80 multi-finger combinations  (data/t5.2021.07.08/*.mat)
#
# Two output variants per dataset:
#
#  (1) 'faithful' -- the paper's exact pipeline (verbatim from "figure 3B,
#      C.ipynb", the input to its linear-SVM decoders). The only edits are
#      np.int -> int and building the combination-dict key from python ints
#      (newer numpy prints np.int64 in repr, which broke the lookup).
#        B: per-trial Gaussian smoothing -> drop 5 noisy electrodes -> drop
#           electrodes with mean FR < 10 Hz -> per-block z-score -> preprocess().
#        C: per-block z-score -> preprocess().   (no smoothing / channel pruning)
#      preprocess() windows in time, rotates electrodes by full-rank PCA (a
#      *centred* rotation -- no dim reduction with n_elec_pcs = #electrodes),
#      then sums into n_t_bins (=1, i.e. over the whole trial). Both the
#      z-score and the PCA centring make the representation mean-subtracted
#      (signed), which is why 'faithful' has negative entries.
#
#  (2) 'nonneg' -- same normalization scale, but mean-subtraction removed so
#      the representation stays in the non-negative orthant:
#        - skip the PCA rotation, stay in electrode space (full-rank PCA is a
#          centred rotation; dropping it leaves Fig-3 decoding unchanged, since
#          a linear SVM is invariant to a centred rotation);
#        - per-block normalization divides by the std but does NOT subtract the
#          mean (preserves the paper's block gain-equalisation).
#      For the gestures the source is genuine spike counts, so this is already
#      >= 0 and anchored at true zero firing. The finger-combo source Y_batches
#      is delivered ALREADY z-scored (mean ~0, ~84% of values < 0), so there is
#      no raw firing to recover; we re-anchor each electrode by subtracting its
#      empirical minimum (its zero-firing floor) -- a per-feature constant that
#      leaves the condition geometry (pairwise distances) unchanged.
#
# Saved to SAVE_DIR/shah_fig3_representations.pkl: for each dataset the single-
# trial decoder input Y_2d (n_trials x n_features) and per-trial labels, plus
# the condition-averaged representation reps (n_conditions x n_features) -- the
# trial-averaged neural geometry -- in both 'faithful' and 'nonneg' forms.
# ----------------------------------------------------------------------------


def preprocess(Y_use, t_start, t_end, n_t_bins, n_elec_pcs):
    """Notebook's preprocess(): time-window, full-rank PCA over electrodes,
    then sum activity into n_t_bins time bins. Returns (trials, n_t_bins, pcs)."""
    Y_use_ = Y_use[:, t_start:t_end, :]

    shp = Y_use_.shape
    Y_use_ = np.reshape(Y_use_, [-1, shp[2]])
    Y_use_ = PCA(n_components=int(n_elec_pcs)).fit_transform(Y_use_)
    Y_use_ = np.reshape(Y_use_, [shp[0], shp[1], -1])

    n_t_bins = int(n_t_bins)
    Y_use_final = np.zeros((Y_use_.shape[0], n_t_bins, Y_use_.shape[2]))
    delta_T = Y_use_.shape[1] / n_t_bins
    for ibin in range(n_t_bins):
        Y_use_final[:, ibin, :] = Y_use_[:, int(ibin * delta_T):int((ibin + 1) * delta_T), :].sum(1)
    return Y_use_final


def bin_time(Y_use, t_start, t_end, n_t_bins):
    """Like preprocess() but WITHOUT the PCA rotation -- stays in electrode
    space so non-negativity is preserved. Returns (trials, n_t_bins, elec)."""
    Y_use_ = Y_use[:, t_start:t_end, :]
    n_t_bins = int(n_t_bins)
    out = np.zeros((Y_use_.shape[0], n_t_bins, Y_use_.shape[2]))
    delta_T = Y_use_.shape[1] / n_t_bins
    for ibin in range(n_t_bins):
        out[:, ibin, :] = Y_use_[:, int(ibin * delta_T):int((ibin + 1) * delta_T), :].sum(1)
    return out


def normalize_blocks(Y, block, center=True):
    """Per-block, per-electrode normalization. center=True -> z-score (paper's
    pipeline). center=False -> divide by std only, no mean-subtraction (keeps
    the same scale but preserves non-negativity of non-negative inputs)."""
    Y = Y.copy()
    for b in np.unique(block):
        sel = block == b
        Ys = Y[sel]
        Y2d = np.reshape(Ys, [-1, Ys.shape[-1]])
        sd = np.sqrt(np.var(Y2d, 0))
        Y2d = (Y2d - Y2d.mean(0)) / sd if center else Y2d / sd
        Y[sel] = np.reshape(Y2d, Ys.shape)
    return Y


def condition_average(Y_2d, labels):
    """Trial-average the (n_trials x n_features) representation within each
    unique label. Returns (uniq_labels, reps) with reps (n_cond x n_features)."""
    uniq = np.unique(labels)
    reps = np.stack([Y_2d[labels == c].mean(0) for c in uniq])
    return uniq, reps


#%%
# ============================================================================
# Figure 3B -- 38 natural gestures
# ============================================================================

data = sio.loadmat(os.path.join(LOAD_DIR, 'natural_gestures.mat'))

Y = data['Y']
Z = np.squeeze(data['Z'])
block_ = np.squeeze(data['block_batches'])
block = np.squeeze(np.array([block_[itr][0] for itr in range(len(block_))]))

gestures = {1: "Idle",
    2: "SignA", 3: "SignB", 4: "SignC", 5: "SignK", 6: "SignO", 7: "SignR",
    8: "SignS", 9: "SignT", 10: "SignU", 11: "SignV", 12: "SignW", 13: "SignY",
    14: "SignYOLO", 15: "AllFingerExtension", 16: "AllFingerFlexion",
    17: "AllFingerThumbExtension", 18: "AllFingerThumbFlexion",
    19: "IndexFingerExtension", 20: "IndexFingerFlexion", 21: "LittleFingerExtension",
    22: "LittleFingerFlexion", 23: "MiddleFingerExtension", 24: "MiddleFingerFlexion",
    25: "RingFingerExtension", 26: "RingFingerFlexion", 27: "ThumbExtension",
    28: "ThumbFlexion", 29: "SignAbduction", 30: "SignAlternateFingers", 31: "SignD",
    32: "SignE", 33: "SignF", 34: "SignG", 35: "SignH", 36: "SignI", 37: "SignL",
    38: "SignThumbsUp", 39: "SignVulcan"}

# Drop "Idle" (Z==1); gestures 2..39 -> the 38 natural gestures.
selected_trials = np.where(Z >= 2)[0]
gest_labels = Z[selected_trials]                  # gesture id per trial

# Per-trial Gaussian smoothing of each electrode's spike-count time series
# (sigma=2 bins; the notebook drops the first time sample with YY[1:, ielec]).
Y_use, block_use = [], []
for itr in selected_trials:
    YY = Y[0, itr]
    YYY = [scipy.ndimage.gaussian_filter1d(YY[1:, ielec].astype(np.float32), sigma=2)
           for ielec in range(YY.shape[1])]
    Y_use.append(np.array(YYY).T)
    block_use.append(block[itr])
Y_use = np.array(Y_use)                           # (n_trials, time, 192)
block_use = np.squeeze(np.array(block_use))

# Remove 5 noisy electrodes, then electrodes below 10 Hz mean firing rate.
remove_idx = [2, 16, 40, 55, 88]
remaining_idx = np.setdiff1d(np.arange(Y_use.shape[2]), remove_idx)
Y_use = Y_use[:, :, remaining_idx]
fr = Y_use.mean(0).mean(0) / (10 / 1000)          # mean count per 10 ms bin -> Hz
gest_Y_raw = Y_use[:, :, fr > 10]                 # >= 0 (smoothed spike counts)

p_final = [0, gest_Y_raw.shape[1], 1, gest_Y_raw.shape[-1]]

# (1) faithful: per-block z-score -> full-rank PCA -> sum over trial.
Yg = normalize_blocks(gest_Y_raw, block_use, center=True)
gest_Y_2d = np.reshape(preprocess(Yg, *p_final), [gest_Y_raw.shape[0], -1])
gest_cond, gest_reps = condition_average(gest_Y_2d, gest_labels)

# (2) nonneg: per-block divide-by-std (no centring) -> sum over trial, no PCA.
Yg_nn = normalize_blocks(gest_Y_raw, block_use, center=False)
gest_Y_2d_nn = np.reshape(bin_time(Yg_nn, *p_final[:3]), [gest_Y_raw.shape[0], -1])
gest_cond_nn, gest_reps_nn = condition_average(gest_Y_2d_nn, gest_labels)

gest_names = [gestures[int(c)] for c in gest_cond]
print('[gestures] faithful reps', gest_reps.shape, 'min %.3f' % gest_reps.min(),
      '| nonneg reps', gest_reps_nn.shape, 'min %.3f' % gest_reps_nn.min())


#%%
# ============================================================================
# Figure 3C -- 80 multi-finger combinations
# ============================================================================

fc_dir = os.path.join(LOAD_DIR, 't5.2021.07.08/')
files = sorted(os.listdir(fc_dir))

X, Y, Z, block_id = [], [], [], []
for iifile, ifile in enumerate(files):
    dat = sio.loadmat(os.path.join(fc_dir, ifile))
    for ibatch in range(dat['X_batches'].shape[1]):
        # Over-long batches are corrupted examples at the start of a block.
        if dat['X_batches'][0, ibatch].shape[0] > 200:
            continue
        X.append(dat['X_batches'][0, ibatch])
        Y.append(dat['Y_batches'][0, ibatch])
        Z.append(dat['target_batches'][0, ibatch])
        block_id.append(iifile)
block_id = np.array(block_id)

# Trim every example to the shortest trial length, then stack.
min_len = np.min([x.shape[0] for x in X])
Y = np.array([y[:min_len, ...] for y in Y])       # (n_trials, time, 192) -- already z-scored at source
Z = np.array([z[:min_len, ...] for z in Z])       # (n_trials, time, 5) finger targets
fc_Y_raw = Y                                       # keep pre-normalization copy


def discretize_z(z):
    """Map a continuous finger target (centred at 0.5) to {-1, 0, +1}."""
    z = np.array(z)
    z_new = (z - 0.5)
    z_new[z_new < -0.05] = -1
    z_new[z_new > 0.05] = 1
    z_new[np.abs(z_new) < 0.05] = 0
    return z_new


# 3^4 = 81 combinations of {flex,rest,extend} over 4 fingers; the notebook
# appends a duplicated 5th column to match the 5-DOF target vector.
finger_combs = list(itertools.product(*[[-1, 0, 1]] * 4))
finger_combs = [list(f) + [list(f)[-1]] for f in finger_combs]
finger_comb_dict = dict(zip([str(f) for f in finger_combs], range(1, len(finger_combs) + 1)))
finger_comb_dict_inverse = dict(zip(range(1, len(finger_combs) + 1), finger_combs))

# Label each trial. Trials that START at rest (z[0]==0) are labelled by the
# combination they MOVE TO (sign=+1); trials that start mid-movement are the
# return-to-rest phase (sign=-1) and are dropped. So label>0 = movement onsets.
finger_comb_label = []
for z in Z:
    z_new = discretize_z(z[0, :])
    sign = -1
    if np.sum(np.abs(z_new)) == 0:
        z_new = discretize_z(z[-1, :])
        sign = 1
    z_new = [int(x) for x in z_new.astype(int)]
    finger_comb_label.append(sign * finger_comb_dict[str(z_new)])
finger_comb_label = np.array(finger_comb_label)

# Keep movement-onset trials.
move = finger_comb_label > 0
fing = finger_comb_label[move]
block_move = block_id[move]
p_final = [0, fc_Y_raw.shape[1], 1, fc_Y_raw.shape[-1]]

# (1) faithful: per-block z-score -> full-rank PCA -> sum over trial.
Yc = normalize_blocks(fc_Y_raw[move], block_move, center=True)
fc_Y_2d = np.reshape(preprocess(Yc, *p_final), [move.sum(), -1])
fc_cond_all, fc_reps_all = condition_average(fc_Y_2d, fing)

# (2) nonneg: per-block divide-by-std (no centring) -> sum over trial, no PCA.
# The source is already z-scored, so re-anchor each electrode at its empirical
# minimum (zero-firing floor) -- a per-feature constant, geometry unchanged.
Yc_nn = normalize_blocks(fc_Y_raw[move], block_move, center=False)
fc_Y_2d_nn = np.reshape(bin_time(Yc_nn, *p_final[:3]), [move.sum(), -1])
fc_Y_2d_nn = fc_Y_2d_nn - fc_Y_2d_nn.min(0, keepdims=True)
fc_cond_all_nn, fc_reps_all_nn = condition_average(fc_Y_2d_nn, fing)

# The all-rest combination [0,0,0,0,0] is label 41 (idle). Exclude it -> the
# 80 finger-MOVEMENT combinations.
REST_LABEL = finger_comb_dict[str([0, 0, 0, 0, 0])]          # == 41
# keep = fc_cond_all != REST_LABEL
keep = np.ones(len(fc_cond_all)) > 0
fc_cond = fc_cond_all[keep]                                  # (80,) combo labels
fc_reps = fc_reps_all[keep]                                 # faithful (80, n_feat)
fc_reps_nn = fc_reps_all_nn[keep]                           # nonneg (80, n_feat)
fc_comb_vectors = np.array([finger_comb_dict_inverse[int(c)] for c in fc_cond])  # (80, 5)

print('[combos] faithful reps', fc_reps.shape, 'min %.3f' % fc_reps.min(),
      '| nonneg reps', fc_reps_nn.shape, 'min %.3f' % fc_reps_nn.min())


#%%
# ============================================================================
# Save
# ============================================================================

out = {
    'natural_gestures': {
        'trial_labels': gest_labels,    # (847,) gesture id (2..39)
        'cond_labels': gest_cond,       # (38,) gesture ids
        'cond_names': gest_names,       # (38,) gesture names
        # faithful (paper pipeline, mean-subtracted)
        'Y_2d': gest_Y_2d,              # (847, 107) single-trial decoder input
        'reps': gest_reps,              # (38, 107) trial-averaged representation
        # non-negative variant (electrode space, no centring)
        'Y_2d_nonneg': gest_Y_2d_nn,    # (847, 107) >= 0
        'reps_nonneg': gest_reps_nn,    # (38, 107) >= 0, anchored at zero firing
    },
    'finger_combinations': {
        'trial_labels': fing,           # (n_trials,) combination label (1..81)
        'cond_labels': fc_cond,         # (80,) movement-combination labels
        'comb_vectors': fc_comb_vectors,# (80, 5) {-1,0,1} per-finger code
        'rest_label': REST_LABEL,       # 41 = excluded all-rest condition
        # faithful (paper pipeline, mean-subtracted)
        'Y_2d': fc_Y_2d,                # (n_trials, 192) single-trial
        'reps': fc_reps,                # (80, 192) trial-averaged representation
        'reps_with_rest': fc_reps_all,  # (81, 192) incl. rest, if needed
        'cond_labels_with_rest': fc_cond_all,
        # non-negative variant (electrode space, per-electrode min re-anchored)
        'Y_2d_nonneg': fc_Y_2d_nn,      # (n_trials, 192) >= 0
        'reps_nonneg': fc_reps_nn,      # (80, 192) >= 0
        'reps_nonneg_with_rest': fc_reps_all_nn,
    },
}

# save_path = os.path.join(SAVE_DIR, 'shah_fig3_representations.pkl')
# with open(save_path, 'wb') as f:
#     pkl.dump(out, f)
# print('saved ->', save_path)

#%%

## all pairwise CCGPs
clf = LinearSVC()

train = np.zeros((4,4))
test = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        
        if i == j:
            train[i,j] = np.nan
            test[i,j] = np.nan
            continue
        
        deez = np.abs(Y[:,i]) > 0
        
        trn = Y[:,j] > 0
        tst = Y[:,j] < 0
        
        clf.fit(X[deez][trn[deez]], Y[:,i][deez][trn[deez]])
        
        train[i,j] = clf.score(X[deez][trn[deez]], Y[:,i][deez][trn[deez]])
        test[i,j] = clf.score(X[deez][tst[deez]], Y[:,i][deez][tst[deez]])

#%%

train = np.zeros((4,4))
test = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        
        if i == j:
            train[i,j] = np.nan
            test[i,j] = np.nan
            continue
        
        deez = Y[:,i] >= 0
        
        trn = Y[:,j] >= 0
        tst = Y[:,j] < 0
        
        clf.fit(X[deez][trn[deez]], Y[:,i][deez][trn[deez]])
        
        train[i,j] = clf.score(X[deez][trn[deez]], Y[:,i][deez][trn[deez]])
        test[i,j] = clf.score(X[deez][tst[deez]], Y[:,i][deez][tst[deez]])

#%%



# X_ = D - D.mean(0)
# X_ = D

# X_ = out['finger_combinations']['Y_2d_nonneg']
X_ = 1*out['finger_combinations']['reps_nonneg']

X_ = X_ / X_.std(0)

# mod = old_bae_models.SemiBMF(8,
#                          nonneg=True, 
#                          tree_reg=1,
#                          weight_pr_reg=1,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=0,
#                          sparse_reg=1,
#                          )

# mod = old_bae_models.SemiBMF(4,
#                          nonneg=True,
#                          # nonneg=False,
#                          # tree_reg=1e-1,
#                          tree_reg=0,
#                          weight_pr_reg=1,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=1e-2,
#                          # sparse_reg=1e-2,
#                          sparse_reg=0,
#                          # fit_intercept=False,
#                          )

mod = bae_models.JBMF(8,
                         nonneg=True,
                         # nonneg=False,
                         # fit_intercept=False,
                         tree_reg=1e-1,
                         weight_pr_reg=1,
                         weight_l2_reg=1e-1,
                         weight_l1_reg=1e-2,
                         sparse_reg=1e-1,
                         # J_loss='mle',
                         J_loss='rple',
                         # J_l1_reg=1e-3,
                         # J_lr=1e-3,
                         J_lr=0,
                         # slab=True,
                         # slab_prior=0.1,
                         )

# mod = old_bae_models.SpikeNMF(11,
#                          nonneg=True, 
#                          sparse_reg=1,
#                          weight_pr_reg=1,
#                          tree_reg=0,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=0,
#                          )

# mod = old_bae_models.SpikeNMF(2,
#                          nonneg=False, 
#                          sparse_reg=0,
#                          weight_pr_reg=0,
#                          tree_reg=0,
#                          weight_l2_reg=1e-2,
#                          weight_l1_reg=1e-2,
#                          fit_intercept=False,
#                          slab_prior=1,
#                          )

# mod = old_bae_models.KernelBMF2(12,
#                            sparse_reg=1,
#                            tree_reg=100,
#                            uniform_scale=False,
#                            # l1_reg=0,
#                            )

en = mod.fit(X_  / X_.std(),
             period=100, 
             initial_temp=100,
             decay_rate=0.88,
             min_temp=1, 
             scl_lr=1e-4,
             # W_lr=0.05,
             # b_lr=0.05,
             # lr=0.05,
             hot_start=True,
             )

# en = mod.fit(X_ / X_.std(),
#              period=100, 
#              initial_temp=10,
#              decay_rate=0.9, 
#              min_temp=1e-4, 
#              scl_lr=0,
#              )

samps = mod.sample(X_ / X_.std(), n_samp=1000)
# samps = mod.sample(X_ , n_samp=1000, slab=False)
# samps = mod.sample(X_ / X_.std(), n_samp=1000, slab=False)
# 
# samps = np.mod(samps + (samps.mean(1,keepdims=True) > 0.5), 2)

plt.imshow(samps.mean(0))

#%%

#%%

Y = out['finger_combinations']['comb_vectors'][:,:-1]
S = np.hstack([1*(Y > 0), 1*(Y < 0)])

rep = X_
# rep = Usc
# signal = 1
# noise = 0
# rep = signal*S@np.random.randn(S.shape[-1], X_.shape[-1]) + noise*np.random.randn(*X_.shape)

ps = np.zeros((S.shape[1],S.shape[1]))
for i in range(S.shape[1]):
    for j in range(S.shape[1]):
        
        edges = 1*((util.yuke(S) == 1) * (S[:,[i]] != S[:,[i]].T))
        aye_i, jay_i = np.where(np.triu(edges))
        
        edges = 1*((util.yuke(S) == 1) * (S[:,[j]] != S[:,[j]].T))
        aye_j, jay_j = np.where(np.triu(edges))
        
        cs = util.cosine_sim((rep[aye_i] - rep[jay_i]).T, (rep[aye_j] - rep[jay_j]).T)
        
        if i == j:
            ps[i,j] = np.sum(np.triu(cs, k=1)) / spc.binom(len(aye_i),2)
        else:
            ps[i,j] = np.trace(cs) / len(aye_i)

plt.imshow(ps, 'bwr', vmin=-1, vmax=1)

