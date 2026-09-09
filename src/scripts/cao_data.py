CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/saves/'
LOAD_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/cao_data/'

import os, sys, glob, itertools
sys.path.append(CODE_DIR)

import numpy as np
import pickle as pkl

import scipy.io as sio

import matplotlib.pyplot as plt

#%%
# ----------------------------------------------------------------------------
# Cao et al. -- human MTL single-unit responses to natural object/scene images
# ("identity / category neurons"). Epilepsy patients with depth electrodes in
# amygdala (A) and hippocampus (anterior AH / posterior PH), both hemispheres
# (L*/R*). Two image sets, each 500 images, each its own pooled-across-sessions
# spike file:
#   CoCo     -- 10 object categories x 50 images   (CoCo_Spikes.mat,     512 units, 18 sess)
#   ImageNet -- 50 object categories x 10 images    (ImageNet_Spikes.mat, 1204 units, 28 sess)
#
# Task: 550 trials/session = 500 unique images + 50 one-back repeats; 2 s trial,
# stimulus onset 500 ms after trial start. (The behaviorData/Events Files give
# the raw per-trial timing, but we don't need them -- see below.)
#
# ---- Why this loader is short -------------------------------------------------
# Unlike the other datasets, the per-image neural representation here is ALREADY
# BUILT into '*_Spikes.mat'. The bundled `FR` struct is the *image-sorted*
# firing rate (the `FR_sort` output of the authors' Analysis/SortFR.m): for unit
# c, every field is length 500 and indexed in canonical image order, where image
# j == `vImg[j]` (airplane-01 ... zebra-50 for CoCo). SortFR.m reorders the raw
# per-trial counts into this image order (dropping the 50 one-back trials and
# averaging any image shown more than once), so index j means the SAME image for
# EVERY unit across EVERY session -- exactly the alignment a pseudopopulation
# needs. Stacking `FR[c].countAll` over units therefore gives an
# (n_image x n_neuron) representation directly; no spike-time alignment required.
#
# `FR` field = response window (firing rate in Hz, baseline-able):
#   countAll        250-1250 ms post-stimulus   <- the paper's main response window
#   countAllEarly   0-750 ms ; countAllLate 750-1500 ms
#   countBaseline   -250..0 ms pre-stimulus     <- for baseline subtraction
#   countEntireTrial 0-2000 ms ; meanOveralFR scalar ; PSTH (500 x 36) 250 ms/50 ms bins
# (windows are relative to stimulus onset; counts normalized to Hz.)
#
# Quality selection: `vKeep` (the authors' kept units, mean FR > 0.15 Hz and not
# epileptic-zone-rejected) vs `vReject`; `areaCell` is the MTL subregion per unit.
#
# ---- Missing entries ----------------------------------------------------------
# A handful of (unit, image) entries are NaN: images whose trial was dropped for
# that session (interruptions). It's a tiny fraction (CoCo 49/226500,
# ImageNet 3648/437000, affecting ~32-49 units). These are LEFT AS NaN in the
# returned `reps` -- impute them however you like during model fitting -- and
# `nan_mask` (True where missing) tells you exactly which. `reps_baselined` keeps
# the same NaN entries so the masks line up.
#
# Output (per dataset) saved to SAVE_DIR/cao_representations.pkl:
#   reps            (n_image x n_keep)  mean-response pseudopop (countAll); NaN = missing
#   reps_baselined  (n_image x n_keep)  countAll - countBaseline (signed); NaN = missing
#   nan_mask        (n_image x n_keep)  True where the entry is missing
#   image_names     (n_image,)          file names, canonical order
#   category        (n_image,)          category string per image
#   cat_labels      (n_image,)          integer category id (0..n_cat-1)
#   categories      (n_cat,)            unique category names
#   unit_area       (n_keep,)           MTL subregion per kept unit
#   unit_session    (n_keep,)           session id per kept unit
#   keep_idx        (n_keep,)           index of kept units into the full unit list
# ----------------------------------------------------------------------------


def load_cao(task, response_field='countAll', use_keep=True):
    """Build the (n_image x n_neuron) pseudopopulation for one Cao image set.

    task            : 'CoCo' or 'ImageNet'
    response_field  : which FR window to use as the response (default 'countAll',
                      the paper's 250-1250 ms window)
    use_keep        : restrict to the authors' quality-kept units (`vKeep`)

    Returns a dict (see module docstring for keys)."""

    m = sio.loadmat(os.path.join(LOAD_DIR, task + '_Spikes.mat'),
                    squeeze_me=True, struct_as_record=False)

    FR = m['FR']                      # (n_unit,) struct, image-sorted FR per unit
    vImg = m['vImg']                  # (500,) image file info, canonical order
    n_unit = len(FR)

    # --- image labels (canonical order: index j == vImg[j]) -------------------
    image_names = np.array([vImg[j].name for j in range(len(vImg))])
    category = np.array([n.rsplit('-', 1)[0] for n in image_names])
    categories, cat_labels = np.unique(category, return_inverse=True)

    # --- which units to keep --------------------------------------------------
    if use_keep:
        keep_idx = np.asarray(m['vKeep'], int) - 1        # vKeep is 1-indexed (MATLAB)
    else:
        keep_idx = np.arange(n_unit)

    # --- stack the per-image response into (n_image x n_unit) -----------------
    def stack(field):
        return np.stack([np.asarray(getattr(FR[c], field), float)
                         for c in keep_idx], axis=1)       # (n_image x n_keep)

    reps = stack(response_field)               # NaN = missing entry, left for you to impute
    nan_mask = np.isnan(reps)

    # baseline-subtracted (signed) variant; missing entries stay NaN (mask lines up)
    reps_baselined = reps - stack('countBaseline')

    # --- per-unit metadata ----------------------------------------------------
    unit_area = np.asarray(m['areaCell'], object)[keep_idx]
    unit_session = np.asarray(m['vCell'], int)[keep_idx]

    return {
        'reps': reps,
        'reps_baselined': reps_baselined,
        'nan_mask': nan_mask,
        'image_names': image_names,
        'category': category,
        'cat_labels': cat_labels,
        'categories': categories,
        'unit_area': unit_area,
        'unit_session': unit_session,
        'keep_idx': keep_idx,
    }


#%%
# build both datasets
out = {task: load_cao(task) for task in ['CoCo', 'ImageNet']}

for task, d in out.items():
    print(f"[{task}] reps {d['reps'].shape} (img x unit) | "
          f"{len(d['categories'])} categories | "
          f"missing (NaN): {d['nan_mask'].sum()} entries in "
          f"{d['nan_mask'].any(0).sum()} units | "
          f"areas: {sorted(set(d['unit_area']))}")

#%%
# quick look: category-block structure of the mean-response RDM
task = 'CoCo'
d = out[task]
X = d['reps'][:,deez]
Xz = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-8)   # z-score each unit (NaN-safe)
R = np.ma.corrcoef(np.ma.masked_invalid(Xz))            # image x image similarity, NaN-aware

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(X, aspect='auto', cmap='magma')
ax[0].set(title=f'{task}: mean response (img x unit)', xlabel='unit', ylabel='image')
ax[1].imshow(R, cmap='RdBu_r', vmin=-1, vmax=1)
ax[1].set(title='image x image correlation', xlabel='image', ylabel='image')
# category boundaries
bnds = np.where(np.diff(d['cat_labels']))[0] + 0.5
for b in bnds:
    ax[1].axhline(b, c='k', lw=0.3); ax[1].axvline(b, c='k', lw=0.3)
plt.tight_layout()
plt.show()

#%%
# save
save_path = os.path.join(SAVE_DIR, 'cao_representations.pkl')
# with open(save_path, 'wb') as f:
#     pkl.dump(out, f)
# print('saved ->', save_path)
