"""
Standalone interpretability search for Hunt et al. data.
Run from the scripts directory with the right conda env activated:

    python run_hunt_search.py

Prints text descriptions of the top results to stdout and saves all
results to pickle files for later inspection / reproduction.
"""

import matplotlib
matplotlib.use('Agg')   # suppress all GUI calls before any other import

import sys, os, pickle
sys.path.insert(0, 'C:/Users/mmall/OneDrive/Documents/github/repler/src/')

import numpy as np
import hunt_data as _hd

X     = _hd.X
conds = _hd.conds
area  = _hd.area

print(f'Data loaded: X shape = {X.shape}')
print(f'Areas: { {a: int((area==a).sum()) for a in ["OFC","ACC","DLPFC"]} }')

# ---- Phase 1: broad search (KernelBMF2 / BiPCA / SemiBMF quick) -------------
print('\n' + '='*60)
print('Phase 1: broad search (300 trials × 3 runs, quick schedule)')
print('='*60)

broad = _hd.broad_search(X, conds, area, n_trials=300, n_runs=3, seed=42)

print(f'\n--- Top 20 results ---\n')
for i, r in enumerate(broad[:20]):
    _hd.describe_result(r, conds, rank=i+1)
    print()

# ---- Phase 2: focused refinement --------------------------------------------
print('='*60)
print('Phase 2: focused refinement (top 20 × 8 runs, full schedule)')
print('='*60)

refined = _hd.focused_search(X, conds, area, broad[:20], n_runs=8, seed=0)

print(f'\n--- Top 10 after refinement ---\n')
for i, r in enumerate(refined[:10]):
    _hd.describe_result(r, conds, rank=i+1)
    print()

with open('hunt_search_results.pkl', 'wb') as f:
    pickle.dump({'broad': broad, 'refined': refined}, f)
print('Saved to hunt_search_results.pkl')

# ---- Phase 3: SemiBMF dedicated search (full schedule) ----------------------
print('\n' + '='*60)
print('Phase 3: SemiBMF search (50 trials × 5 runs, full schedule)')
print('='*60)

semibmf = _hd.semibmf_search(X, conds, area, n_trials=50, n_runs=5, seed=123)

print(f'\n--- Top 15 SemiBMF results ---\n')
for i, r in enumerate(semibmf[:15]):
    _hd.describe_result(r, conds, rank=i+1)
    print()

with open('hunt_search_semibmf.pkl', 'wb') as f:
    pickle.dump({'semibmf': semibmf}, f)
print('Saved to hunt_search_semibmf.pkl')

# ---- Phase 4: stimulus-specific search --------------------------------------
print('\n' + '='*60)
print('Phase 4: stimulus search (60 trials × 5 runs, full schedule)')
print('Sparse SemiBMF, tree_reg=0, small K — targeting rank-exact patterns')
print('='*60)

stim = _hd.stimulus_search(X, conds, area, n_trials=60, n_runs=5, seed=77)

print(f'\n--- Top 15 stimulus search results ---\n')
for i, r in enumerate(stim[:15]):
    _hd.describe_result(r, conds, rank=i+1)
    print()

with open('hunt_search_stimulus.pkl', 'wb') as f:
    pickle.dump({'stimulus': stim}, f)
print('Saved to hunt_search_stimulus.pkl')

# ---- Phase 5: SpikeNMF search -----------------------------------------------
print('\n' + '='*60)
print('Phase 5: SpikeNMF search (60 trials × 5 runs, full schedule)')
print('Spike-and-slab prior — targets stimulus-identity coding in OFC')
print('='*60)

spikenmf = _hd.spikenmf_search(X, conds, area, n_trials=60, n_runs=5, seed=99)

print(f'\n--- Top 15 SpikeNMF results ---\n')
for i, r in enumerate(spikenmf[:15]):
    _hd.describe_result(r, conds, rank=i+1)
    print()

with open('hunt_search_spikenmf.pkl', 'wb') as f:
    pickle.dump({'spikenmf': spikenmf}, f)
print('Saved to hunt_search_spikenmf.pkl')
