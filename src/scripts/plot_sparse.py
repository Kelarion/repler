"""
Plot the sweep_sparse.py results, run_local.py-style.

For a given (structure, model) it lays out rows = metrics, cols = generative
temperature, x-axis = N, one line per split value (tree_reg or pr_reg), with the
NMF baseline overlaid.  Saves PNGs to figs_sweep/ and supports #%% interactive
re-plotting.

    python plot_sparse.py
"""

import sys, os, pickle
sys.path.insert(0, 'C:/Users/mmall/OneDrive/Documents/github/repler/src/')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

RESULTS = pickle.load(open('sweep_sparse_results.pkl', 'rb'))
METRICS = ['hamming', 'nbs', 'feat', 'entropy']   # feat = W recovery (lower=better)
os.makedirs('figs_sweep', exist_ok=True)


def subset(rows, **conds):
    def ok(r):
        for k, v in conds.items():
            rv = r[k]
            if isinstance(v, float) and np.isnan(v):
                if not (isinstance(rv, float) and np.isnan(rv)):
                    return False
            elif rv != v:
                return False
        return True
    return [r for r in rows if ok(r)]


def _xy(rows, metric):
    rows = sorted(rows, key=lambda r: r['N'])
    return (np.array([r['N'] for r in rows]),
            np.array([r[metric + '_mean'] for r in rows]),
            np.array([r[metric + '_std'] for r in rows]))


def plot_grid(structure, model, split='tree_reg', fixed_pr=1.0, fixed_tree=1.0,
              fixed_sparse=1.0, metrics=METRICS, save=True):
    """rows = metrics, cols = gen_temp, lines = `split`; NMF overlaid.
    The two reg dims that aren't `split` are pinned (fixed_pr/tree/sparse)."""
    temps = sorted({r['gen_temp'] for r in RESULTS})
    vals = sorted({r[split] for r in subset(RESULTS, structure=structure, model=model)
                   if not (isinstance(r[split], float) and np.isnan(r[split]))})
    cols = cm.viridis(np.linspace(0, 1, len(vals)))
    pin = {'tree_reg': fixed_tree, 'pr_reg': fixed_pr, 'sparse_reg': fixed_sparse}
    other = {k: v for k, v in pin.items() if k != split}   # pin the non-split regs

    fig, axes = plt.subplots(len(metrics), len(temps), squeeze=False,
                             figsize=(3.4 * len(temps), 2.8 * len(metrics)))
    for i, metric in enumerate(metrics):
        for j, T in enumerate(temps):
            ax = axes[i][j]
            for c, v in zip(cols, vals):
                rows = subset(RESULTS, structure=structure, model=model,
                              gen_temp=T, **{split: v}, **other)
                if not rows:
                    continue
                N, y, e = _xy(rows, metric)
                ax.errorbar(N, y, yerr=e, marker='.', color=c, lw=1.8,
                            label=f'{split}={v:g}')
            nmf = subset(RESULTS, structure=structure, model='NMF', gen_temp=T)
            if nmf and model != 'NMF' and metric != 'entropy':
                N, y, e = _xy(nmf, metric)
                ax.plot(N, y, 'k--', marker='x', lw=1.5, label='NMF')
            ax.set_xscale('log', base=2)
            if i == 0:
                ax.set_title(f'gen T = {T:g}')
            if i == len(metrics) - 1:
                ax.set_xlabel('N')
            if j == 0:
                ax.set_ylabel(metric)
    axes[0][0].legend(fontsize=7)
    keep = ', '.join(f'{k}={v:g}' for k, v in other.items())
    fig.suptitle(f'{structure} | {model} | split by {split} ({keep})')
    fig.tight_layout()
    if save:
        fn = f'figs_sweep/{structure}_{model}_by-{split}.png'
        fig.savefig(fn, dpi=110); plt.close(fig)
        print('saved', fn)


if __name__ == "__main__":
    models = sorted({r['model'] for r in RESULTS} - {'NMF'})
    structs = sorted({r['structure'] for r in RESULTS})
    for s in structs:
        for m in models:
            plot_grid(s, m, split='tree_reg',   fixed_pr=1.0, fixed_sparse=1.0)
            plot_grid(s, m, split='pr_reg',     fixed_tree=1.0, fixed_sparse=1.0)
            plot_grid(s, m, split='sparse_reg', fixed_tree=1.0, fixed_pr=1.0)

# %%  interactive: tweak and re-run a single figure
# plot_grid('tree', 'SpikeNMF', split='tree_reg', fixed_pr=1.0, save=False)
# plt.show()
