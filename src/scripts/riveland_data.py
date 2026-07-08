"""
Replicating Riveland & Pouget (Nat. Neurosci. 2024), "Generalization in
sensorimotor networks configured with natural language instructions", Figure 3a-d,
and extracting task representations for disentanglement analysis.

Fig 3a-d are PCA scatter plots of *sensorimotor* representations (the RNN hidden
state at stimulus onset) for the four decision-making modality tasks
    DMMod1, AntiDMMod1, DMMod2, AntiDMMod2
which form a 2x2 factorial:  {Pro, Anti} x {Mod1, Mod2}.
Each panel is a different instruction model, all loaded from the swap9 holdout
(which holds out AntiDMMod1 among 5 tasks, so it is shown *zero-shot*):
    a  sbertNetL_lin   (language)     b  gptNetXL_lin  (language)
    c  combNet         (StructureNet) d  simpleNet     (one-hot rule)

The paper's own figure code is in
    riveland_data/instructRNN/plotting/plot_figs.py  ->  plot_scatter(...)
but the trained models (NN_simData/) are not distributed, so this script (a)
trains the two GPU-cheap, download-free rule models (simpleNet, combNet) on the
swap9 holdout, (b) re-implements the representation extraction + PCA scatter, and
(c) adds functions to pull representations at each locus (sensorimotor / rule /
language) organized by the tasks' factorial structure, so disentanglement can be
tested (CCGP, parallelism, and a bridge to the BMF pipeline in this repo).

Fixes applied to the reference snapshot (numpy>=2 / attrs):
  * np.NaN / np.NAN  ->  np.nan  (reference uses removed aliases)
  * make_default_model('simpleNet') passes model_name both positionally and by
    keyword into an attrs config -> collision; patched below.

Run as a script to (re)train and regenerate figures:
    python scripts/riveland_data.py --train --plot
or import the functions and use the #%% cells interactively (Spyder style).
"""
#%%
import os
import sys
import argparse

import numpy as np
np.NaN = np.nan   # numpy>=2 compat for the reference code
np.NAN = np.nan

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.svm as svm

# ----------------------------------------------------------------------------
# Machine-specific paths (edit for the current machine, cf. other *_data.py)
# ----------------------------------------------------------------------------
CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/repler/src/'
RIVELAND_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/riveland_data'
SAVE_DIR = 'C:/Users/mmall/OneDrive/Documents/uni/columbia/main/data/riveland_data/figs_repl'

# The reference package resolves instruction dicts and model I/O relative to
# $MODEL_FOLDER; the on-the-fly streaming trainer below avoids the ~50GB disk
# cache that TaskDataSet.check_data_build() would otherwise write here.
MODEL_FOLDER = RIVELAND_DIR + '/NN_simData'
os.environ['MODEL_FOLDER'] = MODEL_FOLDER
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

sys.path.insert(0, CODE_DIR)
sys.path.insert(0, RIVELAND_DIR)

# Running this file directly puts src/scripts on sys.path, whose transformers.py
# would shadow the HuggingFace `transformers` package that the language models
# import. Drop this file's own directory so the real package resolves.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import instructRNN.models.full_models as full_models
from instructRNN.tasks.tasks import (TASK_LIST, SWAPS_DICT, DICH_DICT,
                                      construct_trials)
from instructRNN.instructions.instruct_utils import get_task_info
from instructRNN.tasks.task_criteria import isCorrect
from instructRNN.analysis.model_analysis import (get_task_reps, get_rule_reps,
                                                 get_instruct_reps, reduce_rep,
                                                 get_reps_from_tasks)
from instructRNN.plotting.plotting import get_task_color

os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

# The four DM-modality tasks of Fig 3, and their 2x2 factor labels
FIG3_TASKS = ['DMMod1', 'AntiDMMod1', 'DMMod2', 'AntiDMMod2']
FACTORS = {
    'DMMod1':     {'pro_anti': 'Pro',  'modality': 'Mod1'},
    'AntiDMMod1': {'pro_anti': 'Anti', 'modality': 'Mod1'},
    'DMMod2':     {'pro_anti': 'Pro',  'modality': 'Mod2'},
    'AntiDMMod2': {'pro_anti': 'Anti', 'modality': 'Mod2'},
}


# ----------------------------------------------------------------------------
# make_default_model fix (attrs model_name collision for simpleNet)
# ----------------------------------------------------------------------------
_orig_make_default_model = full_models.make_default_model

def make_default_model(model_str):
    """Corrected constructor: simpleNet's reference path double-passes model_name."""
    if model_str == 'simpleNet':
        return full_models.SimpleNet()          # model_name defaults to 'simpleNet'
    return _orig_make_default_model(model_str)


# ============================================================================
#  TRAINING  (streaming; faithful to the paper but no 50GB trial cache)
# ============================================================================
def stream_train(model, holdouts, seed=0, num_batches=2400, batch_len=64,
                 min_run_epochs=5, max_epochs=60, init_lr=1e-3, gamma=0.95,
                 checker_threshold=0.95, check_duration=3, verbose_every=200):
    """Train a sensorimotor model on all tasks except `holdouts`.

    Mirrors instructRNN.trainers.model_trainer (Adam, ExponentialLR, grad-value
    clip 0.5, masked MSE, isCorrect-based early stop at `checker_threshold`) but
    samples trials on the fly with construct_trials -> no disk build.
    Returns dict of per-task rolling correct history.
    """
    from instructRNN.trainers.base_trainer import masked_MSE_Loss

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tasks = [t for t in TASK_LIST if t not in holdouts]
    model.to(DEVICE)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    correct_hist = {t: [] for t in train_tasks}
    step = 0
    for epoch in range(max_epochs):
        order = [train_tasks[i] for i in np.random.randint(0, len(train_tasks), num_batches)]
        for task in order:
            ins, tar, mask, tar_dir, _ = construct_trials(task, batch_len, return_tensor=True)
            info = get_task_info(batch_len, task, model.info_type)
            opt.zero_grad()
            out, _ = model(ins.to(DEVICE), info)
            loss = masked_MSE_Loss(out, tar.to(DEVICE), mask.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            opt.step()
            correct_hist[task].append(round(float(np.mean(isCorrect(out, tar, tar_dir))), 3))
            if step % verbose_every == 0:
                recent = np.mean([np.mean(v[-check_duration:]) for v in correct_hist.values() if v])
                print(f'  ep{epoch} step{step} task={task:11s} '
                      f'loss={loss.item():.3e} mean_recent_correct={recent:.3f}', flush=True)
            step += 1
        sched.step()

        # early-stop check: every trained task above threshold over its last few batches
        if epoch >= min_run_epochs - 1:
            latest = [np.array(v[-check_duration:]) for v in correct_hist.values() if len(v) >= check_duration]
            if latest and np.all(np.array(latest) > checker_threshold):
                print(f'  >> reached {checker_threshold} on all tasks at epoch {epoch}', flush=True)
                break
    return correct_hist


def train_and_save(model_name, holdout_label='swap9', seed=0, **kwargs):
    """Train `model_name` on a swap holdout and save with the paper's naming so
    model.load_model(EXP/holdout/model_name, suffix='_seedN') can reload it."""
    holdouts = list(SWAPS_DICT[holdout_label])
    print(f'[{model_name}] holdout {holdout_label} -> {holdouts}', flush=True)
    model = make_default_model(model_name)
    stream_train(model, holdouts, seed=seed, **kwargs)
    save_dir = f'{MODEL_FOLDER}/swap_holdouts/{holdout_label}/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    model.save_model(save_dir, suffix=f'_seed{seed}')
    print(f'[{model_name}] saved to {save_dir}/{model_name}_seed{seed}.pt', flush=True)
    return model


def load_model(model_name, holdout_label='swap9', seed=0):
    model = make_default_model(model_name)
    load_dir = f'{MODEL_FOLDER}/swap_holdouts/{holdout_label}/{model_name}'
    model.load_model(load_dir, suffix=f'_seed{seed}')
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================================
#  REPRESENTATION EXTRACTION  (the three loci Fig 3 contrasts)
# ============================================================================
def extract_task_reps(model, tasks=FIG3_TASKS, rep_depth='task', num_trials=50,
                      epoch='stim_start', instruct_mode='combined'):
    """Extract task representations at a chosen locus.

    rep_depth:
        'task'  -> sensorimotor RNN hidden state (Fig 3 uses this): the recurrent
                   activity at `epoch` (default stimulus onset).  Shape
                   (n_tasks, num_trials, rnn_hidden_dim).
        'rule'  -> the model's rule embedding (simpleNet/combNet only).  Shape
                   (n_tasks, 1, rule_dim).
        int / 'full' / 'bow' -> language-model instruction embedding at that
                   transformer depth (instruct models only).  Shape
                   (n_tasks, n_instructions, lang_dim).

    Returns reps for the requested `tasks` (subset of TASK_LIST), in that order.
    """
    if rep_depth == 'task':
        reps = get_task_reps(model, epoch=epoch, num_trials=num_trials,
                             main_var=True, instruct_mode=instruct_mode)
    elif rep_depth == 'rule':
        assert model.info_type in ('rule', 'comb'), 'rule reps only for rule models'
        reps = get_rule_reps(model)
    else:
        assert hasattr(model, 'langModel'), 'language reps need an instruct model'
        reps = get_instruct_reps(model.langModel, depth=rep_depth, instruct_mode=instruct_mode)
    return get_reps_from_tasks(reps, tasks)


def condition_mean_reps(model, tasks=FIG3_TASKS, **kw):
    """(n_tasks, dim) trial-averaged representation per task -- the natural input
    to a (task x unit) factorization / RSA / BMF analysis."""
    reps = extract_task_reps(model, tasks=tasks, **kw)
    return reps.mean(axis=1)


def reps_to_bmf_matrix(model, tasks=TASK_LIST, **kw):
    """Bridge to this repo's BMF pipeline: a (task x unit) matrix X of trial-mean
    sensorimotor activity, ready for e.g. bae_models.SemiBMF to factor
    X ~ f(S @ W) and test whether the binary latents S recover the task factors."""
    X = condition_mean_reps(model, tasks=tasks, **kw)
    return X, list(tasks)


# ============================================================================
#  FIGURE 3 SCATTER  (PCA of the representations)
# ============================================================================
def plot_fig3_scatter(model, tasks=FIG3_TASKS, rep_depth='task', dims=3,
                      num_trials=50, epoch='stim_start', instruct_mode='combined',
                      title=None, save_path=None, s=12):
    """Re-implementation of instructRNN.plotting.plot_scatter for Fig 3a-d:
    PCA of the per-trial representations, colored by task."""
    reps = extract_task_reps(model, tasks=tasks, rep_depth=rep_depth,
                             num_trials=num_trials, epoch=epoch,
                             instruct_mode=instruct_mode)
    reduced, var = reduce_rep(reps, pcs=list(range(dims)))  # (n_tasks, n_trials, dims)

    with plt.style.context('ggplot'):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection='3d') if dims == 3 else fig.add_subplot()
        for i, task in enumerate(tasks):
            c = get_task_color(task)
            pts = reduced[i]
            if dims == 3:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=c, s=s,
                           edgecolor='white', linewidth=0.3, label=task)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], color=c, s=s,
                           edgecolor='white', linewidth=0.3, label=task)
        ev = '' if var is None else f'  (PC var: {np.round(var[:dims], 2)})'
        ax.set_title((title or f'{model.model_name}  [{rep_depth}]') + ev, fontsize=9)
        for setter in ('set_xticklabels', 'set_yticklabels', 'set_zticklabels'):
            if hasattr(ax, setter):
                getattr(ax, setter)([])
        ax.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'saved {save_path}', flush=True)
    return fig, reduced, var


# ============================================================================
#  DISENTANGLEMENT MEASURES
# ============================================================================
def factorial_reps(model, tasks=FIG3_TASKS, **kw):
    """Return trial-mean reps arranged by the {Pro,Anti} x {Mod1,Mod2} grid, plus
    the two factor-difference vectors.  If the geometry is disentangled the Pro->Anti
    displacement is the same across modalities (a parallelogram)."""
    R = {t: r for t, r in zip(tasks, condition_mean_reps(model, tasks=tasks, **kw))}
    pro_anti_vecs = [R['AntiDMMod1'] - R['DMMod1'], R['AntiDMMod2'] - R['DMMod2']]
    mod_vecs = [R['DMMod2'] - R['DMMod1'], R['AntiDMMod2'] - R['AntiDMMod1']]
    return R, np.array(pro_anti_vecs), np.array(mod_vecs)


def parallelism_score(vecs):
    """Cosine similarity between the two displacement vectors of a factor -- 1.0
    means the factor is encoded by a single shared axis (disentangled)."""
    a, b = vecs[0], vecs[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def factor_parallelism(model, tasks=FIG3_TASKS, **kw):
    """Parallelism scores for both factors of the 2x2 DM-Mod set."""
    _, pro_anti_vecs, mod_vecs = factorial_reps(model, tasks=tasks, **kw)
    return {'pro_anti': parallelism_score(pro_anti_vecs),
            'modality': parallelism_score(mod_vecs)}


def ccgp_factor(reps, group_a, group_b, cross_a, cross_b, max_iter=200000):
    """Cross-condition generalization performance for one dichotomy: train a linear
    classifier to separate group_a vs group_b, test on a *held-out* pair
    (cross_a vs cross_b).  reps is (n_tasks_in_subset, n_trials, dim) indexed to
    match the label lists.  High CCGP => the factor is linearly abstract."""
    def stack(names, reps_dict):
        return np.concatenate([reps_dict[n] for n in names], axis=0)

    n_tr = next(iter(reps.values())).shape[0]
    clf = svm.LinearSVC(max_iter=max_iter, dual=False, tol=1e-5, random_state=0)
    Xtr = np.concatenate([stack(group_a, reps), stack(group_b, reps)], axis=0)
    ytr = np.array([0] * (len(group_a) * n_tr) + [1] * (len(group_b) * n_tr))
    clf.fit(Xtr, ytr)
    Xte = np.concatenate([stack(cross_a, reps), stack(cross_b, reps)], axis=0)
    yte = np.array([0] * (len(cross_a) * n_tr) + [1] * (len(cross_b) * n_tr))
    return float(clf.score(Xte, yte))


def dm_mod_ccgp(model, num_trials=50, **kw):
    """CCGP of each DM-Mod factor: can a classifier trained on one modality's
    Pro/Anti (or one Pro/Anti's modalities) generalize to the other?  This is the
    disentanglement measure of Fig 3 (paper's get_dich_CCGP restricted to this 2x2)."""
    reps = extract_task_reps(model, tasks=FIG3_TASKS, rep_depth='task',
                             num_trials=num_trials, **kw)
    rd = {t: reps[i] for i, t in enumerate(FIG3_TASKS)}
    # Pro vs Anti: train within Mod1, test on Mod2 (and vice versa), average
    pro_anti = np.mean([
        ccgp_factor(rd, ['DMMod1'], ['AntiDMMod1'], ['DMMod2'], ['AntiDMMod2']),
        ccgp_factor(rd, ['DMMod2'], ['AntiDMMod2'], ['DMMod1'], ['AntiDMMod1']),
    ])
    # Mod1 vs Mod2: train within Pro, test on Anti (and vice versa), average
    modality = np.mean([
        ccgp_factor(rd, ['DMMod1'], ['DMMod2'], ['AntiDMMod1'], ['AntiDMMod2']),
        ccgp_factor(rd, ['AntiDMMod1'], ['AntiDMMod2'], ['DMMod1'], ['DMMod2']),
    ])
    return {'pro_anti': float(pro_anti), 'modality': float(modality)}


def disentanglement_report(model, num_trials=100):
    """One-call summary for a trained model: parallelism + CCGP of both factors."""
    par = factor_parallelism(model, num_trials=num_trials)
    ccgp = dm_mod_ccgp(model, num_trials=num_trials)
    print(f'\n== {model.model_name} disentanglement (DM-Mod 2x2) ==')
    print(f'  parallelism  pro/anti={par["pro_anti"]:+.3f}  modality={par["modality"]:+.3f}')
    print(f'  CCGP         pro/anti={ccgp["pro_anti"]:.3f}   modality={ccgp["modality"]:.3f}')
    return {'parallelism': par, 'ccgp': ccgp}


# ============================================================================
#  SCRIPT ENTRY / #%% CELLS
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', action='store_true', help='train the rule models')
    ap.add_argument('--plot', action='store_true', help='make Fig-3 scatters')
    ap.add_argument('--models', nargs='+', default=['simpleNet', 'combNet'])
    ap.add_argument('--holdout', default='swap9')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_epochs', type=int, default=60)
    ap.add_argument('--min_run_epochs', type=int, default=5)
    args = ap.parse_args()

    models = {}
    for name in args.models:
        if args.train:
            models[name] = train_and_save(name, args.holdout, args.seed,
                                          max_epochs=args.max_epochs,
                                          min_run_epochs=args.min_run_epochs)
        else:
            models[name] = load_model(name, args.holdout, args.seed)

    for name, model in models.items():
        if args.plot:
            plot_fig3_scatter(model, save_path=f'{SAVE_DIR}/fig3_{name}_task.png')
        disentanglement_report(model)


if __name__ == '__main__':
    main()

#%% interactive: load an already-trained model and inspect
# model = load_model('simpleNet', 'swap9', seed=0)
# fig, reduced, var = plot_fig3_scatter(model, save_path=f'{SAVE_DIR}/fig3_simpleNet_task.png')
# print(disentanglement_report(model))
# X, task_names = reps_to_bmf_matrix(model)   # -> feed to bae_models.SemiBMF
