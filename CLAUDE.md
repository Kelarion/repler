# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`repler` is a personal computational-neuroscience / representation-learning research monorepo (the name has a typo the author kept). It contains years of idiosyncratic experiments. There is **no test suite, no linter, no CI** — code is run interactively or dispatched to a cluster. Two largely independent bodies of code live here:

1. **BMF / Binary Matrix Factorization (the BAE family)** — the actively-developed core. Factorizes a data matrix `X ≈ f(S @ W)` where `S` is a binary (or spike-and-slab) latent matrix and `W` continuous weights, fit by simulated annealing.
2. **Neural-network experiments** — PyTorch models trained on synthetic tasks, orchestrated by an Experiment/Task/Model framework and dispatched as cluster array jobs.

## Layout and import model

- `src/` is the importable root. Modules import each other by bare name (`import util`, `import bae_models`), so **code must be run with `src/` on `sys.path`**. Scripts do this via `sys.path.append(CODE_DIR)` with a hardcoded `CODE_DIR` at the top of the file — expect to edit that path for the current machine.
- `src/scripts/` holds ~140 one-off analysis scripts. They are not libraries; most are organized into `#%%` cells meant to be run block-by-block in Spyder/an interactive console, not start-to-finish. `hunt_data.py` (+ its standalone batch runner `run_hunt_search.py`) is the current active work: RSA + BMF interpretability search on Hunt et al. monkey OFC/ACC/DLPFC data.
- Machine-specific paths (`CODE_DIR`, `SAVE_DIR`, `LOAD_DIR`) are switched by hostname in `src/cloud.py` (`agnello` = local WSL, else `burg` = Columbia cluster), and hardcoded at the top of most scripts.

## Environment & build

- Conda env named `pete`: `conda env create -f src/pete.yaml` (cluster) or `src/pete_local.yaml` (local). `src/requirements.txt` is the pip equivalent. PyTorch is installed from the cu126 index.
- **Cython extensions** (`spbae`, optimized BMF search kernels) must be compiled in place before importing:
  ```bash
  cd src && python spbae_setup.py build_ext --inplace   # builds spbae_search.pyx, spbae_util.pyx -> .pyd
  ```
  `setup.py` separately builds the throwaway `test.pyx`. The `.c`/`.html`/`.pyd` artifacts are checked in; regenerate them after editing any `.pyx`/`.pxd`.

## BMF architecture (bae_*.py, spbae_*)

This is the part to understand before touching model code. There are **two generations** of the stack. The plain `bae_*` names are the current one; the superseded generation kept its old contents under `old_bae_*`.

### Current stack — `bae_models`, `bae_weights`, `bae_priors`, `bae_search`

Composition rather than inheritance: a model is an **operator** (the decoder side) plus a **latent prior** (the latent side), and its E-step kernel is *compiled* from plugins instead of hand-written per model.

- **`bae_models.py`** — the model classes (`SemiBMF`, `JBMF`, `BiPCA`, `SCPD`/`JSCPD`, `RRBMF`/`JRRBMF`, `ConvBMF`, `KernelBMF`), `@dataclass`es over `BMF` / `LinearGaussianBMF`. The base owns the annealing loop (`fit`), one templated `EStep(X, S, Z)` / `MStep(X, ES, S)` pair shared by every linear-Gaussian model, imputation (`fit(mask=…)`), Gibbs `sample`, `loss`, `loglikelihood`, and the variational `elbo`. `n_chains > 1` runs C independent chains as a leading array axis; `best_chain` / `collapse` pick the winner (ranked on the ELBO, not the MSE).
- **`bae_weights.py`** — the operators: `AffineOperator`, `Procrustes` (orthonormal W + one scale, backs BiPCA), `CPOperator`, `ReducedRankOp`, `ConvOperator`, `TorchMatrixOp`. Each supplies `forward` / `drive` / `gram` / `backward`.
- **`bae_priors.py`** — the latent side, and the owner of the latent state (`S`, `Z`, `StS`): `LatentPrior` (independent Bernoulli + sparsity/tree penalties), `BoltzmannPrior` (pairwise Ising coupling in the `{0,1}` coding, fit by pseudolikelihood or PCD maximum likelihood), `MRFPrior`, plus the prior-temperature schedules. `slab=True` is a flag on any prior, not a subclass — it replaces the old `SpikeNMF`.
- **`bae_search.py`** — the `@njit` E-step kernels, composed from three plugins (link × prior × operator scaffold) and memoized per combination, so a new model normally needs no new kernel.
- **`bae_experiments.py`** — synthetic generators used to exercise the above.

### Superseded stack — `old_bae_*`

Still imported by many `scripts/` (notably `hunt_data.py`) and kept working, but not developed.

- **`old_bae_models.py`** — the previous model classes (`SemiBMF`, `SpikeNMF`, `BiPCA`, `KernelBMF`/`KernelBMF2`, `JBMF`, `SCPD`, `ConvBMF`, `RRBMF`, …), each implementing its own `init_params`/`init_latents`, `EStep(S, X)`, `MStep(ES, X)`, `__call__(S)`, `loss`, `loglikelihood`, `sample`. State lives on the instance (`self.S`, `self.W`, `self.scl`, `self.temp`). In `SpikeNMF` `self.S` is the **binary spike** and `self.Z` the **rectified continuous multipliers**, but `EStep`/`grad_step` *return* the product `S*Z`.
- **`old_bae_search.py`** — one hand-written `@njit` coordinate-descent kernel per model (`bae`, `sbmf`, `bpca`, `snmf`, `kerbmf*`, `convbmf`, …).
- **`old_bae.py`** — thin wrappers tying those together for common fit patterns. `old_bae_sparse.py` (geoopt) and `old_bae_gpu.py` (jax) are unused side experiments.

Note the argument order flipped between the generations — `EStep(S, X)` then, `EStep(X, S)` now — which is why `bae_util._estep`/`_mstep` dispatch on the signature so `bicv` can serve both.

### Shared

- **`bae_util.py`** — used by BOTH stacks, which is why it kept its name: the `Neal` annealing scheduler and the cross-validation layer. **`impcv` (imputation-based CV) is the primary model-selection tool** — it masks a fraction of entries, refits while imputing them, and reports held-out vs train log-likelihood. `bicv`/`fpcv`/`kerimpcv`/`splitfit`/`multifit` are alternatives. It also holds the numba-side helpers the search kernels call (`log_ndtr`, `sample_trunc_norm`) and `_seed_all` (numba keeps its own RNG — `np.random.seed` alone does not make a fit reproducible).
- **`spbae_search.pyx` / `spbae_util.pyx`** — Cython reimplementations of the old kernels for sparse/large problems, exposed via the compiled `spbae` module and used by `spbae_models.py`.
- `df_util.py` / `distance_factorization.py` / `df_models.py` are the older distance-factorization predecessors; `bae_util` and `old_bae_models` import helpers (e.g. `permham`) from `df_util`.

Annealing detail: `fit(initial_temp=…, decay_rate=…, period=…, min_temp=…)` sets `self.temp` on a decaying schedule; higher temp = more stochastic latent flips. Fits are multi-start (run several times, keep best) because the objective is non-convex.

## Neural-network experiment framework

- **`super_experiments.py`** — base `Task`, `Model`/`PTModel`, and `Experiment` classes (composition over inheritance; `Experiment(task, model).run()` then `.save_experiment(SAVE_DIR)`). `experiments.py` has the concrete experiment subclasses; `experiments_old.py` is legacy.
- **`students.py`** — PyTorch network modules. **`tasks.py`**, **`grammars.py`**, `recurrent_tasks.py` — synthetic datasets/labelings. `util.py`, `pt_util.py`, `plotting.py`, `anime.py` — general/torch/plot/animation helpers.
- **Cluster sweeps**: `server_utils.py` builds the Cartesian product of parameter dicts (`ParamSet`/`ParamIter`) and `send_to_server(...)` pickles per-config `task_*.pkl`/`model_*.pkl` files and submits a SLURM array job from `job_script_template.sh`. Each array task runs `run_experiment.py $IDX $NDAT`, which unpickles its task+model by index, runs the experiment, and saves. To launch a sweep, edit and run a `scripts/send_*.py` script (e.g. `send_experiments.py`).

## Conventions to follow

- Match the surrounding file's style. New analysis is a `#%%`-celled script in `scripts/` with a hardcoded `CODE_DIR`/`SAVE_DIR`/`LOAD_DIR` header, not a packaged module.
- A new BMF model in the current stack is a `@dataclass` over `LinearGaussianBMF` that assembles an operator from `bae_weights` and a prior from `bae_priors` — it inherits the E/M-step and normally needs no new kernel. Only a genuinely new decoder shape needs a new operator (and, if its gram is not dense, a new scaffold in `bae_search.py`).
- Heavy inner loops are numba `@njit` or Cython — keep them dependency-free (plain numpy/math, no Python objects) so they stay compilable.
