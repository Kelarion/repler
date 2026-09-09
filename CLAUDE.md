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

This is the part to understand before touching model code. The pieces:

- **`bae_models.py`** — model classes (`SemiBMF`, `SpikeNMF`, `BiPCA`, `KernelBMF`/`KernelBMF2`, `JBMF`, `SCPD`, `ConvBMF`, `RRBMF`, …), all `@dataclass`es subclassing `BMF`. The `BMF` base defines the fitting loop (`fit` runs annealing; `grad_step` = one `EStep` then `MStep`) and the contract each subclass must implement:
  - `init_params` / `init_latents` (called by `initialize`)
  - `EStep(S, X)` — update binary latents `S` (delegates to a numba search kernel)
  - `MStep(ES, X)` — update continuous params (`W`, `scl`, intercept `b`), returns loss
  - `__call__(S)` — reconstruct `X̂`; `loss`; `loglikelihood`; `sample` (Gibbs sampling of `S`)
  - State lives on the instance: `self.S` (latents), `self.W`, `self.scl`, `self.temp`. In `SpikeNMF`, `self.S` is the **binary spike** and `self.Z` the **rectified continuous multipliers** — but its `EStep`/`grad_step` *return* the product `S*Z` (that product is the continuous `ES` passed to `MStep` and used to reconstruct `X`).
- **`bae_search.py`** — `@njit` (numba) coordinate-descent kernels that perform the discrete E-step (`bae`, `sbmf`, `bpca`, `snmf`, `kerbmf*`, `convbmf`, …). One kernel per model; this is where the per-element flip probabilities and regularizers (sparsity, `tree_reg`) are computed.
- **`spbae_search.pyx` / `spbae_util.pyx`** — Cython reimplementations of the same kernels for sparse/large problems, exposed via the compiled `spbae` module and used by `spbae_models.py`.
- **`bae_util.py`** — the optimizer/eval layer, *not* the models: the `Neal` annealing scheduler, and cross-validation routines. **`impcv` (imputation-based CV) is the primary model-selection tool** — it masks a fraction of entries, refits while imputing them, and reports held-out vs train log-likelihood. `bicv`/`fpcv`/`kerimpcv`/`splitfit`/`multifit` are alternatives.
- **`bae.py`** — thin wrapper functions tying the above together for common fit patterns.
- `df_util.py` / `distance_factorization.py` / `df_models.py` are the older distance-factorization predecessors; `bae_util` and `bae_models` import helpers (e.g. `permham`) from `df_util`.

Annealing detail: `fit(initial_temp=…, decay_rate=…, period=…, min_temp=…)` sets `self.temp` on a decaying schedule; higher temp = more stochastic latent flips. Fits are multi-start (run several times, keep best) because the objective is non-convex.

## Neural-network experiment framework

- **`super_experiments.py`** — base `Task`, `Model`/`PTModel`, and `Experiment` classes (composition over inheritance; `Experiment(task, model).run()` then `.save_experiment(SAVE_DIR)`). `experiments.py` has the concrete experiment subclasses; `experiments_old.py` is legacy.
- **`students.py`** — PyTorch network modules. **`tasks.py`**, **`grammars.py`**, `recurrent_tasks.py` — synthetic datasets/labelings. `util.py`, `pt_util.py`, `plotting.py`, `anime.py` — general/torch/plot/animation helpers.
- **Cluster sweeps**: `server_utils.py` builds the Cartesian product of parameter dicts (`ParamSet`/`ParamIter`) and `send_to_server(...)` pickles per-config `task_*.pkl`/`model_*.pkl` files and submits a SLURM array job from `job_script_template.sh`. Each array task runs `run_experiment.py $IDX $NDAT`, which unpickles its task+model by index, runs the experiment, and saves. To launch a sweep, edit and run a `scripts/send_*.py` script (e.g. `send_experiments.py`).

## Conventions to follow

- Match the surrounding file's style. New analysis is a `#%%`-celled script in `scripts/` with a hardcoded `CODE_DIR`/`SAVE_DIR`/`LOAD_DIR` header, not a packaged module.
- A new BMF model is a `@dataclass(BMF)` implementing the E/M-step contract above, with its discrete E-step as a new `@njit` kernel in `bae_search.py` (mirror an existing one).
- Heavy inner loops are numba `@njit` or Cython — keep them dependency-free (plain numpy/math, no Python objects) so they stay compilable.
