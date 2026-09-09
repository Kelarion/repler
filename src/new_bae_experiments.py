"""
PROTOTYPE: server-interface Task + Model for running `new_bae_models` numerics
sweeps, in the style of scripts/send_numerics_experiments.py.

Two pieces, both drop-in for the super_experiments / server_utils machinery:

  StructuredCats  -- a CatTask (like SchurCategories / GridCategories) whose
                     latents come from a disjoint union of undirected graphical
                     models.  This is exp.SparseStructured reorganized so that
                     EVERY field is a folder-nameable, sweepable scalar.  The
                     block structure is carried by a compact STRING spec instead
                     of a list-of-dicts (see `parse_spec`), which is what let the
                     original break both parse_params (a list is read as a sweep
                     set) and folder_hierarchy (a list of dicts stringifies to an
                     illegal path).  Structure-specific arguments ride in an
                     optional (k=v,...) group per block: 'grid6(d=3)',
                     'tree4(alpha=2)', 'tree4x2(rho=0.5,beta=1)'.

  NewBMF          -- the BMFModel that experiments.py never had for the
                     new_bae_models API.  Same run_model/(loss, S, time) contract
                     as experiments.KBMF / experiments.SBMF, so BMFModel.fit and
                     all its recovery metrics work unchanged.

Run a sweep from scripts/send_sparse_numerics.py (kept separate so the pickled
class references resolve as `new_bae_experiments.NewBMF`, not `__main__.NewBMF`).

Improvement suggestions are collected at the bottom of this file.
"""

import re
from time import time
from dataclasses import dataclass, fields as dc_fields

import numpy as np
import scipy.linalg as la

import util
import df_util
import bae_util
import experiments as exp          # CatTask + BMFModel live here
import new_bae_models as nbm


# ===========================================================================
#  Task:  StructuredCats  (folder-friendly reorganization of SparseStructured)
# ===========================================================================

# struct abbreviations used in the block spec mini-language
_STRUCTS = {'cat': 'categorical', 'tree': 'tree', 'grid': 'grid', 'none': 'none'}
#            struct   K       xblowup        (k=v,k=v)
_BLOCK_RE = re.compile(r'^([a-z]+)(\d+)(?:x(\d+))?(?:\(([^()]*)\))?$')


def _coerce(v):
    """int -> float -> bool -> str, in that order (spec kwarg values are text)."""
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v in ('True', 'False'):
        return v == 'True'
    return v


def _parse_kwargs(s):
    """'d=2,alpha=0.5' -> {'d': 2, 'alpha': 0.5}."""
    kw = {}
    for pair in s.split(','):
        if not pair.strip():
            continue
        key, sep, val = pair.partition('=')
        if not sep:
            raise ValueError(f'bad kwarg {pair!r} (want key=value)')
        kw[key.strip()] = _coerce(val.strip())
    return kw


def _split_blocks(spec):
    """Split on '+' at paren depth 0, so kwargs values (e.g. a '1e+5') are safe."""
    parts, depth, cur = [], 0, ''
    for ch in spec:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == '+' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts


def parse_spec(spec):
    """Parse a block spec string into df_util.UndirectedModel blocks.

    Grammar:  blocks joined by '+', each block  <struct><K>[x<blowup>][(k=v,...)]
        'cat8'              categorical, K=8
        'cat3x3'            categorical, K=3, node blow-up 3   (-> 9 dims)
        'tree4'             tree, K=4
        'tree4(alpha=2)'    tree, K=4, alpha=2 forwarded to random_tree_couplings
        'grid6(d=3)'        grid, K=6, d=3  (grid *requires* d)
        'tree4x2(rho=0.5,beta=1)'   blow-up 2 AND structure kwargs
        'cat5+tree4x2+none3'        three disjoint blocks, unioned

    The optional (k=v,...) group carries the structure-specific arguments of the
    underlying df_util couplings builder: grid needs `d`; tree takes rho/alpha/beta;
    categorical/none take none.  Values are coerced int -> float -> bool -> str.
    Blow-up (xN) stays separate -- it is a UndirectedModel operation, not a
    coupling kwarg -- and must precede the paren group.

    Returns a single (possibly unioned) df_util.UndirectedModel.  The whole spec
    stays a scalar string: parse_params leaves it fixed (or sweep it with
    su.Set([...])), and it hashes into the folder like any other arg.
    """
    models = []
    for token in _split_blocks(spec):
        m = _BLOCK_RE.match(token.strip())
        if m is None:
            raise ValueError(f'bad block spec {token!r} '
                             f'(want e.g. "cat8", "tree4x2", "grid6(d=3)")')
        abbr, K, blow, kwargs = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if abbr not in _STRUCTS:
            raise ValueError(f'unknown struct {abbr!r} in {token!r}; '
                             f'known: {sorted(_STRUCTS)}')
        models.append(df_util.UndirectedModel(
            K=K, struct=_STRUCTS[abbr], blowup=int(blow) if blow else 1,
            kwargs=_parse_kwargs(kwargs) if kwargs else None))
    model = models[0]
    for m in models[1:]:
        model = model.union(m)
    return model


@dataclass(kw_only=True)
class StructuredCats(exp.CatTask):
    """Latents from a disjoint union of undirected graphical models.

    Same generative model as exp.SparseStructured, but the block structure is a
    scalar string (`spec`, see parse_spec) rather than a list of dicts, so the
    task drops straight into a server_utils sweep with clean folder names.

    Inherited CatTask scalars: samps, snr, ratio, orth, seed, nonneg.
    """

    spec: str = 'cat8'      # block mini-language, e.g. 'cat5+tree4x2'
    N: int = 64             # number of observations
    temp: float = 1.0       # Gibbs temperature of the generative sampler
    slab: bool = False      # multiply binary latents by Gamma magnitudes

    def gen_latents(self):
        model = parse_spec(self.spec)
        S = model.sample(self.N, temp=self.temp)                   # (N, K) binary
        if self.slab:
            S = S * np.random.gamma(8, 1 / 8, size=S.shape)
        # true generative couplings, with the local fields h on the diagonal.
        # Returned as an `extras` dict (see CatTask.sample): this is the only
        # CatTask with a graphical-model ground truth, so the base collects it
        # generically rather than every task carrying a Jtrue.
        Jtrue = model.J + np.diag(model.h)
        return S, {'Jtrue': Jtrue}


# ===========================================================================
#  Model:  NewBMF  (BMFModel wrapper around new_bae_models)
# ===========================================================================

# kinds exposed to sweeps -> new_bae_models classes.  Every one takes dim_hid as
# its first positional arg and yields latents via the `.S` property.
KINDS = {
    'SemiBMF': nbm.SemiBMF,
    'JBMF':    nbm.JBMF,
    'BiPCA':   nbm.BiPCA,
    'RRBMF':   nbm.RRBMF,
    'SCPD':    nbm.SCPD,
}


def _ctor_kwargs(cls, kw):
    """Keep only the kwargs that `cls` (a dataclass) actually declares -- lets one
    flat kwarg dict feed models with different constructors (e.g. BiPCA has no
    weight_pr_reg / nonneg)."""
    names = {f.name for f in dc_fields(cls)}
    return {k: v for k, v in kw.items() if k in names}


def _prior_coupling(prior, best=None):
    """The fitted prior's coupling `J` (m, m), or None if it has no coupling.
    Both structured priors keep it symmetric (BoltzmannPrior continuous, MRFPrior
    in {-1,0,+1}), so this just picks the chain when the prior is chain-batched."""
    J = getattr(prior, 'J', None)
    if not isinstance(J, np.ndarray):
        return None
    return J if best is None else J[best]


@dataclass
class NewBMF(exp.BMFModel):
    """new_bae_models fit under the BMFModel contract (mirror of exp.KBMF/SBMF).

    `kind` picks the concrete model (see KINDS).  Regularizers are the flat
    super-set; each is routed to the chosen model only if it declares it.
    """

    dim_hid: float = None        # rank as a fraction of the true latent dim (see BMFModel.fit)
    kind: str = 'JBMF'

    # regularizers (filtered per model)
    nonneg: bool = False
    sparse_reg: float = 0.0
    tree_reg: float = 1e-2
    weight_pr_reg: float = 1e-2
    weight_l2_reg: float = 1e-2
    J_lr: float = 0.0
    J_prior: str = 'boltzmann'   # BiPCA only: 'none' | 'boltzmann' | 'mrf'.  'mrf' is
                                 # the sign-constrained coupling, needs n_chains > 1,
                                 # and ignores J_lr (it samples J, it does not descend
                                 # it) -- pair it with binarize_J=False.
    J_temp: float = 0.0          # 'mrf' only: sampler temperature over J (<=0 = ICM)
    J_sweeps: int = 1            # 'mrf' only: Gibbs sweeps over J per M-step

    # annealing schedule (matches new_bae_models.BMF.fit)
    T0: float = 10.0
    decay_rate: float = 0.9
    period: int = 8
    min_temp: float = 1.0
    max_iter: int = None

    # init / misc
    scl_lr: float = 1e-3
    lr: float = 1e-3
    n_chains: int = 1
    folds: int = None
    hot_start: bool = True
    resample_dead: bool = True
    rescale: bool = True         # fit on X / X.std() (helps convergence)
    posterior_samps: int = 12    # Gibbs draws scored + averaged by BMFModel.fit
    binarize_J: bool = True      # extra_metrics: split the recovered coupling into
                                 # edges/non-edges (2-means) before scoring it.
                                 # Needed for a continuous J; set False for J_prior=
                                 # 'mrf', whose J is already {-1,0,+1}.

    def run_model(self, X, h):

        ModelCls = KINDS[self.kind]
        ctor = dict(nonneg=self.nonneg, sparse_reg=self.sparse_reg,
                    tree_reg=self.tree_reg, weight_pr_reg=self.weight_pr_reg,
                    weight_l2_reg=self.weight_l2_reg, J_lr=self.J_lr,
                    J_prior=self.J_prior, J_temp=self.J_temp,
                    J_sweeps=self.J_sweeps, n_chains=self.n_chains)
        mod = ModelCls(h, **_ctor_kwargs(ModelCls, ctor))

        # let nonneg models recover columns that die during the fit
        op = getattr(mod, 'operator', None)
        if op is not None and hasattr(op, 'resample_dead'):
            op.resample_dead = self.resample_dead

        # copy either way so `fit`'s in-place imputation never mutates caller's X
        Xs = X / X.std() if self.rescale else 1.0 * X

        # optional imputation CV: hold out fold 0 (~1/folds of the entries); fit
        # imputes those in place, so keep a pristine copy to score them against.
        if self.folds is not None:
            mask = (bae_util._kfold_assignment(X.shape, self.folds) == 0)
            Xtrue = 1.0 * Xs
        else:
            mask = None

        t0 = time()
        en = mod.fit(Xs, initial_temp=self.T0, decay_rate=self.decay_rate,
                     period=self.period, min_temp=self.min_temp,
                     max_iter=self.max_iter, scl_lr=self.scl_lr,
                     lr=self.lr, mask=mask,
                     hot_start=self.hot_start, verbose=False)
        T = time() - t0

        # posterior draws (n_samp, N, K) at the final annealed temperature;
        # BMFModel.fit scores each and averages the metrics.
        samps = mod.sample(Xs, n_samp=self.posterior_samps)

        # held-out vs train log-likelihood on the imputed fit (impcv-style):
        # reconstruct from the (partially imputed) data, score against the true
        # values.  Stashed for extra_metrics; None when not imputing.
        if mask is not None:
            ll = mod.loglikelihood(Xtrue, mod(samps))
            self._heldout = {'test_ll': float(np.mean(ll[..., mask])),
                             'train_ll': float(np.mean(ll[..., ~mask]))}
        else:
            self._heldout = None

        self._model = mod       # kept so extra_metrics can read latent_prior.J
        return en[-1], samps, T

    def extra_metrics(self, X, Strue, Ss, Jtrue=None, **rest):
        """Per-fit metrics: held-out log-likelihood and coupling recovery.

        Both are optional and self-gating: `test_ll`/`train_ll` appear only when
        fitting with imputation (folds set, see run_model); the coupling scores
        appear only when the task supplied Jtrue AND the chosen model's
        latent_prior has a coupling (BoltzmannPrior's J, MRFPrior's J).
        """
        out = {}
        if getattr(self, '_heldout', None) is not None:
            out.update(self._heldout)          # test_ll, train_ll

        mod = getattr(self, '_model', None)
        prior = getattr(mod, 'latent_prior', None)
        best = mod.best_chain() if (mod is not None and mod._multi) else None
        Jeff = _prior_coupling(prior, best)    # recovered symmetric coupling
        if Jtrue is None or Jeff is None:
            out.update({'j_cos': np.nan, 'j_ged': np.nan})
            return out
        import networkx as nx

        # align recovered latents to the true ones (thresholded posterior mean),
        # then reorder both coupling matrices to the matched set so they compare.
        Shat = 1 * (Ss.mean(0) > 0.9)
        aye, jay = df_util.permham_idx(Strue, Shat)
        sgn = np.sign(np.diag((2*Strue[:,aye]-1).T@(2*Shat[:,jay]-1)))

        if self.binarize_J:
            # a continuous coupling has no scale in common with Jtrue's {-1,0,1}, so
            # keep only the "significant" edges (2-means on |off-diagonal|) and take
            # the sign of what survives.
            edge = df_util.binarize(np.abs(util.vec(Jeff))[None], axis=1).squeeze()
            # thresh = np.mean(np.abs(util.vec(Jeff))[np.abs(util.vec(Jeff))> 1e-2]) * 0.1
            # edge = np.abs(util.vec(Jeff)) > thresh
            mask = util.mat(edge)
            Jhat = np.outer(sgn,sgn) * np.sign((mask*Jeff)[np.ix_(jay, jay)])
        else:
            # MRFPrior's coupling is already in {-1,0,+1}: it is directly comparable
            # to Jtrue, and running it through the 2-means split would only risk
            # inventing a threshold where the values are already discrete (and it is
            # ill-posed when J comes out all-zero or all-nonzero).
            Jhat = np.outer(sgn,sgn) * Jeff[np.ix_(jay, jay)]
        Jtru = Jtrue[np.ix_(aye, aye)] - np.diag(np.diag(Jtrue)[aye])

        j_cos = np.abs(util.vec(Jhat)@util.vec(Jtru)) / (la.norm(util.vec(Jtru))*la.norm(util.vec(Jhat)))
        j_ged = nx.graph_edit_distance(
            nx.Graph(np.abs(Jhat)),
            nx.Graph(np.abs(Jtru)))

        out.update({'j_cos': j_cos, 'j_ged': j_ged})
        return out


# ===========================================================================
#  Suggestions for improving this infrastructure (see module docstring)
# ===========================================================================
#
# 1. parse_params/folder_hierarchy vs. structured params.  The real barrier is
#    that server_utils treats `type(v) is list` as a sweep set and stringifies
#    every value into a path.  Any non-scalar task/model arg (list, dict) breaks
#    BOTH.  StructuredCats sidesteps it with a string spec, but the general fix
#    is a small hook: let a Task/Model expose `folder_name(key, val)` /
#    `sweep_values(key, val)` so a structured arg can define its own short slug
#    and its own (non-)expansion.  Then list-of-dicts specs could stay.
#
# 2. `it` leakage in BMFModel subclasses.  experiments.BAE.run_model references
#    `X[it]` with `it` undefined in its scope (works only by accident of a module
#    global).  KBMF/SBMF/NewBMF correctly use `X`.  Worth fixing BAE to match.
#
# 3. SparseStructured latent bug.  Its per-block dict can carry a 'blowup' key
#    (per the docstring example) that is left in `kwargs` and forwarded into
#    category_couplings/random_tree_couplings, which don't accept it -> TypeError.
#    StructuredCats handles blow-up explicitly, so it can't happen here.
#
# 4. Posterior scoring (DONE).  NewBMF returns `posterior_samps` Gibbs draws and
#    BMFModel.fit (refactored into `score` + a sample-averaging loop) averages the
#    recovery metrics over them, instead of scoring the single point estimate S.
#    A natural extension is a posterior-uncertainty metric (mean marginal entropy
#    of <S>), which needs the sample stack `score` currently reduces away.
