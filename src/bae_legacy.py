"""
bae_legacy.py  --  load BMF fits pickled before the bae_* rename and cleanup
============================================================================

Fits saved from the old stack name their classes `new_bae_models.BiPCA`,
`new_bae_weights.Procrustes`, `new_bae_priors.BoltzmannPriorNP`, none of which
exist any more.  `load(path)` reads them anyway and returns a model built out of
today's classes.

Two things make this more than a rename.

1.  The stored attribute sets are older than today's classes -- no `n_chains`, no
    prior `temp`, no `m_step` / `var_update`, `lr` instead of `b_lr`/`W_lr`.  Any
    field a rescued object is missing is filled from the current default, so the
    result is a fully-formed modern object rather than a half-populated one.

2.  `BoltzmannPriorNP` stored its coupling in the SPIN convention and folded it
    for the search through a `coupling()` that did not reproduce the conditional
    log-odds of either convention (this is the bug the cleanup fixed).  The codes
    in a saved fit were nevertheless sampled under that fold, so the rescue
    preserves the EFFECTIVE coupling -- the (Jc, hc) the search actually saw --
    rather than the nominal parameters:

        J_new = Jc_old / 2 ,   h_new = hc_old / 2

    which is exactly the (J, h) whose `coupling()` returns (Jc_old, hc_old) in
    today's {0,1} parameterization.  The saved fit therefore keeps behaving
    exactly as it did.  The original arrays are kept as `J_spin` / `h_spin`.

    The consequence: a rescued prior REPRODUCES its fit but is not a coupling the
    current `learn()` would have produced.  Use it for inference and analysis;
    refitting from it continues a trajectory that was shaped by the old bug.

Pickle never calls `__init__` -- it makes a bare instance and updates `__dict__` --
so nothing here needs the old code to be importable, and `src/.bak_lgbmf_cleanup/`
is not required.

    import bae_legacy
    mod = bae_legacy.load('best_model.pkl')      # a real bae_models.BiPCA
    bae_legacy.convert_file('best_model.pkl', 'best_model_bae.pkl')
"""

import dataclasses
import pickle

import numpy as np

import bae_models
import bae_priors
import bae_weights

# modules that no longer exist; anything pickled under them is ghosted on load
OLD_MODULES = ('new_bae_models', 'new_bae_weights', 'new_bae_priors',
               'new_bae_search')


class _Ghost:
    """Placeholder a legacy instance is unpickled into: pickle fills __dict__
    without calling __init__, so an empty class is enough to carry the state."""
    _legacy = '?'


def _dropped(*args, **kwargs):
    """Stand-in for the pickled numba dispatcher (operator._kernel).  The kernel
    is rebuilt from today's search anyway, so the stored one is discarded rather
    than reconstructed -- which also avoids compiling on load."""
    return None


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'numba' in module:
            return _dropped
        if module in OLD_MODULES:
            return type(name, (_Ghost,), {'_legacy': f'{module}.{name}'})
        return super().find_class(module, name)


# ---------------------------------------------------------------------------
#  the old coupling fold, reproduced so the rescue can invert it
# ---------------------------------------------------------------------------

def _legacy_coupling(J_W, J_h):
    """`BoltzmannPriorNP.coupling()` verbatim: the (Jc, hc) the old search saw."""
    Jsym = (J_W + J_W.T) / 2
    Jbin = 4 * Jsym + 2 * np.diag(J_h - 2 * Jsym.sum(1))
    Jc = -Jbin.copy()
    np.fill_diagonal(Jc, 0.0)
    return Jc, -0.5 * np.diag(Jbin).copy()


# ---------------------------------------------------------------------------
#  migration
# ---------------------------------------------------------------------------

def _fill_defaults(obj):
    """Give `obj` every dataclass field of its class that the old state lacked."""
    for f in dataclasses.fields(type(obj)):
        if hasattr(obj, f.name):
            continue
        if f.default is not dataclasses.MISSING:
            setattr(obj, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:   # noqa: B009
            setattr(obj, f.name, f.default_factory())
    return obj


def _state(ghost):
    return dict(vars(ghost))


def _rebuild(ghost, module, skip=('_kernel', '_legacy')):
    """A bare instance of today's class of the same name, carrying the old
    state.  Everything is copied across -- the two stacks agree on almost every
    attribute name -- and the caller fixes up whatever genuinely changed."""
    name = ghost._legacy.rsplit('.', 1)[-1]
    cls = getattr(module, name, None)
    if cls is None:
        raise KeyError(f"{ghost._legacy} has no counterpart in "
                       f"{module.__name__}; migrate it by hand")
    new = cls.__new__(cls)
    for k, v in vars(ghost).items():
        if k in skip:
            continue
        setattr(new, k, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
    new.n_chains = getattr(ghost, 'n_chains', 1)
    return new


def migrate_prior(ghost, **over):
    """A legacy latent prior -> today's.  `BoltzmannPriorNP` is the one class
    that was both renamed and reparameterized (see the module docstring)."""
    name = ghost._legacy.rsplit('.', 1)[-1]
    s = _state(ghost)

    if name == 'BoltzmannPriorNP':
        new = bae_priors.BoltzmannPrior.__new__(bae_priors.BoltzmannPrior)
        for k, v in s.items():
            if k in ('_legacy', 'J_W', 'J_h'):
                continue
            setattr(new, k, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        Jc, hc = _legacy_coupling(np.asarray(s['J_W']), np.asarray(s['J_h']))
        new.J, new.h = Jc / 2.0, hc / 2.0           # coupling() -> (Jc, hc) again
        new.J_spin = np.array(s['J_W'], copy=True)  # provenance; unused by the model
        new.h_spin = np.array(s['J_h'], copy=True)
        new.n_chains = s.get('n_chains', 1)
    elif name == 'BoltzmannPrior':
        raise NotImplementedError(
            "a torch-era BoltzmannPrior cannot be migrated: its coupling lived in "
            "nn.Linear parameters that are gone.  Refit, or rescue by hand.")
    elif name == 'MRFPrior':
        raise NotImplementedError(
            "MRFPrior's coupling convention also changed; migrate it by hand "
            "(see MRFPrior.coupling in bae_priors).")
    else:
        new = _rebuild(ghost, bae_priors)

    for k, v in over.items():
        setattr(new, k, v)
    return _fill_defaults(new)


def migrate_operator(ghost, m_step='legacy', **over):
    """A legacy operator -> today's.  `Procrustes` is the one that changed shape:
    its single `lr` became `b_lr`/`W_lr`, `scl` became an array, and its M-step
    was rewritten -- so `m_step` defaults to 'legacy', the rule that produced the
    fit, and continuing the fit stays in that regime."""
    new = _rebuild(ghost, bae_weights)
    if isinstance(new, bae_weights.Procrustes):
        s = _state(ghost)
        new.scl = np.asarray(s.get('scl', 1.0), dtype=float)
        lr = float(s.get('lr', s.get('b_lr', 1.0)))   # the old single decoder rate
        new.b_lr = new.W_lr = lr
        new.m_step = m_step
    for k, v in over.items():
        setattr(new, k, v)
    return _fill_defaults(new)


def migrate_model(ghost, m_step='legacy', var_update='legacy', **over):
    """A legacy model -> today's, with its operator and prior migrated too."""
    s = _state(ghost)
    new = _rebuild(ghost, bae_models, skip=('_kernel', '_legacy',
                                            'operator', 'latent_prior'))
    new.operator = migrate_operator(s['operator'], m_step=m_step)
    new.latent_prior = migrate_prior(s['latent_prior'])
    new.dim_hid = s.get('dim_hid', getattr(new.operator, 'dim_hid', None))

    new.var_update = var_update
    if isinstance(new, bae_models.BiPCA):
        new.m_step = m_step
    if not hasattr(new, 'J_prior'):
        new.J_prior = ('boltzmann' if isinstance(new.latent_prior,
                                                 bae_priors.BoltzmannPrior)
                       else 'none')
    new._Ximp = None
    new._best_op = None
    new.outs = list(s.get('outs', []))
    for k, v in over.items():
        setattr(new, k, v)
    _fill_defaults(new)

    # the stored numba kernel was dropped on load; rebuild from today's search
    new.operator.build_search(new.latent_prior.link,
                              new.latent_prior.prior_plugin,
                              debug=getattr(new, 'debug', False))
    return new


def migrate(obj, **kw):
    """Migrate a legacy object, or walk a list/tuple/dict of them."""
    if isinstance(obj, _Ghost):
        kind = obj._legacy.rsplit('.', 1)[-1]
        if kind.endswith(('Operator', 'Op')) or kind == 'Procrustes':
            return migrate_operator(obj, **kw)
        if 'Prior' in kind:
            return migrate_prior(obj, **kw)
        return migrate_model(obj, **kw)
    if isinstance(obj, dict):
        return {k: migrate(v, **kw) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(migrate(v, **kw) for v in obj)
    return obj


# ---------------------------------------------------------------------------

def load_raw(path):
    """Unpickle without migrating -- legacy objects come back as ghosts."""
    with open(path, 'rb') as fh:
        return LegacyUnpickler(fh).load()


def load(path, **kw):
    """Unpickle a legacy fit and return it rebuilt on today's classes."""
    return migrate(load_raw(path), **kw)


def convert_file(src, dst, **kw):
    """Migrate `src` and write it to `dst` (the original is left alone)."""
    obj = load(src, **kw)
    with open(dst, 'wb') as fh:
        pickle.dump(obj, fh, protocol=4)
    return obj
