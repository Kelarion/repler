"""
new_bae_priors_parallel.py  --  PROTOTYPE (parallel chains)
===========================================================

Parallel-chains variant of new_bae_priors.LatentPrior.  The whole latent side is
identical to the serial one except that every state array carries a leading chain
axis C, and the two numpy operations that build it (the sign-of-drive spike and
S^T S) become chain-batched einsums -- no Python loop over chains.

  S    (C, n, m)     per-chain binary spike
  Z    (C, n, m)     per-chain effective latent (== S with no slab)
  StS  (C, m, m)     per-chain S^T S

Scope of the prototype: the plain, no-slab LatentPrior (which backs SemiBMF).
`slab=True` and the structured BoltzmannPrior are straightforward extensions and
are sketched at the bottom -- the only real question there is per-chain J, which
is just another leading axis on J_W / J_h (C, m, m) / (C, m).
"""

import numpy as np
from dataclasses import dataclass

import new_bae_search_parallel as nbsp


@dataclass
class ParallelLatentPrior:
    """Independent Bernoulli prior (sparsity + tree), replicated across C chains.
    Same contract as new_bae_priors.LatentPrior -- link / prior_plugin / init /
    coupling / learn -- with a leading chain axis on the state."""

    n_chains: int = 8
    sparse_reg: float = 0.0
    tree_reg: float = 1e-2
    slab: bool = False           # prototype: no-slab only (see note at bottom)
    slab_prior: float = 1.0

    @property
    def link(self):
        # same choice as the serial prior; the parallel scaffold takes the same
        # (score, aux_update) pair (imported straight through from new_bae_search).
        return nbsp.SLAB_LINK if self.slab else nbsp.BINARY_LINK

    @property
    def prior_plugin(self):
        return nbsp.PRIOR_PLAIN

    def init_params(self, S0, **opt_args):
        return None

    # ---- latent state, chain-batched --------------------------------------
    def init_latents(self, drive):
        """`drive` is the operator's per-chain drive, shape (C, n, m) -- distinct
        across chains because each chain has its own W, which is exactly what makes
        the chains explore different basins.  Spike / StS / Z are the serial code
        with a `c` einsum axis."""
        self.S = 1.0 * (drive >= 0)                       # (C, n, m)
        self.StS = self.S.transpose(0, 2, 1) @ self.S     # (C, m, m), batched BLAS
        self.Z = (drive * self.S) if self.slab else self.S
        self.init_params(self.S)

    def coupling(self):
        """Zeros for the plain prior; the scaffold's coupling loop then adds
        nothing.  Chain-axis shapes so the kernel can index Jc[c] / hc[c]."""
        C, _, m = self.S.shape
        return np.zeros((C, m, m)), np.zeros((C, m))

    def learn(self, ES):
        return None


# ---------------------------------------------------------------------------
#  NOTE -- generalizing to slab and to the Boltzmann prior
# ---------------------------------------------------------------------------
#
#  * slab=True  needs nothing here: init_latents already branches, the link
#    property already returns SLAB_LINK, and the parallel scaffold already threads
#    Z / mu / nu per chain.  The only caveat is that the truncated-normal draw in
#    aux_slab runs inside the prange -- fine, numba's RNG is thread-local.
#
#  * BoltzmannPrior would carry per-chain coupling J_W (C, m, m) and field
#    J_h (C, m); `coupling()` returns the same additive (Jc, hc) split with a
#    leading axis, and `learn()` does the pseudolikelihood/MLE gradient batched
#    over chains (einsum with a `c` axis, exactly like the operator's backward).
#    The MLE sampler's fantasy particles gain a chain axis too.  No new kernel.
