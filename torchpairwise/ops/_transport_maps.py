"""Streamed conditional maps of finite-iteration balanced entropic transport."""
import torch
from ._validation import _check_chunk_size
from ._sinkhorn import _parameters


def _map(x, y, source_weights, target_weights, regularization, max_iter, p, chunk_size, displacement):
    _parameters(dict(p=p, regularization=regularization, max_iter=max_iter))
    _check_chunk_size(chunk_size)
    return torch.ops.torchpairwise.sinkhorn_map(
        x, y, source_weights, target_weights, regularization, max_iter, p,
        chunk_size, displacement)


def sinkhorn_barycentric_map(x, y, *, source_weights=None, target_weights=None,
                              regularization=0.1, max_iter=100, p=2, chunk_size=128):
    """Map source points [...,N,D] into target coordinates [...,M,D].

    Uses normalized nonnegative masses (uniform by default) and ||x-y||**p
    entropic ground costs, p=1 or 2. Fixed alternating log-domain iterations
    approximate balanced OT; final conditional rows normalize to one. Zero
    source-mass rows return zero. Batch axes broadcast, including mass batches.
    Forward/backward stream cost tiles; differentiating iterations retains
    O(max_iter*(N+M)) dual state. First derivatives include mass/solver dependence.
    No full plan or pair-displacement tensor is returned or saved.
    """
    return _map(x, y, source_weights, target_weights, regularization, max_iter, p,
                chunk_size, False)


def sinkhorn_displacement(x, y, *, source_weights=None, target_weights=None,
                          regularization=0.1, max_iter=100, p=2, chunk_size=128):
    """Barycentric map minus source positions; zero-source-mass rows return zero."""
    return _map(x, y, source_weights, target_weights, regularization, max_iter, p,
                chunk_size, True)


__all__ = ['sinkhorn_barycentric_map', 'sinkhorn_displacement']
