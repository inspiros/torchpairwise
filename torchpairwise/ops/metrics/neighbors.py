"""Dense neighbor masks; use exact k and stable column-order tie breaking."""
import torch
from .._batching import call_pairwise
from .._registry import DISTANCES, resolve


def _distances(x1, x2, metric, exclude_self, kwargs):
    self_comparison = x2 is None
    exclude = self_comparison if exclude_self is None else exclude_self
    result = call_pairwise(resolve(metric, DISTANCES).op, x1, x2,
                           **{k: v for k, v in kwargs.items() if v is not None})
    dists = result[0] if isinstance(result, tuple) else result
    if exclude and dists.shape[-1] != dists.shape[-2]:
        raise ValueError('Self exclusion requires equal sample counts')
    return dists, exclude


def k_neighbors_mask(x1, x2=None, k=1, metric='euclidean', *, exclude_self=None,
                     w=None, V=None, VI=None, p=None, base=None, shuffle=None, generator=None,
                     batched=None, feature_dim=None,
                     squared=None, x1_lengths=None, x2_lengths=None, chunk_size=None, bin_positions=None,
          projections=None, regularization=None, max_iter=None, kernel=None, length_scale=None, alpha=None):
    """Select exactly k nearest columns per row, excluding self when x2 is omitted.

    Equal distances are ordered by column index. k must not exceed the number
    of eligible columns. Output has the broadcast all-pairs shape and bool dtype.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError('k must be a non-negative integer')
    with torch.no_grad():
        dists, exclude = _distances(x1, x2, metric, exclude_self,
                                   dict(w=w, V=V, VI=VI, p=p, base=base, shuffle=shuffle,
                                        generator=generator, batched=batched, feature_dim=feature_dim,
                                        squared=squared, x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size, bin_positions=bin_positions,
                                    projections=projections, regularization=regularization, max_iter=max_iter, kernel=kernel, length_scale=length_scale, alpha=alpha))
        available = max(0, dists.shape[-1] - int(exclude))
        if k > available:
            raise ValueError(f'k cannot exceed the number of eligible columns ({available})')
        return torch.ops.torchpairwise.neighbor_mask_from_distances(
            dists, 'k', k, None, bool(exclude))


def radius_neighbors_mask(x1, x2=None, epsilon=0., metric='euclidean', *, exclude_self=None,
                          w=None, V=None, VI=None, p=None, base=None, shuffle=None, generator=None,
                          batched=None, feature_dim=None,
                          squared=None, x1_lengths=None, x2_lengths=None, chunk_size=None, bin_positions=None,
          projections=None, regularization=None, max_iter=None, kernel=None, length_scale=None, alpha=None):
    """Select distances <= epsilon, excluding self when x2 is omitted."""
    if not epsilon >= 0:
        raise ValueError('epsilon must be non-negative')
    with torch.no_grad():
        dists, exclude = _distances(x1, x2, metric, exclude_self,
                                   dict(w=w, V=V, VI=VI, p=p, base=base, shuffle=shuffle,
                                        generator=generator, batched=batched, feature_dim=feature_dim,
                                        squared=squared, x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size, bin_positions=bin_positions,
                                    projections=projections, regularization=regularization, max_iter=max_iter, kernel=kernel, length_scale=length_scale, alpha=alpha))
        return torch.ops.torchpairwise.neighbor_mask_from_distances(
            dists, 'radius', None, epsilon, bool(exclude))


__all__ = ['k_neighbors_mask', 'radius_neighbors_mask']


def farthest_point_sampling(D, k, *, start=0):
    """Greedily select k distinct samples from a scalar self-distance map.

    D is an entrywise [...,N,N] lazy source or dense Tensor; linear-only maps
    are rejected. Starting at start, each native step reads one row and keeps
    the minimum distance to selected samples, using O(N*k) entry evaluations.
    Ties choose the lowest unselected index. Returns int64 indices [...,k] and
    insertion distances [...,k] in D's dtype (the first distance is infinity).
    Inference-only: use explicit torch.no_grad() for gradient-bearing inputs.
    """
    from ...linalg import _entrywise_lowrank_source, _packet
    if type(k) is not int or type(start) is not int:
        raise TypeError('farthest_point_sampling k and start must be integers')
    D = _entrywise_lowrank_source(D, 'farthest_point_sampling')
    torch._check(D.shape[-2] == D.shape[-1], lambda: 'farthest_point_sampling requires a square self-distance operator')
    anchor = torch.empty((), dtype=D.dtype, device=D.device)
    return torch.ops.torchpairwise.linear_farthest_point_sampling(
        *_packet(D), list(D.shape), anchor, k, start)


def kmeans_plusplus(D, k, *, generator=None):
    """Stream k distinct k-means++ seeds from squared self-distances.

    D is a scalar entrywise [...,N,N] lazy source or dense Tensor containing
    squared distances; do not pass unsquared Euclidean distances. The first
    index is uniform; subsequent draws are proportional to minimum D over
    selected rows. Each step reads one row, for O(N*k) entries. Zero remaining
    mass selects the lowest unselected index. Returns int64 indices [...,k].

    generator must match D's device. The same generator state, device and input
    shape reproduce draws; determinism across devices is not promised. Without
    a generator the current device RNG is consumed. Inference-only: use explicit
    torch.no_grad() for gradient-bearing inputs. Linear-only maps are rejected.
    """
    from ...linalg import _entrywise_lowrank_source, _packet
    if type(k) is not int:
        raise TypeError('kmeans_plusplus k must be an integer')
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError('kmeans_plusplus generator must be a torch.Generator or None')
    D = _entrywise_lowrank_source(D, 'kmeans_plusplus')
    torch._check(D.shape[-2] == D.shape[-1], lambda: 'kmeans_plusplus requires a square squared self-distance operator')
    anchor = torch.empty((), dtype=D.dtype, device=D.device)
    return torch.ops.torchpairwise.linear_kmeans_plusplus(
        *_packet(D), list(D.shape), anchor, k, generator)


__all__ += ['farthest_point_sampling', 'kmeans_plusplus']
