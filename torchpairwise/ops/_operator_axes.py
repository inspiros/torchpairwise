"""Dimension metadata shared by scalar and residual operator frontends."""
from torch import Tensor


def _operator_dims(dim, ndim, logical_rank):
    if dim is None or (isinstance(dim, (tuple, list)) and len(dim) == 0):
        return tuple(range(-logical_rank, 0))
    if type(dim) is int:
        dims = (dim,)
    elif isinstance(dim, (tuple, list)):
        dims = tuple(dim)
    else:
        raise TypeError('dim must be an integer, tuple/list of integers, or None')
    result = []
    for axis in dims:
        if type(axis) is not int:
            raise TypeError('dim entries must be integers')
        if axis < -ndim or axis >= ndim:
            raise ValueError('dim is outside the operator rank')
        normalized = axis if axis < 0 else axis - ndim
        if normalized < -logical_rank:
            raise ValueError('Operator reductions preserve batch axes')
        if normalized in result:
            raise ValueError('dim contains a repeated axis')
        result.append(normalized)
    return tuple(sorted(result))


def _scalar_dim(dim, operator, weights=None):
    # Canonical negative axes do not need source metadata. Preserve native
    # parameter validation (and avoid re-walking a DAG) for ordinary reductions.
    dims = (dim,) if type(dim) is int else dim if isinstance(dim, (tuple, list)) else ()
    positive = any(type(axis) is int and axis >= 0 for axis in dims)
    ndim = len(operator.shape) if positive else 2
    if positive and weights is not None:
        ndim = max(ndim, weights.ndim + 1 if isinstance(weights, Tensor) else len(weights.shape))
    axes = _operator_dims(dim, ndim, 2)
    return None if len(axes) == 2 else axes[0]
