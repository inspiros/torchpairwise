"""Validation adapters for native finite-iteration Sinkhorn divergence."""
import numbers
import torch
from ._batching import is_finite_real, validate_values


def _parameters(kwargs):
    p = kwargs.get('p', 2)
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or p not in (1, 2):
        raise ValueError('Sinkhorn p must be 1 or 2')
    epsilon = kwargs.get('regularization', 0.1)
    if (isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real)
            or not is_finite_real(epsilon) or epsilon <= 0):
        raise ValueError('regularization must be a positive finite real scalar')
    max_iter = kwargs.get('max_iter', 100)
    if type(max_iter) is not int:
        raise TypeError('max_iter must be an integer')
    if max_iter < 1:
        raise ValueError('max_iter must be positive')
    return p, epsilon, max_iter


def validate_sinkhorn(x, y, kwargs):
    p, epsilon, max_iter = _parameters(kwargs)
    for value, lengths in ((x, kwargs.get('x1_lengths')), (y, kwargs.get('x2_lengths'))):
        valid = torch.ones(value.shape[:-1], device=value.device, dtype=torch.bool)
        if lengths is not None:
            valid = torch.arange(value.shape[-2], device=value.device) < lengths.unsqueeze(-1)
        validate_values((value.isfinite() | ~valid[..., None]).all(), 'Sinkhorn valid coordinates must be finite')
    if not torch.finfo(x.dtype).tiny <= epsilon <= torch.finfo(x.dtype).max:
        raise ValueError('regularization must be a positive normal finite value in the compute dtype')


def evaluate_sinkhorn(spec, x, y, kwargs):
    validate_sinkhorn(x, y, kwargs)
    return torch.ops.torchpairwise.transport_comparisons(x, y, 'sinkhorn', **kwargs)
