"""Validation adapters for native empirical energy/MMD comparisons."""
from typing import Optional

import numbers
import torch
from torch import Tensor
from ._batching import validate_values
from ._batching import is_finite_real


def validate_distribution(spec, x, y, kwargs):
    squared = kwargs.get('squared', False)
    if type(squared) is not bool:
        raise TypeError('squared must be a boolean')
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError('Distribution comparisons require float32 or float64 inputs outside autocast')
    energy = spec.op == 'energy_distances'
    kernel = None if energy else kwargs.get('kernel', 'rbf')
    length_scale = kwargs.get('length_scale', 1.)
    if not energy:
        from ._registry import KERNELS
        if (not isinstance(kernel, str) or
                kernel not in (*KERNELS, *(s.op for s in KERNELS.values()), 'euclidean_laplacian')):
            raise ValueError('Unknown MMD kernel; select a named pairwise kernel')
        if kernel in ('sigmoid', 'sigmoid_kernel'):
            raise ValueError('sigmoid is not generally positive semidefinite and cannot define an MMD distance')
        if (isinstance(length_scale, bool) or not isinstance(length_scale, numbers.Real)
                or not is_finite_real(length_scale) or length_scale <= 0):
            raise ValueError('length_scale must be a positive finite scalar')
    for value, lengths in ((x, kwargs.get('x1_lengths')), (y, kwargs.get('x2_lengths'))):
        valid = torch.ones(value.shape[:-1], device=value.device, dtype=torch.bool)
        if lengths is not None:
            valid = torch.arange(value.shape[-2], device=value.device) < lengths.unsqueeze(-1)
        validate_values((torch.isfinite(value) | ~valid.unsqueeze(-1)).all(),
                        'Distribution comparison valid point coordinates must be finite')


def evaluate_distribution(spec, x, y, kwargs):
    validate_distribution(spec, x, y, kwargs)
    return torch.ops.torchpairwise.builtin_comparisons(x, y, spec.op, **kwargs)


def _mmd_statistics(x1, x2, mode, kernel, estimator, length_scale, gamma,
                    degree, coef0, alpha, nu, batched, feature_dim,
                    x1_lengths, x2_lengths, chunk_size):
    if estimator not in ('biased', 'unbiased'):
        raise ValueError('estimator must be biased or unbiased')
    if not isinstance(kernel, str):
        raise TypeError('kernel must be a string')
    for name, value in (('length_scale', length_scale), ('gamma', gamma),
                        ('coef0', coef0), ('alpha', alpha), ('nu', nu)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Real)):
            raise TypeError(f'{name} must be a real scalar')
    if degree is not None and type(degree) is not int:
        raise TypeError('degree must be an integer')
    return torch.ops.torchpairwise.mmd_statistics(
        x1, x2, mode, kernel, estimator, length_scale, gamma, degree, coef0,
        alpha, nu, x1_lengths, x2_lengths, chunk_size, batched, feature_dim)


def mmd_statistics(x1: Tensor, x2: Optional[Tensor] = None, *, kernel: str = 'rbf', estimator: str = 'unbiased',
                   length_scale: float = 1., gamma: Optional[float] = None,
                   degree: Optional[int] = None, coef0: Optional[float] = None,
                   alpha: Optional[float] = None, nu: Optional[float] = None,
                   batched: Optional[bool] = None, feature_dim: int = -1,
                   x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                   chunk_size: Optional[int] = None) -> Tensor:
    """Signed squared empirical MMD; estimator is unbiased (default) or biased.

    Inputs are [...,N,P,C] and [...,M,Q,C]. Unbiased estimates exclude
    within-cloud diagonal terms and require at least two valid points in each
    cloud. Cross terms include every pair, including when comparing a cloud
    with itself. Negative estimates are retained; no root/clamp is applied.
    Select polynomial with degree=3, gamma=1/C, coef0=1 for the statistic used
    by KID on supplied features. No encoder or subset-averaging protocol is
    supplied. Kernel options/domains follow mmd_distances. Native first
    gradients, CPU/CUDA autocast, compile/export and prefix lengths are supported.
    """
    return _mmd_statistics(x1, x2, 'all_pairs', kernel, estimator, length_scale,
                           gamma, degree, coef0, alpha, nu, batched, feature_dim,
                           x1_lengths, x2_lengths, chunk_size)


def paired_mmd_statistics(x1: Tensor, x2: Tensor, *, kernel: str = 'rbf', estimator: str = 'unbiased',
                   length_scale: float = 1., gamma: Optional[float] = None,
                   degree: Optional[int] = None, coef0: Optional[float] = None,
                   alpha: Optional[float] = None, nu: Optional[float] = None,
                   batched: Optional[bool] = None, feature_dim: int = -1,
                   x1_lengths: Optional[Tensor] = None, x2_lengths: Optional[Tensor] = None,
                   chunk_size: Optional[int] = None) -> Tensor:
    """Paired signed MMD statistics; see mmd_statistics."""
    return _mmd_statistics(x1, x2, 'paired', kernel, estimator, length_scale,
                           gamma, degree, coef0, alpha, nu, batched, feature_dim,
                           x1_lengths, x2_lengths, chunk_size)


def condensed_mmd_statistics(x1: Tensor, *, kernel: str = 'rbf', estimator: str = 'unbiased',
                   length_scale: float = 1., gamma: Optional[float] = None,
                   degree: Optional[int] = None, coef0: Optional[float] = None,
                   alpha: Optional[float] = None, nu: Optional[float] = None,
                   batched: Optional[bool] = None, feature_dim: int = -1,
                   x1_lengths: Optional[Tensor] = None,
                   chunk_size: Optional[int] = None) -> Tensor:
    """Condensed signed MMD statistics; see mmd_statistics."""
    return _mmd_statistics(x1, None, 'condensed', kernel, estimator, length_scale,
                           gamma, degree, coef0, alpha, nu, batched, feature_dim,
                           x1_lengths, None, chunk_size)


__all__ = ['mmd_statistics', 'paired_mmd_statistics', 'condensed_mmd_statistics']
