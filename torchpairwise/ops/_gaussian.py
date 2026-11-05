"""Native Gaussian Wasserstein adapters; algorithms and derivatives live in C++."""
import torch


def _validate_inputs(tensors, name):
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError(f'{name} inputs must be tensors')
    if any(value.layout != torch.strided for value in tensors):
        raise TypeError(f'{name} inputs must be dense strided tensors')
    first = tensors[0]
    if any(value.device != first.device for value in tensors):
        raise ValueError(f'{name} inputs must have the same device')
    if any(value.dtype != first.dtype for value in tensors):
        raise ValueError(f'{name} inputs must have the same dtype')
    allowed = (torch.float32, torch.float64)
    if torch.is_autocast_enabled(first.device.type):
        allowed += (torch.float16, torch.bfloat16)
    if first.dtype not in allowed:
        raise TypeError(f'{name} inputs require float32 or float64 outside autocast')


def _call(means, covariances, other_means, other_covariances, *, mode, squared):
    if type(squared) is not bool:
        raise TypeError('squared must be a boolean')
    _validate_inputs((means, covariances, other_means, other_covariances), 'Gaussian')
    if mode == 'condensed':
        other_means = other_covariances = None
    try:
        return torch.ops.torchpairwise.gaussian_comparisons(
            means, covariances, other_means, other_covariances,
            'all_pairs' if mode == 'allpairs' else mode, squared)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def gaussian_wasserstein_distances(x1_mean, x1_covariance, x2_mean=None, x2_covariance=None, *, squared=False):
    """Gaussian 2-Wasserstein distances, returning ``[...,N,M]``.

    Means have shape ``[...,N,D]`` and covariances ``[...,N,D,D]``; only
    leading batch axes broadcast. Omit both second inputs for self distances.
    Covariances must be finite, symmetric and PSD. Symmetry and negative
    eigenvalues tolerate relative roundoff of ``64*D*finfo(dtype).eps``;
    accepted covariances are symmetrized and negative eigenvalues clipped.

    Supports float32/float64 on CPU/CUDA and promotes float16/bfloat16 to
    float32 under autocast. First derivatives are stable on positive definite
    covariances, including repeated eigenvalues. At singular PSD boundaries
    no general finite derivative exists: the square-root Sylvester derivative
    is set to zero on zero/zero eigenspaces, and an SVD-selected Procrustes
    minimizer defines a finite continuation. Higher derivatives are unsupported.
    Unsquared distances use zero derivative at coincident distributions.
    """
    if (x2_mean is None) != (x2_covariance is None):
        raise ValueError('Supply both x2_mean and x2_covariance, or omit both')
    if x2_mean is None:
        x2_mean, x2_covariance = x1_mean, x1_covariance
    return _call(x1_mean, x1_covariance, x2_mean, x2_covariance, mode='allpairs', squared=squared)


def paired_gaussian_wasserstein_distances(x1_mean, x1_covariance, x2_mean, x2_covariance, *, squared=False):
    """Aligned Gaussian distances ``[...,N]``; see :func:`gaussian_wasserstein_distances`."""
    return _call(x1_mean, x1_covariance, x2_mean, x2_covariance, mode='paired', squared=squared)


def condensed_gaussian_wasserstein_distances(means, covariances, *, squared=False):
    """Gaussian self distances in row-major upper-triangle order ``[...,N*(N-1)//2]``.

    See :func:`gaussian_wasserstein_distances` for validation and gradient policy.
    """
    return _call(means, covariances, means, covariances, mode='condensed', squared=squared)
