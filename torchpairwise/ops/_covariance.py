"""Native SPD covariance geometry adapters."""
import torch
from ._gaussian import _validate_inputs


def _call(a, b, mode, metric, squared):
    if type(squared) is not bool:
        raise TypeError('squared must be a boolean')
    _validate_inputs((a, b), 'Covariance')
    try:
        return torch.ops.torchpairwise.covariance_comparisons(
            a, None if mode == 'condensed' else b, metric,
            'all_pairs' if mode == 'allpairs' else mode, squared)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def log_euclidean_distances(x1, x2=None, *, squared=False):
    """Log-Euclidean SPD covariance distances ``||log(A)-log(B)||_F``.

    Inputs have shape ``[...,N,D,D]`` and ``[...,M,D,D]``; only leading
    batch axes broadcast. Returns ``[...,N,M]``. Omit x2 for self distances.
    ``squared=True`` returns the square of the distance.

    Covariances must be finite, symmetric, and strictly positive definite.
    Symmetry tolerates relative roundoff ``64*D*finfo(dtype).eps``; accepted
    inputs are symmetrized. No eigenvalue clipping or jitter is applied.
    Float32/float64 are supported on CPU/CUDA. Autocast promotes float16 and
    bfloat16 inputs to float32. Stable first derivatives include repeated
    positive eigenvalues; higher derivatives are unsupported. At identical
    covariances the distance and chosen subgradient are exactly zero.

    Sample/decomposition work is tiled. Autograd retains intermediates for
    all tiles; training memory can still scale as ``N*M*D*D`` per batch.
    """
    return _call(x1, x1 if x2 is None else x2, 'allpairs', 'log_euclidean', squared)


def paired_log_euclidean_distances(x1, x2, *, squared=False):
    """Aligned Log-Euclidean distances ``[...,N]``; see :func:`log_euclidean_distances`."""
    return _call(x1, x2, 'paired', 'log_euclidean', squared)


def condensed_log_euclidean_distances(x, *, squared=False):
    """Log-Euclidean self distances in row-major upper-triangle order.

    Returns ``[...,N*(N-1)//2]``. See :func:`log_euclidean_distances`.
    """
    return _call(x, x, 'condensed', 'log_euclidean', squared)


def affine_invariant_distances(x1, x2=None, *, squared=False):
    """Affine-invariant Riemannian SPD distance ``||log(A^-1/2 B A^-1/2)||_F``.

    Returns ``[...,N,M]``. ``squared=True`` returns its square. Invariant
    under simultaneous invertible congruence ``A -> P A P.T``. Computation
    uses Cholesky whitening and stable scalar spectral derivatives. Near
    equality it diagonalizes the whitened difference and uses log1p.
    See :func:`log_euclidean_distances` for shape, domain, dtype, gradient,
    and memory policies. Very ill-conditioned inputs may lose numerical
    positive definiteness during whitening and raise ValueError in eager mode.
    """
    return _call(x1, x1 if x2 is None else x2, 'allpairs', 'affine_invariant', squared)


def paired_affine_invariant_distances(x1, x2, *, squared=False):
    """Aligned AIRM distances ``[...,N]``; see :func:`affine_invariant_distances`."""
    return _call(x1, x2, 'paired', 'affine_invariant', squared)


def condensed_affine_invariant_distances(x, *, squared=False):
    """AIRM self distances in row-major upper-triangle order ``[...,N*(N-1)//2]``."""
    return _call(x, x, 'condensed', 'affine_invariant', squared)


def stein_distances(x1, x2=None, *, squared=False):
    """Rooted Jensen-Bregman LogDet (Stein) SPD covariance distances.

    With ``S(A,B) = logdet((A+B)/2) - (logdet(A)+logdet(B))/2``, returns
    ``sqrt(S)`` by default and the divergence ``S`` when ``squared=True``.
    No extra factor of two is used. Output shape is ``[...,N,M]``.
    Invariant under simultaneous invertible congruence. Uses a cancellation-
    resistant log-cosh formula of generalized log eigenvalues.
    See :func:`affine_invariant_distances` and :func:`log_euclidean_distances`
    for validation, gradient, dtype, shape, and memory policies.
    """
    return _call(x1, x1 if x2 is None else x2, 'allpairs', 'stein', squared)


def paired_stein_distances(x1, x2, *, squared=False):
    """Aligned rooted Stein distances ``[...,N]``; see :func:`stein_distances`."""
    return _call(x1, x2, 'paired', 'stein', squared)


def condensed_stein_distances(x, *, squared=False):
    """Rooted Stein self distances in row-major upper-triangle order ``[...,N*(N-1)//2]``."""
    return _call(x, x, 'condensed', 'stein', squared)
