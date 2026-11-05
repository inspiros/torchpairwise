"""Thin adapters for native weighted Gaussian-mixture comparisons."""
import numbers
import torch
from ._validation import _check_chunk_size
from ._gaussian import _validate_inputs
from ._batching import is_finite_real


def _positive_integer(value, name):
    if type(value) is not int:
        raise TypeError(f'{name} must be an integer')
    if value < 1:
        raise ValueError(f'{name} must be positive')


def _call(means, covariances, weights, other_means, other_covariances, other_weights, *,
          family, mode, x1_lengths=None, x2_lengths=None, chunk_size=8,
          component_chunk_size=None, squared=None, regularization=None, max_iter=None):
    if squared is not None and type(squared) is not bool:
        raise TypeError('squared must be a boolean')
    for name, size in (('chunk_size', chunk_size), ('component_chunk_size', component_chunk_size)):
        if name == 'component_chunk_size' and size is None:
            continue
        if type(size) is not int:
            raise TypeError(f'{name} must be an integer')
        _check_chunk_size(size, name)
    if max_iter is not None:
        _positive_integer(max_iter, 'max_iter')
    supplied = (other_means is not None, other_covariances is not None, other_weights is not None)
    if any(supplied) and not all(supplied):
        raise ValueError('Supply all three second mixture tensors, or omit all three')
    tensors = (means, covariances, weights)
    if all(supplied):
        tensors += (other_means, other_covariances, other_weights)
    _validate_inputs(tensors, 'Mixture')
    if regularization is not None and (isinstance(regularization, bool)
            or not isinstance(regularization, numbers.Real)
            or not is_finite_real(regularization) or regularization <= 0):
        raise ValueError('regularization must be a positive finite real scalar')
    for length in (x1_lengths, x2_lengths):
        if length is not None and (not isinstance(length, torch.Tensor) or length.dtype != torch.int64):
            raise TypeError('Mixture lengths must be int64 tensors')
    try:
        return torch.ops.torchpairwise.mixture_comparisons(
            means, covariances, weights, other_means, other_covariances, other_weights,
            family, 'all_pairs' if mode == 'allpairs' else mode, x1_lengths, x2_lengths,
            squared, regularization, max_iter, chunk_size, component_chunk_size)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def gaussian_mixture_overlap_kernel(x1_mean, x1_covariance, x1_weights, x2_mean=None,
                                    x2_covariance=None, x2_weights=None, *, x1_lengths=None,
                                    x2_lengths=None, chunk_size=8, component_chunk_size=32):
    """Exact density overlap integral, returning ``[...,N,M]``.

    Inputs are means ``[...,N,K,D]``, SPD covariances ``[...,N,K,D,D]``
    and explicit nonnegative weights ``[...,N,K]``. Weights are normalized per
    sample; only leading batch axes broadcast. Optional int64 lengths
    ``[...,N]`` select valid component prefixes. Padded values are ignored.
    All valid covariances, including zero-weight ones, must be SPD.

    CPU/CUDA float32/float64; autocast promotes half/bfloat16 to float32.
    ``chunk_size`` bounds mixture-sample tiles; ``component_chunk_size`` bounds
    density component tiles. Gradients follow the analytic torch composition;
    zero weights use a finite continuation, not a boundary derivative guarantee.
    Outputs can underflow to zero; CS divergence uses log overlaps directly.
    See ``docs/gaussian-mixtures.md`` for transport and numerical contracts.
    """
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='overlap', mode='allpairs', x1_lengths=x1_lengths, x2_lengths=x2_lengths,
                 chunk_size=chunk_size, component_chunk_size=component_chunk_size)


def paired_gaussian_mixture_overlap_kernel(x1_mean, x1_covariance, x1_weights, x2_mean,
                                           x2_covariance, x2_weights, *, x1_lengths=None,
                                           x2_lengths=None, chunk_size=8, component_chunk_size=32):
    """Aligned density overlaps ``[...,N]``; see :func:`gaussian_mixture_overlap_kernel`."""
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='overlap', mode='paired', x1_lengths=x1_lengths, x2_lengths=x2_lengths,
                 chunk_size=chunk_size, component_chunk_size=component_chunk_size)


def condensed_gaussian_mixture_overlap_kernel(means, covariances, weights, *, x1_lengths=None,
                                              chunk_size=8, component_chunk_size=32):
    """Self overlaps in row-major upper-triangle order; see density overlap contract."""
    return _call(means, covariances, weights, None, None, None, family='overlap', mode='condensed',
                 x1_lengths=x1_lengths, chunk_size=chunk_size, component_chunk_size=component_chunk_size)


def gaussian_mixture_l2_distances(x1_mean, x1_covariance, x1_weights, x2_mean=None,
                                 x2_covariance=None, x2_weights=None, *, squared=False,
                                 x1_lengths=None, x2_lengths=None, chunk_size=8, component_chunk_size=32):
    """L2 distances between normalized densities; ``squared`` returns the squared norm.

    Uses analytic overlap sums. Negative cancellation residuals are clamped to
    zero; unsquared values use zero derivative at zero. See density input contract.
    """
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='l2', mode='allpairs', squared=squared, x1_lengths=x1_lengths,
                 x2_lengths=x2_lengths, chunk_size=chunk_size, component_chunk_size=component_chunk_size)


def paired_gaussian_mixture_l2_distances(x1_mean, x1_covariance, x1_weights, x2_mean,
                                        x2_covariance, x2_weights, *, squared=False,
                                        x1_lengths=None, x2_lengths=None, chunk_size=8, component_chunk_size=32):
    """Aligned density L2 distances; see :func:`gaussian_mixture_l2_distances`."""
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='l2', mode='paired', squared=squared, x1_lengths=x1_lengths,
                 x2_lengths=x2_lengths, chunk_size=chunk_size, component_chunk_size=component_chunk_size)


def condensed_gaussian_mixture_l2_distances(means, covariances, weights, *, squared=False,
                                           x1_lengths=None, chunk_size=8, component_chunk_size=32):
    """Self density L2 distances in row-major upper-triangle order."""
    return _call(means, covariances, weights, None, None, None, family='l2', mode='condensed',
                 squared=squared, x1_lengths=x1_lengths, chunk_size=chunk_size,
                 component_chunk_size=component_chunk_size)


def gaussian_mixture_cauchy_schwarz_divergences(x1_mean, x1_covariance, x1_weights, x2_mean=None,
                                              x2_covariance=None, x2_weights=None, *, x1_lengths=None,
                                              x2_lengths=None, chunk_size=8, component_chunk_size=32):
    """Natural-log CS divergence ``-log(K(P,Q)/sqrt(K(P,P)*K(Q,Q)))``.

    Uses stable log overlaps, with roundoff-sized negative results clamped to
    zero. This convention has no factor of two. See density input contract.
    """
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='cs', mode='allpairs', x1_lengths=x1_lengths, x2_lengths=x2_lengths,
                 chunk_size=chunk_size, component_chunk_size=component_chunk_size)


def paired_gaussian_mixture_cauchy_schwarz_divergences(x1_mean, x1_covariance, x1_weights, x2_mean,
                                                     x2_covariance, x2_weights, *, x1_lengths=None,
                                                     x2_lengths=None, chunk_size=8, component_chunk_size=32):
    """Aligned CS divergences; see :func:`gaussian_mixture_cauchy_schwarz_divergences`."""
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='cs', mode='paired', x1_lengths=x1_lengths, x2_lengths=x2_lengths,
                 chunk_size=chunk_size, component_chunk_size=component_chunk_size)


def condensed_gaussian_mixture_cauchy_schwarz_divergences(means, covariances, weights, *, x1_lengths=None,
                                                        chunk_size=8, component_chunk_size=32):
    """Self CS divergences in row-major upper-triangle order."""
    return _call(means, covariances, weights, None, None, None, family='cs', mode='condensed',
                 x1_lengths=x1_lengths, chunk_size=chunk_size, component_chunk_size=component_chunk_size)


def gaussian_mixture_wasserstein_distances(x1_mean, x1_covariance, x1_weights, x2_mean=None,
                                          x2_covariance=None, x2_weights=None, *, squared=False,
                                          x1_lengths=None, x2_lengths=None, chunk_size=8):
    """Component-restricted MW2 using exact weighted discrete OT of Gaussian W2².

    Returns the square root unless ``squared=True``. Covariances may be PSD.
    This generally differs from W2 between full mixture densities. At most 4096
    positive-mass component pairs per comparison; larger supports should use
    entropic transport. The exact CPU-controlled solver synchronizes CUDA and
    is available as an opaque native operator to compile/export/FakeTensor. ``chunk_size`` is accepted for
    consistency; the solver evaluates one mixture pair at a time.

    Selected-plan and dual-potential envelope derivatives support coordinates,
    SPD covariances and positive weights. Ties select a subgradient. Singular
    covariances and zero weights have finite continuations without a general
    derivative guarantee. Higher derivatives are unsupported.
    """
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='wasserstein', mode='allpairs', squared=squared, x1_lengths=x1_lengths,
                 x2_lengths=x2_lengths, chunk_size=chunk_size)


def paired_gaussian_mixture_wasserstein_distances(x1_mean, x1_covariance, x1_weights, x2_mean,
                                                 x2_covariance, x2_weights, *, squared=False,
                                                 x1_lengths=None, x2_lengths=None, chunk_size=8):
    """Aligned MW2 distances; see :func:`gaussian_mixture_wasserstein_distances`."""
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='wasserstein', mode='paired', squared=squared, x1_lengths=x1_lengths,
                 x2_lengths=x2_lengths, chunk_size=chunk_size)


def condensed_gaussian_mixture_wasserstein_distances(means, covariances, weights, *, squared=False,
                                                    x1_lengths=None, chunk_size=8):
    """Self MW2 distances in row-major upper-triangle order."""
    return _call(means, covariances, weights, None, None, None, family='wasserstein', mode='condensed',
                 squared=squared, x1_lengths=x1_lengths, chunk_size=chunk_size)


def gaussian_mixture_entropic_transport_costs(x1_mean, x1_covariance, x1_weights, x2_mean=None,
                                             x2_covariance=None, x2_weights=None, *, regularization=.1,
                                             max_iter=100, x1_lengths=None, x2_lengths=None, chunk_size=8):
    """Estimate min <pi,C> + epsilon*KL(pi || a tensor-product b), C=Gaussian W2².

    Reports the full regularized objective, without a square root or debiasing.
    Uses exactly ``max_iter`` symmetric half-relaxed log-scaling updates and a
    dual-objective estimate. Finite budgets need not converge. Autograd follows
    all updates; no envelope gradient is substituted. Covariances may be PSD.
    Ground storage is O(chunk_size²*K*L); component chunking is not supported
    because Sinkhorn needs the complete ground matrix for each sample tile.
    """
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='entropic', mode='allpairs', regularization=regularization, max_iter=max_iter,
                 x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def paired_gaussian_mixture_entropic_transport_costs(x1_mean, x1_covariance, x1_weights, x2_mean,
                                                    x2_covariance, x2_weights, *, regularization=.1,
                                                    max_iter=100, x1_lengths=None, x2_lengths=None, chunk_size=8):
    """Aligned full entropic costs; see :func:`gaussian_mixture_entropic_transport_costs`."""
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='entropic', mode='paired', regularization=regularization, max_iter=max_iter,
                 x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def condensed_gaussian_mixture_entropic_transport_costs(means, covariances, weights, *, regularization=.1,
                                                       max_iter=100, x1_lengths=None, chunk_size=8):
    """Self entropic costs in row-major upper-triangle order."""
    return _call(means, covariances, weights, None, None, None, family='entropic', mode='condensed',
                 regularization=regularization, max_iter=max_iter, x1_lengths=x1_lengths, chunk_size=chunk_size)


def gaussian_mixture_sinkhorn_divergences(x1_mean, x1_covariance, x1_weights, x2_mean=None,
                                         x2_covariance=None, x2_weights=None, *, regularization=.1,
                                         max_iter=100, x1_lengths=None, x2_lengths=None, chunk_size=8):
    """Debiased entropic costs OT(P,Q)-OT(P,P)/2-OT(Q,Q)/2.

    Uses Gaussian W2² component ground costs, no root and no nonnegative clamp.
    Finite-iteration dual estimates may be negative; general positivity for this
    Gaussian ground cost is not promised. See entropic transport cost contract.
    """
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='sinkhorn', mode='allpairs', regularization=regularization, max_iter=max_iter,
                 x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def paired_gaussian_mixture_sinkhorn_divergences(x1_mean, x1_covariance, x1_weights, x2_mean,
                                                x2_covariance, x2_weights, *, regularization=.1,
                                                max_iter=100, x1_lengths=None, x2_lengths=None, chunk_size=8):
    """Aligned debiased entropic costs; see :func:`gaussian_mixture_sinkhorn_divergences`."""
    return _call(x1_mean, x1_covariance, x1_weights, x2_mean, x2_covariance, x2_weights,
                 family='sinkhorn', mode='paired', regularization=regularization, max_iter=max_iter,
                 x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def condensed_gaussian_mixture_sinkhorn_divergences(means, covariances, weights, *, regularization=.1,
                                                   max_iter=100, x1_lengths=None, chunk_size=8):
    """Self debiased entropic costs in row-major upper-triangle order."""
    return _call(means, covariances, weights, None, None, None, family='sinkhorn', mode='condensed',
                 regularization=regularization, max_iter=max_iter, x1_lengths=x1_lengths, chunk_size=chunk_size)
