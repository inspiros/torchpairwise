"""Validated Python entry points for native sequence dynamic programming."""
import numbers

import torch
from ._validation import _check_chunk_size

from ._batching import validate_values
from ._batching import is_finite_real


def _prepare(x, y, lx, ly, *, mode, chunk_size, window):
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError('Sequence inputs must be tensors')
    if x.layout != torch.strided or y.layout != torch.strided:
        raise TypeError('Sequence inputs must be dense strided tensors')
    if x.ndim < 3 or y.ndim < 3:
        raise ValueError('Sequences must have shape [...,N,T,D]')
    if x.shape[-1] != y.shape[-1]:
        raise ValueError('Sequence feature dimensions must match')
    if min(x.shape[-2:]) == 0 or min(y.shape[-2:]) == 0:
        raise ValueError('Sequence time and feature dimensions must be non-empty')
    if mode == 'paired' and x.shape[-3] != y.shape[-3]:
        raise ValueError('Paired sequence inputs must have equal sample counts')
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError('Sequence inputs must have the same device and dtype')
    if x.device.type not in ('cpu', 'cuda'):
        raise ValueError('Sequence inputs must be on CPU or CUDA')
    allowed = ((torch.float32, torch.float64, torch.float16, torch.bfloat16)
               if torch.is_autocast_enabled(x.device.type) else (torch.float32, torch.float64))
    if x.dtype not in allowed:
        raise TypeError('Sequence inputs require float32 or float64 outside autocast')
    if type(chunk_size) is not int:
        raise TypeError('chunk_size must be an integer')
    _check_chunk_size(chunk_size)
    if window is not None:
        if type(window) is not int:
            raise TypeError('window must be an integer or None')
        if window < 0:
            raise ValueError('window must be non-negative')
    original = x, y, lx, ly
    same = y is x and ly is lx
    shapes = [x.shape[:-3], y.shape[:-3]]
    lengths = []
    for value, length, name in ((x, lx, 'x1_lengths'), (y, ly, 'x2_lengths')):
        if length is None:
            length = torch.full(value.shape[:-2], value.shape[-2],
                                dtype=torch.int64, device=value.device)
        else:
            if not isinstance(length, torch.Tensor) or length.dtype != torch.int64:
                raise TypeError(f'{name} must be an int64 tensor')
            if length.device != value.device:
                raise ValueError(f'{name} must be on the input device')
            if length.ndim < 1 or length.shape[-1] != value.shape[-3]:
                raise ValueError(f'{name} must end with the input sample count')
            if length.ndim - 1 > max(x.ndim, y.ndim) - 3:
                raise ValueError(f'{name} cannot introduce additional batch axes')
            validate_values(((length >= 1) & (length <= value.shape[-2])).all(),
                            f'{name} must be between 1 and the padded time dimension')
        shapes.append(length.shape[:-1])
        lengths.append(length)
    try:
        batch = torch.broadcast_shapes(*shapes)
    except RuntimeError as exc:
        raise ValueError('Sequence and length batch dimensions are not broadcastable') from exc
    lx = lengths[0].expand(*batch, x.shape[-3])
    ly = lx if same else lengths[1].expand(*batch, y.shape[-3])
    x = x.expand(*batch, *x.shape[-3:])
    y = x if same else y.expand(*batch, *y.shape[-3:])
    for value, length in ((x, lx), (y, ly)):
        valid = torch.arange(value.shape[-2], device=value.device) < length[..., None]
        validate_values((torch.isfinite(value) | ~valid[..., None]).all(),
                        'Valid sequence frames must be finite')
    return original


def _call(x, y, lx, ly, *, mode, squared=False, gamma=None, window=None, chunk_size=16):
    if type(squared) is not bool:
        raise TypeError('squared must be a boolean')
    if gamma is not None:
        if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real):
            raise TypeError('gamma must be a positive finite real scalar')
        if not is_finite_real(gamma) or gamma <= 0:
            raise ValueError('gamma must be a positive finite real scalar')
        gamma = float(gamma)
    x, y, lx, ly = _prepare(x, y, lx, ly, mode=mode, chunk_size=chunk_size, window=window)
    computation_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    if gamma is not None and not torch.finfo(computation_dtype).tiny <= gamma <= torch.finfo(computation_dtype).max:
        raise ValueError('gamma must be representable as a positive normal value in the computation dtype')
    return torch.ops.torchpairwise.sequence_comparisons(
        x, None if mode == 'condensed' else y,
        'all_pairs' if mode == 'allpairs' else mode, squared, gamma,
        lx, ly, window, chunk_size)


def dtw_distances(x1, x2=None, *, x1_lengths=None, x2_lengths=None,
                  squared=False, window=None, chunk_size=16):
    """Dynamic time warping with squared Euclidean frame costs, ``[...,N,M]``.

    Inputs are ordered, non-empty samples ``[T,D]`` in collections
    ``[...,N,T,D]``. Only batch axes broadcast; the two padded T axes may differ.
    Optional int64 lengths ``[...,N]`` select non-empty valid prefixes on the
    input device. Padding is ignored, may be non-finite and has zero gradients.
    Omit x2 for self comparisons (x1_lengths also supplies x2_lengths by default).

    Minimize the sum of squared Euclidean frame costs along paths from (0,0)
    to (length1-1,length2-1), with steps (1,0), (0,1), (1,1). Return its square
    root by default, or the unrooted sum with squared=True; no path-length
    normalization. Optional non-negative integer window restricts |i-j|<=window
    in unscaled frame indices. Infeasible pairs return +inf with zero gradients.
    Neither form is a triangle metric or separates repeated-frame sequences.

    CPU/CUDA float32/float64; half/bfloat16 require autocast and produce float32.
    First derivatives only: equally split adjoints between tied predecessors;
    rooted zero costs have zero derivatives. chunk_size bounds sample pairs
    per batch in a tile. Forward uses linear wavefront memory; backward
    recomputes a quadratic table for one tile in C++/ATen.
    See docs/sequence-comparisons.md.
    """
    if x2 is None:
        x2 = x1
        if x2_lengths is None:
            x2_lengths = x1_lengths
    return _call(x1, x2, x1_lengths, x2_lengths, mode='allpairs', squared=squared,
                 window=window, chunk_size=chunk_size)


def paired_dtw_distances(x1, x2, *, x1_lengths=None, x2_lengths=None,
                         squared=False, window=None, chunk_size=16):
    """Aligned DTW ``[...,N]``; see :func:`dtw_distances`."""
    return _call(x1, x2, x1_lengths, x2_lengths, mode='paired', squared=squared,
                 window=window, chunk_size=chunk_size)


def condensed_dtw_distances(x, *, x_lengths=None, squared=False, window=None, chunk_size=16):
    """Self DTW in row-major i<j order ``[...,N*(N-1)//2]``."""
    return _call(x, x, x_lengths, x_lengths, mode='condensed', squared=squared,
                 window=window, chunk_size=chunk_size)


def soft_dtw_divergences(x1, x2=None, *, gamma, x1_lengths=None, x2_lengths=None,
                         window=None, chunk_size=16):
    """Debiased Soft-DTW ``Sγ(x,y) - (Sγ(x,x)+Sγ(y,y))/2``, ``[...,N,M]``.

    Sγ replaces each DTW minimum by -gamma*logsumexp(-predecessors/gamma).
    gamma is an explicit positive finite real scalar in squared-feature units,
    representable as a positive normal value in the computation dtype. Frame
    cost is squared Euclidean. Output is unrooted, unclamped and unnormalized;
    raw Soft-DTW is not returned. The same window applies to all three terms.
    Identical valid sequences return exactly zero with zero whole-pair gradient.
    An infeasible cross alignment returns +inf with zero whole-pair gradients.
    Nonnegativity/separation theorems require conditions on the ground cost and
    alignment family; no blanket guarantee is claimed for this cost/window API.
    This divergence is not a triangle metric. See :func:`dtw_distances` for
    lengths, shape, dtype, chunking and first-derivative contracts.
    """
    if gamma is None:
        raise TypeError('gamma must be a positive finite real scalar')
    if x2 is None:
        x2 = x1
        if x2_lengths is None:
            x2_lengths = x1_lengths
    return _call(x1, x2, x1_lengths, x2_lengths, mode='allpairs', gamma=gamma,
                 window=window, chunk_size=chunk_size)


def paired_soft_dtw_divergences(x1, x2, *, gamma, x1_lengths=None, x2_lengths=None,
                                window=None, chunk_size=16):
    """Aligned debiased Soft-DTW ``[...,N]``; see :func:`soft_dtw_divergences`."""
    if gamma is None:
        raise TypeError('gamma must be a positive finite real scalar')
    return _call(x1, x2, x1_lengths, x2_lengths, mode='paired', gamma=gamma,
                 window=window, chunk_size=chunk_size)


def condensed_soft_dtw_divergences(x, *, gamma, x_lengths=None, window=None, chunk_size=16):
    """Self Soft-DTW divergence in row-major i<j order ``[...,N*(N-1)//2]``."""
    if gamma is None:
        raise TypeError('gamma must be a positive finite real scalar')
    return _call(x, x, x_lengths, x_lengths, mode='condensed', gamma=gamma,
                 window=window, chunk_size=chunk_size)
