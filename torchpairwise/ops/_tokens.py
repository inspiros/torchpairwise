"""Integer-token sequence comparisons with bounded pairwise DP workspaces."""
import torch
from ._validation import _check_chunk_size

from ._batching import validate_values


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _lengths(value, x, name):
    if value is None:
        return torch.full(x.shape[:-1], x.shape[-1], dtype=torch.int64, device=x.device)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int64:
        raise TypeError(f'{name} must be an int64 tensor')
    if value.layout != torch.strided or value.device != x.device:
        raise ValueError(f'{name} must be dense and on the input device')
    try:
        value = value.expand(x.shape[:-1])
    except RuntimeError as exc:
        raise ValueError(f'{name} must broadcast to the input sample counts') from exc
    validate_values(((value >= 0) & (value <= x.shape[-1])).all(),
                    f'{name} must lie between zero and the padded sequence length')
    return value


def _call(x1, x2, *, mode, substitution_cost, x1_lengths=None,
          x2_lengths=None, chunk_size=256):
    if mode == 'paired' and x2 is None:
        raise TypeError('Paired comparisons require both input tensors')
    if x2 is None and x2_lengths is None:
        x2_lengths = x1_lengths
    x2 = x1 if x2 is None else x2
    if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
        raise TypeError('Token inputs must be tensors')
    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError('Token inputs must have shape [...,N,T]')
    if x1.layout != torch.strided or x2.layout != torch.strided:
        raise TypeError('Token inputs must be dense strided tensors')
    if x1.dtype not in _INTEGER_DTYPES or x2.dtype not in _INTEGER_DTYPES:
        raise TypeError('Token inputs require integer dtypes, excluding bool')
    if x1.dtype != x2.dtype or x1.device != x2.device:
        raise ValueError('Token inputs must have the same dtype and device')
    if x1.device.type not in ('cpu', 'cuda'):
        raise ValueError('Token inputs must be on CPU or CUDA')
    _check_chunk_size(chunk_size)
    n, m = x1.shape[-2], x2.shape[-2]
    if mode == 'paired' and n != m:
        raise ValueError('Paired inputs must have equal sample counts')
    try:
        torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    except RuntimeError as exc:
        raise ValueError('Token batch dimensions are not broadcastable') from exc
    _lengths(x1_lengths, x1, 'x1_lengths')
    _lengths(x2_lengths, x2, 'x2_lengths')
    return torch.ops.torchpairwise.token_comparisons(
        x1, None if mode == 'condensed' else x2,
        'all_pairs' if mode == 'allpairs' else mode, substitution_cost,
        x1_lengths, x2_lengths, chunk_size)


def levenshtein_distances(x1, x2=None, *, x1_lengths=None, x2_lengths=None, chunk_size=256):
    """Unit insertion/deletion/substitution edit distance for integer token rows.

    Inputs [...,N,T] and [...,M,U] produce int64 [...,N,M]. Length tensors
    broadcast to each input's [...,N] sample shape; empty sequences are valid.
    Tokenization is user-owned. No derivatives or normalization are applied.
    """
    return _call(x1, x2, mode='allpairs', substitution_cost=1,
                 x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def paired_levenshtein_distances(x1, x2, *, x1_lengths=None, x2_lengths=None, chunk_size=256):
    """Levenshtein distances between corresponding token sequences."""
    return _call(x1, x2, mode='paired', substitution_cost=1,
                 x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def condensed_levenshtein_distances(x, *, x_lengths=None, chunk_size=256):
    """Upper-triangle Levenshtein distances in row-major pdist order."""
    return _call(x, x, mode='condensed', substitution_cost=1,
                 x1_lengths=x_lengths, x2_lengths=x_lengths, chunk_size=chunk_size)


def indel_distances(x1, x2=None, *, x1_lengths=None, x2_lengths=None, chunk_size=256):
    """Insertion/deletion distance = len(x)+len(y)-2*LCS_length(x,y)."""
    return _call(x1, x2, mode='allpairs', substitution_cost=2,
                 x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def paired_indel_distances(x1, x2, *, x1_lengths=None, x2_lengths=None, chunk_size=256):
    """Insertion/deletion distances between corresponding token sequences."""
    return _call(x1, x2, mode='paired', substitution_cost=2,
                 x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def condensed_indel_distances(x, *, x_lengths=None, chunk_size=256):
    """Upper-triangle insertion/deletion distances in row-major pdist order."""
    return _call(x, x, mode='condensed', substitution_cost=2,
                 x1_lengths=x_lengths, x2_lengths=x_lengths, chunk_size=chunk_size)


__all__ = ['levenshtein_distances', 'paired_levenshtein_distances',
           'condensed_levenshtein_distances', 'indel_distances',
           'paired_indel_distances', 'condensed_indel_distances']
