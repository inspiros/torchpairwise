"""Validation adapters for native finite-direction sliced Wasserstein."""
import numbers
import torch
from ._batching import prepare_pointsets, validate_values


def _validate(x, y, projections, p):
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or p not in (1, 2):
        raise ValueError('Sliced Wasserstein p must be 1 or 2')
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError('Sliced Wasserstein requires float32 or float64 inputs outside autocast')
    if (not isinstance(projections, torch.Tensor) or projections.ndim != 2
            or projections.shape[0] != x.shape[-1] or projections.shape[1] == 0):
        raise ValueError('projections must have shape [C,L] with at least one direction')
    if projections.layout != torch.strided:
        raise TypeError('projections must be a dense strided tensor')
    if projections.dtype != x.dtype or projections.device != x.device:
        raise ValueError('projections must have the input dtype and device')
    norms = torch.linalg.vector_norm(projections, dim=0)
    tolerance = 1e-5 if x.dtype == torch.float32 else 1e-10
    validate_values(torch.isfinite(projections).all()
                    & torch.isclose(norms, torch.ones_like(norms),
                                    rtol=tolerance, atol=tolerance).all(),
                    'projections must contain finite unit direction columns')


def _prepared_sliced(x, y, *, projections, p=2, x1_lengths=None,
                     x2_lengths=None, chunk_size=None, paired=False):
    _validate(x, y, projections, p)
    for value, lengths in ((x, x1_lengths), (y, x2_lengths)):
        valid = torch.ones(value.shape[:-1], device=value.device, dtype=torch.bool)
        if lengths is not None:
            valid = torch.arange(value.shape[-2], device=value.device) < lengths.unsqueeze(-1)
        validate_values((torch.isfinite(value) | ~valid.unsqueeze(-1)).all(),
                        'Sliced Wasserstein valid point coordinates must be finite')
    return torch.ops.torchpairwise.transport_comparisons(
        x, y, 'sliced_wasserstein', 'paired' if paired else 'all_pairs', p=p,
        projections=projections, x1_lengths=x1_lengths, x2_lengths=x2_lengths, chunk_size=chunk_size)


def evaluate_sliced(spec, x, y, kwargs):
    return _prepared_sliced(x, x if y is None else y, **kwargs)


def paired_sliced(spec, x1, x2, kwargs):
    x, y, _, kwargs = prepare_pointsets(spec, x1, x2, kwargs, paired=True)
    return _prepared_sliced(x, y, paired=True, **kwargs)


def condensed_sliced(spec, x, kwargs):
    if kwargs.get('x2_lengths') is not None:
        raise ValueError('Condensed point sets accept x1_lengths only')
    x, _, _, kwargs = prepare_pointsets(spec, x, None, kwargs)
    _validate(x, x, kwargs['projections'], kwargs.get('p', 2))
    kwargs = dict(kwargs)
    kwargs.pop('x2_lengths', None)
    return torch.ops.torchpairwise.transport_comparisons(x, None, 'sliced_wasserstein', 'condensed', **kwargs)
