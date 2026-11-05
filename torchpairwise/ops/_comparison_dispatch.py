"""Thin public selectors over the native comparison catalog and family kernels."""
from typing import NamedTuple, Optional, Tuple, Union
import torch
from ._registry import DISTANCES, KERNELS
from ._validation import _check_chunk_size

Collection = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

class ComparisonInfo(NamedTuple):
    name: str
    family: str
    inputs: str
    modes: tuple
    parameters: tuple
    symmetric: bool
    differentiable: bool
    aliases: tuple
    role: str
    sample_ndims: tuple

# Native metadata is authoritative for public selector scope in both languages.
_CATALOG = {row[0]: ComparisonInfo(*(tuple(value) if isinstance(value, list) else value for value in row))
            for row in torch.ops.torchpairwise.comparison_catalog()}
_ROLE_LOOKUP = {}
for _role in ('any', 'distance', 'kernel'):
    _entries = [entry for entry in _CATALOG.values() if _role == 'any' or entry.role == _role]
    _lookup = {entry.name: entry.name for entry in _entries}
    for _entry in _entries:
        for _alias in _entry.aliases:
            _lookup.setdefault(_alias, _entry.name)
    _ROLE_LOOKUP[_role] = _lookup
_LOOKUP = _ROLE_LOOKUP['any']
_VECTOR_SPECS = {spec.op: (spec, registry is KERNELS)
                 for registry in (DISTANCES, KERNELS) for spec in registry.values()}


def list_comparisons(*, family=None, role=None):
    """Canonical metadata: names, roles, layouts, four modes and sample ranks.

    role is distance (including divergences/costs), kernel, score or statistic.
    Single mode consumes one sample per leading batch, without an N/M axis.
    """
    if family is not None and family not in {info.family for info in _CATALOG.values()}:
        raise ValueError(f'Unknown comparison family {family!r}')
    if role is not None and role not in ('distance', 'kernel', 'score', 'statistic'):
        raise ValueError(f'Unknown comparison role {role!r}')
    return tuple(info for name, info in sorted(_CATALOG.items())
                 if (family is None or info.family == family) and (role is None or info.role == role))


def _structured(value, count, name):
    if not isinstance(value, tuple) or len(value) != count or any(not isinstance(t, torch.Tensor) for t in value):
        labels = '(means, covariances)' if count == 2 else '(means, covariances, weights)'
        raise TypeError(f'{name} must be a tuple of tensors {labels}')
    return value



def _select(x1, x2, metric, mode, role='any', selector='compare', **kwargs):
    lookup = _ROLE_LOOKUP[role]
    if not isinstance(metric, str) or metric not in lookup:
        raise ValueError(f'Unknown comparison {metric!r} for role {role}; use list_comparisons()')
    info = _CATALOG[lookup[metric]]
    if mode not in ('all_pairs', 'paired', 'condensed', 'single'):
        raise ValueError('mode must be all_pairs, paired, condensed or single')
    if mode not in info.modes:
        raise ValueError(f'{info.name} does not support {mode} mode; condensed requires a supported symmetric comparison')
    if mode in ('paired', 'single') and x2 is None:
        raise ValueError('Paired/single comparisons require x2')
    if mode == 'condensed' and x2 is not None:
        raise ValueError('Condensed comparisons accept x1 only')
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if mode == 'condensed':
        if 'x2_lengths' in kwargs:
            raise TypeError('Condensed comparisons do not accept x2_lengths')
        if info.family in ('sequence', 'tokens') and 'x_lengths' in kwargs:
            if 'x1_lengths' in kwargs:
                raise TypeError('Supply x1_lengths or x_lengths, not both')
            kwargs['x1_lengths'] = kwargs.pop('x_lengths')
    unused = set(kwargs) - set(info.parameters)
    if unused:
        raise TypeError(f'{info.name} does not accept parameters: {", ".join(sorted(unused))}')
    for key in ('chunk_size', 'component_chunk_size'):
        if key in kwargs:
            _check_chunk_size(kwargs[key], key)
    if 'resolutions' in kwargs and kwargs['resolutions'] and isinstance(kwargs['resolutions'][0], (tuple, list)):
        # Named audio APIs accept triples; native APIs use flattened triples.
        triples = kwargs['resolutions']
        if any(len(triple) != 3 for triple in triples):
            raise ValueError('Each resolution must contain n_fft, hop_length, win_length')
        kwargs['resolutions'] = [v for n, hop, win in triples for v in (n, max(1, n//4) if hop is None else hop, n if win is None else win)]
    structured = info.family in ('gaussian', 'mixture')
    if structured:
        count = 2 if info.family == 'gaussian' else 3
        a = _structured(x1, count, 'x1')
        b = None if x2 is None else _structured(x2, count, 'x2')
    else:
        if not isinstance(x1, torch.Tensor) or (x2 is not None and not isinstance(x2, torch.Tensor)):
            raise TypeError(f'{info.family} collections must be tensors')
        a, b = (x1,), None if x2 is None else (x2,)
        spec = _VECTOR_SPECS.get(info.name, (None,))[0]
        # Share the named adapters' argument preflight (types, layouts and
        # required options). Under autocast they validate cast copies, so the
        # native checks alone apply there.
        if (spec is not None and mode != 'single'
                and not (x1.device.type in ('cpu', 'cuda') and torch.is_autocast_enabled(x1.device.type))):
            from ._batching import validate_comparison
            validate_comparison(spec, x1, x2, kwargs, mode='paired' if mode == 'paired' else 'all_pairs')
    try:
        if selector == 'cdist':
            return (torch.ops.torchpairwise.cdist.collections(a, b, metric, **kwargs) if structured else
                    torch.ops.torchpairwise.cdist.default(x1, x2, metric, **kwargs))
        if selector == 'pdist':
            return (torch.ops.torchpairwise.pdist.collections(a, metric, **kwargs) if structured else
                    torch.ops.torchpairwise.pdist.default(x1, metric, **kwargs))
        return torch.ops.torchpairwise.compare(a, b, metric, mode, role, **kwargs)
    except RuntimeError as exc:
        # Match the named Python adapters; numerical validation remains native.
        raise ValueError(str(exc)) from exc


def compare(x1: Collection, x2: Optional[Collection] = None, metric: str = 'euclidean',
            *, mode: str = 'all_pairs', **kwargs):
    """Compare collections, or single samples with mode='single'.

    Collection layouts/modes are described by list_comparisons(). Single inputs
    omit the sample-count axis and broadcast leading batch axes. All generic
    modes return a Tensor; use named directed Hausdorff for witness indices.
    """
    return _select(x1, x2, metric, mode, **kwargs)


def single_comparison(x1: Collection, x2: Collection, metric: str = 'euclidean', **kwargs):
    """One comparison per leading batch; canonical sample dimensions are retained."""
    return compare(x1, x2, metric, mode='single', **kwargs)


__all__ = ['ComparisonInfo', 'compare', 'single_comparison', 'list_comparisons']
