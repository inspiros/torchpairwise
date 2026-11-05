"""Within-call representation reuse for user-owned encoders and comparisons."""
import math

import torch
from ._validation import _check_chunk_size


def _collection(x, sample_ndim, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'{name} must be a Tensor')
    if x.layout != torch.strided:
        raise TypeError(f'{name} must be a dense strided Tensor')
    if x.ndim < sample_ndim + 1:
        raise ValueError(f'{name} must have a sample-count axis and sample_ndim sample axes')
    prefix = x.shape[:-sample_ndim]
    count = math.prod(prefix)
    if count == 0:
        raise ValueError(f'{name} contains no samples; empty feature collections are unsupported')
    return prefix, count


def _validate_features(tree, count, path='features'):
    if isinstance(tree, torch.Tensor):
        if tree.layout != torch.strided:
            raise TypeError(f'{path} must be a dense strided Tensor')
        if tree.ndim < 1 or tree.shape[0] != count:
            raise ValueError(f'{path} must have leading encoded sample count {count}')
        return 1
    if type(tree) in (tuple, list):
        leaves = sum(_validate_features(value, count, f'{path}[{i}]') for i, value in enumerate(tree))
    elif type(tree) is dict:
        leaves = sum(_validate_features(value, count, f'{path}[{key!r}]') for key, value in tree.items())
    else:
        raise TypeError(f'{path} must be a Tensor or an ordinary tuple/list/dict of tensor features')
    if leaves == 0:
        raise ValueError(f'{path} must contain at least one tensor feature leaf')
    return leaves


def _merge(parts, path='features'):
    first = parts[0]
    if isinstance(first, torch.Tensor):
        for other in parts[1:]:
            if not isinstance(other, torch.Tensor):
                raise TypeError(f'{path} structure must be consistent across encoder chunks')
            if first.shape[1:] != other.shape[1:]:
                raise ValueError(f'{path} feature shape must be consistent across encoder chunks')
            if first.dtype != other.dtype or first.device != other.device:
                raise ValueError(f'{path} dtype and device must be consistent across encoder chunks')
        return torch.cat(parts, dim=0)
    if any(type(part) is not type(first) for part in parts[1:]):
        raise TypeError(f'{path} structure must be consistent across encoder chunks')
    if type(first) is dict:
        if any(part.keys() != first.keys() for part in parts[1:]):
            raise ValueError(f'{path} dictionary keys must be consistent across encoder chunks')
        return {key: _merge([part[key] for part in parts], f'{path}[{key!r}]') for key in first}
    if any(len(part) != len(first) for part in parts[1:]):
        raise ValueError(f'{path} structure length must be consistent across encoder chunks')
    values = [_merge([part[i] for part in parts], f'{path}[{i}]') for i in range(len(first))]
    return tuple(values) if type(first) is tuple else values


def _restore(tree, prefix):
    if isinstance(tree, torch.Tensor):
        return tree.reshape(*prefix, *tree.shape[1:])
    if type(tree) is dict:
        return {key: _restore(value, prefix) for key, value in tree.items()}
    values = [_restore(value, prefix) for value in tree]
    return tuple(values) if type(tree) is tuple else values


class PreparedFeatureComparison:
    """Explicitly prepared features; no encoder is run by this binding's consumers."""
    def __init__(self, owner, features1, features2, mode=None, single=False):
        self._owner = owner
        self.features1, self.features2, self.mode = features1, features2, mode
        self._single = single

    def __call__(self, *args, **kwargs):
        selected = kwargs.pop('mode', None)
        selected = self.mode if selected is None else selected
        self._owner._check_mode(selected)
        if selected is not None:
            kwargs['mode'] = selected
        return self._owner._compare(self.features1, self.features2, *args, **kwargs)

    def _operator(self, **kwargs):
        """Bind a supported streaming head, preserving the prepared feature graph."""
        if self._single:
            raise ValueError('streaming operators require prepared collections, not single-mode samples')
        return self._owner._bind_operator(self.features1, self.features2, self.mode, **kwargs)


class FeatureComparison:
    """Caller-owned encoder/head with explicit encode, prepare and lazy binding paths."""
    def __init__(self, encoder, metric, head, mode, sample_ndim, encode_chunk_size, metric_kwargs):
        self._two_towers = isinstance(encoder, tuple)
        self.query_encoder, self.reference_encoder = encoder if self._two_towers else (encoder, encoder)
        self.metric, self.head, self.metric_kwargs = metric, head, metric_kwargs
        self.mode = ('all_pairs' if metric is not None else getattr(head, 'mode', None)) if mode is None else mode
        self.sample_ndim, self.encode_chunk_size = sample_ndim, encode_chunk_size
        self._check_mode(self.mode)

    @property
    def encoder(self):
        return (self.query_encoder, self.reference_encoder) if self._two_towers else self.query_encoder

    def _check_mode(self, mode):
        if self._two_towers and mode == 'condensed':
            raise ValueError('condensed mode requires symmetric features; use one encoder or all_pairs with two towers')

    def _encode(self, x, encoder, *, single=False):
        if single:
            if not isinstance(x, torch.Tensor) or x.ndim < self.sample_ndim:
                raise ValueError('single inputs must have sample_ndim sample axes')
            x = x.unsqueeze(-self.sample_ndim - 1)
        prefix, count = _collection(x, self.sample_ndim, 'input')
        flat = x.reshape(count, *x.shape[-self.sample_ndim:])
        size = self.encode_chunk_size
        if size is None or count <= size:
            result = encoder(flat)
            _validate_features(result, count)
        else:
            parts = []
            for start in range(0, count, size):
                stop = min(start + size, count)
                part = encoder(flat[start:stop])
                _validate_features(part, stop - start)
                parts.append(part)
            result = _merge(parts)
        restored = _restore(result, prefix)
        if not single:
            return restored
        def remove_collection(tree):
            if isinstance(tree, torch.Tensor):
                return tree.squeeze(len(prefix) - 1)
            if type(tree) is dict:
                return {key: remove_collection(value) for key, value in tree.items()}
            values = [remove_collection(value) for value in tree]
            return tuple(values) if type(tree) is tuple else values
        return remove_collection(restored)

    def encode(self, x, *, single=False, tower=None):
        """Extract reusable features, selecting query/reference for two towers."""
        if type(single) is not bool:
            raise TypeError('single must be a boolean')
        if self._two_towers:
            if tower not in ('query', 'reference'):
                raise ValueError("two encoders require tower='query' or tower='reference'")
            encoder = self.query_encoder if tower == 'query' else self.reference_encoder
        else:
            if tower is not None:
                raise ValueError('tower must be None with a single encoder')
            encoder = self.query_encoder
        return self._encode(x, encoder, single=single)

    def _compare(self, features1, features2, *args, **kwargs):
        if self.metric is not None:
            from ._comparison_dispatch import compare
            return compare(features1, features2, *args, metric=self.metric, **{**self.metric_kwargs, **kwargs})
        if features2 is None:
            return self.head(features1, *args, **kwargs)
        return self.head(features1, features2, *args, **kwargs)

    def _prepare(self, x1, x2, single, mode=None):
        self._check_mode(mode)
        # Validate both original layouts before running possibly stateful encoders.
        for name, value in (('x1', x1), ('x2', x2)):
            if value is not None:
                if single:
                    if not isinstance(value, torch.Tensor) or value.ndim < self.sample_ndim:
                        raise ValueError('single inputs must have sample_ndim sample axes')
                    value = value.unsqueeze(-self.sample_ndim - 1)
                _collection(value, self.sample_ndim, name)
        a = self._encode(x1, self.query_encoder, single=single)
        if self._two_towers:
            b = self._encode(x1 if x2 is None else x2, self.reference_encoder, single=single)
        else:
            b = None if x2 is None else a if x2 is x1 else self._encode(x2, self.query_encoder, single=single)
        return PreparedFeatureComparison(self, a, b, mode, single)

    def prepare(self, x1, x2=None, *, mode=None):
        """Encode now for explicit reuse across calls/consumers; owns no hidden cache."""
        selected = self.mode if mode is None else mode
        if selected == 'single' and x2 is None:
            raise TypeError('single mode requires both input tensors')
        return self._prepare(x1, x2, selected == 'single', selected)

    def __call__(self, x1, x2=None, *args, **kwargs):
        selected = kwargs.pop('mode', None)
        return self.prepare(x1, x2, mode=selected)(*args, **kwargs)

    def _bind_operator(self, features1, features2, mode, **kwargs):
        self._check_mode(kwargs.get('mode', mode))
        if self.metric is not None:
            from ._lazy import lazy
            selected = kwargs.pop('mode', mode)
            if selected != 'all_pairs':
                raise ValueError("metric= heads require mode='all_pairs' for lazy binding")
            if not isinstance(features1, torch.Tensor) or (features2 is not None and not isinstance(features2, torch.Tensor)):
                raise TypeError('metric= heads need one feature tensor; use head= for feature trees')
            return lazy(features1, features2, self.metric, **{**self.metric_kwargs, **kwargs})
        from ._custom import CustomComparison
        if isinstance(self.head, CustomComparison):
            return self.head._operator(features1, features2, **kwargs)
        raise TypeError('head=... is an eager collection head; wrap a pairwise function with '
                        'tp.make_pairwise (or use metric=) to stream')

    def _operator(self, x1, x2=None, **kwargs):
        """Encode each original collection once, then bind a streamed all-pairs head."""
        mode = kwargs.pop('mode', None)
        return self.prepare(x1, x2, mode=mode)._operator(**kwargs)


def make_feature_pairwise(encoder, metric=None, *, head=None, mode=None, sample_ndim=1,
                          encode_chunk_size=None, **metric_kwargs):
    """Compose one encoder or query/reference towers with an explicit head.

    Use metric= for built-in distances/kernels, or head= for a CustomComparison
    or eager collection callable. encode(X) extracts features; prepare(X,Y)
    binds them for reuse. Encoder chunks preserve gradients and model mode.
    Empty input collections are rejected without probing encoder output metadata.
    """
    if metric is not None and not isinstance(metric, str):
        raise TypeError('metric must be a string name; pass callable comparisons with head=')
    if (metric is None) == (head is None):
        raise TypeError('provide exactly one of metric or head=')
    if head is not None:
        if not callable(head):
            raise TypeError('head= must be callable')
        if metric_kwargs:
            raise TypeError('metric keyword arguments require metric=; bind head= arguments in the callable')
    if isinstance(encoder, tuple):
        if len(encoder) != 2 or any(not callable(tower) for tower in encoder):
            raise TypeError('encoder must be callable or a (query_encoder, reference_encoder) tuple of callables')
    elif not callable(encoder):
        raise TypeError('encoder must be callable or a (query_encoder, reference_encoder) tuple of callables')
    if type(sample_ndim) is not int or sample_ndim < 1:
        raise ValueError('sample_ndim must be a positive integer')
    _check_chunk_size(encode_chunk_size, 'encode_chunk_size', allow_none=True)
    return FeatureComparison(encoder, metric, head, mode, sample_ndim, encode_chunk_size, metric_kwargs)


__all__ = ['FeatureComparison', 'PreparedFeatureComparison', 'make_feature_pairwise']
