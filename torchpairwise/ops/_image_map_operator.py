"""Lazy image event programs and native sample/event streaming reductions."""
import math
import torch
from ._validation import _check_chunk_size
from ._image import _prepare, _window_options, _positive_scalar
from ._operator_axes import _operator_dims
from ._operator_algebra import ScalarExpression
from ._pairwise_operator import _ScalarOperator


_DEFAULT_DIMS = object()
_REDUCERS = ('sum', 'mean', 'var', 'std', 'min', 'max', 'prod', 'all', 'any')


class ImageMapOperator:
    """Lazy aligned image maps [...,N,M,C,Hout,Wout].

    Event-only reductions stay lazy, preserving remaining event axes. Once all
    events are contracted the result is a scalar expression. Selecting N or M
    returns a terminal tensor, computed by native sample tiling.
    Signed residual follows image_maps (Y-X).
    """
    def __init__(self, x1, x2=None, *, comparison='squared_error', data_range=None,
                 window_size=11, sigma=1.5, k1=.01, k2=.03, chunk_size=16):
        if comparison not in ('residual', 'absolute_error', 'squared_error', 'ssim'):
            raise ValueError('Unknown image map comparison')
        _check_chunk_size(chunk_size)
        if not isinstance(x1, torch.Tensor) or (x2 is not None and not isinstance(x2, torch.Tensor)):
            raise TypeError('Image inputs must be tensors')
        y = x1 if x2 is None else x2
        if x1.ndim < 4 or y.ndim < 4:
            raise ValueError('Images require shape [...,N,C,H,W]')
        for a, b in zip(x1.shape[-3:], y.shape[-3:]):
            torch._check(a == b)
        if not torch.compiler.is_compiling():
            _prepare(x1, y, 'all_pairs', chunk_size)
            if comparison == 'ssim':
                data_range = _positive_scalar(data_range, 'data_range')
                _window_options(x1, window_size, sigma, k1, k2, None)
            elif data_range is not None or (window_size, sigma, k1, k2) != (11, 1.5, .01, .03):
                raise ValueError('SSIM options require comparison=ssim')
        self._program, self._corrections = (), ()
        self.x1, self.x2, self.comparison = x1, x2, comparison
        self.chunk_size = chunk_size
        self.options = dict(comparison=comparison, data_range=data_range,
                            window_size=window_size, sigma=sigma, k1=k1, k2=k2)

    @property
    def event_shape(self):
        c, h, w = self.x1.shape[-3:]
        crop = self.options['window_size'] - 1 if self.comparison == 'ssim' else 0
        shape = [c, h-crop, w-crop]
        for k in range(len(self._corrections)):
            _, mask, keep = self._program[3*k:3*k+3]
            shape = [1 if mask & (1 << d) else size for d, size in enumerate(shape)] if keep else [size for d, size in enumerate(shape) if not mask & (1 << d)]
        return torch.Size(shape)

    @property
    def shape(self):
        x, y = self.x1, self.x1 if self.x2 is None else self.x2
        return torch.Size((*torch.broadcast_shapes(x.shape[:-4], y.shape[:-4]),
                           x.shape[-4], y.shape[-4], *self.event_shape))

    @property
    def dtype(self):
        dtype = self.x1.dtype
        if dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(self.device.type):
            dtype = torch.float32
        for code in self._program[::3]:
            if code in (7, 8):
                dtype = torch.bool
            elif code in (0, 6) and dtype in (torch.bool, torch.int64):
                dtype = torch.int64
        return dtype

    @property
    def device(self):
        return self.x1.device

    def reduce(self, reduction='mean', dim=_DEFAULT_DIMS, *, correction=None, keepdim=False):
        """Reduce current event axes lazily, or N/M jointly into a tensor.

        Omitted dim contracts every current event axis. Explicit None/[] selects
        N, M and all current events. Batch axes are never reduced.
        """
        rank = len(self.event_shape)
        axes = _operator_dims(tuple(range(-rank, 0)) if dim is _DEFAULT_DIMS else dim,
                              len(self.shape), rank + 2)
        if reduction not in _REDUCERS:
            raise ValueError('Unsupported image-map event reduction')
        if type(keepdim) is not bool:
            raise TypeError('keepdim must be a boolean')
        if correction is not None and reduction not in ('var', 'std'):
            raise ValueError('correction is only used by var/std')
        if correction is not None and (type(correction) not in (int, float) or not math.isfinite(correction)):
            raise ValueError('correction must be a finite real number')
        if reduction in ('mean', 'var', 'std') and not self.dtype.is_floating_point:
            raise TypeError('mean/var/std require floating point entries')
        if any(axis < -rank for axis in axes):
            options = {key: val for key, val in self.options.items() if key != 'comparison'}
            return torch.ops.torchpairwise.image_event_reduction(
                self.x1, self.x2, self.comparison, list(self._program),
                list(self._corrections), reduction, list(axes), keepdim,
                correction, **options, chunk_size=self.chunk_size)
        mask = sum(1 << (axis + rank) for axis in axes)
        view = object.__new__(ImageMapOperator)
        view.x1, view.x2, view.comparison = self.x1, self.x2, self.comparison
        view.options, view.chunk_size = self.options, self.chunk_size
        view._program = (*self._program, _REDUCERS.index(reduction), mask, int(keepdim))
        view._corrections = (*self._corrections, 1. if correction is None else float(correction))
        if view.event_shape:
            return view
        return ScalarExpression('identity', _ImageMapReduction(view))

    def sum(self, dim=_DEFAULT_DIMS, *, keepdim=False):
        return self.reduce('sum', dim, keepdim=keepdim)

    def mean(self, dim=_DEFAULT_DIMS, *, keepdim=False):
        return self.reduce('mean', dim, keepdim=keepdim)

    def min(self, dim=_DEFAULT_DIMS, *, keepdim=False):
        return self.reduce('min', dim, keepdim=keepdim)

    def max(self, dim=_DEFAULT_DIMS, *, keepdim=False):
        return self.reduce('max', dim, keepdim=keepdim)

    def prod(self, dim=_DEFAULT_DIMS, *, keepdim=False):
        return self.reduce('prod', dim, keepdim=keepdim)

    def all(self, dim=_DEFAULT_DIMS, *, keepdim=False):
        return self.reduce('all', dim, keepdim=keepdim)

    def any(self, dim=_DEFAULT_DIMS, *, keepdim=False):
        return self.reduce('any', dim, keepdim=keepdim)

    def var(self, dim=_DEFAULT_DIMS, *, correction=1, keepdim=False):
        return self.reduce('var', dim, correction=correction, keepdim=keepdim)

    def std(self, dim=_DEFAULT_DIMS, *, correction=1, keepdim=False):
        return self.reduce('std', dim, correction=correction, keepdim=keepdim)

    def topk(self, k, dim=-1, largest=True, sorted=True):
        """Terminal values/indices along one current event axis.

        Ties use the lower axis index; NaNs rank highest. Batch/sample axes
        persist. sorted=False permits sorted output, as in torch.topk.
        """
        if type(k) is not int or k < 0:
            raise ValueError('k must be a nonnegative integer')
        if type(dim) is not int:
            raise TypeError('topk requires one integer dim')
        rank = len(self.event_shape)
        axes = _operator_dims(dim, len(self.shape), rank + 2)
        if not rank or axes[0] < -rank:
            raise ValueError('Image topk requires one current event axis; contract events before sample topk')
        if type(largest) is not bool or type(sorted) is not bool:
            raise TypeError('largest and sorted must be booleans')
        if self.dtype == torch.bool:
            raise TypeError('topk requires numeric entries, not boolean')
        options = {key: value for key, value in self.options.items() if key != 'comparison'}
        values, indices = torch.ops.torchpairwise.image_event_topk(
            self.x1, self.x2, self.comparison, list(self._program),
            list(self._corrections), k, dim, largest, sorted,
            **options, chunk_size=self.chunk_size)
        return torch.return_types.topk((values, indices))


class _ImageMapReduction(_ScalarOperator):
    def __init__(self, source):
        self.source = source
        self.x1, self.x2, self.chunk_size = source.x1, source.x2, source.chunk_size
        self.metric, self._family, self._transposed = 'event', 'image_event', False
        self.metric_kwargs = dict(source.options, event_program=source._program,
                                  event_corrections=source._corrections)

    @property
    def shape(self):
        return self.source.shape

    @property
    def dtype(self):
        return self.source.dtype

    @property
    def device(self):
        return self.source.device


def image_map_operator(x1, x2=None, *, comparison='squared_error', data_range=None,
                       window_size=11, sigma=1.5, k1=.01, k2=.03, chunk_size=16):
    """Bind aligned maps; reduce events lazily before scalar streaming."""
    return ImageMapOperator(x1, x2, comparison=comparison, data_range=data_range,
        window_size=window_size, sigma=sigma, k1=k1, k2=k2, chunk_size=chunk_size)


__all__ = ['ImageMapOperator', 'image_map_operator']
