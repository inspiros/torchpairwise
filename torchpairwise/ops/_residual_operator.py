"""A bounded residual feature source with PyTorch-style reduction axes."""
import math
import torch
from ._validation import _check_chunk_size
from ._pairwise_operator import _ScalarOperator
from ._operator_algebra import ScalarExpression
from ._operator_axes import _operator_dims


class ResidualOperator:
    """Lazy G[...,i,j,d] = transform(X[...,i,d] - Y[...,j,d]).

    Feature-only reductions return a lazy scalar expression. Reductions
    including N/M stream directly into Tensors. Dimensions may be integers or
    tuples/lists, with batch axes preserved. Generic feature algebra is deferred.
    """
    def __init__(self, x1, x2=None, *, transform='identity', chunk_size=128,
                 feature_chunk_size=128):
        if not isinstance(x1, torch.Tensor) or (x2 is not None and not isinstance(x2, torch.Tensor)):
            raise TypeError('Residual inputs must be tensors')
        y = x1 if x2 is None else x2
        if x1.ndim < 2 or y.ndim < 2 or x1.layout != torch.strided or y.layout != torch.strided:
            raise ValueError('Residual inputs must be dense with shape [...,N,D]')
        if transform not in ('identity', 'abs', 'square'):
            raise ValueError('transform must be identity, abs or square')
        for name, size in (('chunk_size', chunk_size), ('feature_chunk_size', feature_chunk_size)):
            _check_chunk_size(size, name)
        torch._check(x1.shape[-1] == y.shape[-1])
        if x1.device != y.device:
            raise ValueError('Residual inputs must share a device')
        # Metadata follows the same promotion policy as the native preparation.
        def dtype(t):
            return torch.float32 if t.dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(t.device.type) else t.dtype
        if dtype(x1) != dtype(y) or dtype(x1) not in (torch.float32, torch.float64):
            raise ValueError('Residual inputs require matching float32/float64 dtype; low precision requires autocast')
        self.x1, self.x2 = x1, x2
        self.transform = transform
        self.chunk_size, self.feature_chunk_size = chunk_size, feature_chunk_size

    @property
    def shape(self):
        x, y = self.x1, self.x1 if self.x2 is None else self.x2
        return torch.Size((*torch.broadcast_shapes(x.shape[:-2], y.shape[:-2]),
                           x.shape[-2], y.shape[-2], x.shape[-1]))

    @property
    def dtype(self):
        dtype = self.x1.dtype
        return torch.float32 if dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(self.device.type) else dtype

    @property
    def device(self):
        return self.x1.device

    def reduce(self, reduction='sum', dim=-1, *, correction=None, keepdim=False):
        """Reduce selected N/M/D axes, preserving batch axes.

        D alone produces a lazy scalar source; any sample-axis selection returns
        a Tensor. Tuple/list dimensions form one joint reduction. None/empty
        select all N/M/D. correction applies to var/std, defaulting to 1.
        """
        axes = _operator_dims(dim, len(self.shape), 3)
        if reduction not in ('sum', 'mean', 'var', 'std', 'min', 'max', 'prod', 'all', 'any'):
            raise ValueError('Unsupported residual reduction')
        if correction is not None and reduction not in ('var', 'std'):
            raise ValueError('correction is only used by var/std')
        if correction is not None and (type(correction) not in (int, float) or not math.isfinite(correction)):
            raise ValueError('correction must be a finite real number')
        if type(keepdim) is not bool:
            raise TypeError('keepdim must be a boolean')
        if axes != (-1,):
            return torch.ops.torchpairwise.residual_reduction(
                self.x1, self.x2, reduction, self.transform, list(axes),
                keepdim, correction, self.chunk_size, self.feature_chunk_size)
        if keepdim:
            raise ValueError('Feature-only keepdim=True is unsupported for a lazy scalar source; use keepdim=False')
        leaf = _ResidualReduction(self, reduction, 1. if correction is None else float(correction))
        return ScalarExpression('identity', leaf)

    def sum(self, dim=-1, *, keepdim=False):
        return self.reduce('sum', dim, keepdim=keepdim)

    def mean(self, dim=-1, *, keepdim=False):
        return self.reduce('mean', dim, keepdim=keepdim)

    def min(self, dim=-1, *, keepdim=False):
        return self.reduce('min', dim, keepdim=keepdim)

    def max(self, dim=-1, *, keepdim=False):
        return self.reduce('max', dim, keepdim=keepdim)

    def prod(self, dim=-1, *, keepdim=False):
        return self.reduce('prod', dim, keepdim=keepdim)

    def all(self, dim=-1, *, keepdim=False):
        return self.reduce('all', dim, keepdim=keepdim)

    def any(self, dim=-1, *, keepdim=False):
        return self.reduce('any', dim, keepdim=keepdim)

    def var(self, dim=-1, *, correction=1, keepdim=False):
        return self.reduce('var', dim, correction=correction, keepdim=keepdim)

    def std(self, dim=-1, *, correction=1, keepdim=False):
        return self.reduce('std', dim, correction=correction, keepdim=keepdim)

    def topk(self, k, dim=-1, largest=True, sorted=True):
        """Return terminal [...,N,M,k] values/feature indices without N*M*D.

        One feature axis only. Ties use the lower feature index; NaNs rank
        above finite values. sorted=False permits a sorted result.
        """
        if type(k) is not int or k < 0:
            raise ValueError('k must be a nonnegative integer')
        if type(dim) is not int:
            raise TypeError('topk requires one integer dim')
        if _operator_dims(dim, len(self.shape), 3) != (-1,):
            raise ValueError('Residual topk currently supports only the feature axis')
        if type(largest) is not bool or type(sorted) is not bool:
            raise TypeError('largest and sorted must be booleans')
        values, indices = torch.ops.torchpairwise.residual_topk(
            self.x1, k, self.x2, dim, largest, sorted, self.transform,
            self.chunk_size, self.feature_chunk_size)
        return torch.return_types.topk((values, indices))


class _ResidualReduction(_ScalarOperator):
    """Metadata source for the typed native expression frontend."""
    def __init__(self, source, reduction, correction):
        self.source = source
        self.x1, self.x2 = source.x1, source.x2
        self.chunk_size = source.chunk_size
        self.metric, self._family, self._transposed = reduction, 'residual', False
        self.metric_kwargs = dict(transform=source.transform, correction=correction,
                                  feature_chunk_size=source.feature_chunk_size)

    @property
    def shape(self):
        return self.source.shape[:-1]

    @property
    def dtype(self):
        return torch.bool if self.metric in ('all', 'any') else self.source.dtype

    @property
    def device(self):
        return self.source.device


def residual_operator(x1, x2=None, *, transform='identity', chunk_size=128,
                      feature_chunk_size=128):
    """Bind signed/absolute/squared residual features without allocating N*M*D."""
    return ResidualOperator(x1, x2, transform=transform, chunk_size=chunk_size,
                            feature_chunk_size=feature_chunk_size)


__all__ = ['ResidualOperator', 'residual_operator']
