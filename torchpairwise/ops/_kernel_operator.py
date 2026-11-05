"""Reusable kernel data/options binding over native streamed operations."""
import torch
from ._validation import _check_chunk_size
from ._registry import KERNELS, resolve
from ._batching import validate_comparison
from ._pairwise_operator import PairwiseOperator, _backends
from ._comparison_dispatch import _LOOKUP


def _kernel_name(kernel):
    # Full public function names and short registry names share one capability
    # table. The native selector uses the same canonical comparison catalog.
    for name, spec in KERNELS.items():
        if kernel == spec.op:
            return name
    if isinstance(kernel, str) and kernel in KERNELS:
        return kernel
    name = _LOOKUP.get(kernel) if isinstance(kernel, str) else None
    if name is not None and name.endswith("_kernel"):
        return name
    resolve(kernel, KERNELS)
    return kernel


def _kernel_operation(x1, x2, kernel, operation, *, left=None, right=None,
                      rows=None, columns=None, reduction='sum', dim=None,
                      keepdim=False, exclude_self=False, chunk_size=None, **kwargs):
    kernel = _kernel_name(kernel)
    if kernel not in KERNELS:
        if operation in ('displacement', 'displacement_normalized'):
            raise ValueError('Geometric kernel operations require vector inputs')
        return PairwiseOperator(x1, x2, metric=kernel, chunk_size=chunk_size, **kwargs)._evaluate(
            operation, left=left, right=right, rows=rows, columns=columns,
            reduction=reduction, dim=dim, keepdim=keepdim, exclude_self=exclude_self)
    spec = resolve(kernel, KERNELS)
    if isinstance(x1, torch.Tensor) and x1.device.type in ('cpu', 'cuda') and torch.is_autocast_enabled(x1.device.type):
        from .._integration import autocast_inputs
        with torch.autocast(x1.device.type, enabled=False):
            x1, x2, promoted = autocast_inputs(x1, x2, dict(kwargs, left=left, right=right))
            left, right = promoted.pop('left'), promoted.pop('right')
            return _kernel_operation(x1, x2, kernel, operation, left=left, right=right,
                                     rows=rows, columns=columns, reduction=reduction,
                                     dim=dim, keepdim=keepdim, exclude_self=exclude_self,
                                     chunk_size=chunk_size, **promoted)
    _check_chunk_size(chunk_size, allow_none=True)
    if dim is not None and (type(dim) is not int or dim not in (-1, -2)):
        raise ValueError('dim must be None, -1 or -2')
    if type(keepdim) is not bool or type(exclude_self) is not bool:
        raise TypeError('keepdim and exclude_self must be booleans')
    validate_comparison(spec, x1, x2, kwargs)
    return torch.ops.torchpairwise.kernel_operations(
        x1, x2, kernel, operation, left, right, rows, columns, reduction,
        dim, keepdim, exclude_self, chunk_size, **kwargs)


class KernelOperator(PairwiseOperator):
    """Kernel-specialized PairwiseOperator with geometric interactions.

    The twelve common matrix operations are inherited. This facade retains
    kernel-only selection, the RBF default and legacy native entry points.
    """
    def __init__(self, x1, x2=None, *, kernel='rbf', chunk_size=None, **kernel_kwargs):
        self.kernel = _kernel_name(kernel)
        super().__init__(x1, x2, metric=KERNELS[self.kernel].op if self.kernel in KERNELS else self.kernel,
                         chunk_size=chunk_size, **kernel_kwargs)
        self.kernel_kwargs = self.metric_kwargs

    @property
    def T(self):
        """Logical transpose, retaining the kernel specialization."""
        result = KernelOperator(self.x1, self.x2, kernel=self.kernel,
            chunk_size=self.chunk_size, **self.kernel_kwargs)
        result._transposed = not self._transposed
        _backends._set_backend(result, getattr(self, '_backend', None))
        return result

    def _inputs(self):
        if self._transposed:
            return self.x1 if self.x2 is None else self.x2, self.x1
        return self.x1, self.x2

    def _native_evaluate(self, operation, **options):
        if operation == 'residual_outer_product':
            x, y = self._inputs()
            return _kernel_residual_outer_product(x, y, kernel=self.kernel,
                chunk_size=self.chunk_size, **options, **self.kernel_kwargs)
        if self.kernel not in KERNELS:
            return super()._native_evaluate(operation, **options)
        # Registered vector kernels are symmetric, so swapping operands also
        # gives geometric methods the correct transposed coordinate frame.
        x, y = self._inputs()
        return _kernel_operation(x, y, self.kernel, operation,
            chunk_size=self.chunk_size, **options, **self.kernel_kwargs)

    def displacement(self, *, normalize=False):
        """Sum Kij*(Yj-Xi); optionally divide by row mass (zero mass -> zero)."""
        if type(normalize) is not bool:
            raise TypeError('normalize must be a boolean')
        if self.kernel not in KERNELS:
            raise ValueError('Geometric kernel operations require vector inputs')
        return self._evaluate('displacement_normalized' if normalize else 'displacement')

    def residual_outer_product(self, *, normalize=False, center=False):
        """Sum Kij*delta*delta.T; optionally center and divide by row mass."""
        if self.kernel not in KERNELS:
            raise ValueError('Geometric kernel operations require vector inputs')
        return self._evaluate('residual_outer_product', normalize=normalize, center=center)


def _kernel_residual_outer_product(x1, x2=None, *, kernel='rbf', normalize=False,
                                   center=False, chunk_size=None, **kernel_kwargs):
    """Stream weighted residual outer products, returning [...,N,D,D].

    Residuals are Yj-Xi. Center subtracts the row weighted mean residual;
    normalize divides by row mass. Zero mass gives zero for either option.
    Signed kernels are algebraically allowed without a covariance/PSD guarantee.
    Sample/channel tiles avoid a pair-of-points outer-product intermediate.
    """
    if type(normalize) is not bool or type(center) is not bool:
        raise TypeError('normalize and center must be booleans')
    _check_chunk_size(chunk_size, allow_none=True)
    kernel = _kernel_name(kernel)
    if kernel not in KERNELS:
        raise ValueError('Geometric kernel operations require vector inputs')
    validate_comparison(resolve(kernel, KERNELS), x1, x2, kernel_kwargs)
    return torch.ops.torchpairwise.kernel_residual_outer_product(
        x1, x2, kernel, normalize, center, chunk_size, **kernel_kwargs)


__all__ = ['KernelOperator']
