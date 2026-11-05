"""Public constructors for lazy pairwise matrices and linear maps."""
from ._pairwise_operator import LazyPairwise, PairwiseOperator
from ._kernel_operator import KernelOperator
from ._matrix_operator import MatrixOperator
from ._registry import KERNELS
from ._comparison_dispatch import _LOOKUP
from .. import backends as _backends

_KERNEL_NAMES = frozenset(KERNELS) | frozenset(spec.op for spec in KERNELS.values())


def lazy(x1, x2=None, metric='euclidean', *, chunk_size=None, backend=None, **options):
    """Bind a comparison, kernel or feature comparison without evaluating entries.

    Construct inside forward for dynamic torch.compile and torch.export inputs.
    Arbitrary Python comparisons/encoders retain their own deployment contracts.
    """
    from ._custom import CustomComparison
    from ._features import FeatureComparison, PreparedFeatureComparison
    if backend is not None:
        _backends._validate(backend)
    if isinstance(x1, PreparedFeatureComparison):
        if x2 is not None or metric != 'euclidean':
            raise TypeError('A prepared comparison already binds its inputs and metric')
        result = x1._operator(chunk_size=chunk_size, **options)
        return _backends._set_backend(result, backend)
    if isinstance(metric, (CustomComparison, FeatureComparison)):
        result = metric._operator(x1, x2, chunk_size=chunk_size, **options)
        return _backends._set_backend(result, backend)
    if isinstance(metric, str) and (metric in _KERNEL_NAMES or _LOOKUP.get(metric, metric).endswith('_kernel')):
        result = KernelOperator(x1, x2, kernel=metric, chunk_size=chunk_size, **options)
    else:
        result = PairwiseOperator(x1, x2, metric=metric, chunk_size=chunk_size, **options)
    return _backends._set_backend(result, backend)


def as_lazy(matrix, *, chunk_size=None):
    """Bind a caller-owned dense [...,N,M] tensor as an entrywise leaf."""
    return MatrixOperator(matrix, chunk_size=chunk_size)


def eye_like(A):
    """Linear identity with the square shape, dtype and device of A."""
    from ._linear import _identity
    return _identity(A)


def diag_embed(values):
    """Linear diagonal map for [...,N] values."""
    from ._linear import _diagonal
    return _diagonal(values)


def block(blocks):
    """Assemble a rectangular grid of lazy matrices or dense tensor leaves."""
    from ._linear import _block
    return _block(blocks)


def group_projection(labels, *, like):
    """Linear map that replaces sample values with their group means."""
    from ._linear import _group_projection
    return _group_projection(labels, like=like)


__all__ = ['LazyPairwise', 'lazy', 'as_lazy', 'eye_like', 'diag_embed', 'block', 'group_projection']
