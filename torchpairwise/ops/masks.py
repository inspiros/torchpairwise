"""Structural pair masks for lazy consumers, lowered to native expression nodes.

A mask selects eligible (i, j) entries of a logical [...,N,M] matrix. Consumers
accept ``mask=`` and evaluate ``consumer(where(mask, A, fill))`` with the fill
that removes an entry from that reduction; mean/var/std normalize by the
eligible count. Masks are specifications: they bind no shape until a consumer
lowers them against its operand, so one mask can serve several operators.
"""
import torch

__all__ = ['PairMask', 'same_group', 'different_group', 'causal', 'band', 'diagonal']


class PairMask:
    """Composable pair predicate; combine with &, | and ~."""
    def __init__(self, kind, *operands):
        self._kind, self._operands = kind, operands

    def __and__(self, other):
        return PairMask('and', self, _as_mask(other))

    def __rand__(self, other):
        return PairMask('and', _as_mask(other), self)

    def __or__(self, other):
        return PairMask('or', self, _as_mask(other))

    def __ror__(self, other):
        return PairMask('or', _as_mask(other), self)

    def __invert__(self):
        return PairMask('not', self)

    def __repr__(self):
        return f'PairMask({self._kind!r})'

    def _lower(self, like):
        from ._operator_algebra import ScalarExpression
        kind, operands = self._kind, self._operands
        if kind in ('and', 'or'):
            left, right = (_lower(value, like) for value in operands)
            return ScalarExpression('logical_' + kind, left, right)
        if kind == 'not':
            return ScalarExpression('logical_not', _lower(operands[0], like))
        if kind in ('same_group', 'different_group'):
            rows, columns = operands
            if columns is None:
                if like.shape[-2] != like.shape[-1]:
                    raise ValueError('one label tensor requires a square operand; pass labels for both axes')
                columns = rows
            for name, labels, axis in (('row', rows, -2), ('column', columns, -1)):
                if labels.shape[-1] != like.shape[axis]:
                    raise ValueError(f'{name} labels must have shape [...,{like.shape[axis]}]')
                if labels.device != like.device:
                    raise ValueError('labels must be on the operand device')
            left = ScalarExpression('row_values', like, rows)
            right = ScalarExpression('column_values', like, columns)
            return ScalarExpression('eq' if kind == 'same_group' else 'ne', left, right)
        rows, columns = ScalarExpression('row_index', like), ScalarExpression('column_index', like)
        if kind == 'diagonal':
            return ScalarExpression('eq', columns, rows)
        offset = ScalarExpression('sub', columns, rows)  # j - i
        if kind == 'causal':
            return ScalarExpression('le', offset, operands[0])
        lower, upper = operands
        return ScalarExpression('logical_and', ScalarExpression('ge', offset, -lower),
                                ScalarExpression('le', offset, upper))


def _labels(labels, name):
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f'{name} must be a tensor')
    if labels.layout != torch.strided or labels.ndim < 1:
        raise ValueError(f'{name} must be a dense tensor with shape [...,N]')
    if labels.is_complex():
        raise TypeError(f'{name} must be real or integer labels')
    return labels


def _offset(value, name):
    if type(value) is not int:
        raise TypeError(f'{name} must be an integer')
    return value


def same_group(labels, other=None):
    """Keep pairs whose row label equals the column label.

    labels:[...,N] labels rows; other:[...,M] labels columns (defaults to
    labels, requiring N == M). Batch axes broadcast with the operand.
    """
    return PairMask('same_group', _labels(labels, 'labels'),
                    None if other is None else _labels(other, 'other'))


def different_group(labels, other=None):
    """Keep pairs whose row and column labels differ."""
    return PairMask('different_group', _labels(labels, 'labels'),
                    None if other is None else _labels(other, 'other'))


def causal(offset=0):
    """Keep j <= i + offset (offset=0 keeps the diagonal and its lower part)."""
    return PairMask('causal', _offset(offset, 'offset'))


def band(lower=0, upper=0):
    """Keep -lower <= j - i <= upper."""
    lower, upper = _offset(lower, 'lower'), _offset(upper, 'upper')
    if lower + upper < 0:
        raise ValueError('band is empty: lower + upper must be nonnegative')
    return PairMask('band', lower, upper)


def diagonal():
    """Keep i == j; ~diagonal() keeps the off-diagonal entries."""
    return PairMask('diagonal')


def _as_mask(value):
    from ._pairwise_operator import LazyPairwise
    if isinstance(value, (PairMask, LazyPairwise)):
        return value
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError('masks combine with PairMask, Boolean lazy expressions or dense Boolean tensors')


def _lower(mask, like):
    """Boolean scalar expression with like's logical [...,N,M] axes."""
    lowered = _lower_axes(mask, like)
    batch = tuple(like.shape[:-2])
    try:
        fits = torch.broadcast_shapes(tuple(lowered.shape[:-2]), batch) == batch
    except RuntimeError:
        fits = False
    if not fits:
        # A mask that adds batch axes would make positive consumer dims ambiguous.
        raise ValueError(f'mask batch axes must broadcast to the operand batch shape {list(batch)}')
    return lowered


def _lower_axes(mask, like):
    from ._pairwise_operator import LazyPairwise
    from ._matrix_operator import MatrixOperator
    if isinstance(mask, PairMask):
        return mask._lower(like)
    if isinstance(mask, torch.Tensor):
        if mask.dtype != torch.bool or mask.ndim < 2:
            raise TypeError('a dense mask must be a Boolean tensor with shape [...,N,M]')
        mask = MatrixOperator(mask)
    if not isinstance(mask, LazyPairwise):
        raise TypeError('mask must be a PairMask, a Boolean lazy expression or a dense Boolean tensor')
    mask._require_entrywise('mask')
    if mask.dtype != torch.bool:
        raise TypeError('a lazy mask must have Boolean entries, for example D < radius')
    if tuple(mask.shape[-2:]) != tuple(like.shape[-2:]):
        raise ValueError('mask and operand must have the same N and M axes')
    return mask


def _fill(reduction, dtype, *, largest=None):
    """Entry value that a masked-out position contributes to this reduction."""
    if reduction in ('sum', 'count_nonzero'):
        return 0
    if reduction == 'prod':
        return 1
    if reduction in ('all', 'any'):
        return reduction == 'all'
    if reduction == 'logsumexp':
        return -float('inf')
    if reduction in ('min', 'max'):
        largest = reduction == 'max'
    if dtype == torch.bool:
        return not largest
    if dtype.is_floating_point:
        return -float('inf') if largest else float('inf')
    info = torch.iinfo(dtype)
    return info.min if largest else info.max
