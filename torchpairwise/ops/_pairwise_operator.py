"""Scalar comparison operators backed by the shared native streaming engine."""
import torch
from .. import backends as _backends
from ._validation import _check_chunk_size
from ._comparison_dispatch import _LOOKUP, _VECTOR_SPECS, _CATALOG, _structured
from ._batching import PARAMETER_NDIM, _pointset_layout
from ._operator_axes import _scalar_dim


class LazyPairwise:
    """Common matrix consumers; subclasses supply native sources and metadata."""
    @property
    def entrywise(self):
        """Whether native entrywise expressions can consume this node."""
        return True

    @property
    def ndim(self):
        return len(self.shape)

    def size(self, dim=None):
        if dim is not None and type(dim) is not int:
            raise TypeError('size dim must be an integer')
        return self.shape if dim is None else self.shape[dim]

    @property
    def mT(self):
        return self.T

    def _require_entrywise(self, operation):
        if not self.entrywise:
            raise TypeError(f'{operation} needs entrywise capability; this node has linear capability; '
                            'materialize with .to_dense() or build the expression entrywise')

    def _evaluate(self, operation, **options):
        """Single evaluation boundary for lazy consumers."""
        if operation == 'softmax_matmul' and self.entrywise:
            from ._operator_algebra import ScalarExpression
            if not isinstance(self, ScalarExpression):
                return ScalarExpression('identity', self)._evaluate(operation, **options)
        if self.entrywise:
            result = _backends._dispatch(self, operation, options)
            if result is not NotImplemented:
                return result
        return self._native_evaluate(operation, **options)

    def to_dense(self):
        return self.block(torch.arange(self.shape[-2], device=self.device),
                          torch.arange(self.shape[-1], device=self.device))

    def softmax(self, dim=-1, *, exclude_self=False, mask=None):
        self._require_entrywise('softmax')
        if mask is not None:
            return self._masked_operand(mask, 'logsumexp').softmax(dim, exclude_self=exclude_self)
        if type(dim) is not int or _scalar_dim(dim, self) != -1:
            raise ValueError('softmax supports dim=-1; use A.mT.softmax(-1) for dim=-2')
        if type(exclude_self) is not bool:
            raise TypeError('exclude_self must be a boolean')
        from ._linear import LinearOperator
        return LinearOperator(self, _kind='softmax', _right=exclude_self)

    def centered(self, reference_mean=None):
        from ._linear import _as_linear
        return _as_linear(self).centered(reference_mean)

    def __rmatmul__(self, values):
        if not isinstance(values, torch.Tensor):
            return NotImplemented
        if values.ndim == 1:
            return self.mT.matmul(values)
        return self.mT.matmul(values.transpose(-2, -1)).transpose(-2, -1)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else dict(kwargs)
        name = func.__name__
        if (getattr(func, '__module__', '') or '').startswith('torch.linalg') or name.startswith('linalg_'):
            raise TypeError(f'{name} is not lazy; use torchpairwise.linalg (tp.linalg) or materialize with .to_dense()')
        binary = {'add': 'add', '__add__': 'add', '__radd__': 'add',
                  'sub': 'sub', 'subtract': 'sub', '__sub__': 'sub', '__rsub__': 'sub',
                  'mul': 'mul', 'multiply': 'mul', '__mul__': 'mul', '__rmul__': 'mul',
                  'div': 'div', 'true_divide': 'div', 'pow': 'pow',
                  'minimum': 'minimum', 'maximum': 'maximum',
                  'lt': 'lt', 'less': 'lt', 'le': 'le', 'less_equal': 'le',
                  'gt': 'gt', 'greater': 'gt', 'ge': 'ge', 'greater_equal': 'ge',
                  'eq': 'eq', 'ne': 'ne', 'logical_and': 'logical_and',
                  'logical_or': 'logical_or'}
        if name in binary:
            from ._operator_algebra import _binary
            left = args[0] if args else kwargs.pop('input')
            right = args[1] if len(args) > 1 else kwargs.pop('exponent' if binary[name] == 'pow' else 'other')
            if name.startswith('__r'):
                left, right = right, left
            alpha = kwargs.pop('alpha', 1) if binary[name] in ('add', 'sub') else 1
            if kwargs:
                raise TypeError(f'Unsupported {name} options: {tuple(kwargs)}')
            if alpha != 1:
                right = _binary('mul', right, alpha) if isinstance(right, LazyPairwise) else right * alpha
            return _binary(binary[name], left, right)
        if name == 'where':
            from ._operator_algebra import ScalarExpression, _check_scalar
            condition = args[0] if args else kwargs.pop('condition')
            left = args[1] if len(args) > 1 else kwargs.pop('input')
            right = args[2] if len(args) > 2 else kwargs.pop('other')
            if len(args) > 3 or kwargs:
                raise TypeError(f'Unsupported {name} options: {tuple(kwargs)}')
            for operand in (condition, left, right):
                if isinstance(operand, LazyPairwise):
                    operand._require_entrywise('where')
                else:
                    _check_scalar(operand)
            return ScalarExpression('where', condition, (left, right))
        if name in ('block_diag', 'hstack', 'vstack'):
            from ._linear import _as_linear, _block, _diagonal
            nodes = args if name == 'block_diag' else (args[0] if args else kwargs.pop('tensors'))
            if kwargs:
                raise TypeError(f'Unsupported {name} options: {tuple(kwargs)}')
            if name == 'block_diag':
                rows = []
                for i, a in enumerate(nodes):
                    row = []
                    for j, b in enumerate(nodes):
                        if i == j:
                            row.append(a)
                        else:
                            # An empty contraction represents rectangular zero
                            # blocks without allocating their N-by-M entries.
                            row.append(_as_linear(torch.empty((a.shape[-2], 0),
                                dtype=a.dtype, device=a.device))._compose(_as_linear(
                                    torch.empty((0, b.shape[-1]), dtype=a.dtype, device=a.device))))
                    rows.append(row)
                return _block(rows)
            nodes = [_as_linear(node) for node in nodes]
            result = nodes[0]
            for node in nodes[1:]:
                result = result._hstack(node) if name == 'hstack' else result._vstack(node)
            return result
        value = args[0] if args else kwargs.pop('input', None)
        rest = args[1:]
        if name in ('matmul', '__matmul__', '__rmatmul__', 'mm', 'mv'):
            right = rest[0] if rest else kwargs.pop('other', kwargs.pop('mat2', kwargs.pop('vec', None)))
            if kwargs:
                raise TypeError(f'Unsupported {name} options: {tuple(kwargs)}')
            if name == '__rmatmul__':
                value, right = right, value
            if (func is torch.Tensor.matmul and isinstance(value, torch.Tensor)
                    and torch.compiler.is_dynamo_compiling()):
                # Dynamo delivers both `A @ v` and `v @ A` as Tensor.matmul(v, A).
                raise TypeError('Inside torch.compile, `@` between a lazy object and a tensor is ambiguous; '
                                'use A.matmul(v), torch.matmul(A, v) or torch.matmul(v, A)')
            if isinstance(value, LazyPairwise):
                return value.__matmul__(right)
            return right.__rmatmul__(value)
        if name in ('transpose', 'mT'):
            dim0 = rest[0] if rest else kwargs.pop('dim0', -2)
            dim1 = rest[1] if len(rest) > 1 else kwargs.pop('dim1', -1)
            valid = all(type(dim) is int and -value.ndim <= dim < value.ndim for dim in (dim0, dim1))
            if kwargs or not valid or {dim0 % value.ndim, dim1 % value.ndim} != {value.ndim-2, value.ndim-1}:
                raise TypeError('lazy transpose supports only the last two axes; materialize with .to_dense()')
            return value.mT
        if name == 'diagonal':
            offset = rest[0] if rest else kwargs.pop('offset', 0)
            dim1 = rest[1] if len(rest) > 1 else kwargs.pop('dim1', 0)
            dim2 = rest[2] if len(rest) > 2 else kwargs.pop('dim2', 1)
            valid = all(type(dim) is int and -value.ndim <= dim < value.ndim for dim in (dim1, dim2))
            if type(offset) is not int or offset != 0 or kwargs or not valid or {dim1 % value.ndim, dim2 % value.ndim} != {value.ndim-2, value.ndim-1}:
                raise TypeError('lazy diagonal requires offset 0 and the last two axes; materialize with .to_dense()')
            return value.diagonal()
        if name == 'trace':
            if value.ndim != 2 or rest or kwargs:
                raise TypeError('trace requires a two-dimensional lazy matrix')
            return value.diagonal().sum()
        methods = {'negative': '__neg__', 'neg': '__neg__', '__neg__': '__neg__',
                   'exp': 'exp', 'square': 'square', 'sqrt': 'sqrt', 'log': 'log',
                   'sin': 'sin', 'cos': 'cos', 'tanh': 'tanh', 'sigmoid': 'sigmoid',
                   'abs': 'abs', 'absolute': 'abs', 'logical_not': 'logical_not',
                   'clamp': 'clamp', 'sum': 'sum', 'mean': 'mean',
                   'amin': 'amin', 'amax': 'amax', 'min': 'min', 'max': 'max',
                   'argmin': 'argmin', 'argmax': 'argmax', 'prod': 'prod',
                   'all': 'all', 'any': 'any', 'var': 'var', 'std': 'std',
                   'logsumexp': 'logsumexp', 'count_nonzero': 'count_nonzero',
                   'topk': 'topk', 'median': 'median', 'quantile': 'quantile', 'softmax': 'softmax'}
        if name in methods:
            return getattr(value, methods[name])(*rest, **kwargs)
        raise TypeError(f'{name} is not lazy; materialize with .to_dense() or use torchpairwise.linalg')

    def __add__(self, other):
        from ._operator_algebra import _binary
        return _binary('add', self, other)

    def __radd__(self, other):
        from ._operator_algebra import _binary
        return _binary('add', other, self)

    def __sub__(self, other):
        from ._operator_algebra import _binary
        return _binary('sub', self, other)

    def __rsub__(self, other):
        from ._operator_algebra import _binary
        return _binary('sub', other, self)

    def __mul__(self, other):
        from ._operator_algebra import _binary
        return _binary('mul', self, other)

    def __rmul__(self, other):
        from ._operator_algebra import _binary
        return _binary('mul', other, self)

    def __neg__(self):
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('neg', self)

    def square(self):
        self._require_entrywise('square')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('square', self)

    def exp(self):
        self._require_entrywise('exp')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('exp', self)

    def sqrt(self):
        self._require_entrywise('sqrt')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('sqrt', self)

    def log(self):
        self._require_entrywise('log')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('log', self)

    def sin(self):
        self._require_entrywise('sin')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('sin', self)

    def cos(self):
        self._require_entrywise('cos')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('cos', self)

    def tanh(self):
        self._require_entrywise('tanh')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('tanh', self)

    def sigmoid(self):
        self._require_entrywise('sigmoid')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('sigmoid', self)

    def abs(self):
        self._require_entrywise('abs')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('abs', self)

    def logical_not(self):
        self._require_entrywise('logical_not')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('logical_not', self)

    def __truediv__(self, other):
        self._require_entrywise('div')
        from ._operator_algebra import _binary
        return _binary('div', self, other)

    def __rtruediv__(self, other):
        self._require_entrywise('div')
        from ._operator_algebra import _binary
        return _binary('div', other, self)

    def __pow__(self, other):
        self._require_entrywise('pow')
        from ._operator_algebra import _binary
        return _binary('pow', self, other)

    def __rpow__(self, other):
        self._require_entrywise('pow')
        from ._operator_algebra import _binary
        return _binary('pow', other, self)

    def minimum(self, other):
        self._require_entrywise('minimum')
        from ._operator_algebra import _binary
        return _binary('minimum', self, other)

    def maximum(self, other):
        self._require_entrywise('maximum')
        from ._operator_algebra import _binary
        return _binary('maximum', self, other)

    def __lt__(self, other):
        self._require_entrywise('lt')
        from ._operator_algebra import _binary
        return _binary('lt', self, other)

    def __le__(self, other):
        self._require_entrywise('le')
        from ._operator_algebra import _binary
        return _binary('le', self, other)

    def __gt__(self, other):
        self._require_entrywise('gt')
        from ._operator_algebra import _binary
        return _binary('gt', self, other)

    def __ge__(self, other):
        self._require_entrywise('ge')
        from ._operator_algebra import _binary
        return _binary('ge', self, other)

    def eq(self, other):
        self._require_entrywise('eq')
        from ._operator_algebra import _binary
        return _binary('eq', self, other)

    def ne(self, other):
        self._require_entrywise('ne')
        from ._operator_algebra import _binary
        return _binary('ne', self, other)

    def logical_and(self, other):
        self._require_entrywise('logical_and')
        from ._operator_algebra import _binary
        return _binary('logical_and', self, other)

    def logical_or(self, other):
        self._require_entrywise('logical_or')
        from ._operator_algebra import _binary
        return _binary('logical_or', self, other)

    def __abs__(self):
        return self.abs()

    def __and__(self, other):
        from .masks import PairMask
        if isinstance(other, PairMask):  # defer to PairMask.__rand__
            return NotImplemented
        return self.logical_and(other)

    def __or__(self, other):
        from .masks import PairMask
        if isinstance(other, PairMask):
            return NotImplemented
        return self.logical_or(other)

    def __invert__(self):
        return self.logical_not()

    def clamp(self, min=None, max=None):
        self._require_entrywise('clamp')
        from ._operator_algebra import ScalarExpression, _check_scalar
        if min is None and max is None:
            raise ValueError('clamp requires min or max')
        result = self
        for operation, bound in (('clamp_min', min), ('clamp_max', max)):
            if bound is not None:
                _check_scalar(bound)
                result = ScalarExpression(operation, result, bound)
        return result

    def where(self, condition, other):
        self._require_entrywise('where')
        from ._operator_algebra import ScalarExpression, _check_scalar
        for operand in (condition, other):
            if isinstance(operand, LazyPairwise):
                operand._require_entrywise('where')
            else:
                _check_scalar(operand)
        return ScalarExpression('where', condition, (self, other))

    def row_values(self, values):
        self._require_entrywise('row_values')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('row_values', self, values)

    def column_values(self, values):
        self._require_entrywise('column_values')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('column_values', self, values)

    def row_index(self):
        self._require_entrywise('row_index')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('row_index', self)

    def column_index(self):
        self._require_entrywise('column_index')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('column_index', self)

    def add_rows(self, values):
        self._require_entrywise('add_rows')
        from ._operator_algebra import ScalarExpression
        return self + ScalarExpression('row_values', self, values)

    def add_columns(self, values):
        self._require_entrywise('add_columns')
        from ._operator_algebra import ScalarExpression
        return self + ScalarExpression('column_values', self, values)

    def weight_rows(self, weights):
        self._require_entrywise('weight_rows')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('rows', self, weights)

    def weight_columns(self, weights):
        self._require_entrywise('weight_columns')
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('columns', self, weights)

    def matmul(self, values):
        """A @ values; [M] or [...,M,C], in sample/channel tiles."""
        if isinstance(values, LazyPairwise):
            from ._linear import _as_linear
            return _as_linear(self)._compose(values)
        return self._evaluate('matmul', right=values)

    def softmax_matmul(self, values, *, exclude_self=False, mask=None):
        """softmax(A, dim=-1) @ values without retaining the score matrix.

        Normalize over references M, preserving batch and query axes. Empty
        reference sets and entirely excluded rows return zero. Other nonfinite
        scores follow torch.softmax. Values have shape [M] or [...,M,C].
        mask= normalizes over eligible pairs only; see docs/pair-masks.md.
        """
        self._require_entrywise('softmax_matmul')
        if mask is not None:
            return self._masked_operand(mask, 'logsumexp').softmax_matmul(values, exclude_self=exclude_self)
        if type(exclude_self) is not bool:
            raise TypeError('exclude_self must be a boolean')
        from ._operator_algebra import ScalarExpression
        expression = self if isinstance(self, ScalarExpression) else ScalarExpression('identity', self)
        return expression._evaluate('softmax_matmul', right=values, exclude_self=exclude_self)

    def __matmul__(self, values):
        if isinstance(values, LazyPairwise):
            from ._linear import _as_linear
            return _as_linear(self)._compose(values)
        return self.matmul(values)

    def bilinear(self, left, right):
        """left.T @ A @ right, retaining neither A nor A @ right."""
        self._require_entrywise('bilinear')
        return self._evaluate('bilinear', left=left, right=right)

    def quadratic(self, values):
        """values.T @ A @ values; square A required, symmetry/PSD not assumed."""
        self._require_entrywise('quadratic')
        return self._evaluate('quadratic', left=values)

    def frobenius_inner_product(self, other, *, center=False, normalize=False):
        """Sum entry products, optionally double-center and normalize both sources.

        With center=True and normalize=True this is centered kernel alignment
        for two Gram operators over the same observations. Returns one scalar
        per broadcast batch; a zero norm has value and first derivatives zero.
        """
        self._require_entrywise('frobenius_inner_product')
        from ._operator_algebra import ScalarExpression
        if isinstance(other, LazyPairwise):
            other._require_entrywise('pairwise statistic')
        if not isinstance(other, _ScalarOperator):
            raise TypeError('other must be a scalar operator; wrap dense matrices with tp.as_lazy')
        if type(center) is not bool or type(normalize) is not bool:
            raise TypeError('center and normalize must be booleans')
        expression = self if isinstance(self, ScalarExpression) else ScalarExpression('identity', self)
        packet, left_root, right_root = expression._packet(other)
        operation = ('centered_' if center else '') + ('alignment' if normalize else 'frobenius')
        return expression._evaluate(operation, _return_tuple=True, packet=packet, correction=0.,
            value_root=left_root, weight_root=right_root)[0]

    def hsic(self, other, *, unbiased=False):
        """HSIC for symmetric Gram sources over the same observations.

        Biased HSIC uses (N-1)**2 (N >= 2). Unbiased HSIC U-centers
        off-diagonal entries (N >= 4); estimates may be negative.
        """
        self._require_entrywise('hsic')
        from ._operator_algebra import ScalarExpression
        if isinstance(other, LazyPairwise):
            other._require_entrywise('pairwise statistic')
        if not isinstance(other, _ScalarOperator):
            raise TypeError('other must be a scalar operator; wrap dense matrices with tp.as_lazy')
        if type(unbiased) is not bool:
            raise TypeError('unbiased must be a boolean')
        expression = self if isinstance(self, ScalarExpression) else ScalarExpression('identity', self)
        packet, left_root, right_root = expression._packet(other)
        return expression._evaluate('hsic_unbiased' if unbiased else 'hsic', _return_tuple=True,
            packet=packet, correction=0., value_root=left_root, weight_root=right_root)[0]

    def diagonal(self):
        self._require_entrywise('diagonal')
        return self._evaluate('diagonal')

    def block(self, rows, columns):
        """Materialize only the selected rows/columns; repeated IDs are allowed."""
        self._require_entrywise('block')
        return self._evaluate('block', rows=rows, columns=columns)

    def topk(self, k, dim=-1, largest=True, sorted=True, *, exclude_self=False, mask=None):
        """Select k entries along one N/M axis, returning values and indices.

        Equal values use the lower axis index. NaNs rank above finite values,
        as in torch.topk. sorted=False permits a sorted result. Terminal tensors
        replace the selected axis with k; batch axes are preserved.
        """
        self._require_entrywise('topk')
        if type(k) is not int or k < 0:
            raise ValueError('k must be a nonnegative integer')
        if type(dim) is not int:
            raise TypeError('topk requires one integer dim')
        dim = _scalar_dim(dim, self)
        if any(type(value) is not bool for value in (largest, sorted, exclude_self)):
            raise TypeError('largest, sorted and exclude_self must be booleans')
        if mask is not None:
            return self._masked_operand(mask, 'topk', largest=largest).topk(
                k, dim, largest, sorted, exclude_self=exclude_self)
        from ._operator_algebra import ScalarExpression
        expression = self if isinstance(self, ScalarExpression) else ScalarExpression('identity', self)
        values, indices = expression._evaluate('reduce', _return_tuple=True, k=k, dim=dim, largest=largest,
            sorted=sorted, exclude_self=exclude_self)
        return torch.return_types.topk((values, indices))

    def median(self, dim=-1, *, keepdim=False, exclude_self=False, mask=None):
        """Exact lower-middle values along one N/M axis (terminal tensor).

        Stable ties follow the original axis order; a NaN selects the first
        logical NaN. Workspace retains complete rows in budgeted row blocks.
        """
        self._require_entrywise('median')
        _reject_mask(mask, 'median')
        return self._selection('median', .5, dim, keepdim, exclude_self, 'lower')

    def quantile(self, q, dim=-1, *, keepdim=False, interpolation='linear', exclude_self=False, mask=None):
        """Exact scalar-q quantile along one N/M axis (terminal tensor).

        q is a finite Python real in [0,1]. Float32/64 entries are required.
        Interpolation matches torch.quantile; ties use stable axis order.
        A NaN propagates through the last stable sorted NaN. Complete rows
        are retained in budgeted row blocks; this is not constant-state.
        """
        self._require_entrywise('quantile')
        _reject_mask(mask, 'quantile')
        return self._selection('quantile', q, dim, keepdim, exclude_self, interpolation)

    def _selection(self, reduction, q, dim, keepdim, exclude_self, interpolation):
        import math
        if type(q) not in (int, float):
            raise TypeError('q must be a real Python scalar')
        if not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('q must be finite and between zero and one')
        if type(dim) is not int:
            raise TypeError('median/quantile require one integer dim')
        dim = _scalar_dim(dim, self)
        if type(keepdim) is not bool or type(exclude_self) is not bool:
            raise TypeError('keepdim and exclude_self must be booleans')
        if interpolation not in ('linear', 'lower', 'higher', 'nearest', 'midpoint'):
            raise ValueError('Unknown quantile interpolation')
        from ._operator_algebra import ScalarExpression
        expression = self if isinstance(self, ScalarExpression) else ScalarExpression('identity', self)
        return expression._evaluate('_selection', reduction=reduction, q=float(q), dim=dim,
                                    keepdim=keepdim, exclude_self=exclude_self, interpolation=interpolation)

    def reduce(self, reduction='sum', dim=None, *, keepdim=False, exclude_self=False, correction=None, mask=None):
        """Reduce both sample axes (None), references (-1) or queries (-2).

        Integers and tuples/lists use axes of the advertised shape; positive
        indices are accepted. None/empty selects N/M together; batches persist.

        All reducers preserve batch axes. var/std/logsumexp/count_nonzero
        use the native scalar-expression statistics engine. Min/max return values,
        propagate NaNs and choose first logical row-major ties. Empty sum/prod/
        all/any use 0/1/True/False; empty mean is NaN; empty min/max raise.
        exclude_self removes the positional diagonal and requires square A.
        all/any use numeric nonzero truth; arbitrary predicates are separate.
        mask= reduces where(mask, A, fill) with the reduction's identity fill;
        mean/var/std normalize by the eligible count (docs/pair-masks.md).
        """
        self._require_entrywise('reduce')
        if mask is not None:
            if reduction in ('mean', 'var', 'std'):
                if reduction == 'mean' and correction is not None:
                    raise ValueError('correction is only used by var/std')
                from . import masks
                weights = masks._lower(mask, self)
                return self._weighted_statistics_call(weights, reduction, dim, keepdim, exclude_self,
                    0. if reduction == 'mean' else 1. if correction is None else correction)
            return self._masked_operand(mask, reduction).reduce(
                reduction, dim, keepdim=keepdim, exclude_self=exclude_self, correction=correction)
        multiple = isinstance(dim, (tuple, list))
        dim = _scalar_dim(dim, self)
        if type(keepdim) is not bool or type(exclude_self) is not bool:
            raise TypeError('keepdim and exclude_self must be booleans')
        if reduction in ('var', 'std', 'logsumexp', 'count_nonzero'):
            if correction is not None and reduction not in ('var', 'std'):
                raise ValueError('correction is only used by var/std')
            if correction is not None and type(correction) not in (int, float):
                raise TypeError('correction must be a real number')
            from ._operator_algebra import ScalarExpression
            expression = self if isinstance(self, ScalarExpression) else ScalarExpression('identity', self)
            if multiple:
                return expression._axes(reduction, [] if dim is None else [dim],
                    keepdim, exclude_self, 1. if correction is None else float(correction))
            return expression._statistics(reduction, dim, keepdim, exclude_self,
                                          1. if correction is None else float(correction))
        if correction is not None:
            raise ValueError('correction is only used by var/std')
        if multiple and not (reduction == 'prod' and hasattr(self, 'matrix')):
            from ._operator_algebra import ScalarExpression
            expression = self if isinstance(self, ScalarExpression) else ScalarExpression('identity', self)
            return expression._axes(reduction, [] if dim is None else [dim], keepdim, exclude_self, None)
        return self._evaluate('reduce', reduction=reduction, dim=dim,
            keepdim=keepdim, exclude_self=exclude_self)

    def sum(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self.reduce('sum', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def mean(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self.reduce('mean', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def amin(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self.reduce('min', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def amax(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self.reduce('max', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def _extreme(self, largest, dim, keepdim, exclude_self, mask=None):
        self._require_entrywise('max' if largest else 'min')
        if type(dim) is not int:
            raise TypeError('min/max require one integer dim')
        axis = _scalar_dim(dim, self)
        if type(keepdim) is not bool:
            raise TypeError('keepdim must be a boolean')
        if not largest and self.dtype.is_floating_point:
            # Reversing float scores lets the existing largest/nan_first
            # selector choose the first NaN for min as well as max. Keep integer
            # minima on the direct path to avoid negating the minimum integer.
            result = (-self).topk(1, axis, largest=True, exclude_self=exclude_self, mask=mask)
            values, indices = -result.values, result.indices
        else:
            result = self.topk(1, axis, largest=largest, exclude_self=exclude_self, mask=mask)
            values, indices = result.values, result.indices
        if not keepdim:
            values, indices = values.squeeze(axis), indices.squeeze(axis)
        kind = torch.return_types.max if largest else torch.return_types.min
        return kind((values, indices))

    def min(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        if dim is None:
            return self.amin(keepdim=keepdim, exclude_self=exclude_self, mask=mask)
        return self._extreme(False, dim, keepdim, exclude_self, mask)

    def max(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        if dim is None:
            return self.amax(keepdim=keepdim, exclude_self=exclude_self, mask=mask)
        return self._extreme(True, dim, keepdim, exclude_self, mask)

    def _argextreme(self, largest, dim, keepdim, exclude_self, mask=None):
        if type(keepdim) is not bool:
            raise TypeError('keepdim must be a boolean')
        if dim is not None:
            return self._extreme(largest, dim, keepdim, exclude_self, mask).indices
        # Flatten only the logical sample axes, preserving batch axes.
        rows = self._extreme(largest, -1, False, exclude_self, mask)
        from ._matrix_operator import MatrixOperator
        column = MatrixOperator(rows.values.unsqueeze(-2))._extreme(largest, -1, True, False)
        row = column.indices.squeeze(-2)
        indices = row.squeeze(-1) * self.shape[-1] + rows.indices.gather(-1, row).squeeze(-1)
        return indices.unsqueeze(-1).unsqueeze(-1) if keepdim else indices

    def argmin(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self._argextreme(False, dim, keepdim, exclude_self, mask)

    def argmax(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self._argextreme(True, dim, keepdim, exclude_self, mask)


    def prod(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self.reduce('prod', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def all(self, dim=None, *, keepdim=False, exclude_self=False, predicate=None, mask=None):
        if predicate is not None:
            _reject_mask(mask, 'predicate all')
            return self._predicate_reduce('all', predicate, dim, keepdim, exclude_self)
        return self.reduce('all', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def any(self, dim=None, *, keepdim=False, exclude_self=False, predicate=None, mask=None):
        if predicate is not None:
            _reject_mask(mask, 'predicate any')
            return self._predicate_reduce('any', predicate, dim, keepdim, exclude_self)
        return self.reduce('any', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def _predicate_reduce(self, operation, predicate, dim, keepdim, exclude_self):
        """Explicit Python tile callback escape hatch; not compile/exportable."""
        self._require_entrywise(operation)
        if not callable(predicate):
            raise TypeError('predicate must be callable and is supported only for all/any')
        axis = _scalar_dim(dim, self)
        if type(keepdim) is not bool or type(exclude_self) is not bool:
            raise TypeError('keepdim and exclude_self must be booleans')
        n, m = self.shape[-2:]
        if exclude_self and n != m:
            raise ValueError('exclude_self=True requires equal sample counts')
        identity = operation == 'all'
        output_shape = (*self.shape[:-2], *((n,) if axis == -1 else (m,) if axis == -2 else ()))
        result = torch.full(output_shape, identity, dtype=torch.bool, device=self.device)
        size = getattr(self, 'chunk_size', None) or 128
        with torch.no_grad():
            for start in range(0, n, size):
                rows = torch.arange(start, min(n, start + size), device=self.device)
                for other in range(0, m, size):
                    columns = torch.arange(other, min(m, other + size), device=self.device)
                    tile = self.block(rows, columns)
                    truth = predicate(tile)
                    if not isinstance(truth, torch.Tensor) or truth.dtype != torch.bool:
                        raise TypeError('predicate must return a Boolean tensor')
                    if truth.shape != tile.shape or truth.device != tile.device or truth.layout != torch.strided:
                        raise ValueError('predicate must preserve the distance tile shape, device and dense layout')
                    if exclude_self:
                        truth = torch.where(rows[:, None] != columns[None, :], truth, identity)
                    dims = (-2, -1) if axis is None else axis
                    reduced = truth.all(dims) if identity else truth.any(dims)
                    index = (..., slice(start, start + rows.numel())) if axis == -1 else (
                        (..., slice(other, other + columns.numel())) if axis == -2 else (...,))
                    result[index] = result[index] & reduced if identity else result[index] | reduced
        if keepdim:
            return result.unsqueeze(-1).unsqueeze(-1) if axis is None else result.unsqueeze(axis)
        return result

    def var(self, dim=None, *, correction=1, keepdim=False, exclude_self=False, mask=None):
        """Stable sample variance, preserving batch axes; correction defaults to 1."""
        return self.reduce('var', dim, correction=correction, keepdim=keepdim, exclude_self=exclude_self,
                           mask=mask)

    def std(self, dim=None, *, correction=1, keepdim=False, exclude_self=False, mask=None):
        return self.reduce('std', dim, correction=correction, keepdim=keepdim, exclude_self=exclude_self,
                           mask=mask)

    def logsumexp(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self.reduce('logsumexp', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def count_nonzero(self, dim=None, *, keepdim=False, exclude_self=False, mask=None):
        return self.reduce('count_nonzero', dim, keepdim=keepdim, exclude_self=exclude_self, mask=mask)

    def _weighted_statistics_call(self, weights, reduction, dim, keepdim, exclude_self, correction, mask=None):
        self._require_entrywise('_weighted_statistics_call')
        multiple = isinstance(dim, (tuple, list))
        if not isinstance(weights, (torch.Tensor, _ScalarOperator)):
            raise TypeError('Weights must be an axis confidence tensor or scalar operator')
        dim = _scalar_dim(dim, self, weights)
        if type(keepdim) is not bool or type(exclude_self) is not bool:
            raise TypeError('keepdim and exclude_self must be booleans')
        if type(correction) not in (int,float):
            raise TypeError('correction must be a real number')
        from ._operator_algebra import ScalarExpression
        if isinstance(weights,torch.Tensor):
            if dim is None:
                raise ValueError('Tensor confidence weights require dim=-1 or -2; wrap pairwise weights with tp.as_lazy')
            if weights.ndim < 1:
                raise ValueError('Confidence weights must have shape [...,M] or [...,N]')
            if mask is not None:
                weights = ScalarExpression('column_values' if dim == -1 else 'row_values', self, weights)
            else:
                weights = ScalarExpression('columns' if dim == -1 else 'rows', True, weights)
        elif isinstance(weights, LazyPairwise) and not weights.entrywise:
            weights._require_entrywise('weighted statistics')
        elif not isinstance(weights,_ScalarOperator):
            raise TypeError('Weights must be an axis confidence tensor or scalar operator')
        if mask is not None:
            # Masked-out pairs get zero confidence, so eligible weights normalize.
            from . import masks
            weights = weights.where(masks._lower(mask, self), 0)
        expression = self if isinstance(self,ScalarExpression) else ScalarExpression('identity',self)
        return expression._weighted_statistics(weights,reduction,dim,keepdim,exclude_self,float(correction),
            ([] if dim is None else [dim]) if multiple else None)

    def weighted_mean(self, weights, dim=-1, *, keepdim=False, exclude_self=False, mask=None):
        """Normalize nonnegative confidence/operator weights over the sample axis."""
        return self._weighted_statistics_call(weights,'mean',dim,keepdim,exclude_self,0.,mask)

    def weighted_var(self, weights, dim=-1, *, correction=0, keepdim=False, exclude_self=False, mask=None):
        """Reliability-weighted variance; correction=0 population, 1 unbiased."""
        return self._weighted_statistics_call(weights,'var',dim,keepdim,exclude_self,correction,mask)

    def weighted_std(self, weights, dim=-1, *, correction=0, keepdim=False, exclude_self=False, mask=None):
        return self._weighted_statistics_call(weights,'std',dim,keepdim,exclude_self,correction,mask)

    def masked(self, mask, value=0):
        """Lazy where(mask, A, value) for products, solvers and further algebra."""
        from . import masks
        return self.where(masks._lower(mask, self), value)

    def _masked_operand(self, mask, reduction, *, largest=None):
        """where(mask, A, fill), with the fill that removes an entry from reduction."""
        from . import masks
        return self.masked(mask, masks._fill(reduction, self.dtype, largest=largest))


def _reject_mask(mask, operation):
    if mask is not None:
        raise ValueError(f'{operation} does not accept mask=; use A.masked(mask, value) for an explicit fill')


_ScalarOperator = LazyPairwise


class PairwiseOperator(_ScalarOperator):
    """Bind scalar comparison inputs and metric settings without storing their matrix.

    Names/aliases follow compare(). Full public names distinguish similarities
    from distances. Tensor mutations remain visible on subsequent calls. Build
    the binding inside forward for dynamic compilation/export. Built-in execution
    and first backward are native; no Python tile callback is used.
    """
    def __init__(self, x1, x2=None, *, metric='euclidean', chunk_size=None, **metric_kwargs):
        if not isinstance(metric, str) or metric not in _LOOKUP:
            raise ValueError(f'Unknown comparison {metric!r}; use list_comparisons()')
        name = _LOOKUP[metric]
        self._family = _CATALOG[name].family
        self._ordered_family = self._family in ('sequence', 'tokens') or name in ('mmd_distances', 'mmd_statistics')
        self._media_family = self._family in ('image', 'audio')
        self._structured_family = self._family in ('covariance', 'gaussian', 'mixture')
        if name not in _VECTOR_SPECS and not self._structured_family and not self._media_family and not self._ordered_family:
            raise ValueError(f'Operator streaming does not yet support {self._family} comparisons')
        self._spec, self._is_kernel = _VECTOR_SPECS.get(name, (None, False))
        self.metric, self.x1, self.x2 = name, x1, x2
        self.chunk_size, self.metric_kwargs = chunk_size, dict(metric_kwargs)
        self._transposed = False
        allowed = set(_CATALOG[name].parameters if self._structured_family or self._media_family or self._ordered_family else self._spec.parameters)
        if 'chunk_size' in allowed:
            allowed.remove('chunk_size')
            if (not self._structured_family and not self._media_family and not self._ordered_family) or name in ('mmd_distances', 'mmd_statistics'):
                allowed.add('metric_chunk_size')
        if not self._structured_family and not self._media_family and self._spec is not None and self._spec.event_ndim == 3:
            allowed.update(('batched', 'feature_dim'))
        unused = set(metric_kwargs) - allowed
        if unused:
            raise TypeError(f'{name} does not accept parameter {sorted(unused)[0]!r}')
        if self._family in ('gaussian', 'mixture'):
            count = 2 if self._family == 'gaussian' else 3
            _structured(x1, count, 'x1')
            if x2 is not None:
                _structured(x2, count, 'x2')
        elif not isinstance(x1, torch.Tensor) or (x2 is not None and not isinstance(x2, torch.Tensor)):
            raise TypeError('Operator inputs must be tensors')
        _check_chunk_size(chunk_size, allow_none=True)
        for key in ('metric_chunk_size', 'component_chunk_size'):
            if key in metric_kwargs:
                _check_chunk_size(metric_kwargs[key], key, allow_none=True)
        if not self._structured_family and not self._media_family and not self._ordered_family and self._spec.event_ndim == 2 and (x1.ndim < 2 or (x2 is not None and x2.ndim < 2)):
            raise ValueError('Vector inputs must have rank >= 2')

    def _layout(self, value):
        if not self._structured_family and not self._media_family and self._spec is not None and self._spec.event_ndim == 3:
            return _pointset_layout(value, self.metric_kwargs.get('batched'), self.metric_kwargs.get('feature_dim', -1))
        return value

    @property
    def shape(self):
        if self._family in ('sequence', 'tokens'):
            rank = 3 if self._family == 'sequence' else 2
            x, y = self.x1, self.x1 if self.x2 is None else self.x2
            shapes = [x.shape[:-rank], y.shape[:-rank]]
            if self._family == 'sequence':
                shapes += [self.metric_kwargs[key].shape[:-1] for key in ('x1_lengths', 'x2_lengths')
                           if isinstance(self.metric_kwargs.get(key), torch.Tensor)]
            n, m = x.shape[-rank], y.shape[-rank]
            return torch.Size((*torch.broadcast_shapes(*shapes), *((m, n) if self._transposed else (n, m))))
        if self.metric == 'mmd_statistics':
            x, y = (_pointset_layout(v, self.metric_kwargs.get('batched'), self.metric_kwargs.get('feature_dim', -1))
                    for v in (self.x1, self.x1 if self.x2 is None else self.x2))
            shapes = [x.shape[:-3], y.shape[:-3]]
            shapes += [self.metric_kwargs[key].shape[:-1] for key in ('x1_lengths', 'x2_lengths')
                       if isinstance(self.metric_kwargs.get(key), torch.Tensor)]
            n, m = x.shape[-3], y.shape[-3]
            return torch.Size((*torch.broadcast_shapes(*shapes), *((m, n) if self._transposed else (n, m))))
        if self._media_family:
            rank = 4 if self._family == 'image' else 3
            x, y = self.x1, self.x1 if self.x2 is None else self.x2
            n, m = x.shape[-rank], y.shape[-rank]
            return torch.Size((*torch.broadcast_shapes(x.shape[:-rank], y.shape[:-rank]),
                               *((m, n) if self._transposed else (n, m))))
        if self._structured_family:
            ranks = (3,) if self._family == 'covariance' else (2, 3) if self._family == 'gaussian' else (3, 4, 2)
            x = (self.x1,) if self._family == 'covariance' else self.x1
            y = x if self.x2 is None else (self.x2,) if self._family == 'covariance' else self.x2
            shapes = [t.shape[:-rank] for collection in (x, y) for t, rank in zip(collection, ranks)]
            for key in ('x1_lengths', 'x2_lengths'):
                if isinstance(self.metric_kwargs.get(key), torch.Tensor):
                    shapes.append(self.metric_kwargs[key].shape[:-1])
            n, m = x[0].shape[-ranks[0]], y[0].shape[-ranks[0]]
            return torch.Size((*torch.broadcast_shapes(*shapes), *((m, n) if self._transposed else (n, m))))
        x, y = self._layout(self.x1), self._layout(self.x1 if self.x2 is None else self.x2)
        rank = self._spec.event_ndim
        shapes = [x.shape[:-rank], y.shape[:-rank]]
        for key, event in (*PARAMETER_NDIM.items(), ('x1_lengths', 1), ('x2_lengths', 1)):
            if isinstance(self.metric_kwargs.get(key), torch.Tensor):
                shapes.append(self.metric_kwargs[key].shape[:-event])
        n, m = x.shape[-rank], y.shape[-rank]
        return torch.Size((*torch.broadcast_shapes(*shapes), *((m, n) if self._transposed else (n, m))))

    @property
    def dtype(self):
        if self._family == 'tokens':
            return torch.int64
        if self._ordered_family:
            return torch.float32 if self.x1.dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(self.x1.device.type) else self.x1.dtype
        if self._media_family:
            return torch.float64 if self.x1.dtype == torch.float64 else torch.float32
        if self._structured_family:
            x = self.x1 if self._family == 'covariance' else self.x1[0]
            return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(x.device.type) else x.dtype
        weight = self.metric_kwargs.get('w')
        if not self._spec.differentiable:
            result = weight.dtype if isinstance(weight, torch.Tensor) else torch.float32
        else:
            result = self.x1.dtype
        if result in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(self.x1.device.type):
            return torch.float32
        return result

    @property
    def device(self):
        return self.x1[0].device if self._family in ('gaussian', 'mixture') else self.x1.device

    @property
    def T(self):
        """Logical transpose, keeping the original directed metric and settings."""
        result = PairwiseOperator(self.x1, self.x2, metric=self.metric,
            chunk_size=self.chunk_size, **self.metric_kwargs)
        result._transposed = not self._transposed
        _backends._set_backend(result, getattr(self, '_backend', None))
        return result

    def _native_evaluate(self, operation, **options):
        if self._ordered_family:
            op = (torch.ops.torchpairwise.sequence_pairwise_operations if self._family == 'sequence'
                  else torch.ops.torchpairwise.token_pairwise_operations if self._family == 'tokens'
                  else torch.ops.torchpairwise.mmd_pairwise_operations)
            return op(self.x1, self.x2, self.metric, operation,
                transposed=self._transposed, chunk_size=self.chunk_size,
                **options, **self.metric_kwargs)
        if self._media_family:
            kwargs = dict(self.metric_kwargs)
            if 'resolutions' in kwargs:
                kwargs['resolutions'] = [v for n, hop, win in kwargs['resolutions']
                    for v in (n, max(1, n//4) if hop is None else hop, n if win is None else win)]
            op = (torch.ops.torchpairwise.image_pairwise_operations if self._family == 'image'
                  else torch.ops.torchpairwise.audio_pairwise_operations)
            return op(self.x1, self.x2, self.metric, operation,
                transposed=self._transposed, chunk_size=self.chunk_size,
                **options, **kwargs)
        if self._structured_family:
            x = (self.x1,) if self._family == 'covariance' else self.x1
            y = None if self.x2 is None else (self.x2,) if self._family == 'covariance' else self.x2
            return torch.ops.torchpairwise.structured_pairwise_operations(x, y,
                self.metric, operation, transposed=self._transposed,
                chunk_size=self.chunk_size, **options, **self.metric_kwargs)
        kwargs = dict(self.metric_kwargs)
        if 'kernel' in kwargs:
            kwargs['family'] = kwargs.pop('kernel')
        return torch.ops.torchpairwise.pairwise_operations(self.x1, self.x2,
            self.metric, self._is_kernel, operation, transposed=self._transposed,
            chunk_size=self.chunk_size, **options, **kwargs)



__all__ = ['LazyPairwise']
