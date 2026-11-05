"""Finite linear-map descriptions over native sources and linear primitives."""
import torch

from ._matrix_operator import MatrixOperator
from ._pairwise_operator import LazyPairwise
from ._operator_axes import _scalar_dim
from ._operator_algebra import ScalarExpression


class LinearOperator(LazyPairwise):
    """Real matrix actions, distinct from entrywise comparison expressions.

    A @ B composes two LinearOperators; A @ tensor applies the resulting map.
    Bind float32/64 sources; factors and RHS tensors share their dtype/device.
    Value mutations are visible. Reconstruct after resizing a bound source.
    """
    def __init__(self, source, *, _kind='source', _right=None):
        self._kind, self._right = _kind, _right
        if _kind == 'lowrank_inverse':
            basis, spectrum, shift = source
            self._left = (basis, spectrum, shift)
            self._shape = torch.Size((*basis.shape[:-1], basis.shape[-2]))
            self._dtype, self._device = shift.dtype, shift.device
            return
        if _kind == 'diagonal':
            if not isinstance(source, torch.Tensor):
                raise TypeError('Diagonal entries must be a Tensor')
            if source.layout != torch.strided or source.ndim < 1 or source.dtype not in (torch.float32, torch.float64):
                raise ValueError('Diagonal entries must be strided float32/float64 [...,N]')
            self._left = source
            self._shape = torch.Size((*source.shape, source.shape[-1]))
            self._dtype, self._device = source.dtype, source.device
            return
        if _kind == 'group':
            indices, counts, like = source
            self._left, self._right = indices, counts
            self._shape = torch.Size((*indices.shape, indices.shape[-1]))
            self._dtype, self._device = like.dtype, like.device
            return
        if _kind == 'source':
            if isinstance(source, torch.Tensor):
                source = MatrixOperator(source)
            if not isinstance(source, LazyPairwise):
                raise TypeError('Expected a dense Tensor or LazyPairwise; contract event axes first')
            self._left = source
            self._shape, self._dtype, self._device = source.shape, source.dtype, source.device
            if self._dtype not in (torch.float32, torch.float64):
                raise TypeError('Linear operators require float32 or float64 entries')
            return
        self._left = source
        self._shape, self._dtype, self._device = source.shape, source.dtype, source.device
        if _kind == 'transpose':
            self._shape = torch.Size((*source.shape[:-2], source.shape[-1], source.shape[-2]))
        elif _kind == 'softmax':
            if not source.entrywise:
                raise TypeError('softmax requires entrywise capability; materialize with .to_dense()')
        elif _kind == 'reference_center':
            mean = _right
            if (not isinstance(mean, torch.Tensor) or mean.layout != torch.strided or mean.ndim < 1
                    or mean.dtype != self.dtype or mean.device != self.device):
                raise ValueError('Reference mean must be a strided [...,M] Tensor matching dtype/device')
            torch._check(mean.shape[-1] == source.shape[-1], lambda: 'Reference mean must match reference axis')
            torch._check(source.shape[-1] > 0, lambda: 'Reference centering requires a nonempty reference axis')
            self._shape = torch.Size((*torch.broadcast_shapes(source.shape[:-2], mean.shape[:-1]), *source.shape[-2:]))
        elif _kind in ('add', 'sub', 'compose', 'hstack', 'vstack'):
            other = _right
            if not isinstance(other, LinearOperator):
                raise TypeError('Linear algebra requires LinearOperator operands; use tp.as_lazy for dense tensors')
            if other.dtype != self.dtype or other.device != self.device:
                raise ValueError('Linear operators must share dtype and device')
            if _kind == 'compose':
                torch._check(source.shape[-1] == other.shape[-2],
                             lambda: 'Linear composition contracted dimensions must match')
            elif _kind == 'hstack':
                torch._check(source.shape[-2] == other.shape[-2], lambda: 'Block row heights must agree')
            elif _kind == 'vstack':
                torch._check(source.shape[-1] == other.shape[-1], lambda: 'Block column widths must agree')
            else:
                torch._check(source.shape[-2] == other.shape[-2])
                torch._check(source.shape[-1] == other.shape[-1])
            rows = source.shape[-2] + other.shape[-2] if _kind == 'vstack' else source.shape[-2]
            columns = (source.shape[-1] + other.shape[-1] if _kind == 'hstack'
                       else other.shape[-1] if _kind == 'compose' else source.shape[-1])
            self._shape = torch.Size((*torch.broadcast_shapes(source.shape[:-2], other.shape[:-2]), rows, columns))
        elif _kind in ('scale', 'add_diagonal'):
            if _kind == 'add_diagonal':
                torch._check(source.shape[-2] == source.shape[-1],
                             lambda: 'Diagonal shifts require a square linear operator')
            if isinstance(_right, torch.Tensor):
                if (_right.layout != torch.strided or _right.ndim != 0
                        or _right.dtype != self.dtype or _right.device != self.device):
                    raise ValueError('Linear factors must be scalar tensors matching dtype/device')
            elif type(_right) not in (int, float):
                raise TypeError('Linear factors must be real numbers or scalar tensors; compose maps explicitly')

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def T(self):
        return self._transpose()

    def _transpose(self):
        return LinearOperator(self, _kind='transpose')

    def centered(self, reference_mean=None):
        """Center using this matrix's means, or a supplied training reference mean.

        Without an argument: H_N A H_M. With mean [...,M]: (A-1*mean) H_M,
        suitable for test-to-training kernels. Supply K_train.mean(dim=-2),
        using the same reference samples/order; test queries never set the mean.
        Reference mean batches broadcast, dtype/device match, M must be nonempty.
        The mean remains referenced and supports gradients; snapshot explicitly
        with detach().clone() for inference reuse. No centered matrix is stored.
        """
        return LinearOperator(self, _kind='center' if reference_mean is None else 'reference_center',
                              _right=reference_mean)

    def _compose(self, other):
        return LinearOperator(self, _kind='compose', _right=_as_linear(other))

    def _hstack(self, other):
        """Horizontal concatenation [self,other], combining split RHS actions."""
        return LinearOperator(self, _kind='hstack', _right=_as_linear(other))

    def _vstack(self, other):
        """Vertical concatenation [self;other], concatenating result blocks."""
        return LinearOperator(self, _kind='vstack', _right=_as_linear(other))

    def _scale(self, factor):
        return LinearOperator(self, _kind='scale', _right=factor)

    def _add_diagonal(self, factor):
        """A + factor*I for square A; this is not an entrywise scalar addition."""
        return LinearOperator(self, _kind='add_diagonal', _right=factor)

    @property
    def entrywise(self):
        return False

    def __neg__(self):
        return self._scale(-1.)

    def __matmul__(self, other):
        return self._compose(other) if isinstance(other, LazyPairwise) else self.matmul(other)

    def to_dense(self):
        if self.shape[-1] == 0:
            return torch.empty(self.shape, dtype=self.dtype, device=self.device)
        identity = torch.eye(self.shape[-1], dtype=self.dtype, device=self.device)
        return self._evaluate('matmul', right=identity)

    def reduce(self, reduction='sum', dim=None, *, keepdim=False, exclude_self=False, correction=None, mask=None):
        if mask is not None:
            raise TypeError('mask= needs entrywise capability; mask the entrywise operand before composing')
        if reduction not in ('sum', 'mean'):
            raise TypeError(f'{reduction} needs entrywise capability; materialize with .to_dense() or build the expression entrywise')
        if type(keepdim) is not bool or type(exclude_self) is not bool:
            raise TypeError('keepdim and exclude_self must be booleans')
        if exclude_self:
            raise TypeError('exclude_self needs entrywise capability; materialize with .to_dense()')
        if correction is not None:
            raise ValueError('correction is only used by var/std')
        return self._evaluate('reduce', reduction=reduction, dim=_scalar_dim(dim, self), keepdim=keepdim)

    def matmul(self, values):
        if isinstance(values, LazyPairwise):
            return self._compose(values)
        return self._evaluate('matmul', right=values)

    def _native_evaluate(self, operation, **options):
        if operation == 'matmul':
            return self._matmul(options['right'])
        if operation == 'reduce' and options['reduction'] in ('sum', 'mean'):
            dim, keepdim = options.get('dim'), options.get('keepdim', False)
            source = self.T if dim == -2 else self
            ones = torch.ones(source.shape[-1], dtype=self.dtype, device=self.device)
            result = source._evaluate('matmul', right=ones)
            if dim is None:
                result = result.sum(-1, keepdim=keepdim)
                if keepdim:
                    result = result.unsqueeze(-1)
            elif keepdim:
                result = result.unsqueeze(dim)
            if options['reduction'] == 'mean':
                count = self.shape[-2] * self.shape[-1] if dim is None else self.shape[dim]
                result = result / count
            return result
        raise TypeError(f'{operation} needs entrywise capability; materialize with .to_dense() or build the expression entrywise')

    def _apply(self, values, transpose=False):
        kind = self._kind
        if kind == 'lowrank_inverse':
            return torch.ops.torchpairwise.linear_lowrank_apply(values, *self._left)
        if kind == 'diagonal':
            return torch.ops.torchpairwise.linear_diagonal_apply(values, self._left)
        if kind == 'group':
            return torch.ops.torchpairwise.group_projection_apply(values, self._left, self._right)
        if kind == 'source':
            return (self._left.T if transpose else self._left)._evaluate('matmul', right=values)
        if kind == 'softmax':
            if not transpose:
                return self._left._evaluate('softmax_matmul', right=values, exclude_self=self._right)
            if not self._right:
                return _softmax_expression(self._left).T._evaluate('matmul', right=values)
            # The softmax primitive contracts only on the right and the scalar
            # bytecode cannot mask a diagonal. Read bounded column panels with
            # basis RHS blocks, then contract each transposed panel.
            width = 128
            columns = torch.arange(self.shape[-1], device=self.device)
            panels = []
            for start in range(0, self.shape[-1], width):
                stop = min(start + width, self.shape[-1])
                selected = torch.arange(start, stop, device=self.device)
                basis = (columns.unsqueeze(-1) == selected.unsqueeze(0)).to(self.dtype)
                panel = self._left._evaluate('softmax_matmul', right=basis, exclude_self=self._right)
                panels.append(panel.transpose(-1, -2) @ values)
            if not panels:
                batch = torch.broadcast_shapes(self.shape[:-2], values.shape[:-2])
                return values.new_zeros((*batch, 0, values.shape[-1]))
            return torch.cat(panels, dim=-2)
        if kind == 'transpose':
            return self._left._apply(values, not transpose)
        if kind == 'compose':
            inner = self._left if transpose else self._right
            outer = self._right if transpose else self._left
            return outer._apply(inner._apply(values, transpose), transpose)
        transform = torch.ops.torchpairwise.linear_transform
        if kind in ('hstack', 'vstack'):
            split = (kind == 'hstack') != transpose
            if split:
                width = self._left.shape[-2 if transpose else -1]
                return transform(self._left._apply(values.narrow(-2, 0, width), transpose), 'add',
                    self._right._apply(values.narrow(-2, width, values.shape[-2]-width), transpose))
            return transform(self._left._apply(values, transpose), 'vstack',
                             self._right._apply(values, transpose))
        if kind == 'center':
            return transform(self._left._apply(transform(values, 'center'), transpose), 'center')
        if kind == 'reference_center':
            rhs = values if transpose else transform(values, 'center')
            return torch.ops.torchpairwise.linear_reference_center(
                self._left._apply(rhs, transpose), rhs, self._right, transpose)
        value = self._left._apply(values, transpose)
        if kind in ('add', 'sub'):
            return transform(value, kind, self._right._apply(values, transpose))
        factor = self._right if isinstance(self._right, torch.Tensor) else torch.scalar_tensor(
            self._right, dtype=self.dtype, device=self.device)
        return transform(value, kind, values if kind == 'add_diagonal' else None, factor)

    def _matmul(self, values):
        """Apply to [M] or [...,M,R], preserving broadcast batches."""
        if not isinstance(values, torch.Tensor):
            raise TypeError('matmul expects a Tensor; use @ for another LazyPairwise')
        if (values.layout != torch.strided or values.ndim < 1
                or values.dtype != self.dtype or values.device != self.device):
            raise ValueError('Linear RHS must be dense and match operator dtype/device')
        vector = values.ndim == 1
        block = values.unsqueeze(-1) if vector else values
        torch._check(block.shape[-1] > 0, lambda: 'Linear RHS channel count must be positive')
        torch._check(block.shape[-2] == self.shape[-1], lambda: 'Linear RHS sample count mismatch')
        expected = (*torch.broadcast_shapes(self.shape[:-2], block.shape[:-2]),
                    self.shape[-2], block.shape[-1])
        result = self._apply(block)
        if result.ndim != len(expected):
            raise ValueError('Bound source shape changed; reconstruct the linear operator')
        for actual, wanted in zip(result.shape, expected):
            torch._check(actual == wanted, lambda: 'Bound source shape changed; reconstruct the linear operator')
        return result.squeeze(-1) if vector else result


def _as_linear(source):
    """Adapt a dense matrix or scalar source without evaluating its entries."""
    return source if isinstance(source, LinearOperator) else LinearOperator(source)


def _diagonal(values):
    """Create diag(values) from [...,N] float32/64 entries without a dense matrix.

    Leading batches broadcast with RHS blocks. Entries may be zero/negative,
    support gradients, and remain referenced: value mutations affect actions.
    Reconstruct the map after resizing entries. Use linalg.jacobi for a checked
    positive reciprocal diagonal inverse action.
    """
    return LinearOperator(values, _kind='diagonal')


def _block(blocks):
    """Assemble a rectangular grid of explicit linear-map blocks lazily.

    Each entry is a LinearOperator, scalar source or dense Tensor; rows/columns
    use consistent partitions and all entries share float32/64 dtype/device.
    Leading batches broadcast. None/scalar placeholders are not supported;
    supply a suitably shaped Tensor for small explicit zero blocks. No full
    generated matrix or padding is allocated. Actions split the RHS and combine
    block products; transpose, first gradients, compilation and native solvers
    use the same finite block graph. Empty sample partitions are allowed.
    """
    if (not isinstance(blocks, (list, tuple)) or not blocks
            or not isinstance(blocks[0], (list, tuple)) or not blocks[0]):
        raise ValueError('block requires a nonempty rectangular grid')
    width, rows, reference = len(blocks[0]), [], None
    for row in blocks:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError('block requires a rectangular grid')
        maps = [_as_linear(value) for value in row]
        if reference is None:
            reference = maps
        joined = maps[0]
        for j, value in enumerate(maps):
            torch._check(value.shape[-1] == reference[j].shape[-1], lambda: 'Block column widths must agree')
            if j:
                joined = joined._hstack(value)
        rows.append(joined)
    result = rows[0]
    for row in rows[1:]:
        result = result._vstack(row)
    return result


def _group_projection(labels, *, like):
    """Map [...,N,N] replacing sample values by their group mean.

    labels is strided int64 [...,N], with arbitrary signed IDs; batches broadcast
    with RHS blocks. like supplies only float32/float64 dtype and device. Labels
    must share that device. The partition is snapshotted at construction: later
    label mutations have no effect. No N-by-N matrix or one-hot table is stored.
    """
    if not isinstance(labels, torch.Tensor) or not isinstance(like, (torch.Tensor, LazyPairwise)):
        raise TypeError('Group labels must be a Tensor and like must be a Tensor or LazyPairwise')
    if labels.layout != torch.strided or labels.ndim < 1 or labels.dtype != torch.int64:
        raise ValueError('Group labels must be dense int64 with shape [...,N]')
    if (isinstance(like, torch.Tensor) and like.layout != torch.strided) or like.dtype not in (torch.float32, torch.float64):
        raise ValueError('Group projection like must be strided float32/float64')
    if labels.device != like.device:
        raise ValueError('Group labels and like must share a device')
    indices, counts = torch.ops.torchpairwise._group_partition(labels)
    return LinearOperator((indices, counts, like), _kind='group')


def _identity(reference):
    torch._check(reference.shape[-2] == reference.shape[-1], lambda: 'eye_like requires a square operator')
    result = _diagonal(torch.ones((*reference.shape[:-2], reference.shape[-1]),
                                  dtype=reference.dtype, device=reference.device))
    result._is_identity = True
    return result


def _identity_factor(value):
    if not isinstance(value, LinearOperator):
        return None
    if getattr(value, '_is_identity', False):
        return 1.
    if value._kind == 'scale':
        factor = _identity_factor(value._left)
        # Reuse the stored factor: a product computed here would be an
        # autograd node shared by, and freed after, the first backward.
        if factor is not None and type(factor) is float and factor == 1.:
            return value._right
        if factor is not None and not isinstance(factor, torch.Tensor) and not isinstance(value._right, torch.Tensor):
            return factor * value._right
    return None


def _linear_binary(operation, left, right):
    left_lazy, right_lazy = isinstance(left, LazyPairwise), isinstance(right, LazyPairwise)
    if operation == 'mul':
        if left_lazy and right_lazy:
            raise TypeError('Hadamard product needs two entrywise operands; use `@` to compose')
        source, factor = (left, right) if left_lazy else (right, left)
        return _as_linear(source)._scale(factor)
    if not left_lazy or not right_lazy:
        raise TypeError('Scalar addition needs entrywise capability; use c * tp.eye_like(A) for a diagonal shift or materialize with .to_dense()')
    factor = _identity_factor(right)
    if factor is not None:
        combined = LinearOperator(_as_linear(left), _kind=operation, _right=_as_linear(right))
        if combined.shape != left.shape or (operation == 'sub' and isinstance(factor, torch.Tensor)):
            return combined
        return _as_linear(left)._add_diagonal(factor if operation == 'add' else -factor)
    if operation == 'add':
        factor = _identity_factor(left)
        if factor is not None:
            combined = LinearOperator(_as_linear(left), _kind=operation, _right=_as_linear(right))
            if combined.shape != right.shape:
                return combined
            return _as_linear(right)._add_diagonal(factor)
    return LinearOperator(_as_linear(left), _kind=operation, _right=_as_linear(right))


def _softmax_expression(source):
    """exp(A - logsumexp(A, -1)) in the native scalar bytecode.

    Only the row log-normalizers are materialized. Fully -inf rows give NaN,
    as torch.softmax does.
    """
    shift = ScalarExpression('rows', 1., source.logsumexp(-1))
    return _SoftmaxExpression('exp', ScalarExpression('sub', source, shift), source=source)


def _softmax_solver_source(source, exclude_self):
    """Prepare a softmax map for native solver packets."""
    if exclude_self:
        raise TypeError('Native solver plans cannot mask a softmax diagonal with the existing bytecode; '
                        'use .to_dense() explicitly for a softmax node with exclude_self=True')
    if source.shape[-1] == 0:
        return _as_linear(source)
    return _as_linear(_softmax_expression(source))


# Row-weighted constants are already supported by the native scalar program
# (also used for weighted statistics). Carry the score source's matrix metadata
# because such a constant has no standalone N/M shape on the Python surface.
class _SoftmaxExpression(ScalarExpression):
    def __init__(self, operation, left, right=None, *, source):
        super().__init__(operation, left, right)
        self._score_source = source

    def _metadata(self):
        return self._score_source.shape, self._score_source.dtype, self._score_source.device
