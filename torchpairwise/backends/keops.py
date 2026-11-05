"""Restricted KeOps lowering for eager, unbatched scalar pairwise graphs."""
from functools import lru_cache
import importlib
import math

import torch

# Calibrated on a Quadro RTX 6000, pykeops 2.3 (develop/step3/calibrate.py):
# float32 KeOps wins from ~2**24 pairs at D<=16 and everywhere at N=M=1e4,
# D<=64; float64 KeOps is slower except for large softmax products.
AUTO_MIN_PAIRS = 2**25
AUTO_MAX_DIM = 64
_stats = {'calls': 0, 'formula': None}


@lru_cache(maxsize=1)
def _load():
    try:
        return importlib.import_module('pykeops.torch').LazyTensor
    except ImportError:
        return None


def available():
    """Whether pykeops can be imported (the import result is cached)."""
    return _load() is not None


class _Unsupported(Exception):
    pass


def _require(condition):
    if not condition:
        raise _Unsupported


class _Formula:
    def __init__(self, source, auto):
        from ..ops._pairwise_operator import PairwiseOperator
        from ..ops._operator_algebra import ScalarExpression
        self.leaf_type, self.expression_type = PairwiseOperator, ScalarExpression
        self.dtype, self.device = source.dtype, source.device
        _require(self.dtype in (torch.float32, torch.float64))
        _require(self.device.type in ('cpu', 'cuda'))
        _require(not torch.is_autocast_enabled(self.device.type))
        _require(len(source.shape) == 2 and all(s > 0 for s in source.shape))
        self.shape, self.auto = source.shape, auto
        self.nonfinite = False
        self.LazyTensor, self.tensors, self.forward_only = _load(), [], False

    def tensor(self, value):
        _require(isinstance(value, torch.Tensor) and value.layout == torch.strided)
        _require(value.device == self.device)
        return value.to(dtype=self.dtype).contiguous()

    def pm(self, value):
        if isinstance(value, torch.Tensor):
            _require(value.ndim == 0 and not value.is_complex())
            self.tensors.append(value)
            value = self.tensor(value)
        else:
            _require(type(value) in (int, float, bool))
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        return self.LazyTensor(value.reshape(1), axis=2)

    def axis(self, value, axis):
        value = self.tensor(value)
        return self.LazyTensor(value, axis=axis)

    def minimum(self, value, bound):
        return (value - bound).ifelse(bound, value)

    def power(self, base, exponent):
        # PyKeOps 2.3 Pow/Powf/Rsqrt' lose float64 accuracy (~1e-6) and Pow
        # returns NaN for negative bases; Exp/Log/Sqrt/Square are exact.
        return (base.log() * exponent).exp()

    def integer_power(self, base, degree):
        result, square = None, base
        while degree:
            if degree & 1:
                result = square if result is None else result * square
            degree >>= 1
            if degree:
                square = square * square
        return result

    def integer_operand(self, value):
        if isinstance(value, self.expression_type):
            if value._operation in ('row_index', 'column_index'):
                return True
            if value._operation in ('row_values', 'column_values'):
                return value._right.dtype in (torch.uint8, torch.int8, torch.int16,
                                               torch.int32, torch.int64)
            # Differences of exact integers keep their sign under rounding, so
            # half-integer margins stay exact (e.g. causal/band: j - i - offset).
            if value._operation in ('add', 'sub'):
                return self.integer_operand(value._left) and self.integer_operand(value._right)
            if value._operation in ('neg', 'identity'):
                return self.integer_operand(value._left)
            return False
        if isinstance(value, torch.Tensor):
            return value.ndim == 0 and value.dtype in (torch.uint8, torch.int8, torch.int16,
                                                       torch.int32, torch.int64)
        return type(value) in (int, bool)

    def is_margin(self, value):
        if not isinstance(value, self.expression_type):
            return False
        op = value._operation
        if op in ('lt', 'le', 'gt', 'ge', 'eq', 'ne'):
            return self.integer_operand(value._left) and self.integer_operand(value._right)
        if op in ('identity', 'transpose', 'logical_not'):
            return self.is_margin(value._left)
        if op in ('logical_and', 'logical_or'):
            return self.is_margin(value._left) and self.is_margin(value._right)
        return False

    def visit(self, value, transpose=False):
        if isinstance(value, self.expression_type):
            op = value._operation
            if op == 'transpose':
                return self.visit(value._left, not transpose)
            if op in ('row_values', 'column_values', 'row_index', 'column_index'):
                axis = (0 if op in ('row_values', 'row_index') else 1) ^ transpose
                if op.endswith('index'):
                    _require(self.shape[axis] < (2**24 if self.dtype == torch.float32 else 2**53))
                    tensor = torch.arange(self.shape[axis], device=self.device, dtype=self.dtype)
                else:
                    tensor = value._right
                    _require(isinstance(tensor, torch.Tensor) and tensor.ndim == 1)
                    _require(tensor.shape[0] == self.shape[axis])
                    if self.integer_operand(value) and tensor.numel():
                        limit = 2**24 if self.dtype == torch.float32 else 2**53
                        _require(bool((tensor.to(torch.float64).abs().max() < limit).item()))
                    self.tensors.append(tensor)
                return self.axis(tensor[:, None], axis)
            if op == 'where':
                _require(self.is_margin(value._left))
                self.forward_only = True
                cond = self.visit(value._left, transpose)
                a, b = value._right
                return cond.ifelse(self.visit(a, transpose), self.visit(b, transpose))
            _require(op in ('identity', 'add', 'sub', 'mul', 'div', 'pow',
                            'minimum', 'maximum', 'clamp_min', 'clamp_max',
                            'neg', 'square', 'exp', 'sqrt', 'log', 'sin', 'cos',
                            'tanh', 'sigmoid', 'abs', 'logical_not', 'lt', 'le',
                            'gt', 'ge', 'eq', 'ne', 'logical_and', 'logical_or',
                            'rows', 'columns'))
            left = self.visit(value._left, transpose)
            if op == 'identity':
                return left
            if op in ('rows', 'columns'):
                weights = value._right
                axis = (0 if op == 'rows' else 1) ^ transpose
                _require(isinstance(weights, torch.Tensor) and weights.ndim == 1)
                _require(weights.shape[0] == self.shape[axis])
                self.tensors.append(weights)
                return left * self.axis(weights[:, None], axis)
            if op == 'neg':
                return -left
            if op == 'square':
                return left * left
            if op == 'logical_not':
                _require(self.is_margin(value._left))
                return -left
            if op == 'sigmoid':
                return self.pm(1.) / (self.pm(1.) + (-left).exp())
            if op in ('exp', 'sqrt', 'log', 'sin', 'cos', 'tanh', 'abs'):
                _require(hasattr(left, op))
                return getattr(left, op)()
            if op == 'pow':
                exponent = value._right
                _require(not isinstance(exponent, self.expression_type) and
                         not isinstance(exponent, self.leaf_type))
                if isinstance(exponent, torch.Tensor):
                    _require(exponent.ndim == 0 and not exponent.requires_grad)
                    raw = exponent.item()
                else:
                    raw = exponent
                _require(type(raw) in (int, float) and math.isfinite(raw))
                if raw == 0:
                    self.forward_only = True
                    rows = torch.zeros((self.shape[0], 1), dtype=self.dtype, device=self.device)
                    columns = torch.zeros((self.shape[1], 1), dtype=self.dtype, device=self.device)
                    return self.pm(1.) + self.axis(rows, 0) + self.axis(columns, 1)
                if float(raw).is_integer():
                    result = self.integer_power(left, abs(int(raw)))
                    return result if raw > 0 else self.pm(1.) / result
                self.forward_only = True
                return self.power(left, self.visit(exponent, transpose))
            if op in ('minimum', 'maximum'):
                _require(not (isinstance(value._left, (self.expression_type, self.leaf_type)) and
                              isinstance(value._right, (self.expression_type, self.leaf_type))))
                self.forward_only = True
                right = self.visit(value._right, transpose)
                return self.minimum(left, right) if op == 'minimum' else -self.minimum(-left, -right)
            if op in ('lt', 'le', 'gt', 'ge', 'eq', 'ne'):
                _require(self.integer_operand(value._left) and self.integer_operand(value._right))
            right = self.visit(value._right, transpose)
            if op == 'add':
                return left + right
            if op == 'sub':
                return left - right
            if op == 'mul':
                return left * right
            if op == 'div':
                return left / right
            if op in ('clamp_min', 'clamp_max'):
                self.forward_only = True
                return (left - right).ifelse(left, right) if op == 'clamp_min' else (right - left).ifelse(left, right)
            if op in ('logical_and', 'logical_or'):
                _require(self.is_margin(value._left) and self.is_margin(value._right))
                return self.minimum(left, right) if op == 'logical_and' else -self.minimum(-left, -right)
            half = self.pm(.5)
            if op == 'eq':
                return half - (left - right).abs()
            if op == 'ne':
                return (left - right).abs() - half
            if op in ('gt', 'ge'):
                left, right = right, left
            return right - left + (half if op in ('le', 'ge') else -half)
        # Broadcast customs that opted into KeOps lower to their own formula.
        from ..ops._custom import CustomPairwiseOperator
        if type(value) is CustomPairwiseOperator and value.comparison.keops:
            return self.custom(value, transpose)
        if not isinstance(value, self.leaf_type):
            if isinstance(value, torch.Tensor):
                finite = value.ndim != 0 or not value.is_floating_point() or bool(value.detach().isfinite())
            else:
                finite = type(value) is not float or math.isfinite(value)
            self.nonfinite |= not finite
            if self.integer_operand(value):
                raw = value.item() if isinstance(value, torch.Tensor) else value
                _require(abs(raw) < (2**24 if self.dtype == torch.float32 else 2**53))
            return self.pm(value)
        # Custom/feature subclasses must retain their own execution contract.
        from ..ops._kernel_operator import KernelOperator
        _require(type(value) in (self.leaf_type, KernelOperator))
        return self.leaf(value, transpose)

    def custom(self, leaf, transpose):
        c = leaf.comparison
        x, y = leaf.x1, leaf.x1 if leaf.x2 is None else leaf.x2
        _require(c.output_dtype in (None, self.dtype))
        _require(all(isinstance(t, torch.Tensor) and t.ndim == 2 and t.shape[0] > 0 and t.shape[1] > 0 and
                     t.dtype == self.dtype and t.device == self.device for t in (x, y)))
        _require(not self.auto or (self.device.type == 'cuda' and self.dtype == torch.float32 and
                 x.shape[0] * y.shape[0] >= AUTO_MIN_PAIRS and max(x.shape[1], y.shape[1]) <= AUTO_MAX_DIM))
        # A transposed leaf keeps fn's argument order; only the KeOps axes swap.
        flipped = leaf._transposed != transpose
        _require(((y.shape[0], x.shape[0]) if flipped else (x.shape[0], y.shape[0])) == tuple(self.shape))
        _require(all(p.dtype == self.dtype and p.device == self.device and p.ndim <= 1 and p.numel() > 0
                     for p in c.parameters))
        self.tensors.extend((x, y, *c.parameters))
        X, Y = self.axis(x, 1 if flipped else 0), self.axis(y, 0 if flipped else 1)
        parameters = [self.LazyTensor(self.tensor(p).reshape(-1), axis=2) for p in c.parameters]
        try:
            result = c._fn(X, Y, *parameters)
        except Exception as exc:
            raise _Unsupported from exc
        _require(isinstance(result, self.LazyTensor) and result.ndim == 1)
        return result

    def leaf(self, leaf, transpose):
        name, params = leaf.metric, leaf.metric_kwargs
        allowed = {
            'sqeuclidean_distances': (), 'euclidean_distances': (),
            'manhattan_distances': (), 'chebyshev_distances': (),
            'minkowski_distances': ('p',), 'cosine_distances': (),
            'cosine_similarity': (), 'linear_kernel': (),
            'polynomial_kernel': ('degree', 'gamma', 'coef0'),
            'sigmoid_kernel': ('gamma', 'coef0'), 'rbf_kernel': ('gamma',),
            'laplacian_kernel': ('gamma',),
            'rational_quadratic_kernel': ('alpha', 'length_scale'),
            'matern_kernel': ('nu', 'length_scale'),
        }
        _require(name in allowed)
        _require(all(v is None or k in allowed[name] for k, v in params.items()))
        x, y = leaf.x1, leaf.x1 if leaf.x2 is None else leaf.x2
        _require(all(isinstance(t, torch.Tensor) and t.ndim == 2 and
                     t.dtype == self.dtype and t.device == self.device for t in (x, y)))
        _require(x.shape[1] == y.shape[1] and x.shape[1] > 0)
        _require(x.shape[0] > 0 and y.shape[0] > 0)
        _require(not self.auto or (self.device.type == 'cuda' and self.dtype == torch.float32 and
                 x.shape[0] * y.shape[0] >= AUTO_MIN_PAIRS and x.shape[1] <= AUTO_MAX_DIM))
        same = leaf.x2 is None or leaf.x2 is leaf.x1
        if leaf._transposed != transpose:
            x, y = y, x
        _require((x.shape[0], y.shape[0]) == tuple(self.shape))
        self.tensors.extend((x, y))
        X, Y = self.axis(x, 0), self.axis(y, 1)
        delta = X - Y
        d2 = (delta * delta).sum(-1)

        def param(key, default):
            v = params.get(key)
            return default if v is None else v

        def scalar(key, default, positive=False):
            v = param(key, default)
            raw = v.detach().item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v
            _require(type(raw) in (float, int) and math.isfinite(raw))
            _require(not positive or raw > 0)
            return self.pm(v)

        if name == 'minkowski_distances':
            p = param('p', 2.)
            fixed = not isinstance(p, torch.Tensor) or (p.ndim == 0 and not p.requires_grad)
            if fixed:
                power_value = p.item() if isinstance(p, torch.Tensor) else p
                if power_value == 1:
                    name = 'manhattan_distances'
                elif power_value == 2:
                    name = 'euclidean_distances'
                elif power_value == math.inf:
                    name = 'chebyshev_distances'
            if name == 'minkowski_distances':
                self.forward_only = True
                power = scalar('p', 2., positive=True)
                return self.power(self.power(delta.abs(), power).sum(-1), self.pm(1.) / power)
        if name == 'sqeuclidean_distances':
            return d2
        if name == 'euclidean_distances':
            return d2.sqrt()
        if name == 'manhattan_distances':
            return delta.abs().sum(-1)
        if name == 'chebyshev_distances':
            self.forward_only = True
            return delta.abs().max(-1)
        dot = (X * Y).sum(-1)
        if name in ('cosine_similarity', 'cosine_distances'):
            zero, one = self.pm(0.), self.pm(1.)
            norms = (X * X).sum(-1).sqrt() * (Y * Y).sum(-1).sqrt()
            # Zero-norm rows give value and gradient zero, as natively.
            empty = -norms
            cosine = dot / empty.ifelse(one, norms) * empty.ifelse(zero, one)
            if name == 'cosine_similarity':
                return cosine
            two = self.pm(2.)
            distance = one - cosine
            distance = self.minimum((zero - distance).ifelse(zero, distance), two)
            if same:
                i = self.axis(torch.arange(x.shape[0], device=self.device, dtype=self.dtype)[:, None], 0)
                j = self.axis(torch.arange(y.shape[0], device=self.device, dtype=self.dtype)[:, None], 1)
                distance = ((i - j).abs() - self.pm(.5)).ifelse(distance, zero)
            return distance
        if name == 'linear_kernel':
            return dot
        if name in ('polynomial_kernel', 'sigmoid_kernel', 'rbf_kernel', 'laplacian_kernel'):
            gamma = scalar('gamma', 1. / x.shape[1])
            if name == 'rbf_kernel':
                return (-gamma * d2).exp()
            if name == 'laplacian_kernel':
                return (-gamma * delta.abs().sum(-1)).exp()
            affine = gamma * dot + scalar('coef0', 1.)
            if name == 'sigmoid_kernel':
                _require(hasattr(affine, 'tanh'))
                return affine.tanh()
            degree = param('degree', 3)
            _require(type(degree) is int and degree >= 0)
            if degree == 0:
                return self.pm(1.) + affine * self.pm(0.)
            return self.integer_power(affine, degree)
        scale = scalar('length_scale', 1., positive=True)
        if name == 'rational_quadratic_kernel':
            alpha = scalar('alpha', 1., positive=True)
            return self.power(self.pm(1.) + d2 / (self.pm(2.) * alpha * scale * scale), -alpha)
        nu = param('nu', 1.5)
        _require(type(nu) in (float, int) and nu in (.5, 1.5, 2.5))
        r = d2.sqrt() / scale * self.pm(math.sqrt(2 * nu))
        r = self.minimum(r, self.pm(-4 * math.log(torch.finfo(self.dtype).tiny)))
        polynomial = self.pm(1.) if nu == .5 else self.pm(1.) + r
        if nu == 2.5:
            polynomial = polynomial + r * r / self.pm(3.)
        return polynomial * (-r).exp()


def _lower(source, operation, options, *, auto=False):
    try:
        return _evaluate(source, operation, options, auto)
    except _Unsupported:
        return NotImplemented


def _evaluate(source, operation, options, auto):
    common = {'exclude_self', '_return_tuple'}
    if operation in ('matmul', 'softmax_matmul'):
        allowed = common | {'right'}
    elif operation == 'reduce':
        allowed = common | {'reduction', 'dim', 'keepdim', 'k', 'largest', 'sorted', 'correction'}
    else:
        return NotImplemented
    _require(not (set(options) - allowed) and not options.get('exclude_self', False))
    reduction, dim, k = options.get('reduction', 'sum'), options.get('dim'), options.get('k')
    if operation == 'reduce':
        _require(dim in (None, -1, -2))
        if k is not None:
            _require(not auto and type(k) is int and 0 < k <= 64 and dim is not None)
            _require('reduction' not in options)
            _require(not options.get('keepdim', False))
        else:
            _require(not ({'largest', 'sorted'} & set(options)))
            _require(reduction in ('sum', 'mean', 'logsumexp', 'min', 'max'))
            _require(not auto or reduction in ('sum', 'mean', 'logsumexp'))
            _require(dim is not None or reduction in ('sum', 'mean'))
        _require('correction' not in options or reduction == 'logsumexp')
    formula = _Formula(source, auto)
    _require(formula.LazyTensor is not None)
    F = formula.visit(source)
    n, m = formula.shape
    # KeOps' online max-shift exponential sums yield NaN when a row's leading
    # entries are -inf (masked fills), so those formulas stay native.
    _require(not formula.nonfinite or not (operation == 'softmax_matmul' or reduction == 'logsumexp'))
    if formula.forward_only and torch.is_grad_enabled():
        _require(not any(t.requires_grad for t in formula.tensors))
    if operation in ('matmul', 'softmax_matmul'):
        right = options.get('right')
        _require(isinstance(right, torch.Tensor) and right.ndim in (1, 2))
        _require(right.shape[0] == m and right.dtype == formula.dtype and right.device == formula.device)
        _require(right.ndim == 1 or right.shape[1] > 0)
        vector = right.ndim == 1
        V = formula.axis(right[:, None] if vector else right, 1)
    if operation == 'reduce' and k is not None:
        _require(k <= formula.shape[dim])
    _stats['formula'] = str(F)
    _stats['calls'] += 1
    if operation in ('matmul', 'softmax_matmul'):
        result = (F * V).sum(1) if operation == 'matmul' else F.sumsoftmaxweight(V, 1)
        result = result.squeeze(-1) if vector else result
    else:
        axis = 1 if dim in (None, -1) else 0
        if k is not None:
            largest = options.get('largest', True)
            values, indices = (-F if largest else F).Kmin_argKmin(k, axis)
            if largest:
                values = -values
            if dim == -2:
                values, indices = values.T, indices.T
            result = (values, indices.to(torch.int64))
        else:
            result = getattr(F, 'sum' if reduction == 'mean' else reduction)(axis)
            if dim is None:
                result = result.sum().reshape((1, 1) if options.get('keepdim') else ())
            elif options.get('keepdim'):
                result = result.reshape((n, 1) if dim == -1 else (1, m))
            else:
                result = result.reshape(n if dim == -1 else m)
            if reduction == 'mean':
                result = result / (n * m if dim is None else formula.shape[dim])
    if options.get('_return_tuple'):
        return result if isinstance(result, tuple) else (result, torch.empty(0, dtype=torch.int64, device=formula.device))
    return result[0] if isinstance(result, tuple) else result


__all__ = ['available']
