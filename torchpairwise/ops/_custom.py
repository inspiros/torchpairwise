"""Experimental scalar and broadcast-form custom pairwise functions."""
import math
import itertools

import torch
from ._validation import _check_chunk_size
from ._pairwise_operator import _ScalarOperator


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_DEFAULT_CHUNK_SIZE = 1024
_RECOMPILE_LIMIT = 64


class _PairwiseTile(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y, parameters):
        return self.fn(x, y, parameters)


class _PairwiseVjp(_PairwiseTile):
    def forward(self, gradient, x, y, parameters):
        return self.fn(gradient, x, y, parameters)


def _compile_pairs(fn, *, backward=False):
    module = _PairwiseVjp(fn) if backward else _PairwiseTile(fn)
    compiled = torch.compile(module, fullgraph=True, dynamic=True)

    def evaluate(*operands):
        *inputs, parameters = operands
        parameters = list(parameters)
        # Cartesian edge tiles can be expanded views (including zero strides
        # on singleton axes), whereas dense gathers are contiguous. Normalize
        # this metadata before compilation to avoid a graph per tile layout.
        if not torch.is_grad_enabled():
            inputs = [value.detach() for value in inputs]
            parameters = [value.detach() for value in parameters]
        inputs = [value.contiguous().view(-1).view(value.shape) for value in inputs]
        # Dynamic shapes still specialize size-1 axes. Broadcast tiles carry
        # singleton axes and edge tiles may be one wide, so forward/vjp, grad
        # mode and edge shapes give a bounded set of graphs above the default limit.
        with torch._dynamo.config.patch(recompile_limit=_RECOMPILE_LIMIT):
            return compiled(*inputs, parameters)

    return evaluate


def _make_dense_pairwise(fn, *, mode='all_pairs', sample_ndim=1, chunk_size=None,
                  compile=False, symmetric=False, output_dtype=None, _evaluate_pairs=None,
                  form='scalar'):
    """Adapt ``fn(sample1, sample2) -> scalar Tensor`` to collections of samples.

    Inputs have shape ``[*batch, N, *sample]`` with ``sample_ndim`` sample axes.
    Modes are ``all_pairs`` (optional second input), ``paired`` (required second
    input), and ``condensed`` (one input and explicit ``symmetric=True``).
    Leading batch axes broadcast; input dtypes must match. The function owns
    compatibility checks for the two sample shapes, which may differ.

    ``chunk_size`` bounds evaluated pairs across all batches (default 1024).
    ``compile=True`` compiles the reusable vectorized tile with fullgraph=True.
    The scalar must have the input device and ``output_dtype`` (input dtype by
    default). Empty outputs follow this contract without invoking ``fn``.
    See docs/custom-metrics.md for autograd, autocast and callable restrictions.
    """
    if not callable(fn):
        raise TypeError('fn must be callable')
    if mode not in ('all_pairs', 'paired', 'condensed'):
        raise ValueError("mode must be 'all_pairs', 'paired', or 'condensed'")
    if type(sample_ndim) is not int or sample_ndim < 1:
        raise ValueError('sample_ndim must be a positive integer')
    _check_chunk_size(chunk_size, allow_none=True)
    if type(compile) is not bool or type(symmetric) is not bool:
        raise TypeError('compile and symmetric must be booleans')
    if mode == 'condensed' and not symmetric:
        raise ValueError('condensed mode requires explicit symmetric=True')
    if output_dtype is not None and output_dtype not in _FLOAT_DTYPES:
        raise TypeError('output_dtype must be a supported real floating point dtype')
    size = _DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size

    def checked_metric(a, b):
        result = fn(a, b)
        if not isinstance(result, torch.Tensor):
            raise TypeError('fn must return a scalar Tensor')
        if result.ndim != 0:
            raise ValueError('fn must return a scalar Tensor (zero dimensions)')
        if result.layout != torch.strided:
            raise TypeError('fn must return a dense strided scalar Tensor')
        if result.device != a.device:
            raise ValueError('fn must return a scalar on the input device')
        if result.dtype != (a.dtype if output_dtype is None else output_dtype):
            raise TypeError('fn scalar dtype must match output_dtype (input dtype by default)')
        return result

    mapped = torch.vmap(checked_metric, randomness='error') if form == 'scalar' else None

    def evaluate_tile(x, y, coordinates, i, j):
        # Compile the gathers and vectorized function together, so fusion can
        # eliminate pair-feature copies when the compiler supports the metric.
        a, b = x[coordinates + (i,)], y[coordinates + (j,)]
        return mapped(a, b) if _evaluate_pairs is None else _evaluate_pairs(a, b)

    evaluate = torch.compile(evaluate_tile, fullgraph=True) if compile else evaluate_tile

    def pairwise(x1, x2=None):
        if mode == 'paired' and x2 is None:
            raise TypeError('paired mode requires both input tensors')
        if mode == 'condensed' and x2 is not None:
            raise TypeError('condensed mode accepts one input tensor')
        x2 = x1 if x2 is None else x2
        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise TypeError('inputs must be tensors')
        rank = sample_ndim + 1
        if x1.ndim < rank or x2.ndim < rank:
            raise ValueError('inputs must have a sample-count axis and sample_ndim sample axes')
        if x1.layout != torch.strided or x2.layout != torch.strided:
            raise TypeError('inputs must be dense strided tensors')
        if x1.dtype not in _FLOAT_DTYPES or x2.dtype not in _FLOAT_DTYPES:
            raise TypeError('inputs must use a supported real floating point dtype')
        if x1.dtype != x2.dtype or x1.device != x2.device:
            raise ValueError('inputs must have the same dtype and device')
        n, m = x1.shape[-rank], x2.shape[-rank]
        if mode == 'paired' and n != m:
            raise ValueError('paired inputs must have equal sample counts')
        try:
            batch = torch.broadcast_shapes(x1.shape[:-rank], x2.shape[:-rank])
        except RuntimeError as exc:
            raise ValueError('input batch dimensions are not broadcastable') from exc
        if mode == 'all_pairs':
            per_batch, tail = n * m, (n, m)
        elif mode == 'paired':
            per_batch, tail = n, (n,)
        else:
            per_batch = n * (n - 1) // 2
            tail = (per_batch,)
        shape = (*batch, *tail)
        total = math.prod(batch) * per_batch
        dtype = x1.dtype if output_dtype is None else output_dtype
        if total == 0:
            # Empty slices read no values, even if singleton inputs contain NaN.
            # Captured tensors are intentionally unused: fn is never probed.
            zero = x1.narrow(0, 0, 0).sum().to(dtype) + x2.narrow(0, 0, 0).sum().to(dtype)
            return x1.new_empty(shape, dtype=dtype) + zero

        x = x1.expand(*batch, *x1.shape[-rank:])
        y = x2.expand(*batch, *x2.shape[-rank:])
        if form == 'broadcast' and mode == 'all_pairs':
            # Slice each collection independently: never gather Cartesian
            # pair-feature operands. Batch axes stay leading tile axes, so a
            # block holds at most size pairs unless the batch alone exceeds it.
            budget = max(1, size // math.prod(batch))
            rows = min(n, max(1, math.isqrt(budget)))
            columns = min(m, max(1, budget // rows))
            blocks = []
            for i in range(0, n, rows):
                a = x.narrow(-rank, i, min(rows, n - i)).unsqueeze(-rank)
                blocks.append(torch.cat([
                    _evaluate_pairs(a, y.narrow(-rank, j, min(columns, m - j)).unsqueeze(-rank - 1))
                    for j in range(0, m, columns)
                ], dim=-1))
            return torch.cat(blocks, dim=-2)
        row_starts = None
        if mode == 'condensed':
            # O(N) integer metadata, never a square result or global K-by-sample
            # gather. Halve the even factor first to avoid intermediate overflow.
            r = torch.arange(n, device=x.device, dtype=torch.int64)
            factor = 2 * n - r - 1
            row_starts = (r // 2) * factor + (r % 2) * (factor // 2)

        parts = []
        for start in range(0, total, size):
            flat = torch.arange(start, min(start + size, total), device=x.device,
                                dtype=torch.int64)
            pair = flat % per_batch
            if mode == 'all_pairs':
                i, j = pair // m, pair % m
            elif mode == 'paired':
                i = j = pair
            else:
                i = torch.searchsorted(row_starts, pair, right=True) - 1
                j = pair - row_starts[i] + i + 1
            # Preserve broadcast/noncontiguous inputs as views. Flattening an
            # expanded batch could otherwise copy the whole input collection.
            remainder = flat // per_batch
            coordinates = []
            for length in reversed(batch):
                coordinates.append(remainder % length)
                remainder = remainder // length
            coordinates = tuple(reversed(coordinates))
            parts.append(evaluate(x, y, coordinates, i, j))
        return torch.cat(parts).reshape(shape)

    return pairwise


_NAMES = itertools.count()
_LIBRARIES = []  # Keep registrations alive for saved autograd/export graphs.
_DTYPE_CODES = {torch.float16: 5, torch.bfloat16: 15, torch.float32: 6, torch.float64: 7}
_MODES = ('all_pairs', 'paired', 'condensed', 'single')


def _check_mode(mode, symmetric):
    if mode not in _MODES:
        raise ValueError("mode must be 'all_pairs', 'paired', 'condensed', or 'single'")
    if mode == 'condensed' and not symmetric:
        raise ValueError('condensed mode requires explicit symmetric=True')
    return mode


def _check_parameters(parameters, like=None):
    if not isinstance(parameters, (tuple, list)) or any(not isinstance(p, torch.Tensor) for p in parameters):
        raise TypeError('parameters must be a tuple/list of tensors')
    for p in parameters:
        if p.layout != torch.strided or p.dtype not in _FLOAT_DTYPES:
            raise TypeError('parameters must be dense real floating point tensors')
        if like is not None and (p.dtype != like.dtype or p.device != like.device):
            raise ValueError('parameters must have the input dtype and device')
    return tuple(parameters)


def _with_vjp(forward_tile, backward_tile):
    class UserComparisonFunction(torch.autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def forward(x, y, *parameters):
            return forward_tile(x, y, parameters)

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)

        @staticmethod
        def backward(ctx, gradient):
            inputs = ctx.saved_tensors
            gradients = backward_tile(gradient, inputs[0], inputs[1], inputs[2:])
            if not isinstance(gradients, (tuple, list)) or len(gradients) != len(inputs):
                raise ValueError('backward must return one gradient per input and explicit parameter')
            return tuple(gradients)

    return UserComparisonFunction.apply


def _captured_trainable(fn):
    from functools import partial
    found, seen = [], set()
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                found.append(value)
        elif isinstance(value, torch.nn.Module):
            found.extend(p for p in value.parameters() if p.requires_grad)
        elif isinstance(value, partial):
            visit(value.func)
            visit(value.args)
            visit(value.keywords)
        elif isinstance(value, (tuple, list, dict)):
            for part in value.values() if isinstance(value, dict) else value:
                visit(part)
    for cell in getattr(fn, '__closure__', None) or ():
        visit(cell.cell_contents)
    for key in getattr(getattr(fn, '__code__', None), 'co_names', ()):
        visit(getattr(fn, '__globals__', {}).get(key))
    visit(getattr(fn, '__self__', None))
    visit(fn)
    return found


class CustomComparison:
    """User comparison with four output modes and a native tiled source.

    Construct once with make_pairwise. Callbacks must be deterministic for
    streamed replay. Explicit parameters are shared tensors, not sample batches.
    Python callable registrations depend on the user's Python environment;
    registered native paired-tile operation names are usable in LibTorch.
    """
    def __init__(self, fn, *, backward, parameters, mode, sample_ndim,
                 chunk_size, compile, symmetric, output_dtype, name, form, keops=False):
        self.mode, self.sample_ndim = mode, sample_ndim
        self.form, self.keops = form, keops
        self.chunk_size, self.symmetric, self.output_dtype = chunk_size, symmetric, output_dtype
        if isinstance(fn, torch.nn.Module):
            if _check_parameters(parameters):
                raise TypeError('nn.Module parameters are bound automatically; omit parameters=')
            module = fn
            names = tuple(name for name, _ in module.named_parameters())
            parameters = tuple(module.parameters())
            parameter_ids = {id(p) for p in parameters}
            hidden_forward = any(id(p) not in parameter_ids
                                 for p in _captured_trainable(module.forward))

            def fn(x, y, *parameters):
                return torch.func.functional_call(module, dict(zip(names, parameters)), (x, y))
        else:
            hidden_forward = bool(_captured_trainable(fn))
        self.parameters = _check_parameters(parameters)
        self._fn, self._backward = fn, backward
        self._hidden_trainable = hidden_forward or bool(_captured_trainable(backward))
        self._native = isinstance(fn, str)
        self._dense_cache = {}
        if self._native:
            if '::' not in fn or (backward is not None and (not isinstance(backward, str) or '::' not in backward)):
                raise ValueError('native forward/backward names must be fully qualified operator names')
            self.forward_op, self.backward_op = fn, backward or ''
            return
        metric = fn
        def checked_metric(a, b, *parameters):
            result = metric(a, b, *parameters)
            if form == 'broadcast':
                if not isinstance(result, torch.Tensor):
                    raise TypeError('fn must return a broadcast Tensor')
                shape = torch.broadcast_shapes(a.shape[:-sample_ndim], b.shape[:-sample_ndim])
                if result.shape != shape:
                    raise ValueError('fn broadcast output shape must match the broadcast leading axes')
                if result.layout != torch.strided:
                    raise TypeError('fn must return a dense strided broadcast Tensor')
                if result.device != a.device:
                    raise ValueError('fn must return a broadcast Tensor on the input device')
                if result.dtype != (a.dtype if output_dtype is None else output_dtype):
                    raise TypeError('fn broadcast dtype must match output_dtype (input dtype by default)')
                return result
            if not isinstance(result, torch.Tensor):
                raise TypeError('fn must return a scalar Tensor')
            if result.ndim != 0:
                raise ValueError('fn must return a scalar Tensor (zero dimensions)')
            if result.layout != torch.strided:
                raise TypeError('fn must return a dense strided scalar Tensor')
            if result.device != a.device:
                raise ValueError('fn must return a scalar on the input device')
            if result.dtype != (a.dtype if output_dtype is None else output_dtype):
                raise TypeError('fn scalar dtype must match output_dtype (input dtype by default)')
            return result
        mapped = (torch.vmap(checked_metric, in_dims=(0, 0, *(None for _ in self.parameters)),
                             randomness='error') if form == 'scalar' else checked_metric)
        def tile(x, y, parameters):
            return mapped(x, y, *parameters)
        evaluate = _compile_pairs(tile) if compile else tile
        if backward is not None:
            def checked_vjp(g, x, y, *parameters):
                gradients = backward(g, x, y, *parameters)
                if not isinstance(gradients, (tuple, list)) or len(gradients) != 2 + len(parameters):
                    raise ValueError('backward must return one gradient per input and explicit parameter')
                for value, original in zip(gradients, (x, y, *parameters)):
                    if not isinstance(value, torch.Tensor):
                        raise TypeError('backward gradients must be Tensors; use zeros for unused operands')
                    if value.shape != original.shape or value.dtype != original.dtype or value.device != original.device:
                        raise ValueError('backward gradients must match each operand shape, dtype and device')
                return tuple(gradients)
            mapped_vjp = (torch.vmap(checked_vjp, in_dims=(0, 0, 0, *(None for _ in self.parameters)),
                                     randomness='error') if form == 'scalar' else checked_vjp)
            def vjp_tile(g, x, y, parameters):
                values = mapped_vjp(g, x, y, *parameters)
                if form == 'broadcast':
                    return values
                return (*values[:2], *(value.sum(0) for value in values[2:]))
            derivative = _compile_pairs(vjp_tile, backward=True) if compile else vjp_tile
            apply = _with_vjp(evaluate, derivative)
            self._evaluate = lambda x, y, parameters: apply(x, y, *parameters)
        else:
            self._evaluate = evaluate
        local_name = f'metric_{next(_NAMES)}' if name is None else name
        if not isinstance(local_name, str) or not local_name.isidentifier():
            raise ValueError('name must be a simple dispatcher identifier')
        lib = torch.library.Library('torchpairwise_custom', 'FRAGMENT')
        lib.define(f'{local_name}(Tensor x, Tensor y, Tensor[] parameters) -> Tensor')
        lib.impl(local_name, self._evaluate, 'CompositeImplicitAutograd')
        _LIBRARIES.append(lib)
        self.forward_op, self.backward_op = f'torchpairwise_custom::{local_name}', ''

    def __call__(self, x1, x2=None, *, mode=None):
        mode = _check_mode(self.mode if mode is None else mode, self.symmetric)
        if mode in ('paired', 'single') and x2 is None:
            raise TypeError(f'{mode} mode requires both input tensors')
        if mode == 'condensed' and x2 is not None:
            raise TypeError('condensed mode accepts one input tensor')
        _check_parameters(self.parameters, x1 if isinstance(x1, torch.Tensor) else None)
        if self._native:
            return torch.ops.torchpairwise.custom_compare(x1, x2, list(self.parameters),
                self.forward_op, self.backward_op, mode, self.sample_ndim,
                None if self.output_dtype is None else _DTYPE_CODES[self.output_dtype],
                self.chunk_size, self.symmetric, self.form == 'broadcast')
        if mode == 'single':
            if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
                raise TypeError('inputs must be tensors')
            if min(x1.ndim, x2.ndim) < self.sample_ndim:
                raise ValueError('single inputs must have sample_ndim sample axes')
            return self._dense(x1.unsqueeze(-self.sample_ndim - 1),
                               x2.unsqueeze(-self.sample_ndim - 1), 'paired').squeeze(-1)
        return self._dense(x1, x2, mode)

    def _dense(self, x1, x2, mode):
        if mode not in self._dense_cache:
            self._dense_cache[mode] = _make_dense_pairwise(self._fn, mode=mode,
                sample_ndim=self.sample_ndim, chunk_size=self.chunk_size,
                symmetric=self.symmetric, output_dtype=self.output_dtype,
                _evaluate_pairs=lambda a, b: self._evaluate(a, b, self.parameters), form=self.form)
        result = self._dense_cache[mode](x1, x2)
        if result.numel() == 0:
            for p in self.parameters:
                result = result + p.reshape(-1)[:0].sum().to(result.dtype)
        return result

    def _operator(self, x1, x2=None, *, chunk_size=None):
        if self._hidden_trainable:
            raise ValueError('streaming trainable tensors must be explicit parameters, not captured callback state')
        if isinstance(x1, torch.Tensor):
            _check_parameters(self.parameters, x1)
        _check_chunk_size(chunk_size, allow_none=True)
        return CustomPairwiseOperator(self, x1, x2, chunk_size)


class CustomPairwiseOperator(_ScalarOperator):
    def __init__(self, comparison, x1, x2, chunk_size):
        self.comparison, self.x1, self.x2, self.chunk_size = comparison, x1, x2, chunk_size
        self._transposed = False
        rank = comparison.sample_ndim + 1
        if not isinstance(x1, torch.Tensor) or (x2 is not None and not isinstance(x2, torch.Tensor)):
            raise TypeError('operator inputs must be tensors')
        y = x1 if x2 is None else x2
        if min(x1.ndim, y.ndim) < rank:
            raise ValueError('operator inputs must have a sample-count axis and sample axes')
        torch.broadcast_shapes(x1.shape[:-rank], y.shape[:-rank])

    @property
    def shape(self):
        rank = self.comparison.sample_ndim + 1
        x, y = self.x1, self.x1 if self.x2 is None else self.x2
        return torch.Size((*torch.broadcast_shapes(x.shape[:-rank], y.shape[:-rank]), x.shape[-rank], y.shape[-rank]))

    @property
    def dtype(self):
        return self.comparison.output_dtype or self.x1.dtype

    @property
    def device(self):
        return self.x1.device

    @property
    def T(self):
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('transpose', self)

    def _custom_source_packet(self, transpose):
        c = self.comparison
        integers = {'sample_ndim': c.sample_ndim, 'parameter_count': len(c.parameters), 'output_dtype': _DTYPE_CODES[self.dtype]}
        if self.chunk_size is not None:
            integers['chunk_size'] = self.chunk_size
        # Encode the form in the custom descriptor so the shared expression
        # parser can forward it without introducing a new source option.
        forward = c.forward_op + ('|broadcast' if c.form == 'broadcast' else '')
        return ([self.x1] if self.x2 is None else [self.x1, self.x2], forward, 'custom',
                {f'parameter_{k}': p for k, p in enumerate(c.parameters)}, {}, integers,
                {}, {'backward': c.backward_op}, [], [], self._transposed != transpose)

    def _expression(self):
        from ._operator_algebra import ScalarExpression
        return ScalarExpression('identity', self)

    def _native_evaluate(self, operation, **options):
        return self._expression()._evaluate(operation, **options)

    def _axes(self, *args):
        return self._expression()._axes(*args)

    def _statistics(self, *args):
        return self._expression()._statistics(*args)

    def _weighted_statistics(self, *args, **kwargs):
        return self._expression()._weighted_statistics(*args, **kwargs)


def make_pairwise(fn, *, backward=None, parameters=(), mode='all_pairs', sample_ndim=1,
                  chunk_size=None, compile=False, symmetric=False, output_dtype=None, name=None,
                  form='scalar', keops=False):
    """Adapt scalar/broadcast fn(x,y,*parameters), or a native tile operator name.

    backward(g,x,y,*parameters) returns VJPs in input order. Per-call mode
    overrides the factory default. tp.lazy(X,Y,comparison) binds a native scalar
    source. Modules bind their parameters automatically and retain their buffers
    and training mode; omit parameters= when passing a module.
    Native strings name fwd(Tensor,Tensor,Tensor[])->Tensor over paired tiles,
    and optional vjp(Tensor,Tensor,Tensor,Tensor[])->Tensor[].
    form='broadcast' receives broadcastable leading tile axes and reduces the
    sample axes itself. Its VJP returns gradients with the passed operand shapes.
    keops=True also lets the KeOps backend call a broadcast fn on pykeops
    LazyTensors (rows axis 0, columns axis 1, parameters as axis-2 vectors);
    lazy graphs fall back to native tiles when that call or lowering fails.
    """
    if form not in ('scalar', 'broadcast'):
        raise ValueError("form must be 'scalar' or 'broadcast'")
    if not callable(fn) and not isinstance(fn, str):
        raise TypeError('fn must be callable or a registered operator name')
    if backward is not None and not callable(backward) and not isinstance(backward, str):
        raise TypeError('backward must be callable or a registered operator name')
    if not isinstance(fn, str) and backward is not None and not callable(backward):
        raise TypeError('a Python forward requires a callable backward')
    if type(sample_ndim) is not int or sample_ndim < 1:
        raise ValueError('sample_ndim must be a positive integer')
    _check_chunk_size(chunk_size, allow_none=True)
    if type(compile) is not bool or type(symmetric) is not bool:
        raise TypeError('compile and symmetric must be booleans')
    _check_mode(mode, symmetric)
    if output_dtype is not None and output_dtype not in _FLOAT_DTYPES:
        raise TypeError('output_dtype must be a supported real floating point dtype')
    if type(keops) is not bool:
        raise TypeError('keops must be a boolean')
    if keops and (form != 'broadcast' or isinstance(fn, str) or backward is not None or sample_ndim != 1):
        raise ValueError("keops=True requires a Python form='broadcast' fn with sample_ndim=1 and no "
                         'backward; KeOps differentiates the formula itself')
    return CustomComparison(fn, backward=backward, parameters=parameters, mode=mode,
        sample_ndim=sample_ndim, chunk_size=chunk_size, compile=compile,
        symmetric=symmetric, output_dtype=output_dtype, name=name, form=form, keops=keops)


__all__ = ['CustomComparison', 'CustomPairwiseOperator', 'make_pairwise']
