"""Finite scalar expression graphs; numerical preparation and replay are native."""
import torch
from ._pairwise_operator import _ScalarOperator


_REAL_OPTIONS = frozenset(('p', 'base', 'alpha', 'length_scale', 'nu', 'gamma',
    'coef0', 'regularization', 'data_range', 'sigma', 'k1', 'k2', 'energy_floor',
    'log_floor', 'norm_floor', 'correction'))


def _check_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError('Tensor factors must be scalar; use weight_rows/weight_columns or tp.as_lazy')
    elif type(value) not in (int, float, bool):
        raise TypeError('Scalar algebra accepts operators, Python numbers or zero-dimensional tensors')


def _source_packet(value, transpose):
    if hasattr(value, '_custom_source_packet'):
        return value._custom_source_packet(transpose)
    # KernelOperator inherits the canonical PairwiseOperator metadata and inputs.
    if hasattr(value, 'matrix'):
        return ([value.matrix], '_matrix', 'matrix', {}, {},
                {} if value.chunk_size is None else {'chunk_size': value.chunk_size},
                {}, {}, [], [], value._transposed != transpose)
    x = list(value.x1) if value._family in ('gaussian', 'mixture') else [value.x1]
    if value.x2 is not None:
        x += list(value.x2) if value._family in ('gaussian', 'mixture') else [value.x2]
    family = 'mmd' if value.metric in ('mmd_distances', 'mmd_statistics') else value._family
    t, r, i, b, s, rl, il = {}, {}, {}, {}, {}, [], []
    for key, param in value.metric_kwargs.items():
        if param is None:
            continue
        if key == 'event_program':
            il = list(param)
        elif key == 'event_corrections':
            rl = list(param)
        elif key == 'weights':
            rl = list(param); b['weights_set'] = True
        elif key == 'resolutions':
            b['resolutions_set'] = True
            il = [v for n, hop, win in param for v in (n, max(1, n//4) if hop is None else hop, n if win is None else win)]
        elif isinstance(param, torch.Tensor):
            t[key] = param
        elif key in _REAL_OPTIONS:
            r[key] = float(param)
        elif type(param) is bool:
            b[key] = param
        elif type(param) is int:
            i[key] = param
        elif isinstance(param, str):
            s[key] = param
        else:
            raise TypeError(f'Unsupported native expression option {key!r}')
    if value.chunk_size is not None:
        i['chunk_size'] = value.chunk_size
    if family in ('vector', 'point_cloud'):
        b['is_kernel'] = value._is_kernel
    return x, value.metric, family, t, r, i, b, s, rl, il, value._transposed != transpose


def _binary(op, left, right):
    if any(isinstance(value, _ScalarOperator) and not value.entrywise for value in (left, right)):
        if op in ('add', 'sub', 'mul'):
            from ._linear import _linear_binary
            return _linear_binary(op, left, right)
        for value in (left, right):
            if isinstance(value, _ScalarOperator):
                value._require_entrywise(op)
    if not isinstance(left, _ScalarOperator):
        _check_scalar(left)
    if not isinstance(right, _ScalarOperator):
        _check_scalar(right)
    return ScalarExpression(op, left, right)


_BINARY = {'add': 3, 'sub': 4, 'mul': 5, 'div': 12, 'pow': 13,
           'minimum': 14, 'maximum': 15, 'clamp_min': 16, 'clamp_max': 17,
           'lt': 26, 'le': 27, 'gt': 28, 'ge': 29, 'eq': 30, 'ne': 31,
           'logical_and': 32, 'logical_or': 33}
_UNARY = {'square': 6, 'exp': 7, 'neg': 8, 'sqrt': 18, 'log': 19,
          'sin': 20, 'cos': 21, 'tanh': 22, 'sigmoid': 23, 'abs': 24,
          'logical_not': 25}
_AXIS = {'rows': 9, 'columns': 10, 'row_values': 38, 'column_values': 39,
         'row_index': 40, 'column_index': 41}


def row_values(like, tensor):
    like._require_entrywise('row_values')
    return ScalarExpression('row_values', like, tensor)


def column_values(like, tensor):
    like._require_entrywise('column_values')
    return ScalarExpression('column_values', like, tensor)


def row_index(like):
    like._require_entrywise('row_index')
    return ScalarExpression('row_index', like)


def column_index(like):
    like._require_entrywise('column_index')
    return ScalarExpression('column_index', like)


class ScalarExpression(_ScalarOperator):
    """Lazy scalar algebra over [...,N,M] entries, with native first backward.

    Construct through entrywise arithmetic, transforms, masks and axis values.
    * multiplies entries; matmul contracts with a dense tensor. Expressions are
    generic scalar operators and do not inherit kernel geometry or PSD promises.
    """
    def __init__(self, operation, left, right=None):
        self._operation, self._left, self._right = operation, left, right

    @property
    def T(self):
        return ScalarExpression('transpose', self)

    def _metadata(self):
        def metadata(value):
            if isinstance(value, _ScalarOperator):
                return value.shape, value.dtype, value.device
            _check_scalar(value)
            if isinstance(value, torch.Tensor):
                if value.dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(value.device.type):
                    value = value.float()
                return (), value, value.device
            return (), value, None
        op = self._operation
        a, ad, device = metadata(self._left)
        if op == 'identity':
            return a, ad, device
        if op == 'transpose':
            return torch.Size((*a[:-2], a[-1], a[-2])), ad, device
        if op in ('row_index', 'column_index'):
            return a, torch.int64, device
        if op in ('rows', 'columns', 'row_values', 'column_values'):
            weights = self._right
            if not isinstance(weights, torch.Tensor) or weights.ndim < 1:
                raise ValueError('Weights must be tensors with shape [...,N] or [...,M]')
            torch._check(weights.shape[-1] == a[-2 if op in ('rows', 'row_values') else -1])
            if weights.device != device:
                raise ValueError('Expression operands must share a device')
            low = weights.dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(device.type)
            wd = torch.float32 if low else weights.dtype
            dtype = torch.promote_types(ad, wd) if op in ('rows', 'columns') else wd
            return torch.Size((*torch.broadcast_shapes(a[:-2], weights.shape[:-1]), *a[-2:])), dtype, device
        if op in _UNARY:
            probe = torch.empty((0, 0), device=device, dtype=ad)
            return a, getattr(torch, op)(probe).dtype, device
        if op == 'where':
            condition_dtype = ad.dtype if isinstance(ad, torch.Tensor) else ad
            if condition_dtype != torch.bool:
                raise TypeError('where condition must have boolean dtype')
            b, bd, bdevice = metadata(self._right[0])
            c, cd, cdevice = metadata(self._right[1])
            devices = [d for d in (device, bdevice, cdevice) if d is not None]
            if any(d != devices[0] for d in devices):
                raise ValueError('Expression operands must share a device')
            device = devices[0] if devices else None
            shapes = [shape for shape in (a, b, c) if shape]
            for shape in shapes[1:]:
                torch._check(shapes[0][-2] == shape[-2]); torch._check(shapes[0][-1] == shape[-1])
            shape = torch.Size((*torch.broadcast_shapes(*(s[:-2] for s in shapes)), *shapes[0][-2:])) if shapes else torch.Size()
            left = torch.empty((0, 0), device=device, dtype=bd) if b else bd
            right = torch.empty((0, 0), device=device, dtype=cd) if c else cd
            probe = torch.where(torch.empty((0, 0), device=device, dtype=torch.bool), left, right)
            return shape, probe.dtype, device
        b, bd, other_device = metadata(self._right)
        if op in ('clamp_min', 'clamp_max'):
            _check_scalar(self._right)
        if a and b:
            torch._check(a[-2] == b[-2]); torch._check(a[-1] == b[-1])
        if device is not None and other_device is not None and device != other_device:
            raise ValueError('Expression operands must share a device')
        device = device if device is not None else other_device
        shape = torch.Size((*torch.broadcast_shapes(a[:-2], b[:-2]), *(a[-2:] if a else b[-2:])))
        lhs = torch.empty((0, 0), device=device, dtype=ad) if a else ad
        rhs = torch.empty((0, 0), device=device, dtype=bd) if b else bd
        if not a and not b:
            lhs = torch.empty((), device=device, dtype=ad.dtype) if isinstance(ad, torch.Tensor) else torch.tensor(ad, device=device)
            rhs = bd
        # minimum/maximum require tensor operands; scalar nodes inherit their
        # partner's dtype in the native interpreter.
        if op in ('minimum', 'maximum'):
            if not isinstance(lhs, torch.Tensor):
                lhs = torch.full((), lhs, device=device, dtype=rhs.dtype)
            if not isinstance(rhs, torch.Tensor):
                rhs = torch.full((), rhs, device=device, dtype=lhs.dtype)
        return shape, getattr(torch, op)(lhs, rhs).dtype, device

    @property
    def shape(self):
        return self._metadata()[0]

    @property
    def dtype(self):
        return self._metadata()[1]

    @property
    def device(self):
        return self._metadata()[2]

    def _packet(self, weights=None):
        sources, nodes, tensors, numbers, seen = [], [], [], [], []
        codes = {**_BINARY, **_UNARY, **_AXIS}
        def visit(recurse, value, transpose=False):
            for previous, orientation, idx in seen:
                if value is previous and orientation == transpose:
                    return idx
            if isinstance(value, ScalarExpression):
                op = value._operation
                if op in ('transpose', 'identity'):
                    idx = recurse(recurse, value._left, not transpose if op == 'transpose' else transpose)
                    seen.append((value, transpose, idx))
                    return idx
                if op not in codes and op != 'where':
                    raise ValueError(f'Unsupported scalar expression operation {op!r}')
                left = -1 if op in ('row_values', 'column_values', 'row_index', 'column_index') else recurse(recurse, value._left, transpose)
                if op in ('rows', 'columns', 'row_values', 'column_values'):
                    weights = value._right
                    if not isinstance(weights, torch.Tensor) or weights.ndim < 1:
                        raise ValueError('Weights must be tensors with shape [...,N] or [...,M]')
                    right = len(tensors); tensors.append(weights)
                    opcode = codes[op]
                    if transpose:
                        opcode = (19 if opcode in (9, 10) else 77)-opcode
                elif op in ('row_index', 'column_index'):
                    right = -1
                    opcode = 81-codes[op] if transpose else codes[op]
                elif op == 'where':
                    yes = recurse(recurse, value._right[0], transpose)
                    no = recurse(recurse, value._right[1], transpose)
                    right = len(nodes); nodes.append([35, yes, no])
                    opcode = 34
                else:
                    if op in ('clamp_min', 'clamp_max'):
                        _check_scalar(value._right)
                    right = recurse(recurse, value._right, transpose) if op in _BINARY else -1
                    opcode = codes[op]
                node = [opcode, left, right]
            elif isinstance(value, _ScalarOperator):
                node = [0, len(sources), -1]
                sources.append(_source_packet(value, transpose))
            elif isinstance(value, torch.Tensor):
                _check_scalar(value)
                node = [1, len(tensors), -1]; tensors.append(value)
            else:
                _check_scalar(value)
                node = [2, len(numbers), -1]; numbers.append(value)
            idx = len(nodes); nodes.append(node); seen.append((value, transpose, idx))
            return idx
        value_root = visit(visit, self)
        weight_root = visit(visit, weights) if weights is not None else -1
        if not sources:
            # Carry geometry without evaluating the metadata-only like object.
            matrix = torch.zeros((), dtype=self.dtype, device=self.device).expand(self.shape)
            sources.append(([matrix], '_matrix', 'matrix', {}, {}, {}, {}, {}, [], [], False))
        # Flat typed option dictionaries and homogeneous number lists survive
        # export serialization; lists of dictionaries and mixed Scalar[] do not.
        packets = [[source[k] for source in sources] for k in (0,1,2)]
        packets += [{f'{idx}:{key}': param for idx, source in enumerate(sources)
                     for key, param in source[column].items()} for column in (3,4,5,6,7)]
        packets += [{str(idx): source[8] for idx, source in enumerate(sources) if source[8]},
                    {str(idx): source[9] for idx, source in enumerate(sources) if source[9]},
                    [source[10] for source in sources]]
        kinds = [0 if type(v) is bool else 1 if type(v) is int else 2 for v in numbers]
        integers = [int(v) if type(v) in (int,bool) else 0 for v in numbers]
        reals = [v if type(v) is float else 0. for v in numbers]
        packet = (*packets, [value for node in nodes for value in node], tensors,
                  kinds, integers, reals)
        return (packet,value_root,weight_root) if weights is not None else packet

    def _native_evaluate(self, operation, *, packet=None, _return_tuple=False, **options):
        packet = self._packet() if packet is None else packet
        groups, *metadata = packet
        sources = [tensor for group in groups for tensor in group]
        counts = [len(group) for group in groups]
        if operation == '_selection':
            return torch.ops.torchpairwise.expression_operations.selection(
                sources, counts, *metadata, options['reduction'], float(options['q']),
                options['dim'], options['keepdim'], options['exclude_self'],
                options['interpolation'])
        result = torch.ops.torchpairwise.expression_operations.flat(
            sources, counts, *metadata, operation, **options)
        return result if _return_tuple else result[0]

    def _axes(self, reduction, axes, keepdim, exclude_self, correction):
        return self._evaluate('reduce', reduction=reduction, dims=axes, use_axes=True,
            keepdim=keepdim, exclude_self=exclude_self, correction=correction)

    def _weighted_statistics(self, weights, reduction, dim, keepdim, exclude_self, correction, axes=None):
        packet,value_root,weight_root = self._packet(weights)
        return self._evaluate('reduce', packet=packet, reduction=reduction,
            dim=dim if axes is None else None, dims=[] if axes is None else axes,
            use_axes=axes is not None, keepdim=keepdim, exclude_self=exclude_self,
            correction=correction,value_root=value_root,weight_root=weight_root)

    def _statistics(self, reduction, dim, keepdim, exclude_self, correction):
        return self._evaluate('reduce', reduction=reduction, dim=dim, keepdim=keepdim,
            exclude_self=exclude_self, correction=correction)



__all__ = ['ScalarExpression', 'row_values', 'column_values', 'row_index', 'column_index']
