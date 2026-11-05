"""Public batch contracts around the native 2D/3D metric primitives.

Only batch dimensions broadcast. Native kernels receive flattened batches;
expand/reshape autograd reduces gradients back to each operand's original shape.
"""
import math
import torch
from ._validation import _check_chunk_size
from torch._subclasses.fake_tensor import FakeTensor

from ._registry import OPS, PROBABILITY_EXTRA_OPS, VECTOR_EXTRA_OPS, NATIVE_VECTOR_OPS, NATIVE_POINTSET_OPS


PARAMETER_NDIM = {'w': 1, 'V': 1, 'VI': 2}


def is_finite_real(value):
    """math.isfinite for real scalars, also valid for symbolic floats under
    dynamic compilation (comparisons guard instead of failing)."""
    return -math.inf < value < math.inf


def validate_values(condition, message):
    # Keep checks in compiled graphs without reading tensor data on the host.
    # CUDA assertions execute asynchronously; eager callers retain ValueError.
    if torch.compiler.is_compiling() or isinstance(condition, FakeTensor) or condition.device.type == 'meta':
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def prepare(spec, x1, x2, kwargs, *, paired=False):
    if ('w' in spec.parameters and 'w' in kwargs and kwargs['w'] is None
            and spec.op != 'wminkowski_distances'):
        # Optional weights must not leak into unchanged native
        # signatures when a generic paired/condensed caller supplies None.
        kwargs = {key: value for key, value in kwargs.items() if key != 'w'}
    if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
        raise TypeError('x1 and x2 must be tensors')
    rank = spec.event_ndim
    if x1.ndim < rank or x2.ndim < rank:
        raise ValueError(f'Inputs must have at least {rank} dimensions')
    if x1.shape[-1] != x2.shape[-1]:
        raise ValueError('Input feature dimensions must match')
    if x1.shape[-1] == 0:
        raise ValueError('Feature dimension must be non-empty')
    if paired and x1.shape[-rank] != x2.shape[-rank]:
        raise ValueError('Paired inputs must have equal sample counts')
    if x1.device != x2.device:
        raise ValueError('Inputs must be on the same device')
    if x1.dtype != x2.dtype:
        raise ValueError('Inputs must have the same dtype')
    if spec.differentiable and not x1.is_floating_point():
        raise TypeError('This metric requires floating point inputs')
    if spec.op in PROBABILITY_EXTRA_OPS:
        from ._probability_extra import validate_probability_extra
        validate_probability_extra(spec, x1, x2, kwargs)
    if spec.op in VECTOR_EXTRA_OPS:
        from ._vector_extra import validate_vector_extra_inputs
        validate_vector_extra_inputs(spec, x1, x2)
    if spec.op in ('bhattacharyya_distances', 'bhattacharyya_kernel', 'aitchison_distances'):
        if x1.layout != torch.strided or x2.layout != torch.strided:
            raise TypeError('Inputs must be dense strided tensors')
        if x1.dtype not in (torch.float32, torch.float64):
            raise TypeError('These comparisons require float32 or float64 inputs outside autocast')
        from ._compositional import validate_bhattacharyya_inputs, validate_aitchison_inputs
        if spec.op == 'aitchison_distances':
            validate_aitchison_inputs(x1, x2)
        else:
            validate_bhattacharyya_inputs(x1, x2)
    if kwargs.get('w') is not None and 'w' in spec.parameters and spec.boolean:
        if x1.dtype != torch.bool:
            raise TypeError('Boolean distances require bool inputs')
    if spec.op in ('kl_divergences', 'symmetric_kl_divergences'):
        if x1.layout != torch.strided or x2.layout != torch.strided:
            raise TypeError('Inputs must be dense strided tensors')
        if x1.dtype not in (torch.float32, torch.float64):
            raise TypeError('KL requires float32 or float64 inputs outside autocast')
        from ._divergences import validate_kl_inputs
        validate_kl_inputs(x1, x2)
        base = kwargs.get('base')
        if base is not None and (isinstance(base, bool) or not isinstance(base, (int, float))
                                 or not is_finite_real(base) or base <= 1):
            raise ValueError('KL logarithm base must be a finite scalar greater than 1')
    if spec.op in ('cosine_distances', 'correlation_distances') and kwargs.get('w') is not None:
        if x1.layout != torch.strided or x2.layout != torch.strided:
            raise TypeError('Inputs must be dense strided tensors')
        if x1.dtype not in (torch.float32, torch.float64):
            raise TypeError('Weighted cosine/correlation require float32 or float64 inputs outside autocast')
    if spec.op in ('hellinger_distances', 'histogram_intersection_kernel', 'wasserstein_1d_distances', 'nan_euclidean_distances'):
        if x1.layout != torch.strided or x2.layout != torch.strided:
            raise TypeError('Inputs must be dense strided tensors')
    if spec.op == 'haversine_distances' and x1.shape[-1] != 2:
        raise ValueError('Haversine inputs must have exactly two features')
    if rank == 3 and (x1.shape[-2] == 0 or x2.shape[-2] == 0):
        raise ValueError('Input point sets must be non-empty')
    unknown = set(kwargs) - set(spec.parameters)
    if unknown:
        raise TypeError(f'{spec.op} does not accept arguments: {", ".join(sorted(unknown))}')
    if spec.op == 'sliced_wasserstein_distances' and 'projections' not in kwargs:
        raise TypeError('sliced_wasserstein_distances requires projections')
    if spec.op in ('rational_quadratic_kernel', 'matern_kernel'):
        from ._kernels import validate_radial_parameters
        validate_radial_parameters(spec, kwargs)
    if spec.op == 'nan_euclidean_distances' and type(kwargs.get('squared', False)) is not bool:
        raise TypeError('squared must be a boolean')
    if spec.op == 'wasserstein_1d_distances' and kwargs.get('bin_positions') is not None:
        positions = kwargs['bin_positions']
        if not isinstance(positions, torch.Tensor) or positions.ndim != 1 or positions.shape[0] != x1.shape[-1]:
            raise ValueError('bin_positions must be a one-dimensional tensor with D entries')
        if positions.dtype != x1.dtype or positions.device != x1.device:
            raise ValueError('bin_positions must have the input dtype and device')
        widths = positions.diff()
        validate_values(torch.isfinite(positions).all() & (torch.isfinite(widths) & (widths > 0)).all(),
                        'bin_positions must be finite and strictly increasing, with finite gaps')
    chunk_size = kwargs.get('chunk_size')
    if chunk_size is not None:
        if type(chunk_size) is not int:
            raise TypeError('chunk_size must be an integer')
        _check_chunk_size(chunk_size)
        if kwargs.get('shuffle') or kwargs.get('generator') is not None:
            raise ValueError('chunk_size does not support shuffle or generator')
    shapes = [x1.shape[:-rank], x2.shape[:-rank]]
    if rank == 3:
        for key, value in (('x1_lengths', x1), ('x2_lengths', x2)):
            lengths = kwargs.get(key)
            if lengths is None:
                continue
            if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int64:
                raise TypeError(f'{key} must be an int64 tensor')
            if lengths.device != value.device:
                raise ValueError(f'{key} must be on the input device')
            if lengths.ndim < 1 or lengths.shape[-1] != value.shape[-3]:
                raise ValueError(f'{key} must end with the input sample count')
            if lengths.ndim - 1 > max(x1.ndim, x2.ndim) - rank:
                raise ValueError(f'{key} cannot introduce additional batch axes')
            validate_values(((lengths >= 1) & (lengths <= value.shape[-2])).all(),
                            f'{key} must be between 1 and the padded point count')
            shapes.append(lengths.shape[:-1])
    d = x1.shape[-1]
    for key, value in kwargs.items():
        if key not in PARAMETER_NDIM or value is None:
            continue
        ndim = PARAMETER_NDIM[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{key} must be a tensor')
        if value.ndim < ndim or tuple(value.shape[-ndim:]) != (d,) * ndim:
            raise ValueError(f'{key} must end with shape {(d,) * ndim}')
        separate_weights = key == 'w' and not spec.differentiable
        if separate_weights and value.dtype not in (torch.float32, torch.float64):
            raise TypeError('Weights for boolean/Hamming distances must be float32 or float64')
        if (not separate_weights and value.dtype != x1.dtype) or value.device != x1.device:
            raise ValueError(f'{key} must have the same dtype and device as the inputs')
        shapes.append(value.shape[:-ndim])
    try:
        batch = torch.broadcast_shapes(*shapes)
    except RuntimeError as exc:
        raise ValueError('Input and parameter batch dimensions are not broadcastable') from exc
    x1 = x1.expand(*batch, *x1.shape[-rank:])
    x2 = x2.expand(*batch, *x2.shape[-rank:])
    kwargs = dict(kwargs)
    if rank == 3:
        for key in ('x1_lengths', 'x2_lengths'):
            if kwargs.get(key) is not None:
                value = kwargs[key]
                kwargs[key] = value.expand(*batch, value.shape[-1])
    for key, ndim in PARAMETER_NDIM.items():
        if kwargs.get(key) is not None:
            value = kwargs[key]
            kwargs[key] = value.expand(*batch, *value.shape[-ndim:])
    if kwargs.get('p') is not None:
        p = kwargs['p']
        if not p >= 0 or (spec.op == 'wminkowski_distances' and p == 0):
            raise ValueError('p must be non-negative (strictly positive for weighted Minkowski)')
    if kwargs.get('base') is not None and not kwargs['base'] > 1:
        raise ValueError('Jensen-Shannon logarithm base must be greater than 1')
    for key in ('w', 'V'):
        value = kwargs.get(key)
        if value is not None:
            valid = value >= 0 if key == 'w' else value > 0
            validate_values((valid & torch.isfinite(value)).all(),
                            f'{key} must be {"non-negative" if key == "w" else "positive"}')
    if kwargs.get('w') is not None and 'w' in spec.parameters:
        if spec.op in ('minkowski_distances', 'wminkowski_distances'):
            p = kwargs.get('p', 2.)
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not p > 0:
                raise ValueError('Weighted Minkowski p must be positive')
        if spec.op not in ('euclidean_distances', 'sqeuclidean_distances', 'manhattan_distances',
                           'minkowski_distances', 'wminkowski_distances', 'chebyshev_distances', 'canberra_distances'):
            validate_values((kwargs['w'].amax(-1) > 0).all(), 'w must have positive total weight')
    if spec.op == 'jensenshannon_distances':
        for value in (x1, x2):
            validate_values((value >= 0).all() & (value.sum(-1) > 0).all(),
                            'Jensen-Shannon inputs must be non-negative with positive row mass')
    if spec.op == 'nan_euclidean_distances':
        validate_values((~torch.isinf(x1)).all() & (~torch.isinf(x2)).all(),
                        'NaN Euclidean inputs may contain NaNs but not infinities')
    if spec.op in ('hellinger_distances', 'histogram_intersection_kernel', 'wasserstein_1d_distances'):
        for value in (x1, x2):
            validate_values((torch.isfinite(value) & (value >= 0)).all(),
                            'Histogram inputs must be finite and non-negative')
            if spec.op in ('hellinger_distances', 'wasserstein_1d_distances'):
                mass = value.sum(-1)
                validate_values((torch.isfinite(mass) & (mass > 0)).all(),
                                'Histogram inputs must have positive finite row mass')
    if spec.op in ('additive_chi2_kernel', 'chi2_kernel'):
        validate_values((x1 >= 0).all() & (x2 >= 0).all(),
                        'Chi-squared inputs must be non-negative')
    if spec.op in NATIVE_POINTSET_OPS:
        if type(kwargs.get('squared', False)) is not bool:
            raise TypeError('squared must be a boolean')
        if any(kwargs.get(key) is not None for key in ('x1_lengths', 'x2_lengths')) and (
                kwargs.get('shuffle') or kwargs.get('generator') is not None):
            raise ValueError('Shuffling is supported only for Hausdorff without lengths')
    return x1, x2, batch, kwargs


def evaluate(spec, x, y, kwargs):
    # Prepared internal tiles, including explicit Python predicate callbacks,
    # share the same native algorithms as the public comparison APIs.
    from ._registry import KERNELS
    is_kernel = any(item.op == spec.op for item in KERNELS.values())
    return torch.ops.torchpairwise.builtin_comparisons(
        x, y, spec.op, 'all_pairs', is_kernel, **kwargs)


def _pointset_layout(x, batched, feature_dim):
    """Collapse sample-local point axes, leaving native kernels unchanged."""
    if not isinstance(x, torch.Tensor):
        raise TypeError('x1 and x2 must be tensors')
    if x.layout != torch.strided:
        raise TypeError('Point-set inputs must be dense strided tensors; sparse layouts are not supported')
    if batched is not None and type(batched) is not bool:
        raise TypeError('batched must be True, False, or None')
    if type(feature_dim) is not int:
        raise TypeError('feature_dim must be an integer')
    minimum = 4 if batched else 3
    if x.ndim < minimum:
        raise ValueError(f'Point-set inputs must have at least {minimum} dimensions')
    if not -x.ndim <= feature_dim < x.ndim:
        raise ValueError('feature_dim is out of range')
    dim = feature_dim % x.ndim
    # None preserves [...,N,P,C]. Explicit bools fix N at axis 0/1 so
    # arbitrary trailing spatial axes have an unambiguous interpretation.
    sample_dim = x.ndim - 3 if batched is None else int(batched)
    if dim <= sample_dim:
        raise ValueError('feature_dim must follow the batch and sample axes')
    x = x.movedim(dim, -1)
    points = math.prod(x.shape[sample_dim + 1:-1])
    return x.reshape(*x.shape[:sample_dim + 1], points, x.shape[-1])


def prepare_pointsets(spec, x1, x2, kwargs, *, paired=False):
    kwargs = dict(kwargs)
    batched = kwargs.pop('batched', None)
    feature_dim = kwargs.pop('feature_dim', -1)
    if x2 is None:
        x2 = x1
        if kwargs.get('x2_lengths') is None and kwargs.get('x1_lengths') is not None:
            kwargs['x2_lengths'] = kwargs['x1_lengths']
    return prepare(spec, _pointset_layout(x1, batched, feature_dim),
                   _pointset_layout(x2, batched, feature_dim), kwargs, paired=paired)


def validate_comparison(spec, x1, x2, kwargs, *, mode='all_pairs'):
    if spec.event_ndim == 3:
        x, y, _, options = prepare_pointsets(spec, x1, x2, kwargs, paired=mode == 'paired')
        if spec.op not in NATIVE_POINTSET_OPS:
            from ._pointsets import _validate_transport
            _validate_transport(spec, x, y, options)
    else:
        x, y, _, options = prepare(spec, x1, x1 if x2 is None else x2, kwargs,
                                   paired=mode == 'paired')
        if mode == 'paired':
            for name, key in (('mahalanobis_distances', 'VI'), ('seuclidean_distances', 'V')):
                if spec.op == name and options.get(key) is None:
                    raise ValueError(f'{key} is required for paired {name}')


def call_pairwise(name, x1, x2=None, **kwargs):
    spec = OPS[name]
    if (isinstance(x1, torch.Tensor) and spec.differentiable
            and x1.device.type in ('cpu', 'cuda') and torch.is_autocast_enabled(x1.device.type)):
        from .._integration import autocast_inputs
        with torch.autocast(x1.device.type, enabled=False):
            x1, x2, kwargs = autocast_inputs(x1, x2, kwargs)
            return call_pairwise(name, x1, x2, **kwargs)
    validate_comparison(spec, x1, x2, kwargs)
    if spec.op == 'directed_hausdorff_distances':
        return torch.ops.torchpairwise.directed_pointset_comparisons(x1, x2, **kwargs)
    from ._registry import KERNELS
    is_kernel = any(item.op == spec.op for item in KERNELS.values())
    return torch.ops.torchpairwise.builtin_comparisons(x1, x2, spec.op,
                                                     'all_pairs', is_kernel, **kwargs)
