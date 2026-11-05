"""Dispatcher metadata for compilation and export.

Registrations happen after the extension is loaded. The float32 mixed precision
policy for comparison schemas is registered natively, so it also applies to
standalone LibTorch callers. Native primitives supply
custom distances; derivatives use saved witnesses or native backward kernels.
Fake kernels allocate only metadata, including native output strides.
"""
import torch


def autocast_inputs(x, y, kwargs):
    """Promote reduced precision operands, leaving double/integer tensors alone."""
    def cast(value):
        if isinstance(value, torch.Tensor) and value.dtype in (torch.float16, torch.bfloat16):
            return value.float()
        return value

    xx = cast(x)
    yy = xx if y is x else cast(y)
    return xx, yy, {key: cast(value) for key, value in kwargs.items()}


def _lowrank_metadata(features, shift):
    if (features.layout != torch.strided or features.ndim < 2
            or features.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Low-rank setup requires strided float32/float64 [...,N,L] tensors')
    if (shift.layout != torch.strided or shift.ndim != 0 or shift.dtype != features.dtype
            or shift.device != features.device):
        raise ValueError('Low-rank shift must be a scalar tensor matching dtype/device')
    if torch.is_grad_enabled() and (features.requires_grad or shift.requires_grad):
        raise RuntimeError('Low-rank setup is inference-only; use torch.no_grad() or detach inputs explicitly')


def _linear_lowrank_setup_fake(features, shift):
    _lowrank_metadata(features, shift)
    rank = torch.sym_min(features.shape[-2], features.shape[-1])
    return (features.new_empty((*features.shape[:-1], rank), dtype=torch.float64),
            features.new_empty((*features.shape[:-2], rank), dtype=torch.float64),
            shift.new_empty(()))


def _linear_nystrom_setup_fake(cross, landmarks, shift, jitter=0.):
    import math
    _lowrank_metadata(cross, shift)
    _lowrank_metadata(landmarks, shift)
    if not math.isfinite(jitter) or jitter < 0:
        raise ValueError('Nyström jitter must be finite and nonnegative')
    torch._check(landmarks.shape[-2] == landmarks.shape[-1])
    torch._check(cross.shape[-1] == landmarks.shape[-1])
    batch = torch.broadcast_shapes(cross.shape[:-2], landmarks.shape[:-2])
    rank = torch.sym_min(cross.shape[-2], cross.shape[-1])
    return (cross.new_empty((*batch, cross.shape[-2], rank), dtype=torch.float64),
            cross.new_empty((*batch, rank), dtype=torch.float64), shift.new_empty(()))


def _linear_implicit_solve_fake(inputs, counts, metrics, families, tensors, reals, integers,
        flags, strings, real_lists, integer_lists, transposed, source_counts, programs,
        factors, factor_counts, kinds, integer_numbers, real_numbers, linear_program,
        linear_factors, shape, rhs, method, rtol, atol, max_iter, restart, initial=None):
    if method not in ('cg', 'minres', 'gmres'):
        raise ValueError('Implicit solve method must be cg, minres or gmres')
    if initial is not None and initial.requires_grad:
        raise RuntimeError('Implicit solve initial guess is an algorithmic aid; detach it explicitly')
    # Inference metadata contracts apply; derivative tracking is supplied by the
    # native custom autograd Function, not by tracing iterative solver updates.
    with torch.no_grad():
        return _linear_gmres_fake(inputs, counts, metrics, families, tensors, reals,
            integers, flags, strings, real_lists, integer_lists, transposed,
            source_counts, programs, factors, factor_counts, kinds, integer_numbers,
            real_numbers, linear_program, linear_factors, shape, rhs, rtol, atol,
            restart, max_iter, initial)


def _linear_implicit_solve_backward_fake(inputs, counts, metrics, families, tensors, reals,
        integers, flags, strings, real_lists, integer_lists, transposed, source_counts,
        programs, factors, factor_counts, kinds, integer_numbers, real_numbers,
        linear_program, linear_factors, shape, rhs, solution, gradient, method,
        rtol, atol, max_iter, restart):
    return [v.new_empty(v.shape) for v in (*inputs, *factors, *linear_factors, *tensors.values(), rhs)]


def _linear_implicit_lstsq_fake(inputs, counts, metrics, families, tensors, reals, integers,
        flags, strings, real_lists, integer_lists, transposed, source_counts, programs,
        factors, factor_counts, kinds, integer_numbers, real_numbers, linear_program,
        linear_factors, shape, rhs, damp, rtol, atol, max_iter, initial=None):
    import math
    if not math.isfinite(damp) or damp <= 0:
        raise ValueError('Differentiable lstsq requires finite positive damp')
    if initial is not None and initial.requires_grad:
        raise RuntimeError('Implicit lstsq initial is an algorithmic aid; detach it explicitly')
    with torch.no_grad():
        return _linear_lsqr_fake(inputs, counts, metrics, families, tensors, reals,
            integers, flags, strings, real_lists, integer_lists, transposed,
            source_counts, programs, factors, factor_counts, kinds, integer_numbers,
            real_numbers, linear_program, linear_factors, shape, rhs, damp, rtol,
            atol, max_iter, initial)


def _linear_implicit_lstsq_backward_fake(inputs, counts, metrics, families, tensors, reals,
        integers, flags, strings, real_lists, integer_lists, transposed, source_counts,
        programs, factors, factor_counts, kinds, integer_numbers, real_numbers,
        linear_program, linear_factors, shape, rhs, solution, gradient, damp,
        rtol, atol, max_iter):
    return [v.new_empty(v.shape) for v in (*inputs, *factors, *linear_factors, *tensors.values(), rhs)]


def _linear_cg_fake(inputs, counts, metrics, families, tensors, reals, integers,
                    flags, strings, real_lists, integer_lists, transposed,
                    source_counts, programs, factors, factor_counts, kinds,
                    integer_numbers, real_numbers, linear_program, linear_factors,
                    shape, rhs, rtol=1e-5, atol=0., max_iter=1000, initial=None):
    if rhs.ndim < 1 or rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError('CG RHS must be float32/float64 with shape [N] or [...,N,R]')
    if torch.is_grad_enabled():
        values = [*inputs, *factors, *linear_factors, *tensors.values(), rhs]
        if initial is not None:
            values.append(initial)
        if any(value.requires_grad for value in values):
            raise RuntimeError('CG is inference-only; use torch.no_grad() or detach inputs explicitly')
    vector = rhs.ndim == 1
    block = rhs.unsqueeze(-1) if vector else rhs
    torch._check(len(shape) >= 2)
    torch._check(shape[-1] == shape[-2])
    torch._check(block.shape[-2] == shape[-1])
    torch._check(block.shape[-1] > 0)
    batch = torch.broadcast_shapes(shape[:-2], block.shape[:-2])
    out = (*batch, shape[-1]) if vector else (*batch, shape[-1], block.shape[-1])
    diagnostics = batch if vector else (*batch, block.shape[-1])
    return (rhs.new_empty(out), rhs.new_empty(diagnostics, dtype=torch.float64),
            rhs.new_empty(diagnostics, dtype=torch.int64), rhs.new_empty(diagnostics, dtype=torch.bool),
            rhs.new_empty(diagnostics, dtype=torch.int64))


def _linear_minres_fake(inputs, counts, metrics, families, tensors, reals, integers,
                        flags, strings, real_lists, integer_lists, transposed,
                        source_counts, programs, factors, factor_counts, kinds,
                        integer_numbers, real_numbers, linear_program, linear_factors,
                        shape, rhs, rtol=1e-5, atol=0., max_iter=1000, initial=None):
    return _linear_gmres_fake(inputs, counts, metrics, families, tensors, reals,
        integers, flags, strings, real_lists, integer_lists, transposed,
        source_counts, programs, factors, factor_counts, kinds, integer_numbers,
        real_numbers, linear_program, linear_factors, shape, rhs,
        rtol=rtol, atol=atol, max_iter=max_iter, initial=initial)


def _linear_gmres_fake(inputs, counts, metrics, families, tensors, reals, integers,
                       flags, strings, real_lists, integer_lists, transposed,
                       source_counts, programs, factors, factor_counts, kinds,
                       integer_numbers, real_numbers, linear_program, linear_factors,
                       shape, rhs, rtol=1e-5, atol=0., restart=20, max_iter=1000, initial=None):
    # The square solver has CG's output metadata. Reuse rectangular validation
    # for initial/RHS metadata, then omit the normal-residual field.
    torch._check(len(shape) >= 2)
    torch._check(shape[-1] == shape[-2])
    result = _linear_lsqr_fake(inputs, counts, metrics, families, tensors, reals,
        integers, flags, strings, real_lists, integer_lists, transposed,
        source_counts, programs, factors, factor_counts, kinds, integer_numbers,
        real_numbers, linear_program, linear_factors, shape, rhs,
        rtol=rtol, atol=atol, max_iter=max_iter, initial=initial)
    return result[0], result[1], result[3], result[4], result[5]


def _linear_lsqr_fake(inputs, counts, metrics, families, tensors, reals, integers,
                      flags, strings, real_lists, integer_lists, transposed,
                      source_counts, programs, factors, factor_counts, kinds,
                      integer_numbers, real_numbers, linear_program, linear_factors,
                      shape, rhs, damp=0., rtol=1e-5, atol=0., max_iter=1000, initial=None):
    if rhs.ndim < 1 or rhs.dtype not in (torch.float32, torch.float64):
        raise ValueError('LSQR RHS must be float32/float64 with shape [N] or [...,N,R]')
    if torch.is_grad_enabled():
        values = [*inputs, *factors, *linear_factors, *tensors.values(), rhs]
        if initial is not None:
            values.append(initial)
        if any(value.requires_grad for value in values):
            raise RuntimeError('LSQR is inference-only; use torch.no_grad() or detach inputs explicitly')
    vector = rhs.ndim == 1
    block = rhs.unsqueeze(-1) if vector else rhs
    torch._check(len(shape) >= 2)
    torch._check(block.shape[-2] == shape[-2])
    torch._check(block.shape[-1] > 0)
    batch = torch.broadcast_shapes(shape[:-2], block.shape[:-2])
    output = (*batch, shape[-1], block.shape[-1])
    if initial is not None:
        if initial.dtype != rhs.dtype or initial.device != rhs.device:
            raise ValueError('LSQR initial must match RHS dtype/device')
        value = initial.unsqueeze(-1) if vector else initial
        torch._check(value.ndim >= 2)
        torch._check(value.shape[-2] == output[-2])
        torch._check(value.shape[-1] == output[-1])
        actual = torch.broadcast_shapes(value.shape, output)
        torch._check(len(actual) == len(output))
        for a, b in zip(actual, output):
            torch._check(a == b)
    out = (*batch, shape[-1]) if vector else output
    diagnostics = batch if vector else (*batch, block.shape[-1])
    return (rhs.new_empty(out), rhs.new_empty(diagnostics, dtype=torch.float64),
            rhs.new_empty(diagnostics, dtype=torch.float64),
            rhs.new_empty(diagnostics, dtype=torch.int64), rhs.new_empty(diagnostics, dtype=torch.bool),
            rhs.new_empty(diagnostics, dtype=torch.int64))


def _linear_svd_fake(inputs, counts, metrics, families, tensors, reals, integers,
                      flags, strings, real_lists, integer_lists, transposed,
                      source_counts, programs, factors, factor_counts, kinds,
                      integer_numbers, real_numbers, linear_program, linear_factors,
                      shape, anchor, k, subspace_size=None, max_iter=100,
                      rtol=1e-5, atol=0., initial=None):
    torch._check(len(shape) >= 2)
    torch._check(k >= 1)
    torch._check(k <= shape[-1])
    torch._check(k <= shape[-2])
    if subspace_size is not None:
        torch._check(subspace_size >= k)
        torch._check(subspace_size <= shape[-1])
        torch._check(subspace_size <= shape[-2])
    if initial is not None:
        torch._check(initial.ndim >= 2)
        torch._check(initial.shape[-2] == shape[-1])
        width = initial.shape[-1] if subspace_size is None else subspace_size
        torch._check(initial.shape[-1] == width)
        torch._check(width >= k)
        torch._check(width <= shape[-1])
        torch._check(width <= shape[-2])
        batch = torch.broadcast_shapes(shape[:-2], initial.shape[:-2])
        torch._check(len(batch) == len(shape)-2)
        for actual, expected in zip(batch, shape[:-2]):
            torch._check(actual == expected)
        if initial.dtype != anchor.dtype or initial.device != anchor.device:
            raise ValueError('svd initial must match operator dtype/device')
    if torch.is_grad_enabled():
        values = [*inputs, *factors, *linear_factors, *tensors.values(), anchor]
        if initial is not None:
            values.append(initial)
        if any(value.requires_grad for value in values):
            raise RuntimeError('svd is inference-only; use torch.no_grad() or detach inputs explicitly')
    pairs = (*shape[:-2], k)
    return (anchor.new_empty(pairs), anchor.new_empty((*shape[:-2], shape[-2], k)),
            anchor.new_empty((*shape[:-2], shape[-1], k)),
            anchor.new_empty(pairs, dtype=torch.float64), anchor.new_empty(pairs, dtype=torch.int64),
            anchor.new_empty(pairs, dtype=torch.bool), anchor.new_empty(pairs, dtype=torch.int64))


def _spectral_grad_check(inputs, factors, linear_factors, tensors, anchor, probes=None):
    values = [*inputs, *factors, *linear_factors, *tensors.values(), anchor]
    if probes is not None:
        values.append(probes)
    if torch.is_grad_enabled() and any(t.requires_grad for t in values):
        raise RuntimeError('Spectral consumers are inference-only; use torch.no_grad() or detach inputs explicitly')


def _linear_trace_fake(inputs, counts, metrics, families, tensors, reals, integers,
                       flags, strings, real_lists, integer_lists, transposed,
                       source_counts, programs, factors, factor_counts, kinds,
                       integer_numbers, real_numbers, linear_program, linear_factors,
                       shape, anchor, function, method, num_probes=None, probes=None,
                       seed=None, max_iter=None, block_size=16):
    torch._check(len(shape) >= 2)
    torch._check(shape[-1] == shape[-2])
    torch._check(block_size > 0)
    _spectral_grad_check(inputs, factors, linear_factors, tensors, anchor, probes)
    if probes is not None:
        torch._check(probes.ndim >= 2)
        torch._check(probes.shape[-2] == shape[-1])
        torch._check(probes.shape[-1] > 0)
        batch = torch.broadcast_shapes(shape[:-2], probes.shape[:-2])
        torch._check(len(batch) == len(shape) - 2)
        for actual, expected in zip(batch, shape[:-2]):
            torch._check(actual == expected)
        if probes.dtype != anchor.dtype or probes.device != anchor.device:
            raise ValueError('Probes must match operator dtype/device')
    count = 0 if method == 'exact' else probes.shape[-1] if probes is not None else num_probes
    if count is None:
        raise ValueError('Stochastic methods require num_probes or probes')
    if method != 'exact':
        torch._check(count > 0)
    batch = shape[:-2]
    return (anchor.new_empty(batch, dtype=torch.float64),
            anchor.new_empty(batch, dtype=torch.float64),
            anchor.new_empty((*batch, count), dtype=torch.float64),
            anchor.new_empty((*batch, count), dtype=torch.int64))


def _linear_trace_differentiable_fake(inputs, counts, metrics, families, tensors, reals, integers,
        flags, strings, real_lists, integer_lists, transposed, source_counts, programs,
        factors, factor_counts, kinds, integer_numbers, real_numbers, linear_program,
        linear_factors, shape, anchor, function, method, num_probes, probes, seed,
        max_iter, block_size, gradient, solve_rtol, solve_atol, solve_max_iter):
    if probes is not None and probes.requires_grad and torch.is_grad_enabled():
        raise RuntimeError('Spectral probes are fixed algorithmic inputs; detach probes explicitly')
    with torch.no_grad():
        result = _linear_trace_fake(inputs, counts, metrics, families, tensors, reals,
            integers, flags, strings, real_lists, integer_lists, transposed, source_counts,
            programs, factors, factor_counts, kinds, integer_numbers, real_numbers,
            linear_program, linear_factors, shape, anchor, function, method, num_probes,
            probes, seed, max_iter, block_size)
    tape_shape = (0,) if method == 'exact' else probes.shape if probes is not None else (*shape[:-2], shape[-1], num_probes)
    return (*result, anchor.new_empty(tape_shape))


def _linear_trace_backward_fake(inputs, counts, metrics, families, tensors, reals, integers,
        flags, strings, real_lists, integer_lists, transposed, source_counts, programs,
        factors, factor_counts, kinds, integer_numbers, real_numbers, linear_program,
        linear_factors, shape, anchor, probes, upstream, function, method, gradient,
        block_size, solve_rtol, solve_atol, solve_max_iter):
    return [v.new_empty(v.shape) for v in (*inputs, *factors, *linear_factors, *tensors.values())]


def _linear_function_fake(inputs, counts, metrics, families, tensors, reals, integers,
                          flags, strings, real_lists, integer_lists, transposed,
                          source_counts, programs, factors, factor_counts, kinds,
                          integer_numbers, real_numbers, linear_program, linear_factors,
                          shape, values, function, scale=1., max_iter=32, block_size=16):
    torch._check(len(shape) >= 2)
    torch._check(shape[-1] == shape[-2])
    torch._check(values.ndim >= 1)
    torch._check(max_iter > 0)
    torch._check(block_size > 0)
    _spectral_grad_check(inputs, factors, linear_factors, tensors, values)
    vector = values.ndim == 1
    torch._check(values.shape[-1 if vector else -2] == shape[-1])
    columns = 1 if vector else values.shape[-1]
    torch._check(columns > 0)
    batch = torch.broadcast_shapes(shape[:-2], () if vector else values.shape[:-2])
    output = (*batch, shape[-1]) if vector else (*batch, shape[-1], columns)
    diagnostics = batch if vector else (*batch, columns)
    return (values.new_empty(output), values.new_empty(diagnostics, dtype=torch.float64),
            values.new_empty(diagnostics, dtype=torch.int64))


def _linear_eigh_fake(inputs, counts, metrics, families, tensors, reals, integers,
                      flags, strings, real_lists, integer_lists, transposed,
                      source_counts, programs, factors, factor_counts, kinds,
                      integer_numbers, real_numbers, linear_program, linear_factors,
                      shape, anchor, k, subspace_size=None, max_iter=100,
                      rtol=1e-5, atol=0., initial=None):
    torch._check(len(shape) >= 2)
    torch._check(shape[-1] == shape[-2])
    torch._check(k >= 1)
    torch._check(k <= shape[-1])
    if subspace_size is not None:
        torch._check(subspace_size >= k)
        torch._check(subspace_size <= shape[-1])
    if initial is not None:
        torch._check(initial.ndim >= 2)
        torch._check(initial.shape[-2] == shape[-1])
        width = initial.shape[-1] if subspace_size is None else subspace_size
        torch._check(initial.shape[-1] == width)
        torch._check(width >= k)
        torch._check(width <= shape[-1])
        batch = torch.broadcast_shapes(shape[:-2], initial.shape[:-2])
        torch._check(len(batch) == len(shape)-2)
        for actual, expected in zip(batch, shape[:-2]):
            torch._check(actual == expected)
        if initial.dtype != anchor.dtype or initial.device != anchor.device:
            raise ValueError('eigh initial must match operator dtype/device')
    if torch.is_grad_enabled():
        values = [*inputs, *factors, *linear_factors, *tensors.values(), anchor]
        if initial is not None:
            values.append(initial)
        if any(value.requires_grad for value in values):
            raise RuntimeError('eigh is inference-only; use torch.no_grad() or detach inputs explicitly')
    pairs = (*shape[:-2], k)
    return (anchor.new_empty(pairs), anchor.new_empty((*shape[:-2], shape[-1], k)),
            anchor.new_empty(pairs, dtype=torch.float64), anchor.new_empty(pairs, dtype=torch.int64),
            anchor.new_empty(pairs, dtype=torch.bool), anchor.new_empty(pairs, dtype=torch.int64))


def _sinkhorn_solver_fake(inputs, counts, metrics, families, tensors, reals, integers,
                          flags, strings, real_lists, integer_lists, transposed,
                          source_counts, programs, factors, factor_counts, kinds,
                          integer_numbers, real_numbers, linear_program, linear_factors,
                          shape, a, b, epsilon, max_iter=1000, rtol=1e-6, atol=0.,
                          reach=None, epsilon_scaling=None, initial_f=None,
                          initial_g=None, check_every=10):
    torch._check(len(shape) >= 2)
    torch._check(a.ndim >= 1 and b.ndim >= 1)
    torch._check(a.shape[-1] == shape[-2])
    torch._check(b.shape[-1] == shape[-1])
    batch = torch.broadcast_shapes(shape[:-2], a.shape[:-1], b.shape[:-1])
    return (a.new_empty((*batch, shape[-2]), dtype=torch.float64),
            b.new_empty((*batch, shape[-1]), dtype=torch.float64),
            a.new_empty(batch, dtype=torch.int64),
            a.new_empty(batch, dtype=torch.bool),
            a.new_empty(batch, dtype=torch.int64),
            a.new_empty(batch, dtype=torch.float64))


def _box_qp_fake(inputs, counts, metrics, families, tensors, reals, integers,
                 flags, strings, real_lists, integer_lists, transposed,
                 source_counts, programs, factors, factor_counts, kinds,
                 integer_numbers, real_numbers, linear_program, linear_factors,
                 shape, q, lower, upper, equality=None, target=None, initial=None,
                 rtol=1e-5, atol=0., max_iter=1000):
    torch._check(q.ndim >= 1)
    torch._check(len(shape) >= 2)
    torch._check(shape[-1] == shape[-2])
    block = q.unsqueeze(-1) if q.ndim == 1 else q
    torch._check(block.shape[-2] == shape[-1])
    torch._check(block.shape[-1] > 0)
    batch = torch.broadcast_shapes(shape[:-2], block.shape[:-2])
    output = (*batch, shape[-1], block.shape[-1])
    pairs = (*batch, block.shape[-1])
    for value in (lower, upper, equality, initial):
        if value is None:
            continue
        candidate = value.unsqueeze(-1) if value.ndim == 1 else value
        actual = torch.broadcast_shapes(candidate.shape, output)
        torch._check(len(actual) == len(output))
        for a, b in zip(actual, output):
            torch._check(a == b)
        if value.dtype != q.dtype or value.device != q.device:
            raise ValueError('box_qp arguments must match q dtype/device')
    if (equality is None) != (target is None):
        raise ValueError('box_qp equality needs vector and target')
    if target is not None:
        actual = torch.broadcast_shapes(target.shape, pairs)
        torch._check(len(actual) == len(pairs))
        for a, b in zip(actual, pairs):
            torch._check(a == b)
        if target.dtype != q.dtype or target.device != q.device:
            raise ValueError('box_qp target must match q dtype/device')
    if torch.is_grad_enabled():
        values = [*inputs, *factors, *linear_factors, *tensors.values(), q, lower, upper]
        values += [v for v in (equality, target, initial) if v is not None]
        if any(v.requires_grad for v in values):
            raise RuntimeError('box_qp is inference-only; use torch.no_grad() or detach inputs explicitly')
    solution = (*batch, shape[-1]) if q.ndim == 1 else output
    diagnostics = batch if q.ndim == 1 else pairs
    return (q.new_empty(solution), q.new_empty(diagnostics, dtype=torch.float64),
            q.new_empty(diagnostics, dtype=torch.int64), q.new_empty(diagnostics, dtype=torch.bool),
            q.new_empty(diagnostics, dtype=torch.int64), q.new_empty(diagnostics, dtype=torch.float64),
            q.new_empty(diagnostics, dtype=torch.float64))


def _kernel_form_fake(x, y, left, right, *options):
    batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2], left.shape[:-2], right.shape[:-2])
    return x.new_empty((*batch, left.shape[-1], right.shape[-1]))


def _kernel_form_backward_fake(grad, x, y, left, right, *options):
    return tuple(t.new_empty(t.shape) for t in (x, y, left, right))


def _pair_shape(x, y, event_ndim=2):
    torch._check(x.ndim == y.ndim)
    torch._check(x.ndim in (event_ndim, event_ndim + 1))
    torch._check(x.shape[-1] == y.shape[-1])
    if x.ndim > event_ndim:
        torch._check(x.shape[0] == y.shape[0])
    return (*x.shape[:-event_ndim], x.shape[-event_ndim], y.shape[-event_ndim])


def _pair_fake(x1, x2, *args, **kwargs):
    return x1.new_empty(_pair_shape(x1, x2))


def _backward_fake(grad, x1, x2, *args, **kwargs):
    return x1.new_empty(x1.shape), x2.new_empty(x2.shape)


def _vector_compare_fake(x, y, grid, kind, mode, *options):
    batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    n, m = x.shape[-2], y.shape[-2]
    shape = (*batch, n, m) if mode == 0 else (*batch, n if mode == 1 else n * (n - 1) // 2)
    return x.new_empty(shape)


def _vector_compare_backward_fake(grad, x, y, grid, *options):
    return x.new_empty(x.shape), y.new_empty(y.shape), grid.new_empty(grid.shape)


def _weighted_compare_fake(x, y, w, kind, mode, p, self):
    batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2], w.shape[:-1])
    n, m = x.shape[-2], y.shape[-2]
    shape = (*batch, n, m) if mode == 0 else (*batch, n if mode == 1 else n * (n - 1) // 2)
    return w.new_empty(shape)


def _weighted_compare_backward_fake(grad, x, y, w, *options):
    return x.new_empty(x.shape), y.new_empty(y.shape), w.new_empty(w.shape)


def _transport_compare_fake(x, y, xl, yl, directions, kind, mode, *options):
    n, m = x.shape[-3], y.shape[-3]
    shape = (*x.shape[:-3], n, m) if mode == 0 else (*x.shape[:-3], n if mode == 1 else n * (n - 1) // 2)
    return x.new_empty(shape)


def _sequence_cost_fake(x, y, xl, yl, gamma, window, chunk_size, self):
    shape = (*xl.shape, 1) if self else torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    return x.new_empty(shape)


def _sequence_cost_backward_fake(grad, x, y, *options):
    return x.new_empty(x.shape), y.new_empty(y.shape)


def _mmd_features_fake(x, lengths, kernel, gamma, degree, coef0, scale, alpha, nu, statistic, unbiased, chunk_size=128):
    if kernel == 'linear' and not unbiased:
        return [x.new_empty((*x.shape[:-2], x.shape[-1])), x.new_empty((0,))]
    return [x.new_empty(x.shape), x.new_empty((*x.shape[:-2], 1))]


def _mmd_features_backward_fake(gradients, x, *options):
    return x.new_empty(x.shape)


def _mmd_compare_fake(x, y, xl, yl, kernel, mode, *options):
    return _modality_fake(x, y, mode, 3)


def _mmd_compare_backward_fake(grad, x, y, xl, yl, *options):
    return x.new_empty(x.shape), y.new_empty(y.shape)


def _transport_compare_backward_fake(grad, x, y, xl, yl, directions, *options):
    return x.new_empty(x.shape), y.new_empty(y.shape), directions.new_empty(directions.shape)


def _modality_fake(x, y, mode, rank):
    batch = torch.broadcast_shapes(x.shape[:-rank], y.shape[:-rank])
    n, m = x.shape[-rank], y.shape[-rank]
    shape = (*batch, n, m) if mode == 0 else (*batch, n if mode == 1 else n * (n - 1) // 2)
    return x.new_empty(shape)


def _image_fake(x, y, kind, mode, *options):
    return _modality_fake(x, y, mode, 4)


def _audio_fake(x, y, kind, mode, *options):
    return _modality_fake(x, y, mode, 3)


def _legacy_compare_fake(x, y, v, vi, name, mode, kernel, *options):
    n, m = x.shape[1], y.shape[1]
    shape = (x.shape[0], n, m) if mode == 0 else (x.shape[0], n if mode == 1 else n * (n - 1) // 2)
    return x.new_empty(shape, dtype=torch.float32 if not kernel and name in (
        'hamming', 'jaccard', 'kulsinski', 'kulczynski1', 'yule', 'dice',
        'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath') else x.dtype)


def _legacy_backward_fake(grad, x, y, v, vi, *options):
    return tuple(t.new_empty(t.shape) for t in (x, y, v, vi))


def _sequence_fake(x, y, lx, ly, mode, *options):
    n, m = x.shape[-3], y.shape[-3]
    shape = (*x.shape[:-3], n, m) if mode == 0 else (*x.shape[:-3], n if mode == 1 else n * (n - 1) // 2)
    return x.new_empty(shape)


def _token_fake(x, y, lx, ly, mode, *options):
    n, m = x.shape[-2], y.shape[-2]
    shape = (*x.shape[:-2], n, m) if mode == 0 else (*x.shape[:-2], n if mode == 1 else n * (n - 1) // 2)
    return x.new_empty(shape, dtype=torch.int64)


def _structured_fake(x, xc, xw, y, yc, yw, xl, yl, kind, mode, *options):
    n, m = x.shape[1], y.shape[1]
    shape = (x.shape[0], n, m) if mode == 0 else (x.shape[0], n if mode == 1 else n * (n - 1) // 2)
    return x.new_empty(shape)


def _media_features_fake(x, kind, settings, parameters):
    if kind < 3:
        c, h, w, window = settings
        count = c*h*w
        if kind:
            for _ in range(len(parameters)-4 if kind == 2 else 1):
                count += c*h*w + 3*c*(h-window+1)*(w-window+1) + c
                h, w = h//2, w//2
        rank = 4
    else:
        c, t = settings[:2]
        rank = 3
        count = c*t if kind <= 4 else 0
        if kind > 4:
            for k in range(7, len(settings), 3):
                n, hop, win = settings[k:k+3]
                frames = 1 + (t + (2*(n//2) if settings[4] else 0) - n)//hop
                count += c*(n//2+1)*frames
            if kind == 7:
                count *= 2
    torch._check(x.ndim >= rank)
    for axis in range(1, rank):
        torch._check(x.shape[axis-rank] == settings[axis-1])
    return x.new_empty((*x.shape[:-rank+1], count))


def _media_features_backward_fake(gradient, x, *settings):
    return x.new_empty(x.shape)


def _structured_features_fake(x, covariance, weights, lengths, kind, epsilon, iterations, component_tile):
    if kind < 4:
        return [x.new_empty(x.shape), (covariance if kind == 0 else x).new_empty((covariance if kind == 0 else x).shape),
                x.new_empty((0,)), x.new_empty((0,))]
    return [x.new_empty(x.shape), covariance.new_empty(covariance.shape), weights.new_empty(weights.shape),
            x.new_empty((*weights.shape[:-1], 1)) if kind in (5, 6, 9) else x.new_empty((0,))]


def _structured_features_backward_fake(gradients, x, covariance, weights, lengths, *options):
    return [t.new_empty(t.shape) for t in (x, covariance, weights)]


def _structured_backward_fake(grad, x, xc, xw, y, yc, yw, *options):
    return tuple(t.new_empty(t.shape) for t in (x, xc, xw, y, yc, yw))


def _weighted_backward_fake(grad, x1, x2, w, p):
    return (*_backward_fake(grad, x1, x2), w.new_empty(w.shape))


def _mahalanobis_backward_fake(grad, x1, x2, VI):
    return (*_backward_fake(grad, x1, x2), VI.new_empty(VI.shape))


def _hausdorff_fake(x1, x2, *, shuffle=False, generator=None):
    shape = _pair_shape(x1, x2, 3)
    return (x1.new_empty(shape), x1.new_empty(shape, dtype=torch.int64),
            x1.new_empty(shape, dtype=torch.int64))


def _pointset_nn_fake(x, y, x_lengths, y_lengths):
    torch._check(x.ndim == 4)
    shape = (*_pair_shape(x, y, 3), x.shape[-2])
    torch._check(x_lengths.shape == x.shape[:2])
    torch._check(y_lengths.shape == y.shape[:2])
    return x.new_empty(shape), x.new_empty(shape, dtype=torch.int64)


def _pointset_compare_fake(x, y, xl, yl, kind, mode, squared, size, legacy):
    n, m = x.shape[1], y.shape[1]
    shape = (x.shape[0], n, m) if mode == 0 else (x.shape[0], n if mode == 1 else n * (n - 1) // 2)
    indices_shape = shape if kind == 2 else (0,)
    return x.new_empty(shape), x.new_empty(indices_shape, dtype=torch.int64), x.new_empty(indices_shape, dtype=torch.int64)


def _nan_squared_fake(x, y):
    shape = _pair_shape(x, y)
    return x.new_empty(shape), x.new_empty(shape, dtype=torch.int64)


def _binary_fake(x1, x2):
    return x1.new_empty((*_pair_shape(x1, x2), x1.shape[-1]), dtype=torch.bool)


def _binary_reduction_fake(x1, x2):
    return x1.new_empty(_pair_shape(x1, x2), dtype=torch.float32)


def _division_fake(input, other, mode='zero'):
    tensor = input if isinstance(input, torch.Tensor) else other
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        torch._check(input.shape == other.shape)
    dtype = torch.result_type(input, other)
    if not dtype.is_floating_point:
        dtype = torch.float32
    return tensor.new_empty(tensor.shape, dtype=dtype)


def _division_backward_fake(grad, self, other, mode='zero'):
    tensors = [value for value in (self, other) if isinstance(value, torch.Tensor)]
    result = tuple(value.new_empty(value.shape) for value in tensors)
    return result if len(result) == 2 else result[0]


def _streaming_reduce_fake(operands, metric, operation, kernel, rank, k,
                           largest, exclusion, self_cosine, nan_first,
                           size, dtype, mask, reals, integers, flags, family):
    x, y = operands[:2]
    shapes = [x.shape[:-rank], y.shape[:-rank]]
    if family == '_structured':
        feature_rank = 3 if integers[0] < 4 else 4
        events = ((2, feature_rank), (3, feature_rank), (4, 2), (5, 2), (6, 2), (7, 2), (9, 2))
    elif family in ('_sequence', '_mmd'):
        events = ((2, 2), (3, 2), (7, 1), (8, 1), (9, 2))
    else:
        events = ((2, 1), (3, 1), (4, 2), (7, 1), (8, 1), (9, 2))
    for slot, event in events:
        if mask & (1 << slot):
            shapes.append(operands[slot].shape[:-event])
    batch = torch.broadcast_shapes(*shapes)
    transposed = operation.startswith('transpose_')
    operation = operation[10:] if transposed else operation
    shape = (*batch, (y if transposed else x).shape[-rank])
    if operation == 'neighbors':
        shape = (*shape, k)
    elif operation == 'entries':
        shape = (*shape, (x if transposed else y).shape[-rank])
    elif operation in ('kernel_sum', 'matmul'):
        shape = (*shape, operands[9].shape[-1])
    numeric_dtype = {0: torch.uint8, 1: torch.int8, 2: torch.int16, 3: torch.int32, 4: torch.int64, 6: torch.float32, 7: torch.float64, 11: torch.bool}[dtype]
    result = x.new_empty(shape, dtype=torch.bool if operation in ('all', 'any') else numeric_dtype)
    indices = x.new_empty(shape if operation == 'neighbors' else (0,), dtype=torch.int64)
    zeros = x.new_empty(shape if operation == 'prod' else (0,), dtype=torch.int64)
    nonzero = x.new_empty(shape, dtype=numeric_dtype) if operation == 'prod' else x.new_empty((0,))
    return result, indices, zeros, nonzero


def _pairwise_form_fake(operands, left, right, metric, operation, kernel, rank, k,
                        largest, exclusion, self_cosine, nan_first, size, dtype,
                        mask, reals, integers, flags, family):
    result = _streaming_reduce_fake(operands, metric, operation, kernel, rank, k,
        largest, exclusion, self_cosine, nan_first, size, dtype, mask, reals, integers, flags, family)[0]
    batch = torch.broadcast_shapes(result.shape[:-1], left.shape[:-2], right.shape[:-2])
    return left.new_empty((*batch, left.shape[-1], right.shape[-1]))


def _pairwise_form_backward_fake(grad, operands, left, right, *options):
    return [t.new_empty(t.shape) for t in (*operands, left, right)]


def _streaming_reduce_backward_fake(grad, operands, result, indices, zeros, nonzero, *options):
    return [value.new_empty(value.shape) for value in operands]


def _custom_compare_fake(inputs, names, shapes, mode, rank, dtype, size):
    x, y = inputs[:2]
    batch = torch.broadcast_shapes(x.shape[:-rank], y.shape[:-rank])
    n, m = x.shape[-rank], y.shape[-rank]
    shape = (*batch, n, m) if mode == 0 else (*batch, n if mode == 1 else n*(n-1)//2)
    return x.new_empty(shape, dtype={5: torch.float16, 6: torch.float32, 7: torch.float64, 15: torch.bfloat16}[dtype])


def _custom_compare_backward_fake(gradient, inputs, *options):
    return [t.new_empty(t.shape) for t in inputs]


def _expression_consume_fake(operands, names, families, headers, reals, integers,
                             flags, nodes, numbers, operation, dtype, size,
                             exclusion, transposed, form):
    largest = operation == 'maximum'
    op = 'neighbors' if largest else operation
    options = ('_expression', ('transpose_' if transposed else '') + op,
               False, 2, 1, largest, exclusion, False,
               op == 'neighbors', size, dtype,
               3 | ((1 << 9) if op == 'matmul' else 0), [], [], [], '_expression')
    if form:
        value = _pairwise_form_fake(operands[:-2], operands[-2], operands[-1], *options)
        return value, value.new_empty((0,)), value.new_empty((0,)), value.new_empty((0,))
    return _streaming_reduce_fake(operands, *options)


def _expression_topk_fake(operands, names, families, headers, reals, integers,
        flags, nodes, numbers, operation, dtype, size, exclusion, transposed,
        form, k, largest):
    torch._check(k >= 0)
    torch._check(k <= operands[0 if transposed else 1].shape[-2] - int(exclusion))
    return _streaming_reduce_fake(operands, '_expression',
        ('transpose_' if transposed else '') + 'neighbors', False, 2, k,
        largest, exclusion, False, largest, size, dtype, 3, [], [], [], '_expression')


def _expression_selection_fake(operands, names, families, headers, reals, integers,
        flags, nodes, numbers, operation, dtype, size, exclusion, transposed,
        form, q, interpolation):
    torch._check(operands[0 if transposed else 1].shape[-2] - int(exclusion) > 0)
    shape = (*torch.broadcast_shapes(operands[0].shape[:-2], operands[1].shape[:-2]),
             operands[1 if transposed else 0].shape[-2])
    value = operands[0].new_empty(shape)
    return value, value.new_empty((*shape, 2), dtype=torch.int64)


def _expression_statistics_fake(operands, names, families, headers, reals, integers,
        flags, nodes, numbers, operation, dtype, size, exclusion, transposed,
        global_reduction, correction):
    shape = torch.broadcast_shapes(operands[0].shape[:-2], operands[1].shape[:-2])
    if not global_reduction:
        shape = (*shape, operands[1 if transposed else 0].shape[-2])
    result = operands[0].new_empty(shape, dtype=torch.int64 if operation == 'count_nonzero' else operands[0].dtype)
    center = operands[0].new_empty(shape, dtype=torch.float64) if operation in ('var', 'std') else operands[0].new_empty((0,))
    return result, center


def _expression_anchored_statistics_fake(operands, names, families, headers, reals,
        integers, flags, nodes, numbers, operation, dtype, size, exclusion,
        transposed, global_reduction, correction):
    result, center = _expression_statistics_fake(operands, names, families,
        headers, reals, integers, flags, nodes, numbers, operation, dtype, size,
        exclusion, transposed, global_reduction, correction)
    if operation in ('var', 'std'):
        center = center.new_empty((*center.shape, 2))
    return result, center


def _expression_weighted_statistics_fake(operands, names, families, headers, reals,
        integers, flags, nodes, numbers, operation, dtype, size, exclusion,
        transposed, global_reduction, correction, value_root, weight_root):
    shape = torch.broadcast_shapes(operands[0].shape[:-2],operands[1].shape[:-2])
    if not global_reduction:
        shape = (*shape,operands[1 if transposed else 0].shape[-2])
    return (operands[0].new_empty(shape),
            operands[0].new_empty((*shape,5),dtype=torch.float64))


def _expression_statistics_backward_fake(grad, operands, result, center, *options):
    return [torch.empty_like(t, memory_format=torch.contiguous_format) for t in operands]


def _expression_product_fake(operands, names, families, headers, reals, integers,
        flags, nodes, numbers, operation, dtype, size, exclusion, transposed,
        global_reduction, correction):
    shape = torch.broadcast_shapes(operands[0].shape[:-2], operands[1].shape[:-2])
    if not global_reduction:
        shape = (*shape, operands[1 if transposed else 0].shape[-2])
    return (operands[0].new_empty(shape),
            operands[0].new_empty((*shape, 5), dtype=torch.float64))


def _expression_inner_product_fake(operands, names, families, headers, reals,
        integers, flags, nodes, numbers, operation, dtype, size, exclusion,
        transposed, form, left_root, right_root, center, normalize):
    batch = torch.broadcast_shapes(operands[0].shape[:-2], operands[1].shape[:-2])
    n, m = operands[0].shape[-2], operands[1].shape[-2]
    if operation in ('hsic', 'hsic_unbiased'):
        torch._check(n == m)
        torch._check(n >= (4 if operation == 'hsic_unbiased' else 2))
    width = 2 * (n + m) + 9 if center else 5
    return (operands[0].new_empty(batch),
            operands[0].new_empty((*batch, width), dtype=torch.float64))


def _expression_anchored_weighted_statistics_fake(*args):
    result, state = _expression_weighted_statistics_fake(*args)
    return result, state.new_empty((*result.shape, 6 if result.dtype == torch.float64 else 5))


def _expression_softmax_matmul_fake(operands, names, families, headers, reals,
        integers, flags, nodes, numbers, operation, dtype, size, exclusion,
        transposed, form):
    batch = torch.broadcast_shapes(operands[0].shape[:-2], operands[1].shape[:-2],
                                   operands[9].shape[:-2])
    n = operands[1 if transposed else 0].shape[-2]
    return (operands[0].new_empty((*batch, n, operands[9].shape[-1])),
            operands[0].new_empty((*batch, n, 2), dtype=torch.float64))


def _residual_features_fake(x, y, reduction, transform, correction, size, paired):
    torch._check(x.shape[-1] == y.shape[-1])
    if paired:
        torch._check(x.shape[0] == y.shape[0])
    shape = (x.shape[0],) if paired else (x.shape[0], y.shape[0])
    width = 5 if reduction == 'prod' else 1 if reduction in ('all', 'any') else 2
    state = (*shape, width) if reduction not in ('sum', 'mean') else (0,)
    return x.new_empty(shape, dtype=torch.bool if reduction in ('all', 'any') else x.dtype), x.new_empty(state, dtype=torch.float64)


def _residual_topk_fake(x, y, k, largest, transform, sample_size, feature_size):
    torch._check(x.shape[-1] == y.shape[-1])
    torch._check(k >= 0)
    torch._check(k <= x.shape[-1])
    batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    shape = (*batch, x.shape[-2], y.shape[-2], k)
    return x.new_empty(shape), x.new_empty(shape, dtype=torch.int64)


def _residual_topk_backward_fake(grad, x, y, indices, *options):
    return x.new_empty(x.shape), y.new_empty(y.shape)


def _residual_features_backward_fake(grad, x, y, state, *options):
    return torch.empty_like(x), torch.empty_like(y)


def _residual_features_anchored_fake(x, y, reduction, *args):
    result, state = _residual_features_fake(x, y, reduction, *args)
    if reduction in ('var', 'std') and x.dtype == torch.float64:
        state = state.new_empty((*state.shape[:-1], 3))
    return result, state


def _residual_reduce_fake(x, y, reduction, transform, axes, keepdim,
                          correction, sample_size, feature_size):
    batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    torch._check(x.shape[-1] == y.shape[-1])
    logical = (x.shape[-2], y.shape[-2], x.shape[-1])
    kept = tuple(1 if axis in axes else logical[axis] for axis in range(3))
    shape = (*batch, *(kept if keepdim else tuple(logical[axis] for axis in range(3) if axis not in axes)))
    width = 3 if reduction in ('var', 'std') else 5 if reduction == 'prod' else 1 if reduction in ('all', 'any') else 2
    state = (*batch, *kept, width) if reduction not in ('sum', 'mean') else (0,)
    return x.new_empty(shape, dtype=torch.bool if reduction in ('all', 'any') else x.dtype), x.new_empty(state, dtype=torch.float64)


def _expression_consume_backward_fake(grad, operands, result, indices, zeros, nonzero, *options):
    return [t.new_empty(t.shape) for t in operands]


def _residual_reduce_anchored_fake(x, y, reduction, *args):
    result, state = _residual_reduce_fake(x, y, reduction, *args)
    if reduction in ('var', 'std') and x.dtype == torch.float64:
        state = state.new_empty((*state.shape[:-1], 4))
    return result, state


def _transport_map_reduce_fake(x, y, logw, p, epsilon, size, barycentric):
    return x.new_empty(x.shape if barycentric else x.shape[:-1])


def _transport_map_reduce_backward_fake(grad, x, y, logw, result, *options):
    return tuple(t.new_empty(t.shape) for t in (x, y, logw))


def _image_map_values_fake(x, y, comparison, settings, parameters, dtype, paired):
    crop = settings[3]-1 if comparison == 3 else 0
    shape = (x.shape[0],) if paired else (x.shape[0], y.shape[0])
    return x.new_empty((*shape, settings[0], settings[1]-crop, settings[2]-crop),
                       dtype=torch.float32 if dtype == 6 else torch.float64)


def _image_map_contract_fake(x, y, comparison, settings, parameters, dtype,
                              code, mask, keepdim, correction, paired):
    crop = settings[3]-1 if comparison == 3 else 0
    event = (settings[0], settings[1]-crop, settings[2]-crop)
    samples = (x.shape[0],) if paired else (x.shape[0], y.shape[0])
    internal = tuple(1 if mask & (1 << d) else size for d, size in enumerate(event))
    output = internal if keepdim else tuple(size for d, size in enumerate(event) if not mask & (1 << d))
    out_dtype = torch.bool if code >= 7 else torch.float32 if dtype == 6 else torch.float64
    width = 1 if code >= 7 else 7 if code == 6 else 2 if code >= 4 else 3
    return (x.new_empty((*samples, *output), dtype=out_dtype),
            x.new_empty((*samples, *internal, width)))


def _image_event_reduce_fake(x, y, comparison, settings, parameters, dtype,
                             program, corrections, reduction, axes, keepdim,
                             correction, chunk_size, paired):
    crop = settings[3]-1 if comparison == 3 else 0
    event = [settings[0], settings[1]-crop, settings[2]-crop]
    out_dtype = torch.float32 if dtype == 6 else torch.float64
    for k in range(len(corrections)):
        code, mask, keep = program[3*k:3*k+3]
        event = [1 if mask & (1 << d) else size for d, size in enumerate(event)] if keep else [size for d, size in enumerate(event) if not mask & (1 << d)]
        if code in (7, 8):
            out_dtype = torch.bool
        elif code in (0, 6) and out_dtype in (torch.bool, torch.int64):
            out_dtype = torch.int64
    shape = [x.shape[0], *event] if paired else [*torch.broadcast_shapes(x.shape[:-2], y.shape[:-2]), x.shape[-2], y.shape[-2], *event]
    if not reduction:
        return x.new_empty(shape, dtype=out_dtype), x.new_empty((0,))
    integral = not out_dtype.is_floating_point
    if reduction in ('all', 'any'):
        result_dtype = torch.bool
    elif reduction in ('sum', 'prod') and integral:
        result_dtype = torch.int64
    else:
        result_dtype = out_dtype
    batch_rank = len(shape)-2-len(event)
    internal = [1 if d-batch_rank in axes else size for d, size in enumerate(shape)]
    result_shape = internal if keepdim else [size for d, size in enumerate(shape) if d-batch_rank not in axes]
    extra = reduction in ('min', 'max', 'prod', 'all', 'any')
    stats = reduction in ('var', 'std')
    width = 1 if integral else 5 if reduction == 'prod' else 1 if reduction in ('all', 'any') else 2 if extra else 3
    state_shape = (*internal, width) if extra or stats else (0,)
    return (x.new_empty(result_shape, dtype=result_dtype),
            x.new_empty(state_shape, dtype=torch.int64 if integral else torch.float64))


def _image_map_contract_anchored_fake(x, y, comparison, settings, parameters,
        dtype, code, *args):
    result, state = _image_map_contract_fake(x, y, comparison, settings,
        parameters, dtype, code, *args)
    if code in (2, 3) and result.dtype == torch.float64:
        state = state.new_empty((*state.shape[:-1], 4))
    return result, state


def _image_event_topk_fake(x, y, comparison, settings, parameters, dtype,
                           program, corrections, k, axis, largest, chunk_size):
    values, _ = _image_event_reduce_fake(
        x, y, comparison, settings, parameters, dtype, program, corrections,
        '', [], False, 1., chunk_size, False)
    batch_rank = len(torch.broadcast_shapes(x.shape[:-2], y.shape[:-2]))
    shape = list(values.shape)
    torch._check(k >= 0)
    torch._check(k <= shape[batch_rank + 2 + axis])
    shape[batch_rank + 2 + axis] = k
    return x.new_empty(shape, dtype=values.dtype), x.new_empty(shape, dtype=torch.int64)


def _image_event_reduce_anchored_fake(x, y, comparison, settings, parameters,
        dtype, program, corrections, reduction, *args):
    result, state = _image_event_reduce_fake(x, y, comparison, settings,
        parameters, dtype, program, corrections, reduction, *args)
    if reduction in ('var', 'std') and result.dtype == torch.float64:
        state = state.new_empty((*state.shape[:-1], 4))
    return result, state


def _image_map_reduce_fake(x, y, comparison, reduction, settings, parameters,
                           dtype, correction, paired):
    torch._check(x.ndim == 2)
    torch._check(y.ndim == 2)
    if paired:
        torch._check(x.shape[0] == y.shape[0])
    shape = (x.shape[0],) if paired else (x.shape[0], y.shape[0])
    return x.new_empty(shape, dtype=torch.bool if reduction in ('all', 'any') else
                       torch.float32 if dtype == 6 else torch.float64)


def _image_map_fake(x, y, kind, mode, size, window, parameters, single_precision=False):
    batch = torch.broadcast_shapes(x.shape[:-4], y.shape[:-4])
    n, m = x.shape[-4], y.shape[-4]
    shape = (*batch, n, m) if mode == 0 else (*batch, n*(n-1)//2 if mode == 2 else n)
    crop = window-1 if kind == 3 else 0
    shape = (*shape, x.shape[-3], x.shape[-2]-crop, x.shape[-1]-crop)
    return x.new_empty(shape), x.new_empty(shape if mode >= 5 else (0,), dtype=torch.int64)


def _image_map_backward_fake(grad, x, y, indices, *options):
    return x.new_empty(x.shape), y.new_empty(y.shape)


def _kernel_moments_fake(x, y, origins, kernel, size, reals, degree, second=True):
    batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    batch = torch.broadcast_shapes(batch, origins.shape[:-2])
    n, d = x.shape[-2:]
    return (x.new_empty((*batch,n,d,d) if second else (0,), dtype=torch.float64),
            x.new_empty((*batch,n,d), dtype=torch.float64),
            x.new_empty((*batch,n), dtype=torch.float64))


def _kernel_moments_backward_fake(gs, gf, gm, x, y, origins, *options):
    return x.new_empty(x.shape), y.new_empty(y.shape), origins.new_empty(origins.shape)


def _streamed_lowrank_metadata(inputs, tensors, factors, linear_factors,
                              linear_program, shape, anchor, k, initial=None,
                              entrywise=False):
    torch._check(len(shape) >= 2)
    torch._check(shape[-2] == shape[-1])
    torch._check(k >= 1)
    torch._check(k <= shape[-1])
    if (anchor.layout != torch.strided or anchor.ndim != 0
            or anchor.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Low-rank and greedy setup requires a scalar float32/float64 anchor')
    if entrywise and (len(linear_program) != 3 or linear_program[0] != 0):
        raise ValueError('Use a scalar entrywise source, not a linear-only composition')
    if initial is not None:
        if initial.layout != torch.strided:
            raise ValueError('Lanczos initial must be strided')
        torch._check(initial.ndim >= 1)
        torch._check(initial.shape[-1] == shape[-1])
        batch = torch.broadcast_shapes(shape[:-2], initial.shape[:-1])
        torch._check(len(batch) == len(shape)-2)
        for actual, expected in zip(batch, shape[:-2]):
            torch._check(actual == expected)
        if initial.dtype != anchor.dtype or initial.device != anchor.device:
            raise ValueError('Lanczos initial must match operator dtype/device')
    if torch.is_grad_enabled():
        values = [*inputs, *tensors.values(), *factors, *linear_factors, anchor]
        if initial is not None:
            values.append(initial)
        if any(value.requires_grad for value in values):
            raise RuntimeError('Low-rank and greedy setup is inference-only; use torch.no_grad() or detach inputs explicitly')


def _linear_pivoted_cholesky_fake(inputs, counts, metrics, families, tensors, reals,
                                 integers, flags, strings, real_lists, integer_lists,
                                 transposed, source_counts, programs, factors,
                                 factor_counts, kinds, integer_numbers, real_numbers,
                                 linear_program, linear_factors, shape, anchor, rank,
                                 rtol=0., atol=0.):
    torch._check(0 <= rtol < float('inf'))
    torch._check(0 <= atol < float('inf'))
    _streamed_lowrank_metadata(inputs, tensors, factors, linear_factors,
                              linear_program, shape, anchor, rank, entrywise=True)
    return (anchor.new_empty((*shape[:-2], shape[-1], rank)),
            anchor.new_empty((*shape[:-2], rank), dtype=torch.int64),
            anchor.new_empty(shape[:-2]), anchor.new_empty(shape[:-2], dtype=torch.int64))


def _linear_lanczos_fake(inputs, counts, metrics, families, tensors, reals, integers,
                         flags, strings, real_lists, integer_lists, transposed,
                         source_counts, programs, factors, factor_counts, kinds,
                         integer_numbers, real_numbers, linear_program, linear_factors,
                         shape, anchor, k, initial, reorthogonalize=True):
    _streamed_lowrank_metadata(inputs, tensors, factors, linear_factors,
                              linear_program, shape, anchor, k, initial)
    return (anchor.new_empty((*shape[:-2], shape[-1], k)),
            anchor.new_empty((*shape[:-2], k)), anchor.new_empty((*shape[:-2], k-1)),
            anchor.new_empty(shape[:-2], dtype=torch.int64),
            anchor.new_empty(shape[:-2], dtype=torch.int64))


def _linear_love_cache_fake(inputs, counts, metrics, families, tensors, reals, integers,
                            flags, strings, real_lists, integer_lists, transposed,
                            source_counts, programs, factors, factor_counts, kinds,
                            integer_numbers, real_numbers, linear_program, linear_factors,
                            shape, anchor, rank, initial):
    _streamed_lowrank_metadata(inputs, tensors, factors, linear_factors,
                              linear_program, shape, anchor, rank, initial)
    return anchor.new_empty((*shape[:-2], shape[-1], rank))


def _linear_farthest_point_sampling_fake(inputs, counts, metrics, families, tensors,
                                        reals, integers, flags, strings, real_lists,
                                        integer_lists, transposed, source_counts,
                                        programs, factors, factor_counts, kinds,
                                        integer_numbers, real_numbers, linear_program,
                                        linear_factors, shape, anchor, k, start=0):
    _streamed_lowrank_metadata(inputs, tensors, factors, linear_factors,
                              linear_program, shape, anchor, k, entrywise=True)
    torch._check(start >= 0)
    torch._check(start < shape[-1])
    return (anchor.new_empty((*shape[:-2], k), dtype=torch.int64),
            anchor.new_empty((*shape[:-2], k)))


def _linear_kmeans_plusplus_fake(inputs, counts, metrics, families, tensors, reals,
                                integers, flags, strings, real_lists, integer_lists,
                                transposed, source_counts, programs, factors,
                                factor_counts, kinds, integer_numbers, real_numbers,
                                linear_program, linear_factors, shape, anchor, k,
                                generator=None):
    _streamed_lowrank_metadata(inputs, tensors, factors, linear_factors,
                              linear_program, shape, anchor, k, entrywise=True)
    if generator is not None and generator.device != anchor.device:
        raise ValueError('kmeans++ generator must match the operator device')
    return anchor.new_empty((*shape[:-2], k), dtype=torch.int64)


def _register():
    torch.library.register_fake('torchpairwise::linear_pivoted_cholesky')(_linear_pivoted_cholesky_fake)
    torch.library.register_fake('torchpairwise::linear_lanczos')(_linear_lanczos_fake)
    torch.library.register_fake('torchpairwise::linear_love_cache')(_linear_love_cache_fake)
    torch.library.register_fake('torchpairwise::linear_farthest_point_sampling')(_linear_farthest_point_sampling_fake)
    torch.library.register_fake('torchpairwise::linear_kmeans_plusplus')(_linear_kmeans_plusplus_fake)
    torch.library.register_fake('torchpairwise::sinkhorn_solver')(_sinkhorn_solver_fake)
    torch.library.register_fake('torchpairwise::box_qp')(_box_qp_fake)
    torch.library.register_fake('torchpairwise::linear_svd')(_linear_svd_fake)
    torch.library.register_fake('torchpairwise::linear_eigh')(_linear_eigh_fake)
    torch.library.register_fake('torchpairwise::linear_trace')(_linear_trace_fake)
    torch.library.register_fake('torchpairwise::linear_trace_differentiable')(_linear_trace_differentiable_fake)
    torch.library.register_fake('torchpairwise::linear_trace_backward')(_linear_trace_backward_fake)
    torch.library.register_fake('torchpairwise::linear_function')(_linear_function_fake)
    torch.library.register_fake('torchpairwise::linear_implicit_pcg')(_linear_implicit_solve_fake)
    torch.library.register_fake('torchpairwise::linear_implicit_pcg_backward')(_linear_implicit_solve_backward_fake)
    torch.library.register_fake('torchpairwise::linear_implicit_lstsq')(_linear_implicit_lstsq_fake)
    torch.library.register_fake('torchpairwise::linear_implicit_lstsq_backward')(_linear_implicit_lstsq_backward_fake)
    torch.library.register_fake('torchpairwise::linear_implicit_solve')(_linear_implicit_solve_fake)
    torch.library.register_fake('torchpairwise::linear_implicit_solve_backward')(_linear_implicit_solve_backward_fake)
    torch.library.register_fake('torchpairwise::linear_cg')(_linear_cg_fake)
    torch.library.register_fake('torchpairwise::linear_pcg')(_linear_gmres_fake)
    torch.library.register_fake('torchpairwise::linear_lowrank_setup')(_linear_lowrank_setup_fake)
    torch.library.register_fake('torchpairwise::linear_nystrom_setup')(_linear_nystrom_setup_fake)
    torch.library.register_fake('torchpairwise::linear_lsqr')(_linear_lsqr_fake)
    torch.library.register_fake('torchpairwise::linear_gmres')(_linear_gmres_fake)
    torch.library.register_fake('torchpairwise::linear_minres')(_linear_minres_fake)
    torch.library.register_fake('torchpairwise::_custom_compare')(_custom_compare_fake)
    torch.library.register_fake('torchpairwise::_custom_compare_backward')(_custom_compare_backward_fake)
    torch.library.register_fake('torchpairwise::_image_map_contract')(_image_map_contract_fake)
    torch.library.register_fake('torchpairwise::_image_map_contract.anchored')(_image_map_contract_anchored_fake)
    torch.library.register_fake('torchpairwise::_image_map_contract_backward')(_backward_fake)
    torch.library.register_fake('torchpairwise::_image_map_values')(_image_map_values_fake)
    torch.library.register_fake('torchpairwise::_image_map_values_backward')(_backward_fake)
    torch.library.register_fake('torchpairwise::_image_event_reduce')(_image_event_reduce_fake)
    torch.library.register_fake('torchpairwise::_image_event_topk')(_image_event_topk_fake)
    torch.library.register_fake('torchpairwise::_image_event_topk_backward')(_residual_topk_backward_fake)
    torch.library.register_fake('torchpairwise::_image_event_reduce.anchored')(_image_event_reduce_anchored_fake)
    torch.library.register_fake('torchpairwise::_image_event_reduce_backward')(_residual_features_backward_fake)
    torch.library.register_fake('torchpairwise::_image_map_reduce')(_image_map_reduce_fake)
    torch.library.register_fake('torchpairwise::_image_map_reduce_backward')(_backward_fake)
    torch.library.register_fake('torchpairwise::_expression_selection')(_expression_selection_fake)
    torch.library.register_fake('torchpairwise::_expression_selection_backward')(_expression_statistics_backward_fake)
    torch.library.register_fake('torchpairwise::_expression_consume.topk')(_expression_topk_fake)
    torch.library.register_fake('torchpairwise::_expression_consume_backward.topk')(_expression_consume_backward_fake)
    torch.library.register_fake('torchpairwise::_residual_reduce')(_residual_reduce_fake)
    torch.library.register_fake('torchpairwise::_residual_topk')(_residual_topk_fake)
    torch.library.register_fake('torchpairwise::_residual_topk_backward')(_residual_topk_backward_fake)
    torch.library.register_fake('torchpairwise::_residual_reduce.anchored')(_residual_reduce_anchored_fake)
    torch.library.register_fake('torchpairwise::_residual_reduce_backward')(_residual_features_backward_fake)
    torch.library.register_fake('torchpairwise::_residual_features')(_residual_features_fake)
    torch.library.register_fake('torchpairwise::_residual_features.anchored')(_residual_features_anchored_fake)
    torch.library.register_fake('torchpairwise::_residual_features_backward')(_residual_features_backward_fake)
    torch.library.register_fake('torchpairwise::_expression_softmax_matmul')(_expression_softmax_matmul_fake)
    torch.library.register_fake('torchpairwise::_expression_softmax_matmul_backward')(_expression_statistics_backward_fake)
    torch.library.register_fake('torchpairwise::_expression_product')(_expression_product_fake)
    torch.library.register_fake('torchpairwise::_expression_product_backward')(_expression_statistics_backward_fake)
    torch.library.register_fake('torchpairwise::_expression_inner_product')(_expression_inner_product_fake)
    torch.library.register_fake('torchpairwise::_expression_inner_product_backward')(_expression_statistics_backward_fake)
    torch.library.register_fake('torchpairwise::_expression_statistics.weighted')(_expression_weighted_statistics_fake)
    torch.library.register_fake('torchpairwise::_expression_statistics.weighted_anchored')(_expression_anchored_weighted_statistics_fake)
    torch.library.register_fake('torchpairwise::_expression_statistics_backward.weighted')(_expression_statistics_backward_fake)
    torch.library.register_fake('torchpairwise::_expression_statistics')(_expression_statistics_fake)
    torch.library.register_fake('torchpairwise::_expression_statistics.anchored')(_expression_anchored_statistics_fake)
    torch.library.register_fake('torchpairwise::_expression_statistics_backward')(_expression_statistics_backward_fake)
    torch.library.register_fake('torchpairwise::_expression_consume')(_expression_consume_fake)
    torch.library.register_fake('torchpairwise::_expression_consume_backward')(_expression_consume_backward_fake)
    torch.library.register_fake('torchpairwise::_sequence_operator_cost')(_sequence_cost_fake)
    torch.library.register_fake('torchpairwise::_sequence_operator_cost_backward')(_sequence_cost_backward_fake)
    torch.library.register_fake('torchpairwise::_mmd_features')(_mmd_features_fake)
    torch.library.register_fake('torchpairwise::_mmd_features_backward')(_mmd_features_backward_fake)
    torch.library.register_fake('torchpairwise::_media_features')(_media_features_fake)
    torch.library.register_fake('torchpairwise::_media_features_backward')(_media_features_backward_fake)
    torch.library.register_fake('torchpairwise::_structured_features')(_structured_features_fake)
    torch.library.register_fake('torchpairwise::_structured_features_backward')(_structured_features_backward_fake)
    torch.library.register_fake('torchpairwise::_pairwise_form')(_pairwise_form_fake)
    torch.library.register_fake('torchpairwise::_pairwise_form_backward')(_pairwise_form_backward_fake)
    torch.library.register_fake('torchpairwise::_image_map')(_image_map_fake)
    torch.library.register_fake('torchpairwise::_image_map_backward')(_image_map_backward_fake)
    torch.library.register_fake('torchpairwise::_kernel_moments')(_kernel_moments_fake)
    torch.library.register_fake('torchpairwise::_kernel_moments_backward')(_kernel_moments_backward_fake)
    torch.library.register_fake('torchpairwise::_transport_map_reduce')(_transport_map_reduce_fake)
    torch.library.register_fake('torchpairwise::_transport_map_reduce_backward')(_transport_map_reduce_backward_fake)
    torch.library.register_fake('torchpairwise::_kernel_form')(_kernel_form_fake)
    torch.library.register_fake('torchpairwise::_kernel_form_backward')(_kernel_form_backward_fake)
    torch.library.register_fake('torchpairwise::_streaming_reduce')(_streaming_reduce_fake)
    torch.library.register_fake('torchpairwise::_streaming_reduce_backward')(_streaming_reduce_backward_fake)
    torch.library.register_fake('torchpairwise::_legacy_compare')(_legacy_compare_fake)
    torch.library.register_fake('torchpairwise::_legacy_compare_backward')(_legacy_backward_fake)
    torch.library.register_fake('torchpairwise::_vector_compare')(_vector_compare_fake)
    torch.library.register_fake('torchpairwise::_vector_compare_backward')(_vector_compare_backward_fake)
    torch.library.register_fake('torchpairwise::_weighted_compare')(_weighted_compare_fake)
    torch.library.register_fake('torchpairwise::_weighted_compare_backward')(_weighted_compare_backward_fake)
    torch.library.register_fake('torchpairwise::_mmd_compare.statistic')(_mmd_compare_fake)
    torch.library.register_fake('torchpairwise::_mmd_compare_backward.statistic')(_mmd_compare_backward_fake)
    torch.library.register_fake('torchpairwise::_mmd_compare')(_mmd_compare_fake)
    torch.library.register_fake('torchpairwise::_mmd_compare_backward')(_mmd_compare_backward_fake)
    torch.library.register_fake('torchpairwise::_transport_compare')(_transport_compare_fake)
    torch.library.register_fake('torchpairwise::_transport_compare_backward')(_transport_compare_backward_fake)
    for name in ('_image_compare', '_audio_compare'):
        torch.library.register_fake('torchpairwise::' + name)(_image_fake if name == '_image_compare' else _audio_fake)
        torch.library.register_fake('torchpairwise::' + name + '_backward')(_backward_fake)
    torch.library.register_fake('torchpairwise::_sequence_compare')(_sequence_fake)
    torch.library.register_fake('torchpairwise::_sequence_compare_backward')(_backward_fake)
    torch.library.register_fake('torchpairwise::_token_compare')(_token_fake)
    torch.library.register_fake('torchpairwise::_structured_compare')(_structured_fake)
    torch.library.register_fake('torchpairwise::_structured_compare_backward')(_structured_backward_fake)
    primitives = ('_canberra', '_braycurtis', '_haversine', '_ppminkowski',
                  '_wminkowski', '_sqmahalanobis', '_sqjensenshannon', '_snr',
                  '_additive_chi2_kernel', '_histogram_intersection')
    for name in primitives:
        torch.library.register_fake('torchpairwise::' + name)(_pair_fake)
        backward = {'_wminkowski': _weighted_backward_fake,
                    '_sqmahalanobis': _mahalanobis_backward_fake}.get(name, _backward_fake)
        torch.library.register_fake('torchpairwise::' + name + '_backward')(backward)
    torch.library.register_fake('torchpairwise::_directed_hausdorff')(_hausdorff_fake)
    torch.library.register_fake('torchpairwise::_directed_hausdorff_backward')(_backward_fake)
    torch.library.register_fake('torchpairwise::_pointset_nn')(_pointset_nn_fake)
    torch.library.register_fake('torchpairwise::_pointset_nn_backward')(_backward_fake)
    torch.library.register_fake('torchpairwise::_pointset_compare')(_pointset_compare_fake)
    torch.library.register_fake('torchpairwise::_pointset_compare_backward')(_backward_fake)
    torch.library.register_fake('torchpairwise::_nan_sqeuclidean')(_nan_squared_fake)
    torch.library.register_fake('torchpairwise::_nan_sqeuclidean_backward')(_backward_fake)
    for name in ('pwand', 'pwor', 'pwxor', 'pweq', 'pwne', 'pwlt', 'pwgt', 'pwle', 'pwge'):
        torch.library.register_fake('torchpairwise::' + name)(_binary_fake)
    for name in ('pwand_sum', 'pwor_sum', 'pwxor_sum', 'pweq_sum', 'pwne_sum', 'pweq_mean', 'pwne_mean'):
        torch.library.register_fake('torchpairwise::' + name)(_binary_reduction_fake)
    for name in ('prf_div', 'prf_ldiv'):
        for overload in ('', '.Scalar', '.rScalar'):
            torch.library.register_fake('torchpairwise::' + name + overload)(_division_fake)
            torch.library.register_fake('torchpairwise::_' + name + '_backward' + overload)(_division_backward_fake)


_register()
