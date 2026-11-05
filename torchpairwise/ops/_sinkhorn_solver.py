"""Thin bindings for native log-domain transport over scalar lazy costs."""
import math
from enum import IntEnum
from typing import NamedTuple

import torch
from torch.utils._pytree import _register_namedtuple

from ._lazy import as_lazy
from ._linear import _as_linear
from ._pairwise_operator import LazyPairwise
from ..linalg import _packet


class SinkhornStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    NONFINITE = 2


class SinkhornResult(NamedTuple):
    f: torch.Tensor
    g: torch.Tensor
    objective: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor
    marginal_error: torch.Tensor


_register_namedtuple(SinkhornResult, serialized_type_name='torchpairwise.optimize.SinkhornResult')


def _cost(C):
    if isinstance(C, torch.Tensor):
        C = as_lazy(C)
    if not isinstance(C, LazyPairwise):
        raise TypeError('sinkhorn C must be a scalar LazyPairwise or a dense Tensor')
    C._require_entrywise('sinkhorn')
    if C.dtype not in (torch.float32, torch.float64):
        raise TypeError('sinkhorn costs must have float32 or float64 dtype')
    return C


def _mass(value, C, axis, name):
    size = C.shape[axis]
    torch._check(size > 0, lambda: 'sinkhorn requires nonempty sample axes')
    if value is None:
        return torch.ones((size,), dtype=C.dtype, device=C.device) / size
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise TypeError(f'sinkhorn {name} must be a Tensor with a final sample axis')
    if value.dtype != C.dtype or value.device != C.device:
        raise ValueError(f'sinkhorn {name} and cost must share dtype/device')
    torch._check(value.shape[-1] == size, lambda: 'sinkhorn masses must match the cost sample axes')
    return value


def _plan(C, f, g, a, b, epsilon):
    # Include masses inside the exponential to avoid overflow before weighting.
    # Zero-mass entries select -inf before exp and never evaluate log(0).
    log_a = torch.where(a > 0, a, torch.ones_like(a)).log()
    log_b = torch.where(b > 0, b, torch.ones_like(b)).log()
    exponent = (-C / epsilon).add_rows(f / epsilon + log_a).add_columns(g / epsilon + log_b)
    support = C.row_values(a > 0) & C.column_values(b > 0)
    return exponent.where(support, -float('inf')).exp()


def sinkhorn(C, a=None, b=None, *, epsilon, max_iter=1000, rtol=1e-6, atol=0.,
             reach=None, epsilon_scaling=None, initial=None, check_every=10):
    """Entropic transport for any scalar lazy cost [...,N,M], or a dense Tensor.

    Nonnegative masses [...,N]/[...,M] and cost batches broadcast; omitted masses
    are uniform. Balanced masses must have equal totals; no normalization occurs.
    reach>0 applies KL marginal penalties with strength reach. epsilon_scaling in
    (0,1) anneals from a streamed cost scale and warm-starts each stage. initial
    is a pair of potentials. max_iter is the total update budget across stages.

    Native alternating logsumexp updates reuse one expression program. Potentials
    and diagnostics use float64; source comparisons retain their original dtype.
    Potentials are detached. The objective uses a streamed differentiable dual
    evaluation, giving envelope gradients without retaining iteration history.
    Balanced objective values are <a,f>+<b,g>; derivatives use the full KL dual
    with reference a tensor-product b (mass derivatives include its dependence).
    Thus mass derivatives agree with potential derivatives up to constants on
    the fixed-total mass simplex, not arbitrary off-simplex extensions.

    Diagnostics are per broadcast batch: iterations/status int64, converged bool,
    marginal_error float64. Numerical failure returns NONFINITE; inspect status
    before using a result. Native early stopping synchronizes CUDA; capture is
    unsupported. compile/export preserve the native solve without unrolling.
    """
    for name, value in (('epsilon', epsilon), ('rtol', rtol), ('atol', atol)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'sinkhorn {name} must be a finite real number')
    if epsilon <= 0 or rtol < 0 or atol < 0:
        raise ValueError('sinkhorn requires epsilon>0 and nonnegative tolerances')
    if type(max_iter) is not int or max_iter < 0:
        raise ValueError('sinkhorn max_iter must be a nonnegative integer')
    if type(check_every) is not int or check_every <= 0:
        raise ValueError('sinkhorn check_every must be a positive integer')
    if reach is not None and (type(reach) not in (int, float) or not math.isfinite(reach) or reach <= 0):
        raise ValueError('sinkhorn reach must be a finite positive number or None')
    if epsilon_scaling is not None and (type(epsilon_scaling) not in (int, float)
            or not math.isfinite(epsilon_scaling) or not 0 < epsilon_scaling < 1):
        raise ValueError('sinkhorn epsilon_scaling must lie strictly between zero and one')
    C = _cost(C)
    a, b = _mass(a, C, -2, 'a'), _mass(b, C, -1, 'b')
    initial_f, initial_g = None, None
    if initial is not None:
        if not isinstance(initial, (tuple, list)) or len(initial) != 2:
            raise TypeError('sinkhorn initial must be a pair (f, g) of Tensor potentials')
        initial_f, initial_g = initial
        if not all(isinstance(value, torch.Tensor) for value in initial):
            raise TypeError('sinkhorn initial must be a pair (f, g) of Tensor potentials')
    f, g, iterations, converged, status, error = torch.ops.torchpairwise.sinkhorn_solver(
        *_packet(_as_linear(C)), list(C.shape), a, b, float(epsilon), max_iter,
        float(rtol), float(atol), None if reach is None else float(reach),
        None if epsilon_scaling is None else float(epsilon_scaling),
        initial_f, initial_g, check_every)
    f, g = f.detach(), g.detach()
    a64, b64 = a.double(), b.double()
    transport_mass = _plan(C, f, g, a64, b64, epsilon).sum((-2, -1))
    reference_mass = a64.sum(-1) * b64.sum(-1)
    if reach is None:
        value = (a64 * f).sum(-1) + (b64 * g).sum(-1)
        dual = value - epsilon * (transport_mass - reference_mass)
        # The balanced convention omits the KL reference-measure constant.
        objective = dual + (value - dual).detach()
    else:
        objective = (-reach * a64 * torch.expm1(-f / reach)).sum(-1)
        objective = objective + (-reach * b64 * torch.expm1(-g / reach)).sum(-1)
        objective = objective - epsilon * (transport_mass - reference_mass)
    return SinkhornResult(f, g, objective, iterations, converged, status, error)


def sinkhorn_plan(C, result, a=None, b=None, *, epsilon):
    """Lazy plan exp((f+g-C)/epsilon) * a * b, using the solve's masses/epsilon.

    Omitted masses are uniform, as in sinkhorn. Potentials are detached; the plan
    remains differentiable in its explicitly supplied cost and masses. Products
    and reductions stream through the ordinary scalar expression backend.
    """
    C = _cost(C)
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('sinkhorn_plan epsilon must be a finite positive number')
    a, b = _mass(a, C, -2, 'a'), _mass(b, C, -1, 'b')
    return _plan(C, result.f.detach().to(C.dtype), result.g.detach().to(C.dtype),
                 a, b, epsilon)


def sinkhorn_divergence(C_xy, C_xx, C_yy, a=None, b=None, *, epsilon, **options):
    """Debiased OT: OT(a,b) - .5 OT(a,a) - .5 OT(b,b).

    Supply cross and self costs built from the same ground comparison. Positivity
    requires the usual assumptions on that comparison; arbitrary costs need not
    define a nonnegative divergence. Solver options are forwarded to each solve.
    """
    cross = sinkhorn(C_xy, a, b, epsilon=epsilon, **options)
    left = sinkhorn(C_xx, a, a, epsilon=epsilon, **options)
    right = sinkhorn(C_yy, b, b, epsilon=epsilon, **options)
    return cross.objective - .5 * left.objective - .5 * right.objective
