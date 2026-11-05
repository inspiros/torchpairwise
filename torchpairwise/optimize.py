"""Native constrained quadratic optimization over streaming linear maps."""
from enum import IntEnum
from typing import NamedTuple

import torch
from torch.utils._pytree import _register_namedtuple

from .ops._linear import _as_linear
from .linalg import _packet


class QPStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    INFEASIBLE = 2
    NONFINITE = 3
    NONCONVEX = 4
    PROJECTION_FAILED = 5
    STALLED = 6


class QPResult(NamedTuple):
    solution: torch.Tensor
    objective: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor
    optimality: torch.Tensor
    feasibility: torch.Tensor


_register_namedtuple(QPResult, serialized_type_name='torchpairwise.optimize.QPResult')


def _tensor(value, like, name):
    if isinstance(value, torch.Tensor):
        return value
    if type(value) not in (float, int):
        raise TypeError(f'box_qp {name} must be a real number or Tensor')
    return torch.scalar_tensor(value, dtype=like.dtype, device=like.device)


def box_qp(Q, q, *, lower, upper, equality=None, initial=None,
           rtol=1e-5, atol=0., max_iter=1000):
    """Minimize .5*x.T Q x + q.T x within a finite box and one optional equality.

    Q is a real symmetric PSD LazyPairwise or dense matrix.
    q [N] or [...,N,R] defines independent objectives. Operator/q batches
    broadcast. Bounds are finite real scalars or matching-dtype/device tensors
    that broadcast to [...,N,R]; [N] bounds mean per-sample bounds. Use [...,N,1]
    for batch-specific sample vectors. Bounds/initial/equality cannot add batches.

    equality=(a,b) imposes a.T x=b on every problem; a broadcasts like bounds,
    b broadcasts to [...,R]. a is normalized internally by its infinity norm,
    with the same scaling applied to b. Zero a is allowed when b=0.

    The native spectral projected-gradient method uses exact quadratic segment
    minimization. Projection onto box+equality uses safeguarded Newton/bisection.
    Feasible iterates/projection use float64, preserving source/output dtype.
    Optimality is ||x-proj(x-(Qx+q))||_inf. Feasibility is maximum box violation
    and normalized equality residual. Convergence checks both on the true map.
    Objective, optimality and feasibility are float64 diagnostics; status/count
    are int64, converged bool. Diagnostics shape [...] for vector q or [...,R]
    for block q. max_iter=0 projects/evaluates the initial guess (default zero).

    Inference-only: gradient-bearing inputs require explicit torch.no_grad().
    Native early stopping synchronizes CUDA; capture is unsupported. compile and
    export retain one native solve. Symmetry/PSD are assumptions; negative
    curvature detection is not an exhaustive convexity test. Invalid arguments
    raise; numerical failures/nonconvergence return status for inspection.
    """
    if not isinstance(q, torch.Tensor):
        raise TypeError('box_qp q must be a Tensor')
    if type(max_iter) is not int:
        raise TypeError('box_qp max_iter must be an integer')
    if type(rtol) not in (int, float) or type(atol) not in (int, float):
        raise TypeError('box_qp tolerances must be real numbers')
    Q = _as_linear(Q)
    if Q.dtype != q.dtype or Q.device != q.device:
        raise ValueError('box_qp operator and q must share dtype/device')
    torch._check(Q.shape[-2] == Q.shape[-1], lambda: 'box_qp requires a square operator')
    vector, target = None, None
    if equality is not None:
        if not isinstance(equality, (tuple, list)) or len(equality) != 2:
            raise TypeError('box_qp equality must be (vector, target)')
        vector, target = (_tensor(v, q, name) for v, name in zip(equality, ('equality vector', 'equality target')))
    return QPResult(*torch.ops.torchpairwise.box_qp(
        *_packet(Q), list(Q.shape), q, _tensor(lower, q, 'lower'), _tensor(upper, q, 'upper'),
        vector, target, initial, float(rtol), float(atol), max_iter))


from .ops._sinkhorn_solver import (
    sinkhorn, sinkhorn_plan, sinkhorn_divergence, SinkhornResult, SinkhornStatus,
)


__all__ = ['box_qp', 'QPResult', 'QPStatus', 'sinkhorn', 'sinkhorn_plan',
           'sinkhorn_divergence', 'SinkhornResult', 'SinkhornStatus']
