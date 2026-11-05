from enum import IntEnum
from typing import NamedTuple, Optional, Union, Tuple
from torch import Tensor
from .ops._pairwise_operator import LazyPairwise

class QPStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    INFEASIBLE = 2
    NONFINITE = 3
    NONCONVEX = 4
    PROJECTION_FAILED = 5
    STALLED = 6
class QPResult(NamedTuple):
    solution: Tensor
    objective: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor
    optimality: Tensor
    feasibility: Tensor
Bound = Union[float, int, Tensor]
def box_qp(Q: Union[LazyPairwise, Tensor], q: Tensor, *, lower: Bound, upper: Bound,
           equality: Optional[Tuple[Bound, Bound]] = None, initial: Optional[Tensor] = None,
           rtol: float = 1e-5, atol: float = 0., max_iter: int = 1000) -> QPResult: ...

class SinkhornStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    NONFINITE = 2
class SinkhornResult(NamedTuple):
    f: Tensor
    g: Tensor
    objective: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor
    marginal_error: Tensor
def sinkhorn(C: Union[LazyPairwise, Tensor], a: Optional[Tensor] = None,
             b: Optional[Tensor] = None, *, epsilon: float, max_iter: int = 1000,
             rtol: float = 1e-6, atol: float = 0., reach: Optional[float] = None,
             epsilon_scaling: Optional[float] = None,
             initial: Optional[Tuple[Tensor, Tensor]] = None,
             check_every: int = 10) -> SinkhornResult: ...
def sinkhorn_plan(C: Union[LazyPairwise, Tensor], result: SinkhornResult,
                  a: Optional[Tensor] = None, b: Optional[Tensor] = None, *,
                  epsilon: float) -> LazyPairwise: ...
def sinkhorn_divergence(C_xy: Union[LazyPairwise, Tensor],
                        C_xx: Union[LazyPairwise, Tensor],
                        C_yy: Union[LazyPairwise, Tensor],
                        a: Optional[Tensor] = None, b: Optional[Tensor] = None, *,
                        epsilon: float, **options) -> Tensor: ...
