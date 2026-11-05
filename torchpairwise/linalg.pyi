from enum import IntEnum
from typing import NamedTuple, Optional, Union
from torch import Tensor
from .ops._pairwise_operator import LazyPairwise

class CGStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    NONPOSITIVE_CURVATURE = 2
    NONFINITE = 3
    INVALID_PRECONDITIONER = 4
class CGResult(NamedTuple):
    solution: Tensor
    residual_norm: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor

def cg(A: Union[LazyPairwise, Tensor], b: Tensor, *, rtol: float = 1e-5,
       atol: float = 0., max_iter: int = 1000, initial: Optional[Tensor] = None,
       preconditioner: Optional[Union[LazyPairwise, Tensor]] = None) -> CGResult: ...

def jacobi(values: Tensor) -> LazyPairwise: ...
def lowrank_inverse(features: Tensor, shift: Union[float, Tensor]) -> LazyPairwise: ...
def nystrom_inverse(cross: Tensor, landmarks: Tensor, shift: Union[float, Tensor], *, jitter: float = 0.) -> LazyPairwise: ...

class EigenStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    RANK_DEFICIENT_INITIAL = 2
    NONFINITE = 3
    NON_PSD = 4
    LOSS_OF_ORTHOGONALITY = 5
class EigenResult(NamedTuple):
    eigenvalues: Tensor
    eigenvectors: Tensor
    residual_norm: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor

def eigh(A: Union[LazyPairwise, Tensor], *, k: int, subspace_size: Optional[int] = None,
         max_iter: int = 100, rtol: float = 1e-5, atol: float = 0., initial: Optional[Tensor] = None) -> EigenResult: ...

class LSQRStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    BREAKDOWN = 2
    NONFINITE = 3
class LSQRResult(NamedTuple):
    solution: Tensor
    residual_norm: Tensor
    normal_residual_norm: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor

def lsqr(A: Union[LazyPairwise, Tensor], b: Tensor, *, damp: float = 0.,
         rtol: float = 1e-5, atol: float = 0., max_iter: int = 1000,
         initial: Optional[Tensor] = None) -> LSQRResult: ...

class GMRESStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    BREAKDOWN = 2
    NONFINITE = 3
class GMRESResult(NamedTuple):
    solution: Tensor
    residual_norm: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor

def gmres(A: Union[LazyPairwise, Tensor], b: Tensor, *, rtol: float = 1e-5,
          atol: float = 0., restart: int = 20, max_iter: int = 1000,
          initial: Optional[Tensor] = None) -> GMRESResult: ...

class MINRESStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    BREAKDOWN = 2
    NONFINITE = 3
class MINRESResult(NamedTuple):
    solution: Tensor
    residual_norm: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor

def minres(A: Union[LazyPairwise, Tensor], b: Tensor, *, rtol: float = 1e-5,
           atol: float = 0., max_iter: int = 1000, initial: Optional[Tensor] = None) -> MINRESResult: ...


class SolveResult(NamedTuple):
    solution: Tensor
    residual_norm: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor

def solve(A: Union[LazyPairwise, Tensor], b: Tensor, *, method: str,
          rtol: float = 1e-5, atol: float = 0., max_iter: int = 1000,
          initial: Optional[Tensor] = None,
          preconditioner: Optional[Union[LazyPairwise, Tensor]] = None,
          restart: Optional[int] = None, differentiable: bool = False) -> SolveResult: ...

def lstsq(A: Union[LazyPairwise, Tensor], b: Tensor, *, method: str = 'lsqr',
          damp: float = 0., rtol: float = 1e-5, atol: float = 0., max_iter: int = 1000,
          initial: Optional[Tensor] = None, differentiable: bool = False) -> LSQRResult: ...


class SVDStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    RANK_DEFICIENT_INITIAL = 2
    NONFINITE = 3
    LOSS_OF_ORTHOGONALITY = 4
class SVDResult(NamedTuple):
    singular_values: Tensor
    left_vectors: Tensor
    right_vectors: Tensor
    residual_norm: Tensor
    iterations: Tensor
    converged: Tensor
    status: Tensor

def svd(A: Union[LazyPairwise, Tensor], *, k: int,
        subspace_size: Optional[int] = None, max_iter: int = 100,
        rtol: float = 1e-5, atol: float = 0., initial: Optional[Tensor] = None) -> SVDResult: ...

class TraceResult(NamedTuple):
    estimate: Tensor
    standard_error: Tensor
    samples: Tensor
class LogdetResult(NamedTuple):
    estimate: Tensor
    standard_error: Tensor
    samples: Tensor
    iterations: Tensor
class MatrixFunctionResult(NamedTuple):
    action: Tensor
    error_estimate: Tensor
    iterations: Tensor

def trace(A: Union[LazyPairwise, Tensor], *, method: str,
          num_probes: Optional[int] = None, probes: Optional[Tensor] = None,
          seed: Optional[int] = None, block_size: int = 16, differentiable: bool = False) -> TraceResult: ...
def logdet(A: Union[LazyPairwise, Tensor], *, method: str,
           num_probes: Optional[int] = None, probes: Optional[Tensor] = None,
           seed: Optional[int] = None, max_iter: Optional[int] = None,
           block_size: int = 16, differentiable: bool = False, gradient: Optional[str] = None,
           solve_rtol: Optional[float] = None, solve_atol: Optional[float] = None,
           solve_max_iter: Optional[int] = None) -> LogdetResult: ...
def expm_multiply(A: Union[LazyPairwise, Tensor], values: Tensor, *,
                  scale: float = 1., max_iter: int = 32,
                  block_size: int = 16) -> MatrixFunctionResult: ...
def inv_sqrt_multiply(A: Union[LazyPairwise, Tensor], values: Tensor, *,
                      max_iter: int = 32, block_size: int = 16) -> MatrixFunctionResult: ...

class PivotedCholeskyResult(NamedTuple):
    factor: Tensor
    pivots: Tensor
    error: Tensor
    rank: Tensor
class LanczosStatus(IntEnum):
    COMPLETED = 0
    BREAKDOWN = 1
    INVALID_INITIAL = 2
    NONFINITE = 3
class LanczosResult(NamedTuple):
    basis: Tensor
    alpha: Tensor
    beta: Tensor
    iterations: Tensor
    status: Tensor
class LOVECache(NamedTuple):
    root: Tensor

def pivoted_cholesky(A: Union[LazyPairwise, Tensor], rank: int, *,
                     rtol: float = 0., atol: float = 0.) -> PivotedCholeskyResult: ...
def lanczos(A: Union[LazyPairwise, Tensor], k: int, *, initial: Tensor,
            reorthogonalize: bool = True) -> LanczosResult: ...
def love_cache(A: Union[LazyPairwise, Tensor], rank: int, *, initial: Tensor) -> LOVECache: ...
def predictive_variance(cache: LOVECache, cross: Union[LazyPairwise, Tensor],
                        prior_variance: Tensor) -> Tensor: ...
