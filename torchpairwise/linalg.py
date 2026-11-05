"""Native iterative linear algebra over finite streaming linear maps."""
from enum import IntEnum
from typing import NamedTuple
from torch.utils._pytree import _register_namedtuple

import torch

from .ops._linear import LinearOperator, _as_linear, _diagonal, _softmax_solver_source
from .ops._operator_algebra import ScalarExpression


class CGStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    NONPOSITIVE_CURVATURE = 2
    NONFINITE = 3
    INVALID_PRECONDITIONER = 4


class CGResult(NamedTuple):
    solution: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor


_register_namedtuple(CGResult, serialized_type_name='torchpairwise.linalg.CGResult')


def _packet(operator):
    # Source-local scalar programs retain their existing opcode/index contract.
    # Flatten tensor lists and remap source option keys for export serialization.
    inputs, counts, metrics, families = [], [], [], []
    tensors, reals, integers, flags, strings, real_lists, integer_lists = {}, {}, {}, {}, {}, {}, {}
    dictionaries = (tensors, reals, integers, flags, strings, real_lists, integer_lists)
    transposed, source_counts, programs, factors, factor_counts = [], [], [], [], []
    kinds, integer_numbers, real_numbers, linear_program, linear_factors, seen = [], [], [], [], [], []
    codes = {'source': 0, 'add': 1, 'sub': 2, 'compose': 3, 'scale': 4,
             'add_diagonal': 5, 'center': 6, 'transpose': 7, 'group': 8,
             'hstack': 9, 'vstack': 10, 'diagonal': 11, 'lowrank_inverse': 12, 'reference_center': 13}

    def visit(recurse, value):
        for previous, index in seen:
            if value is previous:
                return index
        kind = value._kind
        if kind == 'softmax':
            prepared = _softmax_solver_source(value._left, value._right)
            index = recurse(recurse, prepared)
            seen.append((value, index))
            return index
        if kind == 'lowrank_inverse':
            a, b = len(linear_factors), len(linear_factors) + 1
            linear_factors.extend(value._left)
        elif kind == 'diagonal':
            a, b = len(linear_factors), -1
            linear_factors.append(value._left)
        elif kind == 'group':
            a, b = len(linear_factors), len(linear_factors) + 1
            linear_factors.extend((value._left, value._right))
        elif kind == 'source':
            a, b = len(source_counts), -1
            packet = ScalarExpression('identity', value._left)._packet()
            groups, names, groups_families, *rest = packet
            offset = len(metrics)
            for group in groups:
                counts.append(len(group))
                inputs.extend(group)
            metrics.extend(names)
            families.extend(groups_families)
            for column, output in enumerate(dictionaries):
                for key, parameter in rest[column].items():
                    prefix, separator, suffix = key.partition(':')
                    output[str(int(prefix) + offset) + (separator + suffix if separator else '')] = parameter
            transposed.extend(rest[7])
            source_counts.append(len(groups))
            programs.append(rest[8])
            factors.extend(rest[9])
            factor_counts.append(len(rest[9]))
            kinds.append(rest[10])
            integer_numbers.append(rest[11])
            real_numbers.append(rest[12])
        else:
            a, b = recurse(recurse, value._left), -1
            if kind in ('add', 'sub', 'compose', 'hstack', 'vstack'):
                b = recurse(recurse, value._right)
            elif kind in ('scale', 'add_diagonal', 'reference_center'):
                b = len(linear_factors)
                factor = value._right
                if not isinstance(factor, torch.Tensor):
                    factor = torch.scalar_tensor(factor, dtype=value.dtype, device=value.device)
                linear_factors.append(factor)
        index = len(linear_program) // 3
        linear_program.extend((codes[kind], a, b))
        seen.append((value, index))
        return index

    visit(visit, operator)
    return (inputs, counts, metrics, families, *dictionaries, transposed,
            source_counts, {str(i): v for i, v in enumerate(programs)}, factors, factor_counts,
            {str(i): v for i, v in enumerate(kinds) if v},
            {str(i): v for i, v in enumerate(integer_numbers) if v},
            {str(i): v for i, v in enumerate(real_numbers) if v}, linear_program, linear_factors)


def cg(A, b, *, rtol=1e-5, atol=0., max_iter=1000, initial=None, preconditioner=None):
    """Solve a real SPD system using independent native CG recurrences.

    A accepts a LazyPairwise or dense tensor. b has
    shape [N] or [...,N,R]; use [...,N,1] for batched vectors. Operator and RHS
    batches broadcast. initial broadcasts to the solution shape without adding
    batch dimensions. Float32/64 only; no dtype or device conversion is implicit.

    Convergence requires ||b-Ax|| <= max(atol, rtol*||b||), checked using the true
    residual. Diagnostics have shape [...] for vector b or [...,R] for blocks;
    residual_norm is float64, iterations/status int64, converged bool. max_iter=0
    evaluates the initial guess. Nonconvergence returns diagnostics, not an error.

    preconditioner is a fixed SPD inverse action M approximately A^{-1}, adapted
    like A. It must match [N,N] and broadcast to the solution without adding
    batches. The recurrence uses z=M r, not a solve with M; symmetry/positive
    definiteness are caller assumptions. Nonpositive r.T M r gives status 4,
    nonfinite actions status 3. No preconditioner is inferred automatically.

    This solver is inference-only: use torch.no_grad() for gradient-bearing
    inputs. Native early exit synchronizes CUDA; CUDA graph capture is unsupported.
    torch.compile/export retain one native solve rather than unrolling iterations.
    """
    if type(max_iter) is not int:
        raise TypeError('max_iter must be a nonnegative integer')
    if type(rtol) not in (float, int) or type(atol) not in (float, int):
        raise TypeError('CG tolerances must be real numbers')
    if not isinstance(b, torch.Tensor):
        raise TypeError('CG b must be a Tensor')
    A = _as_linear(A)
    if A.dtype != b.dtype or A.device != b.device:
        raise ValueError('CG operator and RHS must share dtype/device')
    torch._check(A.shape[-2] == A.shape[-1], lambda: 'CG requires a square operator')
    packet_source = A
    solve = torch.ops.torchpairwise.linear_cg
    if preconditioner is not None:
        M = _as_linear(preconditioner)
        torch._check(M.shape[-2] == A.shape[-1], lambda: 'CG preconditioner must match system dimensions')
        torch._check(M.shape[-1] == A.shape[-1], lambda: 'CG preconditioner must match system dimensions')
        batch = torch.broadcast_shapes(A.shape[:-2], b.shape[:-2] if b.ndim > 1 else ())
        expected = torch.broadcast_shapes(M.shape[:-2], batch)
        if len(expected) != len(batch):
            raise ValueError('CG preconditioner cannot add batch dimensions')
        for actual, wanted in zip(expected, batch):
            torch._check(actual == wanted, lambda: 'CG preconditioner cannot add batch dimensions')
        # The final composition stores two graph roots in one shared packet.
        # Native PCG applies each root separately, never CG on the product A*M.
        packet_source = A._compose(M)
        solve = torch.ops.torchpairwise.linear_pcg
    return CGResult(*solve(*_packet(packet_source), list(A.shape), b,
                          float(rtol), float(atol), max_iter, initial))


def jacobi(values):
    """Return the fixed inverse action diag(1/d) for a supplied system diagonal.

    d is strided float32/64 [...,N], finite and strictly positive with a finite
    reciprocal. Entries are computed once, not refreshed after changing d.
    This helper does not extract a diagonal from a composed linear map. Scalar
    streaming sources provide .diagonal(); callers may also compute d directly.
    """
    if not isinstance(values, torch.Tensor):
        raise TypeError('Jacobi diagonal must be a Tensor')
    return _diagonal(torch.ops.torchpairwise.linear_jacobi(values))


def _inverse_shift(like, shift):
    if not isinstance(like, torch.Tensor):
        raise TypeError('Low-rank features/cross must be a Tensor')
    if isinstance(shift, torch.Tensor):
        return shift
    if type(shift) not in (int, float):
        raise TypeError('Low-rank shift must be a real number or scalar Tensor')
    return torch.scalar_tensor(shift, dtype=like.dtype, device=like.device)


def lowrank_inverse(features, shift):
    """Prepare the fixed inverse action of F F.T + shift I, F:[...,N,L].

    Setup accepts finite float32/64 features and a positive finite scalar shift
    (Tensor shifts must match dtype/device). It is inference-only: use explicit
    no-grad for gradient-bearing inputs. A thin double SVD snapshots the state;
    rebuild after changing inputs. Rank-deficient columns and empty axes work.
    No N-by-N inverse/projector is constructed. State stores a double [...,N,r]
    basis and [...,r] spectrum, r=min(N,L), plus scalar shift. Setup costs a thin
    SVD and uses O(NL+L*r) workspace, in addition to vendor SVD workspace.

    The returned linear-capability LazyPairwise has the input dtype/device, supports RHS
    gradients and compile/export, and is suitable for cg(preconditioner=M).
    Internal application uses double arithmetic; source/returned RHS dtype stays
    float32/64. Small shifts relative to the spectrum amplify roundoff. Choose
    rank and shift explicitly; fewer CG steps need not offset setup/action cost.
    """
    alpha = _inverse_shift(features, shift)
    state = torch.ops.torchpairwise.linear_lowrank_setup(features, alpha)
    return LinearOperator(state, _kind='lowrank_inverse')


def nystrom_inverse(cross, landmarks, shift, *, jitter=0.):
    """Inverse action of C (W+jitter I)^-1 C.T + shift I, without an N*N matrix.

    Supply C=K(X,Z):[...,N,L] and W=K(Z,Z):[...,L,L], with matching float32/64
    dtype/device and broadcast batches. W must be finite/symmetric, W+jitter I
    positive definite. jitter is an explicit finite nonnegative number; no
    hidden jitter, landmark selection or retry is performed. shift is a positive
    scalar as in lowrank_inverse. Inference-only setup whitens C via a double
    L-by-L Cholesky solve, then prepares the same spectral inverse action.

    Inputs and O(NL+L*L) setup blocks are materialized. Prepared double state
    stores O(N*min(N,L)) entries; the full kernel remains streamed in CG.
    The output snapshots the inputs, supports RHS gradients, native execution
    and compile/export. With L approaching N, storage becomes quadratic.
    """
    alpha = _inverse_shift(cross, shift)
    if not isinstance(landmarks, torch.Tensor):
        raise TypeError('Nyström landmark matrix must be a Tensor')
    if type(jitter) not in (int, float):
        raise TypeError('Nyström jitter must be a real number')
    state = torch.ops.torchpairwise.linear_nystrom_setup(cross, landmarks, alpha, float(jitter))
    return LinearOperator(state, _kind='lowrank_inverse')


__all__ = ['cg', 'CGResult', 'CGStatus', 'jacobi', 'lowrank_inverse', 'nystrom_inverse']


class EigenStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    RANK_DEFICIENT_INITIAL = 2
    NONFINITE = 3
    NON_PSD = 4
    LOSS_OF_ORTHOGONALITY = 5


class EigenResult(NamedTuple):
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor


_register_namedtuple(EigenResult, serialized_type_name='torchpairwise.linalg.EigenResult')


def eigh(A, *, k, subspace_size=None, max_iter=100, rtol=1e-5, atol=0., initial=None):
    """Leading eigenpairs of a real symmetric PSD streaming map, inference only.

    Returns eigenvalues [...,k] in descending order, eigenvectors [...,N,k],
    float64 residual norms [...,k], integer iterations/status and bool converged.
    Batches stop independently when all k true residuals meet
    ||A u - lambda*u|| <= max(atol, rtol*abs(lambda)), with orthonormal vectors.
    iterations counts subspace updates and is repeated across each batch's k
    sorted ranks. max_iter=0 evaluates the starting subspace without updating it.

    QR-stabilized block subspace iteration uses O(N*r+r*r) solver blocks, where
    r=subspace_size. Default r=min(N,2*k+8), or initial.shape[-1] when supplied.
    Require 1<=k<=r<=N. initial [...,N,r] broadcasts to operator batches without
    adding batches, must have full numerical rank and overlap leading directions.
    A random start is used otherwise, respecting torch's current RNG state.
    Small residuals alone do not certify global leading order if a starting
    subspace misses an invariant direction. Flat spectra may converge slowly.

    No smallest/indefinite eigenvalue support, implicit dense matrix, gradients,
    or CUDA graph capture. PSD/symmetry are caller assumptions; a projected
    negative direction can be detected but this is not an exhaustive PSD test.
    Numerical failures return status; invalid arguments raise. Use explicit
    torch.no_grad() for gradient-bearing inputs. compile/export retain a native
    solve rather than a Python loop. Use atol>0 when retaining null directions.
    """
    if type(k) is not int or type(max_iter) is not int:
        raise TypeError('eigh k and max_iter must be integers')
    if subspace_size is not None and type(subspace_size) is not int:
        raise TypeError('eigh subspace_size must be an integer or None')
    if type(rtol) not in (int, float) or type(atol) not in (int, float):
        raise TypeError('eigh tolerances must be real numbers')
    A = _as_linear(A)
    torch._check(A.shape[-2] == A.shape[-1], lambda: 'eigh requires a square operator')
    anchor = torch.empty((), dtype=A.dtype, device=A.device)
    return EigenResult(*torch.ops.torchpairwise.linear_eigh(
        *_packet(A), list(A.shape), anchor, k, subspace_size, max_iter, float(rtol), float(atol), initial))


__all__ += ['eigh', 'EigenResult', 'EigenStatus']


class LSQRStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    BREAKDOWN = 2
    NONFINITE = 3


class LSQRResult(NamedTuple):
    solution: torch.Tensor
    residual_norm: torch.Tensor
    normal_residual_norm: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor


_register_namedtuple(LSQRResult, serialized_type_name='torchpairwise.linalg.LSQRResult')


def lsqr(A, b, *, damp=0., rtol=1e-5, atol=0., max_iter=1000, initial=None):
    """Minimize ||A x-b||^2 + damp^2 ||x||^2 using native rectangular LSQR.

    A is a real float32/float64 map [...,N,M], scalar source or dense Tensor;
    b is [N] or [...,N,R]. Batches broadcast. solution is [...,M] for vector b
    or [...,M,R] for blocks. initial broadcasts to that shape without adding
    batches; damping always penalizes x itself, not its offset from initial.

    Independent Golub-Kahan recurrences use A and A.T actions without forming
    A.T A. The true normal residual g=A.T(b-Ax)-damp^2*x verifies convergence:
    ||g|| <= max(atol, rtol*||A.T b||). A nonzero data residual can be optimal.
    residual_norm reports ||b-Ax||, normal_residual_norm reports ||g||, both
    float64. Counts/status are int64 and converged bool, shape [...] for a
    vector RHS or [...,R] for blocks. max_iter=0 evaluates the initial guess.
    A zero start with damp=0 targets the minimum-norm least-squares solution;
    a nonzero initial may retain null-space components. Poor conditioning or
    a loose normal-residual tolerance can still give inaccurate coefficients.

    Storage is O((N+M)*R) solver blocks per batch plus source/tile workspace.
    Inference only; explicit no-grad is required for gradient-bearing inputs.
    CUDA graph capture is unsupported. compile/export retain one native solve.
    """
    if type(max_iter) is not int:
        raise TypeError('max_iter must be a nonnegative integer')
    if any(type(value) not in (float, int) for value in (damp, rtol, atol)):
        raise TypeError('LSQR damp and tolerances must be real numbers')
    if not isinstance(b, torch.Tensor):
        raise TypeError('LSQR b must be a Tensor')
    A = _as_linear(A)
    if A.dtype != b.dtype or A.device != b.device:
        raise ValueError('LSQR operator and RHS must share dtype/device')
    return LSQRResult(*torch.ops.torchpairwise.linear_lsqr(
        *_packet(A), list(A.shape), b, float(damp), float(rtol), float(atol), max_iter, initial))


__all__ += ['lsqr', 'LSQRResult', 'LSQRStatus']


class GMRESStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    BREAKDOWN = 2
    NONFINITE = 3


class GMRESResult(NamedTuple):
    solution: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor


_register_namedtuple(GMRESResult, serialized_type_name='torchpairwise.linalg.GMRESResult')


def gmres(A, b, *, rtol=1e-5, atol=0., restart=20, max_iter=1000, initial=None):
    """Solve real nonsymmetric square systems with native restarted GMRES.

    A is a LazyPairwise or dense tensor [...,N,N].
    b is [N] or [...,N,R]; batches broadcast, with independent RHS recurrences.
    initial broadcasts to the solution without adding batches. Float32/64 only.

    restart is a positive integer bounding the Arnoldi basis width, clamped to
    N. max_iter counts total Arnoldi steps per batch/RHS across all cycles, not
    cycles. max_iter=0 evaluates the initial guess. No preconditioner is applied.
    Two modified Gram-Schmidt passes and Givens rotations build the restarted
    least-square problem. Actual ||b-Ax|| <= max(atol,rtol*||b||) verifies every
    candidate and final convergence. Source products/output preserve dtype;
    basis/orthogonalization/scalar workspace and residual norms use float64.

    solution is [...,N] or [...,N,R]. Diagnostics are [...] or [...,R]; norms
    float64, iterations/status int64, convergence bool. Status 0 convergence,
    1 total iteration limit, 2 Arnoldi/triangular breakdown without verified
    convergence, 3 nonfinite arithmetic. Failure retains the current iterate.
    A nonsingular map is needed for a unique solution; singular inconsistent
    systems are not a least-squares API. Restarts may stagnate; small residuals
    do not bound coefficient error for ill-conditioned or nonnormal systems.

    Workspace includes O(N*R*m + R*m*m) double basis/QR blocks per broadcast
    batch, m=min(restart,N,max_iter), plus source preparation/tile workspace.
    Inference only; gradient-bearing inputs need explicit torch.no_grad().
    Native stopping synchronizes CUDA: graph capture unsupported. compile/export
    preserve one opaque native solve rather than unrolling iterations.
    """
    if type(max_iter) is not int or type(restart) is not int:
        raise TypeError('GMRES restart and max_iter must be integers')
    if type(rtol) not in (float, int) or type(atol) not in (float, int):
        raise TypeError('GMRES tolerances must be real numbers')
    if not isinstance(b, torch.Tensor):
        raise TypeError('GMRES b must be a Tensor')
    A = _as_linear(A)
    if A.dtype != b.dtype or A.device != b.device:
        raise ValueError('GMRES operator and RHS must share dtype/device')
    torch._check(A.shape[-2] == A.shape[-1], lambda: 'GMRES requires a square operator')
    return GMRESResult(*torch.ops.torchpairwise.linear_gmres(
        *_packet(A), list(A.shape), b, float(rtol), float(atol), restart, max_iter, initial))


__all__ += ['gmres', 'GMRESResult', 'GMRESStatus']


class MINRESStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    BREAKDOWN = 2
    NONFINITE = 3


class MINRESResult(NamedTuple):
    solution: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor


_register_namedtuple(MINRESResult, serialized_type_name='torchpairwise.linalg.MINRESResult')


def minres(A, b, *, rtol=1e-5, atol=0., max_iter=1000, initial=None):
    """Solve a real symmetric, possibly indefinite system using native MINRES.

    Symmetry is a caller assumption, not exhaustively checked. A is a square
    LazyPairwise or dense Tensor [...,N,N]; b is [N]
    or [...,N,R]. Batches broadcast and RHS columns stop independently. initial
    broadcasts to the solution without adding batches. Float32/float64 only.

    A short Lanczos/QR recurrence retains O(N*R) solver blocks per batch plus
    source/tile workspace, with no stored full basis or restart-width option.
    Internal vectors/scalars are double, source products and returned solution
    retain supplied precision. Actual ||b-Ax|| <= max(atol,rtol*||b||) verifies
    candidates and final convergence. False estimated convergence can trigger a
    reliable restart from the actual residual. max_iter counts all Lanczos steps;
    max_iter=0 evaluates the initial guess. No preconditioner or MINRES-QLP.

    Returns solution [...,N] or [...,N,R], float64 residual_norm, int64 iterations
    and status, bool converged. Diagnostics are [...] or [...,R]. Status 0 success,
    1 iteration limit, 2 Lanczos/QR breakdown without verified convergence,
    3 nonfinite. Failure retains the current iterate. Singular/inconsistent maps
    have no minimum-norm or least-squares guarantee; use LSQR for that objective.
    Finite-precision Lanczos can lose orthogonality and converge slowly.

    Inference only; gradient-bearing inputs require explicit no-grad. Native
    stopping synchronizes CUDA, so graph capture is unsupported. compile/export
    retain an opaque native solve. Small residuals do not bound solution error
    for poorly conditioned maps.
    """
    if type(max_iter) is not int:
        raise TypeError('MINRES max_iter must be an integer')
    if type(rtol) not in (float, int) or type(atol) not in (float, int):
        raise TypeError('MINRES tolerances must be real numbers')
    if not isinstance(b, torch.Tensor):
        raise TypeError('MINRES b must be a Tensor')
    A = _as_linear(A)
    if A.dtype != b.dtype or A.device != b.device:
        raise ValueError('MINRES operator and RHS must share dtype/device')
    torch._check(A.shape[-2] == A.shape[-1], lambda: 'MINRES requires a square operator')
    return MINRESResult(*torch.ops.torchpairwise.linear_minres(
        *_packet(A), list(A.shape), b, float(rtol), float(atol), max_iter, initial))


__all__ += ['minres', 'MINRESResult', 'MINRESStatus']


class SolveResult(NamedTuple):
    solution: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor


_register_namedtuple(SolveResult, serialized_type_name='torchpairwise.linalg.SolveResult')


def solve(A, b, *, method, rtol=1e-5, atol=0., max_iter=1000, initial=None,
          preconditioner=None, restart=None, differentiable=False):
    """Solve a real square streaming system with an explicitly selected method.

    method='cg' assumes SPD; 'minres' assumes symmetry and permits indefinite
    systems; 'gmres' permits nonsymmetric systems. No detection or fallback.
    The supplied method must be a static string for compile/export. A accepts
    a LazyPairwise or dense Tensor [...,N,N]; b:[N] or [...,N,R].
    Batches broadcast, with independent RHS diagnostics and the selected solver's
    initial/tolerance/iteration contracts. Float32/64. Default inference only;
    differentiable=True uses an implicit adjoint solve for first derivatives.

    Only CG accepts preconditioner (a fixed SPD inverse action); only GMRES
    accepts restart (omitted means 20). Unsupported options raise, even when
    their explicit values equal another method's defaults. Nonconvergence
    returns SolveResult with the current iterate and true ||b-Ax||, not an
    automatic retry or error. Check converged/status. Status 0 success, 1 limit,
    3 nonfinite; code 2 is nonpositive curvature for CG or breakdown for the
    other methods; code 4 is an invalid CG preconditioner.

    differentiable=True assumes a nonsingular map and raises if either the
    forward or adjoint solve fails convergence, even in a no-grad scope. It
    differentiates the converged mathematical solution, not iteration decisions;
    loose tolerances/poor conditioning can still make gradients inaccurate.
    Initial guesses must be detached. CG applies a supplied fixed SPD inverse
    preconditioner in both solves; preconditioner-only operands are detached. Shared operands still receive system gradients. Higher derivatives
    are unsupported. Source/factor/RHS gradients replay
    streamed products with no dense matrix cotangent or saved iteration trace.
    Only solution is differentiable; diagnostics are detached.

    No inverse matrix is materialized. Singular least squares belong to lstsq;
    solve does not promise a minimum-norm solution. Native early stopping
    excludes CUDA graph capture; compile/export keep the existing native solve.
    """
    if not isinstance(method, str):
        raise TypeError('solve method must be a string')
    if method not in ('cg', 'minres', 'gmres'):
        raise ValueError("solve method must be explicitly selected from 'cg', 'minres', 'gmres'")
    if preconditioner is not None and method != 'cg':
        raise ValueError("solve preconditioner is supported only with method='cg'")
    if restart is not None and method != 'gmres':
        raise ValueError("solve restart is supported only with method='gmres'")
    if type(differentiable) is not bool:
        raise TypeError('differentiable must be bool')
    if differentiable:
        if type(max_iter) is not int or (restart is not None and type(restart) is not int):
            raise TypeError('solve max_iter/restart must be integers')
        if type(rtol) not in (int, float) or type(atol) not in (int, float):
            raise TypeError('solve tolerances must be real numbers')
        if not isinstance(b, torch.Tensor):
            raise TypeError('solve b must be a Tensor')
        A = _as_linear(A)
        if A.dtype != b.dtype or A.device != b.device:
            raise ValueError('solve operator and RHS must share dtype/device')
        torch._check(A.shape[-2] == A.shape[-1], lambda: 'solve requires a square operator')
        graph = A if preconditioner is None else A._compose(_as_linear(preconditioner))
        operation = (torch.ops.torchpairwise.linear_implicit_solve if preconditioner is None
                     else torch.ops.torchpairwise.linear_implicit_pcg)
        return SolveResult(*operation(*_packet(graph), list(A.shape), b, method,
            float(rtol), float(atol), max_iter, 20 if restart is None else restart, initial))
    if method == 'cg':
        result = cg(A, b, rtol=rtol, atol=atol, max_iter=max_iter, initial=initial,
                    preconditioner=preconditioner)
    elif method == 'minres':
        result = minres(A, b, rtol=rtol, atol=atol, max_iter=max_iter, initial=initial)
    else:
        result = gmres(A, b, rtol=rtol, atol=atol, max_iter=max_iter, initial=initial,
                       restart=20 if restart is None else restart)
    return SolveResult(*result)


def lstsq(A, b, *, method='lsqr', damp=0., rtol=1e-5, atol=0., max_iter=1000, initial=None, differentiable=False):
    """Iterative least squares through native LSQR, returning its LSQRResult.

    Minimize ||Ax-b||^2 + damp^2 ||x||^2 for A:[...,N,M], b:[N] or [...,N,R].
    solution:[...,M] or [...,M,R]. Float32/64 with broadcast batches; default
    inference only. method='lsqr' is the only supported method. The data residual may be
    nonzero at the optimum; convergence verifies the true normal residual.

    With damp=0 and the default zero start, LSQR targets a minimum-norm
    least-squares solution. A nonzero initial may preserve null-space components;
    damping regularizes the final x, not its displacement from the initial.
    This is not an SVD pseudoinverse: no rank estimate/singular values or rcond
    cutoff is computed. Poor conditioning or loose stopping may leave inaccurate
    coefficients. Inspect converged/status and normal_residual_norm.

    differentiable=True requires finite positive damp with its square finite
    and positive in the source dtype. Forward uses LSQR; backward solves the
    damped normal action using CG, without forming a normal matrix, then replays
    source products for first source/factor/RHS gradients. Both solves must
    converge or raise. Damping is a fixed scalar; initial must be detached and
    higher derivatives are unsupported. Poor conditioning/loose tolerances can
    leave inaccurate gradients; the normal action squares singular values.
    Rank-deficient A is allowed because positive damping makes the solution unique.

    Native execution, workspace and compile/export behavior are those of lsqr.
    No normal matrix or pseudoinverse is constructed, and CUDA graph capture is
    unsupported. Default numerical failures return diagnostics; implicit-mode
    failures raise. No dense fallback is used.
    """
    if not isinstance(method, str):
        raise TypeError('lstsq method must be a string')
    if method != 'lsqr':
        raise ValueError("lstsq method must be 'lsqr'")
    if type(differentiable) is not bool:
        raise TypeError('differentiable must be bool')
    if differentiable:
        if type(max_iter) is not int or any(type(v) not in (int, float) for v in (damp, rtol, atol)):
            raise TypeError('lstsq needs integer max_iter and real damping/tolerances')
        if not isinstance(b, torch.Tensor):
            raise TypeError('lstsq b must be a Tensor')
        A = _as_linear(A)
        if A.dtype != b.dtype or A.device != b.device:
            raise ValueError('lstsq operator and RHS must share dtype/device')
        return LSQRResult(*torch.ops.torchpairwise.linear_implicit_lstsq(
            *_packet(A), list(A.shape), b, float(damp), float(rtol), float(atol), max_iter, initial))
    return lsqr(A, b, damp=damp, rtol=rtol, atol=atol, max_iter=max_iter, initial=initial)


__all__ += ['solve', 'SolveResult', 'lstsq']


class SVDStatus(IntEnum):
    CONVERGED = 0
    ITERATION_LIMIT = 1
    RANK_DEFICIENT_INITIAL = 2
    NONFINITE = 3
    LOSS_OF_ORTHOGONALITY = 4


class SVDResult(NamedTuple):
    singular_values: torch.Tensor
    left_vectors: torch.Tensor
    right_vectors: torch.Tensor
    residual_norm: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    status: torch.Tensor


_register_namedtuple(SVDResult, serialized_type_name='torchpairwise.linalg.SVDResult')


def svd(A, *, k, subspace_size=None, max_iter=100, rtol=1e-5, atol=0., initial=None):
    """Leading singular triplets of a real rectangular streaming map [...,N,M].

    Two-sided QR-stabilized block iteration uses A and A.T actions and a small
    r-by-r SVD; no normal matrix or dense source is formed. Require
    1<=k<=r<=min(N,M). r defaults to min(N,M,2*k+8), or initial.shape[-1].
    initial is a full-numerical-rank right search block [...,M,r]; it may
    broadcast to source batches but cannot add batches. Otherwise draws a random
    start from the current RNG. Initial overlap with leading subspaces matters;
    small residuals alone cannot certify global leading order. Flat spectra can
    need more updates or a wider subspace.

    SVDResult contains descending singular_values [...,k], left_vectors [...,N,k],
    right_vectors [...,M,k] (columns, not Vh), and per-triplet residual_norm,
    iterations, converged, status [...,k]. Norms are float64, counts/status int64,
    convergence bool. residual_norm is max(||Av-su||,||A.Tu-sv||); convergence
    requires this <=max(atol,rtol*s) and orthonormal left/right vectors. The
    approximation is (left_vectors*singular_values) @ right_vectors.T.

    Each batch stops after all requested ranks converge; vectors are not locked.
    max_iter counts updates after the initial two-sided extraction; zero checks
    that extraction only. Finite rank collapse is allowed, including zero maps.
    Use positive atol for null triplets. Numerical failures return status2-4
    and NaN estimates; iteration exhaustion retains finite estimates. Malformed
    inputs raise. Width near min(N,M) approaches dense storage: retained state
    uses O((N+M)*r+r*r) double blocks plus source/tile/output workspace.

    Float32/64, broadcast batches, inference only; detach gradient-bearing
    operands or use torch.no_grad(). Native stopping excludes CUDA graph capture.
    compile/export retain one native SVD, not a Python iteration loop.
    """
    if type(k) is not int or type(max_iter) is not int:
        raise TypeError('svd k and max_iter must be integers')
    if subspace_size is not None and type(subspace_size) is not int:
        raise TypeError('svd subspace_size must be an integer or None')
    if type(rtol) not in (int, float) or type(atol) not in (int, float):
        raise TypeError('svd tolerances must be real numbers')
    A = _as_linear(A)
    anchor = torch.empty((), dtype=A.dtype, device=A.device)
    return SVDResult(*torch.ops.torchpairwise.linear_svd(
        *_packet(A), list(A.shape), anchor, k, subspace_size, max_iter,
        float(rtol), float(atol), initial))


__all__ += ['svd', 'SVDResult', 'SVDStatus']


class TraceResult(NamedTuple):
    estimate: torch.Tensor
    standard_error: torch.Tensor
    samples: torch.Tensor


class LogdetResult(NamedTuple):
    estimate: torch.Tensor
    standard_error: torch.Tensor
    samples: torch.Tensor
    iterations: torch.Tensor


class MatrixFunctionResult(NamedTuple):
    action: torch.Tensor
    error_estimate: torch.Tensor
    iterations: torch.Tensor


for _result in (TraceResult, LogdetResult, MatrixFunctionResult):
    _register_namedtuple(_result, serialized_type_name='torchpairwise.linalg.' + _result.__name__)


def _spectral_options(block_size, num_probes=None, seed=None, max_iter=None):
    for name, value in (('block_size', block_size), ('num_probes', num_probes), ('max_iter', max_iter)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(name + ' must be a positive integer')
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError('seed must be a nonnegative integer')


def _spectral_estimate(A, function, method, num_probes, probes, seed, max_iter, block_size,
                       differentiable=False, gradient=None, solve_rtol=None, solve_atol=None,
                       solve_max_iter=None):
    _spectral_options(block_size, num_probes, seed, max_iter)
    if probes is not None and not isinstance(probes, torch.Tensor):
        raise TypeError('probes must be a Tensor')
    if method not in (('exact', 'hutchinson') if function == 'trace' else ('exact', 'slq')):
        raise ValueError('Select exact/hutchinson for trace or exact/slq for SPD logdet')
    if method == 'exact' and any(v is not None for v in (num_probes, probes, seed, max_iter)):
        raise ValueError('Exact methods do not accept stochastic options')
    if probes is not None and (num_probes is not None or seed is not None):
        raise ValueError('Supplied probes cannot be combined with num_probes or seed')
    if method != 'exact' and probes is None and num_probes is None:
        raise ValueError('Stochastic methods require num_probes or explicit probes')
    if type(differentiable) is not bool:
        raise TypeError('differentiable must be bool')
    slq_gradient = differentiable and function == 'logdet' and method == 'slq'
    if gradient is not None and not slq_gradient:
        raise ValueError('gradient is only used by differentiable SLQ logdet')
    if not slq_gradient and any(v is not None for v in (solve_rtol, solve_atol, solve_max_iter)):
        raise ValueError('Backward solve options are only used by differentiable SLQ logdet')
    if slq_gradient and gradient != 'solve':
        raise ValueError("Differentiable SLQ requires explicit gradient='solve'; this estimates the logdet derivative, not finite SLQ's derivative")
    import math
    for name, value in (('solve_rtol', solve_rtol), ('solve_atol', solve_atol)):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(name + ' must be a finite nonnegative number')
    if solve_max_iter is not None and (type(solve_max_iter) is not int or solve_max_iter < 0):
        raise ValueError('solve_max_iter must be a nonnegative integer')
    if differentiable and probes is not None and probes.requires_grad and torch.is_grad_enabled():
        raise RuntimeError('Spectral probes are fixed algorithmic inputs; detach probes explicitly')
    A = _as_linear(A)
    torch._check(A.shape[-1] == A.shape[-2], lambda: 'Spectral consumers require a square operator')
    anchor = torch.empty((), dtype=A.dtype, device=A.device)
    if differentiable:
        result = torch.ops.torchpairwise.linear_trace_differentiable(
            *_packet(A), list(A.shape), anchor, function, method, num_probes, probes,
            seed, max_iter, block_size, 'solve' if slq_gradient else 'exact',
            1e-5 if solve_rtol is None else float(solve_rtol),
            0. if solve_atol is None else float(solve_atol),
            1000 if solve_max_iter is None else solve_max_iter)
        return result[:4]
    return torch.ops.torchpairwise.linear_trace(*_packet(A), list(A.shape), anchor,
        function, method, num_probes, probes, seed, max_iter, block_size)


def trace(A, *, method, num_probes=None, probes=None, seed=None, block_size=16, differentiable=False):
    """Exact or Hutchinson trace of a real square linear map.

    method is required: 'exact' uses a source diagonal when available, otherwise
    blocked basis actions; 'hutchinson' averages z.T A z using Rademacher probes.
    Stochastic use requires num_probes>0 or supplied [...,N,P] probes. Supplied
    probes match dtype/device and may broadcast to, but cannot add, A's batches.
    They must be independent/isotropic for the usual unbiasedness interpretation.
    A need not be symmetric for trace. No full matrix is required; a basis-action
    block can approach quadratic storage if block_size approaches N.

    A nonnegative seed uses a private device RNG without advancing global state;
    without it the global PyTorch RNG is consumed. Fixed seed, device, shape and
    block_size reproduce probes; changing blocking/device may change draws.
    Supplied probes forbid num_probes/seed. Exact forbids all random options.
    block_size bounds concurrently applied columns and must be a positive int.

    Result estimate/standard_error are float64 [...]; samples is [...,P].
    Standard error is sample std/sqrt(P), NaN for P=1; it is not a confidence
    bound. Exact returns zero standard error and [...,0] samples. By default
    inference-only; differentiable=True enables first derivatives of estimate.
    Diagnostics and probes are fixed, not differentiable. Stochastic backward
    retains a snapshot [...,N,P] of probes and replays bounded product blocks,
    without retaining forward product history. Compile/export supported;
    CUDA graph capture and higher derivatives are unsupported.
    """
    return TraceResult(*_spectral_estimate(A, 'trace', method, num_probes, probes,
                                         seed, None, block_size, differentiable)[:3])


def logdet(A, *, method, num_probes=None, probes=None, seed=None, max_iter=None, block_size=16,
           differentiable=False, gradient=None, solve_rtol=None, solve_atol=None, solve_max_iter=None):
    """Log-determinant of a real symmetric positive definite linear map.

    method='exact' explicitly materializes [...,N,N] in column blocks and uses
    double Cholesky. method='slq' averages z.T log(A) z via reorthogonalized
    Lanczos, with max_iter (default 32, capped by N) and the probe/RNG contract
    of trace. No jitter, dense fallback or eigenvalue clamping is implicit.

    Result estimate/standard_error/samples follow trace; iterations is int64
    [...,P] with actual Lanczos steps per probe (empty for exact). Standard error
    measures sampling variation only, NOT Lanczos truncation bias; compare larger
    max_iter separately. A positive observed Ritz spectrum cannot certify global
    SPD, which is the caller's assumption. Nonpositive Ritz values raise.
    max_iter is SLQ-only. Basis memory O(N*block_size*min(N,max_iter)), plus small
    projected eigensystems, input/output and source-tile workspace.

    differentiable=True enables first estimate gradients. Exact uses Cholesky
    inverse actions in blocks. SLQ additionally requires gradient='solve': with
    the SAME fixed probes z, solve A lambda=z and accumulate lambda.T (dA) z/P.
    This is a stochastic estimator of d logdet(A), NOT the derivative of the
    finite SLQ forward value. solve_rtol/atol/max_iter control backward CG
    (defaults 1e-5/0/1000); every solve must converge or backward raises.
    Retains [...,N,P] probe snapshots, not Lanczos/CG history. Diagnostics and
    probes are non-differentiable. Compile/export supported; CUDA graph capture
    and higher derivatives unsupported.
    """
    return LogdetResult(*_spectral_estimate(A, 'logdet', method, num_probes, probes,
                                          seed, max_iter, block_size, differentiable, gradient,
                                          solve_rtol, solve_atol, solve_max_iter))


def _matrix_function(A, values, function, scale, max_iter, block_size):
    import math
    _spectral_options(block_size, max_iter=max_iter)
    if type(scale) not in (int, float) or not math.isfinite(scale):
        raise ValueError('scale must be a finite real number')
    if not isinstance(values, torch.Tensor):
        raise TypeError('Matrix function RHS must be a Tensor')
    A = _as_linear(A)
    if values.dtype != A.dtype or values.device != A.device:
        raise ValueError('Matrix function operator/RHS must share dtype/device')
    torch._check(A.shape[-1] == A.shape[-2], lambda: 'Matrix functions require a square operator')
    return MatrixFunctionResult(*torch.ops.torchpairwise.linear_function(
        *_packet(A), list(A.shape), values, function, float(scale), max_iter, block_size))


def expm_multiply(A, values, *, scale=1., max_iter=32, block_size=16):
    """Approximate exp(scale*A) @ values for a real SYMMETRIC linear map.

    For graph diffusion use scale=-t with a symmetric graph Laplacian. RHS [N]
    or [...,N,R], float32/64 matching A; leading batches broadcast. Each column
    has an independent two-pass reorthogonalized Lanczos basis. max_iter is a
    fixed positive budget capped by N; exact invariant-subspace breakdown ends
    a column early. block_size limits concurrent columns, both accept any
    positive integer. Basis storage is O(N*block_size*min(N,max_iter)); no full
    matrix exponential is constructed. Output action retains input dtype.

    Result error_estimate is float64 [...,R] (or [...] for vector input): the
    norm of the difference of the final two projected actions. It is a heuristic,
    not a certified bound; infinity if only one unfinished step was available,
    zero at numerical invariant-subspace breakdown/zero RHS. iterations reports
    int64 per-column steps. Validate accuracy by increasing max_iter. Symmetry
    is assumed, not established. Inference-only, compile/export supported;
    CUDA graph capture unsupported. This is not a general nonsymmetric expm.
    """
    return _matrix_function(A, values, 'exp', scale, max_iter, block_size)


def inv_sqrt_multiply(A, values, *, max_iter=32, block_size=16):
    """Approximate A^{-1/2} @ values for real SPD A, without an inverse/root.

    Uses the same independent Lanczos projection, shapes, diagnostics, memory
    and inference/compile contract as expm_multiply. SPD is a caller assumption;
    observed nonpositive Ritz values raise. No jitter/clipping is applied.
    A small adjacent-projection difference cannot certify approximation error,
    especially for ill-conditioned A; compare increased max_iter explicitly.
    """
    return _matrix_function(A, values, 'inverse_sqrt', 1., max_iter, block_size)


__all__ += ['trace', 'TraceResult', 'logdet', 'LogdetResult', 'expm_multiply',
            'inv_sqrt_multiply', 'MatrixFunctionResult']


class PivotedCholeskyResult(NamedTuple):
    factor: torch.Tensor
    pivots: torch.Tensor
    error: torch.Tensor
    rank: torch.Tensor


class LanczosStatus(IntEnum):
    COMPLETED = 0
    BREAKDOWN = 1
    INVALID_INITIAL = 2
    NONFINITE = 3


class LanczosResult(NamedTuple):
    basis: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    iterations: torch.Tensor
    status: torch.Tensor


class LOVECache(NamedTuple):
    root: torch.Tensor


_register_namedtuple(PivotedCholeskyResult,
                     serialized_type_name='torchpairwise.linalg.PivotedCholeskyResult')
_register_namedtuple(LanczosResult, serialized_type_name='torchpairwise.linalg.LanczosResult')
_register_namedtuple(LOVECache, serialized_type_name='torchpairwise.linalg.LOVECache')


def _entrywise_lowrank_source(A, operation):
    if isinstance(A, LinearOperator) or (not isinstance(A, torch.Tensor)
                                        and not getattr(A, 'entrywise', False)):
        raise TypeError(operation + ' requires a scalar entrywise operator or dense Tensor; '
                        'build the expression entrywise or materialize with .to_dense()')
    return _as_linear(A)


def pivoted_cholesky(A, rank, *, rtol=0., atol=0.):
    """Greedy streamed PSD factorization with independent batched stopping.

    A is a scalar entrywise symmetric PSD source or dense [...,N,N] Tensor;
    linear-only maps are rejected. Symmetry/PSD are caller assumptions. Each
    pivot reads one column, using O(N*rank) entries and O(N*rank**2) arithmetic.
    The native recurrence uses float64 and returns factor [...,N,rank] in A's
    dtype, int64 pivots [...,rank], residual trace error [...] and effective
    int64 rank [...]. Unused factor columns are zero, unused pivots -1;
    error retains A's dtype. Require 1 <= rank <= N.

    Stops at rank or residual trace <= max(atol, rtol*initial_trace). Use
    lowrank_inverse(result.factor, noise) as cg's preconditioner for K+noise*I;
    result.pivots also supplies automatic Nyström landmark indices (slice to
    each batch's effective rank). Inference-only: use explicit torch.no_grad()
    for gradient-bearing inputs. Native early exit precludes CUDA graph capture.
    """
    if type(rank) is not int:
        raise TypeError('pivoted_cholesky rank must be an integer')
    if type(rtol) not in (int, float) or type(atol) not in (int, float):
        raise TypeError('pivoted_cholesky tolerances must be real numbers')
    A = _entrywise_lowrank_source(A, 'pivoted_cholesky')
    torch._check(A.shape[-2] == A.shape[-1], lambda: 'pivoted_cholesky requires a square operator')
    anchor = torch.empty((), dtype=A.dtype, device=A.device)
    return PivotedCholeskyResult(*torch.ops.torchpairwise.linear_pivoted_cholesky(
        *_packet(A), list(A.shape), anchor, rank, float(rtol), float(atol)))


def lanczos(A, k, *, initial, reorthogonalize=True):
    """Native Lanczos projection of a symmetric streamed map, inference only.

    Scalar sources, dense Tensors and linear compositions such as K+noise*I
    are accepted; symmetry is assumed. initial [...,N] matches A's dtype/device
    and broadcasts to A's batches without adding batches. Internal arithmetic
    is float64; basis [...,N,k], alpha [...,k] and beta [...,k-1] retain A's
    dtype. Full reorthogonalization is enabled by default.

    iterations/status are int64 [...] per batch. Early invariant-subspace
    breakdown sets BREAKDOWN and leaves unused entries zero; slice to iterations
    for the effective tridiagonal. COMPLETED exhausts k steps, NONFINITE reports
    arithmetic failure, INVALID_INITIAL a zero/nonfinite start. Use explicit
    torch.no_grad() for gradient-bearing inputs. Memory O(N*k); compile/export
    retain a native call, and CUDA graph capture is unsupported.
    """
    if type(k) is not int:
        raise TypeError('lanczos k must be an integer')
    if type(reorthogonalize) is not bool:
        raise TypeError('lanczos reorthogonalize must be a bool')
    if not isinstance(initial, torch.Tensor):
        raise TypeError('lanczos initial must be a Tensor')
    A = _as_linear(A)
    torch._check(A.shape[-2] == A.shape[-1], lambda: 'lanczos requires a square operator')
    anchor = torch.empty((), dtype=A.dtype, device=A.device)
    return LanczosResult(*torch.ops.torchpairwise.linear_lanczos(
        *_packet(A), list(A.shape), anchor, k, initial, reorthogonalize))


def love_cache(A, rank, *, initial):
    """Prepare an inference-only LOVE inverse root for SPD A=K+noise*I.

    Runs rank fully reorthogonalized double Lanczos steps from initial [...,N],
    factors its tridiagonal T=L L.T, and stores root=Q L^{-T}, [...,N,rank],
    in A's dtype. Breakdown pads unused columns with zeros. No jitter is added;
    nonpositive projected curvature or numerical failure raises. Use explicit
    torch.no_grad() for gradient-bearing inputs. The initial vector follows
    lanczos's dtype/device/batch contract, typically targets or a mean cache.

    The O(N*rank) snapshot approximates A^{-1} by root @ root.T and can be
    reused across test batches. Increasing rank generally improves accuracy;
    compare ranks on your problem. Rebuild after changing the training system.
    """
    if type(rank) is not int:
        raise TypeError('love_cache rank must be an integer')
    if not isinstance(initial, torch.Tensor):
        raise TypeError('love_cache initial must be a Tensor')
    A = _as_linear(A)
    torch._check(A.shape[-2] == A.shape[-1], lambda: 'love_cache requires a square operator')
    anchor = torch.empty((), dtype=A.dtype, device=A.device)
    return LOVECache(torch.ops.torchpairwise.linear_love_cache(
        *_packet(A), list(A.shape), anchor, rank, initial))


def predictive_variance(cache, cross, prior_variance):
    """Return clamped LOVE posterior marginal variances for a reusable cache.

    cross=K(X*,X) is [...,N*,N]; prior_variance is [...,N*]. Batches broadcast.
    Computes (prior_variance - (cross @ cache.root).square().sum(-1)).clamp_min(0).
    The clamp removes negative approximate variances and has zero derivative
    below zero. The root is held constant; gradients flow through the streamed
    cross action and prior_variance, including test kernel inputs.
    """
    if not isinstance(cache, LOVECache) or not isinstance(cache.root, torch.Tensor):
        raise TypeError('predictive_variance cache must be a LOVECache with a Tensor root')
    if not isinstance(prior_variance, torch.Tensor):
        raise TypeError('predictive_variance prior_variance must be a Tensor')
    cross = _entrywise_lowrank_source(cross, 'predictive_variance')
    root = cache.root
    if root.ndim < 2 or prior_variance.ndim < 1:
        raise ValueError('Use cache root [...,N,r] and prior_variance [...,N*]')
    if (root.dtype != cross.dtype or root.device != cross.device
            or prior_variance.dtype != cross.dtype or prior_variance.device != cross.device):
        raise ValueError('predictive_variance inputs must share dtype/device')
    torch._check(root.shape[-2] == cross.shape[-1], lambda: 'Cache root must match cross training axis')
    torch._check(prior_variance.shape[-1] == cross.shape[-2], lambda: 'Prior variance must match cross test axis')
    return (prior_variance - (cross @ root.detach()).square().sum(-1)).clamp_min(0)


__all__ += ['pivoted_cholesky', 'PivotedCholeskyResult', 'lanczos', 'LanczosResult',
            'LanczosStatus', 'love_cache', 'LOVECache', 'predictive_variance']
