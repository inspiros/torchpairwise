# Traces, Log-Determinants and Matrix Functions

`torchpairwise.linalg` consumes a square real linear map through its
products. Python accepts a lazy pairwise object or dense Tensor. C++ accepts a `LinearOperator` through the same `streaming::linalg`
namespace. Use `tp.as_lazy` to wrap a stored matrix.

All consumers use CPU/CUDA float32/float64. Trace and log-determinant estimates
support opt-in first gradients through `differentiable=True`; the default and
matrix-function actions require explicit `torch.no_grad()` / `at::NoGradGuard`
for gradient-bearing inputs. Compatible `torch.compile` and `torch.export`
graphs retain native forward/backward consumers. Runtime numerical checks and
breakdown use synchronization, so CUDA graph capture is unsupported. Symmetry and positive definiteness are caller assumptions;
observed failures raise, but finite probes cannot establish global SPD.

## Explicit Method Selection

| Entry | Method / Assumption | Result | Full Matrix Required |
| --- | --- | --- | --- |
| `trace(A, method="exact")` | Any square real A | Exact trace, up to roundoff | No |
| `trace(A, method="hutchinson", ...)` | Any square real A | Stochastic trace | No |
| `logdet(A, method="exact")` | Symmetric positive definite A | Double Cholesky log-determinant | **Yes**, explicitly selected |
| `logdet(A, method="slq", ...)` | Symmetric positive definite A | Stochastic Lanczos quadrature | No |
| `expm_multiply(A, V, scale=...)` | Symmetric real A | Approximate exponential action | No |
| `inv_sqrt_multiply(A, V)` | Symmetric positive definite A | Approximate inverse square-root action | No |

No hidden dense fallback, jitter, eigenvalue clipping or solver selection is
performed. A generated comparison map need not be symmetric or SPD; choose a
metric/kernel and algebra that give the required mathematical properties.

## Trace and Random Probes

For independent probes with \(\mathbb E[z_pz_p^\mathsf T]=I\),

\[
\widehat{\operatorname{tr}}(A)=\frac1P\sum_{p=1}^Pz_p^\mathsf TAz_p
\]

```python
import torchpairwise as tp

# X:[N,D], float32/64
with torch.no_grad():
    A = (lambda K: K + .1 * tp.eye_like(K))(tp.lazy(X, metric="rbf", gamma=.2))
    exact = tp.linalg.trace(A, method="exact", block_size=16)
    estimate = tp.linalg.trace(A, method="hutchinson", num_probes=64,
                              seed=17, block_size=16)
    value = estimate.estimate                # [...] float64
    sampling_error = estimate.standard_error  # [...] float64
    samples = estimate.samples               # [...,64] float64
```

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;
at::NoGradGuard no_grad;
s::PairwiseOptions source;
source.comparison.gamma = .2;
auto A = s::aslinearoperator(s::kernel(X, c10::nullopt, "rbf", source))
             .add_diagonal(.1);
s::linalg::TraceOptions options;
options.method = "hutchinson";
options.num_probes = 64;
options.seed = 17;
options.block_size = 16;
auto estimate = s::linalg::trace(A, options);
```

Stochastic methods require a positive integer `num_probes`, or explicit
`probes` of shape `[...,N,P]`. Internal probes are Rademacher (+1/-1). Supplied
probes match A's dtype/device and may broadcast to its existing batch dimensions,
without adding batches; supplying probes forbids `num_probes` and `seed`.
Arbitrary supplied vectors do not automatically have the unbiased stochastic
interpretation. For example, a complete basis scaled by \(\sqrt N\) gives an
exact trace average, but is not a set of independent random probes.

A nonnegative `seed` uses a private device generator, leaving the global RNG
state unchanged. Omitting `seed` consumes PyTorch's device RNG. Fixed seed,
shape, device and blocking reproduce draws; changing `block_size` or device can
change probes. Reuse explicit probes for common random numbers across models.
Exact methods reject stochastic options.

`TraceResult` has `estimate`, `standard_error` and `samples`. The first two
have A's batch shape; samples adds a final probe axis. Standard error is sample
standard deviation divided by \(\sqrt P\), NaN for P=1. It is not a confidence
bound. Exact returns zero standard error and an empty final sample axis.

Exact trace uses a direct source diagonal when available and otherwise applies
blocks of coordinate vectors. It does not require a full N-by-N identity or matrix. With `block_size`
approaching N, a coordinate block and its product can themselves approach
quadratic storage.
`block_size` accepts any positive integer and bounds concurrent applied columns.
Generated source tiles have their own `chunk_size` bound.

## SPD Log-Determinants

For SPD A, \(\log\det A=\operatorname{tr}(\log A)\). Lanczos starts at
\(q_1=z/\|z\|\) and constructs a small symmetric tridiagonal projection T:

\[
z^\mathsf T\log(A)z\approx\|z\|^2e_1^\mathsf T\log(T)e_1
\]

```python
with torch.no_grad():
    result = tp.linalg.logdet(A, method="slq", num_probes=64,
                              seed=17, max_iter=32, block_size=8)
    log_determinant = result.estimate        # [...] float64
    probe_steps = result.iterations         # [...,64] int64
    # Small problems only: explicitly materializes A for a Cholesky reference.
    exact = tp.linalg.logdet(A, method="exact", block_size=16)
```

```cpp
s::linalg::LogdetOptions options;
options.method = "slq";
options.num_probes = 64;
options.seed = 17;
options.max_iter = 32;
options.block_size = 8;
auto result = s::linalg::logdet(A, options);
```

`LogdetResult` adds per-probe `iterations` to the trace result. SLQ applies
independent, fully reorthogonalized Lanczos recurrences per probe, with positive
`max_iter` (default 32, capped by N). Numerical invariant-subspace breakdown
ends a column early. Observed nonpositive Ritz values raise.

There are **two different approximations**: finite random probes and finite
Lanczos depth. `standard_error` describes only probe variation. Increase
`max_iter` separately to check quadrature stability; increasing probe count
does not remove quadrature bias. Exact methods reject `max_iter` and all
stochastic options. Exact checks symmetry within roundoff and SPD by Cholesky;
it uses O(N²) storage and is intended for an explicitly chosen dense reference.

## Matrix-Function Actions

For symmetric A and a column v, Lanczos approximates the action directly:

\[
f(A)v\approx\|v\|Q_k f(T_k)e_1
\]

Only the small projected matrix function is computed. `scale` in the exponential
means \(f(\lambda)=\exp(\mathrm{scale}\,\lambda)\); for whitening,
\(f(\lambda)=\lambda^{-1/2}\).

```python
# L:[N,N] symmetric graph Laplacian as a lazy pairwise object; V:[N,R]
with torch.no_grad():
    diffusion = tp.linalg.expm_multiply(L, V, scale=-.5, max_iter=32)
    diffused = diffusion.action              # [N,R], same dtype as V
    whitened = tp.linalg.inv_sqrt_multiply(A, V, max_iter=32).action
```

```cpp
s::linalg::MatrixFunctionOptions options;
options.max_iter = 32;
options.block_size = 8;
auto diffusion = s::linalg::expm_multiply(L, V, -.5, options);
auto whitened = s::linalg::inv_sqrt_multiply(A, V, options).action;
```

V can have shape `[N]` or `[...,N,R]`; A/V batches broadcast. Use `[...,N,1]`
for batched vectors. Each RHS column has its own recurrence and steps. The
returned `MatrixFunctionResult` contains `action`, `error_estimate` and
`iterations`. Diagnostics have shape `[...]` for a vector RHS or `[...,R]` for
blocks; errors are float64 and counts int64.

`error_estimate` is the norm of the difference of the final two Krylov actions,
**a heuristic, not a certified bound**. It is infinity after one unfinished
step, zero for zero RHS or numerical invariant-subspace breakdown. Small
successive differences can miss approximation error; compare larger `max_iter`,
especially for ill-conditioned inverse roots. The budget is fixed; no tolerance
or automatic convergence guarantee is implied. Exponentials require symmetry,
including indefinite symmetric A; general nonsymmetric matrices are unsupported.

## Estimate Gradients

Set `differentiable=True` to differentiate `result.estimate` with respect to
operator inputs and tensor factors. This supports first derivatives in Python
and C++, including compatible compiled/exported forward and backward graphs.
`standard_error`, `samples` and `iterations` are fixed diagnostics. Probes are
algorithmic inputs; detach trainable probes explicitly. Supplied probes are
snapshotted by value, so later edits do not change backward's random vectors.

For exact trace, blocked product VJPs differentiate the diagonal sum. For
Hutchinson, they differentiate the returned finite-probe estimate:

\[
d\widehat{\operatorname{tr}}(A)=\frac1P\sum_pz_p^\mathsf T(dA)z_p
\]

Exact log-determinants use blocked Cholesky inverse actions:

\[
d\log\det A=\operatorname{tr}(A^{-1}dA)
\]

SLQ requires an **explicit gradient choice**. `gradient="solve"` uses the same
forward probes, solves \(A\lambda_p=z_p\) with CG, then applies product VJPs:

\[
\widehat{d\log\det A}=\frac1P\sum_p\lambda_p^\mathsf T(dA)z_p
\]

This estimates the derivative of the underlying log-determinant. **It is not the
derivative of the finite-depth SLQ forward approximation.** With independent
isotropic probes and exact solves it is unbiased; finite solves introduce
numerical error. Even an exact solve does not remove forward quadrature bias or
finite-probe variation. Finite-SLQ algorithm differentiation is not implemented.

```python
# Operator inputs/factors can require gradients.
trace = tp.linalg.trace(A, method="hutchinson", num_probes=32, seed=17,
                       differentiable=True)
exact = tp.linalg.logdet(A, method="exact", differentiable=True)
slq = tp.linalg.logdet(A, method="slq", num_probes=32, seed=17,
    max_iter=24, differentiable=True, gradient="solve",
    solve_rtol=1e-8, solve_max_iter=500)
slq.estimate.backward()
```

```cpp
s::linalg::TraceOptions trace_options;
trace_options.method = "hutchinson";
trace_options.num_probes = 32;
trace_options.seed = 17;
trace_options.differentiable = true;
auto trace = s::linalg::trace(A, trace_options);
s::linalg::LogdetOptions options;
options.method = "slq";
options.num_probes = 32;
options.seed = 17;
options.max_iter = 24;
options.differentiable = true;
options.gradient = "solve";
options.solve_rtol = 1e-8;
options.solve_max_iter = 500;
auto slq = s::linalg::logdet(A, options);
slq.estimate.backward();
```

`solve_rtol`, `solve_atol`, `solve_max_iter` control backward CG only, defaulting
to 1e-5, 0 and 1000. Every backward solve must converge or an error is raised.
They are accepted only for differentiable SLQ; `max_iter` remains the separate
forward Lanczos budget. Exact methods do not need these options. No backward
preconditioner is inferred. Higher derivatives are unsupported.

Stochastic differentiation stores a fixed `[...,N,P]` probe snapshot and
source operands, then replays products in `block_size` columns. Backward consumes
no new random numbers and stores no CG/Lanczos iteration history. Prepared source
statistics can be reused within replay. Exact logdet backward recomputes a dense
Cholesky factor; exact trace/Hutchinson and SLQ backward use blocked actions. Their source
chunk, RHS block and probe budgets determine workspace.
The usual version checks detect modified operator operands between forward and
backward. See the [kernel likelihood example](https://github.com/inspiros/torchpairwise/blob/master/examples/kernel_likelihood.py)
for bandwidth/noise learning and separate dense gradient checks.

## Memory and Reuse

With b=`block_size`, k=min(N,`max_iter`), SLQ/actions retain O(Nbk) double basis
entries and O(bk²) projected workspace per batch, plus RHS/output, probes/samples
and source preparation/tiles. Action output uses the source dtype. Explicit
probe tensors remain caller-owned; generated probes are created in blocks.
Taking b or k near N can make the basis itself large. On small problems,
Lanczos basis/setup workspace can exceed dense Cholesky storage, and repeated
products can take longer than direct factorization. Trace uses O(Nb) action
workspace; plain source diagonals may use less. Exact logdet intentionally uses
a full double matrix and factorization workspace.

Native solvers prepare source metadata once per call. Plain sources bypass
scalar-expression interpretation, and a stored matrix uses a direct ATen
product. General scalar expressions still evaluate tiles. Public operators
reference inputs so value mutations remain visible; reconstruct after resizing.
There is no persistent cross-call numerical cache or retained Krylov history.

## Explicit Lanczos Bases

`tp.linalg.lanczos(A, k, initial=v)` exposes the streamed symmetric projection
as a basis, tridiagonal coefficients and per-batch stopping diagnostics. It uses
float64 internal work and full reorthogonalization by default. It is
inference-only and accepts symmetric linear compositions as well as scalar
kernels. See [GP prediction](gp-prediction.md#inspecting-the-lanczos-projection)
for shapes, breakdown handling and the reusable LOVE inverse-root cache built
from this primitive. Existing `eigh` and `logdet` retain their own contracts.
