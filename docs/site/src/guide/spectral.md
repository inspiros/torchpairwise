<div class="eyebrow">Streamed operators</div>

# Traces, Log-Determinants and Matrix Functions

`tp.linalg` consumes a square real linear map through its products. Python
accepts a lazy pairwise object or dense Tensor. C++ accepts a `LinearOperator`
through the `streaming::linalg` namespace.

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

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

# X:[N,D], float32/64
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=.2)
    A = K + .1 * tp.eye_like(K)
    exact = tp.linalg.trace(A, method="exact", block_size=16)
    estimate = tp.linalg.trace(A, method="hutchinson", num_probes=64,
                              seed=17, block_size=16)
    value = estimate.estimate                 # [...] float64
    sampling_error = estimate.standard_error  # [...] float64
    samples = estimate.samples                # [...,64] float64
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

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

</div>
</div>

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

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
with torch.no_grad():
    result = tp.linalg.logdet(A, method="slq", num_probes=64,
                              seed=17, max_iter=32, block_size=8)
    log_determinant = result.estimate  # [...] float64
    probe_steps = result.iterations    # [...,64] int64
    # Small problems only: explicitly materializes A for a Cholesky reference.
    exact = tp.linalg.logdet(A, method="exact", block_size=16)
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
s::linalg::LogdetOptions options;
options.method = "slq";
options.num_probes = 64;
options.seed = 17;
options.max_iter = 32;
options.block_size = 8;
auto result = s::linalg::logdet(A, options);
```

</div>
</div>

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

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# L:[N,N] symmetric graph Laplacian as a lazy pairwise object; V:[N,R]
with torch.no_grad():
    diffusion = tp.linalg.expm_multiply(L, V, scale=-.5, max_iter=32)
    diffused = diffusion.action  # [N,R], same dtype as V
    whitened = tp.linalg.inv_sqrt_multiply(A, V, max_iter=32).action
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
s::linalg::MatrixFunctionOptions options;
options.max_iter = 32;
options.block_size = 8;
auto diffusion = s::linalg::expm_multiply(L, V, -.5, options);
auto whitened = s::linalg::inv_sqrt_multiply(A, V, options).action;
```

</div>
</div>

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

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

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

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

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

</div>
</div>

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

## Gaussian Process Prediction

For a Gaussian process with kernel \(K\) and noise variance \(\sigma^2\), the
posterior variance at test inputs needs \(A^{-1}\) for \(A=K+\sigma^2I\).
`tp.linalg.love_cache` prepares an inference-only low-rank inverse root from
native Lanczos: with orthonormal basis \(Q\) and projected tridiagonal
\(T=LL^\mathsf T\), the cache stores \(R=QL^{-\mathsf T}\) (`cache.root`), so
\(A^{-1}\approx RR^\mathsf T\) and no N-by-N matrix is formed.
`predictive_variance` then streams the rectangular test-to-training kernel:

\[
v_i\approx k(x_i,x_i)-\sum_r\bigl[K_{*X}R\bigr]_{ir}^2
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D], targets:[N], test:[T,D]; noise is a positive variance.
with torch.no_grad():
    K = tp.lazy(X, metric="rbf", gamma=.2)
    A = K + noise * tp.eye_like(K)
    cache = tp.linalg.love_cache(A, rank=32, initial=targets)  # root:[N,32]
    cross = tp.lazy(test, X, metric="rbf", gamma=.2)
    prior = tp.lazy(test, metric="rbf", gamma=.2).diagonal()       # [T]
    variance = tp.linalg.predictive_variance(cache, cross, prior)  # [T]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
namespace tp = torchpairwise::ops;
// X:[N,D], targets:[N], test:[T,D]
at::NoGradGuard no_grad;
tp::PairwiseOperationOptions source;
source.comparison.gamma = .2;
auto K = tp::pairwise_expression(X, c10::nullopt, "rbf", true, source);
auto A = tp::aslinearoperator(K).add_diagonal(noise);
auto cache = tp::love_cache(A, 32, targets);  // root:[N,32]
auto cross = tp::aslinearoperator(
    tp::pairwise_expression(test, X, "rbf", true, source));
auto prior = tp::pairwise_expression(test, c10::nullopt, "rbf", true, source)
                 .diagonal();                                  // [T]
auto variance = tp::predictive_variance(cache, cross, prior);  // [T]
```

</div>
</div>

The variance is the latent marginal, clamped at zero; add observation noise
separately for noisy predictions. One cache serves many test batches. Cache
storage is O(Nr), and each call stores O(Tr) projected values. A larger rank
improves the approximation at higher setup and prediction cost, and a starting
vector that misses an eigenspace cannot recover it, so compare ranks on
representative predictions. `tp.linalg.lanczos(A, k, initial=v)` exposes the
Lanczos basis and tridiagonal coefficients that LOVE uses. Setup requires no-grad
for gradient-bearing inputs; gradients flow through the cross kernel and prior,
not through the training solve or cache.

Landmarks and low-rank factors can also be selected without storing a Gram
matrix. `farthest_point_sampling` takes a scalar self-distance map and returns
indices with their insertion distances. `kmeans_plusplus` needs **squared**
distances and returns random seeds. Both read one row per step, evaluating O(Nk)
entries. `tp.linalg.pivoted_cholesky(K, rank)` instead builds a low-rank factor
with its pivots from the diagonal and one column per pivot.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# X:[N,D]; inference-only selectors
with torch.no_grad():
    D = tp.lazy(X, metric="euclidean")
    indices, insertion = tp.neighbors.farthest_point_sampling(D, 16)  # [16], [16]
    D2 = tp.lazy(X, metric="sqeuclidean")
    generator = torch.Generator(device=X.device).manual_seed(7)
    seeds = tp.neighbors.kmeans_plusplus(D2, 16, generator=generator)  # [16]
    pivoted = tp.linalg.pivoted_cholesky(
        tp.lazy(X, metric="rbf", gamma=.2), rank=16, rtol=1e-6)
    factor = pivoted.factor  # [N,16], padded with zero columns
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
namespace tp = torchpairwise::ops;
// X:[N,D]
at::NoGradGuard no_grad;
auto D = tp::aslinearoperator(
    tp::pairwise_expression(X, c10::nullopt, "euclidean"));
auto [indices, insertion] = tp::farthest_point_sampling(D, 16);  // [16], [16]
auto D2 = tp::aslinearoperator(
    tp::pairwise_expression(X, c10::nullopt, "sqeuclidean"));
auto seeds = tp::kmeans_plusplus(D2, 16);  // [16]
tp::PairwiseOperationOptions source;
source.comparison.gamma = .2;
auto K = tp::aslinearoperator(
    tp::pairwise_expression(X, c10::nullopt, "rbf", true, source));
auto pivoted = tp::pivoted_cholesky(K, 16, 1e-6);  // pivoted.factor:[N,16]
```

</div>
</div>

The selected indices can supply the landmarks of a Nyström preconditioner, as in
[low-rank preconditioning](algebra.md#low-rank-and-nyström-preconditioning).
See the [GP prediction notes](https://github.com/inspiros/torchpairwise/blob/master/docs/gp-prediction.md)
for the LOVE construction and selector contracts.

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
