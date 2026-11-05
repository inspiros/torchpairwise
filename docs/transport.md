# Multidimensional Point-Set Transport

The development API provides three distinct point-set comparisons:
`wasserstein_distances`, `sliced_wasserstein_distances` and `sinkhorn_divergences`.
The [shared-grid 1D histogram Wasserstein](missing-and-ordered-data.md) API retains
its separate histogram-mass interpretation. No new runtime dependency is needed.

Multidimensional transport is useful when each sample is a distribution of
points: XYZ clouds, joint color distributions, spatial histograms, or collections
of learned descriptors. Unlike nearest-neighbor Chamfer, transport constrains
how much mass each target point can receive. It captures distributional geometry
and density, rather than only closeness to some target point.

## What Distance Would We Compute?

For points x_i,y_j and normalized nonnegative masses a_i,b_j, define

\[
W_p(a,b)=\left(\min_{\pi\geq0,\ \pi\mathbf1=a,\ \pi^T\mathbf1=b}
  \sum_{ij}\pi_{ij}\|x_i-y_j\|_2^p\right)^{1/p}\]

Here p is the transport order; the ground norm is Euclidean. In particular,
W_1 in several dimensions does not mean Manhattan distance between coordinates.
[SciPy's multidimensional W1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance_nd.html)
solves a discrete transport problem between support points and optional masses.

Computing separate 1D distances for the C coordinate marginals loses joint
structure. For example, equal masses on `{(-1,-1),(1,1)}` and
`{(-1,1),(1,-1)}` have identical x and y marginals, but their Euclidean W1 is 2.
Flattening a spatial histogram onto an arbitrary 1D bin order also changes its
geometry. Spatial bin coordinates must be supplied as support locations.

| Approach | Meaning and tradeoff | Recommended role |
| --- | --- | --- |
| Exact discrete Wp | Optimizes the full mass-constrained coupling, subject to floating-point arithmetic. More expensive than sorting 1D samples. | Implemented for small clouds with an eager CPU-controlled solver, including for CUDA inputs. |
| Sinkhorn divergence | Iterative regularized transport. The debiased divergence subtracts half of each self-transport objective. Regularization and convergence affect results. | Implemented as a fixed-budget differentiable dual-objective estimate. |
| Sliced Wasserstein | Averages powered 1D transport distances over unit projection directions, then takes the pth root. Finite directions approximate the sliced distance, which is a different quantity from full Wp. | Implemented with projection/sort preprocessing and exact projected quantile integration. |

Do not treat a transport-only cost under an entropically optimized coupling,
the complete regularized objective, and a debiased divergence as interchangeable.
For example, [POT's `sinkhorn2`](https://pythonot.github.io/all.html#ot.sinkhorn2)
returns the transport contribution without entropy. Our divergence API uses the
full KL-regularized objective described below. Taking a square root of an
arbitrary Sinkhorn loss does not make it exact W2.

## Point-Set Contract

A transport sample is a collection of support vectors, using the existing
[point-set layout](pointsets.md). Ordinary dense tensors suffice:

| Mode | Inputs | Output |
| --- | --- | --- |
| All-pairs | `[N,P,C]`, `[M,Q,C]` | `[N,M]` |
| Paired | `[N,P,C]`, `[N,Q,C]` | `[N]` |
| Condensed | `[N,P,C]` | `[N*(N-1)//2]` |

The existing batch/layout selectors and prefix lengths apply. Each valid point
has uniform mass within its cloud. Clouds must be nonempty, with finite valid
coordinates; padding may contain NaNs. User-supplied nonuniform masses are not
supported. This differs from `wasserstein_1d_distances`, whose inputs are histogram
masses on shared ordered bins. Projected support values must not be passed to
that function as if they were masses.

All three functions support p=1 and p=2. Sliced Wasserstein requires explicit,
shared unit projection columns `[C,L]`. Callers can generate and retain their
directions;
the distance operation need not hide random sampling. Reuse the same directions
across sample pairs, batches, paired calls and condensed calls to keep results
comparable. A fresh independent draw per pair would break those comparisons.
Its generic metric string is `"sliced_wasserstein"`. Keep projection tensors in
float32 under autocast so quantization does not violate their unit-norm check.

For equal point counts P and uniform masses, project and sort each cloud once.
The finite-direction result is

\[
\widehat{SW}_p(X,Y)=\left(\frac{1}{LP}\sum_{\ell=1}^L\sum_{r=1}^P
  |s_r(X\theta_\ell)-s_r(Y\theta_\ell)|^p\right)^{1/p},
\]

where s_r is the rth sorted projected value. Flattening these sorted projections
reduces comparisons to scaled existing Lp operations. Paired and condensed modes
avoid building a square matrix. Unequal counts and prefix lengths use the exact
projected quantile integral, without truncation or interpolation to an arbitrary
common sample count. Stable sorting selects a derivative at projection ties;
absolute-value and zero-distance derivatives use zero subgradients.

Finite projections can miss differences and do not ensure rotation invariance
when held fixed. Increasing L reduces projection-sampling error, not the
difference between sliced and full Wasserstein. Coordinate scales remain part
of the ground geometry. No equivalence to full Wp or performance advantage is
implied. [POT's sliced implementation](https://pythonot.github.io/gen_modules/ot.sliced.html#ot.sliced.sliced_wasserstein_distance)
provides a reference with explicit projections, p and optional point weights.

## Dense Cloud Output versus Inner Solver Storage

Returning every cloud comparison still needs N*M output entries. It does not
require retaining all point comparisons. A naive dense transport implementation
would additionally hold `[N,M,P,Q]` costs. At N=M=100 and P=Q=1000, float32 costs
alone occupy 40 GB, while the output needs only 40 KB (decimal units).

Tiling cloud comparisons bounds the number of simultaneous transport problems.
For one pair, a conventional dense Sinkhorn iteration costs O(P*Q), after ground
cost construction, with additional iterations for convergence. Online reductions
can avoid storing the full point-cost matrix, though they do not remove the
underlying pair interactions. [GeomLoss](https://www.kernel-operations.io/geomloss/)
already supplies differentiable Sinkhorn divergences and an online backend with
linear memory. This is a concrete use case for the reduction approach discussed
in the [reduction guide](reductions.md), even when the final cloud matrix is dense.

## Exact Transport Execution and Gradients

`wasserstein_distances(X,Y,p=1)` uses a balanced min-cost-flow algorithm. Uniform
marginals become integer capacities without repeating points to a common sample
count. It forms one detached P*Q ground-cost matrix per cloud pair and stores
selected transport edges for the coordinate derivative. W2 includes its square
root; both orders return distances in the original coordinate units.

The optimization loop runs serially on the CPU in double precision, including
when X and Y are CUDA tensors. CUDA costs synchronize and transfer to the host;
selected-edge value/gradient operations execute on the input device, and outputs
have the input dtype/device. This is intended for small clouds and reference
comparisons. It is not a native CUDA transport solver. `chunk_size` is accepted
but does not lower its existing per-pair dense solver storage.

First coordinate gradients differentiate through a chosen optimal plan. At plan
ties this selects a valid generalized gradient; zero distances have zero chosen
gradients. Native opaque transport operators support full-graph compilation,
export and metadata execution. The solver itself still executes on the CPU. [POT's exact `emd2`
solver](https://pythonot.github.io/all.html#ot.emd2) also uses a CPU backend for
backend-compatible inputs; backend compatibility alone does not imply CUDA
optimization.

## Sinkhorn Objective and Iteration Budget

For probability marginals a,b, let

\[
F_\varepsilon(a,b)=\min_{\pi\mathbf1=a,\ \pi^T\mathbf1=b,\ \pi\geq0}
\langle\pi,C\rangle+\varepsilon\,\mathrm{KL}(\pi\Vert a\otimes b),
\quad C_{ij}=\|x_i-y_j\|_2^p\]

`sinkhorn_divergences` estimates
`F(X,Y) - F(X,X)/2 - F(Y,Y)/2`. The default p is 2, with
`regularization=0.1` and `max_iter=100`. There is no 1/p ground-cost factor,
square root, or nonnegative clamp. The result has units of coordinates to power
p; regularization has those same units. Scaling coordinates by s requires scaling
regularization by s**p to preserve the relative regularization strength.

The solver executes exactly `max_iter` symmetric half-relaxed log-domain
updates. Each F is evaluated as a dual objective, equal to the full regularized
primal at convergence. At a finite budget this is an estimate: convergence and
nonnegative debiased results are not guaranteed. Increase the budget and compare
results for the actual workload, especially at small regularization. Symmetric
updates preserve symmetry at finite budgets. Autograd differentiates every
executed update, rather than substituting a converged-solver derivative.

The implementation stores dense point costs for each active cloud pair; explicit
`chunk_size` enables existing tile checkpointing during training. Without it,
unrolled iterations retain potentially large autograd intermediates. Self-cloud
objectives are currently recomputed across tiles/condensed rows. For large-scale
transport, use GeomLoss's established online implementation rather than expecting
this initial implementation to provide comparable scaling. This divergence is
not a metric with a guaranteed triangle inequality, even at convergence.

## Usage

```python
import torch
import torchpairwise as tp

X = torch.rand(3, 20, 2)
Y = torch.rand(4, 15, 2)
# Retain one set of shared directions across every comparison mode.
directions = torch.randn(2, 32)
directions = directions / directions.norm(dim=0)
D = tp.sliced_wasserstein_distances(X, Y, projections=directions)
exact = tp.wasserstein_distances(X, Y, p=1)
regularized = tp.sinkhorn_divergences(X, Y, regularization=.2, max_iter=100)
paired = tp.paired_distances(X, Y[:3], metric="sliced_wasserstein",
                             projections=directions)
condensed = tp.pdist(X, metric="sliced_wasserstein", projections=directions)
```

The generic names are `"wasserstein"`, `"sliced_wasserstein"`, and `"sinkhorn"`.
They work in `cdist`, `paired_distances`, `pdist`, and dense neighbor masks. Named
and generic entry points share autocast promotion to float32. All three execute
through native C++ operators, including their first coordinate gradients.
`transport_comparisons` and `builtin_comparisons` provide the native typed
selection APIs, including the aligned native/Python `cdist` catalog.
Sliced projections/sorts and distribution self terms are reused within forward.
Standard float32/float64 inputs are supported; numerical overflow/underflow limits
still apply. Ground costs must be representable; Sinkhorn also rejects scaled
costs that overflow its compute dtype.

See the [runnable example](../examples/transport.py). [POT](https://pythonot.github.io/quickstart.html)
remains a better fit for a broad transport toolkit, explicit masses, couplings,
and transport-specific algorithms outside this package's current scope.

## Transport over arbitrary lazy costs

For nonuniform masses, arbitrary scalar ground costs, streamed couplings and
barycentric maps, use [`tp.optimize.sinkhorn`](entropic-transport.md).
It accepts built-in metrics, entrywise lazy expressions, custom comparisons,
and dense costs, and reports convergence diagnostics. This is a separate solver
from the point-cloud `sinkhorn_divergences` comparison described above.
