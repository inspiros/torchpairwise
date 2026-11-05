# Gaussian-Mixture Comparisons

These functions compare whole mixtures as samples. A collection uses means
`[...,N,K,D]`, covariances `[...,N,K,D,D]`, and explicit weights `[...,N,K]`.
Here `N` counts mixture samples, `K` is the padded component count, and `D` is
the feature dimension. A second collection may have different sample and
component counts. Only leading batch axes broadcast; sample/component axes
must agree within each collection. Inputs must be dense strided tensors.

Weights must be finite and nonnegative, with positive total mass in every
sample. Each sample is normalized independently, so multiplying its weights
by a positive constant preserves its distribution. Normalization first divides
by the largest weight to avoid overflowing the total. Optional int64
`x1_lengths`/`x2_lengths` tensors `[...,N]` select valid component prefixes.
Lengths range from 1 to `K`. Padding is ignored, including NaNs, and receives
zero gradient. Valid zero-weight components still need valid means and
covariances. Empty sample and batch axes are supported, with connected zero
gradients.

CPU and CUDA float32/float64 inputs are supported. All floating inputs need the
same dtype/device. Under autocast, float16/bfloat16 inputs are promoted to
float32 and return float32 results. No extra package is required.

## Comparisons

Let \(p=\sum_i a_i\mathcal N(\mu_i,\Sigma_i)\) and
\(q=\sum_j b_j\mathcal N(\nu_j,\Lambda_j)\), with normalized weights.

| Function | Returned quantity | Covariances |
| --- | --- | --- |
| `gaussian_mixture_overlap_kernel` | \(K(p,q)=\int p(x)q(x)\,dx\) | SPD |
| `gaussian_mixture_l2_distances` | \(\sqrt{K(p,p)+K(q,q)-2K(p,q)}\) | SPD |
| `gaussian_mixture_cauchy_schwarz_divergences` | \(-\log[K(p,q)/\sqrt{K(p,p)K(q,q)}]\) | SPD |
| `gaussian_mixture_wasserstein_distances` | \(\sqrt{\min_{\pi\in\Pi(a,b)}\langle\pi,C\rangle}\) | PSD |
| `gaussian_mixture_entropic_transport_costs` | Estimate of \(\min_{\pi\in\Pi(a,b)}\langle\pi,C\rangle+\epsilon\operatorname{KL}(\pi\Vert a\otimes b)\) | PSD |
| `gaussian_mixture_sinkhorn_divergences` | \(\operatorname{OT}_\epsilon(p,q)-\tfrac12\operatorname{OT}_\epsilon(p,p)-\tfrac12\operatorname{OT}_\epsilon(q,q)\) | PSD |

The component ground cost for all transport functions is
\(C_{ij}=W_2^2(\mathcal N(\mu_i,\Sigma_i),\mathcal N(\nu_j,\Lambda_j))\),
with no additional factor of one half.

Every function has a `paired_` and `condensed_` variant. All-pairs output is
`[...,N,M]`; paired output is `[...,N]` and requires equal sample counts.
Condensed output is `[...,N*(N-1)//2]`, in row-major strict upper-triangle order,
and accepts `x1_lengths` only. Omit all three second-collection tensors for
self comparisons. Supplying only some of them is an error.

`gaussian_mixture_l2_distances` and `gaussian_mixture_wasserstein_distances`
accept `squared=True`. Entropic costs and Sinkhorn divergences always use the
squared ground scale and never apply a square root.

## Density Overlap, L2, and CS

The component overlap is
\(\int\mathcal N(x;\mu_i,\Sigma_i)\mathcal N(x;\nu_j,\Lambda_j)\,dx
=\mathcal N(\mu_i;\nu_j,\Sigma_i+\Lambda_j)\).
The mixture overlap is the exact weighted component sum. SPD covariances are
required for ordinary Gaussian densities: the PSD policy of Gaussian W2 does
not extend to this family. Covariances must be finite and symmetric within the
relative roundoff tolerance `64*D*finfo(compute_dtype).eps`; accepted inputs are
symmetrized. Cholesky factorization must succeed without adding jitter.

The implementation evaluates log overlaps with Cholesky solves and stable
weighted log sums. CS uses natural logarithms and the convention in the table;
there is **no factor of two**. It stays finite when the overlap itself
underflows to zero, provided log overlaps remain representable. L2 uses scaled
sums to avoid overflowing intermediate density norms. Negative L2/CS roundoff
residuals are clamped to zero. Near-identical densities can still lose relative
accuracy through subtraction; CS or squared L2 may be preferable to an
unsquared L2 loss. Unrepresentable intermediate log overlaps or returned
kernel/L2 values raise an error; rescale the problem or use float64.

`chunk_size=8` bounds tiles of mixture samples. `component_chunk_size=32`
bounds the two component axes of temporary density decompositions. All-pairs
output itself remains dense. These are ordinary differentiable PyTorch
compositions. At zero L2 distance the square-root derivative is set to zero.
Zero weights use a finite continuation; no derivative guarantee is made at
that boundary.

## Component-Restricted Transport

MW2 restricts couplings to transports between Gaussian components. It generally
differs from exact Wasserstein distance between the full mixture densities.
The result depends on the component representation: it answers a component
matching question rather than just a density equality question. No exact
full-density W2 claim is made for any function here.

The exact solver supports at most **4096 positive-mass component pairs per
comparison**. It uses an eager CPU min-cost-flow solve, including for CUDA
inputs; ground costs remain PyTorch computations on the input device, but
CUDA synchronizes to transfer detached costs and masses to the CPU. The solver
uses integer capacities derived from the exact binary input-mass ratios to
avoid floating residual-mass termination cutoffs. CPU floating-point costs and
roundoff-tolerant reduced-cost comparisons still limit numerical precision.
Only one component cost matrix is formed at a time. `chunk_size` is accepted
for API consistency and does not change this serial solver's storage.
The native opaque solver supports `torch.compile`, export and FakeTensor
metadata execution. Capture preserves CPU-controlled execution and CUDA
synchronization. It is intended for small reference problems.

Exact OT gradients use one selected optimal plan for ground costs and optimal
dual potentials for the normalized masses. These are envelope derivatives on
regions with a stable optimum; ties select a subgradient. Positive weights and
SPD covariances have first-derivative support. Gaussian W2 roots use the
existing Sylvester/Procrustes derivative, including repeated positive
eigenvalues. At singular PSD covariance boundaries or zero weights, only a
finite continuation is provided; higher derivatives of transport are
unsupported. Zero unsquared MW2 distance has zero derivative.

## Entropic Costs and Debiasing

For larger mixtures, entropic transport uses `regularization=0.1` and
`max_iter=100` by default. The regularizer is relative entropy against the
product of the normalized input marginals. It is not entropy against counting
measure, and the reported full objective includes the regularizer. Self costs
are generally positive. At convergence, the dual estimate equals the full
regularized primal objective.

Both entropic functions execute exactly `max_iter` symmetric, half-relaxed
log-scaling updates. A finite iteration budget need not converge. Autograd
follows every update, including mass normalization; it does not replace this
with an envelope gradient. The debiased result is returned without a
nonnegative clamp or root. Finite-iteration estimates can be negative, and no
general positivity theorem for this Gaussian component ground cost is claimed.
Check convergence when using these values quantitatively, especially at small
regularization. There is no automatic epsilon schedule or convergence test.

Sample chunking bounds each cost tile. Entropic transport needs complete
component cost matrices and does not accept `component_chunk_size`. Ground
storage grows as `chunk_size**2*K*L`, together with Gaussian decomposition
storage; autograd retains update history. Reduce `chunk_size`, component count,
or `max_iter` to control memory. These functions are PyTorch compositions, not
specialized native transport kernels; no speedup is claimed.

## Example

```python
import torch
import torchpairwise as tp

# Two mixtures, each with three two-dimensional Gaussian components.
means = torch.randn(2, 3, 2, dtype=torch.float64)
covariances = torch.eye(2, dtype=means.dtype).expand(2, 3, 2, 2)
weights = torch.tensor([[1., 2., 1.], [3., 1., 2.]], dtype=means.dtype)

l2 = tp.gaussian_mixture_l2_distances(means, covariances, weights)
mw2 = tp.gaussian_mixture_wasserstein_distances(means, covariances, weights)
sinkhorn = tp.gaussian_mixture_sinkhorn_divergences(
    means, covariances, weights, regularization=.2, max_iter=200)
```

See [the runnable example](../examples/gaussian_mixtures.py) for unequal
component counts, lengths, paired/condensed outputs, and a differentiable loss.

For Gaussian splats, supply a deliberate probability-mass definition. Splat
opacity is not automatically mass; visibility-dependent alpha compositing
also depends on cameras and ordering. These comparisons evaluate spatial
Gaussian geometry/densities and do not evaluate color, spherical harmonics,
visibility, or rendered appearance. Rendering and camera selection remain
user-owned; existing image metrics can compare the resulting views.
