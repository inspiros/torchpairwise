# Entropic transport over lazy costs

`tp.optimize.sinkhorn` solves transport for a scalar lazy cost matrix, with
nonuniform masses and streamed reductions. Costs can be built-in metrics,
entrywise expression graphs, `make_pairwise` comparisons, or dense tensors
wrapped with `tp.as_lazy`. Linear maps without entrywise costs are rejected.

```python
import torch
import torchpairwise as tp

x, y = torch.rand(12, 2), torch.rand(9, 2)
C = tp.lazy(x, y, 'sqeuclidean', chunk_size=4)
a, b = x.new_full((12,), 1/12), y.new_full((9,), 1/9)
fit = tp.optimize.sinkhorn(C, a, b, epsilon=.1)
plan = tp.optimize.sinkhorn_plan(C, fit, a, b, epsilon=.1)
row_mass = plan.sum(-1)
transported_features = plan @ y
barycentric_map = transported_features / row_mass.unsqueeze(-1)
```

The plan stays lazy: reductions and matrix products stream cost tiles. The
barycentric map is defined only on rows with positive transported mass. Explicitly
materializing a plan still requires storage proportional to the number of pairs.
The plan has the cost dtype, so its products accept values of that dtype; the
solver potentials and objective use float64.

## Solver contract

```python
sinkhorn(C, a=None, b=None, *, epsilon, max_iter=1000, rtol=1e-6,
         atol=0., reach=None, epsilon_scaling=None, initial=None,
         check_every=10)
```

`C` has shape `[..., N, M]`; masses have shapes `[..., N]` and `[..., M]`.
Batch axes broadcast. Omitted masses are uniform probability masses. Supplied
masses must be finite and nonnegative, with strictly positive totals; zero
entries are supported. Balanced
problems require equal total masses within relative tolerance `1e-6`. Masses are never
silently normalized. `epsilon` must be positive and uses the same units as `C`.

The result contains `f`, `g`, `objective`, `iterations`, `converged`, `status`,
and `marginal_error`. Balanced error is the row-marginal L1 residual. Unbalanced
error sums both L1 residuals against `a*exp(-f/reach)` and
`b*exp(-g/reach)`, the optimality targets rather than the original masses.
Status values are `SinkhornStatus.CONVERGED`,
`ITERATION_LIMIT`, and `NONFINITE`. Check convergence before using a result as a
converged optimum. `initial=(f, g)` supplies warm-start potentials.

The coupling is

\[
\pi_{ij}=a_i b_j\exp((f_i+g_j-C_{ij})/\varepsilon).
\]

Always pass the same cost, masses, and epsilon to `sinkhorn_plan` that were used
for the solve. The result does not store them.

## Objectives and differentiation

Balanced transport minimizes

\[
\langle\pi,C\rangle+\varepsilon\operatorname{KL}(\pi\mid a\otimes b),
\qquad \pi\mathbf1=a,\quad\pi^T\mathbf1=b.
\]

Here generalized KL includes the `-mass + reference_mass` terms. For probability
masses, the converged balanced dual value is
`(a*f).sum(-1) + (b*g).sum(-1)`. Potentials are determined only up to an opposite
constant shift. Comparisons should use the objective, coupling, or potentials
with their gauge aligned.

For equal totals `t` other than one, the returned balanced value follows the
same potential-sum convention. The generalized-KL objective above differs by
`-epsilon * (t - t**2)` at convergence. Its envelope derivative is used for
the returned objective, so mass gradients of the potential-sum convention differ
off the fixed-total manifold.

With `reach=rho > 0`, marginal constraints become penalties
`rho * KL(pi.sum(-1) | a) + rho * KL(pi.sum(-2) | b)`. The dual is

\[
-\rho\sum_i a_i\operatorname{expm1}(-f_i/\rho)
-\rho\sum_j b_j\operatorname{expm1}(-g_j/\rho)
-\varepsilon\left(\sum_{ij}\pi_{ij}
                    -(\sum_i a_i)(\sum_j b_j)\right).
\]

Updates use the relaxation factor `rho/(rho+epsilon)`. Large reach approaches
balanced transport when the supplied masses have equal totals. The regularized
objective includes entropy and, for unbalanced problems, marginal penalties; it
is different from the transport contribution `(plan * C).sum()`.

Iterations execute in a native operator and return detached potentials. A
native differentiable expression evaluation supplies the envelope derivative
without retaining an iteration graph. At convergence, the cost derivative is
the coupling; gradients also propagate to metric inputs and parameters. Mass
derivatives include the dependence of the KL reference measure on the masses.
Balanced mass perturbations must preserve equal totals; potential gauge shifts
change unconstrained mass gradients by constants. The envelope derivative is an
approximation when the solver stops before convergence. This interface does not
differentiate the iterations or the returned potentials.

## Divergences and composition

```python
C_xx = tp.lazy(x, metric='sqeuclidean', chunk_size=4)
C_yy = tp.lazy(y, metric='sqeuclidean', chunk_size=4)
loss = tp.optimize.sinkhorn_divergence(C, C_xx, C_yy, a, b, epsilon=.1)
```

The divergence is `OT(a,b) - OT(a,a)/2 - OT(b,b)/2`, using the three supplied
costs. Nonnegativity depends on the ground cost and convergence; arbitrary
scalar cost expressions need not define a nonnegative divergence. Identical
inputs and consistent self costs give zero up to numerical error.

Fixed-support barycentres can be composed from repeated solves and updates to
support masses or locations. This API does not provide a barycentre optimizer.
The [point-set transport APIs](transport.md) retain their separate cloud layout
and fixed-cost comparison contract.

## Execution and limits

The native solver builds an expression reduction program once and changes its
row and column vector slots between alternating log-domain updates. It avoids
materializing the full cost matrix. Computation still visits all pairs per
half-step. Potentials and expression arithmetic use float64; cost leaves
evaluate in their original dtype before promotion. Native dispatch also supports standalone C++/LibTorch, compilation,
and export; arbitrary Python custom comparisons retain their own deployment
requirements.

`check_every` controls convergence checks. Early stopping synchronizes device
execution, and CUDA graph capture is not supported. Epsilon scaling
(`epsilon_scaling=q`, `0 < q < 1`) warm-starts progressively smaller regularization
stages, starting at `max(epsilon, abs(max(C)))`, with the streamed maximum
taken globally across batches. It can help small-epsilon problems, but does not guarantee fewer total
iterations. Compare convergence and total work for the actual workload.
