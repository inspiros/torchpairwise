# Probability and Histogram Comparisons

The following operations accept dense floating point inputs of shape
`[..., N, D]`. Only leading batch dimensions broadcast. Named functions and
`cdist(..., metric=...)` return `[..., N, M]`; `paired_distances` compares
corresponding rows and returns `[..., N]`. The symmetric operations also support
`condensed_distances`, in the same upper-triangle order as `pdist`.

Except for Itakura-Saito, each input row is a nonnegative histogram with positive
finite total mass. Rows are normalized independently: \(p_i=x_i/\sum_j x_j\)
and \(q_i=y_i/\sum_j y_j\). Multiplying either row by a positive scalar therefore
does not change the result. Entries must be finite, and zero-total rows are
rejected. No pseudocount is added.

| Function | Dispatch name | Definition | Symmetric |
| --- | --- | --- | --- |
| `total_variation_distances` | `total_variation` | \(\frac12\sum_i|p_i-q_i|\) | Yes |
| `fisher_rao_distances` | `fisher_rao` | \(2\arccos\sum_i\sqrt{p_iq_i}\) | Yes |
| `renyi_divergences(..., alpha=1.)` | `renyi` | \(\frac{\log\sum_i p_i^\alpha q_i^{1-\alpha}}{\alpha-1}\), with the KL limit at \(\alpha=1\) | No |
| `pearson_chi2_divergences` | `pearson_chi2` | \(\sum_i(p_i-q_i)^2/q_i\) | No |
| `neyman_chi2_divergences` | `neyman_chi2` | \(\sum_i(p_i-q_i)^2/p_i\) | No |
| `triangular_discrimination_distances` | `triangular_discrimination` | \(\sum_i(p_i-q_i)^2/(p_i+q_i)\) | Yes |
| `itakura_saito_divergences` | `itakura_saito` | \(\sum_i[x_i/y_i-\log(x_i/y_i)-1]\) | No |

All logarithms are natural. Rényi's `alpha` must be a finite Python scalar
strictly greater than zero. The default `alpha=1.` equals `kl_divergences`.
Pearson's denominator is the second operand; Neyman's is the first, so
`neyman_chi2_divergences(x, y)` equals
`pearson_chi2_divergences(y, x).transpose(-1, -2)`. Triangular discrimination
has no factor of one half; its range on normalized histograms is \([0,2]\).
Total variation lies in \([0,1]\).

Fisher-Rao uses the convention with factor two and range \([0,\pi]\).
The implementation evaluates the equivalent expression
\(4\arcsin(\|\sqrt p-\sqrt q\|_2/2)\), which preserves small distances near
identity even when the Bhattacharyya coefficient rounds to one.

Itakura-Saito compares **raw, strictly positive** entries. It does not normalize
rows. This preserves the usual power-spectrum divergence: scaling both operands
by the same positive factor preserves the result, while scaling just one operand
generally changes it. Finite entries remain admissible when their row sum would
overflow, since no row sum is needed. Zero entries are rejected instead of
assigning an ambiguous boundary extension.

## Support and Gradients

A joint zero bin contributes zero to every normalized comparison. A zero
denominator with a positive numerator gives infinite Pearson or Neyman
divergence. For Rényi, positive \(p_i\) with \(q_i=0\) gives infinity when
\(\alpha\geq1\). For \(0<\alpha<1\), only common positive support contributes
to the power sum; disjoint support gives infinity. Bins with \(p_i=0\)
contribute zero for every \(\alpha>0\), including when \(q_i=0\).

The functions differentiate their ordinary formulas on strictly positive smooth
inputs. At raw zero bins, the chosen derivative is zero, including the
normalization contribution. Total variation uses the zero subgradient at equal
probabilities. Fisher-Rao uses a zero gradient at identity. Infinite pairs
contribute zero gradients for both operands, including output overflow for the
tiled comparisons; finite pairs in the same result keep their gradients.
These conventions do not claim a finite analytic derivative at a support change.

## Precision and Memory

Outside autocast, inputs must have matching `float32` or `float64` dtype. Under
CPU or CUDA autocast, lower precision inputs are evaluated in `float32` and
return `float32`. Positive `float32` inputs use double intermediates before the
final output cast, preserving support and small divergence values. The final
output can still round to zero or overflow its dtype.

The reductions use sample blocks of at most 32 by 32 and feature blocks of at
most 64. Autograd checkpoints feature blocks rather than retaining all pair by
feature intermediates. Fisher-Rao uses the same tiles to reduce squared differences of square
root probabilities. No implementation materializes an entire `N * M * D` buffer.
The all-pairs output still requires `N * M` storage; use paired or condensed
output when those match the task.

```python
import torch
import torchpairwise as tp

x = torch.tensor([[1., 3., 2.], [2., 1., 4.]])
y = torch.tensor([[3., 2., 1.], [1., 4., 2.]])

tv = tp.total_variation_distances(x, y)
renyi = tp.cdist(x, y, metric='renyi', alpha=.5)
paired = tp.paired_distances(x, y, metric='fisher_rao')
triangle = tp.condensed_distances(x, metric='triangular_discrimination')
```
