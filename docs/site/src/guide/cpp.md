<div class="eyebrow">Integration</div>

# Python and C++

Built-in comparisons and reductions execute native forward and first backward.
Python binds tensors, metric names and options; standalone C++ calls the same
algorithms without Python callbacks. Include `<torchpairwise.h>` and link the
[CMake target](install.md). To build your own extension on the installed
package, see [Extending from C++](extensions.md).

## Structured Samples and Options

A Gaussian is a mean/covariance pair; a mixture adds weights. Python uses tuples,
C++ uses `ComparisonCollection`. Explicit parameters use keyword arguments or a
`ComparisonOptions` struct. Unused supplied options are errors.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
# mu:[D], covariance:[D,D], with optional leading batches
value = tp.compare((mu, covariance), (other_mu, other_covariance),
                   metric="gaussian_wasserstein_distances",
                   mode="single", squared=True)
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
tp::ComparisonOptions options;
options.squared = true;
tp::ComparisonCollection x{mu, covariance};
tp::ComparisonCollection y{other_mu, other_covariance};
auto value = tp::compare(x, y, "gaussian_wasserstein_distances",
                         "single", options);
```

</div>
</div>

## Native Source Factories

| Input family | ScalarExpression factory / options |
|---|---|
| Vector/point-cloud distances and vector kernels | `pairwise_expression` / `PairwiseOperationOptions` |
| Covariances, Gaussians, mixtures | `structured_expression` / `StructuredOperationOptions` |
| Images | `media_expression` / `ImageComparisonOptions`, `PairwiseOperationOptions` |
| Audio | `media_expression` / `AudioComparisonOptions`, `PairwiseOperationOptions` |
| Sequences, tokens, MMD statistic | `ordered_expression` / `SequenceOperatorOptions`, `TokenOperatorOptions`, `MMDOperatorOptions` |
| Caller matrix | `matrix_expression` / `MatrixOperationOptions` |

`torchpairwise::streaming` re-exports these factories and adds the short forms
`pairwise`, `kernel`, `matrix`, `residual` and `image_map`, plus the `linalg` and
`optimize` solver namespaces used throughout the [streamed operator guides](streaming.md).
Newer helpers such as Sinkhorn, pair-mask constructors and LOVE live in
`torchpairwise::ops`. Named C++ family APIs also exist. Their native symbol names are not always an
identical per-metric Python name; use the mode-aware `compare` selector for a
uniform call. See [native headers and signatures](https://github.com/inspiros/torchpairwise/blob/master/docs/native-api.md)
and the [LibTorch examples](https://github.com/inspiros/torchpairwise/tree/master/examples/libtorch).

## Compatibility

Match LibTorch's major/minor, ABI, toolchain and CUDA runtime with the library.
Rebuild native consumers and re-export graphs when upgrading changed operator
schemas. Generic cdist/pdist schemas include the broad catalog and collection
overloads; older published binaries do not expose the same surface.
