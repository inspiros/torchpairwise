<div class="eyebrow">Integration</div>

# Precision and Deployment

Floating comparisons generally use matching float32/float64 dense CPU/CUDA
tensors. Under supported CPU/CUDA autocast, half/bfloat16 inputs are promoted
to float32. Direct reduced precision outside autocast is operation-specific;
do not assume catalog-wide support. Boolean/equality comparisons and token
sequences have separate dtype and derivative contracts.

## Autograd

Most numeric metrics and scalar consumers support first derivatives. Higher
order derivatives and general torch.func/vmap transforms are outside the common
built-in contract. At ties, zero distances, singular covariances, empty supports
and nonfinite results, derivatives follow the metric-specific conventions.
Finite precision may still overflow; float64 is not an exact arithmetic mode.

## Compile and Export

Metric names, modes, scalar parameters and source construction are structural.
Construct operators inside forward, keep selection fixed, and use explicit
`.matmul()` spelling for a streamed product. Built-in numerical execution does
not depend on arbitrary Python callbacks. Compiled and exported graphs always use
native execution, even inside a `tp.backend("keops")` context.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
class Query(torch.nn.Module):
    def forward(self, X, Y, V):
        K = tp.lazy(X, Y, metric="rbf", gamma=0.2)
        return K.matmul(V)  # X:[N,D], Y:[M,D], V:[M,C] -> [N,C]

query = torch.compile(Query(), fullgraph=True)
# torch.export.export(Query(), (X,Y,V)) captures the fixed source/consumer.
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
// The same numerical operation is available directly to LibTorch.
tp::PairwiseOperationOptions options;
options.comparison.gamma = 0.2;
auto K = tp::pairwise_expression(X, Y, "rbf", true, options);
auto output = K.matmul(V);  // [N,C]
// Export/compilation is performed by the chosen deployment frontend.
```

</div>
</div>

Export/save/load requires the compatible native library and its operator
registrations in the target environment. Dynamic shape support depends on the
source and operation; captured control flow can specialize shapes. Exact discrete
transport remains CPU-controlled, including CUDA cost transfers; export does not
turn that solver into a GPU-only algorithm.

See [AMP/compile/export contracts](https://github.com/inspiros/torchpairwise/blob/master/docs/pytorch-integration.md)
for supported paths and [numerical conventions](https://github.com/inspiros/torchpairwise/blob/master/docs/api-contracts.md)
for empty, Boolean and boundary behavior. TorchScript compatibility is optional.
