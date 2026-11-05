<div class="eyebrow">Custom Comparisons</div>

# User-Defined Metrics

Define one comparison between intact samples. A vector sample is `[D]`, a point
set `[P,D]` and an image `[C,H,W]`; `sample_ndim` counts those trailing axes.
The function returns one scalar Tensor. Different sample extents are allowed
when the function supports them. **Example: weighted squared distance.** This is
one possible formula; the adapter accepts any scalar comparison satisfying the
callback contract, including asymmetric similarities.

\[
f(x,y;w)=\sum_d w_d(x_d-y_d)^2
\]

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise as tp

weight = torch.nn.Parameter(torch.ones(4))
def forward(x, y, w):
    return ((x - y).square() * w).sum()

metric = tp.make_pairwise(forward, parameters=(weight,), symmetric=True)
X = torch.randn(3, 4)
Y = torch.randn(5, 4)
matrix = metric(X, Y)                         # [3,5]
paired = metric(X, Y[:3], mode="paired")      # [3]
condensed = metric(X, mode="condensed")       # [3]
value = metric(X[0], Y[0], mode="single")     # scalar
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch, No User Registration</div>

```cpp
#include <torchpairwise.h>
namespace tp = torchpairwise::ops;

auto weight = at::ones({4}).set_requires_grad(true);
tp::CustomComparisonOptions options;
options.parameters = {weight};
options.symmetric = true;
auto metric = tp::make_pairwise(
    [](const at::Tensor& x, const at::Tensor& y, at::TensorList p) {
      return ((x - y).square() * p[0]).sum();
    }, options);
auto X = at::randn({3, 4});
auto Y = at::randn({5, 4});
auto matrix = metric(X, Y);                         // [3,5]
auto paired = metric(X, Y.slice(0, 0, 3), "paired"); // [3]
auto condensed = metric(X, c10::nullopt, "condensed"); // [3]
auto value = metric(X[0], Y[0], "single");           // scalar
```

</div>
</div>

A factory `mode` supplies the default; a call can override it. Condensed mode
requires an explicit `symmetric=True` assertion and takes one collection.
Both paired and single modes require the second input. Symmetry is not inferred.

| Mode | Inputs for One Trailing Sample Shape S | Output |
|---|---|---|
| <span class="badge mode-all-pairs">all-pairs</span> | `[...,N,*Sx]`, `[...,M,*Sy]` | `[...,N,M]` |
| <span class="badge mode-paired">paired</span> | `[...,N,*Sx]`, `[...,N,*Sy]` | `[...,N]` |
| <span class="badge mode-condensed">condensed</span> | `[...,N,*S]` | `[...,N*(N-1)//2]` |
| <span class="badge mode-single">single</span> | `[...,*Sx]`, `[...,*Sy]` | `[...]`; scalar without batches |

Leading batches broadcast. Explicit parameters are shared tensors with the
input dtype/device; they are not interpreted as additional batch dimensions.
Inputs and scalar outputs use real floating dtypes. `output_dtype` declares a
different result dtype; the callback must return it. No implicit cast is added.

## Supply a Backward Formula

Ordinary PyTorch forwards use autograd. An optional `backward` supplies a
vector-Jacobian product: given the upstream scalar gradient, return gradients
for x, y and each explicit parameter, in that order.

```python
def backward(g, x, y, w):
    delta = x - y
    return (2 * g * w * delta,
            -2 * g * w * delta,
            g * delta.square())

metric = tp.make_pairwise(forward, backward=backward,
                          parameters=(weight,), symmetric=True, compile=True)
```

Return a Tensor for every gradient, using zeros for unused operands.
With `compile=True`, forward and VJP tiles compile separately.
This uses no full Jacobian. Repeated samples, broadcast dimensions and shared
parameter gradients accumulate according to PyTorch semantics.

## Use an Autograd Function

Pass `torch.autograd.Function.apply` directly, without a separate `backward=`
argument. The adapter uses the function's own backward for dense and streamed
comparisons. In C++, wrap the function's `apply` in a lambda; the factory handles
registration internally.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python Autograd Function</div>

```python
import torch
import torchpairwise as tp

class SquaredDistance(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, y):
        return (x - y).square().sum()

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        dx = 2 * grad_output * (x - y)
        return dx, -dx

custom = tp.make_pairwise(SquaredDistance.apply, symmetric=True, compile=True)
X = torch.randn(3, 4, requires_grad=True)
Y = torch.randn(5, 4, requires_grad=True)
values = custom(X, Y)                          # [3,5]
loss = tp.lazy(X, Y, custom, chunk_size=2).mean()  # scalar
loss.backward()                               # populates X.grad and Y.grad
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ Autograd Function, No User Registration</div>

```cpp
#include <torch/autograd.h>
#include <torchpairwise.h>

class SquaredDistanceFunction
    : public torch::autograd::Function<SquaredDistanceFunction> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& x, const at::Tensor& y) {
    ctx->save_for_backward({x, y});
    return (x - y).square().sum();
  }
  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& gradients) {
    auto inputs = ctx->get_saved_variables();
    auto dx = 2 * gradients[0] * (inputs[0] - inputs[1]);
    return {dx, -dx};
  }
};
auto custom = torchpairwise::ops::make_pairwise(
    [](const at::Tensor& x, const at::Tensor& y) {
      return SquaredDistanceFunction::apply(x, y);
    });
auto X = at::randn({3, 4}).set_requires_grad(true);
auto Y = at::randn({5, 4}).set_requires_grad(true);
auto values = custom(X, Y);                      // [3,5]
auto loss = custom.expression(X, Y).reduce("mean"); // scalar
loss.backward();                                // fills X.grad(), Y.grad()
```

</div>
</div>

Python scalar callbacks are vectorized with `torch.vmap`, so the function needs
separate `forward`/`setup_context` methods and a batching rule. Here all operations
support batching, so `generate_vmap_rule=True` supplies it automatically. A
function calling a custom native/CUDA operation may need an explicit `vmap` rule.
Trainable shared tensors belong in `parameters=` and become extra `apply`
arguments; return their gradients from the function's backward as usual.

## Evaluate Whole Tiles

The default scalar form compares one pair of intact samples and relies on
`torch.vmap`. When the formula already works with ordinary broadcasting, pass
`form="broadcast"` to evaluate a tile at once. The function receives
`x:[*Lx,*S]` and `y:[*Ly,*S]`, reduces the `sample_ndim` sample axes itself and
returns the broadcast leading shape. Cartesian tiles are `[n,1,*S]` versus
`[1,m,*S]` and give `[n,m]`; other modes pass `[P,*S]` and give `[P]`. Broadcast
callbacks need no `vmap` rule, and an optional `backward` must return gradients
with exactly the shapes it was passed.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
cubic = tp.make_pairwise(
    lambda x, y: (x - y).abs().pow(3).sum(-1), form="broadcast")
values = cubic(X, Y)                       # [N,M]
row_sum = tp.lazy(X, Y, cubic).sum(-1)     # [N]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch</div>

```cpp
namespace tp = torchpairwise::ops;
tp::CustomComparisonOptions options;
options.broadcast = true;
auto cubic = tp::make_pairwise(
    [](const at::Tensor& x, const at::Tensor& y, at::TensorList) {
      return (x - y).abs().pow(3).sum(-1);  // [n,1,D] vs [1,m,D] -> [n,m]
    }, options);
auto values = cubic(X, Y);                              // [N,M]
auto row_sum = cubic.expression(X, Y).reduce("sum", -1); // [N]
```

</div>
</div>

An `nn.Module` can be passed instead of a function; its parameters then become
explicit operands, and `parameters=` must be omitted. A Python broadcast function
can additionally opt in to the optional KeOps backend with `keops=True`. It must
use only operations supported by both tensors and KeOps `LazyTensor`s, have
`sample_ndim=1` and no `backward`. Evaluation falls back to native tiles when
lowering fails, so the flag changes the backend rather than the result.

```python
# X:[N,3], Y:[M,3], V:[M,C]
w = torch.ones(3, requires_grad=True)
weighted_rbf = tp.make_pairwise(
    lambda x, y, w: (-((x - y) * (x - y) * w).sum(-1)).exp(),
    form="broadcast", parameters=(w,), keops=True)
with tp.backend("keops"):
    out = tp.lazy(X, Y, weighted_rbf) @ V     # [N,C]; gradients reach X, Y and w
```

Broadcast callbacks never use the fused CUDA reducers. See the
[custom metric notes](https://github.com/inspiros/torchpairwise/blob/master/docs/custom-metrics.md#broadcast-form)
for the shape contract, `nn.Module` metrics and KeOps limits.

## Stream the Same Formula

```python
A = tp.lazy(X, Y, metric, chunk_size=2)     # logical [3,5]
row_mean = A.mean(dim=-1)                  # [3]
values, indices = A.topk(2, largest=False) # [3,2] each
V = torch.randn(5, 2)
product = A.matmul(V)                     # [3,2]

B = tp.lazy(X, Y, metric="sqeuclidean")
combined = (A + 0.2 * B).sum()             # scalar
```

The custom source participates in scalar algebra and the
[shared scalar consumers](../guide/streaming.md#shared-scalar-consumers), including
statistics and selections. The custom formula need not be a mathematical metric.

Forward evaluates tiles without retaining every tile's autograd graph. Backward
replays one tile, computes its VJP and accumulates gradients. Parameters are
packed into a shared contiguous operand per consumer call; packing costs storage
proportional to parameter size, independent of N*M. Inputs, outputs, gradients
and consumer-specific workspace still occupy memory.

For dense Python calls, factory `chunk_size` bounds evaluated pairs across all
batches, default 1024. On `tp.lazy`, `chunk_size` is an independent sample tile
width per axis, default 128, matching the other streaming sources. It bounds
an outer tile of at most `chunk_size**2` pairs, not total memory.

Keep callbacks deterministic and free of side effects so replay is meaningful.
Trainable callback state must be passed through `parameters`; captured trainable
tensors/modules are unsupported for streaming. Dense calls retain ordinary
closure autograd behavior. Scalar callbacks must support `vmap`; compilation
adds compiler requirements. Unsupported operations fail explicitly.

## C++ Callables without Registration

`make_pairwise` accepts a lambda, function pointer or `std::function`. Parameter-free
callbacks need only two Tensor arguments; no dispatcher boilerplate is required:

```cpp
auto metric = tp::make_pairwise(
    [](const at::Tensor& x, const at::Tensor& y) {
      return (x - y).square().sum();
    });
auto values = metric(X, Y);                    // [N,M]
auto A = metric.expression(X, Y);              // logical [N,M]
auto mean = A.reduce("mean", -1);              // [N]
auto product = A.matmul(V);                    // [N,K], V:[M,K]
```

ATen operations keep their ordinary autograd behavior. A callback can also call
`torch::autograd::Function::apply`; its backward is used naturally. Supply an
optional explicit VJP when needed:

```cpp
tp::CustomComparisonBackward backward = [](
    const at::Tensor& g, const at::Tensor& x, const at::Tensor& y,
    at::TensorList p) -> std::vector<at::Tensor> {
  auto delta = x - y;
  auto dx = 2 * g * p[0] * delta;
  return {dx, -dx, g * delta.square()};
};
auto metric = tp::make_pairwise(
    [](const at::Tensor& x, const at::Tensor& y, at::TensorList p) {
      return ((x - y).square() * p[0]).sum();
    }, backward, options);  // options.parameters contains weight
```

Shared trainable tensors belong in `options.parameters`, rather than captures.
Other options mirror Python: `mode`, `sample_ndim`, `output_dtype`, `chunk_size`
and `symmetric`. The returned handle stores these options and Tensor operands;
rebuild it to bind different parameter tensors. Its `.expression` accepts separate
`PairwiseOperationOptions`, including the independent streaming tile width.

By default, C++ scalar callbacks run once per pair inside each bounded tile.
This is convenient but may launch many small GPU operations. Set
`options.vectorized=true` when the callable already accepts paired samples
`x[P,*Sx]`, `y[P,*Sy]` and returns `[P]`. In that mode its VJP receives `g[P]`
and returns tile-shaped input gradients and parameter gradients summed over P.
There is no automatic vectorization or compilation of arbitrary C++ callbacks.

Construct once and reuse. TorchPairwise owns internal registrations and callable
captures for the process lifetime so expressions and backward remain valid after
the handle is destroyed. Capture long-lived state by value; referenced objects
must remain alive during deferred use. Captures cannot be serialized. These are
native in-process callbacks, including for streaming; use the named-operator API
below when you need explicit registration identities in a deployment.

## Native Configuration and Saved Context

`parameters` holds shared Tensor operands, including trainable weights. Fixed
native configuration can be captured by value in both C++ callbacks; it does
not need to be converted to a Tensor:

```cpp
const int power = 3;
const bool average = false;
auto forward = [power, average](const at::Tensor& x, const at::Tensor& y,
                                at::TensorList) {
  auto terms = (x - y).abs().pow(power);
  return average ? terms.mean() : terms.sum();
};
auto backward = [power, average](const at::Tensor& g, const at::Tensor& x,
                                 const at::Tensor& y, at::TensorList)
    -> std::vector<at::Tensor> {
  auto delta = x - y;
  auto dx = g * power * delta.abs().pow(power - 1) * delta.sign();
  if (average) dx = dx / x.numel();
  return {dx, -dx};
};
auto metric = torchpairwise::ops::make_pairwise(forward, backward);
```

The same pattern supports `double`, `std::string` and typed configuration structs.
Python callables can capture ordinary configuration in the same way. Trainable
Tensors must still be explicit parameters so replay can accumulate their gradients.

For values computed during forward, use the autograd function's own context.
In C++, store native values in `ctx->saved_data`, for example
`ctx->saved_data["count"] = x.numel()`, and read them with
`ctx->saved_data["count"].toInt()`. Save Tensors with `save_for_backward`.
Python functions use `setup_context` and the usual context attributes/saved tensors.
Keep per-call state on that context: mutating shared captures to communicate
between forward and backward conflicts with tile replay and concurrent calls.

## Registered C++ and CUDA Operators

Python callbacks are registered under process-local dispatcher names. They keep
a Python numerical dependency. `compile=True` compiles their vectorized metric
tile; it does not translate arbitrary Python code into a LibTorch extension.

For explicit operator identities in exported graphs, provide registered operators
with this paired-tile protocol:

| Operation | Signature and Shapes |
|---|---|
| Forward | `(x[P,*Sx], y[P,*Sy], parameters:Tensor[]) -> values[P]` |
| VJP | `(grad[P], x, y, parameters) -> [dx,dy,*dparameters]` |

Native parameter VJPs are already summed over the P pairs and match each
parameter's original shape. Return defined gradients, including zeros for unused
parameters. Native gradients must match the corresponding operand's dtype/device.
If VJP is omitted, the registered forward must supply its own autograd behavior.

The repository's [native example](https://github.com/inspiros/torchpairwise/blob/master/examples/native_custom_metric.cpp)
registers `torchpairwise_example::weighted_pairs` and `weighted_pairs_vjp` using
ATen operations; they dispatch to CPU/CUDA without a bespoke CUDA kernel. Load
that extension in Python, or link its registrations into the LibTorch application,
before running the calls below. These example operator names are user-provided;
they are not built into TorchPairwise.

<div class="code-pair">
<div class="code-panel" data-language="python">
<div class="code-label">Python, After Loading the User Extension</div>

```python
metric = tp.make_pairwise("torchpairwise_example::weighted_pairs",
    backward="torchpairwise_example::weighted_pairs_vjp",
    parameters=(weight,), symmetric=True)
result = metric(X, Y)                     # [N,M]
mean = tp.lazy(X, Y, metric).mean(-1)      # [N]
```

</div>
<div class="code-panel" data-language="cpp">
<div class="code-label">C++ / LibTorch, After Linking the User Extension</div>

```cpp
#include <torchpairwise.h>
namespace tp = torchpairwise::ops;
auto result = tp::custom_compare(X, Y, {weight},
    "torchpairwise_example::weighted_pairs",
    "torchpairwise_example::weighted_pairs_vjp");  // [N,M]
auto A = tp::custom_expression(X, Y,
    "torchpairwise_example::weighted_pairs",
    "torchpairwise_example::weighted_pairs_vjp", {weight});
auto mean = A.reduce("mean", -1);                // [N]
```

</div>
</div>

For a native forward with its own autograd registration, omit the VJP name
(`backward=""` in C++). For example, register a forward whose implementation uses
`torch::autograd::Function::apply`, or differentiable ATen operations through
`CompositeImplicitAutograd`. The custom API then uses that registered autograd
behavior during replay, without requiring a second named VJP operator. A
`torch::autograd::Function` class alone is not a dispatcher registration.

User operator schemas, numerical implementations and autograd registrations must
remain available in the target process. Export records operator names, not the
implementation of a Python callback. For Python callbacks, an explicit factory
`name` allows recreating the registration before loading a graph; it does not
remove the Python requirement. Shape metadata must also be available to the
chosen compilation/export frontend.

Custom streaming currently requires float32/float64 scalar outputs; the callback
owns input precision and autocast behavior. First derivatives are supported by
the selected function/VJP. Higher derivatives and arbitrary torch.func transforms
are not a shared promise. Empty outputs do not probe the metric on fake samples.
