# Experimental Custom Pairwise Functions

This prototype adapts a PyTorch function that receives two samples plus optional explicit parameters and
returns one scalar tensor by default. The scalar form uses `torch.vmap` to evaluate
pairs; the opt-in broadcast form evaluates whole tiles with ordinary broadcasting.
Both optionally use `torch.compile` to compile the tile computation. It is an experimental API; its signature may change.

```python
import torch
from torchpairwise import make_pairwise

def squared_distance(a, b):
    return (a - b).square().sum()

compare = make_pairwise(squared_distance, compile=True)
x = torch.randn(20, 8)
y = torch.randn(30, 8)
distances = compare(x, y)    # [20, 30]
self_distances = compare(x)  # [20, 20]
```

Create the adapter once and reuse it. Python callbacks receive process-local
paired-tile dispatcher registrations. They need no extension build or new
dependency. They do not become built-in catalog entries.
Importing the current package still requires its normal native extension.
Existing catalog functions can have more specialized implementations; this
adapter makes no performance or numerical-equivalence promise against them.

## Shapes and Modes

The factory is:

```python
def make_pairwise(fn, *, backward=None, parameters=(), mode="all_pairs",
                  sample_ndim=1, chunk_size=None, compile=False, symmetric=False,
                  output_dtype=None, name=None, form="scalar"): ...
```

Each input has shape `[*batch, N, *sample]`. `sample_ndim` is a positive integer
giving the number of axes in `sample`, and defaults to one for vectors. Leading
batch axes broadcast using ordinary PyTorch rules. Both inputs must have the
same dtype and device. Their sample shapes can differ: the user function owns
any compatibility checks. Dense strided tensors, including
noncontiguous tensors and expanded views, are supported. Zero-size sample axes
are allowed if the user's function supports those samples.

| Mode | Call | Output |
| --- | --- | --- |
| `all_pairs` | `compare(x, y)` or `compare(x)` | `[*batch, N, M]` |
| `paired` | `compare(x, y)` with equal sample counts | `[*batch, N]` |
| `condensed` | `compare(x)` | `[*batch, N*(N-1)//2]` |
| `single` | `compare(a, b, mode="single")`, inputs omit N/M axes | `[*batch]` |

Single mode consumes one sample per leading batch and requires both inputs.
A factory mode is the default; every invocation can override it. For example,
`compare(a, b, mode="single")` on `[B,D]` and `[D]` returns `[B]`.

Paired mode evaluates only corresponding samples. Condensed mode evaluates
only `(0,1), (0,2), ..., (0,N-1), (1,2), ...`, in row-major upper-triangle order,
without the diagonal. It requires `symmetric=True` when creating the adapter.
This flag is the caller's assertion that `fn(a,b) == fn(b,a)`; the adapter does
not infer or test symmetry. `all_pairs` and `paired` also accept asymmetric
functions and do not use the symmetry flag to skip work.

```python
paired = make_pairwise(squared_distance, mode="paired")
condensed = make_pairwise(squared_distance, mode="condensed", symmetric=True)
paired_values = paired(x, x)     # [20]
unordered_values = condensed(x)  # [190]
```

An image function receives an intact `[C,H,W]` sample with `sample_ndim=3`:

```python
def image_mean_distance(a, b):
    return (a.mean((-1, -2)) - b.mean((-1, -2))).square().sum()

compare_images = make_pairwise(image_mean_distance, sample_ndim=3, compile=True)
images = torch.randn(2, 10, 3, 16, 16)  # [batch, N, C, H, W]
references = torch.randn(1, 4, 3, 16, 16)
image_values = compare_images(images, references)  # [2, 10, 4]
```

Likewise, `sample_ndim=2` can represent a `[points, channels]` sample. The first
collection may contain P points per sample and the second Q points per sample,
if the function supports those shapes. Samples within each collection have a
fixed shape; the adapter has no built-in ragged representation. The adapter
neither flattens samples nor introduces domain-specific
normalization, masking, resampling, or validation.

## Broadcast Form

Use `form="broadcast"` when the function can evaluate a whole tile with ordinary
PyTorch broadcasting. This avoids `vmap` and the Cartesian copies of samples
required by the scalar form:

```python
compare = make_pairwise(
    lambda x, y: (x - y).abs().pow(3).sum(-1), form="broadcast")
values = compare(x, y)
row_sum = tp.lazy(x, y, compare).sum(-1)
```

The function receives `x:[*Lx,*S]` and `y:[*Ly,*S]`, where `S` has
`sample_ndim` axes and the leading shapes broadcast. It reduces the sample
axes itself and returns exactly `broadcast(Lx,Ly)`. For `sample_ndim=2`, for
example, reduce with `.sum((-2,-1))`. Cartesian tiles pass `[n,1,*S]` and
`[1,m,*S]` and require `[n,m]`; paired, condensed and single tiles pass
`[P,*S]` operands and require `[P]`. Leading batch dimensions follow the same
broadcasting rule. The result must be dense strided, on the input device, with
`output_dtype` (the input dtype by default); the adapter validates this contract.
Empty outputs never invoke the callback.

Explicit `parameters` remain shared tensors and receive accumulated gradients.
An optional `backward(g,x,y,*parameters)` receives exactly the shapes passed to
the forward function, with `g` shaped like its output. Return one gradient per
operand, including each parameter, with exactly that operand's passed shape.
The callback must reduce broadcast axes itself. For a Cartesian Manhattan tile,
for example:

```python
def manhattan(x, y):
    return (x - y).abs().sum(-1)

def manhattan_backward(g, x, y):
    dx = g.unsqueeze(-1) * (x - y).sign()
    return dx.sum_to_size(x.shape), (-dx).sum_to_size(y.shape)

compare = make_pairwise(manhattan, backward=manhattan_backward,
                        form="broadcast")
```

`compile=True` compiles the broadcast tile function and, when supplied, its VJP.
The scalar form remains the default. Broadcast callbacks need no `vmap` rule;
they must still obey the compiler's requirements when compilation is requested.
Dense calls retain their tile graphs; streaming consumers retain no per-tile
graph and replay tiles for the first backward. Broadcast callbacks never run
inside fused CUDA reducers.

### KeOps opt-in

`keops=True` lets `tp.backend("keops")` (and `"auto"`, under its usual size
gate) call the same function on pykeops `LazyTensor`s. Rows are `Vi` and
columns `Vj`; each parameter becomes a `Pm` vector, so parameters must be
zero- or one-dimensional. The function must therefore use only operations that
both tensors and `LazyTensor`s support, and reduce features with `.sum(-1)`:

```python
w = torch.ones(3, requires_grad=True)
weighted_rbf = tp.make_pairwise(
    lambda x, y, w: (-((x - y) * (x - y) * w).sum(-1)).exp(),
    form="broadcast", parameters=(w,), keops=True)
with tp.backend("keops"):
    out = tp.lazy(X, Y, weighted_rbf) @ V  # one KeOps reduction; grads to X, Y, w
```

The opt-in requires `sample_ndim=1` and no `backward=`, because KeOps
differentiates the formula itself. Unbatched float32/float64 inputs are lowered.
If the function raises on `LazyTensor`s, or the graph is outside the KeOps
subset, evaluation falls back to native tiles, so the opt-in never changes
results, only the backend.

## Scalar, Dtype, and Autocast Contract

The result of `fn(a,b)` must be a dense strided, zero-dimensional `Tensor` on
the input device. A Python number, tuple, or tensor of shape `[1]` is rejected.
Inputs support `float16`, `bfloat16`, `float32`, and `float64`. Integer, boolean,
complex, sparse, and quantized inputs are outside this prototype's contract.
The function's constituent operations must support the chosen dtype/device.

By default, each scalar result must have the input dtype. An explicit
`output_dtype` declares a different result dtype from the same four supported
floating types. The adapter validates this declaration; it does not cast the
function's result. For example, float32 accumulation from low-precision inputs
can be written explicitly:

```python
def accumulated_distance(a, b):
    return (a.float() - b.float()).square().sum()

compare_float32 = make_pairwise(accumulated_distance, output_dtype=torch.float32)
```

The adapter preserves the caller's autocast context and makes no automatic
precision promotion. If autocast changes a function's result dtype, declare
that dtype with `output_dtype`, explicitly cast the scalar in the function, or
disable autocast around the call. The same factory's dtype declaration applies
to every invocation. Accumulation precision, overflow, NaNs, infinities,
singularities, and domain checks remain the function author's responsibility.

## Gradients and Empty Collections

Ordinary PyTorch autograd differentiates through the inputs, broadcasting,
indexing, and captured tensors. Nothing is implicitly detached. A captured
parameter can be shared by all samples:

```python
weight = torch.nn.Parameter(torch.ones(8))
weighted = make_pairwise(lambda a, b: ((a - b).square() * weight).sum(),
                         compile=True)
loss = weighted(x, y).sum()
loss.backward()  # accumulates weight.grad
```

First-gradient support depends on the operations in the user function and, when
enabled, the compiler backend. Higher-order differentiation of compiled calls
is not promised. Captured tensors are shared across all pairs; the adapter does
not interpret captured batch dimensions or offer additional mapped parameters.

Empty output shapes are supported, including empty batch axes and condensed
inputs with fewer than two samples. The adapter does **not** invoke the function
on fabricated samples to infer metadata. It returns an empty tensor with the
declared dtype and input device, connected to the input tensors for zero input
gradients. Captured tensors are unused on this path, so `autograd.grad` requires
`allow_unused=True` for them and returns `None`. If only a captured parameter
requires gradients, an empty result has no gradient graph. No validation of the
user function's return contract occurs until an actual pair is evaluated.

## Chunking and Compilation Limits

`chunk_size` is the maximum number of pairs evaluated in one tile **across all
batches**, and must be a positive integer. `None` selects 1024 pairs. The
implementation iterates over tiles in Python and evaluates scalar-form pairs
with `vmap`, or calls the broadcast function on whole tiles. It has no Python
call loop over individual pairs and no such fallback.
Paired mode avoids a dense cross product. Condensed mode uses an O(N) integer
row-offset table and bounded gathers, without a dense square matrix or a
full pair-by-feature temporary. Noncontiguous and broadcast input collections
are indexed directly without first copying a flattened broadcast collection.

Chunking bounds the pair count in gathered samples and forward intermediates
inside each evaluation; a function can still have large per-sample temporary
requirements. The complete output is retained, and concatenation allocates the
final output alongside its tiles. Autograd may retain saved features and graphs
for **every** tile until backward finishes. This is not a bounded-memory
training or recomputation implementation.

`compile=True` compiles the vectorized metric tile with fullgraph=True.
An explicit backward also receives its own compiled vectorized VJP tile. Compilation occurs lazily on
first evaluation. Input validation, index generation, and the outer tile loop
remain ordinary Python/Torch execution. Compiler failures propagate; there is
no silent eager fallback for an unsupported full graph. Whole-adapter export
or whole-adapter fullgraph capture is not part of this prototype's contract.

PyTorch may specialize and recompile for different sample shapes, tile sizes
(including the final partial tile), batch ranks, strides, dtypes, devices,
autocast state, or captured-object guards. Creating a fresh factory for every
call defeats reuse. Compilation can cost more than it saves for small or
one-off workloads. The adapter does not compile each pair separately, and it
does not use TorchScript.

The scalar-form user function must use operations supported by `vmap`, and additionally by
the compiler when requested. Keep it pure: no mutation of inputs or captured
state, Python side effects, `.item()`, NumPy conversion, data-dependent Python
branches, or data-dependent output shapes. Use tensor operations such as
`torch.where` where appropriate. Random operations are rejected by the
`randomness="error"` policy. Custom operators need batching rules; unsupported
operations report PyTorch's error instead of falling back to per-pair Python.

## Custom Autograd Functions and Native Deployment

`make_pairwise(MyFunction.apply)` uses the user function's autograd behavior;
the adapter does not replace its backward formula. Because evaluation uses
`torch.vmap`, an ordinary custom autograd function is not automatically
compatible. Use separate `forward` and `setup_context` methods and provide
`vmap`, or set `generate_vmap_rule=True` when the function's implementation
supports the generated rule. Compilation imposes its additional restrictions.
See [PyTorch's transform requirements](https://docs.pytorch.org/docs/stable/notes/extending.func.html).
Alternatively pass a VJP directly and expose trainable state as explicit Tensor
parameters. The VJP receives the upstream scalar gradient, both samples and the
parameters. Its return order matches the inputs:

```python
weight = torch.nn.Parameter(torch.ones(8))
def forward(a, b, w):
    return ((a - b).square() * w).sum()

def backward(g, a, b, w):
    delta = a - b
    return 2*g*w*delta, -2*g*w*delta, g*delta.square()

custom = make_pairwise(forward, backward=backward, parameters=(weight,),
                       symmetric=True, compile=True)
```

Parameters are shared across pairs and must have the input dtype/device. They
are not interpreted as batched parameters. Their gradients accumulate through
pair reuse and broadcasting. Return a Tensor for every gradient, using zeros for unused operands.
Ordinary forwards can omit backward and use PyTorch autograd instead. Forward and VJP callbacks must both support vectorization and the selected
compiler. The factory manages their autograd context; no user setup_context
method is needed when supplying two ordinary functions.

```python
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
        grad_x = 2 * grad_output * (x - y)
        return grad_x, -grad_x

custom = make_pairwise(SquaredDistance.apply, compile=True)
scores = custom(x, y)  # x:[N,D], y:[M,D] -> [N,M]
```

## nn.Module Metrics

Pass an `nn.Module` directly to `make_pairwise` to expose its parameters as
explicit operands for dense calls and streamed backward:

```python
class WeightedDistance(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(features))
        self.register_buffer("scale", torch.tensor(0.5))

    def forward(self, x, y):
        return ((x - y).square() * self.weight).sum(-1) * self.scale

metric_module = WeightedDistance(8)
custom = make_pairwise(metric_module, form="broadcast")
loss = tp.lazy(x, y, custom).sum()
loss.backward()  # accumulates metric_module.weight.grad
```

The factory takes `tuple(module.parameters())` automatically; combining a module
with nonempty explicit `parameters=` raises `TypeError`. Evaluation uses
`torch.func.functional_call` with those explicit parameters in scalar and
broadcast forms, including `compile=True`. Buffers remain module-owned constants.
The factory never changes `module.training`; callbacks must remain deterministic
and free of state mutation for tile replay. An explicit `backward=` receives the
module's parameters after the two samples, just like other parameterized metrics.
Trainable state captured elsewhere, including by a backward closure, still must
be supplied explicitly for streaming.

An `nn.Module` passed directly as `head=` to `make_feature_pairwise` is instead
an eager collection head. Wrap a pairwise module with `make_pairwise` first to
stream it. Module encoders run eagerly before head streaming; encoder gradients
flow through the encoded features without changing the encoder's mode.

## Native Streaming Sources

`tp.lazy(X, Y, custom, chunk_size=128)` binds the logical all-pairs scalar matrix.
It supports the same scalar algebra and 25 consumers as built-in sources. For
example:

```python
A = tp.lazy(x, y, custom, chunk_size=16)
row_mean = A.mean(dim=-1)  # [...,N]
V = torch.randn(30, 4, device=x.device, dtype=x.dtype)
product = A.matmul(V)                                       # [...,N,4]
nearest_values, nearest_indices = A.topk(3, largest=False)  # [...,N,3]
```

Forward retains no per-tile graphs. Backward replays one tile, obtains its VJP
and accumulates sample/parameter adjoints before releasing the tile. Inputs,
outputs, gradients and consumer-specific workspace still occupy memory.
Parameters are packed into a shared contiguous operand per consumer call;
packing uses space proportional to parameter size, not to N*M.

Dense factory chunk_size counts pairs across batches. Operator chunk_size is
independent: it counts samples per outer tile axis, default 128, so a Cartesian
tile contains at most chunk_size squared pairs. Neither is a total memory cap.
Python dense outputs and saved tile graphs still have the dense-call memory
contract described above. Streaming currently requires float32/float64 outputs;
the callback controls its own input precision and autocast behavior.

Streaming callbacks must be deterministic and free of side effects. Trainable
state must be explicit parameters. Hidden trainable closures/modules are
unsupported; ordinary dense calls keep their existing closure-autograd behavior.
An explicit parameter used by the scalar function receives local per-pair VJPs,
which vmap/autograd accumulates; no full Jacobian is constructed.

Construct custom factories outside a compiled module's forward, then construct
bindings inside forward. Scalar streaming consumers can be compiled/exported
through their native dispatcher boundary. Exported graphs retain callback
operator names, not Python callback code. Recreate Python registrations before
loading a graph in another process; name= supplies an explicit process-level
identifier. Keep model parameters as registered module parameters and pass them
explicitly to the factory. Export does not make Python callbacks Python-free.

## C++ Callables

C++ users can construct the same four-mode adapter without writing operator
registrations. Lambdas, function pointers and `std::function` are accepted:

```cpp
#include <torchpairwise.h>
namespace tp = torchpairwise::ops;
auto metric = tp::make_pairwise(
    [](const at::Tensor& x, const at::Tensor& y) {
      return (x - y).square().sum();
    });
auto values = metric(X, Y);                  // [N,M]
auto single = metric(X[0], Y[0], "single");  // scalar
auto A = metric.expression(X, Y);
auto mean = A.reduce("mean", -1);  // [N]
```

Normal ATen autograd and callbacks using `torch::autograd::Function::apply` work
without a separate VJP. For explicit parameters use a three-argument forward
`(x,y,at::TensorList)` and `CustomComparisonOptions.parameters`. The optional
`CustomComparisonBackward` has signature `(g,x,y,at::TensorList) ->
std::vector<at::Tensor>`; scalar gradients match all operands, including shared
parameters. Pass it as `make_pairwise(forward, backward, options)`. Options also
hold mode, sample_ndim, output_dtype, dense chunk_size and symmetric.
`.expression` takes independent PairwiseOperationOptions for streaming tiles.

Scalar callbacks are invoked per pair in C++ loops. For efficient GPU tiles set
options.vectorized=true: callbacks receive `[P,*sample]` and return `[P]`, with
shared parameter VJPs already summed over P. Callables are not automatically
vectorized or compiled. Set `options.broadcast=true` instead for broadcast
callbacks: Cartesian operands are `[n,1,*sample]` and `[1,m,*sample]`, with
`[n,m]` results; other tiles use `[P,*sample]` and return `[P]`. This option does
not require `vectorized=true`. Its VJP returns gradients with the exact passed
operand shapes, reducing broadcast axes and summing shared parameter gradients.
`broadcast` defaults to false. Inputs/parameters retain the shared custom-source contract.

Construct a factory once and reuse it. Internal registrations and captures live
for the process, ensuring deferred expressions/backward survive handle destruction.
Captured references still require caller-managed lifetime. Trainable state must
be explicit parameters. In-process native callbacks do not have portable export
identity or serialized captures; retain the named registration route for explicit
cross-process deployment. No Python callback is used by the callable C++ path.

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

## Registered Native Operators

Pass qualified paired-tile operator names instead of Python functions for a
native route. Forward has schema `(Tensor x, Tensor y, Tensor[] parameters) ->
Tensor` over P corresponding samples; it returns `[P]`. The optional VJP has
schema `(Tensor grad, Tensor x, Tensor y, Tensor[] parameters) -> Tensor[]`.
It returns x/y gradients plus one defined, already pair-summed gradient per
shared parameter. Shapes, dtype and device must match each original operand.
Use zeros for unused parameters. Without a VJP name, forward must implement its
own autograd behavior.

```python
native = make_pairwise("torchpairwise_example::weighted_pairs",
    backward="torchpairwise_example::weighted_pairs_vjp", parameters=(weight,),
    symmetric=True)
values = native(x,y)
result = tp.lazy(x, y, native).mean(-1)
```

The same registered operators can be bound in LibTorch:

```cpp
#include <torchpairwise.h>
namespace tp = torchpairwise::ops;
auto values = tp::custom_compare(X, Y, {weight},
    "torchpairwise_example::weighted_pairs",
    "torchpairwise_example::weighted_pairs_vjp");
auto A = tp::custom_expression(X, Y,
    "torchpairwise_example::weighted_pairs",
    "torchpairwise_example::weighted_pairs_vjp", {weight});
auto row_mean = A.reduce("mean", -1);
```

Schemas and implementations must remain available in the target process.
Register native numerical/Autograd and compilation metadata appropriate to the
chosen frontend; the wrapper cannot provide user code's implementation.
Python callbacks use the same engine but retain their Python dependency.
See [the runnable native example](../examples/native_custom_metric.py) and its
[C++ implementation](../examples/native_custom_metric.cpp).

Registered native broadcast operators use the same signatures, with the broadcast
shape contract above. Pass `form="broadcast"` to the Python factory. In C++,
`custom_compare` appends `bool broadcast=false` after `symmetric`, and
`custom_expression` appends `bool broadcast=false` after its
`PairwiseOperationOptions` argument. Existing calls keep the paired-tile
protocol; pass `true` in the final argument for broadcast operators. Native
broadcast VJPs likewise return gradients matching the passed operand shapes.
