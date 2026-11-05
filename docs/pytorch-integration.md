# Mixed Precision and Compilation

Inside CPU or CUDA autocast, floating distances and kernels promote float16 and
bfloat16 operands to float32 and run with autocast disabled. Outputs are float32;
float64 inputs and outputs stay float64. This accuracy-first policy covers named
functions, string dispatch, paired and condensed distances/kernels, and is
registered natively, so standalone LibTorch callers get the same policy. Tensor
parameters such as feature weights can remain float32. Gradients return in each
original input's dtype. Boolean metrics and Hamming retain their comparison
semantics.

Use float32/float64 or autocast for portable floating comparisons across the
catalog. Outside autocast, direct float16/bfloat16 support remains specific to
the operation and its ATen backend; unsupported dtypes raise an error rather
than being silently promoted. For example, linear kernels use ATen matmul's
dtype support, while the newer vector comparisons require float32/float64.
Direct float16 Canberra is supported on CPU and CUDA: terms and forward/backward
sums use float32 arithmetic, then results and gradients are cast to float16.
Final values can still overflow or underflow float16's range. For bfloat16
Canberra, use autocast or an explicit float32 cast.

```python
with torch.autocast("cuda", dtype=torch.float16):
    features = model(samples)
    distances = torchpairwise.canberra_distances(features)
    loss = distances.mean()  # float32
scaler.scale(loss).backward()
```

Native comparison and streaming-reduction operators have Fake/Meta shape
registrations for forward and first backward. Public adapters validate inputs;
native selectors own comparison layouts, batching and mode execution. Captured
graphs retain opaque native calls, so compilation can optimize surrounding
operations without replacing the comparison algorithms. Paired linear,
polynomial, sigmoid, squared-Euclidean and RBF formulas use direct row reductions
inside the native implementation. Rational quadratic and Matérn likewise execute
shared C++/ATen formulas in every pairing mode.

```python
compiled_distance = torch.compile(
    torchpairwise.canberra_distances, fullgraph=True, dynamic=True
)
distances = compiled_distance(x, y)
distances.sum().backward()
```

Input ranks and metric choices are structural arguments. Different ranks, metric
names, feature counts or scalar options may produce separate graphs. Domain restrictions still
apply, including positive variances and positive-mass probability rows. Eager
value validation raises `ValueError` (native callers receive a synchronous
`RuntimeError`); graph execution uses tensor assertions, whose errors and timing
depend on the backend. A failed CUDA device assertion leaves the CUDA context
unusable, so supply valid data to compiled CUDA execution. Explicit generators for randomized Hausdorff comparisons,
`torch.func` transforms such as vmap are outside this integration contract.
Supported vector comparisons and streamed consumers provide second derivatives;
other operations may raise for higher derivatives. Higher-order replay retains
tile intermediates and can use O(N*M) memory.

Migrated native vector and weighted comparisons expose pairing modes through
opaque native operators. Weighted dynamic export round trips cover all-pairs,
paired and condensed output with changing sample counts and tensor weights.
The general condensed export uses `Dim("n", min=3)`; N=2 produces a singleton
output dimension that PyTorch specializes, so export that boundary separately.
Other built-in families also assemble condensed output inside native operators.
Their dynamic export tests are family-specific; structural options and singleton
output boundaries still need to be considered when choosing export constraints.

Run `examples/mixed_precision.py` for a compiled training step with autocast and
GradScaler. See `tests/test_compile.py` for backend-specific checks; AOT tracing
and Inductor execution are tested separately.

Paired row reductions use temporary storage proportional to the input size in
eager execution. They avoid an all-pairs matrix but can use more memory than a
native reduction that writes each scalar output directly. Speed depends on
feature count, device, dtype and whether surrounding operations are compiled.

Historical float32 eager forward medians for paired squared Euclidean, measured
before the shared native selector migration, in microseconds (PyTorch 2.12.1,
two CPU threads, Quadro RTX 6000):

| Device | N × D | Native one-by-one route | Row reduction | Speedup |
| --- | --- | ---: | ---: | ---: |
| CPU | 1024 × 256 | 3652 | 258 | 14.1× |
| CPU | 4096 × 128 | 7284 | 428 | 17.0× |
| CUDA | 1024 × 256 | 412 | 84 | 4.9× |
| CUDA | 4096 × 128 | 214 | 85 | 2.5× |

These are medians of three interleaved rounds against the former native
one-by-one comparison route with the same batch preparation. The GPU was shared
with other workloads, so these measurements illustrate the crossover rather
than establish portable speedup guarantees. Eager CUDA peak temporary allocation
for the two larger cases increased from 4/16 KiB to 2/4 MiB. These measurements
do not establish the latency or memory use of the current native selector.

Point-set sliced Wasserstein and Sinkhorn support compiled first coordinate
gradients with structural counts and iteration budgets. Exact multidimensional
Wasserstein uses a native CPU-controlled optimization loop, including for CUDA
inputs. Native opaque calls support compile/export and metadata execution;
CUDA synchronization remains part of runtime. All three share autocast promotion. See the
[transport guide](transport.md) for solver and gradient semantics.

KL/Jeffreys divergences and the weighted vector catalog execute shared native
ATen algorithms with native first gradients and autocast promotion. The public
weighted wrappers have full-graph compile checks and dynamic export checks. Infinite KL pairs and zero raw bins use chosen zero
gradients; weighted correlation with zero supported variance is undefined.
See [histograms](histograms.md) and [feature weights](weighted-distances.md).

Bhattacharyya/Aitchison, energy/MMD and the structured Gaussian W2 entry points
support first gradients and autocast. Gaussian SPD derivatives handle repeated
eigenvalues; singular PSD boundaries have chosen finite continuations and
higher derivatives are unsupported. See
[distribution comparisons](distribution-comparisons.md).

## Compiling and Exporting Operator Programs

Build bindings and expressions inside `forward`, and return the consumer's
tensor results. The [operator guide](operators.md) distinguishes lazy source
steps from terminal consumers. Use `A.matmul(values)` or `torch.matmul(A,
values)` rather than `@` inside `torch.compile`: Dynamo passes `A @ v` and
`v @ A` to a lazy object identically, so `@` with a tensor raises there, and
Dynamo does not trace `@` between two lazy objects (`A.matmul(B)` composes).
Eager code and `torch.export` accept `@`. Algebra and event reduction metadata are assembled
in Python; built-in numerical forward/backward remain native opaque operations.

```python
import torch
import torchpairwise as tp

class ImageNeighborhood(torch.nn.Module):
    def forward(self, images, references, values):
        errors = tp.image_map_operator(
            images, references, comparison="absolute_error", chunk_size=2)
        channels = errors.mean((-2,-1))    # lazy [...,N,M,C]
        scores = -channels.max(-1)        # lazy scalar [...,N,M]
        return (scores.softmax_matmul(values),
                errors.var((-4,-2,-1), correction=0))

images = torch.rand(4, 3, 12, 14)
references = torch.rand(7, 3, 12, 14)
values = torch.rand(7, 2)
model = ImageNeighborhood()
compiled = torch.compile(model, fullgraph=True, dynamic=True)
features, spread = compiled(images, references, values)  # [4,2], [4,3]

n = torch.export.Dim("queries", min=2)
m = torch.export.Dim("references", min=2)
exported = torch.export.export(
    model, (images, references, values),
    dynamic_shapes=({0:n}, {0:m}, {0:m}))
features, spread = exported.module()(images, references, values)
```

This export keeps C/H/W fixed while N/M change together with the payload's M.
Representative native operator tests also cover broadcast batch changes. This is
not a promise that every event dimension, singleton case or comparison family
has the same dynamic constraints. Metric names, stage lists, dimensions and
other structural options specialize the program. A terminal N/M reduction
returns a tensor; later tensor work is ordinary graph computation rather than
another event-program stage.

The exported graph needs the compatible torchpairwise native library loaded at
execution. Exporting does not provide an ONNX implementation or make arbitrary
Python callbacks portable to LibTorch. Standalone callers use typed expression
factories and event facades; see [native entry points](native-api.md#streaming-operator-entry-points).

## Python Compilation, TorchScript and LibTorch

TorchScript is [deprecated by PyTorch](https://docs.pytorch.org/tutorials/advanced/torch_script_custom_ops.html);
new deployment work prioritizes `torch.compile` and `torch.export`. TorchScript
compatibility is optional. Legacy scripted models and
standalone LibTorch remain useful compatibility targets.

These are separate capabilities. Python type hints and `.pyi` stubs describe
types for callers and tools; they do not register a C++ operator or establish
compilation support. `torch.compile` captures Python execution and does not
require type annotations or a `@torch.jit.script` decorator. Its support depends
on the operations, control flow, autograd implementation and selected backend.
Use `fullgraph=True` when checking that execution has no graph breaks.

| Interface | Current scope |
| --- | --- |
| Python eager | The documented public functions, subject to each function's domain and device contract. |
| `torch.compile` | Built-in families use native operators with forward/first-backward tests. Fullgraph checks cover vector/cloud, structured, modality, sequence/token and streaming APIs. Shuffled Hausdorff is excluded. |
| `torch.export` | Native family and streaming operators provide metadata kernels and representative dynamic export/save/load tests, including tuple collections and tensor parameters. Runtime still requires the native extension. |
| `torch.jit.script` | The 23 migrated named functions have direct typed native-op branches. Other public wrappers retain their previous limitations; there is no package-wide guarantee. |
| Native C++ / LibTorch | All built-in metrics, kernels, modes and reductions execute forward/first backward through native algorithms. Typed family APIs and unified collection/single-sample selectors are exposed by the umbrella header; see the [native guide](native-api.md). |

Before the native-vector migration, the public Python Canberra, KL and RBF paths passed a CPU
`torch.compile(..., backend="aot_eager", fullgraph=True, dynamic=True)` values
probe on PyTorch 2.12, while their public wrappers failed scripting. Migrated KL
now calls its native schema when scripted. For a legacy function such as
Canberra, a typed
function calling an existing native schema can instead be scripted:

```python
import torch
import torchpairwise  # loads the extension and registers its operators

def native_canberra(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ops.torchpairwise.canberra_distances(x, y)

scripted = torch.jit.script(native_canberra)
scripted.save("canberra.pt")
```

Define this function in a Python source file so TorchScript can read its source.
The consumer must load/link the extension before executing the saved graph.
This uses the native schema's input and parameter contract, which can be
narrower than the public Python function's. Scripting this example does not
establish scripting support for the Python dispatcher or for other metrics.

The [streaming reductions](reductions.md) and exact transport execute through
native opaque operators. Exact solvers remain CPU-controlled and synchronize
CUDA costs even when captured. User-callable and encoder factories contain
Python orchestration; optional compilation of a comparison tile does not make
an arbitrary callback portable to standalone C++. Built-in public APIs use
native first backward; standalone LibTorch examples verify execution without
Python initialization. Histogram-intersection and NaN-distance compatibility primitives also use native
autograd. Python retains only metadata (Fake) registration; the autocast policy
is registered natively, so it also applies in standalone LibTorch. Use the
documented native APIs and each function's dtype contract for standalone deployment.

See the [native API guide](native-api.md) for the migrated families and typed
parameter design.
