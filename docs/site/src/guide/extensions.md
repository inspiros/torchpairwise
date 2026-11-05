<div class="eyebrow">Integration</div>

# Extending from C++

A Python package that builds on torchpairwise needs only `import torchpairwise`.
A package with its own C++ extension has two routes from the installed wheel,
plus the [CMake package](install.md) for applications without Python.

| Route | Build needs | Tied to |
|---|---|---|
| Dispatcher calls by schema (recommended) | PyTorch headers only | operator names and schemas |
| Native headers | `tp.get_include()`, `tp.library_path()`, C++20 | this exact torchpairwise and PyTorch build |

## Dispatcher Calls

Every public operator is registered as `torchpairwise::<name>`. An extension
looks it up by schema and calls it like any PyTorch operator, so autograd, CUDA
and backend overrides behave as in Python. Nothing links against
torchpairwise, so the extension can ship as a binary wheel; only
`import torchpairwise` is needed before the first call.

<div class="code-pair">
<div class="code-panel" data-language="cpp">
<div class="code-label">mypkg/csrc/ops.cpp</div>

```cpp
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

at::Tensor rbf_predict(const at::Tensor& x, const at::Tensor& centers,
                       const at::Tensor& coefficients, double gamma) {
  // Looked up on first use; the extension may load before torchpairwise.
  static const auto rbf =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::rbf_kernel", "")
          .typed<at::Tensor(const at::Tensor&, const c10::optional<at::Tensor>&,
                            c10::optional<double>)>();
  return rbf.call(x, centers, gamma).matmul(coefficients);
}

TORCH_LIBRARY(mypkg, m) {
  m.def("rbf_predict(Tensor x, Tensor centers, Tensor coefficients, "
        "float gamma) -> Tensor", &rbf_predict);
}
```

</div>
<div class="code-panel" data-language="python">
<div class="code-label">Python</div>

```python
import torch
import torchpairwise  # registers torch.ops.torchpairwise
import mypkg          # loads the extension above

# Prints the schema the C++ signature must match.
print(torch.ops.torchpairwise.rbf_kernel.default._schema)
y = torch.ops.mypkg.rbf_predict(x, centers, coefficients, 0.5)
y.sum().backward()  # gradients flow through torchpairwise
```

</div>
</div>

Public names and schemas are the supported contract. Operators whose names start
with an underscore are private. A change to a public schema is a breaking change
and is listed in the migration notes.

## Native Headers

The C++ API adds what a single operator call cannot express, such as streamed
sources, consumers and solvers. Include `<torchpairwise.h>` and link the
installed library:

```python
import torchpairwise as tp
from torch.utils.cpp_extension import CppExtension

extension = CppExtension(
    'mypkg._C', ['mypkg/csrc/ops.cpp'], include_dirs=[tp.get_include()],
    extra_compile_args=['-std=c++20'], extra_link_args=[tp.library_path()])
```

```cpp
#include <torchpairwise.h>
namespace s = torchpairwise::streaming;

s::PairwiseOptions options;
options.comparison.gamma = gamma;
options.chunk_size = 1024;  // the [N,M] kernel matrix is never formed
auto y = s::kernel(x, centers, "rbf", options).matmul(coefficients);
```

This route binds the extension to one installation: build it with
`pip install --no-build-isolation` against the same PyTorch, rebuild it after
upgrading torchpairwise, and do not ship it as a binary wheel. It is checked on
Linux. Windows wheels ship no import library, so use dispatcher calls there.

The [downstream example](https://github.com/inspiros/torchpairwise/tree/master/examples/downstream)
builds one extension of each kind and checks values and gradients against
Python. Details are in the
[native API notes](https://github.com/inspiros/torchpairwise/blob/master/docs/native-api.md#building-on-torchpairwise-from-c).
