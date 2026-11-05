// A downstream operator that calls torchpairwise through its registered
// schema. Only PyTorch headers are needed and nothing links against
// torchpairwise; load it before the first call (`import torchpairwise` in
// Python, torch::ops::load_library or dlopen in C++).
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace {
// Schema: torchpairwise::rbf_kernel(Tensor x1, Tensor? x2=None,
//                                   float? gamma=None) -> Tensor
at::Tensor rbf_kernel(const at::Tensor& x, const at::Tensor& y, double gamma) {
  // Looked up on first use, so this library may load before torchpairwise.
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::rbf_kernel", "")
          .typed<at::Tensor(const at::Tensor&, const c10::optional<at::Tensor>&,
                            c10::optional<double>)>();
  return op.call(x, y, gamma);
}

// Kernel model prediction K(x, centers) @ coefficients. The call goes through
// the dispatcher, so autograd, CUDA and backend overrides apply as in Python.
at::Tensor rbf_predict(const at::Tensor& x, const at::Tensor& centers,
                       const at::Tensor& coefficients, double gamma) {
  return rbf_kernel(x, centers, gamma).matmul(coefficients);
}
}  // namespace

TORCH_LIBRARY(dispatcher_consumer, m) {
  m.def(
      "rbf_predict(Tensor x, Tensor centers, Tensor coefficients, float gamma)"
      " -> Tensor",
      &rbf_predict);
}
