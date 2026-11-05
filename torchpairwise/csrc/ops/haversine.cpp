#include "haversine.h"

#include <torch/types.h>

namespace torchpairwise::ops {
at::Tensor _haversine(const at::Tensor& x1, const at::Tensor& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_haversine", "")
                       .typed<decltype(_haversine)>();
  return op.call(x1, x2);
}

namespace detail {
std::tuple<at::Tensor, at::Tensor> _haversine_backward(const at::Tensor& grad,
                                                       const at::Tensor& x1,
                                                       const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_haversine_backward", "")
          .typed<decltype(_haversine_backward)>();
  return op.call(grad, x1, x2);
}
}  // namespace detail

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchpairwise::_haversine(Tensor x1, Tensor x2) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchpairwise::_haversine_backward(Tensor grad, Tensor x1, Tensor x2) "
      "-> (Tensor grad_x1, Tensor grad_x2)"));
  // Preserve the legacy private name as a traceable dispatcher alias.
  m.def(TORCH_SELECTIVE_SCHEMA(
            "torchpairwise::__haversine_backward(Tensor grad, Tensor x1, "
            "Tensor x2) -> (Tensor grad_x1, Tensor grad_x2)"),
        TORCH_FN(detail::_haversine_backward));
}
}  // namespace torchpairwise::ops
