#include "wminkowski.h"

#include <torch/types.h>

namespace torchpairwise::ops {
at::Tensor _wminkowski(const at::Tensor& x1, const at::Tensor& x2,
                       const at::Tensor& w, double p) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_wminkowski", "")
                       .typed<decltype(_wminkowski)>();
  return op.call(x1, x2, w, p);
}

namespace detail {
std::tuple<at::Tensor, at::Tensor, at::Tensor> _wminkowski_backward(
    const at::Tensor& grad, const at::Tensor& x1, const at::Tensor& x2,
    const at::Tensor& w, double p) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_wminkowski_backward", "")
          .typed<decltype(_wminkowski_backward)>();
  return op.call(grad, x1, x2, w, p);
}
}  // namespace detail

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("torchpairwise::_wminkowski(Tensor x1, Tensor x2, "
                             "Tensor w, float p=2) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchpairwise::_wminkowski_backward(Tensor grad, Tensor x1, Tensor x2, "
      "Tensor w, float p) -> (Tensor grad_x1, Tensor grad_x2, Tensor grad_w)"));
  // Preserve the legacy private name as a traceable dispatcher alias.
  m.def(TORCH_SELECTIVE_SCHEMA(
            "torchpairwise::__wminkowski_backward(Tensor grad, Tensor x1, "
            "Tensor x2, Tensor w, float p) -> (Tensor grad_x1, Tensor grad_x2, "
            "Tensor grad_w)"),
        TORCH_FN(detail::_wminkowski_backward));
}
}  // namespace torchpairwise::ops
