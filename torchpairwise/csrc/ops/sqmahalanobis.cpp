#include "sqmahalanobis.h"

#include <torch/types.h>

namespace torchpairwise::ops {
at::Tensor _sqmahalanobis(const at::Tensor& x1, const at::Tensor& x2,
                          const at::Tensor& VI) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_sqmahalanobis", "")
                       .typed<decltype(_sqmahalanobis)>();
  return op.call(x1, x2, VI);
}

namespace detail {
std::tuple<at::Tensor, at::Tensor, at::Tensor> _sqmahalanobis_backward(
    const at::Tensor& grad, const at::Tensor& x1, const at::Tensor& x2,
    const at::Tensor& VI) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_sqmahalanobis_backward", "")
          .typed<decltype(_sqmahalanobis_backward)>();
  return op.call(grad, x1, x2, VI);
}
}  // namespace detail

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("torchpairwise::_sqmahalanobis(Tensor x1, Tensor "
                             "x2, Tensor VI) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchpairwise::_sqmahalanobis_backward(Tensor grad, Tensor x1, Tensor "
      "x2, Tensor VI) -> (Tensor grad_x1, Tensor grad_x2, Tensor grad_VI)"));
  // Preserve the legacy private name as a traceable dispatcher alias.
  m.def(TORCH_SELECTIVE_SCHEMA(
            "torchpairwise::__sqmahalanobis_backward(Tensor grad, Tensor x1, "
            "Tensor x2, Tensor VI) -> (Tensor grad_x1, Tensor grad_x2, Tensor "
            "grad_VI)"),
        TORCH_FN(detail::_sqmahalanobis_backward));
}
}  // namespace torchpairwise::ops
