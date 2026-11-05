#include "sqjensenshannon.h"

#include <torch/types.h>

namespace torchpairwise::ops {
at::Tensor _sqjensenshannon(const at::Tensor& x1, const at::Tensor& x2,
                            c10::optional<double> base) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_sqjensenshannon", "")
                       .typed<decltype(_sqjensenshannon)>();
  return op.call(x1, x2, base);
}

namespace detail {
std::tuple<at::Tensor, at::Tensor> _sqjensenshannon_backward(
    const at::Tensor& grad, const at::Tensor& x1, const at::Tensor& x2,
    c10::optional<double> base) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_sqjensenshannon_backward", "")
          .typed<decltype(_sqjensenshannon_backward)>();
  return op.call(grad, x1, x2, base);
}
}  // namespace detail

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("torchpairwise::_sqjensenshannon(Tensor x1, "
                             "Tensor x2, float? base=None) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchpairwise::_sqjensenshannon_backward(Tensor grad, Tensor x1, Tensor "
      "x2, float? base) -> (Tensor grad_x1, Tensor grad_x2)"));
  // Preserve the legacy private name as a traceable dispatcher alias.
  m.def(TORCH_SELECTIVE_SCHEMA(
            "torchpairwise::__sqjensenshannon_backward(Tensor grad, Tensor x1, "
            "Tensor x2, float? base) -> (Tensor grad_x1, Tensor grad_x2)"),
        TORCH_FN(detail::_sqjensenshannon_backward));
}
}  // namespace torchpairwise::ops
