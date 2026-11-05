#include "hausdorff.h"

#include <torch/types.h>

namespace torchpairwise::ops {
std::tuple<at::Tensor, at::Tensor, at::Tensor> _directed_hausdorff(
    const at::Tensor& x1, const at::Tensor& x2, bool shuffle,
    c10::optional<at::Generator> generator) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_directed_hausdorff", "")
          .typed<decltype(_directed_hausdorff)>();
  return op.call(x1, x2, shuffle, generator);
}

namespace detail {
std::tuple<at::Tensor, at::Tensor> _directed_hausdorff_backward(
    const at::Tensor& grad, const at::Tensor& x1, const at::Tensor& x2,
    bool shuffle, c10::optional<at::Generator> generator) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_directed_hausdorff_backward", "")
          .typed<decltype(_directed_hausdorff_backward)>();
  return op.call(grad, x1, x2, shuffle, generator);
}
}  // namespace detail

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchpairwise::_directed_hausdorff(Tensor x1, Tensor x2, *, bool "
      "shuffle=False, Generator? generator=None) -> (Tensor output, Tensor "
      "x1_indices, Tensor x2_indices)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchpairwise::_directed_hausdorff_backward(Tensor grad, Tensor x1, "
      "Tensor x2, *, bool shuffle=False, Generator? generator=None) -> (Tensor "
      "grad_x1, Tensor grad_x2)"));
  // Preserve the legacy private name as a traceable dispatcher alias.
  m.def(TORCH_SELECTIVE_SCHEMA(
            "torchpairwise::__directed_hausdorff_backward(Tensor grad, Tensor "
            "x1, Tensor x2, *, bool shuffle=False, Generator? generator=None) "
            "-> (Tensor grad_x1, Tensor grad_x2)"),
        TORCH_FN(detail::_directed_hausdorff_backward));
}
}  // namespace torchpairwise::ops
