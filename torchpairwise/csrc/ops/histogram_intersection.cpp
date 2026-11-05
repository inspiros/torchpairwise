#include "histogram_intersection.h"

#include <torch/library.h>
namespace torchpairwise::ops {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("_histogram_intersection(Tensor x, Tensor y) -> Tensor");
  m.def(
      "_histogram_intersection_backward(Tensor grad, Tensor x, Tensor y) -> "
      "(Tensor, Tensor)");
}
at::Tensor _histogram_intersection(const at::Tensor& x, const at::Tensor& y) {
  static auto fn =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_histogram_intersection", "")
          .typed<decltype(_histogram_intersection)>();
  return fn.call(x, y);
}
namespace detail {
std::tuple<at::Tensor, at::Tensor> _histogram_intersection_backward(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& y) {
  static auto fn =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_histogram_intersection_backward",
                             "")
          .typed<decltype(_histogram_intersection_backward)>();
  return fn.call(grad, x, y);
}
}  // namespace detail
}  // namespace torchpairwise::ops
