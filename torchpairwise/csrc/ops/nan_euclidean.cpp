#include "nan_euclidean.h"

#include <torch/library.h>
namespace torchpairwise::ops {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_nan_sqeuclidean(Tensor x, Tensor y) -> (Tensor squared, Tensor "
      "counts)");
  m.def(
      "_nan_sqeuclidean_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "counts) -> (Tensor, Tensor)");
}
std::tuple<at::Tensor, at::Tensor> _nan_sqeuclidean(const at::Tensor& x,
                                                    const at::Tensor& y) {
  static auto fn = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_nan_sqeuclidean", "")
                       .typed<decltype(_nan_sqeuclidean)>();
  return fn.call(x, y);
}
namespace detail {
std::tuple<at::Tensor, at::Tensor> _nan_sqeuclidean_backward(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& y,
    const at::Tensor& counts) {
  static auto fn =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_nan_sqeuclidean_backward", "")
          .typed<decltype(_nan_sqeuclidean_backward)>();
  return fn.call(grad, x, y, counts);
}
}  // namespace detail
}  // namespace torchpairwise::ops
