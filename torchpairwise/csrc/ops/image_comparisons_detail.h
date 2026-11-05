#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::image {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using Core = Tensor(const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
                    int64_t, at::ArrayRef<double>);
inline Tensor _image_compare(const Tensor& x, const Tensor& y, int64_t kind,
                             int64_t mode, int64_t size, int64_t window,
                             at::ArrayRef<double> p) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_image_compare", "")
                       .typed<Core>();
  return op.call(x, y, kind, mode, size, window, p);
}

}  // namespace torchpairwise::ops::detail::image

namespace torchpairwise::ops::detail::image {
using MapCore = std::tuple<at::Tensor, at::Tensor>(const at::Tensor&,
                                                   const at::Tensor&, int64_t,
                                                   int64_t, int64_t, int64_t,
                                                   at::ArrayRef<double>, bool);
using MapGrad = std::tuple<at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t, int64_t, int64_t, int64_t, at::ArrayRef<double>);
inline std::tuple<at::Tensor, at::Tensor> image_map(
    const at::Tensor& x, const at::Tensor& y, int64_t kind, int64_t mode,
    int64_t size, int64_t window, at::ArrayRef<double> parameters,
    bool single_precision = false) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_image_map", "")
                       .typed<MapCore>();
  return op.call(x, y, kind, mode, size, window, parameters, single_precision);
}
}  // namespace torchpairwise::ops::detail::image
