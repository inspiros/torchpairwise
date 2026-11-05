#pragma once
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
namespace torchpairwise::ops::detail::transport_maps {
using Core = at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        int64_t, double, int64_t, bool);
using Grad = std::tuple<at::Tensor, at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, int64_t, double, int64_t, bool);
inline at::Tensor reduce(const at::Tensor& x, const at::Tensor& y,
                         const at::Tensor& logw, int64_t p, double epsilon,
                         int64_t size, bool barycentric = false) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_transport_map_reduce", "")
          .typed<Core>();
  return op.call(x, y, logw, p, epsilon, size, barycentric);
}
}  // namespace torchpairwise::ops::detail::transport_maps
