#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::transport {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using Core = Tensor(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                    const Tensor&, int64_t, int64_t, double, int64_t, double,
                    double, int64_t, bool, int64_t);
inline Tensor _transport_compare(const Tensor& x, const Tensor& y,
                                 const Tensor& xl, const Tensor& yl,
                                 const Tensor& directions, int64_t kind,
                                 int64_t mode, double p, int64_t kernel,
                                 double scale, double epsilon,
                                 int64_t iterations, bool squared,
                                 int64_t size) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_transport_compare", "")
          .typed<Core>();
  return op.call(x, y, xl, yl, directions, kind, mode, p, kernel, scale,
                 epsilon, iterations, squared, size);
}
Tensor point_autograd(Tensor x, Tensor y, Tensor xv, Tensor yv, int64_t kind,
                      int64_t kernel, double scale);
Tensor point_kernel(const Tensor& x, const Tensor& y, const Tensor& xv,
                    const Tensor& yv, int64_t kind, int64_t kernel,
                    double scale);
}  // namespace torchpairwise::ops::detail::transport
