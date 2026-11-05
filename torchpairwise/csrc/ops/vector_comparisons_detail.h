#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::vector {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::array<Tensor, 3>;
using CoreSignature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                             int64_t, int64_t, double, double, double, double,
                             bool);
inline Tensor _vector_compare(const Tensor& x, const Tensor& y,
                              const Tensor& grid, int64_t kind, int64_t mode,
                              double base, double length_scale, double alpha,
                              double nu, bool squared) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_vector_compare", "")
                       .typed<CoreSignature>();
  return op.call(x, y, grid, kind, mode, base, length_scale, alpha, nu,
                 squared);
}
Values feature_autograd(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e,
                        Tensor f, Tensor scale, int64_t kind, bool paired,
                        double alpha, double limit);
Values feature_kernel(Triple a, Triple b, const Tensor& scale, int64_t kind,
                      bool paired, double alpha, double limit);
}  // namespace torchpairwise::ops::detail::vector
