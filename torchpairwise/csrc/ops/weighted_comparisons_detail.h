#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::weighted {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using CoreSignature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                             int64_t, int64_t, double, bool);
inline Tensor _weighted_compare(const Tensor& x, const Tensor& y,
                                const Tensor& w, int64_t kind, int64_t mode,
                                double p, bool self) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_weighted_compare", "")
          .typed<CoreSignature>();
  return op.call(x, y, w, kind, mode, p, self);
}
Tensor feature_autograd(Tensor x, Tensor y, Tensor w, int64_t kind, bool paired,
                        double p);
Tensor feature_kernel(const Tensor& x, const Tensor& y, const Tensor& w,
                      int64_t kind, bool paired, double p);
}  // namespace torchpairwise::ops::detail::weighted
