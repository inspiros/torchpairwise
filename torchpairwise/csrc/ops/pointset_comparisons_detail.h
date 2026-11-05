#pragma once
// Internal native bridges shared by forward/backward kernels and autograd.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <array>
#include <vector>
namespace torchpairwise::ops::detail::pointset {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using CoreSignature = Triple(const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, int64_t, int64_t, bool, int64_t,
                             bool);
inline Triple _pointset_compare(const Tensor& x, const Tensor& y,
                                const Tensor& xl, const Tensor& yl,
                                int64_t kind, int64_t mode, bool squared,
                                int64_t size, bool legacy) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_pointset_compare", "")
          .typed<CoreSignature>();
  return op.call(x, y, xl, yl, kind, mode, squared, size, legacy);
}

}  // namespace torchpairwise::ops::detail::pointset
