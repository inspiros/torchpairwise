#pragma once
#include "kernel_operations_detail.h"
namespace torchpairwise::ops::detail::kernel_moments {
using Reals = kernel_operations::Reals;
using Core = std::tuple<at::Tensor, at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::string_view,
    int64_t, Reals, c10::optional<int64_t>, bool);
using Grad = std::tuple<at::Tensor, at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, c10::string_view, int64_t, Reals,
    c10::optional<int64_t>, bool);
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> moments(
    const at::Tensor& x, const at::Tensor& y, const at::Tensor& origins,
    c10::string_view kernel, int64_t size, Reals reals,
    c10::optional<int64_t> degree, bool second = true) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_kernel_moments", "")
                       .typed<Core>();
  return op.call(x, y, origins, kernel, size, std::move(reals), degree, second);
}
}  // namespace torchpairwise::ops::detail::kernel_moments
