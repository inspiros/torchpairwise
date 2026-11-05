#pragma once
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <vector>

#include "builtin_comparisons.h"

namespace torchpairwise::ops::detail::kernel_operations {
using Reals = std::vector<c10::optional<double>>;
inline BuiltinComparisonOptions comparison_options(
    const Reals& reals, c10::optional<int64_t> degree) {
  TORCH_CHECK(reals.size() == 5, "Expected five kernel scalar options");
  BuiltinComparisonOptions q;
  q.gamma = reals[0];
  q.coef0 = reals[1];
  q.length_scale = reals[2];
  q.alpha = reals[3];
  q.nu = reals[4];
  q.degree = degree;
  return q;
}

using Core = at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, c10::string_view, int64_t, Reals,
                        c10::optional<int64_t>);
using Grad = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, c10::string_view, int64_t, Reals,
    c10::optional<int64_t>);
inline at::Tensor kernel_form(const at::Tensor& x, const at::Tensor& y,
                              const at::Tensor& left, const at::Tensor& right,
                              c10::string_view kernel, int64_t size,
                              Reals reals, c10::optional<int64_t> degree) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_kernel_form", "")
                       .typed<Core>();
  return op.call(x, y, left, right, kernel, size, std::move(reals), degree);
}
}  // namespace torchpairwise::ops::detail::kernel_operations
