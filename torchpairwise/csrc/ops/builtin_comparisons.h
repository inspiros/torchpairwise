#pragma once
#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
// Shared options for registry vector/cloud distances and vector kernels.
// Omission differs from explicit defaults; unused options are rejected.
struct BuiltinComparisonOptions {
  c10::optional<at::Tensor> w, V, VI, bin_positions, projections, x1_lengths,
      x2_lengths;
  c10::optional<double> p, base, alpha, length_scale, nu, gamma, coef0,
      regularization;
  c10::optional<int64_t> degree, max_iter, chunk_size, feature_dim;
  c10::optional<bool> squared, shuffle, batched;
  c10::optional<std::string> kernel;
  c10::optional<at::Generator> generator;
};
TORCHPAIRWISE_API at::Tensor builtin_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view metric, c10::string_view mode = "all_pairs",
    bool kernel = false, const BuiltinComparisonOptions& options = {});
}  // namespace torchpairwise::ops

namespace torchpairwise::ops::detail {
// Pure ATen replay used only while recording higher-order backward graphs.
TORCHPAIRWISE_API at::Tensor higher_order_comparison(
    const at::Tensor& x, const at::Tensor& y, c10::string_view name,
    bool kernel, double p = 2., int64_t degree = 3, double gamma = 1.,
    double coef0 = 1., double length_scale = 1., double alpha = 1.,
    double nu = 1.5, bool paired = false);
}  // namespace torchpairwise::ops::detail
