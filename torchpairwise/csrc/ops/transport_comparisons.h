#pragma once
#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
struct TransportComparisonOptions {
  c10::optional<double> p, regularization, length_scale, gamma, coef0, alpha,
      nu;
  c10::optional<int64_t> max_iter, chunk_size, degree;
  c10::optional<bool> squared, batched;
  int64_t feature_dim = -1;
  c10::optional<at::Tensor> x1_lengths, x2_lengths, projections;
  c10::optional<std::string> kernel;
};
TORCHPAIRWISE_API at::Tensor transport_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view metric, c10::string_view mode = "all_pairs",
    const TransportComparisonOptions& options = {});
// Named-kernel empirical MMD. The original transport dispatcher remains valid.
TORCHPAIRWISE_API at::Tensor mmd_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view mode = "all_pairs",
    const TransportComparisonOptions& options = {});
// Signed empirical squared MMD; unbiased excludes within-cloud diagonals.
TORCHPAIRWISE_API at::Tensor mmd_statistics(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view mode = "all_pairs",
    c10::string_view estimator = "unbiased",
    const TransportComparisonOptions& options = {});
}  // namespace torchpairwise::ops
