#pragma once
#include "builtin_comparisons.h"

namespace torchpairwise::ops {
struct StreamingReductionOptions {
  BuiltinComparisonOptions comparison;
  c10::optional<at::Tensor> values;
  c10::optional<int64_t> chunk_size;
  c10::optional<bool> exclude_self;
  int64_t k = 1;
  bool largest = false, keepdim = false;
};
// operation: neighbors, min, max, sum, prod, all, any, kernel_sum.
// Neighbor indices are stable reference indices; other operations return an
// empty index tensor. Arbitrary Python predicates are a separate callback API.
TORCHPAIRWISE_API std::tuple<at::Tensor, at::Tensor> streaming_reductions(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view metric = "euclidean", c10::string_view operation = "sum",
    bool kernel = false, const StreamingReductionOptions& options = {});
}  // namespace torchpairwise::ops
