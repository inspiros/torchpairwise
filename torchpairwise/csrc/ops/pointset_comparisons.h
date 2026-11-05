#pragma once
#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
struct PointsetComparisonOptions {
  c10::optional<bool> squared;
  c10::optional<at::Tensor> x1_lengths, x2_lengths;
  c10::optional<int64_t> chunk_size;
  c10::optional<bool> batched;
  int64_t feature_dim = -1;
  c10::optional<bool> shuffle;
  c10::optional<at::Generator> generator;
};
// Coordinates are dense tensors; ragged clouds use valid prefix lengths.
// batched/feature_dim normalize sample-local spatial axes just as in Python.
TORCHPAIRWISE_API at::Tensor pointset_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view metric, c10::string_view mode = "all_pairs",
    const PointsetComparisonOptions& options = {});
TORCHPAIRWISE_API std::tuple<at::Tensor, at::Tensor, at::Tensor>
directed_pointset_comparisons(const at::Tensor& x1,
                              const c10::optional<at::Tensor>& x2,
                              c10::string_view mode = "all_pairs",
                              const PointsetComparisonOptions& options = {});
}  // namespace torchpairwise::ops
