#pragma once
#include <ATen/ATen.h>

#include <algorithm>
#include <numeric>

namespace torchpairwise::ops::detail {
// Empty dimensions mean all logical operator axes; batch axes are preserved.
inline std::vector<int64_t> operator_axes(at::IntArrayRef dims, int64_t rank,
                                          int64_t logical_rank) {
  std::vector<int64_t> axes;
  if (dims.empty()) {
    axes.resize(logical_rank);
    std::iota(axes.begin(), axes.end(), 0);
    return axes;
  }
  for (auto dim : dims) {
    TORCH_CHECK(dim >= -rank && dim < rank, "dim is outside the operator rank");
    auto axis = (dim < 0 ? dim + rank : dim) - (rank - logical_rank);
    TORCH_CHECK(axis >= 0, "Operator reductions preserve batch axes");
    TORCH_CHECK(std::find(axes.begin(), axes.end(), axis) == axes.end(),
                "dim contains a repeated axis");
    axes.push_back(axis);
  }
  std::sort(axes.begin(), axes.end());
  return axes;
}
}  // namespace torchpairwise::ops::detail
