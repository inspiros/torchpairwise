#pragma once
#include <ATen/ATen.h>

namespace torchpairwise::ops::detail {
// State holds count, mean offset, centered square sum, and a fixed anchor.
// Values have already crossed any public-dtype rounding boundary. This is
// translated-precision handling, not full-range overflow protection.
inline void update_centered_moments(at::Tensor state, const at::Tensor& value,
                                    at::IntArrayRef axes, double count) {
  auto n = state.select(-1, 0), mean = state.select(-1, 1),
       m2 = state.select(-1, 2), anchor = state.select(-1, 3);
  auto first = value;
  for (auto axis : axes)
    first = first.narrow(axis, 0, 1);
  auto safe = at::where(first.isfinite(), first, 0.);
  anchor.copy_(at::where(n.gt(0), anchor, safe));
  auto shifted = value - anchor;
  auto average = shifted.sum(axes, true) / count;
  auto partial = (shifted - average).square().sum(axes, true);
  auto total = n + count, delta = average - mean, ratio = count / total;
  m2.copy_(
      at::where(n.gt(0), m2 + partial + delta.square() * n * ratio, partial));
  mean.copy_(at::where(n.gt(0), mean + delta * ratio, average));
  n.copy_(total);
}
inline at::Tensor centered_moment_delta(const at::Tensor& value,
                                        const at::Tensor& state) {
  return state.size(-1) == 4
             ? (value - state.select(-1, 3)) - state.select(-1, 1)
             : value - state.select(-1, 1);
}
// Preserve old auxiliary metadata without modifying the state saved by AD.
inline at::Tensor legacy_moment_state(const at::Tensor& state, int64_t anchor,
                                      int64_t mean) {
  if (state.size(-1) != anchor + 1)
    return state;
  auto result = state.narrow(-1, 0, anchor).clone();
  result.select(-1, mean).copy_(state.select(-1, mean) +
                                state.select(-1, anchor));
  return result;
}
// Center local ATen var/std inputs too: CPU sum-based means can overflow even
// for finite constants. Detach the numerical origin; var/std's VJP is applied
// directly to the shifted values, without a spurious first-coordinate branch.
inline at::Tensor centered_local_values(const at::Tensor& value) {
  if (value.size(-1) == 0)
    return value;
  auto first = value.narrow(-1, 0, 1).detach();
  return value - at::where(first.isfinite(), first, 0.);
}
}  // namespace torchpairwise::ops::detail
