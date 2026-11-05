#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
// p is accepted only for Minkowski; an explicitly unused option is an error.
// Inputs [...,N,D], [...,M,D] and weights [...,D] broadcast batch axes.
TORCHPAIRWISE_API at::Tensor weighted_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    const at::Tensor& w, c10::string_view metric,
    c10::string_view mode = "all_pairs",
    c10::optional<double> p = c10::nullopt);
}  // namespace torchpairwise::ops
