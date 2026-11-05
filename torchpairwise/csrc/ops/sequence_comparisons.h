#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
// Collections [...,N,T,D]; optional prefix lengths [...,N]. gamma=None
// selects DTW, otherwise the debiased Soft-DTW divergence is returned.
TORCHPAIRWISE_API at::Tensor sequence_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view mode = "all_pairs", bool squared = false,
    c10::optional<double> gamma = c10::nullopt,
    const c10::optional<at::Tensor>& x1_lengths = c10::nullopt,
    const c10::optional<at::Tensor>& x2_lengths = c10::nullopt,
    c10::optional<int64_t> window = c10::nullopt, int64_t chunk_size = 16);
}  // namespace torchpairwise::ops
