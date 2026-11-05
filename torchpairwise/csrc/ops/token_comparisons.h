#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
// substitution_cost=1: Levenshtein; substitution_cost=2: insertion/deletion.
TORCHPAIRWISE_API at::Tensor token_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view mode = "all_pairs", int64_t substitution_cost = 1,
    const c10::optional<at::Tensor>& x1_lengths = c10::nullopt,
    const c10::optional<at::Tensor>& x2_lengths = c10::nullopt,
    int64_t chunk_size = 256);
}  // namespace torchpairwise::ops
