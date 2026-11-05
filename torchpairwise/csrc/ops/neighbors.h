#pragma once

#include <ATen/ATen.h>

#include "../macros.h"
#include "builtin_comparisons.h"

namespace torchpairwise::ops {
// Reusable selection over [...,N,M] distances. Stable sorting puts NaNs
// after finite/infinite candidates and breaks ties by column index.
TORCHPAIRWISE_API at::Tensor neighbor_mask_from_distances(
    const at::Tensor& distances, c10::string_view selection,
    c10::optional<int64_t> k = c10::nullopt,
    c10::optional<double> epsilon = c10::nullopt, bool exclude_self = false);

// Full vector/cloud catalog with the same options as builtin_comparisons.
// Omitted exclude_self excludes the diagonal only when x2 is omitted.
TORCHPAIRWISE_API at::Tensor builtin_neighbor_mask(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view selection, c10::optional<int64_t> k = c10::nullopt,
    c10::optional<double> epsilon = c10::nullopt,
    c10::optional<bool> exclude_self = c10::nullopt,
    c10::string_view metric = "euclidean",
    const BuiltinComparisonOptions& options = {});

TORCHPAIRWISE_API at::Tensor k_neighbors_mask(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    int64_t k = 1, c10::string_view metric = "euclidean",
    const c10::optional<at::Tensor>& w = c10::nullopt,
    const c10::optional<at::Tensor>& V = c10::nullopt,
    const c10::optional<at::Tensor>& VI = c10::nullopt,
    c10::optional<double> p = c10::nullopt,
    c10::optional<double> base = c10::nullopt,
    c10::optional<bool> shuffle = c10::nullopt,
    c10::optional<at::Generator> generator = c10::nullopt);

TORCHPAIRWISE_API at::Tensor radius_neighbors_mask(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    double epsilon = 0, c10::string_view metric = "euclidean",
    const c10::optional<at::Tensor>& w = c10::nullopt,
    const c10::optional<at::Tensor>& V = c10::nullopt,
    const c10::optional<at::Tensor>& VI = c10::nullopt,
    c10::optional<double> p = c10::nullopt,
    c10::optional<double> base = c10::nullopt,
    c10::optional<bool> shuffle = c10::nullopt,
    c10::optional<at::Generator> generator = c10::nullopt);
}  // namespace torchpairwise::ops
