#pragma once
#include <ATen/ATen.h>

#include "../macros.h"
namespace torchpairwise::ops {
TORCHPAIRWISE_API at::Tensor gaussian_comparisons(
    const at::Tensor& mean, const at::Tensor& covariance,
    const c10::optional<at::Tensor>& other_mean = c10::nullopt,
    const c10::optional<at::Tensor>& other_covariance = c10::nullopt,
    c10::string_view mode = "all_pairs", bool squared = false);
TORCHPAIRWISE_API at::Tensor covariance_comparisons(
    const at::Tensor& covariance, const c10::optional<at::Tensor>& other,
    c10::string_view metric, c10::string_view mode = "all_pairs",
    bool squared = false);
TORCHPAIRWISE_API at::Tensor mixture_comparisons(
    const at::Tensor& mean, const at::Tensor& covariance,
    const at::Tensor& weights, const c10::optional<at::Tensor>& other_mean,
    const c10::optional<at::Tensor>& other_covariance,
    const c10::optional<at::Tensor>& other_weights, c10::string_view metric,
    c10::string_view mode = "all_pairs",
    const c10::optional<at::Tensor>& lengths = c10::nullopt,
    const c10::optional<at::Tensor>& other_lengths = c10::nullopt,
    c10::optional<bool> squared = c10::nullopt,
    c10::optional<double> regularization = c10::nullopt,
    c10::optional<int64_t> max_iter = c10::nullopt, int64_t chunk_size = 8,
    c10::optional<int64_t> component_chunk_size = c10::nullopt);
}  // namespace torchpairwise::ops
