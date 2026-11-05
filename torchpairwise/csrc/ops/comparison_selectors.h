#pragma once
#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
// Omission is distinct from supplying a default. Unused options are errors.
// window is an integer DTW band or a string STFT window, according to the
// metric.
struct ComparisonOptions {
  c10::optional<at::Tensor> w;
  c10::optional<at::Tensor> V;
  c10::optional<at::Tensor> VI;
  c10::optional<double> p;
  c10::optional<double> base;
  c10::optional<bool> shuffle;
  c10::optional<at::Generator> generator;
  c10::optional<bool> batched;
  c10::optional<int64_t> feature_dim;
  c10::optional<bool> squared;
  c10::optional<at::Tensor> x1_lengths;
  c10::optional<at::Tensor> x2_lengths;
  c10::optional<int64_t> chunk_size;
  c10::optional<at::Tensor> bin_positions;
  c10::optional<at::Tensor> projections;
  c10::optional<double> regularization;
  c10::optional<int64_t> max_iter;
  c10::optional<std::string> kernel;
  c10::optional<double> length_scale;
  c10::optional<double> alpha;
  c10::optional<double> gamma;
  c10::optional<int64_t> degree;
  c10::optional<double> coef0;
  c10::optional<double> nu;
  c10::optional<double> data_range;
  c10::optional<int64_t> window_size;
  c10::optional<double> sigma;
  c10::optional<double> k1;
  c10::optional<double> k2;
  c10::optional<std::vector<double>> weights;
  c10::optional<bool> zero_mean;
  c10::optional<double> energy_floor;
  c10::optional<double> log_floor;
  c10::optional<double> norm_floor;
  c10::optional<int64_t> n_fft;
  c10::optional<int64_t> hop_length;
  c10::optional<int64_t> win_length;
  c10::IValue window;
  c10::optional<bool> center;
  c10::optional<bool> normalized;
  c10::optional<std::string> pad_mode;
  c10::optional<std::vector<int64_t>> resolutions;
  c10::optional<int64_t> component_chunk_size;
  c10::optional<std::string> estimator;
  c10::optional<at::Tensor> x_lengths;
};
using ComparisonCollection = std::vector<at::Tensor>;
// Single mode removes the collection sample axis; all modes share family
// kernels.
TORCHPAIRWISE_API at::Tensor compare(
    at::TensorList x1,
    const c10::optional<ComparisonCollection>& x2 = c10::nullopt,
    c10::string_view metric = "euclidean", c10::string_view mode = "all_pairs",
    const ComparisonOptions& options = {}, c10::string_view role = "any");
TORCHPAIRWISE_API at::Tensor single_comparison(
    at::TensorList x1, at::TensorList x2, c10::string_view metric = "euclidean",
    const ComparisonOptions& options = {});
TORCHPAIRWISE_API at::Tensor compare(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view metric = "euclidean", c10::string_view mode = "all_pairs",
    const ComparisonOptions& options = {}, c10::string_view role = "any");
TORCHPAIRWISE_API at::Tensor single_comparison(
    const at::Tensor& x1, const at::Tensor& x2,
    c10::string_view metric = "euclidean",
    const ComparisonOptions& options = {});
}  // namespace torchpairwise::ops
