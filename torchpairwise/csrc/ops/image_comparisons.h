#pragma once
#include <ATen/ATen.h>

#include <limits>

#include "../macros.h"
namespace torchpairwise::ops {
struct ImageComparisonOptions {
  double data_range = std::numeric_limits<double>::quiet_NaN();
  int64_t chunk_size = 16, window_size = 11;
  double sigma = 1.5, k1 = .01, k2 = .03;
  std::vector<double> weights{.0448, .2856, .3001, .2363, .1333};
};
// Images [...,N,C,H,W]. Modes all_pairs, paired and condensed; PSNR/SSIM/
// multiscale_ssim have the same conventions as their Python named wrappers.
TORCHPAIRWISE_API at::Tensor image_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view metric, c10::string_view mode = "all_pairs",
    const ImageComparisonOptions& options = {});
struct ImageMapOptions {
  c10::optional<double> data_range;
  int64_t chunk_size = 16, window_size = 11;
  double sigma = 1.5, k1 = .01, k2 = .03;
  bool keepdim = false;
};
TORCHPAIRWISE_API at::Tensor image_maps(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view comparison = "squared_error",
    c10::string_view mode = "all_pairs", c10::string_view reduction = "none",
    const ImageMapOptions& options = {});
}  // namespace torchpairwise::ops
