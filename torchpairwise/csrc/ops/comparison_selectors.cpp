#include "comparison_selectors.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include <algorithm>

#include "audio_comparisons.h"
#include "builtin_comparisons.h"
#include "comparison_selectors_detail.h"
#include "image_comparisons.h"
#include "sequence_comparisons.h"
#include "structured_comparisons.h"
#include "token_comparisons.h"
#include "transport_comparisons.h"
#include "utils/comparison_dispatch.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using Entry =
    std::tuple<std::string, std::string, std::string, std::vector<std::string>,
               std::vector<std::string>, bool, bool, std::vector<std::string>,
               std::string, std::vector<int64_t>>;
// The public selector catalog, shared with Python introspection. Lower-level
// family dispatch tables choose algorithms; they do not define public coverage.
const std::vector<Entry>& catalog() {
  static const std::vector<Entry> entries = {
      {"additive_chi2_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"additive_chi2"},
       "kernel",
       {1}},
      {"affine_invariant_distances",
       "covariance",
       "[...,N,D,D] SPD matrices",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared"},
       true,
       true,
       {"affine_invariant"},
       "distance",
       {2}},
      {"aitchison_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"aitchison"},
       "distance",
       {1}},
      {"angular_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"angular"},
       "distance",
       {1}},
      {"bhattacharyya_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"bhattacharyya"},
       "distance",
       {1}},
      {"bhattacharyya_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"bhattacharyya"},
       "kernel",
       {1}},
      {"braycurtis_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       true,
       {"braycurtis"},
       "distance",
       {1}},
      {"canberra_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       true,
       {"canberra"},
       "distance",
       {1}},
      {"chamfer_distances",
       "point_cloud",
       "[...,N,P,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared", "x1_lengths", "x2_lengths", "chunk_size", "batched",
        "feature_dim"},
       true,
       true,
       {"chamfer"},
       "distance",
       {2}},
      {"chebyshev_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       true,
       {"chebyshev", "linf", "linf_distances"},
       "distance",
       {1}},
      {"chi2_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"gamma"},
       true,
       true,
       {"chi2"},
       "kernel",
       {1}},
      {"clark_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"clark"},
       "distance",
       {1}},
      {"correlation_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       true,
       {"correlation"},
       "distance",
       {1}},
      {"cosine_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       true,
       {"cosine"},
       "distance",
       {1}},
      {"cosine_similarity",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"cosine", "cosine_kernel"},
       "kernel",
       {1}},
      {"dice_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"dice"},
       "distance",
       {1}},
      {"directed_hausdorff_distances",
       "point_cloud",
       "[...,N,P,D]",
       {"all_pairs", "paired", "single"},
       {"shuffle", "generator", "x1_lengths", "x2_lengths", "chunk_size",
        "batched", "feature_dim"},
       false,
       true,
       {"directed_hausdorff"},
       "distance",
       {2}},
      {"dtw_distances",
       "sequence",
       "[...,N,T,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"x1_lengths", "x2_lengths", "squared", "window", "chunk_size"},
       true,
       true,
       {"dtw"},
       "distance",
       {2}},
      {"energy_distances",
       "point_cloud",
       "[...,N,P,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared", "x1_lengths", "x2_lengths", "chunk_size", "batched",
        "feature_dim"},
       true,
       true,
       {"energy"},
       "distance",
       {2}},
      {"euclidean_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       true,
       {"euclidean", "l2", "l2_distances"},
       "distance",
       {1}},
      {"fisher_rao_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"fisher_rao"},
       "distance",
       {1}},
      {"gaussian_mixture_cauchy_schwarz_divergences",
       "mixture",
       "(means [...,N,K,D], covariances [...,N,K,D,D], weights [...,N,K])",
       {"all_pairs", "paired", "condensed", "single"},
       {"x1_lengths", "x2_lengths", "chunk_size", "component_chunk_size"},
       true,
       true,
       {"gaussian_mixture_cauchy_schwarz"},
       "distance",
       {2, 3, 1}},
      {"gaussian_mixture_entropic_transport_costs",
       "mixture",
       "(means [...,N,K,D], covariances [...,N,K,D,D], weights [...,N,K])",
       {"all_pairs", "paired", "condensed", "single"},
       {"regularization", "max_iter", "x1_lengths", "x2_lengths", "chunk_size"},
       true,
       true,
       {"gaussian_mixture_entropic_transport"},
       "distance",
       {2, 3, 1}},
      {"gaussian_mixture_l2_distances",
       "mixture",
       "(means [...,N,K,D], covariances [...,N,K,D,D], weights [...,N,K])",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared", "x1_lengths", "x2_lengths", "chunk_size",
        "component_chunk_size"},
       true,
       true,
       {"gaussian_mixture_l2"},
       "distance",
       {2, 3, 1}},
      {"gaussian_mixture_overlap_kernel",
       "mixture",
       "(means [...,N,K,D], covariances [...,N,K,D,D], weights [...,N,K])",
       {"all_pairs", "paired", "condensed", "single"},
       {"x1_lengths", "x2_lengths", "chunk_size", "component_chunk_size"},
       true,
       true,
       {"gaussian_mixture_overlap"},
       "kernel",
       {2, 3, 1}},
      {"gaussian_mixture_sinkhorn_divergences",
       "mixture",
       "(means [...,N,K,D], covariances [...,N,K,D,D], weights [...,N,K])",
       {"all_pairs", "paired", "condensed", "single"},
       {"regularization", "max_iter", "x1_lengths", "x2_lengths", "chunk_size"},
       true,
       true,
       {"gaussian_mixture_sinkhorn"},
       "distance",
       {2, 3, 1}},
      {"gaussian_mixture_wasserstein_distances",
       "mixture",
       "(means [...,N,K,D], covariances [...,N,K,D,D], weights [...,N,K])",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared", "x1_lengths", "x2_lengths", "chunk_size"},
       true,
       true,
       {"gaussian_mixture_wasserstein"},
       "distance",
       {2, 3, 1}},
      {"gaussian_wasserstein_distances",
       "gaussian",
       "(means [...,N,D], covariances [...,N,D,D])",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared"},
       true,
       true,
       {"gaussian_wasserstein"},
       "distance",
       {1, 2}},
      {"hamming_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"hamming"},
       "distance",
       {1}},
      {"hausdorff_distances",
       "point_cloud",
       "[...,N,P,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"shuffle", "generator", "x1_lengths", "x2_lengths", "chunk_size",
        "batched", "feature_dim"},
       true,
       true,
       {"hausdorff"},
       "distance",
       {2}},
      {"haversine_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"haversine"},
       "distance",
       {1}},
      {"hellinger_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"hellinger"},
       "distance",
       {1}},
      {"histogram_intersection_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"histogram_intersection"},
       "kernel",
       {1}},
      {"indel_distances",
       "tokens",
       "[...,N,T] integer tokens",
       {"all_pairs", "paired", "condensed", "single"},
       {"x1_lengths", "x2_lengths", "chunk_size"},
       true,
       false,
       {"indel"},
       "distance",
       {1}},
      {"itakura_saito_divergences",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "single"},
       {},
       false,
       true,
       {"itakura_saito", "itakura_saito_distances"},
       "distance",
       {1}},
      {"jaccard_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"jaccard"},
       "distance",
       {1}},
      {"jensenshannon_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"base"},
       true,
       true,
       {"jensenshannon"},
       "distance",
       {1}},
      {"kl_divergences",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "single"},
       {"base"},
       false,
       true,
       {"kl", "kl_distances"},
       "distance",
       {1}},
      {"kulczynski1_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"kulczynski1"},
       "distance",
       {1}},
      {"kulsinski_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"kulsinski"},
       "distance",
       {1}},
      {"laplacian_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"gamma"},
       true,
       true,
       {"laplacian"},
       "kernel",
       {1}},
      {"levenshtein_distances",
       "tokens",
       "[...,N,T] integer tokens",
       {"all_pairs", "paired", "condensed", "single"},
       {"x1_lengths", "x2_lengths", "chunk_size"},
       true,
       false,
       {"levenshtein"},
       "distance",
       {1}},
      {"linear_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"linear"},
       "kernel",
       {1}},
      {"log_euclidean_distances",
       "covariance",
       "[...,N,D,D] SPD matrices",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared"},
       true,
       true,
       {"log_euclidean"},
       "distance",
       {2}},
      {"log_spectral_distances",
       "audio",
       "[...,N,C,T]",
       {"all_pairs", "paired", "condensed", "single"},
       {"n_fft", "hop_length", "win_length", "window", "center", "pad_mode",
        "normalized", "log_floor", "chunk_size"},
       true,
       true,
       {"log_spectral"},
       "distance",
       {2}},
      {"lorentzian_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"lorentzian"},
       "distance",
       {1}},
      {"mahalanobis_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"VI"},
       true,
       true,
       {"mahalanobis"},
       "distance",
       {1}},
      {"manhattan_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       true,
       {"cityblock", "cityblock_distances", "l1", "l1_distances", "manhattan"},
       "distance",
       {1}},
      {"matern_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"length_scale", "nu"},
       true,
       true,
       {"matern"},
       "kernel",
       {1}},
      {"minkowski_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"p", "w"},
       true,
       true,
       {"lp", "lp_distances", "minkowski"},
       "distance",
       {1}},
      {"mmd_distances",
       "point_cloud",
       "[...,N,P,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared", "kernel", "length_scale", "gamma", "degree", "coef0",
        "alpha", "nu", "x1_lengths", "x2_lengths", "chunk_size", "batched",
        "feature_dim"},
       true,
       true,
       {"mmd"},
       "distance",
       {2}},
      {"mmd_statistics",
       "point_cloud",
       "[...,N,P,C]",
       {"all_pairs", "paired", "condensed", "single"},
       {"kernel", "estimator", "length_scale", "gamma", "degree", "coef0",
        "alpha", "nu", "batched", "feature_dim", "x1_lengths", "x2_lengths",
        "chunk_size"},
       true,
       true,
       {"mmd_statistic"},
       "statistic",
       {2}},
      {"multiresolution_stft_distances",
       "audio",
       "[...,N,C,T]",
       {"all_pairs", "paired", "single"},
       {"resolutions", "window", "center", "pad_mode", "normalized",
        "log_floor", "norm_floor", "chunk_size"},
       false,
       true,
       {"multiresolution_stft"},
       "distance",
       {2}},
      {"multiscale_ssim_scores",
       "image",
       "[...,N,C,H,W]",
       {"all_pairs", "paired", "condensed", "single"},
       {"data_range", "window_size", "sigma", "k1", "k2", "weights",
        "chunk_size"},
       true,
       true,
       {"multiscale_ssim"},
       "score",
       {3}},
      {"nan_euclidean_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared"},
       true,
       true,
       {"nan_euclidean"},
       "distance",
       {1}},
      {"neyman_chi2_divergences",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "single"},
       {},
       false,
       true,
       {"neyman_chi2", "neyman_chi2_distances"},
       "distance",
       {1}},
      {"pearson_chi2_divergences",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "single"},
       {},
       false,
       true,
       {"pearson_chi2", "pearson_chi2_distances"},
       "distance",
       {1}},
      {"polynomial_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"degree", "gamma", "coef0"},
       true,
       true,
       {"polynomial"},
       "kernel",
       {1}},
      {"psnr_scores",
       "image",
       "[...,N,C,H,W]",
       {"all_pairs", "paired", "condensed", "single"},
       {"data_range", "chunk_size"},
       true,
       true,
       {"psnr"},
       "score",
       {3}},
      {"rational_quadratic_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"length_scale", "alpha"},
       true,
       true,
       {"rational_quadratic"},
       "kernel",
       {1}},
      {"rbf_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"gamma"},
       true,
       true,
       {"rbf"},
       "kernel",
       {1}},
      {"renyi_divergences",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "single"},
       {"alpha"},
       false,
       true,
       {"renyi", "renyi_distances"},
       "distance",
       {1}},
      {"rogerstanimoto_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"rogerstanimoto"},
       "distance",
       {1}},
      {"russellrao_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"russellrao"},
       "distance",
       {1}},
      {"ruzicka_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"continuous_jaccard", "continuous_jaccard_distances", "ruzicka"},
       "distance",
       {1}},
      {"seuclidean_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"V"},
       true,
       true,
       {"seuclidean"},
       "distance",
       {1}},
      {"si_sdr_scores",
       "audio",
       "[...,N,C,T]",
       {"all_pairs", "paired", "single"},
       {"zero_mean", "energy_floor", "chunk_size"},
       false,
       true,
       {"si_sdr"},
       "score",
       {2}},
      {"sigmoid_kernel",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"gamma", "coef0"},
       true,
       true,
       {"sigmoid"},
       "kernel",
       {1}},
      {"signal_snr_scores",
       "audio",
       "[...,N,C,T]",
       {"all_pairs", "paired", "single"},
       {"zero_mean", "energy_floor", "chunk_size"},
       false,
       true,
       {"signal_snr"},
       "score",
       {2}},
      {"sinkhorn_divergences",
       "point_cloud",
       "[...,N,P,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"p", "regularization", "max_iter", "x1_lengths", "x2_lengths",
        "chunk_size", "batched", "feature_dim"},
       true,
       true,
       {"sinkhorn", "sinkhorn_distances"},
       "distance",
       {2}},
      {"sliced_wasserstein_distances",
       "point_cloud",
       "[...,N,P,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"p", "projections", "x1_lengths", "x2_lengths", "chunk_size", "batched",
        "feature_dim"},
       true,
       true,
       {"sliced_wasserstein"},
       "distance",
       {2}},
      {"snr_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "single"},
       {},
       false,
       true,
       {"snr"},
       "distance",
       {1}},
      {"soergel_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"soergel"},
       "distance",
       {1}},
      {"soft_dtw_divergences",
       "sequence",
       "[...,N,T,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"gamma", "x1_lengths", "x2_lengths", "window", "chunk_size"},
       true,
       true,
       {"soft_dtw"},
       "distance",
       {2}},
      {"sokalmichener_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"sokalmichener"},
       "distance",
       {1}},
      {"sokalsneath_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"sokalsneath"},
       "distance",
       {1}},
      {"spectral_convergence_distances",
       "audio",
       "[...,N,C,T]",
       {"all_pairs", "paired", "single"},
       {"n_fft", "hop_length", "win_length", "window", "center", "pad_mode",
        "normalized", "norm_floor", "chunk_size"},
       false,
       true,
       {"spectral_convergence"},
       "distance",
       {2}},
      {"sqeuclidean_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       true,
       {"sqeuclidean"},
       "distance",
       {1}},
      {"ssim_scores",
       "image",
       "[...,N,C,H,W]",
       {"all_pairs", "paired", "condensed", "single"},
       {"data_range", "window_size", "sigma", "k1", "k2", "chunk_size"},
       true,
       true,
       {"ssim", "ssim_index"},
       "score",
       {3}},
      {"stein_distances",
       "covariance",
       "[...,N,D,D] SPD matrices",
       {"all_pairs", "paired", "condensed", "single"},
       {"squared"},
       true,
       true,
       {"stein"},
       "distance",
       {2}},
      {"symmetric_kl_divergences",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"base"},
       true,
       true,
       {"symmetric_kl", "symmetric_kl_distances"},
       "distance",
       {1}},
      {"total_variation_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"total_variation"},
       "distance",
       {1}},
      {"triangular_discrimination_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {},
       true,
       true,
       {"triangular_discrimination"},
       "distance",
       {1}},
      {"wasserstein_1d_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"bin_positions"},
       true,
       true,
       {"wasserstein_1d"},
       "distance",
       {1}},
      {"wasserstein_distances",
       "point_cloud",
       "[...,N,P,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"p", "x1_lengths", "x2_lengths", "chunk_size", "batched",
        "feature_dim"},
       true,
       true,
       {"wasserstein"},
       "distance",
       {2}},
      {"wminkowski_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"p", "w"},
       true,
       true,
       {"wminkowski"},
       "distance",
       {1}},
      {"yule_distances",
       "vector",
       "[...,N,D]",
       {"all_pairs", "paired", "condensed", "single"},
       {"w"},
       true,
       false,
       {"yule"},
       "distance",
       {1}},
  };
  return entries;
}
bool has(const std::vector<std::string>& values, c10::string_view key) {
  for (const auto& value : values)
    if (value == key)
      return true;
  return false;
}
const Entry& resolve(c10::string_view name, c10::string_view role) {
  TORCH_CHECK(role == "any" || role == "distance" || role == "kernel",
              "role must be any, distance or kernel");
  // Exact canonical names win over aliases. Shared short cosine/bhattacharyya
  // names select distance unless the caller explicitly requests kernels.
  for (const auto& entry : catalog())
    if (std::get<0>(entry) == name &&
        (role == "any" || std::get<8>(entry) == role))
      return entry;
  for (const auto& entry : catalog())
    if (has(std::get<7>(entry), name) &&
        (role == "any" || std::get<8>(entry) == role))
      return entry;
  TORCH_CHECK(false, "Unknown ", role, " comparison '", name,
              "'; use list_comparisons()");
}
void validate_options(const Entry& info, const ComparisonOptions& o) {
  const auto& allowed = std::get<4>(info);
  if (o.w.has_value())
    TORCH_CHECK_TYPE(has(allowed, "w"), std::get<0>(info),
                     " does not accept parameter w");
  if (o.V.has_value())
    TORCH_CHECK_TYPE(has(allowed, "V"), std::get<0>(info),
                     " does not accept parameter V");
  if (o.VI.has_value())
    TORCH_CHECK_TYPE(has(allowed, "VI"), std::get<0>(info),
                     " does not accept parameter VI");
  if (o.p.has_value())
    TORCH_CHECK_TYPE(has(allowed, "p"), std::get<0>(info),
                     " does not accept parameter p");
  if (o.base.has_value())
    TORCH_CHECK_TYPE(has(allowed, "base"), std::get<0>(info),
                     " does not accept parameter base");
  if (o.shuffle.has_value())
    TORCH_CHECK_TYPE(has(allowed, "shuffle"), std::get<0>(info),
                     " does not accept parameter shuffle");
  if (o.generator.has_value())
    TORCH_CHECK_TYPE(has(allowed, "generator"), std::get<0>(info),
                     " does not accept parameter generator");
  if (o.batched.has_value())
    TORCH_CHECK_TYPE(has(allowed, "batched"), std::get<0>(info),
                     " does not accept parameter batched");
  if (o.feature_dim.has_value())
    TORCH_CHECK_TYPE(has(allowed, "feature_dim"), std::get<0>(info),
                     " does not accept parameter feature_dim");
  if (o.squared.has_value())
    TORCH_CHECK_TYPE(has(allowed, "squared"), std::get<0>(info),
                     " does not accept parameter squared");
  if (o.x1_lengths.has_value())
    TORCH_CHECK_TYPE(has(allowed, "x1_lengths"), std::get<0>(info),
                     " does not accept parameter x1_lengths");
  if (o.x2_lengths.has_value())
    TORCH_CHECK_TYPE(has(allowed, "x2_lengths"), std::get<0>(info),
                     " does not accept parameter x2_lengths");
  if (o.chunk_size.has_value())
    TORCH_CHECK_TYPE(has(allowed, "chunk_size"), std::get<0>(info),
                     " does not accept parameter chunk_size");
  if (o.bin_positions.has_value())
    TORCH_CHECK_TYPE(has(allowed, "bin_positions"), std::get<0>(info),
                     " does not accept parameter bin_positions");
  if (o.projections.has_value())
    TORCH_CHECK_TYPE(has(allowed, "projections"), std::get<0>(info),
                     " does not accept parameter projections");
  if (o.regularization.has_value())
    TORCH_CHECK_TYPE(has(allowed, "regularization"), std::get<0>(info),
                     " does not accept parameter regularization");
  if (o.max_iter.has_value())
    TORCH_CHECK_TYPE(has(allowed, "max_iter"), std::get<0>(info),
                     " does not accept parameter max_iter");
  if (o.kernel.has_value())
    TORCH_CHECK_TYPE(has(allowed, "kernel"), std::get<0>(info),
                     " does not accept parameter kernel");
  if (o.length_scale.has_value())
    TORCH_CHECK_TYPE(has(allowed, "length_scale"), std::get<0>(info),
                     " does not accept parameter length_scale");
  if (o.alpha.has_value())
    TORCH_CHECK_TYPE(has(allowed, "alpha"), std::get<0>(info),
                     " does not accept parameter alpha");
  if (o.gamma.has_value())
    TORCH_CHECK_TYPE(has(allowed, "gamma"), std::get<0>(info),
                     " does not accept parameter gamma");
  if (o.degree.has_value())
    TORCH_CHECK_TYPE(has(allowed, "degree"), std::get<0>(info),
                     " does not accept parameter degree");
  if (o.coef0.has_value())
    TORCH_CHECK_TYPE(has(allowed, "coef0"), std::get<0>(info),
                     " does not accept parameter coef0");
  if (o.nu.has_value())
    TORCH_CHECK_TYPE(has(allowed, "nu"), std::get<0>(info),
                     " does not accept parameter nu");
  if (o.data_range.has_value())
    TORCH_CHECK_TYPE(has(allowed, "data_range"), std::get<0>(info),
                     " does not accept parameter data_range");
  if (o.window_size.has_value())
    TORCH_CHECK_TYPE(has(allowed, "window_size"), std::get<0>(info),
                     " does not accept parameter window_size");
  if (o.sigma.has_value())
    TORCH_CHECK_TYPE(has(allowed, "sigma"), std::get<0>(info),
                     " does not accept parameter sigma");
  if (o.k1.has_value())
    TORCH_CHECK_TYPE(has(allowed, "k1"), std::get<0>(info),
                     " does not accept parameter k1");
  if (o.k2.has_value())
    TORCH_CHECK_TYPE(has(allowed, "k2"), std::get<0>(info),
                     " does not accept parameter k2");
  if (o.weights.has_value())
    TORCH_CHECK_TYPE(has(allowed, "weights"), std::get<0>(info),
                     " does not accept parameter weights");
  if (o.zero_mean.has_value())
    TORCH_CHECK_TYPE(has(allowed, "zero_mean"), std::get<0>(info),
                     " does not accept parameter zero_mean");
  if (o.energy_floor.has_value())
    TORCH_CHECK_TYPE(has(allowed, "energy_floor"), std::get<0>(info),
                     " does not accept parameter energy_floor");
  if (o.log_floor.has_value())
    TORCH_CHECK_TYPE(has(allowed, "log_floor"), std::get<0>(info),
                     " does not accept parameter log_floor");
  if (o.norm_floor.has_value())
    TORCH_CHECK_TYPE(has(allowed, "norm_floor"), std::get<0>(info),
                     " does not accept parameter norm_floor");
  if (o.n_fft.has_value())
    TORCH_CHECK_TYPE(has(allowed, "n_fft"), std::get<0>(info),
                     " does not accept parameter n_fft");
  if (o.hop_length.has_value())
    TORCH_CHECK_TYPE(has(allowed, "hop_length"), std::get<0>(info),
                     " does not accept parameter hop_length");
  if (o.win_length.has_value())
    TORCH_CHECK_TYPE(has(allowed, "win_length"), std::get<0>(info),
                     " does not accept parameter win_length");
  if (!o.window.isNone())
    TORCH_CHECK_TYPE(has(allowed, "window"), std::get<0>(info),
                     " does not accept parameter window");
  if (o.center.has_value())
    TORCH_CHECK_TYPE(has(allowed, "center"), std::get<0>(info),
                     " does not accept parameter center");
  if (o.normalized.has_value())
    TORCH_CHECK_TYPE(has(allowed, "normalized"), std::get<0>(info),
                     " does not accept parameter normalized");
  if (o.pad_mode.has_value())
    TORCH_CHECK_TYPE(has(allowed, "pad_mode"), std::get<0>(info),
                     " does not accept parameter pad_mode");
  if (o.resolutions.has_value())
    TORCH_CHECK_TYPE(has(allowed, "resolutions"), std::get<0>(info),
                     " does not accept parameter resolutions");
  if (o.component_chunk_size.has_value())
    TORCH_CHECK_TYPE(has(allowed, "component_chunk_size"), std::get<0>(info),
                     " does not accept parameter component_chunk_size");
  if (o.estimator.has_value())
    TORCH_CHECK_TYPE(has(allowed, "estimator"), std::get<0>(info),
                     " does not accept parameter estimator");
}
std::vector<Entry> comparison_catalog() {
  return catalog();
}
}  // namespace
namespace detail {
Tensor select_comparison(at::TensorList input,
                         const c10::optional<ComparisonCollection>& other,
                         c10::string_view raw, c10::string_view mode,
                         c10::string_view role,
                         const ComparisonOptions& options) {
  const auto& info = resolve(raw, role);
  const auto& name = std::get<0>(info);
  const auto& family = std::get<1>(info);
  TORCH_CHECK(has(std::get<3>(info), mode), name, " does not support mode ",
              mode, "; condensed requires a supported symmetric comparison");
  TORCH_CHECK((mode != "paired" && mode != "single") || other.has_value(),
              "Paired/single comparisons require x2");
  TORCH_CHECK(mode != "condensed" || !other.has_value(),
              "Condensed comparisons accept x1 only");
  const auto& ranks = std::get<9>(info);
  TORCH_CHECK(
      input.size() == ranks.size() && (!other || other->size() == ranks.size()),
      family, " requires a collection/tuple of ", ranks.size(), " tensors");
  auto o = options;
  if (o.x_lengths) {
    TORCH_CHECK(
        (family == "sequence" || family == "tokens") && mode == "condensed",
        "x_lengths is only valid for condensed sequence/token comparisons");
    TORCH_CHECK(!o.x1_lengths, "Supply x1_lengths or x_lengths, not both");
    o.x1_lengths = o.x_lengths;
  }
  TORCH_CHECK(mode != "condensed" || !o.x2_lengths,
              "Condensed comparisons do not accept x2_lengths");
  validate_options(info, o);
  ComparisonCollection x(input.begin(), input.end());
  auto y = other;
  bool single = mode == "single";
  if (single) {
    TORCH_CHECK(!o.batched && !o.feature_dim,
                "single mode uses canonical sample dimensions; omit "
                "batched/feature_dim");
    for (size_t k = 0; k < ranks.size(); ++k) {
      TORCH_CHECK(x[k].dim() >= ranks[k] && (*y)[k].dim() >= ranks[k],
                  "Insufficient single-sample rank");
      bool same = x[k].is_same((*y)[k]);
      x[k] = x[k].unsqueeze(-ranks[k] - 1);
      (*y)[k] = same ? x[k] : (*y)[k].unsqueeze(-ranks[k] - 1);
    }
    if (o.x1_lengths)
      o.x1_lengths = o.x1_lengths->unsqueeze(-1);
    if (o.x2_lengths)
      o.x2_lengths = o.x2_lengths->unsqueeze(-1);
    mode = "paired";
  }
  c10::optional<Tensor> b = y ? c10::optional<Tensor>((*y)[0]) : c10::nullopt;
  Tensor result;
  if (family == "vector" ||
      (family == "point_cloud" && name != "mmd_statistics")) {
    BuiltinComparisonOptions p;
    p.w = o.w;
    p.V = o.V;
    p.VI = o.VI;
    p.p = o.p;
    p.base = o.base;
    p.shuffle = o.shuffle;
    p.generator = o.generator;
    p.batched = o.batched;
    p.feature_dim = o.feature_dim;
    p.squared = o.squared;
    p.x1_lengths = o.x1_lengths;
    p.x2_lengths = o.x2_lengths;
    p.chunk_size = o.chunk_size;
    p.bin_positions = o.bin_positions;
    p.projections = o.projections;
    p.regularization = o.regularization;
    p.max_iter = o.max_iter;
    p.kernel = o.kernel;
    p.length_scale = o.length_scale;
    p.alpha = o.alpha;
    p.gamma = o.gamma;
    p.degree = o.degree;
    p.coef0 = o.coef0;
    p.nu = o.nu;
    result = builtin_comparisons(x[0], b, name, mode,
                                 std::get<8>(info) == "kernel", p);
  } else if (family == "image") {
    TORCH_CHECK(o.data_range, "data_range is required");
    ImageComparisonOptions p;
    p.data_range = *o.data_range;
    if (o.chunk_size)
      p.chunk_size = *o.chunk_size;
    if (o.window_size)
      p.window_size = *o.window_size;
    if (o.sigma)
      p.sigma = *o.sigma;
    if (o.k1)
      p.k1 = *o.k1;
    if (o.k2)
      p.k2 = *o.k2;
    if (o.weights)
      p.weights = *o.weights;
    result = image_comparisons(x[0], b, name, mode, p);
  } else if (family == "audio") {
    AudioComparisonOptions p;
    if (o.chunk_size)
      p.chunk_size = *o.chunk_size;
    if (o.energy_floor)
      p.energy_floor = *o.energy_floor;
    if (o.log_floor)
      p.log_floor = *o.log_floor;
    if (o.norm_floor)
      p.norm_floor = *o.norm_floor;
    if (o.n_fft)
      p.n_fft = *o.n_fft;
    if (o.center)
      p.center = *o.center;
    if (o.normalized)
      p.normalized = *o.normalized;
    if (o.pad_mode)
      p.pad_mode = *o.pad_mode;
    if (o.resolutions)
      p.resolutions = *o.resolutions;
    p.zero_mean = o.zero_mean;
    p.hop_length = o.hop_length;
    p.win_length = o.win_length;
    if (!o.window.isNone()) {
      TORCH_CHECK_TYPE(o.window.isString(), "audio window must be a string");
      p.window = o.window.toStringRef();
    }
    result = audio_comparisons(x[0], b, name, mode, p);
  } else if (family == "sequence") {
    TORCH_CHECK(name != "soft_dtw_divergences" || o.gamma,
                "soft_dtw_divergences requires gamma");
    c10::optional<int64_t> window;
    if (!o.window.isNone()) {
      TORCH_CHECK_TYPE(o.window.isInt(), "sequence window must be an integer");
      window = o.window.toInt();
    }
    result = sequence_comparisons(
        x[0], b, mode, o.squared.value_or(false),
        name == "soft_dtw_divergences"
            ? c10::optional<double>(o.gamma.value_or(1.))
            : c10::nullopt,
        o.x1_lengths, o.x2_lengths, window, o.chunk_size.value_or(16));
  } else if (family == "tokens") {
    result = token_comparisons(x[0], b, mode, name == "indel_distances" ? 2 : 1,
                               o.x1_lengths, o.x2_lengths,
                               o.chunk_size.value_or(256));
  } else if (family == "covariance") {
    result =
        covariance_comparisons(x[0], b, detail::canonical_comparison_name(name),
                               mode, o.squared.value_or(false));
  } else if (family == "gaussian") {
    result = gaussian_comparisons(
        x[0], x[1], b, y ? c10::optional<Tensor>((*y)[1]) : c10::nullopt, mode,
        o.squared.value_or(false));
  } else if (family == "mixture") {
    result = mixture_comparisons(
        x[0], x[1], x[2], b, y ? c10::optional<Tensor>((*y)[1]) : c10::nullopt,
        y ? c10::optional<Tensor>((*y)[2]) : c10::nullopt,
        std::vector<const char*>{"overlap", "l2", "cs", "wasserstein",
                                 "entropic", "sinkhorn"}
            .at(detail::structured_comparison_kind(name) - 4),
        mode, o.x1_lengths, o.x2_lengths, o.squared, o.regularization,
        o.max_iter, o.chunk_size.value_or(8), o.component_chunk_size);
  } else {
    TransportComparisonOptions p;
    p.kernel = o.kernel;
    p.length_scale = o.length_scale;
    p.gamma = o.gamma;
    p.degree = o.degree;
    p.coef0 = o.coef0;
    p.alpha = o.alpha;
    p.nu = o.nu;
    p.x1_lengths = o.x1_lengths;
    p.x2_lengths = o.x2_lengths;
    p.chunk_size = o.chunk_size;
    p.batched = o.batched;
    p.feature_dim = o.feature_dim.value_or(-1);
    result = mmd_statistics(x[0], b, mode, o.estimator.value_or("unbiased"), p);
  }
  return single ? result.squeeze(-1) : result;
}
}  // namespace detail
namespace {
Tensor compare_impl(
    at::TensorList x1, const c10::optional<ComparisonCollection>& x2,
    c10::string_view metric, c10::string_view mode, c10::string_view role,
    c10::optional<at::Tensor> w, c10::optional<at::Tensor> V,
    c10::optional<at::Tensor> VI, c10::optional<double> p,
    c10::optional<double> base, c10::optional<bool> shuffle,
    c10::optional<at::Generator> generator, c10::optional<bool> batched,
    c10::optional<int64_t> feature_dim, c10::optional<bool> squared,
    c10::optional<at::Tensor> x1_lengths, c10::optional<at::Tensor> x2_lengths,
    c10::optional<int64_t> chunk_size, c10::optional<at::Tensor> bin_positions,
    c10::optional<at::Tensor> projections, c10::optional<double> regularization,
    c10::optional<int64_t> max_iter, c10::optional<std::string> kernel,
    c10::optional<double> length_scale, c10::optional<double> alpha,
    c10::optional<double> gamma, c10::optional<int64_t> degree,
    c10::optional<double> coef0, c10::optional<double> nu,
    c10::optional<double> data_range, c10::optional<int64_t> window_size,
    c10::optional<double> sigma, c10::optional<double> k1,
    c10::optional<double> k2, c10::optional<std::vector<double>> weights,
    c10::optional<bool> zero_mean, c10::optional<double> energy_floor,
    c10::optional<double> log_floor, c10::optional<double> norm_floor,
    c10::optional<int64_t> n_fft, c10::optional<int64_t> hop_length,
    c10::optional<int64_t> win_length, c10::optional<c10::IValue> window,
    c10::optional<bool> center, c10::optional<bool> normalized,
    c10::optional<std::string> pad_mode,
    c10::optional<std::vector<int64_t>> resolutions,
    c10::optional<int64_t> component_chunk_size,
    c10::optional<std::string> estimator, c10::optional<at::Tensor> x_lengths) {
  ComparisonOptions o;
  o.w = w;
  o.V = V;
  o.VI = VI;
  o.p = p;
  o.base = base;
  o.shuffle = shuffle;
  o.generator = generator;
  o.batched = batched;
  o.feature_dim = feature_dim;
  o.squared = squared;
  o.x1_lengths = x1_lengths;
  o.x2_lengths = x2_lengths;
  o.chunk_size = chunk_size;
  o.bin_positions = bin_positions;
  o.projections = projections;
  o.regularization = regularization;
  o.max_iter = max_iter;
  o.kernel = kernel;
  o.length_scale = length_scale;
  o.alpha = alpha;
  o.gamma = gamma;
  o.degree = degree;
  o.coef0 = coef0;
  o.nu = nu;
  o.data_range = data_range;
  o.window_size = window_size;
  o.sigma = sigma;
  o.k1 = k1;
  o.k2 = k2;
  o.weights = weights;
  o.zero_mean = zero_mean;
  o.energy_floor = energy_floor;
  o.log_floor = log_floor;
  o.norm_floor = norm_floor;
  o.n_fft = n_fft;
  o.hop_length = hop_length;
  o.win_length = win_length;
  o.window = window.value_or(c10::IValue());
  o.center = center;
  o.normalized = normalized;
  o.pad_mode = pad_mode;
  o.resolutions = resolutions;
  o.component_chunk_size = component_chunk_size;
  o.estimator = estimator;
  o.x_lengths = x_lengths;

  return detail::select_comparison(x1, x2, metric, mode, role, o);
}
}  // namespace
Tensor compare(at::TensorList x1, const c10::optional<ComparisonCollection>& x2,
               c10::string_view metric, c10::string_view mode,
               const ComparisonOptions& o, c10::string_view role) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::compare", "")
                       .typed<decltype(compare_impl)>();
  return op.call(
      x1, x2, metric, mode, role, o.w, o.V, o.VI, o.p, o.base, o.shuffle,
      o.generator, o.batched, o.feature_dim, o.squared, o.x1_lengths,
      o.x2_lengths, o.chunk_size, o.bin_positions, o.projections,
      o.regularization, o.max_iter, o.kernel, o.length_scale, o.alpha, o.gamma,
      o.degree, o.coef0, o.nu, o.data_range, o.window_size, o.sigma, o.k1, o.k2,
      o.weights, o.zero_mean, o.energy_floor, o.log_floor, o.norm_floor,
      o.n_fft, o.hop_length, o.win_length,
      o.window.isNone() ? c10::nullopt : c10::optional<c10::IValue>(o.window),
      o.center, o.normalized, o.pad_mode, o.resolutions, o.component_chunk_size,
      o.estimator, o.x_lengths);
}
Tensor compare(const Tensor& x1, const c10::optional<Tensor>& x2,
               c10::string_view metric, c10::string_view mode,
               const ComparisonOptions& o, c10::string_view role) {
  return compare(ComparisonCollection{x1},
                 x2 ? c10::optional<ComparisonCollection>({*x2}) : c10::nullopt,
                 metric, mode, o, role);
}
Tensor single_comparison(at::TensorList x1, at::TensorList x2,
                         c10::string_view metric, const ComparisonOptions& o) {
  return compare(x1, ComparisonCollection(x2.begin(), x2.end()), metric,
                 "single", o);
}
Tensor single_comparison(const Tensor& x1, const Tensor& x2,
                         c10::string_view metric, const ComparisonOptions& o) {
  return compare(x1, x2, metric, "single", o);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("comparison_catalog", TORCH_FN(comparison_catalog));
  m.def(
      "compare(Tensor[] x1, Tensor[]? x2=None, str metric='euclidean', str "
      "mode='all_pairs', str role='any', *, Tensor? w=None, Tensor? V=None, "
      "Tensor? VI=None, float? p=None, float? base=None, bool? shuffle=None, "
      "Generator? generator=None, bool? batched=None, int? feature_dim=None, "
      "bool? squared=None, Tensor? x1_lengths=None, Tensor? x2_lengths=None, "
      "int? chunk_size=None, Tensor? bin_positions=None, Tensor? "
      "projections=None, float? regularization=None, int? max_iter=None, str? "
      "kernel=None, float? length_scale=None, float? alpha=None, float? "
      "gamma=None, int? degree=None, float? coef0=None, float? nu=None, float? "
      "data_range=None, int? window_size=None, float? sigma=None, float? "
      "k1=None, float? k2=None, float[]? weights=None, bool? zero_mean=None, "
      "float? energy_floor=None, float? log_floor=None, float? "
      "norm_floor=None, int? n_fft=None, int? hop_length=None, int? "
      "win_length=None, Any? window=None, bool? center=None, bool? "
      "normalized=None, str? pad_mode=None, int[]? resolutions=None, int? "
      "component_chunk_size=None, str? estimator=None, Tensor? x_lengths=None) "
      "-> Tensor",
      TORCH_FN(compare_impl));
}
}  // namespace torchpairwise::ops
