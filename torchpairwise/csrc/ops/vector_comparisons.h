#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
// Optional fields distinguish omission from an explicitly supplied parameter.
// The generic entry point rejects options not used by the selected metric.
struct VectorComparisonOptions {
  c10::optional<double> base, length_scale, alpha, nu;
  c10::optional<bool> squared;
  c10::optional<at::Tensor> bin_positions;
};

TORCHPAIRWISE_API at::Tensor vector_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view metric, c10::string_view mode = "all_pairs",
    const VectorComparisonOptions& options = {});

// Named entry points also have matching dispatcher schemas.
TORCHPAIRWISE_API at::Tensor angular_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_angular_distances(const at::Tensor& x1,
                                                      const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_angular_distances(const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor lorentzian_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_lorentzian_distances(const at::Tensor& x1,
                                                         const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_lorentzian_distances(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor clark_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_clark_distances(const at::Tensor& x1,
                                                    const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_clark_distances(const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor soergel_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_soergel_distances(const at::Tensor& x1,
                                                      const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_soergel_distances(const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor ruzicka_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_ruzicka_distances(const at::Tensor& x1,
                                                      const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_ruzicka_distances(const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor bhattacharyya_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_bhattacharyya_distances(
    const at::Tensor& x1, const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_bhattacharyya_distances(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor bhattacharyya_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_bhattacharyya_kernel(const at::Tensor& x1,
                                                         const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_bhattacharyya_kernel(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor aitchison_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_aitchison_distances(const at::Tensor& x1,
                                                        const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_aitchison_distances(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor hellinger_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_hellinger_distances(const at::Tensor& x1,
                                                        const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_hellinger_distances(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor histogram_intersection_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_histogram_intersection_kernel(
    const at::Tensor& x1, const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_histogram_intersection_kernel(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor total_variation_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_total_variation_distances(
    const at::Tensor& x1, const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_total_variation_distances(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor fisher_rao_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_fisher_rao_distances(const at::Tensor& x1,
                                                         const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_fisher_rao_distances(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor pearson_chi2_divergences(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_pearson_chi2_divergences(
    const at::Tensor& x1, const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor neyman_chi2_divergences(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_neyman_chi2_divergences(
    const at::Tensor& x1, const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor triangular_discrimination_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_triangular_discrimination_distances(
    const at::Tensor& x1, const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor condensed_triangular_discrimination_distances(
    const at::Tensor& x1);
TORCHPAIRWISE_API at::Tensor itakura_saito_divergences(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_itakura_saito_divergences(
    const at::Tensor& x1, const at::Tensor& x2);
TORCHPAIRWISE_API at::Tensor kl_divergences(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::optional<double> base = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_kl_divergences(
    const at::Tensor& x1, const at::Tensor& x2,
    c10::optional<double> base = c10::nullopt);
TORCHPAIRWISE_API at::Tensor symmetric_kl_divergences(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::optional<double> base = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_symmetric_kl_divergences(
    const at::Tensor& x1, const at::Tensor& x2,
    c10::optional<double> base = c10::nullopt);
TORCHPAIRWISE_API at::Tensor condensed_symmetric_kl_divergences(
    const at::Tensor& x1, c10::optional<double> base = c10::nullopt);
TORCHPAIRWISE_API at::Tensor renyi_divergences(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    double alpha = 1.0);
TORCHPAIRWISE_API at::Tensor paired_renyi_divergences(const at::Tensor& x1,
                                                      const at::Tensor& x2,
                                                      double alpha = 1.0);
TORCHPAIRWISE_API at::Tensor rational_quadratic_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    double length_scale = 1.0, double alpha = 1.0);
TORCHPAIRWISE_API at::Tensor paired_rational_quadratic_kernel(
    const at::Tensor& x1, const at::Tensor& x2, double length_scale = 1.0,
    double alpha = 1.0);
TORCHPAIRWISE_API at::Tensor condensed_rational_quadratic_kernel(
    const at::Tensor& x1, double length_scale = 1.0, double alpha = 1.0);
TORCHPAIRWISE_API at::Tensor matern_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    double length_scale = 1.0, double nu = 1.5);
TORCHPAIRWISE_API at::Tensor paired_matern_kernel(const at::Tensor& x1,
                                                  const at::Tensor& x2,
                                                  double length_scale = 1.0,
                                                  double nu = 1.5);
TORCHPAIRWISE_API at::Tensor condensed_matern_kernel(const at::Tensor& x1,
                                                     double length_scale = 1.0,
                                                     double nu = 1.5);
TORCHPAIRWISE_API at::Tensor nan_euclidean_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    bool squared = false);
TORCHPAIRWISE_API at::Tensor paired_nan_euclidean_distances(
    const at::Tensor& x1, const at::Tensor& x2, bool squared = false);
TORCHPAIRWISE_API at::Tensor condensed_nan_euclidean_distances(
    const at::Tensor& x1, bool squared = false);
TORCHPAIRWISE_API at::Tensor wasserstein_1d_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    const c10::optional<at::Tensor>& bin_positions = c10::nullopt);
TORCHPAIRWISE_API at::Tensor paired_wasserstein_1d_distances(
    const at::Tensor& x1, const at::Tensor& x2,
    const c10::optional<at::Tensor>& bin_positions = c10::nullopt);
TORCHPAIRWISE_API at::Tensor condensed_wasserstein_1d_distances(
    const at::Tensor& x1,
    const c10::optional<at::Tensor>& bin_positions = c10::nullopt);
}  // namespace torchpairwise::ops
