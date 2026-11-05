#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
namespace utils {
template <bool is_boolean_metric = false, bool is_2d_metric = false>
static C10_ALWAYS_INLINE std::pair<at::Tensor, at::Tensor>
check_pairwise_inputs(const at::CheckedFrom& c, const at::Tensor& x1,
                      const c10::optional<at::Tensor>& x2 = c10::nullopt) {
  const auto y = x2.value_or(x1);
  const int64_t rank = is_2d_metric ? 3 : 2;
  // Tensor::size() materializes the whole concrete shape. Query
  // individual symbolic sizes so these checks do not specialize
  // unrelated sample dimensions during compile/export.
  TORCH_CHECK(x1.dim() == rank || x1.dim() == rank + 1,
              "x1 has an invalid rank for ", c);
  TORCH_CHECK(y.dim() == x1.dim(), "x1 and x2 must have the same rank for ", c);
  TORCH_CHECK(x1.sym_size(-1) == y.sym_size(-1),
              "Feature dimensions do not match for ", c);
  TORCH_CHECK(x1.sym_size(-1) > 0, "Feature dimension must be non-empty for ",
              c);
  TORCH_CHECK(x1.device() == y.device(),
              "x1 and x2 must be on the same device for ", c);
  TORCH_CHECK(x1.dim() == rank || x1.sym_size(0) == y.sym_size(0),
              "Batch dimensions do not match for ", c);
  if constexpr (is_2d_metric) {
    TORCH_CHECK(x1.sym_size(-2) > 0 && y.sym_size(-2) > 0,
                "Hausdorff point sets must be non-empty");
  }
  if constexpr (is_boolean_metric) {
    if (x1.scalar_type() != at::kBool || y.scalar_type() != at::kBool)
      TORCH_WARN("Data was converted to bool for metric ", c);
    return std::make_pair(x1.to(at::kBool), y.to(at::kBool));
  } else {
    TORCH_CHECK(x1.scalar_type() == y.scalar_type(),
                "x1 and x2 must have the same dtype for ", c);
    return std::make_pair(x1, y);
  }
}
}  // namespace utils

// ~~~~~ functions ~~~~~
// sklearn
TORCHPAIRWISE_API at::Tensor euclidean_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor haversine_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor manhattan_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor cosine_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor linear_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor polynomial_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    int64_t degree = 3, c10::optional<double> gamma = c10::nullopt,
    double coef0 = 1);

TORCHPAIRWISE_API at::Tensor sigmoid_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::optional<double> gamma = c10::nullopt, double coef0 = 1);

TORCHPAIRWISE_API at::Tensor rbf_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::optional<double> gamma = c10::nullopt);

TORCHPAIRWISE_API at::Tensor laplacian_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::optional<double> gamma = c10::nullopt);

TORCHPAIRWISE_API at::Tensor cosine_similarity(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor additive_chi2_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor chi2_kernel(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    double gamma = 1.0);

// scipy
TORCHPAIRWISE_API std::tuple<at::Tensor, at::Tensor, at::Tensor>
directed_hausdorff_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    bool shuffle = false,
    c10::optional<at::Generator> generator = c10::nullopt);

TORCHPAIRWISE_API at::Tensor minkowski_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    double p = 2);

TORCHPAIRWISE_API at::Tensor wminkowski_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    double p = 2, const c10::optional<at::Tensor>& w = c10::nullopt);

TORCHPAIRWISE_API at::Tensor sqeuclidean_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor correlation_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor hamming_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor jaccard_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor kulsinski_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor kulczynski1_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor seuclidean_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    const c10::optional<at::Tensor>& V = c10::nullopt);

TORCHPAIRWISE_API at::Tensor cityblock_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor mahalanobis_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    const c10::optional<at::Tensor>& VI = c10::nullopt);

TORCHPAIRWISE_API at::Tensor chebyshev_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor braycurtis_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor canberra_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor jensenshannon_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::optional<double> base = c10::nullopt);

TORCHPAIRWISE_API at::Tensor yule_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor dice_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor rogerstanimoto_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor russellrao_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor sokalmichener_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor sokalsneath_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

// others
TORCHPAIRWISE_API at::Tensor snr_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

// aliases
TORCHPAIRWISE_API at::Tensor l1_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor l2_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);

TORCHPAIRWISE_API at::Tensor lp_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    double p = 2);

TORCHPAIRWISE_API at::Tensor linf_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt);
}  // namespace torchpairwise::ops
