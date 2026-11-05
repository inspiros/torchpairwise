#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
struct PivotedCholeskyResult {
  at::Tensor factor, pivots, error, rank;
};
enum class LanczosStatus : int64_t {
  Completed = 0,
  Breakdown = 1,
  InvalidInitial = 2,
  Nonfinite = 3
};
struct LanczosResult {
  at::Tensor basis, alpha, beta, iterations, status;
};
struct LOVECache {
  at::Tensor root;
};
// Scalar entrywise sources only. Symmetry/PSD are caller assumptions.
TORCHPAIRWISE_API PivotedCholeskyResult pivoted_cholesky(
    const LinearOperator& op, int64_t rank, double rtol = 0., double atol = 0.);
TORCHPAIRWISE_API LanczosResult lanczos(const LinearOperator& op, int64_t k,
                                        const at::Tensor& initial,
                                        bool reorthogonalize = true);
TORCHPAIRWISE_API LOVECache love_cache(const LinearOperator& op, int64_t rank,
                                       const at::Tensor& initial);
// Cross inputs remain differentiable; cache state is held constant.
TORCHPAIRWISE_API at::Tensor predictive_variance(
    const LOVECache& cache, const LinearOperator& cross,
    const at::Tensor& prior_variance);
TORCHPAIRWISE_API std::tuple<at::Tensor, at::Tensor> farthest_point_sampling(
    const LinearOperator& distances, int64_t k, int64_t start = 0);
TORCHPAIRWISE_API at::Tensor kmeans_plusplus(
    const LinearOperator& squared_distances, int64_t k,
    const c10::optional<at::Generator>& generator = c10::nullopt);
}  // namespace torchpairwise::ops
