#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
enum class SVDStatus : int64_t {
  Converged = 0,
  IterationLimit = 1,
  RankDeficientInitial = 2,
  Nonfinite = 3,
  LossOfOrthogonality = 4
};
struct SVDOptions {
  int64_t k = 0;
  c10::optional<int64_t> subspace_size;
  int64_t max_iter = 100;
  double rtol = 1e-5, atol = 0.;
  c10::optional<at::Tensor> initial;  // Right search block [...,M,r].
};
struct SVDResult {
  at::Tensor singular_values, left_vectors, right_vectors, residual_norm,
      iterations, converged, status;
};
// Leading singular triplets of a real rectangular map. Inference-only
// two-sided block iteration; no normal matrix is constructed. Right vectors
// are columns [...,M,k], so A approximates U diag(s) V^T.
TORCHPAIRWISE_API SVDResult svd(const LinearOperator& op,
                                const SVDOptions& options = {});
}  // namespace torchpairwise::ops
