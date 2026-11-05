#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
enum class LSQRStatus : int64_t {
  Converged = 0,
  IterationLimit = 1,
  Breakdown = 2,
  Nonfinite = 3
};
struct LSQROptions {
  double damp = 0., rtol = 1e-5, atol = 0.;
  int64_t max_iter = 1000;
  c10::optional<at::Tensor> initial;
};
struct LSQRResult {
  at::Tensor solution, residual_norm, normal_residual_norm, iterations,
      converged, status;
};
// Minimize ||Ax-b||^2 + damp^2 ||x||^2 for real rectangular maps. Independent
// batch/RHS recurrences, inference only; true normal residual checks
// convergence.
TORCHPAIRWISE_API LSQRResult lsqr(const LinearOperator& op,
                                  const at::Tensor& rhs,
                                  const LSQROptions& options = {});
}  // namespace torchpairwise::ops
