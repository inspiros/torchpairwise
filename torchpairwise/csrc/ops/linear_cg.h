#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
enum class CGStatus : int64_t {
  Converged = 0,
  IterationLimit = 1,
  NonpositiveCurvature = 2,
  Nonfinite = 3,
  InvalidPreconditioner = 4
};
struct CGOptions {
  double rtol = 1e-5, atol = 0.;
  int64_t max_iter = 1000;
  c10::optional<at::Tensor> initial;
  // Fixed SPD inverse action M approximately A^{-1}; not the matrix to invert.
  c10::optional<LinearOperator> preconditioner;
};
struct CGResult {
  at::Tensor solution, residual_norm, iterations, converged, status;
};
// Independent real SPD solves per batch/RHS. Inference only; actual residuals
// verify convergence. No implicit dense matrix or inverse is constructed.
TORCHPAIRWISE_API CGResult cg(const LinearOperator& op, const at::Tensor& rhs,
                              const CGOptions& options = {});
}  // namespace torchpairwise::ops
