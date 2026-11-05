#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
enum class MINRESStatus : int64_t {
  Converged = 0,
  IterationLimit = 1,
  Breakdown = 2,
  Nonfinite = 3
};
struct MINRESOptions {
  double rtol = 1e-5, atol = 0.;
  int64_t max_iter = 1000;
  c10::optional<at::Tensor> initial;
};
struct MINRESResult {
  at::Tensor solution, residual_norm, iterations, converged, status;
};
// Real symmetric, possibly indefinite square maps. Short Lanczos/QR recurrence,
// actual residual verification, independent batch/RHS and inference only.
TORCHPAIRWISE_API MINRESResult minres(const LinearOperator& op,
                                      const at::Tensor& rhs,
                                      const MINRESOptions& options = {});
}  // namespace torchpairwise::ops
