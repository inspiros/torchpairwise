#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
enum class GMRESStatus : int64_t {
  Converged = 0,
  IterationLimit = 1,
  Breakdown = 2,
  Nonfinite = 3
};
struct GMRESOptions {
  double rtol = 1e-5, atol = 0.;
  int64_t restart = 20, max_iter = 1000;
  c10::optional<at::Tensor> initial;
};
struct GMRESResult {
  at::Tensor solution, residual_norm, iterations, converged, status;
};
// Restarted real nonsymmetric square solves, independent batch/RHS recurrences.
// max_iter counts Arnoldi steps across all restarts. Inference only.
TORCHPAIRWISE_API GMRESResult gmres(const LinearOperator& op,
                                    const at::Tensor& rhs,
                                    const GMRESOptions& options = {});
}  // namespace torchpairwise::ops
