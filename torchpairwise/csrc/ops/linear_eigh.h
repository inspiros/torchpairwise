#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
enum class EigenStatus : int64_t {
  Converged = 0,
  IterationLimit = 1,
  RankDeficientInitial = 2,
  Nonfinite = 3,
  NonPSD = 4,
  LossOfOrthogonality = 5
};
struct EighOptions {
  int64_t k = 0;  // Required: set explicitly.
  c10::optional<int64_t> subspace_size;
  int64_t max_iter = 100;
  double rtol = 1e-5, atol = 0.;
  c10::optional<at::Tensor> initial;
};
struct EigenResult {
  at::Tensor eigenvalues, eigenvectors, residual_norm, iterations, converged,
      status;
};
// Leading eigenpairs only, for a real symmetric PSD map. Inference-only block
// subspace iteration; residual convergence does not certify global ordering
// when a starting subspace misses an invariant leading direction.
TORCHPAIRWISE_API EigenResult eigh(const LinearOperator& op,
                                   const EighOptions& options = {});
}  // namespace torchpairwise::ops
