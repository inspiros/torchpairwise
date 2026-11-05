#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
enum class QPStatus : int64_t {
  Converged = 0,
  IterationLimit = 1,
  Infeasible = 2,
  Nonfinite = 3,
  Nonconvex = 4,
  ProjectionFailed = 5,
  Stalled = 6
};
struct QPEquality {
  at::Tensor vector, target;
};
struct BoxQPOptions {
  at::Tensor lower, upper;
  c10::optional<QPEquality> equality;
  c10::optional<at::Tensor> initial;
  double rtol = 1e-5, atol = 0.;
  int64_t max_iter = 1000;
};
struct QPResult {
  at::Tensor solution, objective, iterations, converged, status, optimality,
      feasibility;
};
// Real symmetric convex quadratic, finite box, optional one affine equality.
// q [N] or [...,N,R] defines independent objectives; numerical execution is
// native.
TORCHPAIRWISE_API QPResult box_qp(const LinearOperator& Q, const at::Tensor& q,
                                  const BoxQPOptions& options);
}  // namespace torchpairwise::ops
