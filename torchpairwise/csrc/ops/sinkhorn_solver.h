#pragma once
#include "expression_operations.h"

namespace torchpairwise::ops {
enum class SinkhornStatus : int64_t {
  Converged = 0,
  IterationLimit = 1,
  Nonfinite = 2
};
struct SinkhornOptions {
  double epsilon = 1., rtol = 1e-6, atol = 0.;
  int64_t max_iter = 1000, check_every = 10;
  c10::optional<double> reach, epsilon_scaling;
  c10::optional<at::Tensor> initial_f, initial_g;
};
struct SinkhornResult {
  at::Tensor f, g, objective, iterations, converged, status, marginal_error;
};
TORCHPAIRWISE_API SinkhornResult sinkhorn(const ScalarExpression& cost,
                                          const at::Tensor& a,
                                          const at::Tensor& b,
                                          const SinkhornOptions& options = {});
TORCHPAIRWISE_API ScalarExpression sinkhorn_plan(const ScalarExpression& cost,
                                                 const SinkhornResult& result,
                                                 const at::Tensor& a,
                                                 const at::Tensor& b,
                                                 double epsilon);
}  // namespace torchpairwise::ops
