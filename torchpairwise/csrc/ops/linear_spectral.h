#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
struct TraceOptions {
  std::string method;  // Required: "exact" or "hutchinson".
  c10::optional<int64_t> num_probes, seed;
  c10::optional<at::Tensor> probes;  // [...,N,P], same dtype/device as A.
  int64_t block_size = 16;
  bool differentiable = false;
};
struct TraceResult {
  at::Tensor estimate, standard_error, samples;
};
struct LogdetOptions : TraceOptions {
  // method: "exact" (materializes A) or "slq" (matrix-free SPD Lanczos).
  c10::optional<int64_t> max_iter;  // SLQ only, defaults to 32, capped by N.
  // Differentiable SLQ requires explicit "solve": a stochastic gradient of
  // logdet(A), not the derivative of the finite Lanczos approximation.
  c10::optional<std::string> gradient;
  c10::optional<double> solve_rtol, solve_atol;
  c10::optional<int64_t> solve_max_iter;
};
struct LogdetResult {
  at::Tensor estimate, standard_error, samples, iterations;
};
struct MatrixFunctionOptions {
  int64_t max_iter = 32, block_size = 16;
};
struct MatrixFunctionResult {
  at::Tensor action, error_estimate, iterations;
};
// Float32/64 estimates; differentiable=true enables first estimate gradients.
// Seeded probes use a private generator;
// explicit probes disable internal randomness. Standard errors describe probe
// variation, not quadrature error. Symmetry/SPD remain caller assumptions.
TORCHPAIRWISE_API TraceResult trace(const LinearOperator& op,
                                    const TraceOptions& options);
TORCHPAIRWISE_API LogdetResult logdet(const LinearOperator& op,
                                      const LogdetOptions& options);
// Symmetric exp(scale*A) V and SPD A^{-1/2} V. No matrix function is stored.
// Adjacent Krylov projection differences are heuristic error indicators.
TORCHPAIRWISE_API MatrixFunctionResult
expm_multiply(const LinearOperator& op, const at::Tensor& values,
              double scale = 1., const MatrixFunctionOptions& options = {});
TORCHPAIRWISE_API MatrixFunctionResult
inv_sqrt_multiply(const LinearOperator& op, const at::Tensor& values,
                  const MatrixFunctionOptions& options = {});
}  // namespace torchpairwise::ops
