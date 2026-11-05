#pragma once

#include "linear_cg.h"
#include "linear_gmres.h"
#include "linear_lsqr.h"
#include "linear_minres.h"

namespace torchpairwise::ops {
struct SolveOptions {
  // Required: "cg", "minres" or "gmres". No automatic method/fallback.
  std::string method;
  bool differentiable =
      false;  // Implicit first gradients; requires convergence.
  double rtol = 1e-5, atol = 0.;
  int64_t max_iter = 1000;
  c10::optional<at::Tensor> initial;
  c10::optional<LinearOperator>
      preconditioner;              // CG only: fixed SPD inverse action.
  c10::optional<int64_t> restart;  // GMRES only; omitted means 20.
};
struct SolveResult {
  at::Tensor solution, residual_norm, iterations, converged, status;
};
struct LstsqOptions {
  bool differentiable =
      false;  // Implicit first gradients, positive damping only.
  std::string method = "lsqr";
  double damp = 0., rtol = 1e-5, atol = 0.;
  int64_t max_iter = 1000;
  c10::optional<at::Tensor> initial;
};
// solve optionally differentiates the converged solution via an adjoint solve.
// lstsq optionally differentiates with positive damping. Status values
// retain the selected solver's meaning; no dense inverse/pseudoinverse is made.
TORCHPAIRWISE_API SolveResult solve(const LinearOperator& op,
                                    const at::Tensor& rhs,
                                    const SolveOptions& options);
TORCHPAIRWISE_API LSQRResult lstsq(const LinearOperator& op,
                                   const at::Tensor& rhs,
                                   const LstsqOptions& options = {});
}  // namespace torchpairwise::ops
