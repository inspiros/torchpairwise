#include "linear_solve.h"

#include "linear_solve_detail.h"

namespace torchpairwise::ops {
namespace {
template <typename Result>
SolveResult common_result(const Result& result) {
  return {result.solution, result.residual_norm, result.iterations,
          result.converged, result.status};
}
}  // namespace
SolveResult solve(const LinearOperator& op, const at::Tensor& rhs,
                  const SolveOptions& options) {
  TORCH_CHECK(
      options.method == "cg" || options.method == "minres" ||
          options.method == "gmres",
      "solve method must be explicitly selected from 'cg', 'minres', 'gmres'");
  TORCH_CHECK(!options.preconditioner || options.method == "cg",
              "solve preconditioner is supported only with method='cg'");
  TORCH_CHECK(!options.restart || options.method == "gmres",
              "solve restart is supported only with method='gmres'");
  if (options.differentiable) {
    TORCH_CHECK(op.dtype() == rhs.scalar_type() && op.device() == rhs.device(),
                "Implicit solve operator and RHS must share dtype/device");
    auto p = detail::linear::pack_graph(
        (options.preconditioner ? op.compose(*options.preconditioner) : op)
            .graph());
    static auto handle =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_implicit_solve", "")
            .typed<detail::linear::ImplicitSolveSignature>();
    static auto pcg =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_implicit_pcg", "")
            .typed<detail::linear::ImplicitSolveSignature>();
    auto result =
        (options.preconditioner ? pcg : handle)
            .call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(), rhs,
                  options.method, options.rtol, options.atol, options.max_iter,
                  options.restart.value_or(20), options.initial);
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result), std::get<4>(result)};
  }
  if (options.method == "cg") {
    CGOptions selected;
    selected.rtol = options.rtol;
    selected.atol = options.atol;
    selected.max_iter = options.max_iter;
    selected.initial = options.initial;
    selected.preconditioner = options.preconditioner;
    return common_result(cg(op, rhs, selected));
  }
  if (options.method == "minres") {
    MINRESOptions selected;
    selected.rtol = options.rtol;
    selected.atol = options.atol;
    selected.max_iter = options.max_iter;
    selected.initial = options.initial;
    return common_result(minres(op, rhs, selected));
  }
  GMRESOptions selected;
  selected.rtol = options.rtol;
  selected.atol = options.atol;
  selected.max_iter = options.max_iter;
  selected.initial = options.initial;
  selected.restart = options.restart.value_or(20);
  return common_result(gmres(op, rhs, selected));
}
LSQRResult lstsq(const LinearOperator& op, const at::Tensor& rhs,
                 const LstsqOptions& options) {
  TORCH_CHECK(options.method == "lsqr", "lstsq method must be 'lsqr'");
  if (options.differentiable) {
    TORCH_CHECK(op.dtype() == rhs.scalar_type() && op.device() == rhs.device(),
                "Implicit lstsq operator and RHS must share dtype/device");
    auto p = detail::linear::pack_graph(op.graph());
    static auto handle =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::linear_implicit_lstsq", "")
            .typed<detail::linear::ImplicitLstsqSignature>();
    auto result = handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                              rhs, options.damp, options.rtol, options.atol,
                              options.max_iter, options.initial);
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result), std::get<4>(result), std::get<5>(result)};
  }
  LSQROptions selected;
  selected.damp = options.damp;
  selected.rtol = options.rtol;
  selected.atol = options.atol;
  selected.max_iter = options.max_iter;
  selected.initial = options.initial;
  return lsqr(op, rhs, selected);
}
}  // namespace torchpairwise::ops
