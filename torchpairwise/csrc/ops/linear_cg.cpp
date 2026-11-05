#include "linear_cg.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <cmath>

#include "linear_plan_detail.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
using namespace detail::linear;
using Result = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;

Tensor column_norm(const Tensor& v) {
  return at::linalg_vector_norm(v.to(at::kDouble), 2., {-2}, false);
}
Tensor column_dot(const Tensor& a, const Tensor& b) {
  return (a.to(at::kDouble) * b.to(at::kDouble)).sum(-2);
}
Tensor select_columns(const Tensor& mask, const Tensor& yes, const Tensor& no) {
  return at::where(mask.unsqueeze(-2), yes, no);
}
Result solve(const Plan& a, int64_t root, int64_t inverse_root,
             const Tensor& rhs, const c10::optional<Tensor>& initial,
             double rtol, double atol, int64_t max_iter) {
  bool vector = rhs.dim() == 1;
  auto block = vector ? rhs.unsqueeze(-1) : rhs;
  const auto& shape = a.shapes[root];
  auto action = [&](const Tensor& value) { return a.apply(root, value); };
  TORCH_CHECK(shape.back() == shape[shape.size() - 2],
              "CG requires a square operator");
  TORCH_CHECK(block.size(-2) == shape.back(), "CG RHS sample count mismatch");
  auto batch = at::infer_size_symint(
      c10::SymIntArrayRef(shape).slice(0, shape.size() - 2),
      block.sym_sizes().slice(0, block.dim() - 2));
  auto output = batch;
  output.insert(output.end(), {shape.back(), block.sym_size(-1)});
  if (inverse_root >= 0) {
    const auto& mshape = a.shapes[inverse_root];
    TORCH_CHECK(mshape.back() == shape.back() &&
                    mshape[mshape.size() - 2] == shape.back(),
                "CG preconditioner must match the square system dimensions");
    TORCH_CHECK(at::infer_size_symint(
                    c10::SymIntArrayRef(mshape).slice(0, mshape.size() - 2),
                    batch) == batch,
                "CG preconditioner may broadcast to the solution but cannot "
                "add batch dimensions");
  }
  auto b = block.expand_symint(output);
  auto x = at::zeros_symint(output, b.options());
  if (initial) {
    TORCH_CHECK(initial->layout() == at::kStrided &&
                    initial->scalar_type() == rhs.scalar_type() &&
                    initial->device() == rhs.device(),
                "CG initial must match RHS dtype/device");
    auto value = vector ? initial->unsqueeze(-1) : *initial;
    TORCH_CHECK((!vector || initial->dim() >= 1) && value.dim() >= 2 &&
                    value.size(-2) == b.size(-2) &&
                    value.size(-1) == b.size(-1),
                "CG initial must match solution sample/channel dimensions");
    TORCH_CHECK(at::infer_size_symint(value.sym_sizes(), output) == output,
                "CG initial may broadcast to the solution but cannot add batch "
                "dimensions");
    x = value.expand_symint(output).clone();
  }
  auto r = b - action(x);
  auto residual = column_norm(r),
       threshold = (rtol * column_norm(b)).clamp_min(atol);
  auto status = at::ones_like(residual, residual.options().dtype(at::kLong));
  auto iterations = at::zeros_like(status);
  auto finite =
      residual.isfinite() & threshold.isfinite() & column_norm(x).isfinite();
  status = at::where(~finite, 3, at::where(residual <= threshold, 0, status));
  auto active = status.eq(1);
  auto precondition = [&](const Tensor& residual) {
    auto masked = select_columns(active, residual, at::zeros_like(residual));
    return inverse_root >= 0 && max_iter > 0 && active.any().item<bool>()
               ? a.apply(inverse_root, masked)
               : masked;
  };
  auto check_preconditioner = [&](const Tensor& z, const Tensor& rho) {
    if (inverse_root < 0 || max_iter == 0)
      return;
    auto finite = column_norm(z).isfinite() & rho.isfinite();
    status = at::where(active & ~finite, 3, status);
    status = at::where(active & finite & rho.le(0), 4, status);
    active = status.eq(1);
  };
  auto z = precondition(r);
  auto rho = column_dot(r, z);
  check_preconditioner(z, rho);
  auto p = select_columns(active, z, at::zeros_like(r));
  for (int64_t step = 0; step < max_iter && active.any().item<bool>(); ++step) {
    iterations.add_(active.to(at::kLong));
    auto ap = action(p), curvature = column_dot(p, ap);
    auto finite_step =
        column_norm(ap).isfinite() & curvature.isfinite() & rho.isfinite();
    status = at::where(active & ~finite_step, 3, status);
    status = at::where(active & finite_step & curvature.le(0), 2, status);
    active = status.eq(1);
    auto alpha = at::where(
        active, rho / at::where(active, curvature, at::ones_like(curvature)),
        0.);
    auto next_x = x + p * alpha.to(x.scalar_type()).unsqueeze(-2);
    auto next_r = r - ap * alpha.to(r.scalar_type()).unsqueeze(-2);
    auto finite_update =
        column_norm(next_x).isfinite() & column_norm(next_r).isfinite();
    status = at::where(active & ~finite_update, 3, status);
    active = status.eq(1);
    x = select_columns(active, next_x, x);
    r = select_columns(active, next_r, r);
    auto candidate = active & column_norm(r).le(threshold);
    if (candidate.any().item<bool>()) {
      auto actual = b - action(x), actual_norm = column_norm(actual);
      status = at::where(candidate & ~actual_norm.isfinite(), 3, status);
      status = at::where(candidate & actual_norm.le(threshold), 0, status);
      // Restart candidates whose recurrence residual drifted below tolerance.
      r = select_columns(candidate, actual, r);
    }
    active = status.eq(1);
    z = precondition(r);
    auto next_rho = column_dot(r, z);
    check_preconditioner(z, next_rho);
    auto beta =
        at::where(active & ~candidate,
                  next_rho / at::where(active, rho, at::ones_like(rho)), 0.);
    p = select_columns(active, z + p * beta.to(p.scalar_type()).unsqueeze(-2),
                       at::zeros_like(p));
    rho = next_rho;
  }
  residual = column_norm(b - action(x));
  status = at::where(~residual.isfinite(), 3, status);
  auto converged = residual.isfinite() & residual.le(threshold) & status.ne(3);
  status = at::where(converged, 0, at::where(status.eq(0), 1, status));
  if (vector) {
    x = x.squeeze(-1);
    residual = residual.squeeze(-1);
    iterations = iterations.squeeze(-1);
    converged = converged.squeeze(-1);
    status = status.squeeze(-1);
  }
  return {x, residual, iterations, converged, status};
}

Result linear_cg_dispatch(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                          c10::SymIntArrayRef shape, const Tensor& rhs,
                          double rtol, double atol, int64_t max_iter,
                          const c10::optional<Tensor>& initial,
                          bool preconditioned) {
  TORCH_CHECK(
      rhs.layout() == at::kStrided && rhs.dim() >= 1 &&
          (rhs.scalar_type() == at::kFloat || rhs.scalar_type() == at::kDouble),
      "CG RHS must be dense float32/float64 with shape [N] or [...,N,R]");
  TORCH_CHECK(rhs.dim() == 1 || rhs.size(-1) > 0,
              "CG RHS channel count must be positive");
  TORCH_CHECK(
      std::isfinite(rtol) && rtol >= 0 && std::isfinite(atol) && atol >= 0 &&
          max_iter >= 0,
      "CG tolerances must be finite/nonnegative and max_iter nonnegative");
  if (rhs.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(rhs.device().type())
                     .getStream(rhs.device())
                     .is_capturing(),
                "CG uses native early stopping and does not support CUDA graph "
                "capture");
  if (at::GradMode::is_enabled()) {
    auto check = [](const Tensor& t) {
      TORCH_CHECK(!t.requires_grad(),
                  "CG is inference-only; use torch.no_grad() or detach inputs "
                  "explicitly");
    };
    for (const auto& t : inputs)
      check(t);
    for (const auto& t : factors)
      check(t);
    for (const auto& t : linear_factors)
      check(t);
    for (const auto& item : tensors)
      check(item.value());
    check(rhs);
    if (initial)
      check(*initial);
  }
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto graph = unpack_graph(inputs, counts, metrics, families, tensors, reals,
                            integers, flags, strings, real_lists, integer_lists,
                            transposed, source_counts, programs, factors,
                            factor_counts, kinds, integer_numbers, real_numbers,
                            linear_program, linear_factors);
  Plan plan(std::move(graph), rhs);
  auto root = int64_t(plan.graph.nodes.size()) - 1;
  int64_t inverse_root = -1;
  if (preconditioned) {
    // A final composition is a packet container for two shared subgraphs.
    // The recurrence applies A and M separately; it never runs CG on A*M.
    const auto node = plan.graph.nodes.back();
    TORCH_CHECK(node[0] == 3,
                "PCG packet must contain system and inverse-action roots");
    root = node[1];
    inverse_root = node[2];
  }
  TORCH_CHECK(c10::SymIntArrayRef(plan.shapes[root]).equals(shape),
              "Bound CG source shape changed; reconstruct the operator");
  return solve(plan, root, inverse_root, rhs, initial, rtol, atol, max_iter);
}
Result linear_cg_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                      c10::SymIntArrayRef shape, const Tensor& rhs, double rtol,
                      double atol, int64_t max_iter,
                      const c10::optional<Tensor>& initial) {
  return linear_cg_dispatch(TORCHPAIRWISE_LINEAR_PACKET_ARGS, shape, rhs, rtol,
                            atol, max_iter, initial, false);
}
Result linear_pcg_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                       c10::SymIntArrayRef shape, const Tensor& rhs,
                       double rtol, double atol, int64_t max_iter,
                       const c10::optional<Tensor>& initial) {
  return linear_cg_dispatch(TORCHPAIRWISE_LINEAR_PACKET_ARGS, shape, rhs, rtol,
                            atol, max_iter, initial, true);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "linear_cg(Tensor[] inputs, int[] counts, str[] metrics, str[] families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, "
      "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "
      "real_lists, "
      "Dict(str,int[]) integer_lists, bool[] transposed, int[] source_counts, "
      "Dict(str,int[]) programs, "
      "Tensor[] factors, int[] factor_counts, Dict(str,int[]) kinds, "
      "Dict(str,int[]) "
      "integer_numbers, Dict(str,float[]) real_numbers, "
      "int[] linear_program, Tensor[] linear_factors, SymInt[] shape, Tensor "
      "rhs, "
      "float rtol=1e-5, float atol=0., int max_iter=1000, Tensor? "
      "initial=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "linear_pcg(Tensor[] inputs, int[] counts, str[] metrics, str[] "
      "families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, "
      "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "
      "real_lists, "
      "Dict(str,int[]) integer_lists, bool[] transposed, int[] source_counts, "
      "Dict(str,int[]) programs, "
      "Tensor[] factors, int[] factor_counts, Dict(str,int[]) kinds, "
      "Dict(str,int[]) "
      "integer_numbers, Dict(str,float[]) real_numbers, "
      "int[] linear_program, Tensor[] linear_factors, SymInt[] shape, Tensor "
      "rhs, "
      "float rtol=1e-5, float atol=0., int max_iter=1000, Tensor? "
      "initial=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_cg", TORCH_FN(linear_cg_impl));
  m.impl("linear_pcg", TORCH_FN(linear_pcg_impl));
}
}  // namespace
CGResult cg(const LinearOperator& op, const Tensor& rhs,
            const CGOptions& options) {
  TORCH_CHECK(op.dtype() == rhs.scalar_type() && op.device() == rhs.device(),
              "CG operator and RHS must share dtype/device");
  auto graph = op;
  if (options.preconditioner) {
    const auto& m = *options.preconditioner;
    TORCH_CHECK(m.shape()[m.shape().size() - 2] == op.shape().back() &&
                    m.shape().back() == op.shape().back(),
                "CG preconditioner must match the square system dimensions");
    graph = op.compose(m);
  }
  auto p = pack_graph(graph.graph());
  static auto handle = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("torchpairwise::linear_cg", "")
                           .typed<decltype(linear_cg_impl)>();
  static auto pcg_handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_pcg", "")
          .typed<decltype(linear_pcg_impl)>();
  auto result =
      (options.preconditioner ? pcg_handle : handle)
          .call(p.inputs, p.counts, p.metrics, p.families, p.tensors, p.reals,
                p.integers, p.flags, p.strings, p.real_lists, p.integer_lists,
                p.transposed, p.source_counts, p.programs, p.factors,
                p.factor_counts, p.kinds, p.integer_numbers, p.real_numbers,
                p.linear_program, p.linear_factors, op.shape(), rhs,
                options.rtol, options.atol, options.max_iter, options.initial);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result), std::get<4>(result)};
}
}  // namespace torchpairwise::ops
