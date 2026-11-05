#include "linear_lsqr.h"

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
using Result = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;

Tensor column_norm(const Tensor& v) {
  return at::linalg_vector_norm(v.to(at::kDouble), 2., {-2}, false);
}
Tensor select_columns(const Tensor& mask, const Tensor& yes, const Tensor& no) {
  return at::where(mask.unsqueeze(-2), yes, no);
}
Tensor divide_columns(const Tensor& value, const Tensor& divisor) {
  auto safe = at::where(divisor.gt(0), divisor, at::ones_like(divisor));
  return (value.to(at::kDouble) / safe.unsqueeze(-2)).to(value.scalar_type());
}

Result solve(const Plan& a, const Tensor& rhs,
             const c10::optional<Tensor>& initial, double damp, double rtol,
             double atol, int64_t max_iter) {
  bool vector = rhs.dim() == 1;
  auto block = vector ? rhs.unsqueeze(-1) : rhs;
  const auto& shape = a.shapes.back();
  auto rows = shape[shape.size() - 2].expect_int();
  auto columns = shape.back().expect_int();
  TORCH_CHECK(block.size(-2) == rows, "LSQR RHS sample count mismatch");
  auto batch = at::infer_size_symint(
      c10::SymIntArrayRef(shape).slice(0, shape.size() - 2),
      block.sym_sizes().slice(0, block.dim() - 2));
  auto data_shape = batch, solution_shape = batch;
  data_shape.insert(data_shape.end(), {rows, block.sym_size(-1)});
  solution_shape.insert(solution_shape.end(), {columns, block.sym_size(-1)});
  auto b = block.expand_symint(data_shape);
  auto x = at::zeros_symint(solution_shape, b.options());
  if (initial) {
    TORCH_CHECK(initial->layout() == at::kStrided &&
                    initial->scalar_type() == rhs.scalar_type() &&
                    initial->device() == rhs.device(),
                "LSQR initial must match RHS dtype/device");
    auto value = vector ? initial->unsqueeze(-1) : *initial;
    TORCH_CHECK((!vector || initial->dim() >= 1) && value.dim() >= 2 &&
                    value.size(-2) == columns && value.size(-1) == b.size(-1),
                "LSQR initial must match solution sample/channel dimensions");
    TORCH_CHECK(at::infer_size_symint(value.sym_sizes(), solution_shape) ==
                    solution_shape,
                "LSQR initial may broadcast to the solution but cannot add "
                "batch dimensions");
    x = value.expand_symint(solution_shape).clone();
  }
  auto transpose = [&](const Tensor& v) {
    return a.apply(a.graph.nodes.size() - 1, v, true);
  };
  // LSQR on B=[A;damp I]. The regularizer is anchored at zero even when an
  // initial guess is supplied; its correction RHS is [b-Ax0;-damp*x0].
  auto forward_augmented = [&](const Tensor& v) {
    auto product = a(v);
    return damp == 0. ? product : at::cat({product, damp * v}, -2);
  };
  auto transpose_augmented = [&](const Tensor& v) {
    if (damp == 0.)
      return transpose(v);
    return transpose(v.narrow(-2, 0, rows)) +
           damp * v.narrow(-2, rows, columns);
  };
  auto gradient = [&](const Tensor& r) {
    // Keep the normal-residual subtraction in double without changing the
    // precision used by source products or the returned solution.
    return transpose(r).to(at::kDouble) - (x.to(at::kDouble) * damp) * damp;
  };
  auto normal_rhs = column_norm(transpose(b));
  // With positive damping the minimizer is uniquely zero when A^T b is zero.
  // This also handles empty data rows exactly, instead of asking a recurrence
  // to meet a zero absolute/relative threshold after cancellation roundoff.
  // A zero update budget must still evaluate the caller's original guess.
  if (damp > 0. && max_iter > 0) {
    auto exact_zero = normal_rhs.eq(0) & column_norm(b).isfinite() &
                      column_norm(x).isfinite();
    x = select_columns(exact_zero, at::zeros_like(x), x);
  }
  auto r = b - a(x);
  auto residual = column_norm(r), normal = column_norm(gradient(r));
  auto threshold = (rtol * normal_rhs).clamp_min(atol);
  auto status = at::ones_like(normal, normal.options().dtype(at::kLong));
  auto iterations = at::zeros_like(status);
  auto finite = residual.isfinite() & normal.isfinite() & threshold.isfinite() &
                column_norm(x).isfinite();
  status = at::where(~finite, 3, at::where(normal.le(threshold), 0, status));
  auto active = status.eq(1);
  auto u = damp == 0. ? r : at::cat({r, -damp * x}, -2);
  u = select_columns(active, u, at::zeros_like(u));
  auto beta = column_norm(u);
  u = divide_columns(u, beta);
  auto v = transpose_augmented(u);
  auto alpha = column_norm(v);
  v = divide_columns(v, alpha);
  auto w = v.clone(), rhobar = alpha, phibar = beta;
  for (int64_t step = 0; step < max_iter && active.any().item<bool>(); ++step) {
    iterations.add_(active.to(at::kLong));
    auto next_u =
        forward_augmented(v) - u * alpha.to(u.scalar_type()).unsqueeze(-2);
    beta = column_norm(next_u);
    next_u = divide_columns(next_u, beta);
    auto next_v = transpose_augmented(next_u) -
                  v * beta.to(v.scalar_type()).unsqueeze(-2);
    alpha = column_norm(next_v);
    next_v = divide_columns(next_v, alpha);
    auto rho = at::hypot(rhobar, beta);
    auto finite_step =
        beta.isfinite() & alpha.isfinite() & rho.isfinite() & phibar.isfinite();
    status = at::where(active & ~finite_step, 3, status);
    auto zero_rho = rho.eq(0);
    auto safe_rho = at::where(rho.gt(0), rho, at::ones_like(rho));
    auto cosine = rhobar / safe_rho, sine = beta / safe_rho;
    auto theta = sine * alpha;
    rhobar = -cosine * alpha;
    auto phi = cosine * phibar;
    phibar = sine * phibar;
    auto next_x = x + w * (phi / safe_rho).to(x.scalar_type()).unsqueeze(-2);
    auto next_w =
        next_v - w * (theta / safe_rho).to(w.scalar_type()).unsqueeze(-2);
    status = at::where(active & ~column_norm(next_x).isfinite(), 3, status);
    active = status.eq(1);
    x = select_columns(active, next_x, x);
    // The recurrence estimates ||B^T(Bx-b_aug)||. Verify candidates and every
    // final/breakdown step using actual A and A^T actions; never trust drift.
    auto estimate = alpha * (sine * phi).abs();
    auto breakdown = zero_rho | alpha.eq(0) | beta.eq(0);
    auto candidate = active & (estimate.le(threshold) | breakdown);
    if (candidate.any().item<bool>()) {
      r = b - a(x);
      residual = column_norm(r);
      normal = column_norm(gradient(r));
      auto valid = residual.isfinite() & normal.isfinite();
      status = at::where(candidate & ~valid, 3, status);
      status = at::where(candidate & valid & normal.le(threshold), 0, status);
      status = at::where(candidate & breakdown & status.eq(1), 2, status);
    }
    active = status.eq(1);
    u = select_columns(active, next_u, at::zeros_like(next_u));
    v = select_columns(active, next_v, at::zeros_like(next_v));
    w = select_columns(active, next_w, at::zeros_like(next_w));
  }
  r = b - a(x);
  residual = column_norm(r);
  normal = column_norm(gradient(r));
  finite = residual.isfinite() & normal.isfinite() & column_norm(x).isfinite();
  status = at::where(~finite, 3, status);
  auto converged =
      finite & threshold.isfinite() & normal.le(threshold) & status.ne(3);
  status = at::where(converged, 0, at::where(status.eq(0), 1, status));
  if (vector) {
    x = x.squeeze(-1);
    residual = residual.squeeze(-1);
    normal = normal.squeeze(-1);
    iterations = iterations.squeeze(-1);
    converged = converged.squeeze(-1);
    status = status.squeeze(-1);
  }
  return {x, residual, normal, iterations, converged, status};
}

Result linear_lsqr_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                        c10::SymIntArrayRef shape, const Tensor& rhs,
                        double damp, double rtol, double atol, int64_t max_iter,
                        const c10::optional<Tensor>& initial) {
  TORCH_CHECK(
      rhs.layout() == at::kStrided && rhs.dim() >= 1 &&
          (rhs.scalar_type() == at::kFloat || rhs.scalar_type() == at::kDouble),
      "LSQR RHS must be dense float32/float64 with shape [N] or [...,N,R]");
  TORCH_CHECK(rhs.dim() == 1 || rhs.size(-1) > 0,
              "LSQR RHS channel count must be positive");
  TORCH_CHECK(std::isfinite(damp) && damp >= 0 && std::isfinite(rtol) &&
                  rtol >= 0 && std::isfinite(atol) && atol >= 0 &&
                  max_iter >= 0,
              "LSQR damp/tolerances must be finite/nonnegative and max_iter "
              "nonnegative");
  if (rhs.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(rhs.device().type())
                     .getStream(rhs.device())
                     .is_capturing(),
                "LSQR uses native early stopping and does not support CUDA "
                "graph capture");
  if (at::GradMode::is_enabled()) {
    auto check = [](const Tensor& t) {
      TORCH_CHECK(!t.requires_grad(),
                  "LSQR is inference-only; use torch.no_grad() or detach "
                  "inputs explicitly");
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
  auto graph = unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  Plan plan(std::move(graph), rhs);
  TORCH_CHECK(c10::SymIntArrayRef(plan.shapes.back()).equals(shape),
              "Bound LSQR source shape changed; reconstruct the operator");
  return solve(plan, rhs, initial, damp, rtol, atol, max_iter);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "linear_lsqr(Tensor[] inputs, int[] counts, str[] metrics, str[] "
      "families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, "
      "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "
      "real_lists, "
      "Dict(str,int[]) integer_lists, bool[] transposed, int[] source_counts, "
      "Dict(str,int[]) programs, Tensor[] factors, int[] factor_counts, "
      "Dict(str,int[]) kinds, Dict(str,int[]) integer_numbers, "
      "Dict(str,float[]) real_numbers, int[] linear_program, "
      "Tensor[] linear_factors, SymInt[] shape, Tensor rhs, float damp=0., "
      "float rtol=1e-5, float atol=0., int max_iter=1000, Tensor? "
      "initial=None) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_lsqr", TORCH_FN(linear_lsqr_impl));
}
}  // namespace
LSQRResult lsqr(const LinearOperator& op, const Tensor& rhs,
                const LSQROptions& options) {
  TORCH_CHECK(op.dtype() == rhs.scalar_type() && op.device() == rhs.device(),
              "LSQR operator and RHS must share dtype/device");
  auto p = pack_graph(op.graph());
  static auto handle = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("torchpairwise::linear_lsqr", "")
                           .typed<decltype(linear_lsqr_impl)>();
  auto result = handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                            rhs, options.damp, options.rtol, options.atol,
                            options.max_iter, options.initial);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result), std::get<4>(result), std::get<5>(result)};
}
}  // namespace torchpairwise::ops
