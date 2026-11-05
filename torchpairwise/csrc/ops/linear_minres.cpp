#include "linear_minres.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "linear_plan_detail.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
using namespace detail::linear;
using Result = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;

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
             const c10::optional<Tensor>& initial, double rtol, double atol,
             int64_t max_iter) {
  bool vector = rhs.dim() == 1;
  auto block = vector ? rhs.unsqueeze(-1) : rhs;
  const auto& shape = a.shapes.back();
  TORCH_CHECK(shape.back() == shape[shape.size() - 2],
              "MINRES requires a square operator");
  TORCH_CHECK(block.size(-2) == shape.back(),
              "MINRES RHS sample count mismatch");
  auto batch = at::infer_size_symint(
      c10::SymIntArrayRef(shape).slice(0, shape.size() - 2),
      block.sym_sizes().slice(0, block.dim() - 2));
  auto output = batch;
  output.insert(output.end(), {shape.back(), block.sym_size(-1)});
  auto b = block.expand_symint(output);
  auto x = at::zeros_symint(output, b.options());
  if (initial) {
    TORCH_CHECK(initial->layout() == at::kStrided &&
                    initial->scalar_type() == rhs.scalar_type() &&
                    initial->device() == rhs.device(),
                "MINRES initial must match RHS dtype/device");
    auto value = vector ? initial->unsqueeze(-1) : *initial;
    TORCH_CHECK((!vector || initial->dim() >= 1) && value.dim() >= 2 &&
                    value.size(-2) == b.size(-2) &&
                    value.size(-1) == b.size(-1),
                "MINRES initial must match solution sample/channel dimensions");
    TORCH_CHECK(at::infer_size_symint(value.sym_sizes(), output) == output,
                "MINRES initial may broadcast to the solution but cannot add "
                "batch dimensions");
    x = value.expand_symint(output).clone();
  }
  auto r = b - a(x);
  auto residual = column_norm(r),
       threshold = (rtol * column_norm(b)).clamp_min(atol);
  auto status = at::ones_like(residual, residual.options().dtype(at::kLong));
  auto iterations = at::zeros_like(status);
  auto finite =
      residual.isfinite() & threshold.isfinite() & column_norm(x).isfinite();
  status = at::where(~finite, 3, at::where(residual.le(threshold), 0, status));
  auto active = status.eq(1) & iterations.lt(max_iter);
  x = x.to(at::kDouble);
  double epsilon = rhs.scalar_type() == at::kDouble
                       ? std::numeric_limits<double>::epsilon()
                       : std::numeric_limits<float>::epsilon();
  while (active.any().item<bool>()) {
    auto running = active.clone();
    auto r1 = select_columns(running, r.to(at::kDouble),
                             at::zeros_like(r, r.options().dtype(at::kDouble)));
    auto r2 = r1.clone(), beta = column_norm(r2);
    auto old_beta = at::ones_like(beta), dbar = at::zeros_like(beta),
         epsln = at::zeros_like(beta), phibar = beta.clone(),
         cosine = -at::ones_like(beta), sine = at::zeros_like(beta);
    auto w = at::zeros_like(x), w2 = at::zeros_like(x);
    int64_t step = 0;
    while (running.any().item<bool>()) {
      iterations.add_(running.to(at::kLong));
      auto v = divide_columns(r2, beta);
      auto y = a(v.to(rhs.scalar_type())).to(at::kDouble);
      auto action_norm = column_norm(y);
      status = at::where(running & ~action_norm.isfinite(), 3, status);
      running = running & status.eq(1);
      y = select_columns(running, y, at::zeros_like(y));
      if (step > 0) {
        auto ratio =
            beta / at::where(old_beta.gt(0), old_beta, at::ones_like(old_beta));
        y = y - r1 * ratio.unsqueeze(-2);
      }
      auto alpha = (v * y).sum(-2);
      y = y - r2 * (alpha / at::where(beta.gt(0), beta, at::ones_like(beta)))
                       .unsqueeze(-2);
      r1 = r2;
      r2 = y;
      old_beta = beta;
      beta = column_norm(r2);
      auto old_eps = epsln;
      auto delta = cosine * dbar + sine * alpha;
      auto gbar = sine * dbar - cosine * alpha;
      epsln = sine * beta;
      dbar = -cosine * beta;
      auto gamma = at::hypot(gbar, beta);
      auto good = gamma.isfinite() & gamma.gt(0);
      auto denominator = at::where(good, gamma, at::ones_like(gamma));
      cosine = at::where(good, gbar / denominator, 1.);
      sine = at::where(good, beta / denominator, 0.);
      auto phi = cosine * phibar;
      phibar = sine * phibar;
      auto next_w = (v - old_eps.unsqueeze(-2) * w2 - delta.unsqueeze(-2) * w) /
                    denominator.unsqueeze(-2);
      auto next_x = x + phi.unsqueeze(-2) * next_w;
      auto finite_update = column_norm(next_x).isfinite() &
                           column_norm(next_w).isfinite() & gamma.isfinite() &
                           beta.isfinite() & phibar.isfinite();
      status = at::where(running & ~finite_update, 3, status);
      running = running & status.eq(1);
      x = select_columns(running & good, next_x, x);
      w2 = w;
      w = select_columns(running & good, next_w, at::zeros_like(next_w));
      auto breakdown = ~good | beta.le(8. * epsilon * action_norm);
      auto candidate = running & (phibar.abs().le(threshold) | breakdown |
                                  iterations.ge(max_iter));
      if (candidate.any().item<bool>()) {
        // The actual output-precision point, not the higher-precision latent
        // recurrence, must solve the requested system before reporting success.
        auto returned = x.to(rhs.scalar_type());
        auto actual = b - a(returned), actual_norm = column_norm(actual);
        status = at::where(candidate & ~actual_norm.isfinite(), 3, status);
        status = at::where(candidate & status.ne(3) & actual_norm.le(threshold),
                           0, status);
        status = at::where(candidate & status.eq(1) & breakdown, 2, status);
        r = select_columns(candidate, actual, r);
        x = select_columns(candidate, returned.to(at::kDouble), x);
        // A false estimated convergence restarts only those unfinished RHS
        // recurrences from their actual residual, within the total step budget.
        running = running & ~candidate;
      }
      r2 = select_columns(running, r2, at::zeros_like(r2));
      ++step;
    }
    active = status.eq(1) & iterations.lt(max_iter);
  }
  x = x.to(rhs.scalar_type());
  residual = column_norm(b - a(x));
  status = at::where(~residual.isfinite(), 3, status);
  auto converged = residual.isfinite() & threshold.isfinite() &
                   residual.le(threshold) & status.ne(3);
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

Result linear_minres_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                          c10::SymIntArrayRef shape, const Tensor& rhs,
                          double rtol, double atol, int64_t max_iter,
                          const c10::optional<Tensor>& initial) {
  TORCH_CHECK(
      rhs.layout() == at::kStrided && rhs.dim() >= 1 &&
          (rhs.scalar_type() == at::kFloat || rhs.scalar_type() == at::kDouble),
      "MINRES RHS must be dense float32/float64 with shape [N] or [...,N,R]");
  TORCH_CHECK(rhs.dim() == 1 || rhs.size(-1) > 0,
              "MINRES RHS channel count must be positive");
  TORCH_CHECK(std::isfinite(rtol) && rtol >= 0 && std::isfinite(atol) &&
                  atol >= 0 && max_iter >= 0,
              "MINRES tolerances must be finite/nonnegative and max_iter "
              "nonnegative");
  if (rhs.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(rhs.device().type())
                     .getStream(rhs.device())
                     .is_capturing(),
                "MINRES uses native early stopping and does not support CUDA "
                "graph capture");
  if (at::GradMode::is_enabled()) {
    auto check = [](const Tensor& t) {
      TORCH_CHECK(!t.requires_grad(),
                  "MINRES is inference-only; use torch.no_grad() or detach "
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
              "Bound MINRES source shape changed; reconstruct the operator");
  return solve(plan, rhs, initial, rtol, atol, max_iter);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "linear_minres(Tensor[] inputs, int[] counts, str[] metrics, str[] "
      "families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, "
      "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "
      "real_lists, "
      "Dict(str,int[]) integer_lists, bool[] transposed, int[] source_counts, "
      "Dict(str,int[]) programs, Tensor[] factors, int[] factor_counts, "
      "Dict(str,int[]) kinds, Dict(str,int[]) integer_numbers, "
      "Dict(str,float[]) real_numbers, int[] linear_program, "
      "Tensor[] linear_factors, SymInt[] shape, Tensor rhs, "
      "float rtol=1e-5, float atol=0., int max_iter=1000, "
      "Tensor? "
      "initial=None) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_minres", TORCH_FN(linear_minres_impl));
}
}  // namespace
MINRESResult minres(const LinearOperator& op, const Tensor& rhs,
                    const MINRESOptions& options) {
  TORCH_CHECK(op.dtype() == rhs.scalar_type() && op.device() == rhs.device(),
              "MINRES operator and RHS must share dtype/device");
  auto p = pack_graph(op.graph());
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_minres", "")
          .typed<decltype(linear_minres_impl)>();
  auto result = handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                            rhs, options.rtol, options.atol, options.max_iter,
                            options.initial);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result), std::get<4>(result)};
}
}  // namespace torchpairwise::ops
