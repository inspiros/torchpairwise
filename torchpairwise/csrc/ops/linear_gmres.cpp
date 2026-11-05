#include "linear_gmres.h"

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
             int64_t restart, int64_t max_iter) {
  bool vector = rhs.dim() == 1;
  auto block = vector ? rhs.unsqueeze(-1) : rhs;
  const auto& shape = a.shapes.back();
  TORCH_CHECK(shape.back() == shape[shape.size() - 2],
              "GMRES requires a square operator");
  TORCH_CHECK(block.size(-2) == shape.back(),
              "GMRES RHS sample count mismatch");
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
                "GMRES initial must match RHS dtype/device");
    auto value = vector ? initial->unsqueeze(-1) : *initial;
    TORCH_CHECK((!vector || initial->dim() >= 1) && value.dim() >= 2 &&
                    value.size(-2) == b.size(-2) &&
                    value.size(-1) == b.size(-1),
                "GMRES initial must match solution sample/channel dimensions");
    TORCH_CHECK(at::infer_size_symint(value.sym_sizes(), output) == output,
                "GMRES initial may broadcast to the solution but cannot add "
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
  auto width = std::min(restart, shape.back().expect_int());
  double epsilon = rhs.scalar_type() == at::kDouble
                       ? std::numeric_limits<double>::epsilon()
                       : std::numeric_limits<float>::epsilon();
  while (active.any().item<bool>()) {
    auto running = active.clone();
    auto beta = column_norm(r);
    auto base = x.clone();
    std::vector<Tensor> basis{divide_columns(
        select_columns(running, r.to(at::kDouble),
                       at::zeros_like(r, r.options().dtype(at::kDouble))),
        beta)};
    std::vector<std::vector<Tensor>> h;
    std::vector<Tensor> cosine, sine, g{at::where(running, beta, 0.)};
    for (int64_t j = 0; j < width && running.any().item<bool>(); ++j) {
      iterations.add_(running.to(at::kLong));
      auto w = a(basis[j].to(rhs.scalar_type())).to(at::kDouble);
      auto original_norm = column_norm(w);
      status = at::where(running & ~original_norm.isfinite(), 3, status);
      running = running & status.eq(1);
      w = select_columns(running, w, at::zeros_like(w));
      std::vector<Tensor> column;
      // Two modified Gram-Schmidt passes limit loss of orthogonality without
      // retaining anything beyond the restart basis and small Hessenberg block.
      for (int64_t i = 0; i <= j; ++i) {
        auto value = (basis[i] * w).sum(-2);
        column.push_back(value);
        w = w - basis[i] * value.unsqueeze(-2);
      }
      for (int64_t i = 0; i <= j; ++i) {
        auto correction = (basis[i] * w).sum(-2);
        column[i] = column[i] + correction;
        w = w - basis[i] * correction.unsqueeze(-2);
      }
      auto next_norm = column_norm(w);
      auto breakdown = next_norm.le(8. * epsilon * original_norm);
      column.push_back(next_norm);
      basis.push_back(divide_columns(w, next_norm));
      for (int64_t i = 0; i < j; ++i) {
        auto top = cosine[i] * column[i] + sine[i] * column[i + 1];
        column[i + 1] = -sine[i] * column[i] + cosine[i] * column[i + 1];
        column[i] = top;
      }
      auto rho = at::hypot(column[j], column[j + 1]);
      auto good = rho.isfinite() & rho.gt(0);
      auto safe_rho = at::where(good, rho, at::ones_like(rho));
      cosine.push_back(at::where(good, column[j] / safe_rho, 1.));
      sine.push_back(at::where(good, column[j + 1] / safe_rho, 0.));
      column[j] = rho;
      h.push_back(std::move(column));
      auto next_g = -sine[j] * g[j];
      g[j] = cosine[j] * g[j];
      g.push_back(next_g);
      status = at::where(running & (~rho.isfinite() | ~next_norm.isfinite() |
                                    ~g[j].isfinite() | ~next_g.isfinite()),
                         3, status);
      running = running & status.eq(1);
      auto candidate =
          running & (next_g.abs().le(threshold) | breakdown | ~good |
                     iterations.ge(max_iter) | (j + 1 == width));
      if (candidate.any().item<bool>()) {
        // The rotated least-square system is upper triangular. A zero pivot
        // is a breakdown, not permission to manufacture a successful solve.
        std::vector<Tensor> y(j + 1);
        auto valid = candidate.clone();
        for (int64_t i = j; i >= 0; --i) {
          auto value = g[i].clone();
          for (int64_t k = i + 1; k <= j; ++k)
            value = value - h[k][i] * y[k];
          auto pivot = h[i][i];
          auto pivot_ok = pivot.isfinite() & pivot.ne(0);
          valid = valid & pivot_ok;
          y[i] = at::where(
              pivot_ok,
              value / at::where(pivot_ok, pivot, at::ones_like(pivot)), 0.);
        }
        auto proposal = base.to(at::kDouble);
        for (int64_t i = 0; i <= j; ++i)
          proposal = proposal + basis[i] * y[i].unsqueeze(-2);
        proposal = proposal.to(rhs.scalar_type());
        auto finite_proposal = column_norm(proposal).isfinite();
        status = at::where(candidate & ~finite_proposal, 3, status);
        valid = valid & finite_proposal;
        x = select_columns(valid, proposal, x);
        auto actual = b - a(x), actual_norm = column_norm(actual);
        status = at::where(candidate & ~actual_norm.isfinite(), 3, status);
        auto success = candidate & status.ne(3) & actual_norm.le(threshold);
        status = at::where(success, 0, status);
        status = at::where(
            candidate & status.eq(1) & (breakdown | ~good | ~valid), 2, status);
        // A false estimated convergence starts a new cycle from the actual
        // residual. Returned-precision x, not its double proposal, is checked.
        r = select_columns(candidate, actual, r);
        running = running & ~candidate;
      }
    }
    active = status.eq(1) & iterations.lt(max_iter);
  }
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

Result linear_gmres_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                         c10::SymIntArrayRef shape, const Tensor& rhs,
                         double rtol, double atol, int64_t restart,
                         int64_t max_iter,
                         const c10::optional<Tensor>& initial) {
  TORCH_CHECK(
      rhs.layout() == at::kStrided && rhs.dim() >= 1 &&
          (rhs.scalar_type() == at::kFloat || rhs.scalar_type() == at::kDouble),
      "GMRES RHS must be dense float32/float64 with shape [N] or [...,N,R]");
  TORCH_CHECK(rhs.dim() == 1 || rhs.size(-1) > 0,
              "GMRES RHS channel count must be positive");
  TORCH_CHECK(std::isfinite(rtol) && rtol >= 0 && std::isfinite(atol) &&
                  atol >= 0 && restart > 0 && max_iter >= 0,
              "GMRES tolerances must be finite/nonnegative and max_iter "
              "nonnegative and restart positive");
  if (rhs.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(rhs.device().type())
                     .getStream(rhs.device())
                     .is_capturing(),
                "GMRES uses native early stopping and does not support CUDA "
                "graph capture");
  if (at::GradMode::is_enabled()) {
    auto check = [](const Tensor& t) {
      TORCH_CHECK(!t.requires_grad(),
                  "GMRES is inference-only; use torch.no_grad() or detach "
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
              "Bound GMRES source shape changed; reconstruct the operator");
  return solve(plan, rhs, initial, rtol, atol, restart, max_iter);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "linear_gmres(Tensor[] inputs, int[] counts, str[] metrics, str[] "
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
      "float rtol=1e-5, float atol=0., int restart=20, int max_iter=1000, "
      "Tensor? "
      "initial=None) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_gmres", TORCH_FN(linear_gmres_impl));
}
}  // namespace
GMRESResult gmres(const LinearOperator& op, const Tensor& rhs,
                  const GMRESOptions& options) {
  TORCH_CHECK(op.dtype() == rhs.scalar_type() && op.device() == rhs.device(),
              "GMRES operator and RHS must share dtype/device");
  auto p = pack_graph(op.graph());
  static auto handle = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("torchpairwise::linear_gmres", "")
                           .typed<decltype(linear_gmres_impl)>();
  auto result = handle.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), op.shape(),
                            rhs, options.rtol, options.atol, options.restart,
                            options.max_iter, options.initial);
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result), std::get<4>(result)};
}
}  // namespace torchpairwise::ops
