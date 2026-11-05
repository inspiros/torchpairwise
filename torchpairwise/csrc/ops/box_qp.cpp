#include "box_qp.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "linear_plan_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::linear;
using Result =
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
Tensor dot_columns(const Tensor& a, const Tensor& b) {
  return (a.to(at::kDouble) * b.to(at::kDouble)).sum(-2);
}
Tensor max_columns(const Tensor& x) {
  // Empty sample axes have zero infinity norm (including zero-dimensional QPs).
  auto shape = x.sizes().vec();
  shape[shape.size() - 2] = 1;
  return std::get<0>(at::cat({x.abs().to(at::kDouble),
                              at::zeros(shape, x.options().dtype(at::kDouble))},
                             -2)
                         .max(-2));
}
Tensor choose(const Tensor& mask, const Tensor& yes, const Tensor& no) {
  return at::where(mask.unsqueeze(-2), yes, no);
}
Tensor expand_block(const Tensor& value, at::IntArrayRef shape, const Tensor& q,
                    const char* name) {
  TORCH_CHECK(value.defined() && value.layout() == at::kStrided &&
                  value.scalar_type() == q.scalar_type() &&
                  value.device() == q.device(),
              "box_qp ", name, " must be strided and match q dtype/device");
  auto block = value.dim() == 1 ? value.unsqueeze(-1) : value;
  TORCH_CHECK(
      at::infer_size(block.sizes(), shape) == shape, "box_qp ", name,
      " must broadcast to the solution without adding batch dimensions");
  return block.expand(shape);
}
struct Projection {
  Tensor value, success;
};
struct Constraints {
  Tensor lower, upper, a, target, feasible;
  bool equality;
  Constraints(const Tensor& low, const Tensor& high,
              const c10::optional<Tensor>& vector,
              const c10::optional<Tensor>& rhs, const Tensor& q,
              at::IntArrayRef shape) {
    lower = expand_block(low, shape, q, "lower").to(at::kDouble);
    upper = expand_block(high, shape, q, "upper").to(at::kDouble);
    utils::ValueChecks checks;
    checks.require(lower.isfinite().all() & upper.isfinite().all(),
                   "box_qp requires finite lower/upper bounds");
    checks.finish();
    feasible = lower.le(upper).all(-2);
    equality = bool(vector);
    TORCH_CHECK(bool(vector) == bool(rhs),
                "box_qp equality needs both a vector and a target");
    auto pairs = shape.vec();
    pairs.erase(pairs.end() - 2);
    if (!equality) {
      a = at::zeros_like(lower);
      target = at::zeros(pairs, lower.options());
      return;
    }
    a = expand_block(*vector, shape, q, "equality vector").to(at::kDouble);
    TORCH_CHECK(rhs->layout() == at::kStrided &&
                    rhs->scalar_type() == q.scalar_type() &&
                    rhs->device() == q.device(),
                "box_qp equality target must match q dtype/device");
    TORCH_CHECK(at::infer_size(rhs->sizes(), pairs) == pairs,
                "box_qp equality target must broadcast to [...,R]");
    target = rhs->expand(pairs).to(at::kDouble);
    // Normalize the equality for scale-invariant projection and diagnostics.
    auto scale = max_columns(a);
    auto denominator = at::where(scale.gt(0), scale, at::ones_like(scale));
    a = a / denominator.unsqueeze(-2);
    target = target / denominator;
    auto minimum = at::where(a.ge(0), a * lower, a * upper).sum(-2);
    auto maximum = at::where(a.ge(0), a * upper, a * lower).sum(-2);
    feasible = feasible & target.ge(minimum) & target.le(maximum);
  }
  Tensor tangent_gradient(const Tensor& gradient) const {
    auto g = gradient.to(at::kDouble);
    if (!equality)
      return g;
    auto squared = dot_columns(a, a);
    auto normal = dot_columns(a, g) /
                  at::where(squared.gt(0), squared, at::ones_like(squared));
    // Adding a multiple of a leaves projection onto a.T x=b unchanged.
    // Removing it in double prevents float32 equality roundoff from reversing
    // the line-search derivative near stationarity.
    return g - a * normal.unsqueeze(-2);
  }
  Tensor violation(const Tensor& x) const {
    auto v = at::maximum(max_columns((lower - x.to(at::kDouble)).clamp_min(0)),
                         max_columns((x.to(at::kDouble) - upper).clamp_min(0)));
    return equality ? at::maximum(v, (dot_columns(a, x) - target).abs()) : v;
  }
  Tensor tolerance(const Tensor& x, double rtol, double atol) const {
    auto scale = at::ones_like(target);
    if (equality)
      scale = at::maximum(
          scale, at::maximum(target.abs(), dot_columns(a.abs(), x.abs())));
    return (scale * rtol).clamp_min(atol);
  }
  Projection project(const Tensor& input) const {
    auto z = input.to(at::kDouble);
    auto clip = [&](const Tensor& v) {
      return at::minimum(at::maximum(v, lower), upper);
    };
    auto result = clip(z);
    auto finite =
        z.isfinite().all(-2) & a.isfinite().all(-2) & target.isfinite();
    if (!equality)
      return {result.to(input.scalar_type()), finite};
    auto nz = a.ne(0), any = nz.any(-2);
    auto divisor = at::where(nz, a, at::ones_like(a));
    auto first = (z - lower) / divisor, second = (z - upper) / divisor;
    auto inf = std::numeric_limits<double>::infinity();
    // Include a zero slot so empty/zero-coefficient equations have a bracket.
    // Signed extrema bracket the monotone equality residual.
    auto zero = at::zeros_like(target).unsqueeze(-2);
    auto left = std::get<0>(
        at::cat({at::where(nz, at::minimum(first, second), inf), zero}, -2)
            .min(-2));
    auto right = std::get<0>(
        at::cat({at::where(nz, at::maximum(first, second), -inf), zero}, -2)
            .max(-2));
    auto tau = at::zeros_like(target);
    auto success = ~any & target.eq(0) & finite;
    auto active = finite & feasible & any;
    auto end_low = dot_columns(a, at::where(a.ge(0), lower, upper));
    auto end_high = dot_columns(a, at::where(a.ge(0), upper, lower));
    auto at_low = active & target.eq(end_low),
         at_high = active & target.eq(end_high);
    auto lower_extreme =
        at::where(a.gt(0), lower, at::where(a.lt(0), upper, result));
    auto upper_extreme =
        at::where(a.gt(0), upper, at::where(a.lt(0), lower, result));
    result =
        choose(at_low, lower_extreme, choose(at_high, upper_extreme, result));
    success = success | at_low | at_high;
    active = active & ~success & left.isfinite() & right.isfinite();
    // Safeguarded semismooth Newton: the slope is the active-face squared norm.
    // Bisection provides progress when the Newton step crosses a breakpoint.
    constexpr double eps = std::numeric_limits<double>::epsilon();
    for (int step = 0; step < 80 && active.any().item<bool>(); ++step) {
      auto raw = z - tau.unsqueeze(-2) * a;
      auto candidate = clip(raw), gap = dot_columns(a, candidate) - target;
      auto tolerance =
          (dot_columns(a.abs(), candidate.abs()) + target.abs() + 1) *
          (64 * eps);
      auto done = active & gap.abs().le(tolerance);
      result = choose(active, candidate, result);
      success = success | done;
      active = active & ~done;
      if (!active.any().item<bool>())
        break;
      left = at::where(active & gap.gt(0), tau, left);
      right = at::where(active & gap.le(0), tau, right);
      auto free = raw.gt(lower) & raw.lt(upper);
      auto slope = (a.square() * free).sum(-2);
      auto next =
          tau + gap / at::where(slope.gt(0), slope, at::ones_like(slope));
      auto midpoint = left * .5 + right * .5;
      auto accept =
          slope.gt(0) & next.gt(left) & next.lt(right) & next.isfinite();
      tau = at::where(active, at::where(accept, next, midpoint), tau);
    }
    return {result.to(input.scalar_type()), success & finite & feasible};
  }
};
Result iterate(const Plan& Q, const Tensor& original, const Tensor& lower,
               const Tensor& upper, const c10::optional<Tensor>& equality,
               const c10::optional<Tensor>& target,
               const c10::optional<Tensor>& initial, double rtol, double atol,
               int64_t max_iter) {
  bool vector = original.dim() == 1;
  auto q = vector ? original.unsqueeze(-1) : original;
  const auto& matrix = Q.shapes.back();
  TORCH_CHECK(matrix.back() == matrix[matrix.size() - 2],
              "box_qp requires a square operator");
  TORCH_CHECK(q.sym_size(-2) == matrix.back(),
              "box_qp q sample count mismatch");
  auto shape = at::infer_size_symint(
      c10::SymIntArrayRef(matrix).slice(0, matrix.size() - 2),
      q.sym_sizes().slice(0, q.dim() - 2));
  shape.insert(shape.end(), {q.sym_size(-2), q.sym_size(-1)});
  q = q.expand_symint(shape);
  Constraints c(lower, upper, equality, target, original, q.sizes());
  auto x = initial ? expand_block(*initial, q.sizes(), original, "initial")
                   : at::zeros_like(q);
  x = x.to(at::kDouble);
  auto finite = q.isfinite().all(-2) & x.isfinite().all(-2) &
                c.a.isfinite().all(-2) & c.target.isfinite();
  auto state = at::where(
      ~finite, 3,
      at::where(
          ~c.feasible, 2,
          at::ones_like(c.feasible, c.feasible.options().dtype(at::kLong))));
  auto first = c.project(choose(state.eq(1), x, at::zeros_like(x)));
  state = at::where(state.eq(1) & ~first.success, 5, state);
  x = first.value;
  auto iterations = at::zeros_like(state), alpha = at::ones_like(c.target);
  auto optimality =
      at::full_like(c.target, std::numeric_limits<double>::quiet_NaN());
  auto feasibility = c.violation(x), objective = optimality.clone();
  auto converged = at::zeros_like(state, state.options().dtype(at::kBool));
  auto threshold = (max_columns(q).clamp_min(1) * rtol).clamp_min(atol);
  double eps = original.scalar_type() == at::kDouble
                   ? std::numeric_limits<double>::epsilon()
                   : std::numeric_limits<float>::epsilon();
  for (int64_t step = 0;; ++step) {
    // Recompute Qx: reported objective/KKT checks use the true map, not an
    // incrementally updated gradient whose residual might have drifted.
    auto reported = x.to(original.scalar_type());
    auto product = Q(reported);
    auto gradient = product.to(at::kDouble) + q.to(at::kDouble);
    auto tangent = c.tangent_gradient(gradient);
    auto accurate_x = x;
    auto accurate_reported = reported.to(at::kDouble);
    finite = product.isfinite().all(-2) & gradient.isfinite().all(-2);
    state = at::where(state.eq(1) & ~finite, 3, state);
    objective = .5 * dot_columns(reported, product) + dot_columns(q, reported);
    state = at::where(state.eq(1) & ~objective.isfinite(), 3, state);
    auto active = state.eq(1);
    auto mapping = c.project(
        choose(active, accurate_reported - tangent, accurate_reported));
    state = at::where(active & ~mapping.success, 5, state);
    optimality = max_columns(accurate_reported - mapping.value);
    feasibility = c.violation(reported);
    converged = state.eq(1) & optimality.le(threshold) &
                feasibility.le(c.tolerance(reported, rtol, atol));
    state = at::where(converged, 0, state);
    active = state.eq(1);
    if (step >= max_iter || !active.any().item<bool>())
      break;
    iterations.add_(active.to(at::kLong));
    auto projected = c.project(
        choose(active, accurate_x - tangent * alpha.unsqueeze(-2), accurate_x));
    state = at::where(active & ~projected.success, 5, state);
    active = state.eq(1);
    auto d = choose(active, projected.value - accurate_x,
                    at::zeros_like(accurate_x));
    auto qd = Q(d.to(original.scalar_type())).to(at::kDouble);
    auto curvature = dot_columns(d, qd), descent = dot_columns(tangent, d),
         squared = dot_columns(d, d);
    finite = qd.isfinite().all(-2) & curvature.isfinite() & descent.isfinite() &
             squared.isfinite();
    state = at::where(active & ~finite, 3, state);
    auto curvature_bound =
        64 * eps * max_columns(d) * max_columns(qd) * q.size(-2);
    state =
        at::where(active & finite & curvature.lt(-curvature_bound), 4, state);
    active = state.eq(1);
    auto rate = at::where(curvature.gt(0),
                          (-descent / at::where(curvature.gt(0), curvature,
                                                at::ones_like(curvature)))
                              .clamp(0, 1),
                          at::ones_like(curvature));
    auto next = accurate_x + d * rate.unsqueeze(-2);
    auto moved = next.ne(x).any(-2);
    state = at::where(active & ~moved, 6, state);
    active = state.eq(1);
    x = choose(active, next, x);
    // Spectral projected gradient step; exact quadratic line minimization above
    // keeps the update inside the segment between two feasible points.
    alpha = at::where(curvature.gt(0), (squared / curvature).clamp(1e-12, 1e12),
                      alpha);
  }
  converged = state.eq(0);
  auto failed = state.ge(2);
  double nan = std::numeric_limits<double>::quiet_NaN();
  x = x.to(original.scalar_type());
  x = choose(failed, at::full_like(x, nan), x);
  objective = at::where(failed, at::full_like(objective, nan), objective);
  optimality = at::where(failed, at::full_like(optimality, nan), optimality);
  feasibility = at::where(failed, at::full_like(feasibility, nan), feasibility);
  if (vector) {
    x = x.squeeze(-1);
    objective = objective.squeeze(-1);
    iterations = iterations.squeeze(-1);
    converged = converged.squeeze(-1);
    state = state.squeeze(-1);
    optimality = optimality.squeeze(-1);
    feasibility = feasibility.squeeze(-1);
  }
  return {x, objective, iterations, converged, state, optimality, feasibility};
}
Result box_qp_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL, c10::SymIntArrayRef shape,
                   const Tensor& q, const Tensor& lower, const Tensor& upper,
                   const c10::optional<Tensor>& equality,
                   const c10::optional<Tensor>& target,
                   const c10::optional<Tensor>& initial, double rtol,
                   double atol, int64_t max_iter) {
  TORCH_CHECK(
      q.layout() == at::kStrided && q.dim() >= 1 &&
          (q.scalar_type() == at::kFloat || q.scalar_type() == at::kDouble),
      "box_qp q must be strided float32/float64 [N] or [...,N,R]");
  TORCH_CHECK(q.dim() == 1 || q.size(-1) > 0,
              "box_qp q channel count must be positive");
  TORCH_CHECK(
      std::isfinite(rtol) && rtol >= 0 && std::isfinite(atol) && atol >= 0 &&
          max_iter >= 0,
      "box_qp tolerances must be finite/nonnegative and max_iter nonnegative");
  if (q.is_cuda())
    TORCH_CHECK(
        !c10::impl::VirtualGuardImpl(q.device().type())
             .getStream(q.device())
             .is_capturing(),
        "box_qp uses native stopping and does not support CUDA graph capture");
  if (at::GradMode::is_enabled()) {
    auto check = [](const Tensor& t) {
      TORCH_CHECK(!t.requires_grad(),
                  "box_qp is inference-only; use torch.no_grad() or detach "
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
    check(q);
    check(lower);
    check(upper);
    if (equality)
      check(*equality);
    if (target)
      check(*target);
    if (initial)
      check(*initial);
  }
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto graph = unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  Plan plan(std::move(graph), q);
  TORCH_CHECK(c10::SymIntArrayRef(plan.shapes.back()).equals(shape),
              "Bound box_qp source shape changed; reconstruct the operator");
  return iterate(plan, q, lower, upper, equality, target, initial, rtol, atol,
                 max_iter);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "box_qp(Tensor[] inputs, int[] counts, str[] metrics, str[] families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] source_counts, Dict(str,int[]) programs, Tensor[] "
      "factors, int[] factor_counts, Dict(str,int[]) kinds, Dict(str,int[]) "
      "integer_numbers, Dict(str,float[]) real_numbers, int[] linear_program, "
      "Tensor[] linear_factors, SymInt[] shape, Tensor q, Tensor lower, Tensor "
      "upper, Tensor? equality=None, Tensor? target=None, Tensor? "
      "initial=None, float rtol=1e-5, float atol=0., int max_iter=1000) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("box_qp", TORCH_FN(box_qp_impl));
}
}  // namespace
QPResult box_qp(const LinearOperator& Q, const Tensor& q,
                const BoxQPOptions& options) {
  TORCH_CHECK(Q.dtype() == q.scalar_type() && Q.device() == q.device(),
              "box_qp operator and q must share dtype/device");
  TORCH_CHECK(options.lower.defined() && options.upper.defined(),
              "box_qp lower and upper bounds are required");
  auto p = pack_graph(Q.graph());
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::box_qp", "")
                       .typed<decltype(box_qp_impl)>();
  auto fit =
      op.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(p), Q.shape(), q, options.lower,
              options.upper,
              options.equality ? c10::optional<Tensor>(options.equality->vector)
                               : c10::nullopt,
              options.equality ? c10::optional<Tensor>(options.equality->target)
                               : c10::nullopt,
              options.initial, options.rtol, options.atol, options.max_iter);
  return {std::get<0>(fit), std::get<1>(fit), std::get<2>(fit),
          std::get<3>(fit), std::get<4>(fit), std::get<5>(fit),
          std::get<6>(fit)};
}
}  // namespace torchpairwise::ops
