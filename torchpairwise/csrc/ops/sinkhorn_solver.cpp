#include "sinkhorn_solver.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "expression_graph_detail.h"
#include "fused_reducers.h"
#include "linear_plan_detail.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::linear;
namespace fused = detail::fused;
using Result = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
Tensor weighted_sum(const Tensor& mass, const Tensor& value) {
  return (mass * at::where(mass.gt(0), value, at::zeros_like(value))).sum(-1);
}
Result sinkhorn_impl(TORCHPAIRWISE_LINEAR_PACKET_DECL,
                     c10::SymIntArrayRef shape, const Tensor& original_a,
                     const Tensor& original_b, double epsilon, int64_t max_iter,
                     double rtol, double atol, c10::optional<double> reach,
                     c10::optional<double> epsilon_scaling,
                     const c10::optional<Tensor>& initial_f,
                     const c10::optional<Tensor>& initial_g,
                     int64_t check_every) {
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0,
              "sinkhorn epsilon must be finite and positive");
  TORCH_CHECK(max_iter >= 0 && check_every > 0 && std::isfinite(rtol) &&
                  rtol >= 0 && std::isfinite(atol) && atol >= 0,
              "sinkhorn needs nonnegative tolerances/max_iter and positive "
              "check_every");
  TORCH_CHECK(!reach || (std::isfinite(*reach) && *reach > 0),
              "sinkhorn reach must be finite and positive");
  TORCH_CHECK(
      !epsilon_scaling || (std::isfinite(*epsilon_scaling) &&
                           *epsilon_scaling > 0 && *epsilon_scaling < 1),
      "sinkhorn epsilon_scaling must lie strictly between zero and one");
  TORCH_CHECK(
      shape.size() >= 2 && shape[shape.size() - 2] > 0 && shape.back() > 0,
      "sinkhorn requires nonempty cost sample axes");
  TORCH_CHECK(
      original_a.dim() >= 1 && original_b.dim() >= 1 &&
          original_a.sym_size(-1) == shape[shape.size() - 2] &&
          original_b.sym_size(-1) == shape.back(),
      "sinkhorn masses must end in the corresponding cost sample count");
  TORCH_CHECK(
      original_a.is_floating_point() && original_b.is_floating_point() &&
          original_a.device() == original_b.device() &&
          original_a.layout() == at::kStrided &&
          original_b.layout() == at::kStrided,
      "sinkhorn masses must be strided floating tensors on the cost device");
  if (original_a.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(original_a.device().type())
                     .getStream(original_a.device())
                     .is_capturing(),
                "sinkhorn uses native stopping and does not support CUDA graph "
                "capture");
  at::NoGradGuard guard;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto graph = unpack_graph(TORCHPAIRWISE_LINEAR_PACKET_ARGS);
  TORCH_CHECK(graph.nodes.size() == 1 &&
                  graph.nodes[0] == std::vector<int64_t>({0, 0, -1}) &&
                  graph.sources.size() == 1,
              "sinkhorn requires a scalar entrywise cost; pass a LazyPairwise "
              "or dense tensor");
  auto batch = at::infer_size_symint(
      shape.slice(0, shape.size() - 2),
      original_a.sym_sizes().slice(0, original_a.dim() - 1));
  batch = at::infer_size_symint(
      batch, original_b.sym_sizes().slice(0, original_b.dim() - 1));
  auto rows_shape = batch, columns_shape = batch;
  rows_shape.push_back(shape[shape.size() - 2]);
  columns_shape.push_back(shape.back());
  auto a = original_a.to(at::kDouble).expand_symint(rows_shape);
  auto b = original_b.to(at::kDouble).expand_symint(columns_shape);
  TORCH_CHECK((a.isfinite() & a.ge(0)).all().item<bool>() &&
                  (b.isfinite() & b.ge(0)).all().item<bool>(),
              "sinkhorn masses must be finite and nonnegative");
  auto total_a = a.sum(-1), total_b = b.sum(-1);
  TORCH_CHECK((total_a.gt(0) & total_b.gt(0)).all().item<bool>(),
              "sinkhorn needs positive total mass in every batch");
  if (!reach)
    TORCH_CHECK((total_a - total_b)
                    .abs()
                    .le(at::maximum(total_a, total_b) * 1e-6)
                    .all()
                    .item<bool>(),
                "balanced sinkhorn requires equal total masses; pass reach for "
                "unbalanced transport");
  auto warm = [&](const c10::optional<Tensor>& value, const Tensor& mass) {
    if (!value)
      return at::zeros_like(mass);
    TORCH_CHECK(
        value->device() == mass.device() && value->is_floating_point() &&
            at::infer_size(value->sizes(), mass.sizes()) == mass.sizes(),
        "sinkhorn initial potentials must broadcast to their mass vectors on "
        "the cost device");
    return value->to(at::kDouble).expand_as(mass).clone();
  };
  auto f = warm(initial_f, a), g = warm(initial_g, b);
  auto log_a = a.log(), log_b = b.log();
  auto row_slot = at::zeros_like(a), column_slot = at::zeros_like(b);
  auto scale_slot = at::scalar_tensor(-1. / epsilon, a.options());
  auto expression = graph.sources[0];
  auto root = int64_t(expression.nodes.size()) - 1;
  auto factor = int64_t(expression.factors.size());
  expression.factors.insert(expression.factors.end(),
                            {scale_slot, row_slot, column_slot});
  expression.nodes.push_back({1, factor, -1});
  expression.nodes.push_back({5, root, root + 1});
  expression.nodes.push_back({38, -1, factor + 1});
  expression.nodes.push_back({3, root + 2, root + 3});
  expression.nodes.push_back({39, -1, factor + 2});
  expression.nodes.push_back({3, root + 4, root + 5});
  auto prepared = prepare_expression_graph(expression);
  TORCH_CHECK(prepared.values[X].device() == a.device(),
              "sinkhorn cost and masses must share a device");
  TORCH_CHECK(prepared.queries == shape[shape.size() - 2] &&
                  prepared.references == shape.back(),
              "sinkhorn cost shape changed; reconstruct the operator");
#ifdef WITH_CUDA
  // Fused half steps: the bare cost lowers to a register kernel that adds
  // the slots itself, avoiding per-tile launches. Masses that add batch
  // dimensions keep the tile path.
  c10::optional<fused::Plan> plan;
  {
    auto base = prepare_expression_graph(graph.sources[0]);
    base.settings.dtype = int64_t(at::kFloat);
    plan = fused::plan(base.values, base.settings);
    std::vector<int64_t> sizes;
    for (const auto& size : batch)
      sizes.push_back(size.guard_int(__FILE__, __LINE__));
    if (plan && plan->batch != sizes)
      plan.reset();
  }
#endif
  double stage = epsilon, tau = 1.;
  auto reduction = [&](bool transpose, bool maximum = false) {
#ifdef WITH_CUDA
    if (plan && !maximum) {
      fused::count();
      const auto& bias = transpose ? row_slot : column_slot;
      const auto& other = transpose ? column_slot : row_slot;
      // The half step's fixed point: lse = -potential / (stage * tau).
      auto guess = (transpose ? g : f) * (-1. / (stage * tau)) - other;
      auto value = fused::biased_logsumexp_cuda(
          transpose ? fused::swapped(*plan) : *plan, scale_slot,
          bias.reshape({-1, bias.size(-1)}),
          guess.reshape({-1, guess.size(-1)}));
      return value.view(other.sizes()) + other;
    }
#endif
    auto settings = prepared.settings;
    settings.operation = transpose ? "transpose_sum" : "sum";
    settings.dtype = int64_t(at::kDouble);
    auto dimensions = batch_shape(prepared.values, settings);
    auto output_shape = dimensions;
    int64_t n = prepared.values[transpose ? Y : X].size(-2);
    int64_t m = prepared.values[transpose ? X : Y].size(-2);
    output_shape.push_back(n);
    auto output = at::empty(output_shape, a.options());
    int64_t batches = 1;
    for (auto dimension : dimensions)
      batches *= dimension;
    for (int64_t flat = 0; flat < batches; ++flat) {
      auto index = coordinates(flat, dimensions);
      for (int64_t left = 0; left < n; left += settings.size) {
        auto count = std::min(settings.size, n - left);
        auto total = at::full({count}, -std::numeric_limits<double>::infinity(),
                              a.options());
        for (int64_t right = 0; right < m; right += settings.size) {
          auto other = std::min(settings.size, m - right);
          auto local = local_inputs(prepared.values, settings, index, left,
                                    count, right, other);
          auto tile =
              evaluate(local, settings, false, left, right).to(at::kDouble);
          total = maximum ? at::maximum(total, std::get<0>(tile.max(-1)))
                          : at::logaddexp(total, at::logsumexp(tile, -1));
        }
        lookup(output, index, 1).narrow(0, left, count).copy_(total);
      }
    }
    return output;
  };
  auto lse = [&](bool transpose) { return reduction(transpose); };
  if (epsilon_scaling) {
    // Stream the cost maximum, avoiding a dense diameter calculation.
    scale_slot.fill_(1.);
    row_slot.zero_();
    column_slot.zero_();
    stage =
        std::max(epsilon, reduction(false, true).max().abs().item<double>());
  }
  auto state = at::ones_like(total_a, total_a.options().dtype(at::kLong));
  auto iterations = at::zeros_like(state);
  auto error = at::full_like(total_a, std::numeric_limits<double>::infinity());
  for (int64_t step = 0; step < max_iter; ++step) {
    scale_slot.fill_(-1. / stage);
    tau = reach ? *reach / (*reach + stage) : 1.;
    auto active = state.eq(1);
    row_slot.zero_();
    column_slot.copy_(log_b + g / stage);
    auto next_f = at::where(a.gt(0), -stage * tau * lse(false), 0.);
    f = at::where(active.unsqueeze(-1), next_f, f);
    row_slot.copy_(log_a + f / stage);
    column_slot.zero_();
    auto next_g = at::where(b.gt(0), -stage * tau * lse(true), 0.);
    g = at::where(active.unsqueeze(-1), next_g, g);
    iterations.add_(active.to(at::kLong));
    if ((step + 1) % check_every != 0 && step + 1 != max_iter)
      continue;
    row_slot.zero_();
    column_slot.copy_(log_b + g / stage);
    auto row_log = f / stage + lse(false);
    auto row_mass = a * at::where(a.gt(0), row_log, 0.).exp();
    auto target_a = reach ? a * (-f / *reach).exp() : a;
    error = (row_mass - target_a).abs().sum(-1);
    if (reach) {
      row_slot.copy_(log_a + f / stage);
      column_slot.zero_();
      auto column_mass =
          b * at::where(b.gt(0), g / stage + lse(true), 0.).exp();
      error = error + (column_mass - b * (-g / *reach).exp()).abs().sum(-1);
    }
    auto finite =
        f.isfinite().all(-1) & g.isfinite().all(-1) & error.isfinite();
    state = at::where(active & ~finite, 2, state);
    auto done = error.le(atol + rtol * (reach ? total_a + total_b : total_a));
    if (stage > epsilon) {
      // Coarse stages solve only to a loose tolerance, then warm-start target.
      if ((done | error.le(.01 * (total_a + total_b)) | ~finite)
              .all()
              .item<bool>())
        stage = std::max(epsilon, stage * *epsilon_scaling);
    } else
      state = at::where(active & finite & done, 0, state);
    if (!state.eq(1).any().item<bool>())
      break;
  }
  return {f, g, iterations, state.eq(0), state, error};
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "sinkhorn_solver(Tensor[] inputs, int[] counts, str[] metrics, str[] "
      "families, Dict(str,Tensor) tensors, Dict(str,float) reals, "
      "Dict(str,int) integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] source_counts, Dict(str,int[]) programs, Tensor[] "
      "factors, int[] factor_counts, Dict(str,int[]) kinds, Dict(str,int[]) "
      "integer_numbers, Dict(str,float[]) real_numbers, int[] linear_program, "
      "Tensor[] linear_factors, SymInt[] shape, Tensor a, Tensor b, float "
      "epsilon, int max_iter=1000, float rtol=1e-6, float atol=0., float? "
      "reach=None, float? epsilon_scaling=None, Tensor? initial_f=None, "
      "Tensor? initial_g=None, int check_every=10) -> (Tensor, Tensor, Tensor, "
      "Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("sinkhorn_solver", TORCH_FN(sinkhorn_impl));
}
ScalarExpression make_plan(const ScalarExpression& cost,
                           const SinkhornResult& result, const Tensor& a,
                           const Tensor& b, double epsilon,
                           at::ScalarType dtype) {
  auto aa = a.to(dtype), bb = b.to(dtype);
  auto exponent = (-cost / epsilon)
                      .add_rows(result.f.detach().to(dtype) / epsilon +
                                at::where(aa.gt(0), aa, 1.).log())
                      .add_columns(result.g.detach().to(dtype) / epsilon +
                                   at::where(bb.gt(0), bb, 1.).log());
  auto supported =
      cost.row_values(aa.gt(0)).logical_and(cost.column_values(bb.gt(0)));
  return exponent.where(supported, -std::numeric_limits<double>::infinity())
      .exp();
}
}  // namespace
ScalarExpression sinkhorn_plan(const ScalarExpression& cost,
                               const SinkhornResult& result, const Tensor& a,
                               const Tensor& b, double epsilon) {
  return make_plan(cost, result, a, b, epsilon, cost.metadata().dtype);
}
SinkhornResult sinkhorn(const ScalarExpression& cost, const Tensor& a,
                        const Tensor& b, const SinkhornOptions& options) {
  auto linear = LinearOperator(cost);
  auto packet = pack_graph(linear.graph());
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::sinkhorn_solver", "")
                       .typed<decltype(sinkhorn_impl)>();
  auto result =
      op.call(TORCHPAIRWISE_LINEAR_PACKED_ARGS(packet), linear.shape(), a, b,
              options.epsilon, options.max_iter, options.rtol, options.atol,
              options.reach, options.epsilon_scaling, options.initial_f,
              options.initial_g, options.check_every);
  SinkhornResult fit{
      std::get<0>(result), std::get<1>(result), {},
      std::get<2>(result), std::get<3>(result), std::get<4>(result),
      std::get<5>(result)};
  fit.f = fit.f.detach();
  fit.g = fit.g.detach();
  auto aa = a.to(at::kDouble), bb = b.to(at::kDouble);
  auto value =
      options.reach
          ? weighted_sum(aa,
                         -*options.reach * at::expm1(-fit.f / *options.reach)) +
                weighted_sum(
                    bb, -*options.reach * at::expm1(-fit.g / *options.reach))
          : weighted_sum(aa, fit.f) + weighted_sum(bb, fit.g);
  auto mass =
      make_plan(cost, fit, aa, bb, options.epsilon, at::kDouble).reduce("sum");
  auto dual = value - options.epsilon * (mass - aa.sum(-1) * bb.sum(-1));
  // Preserve the requested balanced value while retaining the dual envelope.
  fit.objective = options.reach ? dual : dual + (value - dual).detach();
  return fit;
}
}  // namespace torchpairwise::ops
