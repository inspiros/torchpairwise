#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "expression_statistics_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::streaming {
namespace {
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
struct WeightedMoments {
  Tensor mass, square_mass, mean, m2, scale, anchor;
  bool anchored;
  WeightedMoments(int64_t n, const at::TensorOptions& options, bool anchored)
      : anchored(anchored) {
    mass = at::zeros({n}, options.dtype(at::kDouble));
    square_mass = at::zeros_like(mass);
    mean = at::zeros_like(mass);
    m2 = at::zeros_like(mass);
    scale = at::zeros_like(mass);
    if (anchored)
      anchor = at::zeros_like(mass);
  }
  void merge(const Tensor& values, const Tensor& weights) {
    // Rescale the entire merge state when a larger weight arrives. The
    // normalized statistic is invariant to this scale; sums and squared
    // weights cannot overflow merely because confidences share a large scale.
    auto [block_scale, pivot] = weights.max(-1);
    auto next_scale = at::maximum(scale, block_scale);
    auto safe_scale = at::where(next_scale.gt(0), next_scale, 1.);
    auto factor = scale / safe_scale;
    auto prior_mass = mass * factor, prior_m2 = m2 * factor;
    auto w = weights / safe_scale.unsqueeze(-1);
    auto k = w.sum(-1);
    auto safe_k = at::where(k.gt(0), k, 1.);
    auto value = at::where(weights.eq(0) & ~values.isfinite(), 0., values);
    auto first = value.gather(-1, pivot.unsqueeze(-1)).squeeze(-1);
    first = at::where(first.isfinite(), first, 0.);
    // Keep the anchor separate from its weighted offset across tiles and in
    // backward. Rounding an absolute mean loses small translated variations.
    auto base = first;
    if (anchored) {
      anchor = at::where(prior_mass.eq(0) & k.gt(0), first, anchor);
      base = anchor;
    }
    auto shifted = at::where(w.gt(0), value - base.unsqueeze(-1), 0.);
    auto average = (w * shifted).sum(-1) / safe_k;
    if (!anchored)
      average = base + average;
    auto centered = at::where(
        w.gt(0), (anchored ? shifted : value) - average.unsqueeze(-1), 0.);
    auto partial = (w * centered.square()).sum(-1);
    auto total = prior_mass + k;
    auto ratio = k / at::where(total.gt(0), total, 1.);
    auto delta = average - mean;
    auto next_mean =
        at::where(mean.isfinite() & average.isfinite(), mean + delta * ratio,
                  mean * (1. - ratio) + average * ratio);
    auto next_m2 = prior_m2 + partial + delta.square() * prior_mass * ratio;
    mean = at::where(k.gt(0), at::where(prior_mass.gt(0), next_mean, average),
                     mean);
    m2 = at::where(k.gt(0), at::where(prior_mass.gt(0), next_m2, partial),
                   prior_m2);
    mass = total;
    square_mass = square_mass * factor.square() + w.square().sum(-1);
    scale = next_scale;
  }
  Tensor state() const {
    return anchored
               ? at::stack({mass, square_mass, mean, m2, scale, anchor}, -1)
               : at::stack({mass, square_mass, mean, m2, scale}, -1);
  }
  Tensor value(c10::string_view operation, double correction) const {
    if (operation == "mean")
      return at::where(mass.gt(0), anchored ? anchor + mean : mean, nan);
    auto denominator =
        mass - correction * square_mass / at::where(mass.gt(0), mass, 1.);
    auto result =
        at::where(mass.gt(0) & denominator.gt(0), m2 / denominator, nan);
    return operation == "std" ? result.sqrt() : result;
  }
};
Settings weighted_settings(const ExpressionPacket& p,
                           c10::string_view operation, int64_t dtype,
                           int64_t size, bool exclusion, bool transposed,
                           double correction, int64_t value_root,
                           int64_t weight_root) {
  TORCH_CHECK(operation == "mean" || operation == "var" || operation == "std",
              "Unknown weighted statistic");
  TORCH_CHECK(std::isfinite(correction) && correction >= 0,
              "Weighted correction must be finite and nonnegative");
  TORCH_CHECK(operation != "mean" || correction == 0.,
              "correction is only used by var/std");
  TORCH_CHECK(value_root >= 0 && weight_root >= 0 &&
                  value_root < int64_t(p.nodes.size()) &&
                  weight_root < int64_t(p.nodes.size()),
              "Invalid weighted expression roots");
  return statistics_settings(p, "var", dtype, size, exclusion, transposed);
}
Tensor eligible(int64_t i, int64_t n, int64_t j, int64_t m, const Settings& s,
                const at::TensorOptions& options) {
  if (!s.exclusion)
    return at::ones({n, m}, options.dtype(at::kBool));
  return at::arange(i, i + n, options.dtype(at::kLong))
      .unsqueeze(-1)
      .ne(at::arange(j, j + m, options.dtype(at::kLong)));
}
std::pair<Tensor, Tensor> entries(at::TensorList local, const Settings& s,
                                  int64_t value_root, int64_t weight_root,
                                  int64_t n, int64_t m,
                                  ExpressionReplay* replay = nullptr) {
  auto roots =
      expression_evaluate_roots(local, s, {value_root, weight_root}, replay);
  return {roots[0].expand({n, m}).to(at::kDouble),
          roots[1].expand({n, m}).to(at::kDouble)};
}
}  // namespace
StatisticsOutputs weighted_statistics_kernel(at::TensorList v,
                                             TORCHPAIRWISE_EXPRESSION_DECL,
                                             double correction,
                                             int64_t value_root,
                                             int64_t weight_root) {
  auto s = weighted_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed, correction, value_root,
      weight_root);
  at::NoGradGuard guard;
  utils::DeferredValueChecks checks;
  auto batch = batch_shape(v, s), shape = batch;
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  bool global = form;
  if (!global)
    shape.push_back(n);
  auto result = at::empty(shape, v[X].options().dtype(at::ScalarType(dtype)));
  auto state_shape = shape;
  state_shape.push_back(dtype == int64_t(at::kDouble) ? 6 : 5);
  auto state = at::empty(state_shape, v[X].options().dtype(at::kDouble));
  int64_t batches = 1;
  for (auto d : batch)
    batches *= d;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    // Preserve float32's existing float64 absolute-mean arithmetic; persistent
    // anchor work is needed for float64 entry precision.
    WeightedMoments whole(1, v[X].options(), dtype == int64_t(at::kDouble));
    for (int64_t left = 0; left < n; left += size) {
      auto count = std::min(size, n - left);
      WeightedMoments row(count, v[X].options(), dtype == int64_t(at::kDouble));
      for (int64_t right = 0; right < m; right += size) {
        auto other = std::min(size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other);
        auto [value, weights] =
            entries(local, s, value_root, weight_root, count, other);
        utils::check_values(
            (weights.isfinite() & weights.ge(0)).all(),
            "Statistical weights must be finite and nonnegative");
        weights =
            at::where(eligible(left, count, right, other, s, weights.options()),
                      weights, 0.);
        if (global) {
          value = value.reshape({1, -1});
          weights = weights.reshape({1, -1});
        }
        (global ? whole : row).merge(value, weights);
      }
      if (!global) {
        lookup(result, index, 1)
            .narrow(0, left, count)
            .copy_(row.value(operation, correction));
        lookup(state, index, 2).narrow(0, left, count).copy_(row.state());
      }
    }
    if (global) {
      lookup(result, index, 0)
          .copy_(whole.value(operation, correction).squeeze(0));
      lookup(state, index, 1).copy_(whole.state().squeeze(0));
    }
  }
  checks.finish();
  return {result, state};
}
Tensors weighted_statistics_backward_kernel(
    const Tensor& gradient, at::TensorList v, const Tensor& result,
    const Tensor& state, TORCHPAIRWISE_EXPRESSION_DECL, double correction,
    int64_t value_root, int64_t weight_root) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto s = weighted_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed, correction, value_root,
      weight_root);
  utils::DeferredValueChecks checks;
  Tensors buffers;
  for (const auto& value : v)
    buffers.push_back(at::zeros_like(
        value, value.options().dtype(value.is_floating_point()
                                         ? at::kDouble
                                         : value.scalar_type())));
  auto batch = batch_shape(v, s);
  int64_t batches = 1;
  for (auto d : batch)
    batches *= d;
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  bool global = form, mean = operation == "mean";
  at::AutoGradMode enable(true);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    for (int64_t left = 0; left < n; left += size) {
      auto count = std::min(size, n - left);
      auto saved = lookup(state, index, global ? 1 : 2);
      if (!global)
        saved = saved.narrow(0, left, count);
      auto field = [&](int64_t k) {
        auto t = saved.select(-1, k);
        return global ? t.reshape({1, 1}) : t.unsqueeze(-1);
      };
      auto mass = field(0), square_mass = field(1), center = field(2),
           m2 = field(3), scale = field(4);
      auto safe_mass = at::where(mass.gt(0), mass, 1.);
      auto safe_scale = at::where(scale.gt(0), scale, 1.);
      auto denominator = mass - correction * square_mass / safe_mass;
      auto valid = mass.gt(0) &
                   (mean ? at::ones_like(mass, mass.options().dtype(at::kBool))
                         : denominator.gt(0));
      auto variance = m2 / denominator;
      auto g = lookup(gradient, index, global ? 0 : 1).to(at::kDouble);
      g = global ? g.reshape({1, 1}) : g.narrow(0, left, count).unsqueeze(-1);
      if (operation == "std")
        g = at::where(variance.eq(0), 0., g / (2. * variance.sqrt()));
      for (int64_t right = 0; right < m; right += size) {
        auto other = std::min(size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other,
                                  Tensor(), Tensor(), true);
        ExpressionReplay replay;
        auto [value, weights] =
            entries(local, s, value_root, weight_root, count, other, &replay);
        // Existing compiled graphs may pass the former five-field state.
        auto delta = state.size(-1) == 6 && dtype == int64_t(at::kDouble)
                         ? (value - field(5)) - center
                         : value - center;
        auto w = weights / safe_scale;
        Tensor da, dw;
        if (mean) {
          da = g * w / safe_mass;
          dw = (g / safe_scale) * delta / safe_mass;
        } else {
          da = g * 2. * w * delta / denominator;
          auto derivative =
              1. - correction *
                       (2. * w / safe_mass - square_mass / safe_mass.square());
          dw = (g / safe_scale) * (delta.square() - variance * derivative) /
               denominator;
        }
        da = at::where(valid, da, nan);
        dw = at::where(valid, dw, nan);
        auto allowed =
            eligible(left, count, right, other, s, weights.options());
        auto ignored = weights.eq(0) & ~value.isfinite();
        // A zero-weight value has no influence, even on an invalid row.
        da = at::where(allowed & weights.ne(0), da, 0.);
        dw = at::where(allowed & ~ignored, dw, 0.);
        expression_accumulate_roots(local, {da, dw}, {value_root, weight_root},
                                    buffers, s, index, &replay);
      }
    }
  }
  for (size_t k = 0; k < v.size(); ++k)
    buffers[k] = buffers[k].to(v[k].scalar_type()).contiguous();
  checks.finish();
  return buffers;
}
Tensor expression_weighted_statistics(at::TensorList v, const Settings& s,
                                      c10::string_view operation, bool global,
                                      double correction, int64_t value_root,
                                      int64_t weight_root) {
  const auto& p = s.expression->packet;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_expression_statistics",
                             "weighted_anchored")
          .typed<WeightedStatisticsSignature>();
  return std::get<0>(op.call(v, p.names, p.families, p.headers, p.reals,
                             p.integers, p.flags, p.nodes, p.numbers, operation,
                             s.dtype, s.size, s.exclusion, s.transposed(),
                             global, correction, value_root, weight_root));
}
Tensor legacy_weighted_statistics_state(const Tensor& state) {
  if (state.size(-1) == 5)
    return state;
  return at::cat({state.narrow(-1, 0, 2),
                  (state.select(-1, 2) + state.select(-1, 5)).unsqueeze(-1),
                  state.narrow(-1, 3, 2)},
                 -1);
}
namespace {
StatisticsOutputs legacy_weighted_statistics_kernel(
    at::TensorList v, TORCHPAIRWISE_EXPRESSION_DECL, double correction,
    int64_t value_root, int64_t weight_root) {
  auto [result, state] = weighted_statistics_kernel(
      v, names, families, headers, reals, integers, flags, nodes, numbers,
      operation, dtype, size, exclusion, transposed, form, correction,
      value_root, weight_root);
  return {result, legacy_weighted_statistics_state(state)};
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_expression_statistics.weighted(Tensor[] operands, str[] names, str[] "
      "families, int[][] headers, "
      "float?[][] reals, int?[][] integers, bool?[][] flags, int[][] nodes, "
      "Scalar[] numbers, "
      "str operation, int dtype, int size, bool exclusion, bool transposed, "
      "bool global_reduction, "
      "float correction, int value_root, int weight_root) -> (Tensor, Tensor)");
  m.def(
      "_expression_statistics.weighted_anchored(Tensor[] operands, str[] "
      "names, "
      "str[] families, int[][] headers, float?[][] reals, int?[][] integers, "
      "bool?[][] flags, int[][] nodes, Scalar[] numbers, str operation, "
      "int dtype, int size, bool exclusion, bool transposed, "
      "bool global_reduction, float correction, int value_root, "
      "int weight_root) -> (Tensor, Tensor)");
  m.def(
      "_expression_statistics_backward.weighted(Tensor grad, Tensor[] "
      "operands, Tensor result, Tensor center, "
      "str[] names, str[] families, int[][] headers, float?[][] reals, "
      "int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, "
      "bool transposed, bool global_reduction, float correction, int "
      "value_root, int weight_root) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_expression_statistics.weighted",
         TORCH_FN(legacy_weighted_statistics_kernel));
  m.impl("_expression_statistics.weighted_anchored",
         TORCH_FN(weighted_statistics_kernel));
  m.impl("_expression_statistics_backward.weighted",
         TORCH_FN(weighted_statistics_backward_kernel));
}
}  // namespace
}  // namespace torchpairwise::ops::detail::streaming
