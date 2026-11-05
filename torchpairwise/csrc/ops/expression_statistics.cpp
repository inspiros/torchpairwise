#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "expression_statistics_detail.h"
#include "fused_reducers.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::streaming {
namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();
// Chan merge of centered tile moments. Accumulate floating statistics in
// float64, including the saved mean, to retain small float32 variations on
// large offsets. Empty partials must not contaminate nonfinite valid entries.
struct Moments {
  Tensor count, anchor, mean, m2;
  bool full_range;
  explicit Moments(int64_t rows, const at::TensorOptions& options,
                   bool full_range)
      : count(at::zeros({rows}, options.dtype(at::kLong))),
        anchor(at::zeros({rows}, options.dtype(at::kDouble))),
        mean(at::zeros_like(anchor)),
        m2(at::zeros_like(mean)),
        full_range(full_range) {}
  void merge(const Tensor& values, const Tensor& allowed) {
    auto k = allowed.sum(-1);
    auto denominator = k.clamp_min(1).to(at::kDouble);
    auto pivot = allowed.to(at::kLong).argmax(-1, true);
    auto first = values.gather(-1, pivot).squeeze(-1);
    first = at::where(first.isfinite(), first, 0.);
    auto base = at::where(count.eq(0) & k.gt(0), first, anchor);
    auto shifted = at::where(allowed, values - base.unsqueeze(-1), 0.);
    // Opposite finite extremes can overflow an anchored difference. Rebase
    // those rows to zero; their large spread no longer needs offset precision.
    auto prior_mean = mean;
    if (full_range) {
      auto rebase = (allowed & values.isfinite() & ~shifted.isfinite()).any(-1);
      prior_mean = at::where(rebase, anchor + mean, mean);
      base = at::where(rebase, 0., base);
      shifted = at::where(allowed, values - base.unsqueeze(-1), 0.);
    }
    auto average = shifted.sum(-1) / denominator;
    // A finite row can still overflow its offset sum after rebasing. Compute
    // that mean in normalized units while keeping genuine nonfinite entries.
    if (full_range) {
      auto scale = shifted.abs().amax(-1);
      auto safe_scale = at::where(scale.isfinite() & scale.gt(0), scale, 1.);
      auto scaled_mean =
          ((shifted / safe_scale.unsqueeze(-1)).sum(-1) / denominator) *
          safe_scale;
      auto finite = (~allowed | values.isfinite()).all(-1);
      average = at::where(finite & ~average.isfinite(), scaled_mean, average);
    }
    auto centered = at::where(allowed, shifted - average.unsqueeze(-1), 0.);
    auto partial = centered.square().sum(-1);
    auto total = count + k;
    auto delta = average - prior_mean;
    auto ratio = k.to(at::kDouble) / total.clamp_min(1).to(at::kDouble);
    auto next_mean = prior_mean + delta * ratio;
    if (full_range)
      next_mean = at::where(
          prior_mean.isfinite() & average.isfinite() & ~delta.isfinite(),
          prior_mean * (1. - ratio) + average * ratio, next_mean);
    auto next_m2 =
        m2 + partial + delta.square() * count.to(at::kDouble) * ratio;
    mean = at::where(k.gt(0), at::where(count.gt(0), next_mean, average), mean);
    m2 = at::where(k.gt(0), at::where(count.gt(0), next_m2, partial), m2);
    count = total;
    anchor = base;
  }
  Tensor center() const {
    return at::stack({anchor, mean}, -1);
  }
};
Tensor allowed_entries(int64_t left, int64_t count, int64_t right,
                       int64_t other, const Settings& s,
                       const at::TensorOptions& options) {
  if (!s.exclusion)
    return at::ones({count, other}, options.dtype(at::kBool));
  return at::arange(left, left + count, options.dtype(at::kLong))
      .unsqueeze(-1)
      .ne(at::arange(right, right + other, options.dtype(at::kLong)));
}
}  // namespace
Settings statistics_settings(const ExpressionPacket& packet,
                             c10::string_view reduction, int64_t dtype,
                             int64_t size, bool exclusion, bool transposed) {
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  TORCH_CHECK(reduction == "var" || reduction == "std" ||
                  reduction == "logsumexp" || reduction == "count_nonzero",
              "Unknown operator statistic");
  TORCH_CHECK(reduction == "count_nonzero" || dtype == int64_t(at::kFloat) ||
                  dtype == int64_t(at::kDouble),
              "Floating statistics require float32/float64 entries");
  Settings s{"_expression",
             transposed ? "transpose_sum" : "sum",
             false,
             false,
             exclusion,
             false,
             false,
             2,
             1,
             size,
             dtype,
             3,
             {},
             {},
             {},
             std::string("_expression")};
  s.expression = expression_plan(packet);
  return s;
}
StatisticsOutputs expression_statistics_kernel(at::TensorList v,
                                               TORCHPAIRWISE_EXPRESSION_DECL,
                                               double correction) {
  TORCH_CHECK(std::isfinite(correction), "correction must be finite");
  auto s = statistics_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed);
  bool global = form, moments = operation == "var" || operation == "std";
  // Public float32 entries are bounded before promotion. Even an int64-sized
  // tile cannot overflow float64 sums, differences or squared deviations of
  // finite float32 values. Only float64 entries need the range safeguards.
  bool full_range = dtype == int64_t(at::kDouble);
  bool nonzero = operation == "count_nonzero";
  at::NoGradGuard guard;
  utils::DeferredValueChecks checks;
  auto batch = batch_shape(v, s), shape = batch;
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  if (!global)
    shape.push_back(n);
  auto output_options =
      v[X].options().dtype(nonzero ? at::kLong : at::ScalarType(dtype));
  auto result = at::empty(shape, output_options);
  auto center_shape = shape;
  center_shape.push_back(2);
  auto center = moments
                    ? at::empty(center_shape, v[X].options().dtype(at::kDouble))
                    : at::empty({0}, v[X].options());
#ifdef WITH_CUDA
  if (operation == "logsumexp" && !global)
    if (auto plan = fused::plan(v, s)) {
      fused::count();
      result.copy_(
          fused::logsumexp_cuda(*plan, exclusion).view(result.sizes()));
      checks.finish();
      return {result, center};
    }
#endif
  int64_t batches = 1;
  for (auto dimension : batch)
    batches *= dimension;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    Moments whole(1, v[X].options(), full_range);
    auto total =
        at::full({1}, nonzero ? 0. : -infinity,
                 v[X].options().dtype(nonzero ? at::kLong : at::kDouble));
    for (int64_t left = 0; left < n; left += size) {
      auto count = std::min(size, n - left);
      Moments row(count, v[X].options(), full_range);
      if (!global)
        total = at::full({count}, nonzero ? 0. : -infinity, total.options());
      for (int64_t right = 0; right < m; right += size) {
        auto other = std::min(size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other);
        auto value = evaluate(local, s, false, left, right);
        auto allowed =
            allowed_entries(left, count, right, other, s, value.options());
        if (global) {
          value = value.reshape({1, -1});
          allowed = allowed.reshape({1, -1});
        }
        if (nonzero)
          total = total + (value.ne(0) & allowed).sum(-1);
        else {
          if (moments)
            (global ? whole : row).merge(value.to(at::kDouble), allowed);
          else
            total = at::logaddexp(
                total, at::logsumexp(at::where(allowed, value, -infinity), -1)
                           .to(at::kDouble));
        }
      }
      if (!global) {
        auto value = total;
        if (moments) {
          auto denominator =
              (row.count.to(at::kDouble) - correction).clamp_min(0);
          value = at::where(
              row.count.gt(0), row.m2 / denominator,
              at::full_like(row.m2, std::numeric_limits<double>::quiet_NaN()));
          if (operation == "std")
            value = value.sqrt();
          lookup(center, index, 2).narrow(0, left, count).copy_(row.center());
        }
        lookup(result, index, 1).narrow(0, left, count).copy_(value);
      }
    }
    if (global) {
      auto value = total;
      if (moments) {
        auto denominator =
            (whole.count.to(at::kDouble) - correction).clamp_min(0);
        value = at::where(
            whole.count.gt(0), whole.m2 / denominator,
            at::full_like(whole.m2, std::numeric_limits<double>::quiet_NaN()));
        if (operation == "std")
          value = value.sqrt();
        lookup(center, index, 1).copy_(whole.center().squeeze(0));
      }
      lookup(result, index, 0).copy_(value.squeeze(0));
    }
  }
  checks.finish();
  return {result, center};
}
// Higher-order replay retains tile graphs, including the normalization and
// cotangent. Unlike the ordinary backward, its memory grows with all pairs.
Tensors statistics_higher_order_backward(const Tensor& gradient,
                                         at::TensorList v, Settings s,
                                         bool softmax, bool global,
                                         int64_t value_root = -1,
                                         int64_t weight_root = -1) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  utils::DeferredValueChecks checks;
  auto batch = batch_shape(v, s);
  auto n = v[s.transposed() ? Y : X].size(-2);
  auto m = v[s.transposed() ? X : Y].size(-2);
  Tensors inputs, buffers;
  std::vector<size_t> slots;
  for (size_t slot = 0; slot < v.size(); ++slot) {
    buffers.push_back(v[slot].is_floating_point() ? v[slot] * 0.
                                                  : at::zeros_like(v[slot]));
    if (v[slot].requires_grad()) {
      bool duplicate = false;
      for (const auto& input : inputs)
        duplicate |= input.is_same(v[slot]);
      if (!duplicate) {
        inputs.push_back(v[slot]);
        slots.push_back(slot);
      }
    }
  }
  auto accumulate_output = [&](const Tensor& output, const Tensor& g) {
    if (!output.requires_grad() || inputs.empty())
      return;
    auto gradients =
        torch::autograd::grad({output}, inputs, {g}, true, true, true);
    for (size_t k = 0; k < gradients.size(); ++k)
      if (gradients[k].defined())
        buffers[slots[k]] = buffers[slots[k]] + gradients[k];
  };
  int64_t batches = 1;
  for (auto dimension : batch)
    batches *= dimension;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    Tensor total;
    for (int64_t left = 0; left < n; left += s.size) {
      auto count = std::min(s.size, n - left);
      Tensors tiles, weights;
      for (int64_t right = 0; right < m; right += s.size) {
        auto other = std::min(s.size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other);
        Tensor score;
        if (weight_root >= 0) {
          auto value_settings = s, weight_settings = s;
          auto value_plan = std::make_shared<ExpressionPlan>(*s.expression);
          auto weight_plan = std::make_shared<ExpressionPlan>(*s.expression);
          value_plan->packet.nodes.resize(value_root + 1);
          weight_plan->packet.nodes.resize(weight_root + 1);
          value_settings.expression = value_plan;
          weight_settings.expression = weight_plan;
          score =
              expression_evaluate_higher_order(local, value_settings, false);
          auto weight =
              expression_evaluate_higher_order(local, weight_settings, false);
          auto allowed =
              allowed_entries(left, count, right, other, s, score.options());
          weights.push_back(at::where(allowed, weight, 0.));
          tiles.push_back(at::where(allowed & weight.ne(0), score, 0.));
          continue;
        }
        score = evaluate_higher_order(local, s, false, left, right);
        auto allowed =
            allowed_entries(left, count, right, other, s, score.options());
        tiles.push_back(at::where(allowed, score, -infinity));
      }
      auto scores = tiles.empty() ? at::empty({count, 0}, v[X].options())
                                  : at::cat(tiles, -1);
      if (weight_root >= 0) {
        auto w =
            weights.empty() ? at::empty_like(scores) : at::cat(weights, -1);
        utils::check_values(
            (w.isfinite() & w.ge(0)).all(),
            "Weighted statistics require finite nonnegative weights");
        auto numerator = (scores.to(at::kDouble) * w.to(at::kDouble)).sum(-1);
        auto mass = w.to(at::kDouble).sum(-1);
        if (global) {
          auto partial = at::stack({numerator.sum(), mass.sum()});
          total = total.defined() ? total + partial : partial;
        } else {
          auto output = (numerator / mass).to(at::ScalarType(s.dtype));
          accumulate_output(output,
                            lookup(gradient, index, 1).narrow(0, left, count));
        }
      } else if (softmax) {
        auto payload = lookup(v[Payload], index, 2).to(scores.scalar_type());
        auto probabilities = scores.softmax(-1);
        // Entirely excluded rows have zero mass by the operator contract.
        if (s.exclusion && m == 1)
          probabilities = at::zeros_like(probabilities);
        auto output = probabilities.matmul(payload).to(at::ScalarType(s.dtype));
        auto g = lookup(gradient, index, 2).narrow(0, left, count);
        accumulate_output(output, g);
      } else if (global) {
        auto partial = scores.reshape({-1}).logsumexp(0);
        total = total.defined() ? at::logaddexp(total, partial) : partial;
      } else {
        auto output = scores.logsumexp(-1).to(at::ScalarType(s.dtype));
        auto g = lookup(gradient, index, 1).narrow(0, left, count);
        accumulate_output(output, g);
      }
    }
    if (global && total.defined())
      accumulate_output((weight_root >= 0 ? total[0] / total[1] : total)
                            .to(at::ScalarType(s.dtype)),
                        lookup(gradient, index, 0));
  }
  checks.finish();
  return buffers;
}
Tensors expression_statistics_backward_kernel(
    const Tensor& gradient, at::TensorList v, const Tensor& result,
    const Tensor& center, TORCHPAIRWISE_EXPRESSION_DECL, double correction) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  utils::DeferredValueChecks checks;
  auto s = statistics_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed);
  if (at::GradMode::is_enabled()) {
    TORCH_CHECK(
        operation == "logsumexp",
        "Higher derivatives of this streamed statistic are unsupported");
    return statistics_higher_order_backward(gradient, v, s, false, form);
  }
#ifdef WITH_CUDA
  if (operation == "logsumexp" && !form)
    if (auto plan = fused::plan(v, s))
      if (plan->differentiable &&
          plan->rows.size(2) <= fused::max_gradient_features) {
        fused::count();
        // Cotangent g * exp(value - output), with the rounded output.
        auto ones = at::ones_like(result);
        fused::Probability rows{result, ones, Tensor(), false},
            columns{result, ones, Tensor(), true};
        auto left = gradient.unsqueeze(-1),
             right = at::ones({plan->columns.size(1), 1}, gradient.options());
        Tensors fused_buffers;
        for (const auto& value : v)
          fused_buffers.push_back(at::zeros_like(value));
        fused::operand_gradients(*plan, left, right, exclusion, rows, columns,
                                 fused_buffers, v);
        checks.finish();
        return fused_buffers;
      }
#endif
  Tensors buffers;
  for (const auto& value : v)
    buffers.push_back(at::zeros_like(
        value, value.options().dtype(value.is_floating_point()
                                         ? at::kDouble
                                         : value.scalar_type())));
  auto batch = batch_shape(v, s);
  int64_t batches = 1;
  for (auto dimension : batch)
    batches *= dimension;
  bool global = form;
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  double eligible = double(global ? n : 1) * double(m) -
                    (exclusion ? (global ? double(n) : 1.) : 0.);
  // PyTorch's explicit-axis backward uses the signed degrees of freedom,
  // even though forward clamps it to zero. Preserve its degenerate behavior.
  double denominator = eligible - correction;
  at::AutoGradMode enable(true);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    for (int64_t left = 0; left < n; left += size) {
      auto count = std::min(size, n - left);
      auto select = [&](const Tensor& tensor) {
        return global ? lookup(tensor, index, 0).reshape({1, 1})
                      : lookup(tensor, index, 1)
                            .narrow(0, left, count)
                            .unsqueeze(-1);
      };
      auto g = select(gradient).to(at::kDouble);
      auto output = select(result).to(at::kDouble);
      // Accept the former single-mean auxiliary tensor in direct calls to
      // the existing backward schema; new forwards save anchor/offset pairs.
      bool anchored = operation != "logsumexp" &&
                      center.dim() == result.dim() + 1 && center.size(-1) == 2;
      auto average = operation == "logsumexp" ? output
                     : anchored               ? select(center.select(-1, 1))
                                              : select(center);
      auto anchor = anchored ? select(center.select(-1, 0)) : Tensor();
      if (operation == "std")
        g = at::where(output.eq(0), 0., g / (2. * output));
      for (int64_t right = 0; right < m; right += size) {
        auto other = std::min(size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other,
                                  Tensor(), Tensor(), true);
        ExpressionReplay replay;
        auto value =
            evaluate(local, s, false, left, right, Tensor(), Tensor(), &replay);
        Tensor cotangent;
        if (operation == "logsumexp")
          cotangent = g.to(value.scalar_type()) *
                      (value - output.to(value.scalar_type())).exp();
        else {
          value = value.to(at::kDouble);
          cotangent = g * (2. / denominator) *
                      (anchored ? (value - anchor) - average : value - average);
        }
        auto allowed =
            allowed_entries(left, count, right, other, s, value.options());
        cotangent = at::where(allowed, cotangent, 0.);
        accumulate(local, value, cotangent, buffers, s, index, left, count,
                   right, other, Tensor(), Tensor(), &replay);
      }
    }
  }
  for (size_t slot = 0; slot < v.size(); ++slot)
    buffers[slot] = buffers[slot].to(v[slot].scalar_type()).contiguous();
  checks.finish();
  return buffers;
}
namespace {
StatisticsOutputs legacy_statistics_kernel(at::TensorList v,
                                           TORCHPAIRWISE_EXPRESSION_DECL,
                                           double correction) {
  auto [result, center] = expression_statistics_kernel(
      v, names, families, headers, reals, integers, flags, nodes, numbers,
      operation, dtype, size, exclusion, transposed, form, correction);
  // Keep serialized/default-op metadata stable for existing compiled graphs.
  if (operation == "var" || operation == "std")
    center = center.sum(-1);
  return {result, center};
}
}  // namespace
Tensor expression_statistics(at::TensorList v, const Settings& s,
                             c10::string_view reduction, bool global,
                             double correction) {
  const auto& p = s.expression->packet;
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::_expression_statistics", "anchored")
                       .typed<StatisticsSignature>();
  return std::get<0>(op.call(v, p.names, p.families, p.headers, p.reals,
                             p.integers, p.flags, p.nodes, p.numbers, reduction,
                             s.dtype, s.size, s.exclusion, s.transposed(),
                             global, correction));
}
namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_expression_statistics(Tensor[] operands, str[] names, str[] families, "
      "int[][] headers, "
      "float?[][] reals, int?[][] integers, bool?[][] flags, int[][] nodes, "
      "Scalar[] numbers, "
      "str operation, int dtype, int size, bool exclusion, bool transposed, "
      "bool global_reduction, "
      "float correction) -> (Tensor, Tensor)");
  m.def(
      "_expression_statistics.anchored(Tensor[] operands, str[] names, str[] "
      "families, "
      "int[][] headers, float?[][] reals, int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, bool transposed, bool global_reduction, float "
      "correction) "
      "-> (Tensor, Tensor)");
  m.def(
      "_expression_statistics_backward(Tensor grad, Tensor[] operands, Tensor "
      "result, Tensor center, "
      "str[] names, str[] families, int[][] headers, float?[][] reals, "
      "int?[][] integers, "
      "bool?[][] flags, int[][] nodes, Scalar[] numbers, str operation, int "
      "dtype, int size, "
      "bool exclusion, bool transposed, bool global_reduction, float "
      "correction) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_expression_statistics", TORCH_FN(legacy_statistics_kernel));
  m.impl("_expression_statistics.anchored",
         TORCH_FN(expression_statistics_kernel));
  m.impl("_expression_statistics_backward",
         TORCH_FN(expression_statistics_backward_kernel));
}
}  // namespace
}  // namespace torchpairwise::ops::detail::streaming
