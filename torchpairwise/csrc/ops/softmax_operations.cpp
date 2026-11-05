#include <torch/autograd.h>
#include <torch/library.h>

#include <limits>

#include "fused_reducers.h"
#include "softmax_operations_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::streaming {
Tensors statistics_higher_order_backward(const Tensor& gradient,
                                         at::TensorList v, Settings s,
                                         bool softmax, bool global,
                                         int64_t value_root = -1,
                                         int64_t weight_root = -1);
namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();
Settings softmax_settings(TORCHPAIRWISE_EXPRESSION_DECL) {
  TORCH_CHECK(operation == "softmax_matmul" && !form,
              "Unknown softmax operation");
  auto s = statistics_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      "logsumexp", dtype, size, exclusion, transposed);
  s.mask |= int64_t(1) << Payload;
  return s;
}
Tensor allowed_entries(int64_t left, int64_t count, int64_t right,
                       int64_t other, const Settings& s,
                       const at::TensorOptions& options) {
  if (!s.exclusion)
    return at::ones({count, other}, options.dtype(at::kBool));
  return at::arange(left, left + count, options.dtype(at::kLong))
      .unsqueeze(-1)
      .ne(at::arange(right, right + other, options.dtype(at::kLong)));
}
Tensor probabilities(const Tensor& scores, const Tensor& state,
                     const Tensor& allowed) {
  auto maximum = state.select(-1, 0).unsqueeze(-1);
  // An all -inf row has zero mass and yields NaN, as in torch.softmax.
  // An entirely excluded row instead has zero probabilities by definition.
  // Every finite maximum was selected from scores, so this cast is exact.
  auto shift =
      at::where(maximum.eq(-infinity), 0., maximum).to(scores.scalar_type());
  return at::where(
      allowed,
      (scores - shift).exp() /
          state.select(-1, 1).unsqueeze(-1).to(scores.scalar_type()),
      0.);
}
#ifdef WITH_CUDA
// Fused softmax weighting from a [..., n, 2] (maximum, mass) state, shifted
// as probabilities() shifts it.
fused::Probability probability_state(const Tensor& state, bool on_columns,
                                     const Tensor& center = Tensor()) {
  auto maximum = state.select(-1, 0);
  return {at::where(maximum.eq(-infinity), 0., maximum), state.select(-1, 1),
          center, on_columns};
}
#endif
}  // namespace
StatisticsOutputs softmax_matmul_kernel(at::TensorList v,
                                        TORCHPAIRWISE_EXPRESSION_DECL) {
  auto s = softmax_settings(TORCHPAIRWISE_EXPRESSION_ARGS);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  utils::DeferredValueChecks checks;
  auto batch = batch_shape(v, s), shape = batch, state_shape = batch;
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  auto channels = v[Payload].size(-1);
  shape.insert(shape.end(), {n, channels});
  state_shape.insert(state_shape.end(), {n, 2});
  auto result = at::zeros(shape, v[X].options().dtype(at::ScalarType(dtype)));
  auto state = at::empty(state_shape, v[X].options().dtype(at::kDouble));
  if (exclusion && m == 1) {
    state.select(-1, 0).fill_(-infinity);
    state.select(-1, 1).zero_();
    checks.finish();
    return {result, state};
  }
#ifdef WITH_CUDA
  if (auto plan = fused::plan(v, s)) {
    fused::count();
    auto fused_state = fused::softmax_state_cuda(*plan, exclusion);
    result.copy_(fused::matmul_cuda(*plan, v[Payload], exclusion,
                                    probability_state(fused_state, false))
                     .view(result.sizes()));
    state.copy_(fused_state.view(state.sizes()));
    checks.finish();
    return {result, state};
  }
#endif
  int64_t batches = 1;
  for (auto dimension : batch)
    batches *= dimension;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto payload = lookup(v[Payload], index, 2);
    for (int64_t left = 0; left < n; left += size) {
      auto count = std::min(size, n - left);
      auto maximum = at::full({count}, -infinity, state.options());
      auto mass = at::zeros_like(maximum);
      for (int64_t right = 0; right < m; right += size) {
        auto other = std::min(size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other);
        auto score = evaluate(local, s, false, left, right);
        auto allowed =
            allowed_entries(left, count, right, other, s, score.options());
        score = at::where(allowed, score, -infinity);
        auto next =
            at::maximum(maximum, std::get<0>(score.max(-1)).to(at::kDouble));
        auto shift = at::where(next.eq(-infinity), 0., next);
        auto factor = (maximum - shift).exp();
        // A maximum of score values is exactly representable in score dtype.
        auto weights =
            (score - shift.to(score.scalar_type()).unsqueeze(-1)).exp();
        mass = mass * factor + weights.sum(-1).to(at::kDouble);
        maximum = next;
      }
      auto row_state = lookup(state, index, 2).narrow(0, left, count);
      row_state.select(-1, 0).copy_(maximum);
      row_state.select(-1, 1).copy_(mass);
      // Retain normalized two-pass contraction even for one channel tile:
      // an online unnormalized numerator can overflow for finite payloads
      // whose normalized result is representable (including float64).
      // Channel tiles bound intermediates even for very wide payloads. Scores
      // are replayed per channel tile instead of retaining a dense probability.
      for (int64_t channel = 0; channel < channels; channel += size) {
        auto width = std::min(size, channels - channel);
        auto total = at::zeros({count, width}, state.options());
        auto correction = at::zeros_like(total);
        for (int64_t right = 0; right < m; right += size) {
          auto other = std::min(size, m - right);
          auto local = local_inputs(v, s, index, left, count, right, other);
          auto score = evaluate(local, s, false, left, right);
          auto allowed =
              allowed_entries(left, count, right, other, s, score.options());
          auto p = probabilities(score, row_state, allowed);
          auto values = payload.narrow(0, right, other)
                            .narrow(-1, channel, width)
                            .to(score.scalar_type());
          auto update =
              compensated(total, correction, p.matmul(values).to(at::kDouble));
          total = update.first;
          correction = update.second;
        }
        lookup(result, index, 2)
            .narrow(0, left, count)
            .narrow(-1, channel, width)
            .copy_(total + correction);
      }
    }
  }
  checks.finish();
  return {result, state};
}
Tensors softmax_matmul_backward_kernel(const Tensor& gradient, at::TensorList v,
                                       const Tensor& result,
                                       const Tensor& state,
                                       TORCHPAIRWISE_EXPRESSION_DECL) {
  auto s = softmax_settings(TORCHPAIRWISE_EXPRESSION_ARGS);
  if (at::GradMode::is_enabled())
    return statistics_higher_order_backward(gradient, v, s, true, false);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  utils::DeferredValueChecks checks;
  Tensors buffers;
  for (const auto& value : v)
    buffers.push_back(at::zeros_like(
        value, value.options().dtype(value.is_floating_point()
                                         ? at::kDouble
                                         : value.scalar_type())));
  auto batch = batch_shape(v, s);
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  auto channels = v[Payload].size(-1);
  if (exclusion && m == 1) {
    for (size_t slot = 0; slot < v.size(); ++slot)
      buffers[slot] = buffers[slot].to(v[slot].scalar_type()).contiguous();
    checks.finish();
    return buffers;
  }
#ifdef WITH_CUDA
  if (auto plan = fused::plan(v, s))
    if (plan->differentiable &&
        plan->rows.size(2) <= fused::max_gradient_features) {
      fused::count();
      // Cotangent p * (g . v - center) with the center from the output.
      auto center = (gradient.to(at::kDouble) * result.to(at::kDouble)).sum(-1);
      auto rows = probability_state(state, false, center),
           columns = probability_state(state, true, center);
      auto transposed = fused::swapped(*plan);
      const auto& payload = v[Payload];
      Tensors fused_buffers;
      for (const auto& value : v)
        fused_buffers.push_back(at::zeros_like(value));
      fused::operand_gradients(*plan, gradient, payload, exclusion, rows,
                               columns, fused_buffers, v);
      auto shape = plan->batch;
      shape.insert(shape.end(), {payload.size(-2), payload.size(-1)});
      fused_buffers[Payload] =
          at::sum_to(
              fused::matmul_cuda(transposed, gradient, exclusion, columns)
                  .reshape(shape),
              payload.sizes())
              .contiguous();
      checks.finish();
      return fused_buffers;
    }
#endif
  int64_t batches = 1;
  for (auto dimension : batch)
    batches *= dimension;
  at::AutoGradMode enable(true);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto payload = lookup(v[Payload], index, 2).detach();
    for (int64_t left = 0; left < n; left += size) {
      auto count = std::min(size, n - left);
      auto row_state = lookup(state, index, 2).narrow(0, left, count);
      auto g = lookup(gradient, index, 2).narrow(0, left, count).detach();
      // Recompute the softmax VJP center from tile probabilities, avoiding the
      // rounded output. Sum tile/channel contributions in float64.
      auto center = at::zeros({count, 1}, state.options());
      for (int64_t right = 0; right < m; right += size) {
        auto other = std::min(size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other);
        auto score = evaluate(local, s, false, left, right);
        auto allowed =
            allowed_entries(left, count, right, other, s, score.options());
        auto p = probabilities(score, row_state, allowed);
        for (int64_t channel = 0; channel < channels; channel += size) {
          auto width = std::min(size, channels - channel);
          auto gp = g.narrow(-1, channel, width).to(score.scalar_type());
          auto values = payload.narrow(0, right, other)
                            .narrow(-1, channel, width)
                            .to(score.scalar_type());
          center.add_((p * gp.matmul(values.transpose(0, 1)))
                          .sum(-1, true)
                          .to(at::kDouble));
        }
      }
      for (int64_t right = 0; right < m; right += size) {
        auto other = std::min(size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other,
                                  Tensor(), Tensor(), true);
        ExpressionReplay replay;
        auto score =
            evaluate(local, s, false, left, right, Tensor(), Tensor(), &replay);
        auto allowed =
            allowed_entries(left, count, right, other, s, score.options());
        auto p = probabilities(score, row_state, allowed);
        auto cotangent = at::zeros(p.sizes(), state.options());
        for (int64_t channel = 0; channel < channels; channel += size) {
          auto width = std::min(size, channels - channel);
          auto gp = g.narrow(-1, channel, width).to(score.scalar_type());
          auto values = payload.narrow(0, right, other)
                            .narrow(-1, channel, width)
                            .to(score.scalar_type());
          cotangent.add_(gp.matmul(values.transpose(0, 1)).to(at::kDouble));
          lookup(buffers[Payload], index, 2)
              .narrow(0, right, other)
              .narrow(-1, channel, width)
              .add_(p.transpose(0, 1).matmul(gp).to(at::kDouble));
        }
        cotangent = at::where(allowed,
                              p * (cotangent.to(score.scalar_type()) -
                                   center.to(score.scalar_type())),
                              0.);
        accumulate(local, score, cotangent, buffers, s, index, left, count,
                   right, other, Tensor(), Tensor(), &replay);
      }
    }
  }
  for (size_t slot = 0; slot < v.size(); ++slot)
    buffers[slot] = buffers[slot].to(v[slot].scalar_type()).contiguous();
  checks.finish();
  return buffers;
}
Tensor expression_softmax_matmul(at::TensorList operands, const Settings& s) {
  const auto& p = s.expression->packet;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_expression_softmax_matmul", "")
          .typed<SoftmaxSignature>();
  return std::get<0>(op.call(operands, p.names, p.families, p.headers, p.reals,
                             p.integers, p.flags, p.nodes, p.numbers,
                             "softmax_matmul", s.dtype, s.size, s.exclusion,
                             s.transposed(), false));
}
namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_expression_softmax_matmul(Tensor[] operands, str[] names, str[] "
      "families, "
      "int[][] headers, float?[][] reals, int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, bool transposed, bool form) -> (Tensor, Tensor)");
  m.def(
      "_expression_softmax_matmul_backward(Tensor grad, Tensor[] operands, "
      "Tensor result, Tensor state, str[] names, str[] families, int[][] "
      "headers, "
      "float?[][] reals, int?[][] integers, bool?[][] flags, int[][] nodes, "
      "Scalar[] numbers, str operation, int dtype, int size, bool exclusion, "
      "bool transposed, bool form) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_expression_softmax_matmul", TORCH_FN(softmax_matmul_kernel));
  m.impl("_expression_softmax_matmul_backward",
         TORCH_FN(softmax_matmul_backward_kernel));
}
}  // namespace
}  // namespace torchpairwise::ops::detail::streaming
