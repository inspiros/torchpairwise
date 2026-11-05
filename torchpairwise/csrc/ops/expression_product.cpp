#include <ATen/ExpandUtils.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include "expression_statistics_detail.h"
#include "residual_reducer.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::streaming {
namespace {
Settings product_settings(const ExpressionPacket& packet, int64_t dtype,
                          int64_t size, bool exclusion, bool transposed,
                          c10::string_view operation, double correction) {
  TORCH_CHECK(operation == "prod" && correction == 0.,
              "Invalid product consumer metadata");
  return statistics_settings(packet, "var", dtype, size, exclusion, transposed);
}
Tensor allowed_entries(int64_t i, int64_t n, int64_t j, int64_t m,
                       const Settings& s, const at::TensorOptions& options) {
  if (!s.exclusion)
    return at::ones({n, m}, options.dtype(at::kBool));
  return at::arange(i, i + n, options.dtype(at::kLong))
      .unsqueeze(-1)
      .ne(at::arange(j, j + m, options.dtype(at::kLong)));
}
struct ProductState {
  Tensor raw, state;
  ProductState(int64_t count, const at::TensorOptions& options) {
    raw = at::ones({count}, options);
    state = at::zeros({count, 5}, options.dtype(at::kDouble));
    state.select(-1, 1).fill_(1.);
  }
  void merge(const Tensor& entries) {
    auto value = entries.to(at::kDouble);
    auto zero = value.eq(0), inf = value.isinf(), nan = value.isnan();
    raw.mul_(entries.prod(-1));
    state.select(-1, 0).add_(zero.sum(-1));
    state.select(-1, 1).mul_(at::where(zero | inf | nan, 1., value).prod(-1));
    state.select(-1, 2).add_(value.lt(0).sum(-1));
    state.select(-1, 3).add_(inf.sum(-1));
    state.select(-1, 4).add_(nan.sum(-1));
  }
};
}  // namespace
StatisticsOutputs expression_product_kernel(at::TensorList v,
                                            TORCHPAIRWISE_EXPRESSION_DECL,
                                            double correction) {
  auto s = product_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      dtype, size, exclusion, transposed, operation, correction);
  at::NoGradGuard guard;
  utils::DeferredValueChecks checks;
  auto batch = batch_shape(v, s), shape = batch;
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  bool global = form;
  if (!global)
    shape.push_back(n);
  auto options = v[X].options().dtype(at::ScalarType(dtype));
  auto result = at::ones(shape, options);
  shape.push_back(5);
  auto state = at::zeros(shape, options.dtype(at::kDouble));
  int64_t batches = 1;
  for (auto d : batch)
    batches *= d;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    ProductState whole(1, options);
    for (int64_t i = 0; i < n; i += size) {
      auto count = std::min(size, n - i);
      ProductState row(count, options);
      for (int64_t j = 0; j < m; j += size) {
        auto other = std::min(size, m - j);
        auto local = local_inputs(v, s, index, i, count, j, other);
        auto value = evaluate(local, s, false, i, j).expand({count, other});
        value = at::where(allowed_entries(i, count, j, other, s, options),
                          value, 1.);
        (global ? whole : row).merge(global ? value.reshape({1, -1}) : value);
      }
      if (!global) {
        lookup(result, index, 1).narrow(0, i, count).copy_(row.raw);
        lookup(state, index, 2).narrow(0, i, count).copy_(row.state);
      }
    }
    if (global) {
      lookup(result, index, 0).copy_(whole.raw.squeeze(0));
      lookup(state, index, 1).copy_(whole.state.squeeze(0));
    }
  }
  checks.finish();
  return {result, state};
}
Tensors expression_product_backward_kernel(
    const Tensor& gradient, at::TensorList v, const Tensor& result,
    const Tensor& state, TORCHPAIRWISE_EXPRESSION_DECL, double correction) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto s = product_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      dtype, size, exclusion, transposed, operation, correction);
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
  bool global = form;
  // Match ATen's whole-input zero predicate, including other output rows and
  // batches. This stays on-device and can change during CUDA graph replay.
  auto any_zero = state.select(-1, 0).ne(0).any();
  at::AutoGradMode enable(true);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    for (int64_t i = 0; i < n; i += size) {
      auto count = std::min(size, n - i);
      auto saved = lookup(state, index, global ? 1 : 2);
      auto total = lookup(result, index, global ? 0 : 1).to(at::kDouble);
      auto grad = lookup(gradient, index, global ? 0 : 1).to(at::kDouble);
      if (!global) {
        saved = saved.narrow(0, i, count).unsqueeze(-2);
        total = total.narrow(0, i, count).unsqueeze(-1);
        grad = grad.narrow(0, i, count).unsqueeze(-1);
      } else {
        total = total.reshape({1, 1});
        grad = grad.reshape({1, 1});
      }
      auto field = [&](int64_t k) { return saved.select(-1, k); };
      for (int64_t j = 0; j < m; j += size) {
        auto other = std::min(size, m - j);
        auto local = local_inputs(v, s, index, i, count, j, other, Tensor(),
                                  Tensor(), true);
        ExpressionReplay replay;
        auto value =
            evaluate(local, s, false, i, j, Tensor(), Tensor(), &replay)
                .expand({count, other})
                .to(at::kDouble);
        auto zero = value.eq(0), inf = value.isinf(), nan = value.isnan();
        auto finite = field(1) / at::where(zero | inf | nan, 1., value);
        auto exclusive = residual::product_value(
            field(0) - zero.to(at::kDouble), finite,
            field(2) - value.lt(0).to(at::kDouble),
            field(3) - inf.to(at::kDouble), field(4) - nan.to(at::kDouble));
        auto derivative =
            at::where(any_zero, exclusive, total / at::where(zero, 1., value));
        auto cotangent =
            at::where(allowed_entries(i, count, j, other, s, value.options()),
                      grad * derivative, 0.);
        accumulate(local, value, cotangent, buffers, s, index, i, count, j,
                   other, Tensor(), Tensor(), &replay);
      }
    }
  }
  for (size_t k = 0; k < v.size(); ++k)
    buffers[k] = buffers[k].to(v[k].scalar_type()).contiguous();
  checks.finish();
  return buffers;
}
Outputs streaming_product(at::TensorList v, const Settings& original,
                          bool global) {
  auto values = v.vec();
  auto s = original;
  if (!s.expression) {
    // Adapt an already-prepared built-in source; do not prepare it again or
    // call its public wrapper. The common expression engine owns its VJP.
    auto batch =
        at::infer_size_symint(v[X].sym_sizes().slice(0, v[X].dim() - s.rank),
                              v[Y].sym_sizes().slice(0, v[Y].dim() - s.rank));
    for (int64_t slot = W; slot < Slots; ++slot)
      if (s.has(slot) && s.operand_rank(slot) > 0)
        batch = at::infer_size_symint(
            batch,
            v[slot].sym_sizes().slice(0, v[slot].dim() - s.operand_rank(slot)));
    auto n = v[s.transposed() ? Y : X].sym_size(-s.rank);
    auto m = v[s.transposed() ? X : Y].sym_size(-s.rank);
    ExpressionPacket packet{{s.metric},
                            {s.family.value_or("")},
                            {{s.rank, s.size, s.dtype, s.kernel, s.self_cosine,
                              s.transposed(), s.mask}},
                            {{0, 0, -1}},
                            {s.reals},
                            {s.integers},
                            {s.flags},
                            {}};
    auto xs = batch, ys = batch;
    xs.insert(xs.end(), {n, 0});
    ys.insert(ys.end(), {m, 0});
    values.assign(
        Slots, at::empty({0}, v[X].options().dtype(at::ScalarType(s.dtype))));
    values[X] = at::empty_symint(xs, values[0].options());
    values[Y] = at::empty_symint(ys, values[0].options());
    values[XL] = at::arange(n, v[X].options().dtype(at::kLong));
    values[YL] = at::arange(m, v[X].options().dtype(at::kLong));
    values.insert(values.end(), v.begin(), v.end());
    s = statistics_settings(packet, "var", s.dtype, s.size, s.exclusion, false);
  }
  const auto& p = s.expression->packet;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_expression_product", "")
          .typed<StatisticsSignature>();
  auto result = std::get<0>(op.call(values, p.names, p.families, p.headers,
                                    p.reals, p.integers, p.flags, p.nodes,
                                    p.numbers, "prod", s.dtype, s.size,
                                    s.exclusion, s.transposed(), global, 0.));
  auto empty = at::empty({0}, result.options());
  return {result, empty.to(at::kLong), empty.to(at::kLong), empty};
}
namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_expression_product(Tensor[] operands, str[] names, str[] families, "
      "int[][] headers, float?[][] reals, int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, bool transposed, bool global_reduction, "
      "float correction) -> (Tensor, Tensor)");
  m.def(
      "_expression_product_backward(Tensor grad, Tensor[] operands, Tensor "
      "result, "
      "Tensor state, str[] names, str[] families, int[][] headers, float?[][] "
      "reals, "
      "int?[][] integers, bool?[][] flags, int[][] nodes, Scalar[] numbers, "
      "str operation, int dtype, int size, bool exclusion, bool transposed, "
      "bool global_reduction, float correction) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_expression_product", TORCH_FN(expression_product_kernel));
  m.impl("_expression_product_backward",
         TORCH_FN(expression_product_backward_kernel));
}
}  // namespace
}  // namespace torchpairwise::ops::detail::streaming
