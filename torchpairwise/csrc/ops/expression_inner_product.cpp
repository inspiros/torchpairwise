#include <torch/autograd.h>
#include <torch/library.h>

#include "expression_statistics_detail.h"

namespace torchpairwise::ops::detail::streaming {
namespace {
Settings inner_settings(const ExpressionPacket& packet, int64_t dtype,
                        int64_t size, bool exclusion, bool transposed,
                        bool form, int64_t left_root, int64_t right_root) {
  TORCH_CHECK(dtype == int64_t(at::kFloat) || dtype == int64_t(at::kDouble),
              "Frobenius inner products require floating entries");
  TORCH_CHECK(!exclusion && !transposed && !form,
              "Unused Frobenius inner product metadata");
  TORCH_CHECK(left_root >= 0 && right_root >= 0 &&
                  left_root < int64_t(packet.nodes.size()) &&
                  right_root < int64_t(packet.nodes.size()),
              "Invalid Frobenius expression roots");
  return statistics_settings(packet, "var", dtype, size, false, false);
}
Tensors inner_entries(at::TensorList local, const Settings& s, int64_t a,
                      int64_t b, int64_t n, int64_t m,
                      ExpressionReplay* replay = nullptr) {
  auto roots = expression_evaluate_roots(local, s, {a, b}, replay);
  for (auto& root : roots)
    root = root.expand({n, m}).to(at::kDouble);
  return roots;
}
// Each source keeps row/column mean offsets, an anchor and a grand mean
// offset. Never collapse a large absolute anchor into its small mean offset.
// The final five fields are dot, squared norms and two normalization scales.
Tensor off_diagonal(int64_t i, int64_t ni, int64_t j, int64_t nj,
                    const at::TensorOptions& options) {
  return at::arange(i, i + ni, options.dtype(at::kLong))
      .unsqueeze(-1)
      .ne(at::arange(j, j + nj, options.dtype(at::kLong)));
}
double hsic_denominator(c10::string_view operation, int64_t n, int64_t m,
                        bool center, bool normalize) {
  TORCH_CHECK(operation == "inner_product" || operation == "hsic" ||
                  operation == "hsic_unbiased",
              "Unknown coordinated statistic");
  if (operation == "inner_product")
    return 1.;
  TORCH_CHECK(center && !normalize, "Unused HSIC metadata");
  TORCH_CHECK(n == m, "HSIC requires square Gram operators");
  bool unbiased = operation == "hsic_unbiased";
  TORCH_CHECK(n >= (unbiased ? 4 : 2),
              unbiased ? "Unbiased HSIC requires at least 4 observations"
                       : "Biased HSIC requires at least 2 observations");
  return unbiased ? double(n) * double(n - 3) : double(n - 1) * double(n - 1);
}
Tensor centered_entry(const Tensor& value, const Tensor& state, int64_t source,
                      int64_t n, int64_t m, int64_t i, int64_t ni, int64_t j,
                      int64_t nj) {
  auto base = 2 * (n + m);
  auto rows = state.narrow(0, source * (n + m), n).narrow(0, i, ni);
  auto columns = state.narrow(0, source * (n + m) + n, m).narrow(0, j, nj);
  return ((value - state[base + source]) - rows.unsqueeze(-1)) -
         (columns.unsqueeze(0) - state[base + 2 + source]);
}
Tensor u_centered_adjoint(const Tensor& value, const Tensor& state,
                          int64_t source, int64_t n, int64_t i, int64_t ni,
                          int64_t j, int64_t nj) {
  auto rows = state.narrow(0, source * (2 * n), n);
  auto columns = state.narrow(0, source * (2 * n) + n, n);
  // U is self-adjoint but not idempotent for nonsymmetric matrices. This
  // extra projection vanishes for symmetric Gram sources; retain it so the
  // defined U-centered product also has correct unconstrained matrix VJPs.
  auto row_delta =
      (columns.narrow(0, i, ni) - rows.narrow(0, i, ni)) / double(n - 2);
  auto column_delta =
      (rows.narrow(0, j, nj) - columns.narrow(0, j, nj)) / double(n - 2);
  return at::where(off_diagonal(i, ni, j, nj, value.options()),
                   value - row_delta.unsqueeze(-1) - column_delta.unsqueeze(0),
                   0.);
}
}  // namespace
StatisticsOutputs inner_product_kernel(at::TensorList v,
                                       TORCHPAIRWISE_EXPRESSION_DECL,
                                       int64_t left_root, int64_t right_root,
                                       bool center, bool normalize) {
  auto s = inner_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      dtype, size, exclusion, transposed, form, left_root, right_root);
  at::NoGradGuard guard;
  auto batch = batch_shape(v, s);
  auto n = v[X].size(-2), m = v[Y].size(-2);
  auto divisor = hsic_denominator(operation, n, m, center, normalize);
  bool unbiased = operation == "hsic_unbiased";
  auto result = at::zeros(batch, v[X].options().dtype(at::ScalarType(dtype)));
  auto shape = batch;
  auto base = center ? 2 * (n + m) + 4 : 0;
  shape.push_back(base + 5);
  auto state = at::zeros(shape, v[X].options().dtype(at::kDouble));
  int64_t batches = 1;
  for (auto d : batch)
    batches *= d;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto saved = lookup(state, index, 1);
    if (center && n > 0 && m > 0) {
      // First pass prepares O(N+M) centering state. An origin is chosen once
      // per batch/source; no data-dependent host branch or device sync.
      if (unbiased) {
        auto local = local_inputs(v, s, index, 0, 1, 1, 1);
        auto origin = inner_entries(local, s, left_root, right_root, 1, 1);
        for (int64_t k = 0; k < 2; ++k)
          saved[2 * (n + m) + k].copy_(
              at::where(origin[k][0][0].isfinite(), origin[k][0][0], 0.));
      }
      for (int64_t i = 0; i < n; i += size) {
        auto ni = std::min(size, n - i);
        for (int64_t j = 0; j < m; j += size) {
          auto nj = std::min(size, m - j);
          auto local = local_inputs(v, s, index, i, ni, j, nj);
          auto entries = inner_entries(local, s, left_root, right_root, ni, nj);
          for (int64_t k = 0; k < 2; ++k) {
            auto anchor = saved[2 * (n + m) + k];
            if (!unbiased && i == 0 && j == 0)
              anchor.copy_(
                  at::where(entries[k][0][0].isfinite(), entries[k][0][0], 0.));
            auto shifted =
                unbiased ? at::where(
                               off_diagonal(i, ni, j, nj, entries[k].options()),
                               entries[k], anchor) -
                               anchor
                         : entries[k] - anchor;
            saved.narrow(0, k * (n + m), n)
                .narrow(0, i, ni)
                .add_((shifted / double(unbiased ? n - 2 : m)).sum(-1));
            saved.narrow(0, k * (n + m) + n, m)
                .narrow(0, j, nj)
                .add_((shifted / double(unbiased ? n - 2 : n)).sum(-2));
          }
        }
      }
      for (int64_t k = 0; k < 2; ++k)
        saved[2 * (n + m) + 2 + k].copy_(
            (saved.narrow(0, k * (n + m), n) / double(unbiased ? n - 1 : n))
                .sum());
    }
    // Inner products use centered entries rather than subtracting large raw
    // squared sums. The two roots share leaf evaluation within each tile.
    for (int64_t i = 0; i < n; i += size) {
      auto ni = std::min(size, n - i);
      for (int64_t j = 0; j < m; j += size) {
        auto nj = std::min(size, m - j);
        auto local = local_inputs(v, s, index, i, ni, j, nj);
        auto entries = inner_entries(local, s, left_root, right_root, ni, nj);
        if (center)
          for (int64_t k = 0; k < 2; ++k)
            entries[k] =
                centered_entry(entries[k], saved, k, n, m, i, ni, j, nj);
        if (unbiased)
          for (auto& entry : entries)
            entry = at::where(off_diagonal(i, ni, j, nj, entry.options()),
                              entry, 0.);
        if (normalize) {
          auto a = at::maximum(saved[base + 3], entries[0].abs().amax());
          auto b = at::maximum(saved[base + 4], entries[1].abs().amax());
          auto safe_a = at::where(a.eq(0), 1., a);
          auto safe_b = at::where(b.eq(0), 1., b);
          auto ratio_a = saved[base + 3] / safe_a;
          auto ratio_b = saved[base + 4] / safe_b;
          saved[base].mul_(ratio_a * ratio_b);
          saved[base + 1].mul_(ratio_a.square());
          saved[base + 2].mul_(ratio_b.square());
          entries[0] = entries[0] / safe_a;
          entries[1] = entries[1] / safe_b;
          saved[base + 3].copy_(a);
          saved[base + 4].copy_(b);
          saved[base + 1].add_(entries[0].square().sum());
          saved[base + 2].add_(entries[1].square().sum());
        }
        saved[base].add_((entries[0] * entries[1]).sum());
      }
    }
    auto value = saved[base];
    if (normalize) {
      auto a = saved[base + 1], b = saved[base + 2];
      auto zero = a.eq(0) | b.eq(0);
      auto denominator = at::where(zero, 1., a.sqrt() * b.sqrt());
      value = at::where(zero, 0., value / denominator);
    }
    lookup(result, index, 0).copy_(value / divisor);
  }
  return {result, state};
}
Tensors inner_product_backward_kernel(const Tensor& gradient, at::TensorList v,
                                      const Tensor& result, const Tensor& state,
                                      TORCHPAIRWISE_EXPRESSION_DECL,
                                      int64_t left_root, int64_t right_root,
                                      bool center, bool normalize) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto s = inner_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      dtype, size, exclusion, transposed, form, left_root, right_root);
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
  auto n = v[X].size(-2), m = v[Y].size(-2);
  auto divisor = hsic_denominator(operation, n, m, center, normalize);
  bool unbiased = operation == "hsic_unbiased";
  auto base = center ? 2 * (n + m) + 4 : 0;
  at::AutoGradMode enable(true);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto saved = lookup(state, index, 1);
    auto g = lookup(gradient, index, 0).to(at::kDouble) / divisor;
    auto dot = saved[base], a = saved[base + 1], b = saved[base + 2];
    auto zero = a.eq(0) | b.eq(0);
    auto safe_a = at::where(a.eq(0), 1., a);
    auto safe_b = at::where(b.eq(0), 1., b);
    auto scale_a = at::where(saved[base + 3].eq(0), 1., saved[base + 3]);
    auto scale_b = at::where(saved[base + 4].eq(0), 1., saved[base + 4]);
    auto denominator = safe_a.sqrt() * safe_b.sqrt();
    for (int64_t i = 0; i < n; i += size) {
      auto ni = std::min(size, n - i);
      for (int64_t j = 0; j < m; j += size) {
        auto nj = std::min(size, m - j);
        auto local =
            local_inputs(v, s, index, i, ni, j, nj, Tensor(), Tensor(), true);
        ExpressionReplay replay;
        auto entries =
            inner_entries(local, s, left_root, right_root, ni, nj, &replay);
        if (center)
          for (int64_t k = 0; k < 2; ++k)
            entries[k] =
                centered_entry(entries[k], saved, k, n, m, i, ni, j, nj);
        if (normalize) {
          entries[0] = entries[0] / scale_a;
          entries[1] = entries[1] / scale_b;
        }
        Tensor da = entries[1], db = entries[0];
        if (unbiased) {
          da = u_centered_adjoint(da, saved, 1, n, i, ni, j, nj);
          db = u_centered_adjoint(db, saved, 0, n, i, ni, j, nj);
        }
        if (normalize) {
          da = at::where(
              zero, 0.,
              ((da - (dot / safe_a) * entries[0]) / denominator) / scale_a);
          db = at::where(
              zero, 0.,
              ((db - (dot / safe_b) * entries[1]) / denominator) / scale_b);
        }
        expression_accumulate_roots(local, {g * da, g * db},
                                    {left_root, right_root}, buffers, s, index,
                                    &replay);
      }
    }
  }
  for (size_t k = 0; k < v.size(); ++k)
    buffers[k] = buffers[k].to(v[k].scalar_type()).contiguous();
  return buffers;
}
Tensor expression_inner_product(at::TensorList v, const Settings& s,
                                int64_t left_root, int64_t right_root,
                                bool center, bool normalize,
                                c10::string_view statistic) {
  const auto& p = s.expression->packet;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_expression_inner_product", "")
          .typed<InnerProductSignature>();
  return std::get<0>(op.call(v, p.names, p.families, p.headers, p.reals,
                             p.integers, p.flags, p.nodes, p.numbers, statistic,
                             s.dtype, s.size, false, false, false, left_root,
                             right_root, center, normalize));
}
namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_expression_inner_product(Tensor[] operands, str[] names, str[] "
      "families, "
      "int[][] headers, float?[][] reals, int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, bool transposed, bool form, int left_root, int "
      "right_root, "
      "bool center, bool normalize) -> (Tensor, Tensor)");
  m.def(
      "_expression_inner_product_backward(Tensor grad, Tensor[] operands, "
      "Tensor result, Tensor state, str[] names, str[] families, int[][] "
      "headers, "
      "float?[][] reals, int?[][] integers, bool?[][] flags, int[][] nodes, "
      "Scalar[] numbers, str operation, int dtype, int size, bool exclusion, "
      "bool transposed, bool form, int left_root, int right_root, "
      "bool center, bool normalize) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_expression_inner_product", TORCH_FN(inner_product_kernel));
  m.impl("_expression_inner_product_backward",
         TORCH_FN(inner_product_backward_kernel));
}
}  // namespace
}  // namespace torchpairwise::ops::detail::streaming
