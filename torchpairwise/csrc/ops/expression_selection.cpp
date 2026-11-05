#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>

#include "expression_selection_detail.h"
#include "expression_statistics_detail.h"
#include "ppminkowski.h"
#include "utils/comparison_dispatch.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::streaming {
namespace {
struct SelectionRanks {
  int64_t lower, upper;
  double weight;
};
SelectionRanks selection_ranks(int64_t count, c10::string_view operation,
                               double q, c10::string_view interpolation,
                               int64_t dtype) {
  TORCH_CHECK(count > 0, "median/quantile require an eligible reference");
  TORCH_CHECK(operation == "median" || operation == "quantile",
              "Unknown exact selection consumer");
  TORCH_CHECK(std::isfinite(q) && q >= 0. && q <= 1.,
              "q must be finite and between zero and one");
  TORCH_CHECK(interpolation == "linear" || interpolation == "lower" ||
                  interpolation == "higher" || interpolation == "nearest" ||
                  interpolation == "midpoint",
              "Unknown quantile interpolation");
  TORCH_CHECK(operation != "median" || (q == .5 && interpolation == "lower"),
              "Unused median options");
  if (operation == "median")
    return {(count - 1) / 2, (count - 1) / 2, 0.};
  // Match the entry dtype used by torch.quantile for its rank arithmetic.
  double rank = dtype == int64_t(at::kFloat)
                    ? double(float(q) * float(count - 1))
                    : q * double(count - 1);
  rank = std::clamp(rank, 0., double(count - 1));
  auto lower = int64_t(std::floor(rank)), upper = int64_t(std::ceil(rank));
  auto weight = rank - lower;
  if (interpolation == "nearest") {
    auto nearest = weight < .5   ? lower
                   : weight > .5 ? upper
                                 : (lower % 2 ? upper : lower);
    return {nearest, nearest, 0.};
  }
  if (interpolation == "lower")
    return {lower, lower, 0.};
  if (interpolation == "higher")
    return {upper, upper, 0.};
  return {lower, upper, interpolation == "midpoint" ? .5 : weight};
}
Settings selection_settings(const ExpressionPacket& packet,
                            c10::string_view operation, int64_t dtype,
                            int64_t size, bool exclusion, bool transposed,
                            bool form) {
  TORCH_CHECK(!form, "Exact selection requires one sample axis");
  TORCH_CHECK(operation == "median" || operation == "quantile",
              "Unknown exact selection consumer");
  TORCH_CHECK(dtype != int64_t(at::kBool),
              "Boolean exact selection is unsupported");
  TORCH_CHECK(operation == "median" || dtype == int64_t(at::kFloat) ||
                  dtype == int64_t(at::kDouble),
              "quantile requires float32/float64 entries");
  // Reuse packet/dtype validation without applying a statistic's float-only
  // requirement to integer median values.
  return statistics_settings(packet, "count_nonzero", dtype, size, exclusion,
                             transposed);
}
enum class SelectionSource { General, Matrix, SquaredEuclidean };
SelectionSource selection_source(at::TensorList v, const Settings& s) {
  const auto& p = *s.expression;
  if (p.leaves.size() != 1 ||
      p.packet.nodes != std::vector<std::vector<int64_t>>{{0, 0, -1}} ||
      v.size() != 2 * Slots)
    return SelectionSource::General;
  const auto& leaf = p.leaves[0];
  if (leaf.family && *leaf.family == "_matrix")
    return SelectionSource::Matrix;
  if (!leaf.family && !leaf.kernel && leaf.rank == 2 && leaf.mask == 3 &&
      detail::canonical_comparison_name(leaf.metric) == "sqeuclidean")
    return SelectionSource::SquaredEuclidean;
  return SelectionSource::General;
}
bool selection_swapped(const Settings& s) {
  return s.transposed() != s.expression->leaves[0].transposed();
}
Tensor selection_matrix(at::TensorList v, const Settings& s,
                        const std::vector<int64_t>& index) {
  auto matrix = lookup(v[Slots + X], index, 2);
  return selection_swapped(s) ? matrix.transpose(-1, -2) : matrix;
}
int64_t selection_rows(int64_t width, int64_t size) {
  // This is an element target, not a hard cap: one complete row is required.
  return std::min(size, std::max<int64_t>(1, (int64_t(1) << 20) /
                                                 std::max<int64_t>(1, width)));
}
}  // namespace
StatisticsOutputs expression_selection_kernel(at::TensorList v,
                                              TORCHPAIRWISE_EXPRESSION_DECL,
                                              double q,
                                              c10::string_view interpolation) {
  auto s = selection_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed, form);
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  TORCH_CHECK(!exclusion || n == m,
              "exclude_self requires a square expression");
  auto ranks = selection_ranks(m - int64_t(exclusion), operation, q,
                               interpolation, dtype);
  at::NoGradGuard guard;
  utils::DeferredValueChecks checks;
  auto batch = batch_shape(v, s), shape = batch;
  shape.push_back(n);
  auto result = at::empty(shape, v[X].options().dtype(at::ScalarType(dtype)));
  shape.push_back(2);
  auto witnesses = at::empty(shape, v[X].options().dtype(at::kLong));
  int64_t batches = 1;
  for (auto dimension : batch)
    batches *= dimension;
  auto rows = selection_rows(m, size);
  auto source = selection_source(v, s);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    for (int64_t left = 0; left < n; left += rows) {
      auto count = std::min(rows, n - left);
      auto values = source == SelectionSource::Matrix
                        ? selection_matrix(v, s, index).narrow(0, left, count)
                        : at::empty({count, m}, result.options());
      if (source == SelectionSource::SquaredEuclidean) {
        bool swapped = selection_swapped(s);
        auto a = lookup(v[Slots + (swapped ? Y : X)], index, 2)
                     .narrow(0, left, count);
        auto b = lookup(v[Slots + (swapped ? X : Y)], index, 2);
        for (int64_t right = 0; right < m; right += size) {
          auto other = std::min(size, m - right);
          values.narrow(-1, right, other)
              .copy_(_ppminkowski(a, b.narrow(0, right, other), 2.));
        }
      } else if (source == SelectionSource::General) {
        for (int64_t right = 0; right < m; right += size) {
          auto other = std::min(size, m - right);
          auto local = local_inputs(v, s, index, left, count, right, other);
          values.narrow(-1, right, other)
              .copy_(evaluate(local, s, false, left, right));
        }
      }
      auto row_ids =
          at::arange(left, left + count, witnesses.options()).unsqueeze(-1);
      if (exclusion) {
        auto ids = at::arange(m - 1, witnesses.options()).unsqueeze(0);
        auto columns = ids + ids.ge(row_ids);
        values = values.gather(-1, columns);
      }
      // Stable sort defines ties by their original logical axis index.
      auto [ordered, ids] = at::sort(values, true, -1, false);
      auto lo = ids.select(-1, ranks.lower), hi = ids.select(-1, ranks.upper);
      if (values.is_floating_point()) {
        auto has_nan = ordered.select(-1, ordered.size(-1) - 1).isnan();
        // quantile propagates NaNs through the last stable sorted entry, as
        // torch.quantile does. Median chooses the first logical NaN.
        auto nan_id = operation == "median"
                          ? values.isnan().to(at::kLong).argmax(-1)
                          : ids.select(-1, ids.size(-1) - 1);
        lo = at::where(has_nan, nan_id, lo);
        hi = at::where(has_nan, nan_id, hi);
      }
      auto low_value = values.gather(-1, lo.unsqueeze(-1)).squeeze(-1);
      auto high_value = values.gather(-1, hi.unsqueeze(-1)).squeeze(-1);
      auto value = operation == "quantile" && (interpolation == "linear" ||
                                               interpolation == "midpoint")
                       ? at::lerp(low_value, high_value, ranks.weight)
                       : low_value;
      if (exclusion) {
        auto row = row_ids.squeeze(-1);
        lo = lo + lo.ge(row);
        hi = hi + hi.ge(row);
      }
      lookup(result, index, 1).narrow(0, left, count).copy_(value);
      lookup(witnesses, index, 2)
          .narrow(0, left, count)
          .copy_(at::stack({lo, hi}, -1));
    }
  }
  checks.finish();
  return {result, witnesses};
}
Tensors expression_selection_backward_kernel(
    const Tensor& gradient, at::TensorList v, const Tensor& result,
    const Tensor& indices, TORCHPAIRWISE_EXPRESSION_DECL, double q,
    c10::string_view interpolation) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU),
      no_cuda_autocast(c10::DispatchKey::AutocastCUDA);
  auto s = selection_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed, form);
  auto n = v[transposed ? Y : X].size(-2), m = v[transposed ? X : Y].size(-2);
  auto ranks = selection_ranks(m - int64_t(exclusion), operation, q,
                               interpolation, dtype);
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
  auto source = selection_source(v, s);
  if (source == SelectionSource::Matrix) {
    at::NoGradGuard guard;
    bool swapped = selection_swapped(s);
    for (int64_t flat = 0; flat < batches; ++flat) {
      auto index = coordinates(flat, batch);
      auto destination = lookup(buffers[Slots + X], index, 2);
      if (swapped)
        destination = destination.transpose(-1, -2);
      auto ids = lookup(indices, index, 2);
      auto g = lookup(gradient, index, 1).unsqueeze(-1);
      // Duplicate witnesses combine their weights before multiplying by g,
      // exactly as in the common cotangent construction, including nonfinite g.
      auto same = ids.select(-1, 0).eq(ids.select(-1, 1)).unsqueeze(-1);
      auto lo = at::where(same, 1., at::full_like(g, 1. - ranks.weight));
      auto hi = at::where(same, 0., at::full_like(g, ranks.weight));
      for (int64_t k = 0; k < 2; ++k) {
        auto coefficient = k == 0 ? lo : hi;
        auto value = at::where(coefficient.ne(0), g * coefficient, 0.);
        destination.scatter_add_(-1, ids.narrow(-1, k, 1),
                                 value.to(at::kDouble));
      }
    }
  }
#ifdef WITH_CUDA
  else if (source == SelectionSource::SquaredEuclidean && v[X].is_cuda()) {
    at::NoGradGuard guard;
    bool swapped = selection_swapped(s);
    for (int64_t flat = 0; flat < batches; ++flat) {
      auto index = coordinates(flat, batch);
      auto a = lookup(v[Slots + (swapped ? Y : X)], index, 2);
      auto b = lookup(v[Slots + (swapped ? X : Y)], index, 2);
      auto [da, db] = selection_sqeuclidean_backward_cuda(
          a, b, lookup(gradient, index, 1), lookup(indices, index, 2),
          ranks.weight);
      lookup(buffers[Slots + (swapped ? Y : X)], index, 2).add_(da);
      lookup(buffers[Slots + (swapped ? X : Y)], index, 2).add_(db);
    }
  }
#endif
  else {
    utils::DeferredValueChecks checks;
    at::AutoGradMode enable(true);
    for (int64_t flat = 0; flat < batches; ++flat) {
      auto index = coordinates(flat, batch);
      for (int64_t left = 0; left < n; left += size) {
        auto count = std::min(size, n - left);
        auto ids = lookup(indices, index, 2).narrow(0, left, count);
        auto g =
            lookup(gradient, index, 1).narrow(0, left, count).unsqueeze(-1);
        for (int64_t right = 0; right < m; right += size) {
          auto other = std::min(size, m - right);
          auto columns = at::arange(right, right + other, indices.options());
          auto coefficients =
              ids.select(-1, 0).unsqueeze(-1).eq(columns).to(g.scalar_type()) *
                  (1. - ranks.weight) +
              ids.select(-1, 1).unsqueeze(-1).eq(columns).to(g.scalar_type()) *
                  ranks.weight;
          auto cotangent = at::where(coefficients.ne(0), g * coefficients, 0.);
          auto local = local_inputs(v, s, index, left, count, right, other,
                                    Tensor(), Tensor(), true);
          ExpressionReplay replay;
          auto value = evaluate(local, s, false, left, right, Tensor(),
                                Tensor(), &replay);
          // Replay all entries, including zero cotangents, so composed sources
          // keep their native nonfinite chain rules without retaining a graph.
          accumulate(local, value, cotangent, buffers, s, index, left, count,
                     right, other, Tensor(), Tensor(), &replay);
        }
      }
    }
    checks.finish();
  }
  for (size_t slot = 0; slot < v.size(); ++slot)
    buffers[slot] = buffers[slot].to(v[slot].scalar_type()).contiguous();
  return buffers;
}
Tensor expression_selection(at::TensorList v, const Settings& s,
                            c10::string_view reduction, double q,
                            c10::string_view interpolation) {
  const auto& p = s.expression->packet;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_expression_selection", "")
          .typed<SelectionSignature>();
  return std::get<0>(op.call(v, p.names, p.families, p.headers, p.reals,
                             p.integers, p.flags, p.nodes, p.numbers, reduction,
                             s.dtype, s.size, s.exclusion, s.transposed(),
                             false, q, interpolation));
}
namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_expression_selection(Tensor[] operands, str[] names, str[] families, "
      "int[][] headers, float?[][] reals, int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, bool transposed, bool form, float q, str interpolation) "
      "-> (Tensor, Tensor)");
  m.def(
      "_expression_selection_backward(Tensor grad, Tensor[] operands, "
      "Tensor result, Tensor indices, str[] names, str[] families, "
      "int[][] headers, float?[][] reals, int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, bool transposed, bool form, float q, str interpolation) "
      "-> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_expression_selection", TORCH_FN(expression_selection_kernel));
  m.impl("_expression_selection_backward",
         TORCH_FN(expression_selection_backward_kernel));
}
}  // namespace
}  // namespace torchpairwise::ops::detail::streaming
