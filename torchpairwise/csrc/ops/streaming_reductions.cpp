#include "streaming_reductions.h"

#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <limits>

#include "custom_comparisons_detail.h"
#include "expression_operations_detail.h"
#include "fused_reducers.h"
#include "image_map_operations_detail.h"
#include "media_operations_detail.h"
#include "mmd_comparisons_detail.h"
#include "residual_operations_detail.h"
#include "sequence_comparisons_detail.h"
#include "streaming_reductions_detail.h"
#include "structured_comparisons_detail.h"
#include "token_comparisons_detail.h"
#include "utils/clouds.h"
#include "utils/comparison_dispatch.h"
#include "utils/tiling.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
using at::Tensor;
namespace {
std::tuple<at::Tensor, at::Tensor> streaming_reductions_composite(
    const at::Tensor& input, const c10::optional<at::Tensor>& other,
    c10::string_view metric, c10::string_view operation, bool kernel,
    const StreamingReductionOptions& options);
}

namespace detail::streaming {
BuiltinComparisonOptions comparison_options(const Settings& s,
                                            at::TensorList v) {
  BuiltinComparisonOptions o;
  if (s.has(W))
    o.w = v[W];
  if (s.has(Variance))
    o.V = v[Variance];
  if (s.has(Inverse))
    o.VI = v[Inverse];
  if (s.has(Bins))
    o.bin_positions = v[Bins];
  if (s.has(Projections))
    o.projections = v[Projections];
  if (s.has(XL))
    o.x1_lengths = v[XL];
  if (s.has(YL))
    o.x2_lengths = v[YL];
  o.p = s.reals[0];
  o.base = s.reals[1];
  o.alpha = s.reals[2];
  o.length_scale = s.reals[3];
  o.nu = s.reals[4];
  o.gamma = s.reals[5];
  o.coef0 = s.reals[6];
  o.regularization = s.reals[7];
  o.degree = s.integers[0];
  o.max_iter = s.integers[1];
  o.chunk_size = s.integers[2];
  o.squared = s.flags[0];
  o.shuffle = s.flags[1];
  o.kernel = s.family;
  return o;
}
std::vector<int64_t> batch_shape(at::TensorList v, const Settings& s) {
  auto batch = at::infer_size(v[X].sizes().slice(0, v[X].dim() - s.rank),
                              v[Y].sizes().slice(0, v[Y].dim() - s.rank));
  for (int64_t slot = W; slot < Slots; ++slot)
    if (s.has(slot) && s.operand_rank(slot) > 0)
      batch = at::infer_size(
          batch,
          v[slot].sizes().slice(0, v[slot].dim() - s.operand_rank(slot)));
  return batch;
}
std::vector<int64_t> coordinates(int64_t flat,
                                 const std::vector<int64_t>& batch) {
  std::vector<int64_t> index(batch.size());
  for (int64_t i = int64_t(batch.size()) - 1; i >= 0; --i) {
    index[i] = flat % batch[i];
    flat /= batch[i];
  }
  return index;
}
Tensor lookup(Tensor value, const std::vector<int64_t>& index, int64_t rank) {
  auto dims = value.dim() - rank, offset = int64_t(index.size()) - dims;
  for (int64_t axis = 0; axis < dims; ++axis)
    value = value.select(0, value.size(0) == 1 ? 0 : index[offset + axis]);
  return value;
}
Tensors local_inputs(at::TensorList v, const Settings& s,
                     const std::vector<int64_t>& index, int64_t left,
                     int64_t left_size, int64_t right, int64_t right_size,
                     const Tensor& rows, const Tensor& columns, bool replay) {
  if (s.expression)
    return expression_local_inputs(v, s, index, left, left_size, right,
                                   right_size, rows, columns, replay);
  Tensors result(v.begin(), v.end());
  for (int64_t slot = 0; slot < Slots; ++slot)
    if (s.has(slot)) {
      auto rank = s.operand_rank(slot);
      auto value = rank > 0 ? lookup(v[slot], index, rank) : v[slot];
      if (s.sampled(slot)) {
        bool first = s.first_operand(slot) != s.transposed();
        if (rows.defined()) {
          // Paired matrix entries can index the source directly, without
          // copying selected rows across the source's full column width.
          if (!(s.family && *s.family == "_matrix" && slot == X))
            value = value.index_select(0, first ? rows : columns);
        } else
          value = value.narrow(0, first ? left : right,
                               first ? left_size : right_size);
      }
      if (replay) {
        value = value.detach();
        if (s.differentiable(slot) && value.is_floating_point())
          value.set_requires_grad(true);
      }
      result[slot] = value;
    }
  return result;
}
Tensor evaluate(at::TensorList local, const Settings& s, bool paired,
                int64_t left, int64_t right, const Tensor& rows,
                const Tensor& columns, ExpressionReplay* replay) {
  if (s.expression)
    return expression_evaluate(local, s, paired, replay);
  Tensor result;
  if (s.family && *s.family == "_custom") {
    result = detail::custom::evaluate(local, s, paired);
  } else if (s.family && *s.family == "_matrix") {
    result = paired && rows.defined()
                 ? local[X].index(
                       {s.transposed() ? columns : rows, local[Y].squeeze(-1)})
             : paired ? local[X].gather(-1, local[Y]).squeeze(-1)
                      : local[X].index_select(-1, local[Y].squeeze(-1));
  } else if (s.family && *s.family == "_residual") {
    auto transform = s.integers[1].value();
    result = detail::residual::feature_reduce(local[X], local[Y], s.metric,
                                              transform == 0   ? "identity"
                                              : transform == 1 ? "abs"
                                                               : "square",
                                              s.reals[0].value(),
                                              s.integers[0].value(), paired);
  } else if (s.family && *s.family == "_sequence") {
    result = detail::sequence::prepared_pair(
        local[X], local[Y], local[XL], local[YL], local[W], local[Variance],
        paired, s.flags[0].value(), s.reals[0], s.integers[0], s.size);
  } else if (s.family && *s.family == "_tokens") {
    result =
        detail::tokens::compare(local[X], local[Y], local[XL], local[YL],
                                paired ? 1 : 0, s.integers[0].value(), s.size);
  } else if (s.family && *s.family == "_mmd") {
    detail::mmd::KernelOptions o{
        s.metric,           s.reals[0].value(), s.integers[0].value(),
        s.reals[1].value(), s.reals[2].value(), s.reals[3].value(),
        s.reals[4].value()};
    o.statistic = s.flags[1].value();
    o.unbiased = s.flags[2].value();
    result = detail::mmd::prepared_pair(
        local[X], local[Y], local[XL], local[YL], local[W], local[Variance], o,
        paired, s.flags[0].value(), s.integers[1].value());
  } else if (s.family && *s.family == "_image_event") {
    std::vector<int64_t> settings, program;
    std::vector<double> parameters, corrections;
    for (size_t k = 1; k < 5; ++k)
      settings.push_back(s.integers[k].value());
    for (size_t k = 6; k < s.integers.size(); ++k)
      program.push_back(s.integers[k].value());
    for (size_t k = 0; k < 4; ++k)
      parameters.push_back(s.reals[k].value());
    for (size_t k = 4; k < s.reals.size(); ++k)
      corrections.push_back(s.reals[k].value());
    result = std::get<0>(detail::media::event_reduce(
        local[X], local[Y], s.integers[0].value(), settings, parameters,
        s.integers[5].value(), program, corrections, "", {}, false, 1., s.size,
        paired));
  } else if (s.family && *s.family == "_image_map") {
    std::vector<int64_t> settings;
    std::vector<double> parameters;
    for (size_t k = 1; k + 1 < s.integers.size(); ++k)
      settings.push_back(s.integers[k].value());
    for (size_t k = 0; k + 1 < s.reals.size(); ++k)
      parameters.push_back(s.reals[k].value());
    result = detail::media::map_reduction(
        local[X], local[Y], s.integers[0].value(), s.metric, settings,
        parameters, s.integers.back().value(), s.reals.back().value(), paired);
  } else if (s.family && *s.family == "_media") {
    std::vector<int64_t> settings;
    std::vector<double> parameters;
    for (size_t k = 1; k < s.integers.size(); ++k)
      settings.push_back(s.integers[k].value());
    for (const auto& p : s.reals)
      parameters.push_back(p.value());
    result =
        detail::media::feature_pair(local[X], local[Y], s.integers[0].value(),
                                    paired, settings, parameters)
            .to(at::ScalarType(s.dtype));
  } else if (s.structured()) {
    detail::structured::Features a{local[X], local[W], local[Inverse],
                                   local[Projections]},
        b{local[Y], local[Variance], local[Bins], local[XL]};
    result = detail::structured::feature_pair(
        a, b, s.integers[0].value(), paired, s.flags[0].value_or(false),
        s.reals[0].value_or(.1), s.integers[1].value_or(100),
        s.integers[2].value_or(32));
  } else
    result = builtin_comparisons(local[X], local[Y], s.metric,
                                 paired ? "paired" : "all_pairs", s.kernel,
                                 comparison_options(s, local));
  if (s.transposed() && !paired)
    result = result.transpose(-1, -2);
  if (s.self_cosine) {
    auto i =
        rows.defined()
            ? rows
            : at::arange(left, left + local[s.transposed() ? Y : X].size(0),
                         local[X].options().dtype(at::kLong));
    auto j =
        columns.defined()
            ? columns
            : at::arange(right, right + local[s.transposed() ? X : Y].size(0),
                         local[Y].options().dtype(at::kLong));
    result = at::where(paired ? i.eq(j) : i.unsqueeze(-1).eq(j.unsqueeze(0)),
                       0., result);
  }
  return result;
}
std::pair<Tensor, Tensor> compensated(const Tensor& total,
                                      const Tensor& correction,
                                      const Tensor& term) {
  if (!total.is_floating_point())
    return {total + term, correction};
#ifdef WITH_CUDA
  // Keep ATen replay when a surrounding differentiable composition needs it.
  // Normal tiled forwards use one launch for the same finite/nonfinite rule.
  if (!at::GradMode::is_enabled() && total.is_cuda() &&
      (total.scalar_type() == at::kFloat ||
       total.scalar_type() == at::kDouble) &&
      total.is_contiguous() && correction.is_contiguous() &&
      term.is_contiguous() && total.sizes().equals(correction.sizes()) &&
      total.sizes().equals(term.sizes()) &&
      total.scalar_type() == correction.scalar_type() &&
      total.scalar_type() == term.scalar_type() &&
      total.device() == correction.device() && total.device() == term.device())
    return compensated_cuda(total, correction, term);
#endif
  auto updated = total + term,
       valid = total.isfinite() & term.isfinite() & updated.isfinite();
  auto residual = at::where(total.abs().ge(term.abs()),
                            (total - updated) + term, (term - updated) + total);
  return {updated, at::where(valid, correction + residual, 0.)};
}
Tensor eligible(const Tensor& rows, const Tensor& columns, bool exclusion) {
  return exclusion ? rows.unsqueeze(-1).ne(columns.unsqueeze(0))
                   : at::ones({rows.numel(), columns.numel()},
                              rows.options().dtype(at::kBool));
}
// Keep the tile's exact top-k set in column order. The stable merge below
// resolves ties by input order: previous columns precede all new tile columns.
// CUDA skips pruning: its extra launches cost more than the sorts they save.
std::tuple<Tensor, Tensor, Tensor> prune_topk_tile(Tensor value, Tensor index,
                                                   Tensor allowed,
                                                   const Settings& s) {
  auto width = value.size(-1);
  if (value.is_cuda() || !value.is_floating_point() || s.k <= 1 || width <= s.k)
    return {value, index, allowed};
  auto nan = value.isnan();
  auto category =
      at::where(allowed, (s.nan_first ? ~nan : nan).to(at::kChar), 2);
  // Locate rank k's category without sorting or reading device scalars.
  auto cutoff_category =
      at::where(category.eq(0).sum(-1, true).ge(s.k), 0,
                at::where(category.le(1).sum(-1, true).ge(s.k), 1, 2));
  auto key = at::where(nan, 0., s.largest ? -value : value);
  auto infinity = std::numeric_limits<double>::infinity();
  auto threshold_key =
      at::where(category.lt(cutoff_category), -infinity,
                at::where(category.eq(cutoff_category), key, infinity));
  auto threshold = std::get<0>(at::topk(threshold_key, s.k, -1, false, false))
                       .amax(-1, true);
  // Earlier categories win even at infinite thresholds. Among threshold ties,
  // retain the earliest columns, including equal signed zeros and NaN payloads.
  auto better = category.lt(cutoff_category) |
                (category.eq(cutoff_category) & key.lt(threshold));
  auto tied = category.eq(cutoff_category) & key.eq(threshold);
  auto remaining = s.k - better.sum(-1, true);
  auto keep = better | (tied & tied.cumsum(-1).le(remaining));
  // Exactly k offsets survive. Sorting just those offsets restores column order
  // without a full argsort, nonzero's dynamic allocation, or host
  // synchronization.
  auto offsets = at::arange(width, index.options());
  auto packed = std::get<0>(
      at::topk(at::where(keep, offsets, width), s.k, -1, false, true));
  return {value.gather(-1, packed), index.gather(-1, packed),
          allowed.gather(-1, packed)};
}
std::tuple<Tensor, Tensor, Tensor> select(Tensor value, Tensor index,
                                          Tensor allowed, const Settings& s) {
  if (!value.is_floating_point()) {
    if (s.k != 1) {
      auto order = at::argsort(value.to(at::kLong), true, -1, s.largest);
      value = value.gather(-1, order);
      index = index.gather(-1, order);
      allowed = allowed.gather(-1, order);
      order = at::argsort((~allowed).to(at::kChar), true, -1, false)
                  .narrow(-1, 0, std::min(s.k, value.size(-1)));
      return {value.gather(-1, order), index.gather(-1, order),
              allowed.gather(-1, order)};
    }
    auto sentinel = s.largest ? std::numeric_limits<int64_t>::lowest()
                              : std::numeric_limits<int64_t>::max();
    auto key = at::where(allowed, value.to(at::kLong), sentinel);
    auto best = s.largest ? key.amax(-1, true) : key.amin(-1, true);
    auto first = (allowed & key.eq(best)).to(at::kChar).argmax(-1, true);
    return {value.gather(-1, first), index.gather(-1, first),
            allowed.gather(-1, first)};
  }
  auto nan = value.isnan();
  auto category =
      at::where(allowed, (s.nan_first ? ~nan : nan).to(at::kChar), 2);
  if (s.k == 1) {
    auto smallest = category.amin(-1, true);
    auto key = at::where(nan, 0., s.largest ? -value : value);
    key = at::where(category.eq(smallest), key,
                    std::numeric_limits<double>::infinity());
    auto minimum = key.amin(-1, true);
    auto first = (category.eq(smallest) & key.eq(minimum))
                     .to(at::kChar)
                     .argmax(-1, true);
    return {value.gather(-1, first), index.gather(-1, first),
            allowed.gather(-1, first)};
  }
  auto order = at::argsort(at::where(nan, 0., value), true, -1, s.largest);
  value = value.gather(-1, order);
  index = index.gather(-1, order);
  allowed = allowed.gather(-1, order);
  category = category.gather(-1, order);
  order = at::argsort(category, true, -1, false)
              .narrow(-1, 0, std::min(s.k, category.size(-1)));
  return {value.gather(-1, order), index.gather(-1, order),
          allowed.gather(-1, order)};
}
Outputs streaming_reduce_kernel(at::TensorList v, const Settings& s) {
  utils::DeferredValueChecks checks;
  auto batch = batch_shape(v, s), shape = batch;
  auto n = v[s.transposed() ? Y : X].size(-s.rank),
       m = v[s.transposed() ? X : Y].size(-s.rank);
  shape.push_back(n);
  bool search = s.kind() == "neighbors",
       boolean = s.kind() == "all" || s.kind() == "any",
       product = s.kind() == "prod",
       matmul = (s.kind() == "kernel_sum" || s.kind() == "matmul");
  if (search)
    shape.push_back(s.k);
  if (s.kind() == "entries")
    shape.push_back(m);
  if (matmul)
    shape.push_back(v[Payload].size(-1));
  auto result = at::empty(
      shape,
      v[X].options().dtype(boolean ? at::kBool : at::ScalarType(s.dtype)));
  auto empty = at::empty({0}, v[X].options());
  auto indices = search ? at::empty(shape, v[X].options().dtype(at::kLong))
                        : at::empty({0}, v[X].options().dtype(at::kLong));
  auto zeros = product ? at::zeros(shape, v[X].options().dtype(at::kLong))
                       : at::empty({0}, v[X].options().dtype(at::kLong));
  auto nonzero = product ? at::ones(shape, result.options()) : empty.clone();
#ifdef WITH_CUDA
  if (matmul || s.kind() == "sum")
    if (auto plan = fused::plan(v, s)) {
      fused::count();
      auto payload = matmul ? v[Payload] : at::ones({m, 1}, result.options());
      // Matrix products have no exclusion; reductions drop j == i.
      auto value = fused::matmul_cuda(*plan, payload, !matmul && s.exclusion);
      result.copy_(value.view(result.sizes()));
      checks.finish();
      return {result, indices, zeros, nonzero};
    }
  if (search && s.k >= 1 && s.k <= fused::max_topk &&
      s.k <= m - int64_t(s.exclusion))
    if (auto plan = fused::plan(v, s)) {
      fused::count();
      Tensor values, selected;
      std::tie(values, selected) =
          fused::topk_cuda(*plan, s.k, s.largest, s.nan_first, s.exclusion);
      result.copy_(values.view(result.sizes()));
      indices.copy_(selected.view(indices.sizes()));
      checks.finish();
      return {result, indices, zeros, nonzero};
    }
#endif
  int64_t batches = 1;
  for (auto dim : batch)
    batches *= dim;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    for (int64_t left = 0; left < n && (!search || s.k > 0); left += s.size) {
      auto count = std::min(s.size, n - left);
      auto rows =
          at::arange(left, left + count, v[X].options().dtype(at::kLong));
      if (s.kind() == "diagonal") {
        auto local = local_inputs(v, s, index, left, count, left, count);
        lookup(result, index, 1)
            .narrow(0, left, count)
            .copy_(evaluate(local, s, true, left, left));
      } else if (s.kind() == "entries") {
        for (int64_t right = 0; right < m; right += s.size) {
          auto other = std::min(s.size, m - right);
          auto local = local_inputs(v, s, index, left, count, right, other);
          lookup(result, index, 2)
              .narrow(0, left, count)
              .narrow(1, right, other)
              .copy_(evaluate(local, s, false, left, right));
        }
      } else if (search) {
        auto current = at::empty({count, 0}, result.options());
        auto selected = at::empty({count, 0}, indices.options()),
             allowed = at::empty({count, 0}, v[X].options().dtype(at::kBool));
        for (int64_t right = 0; right < m; right += s.size) {
          auto other = std::min(s.size, m - right);
          auto columns = at::arange(right, right + other, indices.options());
          auto local = local_inputs(v, s, index, left, count, right, other);
          auto tile = evaluate(local, s, false, left, right);
          auto candidates =
              prune_topk_tile(tile, columns.expand({count, other}),
                              eligible(rows, columns, s.exclusion), s);
          auto chosen =
              select(at::cat({current, std::get<0>(candidates)}, -1),
                     at::cat({selected, std::get<1>(candidates)}, -1),
                     at::cat({allowed, std::get<2>(candidates)}, -1), s);
          current = std::get<0>(chosen);
          selected = std::get<1>(chosen);
          allowed = std::get<2>(chosen);
        }
        lookup(result, index, 2).narrow(0, left, count).copy_(current);
        lookup(indices, index, 2).narrow(0, left, count).copy_(selected);
      } else if (matmul) {
        auto payload = lookup(v[Payload], index, 2);
        for (int64_t channel = 0; channel < payload.size(-1);
             channel += s.size) {
          auto channels = std::min(s.size, payload.size(-1) - channel);
          auto total = at::zeros({count, channels}, result.options()),
               correction = at::zeros_like(total);
          for (int64_t right = 0; right < m; right += s.size) {
            auto other = std::min(s.size, m - right);
            auto local = local_inputs(v, s, index, left, count, right, other);
            auto kernel = evaluate(local, s, false, left, right)
                              .to(payload.scalar_type());
            auto term = kernel.matmul(
                payload.narrow(0, right, other).narrow(-1, channel, channels));
            auto update = compensated(total, correction, term);
            total = update.first;
            correction = update.second;
          }
          lookup(result, index, 2)
              .narrow(0, left, count)
              .narrow(-1, channel, channels)
              .copy_(total + correction);
        }
      } else {
        auto total =
            at::full({count}, product || s.kind() == "all", result.options());
        auto correction = at::zeros_like(total),
             zero_count = at::zeros({count}, zeros.options()),
             nonzero_product = at::ones({count}, result.options());
        for (int64_t right = 0; right < m; right += s.size) {
          auto other = std::min(s.size, m - right);
          auto local = local_inputs(v, s, index, left, count, right, other);
          auto value = evaluate(local, s, false, left, right);
          auto columns = at::arange(right, right + other, rows.options());
          auto allowed = eligible(rows, columns, s.exclusion);
          if (boolean) {
            auto truth = at::where(allowed, value.ne(0), s.kind() == "all");
            total = s.kind() == "all" ? total & truth.all(-1)
                                      : total | truth.any(-1);
          } else if (s.kind() == "sum") {
            auto update = compensated(total, correction,
                                      at::where(allowed, value, 0).sum(-1));
            total = update.first;
            correction = update.second;
          } else {
            value = at::where(allowed, value, 1);
            total = total * value.prod(-1);
            zero_count = zero_count + value.eq(0).sum(-1);
            nonzero_product =
                nonzero_product * at::where(value.eq(0), 1, value).prod(-1);
          }
        }
        lookup(result, index, 1)
            .narrow(0, left, count)
            .copy_(s.kind() == "sum" ? total + correction : total);
        if (product) {
          lookup(zeros, index, 1).narrow(0, left, count).copy_(zero_count);
          lookup(nonzero, index, 1)
              .narrow(0, left, count)
              .copy_(nonzero_product);
        }
      }
    }
  }
  checks.finish();
  return {result, indices, zeros, nonzero};
}
// Higher-order replay retains the original operand views and cotangents.
// Its graph deliberately retains all tiles, so memory is no longer bounded.
Tensor evaluate_higher_order(at::TensorList local, const Settings& s,
                             bool paired, int64_t left, int64_t right,
                             const Tensor& rows, const Tensor& columns) {
  if (s.expression)
    return expression_evaluate_higher_order(local, s, paired);
  TORCH_CHECK(
      !s.family || *s.family == "_matrix",
      "Higher derivatives of this native streaming family are unsupported");
  if (s.family && *s.family == "_matrix")
    return evaluate(local, s, paired, left, right, rows, columns);
  auto o = comparison_options(s, local);
  TORCH_CHECK(
      !o.w && !o.V && !o.VI && !o.bin_positions && !o.projections,
      "Higher derivatives of parameterized native comparisons are unsupported");
  auto name = detail::canonical_comparison_name(s.metric);
  auto value = detail::higher_order_comparison(
      local[X], local[Y], name, s.kernel, o.p.value_or(2.),
      o.degree.value_or(3),
      o.gamma.value_or(name == "chi2" ? 1. : 1. / local[X].size(-1)),
      o.coef0.value_or(1.), o.length_scale.value_or(1.), o.alpha.value_or(1.),
      o.nu.value_or(1.5), paired);
  if (s.transposed() && !paired)
    value = value.transpose(-1, -2);
  if (s.self_cosine) {
    auto i =
        rows.defined()
            ? rows
            : at::arange(left, left + local[s.transposed() ? Y : X].size(0),
                         local[X].options().dtype(at::kLong));
    auto j =
        columns.defined()
            ? columns
            : at::arange(right, right + local[s.transposed() ? X : Y].size(0),
                         local[Y].options().dtype(at::kLong));
    value = at::where(paired ? i.eq(j) : i.unsqueeze(-1).eq(j.unsqueeze(0)), 0.,
                      value);
  }
  return value;
}
Tensors streaming_reduce_higher_order(const Tensor& gradient, at::TensorList v,
                                      const Tensor& indices,
                                      const Settings& s) {
  TORCH_CHECK(
      s.kind() != "prod",
      "Higher derivatives of native product reductions are unsupported");
  if (!s.expression && !(s.family && *s.family == "_matrix")) {
    TORCH_CHECK(
        !s.family,
        "Higher derivatives of this native streaming family are unsupported");
    auto name = detail::canonical_comparison_name(s.metric);
    bool supported = false;
    for (auto metric : {"sqeuclidean", "euclidean", "manhattan", "minkowski",
                        "rbf", "laplacian", "linear", "polynomial", "sigmoid",
                        "cosine", "matern", "rational_quadratic"})
      supported |= name == metric;
    TORCH_CHECK(supported, "Higher derivatives of comparison ", name,
                " are unsupported");
  }
  Tensors active, buffers(v.size());
  std::vector<size_t> slots;
  for (size_t slot = 0; slot < v.size(); ++slot) {
    buffers[slot] =
        v[slot].requires_grad() ? v[slot] * 0 : at::zeros_like(v[slot]);
    if (!v[slot].requires_grad())
      continue;
    bool duplicate = false;
    for (const auto& operand : active)
      duplicate |= operand.is_same(v[slot]);
    if (!duplicate) {
      active.push_back(v[slot]);
      slots.push_back(slot);
    }
  }
  auto add_tile = [&](const Tensor& value, const Tensor& cotangent) {
    if (active.empty() || !value.requires_grad())
      return;
    auto derivatives =
        torch::autograd::grad({value}, active, {cotangent}, true, true, true);
    for (size_t j = 0; j < slots.size(); ++j)
      if (derivatives[j].defined())
        buffers[slots[j]] = buffers[slots[j]] + derivatives[j];
  };
  auto batch = batch_shape(v, s);
  int64_t batches = 1;
  for (auto dim : batch)
    batches *= dim;
  auto n = v[s.transposed() ? Y : X].size(-s.rank);
  auto m = v[s.transposed() ? X : Y].size(-s.rank);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    if (s.kind() == "neighbors" || s.kind() == "diagonal") {
      bool diagonal = s.kind() == "diagonal";
      auto length = diagonal ? n : n * s.k;
      auto ids = diagonal ? Tensor() : lookup(indices, index, 2).reshape({-1});
      auto grad = lookup(gradient, index, diagonal ? 1 : 2).reshape({-1});
      for (int64_t start = 0; start < length; start += s.size) {
        auto count = std::min(s.size, length - start);
        auto sequence =
            at::arange(start, start + count, v[X].options().dtype(at::kLong));
        auto rows = diagonal ? sequence : at::floor_divide(sequence, s.k);
        auto columns = diagonal ? sequence : ids.narrow(0, start, count);
        auto local =
            local_inputs(v, s, index, 0, 0, 0, 0, rows, columns, false);
        add_tile(evaluate_higher_order(local, s, true, 0, 0, rows, columns),
                 grad.narrow(0, start, count));
      }
      continue;
    }
    for (int64_t left = 0; left < n; left += s.size) {
      auto count = std::min(s.size, n - left);
      for (int64_t right = 0; right < m; right += s.size) {
        auto other = std::min(s.size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other);
        auto value = evaluate_higher_order(local, s, false, left, right);
        if (s.exclusion) {
          auto rows =
              at::arange(left, left + count, v[X].options().dtype(at::kLong));
          auto columns = at::arange(right, right + other, rows.options());
          value = at::where(eligible(rows, columns, true), value, 0.);
        }
        if (s.kind() == "entries")
          add_tile(value, lookup(gradient, index, 2)
                              .narrow(0, left, count)
                              .narrow(1, right, other));
        else if (s.kind() == "matmul" || s.kind() == "kernel_sum") {
          auto payload = lookup(v[Payload], index, 2).narrow(0, right, other);
          for (int64_t channel = 0; channel < payload.size(-1);
               channel += s.size) {
            auto channels = std::min(s.size, payload.size(-1) - channel);
            auto output = value.to(payload.scalar_type())
                              .matmul(payload.narrow(-1, channel, channels));
            add_tile(output, lookup(gradient, index, 2)
                                 .narrow(0, left, count)
                                 .narrow(-1, channel, channels));
          }
        } else {
          TORCH_CHECK(
              s.kind() == "sum",
              "Higher derivatives of this native reduction are unsupported");
          add_tile(value, lookup(gradient, index, 1)
                              .narrow(0, left, count)
                              .unsqueeze(-1)
                              .expand_as(value));
        }
      }
    }
  }
  return buffers;
}
void accumulate(at::TensorList local, const Tensor& value,
                const Tensor& cotangent, Tensors& buffers, const Settings& s,
                const std::vector<int64_t>& index, int64_t left, int64_t count,
                int64_t right, int64_t other, const Tensor& rows,
                const Tensor& columns, const ExpressionReplay* replay) {
  if (s.expression) {
    expression_accumulate(local, cotangent, buffers, s, index, replay);
    return;
  }
  if (s.family && *s.family == "_matrix") {
    if (!local[X].requires_grad())
      return;
    // The matrix already owns its full values. Scatter cotangents directly;
    // autograd through a row view would allocate count*full_columns per tile.
    at::NoGradGuard guard;
    auto destination = lookup(buffers[X], index, 2);
    auto ids = local[Y].squeeze(-1);
    auto cot = cotangent.to(at::kDouble);
    if (rows.defined()) {
      auto source_rows = s.transposed() ? columns : rows;
      destination.index_put_({source_rows, ids}, cot, true);
    } else {
      auto target = destination.narrow(0, s.transposed() ? right : left,
                                       s.transposed() ? other : count);
      target.index_add_(-1, ids, s.transposed() ? cot.transpose(-1, -2) : cot);
    }
    return;
  }
  Tensors active;
  std::vector<int64_t> slots;
  for (int64_t slot = 0; slot < Slots; ++slot)
    if (s.has(slot) && local[slot].requires_grad() && slot != Payload) {
      active.push_back(local[slot]);
      slots.push_back(slot);
    }
  if (active.empty() || !value.requires_grad())
    return;
  auto derivatives =
      torch::autograd::grad({value}, active, {cotangent}, false, false, true);
  at::NoGradGuard no_grad;
  for (size_t j = 0; j < slots.size(); ++j)
    if (derivatives[j].defined()) {
      auto slot = slots[j], rank = s.operand_rank(slot);
      auto destination =
          rank > 0 ? lookup(buffers[slot], index, rank) : buffers[slot];
      auto derivative = derivatives[j].to(at::kDouble);
      if (s.sampled(slot)) {
        bool first = s.first_operand(slot) != s.transposed();
        if (rows.defined())
          destination.index_add_(0, first ? rows : columns, derivative);
        else
          destination.narrow(0, first ? left : right, first ? count : other)
              .add_(derivative);
      } else
        destination.add_(derivative);
    }
}
Tensors streaming_reduce_backward_kernel(const Tensor& gradient,
                                         at::TensorList v, const Tensor& result,
                                         const Tensor& indices,
                                         const Tensor& zeros,
                                         const Tensor& nonzero,
                                         const Settings& s) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU),
      no_cuda_autocast(c10::DispatchKey::AutocastCUDA);
  if (at::GradMode::is_enabled())
    return streaming_reduce_higher_order(gradient, v, indices, s);
  utils::DeferredValueChecks checks;
#ifdef WITH_CUDA
  if (s.kind() == "matmul" || s.kind() == "kernel_sum" || s.kind() == "sum")
    if (auto plan = fused::plan(v, s))
      if (plan->differentiable &&
          plan->rows.size(2) <= fused::max_gradient_features) {
        fused::count();
        bool matmul = s.kind() != "sum";
        // Analytic pair gradients weighted by g_i . v_j. Every slot is
        // computed, as on the tile path: compiled backward graphs call this
        // with tensors that do not record requires_grad.
        auto left = matmul ? gradient : gradient.unsqueeze(-1);
        auto right =
            matmul ? v[Payload]
                   : at::ones({plan->columns.size(1), 1}, gradient.options());
        bool exclusion = !matmul && s.exclusion;
        Tensors result;
        for (const auto& value : v)
          result.push_back(at::zeros_like(value));
        fused::operand_gradients(*plan, left, right, exclusion, {}, {}, result,
                                 v);
        auto transposed = fused::swapped(*plan);
        if (matmul) {
          auto shape = plan->batch;
          shape.insert(shape.end(), {v[Payload].size(-2), v[Payload].size(-1)});
          result[Payload] =
              at::sum_to(fused::matmul_cuda(transposed, gradient, false)
                             .reshape(shape),
                         v[Payload].sizes())
                  .contiguous();
        }
        checks.finish();
        return result;
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
  for (auto dim : batch)
    batches *= dim;
  auto n = v[s.transposed() ? Y : X].size(-s.rank),
       m = v[s.transposed() ? X : Y].size(-s.rank);
  at::AutoGradMode enable(true);
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    if (s.kind() == "diagonal") {
      for (int64_t start = 0; start < n; start += s.size) {
        auto count = std::min(s.size, n - start);
        auto ids =
            at::arange(start, start + count, v[X].options().dtype(at::kLong));
        auto local = local_inputs(v, s, index, 0, 0, 0, 0, ids, ids, true);
        ExpressionReplay replay;
        auto value = evaluate(local, s, true, 0, 0, ids, ids, &replay);
        accumulate(local, value,
                   lookup(gradient, index, 1).narrow(0, start, count), buffers,
                   s, index, 0, 0, 0, 0, ids, ids, &replay);
      }
      continue;
    }
    if (s.kind() == "neighbors") {
      auto ids = lookup(indices, index, 2).reshape({-1}),
           grad = lookup(gradient, index, 2).reshape({-1});
      for (int64_t start = 0; start < n * s.k; start += s.size) {
        auto count = std::min(s.size, n * s.k - start);
        auto rows = at::floor_divide(
                 at::arange(start, start + count, indices.options()), s.k),
             columns = ids.narrow(0, start, count);
        auto local = local_inputs(v, s, index, 0, 0, 0, 0, rows, columns, true);
        ExpressionReplay replay;
        auto value = evaluate(local, s, true, 0, 0, rows, columns, &replay);
        accumulate(local, value, grad.narrow(0, start, count), buffers, s,
                   index, 0, 0, 0, 0, rows, columns, &replay);
      }
      continue;
    }
    for (int64_t left = 0; left < n; left += s.size) {
      auto count = std::min(s.size, n - left);
      for (int64_t right = 0; right < m; right += s.size) {
        auto other = std::min(s.size, m - right);
        auto local = local_inputs(v, s, index, left, count, right, other,
                                  Tensor(), Tensor(), true);
        ExpressionReplay replay;
        auto value =
            evaluate(local, s, false, left, right, Tensor(), Tensor(), &replay);
        Tensor cotangent;
        if (s.kind() == "entries")
          cotangent = lookup(gradient, index, 2)
                          .narrow(0, left, count)
                          .narrow(1, right, other);
        else if ((s.kind() == "kernel_sum" || s.kind() == "matmul")) {
          auto payload = lookup(v[Payload], index, 2).detach(),
               grad = lookup(gradient, index, 2);
          cotangent = at::zeros_like(value, value.options().dtype(at::kDouble));
          at::NoGradGuard no_grad;
          for (int64_t channel = 0; channel < payload.size(-1);
               channel += s.size) {
            auto channels = std::min(s.size, payload.size(-1) - channel);
            auto g = grad.narrow(0, left, count).narrow(-1, channel, channels);
            cotangent.add_(g.matmul(payload.narrow(0, right, other)
                                        .narrow(-1, channel, channels)
                                        .transpose(-1, -2)));
            lookup(buffers[Payload], index, 2)
                .narrow(0, right, other)
                .narrow(-1, channel, channels)
                .add_(value.detach()
                          .to(g.scalar_type())
                          .transpose(-1, -2)
                          .matmul(g));
          }
          cotangent = cotangent.to(value.scalar_type());
        } else {
          auto grad =
              lookup(gradient, index, 1).narrow(0, left, count).unsqueeze(-1);
          at::NoGradGuard no_grad;
          if (s.kind() == "sum")
            cotangent = grad.expand_as(value);
          else {
            auto zero_count =
                lookup(zeros, index, 1).narrow(0, left, count).unsqueeze(-1);
            auto total =
                lookup(result, index, 1).narrow(0, left, count).unsqueeze(-1);
            auto product =
                lookup(nonzero, index, 1).narrow(0, left, count).unsqueeze(-1);
            auto derivative = at::where(
                zero_count.eq(0), total / at::where(value.eq(0), 1., value),
                at::where(zero_count.eq(1) & value.eq(0), product, 0.));
            cotangent = grad * derivative;
          }
          if (s.exclusion) {
            auto rows =
                at::arange(left, left + count, v[X].options().dtype(at::kLong));
            auto columns = at::arange(right, right + other, rows.options());
            cotangent = at::where(eligible(rows, columns, true), cotangent, 0.);
          }
        }
        accumulate(local, value, cotangent, buffers, s, index, left, count,
                   right, other, Tensor(), Tensor(), &replay);
      }
    }
  }
  for (size_t slot = 0; slot < v.size(); ++slot)
    buffers[slot] = buffers[slot].to(v[slot].scalar_type());
  checks.finish();
  return buffers;
}
Outputs streaming_reduce_impl(at::TensorList v,
                              TORCHPAIRWISE_STREAMING_META_DECL) {
  return streaming_reduce_kernel(
      v, settings(metric, operation, kernel, rank, k, largest, exclusion,
                  self_cosine, nan_first, size, dtype, mask, std::move(reals),
                  std::move(integers), std::move(flags), family));
}
Tensors streaming_reduce_backward_impl(const Tensor& gradient, at::TensorList v,
                                       const Tensor& result,
                                       const Tensor& index, const Tensor& zeros,
                                       const Tensor& nonzero,
                                       TORCHPAIRWISE_STREAMING_META_DECL) {
  return streaming_reduce_backward_kernel(
      gradient, v, result, index, zeros, nonzero,
      settings(metric, operation, kernel, rank, k, largest, exclusion,
               self_cosine, nan_first, size, dtype, mask, std::move(reals),
               std::move(integers), std::move(flags), family));
}

bool cloud_metric(c10::string_view metric) {
  for (auto name : {"chamfer", "hausdorff", "directed_hausdorff", "wasserstein",
                    "sliced_wasserstein", "sinkhorn", "energy", "mmd"})
    if (metric == name || metric == std::string(name) + "_distances" ||
        (metric == "sinkhorn_divergences" && std::string(name) == "sinkhorn"))
      return true;
  return false;
}
}  // namespace detail::streaming

std::tuple<Tensor, Tensor> streaming_reductions(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view metric, c10::string_view operation, bool kernel,
    const StreamingReductionOptions& options) {
  using Signature = std::tuple<Tensor, Tensor>(
      const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
      c10::string_view operation, bool kernel,
      const c10::optional<Tensor>& values, int64_t k, bool largest,
      c10::optional<bool> exclude_self, bool keepdim,
      c10::optional<int64_t> chunk_size, const c10::optional<Tensor>& w,
      const c10::optional<Tensor>& variance,
      const c10::optional<Tensor>& inverse, const c10::optional<Tensor>& bins,
      const c10::optional<Tensor>& projections, const c10::optional<Tensor>& xl,
      const c10::optional<Tensor>& yl, c10::optional<double> p,
      c10::optional<double> base, c10::optional<double> alpha,
      c10::optional<double> length_scale, c10::optional<double> nu,
      c10::optional<double> gamma, c10::optional<double> coef0,
      c10::optional<double> regularization, c10::optional<int64_t> degree,
      c10::optional<int64_t> max_iter, c10::optional<int64_t> metric_chunk_size,
      c10::optional<bool> squared, c10::optional<bool> shuffle,
      c10::optional<bool> batched, c10::optional<int64_t> feature_dim,
      c10::optional<c10::string_view> family,
      const c10::optional<at::Generator>& generator);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::streaming_reductions", "")
          .typed<Signature>();
  return op.call(
      input, other, metric, operation, kernel, options.values, options.k,
      options.largest, options.exclude_self, options.keepdim,
      options.chunk_size, options.comparison.w, options.comparison.V,
      options.comparison.VI, options.comparison.bin_positions,
      options.comparison.projections, options.comparison.x1_lengths,
      options.comparison.x2_lengths, options.comparison.p,
      options.comparison.base, options.comparison.alpha,
      options.comparison.length_scale, options.comparison.nu,
      options.comparison.gamma, options.comparison.coef0,
      options.comparison.regularization, options.comparison.degree,
      options.comparison.max_iter, options.comparison.chunk_size,
      options.comparison.squared, options.comparison.shuffle,
      options.comparison.batched, options.comparison.feature_dim,
      options.comparison.kernel
          ? c10::optional<c10::string_view>(*options.comparison.kernel)
          : c10::nullopt,
      options.comparison.generator);
}
namespace {
using namespace detail::streaming;
std::tuple<Tensor, Tensor> streaming_reductions_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, bool kernel,
    const c10::optional<Tensor>& values, int64_t k, bool largest,
    c10::optional<bool> exclude_self, bool keepdim,
    c10::optional<int64_t> chunk_size, const c10::optional<Tensor>& w,
    const c10::optional<Tensor>& variance, const c10::optional<Tensor>& inverse,
    const c10::optional<Tensor>& bins, const c10::optional<Tensor>& projections,
    const c10::optional<Tensor>& xl, const c10::optional<Tensor>& yl,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<double> alpha, c10::optional<double> length_scale,
    c10::optional<double> nu, c10::optional<double> gamma,
    c10::optional<double> coef0, c10::optional<double> regularization,
    c10::optional<int64_t> degree, c10::optional<int64_t> max_iter,
    c10::optional<int64_t> metric_chunk_size, c10::optional<bool> squared,
    c10::optional<bool> shuffle, c10::optional<bool> batched,
    c10::optional<int64_t> feature_dim, c10::optional<c10::string_view> family,
    const c10::optional<at::Generator>& generator) {
  StreamingReductionOptions o;
  o.values = values;
  o.k = k;
  o.largest = largest;
  o.exclude_self = exclude_self;
  o.keepdim = keepdim;
  o.chunk_size = chunk_size;
  auto& c = o.comparison;
  c.w = w;
  c.V = variance;
  c.VI = inverse;
  c.bin_positions = bins;
  c.projections = projections;
  c.x1_lengths = xl;
  c.x2_lengths = yl;
  c.p = p;
  c.base = base;
  c.alpha = alpha;
  c.length_scale = length_scale;
  c.nu = nu;
  c.gamma = gamma;
  c.coef0 = coef0;
  c.regularization = regularization;
  c.degree = degree;
  c.max_iter = max_iter;
  c.chunk_size = metric_chunk_size;
  c.squared = squared;
  c.shuffle = shuffle;
  c.batched = batched;
  c.feature_dim = feature_dim;
  if (family)
    c.kernel = std::string(*family);
  c.generator = generator;
  return streaming_reductions_composite(x, y, metric, operation, kernel, o);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "streaming_reductions(Tensor x1, Tensor? x2=None, str "
      "metric='euclidean', str operation='sum', bool kernel=False, Tensor? "
      "values=None, int k=1, bool largest=False, bool? exclude_self=None, bool "
      "keepdim=False, int? chunk_size=None, Tensor? w=None, Tensor? V=None, "
      "Tensor? VI=None, Tensor? bin_positions=None, Tensor? projections=None, "
      "Tensor? x1_lengths=None, Tensor? x2_lengths=None, float? p=None, float? "
      "base=None, float? alpha=None, float? length_scale=None, float? nu=None, "
      "float? gamma=None, float? coef0=None, float? regularization=None, int? "
      "degree=None, int? max_iter=None, int? metric_chunk_size=None, bool? "
      "squared=None, bool? shuffle=None, bool? batched=None, int? "
      "feature_dim=None, str? family=None, Generator? generator=None) -> "
      "(Tensor, Tensor)",
      TORCH_FN(streaming_reductions_impl));
  m.def(
      "_streaming_reduce(Tensor[] operands, str metric, str operation, bool "
      "kernel, int rank, int k, bool largest, bool exclusion, bool "
      "self_cosine, bool nan_first, int size, int dtype, int mask, float?[] "
      "reals, int?[] integers, bool?[] flags, str? family) -> (Tensor, Tensor, "
      "Tensor, Tensor)");
  m.def(
      "_streaming_reduce_backward(Tensor grad, Tensor[] operands, Tensor "
      "result, Tensor indices, Tensor zeros, Tensor nonzero, str metric, str "
      "operation, bool kernel, int rank, int k, bool largest, bool exclusion, "
      "bool self_cosine, bool nan_first, int size, int dtype, int mask, "
      "float?[] reals, int?[] integers, bool?[] flags, str? family) -> "
      "Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_streaming_reduce", TORCH_FN(streaming_reduce_impl));
  m.impl("_streaming_reduce_backward",
         TORCH_FN(streaming_reduce_backward_impl));
}

#undef TORCHPAIRWISE_STREAMING_META_DECL
#undef TORCHPAIRWISE_STREAMING_META_TYPES
#undef TORCHPAIRWISE_STREAMING_META_ARGS
}  // namespace
namespace detail::streaming {
PreparedComparison prepare(const Tensor& input,
                           const c10::optional<Tensor>& other,
                           c10::string_view metric, c10::string_view operation,
                           bool kernel,
                           const StreamingReductionOptions& options,
                           bool transposed) {
  bool omitted = !other.has_value(), search = operation == "neighbors" ||
                                              operation == "min" ||
                                              operation == "max";
  TORCH_CHECK(search || operation == "sum" || operation == "prod" ||
                  operation == "all" || operation == "any" ||
                  operation == "kernel_sum" || operation == "matmul",
              "Unknown reduction operation");
  TORCH_CHECK(operation != "kernel_sum" || kernel,
              "kernel_sum requires a kernel");
  TORCH_CHECK((operation == "kernel_sum" || operation == "matmul") ==
                  options.values.has_value(),
              "values are required only for kernel_sum");
  auto comparison = options.comparison;
  TORCH_CHECK(
      !comparison.shuffle.value_or(false) && !comparison.generator.has_value(),
      "Streaming reductions do not support shuffle or generator");
  Tensor x = input, y = other.value_or(input);
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided,
              "Inputs must be dense strided tensors");
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Inputs must have the same dtype and device");
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Reduced operations support CPU and CUDA inputs");
  bool autocast = !x.device().is_meta() &&
                  at::autocast::is_autocast_enabled(x.device().type());
  auto cast = [&](Tensor value) {
    return autocast && (value.scalar_type() == at::kHalf ||
                        value.scalar_type() == at::kBFloat16)
               ? value.to(at::kFloat)
               : value;
  };
  x = cast(x);
  y = cast(y);
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU),
      no_cuda_autocast(c10::DispatchKey::AutocastCUDA);
  TORCH_CHECK(!x.is_floating_point() || x.scalar_type() == at::kFloat ||
                  x.scalar_type() == at::kDouble,
              "Reduced operations require float32 or float64 outside autocast");
  auto rank = cloud_metric(metric) ? int64_t(3) : int64_t(2);
  // Vector tiles are cheap per pair, so a small fixed tile is dominated by
  // dispatch; point-set pairs keep the smaller tile until profiled.
  auto size = options.chunk_size.value_or(
      rank == 3 ? int64_t(128) : utils::default_tile_side());
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  if (rank == 3) {
    x = detail::cloud_layout(x, comparison.batched,
                             comparison.feature_dim.value_or(-1));
    y = detail::cloud_layout(y, comparison.batched,
                             comparison.feature_dim.value_or(-1));
    comparison.batched = c10::nullopt;
    comparison.feature_dim = c10::nullopt;
    if (omitted && !comparison.x2_lengths)
      comparison.x2_lengths = comparison.x1_lengths;
  } else
    TORCH_CHECK(!comparison.batched && !comparison.feature_dim,
                "batched/feature_dim require point-set inputs");
  TORCH_CHECK(x.dim() >= rank && y.dim() >= rank,
              "Inputs have insufficient rank");
  TORCH_CHECK(x.sym_size(-1) > 0 && x.sym_size(-1) == y.sym_size(-1),
              "Features must match and be non-empty");
  if (metric == "seuclidean" || metric == "seuclidean_distances") {
    TORCH_CHECK(comparison.V || omitted,
                "V is required for seuclidean when x2 is passed");
    if (!comparison.V) {
      at::NoGradGuard no_grad;
      comparison.V = x.detach().var({-2}, true, false);
    }
  }
  if (metric == "mahalanobis" || metric == "mahalanobis_distances") {
    TORCH_CHECK(comparison.VI || omitted,
                "VI is required for Mahalanobis when x2 is passed");
    if (!comparison.VI) {
      TORCH_CHECK(x.sym_size(-2) >= 2,
                  "At least two samples are required to infer covariance");
      at::NoGradGuard no_grad;
      auto centered = x - x.mean(-2, true);
      comparison.VI =
          at::linalg_inv(centered.transpose(-1, -2).matmul(centered) /
                         (x.sym_size(-2) - 1))
              .transpose(-1, -2);
    }
  }
  auto empty = at::empty({0}, x.options());
  Tensors v{x, y, empty, empty, empty, empty, empty, empty, empty, empty};
  int64_t mask = 3;
  auto set = [&](int64_t slot, c10::optional<Tensor>& value) {
    if (value) {
      v[slot] = cast(*value);
      value = v[slot];
      mask |= int64_t(1) << slot;
    }
  };
  set(W, comparison.w);
  set(Variance, comparison.V);
  set(Inverse, comparison.VI);
  set(Bins, comparison.bin_positions);
  set(Projections, comparison.projections);
  set(XL, comparison.x1_lengths);
  set(YL, comparison.x2_lengths);
  if (options.values) {
    v[Payload] = cast(*options.values);
    mask |= int64_t(1) << Payload;
  }
  auto n = (transposed ? y : x).sym_size(-rank),
       m = (transposed ? x : y).sym_size(-rank);
  bool exclusion =
      options.exclude_self.value_or(operation == "neighbors" && omitted);
  TORCH_CHECK(!exclusion || n == m,
              "exclude_self=True requires equal sample counts");
  auto k = operation == "neighbors" ? options.k : int64_t(1);
  TORCH_CHECK(!search || (k >= 0 && (k == 0 || k <= m - int64_t(exclusion))),
              "k must be between zero and the eligible reference count");
  TORCH_CHECK(
      (operation != "min" && operation != "max") || m - int64_t(exclusion) > 0,
      "min/max require at least one eligible reference");
  Tensor probe;
  {
    at::NoGradGuard no_grad;
    auto left_options = comparison, right_options = comparison;
    if (left_options.x2_lengths)
      left_options.x2_lengths = left_options.x2_lengths->narrow(-1, 0, 0);
    if (right_options.x1_lengths)
      right_options.x1_lengths = right_options.x1_lengths->narrow(-1, 0, 0);
    probe = builtin_comparisons(x, y.narrow(-rank, 0, 0), metric, "all_pairs",
                                kernel, left_options);
    builtin_comparisons(x.narrow(-rank, 0, 0), y, metric, "all_pairs", kernel,
                        right_options);
  }
  if (options.values) {
    auto value = v[Payload];
    TORCH_CHECK(
        value.layout() == at::kStrided && value.dim() >= 2 &&
            value.sym_size(-2) == m && value.sym_size(-1) > 0,
        "values must have shape [...,M,Cv] with matching M and positive Cv");
    TORCH_CHECK(value.device() == x.device() &&
                    value.scalar_type() == probe.scalar_type(),
                "values must have the input dtype and device");
  }
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - rank),
                                     y.sym_sizes().slice(0, y.dim() - rank));
  for (int64_t slot = W; slot < Slots; ++slot)
    if ((mask & (int64_t(1) << slot)) && ranks[slot] > 0) {
      TORCH_CHECK(v[slot].dim() >= ranks[slot],
                  "Parameter has insufficient rank");
      batch = at::infer_size_symint(
          batch, v[slot].sym_sizes().slice(0, v[slot].dim() - ranks[slot]));
    }
  Settings s{
      std::string(metric),
      (transposed ? "transpose_" : "") +
          (search ? std::string("neighbors") : std::string(operation)),
      kernel,
      operation == "max" || (operation == "neighbors" && options.largest),
      exclusion,
      (!other || input.is_same(*other)) && !kernel &&
          (metric == "cosine" || metric == "cosine_distances"),
      operation == "min" || operation == "max",
      rank,
      k,
      size,
      int64_t(probe.scalar_type()),
      mask,
      {comparison.p, comparison.base, comparison.alpha, comparison.length_scale,
       comparison.nu, comparison.gamma, comparison.coef0,
       comparison.regularization},
      {comparison.degree, comparison.max_iter, comparison.chunk_size},
      {comparison.squared, comparison.shuffle},
      comparison.kernel};
  return {std::move(v), std::move(s), std::move(batch), n, m};
}
}  // namespace detail::streaming
namespace {
std::tuple<Tensor, Tensor> streaming_reductions_composite(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view metric, c10::string_view operation, bool kernel,
    const StreamingReductionOptions& options) {
  auto prepared = detail::streaming::prepare(input, other, metric, operation,
                                             kernel, options);
  auto& v = prepared.values;
  auto& s = prepared.settings;
  auto x = v[detail::streaming::X];
  auto output = _streaming_reduce(v, s);
  auto result = std::get<0>(output);
  if (operation == "min" || operation == "max")
    result = result.squeeze(-1);
  if (options.keepdim && operation != "neighbors" && operation != "kernel_sum")
    result = result.unsqueeze(-1);
  auto indices = operation == "neighbors"
                     ? std::get<1>(output)
                     : at::empty({0}, x.options().dtype(at::kLong));
  return {result, indices};
}
}  // namespace
}  // namespace torchpairwise::ops
