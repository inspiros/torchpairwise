#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "residual_operations.h"
#include "residual_operations_detail.h"
#include "residual_reducer.h"
#include "utils/centered_moments.h"
#include "utils/operator_axes.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using namespace detail::streaming;
Tensor to_double(const Tensor& t) {
  return t.scalar_type() == at::kDouble ? t : t.to(at::kDouble);
}
Tensor to_dtype(const Tensor& t, at::ScalarType dtype) {
  return t.scalar_type() == dtype ? t : t.to(dtype);
}
std::array<bool, 3> reduction_axes(at::IntArrayRef axes) {
  std::array<bool, 3> reduced{false, false, false};
  TORCH_CHECK(!axes.empty(), "A residual reduction requires an axis");
  for (auto axis : axes) {
    TORCH_CHECK(axis >= 0 && axis < 3 && !reduced[axis],
                "Invalid residual axes");
    reduced[axis] = true;
  }
  return reduced;
}
void validate(const Tensor& x, const Tensor& y, c10::string_view reduction,
              c10::string_view transform, double correction, int64_t sample,
              int64_t features) {
  TORCH_CHECK(
      x.dim() >= 2 && y.dim() >= 2 && x.layout() == at::kStrided &&
          y.layout() == at::kStrided && x.device() == y.device() &&
          x.scalar_type() == y.scalar_type() &&
          (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
      "Residual inputs require dense matching float32/float64 tensors");
  TORCH_CHECK(x.sym_size(-1) == y.sym_size(-1),
              "Residual feature counts differ");
  TORCH_CHECK(reduction == "sum" || reduction == "mean" || reduction == "var" ||
                  reduction == "std" ||
                  detail::residual::extra_reduction(reduction),
              "Unsupported residual reduction");
  TORCH_CHECK(
      transform == "identity" || transform == "abs" || transform == "square",
      "Residual transform must be identity, abs or square");
  TORCH_CHECK(std::isfinite(correction), "correction must be finite");
  TORCH_CHECK(sample > 0 && features > 0, "Tile sizes must be positive");
}
Tensor residual_tile(const Tensor& x, const Tensor& y, int64_t i, int64_t ni,
                     int64_t j, int64_t nj, int64_t d, int64_t nd) {
  return x.narrow(0, i, ni).narrow(-1, d, nd).unsqueeze(-2) -
         y.narrow(0, j, nj).narrow(-1, d, nd).unsqueeze(0);
}
Tensor transform_value(const Tensor& value, c10::string_view transform) {
  return transform == "abs"      ? value.abs()
         : transform == "square" ? value.square()
                                 : value;
}
Tensor output_tile(Tensor value, const std::array<bool, 3>& axes,
                   const std::array<int64_t, 3>& start,
                   const std::array<int64_t, 3>& width) {
  for (int64_t axis = 0; axis < 3; ++axis)
    value = value.narrow(axis, axes[axis] ? 0 : start[axis],
                         axes[axis] ? 1 : width[axis]);
  return value;
}
std::vector<int64_t> internal_shape(const Tensor& x, const Tensor& y,
                                    const std::vector<int64_t>& batch,
                                    const std::array<bool, 3>& axes) {
  auto shape = batch;
  shape.push_back(axes[0] ? 1 : x.size(-2));
  shape.push_back(axes[1] ? 1 : y.size(-2));
  shape.push_back(axes[2] ? 1 : x.size(-1));
  return shape;
}
}  // namespace
namespace detail::residual {
FeatureOutputs reduction_kernel(const Tensor& x, const Tensor& y,
                                c10::string_view reduction,
                                c10::string_view transform,
                                at::IntArrayRef axes, bool keepdim,
                                double correction, int64_t sample_size,
                                int64_t feature_size) {
  validate(x, y, reduction, transform, correction, sample_size, feature_size);
  auto reduced = reduction_axes(axes);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto batch = at::infer_size(x.sizes().slice(0, x.dim() - 2),
                              y.sizes().slice(0, y.dim() - 2));
  auto shape = internal_shape(x, y, batch, reduced), state_shape = shape;
  bool moments = reduction == "var" || reduction == "std";
  bool anchored = moments && x.scalar_type() == at::kDouble;
  state_shape.push_back(anchored ? 4 : 3);
  bool extra = extra_reduction(reduction);
  auto state = extra ? reducer_state(shape, x.options(), reduction)
                     : at::zeros(state_shape, x.options().dtype(at::kDouble));
  auto n = x.size(-2), m = y.size(-2), features = x.size(-1);
  std::array<int64_t, 3> sizes{n, m, features};
  for (auto axis : axes)
    TORCH_CHECK(sizes[axis] > 0 || (reduction != "min" && reduction != "max"),
                "min/max require nonempty reduction axes");
  int64_t batches = 1;
  for (auto dimension : batch)
    batches *= dimension;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto a = lookup(x, index, 2), b = lookup(y, index, 2);
    auto saved = lookup(state, index, 4);
    for (int64_t i = 0; i < n; i += sample_size)
      for (int64_t j = 0; j < m; j += sample_size)
        for (int64_t d = 0; d < features; d += feature_size) {
          std::array<int64_t, 3> start{i, j, d};
          std::array<int64_t, 3> width{std::min(sample_size, n - i),
                                       std::min(sample_size, m - j),
                                       std::min(feature_size, features - d)};
          auto raw = residual_tile(a, b, i, width[0], j, width[1], d, width[2]);
          auto values = to_double(transform_value(raw, transform));
          auto target = output_tile(saved, reduced, start, width);
          if (extra) {
            auto opts = x.options().dtype(at::kLong);
            auto ids = (at::arange(i, i + width[0], opts).view({-1, 1, 1}) * m +
                        at::arange(j, j + width[1], opts).view({1, -1, 1})) *
                           features +
                       at::arange(d, d + width[2], opts).view({1, 1, -1});
            reducer_update(target, values, ids, axes, reduction);
            continue;
          }
          auto count = target.select(-1, 0), mean = target.select(-1, 1),
               m2 = target.select(-1, 2);
          double k = 1.;
          for (auto axis : axes)
            k *= double(width[axis]);
          if (anchored) {
            detail::update_centered_moments(target, values, axes, k);
          } else if (moments) {
            auto anchor = values;
            for (auto axis : axes)
              anchor = anchor.narrow(axis, 0, 1);
            auto safe = at::where(anchor.isfinite(), anchor, 0.);
            auto average = safe + (values - safe).sum(axes, true) / k;
            auto partial = (values - average).square().sum(axes, true);
            auto total = count + k;
            auto delta = average - mean;
            auto ratio = k / total;
            m2.copy_(at::where(count.gt(0),
                               m2 + partial + delta.square() * count * ratio,
                               partial));
            mean.copy_(at::where(count.gt(0), mean + delta * ratio, average));
            count.copy_(total);
          } else {
            auto update = compensated(mean, m2, values.sum(axes, true));
            mean.copy_(update.first);
            m2.copy_(update.second);
          }
        }
  }
  auto result = extra ? reducer_result(state, reduction)
                      : state.select(-1, 1) + state.select(-1, 2);
  if (extra)
    result = reducer_result(state, reduction).clone();
  else if (moments) {
    auto count = state.select(-1, 0);
    result = at::where(count.gt(0),
                       state.select(-1, 2) / (count - correction).clamp_min(0),
                       std::numeric_limits<double>::quiet_NaN());
    if (reduction == "std")
      result = result.sqrt();
  } else if (reduction == "mean") {
    std::array<int64_t, 3> sizes{n, m, features};
    double count = 1.;
    for (auto axis : axes)
      count *= double(sizes[axis]);
    result = result / count;
  }
  if (!keepdim)
    for (int64_t axis = 2; axis >= 0; --axis)
      if (reduced[axis])
        result = result.squeeze(int64_t(batch.size()) + axis);
  return {to_dtype(result,
                   boolean_reduction(reduction) ? at::kBool : x.scalar_type()),
          moments || extra ? state : at::empty({0}, state.options())};
}
std::tuple<Tensor, Tensor> reduction_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& state, c10::string_view reduction, c10::string_view transform,
    at::IntArrayRef axes, bool keepdim, double correction, int64_t sample_size,
    int64_t feature_size) {
  validate(x, y, reduction, transform, correction, sample_size, feature_size);
  auto reduced = reduction_axes(axes);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto gx = at::zeros_like(x, x.options().dtype(at::kDouble)),
       gy = at::zeros_like(y, y.options().dtype(at::kDouble));
  auto batch = at::infer_size(x.sizes().slice(0, x.dim() - 2),
                              y.sizes().slice(0, y.dim() - 2));
  auto g = to_double(gradient).reshape(internal_shape(x, y, batch, reduced));
  bool moments = reduction == "var" || reduction == "std";
  auto n = x.size(-2), m = y.size(-2), features = x.size(-1);
  std::array<int64_t, 3> sizes{n, m, features};
  // The saved witness uses a double logical index. Keep the original mask
  // replay if the flattened coordinates cannot all be represented exactly.
  constexpr int64_t exact_limit = int64_t(1) << 53;
  bool exact_ids =
      m == 0 || (n <= exact_limit / m &&
                 (features == 0 || n * m <= exact_limit / features));
  bool witness_only = transform == "identity" && exact_ids &&
                      (reduction == "min" || reduction == "max");
  double eligible = 1.;
  for (auto axis : axes)
    eligible *= double(sizes[axis]);
  int64_t batches = 1;
  for (auto dimension : batch)
    batches *= dimension;
  for (int64_t flat = 0; flat < batches; ++flat) {
    auto index = coordinates(flat, batch);
    auto a = lookup(x, index, 2), b = lookup(y, index, 2);
    auto upstream = lookup(g, index, 3);
    if (witness_only) {
      auto witnesses = lookup(state, index, 4).select(-1, 1);
      auto dx = lookup(gx, index, 2), dy = lookup(gy, index, 2);
      // Visit output-shaped witness blocks. A retained feature axis still
      // obeys feature_size, while contracted axes have just one coordinate.
      for (int64_t i = 0; i < witnesses.size(0); i += sample_size)
        for (int64_t j = 0; j < witnesses.size(1); j += sample_size)
          for (int64_t d = 0; d < witnesses.size(2); d += feature_size) {
            std::array<int64_t, 3> start{i, j, d};
            std::array<int64_t, 3> width{
                std::min(sample_size, witnesses.size(0) - i),
                std::min(sample_size, witnesses.size(1) - j),
                std::min(feature_size, witnesses.size(2) - d)};
            auto ids = output_tile(witnesses, reduced, start, width)
                           .reshape({-1})
                           .to(at::kLong);
            auto cotangent =
                output_tile(upstream, reduced, start, width).reshape({-1});
            auto feature = ids.remainder(features);
            auto rows = at::floor_divide(ids, m * features);
            auto columns = at::floor_divide(ids, features).remainder(m);
            dx.index_put_({rows, feature}, cotangent, true);
            dy.index_put_({columns, feature}, -cotangent, true);
          }
      continue;
    }
    for (int64_t i = 0; i < n; i += sample_size)
      for (int64_t j = 0; j < m; j += sample_size)
        for (int64_t d = 0; d < features; d += feature_size) {
          std::array<int64_t, 3> start{i, j, d};
          std::array<int64_t, 3> width{std::min(sample_size, n - i),
                                       std::min(sample_size, m - j),
                                       std::min(feature_size, features - d)};
          Tensor raw, values;
          if (moments || reduction == "prod" || transform != "identity")
            raw = residual_tile(a, b, i, width[0], j, width[1], d, width[2]);
          if (moments || reduction == "prod")
            values = to_double(transform_value(raw, transform));
          auto cotangent = output_tile(upstream, reduced, start, width);
          if (extra_reduction(reduction)) {
            auto saved =
                output_tile(lookup(state, index, 4), reduced, start, width);
            auto opts = x.options().dtype(at::kLong);
            auto ids = (at::arange(i, i + width[0], opts).view({-1, 1, 1}) * m +
                        at::arange(j, j + width[1], opts).view({1, -1, 1})) *
                           features +
                       at::arange(d, d + width[2], opts).view({1, 1, -1});
            cotangent =
                reducer_gradient(saved, values, ids, cotangent, reduction);
          } else if (moments) {
            auto saved =
                output_tile(lookup(state, index, 4), reduced, start, width);
            if (reduction == "std") {
              auto std = (saved.select(-1, 2) /
                          (saved.select(-1, 0) - correction).clamp_min(0))
                             .sqrt();
              cotangent = at::where(std.eq(0), 0., cotangent / (2. * std));
            }
            cotangent = cotangent * (2. / (eligible - correction)) *
                        detail::centered_moment_delta(values, saved);
          } else {
            if (reduction == "mean")
              cotangent = cotangent / eligible;
            cotangent = cotangent.expand(width);
          }
          if (transform == "abs")
            cotangent = cotangent * raw.sign();
          else if (transform == "square")
            cotangent = cotangent * (2. * raw);
          lookup(gx, index, 2)
              .narrow(0, i, width[0])
              .narrow(-1, d, width[2])
              .add_(cotangent.sum(-2));
          lookup(gy, index, 2)
              .narrow(0, j, width[1])
              .narrow(-1, d, width[2])
              .sub_(cotangent.sum(0));
        }
  }
  return {to_dtype(gx, x.scalar_type()), to_dtype(gy, y.scalar_type())};
}
namespace {
FeatureOutputs legacy_reduction_kernel(const Tensor& x, const Tensor& y,
                                       c10::string_view reduction,
                                       c10::string_view transform,
                                       at::IntArrayRef axes, bool keepdim,
                                       double correction, int64_t sample_size,
                                       int64_t feature_size) {
  auto [result, state] =
      reduction_kernel(x, y, reduction, transform, axes, keepdim, correction,
                       sample_size, feature_size);
  return {result, detail::legacy_moment_state(state, 3, 1)};
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_residual_reduce(Tensor x, Tensor y, str reduction, str transform, "
      "int[] axes, bool keepdim, float correction, int sample_size, "
      "int feature_size) -> (Tensor, Tensor)");
  m.def(
      "_residual_reduce.anchored(Tensor x, Tensor y, str reduction, str "
      "transform, int[] axes, bool keepdim, float correction, int sample_size, "
      "int feature_size) -> (Tensor, Tensor)");
  m.def(
      "_residual_reduce_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "state, "
      "str reduction, str transform, int[] axes, bool keepdim, float "
      "correction, "
      "int sample_size, int feature_size) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_residual_reduce", TORCH_FN(legacy_reduction_kernel));
  m.impl("_residual_reduce.anchored", TORCH_FN(reduction_kernel));
  m.impl("_residual_reduce_backward", TORCH_FN(reduction_backward_kernel));
}
}  // namespace
}  // namespace detail::residual
namespace {
Tensor residual_reduction_impl(const Tensor& x, const c10::optional<Tensor>& y,
                               c10::string_view reduction,
                               c10::string_view transform, at::IntArrayRef dims,
                               bool keepdim, c10::optional<double> correction,
                               int64_t sample_size, int64_t feature_size) {
  TORCH_CHECK(!correction || reduction == "var" || reduction == "std",
              "correction is only used by var/std");
  PairwiseOperationOptions o;
  o.chunk_size = sample_size;
  auto prepared = prepare_residual_operator(
      x, y, reduction, transform, correction.value_or(1.), feature_size, o);
  auto axes = detail::operator_axes(dims, prepared.batch.size() + 3, 3);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_residual_reduce", "anchored")
          .typed<detail::residual::ReductionSignature>();
  return std::get<0>(op.call(prepared.values[X], prepared.values[Y], reduction,
                             transform, axes, keepdim, correction.value_or(1.),
                             sample_size, feature_size));
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "residual_reduction(Tensor x, Tensor? y=None, str reduction='sum', "
      "str transform='identity', int[] dims=[], bool keepdim=False, "
      "float? correction=None, int chunk_size=128, int feature_chunk_size=128) "
      "-> Tensor",
      TORCH_FN(residual_reduction_impl));
}
}  // namespace
Tensor residual_reduce(const Tensor& x, const c10::optional<Tensor>& y,
                       c10::string_view reduction, at::IntArrayRef dims,
                       bool keepdim, c10::optional<double> correction,
                       c10::string_view transform, int64_t chunk_size,
                       int64_t feature_chunk_size) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::residual_reduction", "")
          .typed<decltype(residual_reduction_impl)>();
  return op.call(x, y, reduction, transform, dims, keepdim, correction,
                 chunk_size, feature_chunk_size);
}
}  // namespace torchpairwise::ops
