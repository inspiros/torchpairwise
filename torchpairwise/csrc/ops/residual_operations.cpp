#include "residual_operations.h"

#include <ATen/ExpandUtils.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "residual_operations_detail.h"
#include "residual_reducer.h"
#include "utils/centered_moments.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
Tensor to_double(const Tensor& value) {
  return value.scalar_type() == at::kDouble ? value : value.to(at::kDouble);
}
void validate_options(c10::string_view reduction, c10::string_view transform,
                      double correction, int64_t size) {
  TORCH_CHECK(reduction == "sum" || reduction == "mean" || reduction == "var" ||
                  reduction == "std" ||
                  detail::residual::extra_reduction(reduction),
              "Unsupported residual feature reduction");
  TORCH_CHECK(
      transform == "identity" || transform == "abs" || transform == "square",
      "Residual transform must be identity, abs or square");
  TORCH_CHECK(std::isfinite(correction), "correction must be finite");
  TORCH_CHECK(size > 0, "feature_chunk_size must be positive");
}
Tensor transform_value(const Tensor& residual, c10::string_view transform) {
  return transform == "abs"      ? residual.abs()
         : transform == "square" ? residual.square()
                                 : residual;
}
Tensor residual_tile(const Tensor& x, const Tensor& y, int64_t start,
                     int64_t width, bool paired) {
  auto a = x.narrow(-1, start, width), b = y.narrow(-1, start, width);
  return paired ? a - b : a.unsqueeze(-2) - b.unsqueeze(0);
}
void validate_inputs(const Tensor& x, const Tensor& y, bool paired) {
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided &&
                  x.dim() == 2 && y.dim() == 2,
              "Residual feature core requires dense rank-two sample tiles");
  TORCH_CHECK(
      x.device() == y.device() && x.scalar_type() == y.scalar_type() &&
          (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
      "Residual inputs require matching float32/float64 dtype/device");
  TORCH_CHECK(x.sym_size(-1) == y.sym_size(-1),
              "Residual feature counts differ");
  TORCH_CHECK(!paired || x.sym_size(0) == y.sym_size(0),
              "Paired residual counts differ");
}
}  // namespace
namespace detail::residual {
FeatureOutputs feature_kernel(const Tensor& x, const Tensor& y,
                              c10::string_view reduction,
                              c10::string_view transform, double correction,
                              int64_t size, bool paired) {
  validate_options(reduction, transform, correction, size);
  validate_inputs(x, y, paired);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto shape = std::vector<int64_t>{x.size(0)};
  if (!paired)
    shape.push_back(y.size(0));
  auto total = at::zeros(shape, x.options().dtype(at::kDouble));
  auto mean = at::zeros_like(total), m2 = at::zeros_like(total);
  auto d = x.size(-1);
  bool moments = reduction == "var" || reduction == "std";
  bool anchored = moments && x.scalar_type() == at::kDouble;
  auto origin = anchored ? at::zeros_like(total) : Tensor();
  TORCH_CHECK(d > 0 || (reduction != "min" && reduction != "max"),
              "min/max require a nonempty reduction axis");
  auto extra = extra_reduction(reduction);
  auto extra_state =
      extra ? reducer_state(shape, x.options(), reduction) : at::Tensor();
  int64_t count = 0;
  for (int64_t start = 0; start < d; start += size) {
    auto width = std::min(size, d - start);
    auto values = to_double(
        transform_value(residual_tile(x, y, start, width, paired), transform));
    if (extra) {
      auto ids = at::arange(start, start + width, x.options().dtype(at::kLong));
      reducer_update(extra_state.unsqueeze(-2), values, ids, {-1}, reduction);
    } else if (moments) {
      // Anchor each partial mean to an entry, retaining exact zero variance
      // for constant finite residuals even at a large common offset.
      auto anchor = values.select(-1, 0);
      auto safe = at::where(anchor.isfinite(), anchor, 0.);
      if (anchored && count == 0)
        origin = safe;
      auto shifted = values - (anchored ? origin : safe).unsqueeze(-1);
      auto average = shifted.mean(-1);
      if (!anchored)
        average = safe + average;
      auto partial = ((anchored ? shifted : values) - average.unsqueeze(-1))
                         .square()
                         .sum(-1);
      if (count == 0) {
        mean = average;
        m2 = partial;
      } else {
        auto delta = average - mean;
        double ratio = double(width) / double(count + width);
        mean = mean + delta * ratio;
        m2 = m2 + partial + delta.square() * (double(count) * ratio);
      }
      count += width;
    } else {
      auto update = streaming::compensated(total, m2, values.sum(-1));
      total = update.first;
      m2 = update.second;
    }
  }
  auto result = total + m2;
  auto state = at::empty({0}, total.options());
  if (extra) {
    result = reducer_result(extra_state, reduction).clone();
    state = extra_state;
  } else if (moments) {
    result =
        d == 0 ? at::full_like(total, std::numeric_limits<double>::quiet_NaN())
               : m2 / std::max(0., double(d) - correction);
    if (reduction == "std")
      result = result.sqrt();
    state = anchored ? at::stack({mean, result, origin}, -1)
                     : at::stack({mean, result}, -1);
  } else if (reduction == "mean")
    result = result / double(d);
  return {
      boolean_reduction(reduction) || result.scalar_type() == x.scalar_type()
          ? result
          : result.to(x.scalar_type()),
      state};
}
std::tuple<Tensor, Tensor> feature_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& state, c10::string_view reduction, c10::string_view transform,
    double correction, int64_t size, bool paired) {
  validate_options(reduction, transform, correction, size);
  validate_inputs(x, y, paired);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto d = x.size(-1);
  if (transform == "identity" && (reduction == "min" || reduction == "max") &&
      d <= (int64_t(1) << 53)) {
    // The saved double witness is exact in this range. A feature extremum
    // contributes at one coordinate per pair, without a D-wide mask. Paired
    // rows have no collisions and can copy directly at the public precision.
    auto ids = state.select(-1, 1).to(at::kLong);
    if (paired) {
      auto gx = at::zeros_like(x), gy = at::zeros_like(y);
      auto values = gradient.to(x.scalar_type()).unsqueeze(-1);
      ids = ids.unsqueeze(-1);
      gx.scatter_(1, ids, values);
      gy.scatter_(1, ids, values).neg_();
      return {gx, gy};
    }
    // Repeated selected features must accumulate in double before rounding
    // to the input dtype, as on the dense-mask reduction path.
    auto gx = at::zeros_like(x, x.options().dtype(at::kDouble)),
         gy = at::zeros_like(y, y.options().dtype(at::kDouble));
    auto values = to_double(gradient);
    gx.scatter_add_(1, ids, values);
    gy.scatter_add_(1, ids.transpose(0, 1), values.transpose(0, 1)).neg_();
    return {gx.to(x.scalar_type()), gy.to(y.scalar_type())};
  }
  auto gx = at::zeros_like(x), gy = at::zeros_like(y);
  auto g = to_double(gradient).unsqueeze(-1);
  bool moments = reduction == "var" || reduction == "std";
  if (!extra_reduction(reduction) && !moments && transform == "identity") {
    // A linear feature contraction's VJP is independent of residual values
    // and constant along D. Sum the sample cotangents once, then broadcast
    // into the natural input-gradient outputs; no pair/feature tile is needed.
    if (d > 0) {
      auto cotangent = reduction == "mean" ? g / double(d) : g;
      gx.copy_((paired ? cotangent : cotangent.sum(-2)).expand_as(gx));
      gy.copy_((paired ? -cotangent : -cotangent.sum(0)).expand_as(gy));
    }
    return {gx, gy};
  }
  if (reduction == "std") {
    auto output = state.select(-1, 1).unsqueeze(-1);
    g = at::where(output.eq(0), 0., g / (2. * output));
  }
  for (int64_t start = 0; start < d; start += size) {
    auto width = std::min(size, d - start);
    Tensor residual, value;
    if (moments || reduction == "prod" || transform != "identity")
      residual = residual_tile(x, y, start, width, paired);
    if (moments || reduction == "prod")
      value = to_double(transform_value(residual, transform));
    Tensor cotangent;
    if (extra_reduction(reduction)) {
      auto ids = at::arange(start, start + width, x.options().dtype(at::kLong));
      cotangent =
          reducer_gradient(state.unsqueeze(-2), value, ids, g, reduction);
    } else if (moments) {
      auto delta = state.size(-1) == 3
                       ? (value - state.select(-1, 2).unsqueeze(-1)) -
                             state.select(-1, 0).unsqueeze(-1)
                       : value - state.select(-1, 0).unsqueeze(-1);
      cotangent = g * (2. / (double(d) - correction)) * delta;
    } else {
      auto shape = paired ? std::vector<int64_t>{x.size(0), width}
                          : std::vector<int64_t>{x.size(0), y.size(0), width};
      cotangent = (reduction == "mean" ? g / double(d) : g).expand(shape);
    }
    if (transform == "abs")
      cotangent = cotangent * residual.sign();
    else if (transform == "square")
      cotangent = cotangent * (2. * residual);
    gx.narrow(-1, start, width).copy_(paired ? cotangent : cotangent.sum(-2));
    gy.narrow(-1, start, width).copy_(paired ? -cotangent : -cotangent.sum(0));
  }
  return {gx, gy};
}
Tensor feature_reduce(const Tensor& x, const Tensor& y,
                      c10::string_view reduction, c10::string_view transform,
                      double correction, int64_t size, bool paired) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_residual_features", "anchored")
          .typed<FeatureSignature>();
  return std::get<0>(
      op.call(x, y, reduction, transform, correction, size, paired));
}
namespace {
FeatureOutputs legacy_feature_kernel(const Tensor& x, const Tensor& y,
                                     c10::string_view reduction,
                                     c10::string_view transform,
                                     double correction, int64_t size,
                                     bool paired) {
  auto [result, state] =
      feature_kernel(x, y, reduction, transform, correction, size, paired);
  return {result, detail::legacy_moment_state(state, 2, 0)};
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_residual_features(Tensor x, Tensor y, str reduction, str transform, "
      "float correction, int size, bool paired) -> (Tensor, Tensor)");
  m.def(
      "_residual_features.anchored(Tensor x, Tensor y, str reduction, str "
      "transform, float correction, int size, bool paired) -> (Tensor, "
      "Tensor)");
  m.def(
      "_residual_features_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "state, "
      "str reduction, str transform, float correction, int size, bool paired) "
      "-> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_residual_features", TORCH_FN(legacy_feature_kernel));
  m.impl("_residual_features.anchored", TORCH_FN(feature_kernel));
  m.impl("_residual_features_backward", TORCH_FN(feature_backward_kernel));
}
}  // namespace
}  // namespace detail::residual
namespace detail::streaming {
PreparedComparison prepare_residual_operator(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view reduction, c10::string_view transform, double correction,
    int64_t feature_size, const PairwiseOperationOptions& o) {
  validate_options(reduction, transform, correction, feature_size);
  auto x = input, y = other.value_or(input);
  TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2 && x.layout() == at::kStrided &&
                  y.layout() == at::kStrided,
              "Residual collections require shape [...,N,D]");
  if (operator_autocast_enabled(x)) {
    if (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)
      x = x.to(at::kFloat);
    if (y.scalar_type() == at::kHalf || y.scalar_type() == at::kBFloat16)
      y = y.to(at::kFloat);
  }
  TORCH_CHECK(
      x.device() == y.device() && x.scalar_type() == y.scalar_type() &&
          (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
      "Residual inputs require matching float32/float64 dtype/device; low "
      "precision requires autocast");
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Residual sources require CPU or CUDA");
  TORCH_CHECK(x.sym_size(-1) == y.sym_size(-1),
              "Residual feature counts differ");
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 2),
                                     y.sym_sizes().slice(0, y.dim() - 2));
  auto size = o.chunk_size.value_or(128);
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  Settings s{
      std::string(reduction),
      o.transposed ? "transpose_sum" : "sum",
      false,
      false,
      false,
      false,
      false,
      2,
      1,
      size,
      int64_t(detail::residual::boolean_reduction(reduction) ? at::kBool
                                                             : x.scalar_type()),
      3,
      {correction},
      {feature_size, transform == "identity" ? 0
                     : transform == "abs"    ? 1
                                             : 2},
      {},
      std::string("_residual")};
  Tensors v(Slots, at::empty({0}, x.options()));
  v[X] = x;
  v[Y] = y;
  auto n = o.transposed ? y.sym_size(-2) : x.sym_size(-2);
  auto m = o.transposed ? x.sym_size(-2) : y.sym_size(-2);
  return {std::move(v), std::move(s), std::move(batch), n, m};
}
}  // namespace detail::streaming
ResidualOperator::ResidualOperator(Tensor x, c10::optional<Tensor> y,
                                   std::string transform, int64_t chunk_size,
                                   int64_t feature_chunk_size)
    : x_(std::move(x)),
      y_(std::move(y)),
      transform_(std::move(transform)),
      chunk_size_(chunk_size),
      feature_chunk_size_(feature_chunk_size) {
  validate_options("sum", transform_, 1., feature_chunk_size_);
  TORCH_CHECK(chunk_size_ > 0, "chunk_size must be positive");
}
ScalarExpression ResidualOperator::reduce(
    c10::string_view reduction, int64_t dim,
    c10::optional<double> correction) const {
  TORCH_CHECK(dim == -1 || dim == std::max(x_.dim(), y_ ? y_->dim() : x_.dim()),
              "ResidualOperator currently supports only feature dim=-1; reduce "
              "features before N/M");
  TORCH_CHECK(!correction || reduction == "var" || reduction == "std",
              "correction is only used by var/std");
  validate_options(reduction, transform_, correction.value_or(1.),
                   feature_chunk_size_);
  ExpressionSource source;
  source.inputs = {x_};
  if (y_)
    source.inputs.push_back(*y_);
  source.metric = std::string(reduction);
  source.family = "residual";
  source.strings.insert("transform", transform_);
  source.reals.insert("correction", correction.value_or(1.));
  source.integers.insert("feature_chunk_size", feature_chunk_size_);
  source.integers.insert("chunk_size", chunk_size_);
  return ScalarExpression(std::move(source));
}
ResidualOperator residual_operator(const Tensor& x,
                                   const c10::optional<Tensor>& y,
                                   c10::string_view transform,
                                   int64_t chunk_size,
                                   int64_t feature_chunk_size) {
  return ResidualOperator(x, y, std::string(transform), chunk_size,
                          feature_chunk_size);
}
}  // namespace torchpairwise::ops
