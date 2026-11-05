#include "media_operations.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <functional>
#include <limits>

#include "image_map_operations_detail.h"
#include "media_operations_detail.h"
#include "operator_sources_detail.h"
#include "pairwise_operations_detail.h"
#include "residual_reducer.h"
#include "utils/centered_moments.h"
#include "utils/operator_axes.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
using detail::media::Features;
using detail::media::PreparedInputs;
using namespace detail::streaming;
std::vector<std::vector<int64_t>> feature_shapes(
    int64_t kind, at::IntArrayRef settings, at::ArrayRef<double> parameters) {
  TORCH_CHECK(kind >= 0 && kind < 8, "Unknown media comparison kind");
  std::vector<std::vector<int64_t>> shapes;
  if (kind < 3) {
    TORCH_CHECK(settings.size() == 4 && parameters.size() >= 4,
                "Invalid image feature settings");
    auto c = settings[0], h = settings[1], w = settings[2],
         window = settings[3];
    TORCH_CHECK(c > 0 && h > 0 && w > 0 && window > 0,
                "Invalid image feature dimensions");
    shapes.push_back({c, h, w});
    if (kind == 0)
      return shapes;
    auto levels = kind == 2 ? int64_t(parameters.size()) - 4 : 1;
    TORCH_CHECK(levels > 0, "Multiscale weights must be nonempty");
    for (int64_t level = 0; level < levels; ++level) {
      TORCH_CHECK(h >= window && w >= window, "Image window exceeds scale");
      shapes.insert(shapes.end(), {{c, h, w},
                                   {c, h - window + 1, w - window + 1},
                                   {c, h - window + 1, w - window + 1},
                                   {c, h - window + 1, w - window + 1},
                                   {c, 1, 1}});
      h /= 2;
      w /= 2;
    }
  } else {
    TORCH_CHECK(settings.size() >= 7 && parameters.size() == 3,
                "Invalid audio feature settings");
    auto c = settings[0], t = settings[1];
    TORCH_CHECK(c > 0 && t > 0, "Invalid audio feature dimensions");
    if (kind <= 4)
      return {{c, t}};
    TORCH_CHECK(settings.size() > 7 && (settings.size() - 7) % 3 == 0,
                "Invalid STFT resolution triples");
    for (size_t k = 7; k < settings.size(); k += 3) {
      auto n = settings[k], hop = settings[k + 1], win = settings[k + 2];
      TORCH_CHECK(n > 0 && hop > 0 && win > 0 && win <= n,
                  "Invalid STFT sizes");
      auto frames = 1 + (t + (settings[4] ? 2 * (n / 2) : 0) - n) / hop;
      TORCH_CHECK(frames > 0, "No STFT frames");
      shapes.push_back({c, (n / 2 + 1) * frames});
    }
    if (kind == 7) {
      auto magnitudes = shapes;
      shapes.insert(shapes.end(), magnitudes.begin(), magnitudes.end());
    }
  }
  return shapes;
}
int64_t elements(at::IntArrayRef shape) {
  int64_t count = 1;
  for (auto d : shape)
    count *= d;
  return count;
}
Features unpack(const Tensor& value, int64_t kind, at::IntArrayRef settings,
                at::ArrayRef<double> parameters) {
  Features features;
  int64_t offset = 0;
  for (const auto& shape : feature_shapes(kind, settings, parameters)) {
    auto target = value.sizes().slice(0, value.dim() - 1).vec();
    target.insert(target.end(), shape.begin(), shape.end());
    auto count = elements(shape);
    features.push_back(value.narrow(-1, offset, count).reshape(target));
    offset += count;
  }
  TORCH_CHECK(offset == value.size(-1), "Packed media feature count mismatch");
  return features;
}
Tensor prepare_features(const Tensor& x, int64_t kind, at::IntArrayRef settings,
                        at::ArrayRef<double> parameters) {
  auto shapes = feature_shapes(kind, settings, parameters);
  auto rank = kind < 3 ? 4 : 3;
  TORCH_CHECK(x.dim() >= rank, "Insufficient media feature input rank");
  for (int64_t axis = 1; axis < rank; ++axis)
    TORCH_CHECK(x.size(axis - rank) == settings[axis - 1],
                "Media event dimensions must match feature settings");
  if (x.numel() == 0) {
    auto target = x.sizes().slice(0, x.dim() - rank + 1).vec();
    int64_t count = 0;
    for (const auto& shape : shapes)
      count += elements(shape);
    target.push_back(count);
    return at::empty(target, x.options());
  }
  auto features =
      kind < 3 ? detail::media::image_features(x, kind, settings[3], parameters)
               : detail::media::audio_features(x, kind - 3, settings.slice(2),
                                               parameters);
  for (auto& t : features)
    t = t.flatten(-(rank - 1));
  // Independent output storage is part of the opaque operator's alias contract.
  return features.size() == 1 ? features[0].clone() : at::cat(features, -1);
}
Tensor media_features_kernel(const Tensor& x, int64_t kind,
                             at::IntArrayRef settings,
                             at::ArrayRef<double> parameters) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  return prepare_features(x, kind, settings, parameters);
}
Tensor media_features_backward_kernel(const Tensor& gradient, const Tensor& x,
                                      int64_t kind, at::IntArrayRef settings,
                                      at::ArrayRef<double> parameters) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of media features are unsupported");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  if (x.numel() == 0)
    return at::zeros_like(x).contiguous();
  // Raw pixel features are only a shape-preserving copy/flatten; their VJP
  // needs no inner autograd graph and must own its output storage.
  if (kind == 0)
    return gradient.reshape_as(x).clone(at::MemoryFormat::Contiguous);
  at::AutoGradMode enable(true);
  auto raw = x.detach().set_requires_grad(true);
  auto value = prepare_features(raw, kind, settings, parameters);
  // Pixel-only preparation is a clone/reshape. Its VJP may alias gradient,
  // while this opaque operator's schema promises independent output storage.
  return torch::autograd::grad({value}, {raw}, {gradient})[0].clone(
      at::MemoryFormat::Contiguous);
}
bool map_moments(c10::string_view reduction) {
  return reduction == "var" || reduction == "std";
}
void validate_map_reduction(c10::string_view reduction, double correction) {
  TORCH_CHECK(reduction == "sum" || reduction == "mean" ||
                  map_moments(reduction) ||
                  detail::residual::extra_reduction(reduction),
              "Unsupported image-map event reduction");
  TORCH_CHECK(std::isfinite(correction), "correction must be finite");
}
Tensor pooled_map(const Tensor& a, const Tensor& b, int64_t comparison,
                  c10::string_view reduction, at::IntArrayRef settings,
                  at::ArrayRef<double> parameters, int64_t dtype,
                  double correction, bool paired) {
  validate_map_reduction(reduction, correction);
  TORCH_CHECK(comparison >= 0 && comparison <= 3 && settings.size() == 4 &&
                  parameters.size() == 4 && a.dim() == 2 && b.dim() == 2,
              "Invalid image-map event packet");
  TORCH_CHECK(a.scalar_type() == at::kDouble &&
                  b.scalar_type() == at::kDouble && a.device() == b.device() &&
                  (!paired || a.size(0) == b.size(0)),
              "Image-map core requires matching packed double feature tiles");
  TORCH_CHECK(dtype == int64_t(at::kFloat) || dtype == int64_t(at::kDouble),
              "Image-map output dtype must be float32 or float64");
  if (a.size(0) == 0 || b.size(0) == 0) {
    std::vector<int64_t> shape{a.size(0)};
    if (!paired)
      shape.push_back(b.size(0));
    return at::empty(
        shape, a.options().dtype(detail::residual::boolean_reduction(reduction)
                                     ? at::kBool
                                     : at::ScalarType(dtype)));
  }
  // Preserve the same regional contraction for legacy serialized sources and
  // the statically typed full-event C++ convenience frontend.
  if (detail::media::use_map_contract(a, b, comparison, settings, paired)) {
    int64_t code = -1;
    const char* names[] = {"sum", "mean", "var", "std", "min",
                           "max", "prod", "all", "any"};
    for (int64_t k = 0; k < 9; ++k)
      if (reduction == names[k] && k != 6)
        code = k;
    if (code >= 0)
      return detail::media::map_contract(a, b, comparison, settings, parameters,
                                         dtype, code, 7, false, correction,
                                         paired);
  }
  // The opaque map boundary replays spatial regions in backward rather than
  // retaining a complete map formula graph. Entries still round before pooling.
  auto value = detail::media::map_values_for_replay(a, b, comparison, settings,
                                                    parameters, dtype, paired)
                   .flatten(-3)
                   .to(at::kDouble);
  if ((reduction == "var" || reduction == "std") &&
      dtype == int64_t(at::kDouble))
    value = detail::centered_local_values(value);
  Tensor result;
  if (reduction == "sum")
    result = value.sum(-1);
  else if (reduction == "mean")
    result = value.mean(-1);
  else if (reduction == "var")
    result = value.var(at::IntArrayRef{-1},
                       c10::optional<at::Scalar>(correction), false);
  else if (reduction == "std")
    result = value.std(at::IntArrayRef{-1},
                       c10::optional<at::Scalar>(correction), false);
  else if (reduction == "min")
    result = std::get<0>(value.min(-1));
  else if (reduction == "max")
    result = std::get<0>(value.max(-1));
  else if (reduction == "prod")
    result = value.prod(-1);
  else if (reduction == "all")
    return value.ne(0).all(-1);
  else
    return value.ne(0).any(-1);
  return result.to(at::ScalarType(dtype));
}
Tensor image_map_reduce_kernel(const Tensor& a, const Tensor& b,
                               int64_t comparison, c10::string_view reduction,
                               at::IntArrayRef settings,
                               at::ArrayRef<double> parameters, int64_t dtype,
                               double correction, bool paired) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  return pooled_map(a, b, comparison, reduction, settings, parameters, dtype,
                    correction, paired);
}
std::tuple<Tensor, Tensor> image_map_reduce_backward_kernel(
    const Tensor& gradient, const Tensor& a, const Tensor& b,
    int64_t comparison, c10::string_view reduction, at::IntArrayRef settings,
    at::ArrayRef<double> parameters, int64_t dtype, double correction,
    bool paired) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher image-map event derivatives are unsupported");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  if (a.size(0) == 0 || b.size(0) == 0 ||
      detail::residual::boolean_reduction(reduction))
    return {at::zeros_like(a), at::zeros_like(b)};
  at::AutoGradMode enable(true);
  auto x = a.detach().set_requires_grad(true),
       y = b.detach().set_requires_grad(true);
  auto value = pooled_map(x, y, comparison, reduction, settings, parameters,
                          dtype, correction, paired);
  auto grads = torch::autograd::grad({value}, {x, y}, {gradient});
  return {grads[0].clone(at::MemoryFormat::Contiguous),
          grads[1].clone(at::MemoryFormat::Contiguous)};
}
void check_comparison_packet(const PairwiseOperationOptions& o) {
  const auto& q = o.comparison;
  TORCH_CHECK(!q.w && !q.V && !q.VI && !q.bin_positions && !q.projections &&
                  !q.x1_lengths && !q.x2_lengths && !q.p && !q.base &&
                  !q.alpha && !q.length_scale && !q.nu && !q.gamma &&
                  !q.coef0 && !q.regularization && !q.degree && !q.max_iter &&
                  !q.chunk_size && !q.feature_dim && !q.squared && !q.shuffle &&
                  !q.batched && !q.kernel && !q.generator,
              "Unused vector/cloud comparison options in media operator");
}
PreparedComparison prepare_media_source(PreparedInputs p,
                                        c10::string_view metric,
                                        c10::string_view operation,
                                        const PairwiseOperationOptions& o,
                                        int64_t size, bool autocast) {
  auto batch =
      at::infer_size_symint(p.x.sym_sizes().slice(0, p.x.dim() - p.rank),
                            p.y.sym_sizes().slice(0, p.y.dim() - p.rank));
  std::vector<int64_t> settings;
  for (int64_t axis = 1 - p.rank; axis < 0; ++axis)
    settings.push_back(p.x.sym_size(axis).guard_int(__FILE__, __LINE__));
  settings.insert(settings.end(), p.settings.begin(), p.settings.end());
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto a = detail::media::features(p.x, p.kind, settings, p.parameters);
  auto b = p.x.is_same(p.y)
               ? a
               : detail::media::features(p.y, p.kind, settings, p.parameters);
  auto empty = at::empty({0}, a.options());
  Tensors v(Slots, empty);
  v[X] = a;
  v[Y] = b;
  auto base = operation == "matmul" ? std::string("matmul")
              : operation == "reduce" && o.reduction != "mean"
                  ? o.reduction
                  : std::string("sum");
  bool search = base == "min" || base == "max";
  bool transposed =
      o.transposed != (operation == "reduce" && o.dim && *o.dim == -2);
  Integers integers{p.kind};
  for (auto value : settings)
    integers.push_back(value);
  Reals reals;
  for (auto value : p.parameters)
    reals.push_back(value);
  Settings s{std::string(metric),
             (transposed ? "transpose_" : "") +
                 (search ? std::string("neighbors") : base),
             false,
             base == "max",
             o.exclude_self,
             false,
             search,
             2,
             1,
             size,
             int64_t(p.dtype),
             3,
             std::move(reals),
             std::move(integers),
             {},
             std::string("_media")};
  auto n = v[transposed ? Y : X].sym_size(-2),
       m = v[transposed ? X : Y].sym_size(-2);
  TORCH_CHECK(!o.exclude_self || n == m,
              "exclude_self requires a square operator");
  TORCH_CHECK(!search || m - int64_t(o.exclude_self) > 0,
              "min/max require an eligible reference");
  if (operation == "matmul") {
    TORCH_CHECK(o.right, "matmul requires values");
    auto payload = *o.right;
    if (autocast && (payload.scalar_type() == at::kHalf ||
                     payload.scalar_type() == at::kBFloat16))
      payload = payload.to(at::kFloat);
    if (payload.dim() == 1)
      payload = payload.unsqueeze(-1);
    TORCH_CHECK(payload.layout() == at::kStrided && payload.dim() >= 2 &&
                    payload.sym_size(-2) == m && payload.sym_size(-1) > 0 &&
                    payload.device() == a.device() &&
                    payload.scalar_type() == p.dtype,
                "values must match comparison output dtype/device and have "
                "shape [...,M,C]");
    batch = at::infer_size_symint(
        batch, payload.sym_sizes().slice(0, payload.dim() - 2));
    v[Payload] = payload;
    s.mask |= int64_t(1) << Payload;
  }
  return {std::move(v), std::move(s), std::move(batch), n, m};
}

PairwiseOperationOptions matrix_options(
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transposed, c10::optional<int64_t> chunk_size) {
  PairwiseOperationOptions o;
  o.left = left;
  o.right = right;
  o.rows = rows;
  o.columns = columns;
  o.reduction = std::string(reduction);
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  o.transposed = transposed;
  o.chunk_size = chunk_size;
  return o;
}
Tensor image_operations_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const c10::optional<Tensor>& left,
    const c10::optional<Tensor>& right, const c10::optional<Tensor>& rows,
    const c10::optional<Tensor>& columns, c10::string_view reduction,
    c10::optional<int64_t> dim, bool keepdim, bool exclude_self,
    bool transposed, c10::optional<int64_t> chunk_size,
    c10::optional<double> data_range, int64_t window_size, double sigma,
    double k1, double k2, at::ArrayRef<double> weights) {
  auto o = matrix_options(left, right, rows, columns, reduction, dim, keepdim,
                          exclude_self, transposed, chunk_size);
  validate_pairwise_options(operation, o);
  ImageComparisonOptions options;
  options.data_range = data_range.value_or(NAN);
  options.window_size = window_size;
  options.sigma = sigma;
  options.k1 = k1;
  options.k2 = k2;
  options.weights = weights.vec();
  options.chunk_size = chunk_size.value_or(16);
  return execute_prepared(
      prepare_image_operator(x, y, metric, operation, o, options), operation, o,
      operator_autocast_enabled(x));
}
Tensor audio_operations_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const c10::optional<Tensor>& left,
    const c10::optional<Tensor>& right, const c10::optional<Tensor>& rows,
    const c10::optional<Tensor>& columns, c10::string_view reduction,
    c10::optional<int64_t> dim, bool keepdim, bool exclude_self,
    bool transposed, c10::optional<int64_t> chunk_size,
    c10::optional<bool> zero_mean, double energy_floor, int64_t n_fft,
    c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length,
    c10::string_view window, bool center, c10::string_view pad_mode,
    bool normalized, double log_floor, double norm_floor,
    at::IntArrayRef resolutions) {
  auto o = matrix_options(left, right, rows, columns, reduction, dim, keepdim,
                          exclude_self, transposed, chunk_size);
  validate_pairwise_options(operation, o);
  AudioComparisonOptions options;
  options.zero_mean = zero_mean;
  options.energy_floor = energy_floor;
  options.n_fft = n_fft;
  options.hop_length = hop_length;
  options.win_length = win_length;
  options.window = std::string(window);
  options.center = center;
  options.pad_mode = std::string(pad_mode);
  options.normalized = normalized;
  options.log_floor = log_floor;
  options.norm_floor = norm_floor;
  options.resolutions = resolutions.vec();
  options.chunk_size = chunk_size.value_or(16);
  return execute_prepared(
      prepare_audio_operator(x, y, metric, operation, o, options), operation, o,
      operator_autocast_enabled(x));
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_image_map_reduce(Tensor x, Tensor y, int comparison, str reduction, "
      "int[] settings, float[] parameters, int dtype, float correction, bool "
      "paired) -> Tensor");
  m.def(
      "_image_map_reduce_backward(Tensor gradient, Tensor x, Tensor y, int "
      "comparison, "
      "str reduction, int[] settings, float[] parameters, int dtype, float "
      "correction, "
      "bool paired) -> (Tensor, Tensor)");

  m.def(
      "_media_features(Tensor x, int kind, int[] settings, float[] parameters) "
      "-> Tensor");
  m.def(
      "_media_features_backward(Tensor gradient, Tensor x, int kind, int[] "
      "settings, float[] parameters) -> Tensor");
  m.def(
      "image_pairwise_operations(Tensor x1, Tensor? x2=None, str "
      "metric='ssim', str operation='reduce', Tensor? left=None, Tensor? "
      "right=None, Tensor? rows=None, Tensor? columns=None, str "
      "reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transposed=False, int? chunk_size=None, float? "
      "data_range=None, int window_size=11, float sigma=1.5, float k1=0.01, "
      "float k2=0.03, float[] weights=[0.0448,0.2856,0.3001,0.2363,0.1333]) -> "
      "Tensor",
      TORCH_FN(image_operations_impl));
  m.def(
      "audio_pairwise_operations(Tensor x1, Tensor? x2=None, str "
      "metric='si_sdr', str operation='reduce', Tensor? left=None, Tensor? "
      "right=None, Tensor? rows=None, Tensor? columns=None, str "
      "reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transposed=False, int? chunk_size=None, bool? "
      "zero_mean=None, float energy_floor=1e-8, int n_fft=512, int? "
      "hop_length=None, int? win_length=None, str window='hann', bool "
      "center=True, str pad_mode='constant', bool normalized=False, float "
      "log_floor=1e-7, float norm_floor=1e-7, int[] "
      "resolutions=[512,128,512,1024,256,1024,2048,512,2048]) -> Tensor",
      TORCH_FN(audio_operations_impl));
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_image_map_reduce", TORCH_FN(image_map_reduce_kernel));
  m.impl("_image_map_reduce_backward",
         TORCH_FN(image_map_reduce_backward_kernel));
  m.impl("_media_features", TORCH_FN(media_features_kernel));
  m.impl("_media_features_backward", TORCH_FN(media_features_backward_kernel));
}
}  // namespace
namespace detail::media {
namespace {
Tensor pixel_map_region(const Tensor& x, const Tensor& y, int64_t comparison,
                        int64_t dtype, bool promote = false) {
#ifdef WITH_CUDA
  if (x.is_cuda())
    return pixel_map_region_cuda(x, y, comparison, dtype, promote);
#endif
  // x/y are packed double features. Public rounding precedes promotion even
  // when the consumer accumulates in double. Only fresh map storage is reused.
  auto value = y - x;
  if (comparison == 1)
    value.abs_();
  else if (comparison == 2)
    value.square_();
  value = value.to(at::ScalarType(dtype));
  return promote ? value.to(at::kDouble) : value;
}
Tensor map_feature_region(const Tensor& feature, int64_t index, bool ssim,
                          int64_t halo, int64_t channel, int64_t channels,
                          int64_t row, int64_t rows, int64_t column,
                          int64_t columns) {
  auto value = feature.narrow(-3, channel, channels);
  if (ssim && index == 5)
    return value;  // Spatially constant per-channel scale.
  auto extra = ssim && index < 2 ? halo : 0;
  return value.narrow(-2, row, rows + extra)
      .narrow(-1, column, columns + extra);
}
}  // namespace
Tensor map_values_composite(const Tensor& a, const Tensor& b,
                            int64_t comparison, at::IntArrayRef settings,
                            at::ArrayRef<double> parameters, int64_t dtype,
                            bool paired) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  TORCH_CHECK(comparison >= 0 && comparison <= 3 && settings.size() == 4 &&
                  parameters.size() == 4 && a.dim() == 2 && b.dim() == 2 &&
                  a.scalar_type() == at::kDouble &&
                  b.scalar_type() == at::kDouble && a.device() == b.device() &&
                  (!paired || a.size(0) == b.size(0)),
              "Invalid image event feature packet");
  TORCH_CHECK(dtype == int64_t(at::kFloat) || dtype == int64_t(at::kDouble),
              "Image-map output dtype must be float32 or float64");
  auto kind = comparison == 3 ? 1 : 0;
  auto left = unpack(paired ? a : a.unsqueeze(-2), kind, settings, parameters);
  auto right = unpack(paired ? b : b.unsqueeze(-3), kind, settings, parameters);
  return image_map(left, right, comparison, settings[3], parameters)
      .to(at::ScalarType(dtype));
}
namespace {
bool large_map(const Tensor& a, const Tensor& b, int64_t comparison,
               at::IntArrayRef settings, bool paired) {
  // 4M public map entries. Division avoids overflowing an event-size product.
  if (!a.size(0) || !b.size(0))
    return false;
  int64_t remaining = int64_t(1) << 22;
  auto halo = comparison == 3 ? settings[3] - 1 : 0;
  for (auto size : {a.size(0), paired ? int64_t(1) : b.size(0), settings[0],
                    settings[1] - halo, settings[2] - halo}) {
    TORCH_CHECK(size > 0, "Image map event dimensions must be positive");
    if (size > remaining)
      return true;
    remaining /= size;
  }
  return false;
}
}  // namespace
Tensor map_values_kernel(const Tensor& a, const Tensor& b, int64_t comparison,
                         at::IntArrayRef settings,
                         at::ArrayRef<double> parameters, int64_t dtype,
                         bool paired) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  TORCH_CHECK(
      settings.size() == 4 && parameters.size() == 4 && comparison >= 0 &&
          comparison <= 3 && a.dim() == 2 && b.dim() == 2 &&
          a.scalar_type() == at::kDouble && b.scalar_type() == at::kDouble &&
          a.device() == b.device() && (!paired || a.size(0) == b.size(0)) &&
          (dtype == int64_t(at::kFloat) || dtype == int64_t(at::kDouble)),
      "Invalid image map packet");
  if (!large_map(a, b, comparison, settings, paired))
    return map_values_composite(a, b, comparison, settings, parameters, dtype,
                                paired);
  auto halo = comparison == 3 ? settings[3] - 1 : 0;
  auto shape = paired ? std::vector<int64_t>{a.size(0)}
                      : std::vector<int64_t>{a.size(0), b.size(0)};
  shape.insert(shape.end(),
               {settings[0], settings[1] - halo, settings[2] - halo});
  auto result = at::empty(shape, a.options().dtype(at::ScalarType(dtype)));
  visit_map_regions(
      a, b, comparison, settings, parameters, dtype, paired,
      [&](const Tensor& value, int64_t c, int64_t nc, int64_t h, int64_t nh,
          int64_t w, int64_t nw) {
        result.narrow(-3, c, nc).narrow(-2, h, nh).narrow(-1, w, nw).copy_(
            value);
      });
  return result;
}
void visit_map_regions(const Tensor& a, const Tensor& b, int64_t comparison,
                       at::IntArrayRef settings,
                       at::ArrayRef<double> parameters, int64_t dtype,
                       bool paired, const MapRegionConsumer& consume,
                       bool promote) {
  at::NoGradGuard guard;
  if (!a.size(0) || !b.size(0))
    return;
  bool ssim = comparison == 3;
  auto halo = ssim ? settings[3] - 1 : 0;
  auto left =
      unpack(paired ? a : a.unsqueeze(-2), ssim ? 1 : 0, settings, parameters);
  auto right =
      unpack(paired ? b : b.unsqueeze(-3), ssim ? 1 : 0, settings, parameters);
  auto identical = ssim ? left[0].eq(right[0]).all(-1).all(-1).all(-1)
                        : at::empty({0}, a.options().dtype(at::kBool));
  auto height = settings[1] - halo, width = settings[2] - halo;
  auto pairs = paired ? a.size(0) : a.size(0) * b.size(0);
  auto channels = std::min<int64_t>(settings[0], 32);
  auto budget = std::max<int64_t>(1, (int64_t(1) << 23) / pairs);
  auto columns =
      std::min<int64_t>({width, 256, std::max<int64_t>(1, budget / channels)});
  auto rows = std::min<int64_t>(
      {height, 64, std::max<int64_t>(1, budget / channels / columns)});
  for (int64_t c = 0; c < settings[0]; c += channels)
    for (int64_t h = 0; h < height; h += rows)
      for (int64_t w = 0; w < width; w += columns) {
        auto nc = std::min(channels, settings[0] - c),
             nh = std::min(rows, height - h), nw = std::min(columns, width - w);
        Features x, y;
        for (size_t k = 0; k < left.size(); ++k) {
          x.push_back(
              map_feature_region(left[k], k, ssim, halo, c, nc, h, nh, w, nw));
          y.push_back(
              map_feature_region(right[k], k, ssim, halo, c, nc, h, nh, w, nw));
        }
        auto value =
            ssim ? image_map_with_identity(x, y, comparison, settings[3],
                                           parameters, identical)
                       .to(at::ScalarType(dtype))
                 : pixel_map_region(x[0], y[0], comparison, dtype, promote);
        if (ssim && promote)
          value = value.to(at::kDouble);
        consume(value, c, nc, h, nh, w, nw);
      }
}
std::tuple<Tensor, Tensor> map_backward_regions(
    const MapRegionCotangent& cotangent_for, bool needs_value, const Tensor& a,
    const Tensor& b, int64_t comparison, at::IntArrayRef settings,
    at::ArrayRef<double> parameters, int64_t dtype, bool paired,
    int64_t element_budget) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of image maps are unsupported");
  TORCH_CHECK(settings.size() == 4 && parameters.size() == 4 &&
                  comparison >= 0 && comparison <= 3,
              "Invalid image map VJP packet");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto ga = at::zeros_like(a), gb = at::zeros_like(b);
  if (a.size(0) == 0 || b.size(0) == 0)
    return {ga, gb};
  bool ssim = comparison == 3;
  auto halo = ssim ? settings[3] - 1 : 0;
  auto left =
      unpack(paired ? a : a.unsqueeze(-2), ssim ? 1 : 0, settings, parameters);
  auto right =
      unpack(paired ? b : b.unsqueeze(-3), ssim ? 1 : 0, settings, parameters);
  auto dleft = unpack(paired ? ga : ga.unsqueeze(-2), ssim ? 1 : 0, settings,
                      parameters);
  auto dright = unpack(paired ? gb : gb.unsqueeze(-3), ssim ? 1 : 0, settings,
                       parameters);
  // Identity belongs to the complete pair, not to a locally identical patch.
  auto identical = ssim ? left[0].eq(right[0]).all(-1).all(-1).all(-1)
                        : at::empty({0}, a.options().dtype(at::kBool));
  auto height = settings[1] - halo, width = settings[2] - halo;
  // Bound each replay graph, including sample-pair and channel multiplicity.
  // This target is not a total memory budget: packed features, the outer map,
  // gradients, convolution workspace and consumer state still occupy memory.
  auto pairs = paired ? a.size(0) : a.size(0) * b.size(0);
  auto channels = std::min<int64_t>(settings[0], 32);
  auto per_pair = std::max<int64_t>(1, element_budget / pairs);
  auto columns = std::min<int64_t>(
      {width, 256, std::max<int64_t>(1, per_pair / channels)});
  auto rows = std::min<int64_t>(
      {height, 64, std::max<int64_t>(1, per_pair / channels / columns)});
  for (int64_t c = 0; c < settings[0]; c += channels)
    for (int64_t h = 0; h < height; h += rows)
      for (int64_t w = 0; w < width; w += columns) {
        auto nc = std::min(channels, settings[0] - c),
             nh = std::min(rows, height - h), nw = std::min(columns, width - w);
        if (!ssim) {
          auto x =
              map_feature_region(left[0], 0, false, 0, c, nc, h, nh, w, nw);
          auto y =
              map_feature_region(right[0], 0, false, 0, c, nc, h, nh, w, nw);
          Tensor value;
          if (needs_value)
            value = pixel_map_region(x, y, comparison, dtype);
          auto cotangent =
              cotangent_for(value, c, nc, h, nh, w, nw).to(at::kDouble);
          if (comparison == 1 || comparison == 2) {
            // The difference is fresh storage. Reuse it for the local chain
            // rule; cotangent can alias caller-owned gradients or saved state.
            auto derivative = y - x;
            if (comparison == 1)
              derivative.sign_();
            else
              derivative.mul_(2.);
            // Keep every chain-rule entry, including unselected 0*Inf/NaN.
            cotangent = derivative.mul_(cotangent);
          }
          auto dx =
              map_feature_region(dleft[0], 0, false, 0, c, nc, h, nh, w, nw);
          auto dy =
              map_feature_region(dright[0], 0, false, 0, c, nc, h, nh, w, nw);
          dx.sub_(paired ? cotangent : cotangent.sum(-4, true));
          dy.add_(paired ? cotangent : cotangent.sum(-5, true));
          continue;
        }
        Features local_left, local_right;
        std::vector<Tensor> leaves, targets;
        at::AutoGradMode enable(true);
        for (int operand = 0; operand < 2; ++operand) {
          auto& source = operand == 0 ? left : right;
          auto& gradients = operand == 0 ? dleft : dright;
          auto& local = operand == 0 ? local_left : local_right;
          for (size_t k = 0; k < source.size(); ++k) {
            auto value = map_feature_region(source[k], k, ssim, halo, c, nc, h,
                                            nh, w, nw)
                             .detach();
            // Raw SSIM pixels only determine the nondifferentiable identity
            // predicate; all differentiable paths use prepared feature fields.
            bool differentiable = !ssim || k != 0;
            local.push_back(value.set_requires_grad(differentiable));
            if (differentiable) {
              leaves.push_back(local.back());
              targets.push_back(map_feature_region(gradients[k], k, ssim, halo,
                                                   c, nc, h, nh, w, nw));
            }
          }
        }
        auto value =
            image_map_with_identity(local_left, local_right, comparison,
                                    settings[3], parameters, identical)
                .to(at::ScalarType(dtype));
        Tensor upstream;
        {
          at::NoGradGuard no_grad;
          upstream = cotangent_for(value.detach(), c, nc, h, nh, w, nw);
        }
        auto gradients = torch::autograd::grad({value}, leaves, {upstream},
                                               false, false, true);
        at::NoGradGuard no_grad;
        for (size_t k = 0; k < gradients.size(); ++k)
          if (gradients[k].defined())
            targets[k].add_(gradients[k]);
      }
  return {ga, gb};
}
std::tuple<Tensor, Tensor> identity_linear_backward(
    const Tensor& gradient, const Tensor& a, const Tensor& b,
    at::IntArrayRef settings, int64_t dtype, double divisor, bool paired) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of image maps are unsupported");
  at::NoGradGuard guard;
  auto ga = at::zeros_like(a), gb = at::zeros_like(b);
  if (!a.size(0) || !b.size(0))
    return {ga, gb};
  auto left_shape =
      std::vector<int64_t>{a.size(0), settings[0], settings[1], settings[2]};
  auto right_shape = left_shape;
  right_shape[0] = b.size(0);
  auto dx = ga.view(left_shape), dy = gb.view(right_shape);
  // Keep contracted image axes singleton through conversion and sample sums.
  // Expanding before a dtype conversion would allocate a pairwise pixel map.
  auto rank = gradient.dim();
  auto pairs = paired ? a.size(0) : a.size(0) * b.size(0);
  auto compact_channels = std::min<int64_t>(gradient.size(-3), 32);
  auto channels = gradient.size(-3) == 1 ? settings[0] : compact_channels;
  auto budget = std::max<int64_t>(1, (int64_t(1) << 20) / pairs);
  auto columns = gradient.size(-1) == 1
                     ? settings[2]
                     : std::min<int64_t>(
                           {settings[2], 256,
                            std::max<int64_t>(1, budget / compact_channels)});
  auto compact_columns = gradient.size(-1) == 1 ? 1 : columns;
  auto rows =
      gradient.size(-2) == 1
          ? settings[1]
          : std::min<int64_t>({settings[1], 64,
                               std::max<int64_t>(1, budget / compact_channels /
                                                        compact_columns)});
  // Contracted axes need no replay loop: expanding a view into the existing
  // input gradient allocates no pairwise temporary. Tile only retained axes.
  for (int64_t c = 0; c < settings[0]; c += channels)
    for (int64_t h = 0; h < settings[1]; h += rows)
      for (int64_t w = 0; w < settings[2]; w += columns) {
        auto nc = std::min(channels, settings[0] - c),
             nh = std::min(rows, settings[1] - h),
             nw = std::min(columns, settings[2] - w);
        auto cotangent = gradient;
        std::array<int64_t, 3> offsets{c, h, w}, lengths{nc, nh, nw};
        for (int64_t d = 0; d < 3; ++d) {
          auto axis = rank - 3 + d;
          bool reduced = gradient.size(axis) == 1;
          cotangent = cotangent.narrow(axis, reduced ? 0 : offsets[d],
                                       reduced ? 1 : lengths[d]);
        }
        if (divisor != 1.)
          cotangent = cotangent / divisor;
        // Division takes place in double; the map cast rounds each incoming
        // entry before any sample accumulation, even when public dtype is
        // float.
        cotangent = cotangent.to(at::ScalarType(dtype)).to(at::kDouble);
        auto target_x = dx.narrow(1, c, nc).narrow(2, h, nh).narrow(3, w, nw);
        auto target_y = dy.narrow(1, c, nc).narrow(2, h, nh).narrow(3, w, nw);
        if (paired) {
          target_x.sub_(cotangent.expand_as(target_x));
          target_y.add_(cotangent.expand_as(target_y));
        } else {
          auto shape = cotangent.sizes().vec();
          shape[0] = a.size(0);
          shape[1] = b.size(0);
          cotangent = cotangent.expand(shape);
          target_x.sub_(cotangent.sum(1).expand_as(target_x));
          target_y.add_(cotangent.sum(0).expand_as(target_y));
        }
      }
  return {ga, gb};
}
bool identity_extrema_backward(Tensor dx, Tensor dy, const Tensor& gradient,
                               const Tensor& witnesses, int64_t dtype,
                               bool sample_ids, bool paired,
                               int64_t sample_size) {
  constexpr int64_t exact_limit = int64_t(1) << 53;
  auto features = dx.size(1), n = dx.size(0), m = dy.size(0);
  if (features > exact_limit ||
      (sample_ids && m &&
       (n > exact_limit / m || (features && n * m > exact_limit / features))))
    return false;
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of image maps are unsupported");
  at::NoGradGuard guard;
  if (!witnesses.numel())
    return true;
  // Block the output, not the unreduced pair map. Retained C/H/W axes can
  // still be large. Bound index/cotangent temporaries as well as sample tiles.
  int64_t rank = witnesses.dim(), sample_rank = paired ? 1 : 2;
  std::vector<int64_t> widths(rank), start(rank, 0);
  int64_t budget = int64_t(1) << 18;
  for (int64_t d = rank - 1; d >= 0; --d) {
    auto cap = d < sample_rank
                   ? sample_size
                   : (d == rank - 1 ? 256 : (d == rank - 2 ? 64 : 32));
    widths[d] = std::min({witnesses.size(d), cap, budget});
    budget = std::max<int64_t>(1, budget / widths[d]);
  }
  for (;;) {
    auto ids = witnesses, cotangent = gradient;
    std::vector<int64_t> shape(rank);
    for (int64_t d = 0; d < rank; ++d) {
      shape[d] = std::min(widths[d], witnesses.size(d) - start[d]);
      ids = ids.narrow(d, start[d], shape[d]);
      cotangent = cotangent.narrow(d, start[d], shape[d]);
    }
    ids = ids.reshape({-1}).to(at::kLong);
    // A cast at the public map boundary rounds the incoming cotangent too.
    cotangent =
        cotangent.reshape({-1}).to(at::ScalarType(dtype)).to(at::kDouble);
    auto feature = ids.remainder(features);
    Tensor rows, columns;
    if (sample_ids) {
      rows = at::floor_divide(ids, m * features);
      columns = at::floor_divide(ids, features).remainder(m);
    } else {
      std::vector<int64_t> row_shape(rank, 1), column_shape(rank, 1);
      row_shape[0] = shape[0];
      rows = at::arange(start[0], start[0] + shape[0], ids.options())
                 .reshape(row_shape)
                 .expand(shape)
                 .reshape({-1});
      if (paired)
        columns = rows;
      else {
        column_shape[1] = shape[1];
        columns = at::arange(start[1], start[1] + shape[1], ids.options())
                      .reshape(column_shape)
                      .expand(shape)
                      .reshape({-1});
      }
    }
    // Repeated pixels accumulate in packed double precision, before the
    // feature preparation VJP returns to the user's input dtype.
    dx.index_put_({rows, feature}, -cotangent, true);
    dy.index_put_({columns, feature}, cotangent, true);
    int64_t d = rank - 1;
    for (; d >= 0; --d) {
      start[d] += widths[d];
      if (start[d] < witnesses.size(d))
        break;
      start[d] = 0;
    }
    if (d < 0)
      break;
  }
  return true;
}
std::tuple<Tensor, Tensor> map_values_backward_kernel(
    const Tensor& gradient, const Tensor& a, const Tensor& b,
    int64_t comparison, at::IntArrayRef settings,
    at::ArrayRef<double> parameters, int64_t dtype, bool paired) {
  auto cotangent = [&](const Tensor&, int64_t c, int64_t nc, int64_t h,
                       int64_t nh, int64_t w, int64_t nw) {
    return gradient.narrow(-3, c, nc).narrow(-2, h, nh).narrow(-1, w, nw);
  };
  return map_backward_regions(cotangent, false, a, b, comparison, settings,
                              parameters, dtype, paired);
}
namespace {
const char* contract_name(int64_t code) {
  static const char* names[] = {"sum", "mean", "var", "std", "min",
                                "max", "prod", "all", "any"};
  return names[code];
}
// Full event row-major indices preserve first-witness selection even when
// rectangular regions are visited in a different order. Retained axes have
// separate states, so their contribution to an index never changes a tie.
Tensor contract_ids(const Tensor& a, at::IntArrayRef settings,
                    int64_t comparison, bool paired, int64_t c, int64_t nc,
                    int64_t h, int64_t nh, int64_t w, int64_t nw) {
  auto halo = comparison == 3 ? settings[3] - 1 : 0;
  auto height = settings[1] - halo, width = settings[2] - halo;
  auto rank = paired ? 4 : 5;
  std::vector<int64_t> cs(rank, 1), hs(rank, 1), ws(rank, 1);
  cs[rank - 3] = nc;
  hs[rank - 2] = nh;
  ws[rank - 1] = nw;
  auto options = a.options().dtype(at::kLong);
  return (at::arange(c, c + nc, options).reshape(cs) * height +
          at::arange(h, h + nh, options).reshape(hs)) *
             width +
         at::arange(w, w + nw, options).reshape(ws);
}
// Use indexed min/max rather than amin/amax: the latter can choose the sign
// of a different tied zero on CUDA. Values and VJPs must share one witness.
void contract_extrema_update(Tensor state, const Tensor& value,
                             const Tensor& ids, at::IntArrayRef axes,
                             int64_t code) {
  std::vector<int64_t> order, retained, shape;
  int64_t count = 1;
  for (int64_t d = 0; d < value.dim(); ++d) {
    bool selected = std::find(axes.begin(), axes.end(), d) != axes.end();
    if (!selected) {
      order.push_back(d);
      retained.push_back(value.size(d));
    } else
      count *= value.size(d);
    shape.push_back(selected ? 1 : value.size(d));
  }
  order.insert(order.end(), axes.begin(), axes.end());
  retained.push_back(count);
  auto flat = value.permute(order).reshape(retained);
  auto [best, index] = code == 4 ? flat.min(-1, true) : flat.max(-1, true);
  best = best.reshape(shape);
  auto witness = ids.expand_as(value)
                     .permute(order)
                     .reshape(retained)
                     .gather(-1, index)
                     .reshape(shape)
                     .to(at::kDouble);
  auto old = state.select(-1, 0), old_id = state.select(-1, 1);
  auto better = best.isnan() & ~old.isnan();
  better |=
      ~best.isnan() & ~old.isnan() & (code == 4 ? best.lt(old) : best.gt(old));
  better |= (best.eq(old) | (best.isnan() & old.isnan())) & witness.lt(old_id);
  old.copy_(at::where(better, best, old));
  old_id.copy_(at::where(better, witness, old_id));
}
// Keep all three event axes internally. Region slices of a reduced axis share
// the same state; retained axes address disjoint output slices.
Tensor contract_region(Tensor value, int64_t mask, int64_t c, int64_t nc,
                       int64_t h, int64_t nh, int64_t w, int64_t nw) {
  return value.narrow(-3, mask & 1 ? 0 : c, mask & 1 ? 1 : nc)
      .narrow(-2, mask & 2 ? 0 : h, mask & 2 ? 1 : nh)
      .narrow(-1, mask & 4 ? 0 : w, mask & 4 ? 1 : nw);
}
std::vector<int64_t> contract_shape(const Tensor& a, const Tensor& b,
                                    int64_t comparison,
                                    at::IntArrayRef settings, int64_t mask,
                                    bool paired, bool keepdim) {
  auto shape = paired ? std::vector<int64_t>{a.size(0)}
                      : std::vector<int64_t>{a.size(0), b.size(0)};
  auto halo = comparison == 3 ? settings[3] - 1 : 0;
  std::vector<int64_t> event{settings[0], settings[1] - halo,
                             settings[2] - halo};
  for (int64_t k = 0; k < 3; ++k)
    if (keepdim || !(mask & (1 << k)))
      shape.push_back(mask & (1 << k) ? 1 : event[k]);
  return shape;
}
void validate_contract(const Tensor& a, const Tensor& b, int64_t comparison,
                       at::IntArrayRef settings,
                       at::ArrayRef<double> parameters, int64_t dtype,
                       int64_t code, int64_t mask, double correction,
                       bool paired) {
  TORCH_CHECK(
      comparison >= 0 && comparison <= 3 && settings.size() == 4 &&
          parameters.size() == 4 && a.dim() == 2 && b.dim() == 2 &&
          a.scalar_type() == at::kDouble && b.scalar_type() == at::kDouble &&
          a.device() == b.device() && (!paired || a.size(0) == b.size(0)) &&
          (dtype == int64_t(at::kFloat) || dtype == int64_t(at::kDouble)) &&
          code >= 0 && code <= 8 && mask > 0 && mask < 8 &&
          std::isfinite(correction),
      "Invalid regional image contraction packet");
  auto halo = comparison == 3 ? settings[3] - 1 : 0;
  TORCH_CHECK(settings[0] > 0 && settings[1] > halo && settings[2] > halo &&
                  settings[3] > 0,
              "Image event dimensions must be positive");
}
// First-event products retain ATen's joint prod backward convention. This
// differs from the exceptional-factor convention used by terminal reducers:
// without any input zero, ATen uses result/input, including inf/inf. If any
// row contains a zero, every row uses an exclusive product instead. Keep the
// raw product separately so finite overflow/underflow follows that quotient
// branch rather than removing exceptional factors unconditionally.
Tensor contract_product(at::IntArrayRef shape, at::TensorOptions options) {
  auto sizes = shape.vec();
  sizes.push_back(7);
  // Raw product, zero count, finite nonzero product, negative-sign count,
  // infinity count, NaN count, log magnitude of finite nonzero factors. State
  // is proportional to the contracted output.
  auto state = at::zeros(sizes, options);
  state.select(-1, 0).fill_(1.);
  state.select(-1, 2).fill_(1.);
  return state;
}
Tensor contract_joint_product(Tensor value, at::IntArrayRef axes) {
  std::vector<int64_t> order, shape, output;
  int64_t count = 1;
  for (int64_t d = 0; d < value.dim(); ++d) {
    bool reduced = std::find(axes.begin(), axes.end(), d) != axes.end();
    output.push_back(reduced ? 1 : value.size(d));
    if (reduced)
      count *= value.size(d);
    else {
      order.push_back(d);
      shape.push_back(value.size(d));
    }
  }
  order.insert(order.end(), axes.begin(), axes.end());
  shape.push_back(count);
  return value.permute(order).reshape(shape).prod(-1).reshape(output);
}
void contract_product_update(Tensor state, const Tensor& entries,
                             at::IntArrayRef axes) {
#ifdef WITH_CUDA
  if (entries.is_cuda()) {
    int64_t factors = 1;
    for (auto axis : axes)
      factors *= entries.size(axis);
    // Very short rows with many retained pixels favor ATen's existing
    // vectorized reductions over launching one cooperative block per row.
    if (factors >= 64) {
      image_product_update_cuda(state, entries, axes);
      return;
    }
  }
#endif
  auto value = entries.to(at::kDouble);
  auto zero = value.eq(0), inf = value.isinf(), nan = value.isnan();
  state.select(-1, 0).mul_(contract_joint_product(value, axes));
  state.select(-1, 1).add_(zero.sum(axes, true));
  state.select(-1, 2).mul_(
      contract_joint_product(at::where(zero | inf | nan, 1., value), axes));
  state.select(-1, 3).add_(value.signbit().sum(axes, true));
  state.select(-1, 4).add_(inf.sum(axes, true));
  state.select(-1, 5).add_(nan.sum(axes, true));
  state.select(-1, 6).add_(
      at::where(zero | inf | nan, 1., value.abs()).log().sum(axes, true));
}
Tensor contract_product_gradient(const Tensor& state, const Tensor& entries,
                                 const Tensor& gradient,
                                 const Tensor& any_zero) {
#ifdef WITH_CUDA
  if (entries.is_cuda())
    return image_product_gradient_cuda(state, entries, gradient, any_zero);
#endif
  auto value = entries.to(at::kDouble);
  auto quotient = state.select(-1, 0) / value;
  auto zero = value.eq(0), inf = value.isinf(), nan = value.isnan();
  auto zeros = state.select(-1, 1) - zero.to(at::kDouble);
  auto finite = state.select(-1, 2) / at::where(zero | inf | nan, 1., value);
  auto negatives = state.select(-1, 3) - value.signbit().to(at::kDouble);
  auto infs = state.select(-1, 4) - inf.to(at::kDouble);
  auto nans = state.select(-1, 5) - nan.to(at::kDouble);
  auto negative = negatives.remainder(2).ne(0);
  // Removing one factor can recover a finite exclusive product even when the
  // total finite product overflowed/underflowed. Retain a log magnitude for
  // this rare fallback; ordinary finite products still use division.
  auto log_magnitude =
      state.select(-1, 6) - at::where(zero | inf | nan, 1., value.abs()).log();
  auto magnitude = log_magnitude.exp();
  auto recovered = at::where(negative, -magnitude, magnitude);
  auto total_finite = state.select(-1, 2);
  finite = at::where(total_finite.isfinite() & total_finite.ne(0), finite,
                     recovered);
  auto infinity = at::where(negative, -std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity());
  auto signed_zero =
      at::where(negative, -at::zeros_like(finite), at::zeros_like(finite));
  auto exclusive = at::where(infs.gt(0), infinity, finite);
  exclusive = at::where(zeros.gt(0), signed_zero, exclusive);
  exclusive = at::where(nans.gt(0) | (zeros.gt(0) & infs.gt(0)),
                        std::numeric_limits<double>::quiet_NaN(), exclusive);
  // The predicate stays on device: no CUDA synchronization or data-dependent
  // Python/host branch. The whole local tile shares ATen's zero branch.
  return gradient * at::where(any_zero, exclusive, quotient);
}
EventOutputs map_contract_kernel(const Tensor& a, const Tensor& b,
                                 int64_t comparison, at::IntArrayRef settings,
                                 at::ArrayRef<double> parameters, int64_t dtype,
                                 int64_t code, int64_t mask, bool keepdim,
                                 double correction, bool paired) {
  validate_contract(a, b, comparison, settings, parameters, dtype, code, mask,
                    correction, paired);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto shape = contract_shape(a, b, comparison, settings, mask, paired, true);
  auto state_shape = shape;
  bool anchored = (code == 2 || code == 3) && dtype == int64_t(at::kDouble);
  state_shape.push_back(anchored ? 4 : 3);
  // Sum: compensated total/error. Moments: count/mean/centered square sum.
  // Extrema/truth use their shared first-witness/boolean merge conventions.
  auto state = code == 6   ? contract_product(shape, a.options())
               : code >= 4 ? detail::residual::reducer_state(
                                 shape, a.options(), contract_name(code))
                           : at::zeros(state_shape, a.options());
  bool ssim = comparison == 3;
  auto halo = ssim ? settings[3] - 1 : 0;
  auto left =
      unpack(paired ? a : a.unsqueeze(-2), ssim ? 1 : 0, settings, parameters);
  auto right =
      unpack(paired ? b : b.unsqueeze(-3), ssim ? 1 : 0, settings, parameters);
  if (a.size(0) && b.size(0)) {
    auto identical = ssim ? left[0].eq(right[0]).all(-1).all(-1).all(-1)
                          : at::empty({0}, a.options().dtype(at::kBool));
    auto height = settings[1] - halo, width = settings[2] - halo;
    auto pairs = paired ? a.size(0) : a.size(0) * b.size(0);
    auto channels = std::min<int64_t>(settings[0], 32);
    // Product adjoints carry several exceptional-factor masks and counts.
    auto target = int64_t(1) << (code == 6 ? 20 : 23);
    auto budget = std::max<int64_t>(1, target / pairs);
    auto columns = std::min<int64_t>(
        {width, 256, std::max<int64_t>(1, budget / channels)});
    auto rows = std::min<int64_t>(
        {height, 64, std::max<int64_t>(1, budget / channels / columns)});
    std::vector<int64_t> axes;
    for (int64_t k = 0; k < 3; ++k)
      if (mask & (1 << k))
        axes.push_back(int64_t(shape.size()) - 3 + k);
    for (int64_t c = 0; c < settings[0]; c += channels)
      for (int64_t h = 0; h < height; h += rows)
        for (int64_t w = 0; w < width; w += columns) {
          auto nc = std::min(channels, settings[0] - c),
               nh = std::min(rows, height - h),
               nw = std::min(columns, width - w);
          Features x, y;
          for (size_t k = 0; k < left.size(); ++k) {
            x.push_back(map_feature_region(left[k], k, ssim, halo, c, nc, h, nh,
                                           w, nw));
            y.push_back(map_feature_region(right[k], k, ssim, halo, c, nc, h,
                                           nh, w, nw));
          }
          // Round map entries before accumulating, as on the dense path.
          auto value =
              ssim ? image_map_with_identity(x, y, comparison, settings[3],
                                             parameters, identical)
                         .to(at::ScalarType(dtype))
                   : pixel_map_region(x[0], y[0], comparison, dtype, code != 6);
          if (code == 6) {
            auto target =
                contract_region(state.movedim(-1, 0), mask, c, nc, h, nh, w, nw)
                    .movedim(0, -1);
            // CUDA consumes rounded public entries directly and promotes
            // each factor in registers; the ATen fallback promotes here.
            contract_product_update(target, value, axes);
            continue;
          }
          value = value.to(at::kDouble);
          if (code >= 4) {
            auto target =
                contract_region(state.movedim(-1, 0), mask, c, nc, h, nh, w, nw)
                    .movedim(0, -1);
            auto ids = code <= 5 ? contract_ids(a, settings, comparison, paired,
                                                c, nc, h, nh, w, nw)
                                 : Tensor();
            if (code <= 5)
              contract_extrema_update(target, value, ids, axes, code);
            else
              detail::residual::reducer_update(target, value, ids, axes,
                                               contract_name(code));
            continue;
          }
          auto first =
              contract_region(state.select(-1, 0), mask, c, nc, h, nh, w, nw);
          auto second =
              contract_region(state.select(-1, 1), mask, c, nc, h, nh, w, nw);
          if (code < 2) {
            auto update = detail::streaming::compensated(first, second,
                                                         value.sum(axes, true));
            first.copy_(update.first);
            second.copy_(update.second);
          } else {
            double count = (mask & 1 ? nc : 1) * double(mask & 2 ? nh : 1) *
                           (mask & 4 ? nw : 1);
            if (anchored) {
              auto target = contract_region(state.movedim(-1, 0), mask, c, nc,
                                            h, nh, w, nw)
                                .movedim(0, -1);
              detail::update_centered_moments(target, value, axes, count);
              continue;
            }
            // Anchor locally to retain small variance around a large offset.
            auto anchor = value;
            for (auto axis : axes)
              anchor = anchor.narrow(axis, 0, 1);
            auto centered = value - anchor;
            auto shift = centered.mean(axes, true);
            auto mean = anchor + shift;
            auto m2 = (centered - shift).square().sum(axes, true);
            auto third =
                contract_region(state.select(-1, 2), mask, c, nc, h, nh, w, nw);
            auto total = first + count;
            auto delta = mean - second;
            third.copy_(at::where(
                first.gt(0),
                third + m2 + delta.square() * first * count / total, m2));
            second.copy_(
                at::where(first.gt(0), second + delta * count / total, mean));
            first.copy_(total);
          }
        }
  }
  int64_t count = 1;
  for (int64_t k = 0; k < 3; ++k)
    if (mask & (1 << k))
      count *= k == 0 ? settings[0] : settings[k] - halo;
  Tensor result;
  if (code == 6) {
    result = state.select(-1, 0).clone(at::MemoryFormat::Contiguous);
  } else if (code >= 4) {
    result = detail::residual::reducer_result(state, contract_name(code));
    // Extrema result is a view into state; the schema promises independent,
    // contiguous outputs, including when public dtype is already double.
    if (code <= 5)
      result = result.clone(at::MemoryFormat::Contiguous);
  } else if (code < 2) {
    result = state.select(-1, 0) + state.select(-1, 1);
    if (code == 1)
      result = result / count;
  } else {
    auto denominator = std::max(0., double(count) - correction);
    result = state.select(-1, 2) / denominator;
    if (code == 3)
      result = result.sqrt();
  }
  return {result.to(code >= 7 ? at::kBool : at::ScalarType(dtype))
              .reshape(contract_shape(a, b, comparison, settings, mask, paired,
                                      keepdim)),
          state};
}
std::tuple<Tensor, Tensor> map_contract_backward_kernel(
    const Tensor& gradient, const Tensor& a, const Tensor& b,
    const Tensor& state, int64_t comparison, at::IntArrayRef settings,
    at::ArrayRef<double> parameters, int64_t dtype, int64_t code, int64_t mask,
    bool keepdim, double correction, bool paired) {
  validate_contract(a, b, comparison, settings, parameters, dtype, code, mask,
                    correction, paired);
  if (code >= 7)
    return {at::zeros_like(a), at::zeros_like(b)};
  auto shape = contract_shape(a, b, comparison, settings, mask, paired, true);
  auto upstream = gradient.reshape(shape).to(at::kDouble);
  if (comparison == 0 && (code == 4 || code == 5)) {
    auto ga = at::zeros_like(a), gb = at::zeros_like(b);
    if (identity_extrema_backward(ga, gb, upstream, state.select(-1, 1), dtype,
                                  false, paired, 128))
      return {ga, gb};
  }
  int64_t count = 1;
  auto halo = comparison == 3 ? settings[3] - 1 : 0;
  for (int64_t k = 0; k < 3; ++k)
    if (mask & (1 << k))
      count *= k == 0 ? settings[0] : settings[k] - halo;
  auto denominator = std::max(0., double(count) - correction);
  if (comparison == 0 && (code == 0 || code == 1))
    return identity_linear_backward(upstream, a, b, settings, dtype,
                                    code == 1 ? double(count) : 1., paired);
  auto any_zero = code == 6 ? state.select(-1, 1).gt(0).any() : Tensor();
  auto cotangent = [&](const Tensor& value, int64_t c, int64_t nc, int64_t h,
                       int64_t nh, int64_t w, int64_t nw) {
    auto result = contract_region(upstream, mask, c, nc, h, nh, w, nw);
    if (code == 1)
      result = result / count;
    if (code == 6) {
      auto saved =
          contract_region(state.movedim(-1, 0), mask, c, nc, h, nh, w, nw)
              .movedim(0, -1);
      result = contract_product_gradient(saved, value, result, any_zero);
    } else if (code == 4 || code == 5) {
      auto witness =
          contract_region(state.select(-1, 1), mask, c, nc, h, nh, w, nw);
      auto ids =
          contract_ids(a, settings, comparison, paired, c, nc, h, nh, w, nw);
      result = at::where(ids.eq(witness), result, 0.);
    } else if (code >= 2) {
      auto mean =
          contract_region(state.select(-1, 1), mask, c, nc, h, nh, w, nw);
      auto centered = value.to(at::kDouble);
      if (state.size(-1) == 4) {
        auto anchor =
            contract_region(state.select(-1, 3), mask, c, nc, h, nh, w, nw);
        centered = centered - anchor;
      }
      result = result * (centered - mean);
      if (code == 2)
        result = result * (2. / denominator);
      else {
        auto m2 =
            contract_region(state.select(-1, 2), mask, c, nc, h, nh, w, nw);
        auto std = (m2 / denominator).sqrt();
        result = at::where(std.eq(0), at::zeros_like(result),
                           result / (denominator * std));
      }
    }
    auto expanded = shape;
    expanded[expanded.size() - 3] = nc;
    expanded[expanded.size() - 2] = nh;
    expanded[expanded.size() - 1] = nw;
    // The public map cast also rounds its incoming cotangent.
    return result.expand(expanded).to(at::ScalarType(dtype));
  };
  return map_backward_regions(cotangent, code == 2 || code == 3 || code == 6, a,
                              b, comparison, settings, parameters, dtype,
                              paired, int64_t(1) << (code == 6 ? 20 : 23));
}
}  // namespace
void update_map_extrema(Tensor state, const Tensor& value, const Tensor& ids,
                        at::IntArrayRef axes, bool smallest) {
  contract_extrema_update(state, value, ids, axes, smallest ? 4 : 5);
}
Tensor map_contract(const Tensor& a, const Tensor& b, int64_t comparison,
                    at::IntArrayRef settings, at::ArrayRef<double> parameters,
                    int64_t dtype, int64_t code, int64_t mask, bool keepdim,
                    double correction, bool paired) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_image_map_contract", "anchored")
          .typed<MapContractSignature>();
  return std::get<0>(op.call(a, b, comparison, settings, parameters, dtype,
                             code, mask, keepdim, correction, paired));
}
bool use_map_contract(const Tensor& a, const Tensor& b, int64_t comparison,
                      at::IntArrayRef settings, bool paired) {
  return large_map(a, b, comparison, settings, paired);
}
Tensor map_values(const Tensor& a, const Tensor& b, int64_t comparison,
                  at::IntArrayRef settings, at::ArrayRef<double> parameters,
                  int64_t dtype, bool paired) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_image_map_values", "")
          .typed<MapValuesSignature>();
  return op.call(a, b, comparison, settings, parameters, dtype, paired);
}
Tensor map_values_for_replay(const Tensor& a, const Tensor& b,
                             int64_t comparison, at::IntArrayRef settings,
                             at::ArrayRef<double> parameters, int64_t dtype,
                             bool paired) {
  if (at::GradMode::is_enabled() &&
      !large_map(a, b, comparison, settings, paired))
    return map_values_composite(a, b, comparison, settings, parameters, dtype,
                                paired);
  return map_values(a, b, comparison, settings, parameters, dtype, paired);
}
namespace {
EventOutputs legacy_map_contract_kernel(
    const Tensor& a, const Tensor& b, int64_t comparison,
    at::IntArrayRef settings, at::ArrayRef<double> parameters, int64_t dtype,
    int64_t code, int64_t mask, bool keepdim, double correction, bool paired) {
  auto [result, state] =
      map_contract_kernel(a, b, comparison, settings, parameters, dtype, code,
                          mask, keepdim, correction, paired);
  return {result, detail::legacy_moment_state(state, 3, 1)};
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_image_map_contract(Tensor x, Tensor y, int comparison, int[] settings, "
      "float[] parameters, int dtype, int code, int mask, bool keepdim, "
      "float correction, bool paired) -> (Tensor, Tensor)");
  m.def(
      "_image_map_contract.anchored(Tensor x, Tensor y, int comparison, int[] "
      "settings, float[] parameters, int dtype, int code, int mask, bool "
      "keepdim, "
      "float correction, bool paired) -> (Tensor, Tensor)");
  m.def(
      "_image_map_contract_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "state, int comparison, int[] settings, float[] parameters, int dtype, "
      "int code, int mask, bool keepdim, float correction, bool paired) "
      "-> (Tensor, Tensor)");
  m.def(
      "_image_map_values(Tensor x, Tensor y, int comparison, int[] settings, "
      "float[] parameters, int dtype, bool paired) -> Tensor");
  m.def(
      "_image_map_values_backward(Tensor grad, Tensor x, Tensor y, int "
      "comparison, int[] settings, float[] parameters, int dtype, bool paired) "
      "-> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_image_map_contract", TORCH_FN(legacy_map_contract_kernel));
  m.impl("_image_map_contract.anchored", TORCH_FN(map_contract_kernel));
  m.impl("_image_map_contract_backward",
         TORCH_FN(map_contract_backward_kernel));
  m.impl("_image_map_values", TORCH_FN(map_values_kernel));
  m.impl("_image_map_values_backward", TORCH_FN(map_values_backward_kernel));
}
}  // namespace
Tensor map_reduction(const Tensor& a, const Tensor& b, int64_t comparison,
                     c10::string_view reduction, at::IntArrayRef settings,
                     at::ArrayRef<double> parameters, int64_t dtype,
                     double correction, bool paired) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_image_map_reduce", "")
          .typed<MapReductionSignature>();
  return op.call(a, b, comparison, reduction, settings, parameters, dtype,
                 correction, paired);
}
Tensor feature_pair(const Tensor& a, const Tensor& b, int64_t kind, bool paired,
                    at::IntArrayRef settings, at::ArrayRef<double> parameters) {
  auto left = unpack(paired ? a : a.unsqueeze(-2), kind, settings, parameters);
  auto right = unpack(paired ? b : b.unsqueeze(-3), kind, settings, parameters);
  return kind < 3 ? image_score(left, right, kind, settings[3], parameters)
                  : audio_score(left, right, kind - 3, settings.slice(2),
                                parameters);
}
}  // namespace detail::media
namespace detail::streaming {
PreparedComparison prepare_image_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const PairwiseOperationOptions& matrix,
    const ImageComparisonOptions& comparison) {
  validate_pairwise_options(operation, matrix);
  check_comparison_packet(matrix);
  auto options = comparison;
  options.chunk_size = matrix.chunk_size.value_or(comparison.chunk_size);
  auto p = detail::media::prepare_image(x, y, metric, "all_pairs", options);
  return prepare_media_source(std::move(p), metric, operation, matrix,
                              options.chunk_size, operator_autocast_enabled(x));
}
PreparedComparison prepare_image_map_operator(
    const Tensor& x, const c10::optional<Tensor>& y,
    c10::string_view comparison, c10::string_view reduction, double correction,
    const PairwiseOperationOptions& matrix, const ImageMapOptions& options) {
  validate_map_reduction(reduction, correction);
  TORCH_CHECK(!options.keepdim,
              "Lazy image event contractions require keepdim=False");
  auto kind = comparison == "residual"         ? 0
              : comparison == "absolute_error" ? 1
              : comparison == "squared_error"  ? 2
              : comparison == "ssim"           ? 3
                                               : -1;
  TORCH_CHECK(kind >= 0, "Unknown image map comparison");
  TORCH_CHECK(kind == 3 || (!options.data_range && options.window_size == 11 &&
                            options.sigma == 1.5 && options.k1 == .01 &&
                            options.k2 == .03),
              "SSIM options require comparison=ssim");
  TORCH_CHECK(kind != 3 || options.data_range, "SSIM maps require data_range");
  ImageComparisonOptions q;
  q.data_range = options.data_range.value_or(1.);
  q.window_size = options.window_size;
  q.sigma = options.sigma;
  q.k1 = options.k1;
  q.k2 = options.k2;
  q.chunk_size = matrix.chunk_size.value_or(options.chunk_size);
  auto p = detail::media::prepare_image(x, y, kind == 3 ? "ssim" : "psnr",
                                        "all_pairs", q);
  // Public maps subtract/transform in double then round their entries to dtype.
  // Preserve that policy even for the pixel-only preparation used by PSNR.
  if (p.x.scalar_type() != at::kDouble) {
    auto a = p.x.to(at::kDouble);
    p.y = p.x.is_same(p.y) ? a : p.y.to(at::kDouble);
    p.x = a;
  }
  auto result =
      prepare_media_source(std::move(p), reduction, "diagonal", matrix,
                           q.chunk_size, operator_autocast_enabled(x));
  result.settings.family = "_image_map";
  result.settings.integers[0] = kind;
  result.settings.integers.push_back(result.settings.dtype);
  result.settings.reals.push_back(correction);
  if (detail::residual::boolean_reduction(reduction))
    result.settings.dtype = int64_t(at::kBool);
  return result;
}
PreparedComparison prepare_audio_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const PairwiseOperationOptions& matrix,
    const AudioComparisonOptions& comparison) {
  validate_pairwise_options(operation, matrix);
  check_comparison_packet(matrix);
  auto options = comparison;
  options.chunk_size = matrix.chunk_size.value_or(comparison.chunk_size);
  auto p = detail::media::prepare_audio(x, y, metric, "all_pairs", options);
  p.kind += 3;
  return prepare_media_source(std::move(p), metric, operation, matrix,
                              options.chunk_size, operator_autocast_enabled(x));
}
}  // namespace detail::streaming
Tensor media_pairwise_operations(const Tensor& x,
                                 const c10::optional<Tensor>& y,
                                 c10::string_view metric,
                                 c10::string_view operation,
                                 const PairwiseOperationOptions& matrix,
                                 const ImageComparisonOptions& image) {
  check_comparison_packet(matrix);
  auto m = matrix;
  if (!m.chunk_size)
    m.chunk_size = image.chunk_size;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::image_pairwise_operations", "")
          .typed<decltype(image_operations_impl)>();
  return op.call(x, y, metric, operation, m.left, m.right, m.rows, m.columns,
                 m.reduction, m.dim, m.keepdim, m.exclude_self, m.transposed,
                 m.chunk_size, image.data_range, image.window_size, image.sigma,
                 image.k1, image.k2, image.weights);
}

Tensor media_pairwise_operations(const Tensor& x,
                                 const c10::optional<Tensor>& y,
                                 c10::string_view metric,
                                 c10::string_view operation,
                                 const PairwiseOperationOptions& matrix,
                                 const AudioComparisonOptions& audio) {
  check_comparison_packet(matrix);
  auto m = matrix;
  if (!m.chunk_size)
    m.chunk_size = audio.chunk_size;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::audio_pairwise_operations", "")
          .typed<decltype(audio_operations_impl)>();
  return op.call(x, y, metric, operation, m.left, m.right, m.rows, m.columns,
                 m.reduction, m.dim, m.keepdim, m.exclude_self, m.transposed,
                 m.chunk_size, audio.zero_mean, audio.energy_floor, audio.n_fft,
                 audio.hop_length, audio.win_length, audio.window, audio.center,
                 audio.pad_mode, audio.normalized, audio.log_floor,
                 audio.norm_floor, audio.resolutions);
}
ImageMapOperator::ImageMapOperator(Tensor x, c10::optional<Tensor> y,
                                   std::string comparison,
                                   ImageMapOptions options)
    : x_(std::move(x)),
      y_(std::move(y)),
      comparison_(std::move(comparison)),
      options_(std::move(options)) {}
ScalarExpression ImageMapOperator::reduce(
    c10::string_view reduction, at::IntArrayRef dims,
    c10::optional<double> correction) const {
  TORCH_CHECK(x_.dim() >= 4 && (!y_ || y_->dim() >= 4),
              "Images require [...,N,C,H,W]");
  auto rank = std::max(x_.dim(), y_ ? y_->dim() : x_.dim()) + 1;
  auto axes = detail::operator_axes(dims, rank, 5);
  TORCH_CHECK(
      axes == std::vector<int64_t>({2, 3, 4}),
      "ImageMapOperator currently contracts all C/H/W event axes together");
  TORCH_CHECK(!correction || map_moments(reduction),
              "correction is only used by var/std");
  TORCH_CHECK(!options_.keepdim,
              "Lazy image event contractions require keepdim=False");
  validate_map_reduction(reduction, correction.value_or(1.));
  ExpressionSource source;
  source.inputs = {x_};
  if (y_)
    source.inputs.push_back(*y_);
  source.family = "image_map";
  source.metric = std::string(reduction);
  source.strings.insert("comparison", comparison_);
  source.reals.insert("correction", correction.value_or(1.));
  if (options_.data_range)
    source.reals.insert("data_range", *options_.data_range);
  source.reals.insert("sigma", options_.sigma);
  source.reals.insert("k1", options_.k1);
  source.reals.insert("k2", options_.k2);
  source.integers.insert("window_size", options_.window_size);
  source.integers.insert("chunk_size", options_.chunk_size);
  return ScalarExpression(std::move(source));
}
ImageMapOperator image_map_operator(const Tensor& x,
                                    const c10::optional<Tensor>& y,
                                    c10::string_view comparison,
                                    const ImageMapOptions& options) {
  return ImageMapOperator(x, y, std::string(comparison), options);
}
}  // namespace torchpairwise::ops
