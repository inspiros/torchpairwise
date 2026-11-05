#include <ATen/ExpandUtils.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "image_map_operations_detail.h"
#include "residual_reducer.h"
#include "utils/centered_moments.h"
#include "utils/operator_axes.h"

namespace torchpairwise::ops {
using at::Tensor;
namespace detail::media {
namespace {
using namespace detail::streaming;
constexpr const char* reducers[] = {"sum", "mean", "var", "std", "min",
                                    "max", "prod", "all", "any"};
bool moments(c10::string_view name) {
  return name == "var" || name == "std";
}
void validate_reduction(c10::string_view name, at::ScalarType dtype) {
  bool found = false;
  for (auto item : reducers)
    found |= name == item;
  TORCH_CHECK(found, "Unsupported image event reduction");
  TORCH_CHECK((name != "mean" && !moments(name)) || at::isFloatingType(dtype),
              "mean/var/std require floating point entries");
}
at::ScalarType reduction_dtype(c10::string_view name, at::ScalarType dtype) {
  validate_reduction(name, dtype);
  if (residual::boolean_reduction(name))
    return at::kBool;
  if ((name == "sum" || name == "prod") && !at::isFloatingType(dtype))
    return at::kLong;
  return dtype;
}
// All selected axes are flattened together, in logical row-major order. This
// gives min/max one first witness and prod one joint (rather than nested) VJP.
Tensor local_reduce(Tensor value, c10::string_view name, at::IntArrayRef axes,
                    bool keepdim, double correction) {
  auto dtype = reduction_dtype(name, value.scalar_type());
  std::vector<int64_t> order, retained, shape;
  for (int64_t k = 0; k < value.dim(); ++k) {
    bool selected = std::find(axes.begin(), axes.end(), k) != axes.end();
    if (!selected) {
      order.push_back(k);
      retained.push_back(value.size(k));
    }
    if (keepdim || !selected)
      shape.push_back(selected ? 1 : value.size(k));
  }
  order.insert(order.end(), axes.begin(), axes.end());
  int64_t count = 1;
  for (auto axis : axes)
    count *= value.size(axis);
  retained.push_back(count);
  value = value.permute(order).reshape(retained);
  // Accumulate floating event contractions in double, rounding each stage to
  // its public dtype. Integer stages stay integer, including large products.
  if (value.is_floating_point())
    value = value.to(at::kDouble);
  if (moments(name) && dtype == at::kDouble)
    value = detail::centered_local_values(value);
  Tensor result;
  if (name == "sum")
    result = value.sum(-1);
  else if (name == "mean")
    result = value.mean(-1);
  else if (name == "var")
    result = value.var(at::IntArrayRef{-1},
                       c10::optional<at::Scalar>(correction), false);
  else if (name == "std")
    result = value.std(at::IntArrayRef{-1},
                       c10::optional<at::Scalar>(correction), false);
  else if (name == "min")
    result = std::get<0>(value.min(-1));
  else if (name == "max")
    result = std::get<0>(value.max(-1));
  else if (name == "prod")
    result = value.prod(-1);
  else if (name == "all")
    result = value.ne(0).all(-1);
  else
    result = value.ne(0).any(-1);
  return result.to(dtype).reshape(shape);
}
Tensor program_values(const Tensor& x, const Tensor& y, int64_t comparison,
                      at::IntArrayRef settings, at::ArrayRef<double> parameters,
                      int64_t dtype, at::IntArrayRef program,
                      at::ArrayRef<double> corrections, bool paired) {
  size_t first = 0;
  Tensor value;
  if (!corrections.empty() &&
      use_map_contract(x, y, comparison, settings, paired)) {
    value =
        map_contract(x, y, comparison, settings, parameters, dtype, program[0],
                     program[1], program[2], corrections[0], paired);
    first = 1;
  } else {
    value = map_values_for_replay(x, y, comparison, settings, parameters, dtype,
                                  paired);
  }
  auto sample_rank = paired ? 1 : 2;
  for (size_t k = first; k < corrections.size(); ++k) {
    std::vector<int64_t> axes;
    for (int64_t d = sample_rank; d < value.dim(); ++d)
      if (program[3 * k + 1] & (int64_t(1) << (d - sample_rank)))
        axes.push_back(d);
    value = local_reduce(value, reducers[program[3 * k]], axes,
                         program[3 * k + 2], corrections[k]);
  }
  return value;
}
Tensor output_tile(Tensor value, const std::vector<bool>& reduced, int64_t i,
                   int64_t ni, int64_t j, int64_t nj) {
  value = value.narrow(0, reduced[0] ? 0 : i, reduced[0] ? 1 : ni);
  return value.narrow(1, reduced[1] ? 0 : j, reduced[1] ? 1 : nj);
}
struct Layout {
  std::vector<int64_t> batch, sizes, shape;
  std::vector<bool> reduced;
  int64_t batches = 1;
  double eligible = 1.;
};
Layout layout(const Tensor& x, const Tensor& y, const EventMetadata& metadata,
              at::IntArrayRef axes, c10::string_view reduction) {
  Layout l;
  l.batch = at::infer_size(x.sizes().slice(0, x.dim() - 2),
                           y.sizes().slice(0, y.dim() - 2));
  l.sizes = {x.size(-2), y.size(-2)};
  l.sizes.insert(l.sizes.end(), metadata.shape.begin(), metadata.shape.end());
  l.reduced.resize(l.sizes.size(), false);
  TORCH_CHECK(!axes.empty(), "Terminal reduction requires sample/event axes");
  for (auto axis : axes) {
    TORCH_CHECK(axis >= 0 && axis < int64_t(l.sizes.size()) && !l.reduced[axis],
                "Invalid image event terminal axes");
    l.reduced[axis] = true;
    l.eligible *= l.sizes[axis];
    TORCH_CHECK(l.sizes[axis] > 0 || (reduction != "min" && reduction != "max"),
                "min/max require nonempty reduction axes");
  }
  l.shape = l.batch;
  for (size_t k = 0; k < l.sizes.size(); ++k)
    l.shape.push_back(l.reduced[k] ? 1 : l.sizes[k]);
  for (auto b : l.batch)
    l.batches *= b;
  return l;
}
Tensor witness_ids(const Tensor& value, const Layout& l, int64_t i, int64_t j,
                   at::IntArrayRef offsets = {}) {
  int64_t features = 1;
  for (size_t k = 2; k < l.sizes.size(); ++k)
    features *= l.sizes[k];
  auto opts = value.options().dtype(at::kLong);
  std::vector<int64_t> a(value.dim(), 1), b = a, c = a;
  a[0] = value.size(0);
  b[1] = value.size(1);
  auto ids = (at::arange(i, i + value.size(0), opts).reshape(a) * l.sizes[1] +
              at::arange(j, j + value.size(1), opts).reshape(b)) *
             features;
  if (!offsets.empty()) {
    int64_t stride = 1;
    for (int64_t d = value.dim() - 1; d >= 2; --d) {
      std::vector<int64_t> shape(value.dim(), 1);
      shape[d] = value.size(d);
      ids =
          ids + at::arange(offsets[d - 2], offsets[d - 2] + value.size(d), opts)
                        .reshape(shape) *
                    stride;
      stride *= l.sizes[d];
    }
  } else if (value.dim() > 2) {
    c[2] = features;
    auto flat = at::arange(features, opts).reshape(c);
    auto shape = value.sizes().vec();
    shape[0] = shape[1] = 1;
    ids = ids + flat.reshape(shape);
  }
  return ids;
}
// Sample axes have already been sliced by output_tile. Retained event
// dimensions address disjoint output/state regions; reduced ones stay
// singleton.
Tensor event_region(Tensor value, const Layout& l, at::IntArrayRef offsets,
                    at::IntArrayRef lengths) {
  for (size_t k = 0; k < offsets.size(); ++k)
    value = value.narrow(k + 2, l.reduced[k + 2] ? 0 : offsets[k],
                         l.reduced[k + 2] ? 1 : lengths[k]);
  return value;
}
// Integer-valued truth/count chains must never pass through a floating state.
Tensor integer_state(at::IntArrayRef shape, at::TensorOptions options,
                     c10::string_view name) {
  auto sizes = shape.vec();
  sizes.push_back(1);
  auto state = at::zeros(sizes, options.dtype(at::kLong));
  if (name == "prod" || name == "all")
    state.fill_(1);
  if (name == "min")
    state.fill_(std::numeric_limits<int64_t>::max());
  if (name == "max")
    state.fill_(std::numeric_limits<int64_t>::min());
  return state;
}
void integer_update(Tensor state, Tensor value, at::IntArrayRef axes,
                    c10::string_view name) {
  auto part = local_reduce(value, name, axes, true, 1.).to(at::kLong);
  auto target = state.select(-1, 0);
  if (name == "sum")
    target.add_(part);
  else if (name == "prod")
    target.mul_(part);
  else if (name == "min")
    target.copy_(at::minimum(target, part));
  else if (name == "max")
    target.copy_(at::maximum(target, part));
  else if (name == "all")
    target.copy_(target.ne(0) & part.ne(0));
  else
    target.copy_(target.ne(0) | part.ne(0));
}
EventOutputs event_kernel(const Tensor& x, const Tensor& y, int64_t comparison,
                          at::IntArrayRef settings,
                          at::ArrayRef<double> parameters, int64_t dtype,
                          at::IntArrayRef program,
                          at::ArrayRef<double> corrections,
                          c10::string_view reduction, at::IntArrayRef axes,
                          bool keepdim, double correction, int64_t size,
                          bool paired) {
  auto metadata =
      event_metadata(comparison, settings, dtype, program, corrections);
  TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2 && x.layout() == at::kStrided &&
                  y.layout() == at::kStrided && x.device() == y.device() &&
                  x.scalar_type() == at::kDouble &&
                  y.scalar_type() == at::kDouble && parameters.size() == 4,
              "Image event core requires matching packed double collections");
  TORCH_CHECK(size > 0 && std::isfinite(correction),
              "Invalid image event reduction options");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  if (reduction.empty()) {
    TORCH_CHECK(axes.empty() && !keepdim,
                "Local event program takes no terminal axes");
    return {program_values(x, y, comparison, settings, parameters, dtype,
                           program, corrections, paired),
            at::empty({0}, x.options())};
  }
  TORCH_CHECK(!paired, "Terminal image reductions require Cartesian samples");
  auto out_dtype = reduction_dtype(reduction, metadata.dtype);
  auto l = layout(x, y, metadata, axes, reduction);
  bool integral = !at::isFloatingType(metadata.dtype),
       extra = residual::extra_reduction(reduction), stats = moments(reduction);
  auto shape = l.shape;
  bool anchored = stats && metadata.dtype == at::kDouble;
  shape.push_back(anchored ? 4 : 3);
  auto state = integral ? integer_state(l.shape, x.options(), reduction)
               : extra
                   ? residual::reducer_state(l.shape, x.options(), reduction)
                   : at::zeros(shape, x.options());
  int64_t rank = l.sizes.size(), n = l.sizes[0], m = l.sizes[1];
  for (int64_t flat = 0; flat < l.batches; ++flat) {
    auto index = coordinates(flat, l.batch);
    auto a = lookup(x, index, 2), b = lookup(y, index, 2),
         saved = lookup(state, index, rank + 1);
    for (int64_t i = 0; i < n; i += size)
      for (int64_t j = 0; j < m; j += size) {
        auto ni = std::min(size, n - i), nj = std::min(size, m - j);
        auto left = a.narrow(0, i, ni), right = b.narrow(0, j, nj);
        auto update = [&](Tensor value, at::IntArrayRef offsets) {
          auto target =
              event_region(output_tile(saved, l.reduced, i, ni, j, nj), l,
                           offsets, value.sizes().slice(2));
          if (integral) {
            integer_update(target, value, axes, reduction);
            return;
          }
          value = value.to(at::kDouble);
          if (extra) {
            auto ids = witness_ids(value, l, i, j, offsets);
            if (reduction == "min" || reduction == "max")
              update_map_extrema(target, value, ids, axes, reduction == "min");
            else
              residual::reducer_update(target, value, ids, axes, reduction);
            return;
          }
          auto count = target.select(-1, 0), mean = target.select(-1, 1),
               m2 = target.select(-1, 2);
          double k = 1.;
          for (auto axis : axes)
            k *= value.size(axis);
          if (anchored) {
            detail::update_centered_moments(target, value, axes, k);
          } else if (stats) {
            auto anchor = value;
            for (auto axis : axes)
              anchor = anchor.narrow(axis, 0, 1);
            auto safe = at::where(anchor.isfinite(), anchor, 0.);
            auto average = safe + (value - safe).sum(axes, true) / k;
            auto partial = (value - average).square().sum(axes, true);
            auto total = count + k, delta = average - mean, ratio = k / total;
            m2.copy_(at::where(count.gt(0),
                               m2 + partial + delta.square() * count * ratio,
                               partial));
            mean.copy_(at::where(count.gt(0), mean + delta * ratio, average));
            count.copy_(total);
          } else {
            auto update = compensated(mean, m2, value.sum(axes, true));
            mean.copy_(update.first);
            m2.copy_(update.second);
          }
        };
        if (program.empty() &&
            use_map_contract(left, right, comparison, settings, false)) {
          visit_map_regions(
              left, right, comparison, settings, parameters, dtype, false,
              [&](const Tensor& value, int64_t c, int64_t nc, int64_t h,
                  int64_t nh, int64_t w,
                  int64_t nw) { update(value, {c, h, w}); },
              true);
        } else {
          update(program_values(left, right, comparison, settings, parameters,
                                dtype, program, corrections, false),
                 {});
        }
      }
  }
  Tensor result;
  if (integral)
    result = state.select(-1, 0).clone();
  else if (extra)
    result = residual::reducer_result(state, reduction).clone();
  else if (stats) {
    auto count = state.select(-1, 0);
    result = at::where(count.gt(0),
                       state.select(-1, 2) / (count - correction).clamp_min(0),
                       std::numeric_limits<double>::quiet_NaN());
    if (reduction == "std")
      result = result.sqrt();
  } else {
    result = state.select(-1, 1) + state.select(-1, 2);
    if (reduction == "mean")
      result = result / l.eligible;
  }
  if (!keepdim)
    for (int64_t axis = rank - 1; axis >= 0; --axis)
      if (l.reduced[axis])
        result = result.squeeze(l.batch.size() + axis);
  return {result.to(out_dtype),
          stats || extra ? state : at::empty({0}, state.options())};
}
std::tuple<Tensor, Tensor> event_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& state, int64_t comparison, at::IntArrayRef settings,
    at::ArrayRef<double> parameters, int64_t dtype, at::IntArrayRef program,
    at::ArrayRef<double> corrections, c10::string_view reduction,
    at::IntArrayRef axes, bool keepdim, double correction, int64_t size,
    bool paired) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher image event derivatives are unsupported");
  auto metadata =
      event_metadata(comparison, settings, dtype, program, corrections);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::NoGradGuard guard;
  auto gx = at::zeros_like(x), gy = at::zeros_like(y);
  if (!metadata.differentiable || residual::boolean_reduction(reduction))
    return {gx, gy};
  auto vjp = [&](const Tensor& a, const Tensor& b, const Tensor& upstream,
                 Tensor dx, Tensor dy) {
    at::AutoGradMode enable(true);
    auto left = a.detach().set_requires_grad(true),
         right = b.detach().set_requires_grad(true);
    auto value = program_values(left, right, comparison, settings, parameters,
                                dtype, program, corrections, paired);
    auto grads = torch::autograd::grad({value}, {left, right},
                                       {upstream.to(value.scalar_type())});
    at::NoGradGuard no_grad;
    dx.add_(grads[0]);
    dy.add_(grads[1]);
  };
  if (reduction.empty()) {
    if (x.size(0) && y.size(0))
      vjp(x, y, gradient, gx, gy);
    return {gx, gy};
  }
  auto l = layout(x, y, metadata, axes, reduction);
  int64_t rank = l.sizes.size(), n = l.sizes[0], m = l.sizes[1];
  auto g = gradient.to(at::kDouble).reshape(l.shape);
  for (int64_t flat = 0; flat < l.batches; ++flat) {
    auto index = coordinates(flat, l.batch);
    auto a = lookup(x, index, 2), b = lookup(y, index, 2),
         upstream = lookup(g, index, rank);
    if (program.empty() && comparison == 0 &&
        (reduction == "min" || reduction == "max") &&
        identity_extrema_backward(lookup(gx, index, 2), lookup(gy, index, 2),
                                  upstream,
                                  lookup(state, index, rank + 1).select(-1, 1),
                                  dtype, true, false, size))
      continue;
    for (int64_t i = 0; i < n; i += size)
      for (int64_t j = 0; j < m; j += size) {
        auto ni = std::min(size, n - i), nj = std::min(size, m - j);
        auto left = a.narrow(0, i, ni), right = b.narrow(0, j, nj);
        if (program.empty() && comparison == 0 &&
            (reduction == "sum" || reduction == "mean")) {
          auto [dx, dy] = identity_linear_backward(
              output_tile(upstream, l.reduced, i, ni, j, nj), left, right,
              settings, dtype, reduction == "mean" ? l.eligible : 1., false);
          lookup(gx, index, 2).narrow(0, i, ni).add_(dx);
          lookup(gy, index, 2).narrow(0, j, nj).add_(dy);
          continue;
        }
        if (program.empty() &&
            use_map_contract(left, right, comparison, settings, false)) {
          auto cotangent_for = [&](const Tensor& value, int64_t c, int64_t nc,
                                   int64_t h, int64_t nh, int64_t w,
                                   int64_t nw) {
            auto cotangent =
                event_region(output_tile(upstream, l.reduced, i, ni, j, nj), l,
                             {c, h, w}, {nc, nh, nw});
            std::vector<int64_t> region_shape{ni, nj, nc, nh, nw};
            if (residual::extra_reduction(reduction)) {
              auto saved =
                  event_region(output_tile(lookup(state, index, rank + 1),
                                           l.reduced, i, ni, j, nj),
                               l, {c, h, w}, {nc, nh, nw});
              auto values = value.defined() ? value.to(at::kDouble)
                                            : cotangent.expand(region_shape);
              cotangent = residual::reducer_gradient(
                  saved, values, witness_ids(values, l, i, j, {c, h, w}),
                  cotangent, reduction);
            } else if (moments(reduction)) {
              auto saved =
                  event_region(output_tile(lookup(state, index, rank + 1),
                                           l.reduced, i, ni, j, nj),
                               l, {c, h, w}, {nc, nh, nw});
              auto denominator =
                  (saved.select(-1, 0) - correction).clamp_min(0);
              if (reduction == "std") {
                auto std = (saved.select(-1, 2) / denominator).sqrt();
                cotangent = at::where(std.eq(0), 0., cotangent / (2. * std));
              }
              cotangent =
                  cotangent * (2. / denominator) *
                  detail::centered_moment_delta(value.to(at::kDouble), saved);
            } else if (reduction == "mean")
              cotangent = cotangent / l.eligible;
            return cotangent.expand(region_shape).to(at::ScalarType(dtype));
          };
          auto [dx, dy] = map_backward_regions(
              cotangent_for, moments(reduction) || reduction == "prod", left,
              right, comparison, settings, parameters, dtype, false);
          lookup(gx, index, 2).narrow(0, i, ni).add_(dx);
          lookup(gy, index, 2).narrow(0, j, nj).add_(dy);
          continue;
        }
        // Sum/mean adjoints depend on shape and count, not on the contracted
        // entries. The VJP below replays the program once; do not also rebuild
        // its values just to broadcast an upstream gradient.
        Tensor values;
        if (residual::extra_reduction(reduction) || moments(reduction))
          values = program_values(left, right, comparison, settings, parameters,
                                  dtype, program, corrections, false)
                       .to(at::kDouble);
        auto cotangent = output_tile(upstream, l.reduced, i, ni, j, nj);
        if (residual::extra_reduction(reduction)) {
          auto saved = output_tile(lookup(state, index, rank + 1), l.reduced, i,
                                   ni, j, nj);
          cotangent = residual::reducer_gradient(saved, values,
                                                 witness_ids(values, l, i, j),
                                                 cotangent, reduction);
        } else if (moments(reduction)) {
          auto saved = output_tile(lookup(state, index, rank + 1), l.reduced, i,
                                   ni, j, nj);
          auto denominator = (saved.select(-1, 0) - correction).clamp_min(0);
          if (reduction == "std") {
            auto std = (saved.select(-1, 2) / denominator).sqrt();
            cotangent = at::where(std.eq(0), 0., cotangent / (2. * std));
          }
          cotangent = cotangent * (2. / denominator) *
                      detail::centered_moment_delta(values, saved);
        } else {
          std::vector<int64_t> tile_shape{ni, nj};
          tile_shape.insert(tile_shape.end(), metadata.shape.begin(),
                            metadata.shape.end());
          cotangent = cotangent.expand(tile_shape);
          if (reduction == "mean")
            cotangent = cotangent / l.eligible;
        }
        vjp(left, right, cotangent, lookup(gx, index, 2).narrow(0, i, ni),
            lookup(gy, index, 2).narrow(0, j, nj));
      }
  }
  return {gx, gy};
}
EventOutputs legacy_event_kernel(const Tensor& x, const Tensor& y,
                                 int64_t comparison, at::IntArrayRef settings,
                                 at::ArrayRef<double> parameters, int64_t dtype,
                                 at::IntArrayRef program,
                                 at::ArrayRef<double> corrections,
                                 c10::string_view reduction,
                                 at::IntArrayRef axes, bool keepdim,
                                 double correction, int64_t size, bool paired) {
  auto [result, state] = event_kernel(x, y, comparison, settings, parameters,
                                      dtype, program, corrections, reduction,
                                      axes, keepdim, correction, size, paired);
  return {result, detail::legacy_moment_state(state, 3, 1)};
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_image_event_reduce(Tensor x, Tensor y, int comparison, int[] settings, "
      "float[] parameters, int dtype, int[] program, float[] corrections, str "
      "reduction, int[] axes, bool keepdim, float correction, int chunk_size, "
      "bool paired) -> (Tensor, Tensor)");
  m.def(
      "_image_event_reduce.anchored(Tensor x, Tensor y, int comparison, int[] "
      "settings, float[] parameters, int dtype, int[] program, float[] "
      "corrections, "
      "str reduction, int[] axes, bool keepdim, float correction, int "
      "chunk_size, "
      "bool paired) -> (Tensor, Tensor)");
  m.def(
      "_image_event_reduce_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "state, int comparison, int[] settings, float[] parameters, int dtype, "
      "int[] program, float[] corrections, str reduction, int[] axes, bool "
      "keepdim, float correction, int chunk_size, bool paired) -> (Tensor, "
      "Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_image_event_reduce", TORCH_FN(legacy_event_kernel));
  m.impl("_image_event_reduce.anchored", TORCH_FN(event_kernel));
  m.impl("_image_event_reduce_backward", TORCH_FN(event_backward_kernel));
}
}  // namespace
EventMetadata event_metadata(int64_t comparison, at::IntArrayRef settings,
                             int64_t dtype, at::IntArrayRef program,
                             at::ArrayRef<double> corrections) {
  TORCH_CHECK(
      settings.size() == 4 && comparison >= 0 && comparison <= 3 &&
          program.size() == 3 * corrections.size() &&
          (dtype == int64_t(at::kFloat) || dtype == int64_t(at::kDouble)),
      "Invalid event program metadata");
  auto crop = comparison == 3 ? settings[3] - 1 : 0;
  EventMetadata m{{settings[0], settings[1] - crop, settings[2] - crop},
                  at::ScalarType(dtype),
                  true};
  for (auto d : m.shape)
    TORCH_CHECK(d > 0, "Image event axes must be nonempty");
  for (size_t k = 0; k < corrections.size(); ++k) {
    auto code = program[3 * k], mask = program[3 * k + 1],
         keep = program[3 * k + 2];
    TORCH_CHECK(code >= 0 && code < 9 && mask > 0 &&
                    mask < (int64_t(1) << m.shape.size()) &&
                    (keep == 0 || keep == 1) && std::isfinite(corrections[k]),
                "Invalid image event reduction stage");
    m.dtype = reduction_dtype(reducers[code], m.dtype);
    m.differentiable &= code != 7 && code != 8;
    for (int64_t axis = int64_t(m.shape.size()) - 1; axis >= 0; --axis)
      if (mask & (int64_t(1) << axis)) {
        if (keep)
          m.shape[axis] = 1;
        else
          m.shape.erase(m.shape.begin() + axis);
      }
  }
  return m;
}
EventOutputs event_reduce(const Tensor& x, const Tensor& y, int64_t comparison,
                          at::IntArrayRef settings,
                          at::ArrayRef<double> parameters, int64_t dtype,
                          at::IntArrayRef program,
                          at::ArrayRef<double> corrections,
                          c10::string_view reduction, at::IntArrayRef axes,
                          bool keepdim, double correction, int64_t size,
                          bool paired) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_image_event_reduce", "anchored")
          .typed<EventSignature>();
  return op.call(x, y, comparison, settings, parameters, dtype, program,
                 corrections, reduction, axes, keepdim, correction, size,
                 paired);
}
}  // namespace detail::media
namespace detail::streaming {
PreparedComparison prepare_image_event_operator(
    const Tensor& x, const c10::optional<Tensor>& y,
    c10::string_view comparison, at::IntArrayRef program,
    at::ArrayRef<double> corrections, const PairwiseOperationOptions& matrix,
    const ImageMapOptions& options) {
  auto p =
      prepare_image_map_operator(x, y, comparison, "sum", 1., matrix, options);
  std::vector<int64_t> settings;
  for (size_t k = 1; k < 5; ++k)
    settings.push_back(p.settings.integers[k].value());
  auto dtype = p.settings.integers[5].value();
  auto metadata = detail::media::event_metadata(
      p.settings.integers[0].value(), settings, dtype, program, corrections);
  p.settings.family = "_image_event";
  p.settings.metric = metadata.differentiable ? "event" : "truth_event";
  p.settings.dtype = int64_t(metadata.dtype);
  for (auto v : program)
    p.settings.integers.push_back(v);
  p.settings.reals.pop_back();
  for (auto v : corrections)
    p.settings.reals.push_back(v);
  return p;
}
}  // namespace detail::streaming
namespace {
Tensor image_event_reduction_impl(
    const Tensor& x, const c10::optional<Tensor>& y,
    c10::string_view comparison, at::IntArrayRef program,
    at::ArrayRef<double> corrections, c10::string_view reduction,
    at::IntArrayRef dims, bool keepdim, c10::optional<double> correction,
    c10::optional<double> data_range, int64_t window, double sigma, double k1,
    double k2, int64_t size) {
  TORCH_CHECK(!correction || reduction == "var" || reduction == "std",
              "correction is only used by var/std");
  ImageMapOptions q;
  q.data_range = data_range;
  q.window_size = window;
  q.sigma = sigma;
  q.k1 = k1;
  q.k2 = k2;
  q.chunk_size = size;
  PairwiseOperationOptions o;
  o.chunk_size = size;
  auto p = detail::streaming::prepare_image_event_operator(
      x, y, comparison, program, corrections, o, q);
  std::vector<int64_t> settings;
  for (size_t k = 1; k < 5; ++k)
    settings.push_back(p.settings.integers[k].value());
  std::vector<double> parameters;
  for (size_t k = 0; k < 4; ++k)
    parameters.push_back(p.settings.reals[k].value());
  auto dtype = p.settings.integers[5].value(),
       kind = p.settings.integers[0].value();
  auto metadata = detail::media::event_metadata(kind, settings, dtype, program,
                                                corrections);
  auto rank = metadata.shape.size() + 2;
  auto axes = detail::operator_axes(dims, p.batch.size() + rank, rank);
  return std::get<0>(detail::media::event_reduce(
      p.values[detail::streaming::X], p.values[detail::streaming::Y], kind,
      settings, parameters, dtype, program, corrections, reduction, axes,
      keepdim, correction.value_or(1.), size, false));
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "image_event_reduction(Tensor x, Tensor? y, str comparison, int[] "
      "program, float[] corrections, str reduction, int[] dims, bool "
      "keepdim=False, float? correction=None, float? data_range=None, int "
      "window_size=11, float sigma=1.5, float k1=0.01, float k2=0.03, int "
      "chunk_size=16) -> Tensor",
      TORCH_FN(image_event_reduction_impl));
}
}  // namespace
Tensor image_event_reduce(const Tensor& x, const c10::optional<Tensor>& y,
                          c10::string_view comparison, at::IntArrayRef program,
                          at::ArrayRef<double> corrections,
                          c10::string_view reduction, at::IntArrayRef dims,
                          bool keepdim, c10::optional<double> correction,
                          const ImageMapOptions& q) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::image_event_reduction", "")
          .typed<decltype(image_event_reduction_impl)>();
  return op.call(x, y, comparison, program, corrections, reduction, dims,
                 keepdim, correction, q.data_range, q.window_size, q.sigma,
                 q.k1, q.k2, q.chunk_size);
}

namespace {
detail::media::EventMetadata expression_metadata(
    const Tensor& x, c10::string_view comparison, const ImageMapOptions& q,
    at::IntArrayRef program, at::ArrayRef<double> corrections) {
  TORCH_CHECK(x.dim() >= 4, "Images require [...,N,C,H,W]");
  return detail::media::event_metadata(
      comparison == "ssim" ? 3 : 0,
      {x.size(-3), x.size(-2), x.size(-1), q.window_size},
      int64_t(x.scalar_type() == at::kDouble ? at::kDouble : at::kFloat),
      program, corrections);
}
}  // namespace
ImageMapExpression::ImageMapExpression(Tensor x, c10::optional<Tensor> y,
                                       std::string comparison,
                                       ImageMapOptions options,
                                       std::vector<int64_t> program,
                                       std::vector<double> corrections)
    : x_(std::move(x)),
      y_(std::move(y)),
      comparison_(std::move(comparison)),
      options_(std::move(options)),
      program_(std::move(program)),
      corrections_(std::move(corrections)) {}
ImageMapExpression ImageMapExpression::reduce(
    c10::string_view reduction, at::IntArrayRef dims, bool keepdim,
    c10::optional<double> correction) const {
  TORCH_CHECK(!correction || reduction == "var" || reduction == "std",
              "correction is only used by var/std");
  auto metadata =
      expression_metadata(x_, comparison_, options_, program_, corrections_);
  auto rank = metadata.shape.size();
  auto batch_rank = std::max(x_.dim(), y_ ? y_->dim() : x_.dim()) - 4;
  auto axes = detail::operator_axes(dims, batch_rank + rank + 2, rank + 2);
  int64_t mask = 0, code = -1;
  for (auto axis : axes) {
    TORCH_CHECK(axis >= 2, "Use reduce_samples for terminal N/M reductions");
    mask |= int64_t(1) << (axis - 2);
  }
  const char* names[] = {"sum", "mean", "var", "std", "min",
                         "max", "prod", "all", "any"};
  for (int64_t k = 0; k < 9; ++k)
    if (reduction == names[k])
      code = k;
  TORCH_CHECK(code >= 0 && mask, "Invalid event reduction");
  auto p = program_;
  auto c = corrections_;
  p.insert(p.end(), {code, mask, int64_t(keepdim)});
  c.push_back(correction.value_or(1.));
  expression_metadata(x_, comparison_, options_, p, c);
  return ImageMapExpression(x_, y_, comparison_, options_, std::move(p),
                            std::move(c));
}
ScalarExpression ImageMapExpression::scalar() const {
  TORCH_CHECK(
      expression_metadata(x_, comparison_, options_, program_, corrections_)
          .shape.empty(),
      "scalar() requires every event axis to be contracted");
  TORCH_CHECK(!options_.keepdim,
              "Set keepdim on individual event reduction stages");
  ExpressionSource s;
  s.inputs = {x_};
  if (y_)
    s.inputs.push_back(*y_);
  s.family = "image_event";
  s.metric = "event";
  s.strings.insert("comparison", comparison_);
  if (options_.data_range)
    s.reals.insert("data_range", *options_.data_range);
  s.reals.insert("sigma", options_.sigma);
  s.reals.insert("k1", options_.k1);
  s.reals.insert("k2", options_.k2);
  s.integers.insert("window_size", options_.window_size);
  s.integers.insert("chunk_size", options_.chunk_size);
  s.integer_list = program_;
  s.real_list = corrections_;
  return ScalarExpression(std::move(s));
}
Tensor ImageMapExpression::reduce_samples(
    c10::string_view reduction, at::IntArrayRef dims, bool keepdim,
    c10::optional<double> correction) const {
  return image_event_reduce(x_, y_, comparison_, program_, corrections_,
                            reduction, dims, keepdim, correction, options_);
}
}  // namespace torchpairwise::ops
