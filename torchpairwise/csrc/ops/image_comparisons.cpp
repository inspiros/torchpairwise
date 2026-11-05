#include "image_comparisons.h"

#include <limits>

#include "image_comparisons_detail.h"
#include "media_operations_detail.h"
#include "modality_comparisons_detail.h"
#include "utils/comparison_dispatch.h"

namespace torchpairwise::ops {
namespace {
at::Tensor image_comparisons_composite(const at::Tensor& input,
                                       const c10::optional<at::Tensor>& other,
                                       c10::string_view metric,
                                       c10::string_view mode,
                                       const ImageComparisonOptions& options);
}

namespace {
using namespace modality_detail;
Tensor smooth(const Tensor& image, const Tensor& kernel) {
  auto flat = image.reshape({-1, 1, image.size(-2), image.size(-1)});
  flat = at::conv2d(flat, kernel.reshape({1, 1, 1, -1}));
  flat = at::conv2d(flat, kernel.reshape({1, 1, -1, 1}));
  auto shape = image.sizes().vec();
  shape[shape.size() - 2] = flat.size(-2);
  shape.back() = flat.size(-1);
  return flat.reshape(shape);
}
Tensor ratio(const Tensor& n, const Tensor& d) {
  auto positive = d.gt(0);
  return at::where(positive, n / at::where(positive, d, 1.), 1.);
}
struct Image {
  int64_t kind, window;
  std::vector<double> p;
  Tensor kernel;
  Image(const Tensor& x, int64_t kind, int64_t window,
        at::ArrayRef<double> parameters)
      : kind(kind), window(window), p(parameters.vec()) {
    if (kind == 0)
      return;
    std::vector<double> coefficients;
    for (int64_t i = -window / 2; i <= window / 2; ++i)
      coefficients.push_back(std::abs(double(i)) > 40 * p[1]
                                 ? 0.
                                 : std::exp(-.5 * std::pow(i / p[1], 2)));
    kernel = at::tensor(coefficients, x.options());
    kernel = kernel / kernel.sum();
  }
  Features features(const Tensor& raw) const {
    if (kind == 0)
      return {raw};
    auto scale = raw.detach()
                     .abs()
                     .amax({-1, -2}, true)
                     .clamp_min(std::numeric_limits<double>::min());
    auto image = raw / scale;
    Features result{raw};
    int64_t levels = kind == 2 ? p.size() - 4 : 1;
    for (int64_t level = 0; level < levels; ++level) {
      auto offset = image.mean({-1, -2}, true), centered = image - offset;
      auto cm = smooth(centered, kernel), mean = cm + offset;
      auto variance =
          (smooth(centered.square(), kernel) - cm.square()).clamp_min(0);
      result.insert(result.end(), {centered, cm, mean, variance, scale});
      if (level + 1 < levels) {
        auto pooled = at::avg_pool2d(
            image.reshape({-1, 1, image.size(-2), image.size(-1)}), {2, 2},
            {2, 2});
        auto shape = image.sizes().vec();
        shape[shape.size() - 2] = pooled.size(-2);
        shape.back() = pooled.size(-1);
        image = pooled.reshape(shape);
      }
    }
    return result;
  }
  Pair components(const Features& a, const Features& b,
                  int64_t level = 0) const {
    auto k = 1 + 5 * level;
    auto scale = at::maximum(a[k + 4], b[k + 4]);
    auto ar = (a[k + 4] / scale).detach(), br = (b[k + 4] / scale).detach();
    auto am = a[k + 2] * ar, bm = b[k + 2] * br;
    auto av = a[k + 3] * ar.square(), bv = b[k + 3] * br.square();
    auto covariance =
        (smooth(a[k] * b[k], kernel) - a[k + 1] * b[k + 1]) * ar * br;
    Features constants;
    for (auto parameter : {p[2], p[3]}) {
      auto root = std::log(p[0]) + std::log(parameter) - scale.log();
      constants.push_back((-2 * root.clamp_min(0)).exp());
      constants.push_back((2 * root.clamp_max(0)).exp());
    }
    auto luminance =
        ratio(2 * am * bm * constants[0] + constants[1],
              (am.square() + bm.square()) * constants[0] + constants[1]);
    auto contrast = ratio(2 * covariance * constants[2] + constants[3],
                          (av + bv) * constants[2] + constants[3]);
    return {luminance, contrast};
  }
  Tensor map(
      const Features& a, const Features& b, int64_t comparison,
      const c10::optional<Tensor>& known_identical = c10::nullopt) const {
    if (comparison != 3) {
      auto delta = b[0] - a[0];
      return comparison == 0   ? delta
             : comparison == 1 ? delta.abs()
                               : delta.square();
    }
    auto terms = components(a, b);
    auto value = std::get<0>(terms) * std::get<1>(terms);
    auto identical = known_identical ? *known_identical
                                     : a[0].eq(b[0]).all(-1).all(-1).all(-1);
    return at::where(identical.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 1.,
                     value);
  }
  Tensor score(const Features& a, const Features& b) const {
    if (kind == 0) {
      auto overflow = ~at::isfinite(a[0] - b[0]).all(-1).all(-1).all(-1);
      auto downscale = at::where(overflow, .5, 1.);
      auto factor = downscale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);
      auto delta = a[0] * factor - b[0] * factor;
      auto scale = delta.abs().amax({-3, -2, -1}), positive = scale.gt(0);
      auto safe = at::where(positive, scale, 1.);
      auto mse = (delta / safe.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                     .square()
                     .mean({-3, -2, -1});
      auto value = 20 * std::log10(p[0]) + 20 * downscale.log10() -
                   20 * safe.log10() -
                   10 * at::where(positive, mse, 1.).log10();
      return at::where(positive, value, INFINITY);
    }
    Tensor value;
    Features factors;
    int64_t levels = kind == 2 ? p.size() - 4 : 1;
    for (int64_t level = 0; level < levels; ++level) {
      auto terms = components(a, b, level);
      auto luminance = std::get<0>(terms), contrast = std::get<1>(terms);
      auto ssim = (luminance * contrast).mean({-1, -2});
      if (kind == 1)
        value = ssim.mean(-1);
      else {
        auto factor = level + 1 == levels ? ssim : contrast.mean({-1, -2});
        auto positive = factor.gt(0);
        factors.push_back(at::where(
            positive, at::where(positive, factor, 1.).pow(p[4 + level]), 0.));
      }
    }
    if (kind == 2)
      value = at::stack(factors).prod(0).mean(-1);
    auto identical = a[0].eq(b[0]).all(-1).all(-1).all(-1);
    return at::where(identical, 1., value);
  }
};
Tensor image_compare_kernel(const Tensor& x, const Tensor& y, int64_t kind,
                            int64_t mode, int64_t size, int64_t window,
                            at::ArrayRef<double> p) {
  Image image(x, kind, window, p);
  return assemble(
      x, y, mode, size, 4, [&](const Tensor& t) { return image.features(t); },
      [&](const Features& a, const Features& b) { return image.score(a, b); });
}
Pair image_compare_backward_kernel(const Tensor& grad, const Tensor& x,
                                   const Tensor& y, int64_t kind, int64_t mode,
                                   int64_t size, int64_t window,
                                   at::ArrayRef<double> p) {
  Image image(x, kind, window, p);
  return backward(
      grad, x, y, mode, size, 4,
      [&](const Tensor& t) { return image.features(t); },
      [&](const Features& a, const Features& b) { return image.score(a, b); });
}
using Core = Tensor(const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
                    int64_t, at::ArrayRef<double>);
using Grad = Pair(const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t,
                  int64_t, int64_t, at::ArrayRef<double>);

int64_t kind_for(c10::string_view metric) {
  auto kind = detail::media_comparison_kind(metric);
  TORCH_CHECK(kind >= 0 && kind < 3, "Unsupported image comparison: ", metric);
  return kind;
}
}  // namespace
Tensor image_comparisons(const Tensor& input,
                         const c10::optional<Tensor>& other,
                         c10::string_view metric, c10::string_view mode,
                         const ImageComparisonOptions& options) {
  using Signature = Tensor(
      const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
      c10::string_view mode, double range, int64_t size, int64_t window,
      double sigma, double k1, double k2, at::ArrayRef<double> weights);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::image_comparisons", "")
          .typed<Signature>();
  return op.call(input, other, metric, mode, options.data_range,
                 options.chunk_size, options.window_size, options.sigma,
                 options.k1, options.k2, options.weights);
}
namespace {
Tensor image_comparisons_impl(const Tensor& x, const c10::optional<Tensor>& y,
                              c10::string_view metric, c10::string_view mode,
                              double range, int64_t size, int64_t window,
                              double sigma, double k1, double k2,
                              at::ArrayRef<double> weights) {
  ImageComparisonOptions options;
  options.data_range = range;
  options.chunk_size = size;
  options.window_size = window;
  options.sigma = sigma;
  options.k1 = k1;
  options.k2 = k2;
  options.weights = weights.vec();
  return image_comparisons_composite(x, y, metric, mode, options);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "image_comparisons(Tensor x1, Tensor? x2, str metric, str mode, float "
      "data_range, int chunk_size=16, int window_size=11, float sigma=1.5, "
      "float k1=0.01, float k2=0.03, float[] "
      "weights=[0.0448,0.2856,0.3001,0.2363,0.1333]) -> Tensor",
      TORCH_FN(image_comparisons_impl));
  m.def(
      "_image_compare(Tensor x, Tensor y, int kind, int mode, int size, int "
      "window, float[] parameters) -> Tensor");
  m.def(
      "_image_compare_backward(Tensor grad, Tensor x, Tensor y, int kind, int "
      "mode, int size, int window, float[] parameters) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_image_compare", TORCH_FN(image_compare_kernel));
  m.impl("_image_compare_backward", TORCH_FN(image_compare_backward_kernel));
}

}  // namespace
namespace {
Tensor image_comparisons_composite(const Tensor& input,
                                   const c10::optional<Tensor>& other,
                                   c10::string_view metric,
                                   c10::string_view mode,
                                   const ImageComparisonOptions& options) {
  auto p = detail::media::prepare_image(input, other, metric, mode, options);
  return detail::image::_image_compare(p.x, p.y, p.kind, mode_for(mode),
                                       options.chunk_size, p.settings[0],
                                       p.parameters)
      .reshape_symint(
          output_shape(input, other.value_or(input), 4, mode_for(mode)))
      .to(p.dtype);
}
}  // namespace
namespace {
std::vector<c10::SymInt> image_map_shape(const Tensor& x, const Tensor& y,
                                         int64_t kind, int64_t mode,
                                         int64_t window) {
  auto shape = output_shape(x, y, 4, mode < 3 ? mode : 1);
  shape.push_back(x.sym_size(-3));
  shape.push_back(x.sym_size(-2) - (kind == 3 ? window - 1 : 0));
  shape.push_back(x.sym_size(-1) - (kind == 3 ? window - 1 : 0));
  return shape;
}
Features image_map_tile(const Features& features, int64_t start, int64_t count,
                        int64_t axis) {
  auto result = slice(features, start, count);
  if (axis >= 0)
    for (auto& value : result)
      value = value.unsqueeze(axis);
  return result;
}
Pair image_map_kernel(const Tensor& x, const Tensor& y, int64_t kind,
                      int64_t mode, int64_t size, int64_t window,
                      at::ArrayRef<double> p, bool single_precision) {
  auto batch = batch_shape(x, y, 4);
  int64_t B = 1;
  for (const auto& d : batch)
    B *= d.expect_int();
  auto n = x.size(-4), m = y.size(-4);
  auto shape = image_map_shape(x, y, kind, mode, window);
  std::vector<int64_t> flat{B, mode == 2 ? n * (n - 1) / 2 : n};
  if (mode == 0)
    flat.push_back(m);
  flat.insert(flat.end(),
              {x.size(-3), x.size(-2) - (kind == 3 ? window - 1 : 0),
               x.size(-1) - (kind == 3 ? window - 1 : 0)});
  auto result = at::empty(flat, x.options());
  auto indices = mode >= 5 ? at::zeros(flat, x.options().dtype(at::kLong))
                           : at::empty({0}, x.options().dtype(at::kLong));
  if (mode >= 3)
    result.fill_(mode == 5 ? INFINITY : mode == 6 ? -INFINITY : 0.);
  if (B == 0 || n == 0 || m == 0 || (mode == 2 && n < 2)) {
    if (mode == 4 && m == 0)
      result.fill_(NAN);
    return {result.reshape_symint(shape),
            mode >= 5 ? indices.reshape_symint(shape) : indices};
  }
  Image image(x, kind == 3 ? 1 : 0, window, p);
  auto a = image.features(x), b = x.is_same(y) ? a : image.features(y);
  for (auto& t : a)
    t = normalize(t, batch, 4);
  for (auto& t : b)
    t = normalize(t, batch, 4);
  tiles(
      n, m, mode < 3 ? mode : 0, size,
      [&](int64_t i, int64_t j, int64_t count, int64_t other, int64_t offset) {
        auto value =
            image.map(image_map_tile(a, i, count, mode == 1 ? -1 : 2),
                      image_map_tile(b, j, other, mode == 1 ? -1 : 1), kind);
        // Compare in the public output dtype so ties match materialized maps.
        if (mode >= 5 && single_precision)
          value = value.to(at::kFloat).to(at::kDouble);
        if (mode == 0)
          result.select(1, i).narrow(1, j, other).copy_(value.squeeze(1));
        else if (mode == 1)
          result.narrow(1, i, count).copy_(value);
        else if (mode == 2)
          result.narrow(1, offset, other).copy_(value.squeeze(1));
        else if (mode < 5)
          result.narrow(1, i, count).add_(value.sum(2));
        else {
          auto local = mode == 5 ? value.min(2) : value.max(2);
          auto candidate = std::get<0>(local);
          auto current = result.narrow(1, i, count);
          auto replace =
              (mode == 5 ? candidate.lt(current) : candidate.gt(current)) |
              (candidate.isnan() & ~current.isnan());
          // The first tile initializes winners even when every entry is +/-inf.
          if (j == 0)
            replace = at::ones_like(replace);
          current.copy_(at::where(replace, candidate, current));
          auto winner = indices.narrow(1, i, count);
          winner.copy_(at::where(replace, std::get<1>(local) + j, winner));
        }
      });
  if (mode == 4)
    result.div_(m);
  return {result.reshape_symint(shape),
          mode >= 5 ? indices.reshape_symint(shape) : indices};
}
Pair image_map_backward_kernel(const Tensor& gradient, const Tensor& x,
                               const Tensor& y, const Tensor& indices,
                               int64_t kind, int64_t mode, int64_t size,
                               int64_t window, at::ArrayRef<double> p) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  if (gradient.numel() == 0 || y.size(-4) == 0)
    return {at::zeros_like(x).contiguous(), at::zeros_like(y).contiguous()};
  at::AutoGradMode enable(true);
  auto ax = x.detach().set_requires_grad(true),
       by = y.detach().set_requires_grad(true);
  Image image(x, kind == 3 ? 1 : 0, window, p);
  auto a = image.features(ax), b = image.features(by);
  auto batch = batch_shape(x, y, 4);
  int64_t B = 1;
  for (const auto& d : batch)
    B *= d.expect_int();
  for (auto& t : a)
    t = normalize(t, batch, 4);
  for (auto& t : b)
    t = normalize(t, batch, 4);
  auto flat = gradient.reshape(
      {B, -1, gradient.size(-3), gradient.size(-2), gradient.size(-1)});
  if (mode == 0)
    flat = flat.reshape({B, x.size(-4), y.size(-4), gradient.size(-3),
                         gradient.size(-2), gradient.size(-1)});
  auto winners = mode >= 5 ? indices.reshape_as(flat) : indices;
  Features ga, gb;
  for (const auto& t : a)
    ga.push_back(at::zeros_like(t));
  for (const auto& t : b)
    gb.push_back(at::zeros_like(t));
  tiles(
      x.size(-4), y.size(-4), mode < 3 ? mode : 0, size,
      [&](int64_t i, int64_t j, int64_t count, int64_t other, int64_t offset) {
        Features al, bl, leaves;
        for (const auto& t : a) {
          al.push_back(t.narrow(1, i, count).detach().set_requires_grad(true));
          leaves.push_back(al.back());
        }
        for (const auto& t : b) {
          bl.push_back(t.narrow(1, j, other).detach().set_requires_grad(true));
          leaves.push_back(bl.back());
        }
        auto av = al, bv = bl;
        if (mode != 1) {
          for (auto& t : av)
            t = t.unsqueeze(2);
          for (auto& t : bv)
            t = t.unsqueeze(1);
        }
        auto value = image.map(av, bv, kind);
        Tensor g;
        if (mode == 0)
          g = flat.narrow(1, i, count).narrow(2, j, other);
        else if (mode == 1)
          g = flat.narrow(1, i, count);
        else if (mode == 2)
          g = flat.narrow(1, offset, other).unsqueeze(1);
        else {
          g = flat.narrow(1, i, count).unsqueeze(2).expand_as(value);
          if (mode == 4)
            g = g / y.size(-4);
          if (mode >= 5) {
            auto positions =
                at::arange(j, j + other, x.options().dtype(at::kLong))
                    .reshape({1, 1, other, 1, 1, 1});
            g = at::where(
                winners.narrow(1, i, count).unsqueeze(2).eq(positions), g, 0.);
          }
        }
        auto grads =
            torch::autograd::grad({value}, leaves, {g}, false, false, true);
        at::NoGradGuard guard;
        for (size_t k = 0; k < a.size(); ++k)
          if (grads[k].defined())
            ga[k].narrow(1, i, count).add_(grads[k]);
        for (size_t k = 0; k < b.size(); ++k)
          if (grads[a.size() + k].defined())
            gb[k].narrow(1, j, other).add_(grads[a.size() + k]);
      });
  Features outputs, grads;
  for (size_t k = 0; k < a.size(); ++k)
    if (a[k].requires_grad()) {
      outputs.push_back(a[k]);
      grads.push_back(ga[k]);
    }
  for (size_t k = 0; k < b.size(); ++k)
    if (b[k].requires_grad()) {
      outputs.push_back(b[k]);
      grads.push_back(gb[k]);
    }
  auto g = torch::autograd::grad(outputs, {ax, by}, grads, false, false, true);
  return {(g[0].defined() ? g[0] : at::zeros_like(x)).contiguous(),
          (g[1].defined() ? g[1] : at::zeros_like(y)).contiguous()};
}
Tensor image_maps_impl(const Tensor& x, const c10::optional<Tensor>& other,
                       c10::string_view comparison, c10::string_view mode,
                       c10::string_view reduction, c10::optional<double> range,
                       int64_t size, int64_t window, double sigma, double k1,
                       double k2, bool keepdim) {
  int64_t kind = comparison == "residual"         ? 0
                 : comparison == "absolute_error" ? 1
                 : comparison == "squared_error"  ? 2
                 : comparison == "ssim"           ? 3
                                                  : -1;
  TORCH_CHECK(kind >= 0, "Unknown image map comparison");
  auto pairing = mode_for(mode);
  TORCH_CHECK(pairing != 1 || other, "Paired image maps require two inputs");
  TORCH_CHECK(pairing != 2 || !other, "Condensed image maps require one input");
  TORCH_CHECK(pairing != 2 || kind != 0,
              "Signed residual maps are directed and cannot be condensed");
  auto op = pairing;
  if (reduction != "none") {
    TORCH_CHECK(pairing == 0, "Reference reductions require all_pairs mode");
    op = reduction == "sum"    ? 3
         : reduction == "mean" ? 4
         : reduction == "min"  ? 5
         : reduction == "max"  ? 6
                               : -1;
    TORCH_CHECK(op >= 0, "Image maps support sum, mean, min or max reduction");
  } else
    TORCH_CHECK(!keepdim, "keepdim requires a reduction");
  auto y = other.value_or(x);
  validate(x, y, 4, pairing, size);
  TORCH_CHECK(op < 5 || y.sym_size(-4) > 0,
              "Cannot reduce an empty reference axis with min/max");
  if (kind == 3) {
    TORCH_CHECK(range, "SSIM maps require data_range");
    positive(*range, "data_range");
    TORCH_CHECK(window > 0 && window % 2 == 1,
                "window_size must be positive and odd");
    positive(sigma, "sigma");
    positive(k1, "k1");
    positive(k2, "k2");
    TORCH_CHECK(x.sym_size(-2) >= window && x.sym_size(-1) >= window,
                "Image H/W are too small for the window");
  } else
    TORCH_CHECK(
        !range && window == 11 && sigma == 1.5 && k1 == .01 && k2 == .03,
        "SSIM options require comparison=ssim");
  auto dtype = x.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto a = x.to(at::kDouble), b = x.is_same(y) ? a : y.to(at::kDouble);
  auto value =
      std::get<0>(detail::image::image_map(a, b, kind, op, size, window,
                                           {range.value_or(1.), sigma, k1, k2},
                                           dtype == at::kFloat))
          .to(dtype);
  return keepdim ? value.unsqueeze(-4) : value;
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "image_maps(Tensor x1, Tensor? x2=None, str comparison='squared_error', "
      "str mode='all_pairs', str reduction='none', float? data_range=None, int "
      "chunk_size=16, int window_size=11, float sigma=1.5, float k1=0.01, "
      "float k2=0.03, bool keepdim=False) -> Tensor",
      TORCH_FN(image_maps_impl));
  m.def(
      "_image_map(Tensor x, Tensor y, int kind, int mode, int size, int "
      "window, float[] parameters, bool single_precision=False) -> (Tensor, "
      "Tensor)");
  m.def(
      "_image_map_backward(Tensor grad, Tensor x, Tensor y, Tensor indices, "
      "int kind, int mode, int size, int window, float[] parameters) -> "
      "(Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_image_map", TORCH_FN(image_map_kernel));
  m.impl("_image_map_backward", TORCH_FN(image_map_backward_kernel));
}
}  // namespace
Tensor image_maps(const Tensor& x, const c10::optional<Tensor>& y,
                  c10::string_view comparison, c10::string_view mode,
                  c10::string_view reduction, const ImageMapOptions& o) {
  using Signature =
      Tensor(const Tensor&, const c10::optional<Tensor>&, c10::string_view,
             c10::string_view, c10::string_view, c10::optional<double>, int64_t,
             int64_t, double, double, double, bool);
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::image_maps", "")
                       .typed<Signature>();
  return op.call(x, y, comparison, mode, reduction, o.data_range, o.chunk_size,
                 o.window_size, o.sigma, o.k1, o.k2, o.keepdim);
}
}  // namespace torchpairwise::ops

namespace torchpairwise::ops::detail::media {
using namespace modality_detail;
PreparedInputs prepare_image(const Tensor& input,
                             const c10::optional<Tensor>& other,
                             c10::string_view metric, c10::string_view mode,
                             const ImageComparisonOptions& options) {
  auto kind = kind_for(metric), pairing = mode_for(mode);
  auto y = other.value_or(input);
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(pairing != 2 || !other.has_value(),
              "Condensed comparisons require one input");
  validate(input, y, 4, pairing, options.chunk_size);
  positive(options.data_range, "data_range");
  if (kind != 0) {
    TORCH_CHECK(options.window_size > 0 && options.window_size % 2 == 1,
                "window_size must be positive and odd");
    positive(options.sigma, "sigma");
    positive(options.k1, "k1");
    positive(options.k2, "k2");
    auto minimum = options.window_size;
    if (kind == 2) {
      TORCH_CHECK(!options.weights.empty(), "weights must be non-empty");
      for (auto weight : options.weights)
        positive(weight, "weights");
      for (size_t i = 1; i < options.weights.size(); ++i) {
        TORCH_CHECK(minimum <= std::numeric_limits<int64_t>::max() / 2,
                    "Image H/W are too small for the window and scale count");
        minimum *= 2;
      }
    }
    TORCH_CHECK(input.sym_size(-2) >= minimum && input.sym_size(-1) >= minimum,
                "Image H/W are too small for the window and scale count");
  }
  auto dtype = input.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda(c10::DispatchKey::AutocastCUDA);
  auto x = input.to(kind == 0 ? dtype : at::kDouble);
  y = input.is_same(y) ? x : y.to(kind == 0 ? dtype : at::kDouble);
  std::vector<double> p{options.data_range, options.sigma, options.k1,
                        options.k2};
  if (kind == 2)
    p.insert(p.end(), options.weights.begin(), options.weights.end());
  return {x, y, kind, 4, dtype, {options.window_size}, std::move(p)};
}

Features image_features(const Tensor& x, int64_t kind, int64_t window,
                        at::ArrayRef<double> parameters) {
  return Image(x, kind, window, parameters).features(x);
}
Tensor image_score(const Features& a, const Features& b, int64_t kind,
                   int64_t window, at::ArrayRef<double> parameters) {
  return Image(a[0], kind, window, parameters).score(a, b);
}
Tensor image_map(const Features& a, const Features& b, int64_t comparison,
                 int64_t window, at::ArrayRef<double> parameters) {
  return Image(a[0], comparison == 3 ? 1 : 0, window, parameters)
      .map(a, b, comparison);
}
Tensor image_map_with_identity(const Features& a, const Features& b,
                               int64_t comparison, int64_t window,
                               at::ArrayRef<double> parameters,
                               const Tensor& identical) {
  return Image(a[0], comparison == 3 ? 1 : 0, window, parameters)
      .map(a, b, comparison, identical);
}
}  // namespace torchpairwise::ops::detail::media
