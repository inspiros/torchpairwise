#include "vector_comparisons.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "builtin_comparisons.h"
#include "ppminkowski.h"
#include "utils/tiling.h"
#include "utils/value_checks.h"
#include "vector_comparisons_detail.h"

namespace torchpairwise::ops {
namespace {
at::Tensor angular_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2);
at::Tensor paired_angular_distances_impl(const at::Tensor& x1,
                                         const at::Tensor& x2);
at::Tensor condensed_angular_distances_impl(const at::Tensor& x1);
at::Tensor lorentzian_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2);
at::Tensor paired_lorentzian_distances_impl(const at::Tensor& x1,
                                            const at::Tensor& x2);
at::Tensor condensed_lorentzian_distances_impl(const at::Tensor& x1);
at::Tensor clark_distances_impl(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2);
at::Tensor paired_clark_distances_impl(const at::Tensor& x1,
                                       const at::Tensor& x2);
at::Tensor condensed_clark_distances_impl(const at::Tensor& x1);
at::Tensor soergel_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2);
at::Tensor paired_soergel_distances_impl(const at::Tensor& x1,
                                         const at::Tensor& x2);
at::Tensor condensed_soergel_distances_impl(const at::Tensor& x1);
at::Tensor ruzicka_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2);
at::Tensor paired_ruzicka_distances_impl(const at::Tensor& x1,
                                         const at::Tensor& x2);
at::Tensor condensed_ruzicka_distances_impl(const at::Tensor& x1);
at::Tensor bhattacharyya_distances_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2);
at::Tensor paired_bhattacharyya_distances_impl(const at::Tensor& x1,
                                               const at::Tensor& x2);
at::Tensor condensed_bhattacharyya_distances_impl(const at::Tensor& x1);
at::Tensor bhattacharyya_kernel_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2);
at::Tensor paired_bhattacharyya_kernel_impl(const at::Tensor& x1,
                                            const at::Tensor& x2);
at::Tensor condensed_bhattacharyya_kernel_impl(const at::Tensor& x1);
at::Tensor aitchison_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2);
at::Tensor paired_aitchison_distances_impl(const at::Tensor& x1,
                                           const at::Tensor& x2);
at::Tensor condensed_aitchison_distances_impl(const at::Tensor& x1);
at::Tensor hellinger_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2);
at::Tensor paired_hellinger_distances_impl(const at::Tensor& x1,
                                           const at::Tensor& x2);
at::Tensor condensed_hellinger_distances_impl(const at::Tensor& x1);
at::Tensor histogram_intersection_kernel_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2);
at::Tensor paired_histogram_intersection_kernel_impl(const at::Tensor& x1,
                                                     const at::Tensor& x2);
at::Tensor condensed_histogram_intersection_kernel_impl(const at::Tensor& x1);
at::Tensor total_variation_distances_impl(const at::Tensor& x1,
                                          const c10::optional<at::Tensor>& x2);
at::Tensor paired_total_variation_distances_impl(const at::Tensor& x1,
                                                 const at::Tensor& x2);
at::Tensor condensed_total_variation_distances_impl(const at::Tensor& x1);
at::Tensor fisher_rao_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2);
at::Tensor paired_fisher_rao_distances_impl(const at::Tensor& x1,
                                            const at::Tensor& x2);
at::Tensor condensed_fisher_rao_distances_impl(const at::Tensor& x1);
at::Tensor pearson_chi2_divergences_impl(const at::Tensor& x1,
                                         const c10::optional<at::Tensor>& x2);
at::Tensor paired_pearson_chi2_divergences_impl(const at::Tensor& x1,
                                                const at::Tensor& x2);
at::Tensor neyman_chi2_divergences_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2);
at::Tensor paired_neyman_chi2_divergences_impl(const at::Tensor& x1,
                                               const at::Tensor& x2);
at::Tensor triangular_discrimination_distances_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2);
at::Tensor paired_triangular_discrimination_distances_impl(
    const at::Tensor& x1, const at::Tensor& x2);
at::Tensor condensed_triangular_discrimination_distances_impl(
    const at::Tensor& x1);
at::Tensor itakura_saito_divergences_impl(const at::Tensor& x1,
                                          const c10::optional<at::Tensor>& x2);
at::Tensor paired_itakura_saito_divergences_impl(const at::Tensor& x1,
                                                 const at::Tensor& x2);
at::Tensor kl_divergences_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2,
                               c10::optional<double> base);
at::Tensor paired_kl_divergences_impl(const at::Tensor& x1,
                                      const at::Tensor& x2,
                                      c10::optional<double> base);
at::Tensor symmetric_kl_divergences_impl(const at::Tensor& x1,
                                         const c10::optional<at::Tensor>& x2,
                                         c10::optional<double> base);
at::Tensor paired_symmetric_kl_divergences_impl(const at::Tensor& x1,
                                                const at::Tensor& x2,
                                                c10::optional<double> base);
at::Tensor condensed_symmetric_kl_divergences_impl(const at::Tensor& x1,
                                                   c10::optional<double> base);
at::Tensor renyi_divergences_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2,
                                  double alpha);
at::Tensor paired_renyi_divergences_impl(const at::Tensor& x1,
                                         const at::Tensor& x2, double alpha);
at::Tensor rational_quadratic_kernel_impl(const at::Tensor& x1,
                                          const c10::optional<at::Tensor>& x2,
                                          double length_scale, double alpha);
at::Tensor paired_rational_quadratic_kernel_impl(const at::Tensor& x1,
                                                 const at::Tensor& x2,
                                                 double length_scale,
                                                 double alpha);
at::Tensor condensed_rational_quadratic_kernel_impl(const at::Tensor& x1,
                                                    double length_scale,
                                                    double alpha);
at::Tensor matern_kernel_impl(const at::Tensor& x1,
                              const c10::optional<at::Tensor>& x2,
                              double length_scale, double nu);
at::Tensor paired_matern_kernel_impl(const at::Tensor& x1, const at::Tensor& x2,
                                     double length_scale, double nu);
at::Tensor condensed_matern_kernel_impl(const at::Tensor& x1,
                                        double length_scale, double nu);
at::Tensor nan_euclidean_distances_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2,
                                        bool squared);
at::Tensor paired_nan_euclidean_distances_impl(const at::Tensor& x1,
                                               const at::Tensor& x2,
                                               bool squared);
at::Tensor condensed_nan_euclidean_distances_impl(const at::Tensor& x1,
                                                  bool squared);
at::Tensor wasserstein_1d_distances_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    const c10::optional<at::Tensor>& bin_positions);
at::Tensor paired_wasserstein_1d_distances_impl(
    const at::Tensor& x1, const at::Tensor& x2,
    const c10::optional<at::Tensor>& bin_positions);
at::Tensor condensed_wasserstein_1d_distances_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& bin_positions);
at::Tensor vector_comparisons_composite(const at::Tensor& input,
                                        const c10::optional<at::Tensor>& other,
                                        c10::string_view metric,
                                        c10::string_view mode,
                                        const VectorComparisonOptions& opts);
}  // namespace

namespace {
using at::Tensor;
using Triple = std::array<Tensor, 3>;
using Values = torch::autograd::variable_list;
enum Kind : int64_t {
  Angular,
  Lorentzian,
  Clark,
  Ruzicka,
  KL,
  Jeffreys,
  Bhattacharyya,
  BhattacharyyaKernel,
  Aitchison,
  Hellinger,
  Wasserstein1D,
  Intersection,
  NanEuclidean,
  RationalQuadratic,
  Matern,
  TotalVariation,
  FisherRao,
  Renyi,
  Pearson,
  Neyman,
  Triangular,
  ItakuraSaito
};
constexpr int64_t feature_tile = 64;

Tensor ratio(const Tensor& a, const Tensor& b) {
  auto valid = b.ne(0);
  return at::where(valid, a / at::where(valid, b, 1.), 0.);
}
Tensor safe_sqrt(const Tensor& x) {
  auto valid = x.ne(0);
  return at::where(valid, at::where(valid, x, 1.).sqrt(), 0.);
}
Tensor stable_norm(const Tensor& x) {
  auto scale = x.abs().amax(-1).detach();
  return at::linalg_vector_norm(
             x / at::where(scale.gt(0), scale, 1.).unsqueeze(-1), 2, {-1},
             false) *
         scale;
}
Tensor safe_hypot(const Tensor& a, const Tensor& b) {
  auto valid = a.ne(0) | b.ne(0);
  return at::where(valid, at::hypot(at::where(valid, a, 1.), b), 0.);
}
Tensor relative_entropy(const Tensor& p, const Tensor& q, const Tensor& lp,
                        const Tensor& lq) {
  auto diff = p - q;
  auto near = q.gt(0) & diff.abs().lt(.001 * q);
  auto t = at::where(near, diff, 0.) / at::where(near, q, 1.);
  auto local = q * t.square() *
               (.5 + t * (-1. / 6 + t * (1. / 12 + t * (-1. / 20 + t / 30))));
  return at::where(near, local, p * (lp - lq) - diff);
}
Tensor expm1_remainder(const Tensor& x) {
  auto local_x = at::where(x.abs().lt(.001), x, 0.);
  auto local =
      local_x.square() *
      (.5 +
       local_x * (1. / 6 +
                  local_x * (1. / 24 + local_x * (1. / 120 + local_x / 720))));
  return at::where(x.abs().lt(.001), local, x.expm1() - x);
}
Triple probability(const Tensor& raw) {
  auto x = raw.scalar_type() == at::kFloat ? raw.to(at::kDouble) : raw;
  auto support = x.gt(0);
  x = at::where(support, x, 0.);
  auto mass = x.sum(-1, true);
  return {x / mass,
          at::where(support, at::where(support, x, 1.).log() - mass.log(), 0.),
          support};
}
Tensor root_probability(const Tensor& raw, bool stable) {
  auto x =
      stable && raw.scalar_type() == at::kFloat ? raw.to(at::kDouble) : raw;
  auto positive = x.gt(0);
  if (stable) {
    x = at::where(positive, x, 0.);
    return at::where(positive, at::where(positive, x, 1.).sqrt(), 0.) /
           x.sum(-1, true).sqrt();
  }
  auto p = x / x.sum(-1, true);
  return at::where(p.ne(0), at::where(p.ne(0), p, 1.).sqrt(), 0.);
}

Values feature(Triple a, Triple b, const Tensor& scale, int64_t kind,
               bool paired, double alpha, double limit) {
  if (!paired) {
    for (auto& t : a)
      if (t.defined() && t.numel() != 0)
        t = t.unsqueeze(-2);
    for (auto& t : b)
      if (t.defined() && t.numel() != 0)
        t = t.unsqueeze(-3);
  }
  auto x = a[0], y = b[0];
  if (kind == Angular)
    return {stable_norm(x - y), stable_norm(x + y)};
  if (kind == Lorentzian) {
    auto large = x.abs().gt(std::numeric_limits<double>::max() / 4) |
                 y.abs().gt(std::numeric_limits<double>::max() / 4);
    auto ordinary =
        (at::where(large, 0., x) - at::where(large, 0., y)).abs().log1p();
    auto half =
        (at::where(large, x, 0.) / 2 - at::where(large, y, 0.) / 2).abs();
    return {
        at::where(large, half.log1p() + (half / (1 + half)).log1p(), ordinary)
            .sum(-1)};
  }
  if (kind == Clark) {
    auto maximum = at::maximum(x, y).detach();
    auto divisor = at::where(maximum.gt(0), maximum, 1.);
    x = x / divisor;
    y = y / divisor;
    return {stable_norm(ratio((x - y).abs(), x + y))};
  }
  if (kind == Ruzicka) {
    x = x / scale.unsqueeze(-1);
    y = y / scale.unsqueeze(-1);
    return {(x - y).abs().sum(-1), at::maximum(x, y).sum(-1)};
  }
  if (kind == Intersection)
    return {at::minimum(x, y).sum(-1)};
  if (kind == Wasserstein1D)
    return {((x - y).abs() * scale).sum(-1)};
  if (kind == NanEuclidean) {
    auto valid = ~x.isnan() & ~y.isnan();
    auto diff = at::where(valid, x, 0.) - at::where(valid, y, 0.);
    return {diff.square().sum(-1), valid.sum(-1)};
  }
  if (kind == KL || kind == Jeffreys) {
    auto value = relative_entropy(x, y, a[1], b[1]);
    auto infinite = a[2] & ~b[2];
    if (kind == Jeffreys) {
      value = value + relative_entropy(y, x, b[1], a[1]);
      infinite = infinite | (b[2] & ~a[2]);
    }
    return {value.sum(-1), infinite.any(-1)};
  }
  if (kind == FisherRao)
    return {(x - y).square().sum(-1),
            at::zeros_like((x - y).sum(-1), at::kBool)};
  if (kind == ItakuraSaito) {
    auto diff = x - y;
    auto near = diff.abs().lt(.001 * y);
    auto t = at::where(near, diff, 0.) / at::where(near, y, 1.);
    auto series =
        t.square() * (.5 + t * (-1. / 3 + t * (.25 + t * (-.2 + t / 6))));
    auto log_ratio = x.log() - y.log();
    return {at::where(near, series, expm1_remainder(log_ratio.clamp_max(limit)))
                .sum(-1),
            log_ratio.gt(limit).any(-1)};
  }
  if (kind == Renyi) {
    double beta = alpha - 1;
    auto overlap = a[2] & b[2];
    auto log_ratio = a[1] - b[1];
    Tensor ordinary, log_weight;
    if (alpha >= 2) {
      ordinary = at::where(overlap, log_ratio, -1.e6).amax(-1);
      auto shift =
          at::where(overlap, log_ratio - ordinary.unsqueeze(-1), 0.) * beta;
      log_weight = at::logsumexp(at::where(overlap, a[1] + shift, -1.e6), -1);
    } else {
      ordinary = at::logsumexp(
          at::where(overlap, alpha * a[1] + (1 - alpha) * b[1], -1.e6), -1);
      log_weight = at::zeros_like(ordinary);
    }
    auto t = at::where(overlap, log_ratio, 0.) * beta;
    auto correction =
        (x * expm1_remainder(at::where(t.abs().lt(.1), t, 0.))).sum(-1) / beta;
    auto kl = relative_entropy(x, y, a[1], b[1]).sum(-1);
    return {ordinary,
            log_weight,
            kl + correction,
            t.abs().amax(-1),
            (a[2] & ~b[2]).any(-1),
            overlap.any(-1)};
  }
  if (kind == TotalVariation)
    return {.5 * (x - y).abs().sum(-1),
            at::zeros_like((x - y).sum(-1), at::kBool)};
  if (kind == Triangular)
    return {((x - y) * ratio(x - y, x + y)).sum(-1),
            at::zeros_like((x - y).sum(-1), at::kBool)};
  if (kind == Neyman) {
    std::swap(a, b);
    x = a[0];
    y = b[0];
  }
  auto difference = (x - y).abs();
  auto direct = difference.gt(0);
  auto delta = (a[1] - b[1]).abs();
  auto unequal = delta.gt(0);
  auto gap = -(-delta).expm1();
  auto logarithmic =
      at::maximum(a[1], b[1]) + at::where(unequal, gap, 1.).log();
  logarithmic = at::where(a[2] & ~b[2], a[1], logarithmic);
  logarithmic = at::where(b[2] & ~a[2], b[1], logarithmic);
  auto nonzero = direct | (a[2] != b[2]) | (a[2] & b[2] & unequal);
  auto log_difference =
      at::where(direct, at::where(direct, difference, 1.).log(), logarithmic);
  auto log_value = 2 * log_difference - b[1];
  auto active = nonzero & b[2];
  auto value = at::where(
      active, at::where(active, log_value, 0.).clamp_max(limit).exp(), 0.);
  return {value.sum(-1),
          ((a[2] & ~b[2]) | (active & log_value.gt(limit))).any(-1)};
}

// Native non-reentrant feature replay: retain views, not pair-feature buffers.

Triple slice_features(Triple values, int64_t start, int64_t length) {
  for (auto& t : values)
    if (t.defined() && t.numel() != 0)
      t = t.narrow(-1, start, length);
  return values;
}
Tensor block(const Tensor& raw_x, const Tensor& raw_y, const Tensor& grid,
             int64_t kind, bool paired, double base, double length_scale,
             double alpha, double nu, bool squared) {
  if (kind == Renyi && alpha == 1)
    kind = KL;
  Tensor x = raw_x, y = raw_y;
  const auto dtype = x.scalar_type();
  if (kind <= Aitchison || kind == FisherRao || kind >= Renyi ||
      kind == TotalVariation)
    if (dtype == at::kFloat) {
      x = x.to(at::kDouble);
      y = y.to(at::kDouble);
    }
  // broadcast() has already aligned batch dimensions and dtypes. Flatten
  // multiple leading batch axes for the native 2D/3D distance primitive.
  auto all_pairs_squared_distance = [](const Tensor& a, const Tensor& b) {
    auto shape = a.sizes().vec();
    shape.back() = b.size(-2);
    return _ppminkowski(a.dim() > 3 ? a.flatten(0, -3) : a,
                        b.dim() > 3 ? b.flatten(0, -3) : b, 2)
        .reshape(shape);
  };
  if (kind == RationalQuadratic) {
    auto distance =
        paired ? (x - y).square().sum(-1) : all_pairs_squared_distance(x, y);
    return (-alpha *
            (distance / length_scale / length_scale * .5 / alpha).log1p())
        .exp();
  }
  if (kind == Matern) {
    Tensor distance;
    if (paired) {
      distance = at::linalg_vector_norm(x - y, 2, {-1}, false);
    } else {
      auto value = all_pairs_squared_distance(x, y);
      auto nonzero = value.ne(0);
      // Choose cdist's zero subgradient without evaluating sqrt'(0).
      distance = at::where(nonzero, at::where(nonzero, value, 1.).sqrt(), 0.);
    }
    auto tiny = dtype == at::kFloat ? std::numeric_limits<float>::min()
                                    : std::numeric_limits<double>::min();
    auto r = (distance / length_scale * std::sqrt(2 * nu))
                 .clamp_max(-4 * std::log(tiny));
    if (nu == .5)
      return (-r).exp();
    auto polynomial = 1 + r;
    if (nu == 2.5)
      polynomial = polynomial + r.square() / 3;
    return polynomial * (-r).exp();
  }
  if (kind == Bhattacharyya || kind == BhattacharyyaKernel ||
      kind == Aitchison || kind == Hellinger) {
    Tensor a, b;
    if (kind == Aitchison) {
      a = x.log();
      b = y.log();
      a = a - a.mean(-1, true);
      b = b - b.mean(-1, true);
    } else {
      a = root_probability(x, kind != Hellinger);
      b = root_probability(y, kind != Hellinger);
    }
    if (kind == Aitchison || kind == Hellinger) {
      // cdist's zero subgradient, without its pair-by-feature backward
      // buffer.
      auto dist = paired ? at::linalg_vector_norm(a - b, 2, {-1}, false)
                         : safe_sqrt(all_pairs_squared_distance(a, b));
      return (dist / (kind == Hellinger ? std::sqrt(2.) : 1.)).to(dtype);
    }
    auto coefficient =
        (paired ? (a * b).sum(-1) : a.matmul(b.transpose(-1, -2))).clamp(0, 1);
    if (kind == BhattacharyyaKernel)
      return coefficient.to(dtype);
    auto half_squared = .5 * (paired ? (a - b).square().sum(-1)
                                     : all_pairs_squared_distance(a, b));
    auto near = half_squared.lt(.5), overlap = coefficient.gt(0);
    auto local = -(-at::where(near, half_squared, 0.)).log1p();
    auto ordinary = -at::where(overlap, coefficient, 1.).log();
    return at::where(overlap, at::where(near, local, ordinary), INFINITY)
        .to(dtype);
  }
  // C++ autograd input metadata requires defined tensors, even for unused
  // feature-state slots. Empty device-local operands carry no saved data.
  auto empty = at::empty({0}, x.options());
  Triple a{x, empty, empty}, b{y, empty, empty};
  Tensor scale = at::ones({}, x.options());
  if (kind == Angular) {
    auto sx = x.abs().amax(-1, true).detach(),
         sy = y.abs().amax(-1, true).detach();
    x = x / sx;
    y = y / sy;
    a[0] = x / at::linalg_vector_norm(x, 2, {-1}, true);
    b[0] = y / at::linalg_vector_norm(y, 2, {-1}, true);
  } else if (kind == Ruzicka) {
    auto sx = x.amax(-1).detach(), sy = y.amax(-1).detach();
    scale = paired ? at::maximum(sx, sy)
                   : at::maximum(sx.unsqueeze(-1), sy.unsqueeze(-2));
    scale = at::where(scale.gt(0), scale, 1.);
  } else if (kind == Wasserstein1D) {
    auto d = x.size(-1);
    a[0] = (x / x.sum(-1, true)).cumsum(-1).narrow(-1, 0, d - 1);
    b[0] = (y / y.sum(-1, true)).cumsum(-1).narrow(-1, 0, d - 1);
    scale = grid.numel() ? grid.diff() : at::ones({d - 1}, x.options());
    if (d == 1) {
      auto za = a[0].sum(-1), zb = b[0].sum(-1);
      return (paired ? za + zb : za.unsqueeze(-1) + zb.unsqueeze(-2)) +
             scale.sum();
    }
  } else if (kind == FisherRao) {
    a[0] = root_probability(x, true);
    b[0] = root_probability(y, true);
  } else if (kind == KL || kind == Jeffreys ||
             (kind >= TotalVariation && kind != ItakuraSaito)) {
    a = probability(x);
    b = probability(y);
  }
  double limit = std::log(std::numeric_limits<double>::max()) - 1.e-12;
  Values accumulated;
  for (int64_t k = 0; k < a[0].size(-1); k += feature_tile) {
    auto count = std::min(feature_tile, a[0].size(-1) - k);
    auto left = slice_features(a, k, count),
         right = slice_features(b, k, count);
    auto factor = kind == Wasserstein1D ? scale.narrow(-1, k, count) : scale;
    Values tile;
    if (at::GradMode::is_enabled() &&
        (a[0].requires_grad() || b[0].requires_grad() ||
         factor.requires_grad()))
      tile = detail::vector::feature_autograd(
          left[0], left[1], left[2], right[0], right[1], right[2], factor, kind,
          paired, alpha, limit);
    else
      tile = feature(left, right, factor, kind, paired, alpha, limit);
    if (accumulated.empty()) {
      accumulated = tile;
      continue;
    }
    if (kind == Renyi) {
      if (alpha >= 2) {
        auto maximum = at::maximum(accumulated[0], tile[0]);
        accumulated[1] = at::logaddexp(
            accumulated[1] + (accumulated[0] - maximum) * (alpha - 1),
            tile[1] + (tile[0] - maximum) * (alpha - 1));
        accumulated[0] = maximum;
      } else
        accumulated[0] = at::logaddexp(accumulated[0], tile[0]);
      accumulated[2] = accumulated[2] + tile[2];
      accumulated[3] = at::maximum(accumulated[3], tile[3]);
      accumulated[4] = accumulated[4] | tile[4];
      accumulated[5] = accumulated[5] | tile[5];
    } else {
      for (size_t q = 0; q < tile.size(); ++q) {
        if (tile[q].scalar_type() == at::kBool)
          accumulated[q] = accumulated[q] | tile[q];
        else if (kind == Angular || kind == Clark)
          accumulated[q] = safe_hypot(accumulated[q], tile[q]);
        else
          accumulated[q] = accumulated[q] + tile[q];
      }
    }
  }
  auto value = accumulated[0];
  if (kind == Angular)
    value = 2 * at::atan2(value, accumulated[1]);
  else if (kind == Ruzicka)
    value = ratio(value, accumulated[1]);
  else if (kind == NanEuclidean) {
    auto counts = accumulated[1];
    value = value * (double(x.size(-1)) / counts.clamp_min(1).to(dtype));
    if (!squared)
      value = safe_sqrt(value);
    value = at::where(counts.gt(0), value, NAN);
  } else if (kind == KL || kind == Jeffreys) {
    value = at::where(accumulated[1], INFINITY, value.clamp_min(0) / base);
  } else if (kind == FisherRao)
    value = 4 * at::asin((safe_sqrt(value) * .5).clamp_max(std::sqrt(.5)));
  else if (kind >= TotalVariation) {
    Tensor infinite;
    if (kind == Renyi) {
      double beta = alpha - 1;
      auto ordinary = alpha >= 2 ? value + accumulated[1] / beta : value / beta;
      auto local = accumulated[3].lt(.1) & ~accumulated[4];
      auto residual = beta * at::where(local, accumulated[2], 0.);
      value =
          at::where(local, residual.clamp_min(-.5).log1p() / beta, ordinary);
      infinite = ~accumulated[5] | (accumulated[4] & (alpha > 1));
    } else
      infinite = accumulated[1];
    auto maximum = dtype == at::kFloat
                       ? double(std::numeric_limits<float>::max())
                       : std::numeric_limits<double>::max();
    infinite = infinite | ~value.isfinite() | value.gt(maximum);
    value = at::where(infinite, INFINITY, value.clamp_min(0));
  }
  return value.to(dtype);
}

std::pair<Tensor, Tensor> broadcast(const Tensor& x, const Tensor& y) {
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 2),
                                     y.sym_sizes().slice(0, y.dim() - 2));
  auto xs = batch, ys = batch;
  xs.insert(xs.end(), {x.sym_size(-2), x.sym_size(-1)});
  ys.insert(ys.end(), {y.sym_size(-2), y.sym_size(-1)});
  return {x.expand_symint(xs), y.expand_symint(ys)};
}
// Matrix formulas build no pair-feature intermediate; feature formulas hold
// several float64 pair-feature temporaries per tile.
bool matrix_kind(int64_t kind) {
  return kind == RationalQuadratic || kind == Matern || kind == Bhattacharyya ||
         kind == BhattacharyyaKernel || kind == Aitchison || kind == Hellinger;
}
int64_t pair_width(int64_t kind, int64_t features, bool paired) {
  if (matrix_kind(kind))
    return paired ? features : 1;
  return 4 * std::min(features, feature_tile);
}
Tensor vector_compare_kernel(const Tensor& x, const Tensor& y,
                             const Tensor& grid, int64_t kind, int64_t mode,
                             double base, double length_scale, double alpha,
                             double nu, bool squared) {
  auto expanded = broadcast(x, y);
  auto a = expanded.first, b = expanded.second;
  auto n = a.size(-2), m = b.size(-2);
  auto shape = a.sizes().slice(0, a.dim() - 2).vec();
  shape.push_back(mode == 2 ? n * (n - 1) / 2 : n);
  if (mode == 0)
    shape.push_back(m);
  if (n == 0 || m == 0 || (mode == 2 && n < 2) || a.numel() == 0)
    return at::empty(shape, x.options());
  std::vector<Tensor> parts, row;
  int64_t current = -1, block_columns = 0;
  auto flush = [&]() {
    if (row.empty())
      return;
    auto value = at::cat(row, -1);
    if (mode == 2) {
      auto count = value.size(-2);
      value = value.flatten(-2).index_select(
          -1, utils::condensed_block_index(count, block_columns, a.options()));
    }
    parts.push_back(value);
    row.clear();
  };
  utils::visit_tiles(
      n, m, mode, pair_width(kind, a.size(-1), mode == 1),
      [&](int64_t i, int64_t count, int64_t j, int64_t other, int64_t columns) {
        if (i != current)
          flush();
        current = i;
        block_columns = columns;
        row.push_back(block(a.narrow(-2, i, count), b.narrow(-2, j, other),
                            grid, kind, mode == 1, base, length_scale, alpha,
                            nu, squared));
      });
  flush();
  return at::cat(parts, mode == 0 ? -2 : -1);
}

using CoreSignature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                             int64_t, int64_t, double, double, double, double,
                             bool);
using GradSignature = std::tuple<Tensor, Tensor, Tensor>(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, int64_t,
    int64_t, double, double, double, double, bool);

std::tuple<Tensor, Tensor, Tensor> vector_compare_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& grid, int64_t kind, int64_t mode, double base,
    double length_scale, double alpha, double nu, bool squared) {
  const bool higher = at::GradMode::is_enabled();
  TORCH_CHECK(!higher || kind == RationalQuadratic || kind == Matern,
              "Higher derivatives of native comparison replay are unsupported");
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda_autocast(
      c10::DispatchKey::AutocastCUDA);
  auto gx = at::zeros_like(x, x.options().dtype(at::kDouble));
  auto gy = at::zeros_like(y, y.options().dtype(at::kDouble));
  auto gg = at::zeros_like(grid, grid.options().dtype(at::kDouble));
  if (gradient.numel() == 0) {
    if (higher) {
      auto zero = gradient.sum() * 0;
      return {x * 0 + zero, y * 0 + zero, grid * 0 + zero};
    }
    return {gx.to(x.scalar_type()), gy.to(y.scalar_type()),
            gg.to(grid.scalar_type())};
  }
  auto expanded = broadcast(x, y);
  auto a = expanded.first, b = expanded.second;
  auto n = a.size(-2), m = b.size(-2);
  at::AutoGradMode enable(true);
  int64_t offset = 0, current = -1;
  Tensor block_gradient;
  utils::visit_tiles(
      n, m, mode, pair_width(kind, a.size(-1), mode == 1),
      [&](int64_t i, int64_t count, int64_t j, int64_t other, int64_t columns) {
        Tensor g;
        if (mode == 0)
          g = gradient.narrow(-2, i, count).narrow(-1, j, other);
        else if (mode == 1)
          g = gradient.narrow(-1, i, count);
        else {
          if (i != current) {
            // Scatter this row block's condensed entries into [count, n-i-1].
            auto size = utils::condensed_block_size(n, i, count);
            auto flat = gradient.sizes().vec();
            flat.back() = count * columns;
            auto full = flat;
            full.back() = count;
            full.push_back(columns);
            block_gradient = at::zeros(flat, gradient.options())
                                 .index_copy(-1,
                                             utils::condensed_block_index(
                                                 count, columns, a.options()),
                                             gradient.narrow(-1, offset, size))
                                 .reshape(full);
            offset += size;
            current = i;
          }
          g = block_gradient.narrow(-1, j - i - 1, other);
        }
        auto left = a.narrow(-2, i, count);
        auto right = b.narrow(-2, j, other);
        if (!higher) {
          left = left.detach();
          right = right.detach();
        }
        left.set_requires_grad(true);
        right.set_requires_grad(true);
        auto positions = grid.detach().set_requires_grad(grid.numel() > 0);
        auto value =
            higher ? detail::higher_order_comparison(
                         left, right,
                         kind == Matern ? "matern" : "rational_quadratic", true,
                         2., 3, 1., 1., length_scale, alpha, nu, mode == 1)
                   : block(left, right, positions, kind, mode == 1, base,
                           length_scale, alpha, nu, squared);
        Values operands{left, right};
        if (positions.requires_grad())
          operands.push_back(positions);
        auto grads =
            torch::autograd::grad({value}, operands, {g}, higher, higher, true);
        auto dx = gx.narrow(-2, i, count), dy = gy.narrow(-2, j, other);
        if (grads[0].defined())
          if (higher)
            gx = gx.index_add(
                -2, at::arange(i, i + count, x.options().dtype(at::kLong)),
                at::sum_to(grads[0].to(at::kDouble), dx.sizes()));
          else
            dx.add_(at::sum_to(grads[0].to(at::kDouble), dx.sizes()));
        if (grads[1].defined())
          if (higher)
            gy = gy.index_add(
                -2, at::arange(j, j + other, y.options().dtype(at::kLong)),
                at::sum_to(grads[1].to(at::kDouble), dy.sizes()));
          else
            dy.add_(at::sum_to(grads[1].to(at::kDouble), dy.sizes()));
        if (operands.size() == 3 && grads[2].defined())
          gg.add_(grads[2].to(at::kDouble));
      });
  return {gx.to(x.scalar_type()), gy.to(y.scalar_type()),
          gg.to(grid.scalar_type())};
}

int64_t kind_for(c10::string_view name) {
#define KIND(NAME, VALUE)                        \
  if (name == NAME || name == NAME "_distances") \
  return VALUE
  KIND("angular", Angular);
  KIND("lorentzian", Lorentzian);
  KIND("clark", Clark);
  KIND("soergel", Ruzicka);
  KIND("ruzicka", Ruzicka);
  KIND("continuous_jaccard", Ruzicka);
  KIND("aitchison", Aitchison);
  KIND("hellinger", Hellinger);
  KIND("wasserstein_1d", Wasserstein1D);
  KIND("nan_euclidean", NanEuclidean);
  KIND("total_variation", TotalVariation);
  KIND("fisher_rao", FisherRao);
  KIND("triangular_discrimination", Triangular);
#undef KIND
  if (name == "kl" || name == "kl_divergences")
    return KL;
  if (name == "symmetric_kl" || name == "symmetric_kl_divergences")
    return Jeffreys;
  if (name == "renyi" || name == "renyi_divergences")
    return Renyi;
  if (name == "pearson_chi2" || name == "pearson_chi2_divergences")
    return Pearson;
  if (name == "neyman_chi2" || name == "neyman_chi2_divergences")
    return Neyman;
  if (name == "itakura_saito" || name == "itakura_saito_divergences")
    return ItakuraSaito;
  if (name == "bhattacharyya_distances")
    return Bhattacharyya;
  if (name == "bhattacharyya_kernel")
    return BhattacharyyaKernel;
  if (name == "histogram_intersection" ||
      name == "histogram_intersection_kernel")
    return Intersection;
  if (name == "rational_quadratic" || name == "rational_quadratic_kernel")
    return RationalQuadratic;
  if (name == "matern" || name == "matern_kernel")
    return Matern;
  TORCH_CHECK(false, "Unsupported native vector comparison: ", name);
}
bool symmetric(int64_t kind) {
  return kind != KL && kind != Renyi && kind != Pearson && kind != Neyman &&
         kind != ItakuraSaito;
}
}  // namespace

at::Tensor vector_comparisons(const Tensor& input,
                              const c10::optional<Tensor>& other,
                              c10::string_view metric, c10::string_view mode,
                              const VectorComparisonOptions& opts) {
  using Signature = Tensor(
      const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
      c10::string_view mode, c10::optional<double> base,
      c10::optional<double> length_scale, c10::optional<double> alpha,
      c10::optional<double> nu, c10::optional<bool> squared,
      const c10::optional<Tensor>& bin_positions);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::vector_comparisons", "")
          .typed<Signature>();
  return op.call(input, other, metric, mode, opts.base, opts.length_scale,
                 opts.alpha, opts.nu, opts.squared, opts.bin_positions);
}

namespace {
Tensor vector_comparisons_impl(const Tensor& x, const c10::optional<Tensor>& y,
                               c10::string_view metric, c10::string_view mode,
                               c10::optional<double> base,
                               c10::optional<double> length_scale,
                               c10::optional<double> alpha,
                               c10::optional<double> nu,
                               c10::optional<bool> squared,
                               const c10::optional<Tensor>& bin_positions) {
  return vector_comparisons_composite(
      x, y, metric, mode,
      {base, length_scale, alpha, nu, squared, bin_positions});
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "vector_comparisons(Tensor x1, Tensor? x2, str metric, str "
      "mode='all_pairs', float? base=None, float? length_scale=None, float? "
      "alpha=None, float? nu=None, bool? squared=None, Tensor? "
      "bin_positions=None) -> Tensor",
      TORCH_FN(vector_comparisons_impl));
  m.def(
      "_vector_compare(Tensor x, Tensor y, Tensor grid, int kind, int mode, "
      "float base, float length_scale, float alpha, float nu, bool squared) -> "
      "Tensor");
  m.def(
      "_vector_compare_backward(Tensor grad, Tensor x, Tensor y, Tensor grid, "
      "int kind, int mode, float base, float length_scale, float alpha, float "
      "nu, bool squared) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_vector_compare", TORCH_FN(vector_compare_kernel));
  m.impl("_vector_compare_backward", TORCH_FN(vector_compare_backward_kernel));
}

}  // namespace

// Named API bindings follow.
at::Tensor angular_distances(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::angular_distances", "")
          .typed<decltype(angular_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_angular_distances(const at::Tensor& x1,
                                    const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_angular_distances", "")
          .typed<decltype(paired_angular_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_angular_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_angular_distances", "")
          .typed<decltype(condensed_angular_distances)>();
  return op.call(x1);
}
at::Tensor lorentzian_distances(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::lorentzian_distances", "")
          .typed<decltype(lorentzian_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_lorentzian_distances(const at::Tensor& x1,
                                       const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_lorentzian_distances", "")
          .typed<decltype(paired_lorentzian_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_lorentzian_distances(const at::Tensor& x1) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::condensed_lorentzian_distances", "")
                       .typed<decltype(condensed_lorentzian_distances)>();
  return op.call(x1);
}
at::Tensor clark_distances(const at::Tensor& x1,
                           const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::clark_distances", "")
                       .typed<decltype(clark_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_clark_distances(const at::Tensor& x1, const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_clark_distances", "")
          .typed<decltype(paired_clark_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_clark_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_clark_distances", "")
          .typed<decltype(condensed_clark_distances)>();
  return op.call(x1);
}
at::Tensor soergel_distances(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::soergel_distances", "")
          .typed<decltype(soergel_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_soergel_distances(const at::Tensor& x1,
                                    const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_soergel_distances", "")
          .typed<decltype(paired_soergel_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_soergel_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_soergel_distances", "")
          .typed<decltype(condensed_soergel_distances)>();
  return op.call(x1);
}
at::Tensor ruzicka_distances(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::ruzicka_distances", "")
          .typed<decltype(ruzicka_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_ruzicka_distances(const at::Tensor& x1,
                                    const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_ruzicka_distances", "")
          .typed<decltype(paired_ruzicka_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_ruzicka_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_ruzicka_distances", "")
          .typed<decltype(condensed_ruzicka_distances)>();
  return op.call(x1);
}
at::Tensor bhattacharyya_distances(const at::Tensor& x1,
                                   const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::bhattacharyya_distances", "")
          .typed<decltype(bhattacharyya_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_bhattacharyya_distances(const at::Tensor& x1,
                                          const at::Tensor& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::paired_bhattacharyya_distances", "")
                       .typed<decltype(paired_bhattacharyya_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_bhattacharyya_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_bhattacharyya_distances",
                             "")
          .typed<decltype(condensed_bhattacharyya_distances)>();
  return op.call(x1);
}
at::Tensor bhattacharyya_kernel(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::bhattacharyya_kernel", "")
          .typed<decltype(bhattacharyya_kernel)>();
  return op.call(x1, x2);
}
at::Tensor paired_bhattacharyya_kernel(const at::Tensor& x1,
                                       const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_bhattacharyya_kernel", "")
          .typed<decltype(paired_bhattacharyya_kernel)>();
  return op.call(x1, x2);
}
at::Tensor condensed_bhattacharyya_kernel(const at::Tensor& x1) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::condensed_bhattacharyya_kernel", "")
                       .typed<decltype(condensed_bhattacharyya_kernel)>();
  return op.call(x1);
}
at::Tensor aitchison_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::aitchison_distances", "")
          .typed<decltype(aitchison_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_aitchison_distances(const at::Tensor& x1,
                                      const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_aitchison_distances", "")
          .typed<decltype(paired_aitchison_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_aitchison_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_aitchison_distances", "")
          .typed<decltype(condensed_aitchison_distances)>();
  return op.call(x1);
}
at::Tensor hellinger_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::hellinger_distances", "")
          .typed<decltype(hellinger_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_hellinger_distances(const at::Tensor& x1,
                                      const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_hellinger_distances", "")
          .typed<decltype(paired_hellinger_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_hellinger_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_hellinger_distances", "")
          .typed<decltype(condensed_hellinger_distances)>();
  return op.call(x1);
}
at::Tensor histogram_intersection_kernel(const at::Tensor& x1,
                                         const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::histogram_intersection_kernel", "")
          .typed<decltype(histogram_intersection_kernel)>();
  return op.call(x1, x2);
}
at::Tensor paired_histogram_intersection_kernel(const at::Tensor& x1,
                                                const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::paired_histogram_intersection_kernel", "")
          .typed<decltype(paired_histogram_intersection_kernel)>();
  return op.call(x1, x2);
}
at::Tensor condensed_histogram_intersection_kernel(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::condensed_histogram_intersection_kernel", "")
          .typed<decltype(condensed_histogram_intersection_kernel)>();
  return op.call(x1);
}
at::Tensor total_variation_distances(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::total_variation_distances", "")
          .typed<decltype(total_variation_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_total_variation_distances(const at::Tensor& x1,
                                            const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_total_variation_distances",
                             "")
          .typed<decltype(paired_total_variation_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_total_variation_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::condensed_total_variation_distances", "")
          .typed<decltype(condensed_total_variation_distances)>();
  return op.call(x1);
}
at::Tensor fisher_rao_distances(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::fisher_rao_distances", "")
          .typed<decltype(fisher_rao_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_fisher_rao_distances(const at::Tensor& x1,
                                       const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_fisher_rao_distances", "")
          .typed<decltype(paired_fisher_rao_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_fisher_rao_distances(const at::Tensor& x1) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::condensed_fisher_rao_distances", "")
                       .typed<decltype(condensed_fisher_rao_distances)>();
  return op.call(x1);
}
at::Tensor pearson_chi2_divergences(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::pearson_chi2_divergences", "")
          .typed<decltype(pearson_chi2_divergences)>();
  return op.call(x1, x2);
}
at::Tensor paired_pearson_chi2_divergences(const at::Tensor& x1,
                                           const at::Tensor& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::paired_pearson_chi2_divergences", "")
                       .typed<decltype(paired_pearson_chi2_divergences)>();
  return op.call(x1, x2);
}
at::Tensor neyman_chi2_divergences(const at::Tensor& x1,
                                   const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::neyman_chi2_divergences", "")
          .typed<decltype(neyman_chi2_divergences)>();
  return op.call(x1, x2);
}
at::Tensor paired_neyman_chi2_divergences(const at::Tensor& x1,
                                          const at::Tensor& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::paired_neyman_chi2_divergences", "")
                       .typed<decltype(paired_neyman_chi2_divergences)>();
  return op.call(x1, x2);
}
at::Tensor triangular_discrimination_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::triangular_discrimination_distances", "")
          .typed<decltype(triangular_discrimination_distances)>();
  return op.call(x1, x2);
}
at::Tensor paired_triangular_discrimination_distances(const at::Tensor& x1,
                                                      const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::paired_triangular_discrimination_distances", "")
          .typed<decltype(paired_triangular_discrimination_distances)>();
  return op.call(x1, x2);
}
at::Tensor condensed_triangular_discrimination_distances(const at::Tensor& x1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::condensed_triangular_discrimination_distances",
              "")
          .typed<decltype(condensed_triangular_discrimination_distances)>();
  return op.call(x1);
}
at::Tensor itakura_saito_divergences(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::itakura_saito_divergences", "")
          .typed<decltype(itakura_saito_divergences)>();
  return op.call(x1, x2);
}
at::Tensor paired_itakura_saito_divergences(const at::Tensor& x1,
                                            const at::Tensor& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_itakura_saito_divergences",
                             "")
          .typed<decltype(paired_itakura_saito_divergences)>();
  return op.call(x1, x2);
}
at::Tensor kl_divergences(const at::Tensor& x1,
                          const c10::optional<at::Tensor>& x2,
                          c10::optional<double> base) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::kl_divergences", "")
                       .typed<decltype(kl_divergences)>();
  return op.call(x1, x2, base);
}
at::Tensor paired_kl_divergences(const at::Tensor& x1, const at::Tensor& x2,
                                 c10::optional<double> base) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_kl_divergences", "")
          .typed<decltype(paired_kl_divergences)>();
  return op.call(x1, x2, base);
}
at::Tensor symmetric_kl_divergences(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2,
                                    c10::optional<double> base) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::symmetric_kl_divergences", "")
          .typed<decltype(symmetric_kl_divergences)>();
  return op.call(x1, x2, base);
}
at::Tensor paired_symmetric_kl_divergences(const at::Tensor& x1,
                                           const at::Tensor& x2,
                                           c10::optional<double> base) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::paired_symmetric_kl_divergences", "")
                       .typed<decltype(paired_symmetric_kl_divergences)>();
  return op.call(x1, x2, base);
}
at::Tensor condensed_symmetric_kl_divergences(const at::Tensor& x1,
                                              c10::optional<double> base) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::condensed_symmetric_kl_divergences", "")
          .typed<decltype(condensed_symmetric_kl_divergences)>();
  return op.call(x1, base);
}
at::Tensor renyi_divergences(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2,
                             double alpha) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::renyi_divergences", "")
          .typed<decltype(renyi_divergences)>();
  return op.call(x1, x2, alpha);
}
at::Tensor paired_renyi_divergences(const at::Tensor& x1, const at::Tensor& x2,
                                    double alpha) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_renyi_divergences", "")
          .typed<decltype(paired_renyi_divergences)>();
  return op.call(x1, x2, alpha);
}
at::Tensor rational_quadratic_kernel(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2,
                                     double length_scale, double alpha) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::rational_quadratic_kernel", "")
          .typed<decltype(rational_quadratic_kernel)>();
  return op.call(x1, x2, length_scale, alpha);
}
at::Tensor paired_rational_quadratic_kernel(const at::Tensor& x1,
                                            const at::Tensor& x2,
                                            double length_scale, double alpha) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_rational_quadratic_kernel",
                             "")
          .typed<decltype(paired_rational_quadratic_kernel)>();
  return op.call(x1, x2, length_scale, alpha);
}
at::Tensor condensed_rational_quadratic_kernel(const at::Tensor& x1,
                                               double length_scale,
                                               double alpha) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::condensed_rational_quadratic_kernel", "")
          .typed<decltype(condensed_rational_quadratic_kernel)>();
  return op.call(x1, length_scale, alpha);
}
at::Tensor matern_kernel(const at::Tensor& x1,
                         const c10::optional<at::Tensor>& x2,
                         double length_scale, double nu) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::matern_kernel", "")
                       .typed<decltype(matern_kernel)>();
  return op.call(x1, x2, length_scale, nu);
}
at::Tensor paired_matern_kernel(const at::Tensor& x1, const at::Tensor& x2,
                                double length_scale, double nu) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::paired_matern_kernel", "")
          .typed<decltype(paired_matern_kernel)>();
  return op.call(x1, x2, length_scale, nu);
}
at::Tensor condensed_matern_kernel(const at::Tensor& x1, double length_scale,
                                   double nu) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_matern_kernel", "")
          .typed<decltype(condensed_matern_kernel)>();
  return op.call(x1, length_scale, nu);
}
at::Tensor nan_euclidean_distances(const at::Tensor& x1,
                                   const c10::optional<at::Tensor>& x2,
                                   bool squared) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::nan_euclidean_distances", "")
          .typed<decltype(nan_euclidean_distances)>();
  return op.call(x1, x2, squared);
}
at::Tensor paired_nan_euclidean_distances(const at::Tensor& x1,
                                          const at::Tensor& x2, bool squared) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::paired_nan_euclidean_distances", "")
                       .typed<decltype(paired_nan_euclidean_distances)>();
  return op.call(x1, x2, squared);
}
at::Tensor condensed_nan_euclidean_distances(const at::Tensor& x1,
                                             bool squared) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::condensed_nan_euclidean_distances",
                             "")
          .typed<decltype(condensed_nan_euclidean_distances)>();
  return op.call(x1, squared);
}
at::Tensor wasserstein_1d_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    const c10::optional<at::Tensor>& bin_positions) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::wasserstein_1d_distances", "")
          .typed<decltype(wasserstein_1d_distances)>();
  return op.call(x1, x2, bin_positions);
}
at::Tensor paired_wasserstein_1d_distances(
    const at::Tensor& x1, const at::Tensor& x2,
    const c10::optional<at::Tensor>& bin_positions) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::paired_wasserstein_1d_distances", "")
                       .typed<decltype(paired_wasserstein_1d_distances)>();
  return op.call(x1, x2, bin_positions);
}
at::Tensor condensed_wasserstein_1d_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& bin_positions) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "torchpairwise::condensed_wasserstein_1d_distances", "")
          .typed<decltype(condensed_wasserstein_1d_distances)>();
  return op.call(x1, bin_positions);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("angular_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(angular_distances_impl));
  m.def("paired_angular_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_angular_distances_impl));
  m.def("condensed_angular_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_angular_distances_impl));
  m.def("lorentzian_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(lorentzian_distances_impl));
  m.def("paired_lorentzian_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_lorentzian_distances_impl));
  m.def("condensed_lorentzian_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_lorentzian_distances_impl));
  m.def("clark_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(clark_distances_impl));
  m.def("paired_clark_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_clark_distances_impl));
  m.def("condensed_clark_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_clark_distances_impl));
  m.def("soergel_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(soergel_distances_impl));
  m.def("paired_soergel_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_soergel_distances_impl));
  m.def("condensed_soergel_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_soergel_distances_impl));
  m.def("ruzicka_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(ruzicka_distances_impl));
  m.def("paired_ruzicka_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_ruzicka_distances_impl));
  m.def("condensed_ruzicka_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_ruzicka_distances_impl));
  m.def("bhattacharyya_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(bhattacharyya_distances_impl));
  m.def("paired_bhattacharyya_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_bhattacharyya_distances_impl));
  m.def("condensed_bhattacharyya_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_bhattacharyya_distances_impl));
  m.def("bhattacharyya_kernel(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(bhattacharyya_kernel_impl));
  m.def("paired_bhattacharyya_kernel(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_bhattacharyya_kernel_impl));
  m.def("condensed_bhattacharyya_kernel(Tensor x1) -> Tensor",
        TORCH_FN(condensed_bhattacharyya_kernel_impl));
  m.def("aitchison_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(aitchison_distances_impl));
  m.def("paired_aitchison_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_aitchison_distances_impl));
  m.def("condensed_aitchison_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_aitchison_distances_impl));
  m.def("hellinger_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(hellinger_distances_impl));
  m.def("paired_hellinger_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_hellinger_distances_impl));
  m.def("condensed_hellinger_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_hellinger_distances_impl));
  m.def("histogram_intersection_kernel(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(histogram_intersection_kernel_impl));
  m.def("paired_histogram_intersection_kernel(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_histogram_intersection_kernel_impl));
  m.def("condensed_histogram_intersection_kernel(Tensor x1) -> Tensor",
        TORCH_FN(condensed_histogram_intersection_kernel_impl));
  m.def("total_variation_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(total_variation_distances_impl));
  m.def("paired_total_variation_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_total_variation_distances_impl));
  m.def("condensed_total_variation_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_total_variation_distances_impl));
  m.def("fisher_rao_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(fisher_rao_distances_impl));
  m.def("paired_fisher_rao_distances(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_fisher_rao_distances_impl));
  m.def("condensed_fisher_rao_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_fisher_rao_distances_impl));
  m.def("pearson_chi2_divergences(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(pearson_chi2_divergences_impl));
  m.def("paired_pearson_chi2_divergences(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_pearson_chi2_divergences_impl));
  m.def("neyman_chi2_divergences(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(neyman_chi2_divergences_impl));
  m.def("paired_neyman_chi2_divergences(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_neyman_chi2_divergences_impl));
  m.def(
      "triangular_discrimination_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(triangular_discrimination_distances_impl));
  m.def(
      "paired_triangular_discrimination_distances(Tensor x1, Tensor x2) -> "
      "Tensor",
      TORCH_FN(paired_triangular_discrimination_distances_impl));
  m.def("condensed_triangular_discrimination_distances(Tensor x1) -> Tensor",
        TORCH_FN(condensed_triangular_discrimination_distances_impl));
  m.def("itakura_saito_divergences(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(itakura_saito_divergences_impl));
  m.def("paired_itakura_saito_divergences(Tensor x1, Tensor x2) -> Tensor",
        TORCH_FN(paired_itakura_saito_divergences_impl));
  m.def(
      "kl_divergences(Tensor x1, Tensor? x2=None, float? base=None) -> Tensor",
      TORCH_FN(kl_divergences_impl));
  m.def(
      "paired_kl_divergences(Tensor x1, Tensor x2, float? base=None) -> Tensor",
      TORCH_FN(paired_kl_divergences_impl));
  m.def(
      "symmetric_kl_divergences(Tensor x1, Tensor? x2=None, float? base=None) "
      "-> Tensor",
      TORCH_FN(symmetric_kl_divergences_impl));
  m.def(
      "paired_symmetric_kl_divergences(Tensor x1, Tensor x2, float? base=None) "
      "-> Tensor",
      TORCH_FN(paired_symmetric_kl_divergences_impl));
  m.def(
      "condensed_symmetric_kl_divergences(Tensor x1, float? base=None) -> "
      "Tensor",
      TORCH_FN(condensed_symmetric_kl_divergences_impl));
  m.def(
      "renyi_divergences(Tensor x1, Tensor? x2=None, float alpha=1.0) -> "
      "Tensor",
      TORCH_FN(renyi_divergences_impl));
  m.def(
      "paired_renyi_divergences(Tensor x1, Tensor x2, float alpha=1.0) -> "
      "Tensor",
      TORCH_FN(paired_renyi_divergences_impl));
  m.def(
      "rational_quadratic_kernel(Tensor x1, Tensor? x2=None, float "
      "length_scale=1.0, float alpha=1.0) -> Tensor",
      TORCH_FN(rational_quadratic_kernel_impl));
  m.def(
      "paired_rational_quadratic_kernel(Tensor x1, Tensor x2, float "
      "length_scale=1.0, float alpha=1.0) -> Tensor",
      TORCH_FN(paired_rational_quadratic_kernel_impl));
  m.def(
      "condensed_rational_quadratic_kernel(Tensor x1, float length_scale=1.0, "
      "float alpha=1.0) -> Tensor",
      TORCH_FN(condensed_rational_quadratic_kernel_impl));
  m.def(
      "matern_kernel(Tensor x1, Tensor? x2=None, float length_scale=1.0, float "
      "nu=1.5) -> Tensor",
      TORCH_FN(matern_kernel_impl));
  m.def(
      "paired_matern_kernel(Tensor x1, Tensor x2, float length_scale=1.0, "
      "float nu=1.5) -> Tensor",
      TORCH_FN(paired_matern_kernel_impl));
  m.def(
      "condensed_matern_kernel(Tensor x1, float length_scale=1.0, float "
      "nu=1.5) -> Tensor",
      TORCH_FN(condensed_matern_kernel_impl));
  m.def(
      "nan_euclidean_distances(Tensor x1, Tensor? x2=None, bool squared=False) "
      "-> Tensor",
      TORCH_FN(nan_euclidean_distances_impl));
  m.def(
      "paired_nan_euclidean_distances(Tensor x1, Tensor x2, bool "
      "squared=False) -> Tensor",
      TORCH_FN(paired_nan_euclidean_distances_impl));
  m.def(
      "condensed_nan_euclidean_distances(Tensor x1, bool squared=False) -> "
      "Tensor",
      TORCH_FN(condensed_nan_euclidean_distances_impl));
  m.def(
      "wasserstein_1d_distances(Tensor x1, Tensor? x2=None, Tensor? "
      "bin_positions=None) -> Tensor",
      TORCH_FN(wasserstein_1d_distances_impl));
  m.def(
      "paired_wasserstein_1d_distances(Tensor x1, Tensor x2, Tensor? "
      "bin_positions=None) -> Tensor",
      TORCH_FN(paired_wasserstein_1d_distances_impl));
  m.def(
      "condensed_wasserstein_1d_distances(Tensor x1, Tensor? "
      "bin_positions=None) -> Tensor",
      TORCH_FN(condensed_wasserstein_1d_distances_impl));
}
namespace detail::vector {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::array<Tensor, 3>;
Values feature_kernel(Triple a, Triple b, const Tensor& scale, int64_t kind,
                      bool paired, double alpha, double limit) {
  return feature(a, b, scale, kind, paired, alpha, limit);
}
}  // namespace detail::vector
namespace {
at::Tensor angular_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "angular_distances", "all_pairs", options);
}

at::Tensor paired_angular_distances_impl(const at::Tensor& x1,
                                         const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "angular_distances", "paired", options);
}

at::Tensor condensed_angular_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "angular_distances", "condensed",
                            options);
}

at::Tensor lorentzian_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "lorentzian_distances", "all_pairs",
                            options);
}

at::Tensor paired_lorentzian_distances_impl(const at::Tensor& x1,
                                            const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "lorentzian_distances", "paired", options);
}

at::Tensor condensed_lorentzian_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "lorentzian_distances",
                            "condensed", options);
}

at::Tensor clark_distances_impl(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "clark_distances", "all_pairs", options);
}

at::Tensor paired_clark_distances_impl(const at::Tensor& x1,
                                       const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "clark_distances", "paired", options);
}

at::Tensor condensed_clark_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "clark_distances", "condensed",
                            options);
}

at::Tensor soergel_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "soergel_distances", "all_pairs", options);
}

at::Tensor paired_soergel_distances_impl(const at::Tensor& x1,
                                         const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "soergel_distances", "paired", options);
}

at::Tensor condensed_soergel_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "soergel_distances", "condensed",
                            options);
}

at::Tensor ruzicka_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "ruzicka_distances", "all_pairs", options);
}

at::Tensor paired_ruzicka_distances_impl(const at::Tensor& x1,
                                         const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "ruzicka_distances", "paired", options);
}

at::Tensor condensed_ruzicka_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "ruzicka_distances", "condensed",
                            options);
}

at::Tensor bhattacharyya_distances_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "bhattacharyya_distances", "all_pairs",
                            options);
}

at::Tensor paired_bhattacharyya_distances_impl(const at::Tensor& x1,
                                               const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "bhattacharyya_distances", "paired",
                            options);
}

at::Tensor condensed_bhattacharyya_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "bhattacharyya_distances",
                            "condensed", options);
}

at::Tensor bhattacharyya_kernel_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "bhattacharyya_kernel", "all_pairs",
                            options);
}

at::Tensor paired_bhattacharyya_kernel_impl(const at::Tensor& x1,
                                            const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "bhattacharyya_kernel", "paired", options);
}

at::Tensor condensed_bhattacharyya_kernel_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "bhattacharyya_kernel",
                            "condensed", options);
}

at::Tensor aitchison_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "aitchison_distances", "all_pairs",
                            options);
}

at::Tensor paired_aitchison_distances_impl(const at::Tensor& x1,
                                           const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "aitchison_distances", "paired", options);
}

at::Tensor condensed_aitchison_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "aitchison_distances",
                            "condensed", options);
}

at::Tensor hellinger_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "hellinger_distances", "all_pairs",
                            options);
}

at::Tensor paired_hellinger_distances_impl(const at::Tensor& x1,
                                           const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "hellinger_distances", "paired", options);
}

at::Tensor condensed_hellinger_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "hellinger_distances",
                            "condensed", options);
}

at::Tensor histogram_intersection_kernel_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "histogram_intersection_kernel",
                            "all_pairs", options);
}

at::Tensor paired_histogram_intersection_kernel_impl(const at::Tensor& x1,
                                                     const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "histogram_intersection_kernel", "paired",
                            options);
}

at::Tensor condensed_histogram_intersection_kernel_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "histogram_intersection_kernel",
                            "condensed", options);
}

at::Tensor total_variation_distances_impl(const at::Tensor& x1,
                                          const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "total_variation_distances", "all_pairs",
                            options);
}

at::Tensor paired_total_variation_distances_impl(const at::Tensor& x1,
                                                 const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "total_variation_distances", "paired",
                            options);
}

at::Tensor condensed_total_variation_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "total_variation_distances",
                            "condensed", options);
}

at::Tensor fisher_rao_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "fisher_rao_distances", "all_pairs",
                            options);
}

at::Tensor paired_fisher_rao_distances_impl(const at::Tensor& x1,
                                            const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "fisher_rao_distances", "paired", options);
}

at::Tensor condensed_fisher_rao_distances_impl(const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt, "fisher_rao_distances",
                            "condensed", options);
}

at::Tensor pearson_chi2_divergences_impl(const at::Tensor& x1,
                                         const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "pearson_chi2_divergences", "all_pairs",
                            options);
}

at::Tensor paired_pearson_chi2_divergences_impl(const at::Tensor& x1,
                                                const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "pearson_chi2_divergences", "paired",
                            options);
}

at::Tensor neyman_chi2_divergences_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "neyman_chi2_divergences", "all_pairs",
                            options);
}

at::Tensor paired_neyman_chi2_divergences_impl(const at::Tensor& x1,
                                               const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "neyman_chi2_divergences", "paired",
                            options);
}

at::Tensor triangular_discrimination_distances_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "triangular_discrimination_distances",
                            "all_pairs", options);
}

at::Tensor paired_triangular_discrimination_distances_impl(
    const at::Tensor& x1, const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "triangular_discrimination_distances",
                            "paired", options);
}

at::Tensor condensed_triangular_discrimination_distances_impl(
    const at::Tensor& x1) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, c10::nullopt,
                            "triangular_discrimination_distances", "condensed",
                            options);
}

at::Tensor itakura_saito_divergences_impl(const at::Tensor& x1,
                                          const c10::optional<at::Tensor>& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "itakura_saito_divergences", "all_pairs",
                            options);
}

at::Tensor paired_itakura_saito_divergences_impl(const at::Tensor& x1,
                                                 const at::Tensor& x2) {
  VectorComparisonOptions options;
  return vector_comparisons(x1, x2, "itakura_saito_divergences", "paired",
                            options);
}

at::Tensor kl_divergences_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2,
                               c10::optional<double> base) {
  VectorComparisonOptions options;
  options.base = base;
  return vector_comparisons(x1, x2, "kl_divergences", "all_pairs", options);
}

at::Tensor paired_kl_divergences_impl(const at::Tensor& x1,
                                      const at::Tensor& x2,
                                      c10::optional<double> base) {
  VectorComparisonOptions options;
  options.base = base;
  return vector_comparisons(x1, x2, "kl_divergences", "paired", options);
}

at::Tensor symmetric_kl_divergences_impl(const at::Tensor& x1,
                                         const c10::optional<at::Tensor>& x2,
                                         c10::optional<double> base) {
  VectorComparisonOptions options;
  options.base = base;
  return vector_comparisons(x1, x2, "symmetric_kl_divergences", "all_pairs",
                            options);
}

at::Tensor paired_symmetric_kl_divergences_impl(const at::Tensor& x1,
                                                const at::Tensor& x2,
                                                c10::optional<double> base) {
  VectorComparisonOptions options;
  options.base = base;
  return vector_comparisons(x1, x2, "symmetric_kl_divergences", "paired",
                            options);
}

at::Tensor condensed_symmetric_kl_divergences_impl(const at::Tensor& x1,
                                                   c10::optional<double> base) {
  VectorComparisonOptions options;
  options.base = base;
  return vector_comparisons(x1, c10::nullopt, "symmetric_kl_divergences",
                            "condensed", options);
}

at::Tensor renyi_divergences_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2,
                                  double alpha) {
  VectorComparisonOptions options;
  options.alpha = alpha;
  return vector_comparisons(x1, x2, "renyi_divergences", "all_pairs", options);
}

at::Tensor paired_renyi_divergences_impl(const at::Tensor& x1,
                                         const at::Tensor& x2, double alpha) {
  VectorComparisonOptions options;
  options.alpha = alpha;
  return vector_comparisons(x1, x2, "renyi_divergences", "paired", options);
}

at::Tensor rational_quadratic_kernel_impl(const at::Tensor& x1,
                                          const c10::optional<at::Tensor>& x2,
                                          double length_scale, double alpha) {
  VectorComparisonOptions options;
  options.length_scale = length_scale;
  options.alpha = alpha;
  return vector_comparisons(x1, x2, "rational_quadratic_kernel", "all_pairs",
                            options);
}

at::Tensor paired_rational_quadratic_kernel_impl(const at::Tensor& x1,
                                                 const at::Tensor& x2,
                                                 double length_scale,
                                                 double alpha) {
  VectorComparisonOptions options;
  options.length_scale = length_scale;
  options.alpha = alpha;
  return vector_comparisons(x1, x2, "rational_quadratic_kernel", "paired",
                            options);
}

at::Tensor condensed_rational_quadratic_kernel_impl(const at::Tensor& x1,
                                                    double length_scale,
                                                    double alpha) {
  VectorComparisonOptions options;
  options.length_scale = length_scale;
  options.alpha = alpha;
  return vector_comparisons(x1, c10::nullopt, "rational_quadratic_kernel",
                            "condensed", options);
}

at::Tensor matern_kernel_impl(const at::Tensor& x1,
                              const c10::optional<at::Tensor>& x2,
                              double length_scale, double nu) {
  VectorComparisonOptions options;
  options.length_scale = length_scale;
  options.nu = nu;
  return vector_comparisons(x1, x2, "matern_kernel", "all_pairs", options);
}

at::Tensor paired_matern_kernel_impl(const at::Tensor& x1, const at::Tensor& x2,
                                     double length_scale, double nu) {
  VectorComparisonOptions options;
  options.length_scale = length_scale;
  options.nu = nu;
  return vector_comparisons(x1, x2, "matern_kernel", "paired", options);
}

at::Tensor condensed_matern_kernel_impl(const at::Tensor& x1,
                                        double length_scale, double nu) {
  VectorComparisonOptions options;
  options.length_scale = length_scale;
  options.nu = nu;
  return vector_comparisons(x1, c10::nullopt, "matern_kernel", "condensed",
                            options);
}

at::Tensor nan_euclidean_distances_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2,
                                        bool squared) {
  VectorComparisonOptions options;
  options.squared = squared;
  return vector_comparisons(x1, x2, "nan_euclidean_distances", "all_pairs",
                            options);
}

at::Tensor paired_nan_euclidean_distances_impl(const at::Tensor& x1,
                                               const at::Tensor& x2,
                                               bool squared) {
  VectorComparisonOptions options;
  options.squared = squared;
  return vector_comparisons(x1, x2, "nan_euclidean_distances", "paired",
                            options);
}

at::Tensor condensed_nan_euclidean_distances_impl(const at::Tensor& x1,
                                                  bool squared) {
  VectorComparisonOptions options;
  options.squared = squared;
  return vector_comparisons(x1, c10::nullopt, "nan_euclidean_distances",
                            "condensed", options);
}

at::Tensor wasserstein_1d_distances_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    const c10::optional<at::Tensor>& bin_positions) {
  VectorComparisonOptions options;
  options.bin_positions = bin_positions;
  return vector_comparisons(x1, x2, "wasserstein_1d_distances", "all_pairs",
                            options);
}

at::Tensor paired_wasserstein_1d_distances_impl(
    const at::Tensor& x1, const at::Tensor& x2,
    const c10::optional<at::Tensor>& bin_positions) {
  VectorComparisonOptions options;
  options.bin_positions = bin_positions;
  return vector_comparisons(x1, x2, "wasserstein_1d_distances", "paired",
                            options);
}

at::Tensor condensed_wasserstein_1d_distances_impl(
    const at::Tensor& x1, const c10::optional<at::Tensor>& bin_positions) {
  VectorComparisonOptions options;
  options.bin_positions = bin_positions;
  return vector_comparisons(x1, c10::nullopt, "wasserstein_1d_distances",
                            "condensed", options);
}

at::Tensor vector_comparisons_composite(const Tensor& input,
                                        const c10::optional<Tensor>& other,
                                        c10::string_view metric,
                                        c10::string_view mode,
                                        const VectorComparisonOptions& opts) {
  auto kind = kind_for(metric);
  int64_t pairing = mode == "all_pairs"   ? 0
                    : mode == "paired"    ? 1
                    : mode == "condensed" ? 2
                                          : -1;
  TORCH_CHECK(pairing >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(pairing != 2 || (!other.has_value() && symmetric(kind)),
              "Condensed output requires one input and a symmetric metric");
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(!opts.base.has_value() || kind == KL || kind == Jeffreys,
              "Unused parameter: base");
  TORCH_CHECK(
      !opts.alpha.has_value() || kind == Renyi || kind == RationalQuadratic,
      "Unused parameter: alpha");
  TORCH_CHECK(!opts.length_scale.has_value() || kind == RationalQuadratic ||
                  kind == Matern,
              "Unused parameter: length_scale");
  TORCH_CHECK(!opts.nu.has_value() || kind == Matern, "Unused parameter: nu");
  TORCH_CHECK(!opts.squared.has_value() || kind == NanEuclidean,
              "Unused parameter: squared");
  TORCH_CHECK(!opts.bin_positions.has_value() || kind == Wasserstein1D,
              "Unused parameter: bin_positions");
  double base = opts.base.has_value() ? std::log(*opts.base) : 1.;
  double alpha = opts.alpha.value_or(1.),
         length_scale = opts.length_scale.value_or(1.),
         nu = opts.nu.value_or(1.5);
  TORCH_CHECK(
      !opts.base.has_value() || (std::isfinite(*opts.base) && *opts.base > 1),
      "base must be finite and greater than one");
  TORCH_CHECK(std::isfinite(alpha) && alpha > 0,
              "alpha must be positive and finite");
  TORCH_CHECK(std::isfinite(length_scale) && length_scale > 0,
              "length_scale must be positive and finite");
  TORCH_CHECK(nu == .5 || nu == 1.5 || nu == 2.5, "nu must be 0.5, 1.5 or 2.5");
  Tensor x = input, y = other.value_or(input),
         grid = opts.bin_positions.value_or(at::empty({0}, input.options()));
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided,
              "Inputs must be dense strided tensors");
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Inputs must have matching dtype and device");
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Native vector comparisons support CPU, CUDA and metadata execution");
  if (!x.device().is_meta() &&
      at::autocast::is_autocast_enabled(x.device().type())) {
    if (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) {
      x = x.to(at::kFloat);
      y = y.to(at::kFloat);
    }
    if (grid.scalar_type() == at::kHalf || grid.scalar_type() == at::kBFloat16)
      grid = grid.to(at::kFloat);
  }
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda_autocast(
      c10::DispatchKey::AutocastCUDA);
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "Inputs must be float32 or float64 outside autocast");
  TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2 &&
                  x.sym_size(-1) == y.sym_size(-1) && x.sym_size(-1) > 0,
              "Inputs must have matching nonempty features and rank >= 2");
  TORCH_CHECK(pairing != 1 || x.sym_size(-2) == y.sym_size(-2),
              "Paired inputs must have equal sample counts");
  broadcast(x, y);
  utils::ValueChecks checks;
  for (auto& value : {x, y}) {
    if (kind == NanEuclidean) {
      checks.require((~value.isinf()).all(),
                     "NaN Euclidean inputs cannot contain infinities");
      continue;
    }
    bool nonnegative = kind != Angular && kind != Lorentzian &&
                       kind != Aitchison && kind != ItakuraSaito &&
                       kind != RationalQuadratic && kind != Matern;
    if (kind != RationalQuadratic && kind != Matern)
      checks.require(
          (value.isfinite() &
           (nonnegative ? value.ge(0) : at::ones_like(value, at::kBool)))
              .all(),
          "Inputs must be finite and non-negative where required");
    if (kind == Angular)
      checks.require(value.abs().amax(-1).gt(0).all(),
                     "Angular inputs require nonzero rows");
    if (kind == Aitchison || kind == ItakuraSaito)
      checks.require(value.gt(0).all(), "Inputs must be strictly positive");
    bool mass = (kind >= KL && kind <= BhattacharyyaKernel) ||
                kind == Hellinger || kind == Wasserstein1D ||
                (kind >= TotalVariation && kind != ItakuraSaito);
    if (mass) {
      auto total = value.sum(-1);
      checks.require((total.isfinite() & total.gt(0)).all(),
                     "Inputs must have positive finite row mass");
    }
  }
  if (opts.bin_positions.has_value()) {
    TORCH_CHECK(grid.dim() == 1 && grid.size(0) == x.sym_size(-1) &&
                    grid.device() == x.device() &&
                    grid.scalar_type() == x.scalar_type(),
                "bin_positions must match features, dtype and device");
    auto widths = grid.diff();
    checks.require(
        (grid.isfinite().all() & (widths.isfinite() & widths.gt(0)).all()),
        "bin_positions must be finite and strictly increasing with finite "
        "gaps");
  }
  checks.finish();
  return detail::vector::_vector_compare(x, y, grid, kind, pairing, base,
                                         length_scale, alpha, nu,
                                         opts.squared.value_or(false));
}
}  // namespace
}  // namespace torchpairwise::ops
