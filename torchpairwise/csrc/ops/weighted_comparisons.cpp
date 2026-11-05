#include "weighted_comparisons.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>

#include "ppminkowski.h"
#include "utils/tiling.h"
#include "utils/value_checks.h"
#include "weighted_comparisons_detail.h"
#include "wminkowski.h"

namespace torchpairwise::ops {
namespace {
at::Tensor weighted_comparisons_impl(const at::Tensor& input,
                                     const c10::optional<at::Tensor>& other,
                                     const at::Tensor& weights,
                                     c10::string_view metric,
                                     c10::string_view mode,
                                     c10::optional<double> exponent);
}

namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
using Triple = std::array<Tensor, 3>;
enum Kind : int64_t {
  Euclidean,
  Squared,
  Manhattan,
  Minkowski,
  Chebyshev,
  Canberra,
  BrayCurtis,
  Cosine,
  Correlation,
  Hamming,
  Jaccard,
  Kulsinski,
  Kulczynski1,
  Yule,
  Dice,
  RogersTanimoto,
  RussellRao,
  SokalMichener,
  SokalSneath,
  BrayDenominator
};
constexpr int64_t feature_tile = 64;
bool matrix_formula(int64_t kind, double p) {
  // These formulas use matrix products or existing native reductions, with
  // no N*M*D intermediate. Preserve their full-size optimized operations.
  return kind == Cosine || kind == Correlation || kind >= Jaccard ||
         (kind <= Minkowski && kind != Squared) || kind == Chebyshev ||
         std::isinf(p);
}
// Budget width per pair: feature formulas hold several float64 pair-feature
// temporaries for one feature tile.
int64_t tile_width(int64_t kind, double p, int64_t features) {
  return matrix_formula(kind, p) ? 1 : 4 * std::min(features, feature_tile);
}

Tensor ratio(const Tensor& a, const Tensor& b) {
  auto nz = b.ne(0);
  return at::where(nz, a / at::where(nz, b, 1.), 0.);
}
Tensor power(const Tensor& x, double p) {
  auto nz = x.ne(0);
  return at::where(nz, at::where(nz, x, 1.).pow(p), 0.);
}

// Elementwise formulas replay at most one feature tile during first backward.
Tensor feature(const Tensor& x, const Tensor& y, const Tensor& w, int64_t kind,
               bool paired, double p) {
  auto a = paired ? x : x.unsqueeze(-2), b = paired ? y : y.unsqueeze(-3);
  auto weight = paired ? w.unsqueeze(-2) : w.unsqueeze(-2).unsqueeze(-2);
  if (kind == BrayCurtis || kind == BrayDenominator) {
    a = a * weight;
    b = b * weight;
    return (kind == BrayCurtis ? (a - b).abs() : (a + b).abs()).sum(-1);
  }
  if (kind == Hamming)
    return (a.ne(b).to(w.scalar_type()) * weight).sum(-1);
  if (kind == Canberra)
    return (ratio((a - b).abs(), a.abs() + b.abs()) * weight).sum(-1);
  return (power((a - b).abs(), p) * weight).sum(-1);
}

Tensor rows(const Tensor& raw, const Tensor& w, bool centered) {
  auto weight = w.unsqueeze(-2);
  auto x = at::where(weight.gt(0), raw, 0.);
  auto scale = x.abs().amax(-1, true);
  x = x / at::where(scale.gt(0), scale, 1.);
  if (centered) {
    auto pivot = w.argmax(-1, true).unsqueeze(-2);
    auto shape = x.sym_sizes().vec();
    shape.back() = 1;
    auto shifted = x - x.gather(-1, pivot.expand_symint(shape));
    x = shifted - (shifted * weight).sum(-1, true);
  }
  auto squared = (x.square() * weight).sum(-1, true);
  if (centered)
    return x / squared.sqrt();
  auto valid = squared.gt(0);
  return at::where(valid, x / at::where(valid, squared, 1.).sqrt(), 0.);
}

Tensor block(const Tensor& raw_x, const Tensor& raw_y, const Tensor& raw_w,
             int64_t kind, bool paired, double p) {
  Tensor w = raw_w;
  auto support = w.gt(0).unsqueeze(-2);
  // A floating scalar zero would promote integer/bool Hamming operands and
  // could erase differences between large, exactly represented integers.
  auto x = at::where(support, raw_x, at::zeros_like(raw_x));
  auto y = at::where(support, raw_y, at::zeros_like(raw_y));
  if (kind == Cosine || kind == Correlation) {
    w = w / w.amax(-1, true);
    w = w / w.sum(-1, true);
    auto a = rows(x, w, kind == Correlation),
         b = rows(y, w, kind == Correlation);
    auto value = paired ? (a * w.unsqueeze(-2) * b).sum(-1)
                        : (a * w.unsqueeze(-2)).matmul(b.transpose(-1, -2));
    return (1 - value).clamp(0, 2);
  }
  if (kind == Chebyshev || std::isinf(p)) {
    // Ties split evenly and NaN propagates, as amax does for paired inputs.
    auto value = paired ? (x - y).abs().amax(-1) : [&] {
      auto n = x.size(-2), m = y.size(-2), d = x.size(-1);
      auto shape = x.sizes().slice(0, x.dim() - 2).vec();
      shape.insert(shape.end(), {n, m});
      return _ppminkowski(x.reshape({-1, n, d}), y.reshape({-1, m, d}),
                          INFINITY)
          .reshape(shape);
    }();
    auto shape = w.sizes().slice(0, w.dim() - 1).vec();
    shape.push_back(1);
    if (!paired)
      shape.push_back(1);
    return value + w.narrow(-1, 0, 0).sum(-1).reshape(shape);
  }
  if (kind <= Minkowski && kind != Squared && !paired) {
    // Reuse the original optimized CPU/CUDA primitive after broadcasting.
    auto n = x.size(-2), m = y.size(-2), d = x.size(-1);
    auto shape = x.sizes().slice(0, x.dim() - 2).vec();
    shape.insert(shape.end(), {n, m});
    return _wminkowski(x.reshape({-1, n, d}), y.reshape({-1, m, d}),
                       w.reshape({-1, d}), p)
        .reshape(shape);
  }
  if (kind == BrayCurtis) {
    w = w / w.amax(-1, true);
    // Feature tiles keep pair-feature state out of the retained graph.
    // Bray-Curtis is a ratio of two reductions, unlike additive metrics.
    Tensor numerator, denominator;
    for (int64_t k = 0; k < x.size(-1); k += feature_tile) {
      auto count = std::min(feature_tile, x.size(-1) - k);
      auto aa = x.narrow(-1, k, count), bb = y.narrow(-1, k, count),
           ww = w.narrow(-1, k, count);
      auto replay =
          at::GradMode::is_enabled() &&
          (aa.requires_grad() || bb.requires_grad() || ww.requires_grad());
      auto num = replay ? detail::weighted::feature_autograd(
                              aa, bb, ww, int64_t(BrayCurtis), paired, p)
                        : feature(aa, bb, ww, BrayCurtis, paired, p);
      auto den = replay ? detail::weighted::feature_autograd(
                              aa, bb, ww, int64_t(BrayDenominator), paired, p)
                        : feature(aa, bb, ww, BrayDenominator, paired, p);
      numerator = numerator.defined() ? numerator + num : num;
      denominator = denominator.defined() ? denominator + den : den;
    }
    return numerator / denominator;
  }
  if (kind <= Canberra || kind == Hamming) {
    if (kind == Hamming)
      w = w / w.amax(-1, true);
    Tensor total;
    for (int64_t k = 0; k < x.size(-1); k += feature_tile) {
      auto count = std::min(feature_tile, x.size(-1) - k);
      auto a = x.narrow(-1, k, count), b = y.narrow(-1, k, count),
           weight = w.narrow(-1, k, count);
      auto value = at::GradMode::is_enabled() &&
                           (a.requires_grad() || b.requires_grad() ||
                            weight.requires_grad())
                       ? detail::weighted::feature_autograd(a, b, weight, kind,
                                                            paired, p)
                       : feature(a, b, weight, kind, paired, p);
      total = total.defined() ? total + value : value;
    }
    if (kind == Hamming)
      return total / (paired ? w.sum(-1).unsqueeze(-1)
                             : w.sum(-1).unsqueeze(-1).unsqueeze(-1));
    return kind == Canberra || kind == Squared || p == 1 ? total
                                                         : power(total, 1 / p);
  }
  w = w / w.amax(-1, true);
  auto a = x.to(w.scalar_type()), b = y.to(w.scalar_type()),
       ww = w.unsqueeze(-2);
  Tensor c11, c10, c01, c00, total = w.sum(-1).unsqueeze(-1);
  if (paired) {
    c11 = (a * b * ww).sum(-1);
    c10 = (a * (1 - b) * ww).sum(-1);
    c01 = ((1 - a) * b * ww).sum(-1);
    c00 = ((1 - a) * (1 - b) * ww).sum(-1);
  } else {
    c11 = (a * ww).matmul(b.transpose(-1, -2));
    c10 = (a * ww).matmul((1 - b).transpose(-1, -2));
    c01 = ((1 - a) * ww).matmul(b.transpose(-1, -2));
    c00 = ((1 - a) * ww).matmul((1 - b).transpose(-1, -2));
    total = total.unsqueeze(-1);
  }
  auto different = c10 + c01;
  if (kind == Jaccard)
    return ratio(different, c11 + different);
  if (kind == Kulsinski)
    return (different - c11 + total) / (different + total);
  if (kind == Kulczynski1)
    return c11 / different;
  if (kind == Dice)
    return different / (2 * c11 + different);
  if (kind == RogersTanimoto || kind == SokalMichener)
    return 2 * different / (c11 + c00 + 2 * different);
  if (kind == RussellRao)
    return (total - c11) / total;
  if (kind == SokalSneath)
    return 2 * different / (c11 + 2 * different);
  return ratio(2 * c10 * c01, c11 * c00 + c10 * c01);
}

Triple broadcast(const Tensor& x, const Tensor& y, const Tensor& w) {
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 2),
                                     y.sym_sizes().slice(0, y.dim() - 2));
  batch = at::infer_size_symint(batch, w.sym_sizes().slice(0, w.dim() - 1));
  auto xs = batch, ys = batch, ws = batch;
  xs.insert(xs.end(), {x.sym_size(-2), x.sym_size(-1)});
  ys.insert(ys.end(), {y.sym_size(-2), y.sym_size(-1)});
  ws.push_back(w.sym_size(-1));
  return {x.expand_symint(xs), y.expand_symint(ys), w.expand_symint(ws)};
}
Tensor weighted_compare_kernel(const Tensor& x, const Tensor& y,
                               const Tensor& w, int64_t kind, int64_t mode,
                               double p, bool self) {
  auto expanded = broadcast(x, y, w);
  auto a = expanded[0], b = expanded[1], weight = expanded[2];
  auto n = a.size(-2), m = b.size(-2);
  auto shape = a.sizes().slice(0, a.dim() - 2).vec();
  shape.push_back(mode == 2 ? n * (n - 1) / 2 : n);
  if (mode == 0)
    shape.push_back(m);
  if (a.numel() == 0 || m == 0 || (mode == 2 && n < 2))
    return at::empty(shape, w.options());
  if (mode == 1 || (mode == 0 && matrix_formula(kind, p))) {
    auto result = block(a, b, weight, kind, mode == 1, p);
    if (self && kind == Cosine) {
      if (mode == 1)
        result.zero_();
      else
        result.diagonal(0, -2, -1).zero_();
    }
    return result;
  }
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
      n, m, mode, tile_width(kind, p, a.size(-1)),
      [&](int64_t i, int64_t count, int64_t j, int64_t other, int64_t columns) {
        if (i != current)
          flush();
        current = i;
        block_columns = columns;
        row.push_back(block(a.narrow(-2, i, count), b.narrow(-2, j, other),
                            weight, kind, false, p));
      });
  flush();
  auto output = at::cat(parts, mode == 0 ? -2 : -1);
  if (self && kind == Cosine && mode == 0)
    output.diagonal(0, -2, -1).zero_();
  return output;
}

using CoreSignature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                             int64_t, int64_t, double, bool);
using GradSignature = std::tuple<Tensor, Tensor, Tensor>(const Tensor&,
                                                         const Tensor&,
                                                         const Tensor&,
                                                         const Tensor&, int64_t,
                                                         int64_t, double, bool);

std::tuple<Tensor, Tensor, Tensor> weighted_compare_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y, const Tensor& w,
    int64_t kind, int64_t mode, double p, bool self) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda_autocast(
      c10::DispatchKey::AutocastCUDA);
  auto expanded = broadcast(x, y, w);
  auto a = expanded[0], b = expanded[1], weight = expanded[2];
  auto gx = at::zeros_like(a, a.options().dtype(at::kDouble));
  auto gy = at::zeros_like(b, b.options().dtype(at::kDouble));
  auto gw = at::zeros_like(weight, weight.options().dtype(at::kDouble));
  auto n = a.size(-2), m = b.size(-2);
  if (gradient.numel() != 0 && !(self && kind == Cosine && mode == 1)) {
    at::AutoGradMode enable(true);
    bool differentiable = kind < Hamming;
    auto replay = [&](int64_t i, int64_t count, int64_t j, int64_t other,
                      bool paired, Tensor g) {
      auto left = a.narrow(-2, i, count).detach(),
           right = b.narrow(-2, j, other).detach();
      auto ww = weight.detach().set_requires_grad(true);
      if (differentiable) {
        left.set_requires_grad(true);
        right.set_requires_grad(true);
      }
      auto value = block(left, right, ww, kind, paired, p);
      if (mode == 0 && self && kind == Cosine) {
        auto ii = at::arange(i, i + count, x.options().dtype(at::kLong))
                      .unsqueeze(-1);
        auto jj = at::arange(j, j + other, y.options().dtype(at::kLong));
        g = at::where(ii.eq(jj), 0., g);
      }
      Values inputs = differentiable ? Values{left, right, ww} : Values{ww};
      auto grads =
          torch::autograd::grad({value}, inputs, {g}, false, false, true);
      if (differentiable) {
        if (grads[0].defined())
          gx.narrow(-2, i, count).add_(grads[0].to(at::kDouble));
        if (grads[1].defined())
          gy.narrow(-2, j, other).add_(grads[1].to(at::kDouble));
      }
      if (grads.back().defined())
        gw.add_(grads.back().to(at::kDouble));
    };
    if (mode == 1 || (mode == 0 && matrix_formula(kind, p))) {
      replay(0, n, 0, mode == 1 ? n : m, mode == 1, gradient);
    } else {
      int64_t offset = 0, current = -1;
      Tensor block_gradient;
      utils::visit_tiles(
          n, m, mode, tile_width(kind, p, a.size(-1)),
          [&](int64_t i, int64_t count, int64_t j, int64_t other,
              int64_t columns) {
            if (mode == 0) {
              replay(i, count, j, other, false,
                     gradient.narrow(-2, i, count).narrow(-1, j, other));
              return;
            }
            if (i != current) {
              auto size = utils::condensed_block_size(n, i, count);
              auto flat = gradient.sizes().vec();
              flat.back() = count * columns;
              auto full = flat;
              full.back() = count;
              full.push_back(columns);
              block_gradient =
                  at::zeros(flat, gradient.options())
                      .index_copy(-1,
                                  utils::condensed_block_index(count, columns,
                                                               a.options()),
                                  gradient.narrow(-1, offset, size))
                      .reshape(full);
              offset += size;
              current = i;
            }
            replay(i, count, j, other, false,
                   block_gradient.narrow(-1, j - i - 1, other));
          });
    }
  }
  return {at::sum_to(gx, x.sizes()).to(x.scalar_type()),
          at::sum_to(gy, y.sizes()).to(y.scalar_type()),
          at::sum_to(gw, w.sizes()).to(w.scalar_type())};
}

int64_t kind_for(c10::string_view metric) {
#define KIND(NAME, VALUE)                            \
  if (metric == NAME || metric == NAME "_distances") \
  return VALUE
  KIND("euclidean", Euclidean);
  KIND("sqeuclidean", Squared);
  KIND("manhattan", Manhattan);
  KIND("minkowski", Minkowski);
  KIND("wminkowski", Minkowski);
  KIND("chebyshev", Chebyshev);
  KIND("canberra", Canberra);
  KIND("braycurtis", BrayCurtis);
  KIND("cosine", Cosine);
  KIND("correlation", Correlation);
  KIND("hamming", Hamming);
  KIND("jaccard", Jaccard);
  KIND("kulsinski", Kulsinski);
  KIND("kulczynski1", Kulczynski1);
  KIND("yule", Yule);
  KIND("dice", Dice);
  KIND("rogerstanimoto", RogersTanimoto);
  KIND("russellrao", RussellRao);
  KIND("sokalmichener", SokalMichener);
  KIND("sokalsneath", SokalSneath);
  KIND("cityblock", Manhattan);
  KIND("l1", Manhattan);
  KIND("l2", Euclidean);
  KIND("lp", Minkowski);
  KIND("linf", Chebyshev);
#undef KIND
  TORCH_CHECK(false, "Unsupported weighted comparison: ", metric);
}
}  // namespace

Tensor weighted_comparisons(const Tensor& input,
                            const c10::optional<Tensor>& other,
                            const Tensor& weights, c10::string_view metric,
                            c10::string_view mode,
                            c10::optional<double> exponent) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::weighted_comparisons", "")
          .typed<decltype(weighted_comparisons)>();
  return op.call(input, other, weights, metric, mode, exponent);
}

namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "weighted_comparisons(Tensor x1, Tensor? x2, Tensor w, str metric, str "
      "mode='all_pairs', float? p=None) -> Tensor",
      TORCH_FN(weighted_comparisons_impl));
  m.def(
      "_weighted_compare(Tensor x, Tensor y, Tensor w, int kind, int mode, "
      "float p, bool self) -> Tensor");
  m.def(
      "_weighted_compare_backward(Tensor grad, Tensor x, Tensor y, Tensor w, "
      "int kind, int mode, float p, bool self) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_weighted_compare", TORCH_FN(weighted_compare_kernel));
  m.impl("_weighted_compare_backward",
         TORCH_FN(weighted_compare_backward_kernel));
}

}  // namespace
namespace detail::weighted {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
Tensor feature_kernel(const Tensor& x, const Tensor& y, const Tensor& w,
                      int64_t kind, bool paired, double p) {
  return feature(x, y, w, kind, paired, p);
}
}  // namespace detail::weighted
namespace {
Tensor weighted_comparisons_impl(const Tensor& input,
                                 const c10::optional<Tensor>& other,
                                 const Tensor& weights, c10::string_view metric,
                                 c10::string_view mode,
                                 c10::optional<double> exponent) {
  auto kind = kind_for(metric);
  int64_t pairing = mode == "all_pairs"   ? 0
                    : mode == "paired"    ? 1
                    : mode == "condensed" ? 2
                                          : -1;
  TORCH_CHECK(pairing >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(pairing != 2 || !other.has_value(),
              "Condensed comparisons require one input");
  TORCH_CHECK(!exponent.has_value() || kind == Minkowski,
              "Unused parameter: p");
  double p = kind == Manhattan   ? 1.
             : kind == Chebyshev ? INFINITY
                                 : exponent.value_or(2.);
  TORCH_CHECK(p > 0, "Weighted Minkowski p must be positive");
  bool self = !other.has_value() || input.is_same(*other);
  Tensor x = input, y = other.value_or(input), w = weights;
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided &&
                  w.layout() == at::kStrided,
              "Inputs and weights must be dense strided tensors");
  TORCH_CHECK(x.device() == y.device() && x.device() == w.device() &&
                  x.scalar_type() == y.scalar_type(),
              "Inputs must have matching dtype and device; weights must be on "
              "the input device");
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Weighted comparisons support CPU, CUDA and metadata execution");
  if (!x.device().is_meta() &&
      at::autocast::is_autocast_enabled(x.device().type())) {
    if (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) {
      x = x.to(at::kFloat);
      y = self ? x : y.to(at::kFloat);
    }
    if (w.scalar_type() == at::kHalf || w.scalar_type() == at::kBFloat16)
      w = w.to(at::kFloat);
  }
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda_autocast(
      c10::DispatchKey::AutocastCUDA);
  TORCH_CHECK(w.scalar_type() == at::kFloat || w.scalar_type() == at::kDouble,
              "Weights must be float32 or float64 outside autocast");
  TORCH_CHECK(kind >= Hamming || x.scalar_type() == w.scalar_type(),
              "Weights must have the input dtype");
  TORCH_CHECK(kind <= Hamming || x.scalar_type() == at::kBool,
              "Boolean distances require bool inputs");
  TORCH_CHECK(kind >= Hamming || x.scalar_type() == at::kFloat ||
                  x.scalar_type() == at::kDouble,
              "Inputs must be float32 or float64 outside autocast");
  TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2 && w.dim() >= 1,
              "Inputs must have rank >= 2 and weights rank >= 1");
  TORCH_CHECK(x.sym_size(-1) > 0 && x.sym_size(-1) == y.sym_size(-1) &&
                  x.sym_size(-1) == w.sym_size(-1),
              "Inputs and weights must have matching nonempty features");
  TORCH_CHECK(pairing != 1 || x.sym_size(-2) == y.sym_size(-2),
              "Paired inputs must have equal sample counts");
  broadcast(x, y, w);
  utils::ValueChecks checks;
  checks.require((w.isfinite() & w.ge(0)).all(),
                 "w must be finite and non-negative");
  if (kind >= BrayCurtis)
    checks.require(w.amax(-1).gt(0).all(), "w must have positive total weight");
  checks.finish();
  if (kind >= Hamming) {
    x = x.detach();
    y = y.detach();
  }
  return detail::weighted::_weighted_compare(x, y, w, kind, pairing, p, self);
}
}  // namespace
}  // namespace torchpairwise::ops
