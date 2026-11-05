#include "builtin_comparisons.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <set>

#include "builtin_comparisons_detail.h"
#include "pairwise_metrics.h"
#include "pointset_comparisons.h"
#include "transport_comparisons.h"
#include "utils/comparison_dispatch.h"
#include "utils/tiling.h"
#include "utils/value_checks.h"
#include "vector_comparisons.h"
#include "weighted_comparisons.h"

namespace torchpairwise::ops {
namespace {
at::Tensor builtin_comparisons_composite(const at::Tensor& input,
                                         const c10::optional<at::Tensor>& other,
                                         c10::string_view raw,
                                         c10::string_view mode, bool kernel,
                                         const BuiltinComparisonOptions& o);
}

namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
using Grad = std::tuple<Tensor, Tensor, Tensor, Tensor>;
bool contains(c10::string_view name, std::initializer_list<const char*> names) {
  for (auto item : names)
    if (name == item)
      return true;
  return false;
}

Tensor basic(const Tensor& x, const Tensor& y, const Tensor& v,
             const Tensor& vi, c10::string_view name, bool kernel, double p,
             double base, int64_t degree, double gamma, double coef0,
             bool self) {
  c10::optional<Tensor> other = self ? c10::nullopt : c10::optional<Tensor>(y);
  return detail::call_named_comparison(x, other, name, kernel, p, base, degree,
                                       gamma, coef0, v, vi);
}
// Zero subgradient at an exactly zero distance, matching the named metrics.
Tensor distance_sqrt(const Tensor& value) {
  auto nonzero = value.ne(0);
  return at::where(nonzero, at::where(nonzero, value, 1.).sqrt(), 0.);
}
Tensor legacy_compare_kernel(const Tensor& x, const Tensor& y, const Tensor& v,
                             const Tensor& vi, c10::string_view name,
                             int64_t mode, bool kernel, double p, double base,
                             int64_t degree, double gamma, double coef0,
                             bool self) {
  auto batch = x.size(0), n = x.size(1), d = x.size(2);
  if (mode == 0)
    return basic(x, y, v, vi, name, kernel, p, base, degree, gamma, coef0,
                 self);
  if (mode == 1) {
    if (kernel && contains(name, {"linear", "polynomial", "sigmoid"})) {
      auto dot = (x * y).sum(-1);
      if (name == "linear")
        return dot;
      dot = dot * gamma + coef0;
      return name == "polynomial" ? dot.pow(degree) : dot.tanh();
    }
    if (name == "sqeuclidean" || (kernel && name == "rbf")) {
      auto value = (x - y).square().sum(-1);
      return kernel ? (-gamma * value).exp() : value;
    }
    if (!kernel && (name == "mahalanobis" || name == "seuclidean")) {
      // Corresponding rows share V/VI; avoid per-sample [D,D] copies.
      auto delta = x - y;
      return distance_sqrt(name == "mahalanobis"
                               ? (delta.matmul(vi) * delta).sum(-1)
                               : (delta.square() / v.unsqueeze(-2)).sum(-1));
    }
    auto result =
        basic(x.reshape({batch * n, 1, d}), y.reshape({batch * n, 1, d}), v, vi,
              name, kernel, p, base, degree, gamma, coef0, self);
    return result.reshape({batch, n});
  }
  if (n < 2)
    return basic(x.narrow(1, 0, 0), y.narrow(1, 0, 0), v, vi, name, kernel, p,
                 base, degree, gamma, coef0, false)
        .flatten(-2);
  // Row blocks against their column suffix, gathered in condensed order.
  std::vector<Tensor> parts;
  for (int64_t i = 0, count; i < n - 1; i += count) {
    auto columns = n - i - 1;
    count = utils::tile_rows(columns, 1, columns);
    auto block = basic(x.narrow(1, i, count), y.slice(1, i + 1), v, vi, name,
                       kernel, p, base, degree, gamma, coef0, false);
    parts.push_back(block.reshape({batch, count * columns})
                        .index_select(-1, utils::condensed_block_index(
                                              count, columns, x.options())));
  }
  return at::cat(parts, -1);
}
using Core = Tensor(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                    c10::string_view, int64_t, bool, double, double, int64_t,
                    double, double, bool);
using Backward = Grad(const Tensor&, const Tensor&, const Tensor&,
                      const Tensor&, const Tensor&, c10::string_view, int64_t,
                      bool, double, double, int64_t, double, double, bool);

Grad legacy_compare_backward_kernel(const Tensor& g, const Tensor& x,
                                    const Tensor& y, const Tensor& v,
                                    const Tensor& vi, c10::string_view name,
                                    int64_t mode, bool kernel, double p,
                                    double base, int64_t degree, double gamma,
                                    double coef0, bool self) {
  const bool higher = at::GradMode::is_enabled();
  if (higher) {
    TORCH_CHECK(contains(name, {"sqeuclidean", "euclidean", "manhattan",
                                "minkowski", "cosine"}) ||
                    (kernel && contains(name, {"rbf", "laplacian", "linear",
                                               "polynomial", "sigmoid"})),
                "Higher derivatives of comparison ", name, " are unsupported");
    TORCH_CHECK(name != "minkowski" || (p > 0 && std::isfinite(p)),
                "Higher derivatives of minkowski require finite p > 0");
  }
  auto gx = at::zeros_like(x), gy = at::zeros_like(y), gv = at::zeros_like(v),
       gi = at::zeros_like(vi);
  if (g.numel() == 0) {
    if (higher) {
      auto zero = g.sum() * 0;
      return {x * 0 + zero, y * 0 + zero, v * 0 + zero, vi * 0 + zero};
    }
    return {gx, gy, gv, gi};
  }
  at::AutoGradMode enable(true);
  // Replay budgeted row tiles; condensed blocks scatter into their suffix.
  auto n = x.size(1), m = y.size(1), d = x.size(2);
  int64_t offset = 0;
  for (int64_t i = 0, count; i < (mode == 2 ? n - 1 : n); i += count) {
    int64_t start = 0, extent = m;
    if (mode == 2) {
      start = i + 1;
      extent = m - start;
      count = utils::tile_rows(extent, 1, n - 1 - i);
    } else if (mode == 1) {
      count = utils::tile_rows(1, d, n - i);
      start = i;
      extent = count;
    } else {
      count = utils::tile_rows(m, 1, n - i);
    }
    auto a = x.narrow(1, i, count), b = y.narrow(1, start, extent);
    if (!higher) {
      a = a.detach();
      b = b.detach();
    }
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    auto vv = v.detach().set_requires_grad(v.numel() > 0),
         iv = vi.detach().set_requires_grad(vi.numel() > 0);
    // Self cosine needs the original global diagonal, not tile-local identity.
    auto result = higher ? detail::higher_order_comparison(
                               a, b, name, kernel, p, degree, gamma, coef0, 1.,
                               1., 1.5, mode == 1)
                         : legacy_compare_kernel(
                               a, b, vv, iv, name, mode == 2 ? 0 : mode, kernel,
                               p, base, degree, gamma, coef0, false);
    if (self && name == "cosine" && !kernel) {
      if (mode == 0) {
        auto row = at::arange(i, i + count, x.options().dtype(at::kLong))
                       .unsqueeze(-1);
        auto col = at::arange(m, x.options().dtype(at::kLong)).unsqueeze(0);
        result = at::where(row.eq(col), 0., result);
      } else if (mode == 1)
        result = at::zeros_like(result) + result * 0;
    }
    Tensor grad;
    if (mode == 0)
      grad = g.narrow(1, i, count);
    else if (mode == 1)
      grad = g.narrow(-1, i, count);
    else {
      auto size = utils::condensed_block_size(n, i, count);
      grad =
          at::zeros({x.size(0), count * extent}, g.options())
              .index_copy(
                  -1, utils::condensed_block_index(count, extent, x.options()),
                  g.narrow(-1, offset, size))
              .reshape({x.size(0), count, extent});
      offset += size;
    }
    Values inputs{a, b};
    if (v.numel())
      inputs.push_back(vv);
    if (vi.numel())
      inputs.push_back(iv);
    auto gradients =
        torch::autograd::grad({result}, inputs, {grad}, higher, higher, true);
    if (gradients[0].defined())
      if (higher)
        gx = gx.index_add(
            1, at::arange(i, i + count, x.options().dtype(at::kLong)),
            gradients[0]);
      else
        gx.narrow(1, i, count).add_(gradients[0]);
    if (gradients[1].defined())
      if (higher)
        gy = gy.index_add(
            1, at::arange(start, start + extent, y.options().dtype(at::kLong)),
            gradients[1]);
      else
        gy.narrow(1, start, extent).add_(gradients[1]);
    int k = 2;
    if (v.numel()) {
      if (gradients[k].defined())
        gv.add_(gradients[k]);
      ++k;
    }
    if (vi.numel() && gradients[k].defined())
      gi.add_(gradients[k]);
  }
  return {gx, gy, gv, gi};
}
}  // namespace

namespace detail {
Tensor higher_order_comparison(const Tensor& x, const Tensor& y,
                               c10::string_view name, bool kernel, double p,
                               int64_t degree, double gamma, double coef0,
                               double length_scale, double alpha, double nu,
                               bool paired) {
  auto a = paired ? x : x.unsqueeze(-2);
  auto b = paired ? y : y.unsqueeze(-3);
  if (kernel && contains(name, {"linear", "polynomial", "sigmoid"})) {
    auto dot = (a * b).sum(-1);
    if (name == "linear")
      return dot;
    dot = gamma * dot + coef0;
    return name == "polynomial" ? dot.pow(degree) : dot.tanh();
  }
  auto delta = a - b;
  if (name == "cosine") {
    auto nx = distance_sqrt(a.square().sum(-1));
    auto ny = distance_sqrt(b.square().sum(-1));
    auto ua = at::where(nx.gt(0).unsqueeze(-1),
                        a / at::where(nx.gt(0), nx, 1.).unsqueeze(-1), 0.);
    auto ub = at::where(ny.gt(0).unsqueeze(-1),
                        b / at::where(ny.gt(0), ny, 1.).unsqueeze(-1), 0.);
    auto similarity = (ua * ub).sum(-1);
    return kernel ? similarity : (1 - similarity).clamp(0, 2);
  }
  if (name == "manhattan" || (kernel && name == "laplacian")) {
    auto value = delta.abs().sum(-1);
    return kernel ? (-gamma * value).exp() : value;
  }
  if (name == "minkowski") {
    TORCH_CHECK(p > 0 && std::isfinite(p),
                "Higher derivatives of minkowski require finite p > 0");
    auto value = delta.abs().pow(p).sum(-1);
    auto valid = value.ne(0);
    return at::where(valid, at::where(valid, value, 1.).pow(1. / p), 0.);
  }
  TORCH_CHECK(contains(name, {"sqeuclidean", "euclidean", "rbf", "matern",
                              "rational_quadratic"}),
              "Higher derivatives of comparison ", name, " are unsupported");
  auto value = delta.square().sum(-1).clamp_min(0);
  if (name == "sqeuclidean")
    return value;
  if (name == "euclidean")
    return distance_sqrt(value);
  if (name == "rbf")
    return (-gamma * value).exp();
  if (name == "rational_quadratic")
    return (-alpha * (value / length_scale / length_scale * .5 / alpha).log1p())
        .exp();
  TORCH_CHECK(nu == .5 || nu == 1.5 || nu == 2.5,
              "Higher derivatives of matern require nu 0.5, 1.5 or 2.5");
  auto r = distance_sqrt(value) / length_scale * std::sqrt(2 * nu);
  const double tiny = x.scalar_type() == at::kFloat
                          ? std::numeric_limits<float>::min()
                          : std::numeric_limits<double>::min();
  r = r.clamp_max(-4 * std::log(tiny));
  if (nu == .5)
    return (-r).exp();
  auto polynomial = 1 + r;
  if (nu == 2.5)
    polynomial = polynomial + r.square() / 3;
  return polynomial * (-r).exp();
}
}  // namespace detail

Tensor builtin_comparisons(const Tensor& input,
                           const c10::optional<Tensor>& other,
                           c10::string_view raw, c10::string_view mode,
                           bool kernel, const BuiltinComparisonOptions& o) {
  using Signature = Tensor(
      const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
      c10::string_view mode, bool is_kernel, const c10::optional<Tensor>& w,
      const c10::optional<Tensor>& V, const c10::optional<Tensor>& VI,
      const c10::optional<Tensor>& bin_positions,
      const c10::optional<Tensor>& projections,
      const c10::optional<Tensor>& x1_lengths,
      const c10::optional<Tensor>& x2_lengths, c10::optional<double> p,
      c10::optional<double> base, c10::optional<double> alpha,
      c10::optional<double> length_scale, c10::optional<double> nu,
      c10::optional<double> gamma, c10::optional<double> coef0,
      c10::optional<double> regularization, c10::optional<int64_t> degree,
      c10::optional<int64_t> max_iter, c10::optional<int64_t> chunk_size,
      c10::optional<int64_t> feature_dim, c10::optional<bool> squared,
      c10::optional<bool> shuffle, c10::optional<bool> batched,
      c10::optional<c10::string_view> kernel,
      c10::optional<at::Generator> generator);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::builtin_comparisons", "")
          .typed<Signature>();
  return op.call(
      input, other, raw, mode, kernel, o.w, o.V, o.VI, o.bin_positions,
      o.projections, o.x1_lengths, o.x2_lengths, o.p, o.base, o.alpha,
      o.length_scale, o.nu, o.gamma, o.coef0, o.regularization, o.degree,
      o.max_iter, o.chunk_size, o.feature_dim, o.squared, o.shuffle, o.batched,
      o.kernel ? c10::optional<c10::string_view>(*o.kernel) : c10::nullopt,
      o.generator);
}
namespace {
Tensor builtin_comparisons_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view mode, bool is_kernel, const c10::optional<Tensor>& w,
    const c10::optional<Tensor>& V, const c10::optional<Tensor>& VI,
    const c10::optional<Tensor>& bin_positions,
    const c10::optional<Tensor>& projections,
    const c10::optional<Tensor>& x1_lengths,
    const c10::optional<Tensor>& x2_lengths, c10::optional<double> p,
    c10::optional<double> base, c10::optional<double> alpha,
    c10::optional<double> length_scale, c10::optional<double> nu,
    c10::optional<double> gamma, c10::optional<double> coef0,
    c10::optional<double> regularization, c10::optional<int64_t> degree,
    c10::optional<int64_t> max_iter, c10::optional<int64_t> chunk_size,
    c10::optional<int64_t> feature_dim, c10::optional<bool> squared,
    c10::optional<bool> shuffle, c10::optional<bool> batched,
    c10::optional<c10::string_view> kernel,
    c10::optional<at::Generator> generator) {
  BuiltinComparisonOptions o;
  o.w = w;
  o.V = V;
  o.VI = VI;
  o.bin_positions = bin_positions;
  o.projections = projections;
  o.x1_lengths = x1_lengths;
  o.x2_lengths = x2_lengths;
  o.p = p;
  o.base = base;
  o.alpha = alpha;
  o.length_scale = length_scale;
  o.nu = nu;
  o.gamma = gamma;
  o.coef0 = coef0;
  o.regularization = regularization;
  o.degree = degree;
  o.max_iter = max_iter;
  o.chunk_size = chunk_size;
  o.feature_dim = feature_dim;
  o.squared = squared;
  o.shuffle = shuffle;
  o.batched = batched;
  if (kernel.has_value())
    o.kernel = std::string(*kernel);
  o.generator = generator;
  return builtin_comparisons_composite(x, y, metric, mode, is_kernel, o);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "builtin_comparisons(Tensor x1, Tensor? x2, str metric, str "
      "mode='all_pairs', bool is_kernel=False, Tensor? w=None, Tensor? V=None, "
      "Tensor? VI=None, Tensor? bin_positions=None, Tensor? projections=None, "
      "Tensor? x1_lengths=None, Tensor? x2_lengths=None, float? p=None, float? "
      "base=None, float? alpha=None, float? length_scale=None, float? nu=None, "
      "float? gamma=None, float? coef0=None, float? regularization=None, int? "
      "degree=None, int? max_iter=None, int? chunk_size=None, int? "
      "feature_dim=None, bool? squared=None, bool? shuffle=None, bool? "
      "batched=None, str? kernel=None, Generator? generator=None) -> Tensor",
      TORCH_FN(builtin_comparisons_impl));
  m.def(
      "_legacy_compare(Tensor x, Tensor y, Tensor v, Tensor vi, str name, int "
      "mode, bool kernel, float p, float base, int degree, float gamma, float "
      "coef0, bool self) -> Tensor");
  m.def(
      "_legacy_compare_backward(Tensor grad, Tensor x, Tensor y, Tensor v, "
      "Tensor vi, str name, int mode, bool kernel, float p, float base, int "
      "degree, float gamma, float coef0, bool self) -> (Tensor, Tensor, "
      "Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_legacy_compare", TORCH_FN(legacy_compare_kernel));
  m.impl("_legacy_compare_backward", TORCH_FN(legacy_compare_backward_kernel));
}

}  // namespace
namespace detail::builtin {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
bool contains_name(c10::string_view name,
                   std::initializer_list<const char*> names) {
  return contains(name, names);
}
}  // namespace detail::builtin
namespace {
Tensor builtin_comparisons_composite(const Tensor& input,
                                     const c10::optional<Tensor>& other,
                                     c10::string_view raw,
                                     c10::string_view mode, bool kernel,
                                     const BuiltinComparisonOptions& o) {
  auto name = detail::canonical_comparison_name(raw);
  int64_t pairing = mode == "all_pairs"   ? 0
                    : mode == "paired"    ? 1
                    : mode == "condensed" ? 2
                                          : -1;
  TORCH_CHECK(pairing >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(pairing != 2 || !other.has_value(),
              "Condensed comparisons require one input");
  TORCH_CHECK(
      pairing != 2 || kernel ||
          !contains(name, {"snr", "kl", "renyi", "pearson_chi2", "neyman_chi2",
                           "itakura_saito", "directed_hausdorff"}),
      "Condensed output requires a symmetric metric");
  auto family = detail::comparison_family(name, kernel);
  std::set<std::string> allowed;
  bool point = family == detail::ComparisonFamily::Pointset;
  bool transport = family == detail::ComparisonFamily::Transport;
  bool modern = family == detail::ComparisonFamily::Vector;
  if (point || transport) {
    allowed = {"x1_lengths", "x2_lengths", "chunk_size", "batched",
               "feature_dim"};
    if (contains(name, {"chamfer", "energy", "mmd"}))
      allowed.insert("squared");
    if (point && name != "chamfer") {
      allowed.insert("shuffle");
      allowed.insert("generator");
    }
    if (contains(name, {"wasserstein", "sliced_wasserstein", "sinkhorn"}))
      allowed.insert("p");
    if (name == "sliced_wasserstein")
      allowed.insert("projections");
    if (name == "sinkhorn") {
      allowed.insert("regularization");
      allowed.insert("max_iter");
    }
    if (name == "mmd") {
      allowed.insert("kernel");
      allowed.insert("length_scale");
      allowed.insert("gamma");
      allowed.insert("degree");
      allowed.insert("coef0");
      allowed.insert("alpha");
      allowed.insert("nu");
    }
  } else if (kernel) {
    if (contains(name, {"rbf", "laplacian", "polynomial", "sigmoid", "chi2"}))
      allowed.insert("gamma");
    if (contains(name, {"polynomial", "sigmoid"}))
      allowed.insert("coef0");
    if (name == "polynomial")
      allowed.insert("degree");
    if (contains(name, {"rational_quadratic", "matern"}))
      allowed.insert("length_scale");
    if (name == "rational_quadratic")
      allowed.insert("alpha");
    if (name == "matern")
      allowed.insert("nu");
  } else {
    if (contains(
            name,
            {"euclidean",      "sqeuclidean", "manhattan",     "chebyshev",
             "minkowski",      "wminkowski",  "cosine",        "correlation",
             "canberra",       "braycurtis",  "hamming",       "jaccard",
             "kulsinski",      "kulczynski1", "yule",          "dice",
             "rogerstanimoto", "russellrao",  "sokalmichener", "sokalsneath"}))
      allowed.insert("w");
    if (contains(name, {"minkowski", "wminkowski"}))
      allowed.insert("p");
    if (contains(name, {"jensenshannon", "kl", "symmetric_kl"}))
      allowed.insert("base");
    if (name == "seuclidean")
      allowed.insert("V");
    if (name == "mahalanobis")
      allowed.insert("VI");
    if (name == "renyi")
      allowed.insert("alpha");
    if (name == "nan_euclidean")
      allowed.insert("squared");
    if (name == "wasserstein_1d")
      allowed.insert("bin_positions");
  }
#define CHECK_OPTION(FIELD)                                  \
  TORCH_CHECK(!o.FIELD.has_value() || allowed.count(#FIELD), \
              "Unused parameter: " #FIELD)
  CHECK_OPTION(w);
  CHECK_OPTION(V);
  CHECK_OPTION(VI);
  CHECK_OPTION(bin_positions);
  CHECK_OPTION(projections);
  CHECK_OPTION(x1_lengths);
  CHECK_OPTION(x2_lengths);
  CHECK_OPTION(p);
  CHECK_OPTION(base);
  CHECK_OPTION(alpha);
  CHECK_OPTION(length_scale);
  CHECK_OPTION(nu);
  CHECK_OPTION(gamma);
  CHECK_OPTION(coef0);
  CHECK_OPTION(regularization);
  CHECK_OPTION(degree);
  CHECK_OPTION(max_iter);
  CHECK_OPTION(chunk_size);
  CHECK_OPTION(feature_dim);
  CHECK_OPTION(squared);
  CHECK_OPTION(shuffle);
  CHECK_OPTION(batched);
  CHECK_OPTION(kernel);
  CHECK_OPTION(generator);
#undef CHECK_OPTION
  if (point) {
    PointsetComparisonOptions q;
    q.squared = o.squared;
    q.x1_lengths = o.x1_lengths;
    q.x2_lengths = o.x2_lengths;
    q.chunk_size = o.chunk_size;
    q.batched = o.batched;
    q.feature_dim = o.feature_dim.value_or(-1);
    q.shuffle = o.shuffle;
    q.generator = o.generator;
    return pointset_comparisons(input, other, name, mode, q);
  }
  if (transport) {
    TransportComparisonOptions q;
    q.p = o.p;
    q.squared = o.squared;
    q.projections = o.projections;
    q.regularization = o.regularization;
    q.max_iter = o.max_iter;
    q.gamma = o.gamma;
    q.degree = o.degree;
    q.coef0 = o.coef0;
    q.alpha = o.alpha;
    q.nu = o.nu;
    q.kernel = o.kernel;
    q.length_scale = o.length_scale;
    q.x1_lengths = o.x1_lengths;
    q.x2_lengths = o.x2_lengths;
    q.chunk_size = o.chunk_size;
    q.batched = o.batched;
    q.feature_dim = o.feature_dim.value_or(-1);
    return transport_comparisons(input, other, name, mode, q);
  }
  if (modern) {
    VectorComparisonOptions q;
    q.base = o.base;
    q.length_scale = o.length_scale;
    q.alpha = o.alpha;
    q.nu = o.nu;
    q.squared = o.squared;
    q.bin_positions = o.bin_positions;
    auto selector =
        name == "bhattacharyya"
            ? (kernel ? "bhattacharyya_kernel" : "bhattacharyya_distances")
            : name;
    return vector_comparisons(input, other, selector, mode, q);
  }
  if (o.w.has_value())
    return weighted_comparisons(input, other, *o.w, name, mode, o.p);
  Tensor x = input, y = other.value_or(input);
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided,
              "Inputs must be dense strided tensors");
  TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2 &&
                  x.sym_size(-1) == y.sym_size(-1) && x.sym_size(-1) > 0,
              "Inputs must have matching nonempty features and rank >= 2");
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Inputs must have matching dtype and device");
  TORCH_CHECK(pairing != 1 || x.sym_size(-2) == y.sym_size(-2),
              "Paired inputs must have equal sample counts");
  if (!x.device().is_meta() &&
      at::autocast::is_autocast_enabled(x.device().type()) &&
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)) {
    x = x.to(at::kFloat);
    y = y.to(at::kFloat);
  }
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto v = o.V.value_or(at::empty({0}, x.options())),
       vi = o.VI.value_or(at::empty({0}, x.options()));
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 2),
                                     y.sym_sizes().slice(0, y.dim() - 2));
  if (o.V.has_value()) {
    TORCH_CHECK(v.dim() >= 1 && v.sym_size(-1) == x.sym_size(-1) &&
                    v.device() == x.device() &&
                    v.scalar_type() == x.scalar_type(),
                "V must match input features, dtype and device");
    batch = at::infer_size_symint(batch, v.sym_sizes().slice(0, v.dim() - 1));
  }
  if (o.VI.has_value()) {
    TORCH_CHECK(vi.dim() >= 2 && vi.sym_size(-1) == x.sym_size(-1) &&
                    vi.sym_size(-2) == x.sym_size(-1) &&
                    vi.device() == x.device() &&
                    vi.scalar_type() == x.scalar_type(),
                "VI must match input features, dtype and device");
    batch = at::infer_size_symint(batch, vi.sym_sizes().slice(0, vi.dim() - 2));
  }
  auto expand = [&](const Tensor& a, int64_t rank) {
    auto shape = batch;
    for (int64_t i = -rank; i < 0; ++i)
      shape.push_back(a.sym_size(i));
    return a.expand_symint(shape);
  };
  x = expand(x, 2);
  y = expand(y, 2);
  if (name == "seuclidean" && !o.V.has_value()) {
    TORCH_CHECK(!other.has_value(),
                "V is required for seuclidean when x2 is passed");
    at::NoGradGuard guard;
    v = x.var(at::IntArrayRef{-2});
  }
  if (name == "mahalanobis" && !o.VI.has_value()) {
    TORCH_CHECK(!other.has_value() && x.sym_size(-2) >= 2,
                "VI is required for cross/paired Mahalanobis; at least two "
                "self samples infer covariance");
    at::NoGradGuard guard;
    auto a = x - x.mean(-2, true);
    vi = (a.transpose(-1, -2).matmul(a) / (x.sym_size(-2) - 1))
             .inverse()
             .transpose(-1, -2);
  }
  auto d = x.sym_size(-1), n = x.sym_size(-2), m = y.sym_size(-2);
  c10::SymInt b = 1;
  for (auto s : batch)
    b *= s;
  x = x.reshape_symint({b, n, d});
  y = y.reshape_symint({b, m, d});
  if (name == "seuclidean") {
    v = expand(v, 1).reshape_symint({b, d});
    utils::check_values((v.isfinite() & v.gt(0)).all(),
                        "V must contain positive finite variances");
  }
  if (name == "mahalanobis")
    vi = expand(vi, 2).reshape_symint({b, d, d});
  double power = o.p.value_or(2.);
  TORCH_CHECK(power >= 0 && (name != "wminkowski" || power > 0),
              "p must be non-negative (positive for wminkowski)");
  TORCH_CHECK(!o.base.has_value() || *o.base > 1,
              "base must be greater than one");
  TORCH_CHECK(!o.gamma.has_value() || std::isfinite(*o.gamma),
              "gamma must be finite");
  TORCH_CHECK(!o.coef0.has_value() || std::isfinite(*o.coef0),
              "coef0 must be finite");
  // gamma's feature-dependent default remains symbolic in the public graph.
  auto g = o.gamma.value_or(
      name == "chi2" ? 1. : 1. / double(d.guard_int(__FILE__, __LINE__)));
  auto out = detail::builtin::_legacy_compare(
      x, y, v, vi, name, pairing, kernel, power, o.base.value_or(NAN),
      o.degree.value_or(3), g, o.coef0.value_or(1.),
      (!other.has_value() || (name == "cosine" && input.is_same(*other))));
  auto shape = batch;
  shape.push_back(pairing == 2 ? n * (n - 1) / 2 : n);
  if (pairing == 0)
    shape.push_back(m);
  return out.reshape_symint(shape);
}
}  // namespace
}  // namespace torchpairwise::ops
