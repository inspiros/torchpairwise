#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>

#include "builtin_comparisons.h"
#include "mmd_comparisons_detail.h"
#include "transport_comparisons.h"
#include "utils/clouds.h"
#include "utils/comparison_dispatch.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::mmd {
Tensor point_higher_order(const Tensor& x, const Tensor& y, const Tensor& xv,
                          const Tensor& yv, const KernelOptions& o);
}
namespace torchpairwise::ops {
namespace {
using at::Tensor;
using detail::mmd::KernelOptions;

Tensor lengths(const Tensor& x, const Tensor& raw) {
  return raw.numel() ? raw
                     : at::full(x.sizes().slice(0, x.dim() - 2), x.size(-2),
                                x.options().dtype(at::kLong));
}
std::tuple<Tensor, Tensor, Tensor> clean(const Tensor& x, const Tensor& len) {
  auto valid = at::arange(x.size(-2), len.options()).lt(len.unsqueeze(-1));
  TORCH_CHECK((x.isfinite() | ~valid.unsqueeze(-1)).all().item<bool>(),
              "Distribution comparison valid point coordinates must be finite");
  return {at::where(valid.unsqueeze(-1), x, x.narrow(-2, 0, 1)), valid,
          len.to(x.scalar_type())};
}
Tensor mean_pair(const Tensor& x, const Tensor& y, const Tensor& xv,
                 const Tensor& yv, const Tensor& xl, const Tensor& yl,
                 const KernelOptions& o, bool self = false,
                 bool higher_order = false) {
  auto result =
      x.narrow(-2, 0, 0).sum({-1, -2}) + y.narrow(-2, 0, 0).sum({-1, -2});
  for (int64_t i = 0; i < x.size(-2); i += o.chunk_size)
    for (int64_t j = 0; j < y.size(-2); j += o.chunk_size) {
      auto a = x.narrow(-2, i, std::min(o.chunk_size, x.size(-2) - i));
      auto b = y.narrow(-2, j, std::min(o.chunk_size, y.size(-2) - j));
      auto av = xv.narrow(-1, i, a.size(-2)), bv = yv.narrow(-1, j, b.size(-2));
      auto tile = o;
      tile.exclude_diagonal = self && o.unbiased && i == j;
      result = result + (higher_order ? detail::mmd::point_higher_order(
                                            a, b, av, bv, tile)
                         : at::GradMode::is_enabled() &&
                                 (a.requires_grad() || b.requires_grad())
                             ? detail::mmd::point_autograd(a, b, av, bv, tile)
                             : detail::mmd::point_kernel(a, b, av, bv, tile));
    }
  return self && o.unbiased ? result / xl / (xl - 1) : result / xl / yl;
}
Tensor pair_value(const Tensor& raw_x, const Tensor& raw_y, const Tensor& xl,
                  const Tensor& yl, const KernelOptions& o, bool squared) {
  auto [x, xv, al] = clean(raw_x, xl);
  auto [y, yv, bl] = clean(raw_y, yl);
  // Checked once per comparison; point kernels run once per point tile.
  if (o.name == "chi2" || o.name == "additive_chi2")
    TORCH_CHECK((x.ge(0).all() & y.ge(0).all()).item<bool>(),
                "Chi-squared MMD inputs must be non-negative");
  if (o.name == "linear" && !o.unbiased) {
    auto a = at::where(xv.unsqueeze(-1), x, 0.).sum(-2) / al.unsqueeze(-1);
    auto b = at::where(yv.unsqueeze(-1), y, 0.).sum(-2) / bl.unsqueeze(-1);
    auto value = at::linalg_vector_norm(a - b, 2, {-1}, false);
    return squared || o.statistic ? value.square() : value;
  }
  auto value = (mean_pair(x, x, xv, xv, al, al, o, true) +
                mean_pair(y, y, yv, yv, bl, bl, o, true) -
                2 * mean_pair(x, y, xv, yv, al, bl, o));
  if (o.statistic)
    return value;
  value = value.clamp_min(0);
  return squared ? value : detail::safe_root(value);
}
Tensor mmd_compare_common(const Tensor& x, const Tensor& y,
                          const Tensor& raw_xl, const Tensor& raw_yl,
                          c10::string_view name, int64_t mode, double gamma,
                          int64_t degree, double coef0, double scale,
                          double alpha, double nu, bool squared,
                          int64_t statistic = 0, bool higher_order = false) {
  auto shape = detail::cloud_output_shape(x, y, mode);
  auto n = x.size(-3), m = y.size(-3);
  if (statistic) {
    // Run value validation inside the opaque call, so eager CUDA callers get
    // an ordinary exception and captured graphs retain the runtime check.
    int64_t minimum = statistic == 2 ? 2 : 1;
    for (const auto& item :
         {std::make_pair(x, raw_xl), std::make_pair(y, raw_yl)})
      if (item.second.numel())
        TORCH_CHECK(
            (item.second.ge(minimum) & item.second.le(item.first.size(-2)))
                .all()
                .item<bool>(),
            statistic == 2
                ? "Unbiased MMD requires at least two valid points per cloud, "
                  "within the padded point count"
                : "Lengths must be between 1 and the padded point count");
  }
  if (!x.numel() || !y.numel() || (mode == 2 && n < 2))
    return at::empty_symint(shape, x.options());
  KernelOptions o{std::string(name), gamma, degree, coef0, scale, alpha, nu};
  o.statistic = statistic != 0;
  o.unbiased = statistic == 2;
  auto xl = lengths(x, raw_xl), yl = lengths(y, raw_yl);
  auto [a, av, al] = clean(x, xl);
  auto [b, bv, bl] = clean(y, yl);
  if (o.name == "chi2" || o.name == "additive_chi2")
    TORCH_CHECK((a.ge(0).all() & b.ge(0).all()).item<bool>(),
                "Chi-squared MMD inputs must be non-negative");
  Tensor own_a, own_b;
  if (o.name == "linear" && !o.unbiased) {
    a = at::where(av.unsqueeze(-1), a, 0.).sum(-2) / al.unsqueeze(-1);
    b = at::where(bv.unsqueeze(-1), b, 0.).sum(-2) / bl.unsqueeze(-1);
  } else {
    auto self = [&](const Tensor& points, const Tensor& valid,
                    const Tensor& count) {
      std::vector<Tensor> values;
      for (int64_t i = 0; i < points.size(-3); ++i)
        values.push_back(mean_pair(points.select(-3, i), points.select(-3, i),
                                   valid.select(-2, i), valid.select(-2, i),
                                   count.select(-1, i), count.select(-1, i), o,
                                   true, higher_order));
      return at::stack(values, -1);
    };
    own_a = self(a, av, al);
    own_b = self(b, bv, bl);
  }
  std::vector<Tensor> rows;
  for (int64_t i = 0; i < n; ++i) {
    std::vector<Tensor> columns;
    for (int64_t j = mode == 2 ? i + 1 : (mode == 1 ? i : 0);
         j < (mode == 1 ? i + 1 : m); ++j) {
      Tensor value;
      if (o.name == "linear" && !o.unbiased) {
        auto delta = a.select(-2, i) - b.select(-2, j);
        if (higher_order && o.statistic)
          value = delta.square().sum(-1);
        else {
          value = at::linalg_vector_norm(delta, 2, {-1}, false);
          if (squared || o.statistic)
            value = value.square();
        }
      } else {
        auto cross =
            mean_pair(a.select(-3, i), b.select(-3, j), av.select(-2, i),
                      bv.select(-2, j), al.select(-1, i), bl.select(-1, j), o,
                      false, higher_order);
        value = own_a.select(-1, i) + own_b.select(-1, j) - 2 * cross;
        if (!o.statistic) {
          value = value.clamp_min(0);
          if (!squared)
            value = detail::safe_root(value);
        }
      }
      columns.push_back(value);
    }
    if (!columns.empty())
      rows.push_back(at::stack(columns, -1));
  }
  return mode == 0 ? at::stack(rows, -2) : at::cat(rows, -1);
}
std::tuple<Tensor, Tensor> mmd_compare_backward_common(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& raw_xl, const Tensor& raw_yl, c10::string_view name,
    int64_t mode, double gamma, int64_t degree, double coef0, double scale,
    double alpha, double nu, bool squared, int64_t statistic = 0) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU),
      no_cuda_autocast(c10::DispatchKey::AutocastCUDA);
  if (at::GradMode::is_enabled()) {
    if (!gradient.numel())
      return {x * 0., y * 0.};
    TORCH_CHECK(statistic,
                "Higher derivatives of MMD distances are unsupported");
    auto value =
        mmd_compare_common(x, y, raw_xl, raw_yl, name, mode, gamma, degree,
                           coef0, scale, alpha, nu, squared, statistic, true);
    std::vector<Tensor> inputs;
    if (x.requires_grad())
      inputs.push_back(x);
    if (y.requires_grad() && !x.is_same(y))
      inputs.push_back(y);
    auto gradients =
        torch::autograd::grad({value}, inputs, {gradient}, true, true, true);
    Tensor gx = x * 0., gy = y * 0.;
    size_t slot = 0;
    if (x.requires_grad()) {
      if (gradients[slot].defined())
        gx = gx + gradients[slot];
      ++slot;
    }
    if (y.requires_grad() && !x.is_same(y) && gradients[slot].defined())
      gy = gy + gradients[slot];
    return {gx, gy};
  }
  auto gx = at::zeros(x.sizes(), x.options().dtype(at::kDouble));
  auto gy = at::zeros(y.sizes(), y.options().dtype(at::kDouble));
  if (!gradient.numel())
    return {gx.to(x.scalar_type()), gy.to(y.scalar_type())};
  KernelOptions o{std::string(name), gamma, degree, coef0, scale, alpha, nu};
  o.statistic = statistic != 0;
  o.unbiased = statistic == 2;
  auto xl = lengths(x, raw_xl), yl = lengths(y, raw_yl);
  int64_t offset = 0, n = x.size(-3), m = y.size(-3);
  at::AutoGradMode enable(true);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = mode == 2 ? i + 1 : (mode == 1 ? i : 0);
         j < (mode == 1 ? i + 1 : m); ++j) {
      auto a = x.select(-3, i).detach().set_requires_grad(true);
      auto b = y.select(-3, j).detach().set_requires_grad(true);
      auto value =
          pair_value(a, b, xl.select(-1, i), yl.select(-1, j), o, squared);
      auto g = mode == 0
                   ? gradient.select(-2, i).select(-1, j)
                   : gradient.select(-1, mode == 1 ? i : offset + j - i - 1);
      auto grads =
          torch::autograd::grad({value}, {a, b}, {g}, false, false, true);
      if (grads[0].defined())
        gx.select(-3, i).add_(grads[0].to(at::kDouble));
      if (grads[1].defined())
        gy.select(-3, j).add_(grads[1].to(at::kDouble));
    }
    if (mode == 2)
      offset += n - i - 1;
  }
  return {gx.to(x.scalar_type()), gy.to(y.scalar_type())};
}
template <bool Statistic>
Tensor mmd_compare_kernel(const Tensor& x, const Tensor& y, const Tensor& xl,
                          const Tensor& yl, c10::string_view name, int64_t mode,
                          double gamma, int64_t degree, double coef0,
                          double scale, double alpha, double nu, bool flag) {
  utils::DeferredValueChecks checks;
  auto result =
      mmd_compare_common(x, y, xl, yl, name, mode, gamma, degree, coef0, scale,
                         alpha, nu, flag, Statistic ? (flag ? 2 : 1) : 0);
  checks.finish();
  return result;
}
template <bool Statistic>
std::tuple<Tensor, Tensor> mmd_compare_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y, const Tensor& xl,
    const Tensor& yl, c10::string_view name, int64_t mode, double gamma,
    int64_t degree, double coef0, double scale, double alpha, double nu,
    bool flag) {
  utils::DeferredValueChecks checks;
  auto result = mmd_compare_backward_common(
      gradient, x, y, xl, yl, name, mode, gamma, degree, coef0, scale, alpha,
      nu, flag, Statistic ? (flag ? 2 : 1) : 0);
  checks.finish();
  return result;
}
std::vector<Tensor> prepare_operator_features(const Tensor& raw,
                                              const Tensor& len,
                                              const KernelOptions& o) {
  if (raw.numel() == 0) {
    auto shape = raw.sizes().slice(0, raw.dim() - 2).vec();
    if (o.name == "linear" && !o.unbiased) {
      shape.push_back(raw.size(-1));
      return {at::empty(shape, raw.options()), at::empty({0}, raw.options())};
    }
    shape.push_back(1);
    return {raw.clone(), at::empty(shape, raw.options())};
  }
  auto minimum = o.unbiased ? 2 : 1;
  utils::check_values(
      (len.ge(minimum) & len.le(raw.size(-2))).all(),
      o.unbiased ? "Unbiased MMD requires at least two valid points per cloud, "
                   "within the padded point count"
                 : "Lengths must be between 1 and the padded point count");
  auto [x, valid, count] = clean(raw, len);
  if (o.name == "chi2" || o.name == "additive_chi2")
    utils::check_values(x.ge(0).all(),
                        "Chi-squared MMD inputs must be non-negative");
  if (o.name == "linear" && !o.unbiased)
    return {at::where(valid.unsqueeze(-1), x, 0.).sum(-2) / count.unsqueeze(-1),
            at::empty({0}, x.options())};
  std::vector<Tensor> own;
  for (int64_t i = 0; i < x.size(-3); ++i) {
    auto a = x.select(-3, i), v = valid.select(-2, i), c = count.select(-1, i);
    own.push_back(mean_pair(a, a, v, v, c, c, o, true));
  }
  return {x, at::stack(own, -1).unsqueeze(-1)};
}
std::vector<Tensor> mmd_features_kernel(const Tensor& x, const Tensor& length,
                                        c10::string_view name, double gamma,
                                        int64_t degree, double coef0,
                                        double scale, double alpha, double nu,
                                        bool statistic, bool unbiased,
                                        int64_t chunk_size) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  KernelOptions o{std::string(name), gamma, degree, coef0, scale, alpha, nu};
  o.statistic = statistic;
  o.unbiased = unbiased;
  TORCH_CHECK(chunk_size > 0, "MMD point chunk_size must be positive");
  o.chunk_size = chunk_size;
  auto result = prepare_operator_features(x, length, o);
  for (auto& t : result)
    t = t.clone();
  return result;
}
Tensor mmd_features_backward_kernel(at::TensorList gradients, const Tensor& x,
                                    const Tensor& length, c10::string_view name,
                                    double gamma, int64_t degree, double coef0,
                                    double scale, double alpha, double nu,
                                    bool statistic, bool unbiased,
                                    int64_t chunk_size) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of MMD features are unsupported");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  if (!x.numel())
    return at::zeros_like(x).contiguous();
  at::AutoGradMode enable(true);
  auto raw = x.detach().set_requires_grad(true);
  KernelOptions o{std::string(name), gamma, degree, coef0, scale, alpha, nu};
  o.statistic = statistic;
  o.unbiased = unbiased;
  TORCH_CHECK(chunk_size > 0, "MMD point chunk_size must be positive");
  o.chunk_size = chunk_size;
  auto features = prepare_operator_features(raw, length, o);
  std::vector<Tensor> outputs, cotangents;
  for (size_t k = 0; k < features.size(); ++k)
    if (features[k].requires_grad() && features[k].numel()) {
      outputs.push_back(features[k]);
      cotangents.push_back(gradients[k]);
    }
  return torch::autograd::grad(outputs, {raw}, cotangents)[0].contiguous();
}
Tensor mmd_comparisons_common(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view mode, c10::string_view raw, bool squared, double scale,
    c10::optional<double> gamma, c10::optional<int64_t> degree,
    c10::optional<double> coef0, c10::optional<double> alpha,
    c10::optional<double> nu, const c10::optional<Tensor>& xl,
    const c10::optional<Tensor>& yl, c10::optional<int64_t> size,
    c10::optional<bool> batched, int64_t feature_dim,
    c10::optional<c10::string_view> estimator) {
  auto p = detail::mmd::prepare(input, other, mode, raw, squared, scale, gamma,
                                degree, coef0, alpha, nu, xl, yl, size, batched,
                                feature_dim, estimator);
  const auto& c = p.clouds;
  return p.options.statistic
             ? detail::mmd::_mmd_statistic(c.x, c.y, c.xl, c.yl, p.options,
                                           p.mode)
             : detail::mmd::_mmd_compare(c.x, c.y, c.xl, c.yl, p.options,
                                         p.mode, p.squared);
}
Tensor mmd_comparisons_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view mode,
    c10::string_view kernel, bool squared, double scale,
    c10::optional<double> gamma, c10::optional<int64_t> degree,
    c10::optional<double> coef0, c10::optional<double> alpha,
    c10::optional<double> nu, const c10::optional<Tensor>& xl,
    const c10::optional<Tensor>& yl, c10::optional<int64_t> size,
    c10::optional<bool> batched, int64_t feature_dim) {
  return mmd_comparisons_common(x, y, mode, kernel, squared, scale, gamma,
                                degree, coef0, alpha, nu, xl, yl, size, batched,
                                feature_dim, c10::nullopt);
}
Tensor mmd_statistics_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view mode,
    c10::string_view kernel, c10::string_view estimator, double scale,
    c10::optional<double> gamma, c10::optional<int64_t> degree,
    c10::optional<double> coef0, c10::optional<double> alpha,
    c10::optional<double> nu, const c10::optional<Tensor>& xl,
    const c10::optional<Tensor>& yl, c10::optional<int64_t> size,
    c10::optional<bool> batched, int64_t feature_dim) {
  return mmd_comparisons_common(x, y, mode, kernel, true, scale, gamma, degree,
                                coef0, alpha, nu, xl, yl, size, batched,
                                feature_dim, estimator);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_mmd_features(Tensor x, Tensor lengths, str kernel, float gamma, int "
      "degree, float coef0, float scale, float alpha, float nu, bool "
      "statistic, bool unbiased, int chunk_size=128) -> Tensor[]");
  m.def(
      "_mmd_features_backward(Tensor[] gradients, Tensor x, Tensor lengths, "
      "str kernel, float gamma, int degree, float coef0, float scale, float "
      "alpha, float nu, bool statistic, bool unbiased, int chunk_size=128) -> "
      "Tensor");
  m.def(
      "mmd_statistics(Tensor x1, Tensor? x2=None, str mode='all_pairs', str "
      "kernel='rbf', str estimator='unbiased', float length_scale=1., float? "
      "gamma=None, int? degree=None, float? coef0=None, float? alpha=None, "
      "float? nu=None, Tensor? x1_lengths=None, Tensor? x2_lengths=None, int? "
      "chunk_size=None, bool? batched=None, int feature_dim=-1) -> Tensor",
      TORCH_FN(mmd_statistics_impl));
  m.def(
      "_mmd_compare.statistic(Tensor x, Tensor y, Tensor xl, Tensor yl, str "
      "kernel, int mode, float gamma, int degree, float coef0, float scale, "
      "float alpha, float nu, bool unbiased) -> Tensor");
  m.def(
      "_mmd_compare_backward.statistic(Tensor grad, Tensor x, Tensor y, Tensor "
      "xl, Tensor yl, str kernel, int mode, float gamma, int degree, float "
      "coef0, float scale, float alpha, float nu, bool unbiased) -> (Tensor, "
      "Tensor)");
  m.def(
      "mmd_comparisons(Tensor x1, Tensor? x2=None, str mode='all_pairs', str "
      "kernel='rbf', bool squared=False, float length_scale=1., float? "
      "gamma=None, int? degree=None, float? coef0=None, float? alpha=None, "
      "float? nu=None, Tensor? x1_lengths=None, Tensor? x2_lengths=None, int? "
      "chunk_size=None, bool? batched=None, int feature_dim=-1) -> Tensor",
      TORCH_FN(mmd_comparisons_impl));
  m.def(
      "_mmd_compare(Tensor x, Tensor y, Tensor xl, Tensor yl, str kernel, int "
      "mode, float gamma, int degree, float coef0, float scale, float alpha, "
      "float nu, bool squared) -> Tensor");
  m.def(
      "_mmd_compare_backward(Tensor grad, Tensor x, Tensor y, Tensor xl, "
      "Tensor yl, str kernel, int mode, float gamma, int degree, float coef0, "
      "float scale, float alpha, float nu, bool squared) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_mmd_features", TORCH_FN(mmd_features_kernel));
  m.impl("_mmd_features_backward", TORCH_FN(mmd_features_backward_kernel));
  m.impl("_mmd_compare.statistic", TORCH_FN(mmd_compare_kernel<true>));
  m.impl("_mmd_compare_backward.statistic",
         TORCH_FN(mmd_compare_backward_kernel<true>));
  m.impl("_mmd_compare", TORCH_FN(mmd_compare_kernel<false>));
  m.impl("_mmd_compare_backward", TORCH_FN(mmd_compare_backward_kernel<false>));
}
}  // namespace
Tensor mmd_comparisons(const Tensor& x, const c10::optional<Tensor>& y,
                       c10::string_view mode,
                       const TransportComparisonOptions& o) {
  TORCH_CHECK(!o.p.has_value() && !o.regularization.has_value() &&
                  !o.max_iter.has_value() && !o.projections.has_value(),
              "Unused transport parameter for MMD");
  using Signature =
      Tensor(const Tensor&, const c10::optional<Tensor>&, c10::string_view,
             c10::string_view, bool, double, c10::optional<double>,
             c10::optional<int64_t>, c10::optional<double>,
             c10::optional<double>, c10::optional<double>,
             const c10::optional<Tensor>&, const c10::optional<Tensor>&,
             c10::optional<int64_t>, c10::optional<bool>, int64_t);
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::mmd_comparisons", "")
                       .typed<Signature>();
  return op.call(x, y, mode, o.kernel.value_or("rbf"),
                 o.squared.value_or(false), o.length_scale.value_or(1.),
                 o.gamma, o.degree, o.coef0, o.alpha, o.nu, o.x1_lengths,
                 o.x2_lengths, o.chunk_size, o.batched, o.feature_dim);
}
Tensor mmd_statistics(const Tensor& x, const c10::optional<Tensor>& y,
                      c10::string_view mode, c10::string_view estimator,
                      const TransportComparisonOptions& o) {
  TORCH_CHECK(!o.p.has_value() && !o.regularization.has_value() &&
                  !o.max_iter.has_value() && !o.projections.has_value() &&
                  !o.squared.has_value(),
              "Unused parameter for MMD statistic");
  using Signature =
      Tensor(const Tensor&, const c10::optional<Tensor>&, c10::string_view,
             c10::string_view, c10::string_view, double, c10::optional<double>,
             c10::optional<int64_t>, c10::optional<double>,
             c10::optional<double>, c10::optional<double>,
             const c10::optional<Tensor>&, const c10::optional<Tensor>&,
             c10::optional<int64_t>, c10::optional<bool>, int64_t);
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::mmd_statistics", "")
                       .typed<Signature>();
  return op.call(x, y, mode, o.kernel.value_or("rbf"), estimator,
                 o.length_scale.value_or(1.), o.gamma, o.degree, o.coef0,
                 o.alpha, o.nu, o.x1_lengths, o.x2_lengths, o.chunk_size,
                 o.batched, o.feature_dim);
}
namespace detail::mmd {
Tensor point_higher_order(const Tensor& x, const Tensor& y, const Tensor& xv,
                          const Tensor& yv, const KernelOptions& o) {
  double gamma = o.gamma < 0 ? 1. / double(x.size(-1)) : o.gamma;
  if (o.name == "rbf" && o.gamma < 0)
    gamma = .5 / o.scale / o.scale;
  Tensor value;
  if (o.name == "euclidean_laplacian") {
    auto distance = detail::higher_order_comparison(x, y, "euclidean", false);
    value = (-distance / o.scale).exp();
  } else
    value =
        detail::higher_order_comparison(x, y, o.name, true, 2., o.degree, gamma,
                                        o.coef0, o.scale, o.alpha, o.nu);
  auto valid = xv.unsqueeze(-1) & yv.unsqueeze(-2);
  if (o.exclude_diagonal)
    valid =
        valid & ~at::eye(x.size(-2), y.size(-2), x.options().dtype(at::kBool));
  return at::where(valid, value, 0.).sum({-1, -2});
}
Tensor point_kernel(const Tensor& x, const Tensor& y, const Tensor& xv,
                    const Tensor& yv, const KernelOptions& o) {
  Tensor value;
  if (o.name == "euclidean_laplacian")
    value = (-at::cdist(x, y, 2, 2) / o.scale).exp();
  else {
    BuiltinComparisonOptions q;
    if (o.name == "rbf")
      q.gamma = o.gamma < 0 ? .5 / o.scale / o.scale : o.gamma;
    if (o.name == "laplacian" || o.name == "chi2") {
      if (o.gamma >= 0)
        q.gamma = o.gamma;
    }
    if (o.name == "polynomial") {
      if (o.gamma >= 0)
        q.gamma = o.gamma;
      q.degree = o.degree;
      q.coef0 = o.coef0;
    }
    if (o.name == "rational_quadratic" || o.name == "matern")
      q.length_scale = o.scale;
    if (o.name == "rational_quadratic")
      q.alpha = o.alpha;
    if (o.name == "matern")
      q.nu = o.nu;
    value = builtin_comparisons(x, y, o.name, "all_pairs", true, q);
  }
  auto valid = xv.unsqueeze(-1) & yv.unsqueeze(-2);
  if (o.exclude_diagonal)
    valid =
        valid & ~at::eye(x.size(-2), y.size(-2), x.options().dtype(at::kBool));
  return at::where(valid, value, 0.).sum({-1, -2});
}
}  // namespace detail::mmd
}  // namespace torchpairwise::ops

namespace torchpairwise::ops::detail::mmd {
PreparedInputs prepare(const Tensor& input, const c10::optional<Tensor>& other,
                       c10::string_view mode, c10::string_view raw,
                       bool squared, double scale, c10::optional<double> gamma,
                       c10::optional<int64_t> degree,
                       c10::optional<double> coef0, c10::optional<double> alpha,
                       c10::optional<double> nu,
                       const c10::optional<Tensor>& xl,
                       const c10::optional<Tensor>& yl,
                       c10::optional<int64_t> size, c10::optional<bool> batched,
                       int64_t feature_dim,
                       c10::optional<c10::string_view> estimator) {
  int64_t pairing = mode == "all_pairs"   ? 0
                    : mode == "paired"    ? 1
                    : mode == "condensed" ? 2
                                          : -1;
  TORCH_CHECK(pairing >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(pairing != 2 || (!other.has_value() && !yl.has_value()),
              "Condensed point sets require one input and x1_lengths only");
  TORCH_CHECK(!size.has_value() || *size > 0, "chunk_size must be positive");
  TORCH_CHECK(std::isfinite(scale) && scale > 0,
              "length_scale must be a positive finite scalar");
  // Preserve the original Euclidean Laplacian spelling. The full registry
  // name deliberately selects its Manhattan-distance kernel instead.
  auto name = raw == "laplacian" || raw == "euclidean_laplacian"
                  ? std::string("euclidean_laplacian")
                  : detail::canonical_comparison_name(raw);
  if (name != "euclidean_laplacian")
    detail::comparison_family(name, true);
  TORCH_CHECK(name != "sigmoid",
              "sigmoid is not generally positive semidefinite and cannot "
              "define an MMD distance");
  auto uses_gamma = name == "rbf" || name == "laplacian" ||
                    name == "polynomial" || name == "chi2";
  TORCH_CHECK(!gamma.has_value() || uses_gamma, "Unused parameter: gamma");
  TORCH_CHECK(!degree.has_value() || name == "polynomial",
              "Unused parameter: degree");
  TORCH_CHECK(!coef0.has_value() || name == "polynomial",
              "Unused parameter: coef0");
  TORCH_CHECK(!alpha.has_value() || name == "rational_quadratic",
              "Unused parameter: alpha");
  TORCH_CHECK(!nu.has_value() || name == "matern", "Unused parameter: nu");
  TORCH_CHECK(!gamma.has_value() || (std::isfinite(*gamma) && *gamma >= 0),
              "MMD gamma must be finite and non-negative");
  TORCH_CHECK(degree.value_or(3) >= 0,
              "MMD polynomial degree must be non-negative");
  TORCH_CHECK(std::isfinite(coef0.value_or(1.)) && coef0.value_or(1.) >= 0,
              "MMD polynomial coef0 must be finite and non-negative");
  TORCH_CHECK(std::isfinite(alpha.value_or(1.)) && alpha.value_or(1.) > 0,
              "alpha must be a positive finite scalar");
  TORCH_CHECK(nu.value_or(1.5) == .5 || nu.value_or(1.5) == 1.5 ||
                  nu.value_or(1.5) == 2.5,
              "nu must be 0.5, 1.5 or 2.5");
  auto x = input, y = other.value_or(input);
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "MMD supports CPU, CUDA and metadata execution");
  if (!x.device().is_meta() &&
      at::autocast::is_autocast_enabled(x.device().type()) &&
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)) {
    x = x.to(at::kFloat);
    y = y.to(at::kFloat);
  }
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU),
      no_cuda_autocast(c10::DispatchKey::AutocastCUDA);
  TORCH_CHECK(
      x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
      "Distribution comparisons require float32 or float64 outside autocast");
  auto clouds = detail::cloud_inputs(
      x, other.has_value() ? c10::optional<Tensor>(y) : c10::nullopt, xl, yl,
      batched, feature_dim, pairing, estimator.has_value());
  TORCH_CHECK(!estimator.has_value() || *estimator == "biased" ||
                  *estimator == "unbiased",
              "estimator must be biased or unbiased");
  bool unbiased = estimator.has_value() && *estimator == "unbiased";
  if (unbiased) {
    for (const auto& item :
         {std::make_pair(clouds.x, xl.has_value()),
          std::make_pair(clouds.y, yl.has_value() || (!other.has_value() &&
                                                      xl.has_value()))}) {
      if (!item.second)
        TORCH_CHECK(
            item.first.sym_size(-2) >= 2 || item.first.sym_size(-3) == 0,
            "Unbiased MMD requires at least two valid points per cloud");
    }
  }
  KernelOptions o{
      name,  gamma.value_or(-1.), degree.value_or(3), coef0.value_or(1.),
      scale, alpha.value_or(1.),  nu.value_or(1.5)};
  o.statistic = estimator.has_value();
  o.unbiased = unbiased;
  return {std::move(clouds), std::move(o), pairing, squared};
}
}  // namespace torchpairwise::ops::detail::mmd

namespace torchpairwise::ops::detail::mmd {
Tensor prepared_pair(const Tensor& x, const Tensor& y, const Tensor& xl,
                     const Tensor& yl, const Tensor& ax, const Tensor& by,
                     const KernelOptions& options, bool paired, bool squared,
                     int64_t chunk_size) {
  auto o = options;
  o.chunk_size = chunk_size;
  auto a =
      paired ? x : x.unsqueeze(-(o.name == "linear" && !o.unbiased ? 2 : 3));
  auto b =
      paired ? y : y.unsqueeze(-(o.name == "linear" && !o.unbiased ? 3 : 4));
  if (o.name == "linear" && !o.unbiased) {
    auto value = at::linalg_vector_norm(a - b, 2, {-1}, false);
    return squared || o.statistic ? value.square() : value;
  }
  auto al = paired ? xl : xl.unsqueeze(-1), bl = paired ? yl : yl.unsqueeze(-2);
  auto av = at::arange(a.size(-2), al.options()).lt(al.unsqueeze(-1));
  auto bv = at::arange(b.size(-2), bl.options()).lt(bl.unsqueeze(-1));
  auto cross = mean_pair(a, b, av, bv, al.to(a.scalar_type()),
                         bl.to(b.scalar_type()), o);
  auto sa = ax.squeeze(-1), sb = by.squeeze(-1);
  if (!paired) {
    sa = sa.unsqueeze(-1);
    sb = sb.unsqueeze(-2);
  }
  auto value = sa + sb - 2 * cross;
  if (o.statistic)
    return value;
  value = value.clamp_min(0);
  return squared ? value : detail::safe_root(value);
}
}  // namespace torchpairwise::ops::detail::mmd
