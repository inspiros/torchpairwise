#include "structured_comparisons.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <limits>

#include "structured_comparisons_detail.h"
#include "transport_solver.h"
#include "utils/eigh.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
at::Tensor gaussian_comparisons_impl(
    const at::Tensor& mean, const at::Tensor& covariance,
    const c10::optional<at::Tensor>& other_mean,
    const c10::optional<at::Tensor>& other_covariance, c10::string_view mode,
    bool squared);
at::Tensor covariance_comparisons_impl(const at::Tensor& covariance,
                                       const c10::optional<at::Tensor>& other,
                                       c10::string_view metric,
                                       c10::string_view mode, bool squared);
at::Tensor mixture_comparisons_impl(
    const at::Tensor& mean, const at::Tensor& covariance,
    const at::Tensor& weights, const c10::optional<at::Tensor>& other_mean,
    const c10::optional<at::Tensor>& other_covariance,
    const c10::optional<at::Tensor>& other_weights, c10::string_view metric,
    c10::string_view mode, const c10::optional<at::Tensor>& lengths,
    const c10::optional<at::Tensor>& other_lengths, c10::optional<bool> squared,
    c10::optional<double> regularization, c10::optional<int64_t> max_iter,
    int64_t chunk_size, c10::optional<int64_t> component_chunk_size);
}  // namespace

namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
using Collection = std::array<Tensor, 4>;
Tensor sym(const Tensor& x) {
  return x * .5 + x.transpose(-1, -2) * .5;
}
void check(const Tensor& predicate, const char* message) {
  TORCH_CHECK(predicate.all().item<bool>(), message);
}
Tensor safe_sqrt(const Tensor& value) {
  auto positive = value.gt(0);
  return at::where(positive, at::where(positive, value, 1.).sqrt(), 0.);
}
double eps(const Tensor& x) {
  return x.scalar_type() == at::kDouble ? std::numeric_limits<double>::epsilon()
                                        : std::numeric_limits<float>::epsilon();
}
void symmetric_check(const Tensor& x) {
  check(x.isfinite(), "Covariances must be finite");
  auto tolerance = 64 * x.size(-1) * eps(x) * x.abs().amax({-1, -2}, true);
  check((x - x.transpose(-1, -2)).abs().le(tolerance),
        "Covariances must be symmetric");
}

Tensor whiten(const Tensor& factor, const Tensor& value) {
  auto left = at::linalg_solve_triangular(factor, value, false);
  return sym(at::linalg_solve_triangular(factor, left.transpose(-1, -2), false)
                 .transpose(-1, -2));
}
Tensor logcosh(const Tensor& x) {
  auto magnitude = x.abs();
  auto near = (2 * (magnitude.clamp_max(1) * .5).sinh().square()).log1p();
  auto far = magnitude + (-2 * magnitude).exp().log1p() - std::log(2.);
  return at::where(magnitude.le(1), near, far);
}

Tensor root(const Tensor& x) {
  symmetric_check(x);
  return detail::structured::matrix_root_autograd(x);
}
Tensor gaussian(const Tensor& am, const Tensor& ar, const Tensor& bm,
                const Tensor& br, bool squared) {
  auto value =
      (am - bm).square().sum(-1) + detail::structured::bures_autograd(ar, br);
  return squared ? value : safe_sqrt(value);
}
Tensor cov_pair(const Tensor& a, const Tensor& ta, const Tensor& b,
                const Tensor& tb, int64_t kind, bool squared) {
  Tensor value;
  if (kind == 1) {
    auto residual = ta - tb;
    residual = at::where(a.eq(b).all(-1).all(-1).unsqueeze(-1).unsqueeze(-1),
                         0., residual);
    value = residual.square().sum({-1, -2});
  } else
    value = detail::structured::congruence_autograd(a, b, ta, kind);
  return squared ? value : safe_sqrt(value);
}
Tensor cov_transform(const Tensor& x, int64_t kind) {
  if (!x.numel())
    return x;
  if (kind == 1)
    return detail::structured::matrix_log_autograd(x);
  at::NoGradGuard guard;
  auto [factor, info] = at::linalg_cholesky_ex(x);
  check(info.eq(0) & factor.isfinite().all(-1).all(-1),
        "Covariances must be strictly positive definite with finite Cholesky "
        "factors");
  return factor;
}
Collection collection(Tensor mean, Tensor covariance, Tensor weights,
                      const Tensor& lengths, bool density) {
  auto k = mean.size(-2), d = mean.size(-1);
  if (lengths.numel())
    check(lengths.ge(1) & lengths.le(k),
          "Mixture lengths must be between 1 and the component count");
  auto valid = lengths.numel()
                   ? at::arange(k, lengths.options()).lt(lengths.unsqueeze(-1))
                   : at::ones_like(weights, weights.options().dtype(at::kBool));
  mean = at::where(valid.unsqueeze(-1), mean, 0.);
  covariance = at::where(valid.unsqueeze(-1).unsqueeze(-1), covariance,
                         at::eye(d, mean.options()));
  weights = at::where(valid, weights, 0.);
  check(mean.isfinite(), "Mixture valid means must be finite");
  check(weights.isfinite() & weights.ge(0),
        "Mixture valid weights must be finite and non-negative");
  auto scale = weights.amax(-1, true);
  check(scale.gt(0), "Each mixture must have positive total weight");
  weights = weights / scale;
  weights = weights / weights.sum(-1, true);
  if (density) {
    symmetric_check(covariance);
    covariance = sym(covariance);
    auto [factor, info] = at::linalg_cholesky_ex(covariance);
    check(info.eq(0), "Density mixture covariances must be positive definite");
  } else
    covariance = root(covariance);
  auto positive = weights.gt(0);
  auto logs =
      at::where(positive, at::where(positive, weights, 1.).log(), -INFINITY);
  return {mean, covariance, weights, logs};
}
Collection slice(Collection c, int64_t start, int64_t count) {
  for (int i = 0; i < 4; ++i)
    c[i] = c[i].narrow(i == 0 ? -3 : i == 1 ? -4 : -2, start, count);
  return c;
}
Collection align(Collection c, bool rows) {
  std::array<int64_t, 4> axes = rows ? std::array<int64_t, 4>{-3, -4, -2, -2}
                                     : std::array<int64_t, 4>{-4, -5, -3, -3};
  for (int i = 0; i < 4; ++i)
    c[i] = c[i].unsqueeze(axes[i]);
  return c;
}
Tensor overlap(const Collection& a, const Collection& b, int64_t tile) {
  std::vector<Tensor> terms;
  double floor = a[0].scalar_type() == at::kDouble
                     ? std::numeric_limits<double>::lowest()
                     : std::numeric_limits<float>::lowest();
  auto k = a[0].size(-2), l = b[0].size(-2), d = a[0].size(-1);
  for (int64_t i = 0; i < k; i += tile)
    for (int64_t j = 0; j < l; j += tile) {
      auto ni = std::min(tile, k - i), nj = std::min(tile, l - j);
      auto delta = a[0].narrow(-2, i, ni).unsqueeze(-2) -
                   b[0].narrow(-2, j, nj).unsqueeze(-3);
      auto covariance = a[1].narrow(-3, i, ni).unsqueeze(-3) +
                        b[1].narrow(-3, j, nj).unsqueeze(-4);
      auto factor = at::linalg_cholesky(covariance);
      auto white =
          at::linalg_solve_triangular(factor, delta.unsqueeze(-1), false)
              .squeeze(-1);
      auto log_density = -.5 * d * std::log(2 * std::acos(-1.)) -
                         factor.diagonal(0, -2, -1).log().sum(-1) -
                         .5 * white.square().sum(-1);
      check(log_density.isfinite(),
            "Mixture log overlaps must be finite; rescale inputs");
      auto active = a[2].narrow(-1, i, ni).unsqueeze(-1).gt(0) &
                    b[2].narrow(-1, j, nj).unsqueeze(-2).gt(0);
      auto al = a[3].narrow(-1, i, ni).unsqueeze(-1),
           bl = b[3].narrow(-1, j, nj).unsqueeze(-2);
      auto weighted = log_density + at::where(al.isfinite(), al, 0.) +
                      at::where(bl.isfinite(), bl, 0.);
      auto term =
          at::logsumexp(at::where(active, weighted, floor).flatten(-2), {-1});
      terms.push_back(at::where(active.any(-1).any(-1), term, floor));
    }
  return at::logsumexp(at::stack(terms, -1), {-1});
}
Tensor ground(const Collection& a, const Collection& b) {
  auto value = gaussian(a[0].unsqueeze(-2), a[1].unsqueeze(-3),
                        b[0].unsqueeze(-3), b[1].unsqueeze(-4), true);
  check(value.isfinite(),
        "Mixture Gaussian W2 costs must be finite; rescale inputs");
  return value;
}
Tensor entropic(const Collection& a, const Collection& b, double epsilon,
                int64_t iterations) {
  auto cost = ground(a, b) / epsilon;
  check(cost.isfinite(),
        "Mixture costs / regularization must be finite; rescale inputs");
  auto f = at::zeros_like(a[3]), g = at::zeros_like(b[3]);
  for (int64_t i = 0; i < iterations; ++i) {
    auto nf = -at::logsumexp(b[3].unsqueeze(-2) + g.unsqueeze(-2) - cost, {-1});
    auto ng = -at::logsumexp(a[3].unsqueeze(-1) + f.unsqueeze(-1) - cost, {-2});
    f = at::where(a[3].isfinite(), (f + nf) * .5, 0.);
    g = at::where(b[3].isfinite(), (g + ng) * .5, 0.);
  }
  auto plan = a[3].unsqueeze(-1) + b[3].unsqueeze(-2) + f.unsqueeze(-1) +
              g.unsqueeze(-2) - cost;
  auto correction = -at::logsumexp(plan.flatten(-2), {-1}).expm1();
  return epsilon *
         ((a[3].exp() * f).sum(-1) + (b[3].exp() * g).sum(-1) + correction);
}
std::vector<double> doubles(const Tensor& value) {
  auto cpu = value.detach().to(at::kCPU, at::kDouble).contiguous();
  return std::vector<double>(cpu.data_ptr<double>(),
                             cpu.data_ptr<double>() + cpu.numel());
}
Tensor exact(Collection a, Collection b) {
  auto ia = at::nonzero(a[2].gt(0)).flatten(),
       ib = at::nonzero(b[2].gt(0)).flatten();
  TORCH_CHECK(ia.numel() * ib.numel() <= 4096,
              "Exact mixture transport supports at most 4096 active component "
              "pairs; use entropic transport");
  for (int i = 0; i < 4; ++i) {
    a[i] = a[i].index_select(0, ia);
    b[i] = b[i].index_select(0, ib);
  }
  auto cost = ground(a, b);
  auto plan = detail::solve_transport(cost.detach().to(at::kCPU, at::kDouble),
                                      doubles(a[2]), doubles(b[2]));
  auto row = at::tensor(plan.rows, at::TensorOptions().dtype(at::kLong))
                 .to(cost.device()),
       column = at::tensor(plan.columns, at::TensorOptions().dtype(at::kLong))
                    .to(cost.device());
  auto options = cost.options().dtype(at::kDouble);
  auto mass = at::tensor(plan.masses, options);
  auto value = (mass * cost.index({row, column}).to(at::kDouble)).sum();
  auto alpha = at::tensor(plan.alpha, options),
       beta = at::tensor(plan.beta, options);
  alpha = alpha - alpha[0];
  beta = beta - beta[0];
  auto mass_term = (alpha * a[2].to(at::kDouble)).sum() +
                   (beta * b[2].to(at::kDouble)).sum();
  check(mass_term.isfinite(),
        "Mixture transport dual must be finite; rescale inputs");
  return (value + (mass_term - mass_term.detach())).to(cost.scalar_type());
}
Tensor mixture_pair(const Collection& a, const Collection& b, int64_t kind,
                    bool squared, double epsilon, int64_t iterations,
                    int64_t tile, const Tensor& self_a = Tensor(),
                    const Tensor& self_b = Tensor()) {
  if (kind == 4) {
    auto value = overlap(a, b, tile).exp();
    check(value.isfinite(),
          "Mixture overlaps must fit the structured_compare_kernel dtype");
    return value;
  }
  if (kind == 5 || kind == 6) {
    auto cross = overlap(a, b, tile),
         aa = self_a.defined() ? self_a : overlap(a, a, tile),
         bb = self_b.defined() ? self_b : overlap(b, b, tile);
    if (kind == 6)
      return (.5 * aa + .5 * bb - cross).clamp_min(0);
    auto maximum = at::maximum(at::maximum(aa, bb), cross);
    auto scaled = ((aa - maximum).exp() + (bb - maximum).exp() -
                   2 * (cross - maximum).exp())
                      .clamp_min(0);
    auto positive = scaled.gt(0);
    auto logs = at::where(positive,
                          maximum + at::where(positive, scaled, 1.).log(), 0.);
    auto value = at::where(positive, logs.exp(), 0.);
    check(value.isfinite(),
          "Mixture L2 values must fit the structured_compare_kernel dtype");
    return squared ? value : safe_sqrt(value);
  }
  if (kind == 8)
    return entropic(a, b, epsilon, iterations);
  if (kind == 9)
    return entropic(a, b, epsilon, iterations) -
           .5 * (self_a.defined() ? self_a
                                  : entropic(a, a, epsilon, iterations)) -
           .5 * (self_b.defined() ? self_b
                                  : entropic(b, b, epsilon, iterations));
  auto value = exact(a, b);
  return squared ? value : safe_sqrt(value);
}
std::vector<Tensor> structured_features_kernel(
    const Tensor& x, const Tensor& covariance, const Tensor& weights,
    const Tensor& lengths, int64_t kind, double epsilon, int64_t iterations,
    int64_t component_tile) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto result = detail::structured::prepare_features(
      x, covariance, weights, lengths, kind, epsilon, iterations,
      component_tile);
  std::vector<Tensor> values;
  for (const auto& t : result)
    values.push_back(t.clone());
  return values;
}
std::vector<Tensor> structured_features_backward_kernel(
    at::TensorList gradients, const Tensor& x, const Tensor& covariance,
    const Tensor& weights, const Tensor& lengths, int64_t kind, double epsilon,
    int64_t iterations, int64_t component_tile) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of structured features are unsupported");
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  at::AutoGradMode enable(true);
  std::array<Tensor, 3> raw{x.detach(), covariance.detach(), weights.detach()};
  Values active, outputs, cotangents;
  std::vector<int> slots;
  for (int k = 0; k < 3; ++k)
    if (raw[k].numel()) {
      raw[k].set_requires_grad(true);
      active.push_back(raw[k]);
      slots.push_back(k);
    }
  auto features = detail::structured::prepare_features(
      raw[0], raw[1], raw[2], lengths, kind, epsilon, iterations,
      component_tile);
  for (int k = 0; k < 4; ++k)
    if (features[k].requires_grad() && features[k].numel()) {
      outputs.push_back(features[k]);
      cotangents.push_back(gradients[k]);
    }
  Values result{at::zeros_like(x), at::zeros_like(covariance),
                at::zeros_like(weights)};
  if (!outputs.empty() && !active.empty()) {
    auto grads =
        torch::autograd::grad(outputs, active, cotangents, false, false, true);
    for (size_t k = 0; k < slots.size(); ++k)
      if (grads[k].defined())
        result[slots[k]] = grads[k].contiguous();
  }
  return result;
}
Tensor structured_compare_kernel(const Tensor& x, const Tensor& xc,
                                 const Tensor& xw, const Tensor& y,
                                 const Tensor& yc, const Tensor& yw,
                                 const Tensor& xl, const Tensor& yl,
                                 int64_t kind, int64_t mode, bool squared,
                                 double epsilon, int64_t iterations,
                                 int64_t tile, int64_t component_tile) {
  auto n = x.size(1), m = y.size(1), batch = x.size(0);
  std::vector<int64_t> shape =
      mode == 0 ? std::vector<int64_t>{batch, n, m}
                : std::vector<int64_t>{batch, mode == 2 ? n * (n - 1) / 2 : n};
  Tensor a = x, b = y, ta, tb;
  Collection aa, bb;
  if (kind == 0) {
    check(x.isfinite(), "Gaussian means must be finite");
    check(y.isfinite(), "Gaussian means must be finite");
    ta = root(xc);
    tb = root(yc);
  } else if (kind < 4) {
    symmetric_check(a);
    symmetric_check(b);
    a = sym(a);
    b = sym(b);
    ta = cov_transform(a, kind);
    tb = cov_transform(b, kind);
  } else {
    aa = collection(x, xc, xw, xl, kind <= 6);
    bb = collection(y, yc, yw, yl, kind <= 6);
  }
  // Validate and transform both operands before handling an empty result.
  // A nonempty operand remains subject to its domain even when the opposite
  // collection is empty or condensed output contains no comparison pairs.
  if (!batch || !n || !m || (mode == 2 && n < 2))
    return at::zeros(shape, x.options());
  if (kind == 7) {
    std::vector<Tensor> values;
    for (int64_t bi = 0; bi < batch; ++bi)
      for (int64_t i = 0; i < (mode == 2 ? n - 1 : n); ++i)
        for (int64_t j = mode == 2   ? i + 1
                         : mode == 1 ? i
                                     : 0;
             j < (mode == 1 ? i + 1 : m); ++j) {
          Collection left, right;
          for (int k = 0; k < 4; ++k) {
            left[k] = aa[k][bi][i];
            right[k] = bb[k][bi][j];
          }
          values.push_back(mixture_pair(left, right, kind, squared, epsilon,
                                        iterations, component_tile));
        }
    return at::stack(values).reshape(shape);
  }
  std::vector<Tensor> rows;
  for (int64_t i = 0; i < (mode == 2 ? n - 1 : n); i += mode == 2 ? 1 : tile) {
    int64_t count = std::min(mode == 2 ? int64_t(1) : tile, n - i);
    std::vector<Tensor> columns;
    for (int64_t j = mode == 2   ? i + 1
                     : mode == 1 ? i
                                 : 0;
         j < (mode == 1 ? i + 1 : m); j += tile) {
      int64_t other = mode == 1 ? count : std::min(tile, m - j);
      Tensor value;
      if (kind < 4) {
        auto left = a.narrow(1, i, count), right = b.narrow(1, j, other);
        auto lt = ta.narrow(1, i, count), rt = tb.narrow(1, j, other);
        if (mode != 1) {
          left = left.unsqueeze(2);
          right = right.unsqueeze(1);
          lt = lt.unsqueeze(2);
          rt = rt.unsqueeze(1);
        }
        value = kind == 0 ? gaussian(left, lt, right, rt, squared)
                          : cov_pair(left, lt, right, rt, kind, squared);
      } else {
        auto left = slice(aa, i, count), right = slice(bb, j, other);
        if (mode != 1) {
          left = align(left, true);
          right = align(right, false);
        }
        value = mixture_pair(left, right, kind, squared, epsilon, iterations,
                             component_tile);
      }
      columns.push_back(value);
    }
    if (!columns.empty()) {
      auto row = at::cat(columns, -1);
      rows.push_back(mode == 2 ? row.squeeze(1) : row);
    }
  }
  return at::cat(rows, mode == 0 ? -2 : -1);
}
using GradResult = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
using Signature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, const Tensor&, const Tensor&,
                         const Tensor&, const Tensor&, int64_t, int64_t, bool,
                         double, int64_t, int64_t, int64_t);
using GradSignature = GradResult(const Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&,
                                 int64_t, int64_t, bool, double, int64_t,
                                 int64_t, int64_t);

GradResult structured_compare_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& xc, const Tensor& xw,
    const Tensor& y, const Tensor& yc, const Tensor& yw, const Tensor& xl,
    const Tensor& yl, int64_t kind, int64_t mode, bool squared, double epsilon,
    int64_t iterations, int64_t tile, int64_t component_tile) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  std::array<Tensor, 6> raw = {x, xc, xw, y, yc, yw}, result;
  for (int i = 0; i < 6; ++i)
    result[i] = at::zeros_like(raw[i]);
  if (gradient.numel()) {
    at::AutoGradMode enable(true);
    Values inputs;
    std::vector<int> positions;
    for (int i = 0; i < 6; ++i) {
      raw[i] = raw[i].detach();
      if (raw[i].numel()) {
        raw[i].set_requires_grad(true);
        inputs.push_back(raw[i]);
        positions.push_back(i);
      }
    }
    auto value = structured_compare_kernel(
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], xl, yl, kind, mode,
        squared, epsilon, iterations, tile, component_tile);
    auto gradients =
        torch::autograd::grad({value}, inputs, {gradient}, false, false, true);
    for (size_t i = 0; i < positions.size(); ++i)
      if (gradients[i].defined())
        result[positions[i]] = gradients[i];
  }
  return {result[0], result[1], result[2], result[3], result[4], result[5]};
}
int64_t pairing(c10::string_view mode, bool other) {
  int64_t result = mode == "all_pairs"   ? 0
                   : mode == "paired"    ? 1
                   : mode == "condensed" ? 2
                                         : -1;
  TORCH_CHECK(result >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(result != 1 || other, "Paired comparisons require two inputs");
  TORCH_CHECK(result != 2 || !other, "Condensed comparisons require one input");
  return result;
}
void input_check(const std::vector<Tensor>& tensors) {
  const auto& first = tensors[0];
  for (const auto& x : tensors) {
    TORCH_CHECK(x.layout() == at::kStrided,
                "Structured inputs must be dense strided tensors");
    TORCH_CHECK(
        x.device() == first.device() && x.scalar_type() == first.scalar_type(),
        "Structured inputs must have matching dtype and device");
  }
  TORCH_CHECK(first.device().is_cpu() || first.device().is_cuda() ||
                  first.device().is_meta(),
              "Structured inputs support CPU and CUDA");
  auto dtype = first.scalar_type();
  bool autocast = !first.device().is_meta() &&
                  at::autocast::is_autocast_enabled(first.device().type());
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble ||
                  (autocast && (dtype == at::kHalf || dtype == at::kBFloat16)),
              "Structured inputs require float32 or float64 outside autocast");
}
Tensor promote(Tensor x) {
  return x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16
             ? x.to(at::kFloat)
             : x;
}
std::vector<c10::SymInt> batch_shape(
    const std::vector<std::pair<Tensor, int64_t>>& values) {
  std::vector<c10::SymInt> batch;
  for (auto& v : values) {
    auto shape = v.first.sym_sizes().slice(0, v.first.dim() - v.second);
    try {
      batch = at::infer_size_symint(batch, shape);
    } catch (const c10::Error&) {
      TORCH_CHECK(false, "Structured batch dimensions are not broadcastable");
    }
  }
  return batch;
}
Tensor expand_flat(const Tensor& x, const std::vector<c10::SymInt>& batch,
                   int64_t tail) {
  auto shape = batch;
  for (int64_t i = -tail; i < 0; ++i)
    shape.push_back(x.sym_size(i));
  auto expanded = x.expand_symint(shape);
  std::vector<c10::SymInt> flat{1};
  for (auto d : batch)
    flat[0] *= d;
  flat.insert(flat.end(), shape.end() - tail, shape.end());
  return expanded.reshape_symint(flat);
}
Tensor output_shape(const Tensor& result,
                    const std::vector<c10::SymInt>& batch) {
  auto shape = batch;
  auto sizes = result.sym_sizes();
  shape.insert(shape.end(), sizes.begin() + 1, sizes.end());
  return result.reshape_symint(shape);
}
Tensor empty(const Tensor& x, bool integer = false) {
  return at::empty({0}, integer ? x.options().dtype(at::kLong) : x.options());
}

// Broadcasting a nonempty raw operand against a zero batch discards its values.
// Validate those operands through an opaque operator before expansion; the
// scalar assertion keeps this validation live in exported/compiled programs.
// Ordinary nonempty batches already validate inside the comparison core.
Tensor validate_raw(const Tensor& value, int64_t kind) {
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  if (kind == 0)
    check(value.isfinite(), "Gaussian means must be finite");
  else if (kind == 1)
    root(value);
  else {
    symmetric_check(value);
    cov_transform(sym(value), 2);
  }
  return at::ones({}, value.options().dtype(at::kBool));
}
Tensor validate_raw_meta(const Tensor& value, int64_t kind) {
  return at::empty({}, value.options().dtype(at::kBool));
}
void validate_erased(const std::vector<c10::SymInt>& batch,
                     const std::vector<std::pair<Tensor, int64_t>>& values) {
  c10::SymInt count = 1;
  for (const auto& size : batch)
    count *= size;
  if (count != 0)
    return;
  using RawSignature = Tensor(const Tensor&, int64_t);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_structured_validate_raw", "")
          .typed<RawSignature>();
  for (const auto& item : values)
    utils::check_values(op.call(promote(item.first), item.second),
                        "Structured raw validation failed");
}
}  // namespace

Tensor gaussian_comparisons(const Tensor& mean, const Tensor& covariance,
                            const c10::optional<Tensor>& other_mean,
                            const c10::optional<Tensor>& other_covariance,
                            c10::string_view mode, bool squared) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::gaussian_comparisons", "")
          .typed<decltype(gaussian_comparisons)>();
  return op.call(mean, covariance, other_mean, other_covariance, mode, squared);
}
Tensor covariance_comparisons(const Tensor& covariance,
                              const c10::optional<Tensor>& other,
                              c10::string_view metric, c10::string_view mode,
                              bool squared) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::covariance_comparisons", "")
          .typed<decltype(covariance_comparisons)>();
  return op.call(covariance, other, metric, mode, squared);
}
Tensor mixture_comparisons(
    const Tensor& mean, const Tensor& covariance, const Tensor& weights,
    const c10::optional<Tensor>& other_mean,
    const c10::optional<Tensor>& other_covariance,
    const c10::optional<Tensor>& other_weights, c10::string_view metric,
    c10::string_view mode, const c10::optional<Tensor>& lengths,
    const c10::optional<Tensor>& other_lengths, c10::optional<bool> squared,
    c10::optional<double> regularization, c10::optional<int64_t> max_iter,
    int64_t chunk_size, c10::optional<int64_t> component_chunk_size) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::mixture_comparisons", "")
          .typed<decltype(mixture_comparisons)>();
  return op.call(mean, covariance, weights, other_mean, other_covariance,
                 other_weights, metric, mode, lengths, other_lengths, squared,
                 regularization, max_iter, chunk_size, component_chunk_size);
}
namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "gaussian_comparisons(Tensor mean, Tensor covariance, Tensor? "
      "other_mean=None, Tensor? other_covariance=None, str mode='all_pairs', "
      "bool squared=False) -> Tensor",
      TORCH_FN(gaussian_comparisons_impl));
  m.def(
      "covariance_comparisons(Tensor covariance, Tensor? other, str metric, "
      "str mode='all_pairs', bool squared=False) -> Tensor",
      TORCH_FN(covariance_comparisons_impl));
  m.def(
      "mixture_comparisons(Tensor mean, Tensor covariance, Tensor weights, "
      "Tensor? other_mean, Tensor? other_covariance, Tensor? other_weights, "
      "str metric, str mode='all_pairs', Tensor? lengths=None, Tensor? "
      "other_lengths=None, bool? squared=None, float? regularization=None, "
      "int? max_iter=None, int chunk_size=8, int? component_chunk_size=None) "
      "-> Tensor",
      TORCH_FN(mixture_comparisons_impl));
  m.def(
      "_structured_features(Tensor x, Tensor covariance, Tensor weights, "
      "Tensor lengths, int kind, float epsilon, int iterations, int "
      "component_tile) -> Tensor[]");
  m.def(
      "_structured_features_backward(Tensor[] gradients, Tensor x, Tensor "
      "covariance, Tensor weights, Tensor lengths, int kind, float epsilon, "
      "int iterations, int component_tile) -> Tensor[]");
  m.def("_structured_validate_raw(Tensor value, int kind) -> Tensor");
  m.def(
      "_structured_compare(Tensor x, Tensor xc, Tensor xw, Tensor y, Tensor "
      "yc, Tensor yw, Tensor xl, Tensor yl, int kind, int mode, bool squared, "
      "float epsilon, int iterations, int tile, int component_tile) -> Tensor");
  m.def(
      "_structured_compare_backward(Tensor grad, Tensor x, Tensor xc, Tensor "
      "xw, Tensor y, Tensor yc, Tensor yw, Tensor xl, Tensor yl, int kind, int "
      "mode, bool squared, float epsilon, int iterations, int tile, int "
      "component_tile) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_structured_features", TORCH_FN(structured_features_kernel));
  m.impl("_structured_features_backward",
         TORCH_FN(structured_features_backward_kernel));
  m.impl("_structured_validate_raw", TORCH_FN(validate_raw));
  m.impl("_structured_compare", TORCH_FN(structured_compare_kernel));
  m.impl("_structured_compare_backward",
         TORCH_FN(structured_compare_backward_kernel));
}
TORCH_LIBRARY_IMPL(torchpairwise, Meta, m) {
  m.impl("_structured_validate_raw", TORCH_FN(validate_raw_meta));
}

}  // namespace
namespace detail::structured {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using GradResult = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
std::array<Tensor, 3> matrix_root_kernel(Tensor x) {
  auto [values, vectors] = utils::eigh(sym(x));
  auto tolerance = 64 * x.size(-1) * eps(x) * values.abs().amax(-1, true);
  check(values.ge(-tolerance),
        "Gaussian covariances must be positive semidefinite");
  auto roots = values.clamp_min(0).sqrt();
  return {at::matmul(vectors * roots.unsqueeze(-2), vectors.transpose(-1, -2)),
          roots, vectors};
}
Values matrix_root_backward_kernel(const Tensor& gradient,
                                   const Values& saved) {
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of structured comparisons are unsupported");
  auto s = saved;
  auto roots = s[0], q = s[1];
  auto local = at::matmul(at::matmul(q.transpose(-1, -2), sym(gradient)), q);
  auto denominator = roots.unsqueeze(-1) + roots.unsqueeze(-2);
  auto positive = denominator.gt(0);
  local = at::where(positive, local / at::where(positive, denominator, 1.), 0.);
  return {at::matmul(at::matmul(q, local), q.transpose(-1, -2))};
}
std::array<Tensor, 3> bures_kernel(Tensor a, Tensor b) {
  auto [u, s, vh] = at::linalg_svd(at::matmul(b.transpose(-1, -2), a), false);
  auto rotation = at::matmul(u, vh);
  auto residual = a - at::matmul(b, rotation);
  auto identical = a.eq(b).all(-1).all(-1);
  residual = at::where(identical.unsqueeze(-1).unsqueeze(-1), 0., residual);
  return {residual.square().sum({-1, -2}), residual, rotation};
}
Values bures_backward_kernel(const Tensor& gradient, const Values& saved,
                             at::IntArrayRef a_shape, at::IntArrayRef b_shape) {
  auto s = saved;
  auto derivative = 2 * gradient.unsqueeze(-1).unsqueeze(-1) * s[0];
  return {at::sum_to(derivative, a_shape),
          at::sum_to(-at::matmul(derivative, s[1].transpose(-1, -2)), b_shape)};
}
std::array<Tensor, 3> matrix_log_kernel(Tensor x) {
  auto [v, q] = utils::eigh(x);
  check(v.isfinite() & v.gt(0),
        "Covariances must be strictly positive definite with finite "
        "eigenvalues");
  return {at::matmul(q * v.log().unsqueeze(-2), q.transpose(-1, -2)), v, q};
}
Values matrix_log_backward_kernel(const Tensor& gradient, const Values& saved) {
  auto s = saved;
  auto v = s[0], q = s[1];
  auto hi = at::maximum(v.unsqueeze(-1), v.unsqueeze(-2)),
       lo = at::minimum(v.unsqueeze(-1), v.unsqueeze(-2));
  auto difference = hi - lo, relative = difference / hi;
  auto log_ratio = at::where(
      relative.le(.5), -(-relative.clamp_max(.5)).log1p(), hi.log() - lo.log());
  auto positive = difference.gt(0);
  auto divided =
      at::where(positive, log_ratio / at::where(positive, difference, 1.),
                hi.reciprocal());
  auto local = at::matmul(at::matmul(q.transpose(-1, -2), sym(gradient)), q);
  auto derivative = at::where(local.eq(0), 0., local * divided);
  return {sym(at::matmul(at::matmul(q, derivative), q.transpose(-1, -2)))};
}
std::array<Tensor, 4> congruence_kernel(Tensor a, Tensor b, Tensor factor,
                                        int64_t kind) {
  auto delta = whiten(factor, b - a), relative = whiten(factor, b);
  auto near = delta.abs().sum(-1).amax(-1).le(.5);
  auto matrix = at::where(near.unsqueeze(-1).unsqueeze(-1), delta, relative);
  check(matrix.isfinite(), "Whitened covariances must be finite");
  auto [spectrum, q] = utils::eigh(matrix);
  auto values = at::where(near.unsqueeze(-1), 1 + spectrum, spectrum);
  check(values.isfinite() & values.gt(0),
        "Whitened covariances must remain strictly positive definite");
  auto logs = at::where(near.unsqueeze(-1),
                        at::where(near.unsqueeze(-1), spectrum, 0.).log1p(),
                        at::where(near.unsqueeze(-1), 1., spectrum).log());
  auto identical = a.eq(b).all(-1).all(-1);
  logs = at::where(identical.unsqueeze(-1), 0., logs);
  auto vectors = at::linalg_solve_triangular(factor.transpose(-1, -2), q, true);
  return {kind == 2 ? logs.square().sum(-1) : logcosh(logs * .5).sum(-1),
          vectors, values, logs};
}
Values congruence_backward_kernel(const Tensor& gradient, const Values& saved,
                                  at::IntArrayRef a_shape,
                                  at::IntArrayRef b_shape, int64_t kind) {
  auto s = saved;
  auto coefficient = kind == 2 ? 2 * s[2] : .5 * (s[2] * .5).tanh();
  auto weight = gradient.unsqueeze(-1) * coefficient;
  auto da =
      sym(at::matmul(s[0] * (-weight).unsqueeze(-2), s[0].transpose(-1, -2)));
  auto db = sym(
      at::matmul(s[0] * (weight / s[1]).unsqueeze(-2), s[0].transpose(-1, -2)));
  return {at::sum_to(da, a_shape), at::sum_to(db, b_shape), Tensor(), Tensor()};
}

}  // namespace detail::structured
namespace detail::structured {
Features prepare_features(const Tensor& x, const Tensor& covariance,
                          const Tensor& weights, const Tensor& lengths,
                          int64_t kind, double epsilon, int64_t iterations,
                          int64_t component_tile) {
  auto vacant = at::empty({0}, x.options());
  if (kind == 0) {
    check(x.isfinite(), "Gaussian means must be finite");
    return {x, root(covariance), vacant, vacant};
  }
  if (kind < 4) {
    symmetric_check(x);
    auto a = sym(x);
    return {a, cov_transform(a, kind), vacant, vacant};
  }
  auto c = collection(x, covariance, weights, lengths, kind <= 6);
  Tensor self = vacant;
  if (kind == 5 || kind == 6)
    self = overlap(c, c, component_tile).unsqueeze(-1);
  if (kind == 9)
    self = entropic(c, c, epsilon, iterations).unsqueeze(-1);
  return {c[0], c[1], c[2], self};
}
Tensor feature_pair(const Features& a, const Features& b, int64_t kind,
                    bool paired, bool squared, double epsilon,
                    int64_t iterations, int64_t component_tile) {
  if (kind < 4) {
    auto x = a[0], y = b[0], xt = a[1], yt = b[1];
    if (!paired) {
      x = x.unsqueeze(1);
      y = y.unsqueeze(0);
      xt = xt.unsqueeze(1);
      yt = yt.unsqueeze(0);
    }
    return kind == 0 ? gaussian(x, xt, y, yt, squared)
                     : cov_pair(x, xt, y, yt, kind, squared);
  }
  auto restore = [](const Features& f) {
    auto positive = f[2].gt(0);
    auto logs =
        at::where(positive, at::where(positive, f[2], 1.).log(), -INFINITY);
    return Collection{f[0], f[1], f[2], logs};
  };
  auto left = restore(a), right = restore(b);
  // Exact transport solves one component problem per sample pair.
  if (kind == 7) {
    std::vector<Tensor> values;
    for (int64_t i = 0; i < a[0].size(0); ++i)
      for (int64_t j = paired ? i : 0; j < (paired ? i + 1 : b[0].size(0));
           ++j) {
        Collection aa, bb;
        for (int k = 0; k < 4; ++k) {
          aa[k] = left[k][i];
          bb[k] = right[k][j];
        }
        values.push_back(mixture_pair(aa, bb, kind, squared, epsilon,
                                      iterations, component_tile));
      }
    return at::stack(values).reshape(
        paired ? std::vector<int64_t>{a[0].size(0)}
               : std::vector<int64_t>{a[0].size(0), b[0].size(0)});
  }
  Tensor sa, sb;
  if (kind == 5 || kind == 6 || kind == 9) {
    sa = a[3].squeeze(-1);
    sb = b[3].squeeze(-1);
  }
  if (!paired) {
    for (int k = 0; k < 4; ++k) {
      left[k] = left[k].unsqueeze(1);
      right[k] = right[k].unsqueeze(0);
    }
    if (sa.defined()) {
      sa = sa.unsqueeze(1);
      sb = sb.unsqueeze(0);
    }
  }
  return mixture_pair(left, right, kind, squared, epsilon, iterations,
                      component_tile, sa, sb);
}
}  // namespace detail::structured
namespace detail::structured {
PreparedInputs prepare_gaussians(const Tensor& mean, const Tensor& covariance,
                                 const c10::optional<Tensor>& other_mean,
                                 const c10::optional<Tensor>& other_covariance,
                                 c10::string_view mode, bool squared) {
  TORCH_CHECK(other_mean.has_value() == other_covariance.has_value(),
              "Supply both second Gaussian tensors or omit both");
  auto pair = pairing(mode, other_mean.has_value());
  auto ym = other_mean.value_or(mean),
       yc = other_covariance.value_or(covariance);
  input_check({mean, covariance, ym, yc});
  for (auto p :
       std::vector<std::pair<Tensor, Tensor>>{{mean, covariance}, {ym, yc}}) {
    auto m = p.first, c = p.second;
    TORCH_CHECK(
        m.dim() >= 2 && c.dim() >= 3,
        "Gaussian means/covariances must have shape [...,N,D] / [...,N,D,D]");
    TORCH_CHECK(m.sym_size(-1) > 0 && c.sym_size(-1) == m.sym_size(-1) &&
                    c.sym_size(-2) == m.sym_size(-1) &&
                    c.sym_size(-3) == m.sym_size(-2),
                "Gaussian covariance dimensions must match means");
  }
  TORCH_CHECK(mean.sym_size(-1) == ym.sym_size(-1),
              "Gaussian feature dimensions must match");
  TORCH_CHECK(pair != 1 || mean.sym_size(-2) == ym.sym_size(-2),
              "Paired Gaussians must have equal sample counts");
  auto batch = batch_shape({{mean, 2}, {covariance, 3}, {ym, 2}, {yc, 3}});
  validate_erased(batch, {{mean, 0}, {covariance, 1}, {ym, 0}, {yc, 1}});
  auto x = expand_flat(promote(mean), batch, 2),
       c = expand_flat(promote(covariance), batch, 3),
       y = expand_flat(promote(ym), batch, 2),
       d = expand_flat(promote(yc), batch, 3);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  return {{x, c, empty(x), y, d, empty(x), empty(x, true), empty(x, true)},
          batch,
          0,
          pair,
          squared,
          .1,
          100,
          16,
          32};
}

PreparedInputs prepare_covariances(const Tensor& covariance,
                                   const c10::optional<Tensor>& other,
                                   c10::string_view metric,
                                   c10::string_view mode, bool squared) {
  auto pair = pairing(mode, other.has_value());
  int64_t kind = metric == "log_euclidean"      ? 1
                 : metric == "affine_invariant" ? 2
                 : metric == "stein"            ? 3
                                                : -1;
  TORCH_CHECK(kind > 0, "Unsupported covariance metric: ", metric);
  auto y = other.value_or(covariance);
  input_check({covariance, y});
  for (const auto& x : {covariance, y})
    TORCH_CHECK(
        x.dim() >= 3 && x.sym_size(-1) > 0 && x.sym_size(-1) == x.sym_size(-2),
        "Covariances must have shape [...,N,D,D] with nonempty square "
        "matrices");
  TORCH_CHECK(covariance.sym_size(-1) == y.sym_size(-1),
              "Covariance feature dimensions must match");
  TORCH_CHECK(pair != 1 || covariance.sym_size(-3) == y.sym_size(-3),
              "Paired covariances must have equal sample counts");
  auto batch = batch_shape({{covariance, 3}, {y, 3}});
  validate_erased(batch, {{covariance, 2}, {y, 2}});
  auto a = expand_flat(promote(covariance), batch, 3),
       b = expand_flat(promote(y), batch, 3);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  return {{a, empty(a), empty(a), b, empty(a), empty(a), empty(a, true),
           empty(a, true)},
          batch,
          kind,
          pair,
          squared,
          .1,
          100,
          16,
          32};
}

PreparedInputs prepare_mixtures(
    const Tensor& mean, const Tensor& covariance, const Tensor& weights,
    const c10::optional<Tensor>& other_mean,
    const c10::optional<Tensor>& other_covariance,
    const c10::optional<Tensor>& other_weights, c10::string_view metric,
    c10::string_view mode, const c10::optional<Tensor>& lengths,
    const c10::optional<Tensor>& other_lengths, c10::optional<bool> squared,
    c10::optional<double> regularization, c10::optional<int64_t> max_iter,
    int64_t chunk_size, c10::optional<int64_t> component_chunk_size) {
  TORCH_CHECK(other_mean.has_value() == other_covariance.has_value() &&
                  other_mean.has_value() == other_weights.has_value(),
              "Supply all three second mixture tensors or omit all three");
  auto pair = pairing(mode, other_mean.has_value());
  auto ym = other_mean.value_or(mean),
       yc = other_covariance.value_or(covariance),
       yw = other_weights.value_or(weights);
  auto yl = other_lengths.has_value()
                ? other_lengths
                : (!other_mean.has_value() ? lengths : c10::nullopt);
  input_check({mean, covariance, weights, ym, yc, yw});
  int64_t kind = metric == "overlap"       ? 4
                 : metric == "l2"          ? 5
                 : metric == "cs"          ? 6
                 : metric == "wasserstein" ? 7
                 : metric == "entropic"    ? 8
                 : metric == "sinkhorn"    ? 9
                                           : -1;
  TORCH_CHECK(kind >= 4, "Unsupported mixture metric: ", metric);
  TORCH_CHECK(!squared.has_value() || kind == 5 || kind == 7,
              "Unused parameter: squared");
  TORCH_CHECK(!regularization.has_value() || kind >= 8,
              "Unused parameter: regularization");
  TORCH_CHECK(!max_iter.has_value() || kind >= 8, "Unused parameter: max_iter");
  TORCH_CHECK(!component_chunk_size.has_value() || kind <= 6,
              "Unused parameter: component_chunk_size");
  TORCH_CHECK(chunk_size > 0 && component_chunk_size.value_or(32) > 0,
              "Mixture chunk sizes must be positive");
  auto epsilon = regularization.value_or(.1);
  auto iterations = max_iter.value_or(100);
  TORCH_CHECK(
      iterations > 0 && std::isfinite(epsilon) && epsilon > 0,
      "Mixture regularization and max_iter must be positive and finite");
  std::vector<std::pair<Tensor, int64_t>> shapes;
  auto validate = [&](const Tensor& m, const Tensor& c, const Tensor& w,
                      const c10::optional<Tensor>& l) {
    TORCH_CHECK(m.dim() >= 3 && c.dim() >= 4 && w.dim() >= 2,
                "Mixture shapes must be [...,N,K,D], [...,N,K,D,D], [...,N,K]");
    auto n = m.sym_size(-3), k = m.sym_size(-2), d = m.sym_size(-1);
    TORCH_CHECK(k > 0 && d > 0 && c.sym_size(-4) == n && c.sym_size(-3) == k &&
                    c.sym_size(-2) == d && c.sym_size(-1) == d &&
                    w.sym_size(-2) == n && w.sym_size(-1) == k,
                "Mixture sample, component and feature dimensions must match");
    shapes.insert(shapes.end(), {{m, 3}, {c, 4}, {w, 2}});
    if (l.has_value()) {
      TORCH_CHECK(l->dim() >= 1 && l->sym_size(-1) == n &&
                      l->scalar_type() == at::kLong &&
                      l->layout() == at::kStrided && l->device() == m.device(),
                  "Mixture lengths must be dense int64 input-device tensors "
                  "ending in the sample count");
      shapes.push_back({*l, 1});
    }
  };
  validate(mean, covariance, weights, lengths);
  validate(ym, yc, yw, yl);
  TORCH_CHECK(mean.sym_size(-1) == ym.sym_size(-1),
              "Mixture features must match");
  TORCH_CHECK(pair != 1 || mean.sym_size(-3) == ym.sym_size(-3),
              "Paired mixtures must have equal sample counts");
  auto batch = batch_shape(shapes);
  auto x = expand_flat(promote(mean), batch, 3),
       c = expand_flat(promote(covariance), batch, 4),
       w = expand_flat(promote(weights), batch, 2),
       y = expand_flat(promote(ym), batch, 3),
       d = expand_flat(promote(yc), batch, 4),
       v = expand_flat(promote(yw), batch, 2);
  if (kind >= 8) {
    double tiny = x.scalar_type() == at::kDouble
                      ? std::numeric_limits<double>::min()
                      : std::numeric_limits<float>::min(),
           maximum = x.scalar_type() == at::kDouble
                         ? std::numeric_limits<double>::max()
                         : std::numeric_limits<float>::max();
    TORCH_CHECK(epsilon >= tiny && epsilon <= maximum,
                "regularization must be a positive normal finite value in the "
                "structured_compare_kernel dtype");
  }
  auto lx = lengths.has_value() ? expand_flat(*lengths, batch, 1)
                                : empty(x, true),
       ly = yl.has_value() ? expand_flat(*yl, batch, 1) : empty(x, true);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  return {{x, c, w, y, d, v, lx, ly},       batch,   kind,       pair,
          squared.value_or(false),          epsilon, iterations, chunk_size,
          component_chunk_size.value_or(32)};
}
}  // namespace detail::structured
namespace {
Tensor run_prepared(const detail::structured::PreparedInputs& p) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  const auto& v = p.operands;
  return output_shape(
      detail::structured::_structured_compare(
          v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], p.kind, p.mode,
          p.squared, p.epsilon, p.iterations, p.tile, p.component_tile),
      p.batch);
}
Tensor gaussian_comparisons_impl(const Tensor& mean, const Tensor& covariance,
                                 const c10::optional<Tensor>& other_mean,
                                 const c10::optional<Tensor>& other_covariance,
                                 c10::string_view mode, bool squared) {
  return run_prepared(detail::structured::prepare_gaussians(
      mean, covariance, other_mean, other_covariance, mode, squared));
}
Tensor covariance_comparisons_impl(const Tensor& covariance,
                                   const c10::optional<Tensor>& other,
                                   c10::string_view metric,
                                   c10::string_view mode, bool squared) {
  return run_prepared(detail::structured::prepare_covariances(
      covariance, other, metric, mode, squared));
}
Tensor mixture_comparisons_impl(
    const Tensor& mean, const Tensor& covariance, const Tensor& weights,
    const c10::optional<Tensor>& other_mean,
    const c10::optional<Tensor>& other_covariance,
    const c10::optional<Tensor>& other_weights, c10::string_view metric,
    c10::string_view mode, const c10::optional<Tensor>& lengths,
    const c10::optional<Tensor>& other_lengths, c10::optional<bool> squared,
    c10::optional<double> regularization, c10::optional<int64_t> max_iter,
    int64_t chunk_size, c10::optional<int64_t> component_chunk_size) {
  return run_prepared(detail::structured::prepare_mixtures(
      mean, covariance, weights, other_mean, other_covariance, other_weights,
      metric, mode, lengths, other_lengths, squared, regularization, max_iter,
      chunk_size, component_chunk_size));
}
}  // namespace
}  // namespace torchpairwise::ops
