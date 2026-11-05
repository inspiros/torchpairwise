#include "transport_maps.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include "transport_maps_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
using at::Tensor;
Tensor ground_cost(const Tensor& x, const Tensor& y, int64_t p) {
  auto distance = at::cdist(x, y, 2, 2);
  return p == 2 ? distance.square() : distance;
}
Tensor log_normalizer(const Tensor& x, const Tensor& y, const Tensor& logw,
                      int64_t p, double epsilon, int64_t size) {
  std::vector<Tensor> rows;
  for (int64_t i = 0; i < x.size(-2); i += size) {
    auto n = std::min(size, x.size(-2) - i);
    auto a = x.narrow(-2, i, n);
    auto total = at::full_like(a.select(-1, 0), -INFINITY);
    for (int64_t j = 0; j < y.size(-2); j += size) {
      auto m = std::min(size, y.size(-2) - j);
      auto cost = ground_cost(a, y.narrow(-2, j, m), p) / epsilon;
      utils::check_values(cost.isfinite().all(),
                          "Sinkhorn map scaled costs must be finite; rescale "
                          "inputs or regularization");
      auto part = at::logsumexp(logw.narrow(-1, j, m).unsqueeze(-2) - cost, -1);
      total = at::logaddexp(total, part);
    }
    rows.push_back(total);
  }
  return at::cat(rows, -1);
}
Tensor transport_map_reduce_kernel(const Tensor& x, const Tensor& y,
                                   const Tensor& logw, int64_t p,
                                   double epsilon, int64_t size,
                                   bool barycentric) {
  utils::DeferredValueChecks checks;
  auto norm = log_normalizer(x, y, logw, p, epsilon, size);
  if (!barycentric) {
    checks.finish();
    return norm;
  }
  std::vector<Tensor> rows;
  for (int64_t i = 0; i < x.size(-2); i += size) {
    auto n = std::min(size, x.size(-2) - i);
    auto a = x.narrow(-2, i, n);
    auto total = at::zeros_like(a);
    for (int64_t j = 0; j < y.size(-2); j += size) {
      auto m = std::min(size, y.size(-2) - j);
      auto b = y.narrow(-2, j, m);
      auto plan =
          (logw.narrow(-1, j, m).unsqueeze(-2) -
           ground_cost(a, b, p) / epsilon - norm.narrow(-1, i, n).unsqueeze(-1))
              .exp();
      total.add_(plan.matmul(b));
    }
    rows.push_back(total);
  }
  checks.finish();
  return at::cat(rows, -2);
}
std::tuple<Tensor, Tensor, Tensor> transport_map_reduce_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& logw, const Tensor& result, int64_t p, double epsilon,
    int64_t size, bool barycentric) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  utils::DeferredValueChecks checks;
  auto gx = at::zeros(x.sizes(), x.options());
  auto gy = at::zeros(y.sizes(), y.options());
  auto gw = at::zeros(logw.sizes(), logw.options());
  // These internal inputs have already been broadcast to a shared batch.
  auto norm =
      barycentric ? log_normalizer(x, y, logw, p, epsilon, size) : result;
  for (int64_t i = 0; i < x.size(-2); i += size) {
    auto n = std::min(size, x.size(-2) - i);
    auto a = x.narrow(-2, i, n).detach();
    auto g = gradient.narrow(barycentric ? -2 : -1, i, n);
    for (int64_t j = 0; j < y.size(-2); j += size) {
      auto m = std::min(size, y.size(-2) - j);
      auto b = y.narrow(-2, j, m).detach();
      Tensor score_gradient, dx, dy;
      {
        at::AutoGradMode enable(true);
        auto aa = a.detach().set_requires_grad(true);
        auto bb = b.detach().set_requires_grad(true);
        auto cost = ground_cost(aa, bb, p);
        {
          at::NoGradGuard guard;
          auto plan =
              (logw.narrow(-1, j, m).unsqueeze(-2) - cost.detach() / epsilon -
               norm.narrow(-1, i, n).unsqueeze(-1))
                  .exp();
          if (barycentric) {
            score_gradient =
                plan * (g.matmul(b.transpose(-1, -2)) -
                        (g * result.narrow(-2, i, n)).sum(-1, true));
            gy.narrow(-2, j, m).add_(plan.transpose(-1, -2).matmul(g));
          } else {
            score_gradient = plan * g.unsqueeze(-1);
          }
        }
        auto grads = torch::autograd::grad(
            {cost}, {aa, bb}, {-score_gradient / epsilon}, false, false);
        dx = grads[0];
        dy = grads[1];
      }
      gx.narrow(-2, i, n).add_(dx);
      gy.narrow(-2, j, m).add_(dy);
      gw.narrow(-1, j, m).add_(score_gradient.sum(-2));
    }
  }
  checks.finish();
  return {gx, gy, gw};
}
Tensor sinkhorn_map_impl(const Tensor& input, const Tensor& other,
                         const c10::optional<Tensor>& source_weights,
                         const c10::optional<Tensor>& target_weights,
                         double epsilon, int64_t iterations, int64_t p,
                         int64_t size, bool displacement) {
  TORCH_CHECK(p == 1 || p == 2, "Sinkhorn map p must be 1 or 2");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0,
              "regularization must be positive and finite");
  TORCH_CHECK(iterations > 0 && size > 0,
              "max_iter and chunk_size must be positive");
  TORCH_CHECK(input.layout() == at::kStrided &&
                  other.layout() == at::kStrided && input.dim() >= 2 &&
                  other.dim() >= 2,
              "Sinkhorn map inputs must be dense with rank >= 2");
  TORCH_CHECK(input.device() == other.device() &&
                  input.scalar_type() == other.scalar_type(),
              "Sinkhorn map inputs must share dtype and device");
  TORCH_CHECK(input.device().is_cpu() || input.device().is_cuda() ||
                  input.device().is_meta(),
              "Sinkhorn maps support CPU, CUDA and metadata execution");
  TORCH_CHECK(input.sym_size(-1) == other.sym_size(-1) &&
                  input.sym_size(-1) > 0 && input.sym_size(-2) > 0 &&
                  other.sym_size(-2) > 0,
              "Sinkhorn maps require nonempty collections and matching "
              "positive feature dimensions");
  bool amp = !input.is_meta() &&
             at::autocast::is_autocast_enabled(input.device().type());
  auto cast = [&](const Tensor& value) {
    return amp && (value.scalar_type() == at::kHalf ||
                   value.scalar_type() == at::kBFloat16)
               ? value.to(at::kFloat)
               : value;
  };
  auto x = cast(input), y = cast(other);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "Sinkhorn maps require float32/float64 outside autocast");
  auto tiny = x.scalar_type() == at::kFloat
                  ? double(std::numeric_limits<float>::min())
                  : std::numeric_limits<double>::min();
  auto maximum = x.scalar_type() == at::kFloat
                     ? double(std::numeric_limits<float>::max())
                     : std::numeric_limits<double>::max();
  TORCH_CHECK(epsilon >= tiny && epsilon <= maximum,
              "regularization must be normal in the compute dtype");
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 2),
                                     y.sym_sizes().slice(0, y.dim() - 2));
  auto weights = [&](const c10::optional<Tensor>& raw, const Tensor& points) {
    auto shape = points.sym_sizes().slice(0, points.dim() - 1).vec();
    auto value = raw ? cast(*raw) : at::ones_symint(shape, points.options());
    TORCH_CHECK(value.layout() == at::kStrided && value.dim() >= 1 &&
                    value.device() == x.device() &&
                    value.scalar_type() == x.scalar_type() &&
                    value.sym_size(-1) == points.sym_size(-2),
                "Masses must match input dtype/device and sample count");
    batch = at::infer_size_symint(batch,
                                  value.sym_sizes().slice(0, value.dim() - 1));
    return value;
  };
  auto a = weights(source_weights, x), b = weights(target_weights, y);
  auto xs = batch, ys = batch, as = batch, bs = batch;
  xs.insert(xs.end(), {x.sym_size(-2), x.sym_size(-1)});
  ys.insert(ys.end(), {y.sym_size(-2), y.sym_size(-1)});
  as.push_back(x.sym_size(-2));
  bs.push_back(y.sym_size(-2));
  x = x.expand_symint(xs);
  y = y.expand_symint(ys);
  a = a.expand_symint(as);
  b = b.expand_symint(bs);
  utils::DeferredValueChecks deferred;
  utils::ValueChecks checks;
  checks.require(x.isfinite().all() & y.isfinite().all(),
                 "Sinkhorn map coordinates must be finite");
  checks.require(
      (a.isfinite() & a.ge(0)).all() & (b.isfinite() & b.ge(0)).all(),
      "Sinkhorn map masses must be finite and nonnegative");
  auto atotal = a.sum(-1, true), btotal = b.sum(-1, true);
  checks.require((atotal.isfinite() & atotal.gt(0)).all() &
                     (btotal.isfinite() & btotal.gt(0)).all(),
                 "Sinkhorn map masses must have a positive finite total");
  checks.finish();
  auto positive_a = a.gt(0), positive_b = b.gt(0);
  auto loga = at::where(
      positive_a, at::where(positive_a, a, 1.).log() - atotal.log(), -INFINITY);
  auto logb = at::where(
      positive_b, at::where(positive_b, b, 1.).log() - btotal.log(), -INFINITY);
  auto v = at::zeros_like(logb);
  for (int64_t t = 0; t < iterations; ++t) {
    auto u = loga - detail::transport_maps::reduce(x, y, v, p, epsilon, size);
    v = logb - detail::transport_maps::reduce(y, x, u, p, epsilon, size);
  }
  auto result = detail::transport_maps::reduce(x, y, v, p, epsilon, size, true);
  if (displacement)
    result = result - x;
  result = at::where(positive_a.unsqueeze(-1), result, 0.);
  deferred.finish();
  return result;
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "sinkhorn_map(Tensor x, Tensor y, Tensor? source_weights=None, Tensor? "
      "target_weights=None, "
      "float regularization=0.1, int max_iter=100, int p=2, int "
      "chunk_size=128, bool displacement=False) -> Tensor",
      TORCH_FN(sinkhorn_map_impl));
  m.def(
      "_transport_map_reduce(Tensor x, Tensor y, Tensor logw, int p, float "
      "epsilon, int size, bool barycentric) -> Tensor");
  m.def(
      "_transport_map_reduce_backward(Tensor grad, Tensor x, Tensor y, Tensor "
      "logw, Tensor result, "
      "int p, float epsilon, int size, bool barycentric) -> (Tensor, Tensor, "
      "Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_transport_map_reduce", TORCH_FN(transport_map_reduce_kernel));
  m.impl("_transport_map_reduce_backward",
         TORCH_FN(transport_map_reduce_backward_kernel));
}
}  // namespace
Tensor sinkhorn_map(const Tensor& x, const Tensor& y,
                    const SinkhornMapOptions& o) {
  using Signature = Tensor(
      const Tensor&, const Tensor&, const c10::optional<Tensor>&,
      const c10::optional<Tensor>&, double, int64_t, int64_t, int64_t, bool);
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::sinkhorn_map", "")
                       .typed<Signature>();
  return op.call(x, y, o.source_weights, o.target_weights, o.regularization,
                 o.max_iter, o.p, o.chunk_size, o.displacement);
}
}  // namespace torchpairwise::ops
