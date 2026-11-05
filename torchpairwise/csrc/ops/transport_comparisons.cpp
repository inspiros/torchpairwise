#include "transport_comparisons.h"

#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <queue>

#include "transport_comparisons_detail.h"
#include "utils/clouds.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
at::Tensor transport_comparisons_composite(
    const at::Tensor& input, const c10::optional<at::Tensor>& other,
    c10::string_view metric, c10::string_view mode,
    const TransportComparisonOptions& options);
}

namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
enum Kind : int64_t { Energy, MMD, Sliced, Sinkhorn, Exact };
enum Kernel : int64_t { RBF, Laplacian, Linear };
Tensor lengths(const Tensor& x, const Tensor& len) {
  return len.numel() ? len
                     : at::full(x.sizes().slice(0, x.dim() - 2), x.size(-2),
                                x.options().dtype(at::kLong));
}
std::tuple<Tensor, Tensor, Tensor> clean(const Tensor& x, const Tensor& len,
                                         bool projection = false) {
  auto valid = at::arange(x.size(-2), x.options().dtype(at::kLong))
                   .lt(len.unsqueeze(-1));
  TORCH_CHECK((x.isfinite() | ~valid.unsqueeze(-1)).all().item<bool>(),
              "Distribution comparison valid point coordinates must be finite");
  return {at::where(valid.unsqueeze(-1), x,
                    projection ? at::zeros_like(x) : x.narrow(-2, 0, 1)),
          valid, len.to(x.scalar_type())};
}

Tensor point_tile(const Tensor& x, const Tensor& y, const Tensor& xv,
                  const Tensor& yv, int64_t kind, int64_t kernel,
                  double scale) {
  auto distance = at::cdist(x, y, 2, 2);
  Tensor value = distance;
  if (kind == MMD)
    value = kernel == RBF ? (-.5 * (distance / scale).square()).exp()
                          : (-distance / scale).exp();
  return at::where(xv.unsqueeze(-1) & yv.unsqueeze(-2), value, 0.)
      .sum({-1, -2});
}

Tensor mean_pair(const Tensor& x, const Tensor& y, const Tensor& xv,
                 const Tensor& yv, const Tensor& xl, const Tensor& yl,
                 int64_t kind, int64_t kernel, double scale) {
  auto result =
      x.narrow(-2, 0, 0).sum({-1, -2}) + y.narrow(-2, 0, 0).sum({-1, -2});
  for (int64_t i = 0; i < x.size(-2); i += 128)
    for (int64_t j = 0; j < y.size(-2); j += 128) {
      auto n = std::min(int64_t(128), x.size(-2) - i),
           m = std::min(int64_t(128), y.size(-2) - j);
      auto a = x.narrow(-2, i, n), b = y.narrow(-2, j, m),
           av = xv.narrow(-1, i, n), bv = yv.narrow(-1, j, m);
      auto value =
          at::GradMode::is_enabled() && (a.requires_grad() || b.requires_grad())
              ? detail::transport::point_autograd(a, b, av, bv, kind, kernel,
                                                  scale)
              : point_tile(a, b, av, bv, kind, kernel, scale);
      result = result + value;
    }
  return result / xl / yl;
}

Tensor quantile(const Tensor& a, const Tensor& b, const Tensor& al,
                const Tensor& bl, double p) {
  auto aticks = at::minimum(at::arange(1, a.size(-2) + 1, al.options()),
                            al.unsqueeze(-1)) *
                bl.unsqueeze(-1);
  auto bticks = at::minimum(at::arange(1, b.size(-2) + 1, bl.options()),
                            bl.unsqueeze(-1)) *
                al.unsqueeze(-1);
  auto ticks = std::get<0>(
      at::cat({at::zeros_like(al).unsqueeze(-1), aticks, bticks}, -1).sort(-1));
  auto left = ticks.narrow(-1, 0, ticks.size(-1) - 1);
  auto ai = at::minimum(at::div(left, bl.unsqueeze(-1), "floor"),
                        al.unsqueeze(-1) - 1);
  auto bi = at::minimum(at::div(left, al.unsqueeze(-1), "floor"),
                        bl.unsqueeze(-1) - 1);
  auto shape = ai.sizes().vec();
  shape.push_back(a.size(-1));
  auto as = ai.sizes().slice(0, ai.dim() - 1).vec();
  as.insert(as.end(), {a.size(-2), a.size(-1)});
  auto bs = ai.sizes().slice(0, ai.dim() - 1).vec();
  bs.insert(bs.end(), {b.size(-2), b.size(-1)});
  auto av = a.expand(as).gather(-2, ai.unsqueeze(-1).expand(shape));
  auto bv = b.expand(bs).gather(-2, bi.unsqueeze(-1).expand(shape));
  auto difference = (av - bv).abs();
  auto mass = ticks.diff().to(a.scalar_type()) / (al * bl).unsqueeze(-1);
  return ((p == 1 ? difference : difference.square()).mean(-1) * mass).sum(-1);
}
Tensor projected(const Tensor& x, const Tensor& len, const Tensor& directions) {
  auto cleaned = clean(x, len, true);
  auto valid = std::get<1>(cleaned);
  auto values = std::get<0>(cleaned)
                    .matmul(directions)
                    .masked_fill(~valid.unsqueeze(-1), INFINITY);
  return std::get<0>(at::sort(values, true, -2, false));
}

Tensor dual(const Tensor& x, const Tensor& y, const Tensor& la,
            const Tensor& lb, double p, double epsilon, int64_t iterations) {
  auto cost = at::cdist(x, y, 2, 2);
  if (p == 2)
    cost = cost.square();
  cost = cost / epsilon;
  TORCH_CHECK(cost.isfinite().all().item<bool>(),
              "Sinkhorn ground costs / regularization must be finite; rescale "
              "inputs or regularization");
  auto f = at::zeros_like(la), g = at::zeros_like(lb);
  for (int64_t iteration = 0; iteration < iterations; ++iteration) {
    auto nf = -at::logsumexp(lb.unsqueeze(-2) + g.unsqueeze(-2) - cost, -1);
    auto ng = -at::logsumexp(la.unsqueeze(-1) + f.unsqueeze(-1) - cost, -2);
    f = at::where(la.isfinite(), (f + nf) * .5, 0.);
    g = at::where(lb.isfinite(), (g + ng) * .5, 0.);
  }
  auto log_plan = la.unsqueeze(-1) + lb.unsqueeze(-2) + f.unsqueeze(-1) +
                  g.unsqueeze(-2) - cost;
  auto correction = -at::logsumexp(log_plan.flatten(-2), -1).expm1();
  return epsilon *
         ((la.exp() * f).sum(-1) + (lb.exp() * g).sum(-1) + correction);
}

// Uniform min-cost flow uses exact integer capacities; no expansion to lcm
// support size. All solver data are detached CPU doubles; selected edges alone
// enter the differentiable input-device objective.
struct Edge {
  int64_t target, reverse, capacity;
  double cost;
};
Tensor exact_pair(const Tensor& x, const Tensor& y, double p) {
  Tensor cost;
  std::vector<int64_t> rows, columns;
  std::vector<double> masses;
  {
    at::NoGradGuard guard;
    cost = at::cdist(x.to(at::kDouble), y.to(at::kDouble), 2, 2);
    if (p == 2)
      cost = cost.square();
    TORCH_CHECK(
        cost.isfinite().all().item<bool>(),
        "Wasserstein ground costs must be finite; rescale the coordinates");
    cost = cost.cpu().contiguous();
    auto prices = cost.accessor<double, 2>();
    int64_t n = x.size(0), m = y.size(0), divisor = std::gcd(n, m),
            supply = m / divisor, demand = n / divisor;
    TORCH_CHECK(n <= std::numeric_limits<int64_t>::max() / supply,
                "Transport integer capacity overflow");
    int64_t total = n * supply, source = n + m, sink = source + 1;
    std::vector<std::vector<Edge>> graph(n + m + 2);
    auto add = [&](int64_t a, int64_t b, int64_t capacity, double price) {
      auto position = graph[a].size();
      graph[a].push_back({b, int64_t(graph[b].size()), capacity, price});
      graph[b].push_back({a, int64_t(position), 0, -price});
      return int64_t(position);
    };
    double scale = cost.max().item<double>();
    if (scale == 0)
      scale = 1;
    std::vector<std::vector<int64_t>> transport(n, std::vector<int64_t>(m));
    for (int64_t i = 0; i < n; ++i) {
      add(source, i, supply, 0);
      for (int64_t j = 0; j < m; ++j)
        transport[i][j] = add(i, n + j, total, prices[i][j] / scale);
    }
    for (int64_t j = 0; j < m; ++j)
      add(n + j, sink, demand, 0);
    std::vector<double> potential(graph.size(), 0.);
    int64_t remaining = total;
    while (remaining) {
      std::vector<double> distance(graph.size(), INFINITY);
      std::vector<std::pair<int64_t, int64_t>> previous(graph.size(), {-1, -1});
      using Item = std::pair<double, int64_t>;
      std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
      distance[source] = 0;
      queue.push({0., source});
      while (!queue.empty()) {
        auto [current, u] = queue.top();
        queue.pop();
        if (current != distance[u])
          continue;
        for (size_t index = 0; index < graph[u].size(); ++index) {
          auto& edge = graph[u][index];
          if (!edge.capacity)
            continue;
          auto reduced = edge.cost + potential[u] - potential[edge.target];
          auto maximum =
              std::max({1., std::abs(edge.cost), std::abs(potential[u]),
                        std::abs(potential[edge.target])});
          double tolerance = 64 * (std::nextafter(maximum, INFINITY) - maximum);
          TORCH_CHECK(reduced >= -tolerance,
                      "Wasserstein solver lost numerical dual feasibility");
          auto candidate = current + std::max(0., reduced);
          if (candidate < distance[edge.target]) {
            distance[edge.target] = candidate;
            previous[edge.target] = {u, int64_t(index)};
            queue.push({candidate, edge.target});
          }
        }
      }
      TORCH_CHECK(
          previous[sink].first >= 0,
          "Wasserstein solver could not find a feasible transport plan");
      for (size_t v = 0; v < graph.size(); ++v)
        if (std::isfinite(distance[v]))
          potential[v] += distance[v];
      int64_t amount = remaining;
      for (auto v = sink; v != source;) {
        auto [u, k] = previous[v];
        amount = std::min(amount, graph[u][k].capacity);
        v = u;
      }
      for (auto v = sink; v != source;) {
        auto [u, k] = previous[v];
        auto& edge = graph[u][k];
        edge.capacity -= amount;
        graph[v][edge.reverse].capacity += amount;
        v = u;
      }
      remaining -= amount;
    }
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < m; ++j) {
        auto flow = total - graph[i][transport[i][j]].capacity;
        if (flow) {
          rows.push_back(i);
          columns.push_back(j);
          masses.push_back(double(flow) / total);
        }
      }
  }
  auto row = at::tensor(rows, x.options().dtype(at::kLong)),
       column = at::tensor(columns, x.options().dtype(at::kLong));
  auto mass = at::tensor(masses, x.options().dtype(at::kDouble));
  auto difference = x.index_select(0, row).to(at::kDouble) -
                    y.index_select(0, column).to(at::kDouble);
  auto value =
      p == 1 ? (mass * at::linalg_vector_norm(difference, 2, {-1}, false)).sum()
             : detail::safe_root((mass * difference.square().sum(-1)).sum());
  return value.to(x.scalar_type());
}

Tensor pair_value(const Tensor& raw_x, const Tensor& raw_y, const Tensor& xl,
                  const Tensor& yl, const Tensor& directions, int64_t kind,
                  double p, int64_t kernel, double scale, double epsilon,
                  int64_t iterations, bool squared) {
  if (kind == Exact) {
    auto batch = raw_x.sizes().slice(0, raw_x.dim() - 2).vec();
    auto count = raw_x.numel() / (raw_x.size(-2) * raw_x.size(-1));
    auto x = raw_x.reshape({count, raw_x.size(-2), raw_x.size(-1)}),
         y = raw_y.reshape({count, raw_y.size(-2), raw_y.size(-1)});
    auto al = xl.reshape({count}).cpu(), bl = yl.reshape({count}).cpu();
    std::vector<Tensor> values;
    for (int64_t i = 0; i < count; ++i)
      values.push_back(exact_pair(x[i].narrow(0, 0, al[i].item<int64_t>()),
                                  y[i].narrow(0, 0, bl[i].item<int64_t>()), p));
    return at::stack(values).reshape(batch);
  }
  if (kind == Sliced) {
    auto a = projected(raw_x, xl, directions),
         b = projected(raw_y, yl, directions);
    auto value = quantile(a, b, xl, yl, p);
    return p == 1 ? value : detail::safe_root(value);
  }
  auto cx = clean(raw_x, xl), cy = clean(raw_y, yl);
  auto x = std::get<0>(cx), y = std::get<0>(cy), xv = std::get<1>(cx),
       yv = std::get<1>(cy);
  auto al = std::get<2>(cx), bl = std::get<2>(cy);
  if (kind == Sinkhorn) {
    auto la = at::where(xv, -al.log().unsqueeze(-1), -INFINITY),
         lb = at::where(yv, -bl.log().unsqueeze(-1), -INFINITY);
    return dual(x, y, la, lb, p, epsilon, iterations) -
           .5 * (dual(x, x, la, la, p, epsilon, iterations) +
                 dual(y, y, lb, lb, p, epsilon, iterations));
  }
  Tensor value;
  if (kind == MMD && kernel == Linear) {
    auto a = at::where(xv.unsqueeze(-1), x, 0.).sum(-2) / al.unsqueeze(-1);
    auto b = at::where(yv.unsqueeze(-1), y, 0.).sum(-2) / bl.unsqueeze(-1);
    value = at::linalg_vector_norm(a - b, 2, {-1}, false);
    return squared ? value.square() : value;
  }
  auto cross = mean_pair(x, y, xv, yv, al, bl, kind, kernel, scale);
  auto own_x = mean_pair(x, x, xv, xv, al, al, kind, kernel, scale),
       own_y = mean_pair(y, y, yv, yv, bl, bl, kind, kernel, scale);
  value =
      (kind == Energy ? 2 * cross - own_x - own_y : own_x + own_y - 2 * cross)
          .clamp_min(0);
  return squared ? value : detail::safe_root(value);
}

// Cache one scalar self objective per cloud. Enumerating clouds here keeps
// temporary point-cost storage independent of the number of cloud partners.
Tensor self_expectations(const Tensor& x, const Tensor& valid,
                         const Tensor& count, const Tensor& log_mass,
                         int64_t kind, double p, int64_t kernel, double scale,
                         double epsilon, int64_t iterations) {
  std::vector<Tensor> values;
  for (int64_t i = 0; i < x.size(-3); ++i) {
    auto cloud = x.select(-3, i);
    if (kind == Sinkhorn) {
      auto mass = log_mass.select(-2, i);
      values.push_back(dual(cloud, cloud, mass, mass, p, epsilon, iterations));
    } else {
      auto mask = valid.select(-2, i), length = count.select(-1, i);
      values.push_back(mean_pair(cloud, cloud, mask, mask, length, length, kind,
                                 kernel, scale));
    }
  }
  return at::stack(values, -1);
}

Tensor transport_compare_kernel(const Tensor& x, const Tensor& y,
                                const Tensor& raw_xl, const Tensor& raw_yl,
                                const Tensor& directions, int64_t kind,
                                int64_t mode, double p, int64_t kernel,
                                double scale, double epsilon,
                                int64_t iterations, bool squared,
                                int64_t size) {
  auto shape = detail::cloud_output_shape(x, y, mode);
  auto n = x.size(-3), m = y.size(-3);
  if (x.numel() == 0 || y.numel() == 0 || (mode == 2 && n < 2))
    return at::empty_symint(shape, x.options());
  auto xl = lengths(x, raw_xl), yl = lengths(y, raw_yl);
  std::vector<Tensor> rows;
  // Each cloud pair is independent. Avoid an N*M*P*Q cost tensor. Sliced
  // projections/sorts and self distribution expectations are cached below.
  Tensor a, b, av, bv, al, bl, la, lb, own_a, own_b;
  if (kind == Sliced) {
    a = projected(x, xl, directions);
    b = projected(y, yl, directions);
  } else if (kind == Energy || kind == MMD || kind == Sinkhorn) {
    auto cx = clean(x, xl), cy = clean(y, yl);
    a = std::get<0>(cx);
    b = std::get<0>(cy);
    av = std::get<1>(cx);
    bv = std::get<1>(cy);
    al = std::get<2>(cx);
    bl = std::get<2>(cy);
    if (kind == MMD && kernel == Linear) {
      a = at::where(av.unsqueeze(-1), a, 0.).sum(-2) / al.unsqueeze(-1);
      b = at::where(bv.unsqueeze(-1), b, 0.).sum(-2) / bl.unsqueeze(-1);
    } else {
      if (kind == Sinkhorn) {
        la = at::where(av, -al.log().unsqueeze(-1), -INFINITY);
        lb = at::where(bv, -bl.log().unsqueeze(-1), -INFINITY);
      }
      own_a = self_expectations(a, av, al, la, kind, p, kernel, scale, epsilon,
                                iterations);
      own_b = self_expectations(b, bv, bl, lb, kind, p, kernel, scale, epsilon,
                                iterations);
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    std::vector<Tensor> columns;
    for (int64_t j = mode == 2 ? i + 1 : (mode == 1 ? i : 0);
         j < (mode == 1 ? i + 1 : m); ++j) {
      Tensor value;
      if (kind == Sliced) {
        value = quantile(a.select(-3, i), b.select(-3, j), xl.select(-1, i),
                         yl.select(-1, j), p);
        if (p == 2)
          value = detail::safe_root(value);
      } else if (kind == MMD && kernel == Linear) {
        value = at::linalg_vector_norm(a.select(-2, i) - b.select(-2, j), 2,
                                       {-1}, false);
        if (squared)
          value = value.square();
      } else if (kind == Energy || kind == MMD) {
        auto cross =
            mean_pair(a.select(-3, i), b.select(-3, j), av.select(-2, i),
                      bv.select(-2, j), al.select(-1, i), bl.select(-1, j),
                      kind, kernel, scale);
        value = (kind == Energy
                     ? 2 * cross - own_a.select(-1, i) - own_b.select(-1, j)
                     : own_a.select(-1, i) + own_b.select(-1, j) - 2 * cross)
                    .clamp_min(0);
        if (!squared)
          value = detail::safe_root(value);
      } else if (kind == Sinkhorn) {
        value = dual(a.select(-3, i), b.select(-3, j), la.select(-2, i),
                     lb.select(-2, j), p, epsilon, iterations) -
                .5 * (own_a.select(-1, i) + own_b.select(-1, j));
      } else
        value = pair_value(x.select(-3, i), y.select(-3, j), xl.select(-1, i),
                           yl.select(-1, j), directions, kind, p, kernel, scale,
                           epsilon, iterations, squared);
      columns.push_back(value);
    }
    if (!columns.empty())
      rows.push_back(at::stack(columns, -1));
  }
  return mode == 2   ? at::cat(rows, -1)
         : mode == 1 ? at::cat(rows, -1)
                     : at::stack(rows, -2);
}

using Core = Tensor(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                    const Tensor&, int64_t, int64_t, double, int64_t, double,
                    double, int64_t, bool, int64_t);
using Grad = std::tuple<Tensor, Tensor, Tensor>(const Tensor&, const Tensor&,
                                                const Tensor&, const Tensor&,
                                                const Tensor&, const Tensor&,
                                                int64_t, int64_t, double,
                                                int64_t, double, double,
                                                int64_t, bool, int64_t);

std::tuple<Tensor, Tensor, Tensor> transport_compare_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y,
    const Tensor& raw_xl, const Tensor& raw_yl, const Tensor& directions,
    int64_t kind, int64_t mode, double p, int64_t kernel, double scale,
    double epsilon, int64_t iterations, bool squared, int64_t size) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU),
      no_cuda_autocast(c10::DispatchKey::AutocastCUDA);
  auto gx = at::zeros(x.sizes(), x.options().dtype(at::kDouble)),
       gy = at::zeros(y.sizes(), y.options().dtype(at::kDouble));
  auto gd = at::zeros_like(directions, directions.options().dtype(at::kDouble));
  if (gradient.numel() == 0)
    return {gx.to(x.scalar_type()), gy.to(y.scalar_type()),
            gd.to(directions.scalar_type())};
  auto xl = lengths(x, raw_xl), yl = lengths(y, raw_yl);
  auto n = x.size(-3), m = y.size(-3);
  int64_t offset = 0;
  at::AutoGradMode enable(true);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = mode == 2 ? i + 1 : (mode == 1 ? i : 0);
         j < (mode == 1 ? i + 1 : m); ++j) {
      auto a = x.select(-3, i).detach().set_requires_grad(true),
           b = y.select(-3, j).detach().set_requires_grad(true);
      auto d = directions.detach().set_requires_grad(kind == Sliced);
      auto value = pair_value(a, b, xl.select(-1, i), yl.select(-1, j), d, kind,
                              p, kernel, scale, epsilon, iterations, squared);
      auto g = mode == 0
                   ? gradient.select(-2, i).select(-1, j)
                   : gradient.select(-1, mode == 1 ? i : offset + j - i - 1);
      Values inputs{a, b};
      if (kind == Sliced)
        inputs.push_back(d);
      auto grads =
          torch::autograd::grad({value}, inputs, {g}, false, false, true);
      if (grads[0].defined())
        gx.select(-3, i).add_(grads[0].to(at::kDouble));
      if (grads[1].defined())
        gy.select(-3, j).add_(grads[1].to(at::kDouble));
      if (kind == Sliced && grads[2].defined())
        gd.add_(grads[2].to(at::kDouble));
    }
    if (mode == 2)
      offset += n - i - 1;
  }
  return {gx.to(x.scalar_type()), gy.to(y.scalar_type()),
          gd.to(directions.scalar_type())};
}
int64_t kind_for(c10::string_view metric) {
  if (metric == "energy" || metric == "energy_distances")
    return Energy;
  if (metric == "mmd" || metric == "mmd_distances")
    return MMD;
  if (metric == "sliced_wasserstein" ||
      metric == "sliced_wasserstein_distances")
    return Sliced;
  if (metric == "sinkhorn" || metric == "sinkhorn_divergences")
    return Sinkhorn;
  if (metric == "wasserstein" || metric == "wasserstein_distances")
    return Exact;
  TORCH_CHECK(false, "Unsupported transport/distribution comparison: ", metric);
}
}  // namespace

Tensor transport_comparisons(const Tensor& input,
                             const c10::optional<Tensor>& other,
                             c10::string_view metric, c10::string_view mode,
                             const TransportComparisonOptions& options) {
  if (metric == "mmd" || metric == "mmd_distances")
    return mmd_comparisons(input, other, mode, options);
  TORCH_CHECK(!options.gamma.has_value() && !options.degree.has_value() &&
                  !options.coef0.has_value() && !options.alpha.has_value() &&
                  !options.nu.has_value(),
              "Unused MMD kernel parameter");
  using Signature = Tensor(
      const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
      c10::string_view mode, c10::optional<double> p,
      c10::optional<bool> squared, const c10::optional<Tensor>& projections,
      c10::optional<double> regularization, c10::optional<int64_t> max_iter,
      c10::optional<c10::string_view> kernel,
      c10::optional<double> length_scale, const c10::optional<Tensor>& xl,
      const c10::optional<Tensor>& yl, c10::optional<int64_t> chunk_size,
      c10::optional<bool> batched, int64_t feature_dim);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::transport_comparisons", "")
          .typed<Signature>();
  return op.call(input, other, metric, mode, options.p, options.squared,
                 options.projections, options.regularization, options.max_iter,
                 options.kernel
                     ? c10::optional<c10::string_view>(*options.kernel)
                     : c10::nullopt,
                 options.length_scale, options.x1_lengths, options.x2_lengths,
                 options.chunk_size, options.batched, options.feature_dim);
}
namespace {
Tensor transport_comparisons_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view mode, c10::optional<double> p, c10::optional<bool> squared,
    const c10::optional<Tensor>& projections,
    c10::optional<double> regularization, c10::optional<int64_t> max_iter,
    c10::optional<c10::string_view> kernel, c10::optional<double> length_scale,
    const c10::optional<Tensor>& xl, const c10::optional<Tensor>& yl,
    c10::optional<int64_t> chunk_size, c10::optional<bool> batched,
    int64_t feature_dim) {
  TransportComparisonOptions opts;
  opts.p = p;
  opts.squared = squared;
  opts.projections = projections;
  opts.regularization = regularization;
  opts.max_iter = max_iter;
  if (kernel.has_value())
    opts.kernel = std::string(*kernel);
  opts.length_scale = length_scale;
  opts.x1_lengths = xl;
  opts.x2_lengths = yl;
  opts.chunk_size = chunk_size;
  opts.batched = batched;
  opts.feature_dim = feature_dim;
  return transport_comparisons_composite(x, y, metric, mode, opts);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "transport_comparisons(Tensor x1, Tensor? x2, str metric, str "
      "mode='all_pairs', float? p=None, bool? squared=None, Tensor? "
      "projections=None, float? regularization=None, int? max_iter=None, str? "
      "kernel=None, float? length_scale=None, Tensor? x1_lengths=None, Tensor? "
      "x2_lengths=None, int? chunk_size=None, bool? batched=None, int "
      "feature_dim=-1) -> Tensor",
      TORCH_FN(transport_comparisons_impl));
  m.def(
      "_transport_compare(Tensor x, Tensor y, Tensor xl, Tensor yl, Tensor "
      "directions, int kind, int mode, float p, int kernel, float scale, float "
      "epsilon, int iterations, bool squared, int size) -> Tensor");
  m.def(
      "_transport_compare_backward(Tensor grad, Tensor x, Tensor y, Tensor xl, "
      "Tensor yl, Tensor directions, int kind, int mode, float p, int kernel, "
      "float scale, float epsilon, int iterations, bool squared, int size) -> "
      "(Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_transport_compare", TORCH_FN(transport_compare_kernel));
  m.impl("_transport_compare_backward",
         TORCH_FN(transport_compare_backward_kernel));
}

}  // namespace
namespace detail::transport {
using at::Tensor;
using Values = std::vector<Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
Tensor point_kernel(const Tensor& x, const Tensor& y, const Tensor& xv,
                    const Tensor& yv, int64_t kind, int64_t kernel,
                    double scale) {
  return point_tile(x, y, xv, yv, kind, kernel, scale);
}
}  // namespace detail::transport
namespace {
Tensor transport_comparisons_composite(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view metric, c10::string_view mode,
    const TransportComparisonOptions& options) {
  auto kind = kind_for(metric);
  if (kind == MMD)
    return mmd_comparisons(input, other, mode, options);
  int64_t pairing = mode == "all_pairs"   ? 0
                    : mode == "paired"    ? 1
                    : mode == "condensed" ? 2
                                          : -1;
  TORCH_CHECK(pairing >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(
      pairing != 2 || (!other.has_value() && !options.x2_lengths.has_value()),
      "Condensed point sets require one input and x1_lengths only");
  TORCH_CHECK(!options.squared.has_value() || kind == Energy || kind == MMD,
              "Unused parameter: squared");
  TORCH_CHECK(!options.p.has_value() || kind >= Sliced, "Unused parameter: p");
  TORCH_CHECK(!options.projections.has_value() || kind == Sliced,
              "Unused parameter: projections");
  TORCH_CHECK(
      (!options.regularization.has_value() && !options.max_iter.has_value()) ||
          kind == Sinkhorn,
      "Unused parameter: regularization or max_iter");
  TORCH_CHECK(
      (!options.kernel.has_value() && !options.length_scale.has_value()) ||
          kind == MMD,
      "Unused parameter: kernel or length_scale");
  double p = options.p.value_or(kind == Exact ? 1. : 2.),
         scale = options.length_scale.value_or(1.),
         epsilon = options.regularization.value_or(.1);
  auto iterations = options.max_iter.value_or(100),
       size = options.chunk_size.value_or(16);
  TORCH_CHECK(p == 1 || p == 2, "Transport p must be 1 or 2");
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  TORCH_CHECK(iterations > 0, "max_iter must be positive");
  TORCH_CHECK(std::isfinite(scale) && scale > 0,
              "length_scale must be a positive finite scalar");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0,
              "regularization must be a positive finite real scalar");
  auto name = options.kernel.value_or("rbf");
  int64_t kernel = name == "rbf"         ? RBF
                   : name == "laplacian" ? Laplacian
                   : name == "linear"    ? Linear
                                         : -1;
  TORCH_CHECK(kernel >= 0, "kernel must be rbf, laplacian or linear");
  Tensor x = input, y = other.value_or(input),
         directions = options.projections.value_or(at::empty({0}, x.options()));
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Transport comparisons support CPU, CUDA and metadata execution");
  if (!x.device().is_meta() &&
      at::autocast::is_autocast_enabled(x.device().type())) {
    if (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) {
      x = x.to(at::kFloat);
      y = y.to(at::kFloat);
    }
    if (directions.scalar_type() == at::kHalf ||
        directions.scalar_type() == at::kBFloat16)
      directions = directions.to(at::kFloat);
  }
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU),
      no_cuda_autocast(c10::DispatchKey::AutocastCUDA);
  TORCH_CHECK(
      x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
      "Distribution comparisons require float32 or float64 outside autocast");
  auto clouds = detail::cloud_inputs(
      x, other.has_value() ? c10::optional<Tensor>(y) : c10::nullopt,
      options.x1_lengths, options.x2_lengths, options.batched,
      options.feature_dim, pairing);
  x = clouds.x;
  y = clouds.y;
  if (kind == Sliced) {
    TORCH_CHECK(options.projections.has_value(),
                "Sliced Wasserstein requires projections");
    TORCH_CHECK(directions.layout() == at::kStrided && directions.dim() == 2 &&
                    directions.sym_size(0) == x.sym_size(-1) &&
                    directions.sym_size(1) > 0,
                "projections must be dense with shape [C,L]");
    TORCH_CHECK(directions.device() == x.device() &&
                    directions.scalar_type() == x.scalar_type(),
                "projections must have the input dtype and device");
    auto norm = at::linalg_vector_norm(directions, 2, {0}, false);
    double tolerance = x.scalar_type() == at::kFloat ? 1.e-5 : 1.e-10;
    utils::check_values(
        directions.isfinite().all() &
            at::isclose(norm, at::ones_like(norm), tolerance, tolerance).all(),
        "projections must contain finite unit direction columns");
  }
  if (kind == Sinkhorn) {
    auto minimum = x.scalar_type() == at::kFloat
                       ? std::numeric_limits<float>::min()
                       : std::numeric_limits<double>::min();
    auto maximum = x.scalar_type() == at::kFloat
                       ? double(std::numeric_limits<float>::max())
                       : std::numeric_limits<double>::max();
    TORCH_CHECK(epsilon >= minimum && epsilon <= maximum,
                "regularization must be a positive normal finite value in the "
                "transport_compare_kernel dtype");
  }
  return detail::transport::_transport_compare(
      x, y, clouds.xl, clouds.yl, directions, kind, pairing, p, kernel, scale,
      epsilon, iterations, options.squared.value_or(false), size);
}
}  // namespace
}  // namespace torchpairwise::ops
