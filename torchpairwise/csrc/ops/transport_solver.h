#pragma once
#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

namespace torchpairwise::ops::detail {
// Small unsigned integer used for exact binary marginal capacities. Keeping it
// here avoids an external big-integer dependency in standalone LibTorch builds.
struct Capacity {
  std::vector<uint32_t> limbs;
  Capacity(uint64_t value = 0) {
    if (value) {
      limbs.push_back(uint32_t(value));
      if (value >> 32)
        limbs.push_back(uint32_t(value >> 32));
    }
  }
  bool empty() const {
    return limbs.empty();
  }
  void trim() {
    while (!limbs.empty() && !limbs.back())
      limbs.pop_back();
  }
  bool operator<(const Capacity& b) const {
    if (limbs.size() != b.limbs.size())
      return limbs.size() < b.limbs.size();
    for (size_t i = limbs.size(); i-- > 0;)
      if (limbs[i] != b.limbs[i])
        return limbs[i] < b.limbs[i];
    return false;
  }
  Capacity& operator+=(const Capacity& b) {
    limbs.resize(std::max(limbs.size(), b.limbs.size()), 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < limbs.size(); ++i) {
      uint64_t v =
          uint64_t(limbs[i]) + (i < b.limbs.size() ? b.limbs[i] : 0) + carry;
      limbs[i] = uint32_t(v);
      carry = v >> 32;
    }
    if (carry) {
      limbs.push_back(uint32_t(carry));
    }
    return *this;
  }
  Capacity& operator-=(const Capacity& b) {
    TORCH_CHECK(!(*this < b), "Invalid transport capacity subtraction");
    uint64_t borrow = 0;
    for (size_t i = 0; i < limbs.size(); ++i) {
      uint64_t rhs = (i < b.limbs.size() ? uint64_t(b.limbs[i]) : 0) + borrow;
      uint64_t lhs = limbs[i];
      limbs[i] = uint32_t(lhs - rhs);
      borrow = lhs < rhs;
    }
    trim();
    return *this;
  }
  Capacity shifted(int bits) const {
    if (empty()) {
      return {};
    }
    Capacity out;
    out.limbs.resize(limbs.size() + bits / 32 + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < limbs.size(); ++i) {
      uint64_t v = (uint64_t(limbs[i]) << (bits % 32)) | carry;
      out.limbs[i + bits / 32] = uint32_t(v);
      carry = v >> 32;
    }
    out.limbs.back() = uint32_t(carry);
    out.trim();
    return out;
  }
  Capacity operator*(const Capacity& b) const {
    Capacity out;
    out.limbs.resize(limbs.size() + b.limbs.size(), 0);
    for (size_t i = 0; i < limbs.size(); ++i) {
      uint64_t carry = 0;
      for (size_t j = 0; j < b.limbs.size(); ++j) {
        uint64_t v = uint64_t(limbs[i]) * b.limbs[j] + out.limbs[i + j] + carry;
        out.limbs[i + j] = uint32_t(v);
        carry = v >> 32;
      }
      if (!b.empty())
        out.limbs[i + b.limbs.size()] = uint32_t(carry);
    }
    out.trim();
    return out;
  }
  std::pair<double, int> scaled() const {
    if (empty()) {
      return {0., 0};
    }
    int n = int(limbs.size());
    double value = limbs.back();
    if (n > 1)
      value += std::ldexp(double(limbs[n - 2]), -32);
    if (n > 2)
      value += std::ldexp(double(limbs[n - 3]), -64);
    return {value, 32 * (n - 1)};
  }
};
inline double capacity_ratio(const Capacity& a, const Capacity& b) {
  auto x = a.scaled(), y = b.scaled();
  return std::ldexp(x.first / y.first, x.second - y.second);
}
struct TransportPlan {
  std::vector<int64_t> rows, columns;
  std::vector<double> masses, alpha, beta;
};
inline TransportPlan solve_transport(const at::Tensor& raw_cost,
                                     const std::vector<double>& a,
                                     const std::vector<double>& b) {
  TORCH_CHECK(raw_cost.device().is_cpu() &&
                  raw_cost.scalar_type() == at::kDouble && raw_cost.dim() == 2,
              "Transport solver requires a CPU double cost matrix");
  auto cost = raw_cost.contiguous();
  auto c = cost.accessor<double, 2>();
  int64_t n = a.size(), m = b.size();
  TORCH_CHECK(n && m && cost.size(0) == n && cost.size(1) == m,
              "Transport support and cost dimensions must match");
  auto capacities = [](const std::vector<double>& masses) {
    int minimum = 0;
    std::vector<std::pair<uint64_t, int>> parts;
    for (double mass : masses) {
      TORCH_CHECK(std::isfinite(mass) && mass > 0,
                  "Transport masses must be finite and positive");
      int exponent;
      double fraction = std::frexp(mass, &exponent);
      parts.emplace_back(uint64_t(std::ldexp(fraction, 53)), exponent - 53);
      minimum = std::min(minimum, exponent - 53);
    }
    std::vector<Capacity> result;
    Capacity sum;
    for (auto p : parts) {
      result.push_back(Capacity(p.first).shifted(p.second - minimum));
      sum += result.back();
    }
    return std::make_pair(result, sum);
  };
  auto aa = capacities(a), bb = capacities(b);
  auto total = aa.second * bb.second;
  struct Edge {
    int64_t to, reverse;
    Capacity capacity;
    double price;
  };
  int64_t source = n + m, sink = source + 1;
  std::vector<std::vector<Edge>> graph(n + m + 2);
  auto add = [&](int64_t u, int64_t v, Capacity capacity, double price) {
    int64_t index = graph[u].size();
    graph[u].push_back({v, int64_t(graph[v].size()), capacity, price});
    graph[v].push_back({u, index, {}, -price});
    return index;
  };
  double scale = 0.;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < m; ++j) {
      TORCH_CHECK(std::isfinite(c[i][j]) && c[i][j] >= 0,
                  "Transport costs must be finite and non-negative");
      scale = std::max(scale, c[i][j]);
    }
  if (!scale)
    scale = 1.;
  std::vector<std::vector<int64_t>> edges(n, std::vector<int64_t>(m));
  for (int64_t i = 0; i < n; ++i) {
    add(source, i, aa.first[i] * bb.second, 0.);
    for (int64_t j = 0; j < m; ++j)
      edges[i][j] = add(i, n + j, total, c[i][j] / scale);
  }
  for (int64_t j = 0; j < m; ++j)
    add(n + j, sink, bb.first[j] * aa.second, 0.);
  std::vector<double> potentials(graph.size(), 0.);
  Capacity remaining = total;
  while (!remaining.empty()) {
    std::vector<double> distance(graph.size(), INFINITY);
    std::vector<std::pair<int64_t, int64_t>> previous(graph.size(), {-1, -1});
    using Item = std::pair<double, int64_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    distance[source] = 0.;
    queue.push({0., source});
    while (!queue.empty()) {
      auto [current, u] = queue.top();
      queue.pop();
      if (current != distance[u])
        continue;
      for (int64_t k = 0; k < int64_t(graph[u].size()); ++k) {
        const auto& e = graph[u][k];
        if (e.capacity.empty())
          continue;
        double reduced = e.price + potentials[u] - potentials[e.to];
        double magnitude =
            std::max({1., std::abs(e.price), std::abs(potentials[u]),
                      std::abs(potentials[e.to])});
        double tolerance =
            64 * (std::nextafter(magnitude, INFINITY) - magnitude);
        TORCH_CHECK(reduced >= -tolerance,
                    "Transport solver lost numerical dual feasibility");
        double candidate = current + std::max(0., reduced);
        if (candidate < distance[e.to]) {
          distance[e.to] = candidate;
          previous[e.to] = {u, k};
          queue.push({candidate, e.to});
        }
      }
    }
    TORCH_CHECK(previous[sink].first >= 0,
                "Transport solver could not find a feasible coupling");
    for (size_t v = 0; v < graph.size(); ++v)
      if (std::isfinite(distance[v]))
        potentials[v] += distance[v];
    Capacity amount = remaining;
    for (int64_t v = sink; v != source;) {
      auto [u, k] = previous[v];
      if (graph[u][k].capacity < amount)
        amount = graph[u][k].capacity;
      v = u;
    }
    for (int64_t v = sink; v != source;) {
      auto [u, k] = previous[v];
      auto& e = graph[u][k];
      e.capacity -= amount;
      graph[v][e.reverse].capacity += amount;
      v = u;
    }
    remaining -= amount;
  }
  TransportPlan result;
  for (int64_t i = 0; i < n; ++i) {
    result.alpha.push_back(-potentials[i] * scale);
    for (int64_t j = 0; j < m; ++j) {
      auto flow = total;
      flow -= graph[i][edges[i][j]].capacity;
      if (!flow.empty()) {
        result.rows.push_back(i);
        result.columns.push_back(j);
        result.masses.push_back(capacity_ratio(flow, total));
      }
    }
  }
  for (int64_t j = 0; j < m; ++j) {
    result.beta.push_back(potentials[n + j] * scale);
  }
  return result;
}
}  // namespace torchpairwise::ops::detail
