#include "fused_reducers.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include <atomic>
#include <cmath>

#include <ATen/ExpandUtils.h>

#include "cuda/pair_functors.cuh"
#include "expression_operations_detail.h"
#include "utils/comparison_dispatch.h"

namespace torchpairwise::ops::detail::fused {
namespace {
using pairs::Feature;
using pairs::Outer;
using streaming::Payload;
using streaming::X;
using streaming::Y;
// Largest feature count with a register-resident row (kernel bucket).
constexpr int64_t max_features = 64;
std::atomic<bool> fused_flag{true};
std::atomic<int64_t> fused_count{0};
bool set_fused_reducers(bool value) {
  return fused_flag.exchange(value);
}
int64_t fused_reducer_calls() {
  return fused_count.load();
}
}  // namespace

bool enabled() {
  return fused_flag.load(std::memory_order_relaxed);
}
void count() {
  fused_count.fetch_add(1, std::memory_order_relaxed);
}

namespace {
using streaming::Settings;
using streaming::Slots;
using streaming::XL;
using streaming::YL;
// Metric of a named leaf: pair functor and scalar parameters, or nullopt.
c10::optional<Plan> metric_plan(at::TensorList v, const Settings& s) {
  constexpr int64_t operands =
      (int64_t(1) << X) | (int64_t(1) << Y) | (int64_t(1) << Payload);
  if (s.expression || s.family || s.self_cosine || s.rank != 2 ||
      (s.mask & ~operands))
    return c10::nullopt;
  const auto &x = v[X], &y = v[Y];
  if (!x.is_cuda() || y.device() != x.device() ||
      (x.scalar_type() != at::kFloat && x.scalar_type() != at::kDouble) ||
      y.scalar_type() != x.scalar_type())
    return c10::nullopt;
  auto features = x.size(-1);
  if (features < 1 || features > max_features || y.size(-1) != features)
    return c10::nullopt;
  // Settings::reals order: p, base, alpha, length_scale, nu, gamma, coef0, ...
  const auto& r = s.reals;
  auto name = canonical_comparison_name(s.metric);
  Plan p{};
  p.differentiable = true;
  for (auto& slot : p.chain_slot)
    slot = -1;
  auto set = [&](Feature feature, Outer outer) {
    p.feature = int(feature);
    p.outer = int(outer);
  };
  if (!s.kernel) {
    // Euclidean (and Minkowski p=2) stays on at::cdist: its GEMM form gives
    // sqrt(rounding residual) at coincident points and NaN for infinite
    // coordinates, which a difference loop would not reproduce.
    if (name == "sqeuclidean")
      set(Feature::Square, Outer::Identity);
    else if (name == "manhattan")
      set(Feature::Abs, Outer::Identity);
    else if (name == "chebyshev") {
      set(Feature::Max, Outer::Identity);
      // Native splits tied maxima evenly; backward stays on the tile path.
      p.differentiable = false;
    } else if (name == "minkowski") {
      auto power = r[0].value_or(2.);
      if (!(power > 0) || std::isinf(power) || power == 2)
        return c10::nullopt;
      if (power == 1)
        // pow(value, 1) is a copy in ATen.
        set(Feature::Abs, Outer::Identity);
      else {
        set(Feature::Power, Outer::Root);
        p.power = power;
        p.a = 1. / power;
      }
    } else
      return c10::nullopt;
  } else {
    auto gamma = r[5].value_or(1. / double(features));
    if (name == "rbf" || name == "laplacian") {
      set(name == "rbf" ? Feature::Square : Feature::Abs, Outer::ScaledExp);
      p.a = -gamma;
    } else if (name == "matern") {
      auto nu = r[4].value_or(1.5);
      if (nu != .5 && nu != 1.5 && nu != 2.5)
        return c10::nullopt;
      set(Feature::Square, nu == .5    ? Outer::Matern05
                           : nu == 1.5 ? Outer::Matern15
                                       : Outer::Matern25);
      p.a = r[3].value_or(1.);  // length scale; reciprocal taken in-dtype
      p.b = std::sqrt(2 * nu);
    } else if (name == "rational_quadratic") {
      set(Feature::Square, Outer::RationalQuadratic);
      p.a = r[3].value_or(1.);  // length scale
      p.b = r[2].value_or(1.);  // alpha
    } else if (name == "linear")
      set(Feature::Product, Outer::Identity);
    else if (name == "polynomial" || name == "sigmoid") {
      // builtin_comparisons' defaults: degree 3, coef0 1, gamma 1 / D.
      set(Feature::Product,
          name == "polynomial" ? Outer::Polynomial : Outer::Sigmoid);
      p.a = gamma;
      p.b = r[6].value_or(1.);
      if (name == "polynomial") {
        auto degree = s.integers[0].value_or(3);
        if (degree < 0)
          return c10::nullopt;
        p.c = double(degree);
      }
    } else
      return c10::nullopt;
  }
  return p;
}
// Lowers the path from an expression's single leaf to its root into the
// chain, false for anything else. Ops off the path are lowered numbers,
// row/column values (op 38/39 nodes and op 9/10 weights) and where
// conditions on boolean row/column values (or their logical and); user scalar
// tensors, other unary ops and a leaf used twice keep the tile path. Row and
// column values are left raw (cast to the value dtype) for plan() to gather.
bool parse_chain(const streaming::ExpressionPlan& e, at::TensorList v,
                 const Tensor& partner, Plan& p) {
  const auto& nodes = e.packet.nodes;
  const auto extras = Slots * (1 + int64_t(e.leaves.size()));
  const auto count = int64_t(nodes.size());
  // Whether a node depends on the leaf.
  std::vector<char> path(count, 0);
  for (int64_t index = 0; index < count; ++index) {
    auto op = nodes[index][0], a = nodes[index][1], b = nodes[index][2];
    if (op == 0)
      path[index] = 1;
    else if (op == 1 || op == 2 || op == 38 || op == 39 || op == 40 ||
             op == 41)
      continue;
    else
      path[index] = (a >= 0 && path[a]) ||
                     (b >= 0 && !streaming::expression_axis_slot(op) &&
                      !streaming::expression_scalar_bound(op) && path[b]);
  }
  auto number = [&](int64_t node, Tensor& value) {
    // Lowered numbers are op 1 nodes tagged -2 (expression_sources.cpp).
    if (nodes[node][0] != 1 || nodes[node][2] != -2 ||
        extras + nodes[node][1] >= int64_t(v.size()))
      return false;
    value = v[extras + nodes[node][1]];
    if (value.dim() != 0 || value.device() != partner.device() ||
        value.is_complex())
      return false;
    // A zero-dimensional operand takes the floating partner's dtype before
    // the tile path's add/sub/mul, so integer and wider numbers cast first.
    if (value.scalar_type() != partner.scalar_type())
      value = value.to(partner.scalar_type());
    return true;
  };
  // Row/column values; conditions must be boolean (a wider floating operand
  // would have widened the expression dtype).
  auto values = [&](int64_t slot, bool condition, Tensor& value,
                    int64_t& gradient) {
    if (slot < 0 || extras + slot >= int64_t(v.size()))
      return false;
    value = v[extras + slot];
    if (value.dim() < 1 || value.device() != partner.device() ||
        value.is_complex() ||
        (condition && value.scalar_type() != at::kBool) ||
        (value.is_floating_point() &&
         value.scalar_type() != partner.scalar_type()))
      return false;
    gradient = value.is_floating_point() ? extras + slot : -1;
    value = value.to(partner.scalar_type());
    return true;
  };
  struct Step {
    int op, axis;
    Tensor constant, values;
    int64_t slot;
  };
  std::vector<Step> steps;  // root first
  auto push = [&](pairs::ChainOp op, int axis = 0, Tensor constant = {},
                  Tensor value = {}, int64_t slot = -1) {
    steps.push_back({int(op), axis, std::move(constant), std::move(value),
                     slot});
  };
  // Operand of a binary op: a number, or a row/column value node.
  auto operand = [&](int64_t node, int& axis, Tensor& constant, Tensor& value,
                     int64_t& slot) {
    auto op = nodes[node][0];
    if (op == 38 || op == 39) {
      axis = op == 38 ? 1 : 2;
      return values(nodes[node][2], false, value, slot);
    }
    axis = 0;
    slot = -1;
    return number(node, constant);
  };
  // A where condition on a row/column value as a Keep (true keeps the value)
  // or Drop step.
  auto mask = [&](int64_t node, bool keep, const Tensor& fill) {
    auto op = nodes[node][0];
    Tensor value;
    int64_t slot;
    if ((op != 38 && op != 39) || !values(nodes[node][2], true, value, slot))
      return false;
    push(keep ? pairs::ChainOp::Keep : pairs::ChainOp::Drop, op == 38 ? 1 : 2,
         fill, value);
    return true;
  };
  // where(a & b, x, fill) == where(a, where(b, x, fill), fill).
  auto condition = [&](int64_t node, bool keep, const Tensor& fill) {
    if (nodes[node][0] != 32)
      return mask(node, keep, fill);
    return keep && mask(nodes[node][1], true, fill) &&
           mask(nodes[node][2], true, fill);
  };
  int64_t current = count - 1;
  if (count == 0 || !path[current])
    return false;
  while (nodes[current][0] != 0) {
    if (int64_t(steps.size()) >= max_chain + 1)
      return false;
    const auto& node = nodes[current];
    auto op = node[0], a = node[1], b = node[2];
    int axis;
    Tensor constant, value;
    int64_t slot;
    if (op == 8 || op == 7) {
      push(op == 8 ? pairs::ChainOp::Neg : pairs::ChainOp::Exp);
      current = a;
    } else if (op == 9 || op == 10) {
      if (!path[a] || !values(b, false, value, slot))
        return false;
      push(pairs::ChainOp::Mul, op == 9 ? 1 : 2, {}, value, slot);
      current = a;
    } else if (op == 3 || op == 4 || op == 5 || op == 12) {
      if (path[a] == path[b])
        return false;
      bool left = path[a];
      if (!operand(left ? b : a, axis, constant, value, slot))
        return false;
      auto code = op == 3   ? pairs::ChainOp::Add
                  : op == 4 ? (left ? pairs::ChainOp::Sub
                                    : pairs::ChainOp::SubFrom)
                  : op == 5 ? pairs::ChainOp::Mul
                            : pairs::ChainOp::Div;
      if (op == 12 && !left)
        return false;
      push(code, axis, constant, value, slot);
      current = left ? a : b;
    } else if (op == 34) {
      const auto& branches = nodes[b];
      if (path[a] || path[branches[1]] == path[branches[2]])
        return false;
      bool keep = path[branches[1]];
      Tensor fill;
      if (!number(keep ? branches[2] : branches[1], fill) ||
          !condition(a, keep, fill))
        return false;
      current = keep ? branches[1] : branches[2];
    } else
      return false;
  }
  if (nodes[current][1] != 0 || int64_t(steps.size()) > max_chain)
    return false;
  p.chain_size = int(steps.size());
  for (int k = 0; k < p.chain_size; ++k) {
    auto& step = steps[steps.size() - 1 - k];
    p.chain_op[k] = step.op;
    p.chain_axis[k] = step.axis;
    p.chain_constant[k] = step.constant;
    p.chain_values[k] = step.values;
    p.chain_slot[k] = step.slot;
  }
  return true;
}
}  // namespace

c10::optional<Plan> plan(at::TensorList v, const Settings& s) {
#ifndef WITH_CUDA
  return c10::nullopt;
#else
  if (!enabled() || at::autocast::is_autocast_enabled(at::kCUDA) ||
      (s.dtype != int64_t(at::kFloat) && s.dtype != int64_t(at::kDouble)))
    return c10::nullopt;
  auto dtype = at::ScalarType(s.dtype);
  if (s.has(Payload) &&
      (!v[Payload].is_cuda() || v[Payload].scalar_type() != dtype))
    return c10::nullopt;
  c10::optional<Plan> p;
  int64_t rows = 0, columns = 0;
  Tensor row_map, column_map;
  if (!s.expression) {
    p = metric_plan(v, s);
    rows = s.transposed() ? Y : X;
    columns = s.transposed() ? X : Y;
  } else {
    // One leaf, whose operands follow the top-level slots; XL/YL map the
    // expression's rows/columns to the leaf's logical rows/columns.
    const auto& e = *s.expression;
    if (e.leaves.size() != 1 || int64_t(v.size()) < 2 * Slots)
      return c10::nullopt;
    const auto& leaf = e.leaves[0];
    auto operands = v.slice(Slots, Slots);
    p = metric_plan(operands, leaf);
    if (!p || operands[X].scalar_type() != dtype ||
        !parse_chain(e, v, operands[X], *p))
      return c10::nullopt;
    rows = Slots + (leaf.transposed() ? Y : X);
    columns = Slots + (leaf.transposed() ? X : Y);
    row_map = v[XL];
    column_map = v[YL];
    if (s.transposed()) {
      std::swap(rows, columns);
      std::swap(row_map, column_map);
      for (int k = 0; k < p->chain_size; ++k)
        if (p->chain_axis[k] != 0)
          p->chain_axis[k] = 3 - p->chain_axis[k];
    }
  }
  // Native forms dot products with a GEMM, so fused values differ from the
  // tile path in rounding; selections promise native's exact ties and keep
  // the tile path.
  if (p && Feature(p->feature) == Feature::Product && s.kind() == "neighbors")
    return c10::nullopt;
  // The fused softmax VJP centers with the rounded output (native recomputes
  // the center from tile probabilities). Polynomial slopes grow as
  // z^(degree-1) and amplify that rounding, so this backward keeps tiles.
  // softmax_matmul settings are the statistics' "sum" with a payload.
  if (p && Outer(p->outer) == Outer::Polynomial && p->c >= 2 &&
      s.kind() == "sum" && s.has(Payload))
    p->differentiable = false;
  // Measured crossovers (step 4.5, Quadro RTX 6000, N 1e4 and 1e5): beyond
  // these the tile path is faster. pow per feature makes Minkowski slower
  // than tiles from D = 8; the Chebyshev max from D = 32; a matrix product
  // with more than one 16-column payload block loses to the tiles' GEMM
  // from D = 8. softmax_matmul ("sum" with a payload) stays fused.
  if (p) {
    const auto features = v[rows].size(-1);
    const auto feature = Feature(p->feature);
    const bool product = s.has(Payload) &&
                         (s.kind() == "matmul" || s.kind() == "kernel_sum");
    if ((feature == Feature::Power && features > 4) ||
        (feature == Feature::Max && features > 16) ||
        (product && v[Payload].size(-1) > 16 && features > 4))
      return c10::nullopt;
  }
  // Raw dispatcher calls may mix devices; the tile path rejects those.
  if (!p || v[rows].scalar_type() != dtype ||
      (s.has(Payload) && v[Payload].device() != v[rows].device()) ||
      (row_map.defined() && row_map.device() != v[rows].device()) ||
      (column_map.defined() && column_map.device() != v[rows].device()))
    return c10::nullopt;
  p->batch = streaming::batch_shape(v, s);
  int64_t batches = 1;
  for (auto size : p->batch)
    batches *= size;
  auto flatten = [&](const Tensor& value, const Tensor& map) {
    auto shape = p->batch;
    shape.insert(shape.end(), {value.size(-2), value.size(-1)});
    auto flat =
        value.expand(shape).reshape({batches, value.size(-2), value.size(-1)});
    return (map.defined() ? flat.index_select(1, map) : flat).contiguous();
  };
  p->row_slot = rows;
  p->column_slot = columns;
  p->row_map = row_map;
  p->column_map = column_map;
  p->rows = flatten(v[rows], row_map);
  p->columns = flatten(v[columns], column_map);
  // Row/column values follow the samples they index: [..., samples] gathered
  // through the same map, kept [1, samples] when shared across the batch.
  for (int k = 0; k < p->chain_size; ++k) {
    auto& value = p->chain_values[k];
    if (!value.defined())
      continue;
    const auto& map = p->chain_axis[k] == 1 ? row_map : column_map;
    if (map.defined() && map.device() != value.device())
      return c10::nullopt;
    auto flat = value.dim() == 1
                    ? value.unsqueeze(0)
                    : flatten(value.unsqueeze(-1), Tensor()).squeeze(-1);
    value = (map.defined() ? flat.index_select(1, map) : flat).contiguous();
  }
  return p;
#endif
}

Tensor to_operand(const Plan& p, const Tensor& gradient, bool rows,
                  at::TensorList v) {
  const auto& target = v[rows ? p.row_slot : p.column_slot];
  const auto& map = rows ? p.row_map : p.column_map;
  auto value = gradient;
  if (map.defined())
    value = at::zeros({gradient.size(0), target.size(-2), gradient.size(2)},
                      gradient.options())
                .index_add_(1, map, gradient);
  auto shape = p.batch;
  shape.insert(shape.end(), {value.size(1), value.size(2)});
  return at::sum_to(value.reshape(shape), target.sizes()).contiguous();
}

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  // Private A/B switch and dispatch counter for tests and benchmarks.
  m.def("_set_fused_reducers(bool enabled) -> bool", &set_fused_reducers);
  m.def("_fused_reducer_calls() -> int", &fused_reducer_calls);
}
}  // namespace torchpairwise::ops::detail::fused
