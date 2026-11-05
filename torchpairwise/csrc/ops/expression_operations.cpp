#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include "expression_operations_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::streaming {
namespace {
int64_t extras_start(const ExpressionPlan& p) {
  return Slots * (1 + int64_t(p.leaves.size()));
}
Tensor number(const at::Scalar& value, const Tensor& like,
              const Tensor& partner = Tensor()) {
  return at::scalar_tensor(
      value, like.options().dtype(expression_number_dtype(value, partner)));
}
Tensor leaf_value(at::TensorList local, const Settings& s, bool paired,
                  const Tensor& rows, const Tensor& columns) {
  if (s.family && *s.family == "_matrix") {
    auto r = s.transposed() ? columns : rows;
    auto c = s.transposed() ? rows : columns;
    auto value = local[X].index(
        {paired ? r : r.unsqueeze(-1), paired ? c : c.unsqueeze(0)});
    return !paired && s.transposed() ? value.transpose(-1, -2) : value;
  }
  return evaluate(local, s, paired, 0, 0, rows, columns);
}
Tensors expression_leaves(at::TensorList local, const ExpressionPlan& p,
                          bool paired) {
  Tensors values;
  for (size_t k = 0; k < p.leaves.size(); ++k)
    values.push_back(leaf_value(local.slice(Slots * (k + 1), Slots),
                                p.leaves[k], paired, local[XL], local[YL]));
  return values;
}
Tensors evaluate_leaves(at::TensorList local, const ExpressionPlan& p,
                        bool paired, ExpressionReplay* replay) {
  // Consumers need detached expression values, but their source VJPs need
  // the source graph. Build it once for this backward tile and release it
  // after accumulation. Forward evaluation continues to record no graph.
  at::AutoGradMode enable(replay != nullptr);
  auto values = expression_leaves(local, p, paired);
  if (replay)
    replay->leaves = values;
  return values;
}
Tensors run_graph(at::TensorList local, const ExpressionPlan& p,
                  const Tensors& leaves, bool paired) {
  // Explicit weighting nodes keep row/column axes distinct even on square
  // tiles.
  Tensors values;
  const auto& nodes = p.packet.nodes;
  // Binary consumers type a number operand by its partner, as Python does.
  auto operand = [&](int64_t index, const Tensor& partner) {
    return nodes[index][0] == 2
               ? number(p.packet.numbers[nodes[index][1]], local[X], partner)
               : values[index];
  };
  for (const auto& node : nodes) {
    auto op = node[0], a = node[1], b = node[2];
    Tensor value;
    if (op == 0)
      value = leaves[a];
    else if (op == 1)
      value = local[extras_start(p) + a];
    else if (op == 2)
      value = number(p.packet.numbers[a], local[X]);
    else if (expression_binary(op) || expression_scalar_bound(op)) {
      auto left = operand(a, values[b]), right = operand(b, values[a]);
      switch (op) {
        case 3:
          value = left + right;
          break;
        case 4:
          value = left - right;
          break;
        case 5:
          value = left * right;
          break;
        case 12:
          value = left / right;
          break;
        case 13:
          value = at::pow(left, right);
          break;
        case 14:
          value = at::minimum(left, right);
          break;
        case 15:
          value = at::maximum(left, right);
          break;
        case 16:
          value = at::clamp_min(left, right);
          break;
        case 17:
          value = at::clamp_max(left, right);
          break;
        case 26:
          value = at::lt(left, right);
          break;
        case 27:
          value = at::le(left, right);
          break;
        case 28:
          value = at::gt(left, right);
          break;
        case 29:
          value = at::ge(left, right);
          break;
        case 30:
          value = at::eq(left, right);
          break;
        case 31:
          value = at::ne(left, right);
          break;
        case 32:
          value = at::logical_and(left, right);
          break;
        case 33:
          value = at::logical_or(left, right);
          break;
      }
    } else if (expression_unary(op)) {
      switch (op) {
        case 6:
          value = values[a].square();
          break;
        case 7:
          value = values[a].exp();
          break;
        case 8:
          value = -values[a];
          break;
        case 18:
          value = values[a].sqrt();
          break;
        case 19:
          value = values[a].log();
          break;
        case 20:
          value = values[a].sin();
          break;
        case 21:
          value = values[a].cos();
          break;
        case 22:
          value = values[a].tanh();
          break;
        case 23:
          value = values[a].sigmoid();
          break;
        case 24:
          value = values[a].abs();
          break;
        case 25:
          value = at::logical_not(values[a]);
          break;
      }
    } else if (expression_axis_slot(op)) {
      auto weights = local[extras_start(p) + b];
      value = paired ? weights
                     : weights.unsqueeze(expression_row_slot(op) ? -1 : 0);
      if (op == 9 || op == 10)
        value = operand(a, weights) * value;
    } else if (op == 40 || op == 41) {
      value = local[op == 40 ? XL : YL];
      if (!paired)
        value = value.unsqueeze(op == 40 ? -1 : 0);
    } else if (op == 34) {
      const auto& branches = nodes[b];
      value = at::where(values[a], operand(branches[1], values[branches[2]]),
                        operand(branches[2], values[branches[1]]));
    }
    // Pair nodes reserve an index but deliberately have no tensor value.
    values.push_back(value);
  }
  return values;
}
Tensor broadcast_tile(const Tensor& value, at::TensorList local, bool paired) {
  TORCH_CHECK(value.defined(), "Pair nodes cannot be expression roots");
  return paired ? value.expand_as(local[XL])
                : value.expand({local[XL].numel(), local[YL].numel()});
}
Settings root_settings(const ExpressionPacket& packet,
                       c10::string_view operation, int64_t dtype, int64_t size,
                       bool exclusion, bool transposed) {
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  TORCH_CHECK(at::ScalarType(dtype) == at::kFloat ||
                  at::ScalarType(dtype) == at::kDouble ||
                  at::ScalarType(dtype) == at::kLong ||
                  at::ScalarType(dtype) == at::kInt ||
                  at::ScalarType(dtype) == at::kShort ||
                  at::ScalarType(dtype) == at::kChar ||
                  at::ScalarType(dtype) == at::kByte ||
                  at::ScalarType(dtype) == at::kBool,
              "Unsupported expression dtype");
  TORCH_CHECK(operation == "sum" || operation == "prod" || operation == "all" ||
                  operation == "any" || operation == "neighbors" ||
                  operation == "maximum" || operation == "matmul" ||
                  operation == "entries" || operation == "diagonal" ||
                  operation == "form",
              "Unknown expression consumer");
  Settings s{"_expression",
             (transposed ? "transpose_" : "") +
                 std::string(operation == "maximum" ? "neighbors" : operation),
             false,
             operation == "maximum",
             exclusion,
             false,
             operation == "neighbors" || operation == "maximum",
             2,
             1,
             size,
             dtype,
             3,
             {},
             {},
             {},
             std::string("_expression")};
  s.expression = expression_plan(packet);
  return s;
}
}  // namespace
at::ScalarType expression_number_dtype(const at::Scalar& value,
                                       const Tensor& partner) {
  if (value.isBoolean())
    return at::kBool;
  if (value.isIntegral(false))
    return at::kLong;
  auto dtype = at::get_default_dtype_as_scalartype();
  // A dimensioned partner keeps its own dtype, and a zero-dimensional one
  // promotes with the default dtype either way, so result dtypes are unchanged.
  return partner.defined() && partner.is_floating_point()
             ? at::promoteTypes(partner.scalar_type(), dtype)
             : dtype;
}
std::vector<at::ScalarType> expression_number_dtypes(at::TensorList operands,
                                                     const ExpressionPlan& plan,
                                                     const Tensors& leaves) {
  at::NoGradGuard guard;
  auto values = run_graph(operands, plan, leaves, false);
  const auto& nodes = plan.packet.nodes;
  std::vector<c10::optional<at::ScalarType>> uses(nodes.size());
  std::vector<bool> conflicting(nodes.size(), false);
  auto use = [&](int64_t index, const Tensor& partner) {
    if (nodes[index][0] != 2)
      return;
    auto dtype =
        expression_number_dtype(plan.packet.numbers[nodes[index][1]], partner);
    conflicting[index] =
        conflicting[index] || (uses[index] && *uses[index] != dtype);
    uses[index] = dtype;
  };
  for (const auto& node : nodes) {
    auto op = node[0], a = node[1], b = node[2];
    if (expression_binary(op) || expression_scalar_bound(op) || op == 35) {
      use(a, values[b]);
      use(b, values[a]);
    } else if (op == 9 || op == 10) {
      use(a, operands[extras_start(plan) + b]);
    } else if (expression_unary(op)) {
      use(a, Tensor());
    }
  }
  std::vector<at::ScalarType> result;
  for (size_t k = 0; k < nodes.size(); ++k) {
    auto fallback =
        nodes[k][0] == 2
            ? expression_number_dtype(plan.packet.numbers[nodes[k][1]],
                                      Tensor())
            : (values[k].defined() ? values[k].scalar_type() : at::kBool);
    // A number shared by differently typed partners keeps the default lowering.
    result.push_back(uses[k] && !conflicting[k] ? *uses[k] : fallback);
  }
  return result;
}
at::ScalarType infer_expression_dtype(at::TensorList operands,
                                      const ExpressionPlan& plan,
                                      const Tensors& leaves, int64_t root) {
  at::NoGradGuard guard;
  auto values = run_graph(operands, plan, leaves, false);
  TORCH_CHECK(root >= -1 && root < int64_t(values.size()),
              "Invalid expression root");
  auto value = root < 0 ? values.back() : values[root];
  TORCH_CHECK(value.defined(), "Pair nodes cannot be expression roots");
  return value.scalar_type();
}
std::shared_ptr<const ExpressionPlan> expression_plan(
    const ExpressionPacket& p) {
  auto plan = std::make_shared<ExpressionPlan>();
  auto n = p.names.size();
  TORCH_CHECK(n > 0 && p.families.size() == n && p.headers.size() == n &&
                  p.reals.size() == n && p.integers.size() == n &&
                  p.flags.size() == n,
              "Invalid expression source packet");
  plan->packet = p;
  for (size_t k = 0; k < n; ++k) {
    const auto& h = p.headers[k];
    TORCH_CHECK(h.size() == 7 && h[0] >= 2 && h[1] > 0,
                "Invalid expression source metadata");
    Settings s{p.names[k],
               h[5] ? "transpose_sum" : "sum",
               bool(h[3]),
               false,
               false,
               bool(h[4]),
               false,
               h[0],
               1,
               h[1],
               h[2],
               0,
               p.reals[k],
               p.integers[k],
               p.flags[k],
               p.families[k].empty()
                   ? c10::nullopt
                   : c10::optional<std::string>(p.families[k])};
    s.mask = h[6];
    plan->leaves.push_back(std::move(s));
  }
  TORCH_CHECK(!p.nodes.empty(), "Expression must contain a root node");
  for (size_t k = 0; k < p.nodes.size(); ++k) {
    const auto& node = p.nodes[k];
    TORCH_CHECK(node.size() == 3, "Invalid expression node");
    auto op = node[0], a = node[1], b = node[2];
    TORCH_CHECK(op == 0 || op == 1 || op == 2 || expression_binary(op) ||
                    expression_unary(op) || expression_axis_slot(op) ||
                    expression_scalar_bound(op) || op == 34 || op == 35 ||
                    op == 40 || op == 41,
                "Invalid expression node opcode");
    auto edge = [&](int64_t id, bool pair = false) {
      TORCH_CHECK(id >= 0 && id < int64_t(k),
                  "Expression nodes must be topologically ordered");
      TORCH_CHECK((p.nodes[id][0] == 35) == pair,
                  "Pair nodes may only be used by where");
    };
    if (op == 0) {
      TORCH_CHECK(a >= 0 && a < int64_t(n) && b == -1, "Invalid source node");
    } else if (op == 1) {
      TORCH_CHECK(a >= 0 && (b == -1 || b == -2), "Invalid scalar tensor node");
    } else if (op == 2) {
      TORCH_CHECK(a >= 0 && a < int64_t(p.numbers.size()) && b == -1,
                  "Invalid number node");
    } else if (op == 40 || op == 41) {
      TORCH_CHECK(a == -1 && b == -1, "Invalid index node");
    } else if (op == 38 || op == 39) {
      TORCH_CHECK(a == -1 && b >= 0, "Invalid axis value node");
    } else {
      edge(a);
      if (expression_binary(op) || expression_scalar_bound(op) || op == 35)
        edge(b);
      else if (op == 34)
        edge(b, true);
      else if (expression_axis_slot(op)) {
        TORCH_CHECK(b >= 0, "Invalid weight tensor node");
      } else {
        TORCH_CHECK(b == -1, "Invalid unary node");
      }
      if (expression_scalar_bound(op)) {
        TORCH_CHECK(p.nodes[b][0] == 1 || p.nodes[b][0] == 2,
                    "Clamp bounds must be scalar nodes");
      }
    }
  }
  TORCH_CHECK(p.nodes.back()[0] != 35, "Pair nodes cannot be expression roots");
  for (size_t k = 0; k < p.nodes.size(); ++k) {
    if (p.nodes[k][0] != 35)
      continue;
    bool used = false;
    for (const auto& node : p.nodes)
      used |= node[0] == 34 && node[2] == int64_t(k);
    TORCH_CHECK(used, "Pair nodes must belong to where");
  }
  return plan;
}
Tensors expression_local_inputs(at::TensorList v, const Settings& s,
                                const std::vector<int64_t>& index, int64_t left,
                                int64_t count, int64_t right, int64_t other,
                                const Tensor& rows, const Tensor& columns,
                                bool replay) {
  const auto& p = *s.expression;
  Tensors result(v.begin(), v.end());
  auto row_ids =
      rows.defined() ? rows : at::arange(left, left + count, v[XL].options());
  auto column_ids = columns.defined()
                        ? columns
                        : at::arange(right, right + other, v[YL].options());
  auto r = v[XL].index_select(0, s.transposed() ? column_ids : row_ids);
  auto c = v[YL].index_select(0, s.transposed() ? row_ids : column_ids);
  result[XL] = r;
  result[YL] = c;
  for (size_t k = 0; k < p.leaves.size(); ++k) {
    auto local = local_inputs(v.slice(Slots * (k + 1), Slots), p.leaves[k],
                              index, 0, 0, 0, 0, r, c, replay);
    std::copy(local.begin(), local.end(), result.begin() + Slots * (k + 1));
  }
  for (const auto& node : p.packet.nodes) {
    auto op = node[0];
    if (op != 1 && !expression_axis_slot(op))
      continue;
    auto slot = extras_start(p) + node[op == 1 ? 1 : 2];
    TORCH_CHECK(slot < int64_t(v.size()),
                "Expression tensor index out of range");
    auto value = op == 1
                     ? v[slot]
                     : lookup(v[slot], index, 1)
                           .index_select(0, expression_row_slot(op) ? r : c);
    if (replay) {
      value = value.detach();
      if (value.is_floating_point())
        // Compiled backward may replay saved intermediates with requires_grad
        // cleared. Differentiate floating operands just as source replay does.
        value.set_requires_grad(true);
    }
    result[slot] = value;
  }
  return result;
}
Tensor expression_evaluate(at::TensorList local, const Settings& s, bool paired,
                           ExpressionReplay* replay) {
  auto leaves = evaluate_leaves(local, *s.expression, paired, replay);
  at::NoGradGuard guard;
  auto value = broadcast_tile(
      run_graph(local, *s.expression, leaves, paired).back().detach(), local,
      paired);
  return s.transposed() && !paired ? value.transpose(-1, -2) : value;
}
Tensor expression_evaluate_higher_order(at::TensorList local, const Settings& s,
                                        bool paired) {
  Tensors leaves;
  const auto& p = *s.expression;
  for (size_t k = 0; k < p.leaves.size(); ++k) {
    auto input = local.slice(Slots * (k + 1), Slots);
    const auto& leaf = p.leaves[k];
    if (leaf.family && *leaf.family == "_matrix")
      leaves.push_back(leaf_value(input, leaf, paired, local[XL], local[YL]));
    else
      leaves.push_back(evaluate_higher_order(input, leaf, paired, 0, 0,
                                             local[XL], local[YL]));
  }
  auto value =
      broadcast_tile(run_graph(local, p, leaves, paired).back(), local, paired);
  return s.transposed() && !paired ? value.transpose(-1, -2) : value;
}
Tensors expression_evaluate_roots(at::TensorList local, const Settings& s,
                                  at::IntArrayRef roots,
                                  ExpressionReplay* replay) {
  auto leaves = evaluate_leaves(local, *s.expression, false, replay);
  at::NoGradGuard guard;
  auto values = run_graph(local, *s.expression, leaves, false);
  Tensors result;
  for (auto root : roots) {
    TORCH_CHECK(root >= 0 && root < int64_t(values.size()),
                "Invalid expression root");
    TORCH_CHECK(values[root].defined(),
                "Pair nodes cannot be expression roots");
    auto value = broadcast_tile(values[root].detach(), local, false);
    result.push_back(
        s.transposed() && value.dim() >= 2 ? value.transpose(-1, -2) : value);
  }
  return result;
}
void expression_accumulate_roots(at::TensorList local,
                                 at::TensorList cotangents,
                                 at::IntArrayRef roots, Tensors& buffers,
                                 const Settings& s,
                                 const std::vector<int64_t>& index,
                                 const ExpressionReplay* replay) {
  TORCH_CHECK(!cotangents.empty() && cotangents.size() == roots.size(),
              "Expression cotangent/root counts differ");
  const auto& cotangent = cotangents[0];
  const auto& p = *s.expression;
  // The cotangent shape distinguishes paired consumers from Cartesian tiles.
  bool paired = cotangent.dim() == 1;
  at::AutoGradMode enable(true);
  auto raw = replay ? replay->leaves : expression_leaves(local, p, paired);
  TORCH_INTERNAL_ASSERT(raw.size() == p.leaves.size());
  Tensors leaves, active;
  std::vector<int64_t> destinations;
  for (size_t k = 0; k < raw.size(); ++k) {
    auto value = raw[k].detach();
    if (value.is_floating_point()) {
      value.set_requires_grad(true);
      active.push_back(value);
      destinations.push_back(k);
    }
    leaves.push_back(value);
  }
  for (const auto& node : p.packet.nodes) {
    if (node[0] != 1 && !expression_axis_slot(node[0]))
      continue;
    auto slot = extras_start(p) + node[node[0] == 1 ? 1 : 2];
    if (!local[slot].requires_grad())
      continue;
    bool seen = false;
    for (auto dest : destinations)
      seen |= dest == int64_t(raw.size()) + slot;
    if (!seen) {
      active.push_back(local[slot]);
      destinations.push_back(raw.size() + slot);
    }
  }
  auto values = run_graph(local, p, leaves, paired);
  Tensors outputs, grads;
  for (size_t k = 0; k < roots.size(); ++k) {
    TORCH_CHECK(roots[k] >= 0 && roots[k] < int64_t(values.size()),
                "Invalid expression root");
    auto value = values[roots[k]];
    TORCH_CHECK(value.defined(), "Pair nodes cannot be expression roots");
    if (!value.requires_grad())
      continue;
    auto cot = s.transposed() && !paired ? cotangents[k].transpose(-1, -2)
                                         : cotangents[k];
    outputs.push_back(value);
    grads.push_back(cot.sum_to_size(value.sizes()).to(value.scalar_type()));
  }
  if (active.empty() || outputs.empty())
    return;
  auto gradients =
      torch::autograd::grad(outputs, active, grads, false, false, true);
  for (size_t q = 0; q < gradients.size(); ++q) {
    if (!gradients[q].defined())
      continue;
    auto dest = destinations[q];
    if (dest < int64_t(raw.size())) {
      auto offset = Slots * (dest + 1);
      const auto& leaf = p.leaves[dest];
      if (leaf.family && *leaf.family == "_matrix") {
        if (!local[offset + X].requires_grad())
          continue;
        at::NoGradGuard guard;
        auto r = leaf.transposed() ? local[YL] : local[XL];
        auto c = leaf.transposed() ? local[XL] : local[YL];
        auto g = leaf.transposed() && !paired ? gradients[q].transpose(-1, -2)
                                              : gradients[q];
        lookup(buffers[offset + X], index, 2)
            .index_put_(
                {paired ? r : r.unsqueeze(-1), paired ? c : c.unsqueeze(0)},
                g.to(buffers[offset + X].scalar_type()), true);
        continue;
      }
      Tensors source_buffers(buffers.begin() + offset,
                             buffers.begin() + offset + Slots);
      accumulate(local.slice(offset, Slots), raw[dest], gradients[q],
                 source_buffers, p.leaves[dest], index, 0, 0, 0, 0, local[XL],
                 local[YL]);
    } else {
      at::NoGradGuard guard;
      auto slot = dest - int64_t(raw.size());
      auto target = buffers[slot];
      int64_t op = 1;
      for (const auto& node : p.packet.nodes)
        if (expression_axis_slot(node[0]) && extras_start(p) + node[2] == slot)
          op = node[0];
      if (op == 1)
        target.add_(gradients[q].to(target.scalar_type()));
      else
        lookup(target, index, 1)
            .index_add_(0, expression_row_slot(op) ? local[XL] : local[YL],
                        gradients[q].to(target.scalar_type()));
    }
  }
}
void expression_accumulate(at::TensorList local, const Tensor& cotangent,
                           Tensors& buffers, const Settings& s,
                           const std::vector<int64_t>& index,
                           const ExpressionReplay* replay) {
  expression_accumulate_roots(local, {cotangent},
                              {int64_t(s.expression->packet.nodes.size()) - 1},
                              buffers, s, index, replay);
}
void expression_select(PreparedComparison& p, c10::string_view operation,
                       const PairwiseOperationOptions& o) {
  auto& v = p.values;
  auto n = p.queries.min(p.references);
  Tensor r, c;
  if (operation == "diagonal") {
    r = at::arange(n, v[XL].options());
    c = r;
  } else {
    r = *o.rows;
    c = *o.columns;
    utils::check_values(
        (r.ge(0) & r.lt(p.queries)).all() &
            (c.ge(0) & c.lt(p.references)).all(),
        "Block indices must be within expression sample bounds");
  }
  auto a = p.settings.transposed() ? c : r, b = p.settings.transposed() ? r : c;
  v[XL] = v[XL].index_select(0, a);
  v[YL] = v[YL].index_select(0, b);
  auto xshape = v[X].sym_sizes().vec(), yshape = v[Y].sym_sizes().vec();
  xshape[xshape.size() - 2] = a.sym_numel();
  yshape[yshape.size() - 2] = b.sym_numel();
  v[X] = at::empty_symint(xshape, v[X].options());
  v[Y] = at::empty_symint(yshape, v[Y].options());
}
Outputs expression_consume(at::TensorList v, TORCHPAIRWISE_EXPRESSION_DECL) {
  auto s = root_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed);
  s.mask |= operation == "matmul" ? (int64_t(1) << Payload) : 0;
  if (!form)
    return streaming_reduce_kernel(v, s);
  auto result = expression_form_kernel(v.slice(0, v.size() - 2),
                                       v[v.size() - 2], v.back(), s);
  auto empty = at::empty({0}, result.options());
  return {result, empty, empty.clone(), empty.clone()};
}
Tensors expression_consume_backward(const Tensor& gradient, at::TensorList v,
                                    const Tensor& result, const Tensor& indices,
                                    const Tensor& zeros, const Tensor& nonzero,
                                    TORCHPAIRWISE_EXPRESSION_DECL) {
  auto s = root_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed);
  s.mask |= operation == "matmul" ? (int64_t(1) << Payload) : 0;
  if (!form)
    return streaming_reduce_backward_kernel(gradient, v, result, indices, zeros,
                                            nonzero, s);
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "Higher derivatives of expression forms are unsupported");
  return expression_form_backward_kernel(gradient, v.slice(0, v.size() - 2),
                                         v[v.size() - 2], v.back(), s);
}
Outputs expression_consume_topk(at::TensorList v, TORCHPAIRWISE_EXPRESSION_DECL,
                                int64_t k, bool largest) {
  TORCH_CHECK(!form && operation == "neighbors", "Invalid top-k consumer");
  auto s = root_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed);
  auto candidates = v[transposed ? X : Y].size(-2) - int64_t(exclusion);
  TORCH_CHECK(k >= 0 && k <= candidates, "k exceeds eligible axis length");
  s.k = k;
  s.largest = largest;
  s.nan_first = largest;
  return streaming_reduce_kernel(v, s);
}
Tensors expression_consume_topk_backward(
    const Tensor& gradient, at::TensorList v, const Tensor& result,
    const Tensor& indices, const Tensor& zeros, const Tensor& nonzero,
    TORCHPAIRWISE_EXPRESSION_DECL, int64_t k, bool largest) {
  auto s = root_settings(
      {names, families, headers, nodes, reals, integers, flags, numbers.vec()},
      operation, dtype, size, exclusion, transposed);
  s.k = k;
  s.largest = largest;
  s.nan_first = largest;
  return streaming_reduce_backward_kernel(gradient, v, result, indices, zeros,
                                          nonzero, s);
}
Outputs expression_topk(at::TensorList v, const Settings& s, int64_t k,
                        bool largest) {
  const auto& p = s.expression->packet;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_expression_consume", "topk")
          .typed<ExpressionTopKSignature>();
  return op.call(v, p.names, p.families, p.headers, p.reals, p.integers,
                 p.flags, p.nodes, p.numbers, "neighbors", s.dtype, s.size,
                 s.exclusion, s.transposed(), false, k, largest);
}
namespace {
Outputs consume(at::TensorList v, const Settings& s, bool form) {
  const auto& p = s.expression->packet;
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::_expression_consume", "")
          .typed<ExpressionSignature>();
  // kind max/min distinction is encoded in a root operation name.
  auto operation = std::string(s.kind());
  if (s.kind() == "neighbors" && s.largest)
    operation = "maximum";
  return op.call(v, p.names, p.families, p.headers, p.reals, p.integers,
                 p.flags, p.nodes, p.numbers, operation, s.dtype, s.size,
                 s.exclusion, s.transposed(), form);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_expression_consume.topk(Tensor[] operands, str[] names, str[] "
      "families, "
      "int[][] headers, float?[][] reals, int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, bool transposed, bool form, int k, bool largest) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "_expression_consume_backward.topk(Tensor grad, Tensor[] operands, "
      "Tensor "
      "result, Tensor indices, Tensor zeros, Tensor nonzero, str[] names, "
      "str[] families, int[][] headers, float?[][] reals, int?[][] integers, "
      "bool?[][] flags, int[][] nodes, Scalar[] numbers, str operation, int "
      "dtype, int size, bool exclusion, bool transposed, bool form, int k, "
      "bool largest) -> Tensor[]");

  m.def(
      "_expression_consume(Tensor[] operands, str[] names, str[] families, "
      "int[][] headers, float?[][] reals, int?[][] integers, bool?[][] flags, "
      "int[][] nodes, Scalar[] numbers, str operation, int dtype, int size, "
      "bool exclusion, bool transposed, bool form) -> (Tensor, Tensor, Tensor, "
      "Tensor)");
  m.def(
      "_expression_consume_backward(Tensor grad, Tensor[] operands, Tensor "
      "result, Tensor indices, Tensor zeros, Tensor nonzero, str[] names, "
      "str[] families, int[][] headers, float?[][] reals, int?[][] integers, "
      "bool?[][] flags, int[][] nodes, Scalar[] numbers, str operation, int "
      "dtype, int size, bool exclusion, bool transposed, bool form) -> "
      "Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_expression_consume.topk", TORCH_FN(expression_consume_topk));
  m.impl("_expression_consume_backward.topk",
         TORCH_FN(expression_consume_topk_backward));
  m.impl("_expression_consume", TORCH_FN(expression_consume));
  m.impl("_expression_consume_backward", TORCH_FN(expression_consume_backward));
}
}  // namespace
Outputs expression_reduce(at::TensorList v, const Settings& s) {
  return consume(v, s, false);
}
Tensor expression_form(at::TensorList v, const Tensor& l, const Tensor& r,
                       const Settings& s) {
  Tensors operands(v.begin(), v.end());
  operands.insert(operands.end(), {l, r});
  return std::get<0>(consume(operands, s, true));
}
}  // namespace torchpairwise::ops::detail::streaming
