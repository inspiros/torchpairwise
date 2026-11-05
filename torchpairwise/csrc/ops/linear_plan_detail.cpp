#include "linear_plan_detail.h"

#include <ATen/ExpandUtils.h>

#include "expression_graph_detail.h"
#include "operator_sources_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops::detail::linear {
template <class T>
void pack(c10::Dict<std::string, T>& output,
          const c10::Dict<std::string, T>& input, int64_t index) {
  for (const auto& item : input)
    output.insert(std::to_string(index) + ":" + item.key(), item.value());
}
Packet pack_graph(const LinearOperatorGraph& graph) {
  Packet p;
  p.linear_factors = graph.factors;
  for (const auto& node : graph.nodes)
    p.linear_program.insert(p.linear_program.end(), node.begin(), node.end());
  for (const auto& expression : graph.sources) {
    auto key = std::to_string(p.source_counts.size());
    p.source_counts.push_back(expression.sources.size());
    p.factor_counts.push_back(expression.factors.size());
    p.factors.insert(p.factors.end(), expression.factors.begin(),
                     expression.factors.end());
    std::vector<int64_t> program, kinds, integers;
    std::vector<double> reals;
    for (const auto& node : expression.nodes)
      program.insert(program.end(), node.begin(), node.end());
    p.programs.insert(key, c10::List<int64_t>(std::move(program)));
    for (const auto& number : expression.numbers) {
      kinds.push_back(number.isBoolean()         ? 0
                      : number.isIntegral(false) ? 1
                                                 : 2);
      integers.push_back(number.isIntegral(true) ? number.toLong() : 0);
      reals.push_back(number.isFloatingPoint() ? number.toDouble() : 0.);
    }
    if (!kinds.empty()) {
      p.kinds.insert(key, c10::List<int64_t>(std::move(kinds)));
      p.integer_numbers.insert(key, c10::List<int64_t>(std::move(integers)));
      p.real_numbers.insert(key, c10::List<double>(std::move(reals)));
    }
    for (const auto& source : expression.sources) {
      auto index = int64_t(p.metrics.size());
      p.counts.push_back(source.inputs.size());
      p.inputs.insert(p.inputs.end(), source.inputs.begin(),
                      source.inputs.end());
      p.metrics.push_back(source.metric);
      p.families.push_back(source.family);
      p.transposed.push_back(source.transposed);
      pack(p.tensors, source.tensors, index);
      pack(p.reals, source.reals, index);
      pack(p.integers, source.integers, index);
      pack(p.flags, source.flags, index);
      pack(p.strings, source.strings, index);
      if (!source.real_list.empty())
        p.real_lists.insert(std::to_string(index),
                            c10::List<double>(source.real_list));
      if (!source.integer_list.empty())
        p.integer_lists.insert(std::to_string(index),
                               c10::List<int64_t>(source.integer_list));
    }
  }
  return p;
}
int64_t source_index(const std::string& key, int64_t size) {
  TORCH_CHECK(!key.empty() &&
                  key.find_first_not_of("0123456789") == std::string::npos &&
                  key.size() < 19,
              "Invalid linear source index");
  auto index = std::stoll(key);
  TORCH_CHECK(index < size, "Linear source index out of range");
  return index;
}
template <class T, class Member>
void unpack(const c10::Dict<std::string, T>& options,
            std::vector<ExpressionSource>& sources, Member member) {
  for (const auto& item : options) {
    auto key = item.key();
    auto colon = key.find(':');
    TORCH_CHECK(colon != std::string::npos, "Invalid linear source option key");
    auto index = source_index(key.substr(0, colon), sources.size());
    (sources[index].*member).insert(key.substr(colon + 1), item.value());
  }
}
LinearOperatorGraph unpack_graph(
    at::TensorList inputs, at::IntArrayRef counts,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef source_counts, const IntLists& programs,
    at::TensorList factors, at::IntArrayRef factor_counts,
    const IntLists& kinds, const IntLists& integer_numbers,
    const RealLists& real_numbers, at::IntArrayRef linear_program,
    at::TensorList linear_factors) {
  auto count = int64_t(counts.size()), leaves = int64_t(source_counts.size());
  TORCH_CHECK(int64_t(metrics.size()) == count &&
                  int64_t(families.size()) == count &&
                  int64_t(transposed.size()) == count,
              "Linear source packet sizes differ");
  TORCH_CHECK(int64_t(programs.size()) == leaves &&
                  int64_t(factor_counts.size()) == leaves &&
                  int64_t(kinds.size()) <= leaves &&
                  integer_numbers.size() == kinds.size() &&
                  real_numbers.size() == kinds.size(),
              "Linear expression packet sizes differ");
  std::vector<ExpressionSource> sources;
  int64_t offset = 0;
  for (int64_t i = 0; i < count; ++i) {
    TORCH_CHECK(counts[i] > 0 && counts[i] <= int64_t(inputs.size()) - offset,
                "Invalid linear source tensor count");
    ExpressionSource s;
    s.inputs = inputs.slice(offset, counts[i]).vec();
    offset += counts[i];
    s.metric = metrics[i];
    s.family = families[i];
    s.transposed = transposed[i];
    sources.push_back(std::move(s));
  }
  TORCH_CHECK(offset == int64_t(inputs.size()), "Unused linear source tensors");
  unpack(tensors, sources, &ExpressionSource::tensors);
  unpack(reals, sources, &ExpressionSource::reals);
  unpack(integers, sources, &ExpressionSource::integers);
  unpack(flags, sources, &ExpressionSource::flags);
  unpack(strings, sources, &ExpressionSource::strings);
  for (const auto& item : real_lists)
    sources[source_index(item.key(), count)].real_list = item.value().vec();
  for (const auto& item : integer_lists)
    sources[source_index(item.key(), count)].integer_list = item.value().vec();
  for (const auto& item : kinds)
    source_index(item.key(), leaves);
  for (const auto& item : integer_numbers)
    source_index(item.key(), leaves);
  for (const auto& item : real_numbers)
    source_index(item.key(), leaves);
  LinearOperatorGraph graph;
  offset = 0;
  int64_t factor_offset = 0;
  for (int64_t i = 0; i < leaves; ++i) {
    TORCH_CHECK(source_counts[i] > 0 && source_counts[i] <= count - offset &&
                    factor_counts[i] >= 0 &&
                    factor_counts[i] <= int64_t(factors.size()) - factor_offset,
                "Invalid linear expression source/factor count");
    ExpressionGraph e;
    e.sources = std::vector<ExpressionSource>(
        sources.begin() + offset, sources.begin() + offset + source_counts[i]);
    e.factors = factors.slice(factor_offset, factor_counts[i]).vec();
    offset += source_counts[i];
    factor_offset += factor_counts[i];
    auto key = std::to_string(i);
    TORCH_CHECK(programs.contains(key), "Missing linear expression program");
    TORCH_CHECK(kinds.contains(key) == integer_numbers.contains(key) &&
                    kinds.contains(key) == real_numbers.contains(key),
                "Missing linear expression number packet");
    auto program = programs.at(key).vec();
    auto number_kinds =
        kinds.contains(key) ? kinds.at(key).vec() : std::vector<int64_t>{};
    auto number_integers = integer_numbers.contains(key)
                               ? integer_numbers.at(key).vec()
                               : std::vector<int64_t>{};
    auto number_reals = real_numbers.contains(key) ? real_numbers.at(key).vec()
                                                   : std::vector<double>{};
    TORCH_CHECK(!program.empty() && program.size() % 3 == 0,
                "Invalid linear scalar expression program");
    for (size_t j = 0; j < program.size(); j += 3)
      e.nodes.push_back({program[j], program[j + 1], program[j + 2]});
    TORCH_CHECK(number_kinds.size() == number_integers.size() &&
                    number_kinds.size() == number_reals.size(),
                "Linear scalar number sizes differ");
    for (size_t j = 0; j < number_kinds.size(); ++j) {
      auto kind = number_kinds[j];
      TORCH_CHECK(kind >= 0 && kind <= 2, "Invalid linear scalar kind");
      e.numbers.push_back(kind == 0   ? at::Scalar(bool(number_integers[j]))
                          : kind == 1 ? at::Scalar(number_integers[j])
                                      : at::Scalar(number_reals[j]));
    }
    graph.sources.push_back(std::move(e));
  }
  TORCH_CHECK(offset == count && factor_offset == int64_t(factors.size()),
              "Unused linear expression sources/factors");
  TORCH_CHECK(!linear_program.empty() && linear_program.size() % 3 == 0,
              "Invalid linear linear program");
  for (size_t i = 0; i < linear_program.size(); i += 3)
    graph.nodes.push_back(
        {linear_program[i], linear_program[i + 1], linear_program[i + 2]});
  graph.factors = linear_factors.vec();
  return graph;
}

Plan::Plan(LinearOperatorGraph g, const Tensor& rhs) : graph(std::move(g)) {
  for (const auto& e : graph.sources) {
    auto p = prepare_expression_graph(e);
    TORCH_CHECK(p.settings.dtype == int64_t(rhs.scalar_type()) &&
                    p.values[X].device() == rhs.device(),
                "Linear sources must match RHS dtype/device");
    sources.push_back(std::move(p));
  }
  std::vector<bool> used_sources(sources.size()),
      used_factors(graph.factors.size());
  for (const auto& node : graph.nodes) {
    auto [kind, a, b] = std::tuple{node[0], node[1], node[2]};
    auto size = int64_t(shapes.size());
    std::vector<c10::SymInt> shape;
    TORCH_CHECK(kind >= 0 && kind <= 13, "Unknown linear linear opcode");
    if (kind == 12) {
      TORCH_CHECK(a >= 0 && b == a + 1 && b + 1 < int64_t(graph.factors.size()),
                  "Invalid low-rank inverse operands");
      const auto& basis = graph.factors[a];
      const auto& spectrum = graph.factors[b];
      const auto& shift = graph.factors[b + 1];
      TORCH_CHECK(basis.layout() == at::kStrided && basis.dim() >= 2 &&
                      basis.scalar_type() == at::kDouble &&
                      spectrum.layout() == at::kStrided &&
                      spectrum.scalar_type() == at::kDouble &&
                      shift.layout() == at::kStrided && shift.dim() == 0 &&
                      shift.scalar_type() == rhs.scalar_type() &&
                      basis.device() == rhs.device() &&
                      spectrum.device() == rhs.device() &&
                      shift.device() == rhs.device(),
                  "Low-rank inverse state dtype/device mismatch");
      auto expected = basis.sym_sizes().vec();
      expected.erase(expected.end() - 2);
      TORCH_CHECK(spectrum.sym_sizes().equals(expected),
                  "Low-rank inverse spectrum shape mismatch");
      used_factors[a] = used_factors[b] = used_factors[b + 1] = true;
      shape = basis.sym_sizes().vec();
      shape.back() = shape[shape.size() - 2];
    } else if (kind == 11) {
      TORCH_CHECK(a >= 0 && a < int64_t(graph.factors.size()) && b == -1,
                  "Invalid linear diagonal operand");
      const auto& d = graph.factors[a];
      TORCH_CHECK(
          d.layout() == at::kStrided && d.dim() >= 1 &&
              d.scalar_type() == rhs.scalar_type() &&
              d.device() == rhs.device(),
          "Linear diagonal requires [...,N] entries matching RHS dtype/device");
      used_factors[a] = true;
      shape = d.sym_sizes().vec();
      shape.push_back(d.sym_size(-1));
    } else if (kind == 8) {
      TORCH_CHECK(a >= 0 && b >= 0 && a < int64_t(graph.factors.size()) &&
                      b < int64_t(graph.factors.size()),
                  "Invalid linear group partition operands");
      used_factors[a] = used_factors[b] = true;
      const auto& ids = graph.factors[a];
      const auto& counts = graph.factors[b];
      TORCH_CHECK(ids.layout() == at::kStrided && ids.dim() >= 1 &&
                      ids.scalar_type() == at::kLong &&
                      counts.layout() == at::kStrided &&
                      counts.scalar_type() == at::kLong &&
                      ids.sym_sizes().equals(counts.sym_sizes()) &&
                      ids.device() == rhs.device() &&
                      counts.device() == rhs.device(),
                  "Linear group partitions require matching int64 [...,N] "
                  "tensors on the RHS device");
      auto n = ids.sym_size(-1);
      utils::ValueChecks checks;
      checks.require((ids.ge(0) & ids.lt(n)).all(),
                     "Invalid linear group partition indices");
      checks.require((counts.ge(0) & counts.le(n)).all(),
                     "Invalid linear group counts");
      checks.finish();
      shape = ids.sym_sizes().vec();
      shape.push_back(n);
    } else if (kind == 0) {
      TORCH_CHECK(a >= 0 && a < int64_t(sources.size()) && b == -1,
                  "Invalid linear source operand");
      used_sources[a] = true;
      auto& p = sources[a];
      shape = p.batch;
      shape.insert(shape.end(), {p.queries, p.references});
    } else {
      TORCH_CHECK(a >= 0 && a < size, "Invalid linear linear edge");
      shape = shapes[a];
      if (kind <= 3 || kind == 9 || kind == 10) {
        TORCH_CHECK(b >= 0 && b < size, "Invalid linear linear edge");
        const auto& other = shapes[b];
        TORCH_CHECK(
            kind == 9    ? shape[shape.size() - 2] == other[other.size() - 2]
            : kind == 10 ? shape.back() == other.back()
            : kind == 3  ? shape.back() == other[other.size() - 2]
                         : (shape.back() == other.back() &&
                           shape[shape.size() - 2] == other[other.size() - 2]),
            "Linear linear dimensions do not match");
        auto rows = shape[shape.size() - 2],
             columns = kind == 3 ? other.back() : shape.back();
        if (kind == 9)
          columns = columns + other.back();
        if (kind == 10)
          rows = rows + other[other.size() - 2];
        shape = at::infer_size_symint(
            c10::SymIntArrayRef(shape).slice(0, shape.size() - 2),
            c10::SymIntArrayRef(other).slice(0, other.size() - 2));
        shape.insert(shape.end(), {rows, columns});
      } else if (kind == 13) {
        TORCH_CHECK(b >= 0 && b < int64_t(graph.factors.size()),
                    "Invalid reference centering factor");
        used_factors[b] = true;
        const auto& mean = graph.factors[b];
        TORCH_CHECK(mean.layout() == at::kStrided && mean.dim() >= 1 &&
                        mean.scalar_type() == rhs.scalar_type() &&
                        mean.device() == rhs.device() &&
                        mean.sym_size(-1) == shape.back() && shape.back() > 0,
                    "Reference mean must match nonempty reference axis and RHS "
                    "dtype/device");
        auto batch = at::infer_size_symint(
            c10::SymIntArrayRef(shape).slice(0, shape.size() - 2),
            mean.sym_sizes().slice(0, mean.dim() - 1));
        batch.insert(batch.end(), {shape[shape.size() - 2], shape.back()});
        shape = std::move(batch);
      } else if (kind == 4 || kind == 5) {
        TORCH_CHECK(b >= 0 && b < int64_t(graph.factors.size()),
                    "Invalid linear factor operand");
        used_factors[b] = true;
        auto& factor = graph.factors[b];
        TORCH_CHECK(factor.dim() == 0 && factor.layout() == at::kStrided &&
                        factor.scalar_type() == rhs.scalar_type() &&
                        factor.device() == rhs.device(),
                    "Linear linear factors must be scalar tensors matching "
                    "dtype/device");
        if (kind == 5)
          TORCH_CHECK(shape.back() == shape[shape.size() - 2],
                      "Linear diagonal shifts require square maps");
      } else {
        TORCH_CHECK(b == -1, "Unused linear linear operand");
        if (kind == 7)
          std::iter_swap(shape.end() - 1, shape.end() - 2);
      }
    }
    shapes.push_back(std::move(shape));
  }
  for (bool used : used_sources)
    TORCH_CHECK(used, "Unused linear source");
  for (bool used : used_factors)
    TORCH_CHECK(used, "Unused linear factor");
}
Tensor Plan::apply(int64_t id, const Tensor& values, bool transpose) const {
  const auto& node = graph.nodes[id];
  auto kind = node[0], a = node[1], b = node[2];
  if (kind == 12)
    return linear_lowrank_apply(values, graph.factors[a], graph.factors[b],
                                graph.factors[b + 1]);
  if (kind == 11)
    return linear_diagonal_apply(values, graph.factors[a]);
  if (kind == 8)
    return group_projection_apply(values, graph.factors[a], graph.factors[b]);
  if (kind == 0) {
    // Preparation/statistics are reused for every iteration. Only RHS blocks
    // change; the existing tiled consumer never stores the dense source.
    auto p = sources[a];
    if (p.settings.family == c10::optional<std::string>("_matrix")) {
      const auto& matrix = p.values[X];
      return (p.settings.transposed() != transpose
                  ? matrix.transpose(-1, -2)
                  : matrix)
          .matmul(values);
    }
    p.settings.operation = p.settings.transposed() != transpose
                               ? "transpose_matmul"
                               : "matmul";
    p.settings.mask |= int64_t(1) << Payload;
    p.values[Payload] = values;
    return std::get<0>(_streaming_reduce(p.values, p.settings));
  }
  if (kind == 7)
    return apply(a, values, !transpose);
  if (kind == 3)
    return apply(transpose ? b : a, apply(transpose ? a : b, values, transpose),
                 transpose);
  if (kind == 9 || kind == 10) {
    bool split = (kind == 9) != transpose;
    if (split) {
      auto width = shapes[a][shapes[a].size() - (transpose ? 2 : 1)];
      return linear_transform(
          apply(a, values.narrow_symint(-2, 0, width), transpose), "add",
          apply(b, values.narrow_symint(-2, width, values.sym_size(-2) - width),
                transpose));
    }
    return linear_transform(apply(a, values, transpose), "vstack",
                            apply(b, values, transpose));
  }
  if (kind == 6)
    return linear_transform(
        apply(a, linear_transform(values, "center"), transpose), "center");
  if (kind == 13) {
    auto rhs = transpose ? values : linear_transform(values, "center");
    return linear_reference_center(apply(a, rhs, transpose), rhs,
                                   graph.factors[b], transpose);
  }
  auto value = apply(a, values, transpose);
  if (kind == 1 || kind == 2)
    return linear_transform(value, kind == 1 ? "add" : "sub",
                            apply(b, values, transpose));
  return linear_transform(
      value, kind == 4 ? "scale" : "add_diagonal",
      kind == 5 ? c10::optional<Tensor>(values) : c10::nullopt,
      graph.factors[b]);
}
Tensor Plan::operator()(const Tensor& values) const {
  return apply(graph.nodes.size() - 1, values);
}
}  // namespace torchpairwise::ops::detail::linear
