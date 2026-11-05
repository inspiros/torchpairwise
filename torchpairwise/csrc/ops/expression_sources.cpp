#include <ATen/ExpandUtils.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include <algorithm>
#include <array>

#include "custom_comparisons_detail.h"
#include "expression_graph_detail.h"
#include "expression_operations.h"
#include "expression_operations_detail.h"
#include "expression_statistics_detail.h"
#include "image_map_operations_detail.h"
#include "operator_sources_detail.h"
#include "residual_operations_detail.h"
#include "softmax_operations_detail.h"
#include "utils/operator_axes.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
using TensorDict = c10::Dict<std::string, Tensor>;
using RealDict = c10::Dict<std::string, double>;
using IntDict = c10::Dict<std::string, int64_t>;
using BoolDict = c10::Dict<std::string, bool>;
using StringDict = c10::Dict<std::string, std::string>;
struct Reader {
  TensorDict t;
  RealDict r;
  IntDict i;
  BoolDict b;
  StringDict s;
  template <class T>
  c10::optional<T> pop(c10::Dict<std::string, T>& d, const char* key) {
    if (!d.contains(key))
      return c10::nullopt;
    auto v = d.at(key);
    d.erase(key);
    return v;
  }
  void finish() const {
    TORCH_CHECK(t.empty() && r.empty() && i.empty() && b.empty() && s.empty(),
                "Unused or incorrectly typed expression source option");
  }
};
PreparedComparison source(const ExpressionSource& a) {
  Reader p{a.tensors.copy(), a.reals.copy(), a.integers.copy(), a.flags.copy(),
           a.strings.copy()};
  PairwiseOperationOptions o;
  o.transposed = a.transposed;
  o.chunk_size = p.pop(p.i, "chunk_size");
  auto count = a.family == "gaussian" ? 2 : a.family == "mixture" ? 3 : 1;
  TORCH_CHECK(int64_t(a.inputs.size()) == count ||
                  int64_t(a.inputs.size()) == count * 2,
              "Expression collection tensor count mismatch");
  auto x = at::TensorList(a.inputs).slice(0, count);
  c10::optional<Tensor> y = int64_t(a.inputs.size()) == count
                                ? c10::nullopt
                                : c10::optional<Tensor>(a.inputs[count]);
  PreparedComparison result;
  if (a.family == "custom") {
    auto rank = p.pop(p.i, "sample_ndim").value_or(1);
    auto dtype = p.pop(p.i, "output_dtype");
    auto backward = p.pop(p.s, "backward").value_or("");
    auto parameter_count = p.pop(p.i, "parameter_count").value_or(0);
    TORCH_CHECK(parameter_count >= 0, "Invalid custom parameter count");
    std::vector<Tensor> parameters;
    for (int64_t k = 0; k < parameter_count; ++k) {
      auto key = "parameter_" + std::to_string(k);
      auto value = p.pop(p.t, key.c_str());
      TORCH_CHECK(value, "Missing custom parameter: ", key);
      parameters.push_back(*value);
    }
    result = detail::custom::prepare(x[0], y, parameters, a.metric, backward,
                                     rank, dtype, o);
  } else if (a.family == "vector" || a.family == "point_cloud") {
    auto& q = o.comparison;
    q.w = p.pop(p.t, "w");
    q.V = p.pop(p.t, "V");
    q.VI = p.pop(p.t, "VI");
    q.bin_positions = p.pop(p.t, "bin_positions");
    q.projections = p.pop(p.t, "projections");
    q.x1_lengths = p.pop(p.t, "x1_lengths");
    q.x2_lengths = p.pop(p.t, "x2_lengths");
    q.p = p.pop(p.r, "p");
    q.base = p.pop(p.r, "base");
    q.alpha = p.pop(p.r, "alpha");
    q.length_scale = p.pop(p.r, "length_scale");
    q.nu = p.pop(p.r, "nu");
    q.gamma = p.pop(p.r, "gamma");
    q.coef0 = p.pop(p.r, "coef0");
    q.regularization = p.pop(p.r, "regularization");
    q.degree = p.pop(p.i, "degree");
    q.max_iter = p.pop(p.i, "max_iter");
    q.chunk_size = p.pop(p.i, "metric_chunk_size");
    q.feature_dim = p.pop(p.i, "feature_dim");
    q.squared = p.pop(p.b, "squared");
    q.shuffle = p.pop(p.b, "shuffle");
    q.batched = p.pop(p.b, "batched");
    q.kernel = p.pop(p.s, "kernel");
    bool kernel = p.pop(p.b, "is_kernel").value_or(false);
    result =
        prepare_pairwise_operator(x[0], y, a.metric, kernel, "diagonal", o);
  } else if (a.family == "residual") {
    auto transform = p.pop(p.s, "transform").value_or("identity");
    auto correction = p.pop(p.r, "correction").value_or(1.);
    auto feature_size = p.pop(p.i, "feature_chunk_size").value_or(128);
    result = prepare_residual_operator(x[0], y, a.metric, transform, correction,
                                       feature_size, o);
  } else if (a.family == "image_map" || a.family == "image_event") {
    ImageMapOptions q;
    q.data_range = p.pop(p.r, "data_range");
    q.window_size = p.pop(p.i, "window_size").value_or(q.window_size);
    q.sigma = p.pop(p.r, "sigma").value_or(q.sigma);
    q.k1 = p.pop(p.r, "k1").value_or(q.k1);
    q.k2 = p.pop(p.r, "k2").value_or(q.k2);
    auto comparison = p.pop(p.s, "comparison").value_or("squared_error");
    auto correction = p.pop(p.r, "correction");
    if (a.family == "image_event") {
      TORCH_CHECK(!correction, "Event corrections use the program list");
      result = prepare_image_event_operator(x[0], y, comparison, a.integer_list,
                                            a.real_list, o, q);
      std::vector<int64_t> settings;
      for (size_t k = 1; k < 5; ++k)
        settings.push_back(result.settings.integers[k].value());
      auto metadata = detail::media::event_metadata(
          result.settings.integers[0].value(), settings,
          result.settings.integers[5].value(), a.integer_list, a.real_list);
      TORCH_CHECK(metadata.shape.empty(),
                  "Scalar image event sources must contract every event axis");
    } else
      result = prepare_image_map_operator(x[0], y, comparison, a.metric,
                                          correction.value_or(1.), o, q);
  } else if (a.family == "matrix") {
    TORCH_CHECK(!y, "Matrix expression takes one input");
    result = prepare_matrix_operator(x[0], "diagonal", o);
  } else if (a.family == "covariance" || a.family == "gaussian" ||
             a.family == "mixture") {
    StructuredOperationOptions q;
    q.matrix = o;
    q.x1_lengths = p.pop(p.t, "x1_lengths");
    q.x2_lengths = p.pop(p.t, "x2_lengths");
    q.squared = p.pop(p.b, "squared");
    q.regularization = p.pop(p.r, "regularization");
    q.max_iter = p.pop(p.i, "max_iter");
    q.component_chunk_size = p.pop(p.i, "component_chunk_size");
    auto other = y ? c10::optional<std::vector<Tensor>>(std::vector<Tensor>(
                         a.inputs.begin() + count, a.inputs.end()))
                   : c10::nullopt;
    result = prepare_structured_operator(x, other, a.metric, "diagonal", q);
  } else if (a.family == "image") {
    ImageComparisonOptions q;
    q.data_range = p.pop(p.r, "data_range").value_or(q.data_range);
    q.window_size = p.pop(p.i, "window_size").value_or(q.window_size);
    q.sigma = p.pop(p.r, "sigma").value_or(q.sigma);
    q.k1 = p.pop(p.r, "k1").value_or(q.k1);
    q.k2 = p.pop(p.r, "k2").value_or(q.k2);
    if (p.pop(p.b, "weights_set").value_or(!a.real_list.empty()))
      q.weights = a.real_list;
    result = prepare_image_operator(x[0], y, a.metric, "diagonal", o, q);
  } else if (a.family == "audio") {
    AudioComparisonOptions q;
    q.zero_mean = p.pop(p.b, "zero_mean");
    q.energy_floor = p.pop(p.r, "energy_floor").value_or(q.energy_floor);
    q.n_fft = p.pop(p.i, "n_fft").value_or(q.n_fft);
    q.hop_length = p.pop(p.i, "hop_length");
    q.win_length = p.pop(p.i, "win_length");
    q.window = p.pop(p.s, "window").value_or(q.window);
    q.center = p.pop(p.b, "center").value_or(q.center);
    q.pad_mode = p.pop(p.s, "pad_mode").value_or(q.pad_mode);
    q.normalized = p.pop(p.b, "normalized").value_or(q.normalized);
    q.log_floor = p.pop(p.r, "log_floor").value_or(q.log_floor);
    q.norm_floor = p.pop(p.r, "norm_floor").value_or(q.norm_floor);
    if (p.pop(p.b, "resolutions_set").value_or(!a.integer_list.empty()))
      q.resolutions = a.integer_list;
    result = prepare_audio_operator(x[0], y, a.metric, "diagonal", o, q);
  } else if (a.family == "sequence") {
    SequenceOperatorOptions q;
    q.squared = p.pop(p.b, "squared").value_or(false);
    q.gamma = p.pop(p.r, "gamma");
    q.window = p.pop(p.i, "window");
    q.x1_lengths = p.pop(p.t, "x1_lengths");
    q.x2_lengths = p.pop(p.t, "x2_lengths");
    result = prepare_sequence_operator(x[0], y, a.metric, "diagonal", o, q);
  } else if (a.family == "tokens") {
    TokenOperatorOptions q;
    q.x1_lengths = p.pop(p.t, "x1_lengths");
    q.x2_lengths = p.pop(p.t, "x2_lengths");
    result = prepare_token_operator(x[0], y, a.metric, "diagonal", o, q);
  } else if (a.family == "mmd") {
    MMDOperatorOptions q;
    q.kernel = p.pop(p.s, "kernel").value_or(q.kernel);
    q.squared = p.pop(p.b, "squared").value_or(false);
    q.length_scale = p.pop(p.r, "length_scale").value_or(q.length_scale);
    q.gamma = p.pop(p.r, "gamma");
    q.degree = p.pop(p.i, "degree");
    q.coef0 = p.pop(p.r, "coef0");
    q.alpha = p.pop(p.r, "alpha");
    q.nu = p.pop(p.r, "nu");
    q.x1_lengths = p.pop(p.t, "x1_lengths");
    q.x2_lengths = p.pop(p.t, "x2_lengths");
    q.batched = p.pop(p.b, "batched");
    q.feature_dim = p.pop(p.i, "feature_dim").value_or(-1);
    q.estimator = p.pop(p.s, "estimator");
    q.metric_chunk_size = p.pop(p.i, "metric_chunk_size");
    result = prepare_mmd_operator(x[0], y, a.metric, "diagonal", o, q);
  } else
    TORCH_CHECK(false, "Unknown expression source family: ", a.family);
  p.finish();
  TORCH_CHECK(
      a.family == "image" || a.family == "image_event" || a.real_list.empty(),
      "Unused real list option");
  TORCH_CHECK(a.family == "audio" || a.family == "image_event" ||
                  a.integer_list.empty(),
              "Unused integer list option");
  return result;
}
}  // namespace
namespace detail::streaming {
PreparedComparison prepare_expression(
    const std::vector<ExpressionSource>& sources,
    const std::vector<std::vector<int64_t>>& nodes, at::TensorList tensors,
    at::ArrayRef<at::Scalar> numbers, c10::string_view operation,
    const PairwiseOperationOptions& original_options, int64_t output_root = -1,
    const c10::optional<std::vector<int64_t>>& axes = c10::nullopt,
    bool topk = false, int64_t other_root = -1) {
  auto o = original_options;
  bool softmax = operation == "softmax_matmul";
  auto validation_options = o;
  if (softmax)
    validation_options.exclude_self = false;
  validate_pairwise_options(softmax ? "matmul" : operation, validation_options);
  TORCH_CHECK(!sources.empty(), "A scalar expression requires a matrix source");
  std::vector<PreparedComparison> leaves;
  for (const auto& a : sources)
    leaves.push_back(source(a));
  auto n = leaves[0].queries, m = leaves[0].references;
  auto device = leaves[0].values[X].device();
  auto batch = leaves[0].batch;
  // Source chunk sizes are explicit bounds, including values above the
  // default. Mixed sources use their smallest requested width.
  int64_t size = leaves[0].settings.size;
  ExpressionPacket packet;
  packet.nodes = nodes;
  packet.numbers = numbers.vec();
  for (auto& leaf : leaves) {
    TORCH_CHECK(leaf.queries == n && leaf.references == m,
                "Expression sources require equal positional N/M dimensions");
    TORCH_CHECK(leaf.values[X].device() == device,
                "Expression sources must share a device");
    batch = at::infer_size_symint(batch, leaf.batch);
    size = std::min(size, leaf.settings.size);
    const auto& s = leaf.settings;
    packet.names.push_back(s.metric);
    packet.families.push_back(s.family.value_or(""));
    packet.headers.push_back({s.rank, s.size, s.dtype, s.kernel, s.self_cosine,
                              s.transposed(), s.mask});
    packet.reals.push_back(s.reals);
    packet.integers.push_back(s.integers);
    packet.flags.push_back(s.flags);
  }
  // Validate edges and scalar indices before lowering.
  auto numbered = expression_plan(packet);
  std::vector<int64_t> tensor_kinds(tensors.size(), -1);
  for (const auto& node : packet.nodes) {
    auto op = node[0];
    if (op != 1 && !expression_axis_slot(op))
      continue;
    auto slot = node[op == 1 ? 1 : 2];
    auto kind = op == 38 ? 9 : op == 39 ? 10 : op;
    TORCH_CHECK(slot >= 0 && slot < int64_t(tensors.size()),
                "Expression tensor index out of range");
    TORCH_CHECK(op != 1 || node[2] == -1, "Invalid scalar tensor node");
    TORCH_CHECK(tensor_kinds[slot] == -1 || tensor_kinds[slot] == kind,
                "Tensor slot cannot mix scalar/row/column roles");
    tensor_kinds[slot] = kind;
  }
  Tensors promoted = tensors.vec();
  bool amp = operator_autocast_enabled(leaves[0].values[X]);
  for (size_t k = 0; k < promoted.size(); ++k) {
    auto& t = promoted[k];
    auto kind = tensor_kinds[k];
    TORCH_CHECK(kind != -1, "Unused expression tensor");
    TORCH_CHECK(
        t.layout() == at::kStrided && t.device() == device && !t.is_complex(),
        "Expression factors/weights must be dense real tensors on the source "
        "device");
    if (amp &&
        (t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16))
      t = t.to(at::kFloat);
    TORCH_CHECK(
        t.scalar_type() != at::kHalf && t.scalar_type() != at::kBFloat16,
        "Low-precision factors require autocast");
    if (kind == 1) {
      TORCH_CHECK(t.dim() == 0,
                  "Tensor scalar factors must have zero dimensions; use "
                  "explicit row/column weights");
    } else {
      TORCH_CHECK(t.dim() >= 1 && t.sym_size(-1) == (kind == 9 ? n : m),
                  "Expression row/column weight count mismatch");
      batch = at::infer_size_symint(batch, t.sym_sizes().slice(0, t.dim() - 1));
    }
  }
  if (axes) {
    TORCH_CHECK(operation == "reduce", "Dimension lists require a reduction");
    auto selected = detail::operator_axes(*axes, batch.size() + 2, 2);
    o.dim = selected.size() == 2 ? c10::nullopt
                                 : c10::optional<int64_t>(selected[0] - 2);
  }
  // Infer entry promotion on empty tiles, before the consumer's accumulation
  // dtype.
  Tensors probe(Slots, at::empty({0}, leaves[0].values[X].options()));
  probe[XL] = at::empty({0}, probe[X].options().dtype(at::kLong));
  probe[YL] = probe[XL];
  for (const auto& leaf : leaves)
    probe.insert(probe.end(), leaf.values.begin(), leaf.values.end());
  for (size_t k = 0; k < promoted.size(); ++k)
    probe.push_back(tensor_kinds[k] == 1
                        ? promoted[k]
                        : at::empty({0}, promoted[k].options()));
  Tensors fake_leaves;
  for (const auto& leaf : leaves)
    fake_leaves.push_back(at::empty(
        {0, 0},
        leaf.values[X].options().dtype(at::ScalarType(leaf.settings.dtype))));
  // Lower scalar numbers once per consumer call. This pins their dtype for
  // backward/export and avoids allocating a device scalar in every pair tile.
  // A floating number takes its partner's floating dtype, so float64 operands
  // see the exact Python value with unchanged result dtypes.
  auto number_dtypes = expression_number_dtypes(probe, *numbered, fake_leaves);
  std::vector<std::tuple<int64_t, at::ScalarType, int64_t>> number_slots;
  for (size_t k = 0; k < packet.nodes.size(); ++k) {
    auto& node = packet.nodes[k];
    if (node[0] != 2)
      continue;
    auto number_index = node[1];
    auto dtype = number_dtypes[k];
    int64_t slot = -1;
    for (const auto& [index, type, existing] : number_slots)
      if (index == number_index && type == dtype)
        slot = existing;
    if (slot < 0) {
      slot = promoted.size();
      number_slots.emplace_back(number_index, dtype, slot);
      promoted.push_back(at::scalar_tensor(
          numbers[number_index], leaves[0].values[X].options().dtype(dtype)));
      tensor_kinds.push_back(1);
      probe.push_back(promoted.back());
    }
    // -2 marks a lowered number: a constant without gradient, which fused
    // reducers may read on device (user scalar tensors keep -1).
    node[0] = 1;
    node[1] = slot;
    node[2] = -2;
  }
  packet.numbers.clear();
  auto plan = expression_plan(packet);
  // A source-free public program is rejected above; validated node edges
  // cannot read past the root.
  auto dtype = infer_expression_dtype(probe, *plan, fake_leaves, output_root);
  if (other_root >= 0)
    dtype = at::promoteTypes(
        dtype, infer_expression_dtype(probe, *plan, fake_leaves, other_root));
  TORCH_CHECK(!softmax || dtype == at::kFloat || dtype == at::kDouble,
              "softmax_matmul requires float32/float64 entries");
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble ||
                  dtype == at::kLong || dtype == at::kInt ||
                  dtype == at::kShort || dtype == at::kChar ||
                  dtype == at::kByte || dtype == at::kBool,
              "Unsupported expression output dtype");
  auto base = operation == "matmul" ? std::string("matmul")
              : operation == "reduce" && o.reduction != "mean"
                  ? o.reduction
                  : std::string("sum");
  bool search = base == "min" || base == "max";
  bool transposed =
      o.transposed != (operation == "reduce" && o.dim && *o.dim == -2);
  if (operation == "matmul" || operation == "bilinear" ||
      operation == "quadratic") {
    if (!at::isFloatingType(dtype)) {
      auto coeff = operation == "matmul" ? o.right : o.left;
      TORCH_CHECK(coeff, "Missing expression coefficient");
      dtype = coeff->scalar_type();
      if (amp && (dtype == at::kHalf || dtype == at::kBFloat16))
        dtype = at::kFloat;
      TORCH_CHECK(
          dtype == at::kFloat || dtype == at::kDouble,
          "Integer expression products require float32/float64 coefficients");
    }
  } else if (operation == "reduce" && (base == "sum" || base == "prod") &&
             !at::isFloatingType(dtype))
    dtype = at::kLong;
  Settings s{"_expression",
             (transposed ? "transpose_" : "") +
                 (search ? std::string("neighbors") : base),
             false,
             base == "max",
             o.exclude_self,
             false,
             search,
             2,
             1,
             o.chunk_size.value_or(size),
             int64_t(dtype),
             3,
             {},
             {},
             {},
             std::string("_expression")};
  s.expression = plan;
  TORCH_CHECK(s.size > 0, "chunk_size must be positive");
  auto logical_n = transposed ? m : n, logical_m = transposed ? n : m;
  TORCH_CHECK(!o.exclude_self || n == m,
              "exclude_self requires a square expression");
  TORCH_CHECK(!search || topk || logical_m - int64_t(o.exclude_self) > 0,
              "min/max require an eligible reference");
  auto opts = leaves[0].values[X].options().dtype(dtype);
  auto xs = batch, ys = batch;
  xs.insert(xs.end(), {n, 0});
  ys.insert(ys.end(), {m, 0});
  Tensors v(Slots, at::empty({0}, opts));
  v[X] = at::empty_symint(xs, opts);
  v[Y] = at::empty_symint(ys, opts);
  v[XL] = at::arange(n, opts.dtype(at::kLong));
  v[YL] = at::arange(m, opts.dtype(at::kLong));
  for (const auto& leaf : leaves)
    v.insert(v.end(), leaf.values.begin(), leaf.values.end());
  v.insert(v.end(), promoted.begin(), promoted.end());
  if (operation == "matmul" || softmax) {
    TORCH_CHECK(o.right, "matmul requires values");
    auto t = *o.right;
    if (amp &&
        (t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16))
      t = t.to(at::kFloat);
    if (t.dim() == 1)
      t = t.unsqueeze(-1);
    TORCH_CHECK(
        t.layout() == at::kStrided && t.dim() >= 2 &&
            t.sym_size(-2) == logical_m && t.sym_size(-1) > 0 &&
            t.device() == device && t.scalar_type() == dtype,
        "values must match expression dtype/device and have shape [...,M,C]");
    batch = at::infer_size_symint(batch, t.sym_sizes().slice(0, t.dim() - 2));
    v[Payload] = t;
    s.mask |= int64_t(1) << Payload;
  }
  return {std::move(v), std::move(s), std::move(batch), logical_n, logical_m};
}
PreparedComparison prepare_expression_graph(const ExpressionGraph& graph) {
  // A plain source has no scalar algebra to interpret. Reuse the same source
  // preparation and its typed tiled consumer for repeated native products.
  if (graph.sources.size() == 1 && graph.nodes.size() == 1 &&
      graph.nodes[0] == std::vector<int64_t>({0, 0, -1}) &&
      graph.factors.empty() && graph.numbers.empty())
    return source(graph.sources[0]);
  return prepare_expression(graph.sources, graph.nodes, graph.factors,
                            graph.numbers, "diagonal", {});
}
}  // namespace detail::streaming
namespace {
template <class T, class Member>
void unpack_options(const c10::Dict<std::string, T>& options,
                    std::vector<ExpressionSource>& sources, Member member) {
  for (const auto& entry : options) {
    auto key = entry.key();
    auto colon = key.find(':');
    TORCH_CHECK(colon != std::string::npos && colon > 0,
                "Invalid expression option key");
    auto prefix = key.substr(0, colon);
    TORCH_CHECK(prefix.find_first_not_of("0123456789") == std::string::npos,
                "Invalid expression source index");
    auto index = std::stoull(prefix);
    TORCH_CHECK(index < sources.size(),
                "Expression option source index out of range");
    (sources[index].*member).insert(key.substr(colon + 1), entry.value());
  }
}
template <class T>
void pack_options(c10::Dict<std::string, T>& output,
                  const c10::Dict<std::string, T>& input, size_t source) {
  for (const auto& item : input)
    output.insert(std::to_string(source) + ":" + item.key(), item.value());
}
using RealLists = c10::Dict<std::string, c10::List<double>>;
using IntLists = c10::Dict<std::string, c10::List<int64_t>>;
Tensor expression_operations_impl_common(
    const std::vector<std::vector<Tensor>>& inputs,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef program, at::TensorList factors,
    at::IntArrayRef number_kinds, at::IntArrayRef integer_numbers,
    at::ArrayRef<double> real_numbers, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transpose, c10::optional<int64_t> chunk_size,
    c10::optional<double> correction,
    c10::optional<std::array<int64_t, 2>> weighted_roots = c10::nullopt,
    const c10::optional<std::vector<int64_t>>& axes = c10::nullopt,
    c10::optional<int64_t> k = c10::nullopt, bool largest = true,
    Tensor* selected = nullptr,
    c10::optional<double> selection_q = c10::nullopt,
    c10::string_view selection_interpolation = "linear") {
  auto n = inputs.size();
  TORCH_CHECK(
      metrics.size() == n && families.size() == n && transposed.size() == n,
      "Expression source packet sizes differ");
  std::vector<ExpressionSource> sources;
  for (size_t k = 0; k < n; ++k) {
    ExpressionSource a;
    a.inputs = inputs[k];
    a.metric = metrics[k];
    a.family = families[k];
    a.transposed = transposed[k];
    auto key = std::to_string(k);
    if (real_lists.contains(key))
      a.real_list = real_lists.at(key).vec();
    if (integer_lists.contains(key))
      a.integer_list = integer_lists.at(key).vec();
    sources.push_back(std::move(a));
  }
  unpack_options(tensors, sources, &ExpressionSource::tensors);
  unpack_options(reals, sources, &ExpressionSource::reals);
  unpack_options(integers, sources, &ExpressionSource::integers);
  unpack_options(flags, sources, &ExpressionSource::flags);
  unpack_options(strings, sources, &ExpressionSource::strings);
  TORCH_CHECK(program.size() % 3 == 0,
              "Expression program must contain opcode/operand triples");
  std::vector<std::vector<int64_t>> nodes;
  for (size_t k = 0; k < program.size(); k += 3)
    nodes.push_back({program[k], program[k + 1], program[k + 2]});
  TORCH_CHECK(number_kinds.size() == integer_numbers.size() &&
                  number_kinds.size() == real_numbers.size(),
              "Expression scalar packet sizes differ");
  std::vector<at::Scalar> numbers;
  for (size_t k = 0; k < number_kinds.size(); ++k) {
    TORCH_CHECK(number_kinds[k] >= 0 && number_kinds[k] <= 2,
                "Invalid expression scalar type");
    numbers.push_back(number_kinds[k] == 0
                          ? at::Scalar(bool(integer_numbers[k]))
                      : number_kinds[k] == 1 ? at::Scalar(integer_numbers[k])
                                             : at::Scalar(real_numbers[k]));
  }
  PairwiseOperationOptions o;
  o.left = left;
  o.right = right;
  o.rows = rows;
  o.columns = columns;
  o.reduction = selection_q  ? "min"
                : correction ? "sum"
                             : std::string(reduction);
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  o.transposed = transpose;
  o.chunk_size = chunk_size;
  bool hsic = operation == "hsic" || operation == "hsic_unbiased";
  bool inner = hsic || operation == "frobenius" ||
               operation == "centered_frobenius" || operation == "alignment" ||
               operation == "centered_alignment";
  if (inner)
    TORCH_CHECK(weighted_roots && correction && *correction == 0. && !dim &&
                    !keepdim && !exclude_self && !transpose && !axes && !k &&
                    !left && !right && !rows && !columns && reduction == "sum",
                "Unused Frobenius inner product option");
  auto p = prepare_expression(
      sources, nodes, factors, numbers, inner ? "reduce" : operation, o,
      weighted_roots ? (*weighted_roots)[0] : -1, axes,
      bool(k) || bool(selection_q), inner ? (*weighted_roots)[1] : -1);
  if (inner)
    return expression_inner_product(
        p.values, p.settings, (*weighted_roots)[0], (*weighted_roots)[1],
        hsic || operation == "centered_frobenius" ||
            operation == "centered_alignment",
        operation == "alignment" || operation == "centered_alignment",
        hsic ? operation : "inner_product");
  if (axes) {
    auto selected = detail::operator_axes(*axes, p.batch.size() + 2, 2);
    dim = selected.size() == 2 ? c10::nullopt
                               : c10::optional<int64_t>(selected[0] - 2);
    o.dim = dim;
  }
  if (selection_q) {
    TORCH_CHECK(dim, "Exact selection requires one sample axis");
    auto result = expression_selection(p.values, p.settings, reduction,
                                       *selection_q, selection_interpolation);
    return keepdim ? result.unsqueeze(*dim == -2 ? -2 : -1) : result;
  }
  if (k) {
    auto result = expression_topk(p.values, p.settings, *k, largest);
    auto values = std::get<0>(result), indices = std::get<1>(result);
    if (o.dim && *o.dim == -2) {
      values = values.transpose(-1, -2);
      indices = indices.transpose(-1, -2);
    }
    *selected = indices;
    return values;
  }
  if (correction) {
    auto result = weighted_roots
                      ? expression_weighted_statistics(
                            p.values, p.settings, reduction, !dim, *correction,
                            (*weighted_roots)[0], (*weighted_roots)[1])
                      : expression_statistics(p.values, p.settings, reduction,
                                              !dim, *correction);
    if (!keepdim)
      return result;
    return dim ? result.unsqueeze(*dim == -2 ? -2 : -1)
               : result.unsqueeze(-1).unsqueeze(-1);
  }
  if (operation == "softmax_matmul") {
    auto result = expression_softmax_matmul(p.values, p.settings);
    return right->dim() == 1 ? result.squeeze(-1) : result;
  }
  return execute_prepared(std::move(p), operation, o,
                          operator_autocast_enabled(inputs[0][0]));
}
Tensor expression_operations_impl(
    const std::vector<std::vector<Tensor>>& inputs,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef program, at::TensorList factors,
    at::IntArrayRef number_kinds, at::IntArrayRef integer_numbers,
    at::ArrayRef<double> real_numbers, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transpose, c10::optional<int64_t> chunk_size) {
  return expression_operations_impl_common(
      inputs, metrics, families, tensors, reals, integers, flags, strings,
      real_lists, integer_lists, transposed, program, factors, number_kinds,
      integer_numbers, real_numbers, operation, left, right, rows, columns,
      reduction, dim, keepdim, exclude_self, transpose, chunk_size,
      c10::nullopt);
}
Tensor expression_statistics_impl(
    const std::vector<std::vector<Tensor>>& inputs,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef program, at::TensorList factors,
    at::IntArrayRef number_kinds, at::IntArrayRef integer_numbers,
    at::ArrayRef<double> real_numbers, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transpose, c10::optional<int64_t> chunk_size,
    double correction) {
  TORCH_CHECK(operation == "reduce", "Statistics require operation=reduce");
  return expression_operations_impl_common(
      inputs, metrics, families, tensors, reals, integers, flags, strings,
      real_lists, integer_lists, transposed, program, factors, number_kinds,
      integer_numbers, real_numbers, operation, left, right, rows, columns,
      reduction, dim, keepdim, exclude_self, transpose, chunk_size, correction);
}
Tensor expression_weighted_statistics_impl(
    const std::vector<std::vector<Tensor>>& inputs,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef program, at::TensorList factors,
    at::IntArrayRef number_kinds, at::IntArrayRef integer_numbers,
    at::ArrayRef<double> real_numbers, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transpose, c10::optional<int64_t> chunk_size,
    double correction, int64_t value_root, int64_t weight_root) {
  TORCH_CHECK(operation == "reduce", "Statistics require operation=reduce");
  return expression_operations_impl_common(
      inputs, metrics, families, tensors, reals, integers, flags, strings,
      real_lists, integer_lists, transposed, program, factors, number_kinds,
      integer_numbers, real_numbers, operation, left, right, rows, columns,
      reduction, dim, keepdim, exclude_self, transpose, chunk_size, correction,
      std::array<int64_t, 2>{value_root, weight_root});
}
Tensor expression_axes_impl(
    const std::vector<std::vector<Tensor>>& inputs,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef program, at::TensorList factors,
    at::IntArrayRef number_kinds, at::IntArrayRef integer_numbers,
    at::ArrayRef<double> real_numbers, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transpose, c10::optional<int64_t> chunk_size,
    c10::optional<double> correction, int64_t value_root, int64_t weight_root,
    at::IntArrayRef dims) {
  TORCH_CHECK(
      operation == "reduce" && !dim,
      "Dimension-list reductions require operation=reduce and dim=None");
  c10::optional<std::array<int64_t, 2>> roots;
  if (value_root >= 0 || weight_root >= 0) {
    TORCH_CHECK(value_root >= 0 && weight_root >= 0 && correction,
                "Weighted axes require both roots and a correction");
    roots = std::array<int64_t, 2>{value_root, weight_root};
  }
  return expression_operations_impl_common(
      inputs, metrics, families, tensors, reals, integers, flags, strings,
      real_lists, integer_lists, transposed, program, factors, number_kinds,
      integer_numbers, real_numbers, operation, left, right, rows, columns,
      reduction, dim, keepdim, exclude_self, transpose, chunk_size, correction,
      roots, dims.vec());
}

std::tuple<Tensor, Tensor> expression_topk_impl(
    const std::vector<std::vector<Tensor>>& inputs,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef program, at::TensorList factors,
    at::IntArrayRef number_kinds, at::IntArrayRef integer_numbers,
    at::ArrayRef<double> real_numbers, int64_t k, int64_t dim, bool largest,
    bool sorted, bool exclude_self, c10::optional<int64_t> chunk_size) {
  TORCH_CHECK(k >= 0, "k must be nonnegative");
  Tensor indices;
  // sorted=False permits a sorted result, matching torch.topk's contract.
  auto result = expression_operations_impl_common(
      inputs, metrics, families, tensors, reals, integers, flags, strings,
      real_lists, integer_lists, transposed, program, factors, number_kinds,
      integer_numbers, real_numbers, "reduce", c10::nullopt, c10::nullopt,
      c10::nullopt, c10::nullopt, largest ? "max" : "min", c10::nullopt, false,
      exclude_self, false, chunk_size, c10::nullopt, c10::nullopt,
      std::vector<int64_t>{dim}, k, largest, &indices);
  return {result, indices};
}
Tensor expression_selection_impl(
    at::TensorList source_inputs, at::IntArrayRef counts,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef program, at::TensorList factors,
    at::IntArrayRef number_kinds, at::IntArrayRef integer_numbers,
    at::ArrayRef<double> real_numbers, c10::string_view reduction, double q,
    int64_t dim, bool keepdim, bool exclude_self,
    c10::string_view interpolation, c10::optional<int64_t> chunk_size) {
  std::vector<std::vector<Tensor>> inputs;
  int64_t offset = 0;
  for (auto count : counts) {
    TORCH_CHECK(count > 0 && offset + count <= int64_t(source_inputs.size()),
                "Invalid flat expression source counts");
    inputs.push_back(source_inputs.slice(offset, count).vec());
    offset += count;
  }
  TORCH_CHECK(offset == int64_t(source_inputs.size()),
              "Unused flat source inputs");
  TORCH_CHECK(reduction == "median" || reduction == "quantile",
              "Unknown exact selection consumer");
  return expression_operations_impl_common(
      inputs, metrics, families, tensors, reals, integers, flags, strings,
      real_lists, integer_lists, transposed, program, factors, number_kinds,
      integer_numbers, real_numbers, "reduce", c10::nullopt, c10::nullopt,
      c10::nullopt, c10::nullopt, reduction, c10::nullopt, keepdim,
      exclude_self, false, chunk_size, c10::nullopt, c10::nullopt,
      std::vector<int64_t>{dim}, c10::nullopt, true, nullptr, q, interpolation);
}
std::tuple<Tensor, Tensor> expression_flat_impl(
    at::TensorList source_inputs, at::IntArrayRef counts,
    const std::vector<std::string>& metrics,
    const std::vector<std::string>& families, const TensorDict& tensors,
    const RealDict& reals, const IntDict& integers, const BoolDict& flags,
    const StringDict& strings, const RealLists& real_lists,
    const IntLists& integer_lists, const c10::List<bool>& transposed,
    at::IntArrayRef program, at::TensorList factors,
    at::IntArrayRef number_kinds, at::IntArrayRef integer_numbers,
    at::ArrayRef<double> real_numbers, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transpose, c10::optional<int64_t> chunk_size,
    c10::optional<double> correction, int64_t value_root, int64_t weight_root,
    at::IntArrayRef dims, bool use_axes, c10::optional<int64_t> k, bool largest,
    bool sorted) {
  std::vector<std::vector<Tensor>> inputs;
  int64_t offset = 0;
  for (auto count : counts) {
    TORCH_CHECK(count > 0 && offset + count <= int64_t(source_inputs.size()),
                "Invalid flat expression source counts");
    inputs.push_back(source_inputs.slice(offset, count).vec());
    offset += count;
  }
  TORCH_CHECK(offset == int64_t(source_inputs.size()),
              "Unused flat source inputs");
  if (k) {
    TORCH_CHECK(operation == "reduce" && !left && !right && !rows && !columns &&
                    !keepdim && !transpose && !correction && !use_axes &&
                    value_root < 0 && weight_root < 0,
                "Unused top-k expression option");
    return expression_topk_impl(
        inputs, metrics, families, tensors, reals, integers, flags, strings,
        real_lists, integer_lists, transposed, program, factors, number_kinds,
        integer_numbers, real_numbers, *k, dim.value_or(-1), largest, sorted,
        exclude_self, chunk_size);
  }
  c10::optional<std::array<int64_t, 2>> roots;
  if (value_root >= 0 || weight_root >= 0) {
    TORCH_CHECK(value_root >= 0 && weight_root >= 0 && correction,
                "Weighted expressions require roots and correction");
    roots = std::array<int64_t, 2>{value_root, weight_root};
  }
  c10::optional<std::vector<int64_t>> axes;
  if (use_axes) {
    TORCH_CHECK(operation == "reduce" && !dim,
                "Axes require reduce with dim=None");
    axes = dims.vec();
  } else
    TORCH_CHECK(dims.empty(), "Unused expression dimensions");
  auto result = expression_operations_impl_common(
      inputs, metrics, families, tensors, reals, integers, flags, strings,
      real_lists, integer_lists, transposed, program, factors, number_kinds,
      integer_numbers, real_numbers, operation, left, right, rows, columns,
      reduction, dim, keepdim, exclude_self, transpose, chunk_size, correction,
      roots, axes);
  return {result, at::empty({0}, result.options().dtype(at::kLong))};
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "expression_operations.selection(Tensor[] source_inputs, int[] counts, "
      "str[] metrics, "
      "str[] families, Dict(str,Tensor) tensors, Dict(str,float) reals, "
      "Dict(str,int) integers, "
      "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "
      "real_lists, "
      "Dict(str,int[]) integer_lists, bool[] transposed, int[] program, "
      "Tensor[] factors, "
      "int[] number_kinds, int[] integer_numbers, float[] real_numbers, str "
      "reduction, "
      "float q=0.5, int dim=-1, bool keepdim=False, bool exclude_self=False, "
      "str interpolation='linear', int? chunk_size=None) -> Tensor",
      TORCH_FN(expression_selection_impl));
  m.def(
      "expression_operations.flat(Tensor[] source_inputs, int[] counts, str[] "
      "metrics, "
      "str[] families, Dict(str,Tensor) tensors, Dict(str,float) reals, "
      "Dict(str,int) integers, "
      "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "
      "real_lists, "
      "Dict(str,int[]) integer_lists, bool[] transposed, int[] program, "
      "Tensor[] factors, "
      "int[] number_kinds, int[] integer_numbers, float[] real_numbers, str "
      "operation='reduce', "
      "Tensor? left=None, Tensor? right=None, Tensor? rows=None, Tensor? "
      "columns=None, "
      "str reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, "
      "bool transpose=False, int? chunk_size=None, float? correction=None, int "
      "value_root=-1, "
      "int weight_root=-1, int[] dims=[], bool use_axes=False, int? k=None, "
      "bool largest=True, "
      "bool sorted=True) -> (Tensor, Tensor)",
      TORCH_FN(expression_flat_impl));

  m.def(
      "expression_operations.topk(Tensor[][] inputs, str[] metrics, str[] "
      "families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, "
      "Dict(str,bool) flags, Dict(str,str) strings, Dict(str,float[]) "
      "real_lists, "
      "Dict(str,int[]) integer_lists, bool[] transposed, int[] program, "
      "Tensor[] factors, "
      "int[] number_kinds, int[] integer_numbers, float[] real_numbers, int k, "
      "int dim=-1, bool largest=True, bool sorted=True, bool "
      "exclude_self=False, "
      "int? chunk_size=None) -> (Tensor, Tensor)",
      TORCH_FN(expression_topk_impl));

  m.def(
      "expression_operations(Tensor[][] inputs, str[] metrics, str[] families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] program, Tensor[] factors, int[] number_kinds, int[] "
      "integer_numbers, float[] real_numbers, str operation='reduce', Tensor? "
      "left=None, Tensor? right=None, Tensor? rows=None, Tensor? columns=None, "
      "str reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transpose=False, int? chunk_size=None) -> "
      "Tensor",
      TORCH_FN(expression_operations_impl));
  m.def(
      "expression_operations.statistics(Tensor[][] inputs, str[] metrics, "
      "str[] families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] program, Tensor[] factors, int[] number_kinds, int[] "
      "integer_numbers, float[] real_numbers, str operation='reduce', Tensor? "
      "left=None, Tensor? right=None, Tensor? rows=None, Tensor? columns=None, "
      "str reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transpose=False, int? chunk_size=None, float "
      "correction=1.) -> "
      "Tensor",
      TORCH_FN(expression_statistics_impl));
  m.def(
      "expression_operations.weighted_statistics(Tensor[][] inputs, str[] "
      "metrics, "
      "str[] families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] program, Tensor[] factors, int[] number_kinds, int[] "
      "integer_numbers, float[] real_numbers, str operation='reduce', Tensor? "
      "left=None, Tensor? right=None, Tensor? rows=None, Tensor? columns=None, "
      "str reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transpose=False, int? chunk_size=None, float "
      "correction=0., int value_root=-1, int weight_root=-1) -> "
      "Tensor",
      TORCH_FN(expression_weighted_statistics_impl));
  m.def(
      "expression_operations.axes(Tensor[][] inputs, str[] "
      "metrics, "
      "str[] families, "
      "Dict(str,Tensor) tensors, Dict(str,float) reals, Dict(str,int) "
      "integers, Dict(str,bool) flags, Dict(str,str) strings, "
      "Dict(str,float[]) real_lists, Dict(str,int[]) integer_lists, bool[] "
      "transposed, int[] program, Tensor[] factors, int[] number_kinds, int[] "
      "integer_numbers, float[] real_numbers, str operation='reduce', Tensor? "
      "left=None, Tensor? right=None, Tensor? rows=None, Tensor? columns=None, "
      "str reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transpose=False, int? chunk_size=None, float? "
      "correction=None, int value_root=-1, int weight_root=-1, int[] dims=[]) "
      "-> "
      "Tensor",
      TORCH_FN(expression_axes_impl));
}
}  // namespace
// Native facade implementation is below; source factories preserve typed family
// options.
struct ScalarExpressionNode {
  int64_t opcode;
  std::shared_ptr<const ScalarExpressionNode> left, right;
  c10::optional<ExpressionSource> source;
  Tensor tensor;
  at::Scalar number = 0;
};
namespace {
std::shared_ptr<const ScalarExpressionNode> node(
    int64_t op, std::shared_ptr<const ScalarExpressionNode> left = {},
    std::shared_ptr<const ScalarExpressionNode> right = {},
    Tensor tensor = Tensor()) {
  return std::make_shared<ScalarExpressionNode>(ScalarExpressionNode{
      op, std::move(left), std::move(right), c10::nullopt, std::move(tensor)});
}
struct ExpressionGeometry {
  std::vector<c10::SymInt> shape;
  c10::optional<c10::Device> device;
};
ExpressionGeometry merge_geometry(ExpressionGeometry a,
                                  const ExpressionGeometry& b) {
  TORCH_CHECK(!a.device || !b.device || a.device == b.device,
              "Expression operands must share a device");
  if (!a.device)
    a.device = b.device;
  if (a.shape.empty()) {
    a.shape = b.shape;
  } else if (!b.shape.empty()) {
    auto rank = a.shape.size();
    TORCH_CHECK(a.shape[rank - 2] == b.shape[b.shape.size() - 2] &&
                    a.shape.back() == b.shape.back(),
                "Expression sources require equal positional N/M dimensions");
    auto batch = at::infer_size_symint(
        c10::SymIntArrayRef(a.shape).slice(0, rank - 2),
        c10::SymIntArrayRef(b.shape).slice(0, b.shape.size() - 2));
    batch.insert(batch.end(), {a.shape[rank - 2], a.shape.back()});
    a.shape = std::move(batch);
  }
  return a;
}
ExpressionGeometry expression_geometry(
    const std::shared_ptr<const ScalarExpressionNode>& root) {
  at::NoGradGuard guard;
  std::vector<std::pair<const ScalarExpressionNode*, ExpressionGeometry>> seen;
  std::function<ExpressionGeometry(
      const std::shared_ptr<const ScalarExpressionNode>&)>
      visit;
  visit = [&](const std::shared_ptr<const ScalarExpressionNode>& n) {
    for (const auto& [ptr, geometry] : seen) {
      if (ptr == n.get())
        return geometry;
    }
    ExpressionGeometry result;
    auto op = n->opcode;
    if (op == 0) {
      auto prepared = source(*n->source);
      result.shape = prepared.batch;
      result.shape.insert(result.shape.end(),
                          {prepared.queries, prepared.references});
      result.device = prepared.values[detail::streaming::X].device();
    } else if (op == 1) {
      result.device = n->tensor.device();
    } else if (op != 2) {
      result = visit(n->left);
      if (op == 11 || expression_axis_slot(op) || op == 40 || op == 41) {
        TORCH_CHECK(result.shape.size() >= 2 || op == 9 || op == 10,
                    "Axis expressions require a matrix shape provider");
        if (op == 11) {
          std::iter_swap(result.shape.end() - 2, result.shape.end() - 1);
        } else if (expression_axis_slot(op)) {
          const auto& tensor = n->tensor;
          TORCH_CHECK(tensor.dim() >= 1,
                      "Weights must be tensors with shape [...,N] or [...,M]");
          TORCH_CHECK(!result.device || tensor.device() == *result.device,
                      "Expression operands must share a device");
          result.device = tensor.device();
          // Scalar weighting expressions acquire their axes from the value
          // root in weighted_statistics; the native role check validates them.
          if (result.shape.empty()) {
            seen.emplace_back(n.get(), result);
            return result;
          }
          auto rank = result.shape.size();
          auto axis = expression_row_slot(op) ? rank - 2 : rank - 1;
          TORCH_CHECK(tensor.sym_size(-1) == result.shape[axis],
                      "Expression row/column weight count mismatch");
          auto batch = at::infer_size_symint(
              c10::SymIntArrayRef(result.shape).slice(0, rank - 2),
              tensor.sym_sizes().slice(0, tensor.dim() - 1));
          batch.insert(batch.end(),
                       {result.shape[rank - 2], result.shape.back()});
          result.shape = std::move(batch);
        }
      } else if (expression_binary(op) || expression_scalar_bound(op) ||
                 op == 34 || op == 35) {
        result = merge_geometry(std::move(result), visit(n->right));
      } else {
        TORCH_CHECK(expression_unary(op), "Unsupported facade expression node");
      }
    }
    seen.emplace_back(n.get(), result);
    return result;
  };
  return visit(root);
}
ExpressionGraph expression_graph(
    const std::shared_ptr<const ScalarExpressionNode>& root,
    const std::shared_ptr<const ScalarExpressionNode>& weights = {},
    ExpressionGeometry* output_geometry = nullptr) {
  auto geometry = expression_geometry(root);
  if (weights)
    geometry =
        merge_geometry(std::move(geometry), expression_geometry(weights));
  if (output_geometry)
    *output_geometry = geometry;
  ExpressionGraph graph;
  auto& sources = graph.sources;
  auto& nodes = graph.nodes;
  auto& factors = graph.factors;
  auto& numbers = graph.numbers;
  std::vector<std::tuple<const ScalarExpressionNode*, bool, int64_t>> seen;
  std::function<int64_t(const std::shared_ptr<const ScalarExpressionNode>&,
                        bool)>
      visit;
  visit = [&](const std::shared_ptr<const ScalarExpressionNode>& n,
              bool transposed) {
    for (const auto& [ptr, t, id] : seen)
      if (ptr == n.get() && t == transposed)
        return id;
    int64_t a = -1, b = -1, op = n->opcode;
    if (op == 11) {
      auto id = visit(n->left, !transposed);
      seen.emplace_back(n.get(), transposed, id);
      return id;
    }
    if (op == 0) {
      a = sources.size();
      auto source = *n->source;
      source.transposed = source.transposed != transposed;
      sources.push_back(std::move(source));
    } else if (op == 1) {
      a = factors.size();
      factors.push_back(n->tensor);
    } else if (op == 2) {
      a = numbers.size();
      numbers.push_back(n->number);
    } else if (op == 38 || op == 39) {
      b = factors.size();
      factors.push_back(n->tensor);
      if (transposed)
        op = 77 - op;
    } else if (op == 40 || op == 41) {
      if (transposed)
        op = 81 - op;
    } else {
      a = visit(n->left, transposed);
      if (op == 9 || op == 10) {
        b = factors.size();
        factors.push_back(n->tensor);
        if (transposed)
          op = 19 - op;
      } else if (expression_binary(op) || expression_scalar_bound(op) ||
                 op == 34 || op == 35) {
        b = visit(n->right, transposed);
      } else {
        TORCH_CHECK(expression_unary(op), "Unsupported facade expression node");
      }
    }
    auto id = int64_t(nodes.size());
    nodes.push_back({op, a, b});
    seen.emplace_back(n.get(), transposed, id);
    return id;
  };
  graph.value_root = visit(root, false);
  graph.weight_root = weights ? visit(weights, false) : -1;
  if (sources.empty()) {
    TORCH_CHECK(geometry.shape.size() >= 2 && geometry.device,
                "A scalar expression requires a matrix source");
    // Match matrix_expression's default source, without visiting the like.
    // The expanded scalar carries geometry without allocating a dense matrix.
    auto options =
        at::TensorOptions().device(*geometry.device).dtype(at::kFloat);
    ExpressionSource matrix;
    matrix.inputs = {at::zeros({}, options).expand_symint(geometry.shape)};
    matrix.metric = "_matrix";
    matrix.family = "matrix";
    sources.push_back(std::move(matrix));
    // No node reads this source. Infer the real entry dtype with the native
    // probes, then give the carrier that dtype, as the Python packet does.
    auto prepared = detail::streaming::prepare_expression(
        sources, nodes, factors, numbers, "diagonal", {}, graph.value_root);
    sources[0].inputs[0] =
        at::zeros({}, options.dtype(at::ScalarType(prepared.settings.dtype)))
            .expand_symint(geometry.shape);
  }
  return graph;
}
template <class T>
void put(c10::Dict<std::string, T>& d, const char* key,
         const c10::optional<T>& v) {
  if (v)
    d.insert(key, *v);
}
void binding(ExpressionSource& s, const PairwiseOperationOptions& o) {
  TORCH_CHECK(!o.left && !o.right && !o.rows && !o.columns && !o.dim &&
                  o.reduction == "sum" && !o.keepdim && !o.exclude_self,
              "Source binding accepts comparison,chunk_size and transpose "
              "options only");
  s.transposed = o.transposed;
  put(s.integers, "chunk_size", o.chunk_size);
  const auto& q = o.comparison;
  put(s.tensors, "w", q.w);
  put(s.tensors, "V", q.V);
  put(s.tensors, "VI", q.VI);
  put(s.tensors, "bin_positions", q.bin_positions);
  put(s.tensors, "projections", q.projections);
  put(s.tensors, "x1_lengths", q.x1_lengths);
  put(s.tensors, "x2_lengths", q.x2_lengths);
  put(s.reals, "p", q.p);
  put(s.reals, "base", q.base);
  put(s.reals, "alpha", q.alpha);
  put(s.reals, "length_scale", q.length_scale);
  put(s.reals, "nu", q.nu);
  put(s.reals, "gamma", q.gamma);
  put(s.reals, "coef0", q.coef0);
  put(s.reals, "regularization", q.regularization);
  put(s.integers, "degree", q.degree);
  put(s.integers, "max_iter", q.max_iter);
  put(s.integers, "metric_chunk_size", q.chunk_size);
  put(s.integers, "feature_dim", q.feature_dim);
  put(s.flags, "squared", q.squared);
  put(s.flags, "shuffle", q.shuffle);
  put(s.flags, "batched", q.batched);
  put(s.strings, "kernel", q.kernel);
  TORCH_CHECK(!q.generator,
              "Expression sources do not support shuffled search generators");
}
ExpressionSource basic(const Tensor& x, const c10::optional<Tensor>& y,
                       c10::string_view metric, c10::string_view family,
                       const PairwiseOperationOptions& o) {
  ExpressionSource s;
  s.inputs = {x};
  if (y)
    s.inputs.push_back(*y);
  s.metric = std::string(metric);
  s.family = std::string(family);
  binding(s, o);
  return s;
}
}  // namespace
ScalarExpression::ScalarExpression(
    std::shared_ptr<const ScalarExpressionNode> n)
    : node_(std::move(n)) {}
ScalarExpression::ScalarExpression(ExpressionSource source) {
  auto n = std::make_shared<ScalarExpressionNode>();
  n->opcode = 0;
  n->source = std::move(source);
  node_ = std::move(n);
}
ScalarExpression::ScalarExpression(Tensor scalar) {
  TORCH_CHECK(scalar.dim() == 0, "Expression tensor factors must be scalar");
  node_ = node(1, {}, {}, std::move(scalar));
}
ScalarExpression::ScalarExpression(at::Scalar scalar) {
  auto n = std::make_shared<ScalarExpressionNode>();
  n->opcode = 2;
  n->number = scalar;
  node_ = std::move(n);
}
ScalarExpression ScalarExpression::operator+(const ScalarExpression& b) const {
  return ScalarExpression(node(3, node_, b.node_));
}
ScalarExpression ScalarExpression::operator-(const ScalarExpression& b) const {
  return ScalarExpression(node(4, node_, b.node_));
}
ScalarExpression ScalarExpression::operator*(const ScalarExpression& b) const {
  return ScalarExpression(node(5, node_, b.node_));
}
ScalarExpression ScalarExpression::operator+(at::Scalar other) const {
  return operator+(ScalarExpression(other));
}
ScalarExpression ScalarExpression::operator-(at::Scalar other) const {
  return operator-(ScalarExpression(other));
}
ScalarExpression ScalarExpression::operator*(at::Scalar other) const {
  return operator*(ScalarExpression(other));
}
ScalarExpression ScalarExpression::operator/(
    const ScalarExpression& other) const {
  return ScalarExpression(node(12, node_, other.node_));
}
ScalarExpression ScalarExpression::operator/(at::Scalar other) const {
  return operator/(ScalarExpression(other));
}
ScalarExpression ScalarExpression::pow(const ScalarExpression& other) const {
  return ScalarExpression(node(13, node_, other.node_));
}
ScalarExpression ScalarExpression::pow(at::Scalar other) const {
  return pow(ScalarExpression(other));
}
ScalarExpression ScalarExpression::minimum(
    const ScalarExpression& other) const {
  return ScalarExpression(node(14, node_, other.node_));
}
ScalarExpression ScalarExpression::minimum(at::Scalar other) const {
  return minimum(ScalarExpression(other));
}
ScalarExpression ScalarExpression::maximum(
    const ScalarExpression& other) const {
  return ScalarExpression(node(15, node_, other.node_));
}
ScalarExpression ScalarExpression::maximum(at::Scalar other) const {
  return maximum(ScalarExpression(other));
}
ScalarExpression ScalarExpression::lt(const ScalarExpression& other) const {
  return ScalarExpression(node(26, node_, other.node_));
}
ScalarExpression ScalarExpression::lt(at::Scalar other) const {
  return lt(ScalarExpression(other));
}
ScalarExpression ScalarExpression::le(const ScalarExpression& other) const {
  return ScalarExpression(node(27, node_, other.node_));
}
ScalarExpression ScalarExpression::le(at::Scalar other) const {
  return le(ScalarExpression(other));
}
ScalarExpression ScalarExpression::gt(const ScalarExpression& other) const {
  return ScalarExpression(node(28, node_, other.node_));
}
ScalarExpression ScalarExpression::gt(at::Scalar other) const {
  return gt(ScalarExpression(other));
}
ScalarExpression ScalarExpression::ge(const ScalarExpression& other) const {
  return ScalarExpression(node(29, node_, other.node_));
}
ScalarExpression ScalarExpression::ge(at::Scalar other) const {
  return ge(ScalarExpression(other));
}
ScalarExpression ScalarExpression::eq(const ScalarExpression& other) const {
  return ScalarExpression(node(30, node_, other.node_));
}
ScalarExpression ScalarExpression::eq(at::Scalar other) const {
  return eq(ScalarExpression(other));
}
ScalarExpression ScalarExpression::ne(const ScalarExpression& other) const {
  return ScalarExpression(node(31, node_, other.node_));
}
ScalarExpression ScalarExpression::ne(at::Scalar other) const {
  return ne(ScalarExpression(other));
}
ScalarExpression ScalarExpression::logical_and(
    const ScalarExpression& other) const {
  return ScalarExpression(node(32, node_, other.node_));
}
ScalarExpression ScalarExpression::logical_and(at::Scalar other) const {
  return logical_and(ScalarExpression(other));
}
ScalarExpression ScalarExpression::logical_or(
    const ScalarExpression& other) const {
  return ScalarExpression(node(33, node_, other.node_));
}
ScalarExpression ScalarExpression::logical_or(at::Scalar other) const {
  return logical_or(ScalarExpression(other));
}
ScalarExpression ScalarExpression::clamp_min(at::Scalar bound) const {
  return ScalarExpression(node(16, node_, ScalarExpression(bound).node_));
}
ScalarExpression ScalarExpression::clamp_min(const Tensor& bound) const {
  return ScalarExpression(node(16, node_, ScalarExpression(bound).node_));
}
ScalarExpression ScalarExpression::clamp_max(at::Scalar bound) const {
  return ScalarExpression(node(17, node_, ScalarExpression(bound).node_));
}
ScalarExpression ScalarExpression::clamp_max(const Tensor& bound) const {
  return ScalarExpression(node(17, node_, ScalarExpression(bound).node_));
}
ScalarExpression ScalarExpression::sqrt() const {
  return ScalarExpression(node(18, node_));
}
ScalarExpression ScalarExpression::log() const {
  return ScalarExpression(node(19, node_));
}
ScalarExpression ScalarExpression::sin() const {
  return ScalarExpression(node(20, node_));
}
ScalarExpression ScalarExpression::cos() const {
  return ScalarExpression(node(21, node_));
}
ScalarExpression ScalarExpression::tanh() const {
  return ScalarExpression(node(22, node_));
}
ScalarExpression ScalarExpression::sigmoid() const {
  return ScalarExpression(node(23, node_));
}
ScalarExpression ScalarExpression::abs() const {
  return ScalarExpression(node(24, node_));
}
ScalarExpression ScalarExpression::logical_not() const {
  return ScalarExpression(node(25, node_));
}
ScalarExpression ScalarExpression::where(const ScalarExpression& condition,
                                         const ScalarExpression& other) const {
  return ScalarExpression(
      node(34, condition.node_, node(35, node_, other.node_)));
}
ScalarExpression ScalarExpression::where(const ScalarExpression& condition,
                                         at::Scalar other) const {
  return where(condition, ScalarExpression(other));
}
ScalarExpression ScalarExpression::row_values(const Tensor& values) const {
  return ScalarExpression(node(38, node_, {}, values));
}
ScalarExpression ScalarExpression::column_values(const Tensor& values) const {
  return ScalarExpression(node(39, node_, {}, values));
}
ScalarExpression ScalarExpression::row_index() const {
  return ScalarExpression(node(40, node_));
}
ScalarExpression ScalarExpression::column_index() const {
  return ScalarExpression(node(41, node_));
}
ScalarExpression ScalarExpression::add_rows(const Tensor& values) const {
  return *this + row_values(values);
}
ScalarExpression ScalarExpression::add_columns(const Tensor& values) const {
  return *this + column_values(values);
}
ScalarExpression ScalarExpression::masked(const ScalarExpression& mask,
                                          at::Scalar value) const {
  return where(mask, value);
}
ScalarExpression ScalarExpression::operator-() const {
  return ScalarExpression(node(8, node_));
}
ScalarExpression ScalarExpression::scale(at::Scalar a) const {
  return *this * ScalarExpression(a);
}
ScalarExpression ScalarExpression::scale(const Tensor& a) const {
  return *this * ScalarExpression(a);
}
ScalarExpression ScalarExpression::shift(at::Scalar a) const {
  return *this + ScalarExpression(a);
}
ScalarExpression ScalarExpression::square() const {
  return ScalarExpression(node(6, node_));
}
ScalarExpression ScalarExpression::exp() const {
  return ScalarExpression(node(7, node_));
}
ScalarExpression ScalarExpression::weight_rows(const Tensor& w) const {
  return ScalarExpression(node(9, node_, {}, w));
}
ScalarExpression ScalarExpression::weight_columns(const Tensor& w) const {
  return ScalarExpression(node(10, node_, {}, w));
}
ScalarExpression ScalarExpression::transpose() const {
  return ScalarExpression(node(11, node_));
}
ExpressionGraph ScalarExpression::graph() const {
  return expression_graph(node_);
}
ScalarExpressionMetadata ScalarExpression::metadata() const {
  at::NoGradGuard guard;
  ExpressionGeometry geometry;
  auto graph = expression_graph(node_, {}, &geometry);
  auto prepared = detail::streaming::prepare_expression(
      graph.sources, graph.nodes, graph.factors, graph.numbers, "diagonal", {});
  auto shape = prepared.batch;
  // Like operands carry batch geometry even when no packet node reads them.
  if (!geometry.shape.empty()) {
    shape =
        at::infer_size_symint(shape, c10::SymIntArrayRef(geometry.shape)
                                         .slice(0, geometry.shape.size() - 2));
  }
  shape.push_back(prepared.queries);
  shape.push_back(prepared.references);
  return {std::move(shape), at::ScalarType(prepared.settings.dtype),
          prepared.values[detail::streaming::X].device()};
}
Tensor ScalarExpression::call(c10::string_view operation,
                              const PairwiseOperationOptions& o,
                              c10::optional<double> correction,
                              const ScalarExpression* weights,
                              const c10::optional<std::vector<int64_t>>& axes,
                              c10::optional<int64_t> k, bool largest,
                              bool sorted, Tensor* selected,
                              c10::string_view interpolation) const {
  auto graph = expression_graph(node_, weights ? weights->node_ : nullptr);
  const auto& sources = graph.sources;
  const auto& nodes = graph.nodes;
  const auto& factors = graph.factors;
  const auto& numbers = graph.numbers;
  auto value_root = graph.value_root, weight_root = graph.weight_root;
  std::vector<std::vector<Tensor>> inputs;
  std::vector<std::string> metrics, families;
  TensorDict tensors;
  RealDict reals;
  IntDict integers;
  BoolDict flags;
  StringDict strings;
  RealLists real_lists;
  IntLists integer_lists;
  c10::List<bool> transposed;
  for (size_t k = 0; k < sources.size(); ++k) {
    const auto& s = sources[k];
    inputs.push_back(s.inputs);
    metrics.push_back(s.metric);
    families.push_back(s.family);
    pack_options(tensors, s.tensors, k);
    pack_options(reals, s.reals, k);
    pack_options(integers, s.integers, k);
    pack_options(flags, s.flags, k);
    pack_options(strings, s.strings, k);
    if (!s.real_list.empty())
      real_lists.insert(std::to_string(k), c10::List<double>(s.real_list));
    if (!s.integer_list.empty())
      integer_lists.insert(std::to_string(k),
                           c10::List<int64_t>(s.integer_list));
    transposed.push_back(s.transposed);
  }
  std::vector<int64_t> program, number_kinds, integer_numbers;
  std::vector<double> real_numbers;
  for (const auto& n : nodes)
    program.insert(program.end(), n.begin(), n.end());
  for (const auto& n : numbers) {
    number_kinds.push_back(n.isBoolean() ? 0 : n.isIntegral(false) ? 1 : 2);
    integer_numbers.push_back(n.isIntegral(true) ? n.toLong() : 0);
    real_numbers.push_back(n.isFloatingPoint() ? n.toDouble() : 0.);
  }
  Tensors source_inputs;
  std::vector<int64_t> counts;
  for (const auto& group : inputs) {
    counts.push_back(group.size());
    source_inputs.insert(source_inputs.end(), group.begin(), group.end());
  }
  if (operation == "median" || operation == "quantile") {
    static auto selection_op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::expression_operations",
                               "selection")
            .typed<decltype(expression_selection_impl)>();
    return selection_op.call(
        source_inputs, counts, metrics, families, tensors, reals, integers,
        flags, strings, real_lists, integer_lists, transposed, program, factors,
        number_kinds, integer_numbers, real_numbers, operation,
        correction.value_or(.5), o.dim.value_or(-1), o.keepdim, o.exclude_self,
        interpolation, o.chunk_size);
  }
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::expression_operations", "flat")
          .typed<decltype(expression_flat_impl)>();
  auto result = op.call(
      source_inputs, counts, metrics, families, tensors, reals, integers, flags,
      strings, real_lists, integer_lists, transposed, program, factors,
      number_kinds, integer_numbers, real_numbers, operation, o.left, o.right,
      o.rows, o.columns, o.reduction, o.dim, o.keepdim, o.exclude_self,
      o.transposed, o.chunk_size, correction, weights ? value_root : -1,
      weight_root, axes.value_or(std::vector<int64_t>{}), bool(axes), k,
      largest, sorted);
  if (selected)
    *selected = std::get<1>(result);
  return std::get<0>(result);
}
std::tuple<Tensor, Tensor> ScalarExpression::topk(int64_t k, int64_t dim,
                                                  bool largest, bool sorted,
                                                  bool exclude_self) const {
  PairwiseOperationOptions o;
  o.dim = dim;
  o.exclude_self = exclude_self;
  Tensor indices;
  auto values = call("reduce", o, c10::nullopt, nullptr, c10::nullopt, k,
                     largest, sorted, &indices);
  return {values, indices};
}
Tensor ScalarExpression::median(int64_t dim, bool keepdim,
                                bool exclude_self) const {
  PairwiseOperationOptions o;
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  return call("median", o, .5, nullptr, c10::nullopt, c10::nullopt, true, true,
              nullptr, "lower");
}
Tensor ScalarExpression::quantile(double q, int64_t dim, bool keepdim,
                                  c10::string_view interpolation,
                                  bool exclude_self) const {
  PairwiseOperationOptions o;
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  return call("quantile", o, q, nullptr, c10::nullopt, c10::nullopt, true, true,
              nullptr, interpolation);
}
Tensor ScalarExpression::matmul(const Tensor& v) const {
  PairwiseOperationOptions o;
  o.right = v;
  return call("matmul", o);
}
Tensor ScalarExpression::softmax_matmul(const Tensor& v,
                                        bool exclude_self) const {
  PairwiseOperationOptions o;
  o.right = v;
  o.exclude_self = exclude_self;
  return call("softmax_matmul", o);
}
Tensor ScalarExpression::bilinear(const Tensor& l, const Tensor& r) const {
  PairwiseOperationOptions o;
  o.left = l;
  o.right = r;
  return call("bilinear", o);
}
Tensor ScalarExpression::quadratic(const Tensor& v) const {
  PairwiseOperationOptions o;
  o.left = v;
  return call("quadratic", o);
}
Tensor ScalarExpression::frobenius_inner_product(const ScalarExpression& other,
                                                 bool center,
                                                 bool normalize) const {
  std::string operation = center ? "centered_" : "";
  operation += normalize ? "alignment" : "frobenius";
  return call(operation, {}, 0., &other);
}
Tensor ScalarExpression::diagonal() const {
  return call("diagonal", {});
}
Tensor ScalarExpression::hsic(const ScalarExpression& other,
                              bool unbiased) const {
  return call(unbiased ? "hsic_unbiased" : "hsic", {}, 0., &other);
}
Tensor ScalarExpression::block(const Tensor& rows,
                               const Tensor& columns) const {
  PairwiseOperationOptions o;
  o.rows = rows;
  o.columns = columns;
  return call("block", o);
}
Tensor ScalarExpression::reduce(c10::string_view reduction,
                                c10::optional<int64_t> dim, bool keepdim,
                                bool exclude_self) const {
  PairwiseOperationOptions o;
  o.reduction = std::string(reduction);
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  if (reduction == "var" || reduction == "std" || reduction == "logsumexp" ||
      reduction == "count_nonzero")
    return statistics(reduction, dim, keepdim, exclude_self);
  return call("reduce", o);
}
Tensor ScalarExpression::reduce(c10::string_view reduction,
                                at::IntArrayRef dims, bool keepdim,
                                bool exclude_self) const {
  if (reduction == "var" || reduction == "std" || reduction == "logsumexp" ||
      reduction == "count_nonzero")
    return statistics(reduction, dims, keepdim, exclude_self);
  PairwiseOperationOptions o;
  o.reduction = std::string(reduction);
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  return call("reduce", o, c10::nullopt, nullptr, dims.vec());
}
Tensor ScalarExpression::statistics(c10::string_view reduction,
                                    at::IntArrayRef dims, bool keepdim,
                                    bool exclude_self,
                                    double correction) const {
  PairwiseOperationOptions o;
  o.reduction = std::string(reduction);
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  return call("reduce", o, correction, nullptr, dims.vec());
}
Tensor ScalarExpression::weighted_statistics(const ScalarExpression& weights,
                                             c10::string_view reduction,
                                             at::IntArrayRef dims, bool keepdim,
                                             bool exclude_self,
                                             double correction) const {
  PairwiseOperationOptions o;
  o.reduction = std::string(reduction);
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  return call("reduce", o, correction, &weights, dims.vec());
}
Tensor ScalarExpression::statistics(c10::string_view reduction,
                                    c10::optional<int64_t> dim, bool keepdim,
                                    bool exclude_self,
                                    double correction) const {
  PairwiseOperationOptions o;
  o.reduction = std::string(reduction);
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  return call("reduce", o, correction);
}
Tensor ScalarExpression::weighted_statistics(const ScalarExpression& weights,
                                             c10::string_view reduction,
                                             c10::optional<int64_t> dim,
                                             bool keepdim, bool exclude_self,
                                             double correction) const {
  PairwiseOperationOptions o;
  o.reduction = std::string(reduction);
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  return call("reduce", o, correction, &weights);
}
Tensor ScalarExpression::weighted_statistics(const Tensor& weights,
                                             c10::string_view reduction,
                                             c10::optional<int64_t> dim,
                                             bool keepdim, bool exclude_self,
                                             double correction) const {
  TORCH_CHECK(dim && (*dim == -1 || *dim == -2),
              "Tensor confidence weights require dim=-1 or -2; wrap pairwise "
              "weights with matrix_expression");
  auto constant = ScalarExpression(at::Scalar(true));
  auto expression = *dim == -1 ? constant.weight_columns(weights)
                               : constant.weight_rows(weights);
  return weighted_statistics(expression, reduction, dim, keepdim, exclude_self,
                             correction);
}
namespace {
ScalarExpression group_mask(const ScalarExpression& like, const Tensor& labels,
                            const c10::optional<Tensor>& other, bool same) {
  auto check_labels = [](const Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.layout() == at::kStrided && tensor.dim() >= 1, name,
                " must be a dense tensor with shape [...,N]");
    TORCH_CHECK(!tensor.is_complex(), name, " must be real or integer labels");
  };
  check_labels(labels, "labels");
  if (other)
    check_labels(*other, "other");
  auto metadata = like.metadata();
  const auto& shape = metadata.shape;
  TORCH_CHECK(shape.size() >= 2, "Masks require a matrix shape provider");
  auto n = shape[shape.size() - 2], m = shape.back();
  TORCH_CHECK(other || n == m,
              "one label tensor requires a square operand; pass labels for "
              "both axes");
  const auto& columns = other ? *other : labels;
  TORCH_CHECK(labels.sym_size(-1) == n, "row labels must have shape [...,", n,
              "]");
  TORCH_CHECK(labels.device() == metadata.device,
              "labels must be on the operand device");
  TORCH_CHECK(columns.sym_size(-1) == m, "column labels must have shape [...,",
              m, "]");
  TORCH_CHECK(columns.device() == metadata.device,
              "labels must be on the operand device");
  auto rows = like.row_values(labels), cols = like.column_values(columns);
  return same ? rows.eq(cols) : rows.ne(cols);
}
}  // namespace
ScalarExpression same_group_mask(const ScalarExpression& like,
                                 const Tensor& labels,
                                 const c10::optional<Tensor>& other) {
  return group_mask(like, labels, other, true);
}
ScalarExpression different_group_mask(const ScalarExpression& like,
                                      const Tensor& labels,
                                      const c10::optional<Tensor>& other) {
  return group_mask(like, labels, other, false);
}
ScalarExpression causal_mask(const ScalarExpression& like, int64_t offset) {
  return (like.column_index() - like.row_index()).le(offset);
}
ScalarExpression band_mask(const ScalarExpression& like, int64_t lower,
                           int64_t upper) {
  // Avoid overflow when checking lower + upper or negating INT64_MIN.
  TORCH_CHECK(lower >= 0 ? upper >= -lower : upper >= 0 && lower >= -upper,
              "band is empty: lower + upper must be nonnegative");
  auto offset = like.column_index() - like.row_index();
  return offset.ge(-lower).logical_and(offset.le(upper));
}
ScalarExpression diagonal_mask(const ScalarExpression& like) {
  return like.column_index().eq(like.row_index());
}
ScalarExpression pairwise_expression(const Tensor& x,
                                     const c10::optional<Tensor>& y,
                                     c10::string_view metric, bool kernel,
                                     const PairwiseOperationOptions& o) {
  auto s = basic(x, y, metric, "vector", o);
  s.flags.insert("is_kernel", kernel);
  return ScalarExpression(std::move(s));
}
ScalarExpression matrix_expression(const Tensor& matrix,
                                   const MatrixOperationOptions& q) {
  PairwiseOperationOptions o;
  o.left = q.left;
  o.right = q.right;
  o.rows = q.rows;
  o.columns = q.columns;
  o.dim = q.dim;
  o.reduction = q.reduction;
  o.keepdim = q.keepdim;
  o.exclude_self = q.exclude_self;
  o.transposed = q.transposed;
  o.chunk_size = q.chunk_size;
  return ScalarExpression(basic(matrix, c10::nullopt, "_matrix", "matrix", o));
}
ScalarExpression structured_expression(
    at::TensorList x, const c10::optional<std::vector<Tensor>>& y,
    c10::string_view metric, const StructuredOperationOptions& q) {
  ExpressionSource s;
  s.inputs = x.vec();
  if (y)
    s.inputs.insert(s.inputs.end(), y->begin(), y->end());
  s.metric = std::string(metric);
  s.family = x.size() == 1   ? "covariance"
             : x.size() == 2 ? "gaussian"
                             : "mixture";
  binding(s, q.matrix);
  put(s.tensors, "x1_lengths", q.x1_lengths);
  put(s.tensors, "x2_lengths", q.x2_lengths);
  put(s.flags, "squared", q.squared);
  put(s.reals, "regularization", q.regularization);
  put(s.integers, "max_iter", q.max_iter);
  put(s.integers, "component_chunk_size", q.component_chunk_size);
  return ScalarExpression(std::move(s));
}
ScalarExpression media_expression(const Tensor& x,
                                  const c10::optional<Tensor>& y,
                                  c10::string_view metric,
                                  const ImageComparisonOptions& q,
                                  const PairwiseOperationOptions& o) {
  auto s = basic(x, y, metric, "image", o);
  if (!o.chunk_size)
    s.integers.insert("chunk_size", q.chunk_size);
  s.reals.insert("data_range", q.data_range);
  s.integers.insert("window_size", q.window_size);
  s.reals.insert("sigma", q.sigma);
  s.reals.insert("k1", q.k1);
  s.reals.insert("k2", q.k2);
  s.flags.insert("weights_set", true);
  s.real_list = q.weights;
  return ScalarExpression(std::move(s));
}
ScalarExpression media_expression(const Tensor& x,
                                  const c10::optional<Tensor>& y,
                                  c10::string_view metric,
                                  const AudioComparisonOptions& q,
                                  const PairwiseOperationOptions& o) {
  auto s = basic(x, y, metric, "audio", o);
  if (!o.chunk_size)
    s.integers.insert("chunk_size", q.chunk_size);
  put(s.flags, "zero_mean", q.zero_mean);
  s.reals.insert("energy_floor", q.energy_floor);
  s.integers.insert("n_fft", q.n_fft);
  put(s.integers, "hop_length", q.hop_length);
  put(s.integers, "win_length", q.win_length);
  s.strings.insert("window", q.window);
  s.flags.insert("center", q.center);
  s.strings.insert("pad_mode", q.pad_mode);
  s.flags.insert("normalized", q.normalized);
  s.reals.insert("log_floor", q.log_floor);
  s.reals.insert("norm_floor", q.norm_floor);
  s.flags.insert("resolutions_set", true);
  s.integer_list = q.resolutions;
  return ScalarExpression(std::move(s));
}
ScalarExpression ordered_expression(const Tensor& x,
                                    const c10::optional<Tensor>& y,
                                    c10::string_view metric,
                                    const SequenceOperatorOptions& q,
                                    const PairwiseOperationOptions& o) {
  auto s = basic(x, y, metric, "sequence", o);
  put(s.tensors, "x1_lengths", q.x1_lengths);
  put(s.tensors, "x2_lengths", q.x2_lengths);
  s.flags.insert("squared", q.squared);
  put(s.reals, "gamma", q.gamma);
  put(s.integers, "window", q.window);
  return ScalarExpression(std::move(s));
}
ScalarExpression ordered_expression(const Tensor& x,
                                    const c10::optional<Tensor>& y,
                                    c10::string_view metric,
                                    const TokenOperatorOptions& q,
                                    const PairwiseOperationOptions& o) {
  auto s = basic(x, y, metric, "tokens", o);
  put(s.tensors, "x1_lengths", q.x1_lengths);
  put(s.tensors, "x2_lengths", q.x2_lengths);
  return ScalarExpression(std::move(s));
}
ScalarExpression ordered_expression(const Tensor& x,
                                    const c10::optional<Tensor>& y,
                                    c10::string_view metric,
                                    const MMDOperatorOptions& q,
                                    const PairwiseOperationOptions& o) {
  auto s = basic(x, y, metric, "mmd", o);
  put(s.tensors, "x1_lengths", q.x1_lengths);
  put(s.tensors, "x2_lengths", q.x2_lengths);
  s.strings.insert("kernel", q.kernel);
  s.flags.insert("squared", q.squared);
  s.reals.insert("length_scale", q.length_scale);
  put(s.reals, "gamma", q.gamma);
  put(s.integers, "degree", q.degree);
  put(s.reals, "coef0", q.coef0);
  put(s.reals, "alpha", q.alpha);
  put(s.reals, "nu", q.nu);
  put(s.flags, "batched", q.batched);
  s.integers.insert("feature_dim", q.feature_dim);
  put(s.strings, "estimator", q.estimator);
  put(s.integers, "metric_chunk_size", q.metric_chunk_size);
  return ScalarExpression(std::move(s));
}
}  // namespace torchpairwise::ops
