#include <ATen/ExpandUtils.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <mutex>
#include <numeric>

#include "custom_comparisons_detail.h"

namespace torchpairwise::ops {
namespace detail::custom {
namespace {
using ForwardSignature = Tensor(const Tensor&, const Tensor&, at::TensorList);
using BackwardSignature = Tensors(const Tensor&, const Tensor&, const Tensor&,
                                  at::TensorList);
template <class Signature>
c10::TypedOperatorHandle<Signature> operation(const std::string& name) {
  // External registrations may be unloaded and recreated under the same name.
  // Unlike our permanent public operators, their handles cannot be cached
  // beyond a call: deregistration invalidates TypedOperatorHandle.
  auto dot = name.find('.');
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(
          name.substr(0, dot).c_str(),
          dot == std::string::npos ? "" : name.substr(dot + 1).c_str())
      .typed<Signature>();
}
std::pair<std::string, std::string> names(c10::string_view encoded) {
  std::string value(encoded);
  auto separator = value.find('|');
  TORCH_CHECK(separator != std::string::npos,
              "Invalid custom comparison descriptor");
  auto end = value.find('|', separator + 1);
  return {value.substr(0, separator),
          value.substr(separator + 1,
                       end == std::string::npos ? end : end - separator - 1)};
}
int64_t broadcast_rank(c10::string_view encoded) {
  std::string value(encoded);
  auto marker = value.find("|broadcast=");
  return marker == std::string::npos ? 0
                                     : std::stoll(value.substr(marker + 11));
}
bool floating(at::ScalarType type) {
  return type == at::kFloat || type == at::kDouble || type == at::kHalf ||
         type == at::kBFloat16;
}
void validate_input(const Tensor& input, int64_t rank) {
  TORCH_CHECK(
      input.layout() == at::kStrided && input.dim() >= rank,
      "Custom inputs must be dense strided collections with sample axes");
  TORCH_CHECK(floating(input.scalar_type()),
              "Custom inputs require real floating point dtypes");
  TORCH_CHECK(input.device().is_cpu() || input.device().is_cuda() ||
                  input.device().is_meta(),
              "Custom comparisons require CPU or CUDA inputs");
}
std::vector<int64_t> shape_values(const Settings& settings) {
  std::vector<int64_t> result;
  for (const auto& value : settings.integers)
    result.push_back(value.value());
  return result;
}
// Flat paired tiles avoid global pair-feature gathers and quadratic index
// arrays.
template <class Visit>
void dense_tiles(at::TensorList v, int64_t mode, int64_t rank, int64_t size,
                 Visit visit) {
  TORCH_CHECK(size > 0 && mode >= 0 && mode <= 2,
              "Invalid custom tiling options");
  auto batch = at::infer_size(v[0].sizes().slice(0, v[0].dim() - rank),
                              v[1].sizes().slice(0, v[1].dim() - rank));
  int64_t batches = std::accumulate(batch.begin(), batch.end(), int64_t(1),
                                    std::multiplies<int64_t>());
  auto n = v[0].size(-rank), m = v[1].size(-rank);
  auto count = mode == 0 ? n * m : mode == 1 ? n : n * (n - 1) / 2;
  Tensor starts;
  if (mode == 2 && count > 0) {
    auto r = at::arange(n, v[0].options().dtype(at::kLong));
    auto factor = 2 * n - r - 1;
    starts = at::floor_divide(r, 2) * factor +
             r.remainder(2) * at::floor_divide(factor, 2);
  }
  for (int64_t b = 0; b < batches; ++b) {
    auto index = streaming::coordinates(b, batch);
    auto x = streaming::lookup(v[0], index, rank);
    auto y = streaming::lookup(v[1], index, rank);
    for (int64_t start = 0; start < count;
         start += std::min(size, count - start)) {
      auto length = std::min(size, count - start);
      auto flat =
          at::arange(start, start + length, v[0].options().dtype(at::kLong));
      Tensor i, j;
      if (mode == 0) {
        i = at::floor_divide(flat, m);
        j = flat.remainder(m);
      } else if (mode == 1) {
        i = j = flat;
      } else {
        i = at::searchsorted(starts, flat, false, true) - 1;
        j = flat - starts.index_select(0, i) + i + 1;
      }
      visit(b, start, index, i, j, x.index_select(0, i), y.index_select(0, j));
    }
  }
}
// Broadcast Cartesian blocks slice each collection independently. No pair
// gather or expanded pair-feature copy is needed in forward or replay.
template <class Visit>
void cartesian_tiles(at::TensorList v, int64_t rank, int64_t size,
                     Visit visit) {
  TORCH_CHECK(size > 0, "Invalid custom tiling options");
  auto batch = at::infer_size(v[0].sizes().slice(0, v[0].dim() - rank),
                              v[1].sizes().slice(0, v[1].dim() - rank));
  int64_t batches = std::accumulate(batch.begin(), batch.end(), int64_t(1),
                                    std::multiplies<int64_t>());
  auto n = v[0].size(-rank), m = v[1].size(-rank);
  if (n == 0 || m == 0)
    return;
  auto rows = std::min(n, std::max(int64_t(1), int64_t(std::sqrt(size))));
  auto columns = std::min(m, std::max(int64_t(1), size / rows));
  for (int64_t b = 0; b < batches; ++b) {
    auto index = streaming::coordinates(b, batch);
    auto x = streaming::lookup(v[0], index, rank);
    auto y = streaming::lookup(v[1], index, rank);
    for (int64_t i = 0; i < n; i += rows)
      for (int64_t j = 0; j < m; j += columns)
        visit(index, i, j, x.narrow(0, i, std::min(rows, n - i)),
              y.narrow(0, j, std::min(columns, m - j)));
  }
}
}  // namespace
std::string operation_names(c10::string_view f, c10::string_view b,
                            int64_t sample_ndim) {
  std::string forward(f), backward(b);
  // Source descriptors use an existing string field: the shared expression
  // source parser remains unchanged. Keep scalar descriptors identical.
  const std::string suffix = "|broadcast";
  bool broadcast = forward.size() >= suffix.size() &&
                   forward.compare(forward.size() - suffix.size(),
                                   suffix.size(), suffix) == 0;
  if (broadcast)
    forward.resize(forward.size() - suffix.size());
  TORCH_CHECK(forward.find("::") != std::string::npos &&
                  forward.find('|') == std::string::npos &&
                  backward.find('|') == std::string::npos,
              "Custom operation names must be qualified dispatcher names");
  operation<ForwardSignature>(forward);
  if (!backward.empty())
    operation<BackwardSignature>(backward);
  return forward + "|" + backward +
         (broadcast ? "|broadcast=" + std::to_string(sample_ndim) : "");
}
Tensors unpack(const Tensor& packed, at::IntArrayRef shapes) {
  TORCH_CHECK(!shapes.empty() && shapes[0] >= 0,
              "Invalid custom parameter metadata");
  Tensors result;
  int64_t cursor = 1, offset = 0;
  for (int64_t k = 0; k < shapes[0]; ++k) {
    TORCH_CHECK(cursor < int64_t(shapes.size()),
                "Invalid custom parameter metadata");
    auto rank = shapes[cursor++];
    TORCH_CHECK(rank >= 0 && cursor + rank <= int64_t(shapes.size()),
                "Invalid parameter rank");
    auto shape = shapes.slice(cursor, rank);
    int64_t count = 1;
    for (auto length : shape) {
      TORCH_CHECK(length >= 0, "Invalid parameter dimension");
      count *= length;
    }
    result.push_back(packed.narrow(0, offset, count).view(shape));
    cursor += rank;
    offset += count;
  }
  TORCH_CHECK(cursor == int64_t(shapes.size()) && offset == packed.numel(),
              "Custom parameter sizes differ");
  return result;
}
Tensor forward(const Tensor& x, const Tensor& y, const Tensor& packed,
               c10::string_view encoded, at::IntArrayRef shapes,
               int64_t dtype) {
  auto value = operation<ForwardSignature>(names(encoded).first)
                   .call(x, y, unpack(packed, shapes));
  auto rank = broadcast_rank(encoded);
  if (rank > 0) {
    auto expected =
        at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - rank),
                              y.sym_sizes().slice(0, y.dim() - rank));
    TORCH_CHECK(value.defined() && value.layout() == at::kStrided,
                "Custom broadcast forward must return a dense strided Tensor");
    TORCH_CHECK(value.sym_sizes() == c10::SymIntArrayRef(expected),
                "Custom broadcast forward output shape must match the "
                "broadcast leading axes");
    TORCH_CHECK(value.device() == x.device(),
                "Custom broadcast forward must return the input device");
    TORCH_CHECK(value.scalar_type() == at::ScalarType(dtype),
                "Custom broadcast forward dtype must match output_dtype");
    return value;
  }
  TORCH_CHECK(value.defined() && value.layout() == at::kStrided &&
                  value.dim() == 1 && value.sym_size(0) == x.sym_size(0) &&
                  value.device() == x.device() &&
                  value.scalar_type() == at::ScalarType(dtype),
              "Custom forward must return a strided [pairs] Tensor of the "
              "declared dtype/device");
  return value;
}
Tensors backward(const Tensor& g, const Tensor& x, const Tensor& y,
                 const Tensor& packed, c10::string_view encoded,
                 at::IntArrayRef shapes) {
  auto parameters = unpack(packed, shapes);
  auto gradients = operation<BackwardSignature>(names(encoded).second)
                       .call(g, x, y, parameters);
  Tensors inputs{x, y};
  inputs.insert(inputs.end(), parameters.begin(), parameters.end());
  TORCH_CHECK(gradients.size() == inputs.size(),
              "Custom backward gradient count mismatch");
  Tensors flat;
  for (size_t k = 0; k < inputs.size(); ++k) {
    const auto& value = gradients[k];
    TORCH_CHECK(value.defined() && value.layout() == at::kStrided &&
                    value.sizes() == inputs[k].sizes() &&
                    value.device() == inputs[k].device() &&
                    value.scalar_type() == inputs[k].scalar_type(),
                "Custom backward gradients must match each input/parameter "
                "shape, dtype and device");
    if (k >= 2)
      flat.push_back(value.reshape({-1}));
  }
  return {gradients[0], gradients[1],
          flat.empty() ? at::empty_like(packed) : at::cat(flat)};
}
Tensor paired(const Tensor& x, const Tensor& y, const Tensor& packed,
              c10::string_view encoded, at::IntArrayRef shapes, int64_t dtype) {
  return names(encoded).second.empty()
             ? forward(x, y, packed, encoded, shapes, dtype)
             : paired_with_vjp(x, y, packed, encoded, shapes, dtype);
}
PreparedComparison prepare(const Tensor& x, const c10::optional<Tensor>& other,
                           at::TensorList parameters, c10::string_view f,
                           c10::string_view b, int64_t sample_ndim,
                           c10::optional<int64_t> output_dtype,
                           const PairwiseOperationOptions& options,
                           bool stream) {
  TORCH_CHECK(sample_ndim > 0, "sample_ndim must be positive");
  int64_t rank = sample_ndim + 1;
  validate_input(x, rank);
  auto y = other.value_or(x);
  validate_input(y, rank);
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Custom inputs must have the same dtype and device");
  auto dtype = at::ScalarType(output_dtype.value_or(int64_t(x.scalar_type())));
  TORCH_CHECK(floating(dtype),
              "Custom output_dtype must be a real floating point dtype");
  TORCH_CHECK(!stream || dtype == at::kFloat || dtype == at::kDouble,
              "Custom streaming currently requires float32/float64 output; "
              "precision is owned by the callback");
  std::vector<int64_t> shapes{int64_t(parameters.size())};
  Tensors flattened;
  for (const auto& p : parameters) {
    TORCH_CHECK(
        p.layout() == at::kStrided && p.device() == x.device() &&
            p.scalar_type() == x.scalar_type(),
        "Custom parameters must be dense tensors with the input dtype/device");
    shapes.push_back(p.dim());
    shapes.insert(shapes.end(), p.sizes().begin(), p.sizes().end());
    flattened.push_back(p.reshape({-1}));
  }
  auto packed =
      flattened.empty() ? at::empty({0}, x.options()) : at::cat(flattened);
  streaming::Integers integers;
  for (auto value : shapes)
    integers.push_back(value);
  auto size = options.chunk_size.value_or(128);
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  Settings s{operation_names(f, b, sample_ndim),
             options.transposed ? "transpose_sum" : "sum",
             false,
             false,
             false,
             false,
             false,
             rank,
             1,
             size,
             int64_t(dtype),
             7,
             {},
             integers,
             {},
             std::string("_custom")};
  Tensors values(streaming::Slots, at::empty({0}, x.options()));
  values[streaming::X] = x;
  values[streaming::Y] = y;
  values[streaming::W] = packed;
  values[streaming::XL] = at::empty({0}, x.options().dtype(at::kLong));
  values[streaming::YL] = values[streaming::XL];
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - rank),
                                     y.sym_sizes().slice(0, y.dim() - rank));
  return {values, s, batch,
          options.transposed ? y.sym_size(-rank) : x.sym_size(-rank),
          options.transposed ? x.sym_size(-rank) : y.sym_size(-rank)};
}
Tensor evaluate(at::TensorList local, const Settings& s, bool corresponding) {
  const auto& x = local[streaming::X];
  const auto& y = local[streaming::Y];
  auto shapes = shape_values(s);
  if (corresponding)
    return paired(x, y, local[streaming::W], s.metric, shapes, s.dtype);
  auto n = x.size(0), m = y.size(0);
  if (broadcast_rank(s.metric) > 0)
    return paired(x.unsqueeze(1), y.unsqueeze(0), local[streaming::W], s.metric,
                  shapes, s.dtype);
  auto xs = x.sizes().vec(), ys = y.sizes().vec();
  xs.insert(xs.begin() + 1, m);
  ys.insert(ys.begin(), n);
  auto xx = x.unsqueeze(1).expand(xs).flatten(0, 1);
  auto yy = y.unsqueeze(0).expand(ys).flatten(0, 1);
  return paired(xx, yy, local[streaming::W], s.metric, shapes, s.dtype)
      .view({n, m});
}
Tensor dense_forward(at::TensorList v, c10::string_view names,
                     at::IntArrayRef shapes, int64_t mode, int64_t rank,
                     int64_t dtype, int64_t size) {
  at::NoGradGuard no_grad;
  TORCH_CHECK(v.size() == 3, "Invalid custom dense operands");
  auto batch = at::infer_size(v[0].sizes().slice(0, v[0].dim() - rank),
                              v[1].sizes().slice(0, v[1].dim() - rank));
  auto n = v[0].size(-rank), m = v[1].size(-rank);
  int64_t count = mode == 0 ? n * m : mode == 1 ? n : n * (n - 1) / 2;
  auto shape = batch;
  if (mode == 0)
    shape.insert(shape.end(), {n, m});
  else
    shape.push_back(count);
  auto result = at::empty(shape, v[0].options().dtype(at::ScalarType(dtype)));
  if (mode == 0 && broadcast_rank(names) > 0) {
    cartesian_tiles(v, rank, size,
                    [&](const auto& index, int64_t i, int64_t j,
                        const Tensor& x, const Tensor& y) {
                      streaming::lookup(result, index, 2)
                          .narrow(0, i, x.size(0))
                          .narrow(1, j, y.size(0))
                          .copy_(paired(x.unsqueeze(1), y.unsqueeze(0), v[2],
                                        names, shapes, dtype));
                    });
    return result;
  }
  auto flat = result.reshape({-1});
  dense_tiles(v, mode, rank, size,
              [&](int64_t b, int64_t start, const auto&, const auto&,
                  const auto&, const Tensor& x, const Tensor& y) {
                flat.narrow(0, b * count + start, x.size(0))
                    .copy_(paired(x, y, v[2], names, shapes, dtype));
              });
  return result;
}
Tensors dense_backward(const Tensor& gradient, at::TensorList v,
                       c10::string_view names, at::IntArrayRef shapes,
                       int64_t mode, int64_t rank, int64_t dtype,
                       int64_t size) {
  Tensors result;
  for (const auto& input : v)
    result.push_back(at::zeros_like(input, input.options().dtype(at::kDouble)));
  if (mode == 0 && broadcast_rank(names) > 0) {
    cartesian_tiles(
        v, rank, size,
        [&](const auto& index, int64_t i, int64_t j, const Tensor& xx,
            const Tensor& yy) {
          at::AutoGradMode enable(true);
          auto x = xx.detach().set_requires_grad(true);
          auto y = yy.detach().set_requires_grad(true);
          auto p = v[2].detach().set_requires_grad(true);
          auto value =
              paired(x.unsqueeze(1), y.unsqueeze(0), p, names, shapes, dtype);
          if (!value.requires_grad())
            return;
          auto g = streaming::lookup(gradient, index, 2)
                       .narrow(0, i, x.size(0))
                       .narrow(1, j, y.size(0));
          auto grads = torch::autograd::grad({value}, {x, y, p}, {g}, false,
                                             false, true);
          at::NoGradGuard guard;
          for (int64_t k = 0; k < 2; ++k)
            if (grads[k].defined())
              streaming::lookup(result[k], index, rank)
                  .narrow(0, k == 0 ? i : j, k == 0 ? x.size(0) : y.size(0))
                  .add_(grads[k].to(at::kDouble));
          if (grads[2].defined())
            result[2].add_(grads[2].to(at::kDouble));
        });
    for (size_t k = 0; k < result.size(); ++k)
      result[k] = result[k].to(v[k].scalar_type());
    return result;
  }
  auto n = v[0].size(-rank), m = v[1].size(-rank);
  int64_t count = mode == 0 ? n * m : mode == 1 ? n : n * (n - 1) / 2;
  auto flat = gradient.reshape({-1});
  dense_tiles(
      v, mode, rank, size,
      [&](int64_t b, int64_t start, const auto& index, const Tensor& i,
          const Tensor& j, const Tensor& xx, const Tensor& yy) {
        at::AutoGradMode enable(true);
        auto x = xx.detach().set_requires_grad(true),
             y = yy.detach().set_requires_grad(true);
        auto p = v[2].detach().set_requires_grad(true);
        auto value = paired(x, y, p, names, shapes, dtype);
        if (!value.requires_grad())
          return;
        auto grads = torch::autograd::grad(
            {value}, {x, y, p}, {flat.narrow(0, b * count + start, x.size(0))},
            false, false, true);
        at::NoGradGuard guard;
        for (int64_t k = 0; k < 2; ++k)
          if (grads[k].defined())
            streaming::lookup(result[k], index, rank)
                .index_add_(0, k == 0 ? i : j, grads[k].to(at::kDouble));
        if (grads[2].defined())
          result[2].add_(grads[2].to(at::kDouble));
      });
  for (size_t k = 0; k < result.size(); ++k)
    result[k] = result[k].to(v[k].scalar_type());
  return result;
}
}  // namespace detail::custom
namespace {
at::Tensor custom_compare_impl(
    const at::Tensor& input, const c10::optional<at::Tensor>& other,
    at::TensorList parameters, c10::string_view forward,
    c10::string_view backward, c10::string_view mode, int64_t sample_ndim,
    c10::optional<int64_t> dtype, c10::optional<int64_t> chunk_size,
    bool symmetric, bool broadcast) {
  TORCH_CHECK(mode == "all_pairs" || mode == "paired" || mode == "condensed" ||
                  mode == "single",
              "Unknown custom comparison mode");
  TORCH_CHECK((mode != "paired" && mode != "single") || other,
              "Paired/single custom comparisons require both inputs");
  TORCH_CHECK(mode != "condensed" || (!other && symmetric),
              "Condensed requires one input and explicit symmetry");
  TORCH_CHECK(sample_ndim > 0, "sample_ndim must be positive");
  bool single = mode == "single";
  auto x = single ? input.unsqueeze(-sample_ndim - 1) : input;
  auto y = other ? c10::optional<at::Tensor>(
                       single ? other->unsqueeze(-sample_ndim - 1) : *other)
                 : c10::nullopt;
  PairwiseOperationOptions options;
  options.chunk_size = chunk_size.value_or(1024);
  auto descriptor = std::string(forward) + (broadcast ? "|broadcast" : "");
  auto p = detail::custom::prepare(x, y, parameters, descriptor, backward,
                                   sample_ndim, dtype, options, false);
  TORCH_CHECK((mode != "paired" && !single) || p.queries == p.references,
              "Paired sample counts must match");
  std::vector<int64_t> shapes;
  for (auto value : p.settings.integers)
    shapes.push_back(value.value());
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_custom_compare", "")
                       .typed<detail::custom::DenseSignature>();
  auto result = op.call({p.values[0], p.values[1], p.values[2]},
                        p.settings.metric, shapes,
                        mode == "all_pairs"   ? 0
                        : mode == "condensed" ? 2
                                              : 1,
                        sample_ndim + 1, p.settings.dtype, p.settings.size);
  return single ? result.squeeze(-1) : result;
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "custom_compare(Tensor x, Tensor? y, Tensor[] parameters, str forward, "
      "str backward='', str mode='all_pairs', int sample_ndim=1, int? "
      "output_dtype=None, int? chunk_size=None, bool symmetric=False, "
      "bool broadcast=False) -> "
      "Tensor");
  m.def(
      "_custom_compare(Tensor[] inputs, str names, int[] shapes, int mode, int "
      "rank, int dtype, int size) -> Tensor");
  m.def(
      "_custom_compare_backward(Tensor gradient, Tensor[] inputs, str names, "
      "int[] shapes, int mode, int rank, int dtype, int size) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeImplicitAutograd, m) {
  m.impl("custom_compare", TORCH_FN(custom_compare_impl));
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_custom_compare", TORCH_FN(detail::custom::dense_forward));
  m.impl("_custom_compare_backward", TORCH_FN(detail::custom::dense_backward));
}
}  // namespace
at::Tensor custom_compare(const at::Tensor& x,
                          const c10::optional<at::Tensor>& y,
                          at::TensorList parameters, c10::string_view forward,
                          c10::string_view backward, c10::string_view mode,
                          int64_t sample_ndim,
                          c10::optional<int64_t> output_dtype,
                          c10::optional<int64_t> chunk_size, bool symmetric,
                          bool broadcast) {
  using Signature = at::Tensor(
      const at::Tensor&, const c10::optional<at::Tensor>&, at::TensorList,
      c10::string_view, c10::string_view, c10::string_view, int64_t,
      c10::optional<int64_t>, c10::optional<int64_t>, bool, bool);
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::custom_compare", "")
                       .typed<Signature>();
  return op.call(x, y, parameters, forward, backward, mode, sample_ndim,
                 output_dtype, chunk_size, symmetric, broadcast);
}
ScalarExpression custom_expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view forward, c10::string_view backward,
    at::TensorList parameters, int64_t sample_ndim,
    c10::optional<int64_t> output_dtype,
    const PairwiseOperationOptions& options, bool broadcast) {
  ExpressionSource source;
  source.inputs = {x};
  if (y)
    source.inputs.push_back(*y);
  source.metric = std::string(forward) + (broadcast ? "|broadcast" : "");
  source.family = "custom";
  source.transposed = options.transposed;
  source.strings.insert("backward", std::string(backward));
  source.integers.insert("sample_ndim", sample_ndim);
  source.integers.insert("parameter_count", parameters.size());
  source.integers.insert("output_dtype",
                         output_dtype.value_or(int64_t(x.scalar_type())));
  if (options.chunk_size)
    source.integers.insert("chunk_size", *options.chunk_size);
  for (size_t k = 0; k < parameters.size(); ++k)
    source.tensors.insert("parameter_" + std::to_string(k), parameters[k]);
  return ScalarExpression(std::move(source));
}

namespace {
struct CallableRegistration {
  torch::Library definitions{torch::Library::FRAGMENT, "torchpairwise_callable",
                             c10::nullopt, __FILE__, __LINE__};
  torch::Library implementations{torch::Library::IMPL, "torchpairwise_callable",
                                 c10::DispatchKey::CompositeImplicitAutograd,
                                 __FILE__, __LINE__};
};

CustomComparisonForward scalar_forward(CustomComparisonForward forward,
                                       int64_t output_dtype) {
  return [forward = std::move(forward), output_dtype](
             const at::Tensor& x, const at::Tensor& y,
             at::TensorList parameters) {
    std::vector<at::Tensor> values;
    values.reserve(x.size(0));
    for (int64_t p = 0; p < x.size(0); ++p) {
      auto value = forward(x[p], y[p], parameters);
      TORCH_CHECK(value.defined() && value.dim() == 0,
                  "Scalar custom forward must return a scalar Tensor");
      values.push_back(std::move(value));
    }
    return values.empty()
               ? at::empty(
                     {0}, x.options().dtype(output_dtype < 0
                                                ? x.scalar_type()
                                                : at::ScalarType(output_dtype)))
               : at::stack(values);
  };
}

CustomComparisonBackward scalar_backward(CustomComparisonBackward backward) {
  return [backward = std::move(backward)](
             const at::Tensor& gradient, const at::Tensor& x,
             const at::Tensor& y, at::TensorList parameters) {
    std::vector<at::Tensor> dx, dy, parameter_gradients;
    for (const auto& parameter : parameters)
      parameter_gradients.push_back(at::zeros_like(parameter));
    dx.reserve(x.size(0));
    dy.reserve(y.size(0));
    for (int64_t p = 0; p < x.size(0); ++p) {
      auto a = x[p], b = y[p];
      auto values = backward(gradient[p], a, b, parameters);
      TORCH_CHECK(values.size() == parameters.size() + 2,
                  "Custom backward must return one gradient per operand");
      for (size_t k = 0; k < values.size(); ++k) {
        const auto& input = k == 0 ? a : k == 1 ? b : parameters[k - 2];
        const auto& value = values[k];
        TORCH_CHECK(value.defined() && value.layout() == at::kStrided &&
                        value.sizes() == input.sizes() &&
                        value.scalar_type() == input.scalar_type() &&
                        value.device() == input.device(),
                    "Scalar custom gradients must match each operand");
      }
      dx.push_back(values[0]);
      dy.push_back(values[1]);
      for (size_t k = 0; k < parameters.size(); ++k)
        parameter_gradients[k] = parameter_gradients[k] + values[k + 2];
    }
    std::vector<at::Tensor> result{
        dx.empty() ? at::empty_like(x) : at::stack(dx),
        dy.empty() ? at::empty_like(y) : at::stack(dy)};
    result.insert(result.end(), parameter_gradients.begin(),
                  parameter_gradients.end());
    return result;
  };
}
}  // namespace

CustomComparison::CustomComparison(std::string forward, std::string backward,
                                   CustomComparisonOptions options)
    : forward_(std::move(forward)),
      backward_(std::move(backward)),
      options_(std::move(options)) {}

at::Tensor CustomComparison::operator()(
    const at::Tensor& x, const c10::optional<at::Tensor>& y) const {
  return compare(x, y, options_.mode);
}

at::Tensor CustomComparison::compare(const at::Tensor& x,
                                     const c10::optional<at::Tensor>& y,
                                     c10::string_view mode) const {
  return custom_compare(x, y, options_.parameters, forward_, backward_, mode,
                        options_.sample_ndim, options_.output_dtype,
                        options_.chunk_size, options_.symmetric,
                        options_.broadcast);
}

ScalarExpression CustomComparison::expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    const PairwiseOperationOptions& options) const {
  return custom_expression(x, y, forward_, backward_, options_.parameters,
                           options_.sample_ndim, options_.output_dtype, options,
                           options_.broadcast);
}

CustomComparison make_pairwise(CustomComparisonForward forward,
                               CustomComparisonBackward backward,
                               CustomComparisonOptions options) {
  TORCH_CHECK(forward, "Custom forward must be callable");
  TORCH_CHECK(options.sample_ndim > 0, "sample_ndim must be positive");
  TORCH_CHECK(!options.chunk_size || *options.chunk_size > 0,
              "chunk_size must be positive");
  TORCH_CHECK(options.mode == "all_pairs" || options.mode == "paired" ||
                  options.mode == "condensed" || options.mode == "single",
              "Invalid custom comparison mode");
  TORCH_CHECK(options.mode != "condensed" || options.symmetric,
              "Condensed mode requires explicit symmetry");
  if (!options.vectorized && !options.broadcast) {
    forward =
        scalar_forward(std::move(forward), options.output_dtype.value_or(-1));
    if (backward)
      backward = scalar_backward(std::move(backward));
  }
  // Own callbacks independently of factory handles and deferred autograd
  // graphs. IDs are unique in this process; they are not serialization names.
  static std::mutex mutex;
  static std::vector<std::unique_ptr<CallableRegistration>> registrations;
  std::lock_guard<std::mutex> lock(mutex);
  auto name = "metric_" + std::to_string(registrations.size());
  auto registration = std::make_unique<CallableRegistration>();
  auto forward_schema =
      name + "(Tensor x, Tensor y, Tensor[] parameters) -> Tensor";
  registration->definitions.def(forward_schema.c_str());
  registration->implementations.impl(name.c_str(), std::move(forward));
  std::string backward_name;
  if (backward) {
    backward_name = name + "_vjp";
    auto backward_schema =
        backward_name +
        "(Tensor grad, Tensor x, Tensor y, Tensor[] parameters) -> Tensor[]";
    registration->definitions.def(backward_schema.c_str());
    registration->implementations.impl(backward_name.c_str(),
                                       std::move(backward));
    backward_name = "torchpairwise_callable::" + backward_name;
  }
  registrations.push_back(std::move(registration));
  return CustomComparison("torchpairwise_callable::" + name,
                          std::move(backward_name), std::move(options));
}
}  // namespace torchpairwise::ops
