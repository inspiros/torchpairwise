#include "pointset_comparisons.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <array>

#include "hausdorff.h"
#include "pointset_comparisons_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
at::Tensor pointset_comparisons_composite(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view metric, c10::string_view mode,
    const PointsetComparisonOptions& options);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
directed_pointset_comparisons_composite(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view mode, const PointsetComparisonOptions& options);
}  // namespace

namespace {
using at::Tensor;
using Triple = std::tuple<Tensor, Tensor, Tensor>;
using Pair = std::tuple<Tensor, Tensor>;
using Values = torch::autograd::variable_list;
enum Kind : int64_t { Chamfer, Hausdorff, Directed };
Tensor root(const Tensor& x) {
  auto nonzero = x.ne(0);
  return at::where(nonzero, at::where(nonzero, x, 1.).sqrt(), 0.);
}
Tensor no_indices(const Tensor& x) {
  return at::empty({0}, x.options().dtype(at::kLong));
}

Triple block(Tensor x, Tensor y, Tensor xl, Tensor yl, int64_t kind,
             bool paired, bool squared, bool legacy, bool shuffle = false,
             c10::optional<at::Generator> generator = c10::nullopt) {
  const auto B = x.size(0), N = x.size(1);
  if (paired) {
    x = x.reshape({B * N, 1, x.size(2), x.size(3)});
    y = y.reshape({B * N, 1, y.size(2), y.size(3)});
    xl = xl.reshape({B * N, 1});
    yl = yl.reshape({B * N, 1});
  }
  Tensor value, xi = no_indices(x), yi = no_indices(x);
  if (legacy) {
    auto forward = _directed_hausdorff(x, y, shuffle, generator);
    value = std::get<0>(forward);
    if (kind == Directed) {
      xi = std::get<1>(forward);
      yi = std::get<2>(forward);
    } else
      value = at::maximum(
          value, std::get<0>(_directed_hausdorff(y, x, shuffle, generator))
                     .transpose(-1, -2));
  } else {
    using Signature =
        Pair(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
    static auto nearest =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchpairwise::_pointset_nn", "")
            .typed<Signature>();
    auto forward = nearest.call(x, y, xl, yl);
    auto distances = std::get<0>(forward);
    if (kind == Directed) {
      auto maximum = distances.max(-1);
      value = root(std::get<0>(maximum));
      xi = std::get<1>(maximum);
      yi = std::get<1>(forward).gather(-1, xi.unsqueeze(-1)).squeeze(-1);
    } else {
      auto reverse = std::get<0>(nearest.call(y, x, yl, xl));
      if (kind == Hausdorff)
        value =
            root(at::maximum(std::get<0>(distances.max(-1)),
                             std::get<0>(reverse.max(-1)).transpose(-1, -2)));
      else {
        if (!squared) {
          distances = root(distances);
          reverse = root(reverse);
        }
        value = distances.sum(-1) / xl.unsqueeze(-1) +
                (reverse.sum(-1) / yl.unsqueeze(-1)).transpose(-1, -2);
      }
    }
  }
  if (paired) {
    value = value.reshape({B, N});
    if (kind == Directed) {
      xi = xi.reshape({B, N});
      yi = yi.reshape({B, N});
    }
  }
  return {value, xi, yi};
}
std::pair<Tensor, Tensor> lengths(const Tensor& x, const Tensor& y,
                                  const Tensor& xl, const Tensor& yl) {
  return {xl.numel() ? xl
                     : at::full({x.size(0), x.size(1)}, x.size(2),
                                x.options().dtype(at::kLong)),
          yl.numel() ? yl
                     : at::full({y.size(0), y.size(1)}, y.size(2),
                                y.options().dtype(at::kLong))};
}
Triple assemble(const Tensor& x, const Tensor& y, const Tensor& raw_xl,
                const Tensor& raw_yl, int64_t kind, int64_t mode, bool squared,
                int64_t size, bool legacy, bool shuffle = false,
                c10::optional<at::Generator> generator = c10::nullopt) {
  const auto B = x.size(0), N = x.size(1), M = y.size(1);
  std::vector<int64_t> shape{B, mode == 2 ? N * (N - 1) / 2 : N};
  if (mode == 0)
    shape.push_back(M);
  if (B == 0 || N == 0 || M == 0 || (mode == 2 && N < 2)) {
    auto indices = kind == Directed
                       ? at::empty(shape, x.options().dtype(at::kLong))
                       : no_indices(x);
    return {at::empty(shape, x.options()), indices, indices.clone()};
  }
  auto lens = lengths(x, y, raw_xl, raw_yl);
  auto xl = lens.first, yl = lens.second;
  std::array<std::vector<Tensor>, 3> rows;
  for (int64_t i = 0; i < N; i += mode == 2 ? 1 : size) {
    const auto count = std::min(mode == 2 ? int64_t(1) : size, N - i);
    std::array<std::vector<Tensor>, 3> columns;
    for (int64_t j = mode == 2 ? i + 1 : (mode == 1 ? i : 0);
         j < (mode == 1 ? i + 1 : M); j += size) {
      auto other = mode == 1 ? count : std::min(size, M - j);
      auto tile = block(x.narrow(1, i, count), y.narrow(1, j, other),
                        xl.narrow(1, i, count), yl.narrow(1, j, other), kind,
                        mode == 1, squared, legacy, shuffle, generator);
      columns[0].push_back(std::get<0>(tile));
      if (kind == Directed) {
        columns[1].push_back(std::get<1>(tile));
        columns[2].push_back(std::get<2>(tile));
      }
    }
    if (columns[0].empty())
      continue;
    for (int64_t k = 0; k < (kind == Directed ? 3 : 1); ++k) {
      auto row = at::cat(columns[k], -1);
      rows[k].push_back(mode == 2 ? row.squeeze(-2) : row);
    }
  }
  auto value = at::cat(rows[0], mode == 0 ? -2 : -1);
  return {
      value,
      kind == Directed ? at::cat(rows[1], mode == 0 ? -2 : -1) : no_indices(x),
      kind == Directed ? at::cat(rows[2], mode == 0 ? -2 : -1) : no_indices(x)};
}
Triple pointset_compare_kernel(const Tensor& x, const Tensor& y,
                               const Tensor& xl, const Tensor& yl, int64_t kind,
                               int64_t mode, bool squared, int64_t size,
                               bool legacy) {
  if (legacy && mode == 0 && size == 0) {
    auto lens = lengths(x, y, xl, yl);
    return block(x, y, lens.first, lens.second, kind, false, false, true);
  }
  return assemble(x, y, xl, yl, kind, mode, squared, size, legacy);
}
using CoreSignature = Triple(const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, int64_t, int64_t, bool, int64_t,
                             bool);
using GradSignature = Pair(const Tensor&, const Tensor&, const Tensor&,
                           const Tensor&, const Tensor&, int64_t, int64_t, bool,
                           int64_t, bool);

Pair pointset_compare_backward_kernel(const Tensor& gradient, const Tensor& x,
                                      const Tensor& y, const Tensor& raw_xl,
                                      const Tensor& raw_yl, int64_t kind,
                                      int64_t mode, bool squared, int64_t size,
                                      bool legacy) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda_autocast(
      c10::DispatchKey::AutocastCUDA);
  auto gx = at::zeros(x.sizes(), x.options().dtype(at::kDouble));
  auto gy = at::zeros(y.sizes(), y.options().dtype(at::kDouble));
  if (gradient.numel() == 0)
    return {gx.to(x.scalar_type()), gy.to(y.scalar_type())};
  auto lens = lengths(x, y, raw_xl, raw_yl);
  auto xl = lens.first, yl = lens.second;
  if (size == 0)
    size = 16;
  auto N = x.size(1), M = y.size(1);
  int64_t offset = 0;
  at::AutoGradMode enable(true);
  for (int64_t i = 0; i < N; i += mode == 2 ? 1 : size) {
    auto count = std::min(mode == 2 ? int64_t(1) : size, N - i);
    for (int64_t j = mode == 2 ? i + 1 : (mode == 1 ? i : 0);
         j < (mode == 1 ? i + 1 : M); j += size) {
      auto other = mode == 1 ? count : std::min(size, M - j);
      auto a = x.narrow(1, i, count).detach().set_requires_grad(true);
      auto b = y.narrow(1, j, other).detach().set_requires_grad(true);
      auto value = std::get<0>(block(a, b, xl.narrow(1, i, count),
                                     yl.narrow(1, j, other), kind, mode == 1,
                                     squared, legacy));
      Tensor g;
      if (mode == 0)
        g = gradient.narrow(-2, i, count).narrow(-1, j, other);
      else if (mode == 1)
        g = gradient.narrow(-1, i, count);
      else
        g = gradient.narrow(-1, offset + j - i - 1, other).unsqueeze(-2);
      auto grads =
          torch::autograd::grad({value}, {a, b}, {g}, false, false, true);
      if (grads[0].defined())
        gx.narrow(1, i, count).add_(grads[0].to(at::kDouble));
      if (grads[1].defined())
        gy.narrow(1, j, other).add_(grads[1].to(at::kDouble));
    }
    if (mode == 2)
      offset += N - i - 1;
  }
  return {gx.to(x.scalar_type()), gy.to(y.scalar_type())};
}

Tensor layout(const Tensor& x, c10::optional<bool> batched,
              int64_t feature_dim) {
  TORCH_CHECK(x.layout() == at::kStrided,
              "Point-set inputs must be dense strided tensors");
  TORCH_CHECK(x.dim() >= (batched.value_or(false) ? 4 : 3),
              "Point-set inputs have insufficient rank");
  TORCH_CHECK(feature_dim >= -x.dim() && feature_dim < x.dim(),
              "feature_dim is out of range");
  auto axis = (feature_dim + x.dim()) % x.dim();
  auto sample = batched.has_value() ? int64_t(*batched) : x.dim() - 3;
  TORCH_CHECK(axis > sample, "feature_dim must follow batch and sample axes");
  auto value = x.movedim(axis, -1);
  auto shape = value.sym_sizes().slice(0, sample + 1).vec();
  c10::SymInt points = 1;
  for (int64_t i = sample + 1; i < value.dim() - 1; ++i)
    points *= value.sym_size(i);
  shape.push_back(points);
  shape.push_back(value.sym_size(-1));
  return value.reshape_symint(shape);
}
int64_t kind_for(c10::string_view name) {
  if (name == "chamfer" || name == "chamfer_distances")
    return Chamfer;
  if (name == "hausdorff" || name == "hausdorff_distances")
    return Hausdorff;
  if (name == "directed_hausdorff" || name == "directed_hausdorff_distances")
    return Directed;
  TORCH_CHECK(false, "Unsupported native point-set comparison: ", name);
}
Triple compare(const Tensor& input, const c10::optional<Tensor>& other,
               int64_t kind, c10::string_view mode,
               const PointsetComparisonOptions& opts) {
  int64_t pairing = mode == "all_pairs"   ? 0
                    : mode == "paired"    ? 1
                    : mode == "condensed" ? 2
                                          : -1;
  TORCH_CHECK(pairing >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(pairing != 2 || (!other.has_value() && kind != Directed),
              "Condensed comparisons require one input and a symmetric metric");
  TORCH_CHECK(pairing != 2 || !opts.x2_lengths.has_value(),
              "Condensed point sets accept x1_lengths only");
  TORCH_CHECK(!opts.squared.has_value() || kind == Chamfer,
              "Unused parameter: squared");
  TORCH_CHECK((!opts.shuffle.has_value() && !opts.generator.has_value()) ||
                  kind != Chamfer,
              "Unused parameter: shuffle or generator");
  TORCH_CHECK(!opts.chunk_size.has_value() || *opts.chunk_size > 0,
              "chunk_size must be positive");
  bool randomized = opts.shuffle.value_or(false) || opts.generator.has_value();
  TORCH_CHECK(!opts.chunk_size.has_value() || !randomized,
              "chunk_size does not support shuffle or generator");
  bool legacy = kind != Chamfer && !opts.x1_lengths.has_value() &&
                !opts.x2_lengths.has_value();
  TORCH_CHECK(legacy || !randomized,
              "Shuffling is supported only for Hausdorff without lengths");
  auto x = layout(input, opts.batched, opts.feature_dim),
       y = other.has_value() ? layout(*other, opts.batched, opts.feature_dim)
                             : x;
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Inputs must share dtype and device");
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Point-set comparisons support CPU, CUDA and metadata execution");
  if (!x.device().is_meta() &&
      at::autocast::is_autocast_enabled(x.device().type()) &&
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16)) {
    x = x.to(at::kFloat);
    y = y.to(at::kFloat);
  }
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda_autocast(
      c10::DispatchKey::AutocastCUDA);
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "Point-set inputs require float32 or float64 outside autocast");
  TORCH_CHECK(x.sym_size(-1) > 0 && x.sym_size(-1) == y.sym_size(-1),
              "Input features must match and be non-empty");
  TORCH_CHECK(x.sym_size(-2) > 0 && y.sym_size(-2) > 0,
              "Input point sets must be non-empty");
  TORCH_CHECK(pairing != 1 || x.sym_size(-3) == y.sym_size(-3),
              "Paired inputs must have equal sample counts");
  auto xl = opts.x1_lengths, yl = opts.x2_lengths;
  if (!other.has_value() && !yl.has_value())
    yl = xl;
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 3),
                                     y.sym_sizes().slice(0, y.dim() - 3));
  auto rank = std::max(x.dim(), y.dim()) - 3;
  for (auto item : {std::make_pair(x, xl), std::make_pair(y, yl)})
    if (item.second.has_value()) {
      const auto& len = *item.second;
      TORCH_CHECK(len.layout() == at::kStrided &&
                      len.scalar_type() == at::kLong &&
                      len.device() == x.device(),
                  "Lengths must be dense int64 tensors on the input device");
      TORCH_CHECK(len.dim() >= 1 && len.sym_size(-1) == item.first.sym_size(-3),
                  "Lengths must end with the sample count");
      TORCH_CHECK(len.dim() - 1 <= rank,
                  "Lengths cannot introduce additional batch axes");
      utils::check_values(
          (len.ge(1) & len.le(item.first.sym_size(-2))).all(),
          "Lengths must be between 1 and the padded point count");
      batch =
          at::infer_size_symint(batch, len.sym_sizes().slice(0, len.dim() - 1));
    }
  c10::SymInt B = 1;
  for (auto& dimension : batch)
    B *= dimension;
  auto normalize = [&](const Tensor& value) {
    auto shape = batch;
    for (int64_t i = -3; i < 0; ++i)
      shape.push_back(value.sym_size(i));
    return value.expand_symint(shape).reshape_symint(
        {B, value.sym_size(-3), value.sym_size(-2), value.sym_size(-1)});
  };
  auto normalize_lengths = [&](const c10::optional<Tensor>& len,
                               const Tensor& value) {
    if (!len.has_value())
      return no_indices(value);
    auto shape = batch;
    shape.push_back(value.sym_size(-3));
    return len->expand_symint(shape).reshape_symint({B, value.sym_size(-3)});
  };
  auto al = normalize_lengths(xl, x), bl = normalize_lengths(yl, y);
  auto a = normalize(x), b = normalize(y);
  Triple result;
  if (legacy && randomized && pairing == 0 && !opts.chunk_size.has_value()) {
    // Preserve the existing early-exit search and its exact witnesses.
    result =
        _directed_hausdorff(a, b, opts.shuffle.value_or(false), opts.generator);
    if (kind == Hausdorff)
      result = {at::maximum(std::get<0>(result),
                            std::get<0>(_directed_hausdorff(
                                            b, a, opts.shuffle.value_or(false),
                                            opts.generator))
                                .transpose(-1, -2)),
                no_indices(x), no_indices(x)};
  } else if (randomized) {
    // Keep forward witnesses in native autograd; never replay the RNG.
    result = assemble(a, b, al, bl, kind, pairing, false, 16, true,
                      opts.shuffle.value_or(false), opts.generator);
  } else
    result = detail::pointset::_pointset_compare(
        a, b, al, bl, kind, pairing, opts.squared.value_or(false),
        opts.chunk_size.value_or(legacy && pairing == 0 ? 0 : 16), legacy);
  auto shape = batch;
  auto N = x.sym_size(-3);
  shape.push_back(pairing == 2 ? N * (N - 1) / 2 : N);
  if (pairing == 0)
    shape.push_back(y.sym_size(-3));
  return {std::get<0>(result).reshape_symint(shape),
          kind == Directed ? std::get<1>(result).reshape_symint(shape)
                           : no_indices(x),
          kind == Directed ? std::get<2>(result).reshape_symint(shape)
                           : no_indices(x)};
}
}  // namespace

Tensor pointset_comparisons(const Tensor& x, const c10::optional<Tensor>& y,
                            c10::string_view metric, c10::string_view mode,
                            const PointsetComparisonOptions& options) {
  using Signature = Tensor(
      const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
      c10::string_view mode, c10::optional<bool> squared,
      const c10::optional<Tensor>& xl, const c10::optional<Tensor>& yl,
      c10::optional<int64_t> size, c10::optional<bool> batched,
      int64_t feature_dim, c10::optional<bool> shuffle,
      c10::optional<at::Generator> generator);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::pointset_comparisons", "")
          .typed<Signature>();
  return op.call(x, y, metric, mode, options.squared, options.x1_lengths,
                 options.x2_lengths, options.chunk_size, options.batched,
                 options.feature_dim, options.shuffle, options.generator);
}
Triple directed_pointset_comparisons(const Tensor& x,
                                     const c10::optional<Tensor>& y,
                                     c10::string_view mode,
                                     const PointsetComparisonOptions& options) {
  TORCH_CHECK(!options.squared.has_value(), "Unused parameter: squared");
  using Signature =
      Triple(const Tensor& x, const c10::optional<Tensor>& y,
             c10::string_view mode, const c10::optional<Tensor>& xl,
             const c10::optional<Tensor>& yl, c10::optional<int64_t> size,
             c10::optional<bool> batched, int64_t feature_dim, bool shuffle,
             c10::optional<at::Generator> generator);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::directed_pointset_comparisons", "")
          .typed<Signature>();
  return op.call(x, y, mode, options.x1_lengths, options.x2_lengths,
                 options.chunk_size, options.batched, options.feature_dim,
                 options.shuffle.value_or(false), options.generator);
}
namespace {
Tensor pointset_comparisons_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view mode, c10::optional<bool> squared,
    const c10::optional<Tensor>& xl, const c10::optional<Tensor>& yl,
    c10::optional<int64_t> size, c10::optional<bool> batched,
    int64_t feature_dim, c10::optional<bool> shuffle,
    c10::optional<at::Generator> generator) {
  return pointset_comparisons_composite(
      x, y, metric, mode,
      {squared, xl, yl, size, batched, feature_dim, shuffle, generator});
}
Triple directed_pointset_comparisons_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view mode,
    const c10::optional<Tensor>& xl, const c10::optional<Tensor>& yl,
    c10::optional<int64_t> size, c10::optional<bool> batched,
    int64_t feature_dim, bool shuffle, c10::optional<at::Generator> generator) {
  return directed_pointset_comparisons_composite(
      x, y, mode,
      {c10::nullopt, xl, yl, size, batched, feature_dim, shuffle, generator});
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "pointset_comparisons(Tensor x1, Tensor? x2, str metric, str "
      "mode='all_pairs', bool? squared=None, Tensor? x1_lengths=None, Tensor? "
      "x2_lengths=None, int? chunk_size=None, bool? batched=None, int "
      "feature_dim=-1, bool? shuffle=None, Generator? generator=None) -> "
      "Tensor",
      TORCH_FN(pointset_comparisons_impl));
  m.def(
      "directed_pointset_comparisons(Tensor x1, Tensor? x2=None, str "
      "mode='all_pairs', Tensor? x1_lengths=None, Tensor? x2_lengths=None, "
      "int? chunk_size=None, bool? batched=None, int feature_dim=-1, bool "
      "shuffle=False, Generator? generator=None) -> (Tensor, Tensor, Tensor)",
      TORCH_FN(directed_pointset_comparisons_impl));
  m.def(
      "_pointset_compare(Tensor x, Tensor y, Tensor xl, Tensor yl, int kind, "
      "int mode, bool squared, int size, bool legacy) -> (Tensor, Tensor, "
      "Tensor)");
  m.def(
      "_pointset_compare_backward(Tensor grad, Tensor x, Tensor y, Tensor xl, "
      "Tensor yl, int kind, int mode, bool squared, int size, bool legacy) -> "
      "(Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_pointset_compare", TORCH_FN(pointset_compare_kernel));
  m.impl("_pointset_compare_backward",
         TORCH_FN(pointset_compare_backward_kernel));
}

}  // namespace
namespace {
Tensor pointset_comparisons_composite(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view mode, const PointsetComparisonOptions& options) {
  return std::get<0>(compare(x, y, kind_for(metric), mode, options));
}

Triple directed_pointset_comparisons_composite(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view mode,
    const PointsetComparisonOptions& options) {
  return compare(x, y, Directed, mode, options);
}
}  // namespace
}  // namespace torchpairwise::ops
