#include "sequence_comparisons.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <tuple>

#include "sequence_comparisons_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
at::Tensor sequence_comparisons_impl(
    const at::Tensor& input, const c10::optional<at::Tensor>& other,
    c10::string_view mode, bool squared, c10::optional<double> gamma,
    const c10::optional<at::Tensor>& x1_lengths,
    const c10::optional<at::Tensor>& x2_lengths, c10::optional<int64_t> window,
    int64_t chunk);
}

namespace {
using at::Tensor;
using Values = torch::autograd::variable_list;
using OptionalDouble = c10::optional<double>;
using OptionalInt = c10::optional<int64_t>;
constexpr double infinity = std::numeric_limits<double>::infinity();

// Each anti-diagonal has the shorter sequence's padded width. Forward retains
// two diagonals; first backward records/replays just the current pair tile.
std::pair<Tensor, Tensor> reduction(const Tensor& predecessors,
                                    OptionalDouble gamma) {
  auto minimum = predecessors.amin(-1), reachable = minimum.isfinite();
  if (!gamma)
    return {minimum, reachable};
  auto baseline = at::where(reachable, minimum, 0.);
  auto total =
      (-(predecessors - baseline.unsqueeze(-1)) / (*gamma)).exp().sum(-1);
  return {baseline - (*gamma) * at::where(reachable, total, 1.).log(),
          reachable};
}
Tensor weights(const Tensor& predecessors, OptionalDouble gamma) {
  auto minimum = predecessors.amin(-1), reachable = minimum.isfinite();
  Tensor weight;
  if (!gamma) {
    auto tied = predecessors.eq(minimum.unsqueeze(-1));
    weight =
        tied.to(predecessors.scalar_type()) / tied.sum(-1, true).clamp_min(1);
  } else {
    auto baseline = at::where(reachable, minimum, 0.);
    auto exponential =
        (-(predecessors - baseline.unsqueeze(-1)) / (*gamma)).exp();
    weight = exponential /
             at::where(reachable, exponential.sum(-1), 1.).unsqueeze(-1);
  }
  return at::where(reachable.unsqueeze(-1), weight, 0.);
}
std::pair<Tensor, std::vector<Tensor>> wavefronts(
    const Tensor& x, const Tensor& y, const Tensor& lx, const Tensor& ly,
    OptionalDouble gamma, OptionalInt window, bool record = false) {
  auto t = x.size(-2), s = y.size(-2);
  auto shape = x.sizes().slice(0, x.dim() - 2).vec(), boundary_shape = shape;
  boundary_shape.push_back(t + 1);
  auto boundary = at::full(boundary_shape, infinity, x.options());
  auto older = at::cat(
      {at::zeros_like(boundary.narrow(-1, 0, 1)), boundary.narrow(-1, 1, t)},
      -1);
  auto previous = boundary;
  std::vector<Tensor> history;
  if (record) {
    history.reserve(t + s + 1);
    history.push_back(older);
    history.push_back(previous);
  }
  auto result = at::full(shape, infinity, x.options());
  auto i = at::arange(1, t + 1, lx.options());
  for (int64_t diagonal = 2; diagonal <= t + s; ++diagonal) {
    auto j = diagonal - i;
    auto valid = i.le(lx.unsqueeze(-1)) & j.ge(1) & j.le(ly.unsqueeze(-1));
    if (window)
      valid = valid & (i - j).abs().le(*window);
    auto other = y.index_select(-2, j.clamp(1, s) - 1);
    auto delta = at::where(valid.unsqueeze(-1), x - other, 0.);
    auto predecessors =
        at::stack({older.narrow(-1, 0, t), previous.narrow(-1, 0, t),
                   previous.narrow(-1, 1, t)},
                  -1);
    auto reduced = reduction(predecessors, gamma);
    auto cells = at::where(valid & reduced.second,
                           delta.square().sum(-1) + reduced.first, infinity);
    auto current = at::cat({boundary.narrow(-1, 0, 1), cells}, -1);
    auto terminal = i.eq(lx.unsqueeze(-1)) & j.eq(ly.unsqueeze(-1));
    result = at::where(terminal, cells, result.unsqueeze(-1)).amin(-1);
    older = previous;
    previous = current;
    if (record)
      history.push_back(current);
  }
  return {result, history};
}
struct Pair {
  Tensor x, y, lx, ly;
  bool swapped;
};
Pair align(const Tensor& a, const Tensor& b, const Tensor& al,
           const Tensor& bl) {
  auto shape = at::infer_size(a.sizes().slice(0, a.dim() - 2),
                              b.sizes().slice(0, b.dim() - 2));
  auto xs = shape, ys = shape;
  xs.insert(xs.end(), {a.size(-2), a.size(-1)});
  ys.insert(ys.end(), {b.size(-2), b.size(-1)});
  Pair pair{a.expand(xs), b.expand(ys), al.expand(shape), bl.expand(shape),
            a.size(-2) > b.size(-2)};
  if (pair.swapped) {
    std::swap(pair.x, pair.y);
    std::swap(pair.lx, pair.ly);
  }
  return pair;
}
Tensor cost(const Tensor& a, const Tensor& b, const Tensor& al,
            const Tensor& bl, OptionalDouble gamma, OptionalInt window) {
  auto pair = align(a, b, al, bl);
  return wavefronts(pair.x, pair.y, pair.lx, pair.ly, gamma, window).first;
}
std::pair<Tensor, Tensor> cost_backward(const Tensor& gradient, const Tensor& a,
                                        const Tensor& b, const Tensor& al,
                                        const Tensor& bl, OptionalDouble gamma,
                                        OptionalInt window) {
  auto pair = align(a, b, al, bl);
  auto x = pair.x, y = pair.y, lx = pair.lx, ly = pair.ly;
  auto history = wavefronts(x, y, lx, ly, gamma, window, true).second;
  auto gx = at::zeros_like(x), gy = at::zeros_like(y);
  auto current_adjoint = at::zeros_like(history[0]),
       previous_adjoint = at::zeros_like(current_adjoint);
  auto zero = current_adjoint.narrow(-1, 0, 1);
  auto t = x.size(-2), s = y.size(-2);
  auto i = at::arange(1, t + 1, lx.options());
  for (int64_t diagonal = t + s; diagonal > 1; --diagonal) {
    auto j = diagonal - i;
    auto terminal = i.eq(lx.unsqueeze(-1)) & j.eq(ly.unsqueeze(-1));
    auto adjoint =
        current_adjoint.narrow(-1, 1, t) + terminal * gradient.unsqueeze(-1);
    auto active = history[diagonal].narrow(-1, 1, t).isfinite();
    adjoint = at::where(active, adjoint, 0.);
    auto predecessors = at::stack({history[diagonal - 2].narrow(-1, 0, t),
                                   history[diagonal - 1].narrow(-1, 0, t),
                                   history[diagonal - 1].narrow(-1, 1, t)},
                                  -1);
    auto contributions = adjoint.unsqueeze(-1) * weights(predecessors, gamma);
    auto to_previous = previous_adjoint +
                       at::cat({contributions.select(-1, 1), zero}, -1) +
                       at::cat({zero, contributions.select(-1, 2)}, -1);
    auto to_older = at::cat({contributions.select(-1, 0), zero}, -1);
    current_adjoint = to_previous;
    previous_adjoint = to_older;
    auto index = j.clamp(1, s) - 1;
    auto delta =
        at::where(active.unsqueeze(-1), x - y.index_select(-2, index), 0.);
    auto derivative = (2 * adjoint.unsqueeze(-1)) * delta;
    gx.add_(derivative);
    gy.scatter_add_(-2, index.unsqueeze(-1).expand_as(derivative), -derivative);
  }
  if (pair.swapped)
    std::swap(gx, gy);
  return {gx.sum_to_size(a.sizes()), gy.sum_to_size(b.sizes())};
}
Tensor identical(const Tensor& a, const Tensor& b, const Tensor& al,
                 const Tensor& bl) {
  auto common = std::min(a.size(-2), b.size(-2));
  return al.eq(bl) &
         a.narrow(-2, 0, common).eq(b.narrow(-2, 0, common)).all(-1).all(-1);
}
Tensor self_cost(const Tensor& x, const Tensor& length, OptionalDouble gamma,
                 OptionalInt window, int64_t chunk) {
  std::vector<Tensor> pieces;
  for (int64_t i = 0; i < x.size(-3); i += chunk) {
    auto size = std::min(chunk, x.size(-3) - i);
    auto a = x.narrow(-3, i, size), l = length.narrow(-1, i, size);
    pieces.push_back(cost(a, a, l, l, gamma, window));
  }
  return at::cat(pieces, -1);
}
Tensor sequence_compare_kernel(const Tensor& x, const Tensor& y,
                               const Tensor& lx, const Tensor& ly, int64_t mode,
                               bool squared, OptionalDouble gamma,
                               OptionalInt window, int64_t chunk) {
  auto n = x.size(-3), m = y.size(-3);
  auto shape = x.sizes().slice(0, x.dim() - 3).vec();
  shape.push_back(mode == 2 ? n * (n - 1) / 2 : n);
  if (mode == 0)
    shape.push_back(m);
  if (n == 0 || m == 0 || x.numel() == 0 || (mode == 2 && n < 2))
    return at::empty(shape, x.options());
  Tensor ax, ay;
  if (gamma) {
    ax = self_cost(x, lx, gamma, window, chunk);
    ay = self_cost(y, ly, gamma, window, chunk);
  }
  auto tile = [&](int64_t i, int64_t size, int64_t j, int64_t other) {
    auto a = x.narrow(-3, i, size), b = y.narrow(-3, j, other);
    auto al = lx.narrow(-1, i, size), bl = ly.narrow(-1, j, other);
    auto value = cost(a, b, al, bl, gamma, window);
    if (gamma) {
      value = at::where(
          value.isfinite(),
          value - .5 * ax.narrow(-1, i, size) - .5 * ay.narrow(-1, j, other),
          infinity);
      return at::where(identical(a, b, al, bl), 0., value);
    }
    if (squared)
      return value;
    auto positive = value.gt(0);
    return at::where(positive, at::where(positive, value, 1.).sqrt(), 0.);
  };
  std::vector<Tensor> rows;
  if (mode == 1) {
    for (int64_t i = 0; i < n; i += chunk) {
      auto size = std::min(chunk, n - i);
      rows.push_back(tile(i, size, i, size));
    }
  } else {
    for (int64_t i = 0; i < (mode == 2 ? n - 1 : n); ++i) {
      std::vector<Tensor> columns;
      for (int64_t j = mode == 2 ? i + 1 : 0; j < m; j += chunk)
        columns.push_back(tile(i, 1, j, std::min(chunk, m - j)));
      auto row = at::cat(columns, -1);
      rows.push_back(mode == 0 ? row.unsqueeze(-2) : row);
    }
  }
  return at::cat(rows, mode == 0 ? -2 : -1);
}
std::tuple<Tensor, Tensor> sequence_compare_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y, const Tensor& lx,
    const Tensor& ly, int64_t mode, bool squared, OptionalDouble gamma,
    OptionalInt window, int64_t chunk) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu_autocast(
      c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda_autocast(
      c10::DispatchKey::AutocastCUDA);
  auto gx = at::zeros_like(x), gy = at::zeros_like(y);
  if (gradient.numel() == 0)
    return {gx, gy};
  auto n = x.size(-3), m = y.size(-3);
  Tensor sx, sy;
  if (gamma) {
    sx = at::zeros_like(lx, x.options());
    sy = at::zeros_like(ly, y.options());
  }
  int64_t offset = 0;
  for (int64_t i = 0; i < (mode == 2 ? n - 1 : n); i += mode == 1 ? chunk : 1) {
    auto size = mode == 1 ? std::min(chunk, n - i) : int64_t(1);
    for (int64_t j = mode == 2   ? i + 1
                     : mode == 1 ? i
                                 : 0;
         j < (mode == 1 ? i + 1 : m); j += chunk) {
      auto other = mode == 1 ? size : std::min(chunk, m - j);
      auto a = x.narrow(-3, i, size), b = y.narrow(-3, j, other);
      auto al = lx.narrow(-1, i, size), bl = ly.narrow(-1, j, other);
      auto grad = mode == 0
                      ? gradient.select(-2, i).narrow(-1, j, other)
                      : gradient.narrow(-1, mode == 1 ? i : offset, other);
      if (mode == 2)
        offset += other;
      if (gamma || !squared) {
        auto value = cost(a, b, al, bl, gamma, window);
        if (gamma) {
          grad =
              at::where(value.isfinite() & ~identical(a, b, al, bl), grad, 0.);
          sx.narrow(-1, i, size).add_((-.5 * grad).sum_to_size(al.sizes()));
          sy.narrow(-1, j, other).add_((-.5 * grad).sum_to_size(bl.sizes()));
        } else {
          auto positive = value.gt(0) & value.isfinite();
          grad = at::where(
              positive, grad / (2 * at::where(positive, value, 1.).sqrt()), 0.);
        }
      }
      auto derivatives = cost_backward(grad, a, b, al, bl, gamma, window);
      gx.narrow(-3, i, size).add_(derivatives.first);
      gy.narrow(-3, j, other).add_(derivatives.second);
    }
  }
  if (gamma) {
    for (int64_t side = 0; side < 2; ++side) {
      auto value = side == 0 ? x : y, length = side == 0 ? lx : ly;
      auto adjoint = side == 0 ? sx : sy, output = side == 0 ? gx : gy;
      for (int64_t i = 0; i < value.size(-3); i += chunk) {
        auto count = std::min(chunk, value.size(-3) - i);
        auto a = value.narrow(-3, i, count), l = length.narrow(-1, i, count);
        auto derivative = cost_backward(adjoint.narrow(-1, i, count), a, a, l,
                                        l, gamma, window);
        output.narrow(-3, i, count).add_(derivative.first + derivative.second);
      }
    }
  }
  return {gx, gy};
}
Tensor operator_cost_kernel(const Tensor& x, const Tensor& y, const Tensor& lx,
                            const Tensor& ly, OptionalDouble gamma,
                            OptionalInt window, int64_t chunk, bool self) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  if (!self)
    return cost(x, y, lx, ly, gamma, window);
  if (x.numel() == 0)
    return at::zeros_like(lx, x.options()).unsqueeze(-1);
  return self_cost(x, lx, gamma, window, chunk).unsqueeze(-1);
}
std::tuple<Tensor, Tensor> operator_cost_backward_kernel(
    const Tensor& gradient, const Tensor& x, const Tensor& y, const Tensor& lx,
    const Tensor& ly, OptionalDouble gamma, OptionalInt window, int64_t chunk,
    bool self) {
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  if (!self)
    return cost_backward(gradient, x, y, lx, ly, gamma, window);
  auto gx = at::zeros_like(x);
  if (gradient.numel())
    for (int64_t i = 0; i < x.size(-3); i += chunk) {
      auto count = std::min(chunk, x.size(-3) - i);
      auto a = x.narrow(-3, i, count), l = lx.narrow(-1, i, count);
      auto derivative = cost_backward(gradient.squeeze(-1).narrow(-1, i, count),
                                      a, a, l, l, gamma, window);
      gx.narrow(-3, i, count).copy_(derivative.first + derivative.second);
    }
  return {gx.contiguous(), at::zeros_like(y).contiguous()};
}
using CoreSignature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, int64_t, bool, OptionalDouble,
                             OptionalInt, int64_t);
using GradSignature = std::tuple<Tensor, Tensor>(const Tensor&, const Tensor&,
                                                 const Tensor&, const Tensor&,
                                                 const Tensor&, int64_t, bool,
                                                 OptionalDouble, OptionalInt,
                                                 int64_t);

Tensor lengths(const Tensor& value, const c10::optional<Tensor>& supplied,
               int64_t batch_rank) {
  if (!supplied)
    return at::full_symint(value.sym_sizes().slice(0, value.dim() - 2),
                           value.sym_size(-2),
                           value.options().dtype(at::kLong));
  auto length = *supplied;
  TORCH_CHECK(length.layout() == at::kStrided &&
                  length.scalar_type() == at::kLong &&
                  length.device() == value.device(),
              "Lengths must be dense int64 tensors on the input device");
  TORCH_CHECK(length.dim() >= 1 && length.sym_size(-1) == value.sym_size(-3),
              "Lengths must end with the input sample count");
  TORCH_CHECK(length.dim() - 1 <= batch_rank,
              "Lengths cannot introduce additional batch axes");
  utils::check_values(
      (length.ge(1) & length.le(value.sym_size(-2))).all(),
      "Lengths must be between 1 and the padded time dimension");
  return length;
}
}  // namespace

Tensor sequence_comparisons(const Tensor& input,
                            const c10::optional<Tensor>& other,
                            c10::string_view mode, bool squared,
                            OptionalDouble gamma,
                            const c10::optional<Tensor>& x1_lengths,
                            const c10::optional<Tensor>& x2_lengths,
                            OptionalInt window, int64_t chunk) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::sequence_comparisons", "")
          .typed<decltype(sequence_comparisons)>();
  return op.call(input, other, mode, squared, gamma, x1_lengths, x2_lengths,
                 window, chunk);
}
namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "_sequence_operator_cost(Tensor x, Tensor y, Tensor xl, Tensor yl, "
      "float? gamma, int? window, int chunk_size, bool self) -> Tensor");
  m.def(
      "_sequence_operator_cost_backward(Tensor grad, Tensor x, Tensor y, "
      "Tensor xl, Tensor yl, float? gamma, int? window, int chunk_size, bool "
      "self) -> (Tensor, Tensor)");
  m.def(
      "sequence_comparisons(Tensor x1, Tensor? x2=None, str mode='all_pairs', "
      "bool squared=False, float? gamma=None, Tensor? x1_lengths=None, Tensor? "
      "x2_lengths=None, int? window=None, int chunk_size=16) -> Tensor",
      TORCH_FN(sequence_comparisons_impl));
  m.def(
      "_sequence_compare(Tensor x, Tensor y, Tensor lx, Tensor ly, int mode, "
      "bool squared, float? gamma, int? window, int chunk_size) -> Tensor");
  m.def(
      "_sequence_compare_backward(Tensor grad, Tensor x, Tensor y, Tensor lx, "
      "Tensor ly, int mode, bool squared, float? gamma, int? window, int "
      "chunk_size) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_sequence_operator_cost", TORCH_FN(operator_cost_kernel));
  m.impl("_sequence_operator_cost_backward",
         TORCH_FN(operator_cost_backward_kernel));
  m.impl("_sequence_compare", TORCH_FN(sequence_compare_kernel));
  m.impl("_sequence_compare_backward",
         TORCH_FN(sequence_compare_backward_kernel));
}

}  // namespace
namespace {
Tensor sequence_comparisons_impl(const Tensor& input,
                                 const c10::optional<Tensor>& other,
                                 c10::string_view mode, bool squared,
                                 OptionalDouble gamma,
                                 const c10::optional<Tensor>& x1_lengths,
                                 const c10::optional<Tensor>& x2_lengths,
                                 OptionalInt window, int64_t chunk) {
  auto v = detail::sequence::prepare(input, other, mode, squared, gamma,
                                     x1_lengths, x2_lengths, window, chunk);
  int64_t pairing = mode == "all_pairs" ? 0 : mode == "paired" ? 1 : 2;
  return detail::sequence::_sequence_compare(v[0], v[1], v[2], v[3], pairing,
                                             squared, gamma, window, chunk);
}
}  // namespace
}  // namespace torchpairwise::ops

namespace torchpairwise::ops::detail::sequence {
std::array<Tensor, 4> prepare(const Tensor& input,
                              const c10::optional<Tensor>& other,
                              c10::string_view mode, bool squared,
                              OptionalDouble gamma,
                              const c10::optional<Tensor>& x1_lengths,
                              const c10::optional<Tensor>& x2_lengths,
                              OptionalInt window, int64_t chunk) {
  int64_t pairing = mode == "all_pairs"   ? 0
                    : mode == "paired"    ? 1
                    : mode == "condensed" ? 2
                                          : -1;
  TORCH_CHECK(pairing >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(pairing != 2 || !other.has_value(),
              "Condensed comparisons require one input");
  TORCH_CHECK(chunk > 0, "chunk_size must be positive");
  TORCH_CHECK(!window || *window >= 0, "window must be non-negative");
  TORCH_CHECK(!gamma || (std::isfinite(*gamma) && *gamma > 0),
              "gamma must be a positive finite real scalar");
  TORCH_CHECK(!gamma || !squared,
              "squared cannot be used with Soft-DTW divergence");
  auto x = input, y = other.value_or(input);
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided,
              "Sequence inputs must be dense strided tensors");
  TORCH_CHECK(x.dim() >= 3 && y.dim() >= 3,
              "Sequences must have shape [...,N,T,D]");
  TORCH_CHECK(x.sym_size(-1) == y.sym_size(-1),
              "Sequence feature dimensions must match");
  TORCH_CHECK(x.sym_size(-1) > 0 && x.sym_size(-2) > 0 && y.sym_size(-2) > 0,
              "Sequence time and feature dimensions must be non-empty");
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Sequence inputs must have the same device and dtype");
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Sequence inputs must be on CPU or CUDA");
  TORCH_CHECK(pairing != 1 || x.sym_size(-3) == y.sym_size(-3),
              "Paired sequence inputs must have equal sample counts");
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
              "Sequence inputs require float32 or float64 outside autocast");
  auto min = x.scalar_type() == at::kFloat
                 ? double(std::numeric_limits<float>::min())
                 : std::numeric_limits<double>::min();
  auto max = x.scalar_type() == at::kFloat
                 ? double(std::numeric_limits<float>::max())
                 : std::numeric_limits<double>::max();
  TORCH_CHECK(!gamma || (*gamma >= min && *gamma <= max),
              "gamma must be representable as a positive normal value in the "
              "computation dtype");
  auto lx = lengths(x, x1_lengths, std::max(x.dim(), y.dim()) - 3);
  auto ly = lengths(y, !other && !x2_lengths ? x1_lengths : x2_lengths,
                    std::max(x.dim(), y.dim()) - 3);
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 3),
                                     y.sym_sizes().slice(0, y.dim() - 3));
  batch = at::infer_size_symint(batch, lx.sym_sizes().slice(0, lx.dim() - 1));
  batch = at::infer_size_symint(batch, ly.sym_sizes().slice(0, ly.dim() - 1));
  auto expand = [&](Tensor value, Tensor length) {
    auto shape = batch;
    shape.insert(shape.end(),
                 {value.sym_size(-3), value.sym_size(-2), value.sym_size(-1)});
    auto ls = batch;
    ls.push_back(value.sym_size(-3));
    value = value.expand_symint(shape);
    length = length.expand_symint(ls);
    auto valid = at::arange(value.sym_size(-2), length.options())
                     .lt(length.unsqueeze(-1));
    utils::check_values((value.isfinite() | ~valid.unsqueeze(-1)).all(),
                        "Valid sequence frames must be finite");
    return std::make_pair(at::where(valid.unsqueeze(-1), value, 0.), length);
  };
  auto a = expand(x, lx), b = expand(y, ly);
  return {a.first, b.first, a.second, b.second};
}
}  // namespace torchpairwise::ops::detail::sequence

namespace torchpairwise::ops::detail::sequence {
Tensor prepared_pair(const Tensor& x, const Tensor& y, const Tensor& lx,
                     const Tensor& ly, const Tensor& sx, const Tensor& sy,
                     bool paired, bool squared, OptionalDouble gamma,
                     OptionalInt window, int64_t chunk) {
  auto a = paired ? x : x.unsqueeze(-3), b = paired ? y : y.unsqueeze(-4);
  auto al = paired ? lx : lx.unsqueeze(-1), bl = paired ? ly : ly.unsqueeze(-2);
  auto value = operator_cost(a, b, al, bl, gamma, window, chunk);
  if (gamma) {
    auto sa = sx.squeeze(-1), sb = sy.squeeze(-1);
    if (!paired) {
      sa = sa.unsqueeze(-1);
      sb = sb.unsqueeze(-2);
    }
    value = at::where(value.isfinite(), value - .5 * sa - .5 * sb, infinity);
    return at::where(identical(a, b, al, bl), 0., value);
  }
  if (squared)
    return value;
  // Keep an infeasible alignment constant even when a consumer supplies a
  // nonfinite adjoint (for example prod over several +inf entries).
  auto positive = value.gt(0) & value.isfinite();
  auto rooted = at::where(positive, at::where(positive, value, 1.).sqrt(), 0.);
  return at::where(value.isposinf(), infinity, rooted);
}
}  // namespace torchpairwise::ops::detail::sequence
