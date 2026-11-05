#include "token_comparisons.h"

#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <torch/library.h>

#include <algorithm>

#include "token_comparisons_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
at::Tensor token_comparisons_impl(const at::Tensor& x,
                                  const c10::optional<at::Tensor>& other,
                                  c10::string_view mode, int64_t substitution,
                                  const c10::optional<at::Tensor>& x1_lengths,
                                  const c10::optional<at::Tensor>& x2_lengths,
                                  int64_t chunk);
}

namespace {
using at::Tensor;
Tensor edit_tile(Tensor a, Tensor b, Tensor al, Tensor bl,
                 int64_t substitution) {
  if (a.size(-1) > b.size(-1)) {
    std::swap(a, b);
    std::swap(al, bl);
  }
  auto t = a.size(-1), u = b.size(-1);
  if (t == 0)
    return bl;
  auto width = at::arange(t + 1, al.options());
  auto infinity = t + u + 1;
  auto previous =
      at::where(width.eq(0), 0, infinity).expand({a.size(0), t + 1});
  auto older = at::full_like(previous, infinity), answer = at::zeros_like(al);
  for (int64_t diagonal = 1; diagonal <= t + u; ++diagonal) {
    auto column = diagonal - width;
    auto valid = column.ge(0) & column.le(u);
    auto above = at::roll(previous, {1}, {-1}),
         corner = at::roll(older, {1}, {-1});
    auto mismatch = a.index_select(-1, (width - 1).clamp_min(0))
                        .ne(b.index_select(-1, (column - 1).clamp(0, u - 1)));
    auto value = at::minimum(at::minimum(previous, above) + 1,
                             corner + mismatch.to(at::kLong) * substitution);
    value = at::where(width.eq(0), column, value);
    value = at::where(column.eq(0), width, value);
    value = at::where(valid, value, infinity);
    auto selected = value.gather(-1, al.unsqueeze(-1)).squeeze(-1);
    answer = at::where((al + bl).eq(diagonal), selected, answer);
    older = previous;
    previous = value;
  }
  return answer;
}
Tensor token_compare_kernel(const Tensor& x, const Tensor& y, const Tensor& lx,
                            const Tensor& ly, int64_t mode,
                            int64_t substitution, int64_t chunk) {
  auto n = x.size(-2), m = y.size(-2),
       count = mode == 0   ? n * m
               : mode == 1 ? n
                           : n * (n - 1) / 2;
  auto batch = x.sizes().slice(0, x.dim() - 2).vec(), shape = batch;
  shape.push_back(mode == 0 ? n : count);
  if (mode == 0)
    shape.push_back(m);
  int64_t total = count;
  for (auto size : batch)
    total *= size;
  if (total == 0)
    return at::empty(shape, x.options().dtype(at::kLong));
  Tensor starts;
  if (mode == 2) {
    auto rows = at::arange(n, lx.options()), factor = 2 * n - rows - 1;
    starts = at::floor_divide(rows, 2) * factor +
             rows.remainder(2) * at::floor_divide(factor, 2);
  }
  std::vector<Tensor> pieces;
  for (int64_t start = 0; start < total; start += chunk) {
    auto flat = at::arange(start, std::min(start + chunk, total), lx.options());
    auto pair = flat.remainder(count);
    Tensor i, j;
    if (mode == 0) {
      i = at::floor_divide(pair, m);
      j = pair.remainder(m);
    } else if (mode == 1) {
      i = pair;
      j = pair;
    } else {
      i = at::searchsorted(starts, pair, false, true) - 1;
      j = pair - starts.index_select(0, i) + i + 1;
    }
    auto remainder = at::floor_divide(flat, count);
    std::vector<at::indexing::TensorIndex> coordinates;
    for (auto dimension = batch.rbegin(); dimension != batch.rend();
         ++dimension) {
      coordinates.emplace_back(remainder.remainder(*dimension));
      remainder = at::floor_divide(remainder, *dimension);
    }
    std::reverse(coordinates.begin(), coordinates.end());
    auto left = coordinates, right = coordinates;
    left.emplace_back(i);
    right.emplace_back(j);
    pieces.push_back(edit_tile(x.index(left), y.index(right), lx.index(left),
                               ly.index(right), substitution));
  }
  return at::cat(pieces).reshape(shape);
}
Tensor lengths(const Tensor& value, const c10::optional<Tensor>& supplied) {
  auto shape = value.sym_sizes().slice(0, value.dim() - 1);
  if (!supplied)
    return at::full_symint(shape, value.sym_size(-1),
                           value.options().dtype(at::kLong));
  auto length = *supplied;
  TORCH_CHECK(length.layout() == at::kStrided &&
                  length.scalar_type() == at::kLong &&
                  length.device() == value.device(),
              "Lengths must be dense int64 tensors on the input device");
  length = length.expand_symint(shape);
  utils::check_values(
      (length.ge(0) & length.le(value.sym_size(-1))).all(),
      "Lengths must lie between zero and the padded sequence length");
  return length;
}
using CoreSignature = Tensor(const Tensor&, const Tensor&, const Tensor&,
                             const Tensor&, int64_t, int64_t, int64_t);
}  // namespace

Tensor token_comparisons(const Tensor& x, const c10::optional<Tensor>& other,
                         c10::string_view mode, int64_t substitution,
                         const c10::optional<Tensor>& x1_lengths,
                         const c10::optional<Tensor>& x2_lengths,
                         int64_t chunk) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::token_comparisons", "")
          .typed<decltype(token_comparisons)>();
  return op.call(x, other, mode, substitution, x1_lengths, x2_lengths, chunk);
}
namespace {
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "token_comparisons(Tensor x1, Tensor? x2=None, str mode='all_pairs', int "
      "substitution_cost=1, Tensor? x1_lengths=None, Tensor? x2_lengths=None, "
      "int chunk_size=256) -> Tensor",
      TORCH_FN(token_comparisons_impl));
  m.def(
      "_token_compare(Tensor x, Tensor y, Tensor lx, Tensor ly, int mode, int "
      "substitution_cost, int chunk_size) -> Tensor");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_token_compare", TORCH_FN(token_compare_kernel));
}

}  // namespace
namespace {
Tensor token_comparisons_impl(const Tensor& x,
                              const c10::optional<Tensor>& other,
                              c10::string_view mode, int64_t substitution,
                              const c10::optional<Tensor>& x1_lengths,
                              const c10::optional<Tensor>& x2_lengths,
                              int64_t chunk) {
  auto v = detail::tokens::prepare(x, other, mode, substitution, x1_lengths,
                                   x2_lengths, chunk);
  int64_t pairing = mode == "all_pairs" ? 0 : mode == "paired" ? 1 : 2;
  return detail::tokens::compare(v[0], v[1], v[2], v[3], pairing, substitution,
                                 chunk);
}
}  // namespace
}  // namespace torchpairwise::ops

namespace torchpairwise::ops::detail::tokens {
std::array<Tensor, 4> prepare(const Tensor& x,
                              const c10::optional<Tensor>& other,
                              c10::string_view mode, int64_t substitution,
                              const c10::optional<Tensor>& x1_lengths,
                              const c10::optional<Tensor>& x2_lengths,
                              int64_t chunk) {
  int64_t pairing = mode == "all_pairs"   ? 0
                    : mode == "paired"    ? 1
                    : mode == "condensed" ? 2
                                          : -1;
  TORCH_CHECK(pairing >= 0, "mode must be all_pairs, paired or condensed");
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(pairing != 2 || !other.has_value(),
              "Condensed comparisons require one input");
  TORCH_CHECK(substitution == 1 || substitution == 2,
              "substitution_cost must be 1 or 2");
  TORCH_CHECK(chunk > 0, "chunk_size must be positive");
  auto y = other.value_or(x);
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided,
              "Token inputs must be dense strided tensors");
  TORCH_CHECK(x.dim() >= 2 && y.dim() >= 2,
              "Token inputs must have shape [...,N,T]");
  TORCH_CHECK(c10::isIntegralType(x.scalar_type(), false) &&
                  c10::isIntegralType(y.scalar_type(), false),
              "Token inputs require integer dtypes, excluding bool");
  TORCH_CHECK(x.scalar_type() == at::kByte || x.scalar_type() == at::kChar ||
                  x.scalar_type() == at::kShort ||
                  x.scalar_type() == at::kInt || x.scalar_type() == at::kLong,
              "Unsupported token integer dtype");
  TORCH_CHECK(x.scalar_type() == y.scalar_type() && x.device() == y.device(),
              "Token inputs must have the same dtype and device");
  TORCH_CHECK(
      x.device().is_cpu() || x.device().is_cuda() || x.device().is_meta(),
      "Token inputs must be on CPU or CUDA");
  TORCH_CHECK(pairing != 1 || x.sym_size(-2) == y.sym_size(-2),
              "Paired inputs must have equal sample counts");
  auto al = lengths(x, x1_lengths),
       bl = lengths(y, !other && !x2_lengths ? x1_lengths : x2_lengths);
  auto batch = at::infer_size_symint(x.sym_sizes().slice(0, x.dim() - 2),
                                     y.sym_sizes().slice(0, y.dim() - 2));
  auto xs = batch, ys = batch, ls = batch, rs = batch;
  xs.insert(xs.end(), {x.sym_size(-2), x.sym_size(-1)});
  ys.insert(ys.end(), {y.sym_size(-2), y.sym_size(-1)});
  ls.push_back(x.sym_size(-2));
  rs.push_back(y.sym_size(-2));
  return {x.expand_symint(xs), y.expand_symint(ys), al.expand_symint(ls),
          bl.expand_symint(rs)};
}
}  // namespace torchpairwise::ops::detail::tokens
