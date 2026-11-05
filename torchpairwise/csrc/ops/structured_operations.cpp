#include "structured_operations.h"

#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include "operator_sources_detail.h"
#include "pairwise_operations_detail.h"
#include "structured_comparisons_detail.h"
#include "utils/comparison_dispatch.h"

namespace torchpairwise::ops {
namespace detail::streaming {
using detail::structured::PreparedInputs;
PreparedComparison prepare_structured_operator(
    at::TensorList x1, const c10::optional<std::vector<Tensor>>& x2,
    c10::string_view metric, c10::string_view operation,
    const StructuredOperationOptions& options) {
  const auto& o = options.matrix;
  validate_pairwise_options(operation, o);
  int64_t kind = detail::structured_comparison_kind(metric);
  TORCH_CHECK(kind >= 0, "Unknown structured operator comparison: ", metric);
  int64_t count = kind == 0 ? 2 : kind < 4 ? 1 : 3;
  TORCH_CHECK(
      int64_t(x1.size()) == count && (!x2 || int64_t(x2->size()) == count),
      "Structured collection tensor count mismatch");
  auto y = x2.value_or(std::vector<Tensor>(x1.begin(), x1.end()));
  bool amp = (x1[0].device().is_cpu() || x1[0].device().is_cuda()) &&
             at::autocast::is_autocast_enabled(x1[0].device().type());
  auto cast = [&](Tensor t) {
    return amp && (t.scalar_type() == at::kHalf ||
                   t.scalar_type() == at::kBFloat16)
               ? t.to(at::kFloat)
               : t;
  };
  auto size = o.chunk_size.value_or(128);
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  const auto& q = o.comparison;
  TORCH_CHECK(!q.w && !q.V && !q.VI && !q.bin_positions && !q.projections &&
                  !q.x1_lengths && !q.x2_lengths && !q.p && !q.base &&
                  !q.alpha && !q.length_scale && !q.nu && !q.gamma &&
                  !q.coef0 && !q.regularization && !q.degree && !q.max_iter &&
                  !q.chunk_size && !q.feature_dim && !q.squared && !q.shuffle &&
                  !q.batched && !q.kernel && !q.generator,
              "Unused vector/cloud comparison options in structured operator");
  TORCH_CHECK(kind >= 4 || (!options.x1_lengths && !options.x2_lengths &&
                            !options.regularization && !options.max_iter &&
                            !options.component_chunk_size),
              "Unused mixture option");
  PreparedInputs p;
  if (kind == 0)
    p = detail::structured::prepare_gaussians(
        x1[0], x1[1], x2 ? c10::optional<Tensor>(y[0]) : c10::nullopt,
        x2 ? c10::optional<Tensor>(y[1]) : c10::nullopt, "all_pairs",
        options.squared.value_or(false));
  else if (kind < 4) {
    constexpr const char* names[] = {"", "log_euclidean", "affine_invariant",
                                     "stein"};
    p = detail::structured::prepare_covariances(
        x1[0], x2 ? c10::optional<Tensor>(y[0]) : c10::nullopt, names[kind],
        "all_pairs", options.squared.value_or(false));
  } else {
    constexpr const char* names[] = {"overlap",     "l2",       "cs",
                                     "wasserstein", "entropic", "sinkhorn"};
    p = detail::structured::prepare_mixtures(
        x1[0], x1[1], x1[2], x2 ? c10::optional<Tensor>(y[0]) : c10::nullopt,
        x2 ? c10::optional<Tensor>(y[1]) : c10::nullopt,
        x2 ? c10::optional<Tensor>(y[2]) : c10::nullopt, names[kind - 4],
        "all_pairs", options.x1_lengths, options.x2_lengths, options.squared,
        options.regularization, options.max_iter, size,
        options.component_chunk_size);
  }
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  const auto& raw = p.operands;
  auto a =
      detail::structured::features(raw[0], raw[1], raw[2], raw[6], kind,
                                   p.epsilon, p.iterations, p.component_tile);
  // Self collections reuse preparation, including normalized mixture
  // weights/self terms.
  bool same = !x2;
  if (x2) {
    same = true;
    for (int64_t k = 0; k < count; ++k)
      same &= x1[k].is_same(y[k]);
    same &= (!options.x1_lengths && !options.x2_lengths) ||
            (options.x1_lengths && options.x2_lengths &&
             options.x1_lengths->is_same(*options.x2_lengths));
  }
  if (!x2 && options.x2_lengths)
    same =
        options.x1_lengths && options.x1_lengths->is_same(*options.x2_lengths);
  auto b = same ? a
                : detail::structured::features(raw[3], raw[4], raw[5], raw[7],
                                               kind, p.epsilon, p.iterations,
                                               p.component_tile);
  auto empty = at::empty({0}, raw[0].options());
  Tensors v(Slots, empty);
  auto reduction = o.reduction;
  auto base = operation == "matmul" ? std::string("matmul")
              : operation == "reduce" && reduction != "mean"
                  ? reduction
                  : std::string("sum");
  bool search = base == "min" || base == "max";
  bool transposed =
      o.transposed != (operation == "reduce" && o.dim && *o.dim == -2);
  auto restore = [&](const Tensor& t) {
    auto shape = p.batch;
    auto dims = t.sym_sizes();
    shape.insert(shape.end(), dims.begin() + 1, dims.end());
    return t.reshape_symint(shape);
  };
  int64_t mask = 0;
  for (int64_t k = 0; k < 4; ++k) {
    bool active = k < 2 || (kind >= 4 && k == 2) ||
                  ((kind == 5 || kind == 6 || kind == 9) && k == 3);
    if (active) {
      v[2 * k] = restore(a[k]);
      v[2 * k + 1] = restore(b[k]);
      mask |= (int64_t(3) << (2 * k));
    }
  }
  Settings s{std::string(metric),
             (transposed ? "transpose_" : "") +
                 (search ? std::string("neighbors") : base),
             false,
             base == "max",
             o.exclude_self,
             false,
             search,
             kind == 0 ? 2 : 3,
             1,
             size,
             int64_t(raw[0].scalar_type()),
             mask,
             {p.epsilon},
             {kind, p.iterations, p.component_tile},
             {p.squared},
             std::string("_structured")};
  auto n = v[transposed ? Y : X].sym_size(-s.rank),
       m = v[transposed ? X : Y].sym_size(-s.rank);
  TORCH_CHECK(!o.exclude_self || n == m,
              "exclude_self requires a square operator");
  TORCH_CHECK(!search || m - int64_t(o.exclude_self) > 0,
              "min/max require at least one eligible reference");
  auto batch = p.batch;
  if (operation == "matmul") {
    TORCH_CHECK(o.right, "matmul requires values");
    auto payload = cast(*o.right);
    if (payload.dim() == 1)
      payload = payload.unsqueeze(-1);
    TORCH_CHECK(payload.layout() == at::kStrided && payload.dim() >= 2 &&
                    payload.sym_size(-2) == m && payload.sym_size(-1) > 0 &&
                    payload.device() == raw[0].device() &&
                    payload.scalar_type() == raw[0].scalar_type(),
                "values must match comparison output dtype/device and have "
                "shape [...,M,C]");
    batch = at::infer_size_symint(
        batch, payload.sym_sizes().slice(0, payload.dim() - 2));
    v[Payload] = payload;
    s.mask |= int64_t(1) << Payload;
  }
  return {std::move(v), std::move(s), std::move(batch), n, m};
}
}  // namespace detail::streaming
namespace {
using namespace detail::streaming;
Tensor structured_operations_composite(
    at::TensorList x1, const c10::optional<std::vector<Tensor>>& x2,
    c10::string_view metric, c10::string_view operation,
    const StructuredOperationOptions& options) {
  auto prepared =
      prepare_structured_operator(x1, x2, metric, operation, options);
  return execute_prepared(std::move(prepared), operation, options.matrix,
                          operator_autocast_enabled(x1[0]));
}
Tensor structured_operations_impl(
    at::TensorList x1, const c10::optional<std::vector<Tensor>>& x2,
    c10::string_view metric, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transposed, c10::optional<int64_t> chunk_size,
    const c10::optional<Tensor>& x1_lengths,
    const c10::optional<Tensor>& x2_lengths, c10::optional<bool> squared,
    c10::optional<double> regularization, c10::optional<int64_t> max_iter,
    c10::optional<int64_t> component_chunk_size) {
  StructuredOperationOptions o;
  auto& m = o.matrix;
  m.left = left;
  m.right = right;
  m.rows = rows;
  m.columns = columns;
  m.reduction = std::string(reduction);
  m.dim = dim;
  m.keepdim = keepdim;
  m.exclude_self = exclude_self;
  m.transposed = transposed;
  m.chunk_size = chunk_size;
  o.x1_lengths = x1_lengths;
  o.x2_lengths = x2_lengths;
  o.squared = squared;
  o.regularization = regularization;
  o.max_iter = max_iter;
  o.component_chunk_size = component_chunk_size;
  return structured_operations_composite(x1, x2, metric, operation, o);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "structured_pairwise_operations(Tensor[] x1, Tensor[]? x2=None, str "
      "metric='gaussian_wasserstein', str operation='reduce', Tensor? "
      "left=None, Tensor? right=None, Tensor? rows=None, Tensor? columns=None, "
      "str reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transposed=False, int? chunk_size=None, "
      "Tensor? x1_lengths=None, Tensor? x2_lengths=None, bool? squared=None, "
      "float? regularization=None, int? max_iter=None, int? "
      "component_chunk_size=None) -> Tensor",
      TORCH_FN(structured_operations_impl));
}
}  // namespace
Tensor structured_pairwise_operations(
    at::TensorList x1, const c10::optional<std::vector<Tensor>>& x2,
    c10::string_view metric, c10::string_view operation,
    const StructuredOperationOptions& o) {
  const auto& m = o.matrix;
  // The structured schema has no vector/cloud parameter packet.
  const auto& q = m.comparison;
  TORCH_CHECK(!q.w && !q.V && !q.VI && !q.bin_positions && !q.projections &&
                  !q.x1_lengths && !q.x2_lengths && !q.p && !q.base &&
                  !q.alpha && !q.length_scale && !q.nu && !q.gamma &&
                  !q.coef0 && !q.regularization && !q.degree && !q.max_iter &&
                  !q.chunk_size && !q.feature_dim && !q.squared && !q.shuffle &&
                  !q.batched && !q.kernel && !q.generator,
              "Unused vector/cloud comparison options in structured operator");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "torchpairwise::structured_pairwise_operations", "")
                       .typed<decltype(structured_operations_impl)>();
  return op.call(x1, x2, metric, operation, m.left, m.right, m.rows, m.columns,
                 m.reduction, m.dim, m.keepdim, m.exclude_self, m.transposed,
                 m.chunk_size, o.x1_lengths, o.x2_lengths, o.squared,
                 o.regularization, o.max_iter, o.component_chunk_size);
}
}  // namespace torchpairwise::ops
