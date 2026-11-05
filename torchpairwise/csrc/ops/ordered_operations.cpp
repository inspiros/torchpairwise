#include "ordered_operations.h"

#include <ATen/ExpandUtils.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include "mmd_comparisons_detail.h"
#include "operator_sources_detail.h"
#include "pairwise_operations_detail.h"
#include "sequence_comparisons_detail.h"
#include "token_comparisons_detail.h"
#include "utils/comparison_dispatch.h"
namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
void validate_ordered_matrix_options(const PairwiseOperationOptions& o) {
  const auto& q = o.comparison;
  TORCH_CHECK(
      !q.w && !q.V && !q.VI && !q.bin_positions && !q.projections &&
          !q.x1_lengths && !q.x2_lengths && !q.p && !q.base && !q.alpha &&
          !q.length_scale && !q.nu && !q.gamma && !q.coef0 &&
          !q.regularization && !q.degree && !q.max_iter && !q.chunk_size &&
          !q.feature_dim && !q.squared && !q.shuffle && !q.batched &&
          !q.kernel && !q.generator,
      "Unused vector/cloud comparison options in sequence/token/MMD operator");
}

PreparedComparison prepare_ordered_consumer(Tensors v, Settings s,
                                            c10::string_view operation,
                                            const PairwiseOperationOptions& o,
                                            bool amp) {
  bool token = s.family && *s.family == "_tokens";
  bool product = operation == "matmul" || operation == "bilinear" ||
                 operation == "quadratic";
  if (token && product) {
    auto payload = operation == "matmul" ? o.right : o.left;
    TORCH_CHECK(payload, "Missing coefficient operand");
    auto dtype = payload->scalar_type();
    if (amp && (dtype == at::kHalf || dtype == at::kBFloat16))
      dtype = at::kFloat;
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
                "Token operator coefficients require float32 or float64 "
                "outside autocast");
    s.dtype = int64_t(dtype);
  }
  auto batch =
      at::infer_size_symint(v[X].sym_sizes().slice(0, v[X].dim() - s.rank),
                            v[Y].sym_sizes().slice(0, v[Y].dim() - s.rank));
  auto n = v[s.transposed() ? Y : X].sym_size(-s.rank),
       m = v[s.transposed() ? X : Y].sym_size(-s.rank);
  TORCH_CHECK(!o.exclude_self || n == m,
              "exclude_self requires a square operator");
  TORCH_CHECK(s.kind() != "neighbors" || m - int64_t(o.exclude_self) > 0,
              "min/max require an eligible reference");
  if (operation == "matmul") {
    TORCH_CHECK(o.right, "matmul requires values");
    auto payload = *o.right;
    if (amp && (payload.scalar_type() == at::kHalf ||
                payload.scalar_type() == at::kBFloat16))
      payload = payload.to(at::kFloat);
    if (payload.dim() == 1)
      payload = payload.unsqueeze(-1);
    TORCH_CHECK(
        payload.layout() == at::kStrided && payload.dim() >= 2 &&
            payload.sym_size(-2) == m && payload.sym_size(-1) > 0 &&
            payload.device() == v[X].device() &&
            payload.scalar_type() == at::ScalarType(s.dtype),
        "values must match output dtype/device and have shape [...,M,C]");
    batch = at::infer_size_symint(
        batch, payload.sym_sizes().slice(0, payload.dim() - 2));
    v[Payload] = payload;
    s.mask |= int64_t(1) << Payload;
  }
  return {std::move(v), std::move(s), std::move(batch), n, m};
}
Settings base_settings(c10::string_view metric, c10::string_view operation,
                       const PairwiseOperationOptions& o, int64_t rank,
                       int64_t size, at::ScalarType dtype,
                       c10::string_view family) {
  auto base = operation == "matmul" ? std::string("matmul")
              : operation == "reduce" && o.reduction != "mean"
                  ? o.reduction
                  : std::string("sum");
  bool search = base == "min" || base == "max";
  bool transpose =
      o.transposed != (operation == "reduce" && o.dim && *o.dim == -2);
  return {std::string(metric),
          (transpose ? "transpose_" : "") +
              (search ? std::string("neighbors") : base),
          false,
          base == "max",
          o.exclude_self,
          false,
          search,
          rank,
          1,
          size,
          int64_t(dtype),
          3,
          {},
          {},
          {},
          std::string(family)};
}
Tensor sequence_operations_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const c10::optional<Tensor>& left,
    const c10::optional<Tensor>& right, const c10::optional<Tensor>& rows,
    const c10::optional<Tensor>& columns, c10::string_view reduction,
    c10::optional<int64_t> dim, bool keepdim, bool exclude_self,
    bool transposed, c10::optional<int64_t> chunk_size, bool squared,
    c10::optional<double> gamma, const c10::optional<Tensor>& x1_lengths,
    const c10::optional<Tensor>& x2_lengths, c10::optional<int64_t> window) {
  PairwiseOperationOptions matrix;
  matrix.left = left;
  matrix.right = right;
  matrix.rows = rows;
  matrix.columns = columns;
  matrix.reduction = std::string(reduction);
  matrix.dim = dim;
  matrix.keepdim = keepdim;
  matrix.exclude_self = exclude_self;
  matrix.transposed = transposed;
  matrix.chunk_size = chunk_size;
  validate_pairwise_options(operation, matrix);
  SequenceOperatorOptions comparison;
  comparison.squared = squared;
  comparison.gamma = gamma;
  comparison.x1_lengths = x1_lengths;
  comparison.x2_lengths = x2_lengths;
  comparison.window = window;
  return execute_prepared(
      prepare_sequence_operator(x, y, metric, operation, matrix, comparison),
      operation, matrix, operator_autocast_enabled(x));
}
Tensor token_operations_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const c10::optional<Tensor>& left,
    const c10::optional<Tensor>& right, const c10::optional<Tensor>& rows,
    const c10::optional<Tensor>& columns, c10::string_view reduction,
    c10::optional<int64_t> dim, bool keepdim, bool exclude_self,
    bool transposed, c10::optional<int64_t> chunk_size,
    const c10::optional<Tensor>& x1_lengths,
    const c10::optional<Tensor>& x2_lengths) {
  PairwiseOperationOptions matrix;
  matrix.left = left;
  matrix.right = right;
  matrix.rows = rows;
  matrix.columns = columns;
  matrix.reduction = std::string(reduction);
  matrix.dim = dim;
  matrix.keepdim = keepdim;
  matrix.exclude_self = exclude_self;
  matrix.transposed = transposed;
  matrix.chunk_size = chunk_size;
  validate_pairwise_options(operation, matrix);
  TokenOperatorOptions comparison;
  comparison.x1_lengths = x1_lengths;
  comparison.x2_lengths = x2_lengths;
  return execute_prepared(
      prepare_token_operator(x, y, metric, operation, matrix, comparison),
      operation, matrix, operator_autocast_enabled(x));
}
Tensor mmd_operations_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const c10::optional<Tensor>& left,
    const c10::optional<Tensor>& right, const c10::optional<Tensor>& rows,
    const c10::optional<Tensor>& columns, c10::string_view reduction,
    c10::optional<int64_t> dim, bool keepdim, bool exclude_self,
    bool transposed, c10::optional<int64_t> chunk_size, c10::string_view kernel,
    bool squared, double length_scale, c10::optional<double> gamma,
    c10::optional<int64_t> degree, c10::optional<double> coef0,
    c10::optional<double> alpha, c10::optional<double> nu,
    const c10::optional<Tensor>& x1_lengths,
    const c10::optional<Tensor>& x2_lengths, c10::optional<bool> batched,
    int64_t feature_dim, c10::optional<c10::string_view> estimator,
    c10::optional<int64_t> metric_chunk_size) {
  PairwiseOperationOptions matrix;
  matrix.left = left;
  matrix.right = right;
  matrix.rows = rows;
  matrix.columns = columns;
  matrix.reduction = std::string(reduction);
  matrix.dim = dim;
  matrix.keepdim = keepdim;
  matrix.exclude_self = exclude_self;
  matrix.transposed = transposed;
  matrix.chunk_size = chunk_size;
  validate_pairwise_options(operation, matrix);
  MMDOperatorOptions comparison;
  comparison.kernel = kernel;
  comparison.squared = squared;
  comparison.length_scale = length_scale;
  comparison.gamma = gamma;
  comparison.degree = degree;
  comparison.coef0 = coef0;
  comparison.alpha = alpha;
  comparison.nu = nu;
  comparison.x1_lengths = x1_lengths;
  comparison.x2_lengths = x2_lengths;
  comparison.batched = batched;
  comparison.feature_dim = feature_dim;
  if (estimator)
    comparison.estimator = std::string(*estimator);
  comparison.metric_chunk_size = metric_chunk_size;
  return execute_prepared(
      prepare_mmd_operator(x, y, metric, operation, matrix, comparison),
      operation, matrix, operator_autocast_enabled(x));
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "sequence_pairwise_operations(Tensor x1, Tensor? x2=None, str "
      "metric='dtw', str operation='reduce', Tensor? left=None, Tensor? "
      "right=None, Tensor? rows=None, Tensor? columns=None, str "
      "reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transposed=False, int? chunk_size=None, bool "
      "squared=False, float? gamma=None, Tensor? x1_lengths=None, Tensor? "
      "x2_lengths=None, int? window=None) -> Tensor",
      TORCH_FN(sequence_operations_impl));
  m.def(
      "token_pairwise_operations(Tensor x1, Tensor? x2=None, str "
      "metric='levenshtein', str operation='reduce', Tensor? left=None, "
      "Tensor? right=None, Tensor? rows=None, Tensor? columns=None, str "
      "reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transposed=False, int? chunk_size=None, "
      "Tensor? x1_lengths=None, Tensor? x2_lengths=None) -> Tensor",
      TORCH_FN(token_operations_impl));
  m.def(
      "mmd_pairwise_operations(Tensor x1, Tensor? x2=None, str metric='mmd', "
      "str operation='reduce', Tensor? left=None, Tensor? right=None, Tensor? "
      "rows=None, Tensor? columns=None, str reduction='sum', int? dim=None, "
      "bool keepdim=False, bool exclude_self=False, bool transposed=False, "
      "int? chunk_size=None, str kernel='rbf', bool squared=False, float "
      "length_scale=1., float? gamma=None, int? degree=None, float? "
      "coef0=None, float? alpha=None, float? nu=None, Tensor? x1_lengths=None, "
      "Tensor? x2_lengths=None, bool? batched=None, int feature_dim=-1, str? "
      "estimator=None, int? metric_chunk_size=None) -> Tensor",
      TORCH_FN(mmd_operations_impl));
}
}  // namespace
namespace detail::streaming {
PreparedComparison prepare_sequence_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const PairwiseOperationOptions& matrix,
    const SequenceOperatorOptions& comparison) {
  validate_pairwise_options(operation, matrix);
  validate_ordered_matrix_options(matrix);
  const auto& chunk_size = matrix.chunk_size;
  const auto& squared = comparison.squared;
  const auto& gamma = comparison.gamma;
  const auto& x1_lengths = comparison.x1_lengths;
  const auto& x2_lengths = comparison.x2_lengths;
  const auto& window = comparison.window;
  auto size = chunk_size.value_or(16);
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  bool amp = (x.device().is_cpu() || x.device().is_cuda()) &&
             at::autocast::is_autocast_enabled(x.device().type());
  auto kind = detail::ordered_comparison_kind(metric);
  TORCH_CHECK(kind == 0 || kind == 1, "Unknown sequence comparison");
  TORCH_CHECK((kind == 1) == bool(gamma), kind == 1
                                              ? "Soft-DTW requires gamma"
                                              : "DTW does not accept gamma");
  auto p = detail::sequence::prepare(x, y, "all_pairs", squared, gamma,
                                     x1_lengths, x2_lengths, window, size);
  if (!y && !x2_lengths) {
    p[1] = p[0];
    p[3] = p[2];
  }
  auto s = base_settings(metric, operation, matrix, 3, size, p[0].scalar_type(),
                         "_sequence");
  s.reals = {gamma};
  s.integers = {window};
  s.flags = {squared};
  Tensors v(Slots, at::empty({0}, p[0].options()));
  v[X] = p[0];
  v[Y] = p[1];
  v[XL] = p[2];
  v[YL] = p[3];
  s.mask |= (1 << XL) | (1 << YL);
  if (gamma) {
    v[W] = detail::sequence::operator_cost(p[0], p[0], p[2], p[2], gamma,
                                           window, size, true);
    v[Variance] = p[0].is_same(p[1])
                      ? v[W]
                      : detail::sequence::operator_cost(
                            p[1], p[1], p[3], p[3], gamma, window, size, true);
    s.mask |= (1 << W) | (1 << Variance);
  }
  return prepare_ordered_consumer(std::move(v), std::move(s), operation, matrix,
                                  amp);
}
PreparedComparison prepare_token_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const PairwiseOperationOptions& matrix,
    const TokenOperatorOptions& comparison) {
  validate_pairwise_options(operation, matrix);
  validate_ordered_matrix_options(matrix);
  const auto& chunk_size = matrix.chunk_size;
  const auto& x1_lengths = comparison.x1_lengths;
  const auto& x2_lengths = comparison.x2_lengths;
  auto size = chunk_size.value_or(256);
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  bool amp = (x.device().is_cpu() || x.device().is_cuda()) &&
             at::autocast::is_autocast_enabled(x.device().type());
  auto kind = detail::ordered_comparison_kind(metric);
  TORCH_CHECK(kind == 2 || kind == 3, "Unknown token comparison");
  auto p = detail::tokens::prepare(x, y, "all_pairs", kind - 1, x1_lengths,
                                   x2_lengths, size);
  auto s =
      base_settings(metric, operation, matrix, 2, size, at::kLong, "_tokens");
  s.integers = {kind - 1};
  Tensors v(Slots, at::empty({0}, x.options()));
  v[X] = p[0];
  v[Y] = p[1];
  v[XL] = p[2];
  v[YL] = p[3];
  s.mask |= (1 << XL) | (1 << YL);
  return prepare_ordered_consumer(std::move(v), std::move(s), operation, matrix,
                                  amp);
}
PreparedComparison prepare_mmd_operator(const Tensor& x,
                                        const c10::optional<Tensor>& y,
                                        c10::string_view metric,
                                        c10::string_view operation,
                                        const PairwiseOperationOptions& matrix,
                                        const MMDOperatorOptions& comparison) {
  validate_pairwise_options(operation, matrix);
  validate_ordered_matrix_options(matrix);
  const auto& chunk_size = matrix.chunk_size;
  const auto& kernel = comparison.kernel;
  const auto& squared = comparison.squared;
  const auto& length_scale = comparison.length_scale;
  const auto& gamma = comparison.gamma;
  const auto& degree = comparison.degree;
  const auto& coef0 = comparison.coef0;
  const auto& alpha = comparison.alpha;
  const auto& nu = comparison.nu;
  const auto& x1_lengths = comparison.x1_lengths;
  const auto& x2_lengths = comparison.x2_lengths;
  const auto& batched = comparison.batched;
  const auto& feature_dim = comparison.feature_dim;
  auto estimator = comparison.estimator
                       ? c10::optional<c10::string_view>(*comparison.estimator)
                       : c10::nullopt;
  const auto& metric_chunk_size = comparison.metric_chunk_size;
  auto size = chunk_size.value_or(16);
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  bool amp = (x.device().is_cpu() || x.device().is_cuda()) &&
             at::autocast::is_autocast_enabled(x.device().type());
  auto kind = detail::ordered_comparison_kind(metric);
  TORCH_CHECK(kind == 4 || kind == 5, "Unknown MMD comparison");
  TORCH_CHECK(kind == 5 || !estimator,
              "estimator applies only to MMD statistics");
  TORCH_CHECK(kind == 4 || !squared, "squared is unused for MMD statistics");
  auto p = detail::mmd::prepare(
      x, y, "all_pairs", kernel, squared, length_scale, gamma, degree, coef0,
      alpha, nu, x1_lengths, x2_lengths, size, batched, feature_dim,
      kind == 5
          ? c10::optional<c10::string_view>(estimator.value_or("unbiased"))
          : c10::nullopt);
  auto full = [](const Tensor& a, Tensor l) {
    return l.numel() || a.sym_size(-3) == 0
               ? l
               : at::full_symint(a.sym_sizes().slice(0, a.dim() - 2),
                                 a.sym_size(-2), a.options().dtype(at::kLong));
  };
  TORCH_CHECK(!metric_chunk_size || *metric_chunk_size > 0,
              "metric_chunk_size must be positive");
  p.options.chunk_size = metric_chunk_size.value_or(128);
  auto lx = full(p.clouds.x, p.clouds.xl), ly = full(p.clouds.y, p.clouds.yl);
  auto a = detail::mmd::features(p.clouds.x, lx, p.options),
       b = !y && !x2_lengths ? a
                             : detail::mmd::features(p.clouds.y, ly, p.options);
  auto& q = p.options;
  bool mean = q.name == "linear" && !q.unbiased;
  auto s = base_settings(q.name, operation, matrix, mean ? 2 : 3, size,
                         a[0].scalar_type(), "_mmd");
  s.reals = {q.gamma, q.coef0, q.scale, q.alpha, q.nu};
  s.integers = {q.degree, q.chunk_size};
  s.flags = {squared, q.statistic, q.unbiased};
  Tensors v(Slots, at::empty({0}, a[0].options()));
  v[X] = a[0];
  v[Y] = b[0];
  v[XL] = lx;
  v[YL] = ly;
  s.mask |= (1 << XL) | (1 << YL);
  if (!mean) {
    v[W] = a[1];
    v[Variance] = b[1];
    s.mask |= (1 << W) | (1 << Variance);
  }
  return prepare_ordered_consumer(std::move(v), std::move(s), operation, matrix,
                                  amp);
}
}  // namespace detail::streaming
Tensor ordered_pairwise_operations(const Tensor& x,
                                   const c10::optional<Tensor>& y,
                                   c10::string_view metric,
                                   c10::string_view operation,
                                   const PairwiseOperationOptions& o,
                                   const SequenceOperatorOptions& q) {
  validate_ordered_matrix_options(o);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::sequence_pairwise_operations", "")
          .typed<decltype(sequence_operations_impl)>();
  return op.call(x, y, metric, operation, o.left, o.right, o.rows, o.columns,
                 o.reduction, o.dim, o.keepdim, o.exclude_self, o.transposed,
                 o.chunk_size, q.squared, q.gamma, q.x1_lengths, q.x2_lengths,
                 q.window);
}
Tensor ordered_pairwise_operations(const Tensor& x,
                                   const c10::optional<Tensor>& y,
                                   c10::string_view metric,
                                   c10::string_view operation,
                                   const PairwiseOperationOptions& o,
                                   const TokenOperatorOptions& q) {
  validate_ordered_matrix_options(o);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::token_pairwise_operations", "")
          .typed<decltype(token_operations_impl)>();
  return op.call(x, y, metric, operation, o.left, o.right, o.rows, o.columns,
                 o.reduction, o.dim, o.keepdim, o.exclude_self, o.transposed,
                 o.chunk_size, q.x1_lengths, q.x2_lengths);
}
Tensor ordered_pairwise_operations(const Tensor& x,
                                   const c10::optional<Tensor>& y,
                                   c10::string_view metric,
                                   c10::string_view operation,
                                   const PairwiseOperationOptions& o,
                                   const MMDOperatorOptions& q) {
  validate_ordered_matrix_options(o);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::mmd_pairwise_operations", "")
          .typed<decltype(mmd_operations_impl)>();
  return op.call(x, y, metric, operation, o.left, o.right, o.rows, o.columns,
                 o.reduction, o.dim, o.keepdim, o.exclude_self, o.transposed,
                 o.chunk_size, q.kernel, q.squared, q.length_scale, q.gamma,
                 q.degree, q.coef0, q.alpha, q.nu, q.x1_lengths, q.x2_lengths,
                 q.batched, q.feature_dim,
                 q.estimator ? c10::optional<c10::string_view>(*q.estimator)
                             : c10::nullopt,
                 q.metric_chunk_size);
}
}  // namespace torchpairwise::ops
