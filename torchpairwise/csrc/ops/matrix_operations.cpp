#include "matrix_operations.h"

#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include "operator_sources_detail.h"
#include "pairwise_operations_detail.h"
#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
using namespace detail::streaming;
Tensor prepare_matrix_value(const Tensor& input, bool amp) {
  TORCH_CHECK(input.layout() == at::kStrided && input.dim() >= 2,
              "Matrix inputs must be dense strided tensors of shape [...,N,M]");
  TORCH_CHECK(input.device().is_cpu() || input.device().is_cuda() ||
                  input.device().is_meta(),
              "Matrix sources require CPU or CUDA");
  auto value = input;
  if (amp && (value.scalar_type() == at::kHalf ||
              value.scalar_type() == at::kBFloat16))
    value = value.to(at::kFloat);
  TORCH_CHECK(
      value.scalar_type() == at::kFloat || value.scalar_type() == at::kDouble ||
          value.scalar_type() == at::kByte ||
          value.scalar_type() == at::kChar ||
          value.scalar_type() == at::kShort ||
          value.scalar_type() == at::kInt || value.scalar_type() == at::kLong ||
          value.scalar_type() == at::kBool,
      "Matrix sources require float32/float64, integer or bool inputs; "
      "half/bfloat16 require autocast");
  return value;
}
Tensor matrix_operations_impl(
    const Tensor& input, c10::string_view operation,
    const c10::optional<Tensor>& left, const c10::optional<Tensor>& right,
    const c10::optional<Tensor>& rows, const c10::optional<Tensor>& columns,
    c10::string_view reduction, c10::optional<int64_t> dim, bool keepdim,
    bool exclude_self, bool transposed, c10::optional<int64_t> chunk_size) {
  PairwiseOperationOptions o;
  o.left = left;
  o.right = right;
  o.rows = rows;
  o.columns = columns;
  o.reduction = std::string(reduction);
  o.dim = dim;
  o.keepdim = keepdim;
  o.exclude_self = exclude_self;
  o.transposed = transposed;
  o.chunk_size = chunk_size;
  validate_pairwise_options(operation, o);
  bool amp = operator_autocast_enabled(input);
  if (!(operation == "block" ||
        (operation == "reduce" && reduction == "prod" && !exclude_self)))
    return execute_prepared(prepare_matrix_operator(input, operation, o),
                            operation, o, amp);
  auto value = prepare_matrix_value(input, amp);
  TORCH_CHECK(chunk_size.value_or(128) > 0, "chunk_size must be positive");
  auto n = value.sym_size(transposed ? -1 : -2),
       m = value.sym_size(transposed ? -2 : -1);
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  // The caller already owns the matrix. A direct ATen product avoids
  // division-based tile adjoints at nonfinite entries. Flattening a
  // noncontiguous matrix for a global product may require a temporary
  // contiguous copy.
  if (operation == "reduce" && reduction == "prod" && !exclude_self) {
    auto logical = transposed ? value.transpose(-1, -2) : value;
    if (dim)
      return logical.prod(*dim, keepdim);
    auto result = logical.flatten(-2).prod(-1, keepdim);
    return keepdim ? result.unsqueeze(-1) : result;
  }
  if (operation == "block") {
    TORCH_CHECK(rows && columns, "block requires rows and columns");
    for (const auto& ids : {*rows, *columns})
      TORCH_CHECK(ids.layout() == at::kStrided && ids.dim() == 1 &&
                      ids.scalar_type() == at::kLong &&
                      ids.device() == value.device(),
                  "Block indices must be int64 vectors on the input device");
    utils::check_values(((rows->ge(0) & rows->lt(n)).all() &
                         (columns->ge(0) & columns->lt(m)).all()),
                        "Block indices must be within matrix sample bounds");
    auto r = transposed ? *columns : *rows, c = transposed ? *rows : *columns;
    auto result =
        value.index({at::indexing::Ellipsis, r.unsqueeze(-1), c.unsqueeze(0)});
    return transposed ? result.transpose(-1, -2) : result;
  }
  TORCH_INTERNAL_ASSERT(false, "Unreachable matrix fast path");
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "matrix_operations(Tensor matrix, str operation='reduce', Tensor? "
      "left=None, Tensor? right=None, Tensor? rows=None, Tensor? columns=None, "
      "str reduction='sum', int? dim=None, bool keepdim=False, bool "
      "exclude_self=False, bool transposed=False, int? chunk_size=None) -> "
      "Tensor",
      TORCH_FN(matrix_operations_impl));
}
}  // namespace
namespace detail::streaming {
PreparedComparison prepare_matrix_operator(const Tensor& input,
                                           c10::string_view operation,
                                           const PairwiseOperationOptions& o) {
  validate_pairwise_options(operation, o);
  bool amp = operator_autocast_enabled(input);
  auto value = prepare_matrix_value(input, amp);
  const auto& left = o.left;
  const auto& right = o.right;
  const auto& reduction = o.reduction;
  const auto& dim = o.dim;
  const auto& exclude_self = o.exclude_self;
  const auto& transposed = o.transposed;
  const auto& chunk_size = o.chunk_size;
  auto size = chunk_size.value_or(128);
  TORCH_CHECK(size > 0, "chunk_size must be positive");
  auto base = operation == "matmul" ? std::string("matmul")
              : operation == "reduce" && reduction != "mean"
                  ? std::string(reduction)
                  : std::string("sum");
  bool search = base == "min" || base == "max";
  bool transpose = transposed != (operation == "reduce" && dim && *dim == -2);
  auto dtype = value.scalar_type();
  if (!value.is_floating_point()) {
    if (operation == "matmul" || operation == "bilinear" ||
        operation == "quadratic") {
      const auto& coefficient = operation == "matmul" ? right : left;
      TORCH_CHECK(coefficient, "Missing matrix coefficient operand");
      dtype = coefficient->scalar_type();
      if (amp && (dtype == at::kHalf || dtype == at::kBFloat16))
        dtype = at::kFloat;
      TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble,
                  "Integer/bool matrix products require float32/float64 "
                  "coefficients outside autocast");
    } else if (operation == "reduce" && (base == "sum" || base == "prod"))
      dtype = at::kLong;
  }
  Settings s{"_matrix",
             (transpose ? "transpose_" : "") +
                 (search ? std::string("neighbors") : base),
             false,
             base == "max",
             exclude_self,
             false,
             search,
             2,
             1,
             size,
             int64_t(dtype),
             3,
             {},
             {},
             {},
             std::string("_matrix")};
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto n = value.sym_size(transpose ? -1 : -2),
       m = value.sym_size(transpose ? -2 : -1);
  TORCH_CHECK(!exclude_self || n == m,
              "exclude_self requires a square operator");
  TORCH_CHECK(!search || m - int64_t(exclude_self) > 0,
              "min/max require an eligible reference");
  auto batch = value.sym_sizes().slice(0, value.dim() - 2).vec();
  Tensors v(Slots, at::empty({0}, value.options()));
  v[X] = value;
  v[Y] = at::arange(value.sym_size(-1), value.options().dtype(at::kLong))
             .unsqueeze(-1);
  if (operation == "matmul") {
    TORCH_CHECK(right, "matmul requires values");
    auto payload = *right;
    if (amp && (payload.scalar_type() == at::kHalf ||
                payload.scalar_type() == at::kBFloat16))
      payload = payload.to(at::kFloat);
    if (payload.dim() == 1)
      payload = payload.unsqueeze(-1);
    TORCH_CHECK(payload.layout() == at::kStrided && payload.dim() >= 2 &&
                    payload.sym_size(-2) == m && payload.sym_size(-1) > 0 &&
                    payload.device() == value.device() &&
                    payload.scalar_type() == dtype,
                "values must match matrix output dtype/device and have shape "
                "[...,M,C]");
    batch = at::infer_size_symint(
        batch, payload.sym_sizes().slice(0, payload.dim() - 2));
    v[Payload] = payload;
    s.mask |= int64_t(1) << Payload;
  }
  return {std::move(v), std::move(s), std::move(batch), n, m};
}
}  // namespace detail::streaming
at::Tensor matrix_operations(const Tensor& matrix, c10::string_view operation,
                             const MatrixOperationOptions& o) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::matrix_operations", "")
          .typed<decltype(matrix_operations_impl)>();
  return op.call(matrix, operation, o.left, o.right, o.rows, o.columns,
                 o.reduction, o.dim, o.keepdim, o.exclude_self, o.transposed,
                 o.chunk_size);
}
}  // namespace torchpairwise::ops
