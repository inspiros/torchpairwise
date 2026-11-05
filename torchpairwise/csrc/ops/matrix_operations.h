#pragma once
#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
struct MatrixOperationOptions {
  c10::optional<at::Tensor> left, right, rows, columns;
  c10::optional<int64_t> chunk_size, dim;
  std::string reduction = "sum";
  bool keepdim = false, exclude_self = false, transposed = false;
};
// Caller-owned real [...,N,M] values, consumed through the shared native
// engine.
TORCHPAIRWISE_API at::Tensor matrix_operations(
    const at::Tensor& matrix, c10::string_view operation = "reduce",
    const MatrixOperationOptions& options = {});
class MatrixOperator {
 public:
  explicit MatrixOperator(at::Tensor matrix,
                          MatrixOperationOptions options = {})
      : matrix_(std::move(matrix)), options_(std::move(options)) {}
  MatrixOperator transpose() const {
    auto result = *this;
    result.options_.transposed = !options_.transposed;
    return result;
  }
  at::Tensor matmul(const at::Tensor& right) const {
    auto o = options_;
    o.right = right;
    return matrix_operations(matrix_, "matmul", o);
  }
  at::Tensor bilinear(const at::Tensor& left, const at::Tensor& right) const {
    auto o = options_;
    o.left = left;
    o.right = right;
    return matrix_operations(matrix_, "bilinear", o);
  }
  at::Tensor quadratic(const at::Tensor& values) const {
    auto o = options_;
    o.left = values;
    return matrix_operations(matrix_, "quadratic", o);
  }
  at::Tensor reduce(c10::string_view reduction = "sum",
                    c10::optional<int64_t> dim = c10::nullopt,
                    bool keepdim = false, bool exclude_self = false) const {
    auto o = options_;
    o.reduction = std::string(reduction);
    o.dim = dim;
    o.keepdim = keepdim;
    o.exclude_self = exclude_self;
    return matrix_operations(matrix_, "reduce", o);
  }
  at::Tensor diagonal() const {
    return matrix_operations(matrix_, "diagonal", options_);
  }
  at::Tensor block(const at::Tensor& rows, const at::Tensor& columns) const {
    auto o = options_;
    o.rows = rows;
    o.columns = columns;
    return matrix_operations(matrix_, "block", o);
  }

 private:
  at::Tensor matrix_;
  MatrixOperationOptions options_;
};
}  // namespace torchpairwise::ops
