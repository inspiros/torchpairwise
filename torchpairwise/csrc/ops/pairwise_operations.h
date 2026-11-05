#pragma once
#include "builtin_comparisons.h"
namespace torchpairwise::ops {
struct PairwiseOperationOptions {
  BuiltinComparisonOptions comparison;
  c10::optional<at::Tensor> left, right, rows, columns;
  c10::optional<int64_t> chunk_size, dim;
  std::string reduction = "sum";
  bool keepdim = false, exclude_self = false, transposed = false;
};
// Uses the native vector/point-cloud comparison catalog. Logical transpose
// preserves the original function orientation and parameter assignment.
TORCHPAIRWISE_API at::Tensor pairwise_operations(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view metric = "euclidean", bool kernel = false,
    c10::string_view operation = "reduce",
    const PairwiseOperationOptions& options = {});
class PairwiseOperator {
 public:
  PairwiseOperator(at::Tensor x1, c10::optional<at::Tensor> x2 = c10::nullopt,
                   std::string metric = "euclidean", bool kernel = false,
                   PairwiseOperationOptions options = {})
      : x1_(std::move(x1)),
        x2_(std::move(x2)),
        metric_(std::move(metric)),
        kernel_(kernel),
        options_(std::move(options)) {}
  PairwiseOperator transpose() const {
    auto result = *this;
    result.options_.transposed = !options_.transposed;
    return result;
  }
  at::Tensor matmul(const at::Tensor& values) const {
    auto o = options_;
    o.right = values;
    return pairwise_operations(x1_, x2_, metric_, kernel_, "matmul", o);
  }
  at::Tensor bilinear(const at::Tensor& left, const at::Tensor& right) const {
    auto o = options_;
    o.left = left;
    o.right = right;
    return pairwise_operations(x1_, x2_, metric_, kernel_, "bilinear", o);
  }
  at::Tensor quadratic(const at::Tensor& values) const {
    auto o = options_;
    o.left = values;
    return pairwise_operations(x1_, x2_, metric_, kernel_, "quadratic", o);
  }
  at::Tensor reduce(c10::string_view reduction = "sum",
                    c10::optional<int64_t> dim = c10::nullopt,
                    bool keepdim = false, bool exclude_self = false) const {
    auto o = options_;
    o.reduction = std::string(reduction);
    o.dim = dim;
    o.keepdim = keepdim;
    o.exclude_self = exclude_self;
    return pairwise_operations(x1_, x2_, metric_, kernel_, "reduce", o);
  }
  at::Tensor diagonal() const {
    return pairwise_operations(x1_, x2_, metric_, kernel_, "diagonal",
                               options_);
  }
  at::Tensor block(const at::Tensor& rows, const at::Tensor& columns) const {
    auto o = options_;
    o.rows = rows;
    o.columns = columns;
    return pairwise_operations(x1_, x2_, metric_, kernel_, "block", o);
  }

 protected:
  at::Tensor x1_;
  c10::optional<at::Tensor> x2_;
  std::string metric_;
  bool kernel_;
  PairwiseOperationOptions options_;
};
}  // namespace torchpairwise::ops
