#pragma once
#include "pairwise_operations.h"
namespace torchpairwise::ops {
struct StructuredOperationOptions {
  PairwiseOperationOptions matrix;
  c10::optional<at::Tensor> x1_lengths, x2_lengths;
  c10::optional<bool> squared;
  c10::optional<double> regularization;
  c10::optional<int64_t> max_iter, component_chunk_size;
};
// Covariance:one tensor; Gaussian:(means,covariances);
// mixture:(means,covariances,weights).
TORCHPAIRWISE_API at::Tensor structured_pairwise_operations(
    at::TensorList x1,
    const c10::optional<std::vector<at::Tensor>>& x2 = c10::nullopt,
    c10::string_view metric = "gaussian_wasserstein",
    c10::string_view operation = "reduce",
    const StructuredOperationOptions& options = {});
class StructuredPairwiseOperator {
 public:
  StructuredPairwiseOperator(
      std::vector<at::Tensor> x1,
      c10::optional<std::vector<at::Tensor>> x2 = c10::nullopt,
      std::string metric = "gaussian_wasserstein",
      StructuredOperationOptions options = {})
      : x1_(std::move(x1)),
        x2_(std::move(x2)),
        metric_(std::move(metric)),
        options_(std::move(options)) {}
  StructuredPairwiseOperator transpose() const {
    auto result = *this;
    result.options_.matrix.transposed = !options_.matrix.transposed;
    return result;
  }
  at::Tensor matmul(const at::Tensor& values) const {
    auto o = options_;
    o.matrix.right = values;
    return structured_pairwise_operations(x1_, x2_, metric_, "matmul", o);
  }
  at::Tensor bilinear(const at::Tensor& left, const at::Tensor& right) const {
    auto o = options_;
    o.matrix.left = left;
    o.matrix.right = right;
    return structured_pairwise_operations(x1_, x2_, metric_, "bilinear", o);
  }
  at::Tensor quadratic(const at::Tensor& values) const {
    auto o = options_;
    o.matrix.left = values;
    return structured_pairwise_operations(x1_, x2_, metric_, "quadratic", o);
  }
  at::Tensor reduce(c10::string_view reduction = "sum",
                    c10::optional<int64_t> dim = c10::nullopt,
                    bool keepdim = false, bool exclude_self = false) const {
    auto o = options_;
    o.matrix.reduction = std::string(reduction);
    o.matrix.dim = dim;
    o.matrix.keepdim = keepdim;
    o.matrix.exclude_self = exclude_self;
    return structured_pairwise_operations(x1_, x2_, metric_, "reduce", o);
  }
  at::Tensor diagonal() const {
    return structured_pairwise_operations(x1_, x2_, metric_, "diagonal",
                                          options_);
  }
  at::Tensor block(const at::Tensor& rows, const at::Tensor& columns) const {
    auto o = options_;
    o.matrix.rows = rows;
    o.matrix.columns = columns;
    return structured_pairwise_operations(x1_, x2_, metric_, "block", o);
  }

 private:
  std::vector<at::Tensor> x1_;
  c10::optional<std::vector<at::Tensor>> x2_;
  std::string metric_;
  StructuredOperationOptions options_;
};
}  // namespace torchpairwise::ops
