#pragma once
#include <ATen/ATen.h>
#include <ATen/core/Dict.h>

#include <memory>

#include "matrix_operations.h"
#include "media_operations.h"
#include "ordered_operations.h"
#include "structured_operations.h"

namespace torchpairwise::ops {
// Typed, finite source packet used by the registered expression frontend.
// Prefer the family factories below when constructing expressions from C++.
struct ExpressionSource {
  std::vector<at::Tensor> inputs;
  std::string metric, family;
  bool transposed = false;
  c10::Dict<std::string, at::Tensor> tensors;
  c10::Dict<std::string, double> reals;
  c10::Dict<std::string, int64_t> integers;
  c10::Dict<std::string, bool> flags;
  c10::Dict<std::string, std::string> strings;
  std::vector<double> real_list;
  std::vector<int64_t> integer_list;
};
// Finite source program shared by native iterative consumers.
struct ExpressionGraph {
  std::vector<ExpressionSource> sources;
  std::vector<std::vector<int64_t>> nodes;
  std::vector<at::Tensor> factors;
  std::vector<at::Scalar> numbers;
  int64_t value_root = -1, weight_root = -1;
};
struct ScalarExpressionNode;
struct ScalarExpressionMetadata {
  std::vector<c10::SymInt> shape;
  at::ScalarType dtype;
  c10::Device device;
};
class TORCHPAIRWISE_API ScalarExpression {
 public:
  explicit ScalarExpression(ExpressionSource source);
  explicit ScalarExpression(at::Tensor scalar);
  explicit ScalarExpression(at::Scalar scalar);
  ScalarExpression operator+(const ScalarExpression& other) const;
  ScalarExpression operator-(const ScalarExpression& other) const;
  ScalarExpression operator*(const ScalarExpression& other) const;
  ScalarExpression operator-() const;
  ScalarExpression operator+(at::Scalar other) const;
  ScalarExpression operator-(at::Scalar other) const;
  ScalarExpression operator*(at::Scalar other) const;
  ScalarExpression operator/(const ScalarExpression& other) const;
  ScalarExpression operator/(at::Scalar other) const;
  ScalarExpression pow(const ScalarExpression& other) const;
  ScalarExpression pow(at::Scalar other) const;
  ScalarExpression minimum(const ScalarExpression& other) const;
  ScalarExpression minimum(at::Scalar other) const;
  ScalarExpression maximum(const ScalarExpression& other) const;
  ScalarExpression maximum(at::Scalar other) const;
  ScalarExpression lt(const ScalarExpression& other) const;
  ScalarExpression lt(at::Scalar other) const;
  ScalarExpression le(const ScalarExpression& other) const;
  ScalarExpression le(at::Scalar other) const;
  ScalarExpression gt(const ScalarExpression& other) const;
  ScalarExpression gt(at::Scalar other) const;
  ScalarExpression ge(const ScalarExpression& other) const;
  ScalarExpression ge(at::Scalar other) const;
  ScalarExpression eq(const ScalarExpression& other) const;
  ScalarExpression eq(at::Scalar other) const;
  ScalarExpression ne(const ScalarExpression& other) const;
  ScalarExpression ne(at::Scalar other) const;
  ScalarExpression logical_and(const ScalarExpression& other) const;
  ScalarExpression logical_and(at::Scalar other) const;
  ScalarExpression logical_or(const ScalarExpression& other) const;
  ScalarExpression logical_or(at::Scalar other) const;
  ScalarExpression clamp_min(at::Scalar bound) const;
  ScalarExpression clamp_min(const at::Tensor& bound) const;
  ScalarExpression clamp_max(at::Scalar bound) const;
  ScalarExpression clamp_max(const at::Tensor& bound) const;
  ScalarExpression sqrt() const;
  ScalarExpression log() const;
  ScalarExpression sin() const;
  ScalarExpression cos() const;
  ScalarExpression tanh() const;
  ScalarExpression sigmoid() const;
  ScalarExpression abs() const;
  ScalarExpression logical_not() const;
  // Select this value where condition is Boolean true, otherwise other.
  ScalarExpression where(const ScalarExpression& condition,
                         const ScalarExpression& other) const;
  ScalarExpression where(const ScalarExpression& condition,
                         at::Scalar other) const;
  // The receiver supplies geometry only; these nodes do not evaluate it.
  ScalarExpression row_values(const at::Tensor& values) const;
  ScalarExpression column_values(const at::Tensor& values) const;
  ScalarExpression row_index() const;
  ScalarExpression column_index() const;
  ScalarExpression add_rows(const at::Tensor& values) const;
  ScalarExpression add_columns(const at::Tensor& values) const;
  ScalarExpression masked(const ScalarExpression& mask,
                          at::Scalar value = 0) const;
  ScalarExpression scale(at::Scalar factor) const;
  ScalarExpression scale(const at::Tensor& factor) const;
  ScalarExpression shift(at::Scalar offset) const;
  ScalarExpression square() const;
  ScalarExpression exp() const;
  ScalarExpression weight_rows(const at::Tensor& weights) const;
  ScalarExpression weight_columns(const at::Tensor& weights) const;
  ScalarExpression transpose() const;
  ScalarExpressionMetadata metadata() const;
  ExpressionGraph graph() const;
  std::tuple<at::Tensor, at::Tensor> topk(int64_t k, int64_t dim = -1,
                                          bool largest = true,
                                          bool sorted = true,
                                          bool exclude_self = false) const;
  // Exact selection over one sample axis; returns values, like min/max.
  at::Tensor median(int64_t dim = -1, bool keepdim = false,
                    bool exclude_self = false) const;
  at::Tensor quantile(double q, int64_t dim = -1, bool keepdim = false,
                      c10::string_view interpolation = "linear",
                      bool exclude_self = false) const;
  at::Tensor matmul(const at::Tensor& values) const;
  at::Tensor softmax_matmul(const at::Tensor& values,
                            bool exclude_self = false) const;
  at::Tensor bilinear(const at::Tensor& left, const at::Tensor& right) const;
  at::Tensor quadratic(const at::Tensor& values) const;
  at::Tensor frobenius_inner_product(const ScalarExpression& other,
                                     bool center = false,
                                     bool normalize = false) const;
  at::Tensor diagonal() const;
  at::Tensor hsic(const ScalarExpression& other, bool unbiased = false) const;
  at::Tensor block(const at::Tensor& rows, const at::Tensor& columns) const;
  at::Tensor var(c10::optional<int64_t> dim = c10::nullopt,
                 double correction = 1., bool keepdim = false,
                 bool exclude_self = false) const {
    return statistics("var", dim, keepdim, exclude_self, correction);
  }
  at::Tensor std(c10::optional<int64_t> dim = c10::nullopt,
                 double correction = 1., bool keepdim = false,
                 bool exclude_self = false) const {
    return statistics("std", dim, keepdim, exclude_self, correction);
  }
  at::Tensor logsumexp(c10::optional<int64_t> dim = c10::nullopt,
                       bool keepdim = false, bool exclude_self = false) const {
    return statistics("logsumexp", dim, keepdim, exclude_self);
  }
  at::Tensor count_nonzero(c10::optional<int64_t> dim = c10::nullopt,
                           bool keepdim = false,
                           bool exclude_self = false) const {
    return statistics("count_nonzero", dim, keepdim, exclude_self);
  }
  at::Tensor weighted_mean(const ScalarExpression& weights,
                           c10::optional<int64_t> dim = -1,
                           bool keepdim = false,
                           bool exclude_self = false) const {
    return weighted_statistics(weights, "mean", dim, keepdim, exclude_self, 0.);
  }
  at::Tensor weighted_var(const ScalarExpression& weights,
                          c10::optional<int64_t> dim = -1,
                          double correction = 0., bool keepdim = false,
                          bool exclude_self = false) const {
    return weighted_statistics(weights, "var", dim, keepdim, exclude_self,
                               correction);
  }
  at::Tensor weighted_std(const ScalarExpression& weights,
                          c10::optional<int64_t> dim = -1,
                          double correction = 0., bool keepdim = false,
                          bool exclude_self = false) const {
    return weighted_statistics(weights, "std", dim, keepdim, exclude_self,
                               correction);
  }
  at::Tensor weighted_statistics(const ScalarExpression& weights,
                                 c10::string_view reduction,
                                 c10::optional<int64_t> dim = -1,
                                 bool keepdim = false,
                                 bool exclude_self = false,
                                 double correction = 0.) const;
  at::Tensor weighted_mean(const at::Tensor& weights,
                           c10::optional<int64_t> dim = -1,
                           bool keepdim = false,
                           bool exclude_self = false) const {
    return weighted_statistics(weights, "mean", dim, keepdim, exclude_self, 0.);
  }
  at::Tensor weighted_var(const at::Tensor& weights,
                          c10::optional<int64_t> dim = -1,
                          double correction = 0., bool keepdim = false,
                          bool exclude_self = false) const {
    return weighted_statistics(weights, "var", dim, keepdim, exclude_self,
                               correction);
  }
  at::Tensor weighted_std(const at::Tensor& weights,
                          c10::optional<int64_t> dim = -1,
                          double correction = 0., bool keepdim = false,
                          bool exclude_self = false) const {
    return weighted_statistics(weights, "std", dim, keepdim, exclude_self,
                               correction);
  }
  at::Tensor weighted_statistics(const at::Tensor& weights,
                                 c10::string_view reduction,
                                 c10::optional<int64_t> dim = -1,
                                 bool keepdim = false,
                                 bool exclude_self = false,
                                 double correction = 0.) const;
  at::Tensor statistics(c10::string_view reduction,
                        c10::optional<int64_t> dim = c10::nullopt,
                        bool keepdim = false, bool exclude_self = false,
                        double correction = 1.) const;
  at::Tensor reduce(c10::string_view reduction = "sum",
                    c10::optional<int64_t> dim = c10::nullopt,
                    bool keepdim = false, bool exclude_self = false) const;
  at::Tensor reduce(c10::string_view reduction, at::IntArrayRef dims,
                    bool keepdim = false, bool exclude_self = false) const;
  at::Tensor reduce(c10::string_view reduction, int64_t dim,
                    bool keepdim = false, bool exclude_self = false) const {
    return reduce(reduction, c10::optional<int64_t>(dim), keepdim,
                  exclude_self);
  }
  at::Tensor statistics(c10::string_view reduction, at::IntArrayRef dims,
                        bool keepdim = false, bool exclude_self = false,
                        double correction = 1.) const;
  at::Tensor statistics(c10::string_view reduction, int64_t dim,
                        bool keepdim = false, bool exclude_self = false,
                        double correction = 1.) const {
    return statistics(reduction, c10::optional<int64_t>(dim), keepdim,
                      exclude_self, correction);
  }
  at::Tensor weighted_statistics(const ScalarExpression& weights,
                                 c10::string_view reduction,
                                 at::IntArrayRef dims, bool keepdim = false,
                                 bool exclude_self = false,
                                 double correction = 0.) const;

 private:
  std::shared_ptr<const ScalarExpressionNode> node_;
  explicit ScalarExpression(std::shared_ptr<const ScalarExpressionNode> node);
  at::Tensor call(
      c10::string_view operation, const PairwiseOperationOptions& options,
      c10::optional<double> correction = c10::nullopt,
      const ScalarExpression* weights = nullptr,
      const c10::optional<std::vector<int64_t>>& axes = c10::nullopt,
      c10::optional<int64_t> k = c10::nullopt, bool largest = true,
      bool sorted = true, at::Tensor* selected = nullptr,
      c10::string_view interpolation = "linear") const;
};
TORCHPAIRWISE_API ScalarExpression
same_group_mask(const ScalarExpression& like, const at::Tensor& labels,
                const c10::optional<at::Tensor>& other = c10::nullopt);
TORCHPAIRWISE_API ScalarExpression
different_group_mask(const ScalarExpression& like, const at::Tensor& labels,
                     const c10::optional<at::Tensor>& other = c10::nullopt);
TORCHPAIRWISE_API ScalarExpression causal_mask(const ScalarExpression& like,
                                               int64_t offset = 0);
TORCHPAIRWISE_API ScalarExpression band_mask(const ScalarExpression& like,
                                             int64_t lower = 0,
                                             int64_t upper = 0);
TORCHPAIRWISE_API ScalarExpression diagonal_mask(const ScalarExpression& like);
TORCHPAIRWISE_API ScalarExpression pairwise_expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
    c10::string_view metric = "euclidean", bool kernel = false,
    const PairwiseOperationOptions& options = {});
TORCHPAIRWISE_API ScalarExpression matrix_expression(
    const at::Tensor& matrix, const MatrixOperationOptions& options = {});
TORCHPAIRWISE_API ScalarExpression structured_expression(
    at::TensorList x, const c10::optional<std::vector<at::Tensor>>& y,
    c10::string_view metric, const StructuredOperationOptions& options = {});
TORCHPAIRWISE_API ScalarExpression media_expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view metric, const ImageComparisonOptions& comparison,
    const PairwiseOperationOptions& options = {});
TORCHPAIRWISE_API ScalarExpression media_expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view metric, const AudioComparisonOptions& comparison,
    const PairwiseOperationOptions& options = {});
TORCHPAIRWISE_API ScalarExpression ordered_expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view metric, const SequenceOperatorOptions& comparison,
    const PairwiseOperationOptions& options = {});
TORCHPAIRWISE_API ScalarExpression ordered_expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view metric, const TokenOperatorOptions& comparison,
    const PairwiseOperationOptions& options = {});
TORCHPAIRWISE_API ScalarExpression ordered_expression(
    const at::Tensor& x, const c10::optional<at::Tensor>& y,
    c10::string_view metric, const MMDOperatorOptions& comparison,
    const PairwiseOperationOptions& options = {});
}  // namespace torchpairwise::ops
