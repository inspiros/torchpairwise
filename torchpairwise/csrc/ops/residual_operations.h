#pragma once
#include "expression_operations.h"

namespace torchpairwise::ops {
// Terminal feature-axis selection, returning [...,N,M,k] values and int64 IDs.
TORCHPAIRWISE_API std::tuple<at::Tensor, at::Tensor> residual_topk(
    const at::Tensor& x, int64_t k,
    const c10::optional<at::Tensor>& y = c10::nullopt, int64_t dim = -1,
    bool largest = true, bool sorted = true,
    c10::string_view transform = "identity", int64_t chunk_size = 128,
    int64_t feature_chunk_size = 128);
// A terminal reduction over any logical N/M/D axes, preserving batch axes.
// Empty dimensions mean all three. Unlike feature source construction, this
// explicit tensor consumer also materializes N*M when only D is selected.
TORCHPAIRWISE_API at::Tensor residual_reduce(
    const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
    c10::string_view reduction = "sum", at::IntArrayRef dims = {},
    bool keepdim = false, c10::optional<double> correction = c10::nullopt,
    c10::string_view transform = "identity", int64_t chunk_size = 128,
    int64_t feature_chunk_size = 128);
// A bounded vector source G[...,i,j,d] = transform(x[...,i,d]-y[...,j,d]).
// Reducing its feature axis constructs a scalar source, without evaluating it.
class TORCHPAIRWISE_API ResidualOperator {
 public:
  ResidualOperator(at::Tensor x, c10::optional<at::Tensor> y = c10::nullopt,
                   std::string transform = "identity", int64_t chunk_size = 128,
                   int64_t feature_chunk_size = 128);
  ScalarExpression reduce(
      c10::string_view reduction, int64_t dim = -1,
      c10::optional<double> correction = c10::nullopt) const;
  ScalarExpression sum(int64_t dim = -1) const {
    return reduce("sum", dim);
  }
  ScalarExpression mean(int64_t dim = -1) const {
    return reduce("mean", dim);
  }
  ScalarExpression min(int64_t dim = -1) const {
    return reduce("min", dim);
  }
  ScalarExpression max(int64_t dim = -1) const {
    return reduce("max", dim);
  }
  ScalarExpression prod(int64_t dim = -1) const {
    return reduce("prod", dim);
  }
  ScalarExpression all(int64_t dim = -1) const {
    return reduce("all", dim);
  }
  ScalarExpression any(int64_t dim = -1) const {
    return reduce("any", dim);
  }
  ScalarExpression var(int64_t dim = -1, double correction = 1.) const {
    return reduce("var", dim, correction);
  }
  ScalarExpression std(int64_t dim = -1, double correction = 1.) const {
    return reduce("std", dim, correction);
  }
  std::tuple<at::Tensor, at::Tensor> topk(int64_t k, int64_t dim = -1,
                                          bool largest = true,
                                          bool sorted = true) const {
    return residual_topk(x_, k, y_, dim, largest, sorted, transform_,
                         chunk_size_, feature_chunk_size_);
  }
  at::Tensor reduce(c10::string_view reduction, at::IntArrayRef dims,
                    c10::optional<double> correction = c10::nullopt,
                    bool keepdim = false) const {
    return residual_reduce(x_, y_, reduction, dims, keepdim, correction,
                           transform_, chunk_size_, feature_chunk_size_);
  }
  at::Tensor sum(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("sum", dims, c10::nullopt, keepdim);
  }
  at::Tensor mean(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("mean", dims, c10::nullopt, keepdim);
  }
  at::Tensor min(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("min", dims, c10::nullopt, keepdim);
  }
  at::Tensor max(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("max", dims, c10::nullopt, keepdim);
  }
  at::Tensor prod(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("prod", dims, c10::nullopt, keepdim);
  }
  at::Tensor all(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("all", dims, c10::nullopt, keepdim);
  }
  at::Tensor any(at::IntArrayRef dims, bool keepdim = false) const {
    return reduce("any", dims, c10::nullopt, keepdim);
  }
  at::Tensor var(at::IntArrayRef dims, double correction = 1.,
                 bool keepdim = false) const {
    return reduce("var", dims, correction, keepdim);
  }
  at::Tensor std(at::IntArrayRef dims, double correction = 1.,
                 bool keepdim = false) const {
    return reduce("std", dims, correction, keepdim);
  }

 private:
  at::Tensor x_;
  c10::optional<at::Tensor> y_;
  std::string transform_;
  int64_t chunk_size_, feature_chunk_size_;
};
TORCHPAIRWISE_API ResidualOperator residual_operator(
    const at::Tensor& x, const c10::optional<at::Tensor>& y = c10::nullopt,
    c10::string_view transform = "identity", int64_t chunk_size = 128,
    int64_t feature_chunk_size = 128);
}  // namespace torchpairwise::ops
