#pragma once
#include "pairwise_operations.h"

namespace torchpairwise::ops {
struct KernelOperationOptions {
  BuiltinComparisonOptions comparison;
  c10::optional<at::Tensor> left, right, rows, columns;
  c10::optional<int64_t> chunk_size, dim;
  std::string reduction = "sum";
  bool keepdim = false, exclude_self = false;
};
// Matrix products, contractions, reductions and selected kernel entries.
// Scalar reductions preserve batch axes; dim is absent, -1 or -2.
TORCHPAIRWISE_API at::Tensor kernel_operations(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view kernel = "rbf", c10::string_view operation = "reduce",
    const KernelOperationOptions& options = {});

TORCHPAIRWISE_API at::Tensor kernel_residual_outer_product(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2 = c10::nullopt,
    c10::string_view kernel = "rbf", bool normalize = false,
    bool center = false, const KernelOperationOptions& options = {});

// Data/options binding only. Algorithms execute through the native free
// function; no kernel/feature cache is retained, and caller tensor mutations
// stay visible.
class KernelOperator : public PairwiseOperator {
 public:
  KernelOperator(at::Tensor x1, c10::optional<at::Tensor> x2 = c10::nullopt,
                 std::string kernel = "rbf",
                 KernelOperationOptions options = {})
      : PairwiseOperator(std::move(x1), std::move(x2), std::move(kernel), true,
                         matrix_options(options)) {}
  KernelOperator transpose() const {
    auto result = *this;
    result.options_.transposed = !options_.transposed;
    return result;
  }
  at::Tensor displacement(bool normalize = false) const {
    return kernel_operations(
        options_.transposed ? x2_.value_or(x1_) : x1_,
        options_.transposed ? c10::optional<at::Tensor>(x1_) : x2_, metric_,
        normalize ? "displacement_normalized" : "displacement",
        geometry_options());
  }
  at::Tensor residual_outer_product(bool normalize = false,
                                    bool center = false) const {
    return kernel_residual_outer_product(
        options_.transposed ? x2_.value_or(x1_) : x1_,
        options_.transposed ? c10::optional<at::Tensor>(x1_) : x2_, metric_,
        normalize, center, geometry_options());
  }

 private:
  static PairwiseOperationOptions matrix_options(
      const KernelOperationOptions& o) {
    PairwiseOperationOptions m;
    m.comparison = o.comparison;
    m.left = o.left;
    m.right = o.right;
    m.rows = o.rows;
    m.columns = o.columns;
    m.chunk_size = o.chunk_size;
    m.dim = o.dim;
    m.reduction = o.reduction;
    m.keepdim = o.keepdim;
    m.exclude_self = o.exclude_self;
    return m;
  }
  KernelOperationOptions geometry_options() const {
    KernelOperationOptions o;
    o.comparison = options_.comparison;
    o.left = options_.left;
    o.right = options_.right;
    o.rows = options_.rows;
    o.columns = options_.columns;
    o.chunk_size = options_.chunk_size;
    o.dim = options_.dim;
    o.reduction = options_.reduction;
    o.keepdim = options_.keepdim;
    o.exclude_self = options_.exclude_self;
    return o;
  }
};

}  // namespace torchpairwise::ops
