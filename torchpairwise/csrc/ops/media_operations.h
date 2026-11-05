#pragma once
#include "audio_comparisons.h"
#include "image_comparisons.h"
#include "pairwise_operations.h"

namespace torchpairwise::ops {
// Family options retain the named comparison's meanings. matrix.chunk_size
// overrides the family default; matrix.comparison must remain unset.
TORCHPAIRWISE_API at::Tensor media_pairwise_operations(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view metric, c10::string_view operation,
    const PairwiseOperationOptions& matrix,
    const ImageComparisonOptions& image);
TORCHPAIRWISE_API at::Tensor media_pairwise_operations(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view metric, c10::string_view operation,
    const PairwiseOperationOptions& matrix,
    const AudioComparisonOptions& audio);

template <typename ComparisonOptions>
class MediaPairwiseOperator {
 public:
  MediaPairwiseOperator(at::Tensor x1, c10::optional<at::Tensor> x2,
                        std::string metric, ComparisonOptions comparison = {},
                        PairwiseOperationOptions matrix = {})
      : x1_(std::move(x1)),
        x2_(std::move(x2)),
        metric_(std::move(metric)),
        comparison_(std::move(comparison)),
        matrix_(std::move(matrix)) {}
  MediaPairwiseOperator transpose() const {
    auto result = *this;
    result.matrix_.transposed = !matrix_.transposed;
    return result;
  }
  at::Tensor matmul(const at::Tensor& values) const {
    auto o = matrix_;
    o.right = values;
    return call("matmul", o);
  }
  at::Tensor bilinear(const at::Tensor& left, const at::Tensor& right) const {
    auto o = matrix_;
    o.left = left;
    o.right = right;
    return call("bilinear", o);
  }
  at::Tensor quadratic(const at::Tensor& values) const {
    auto o = matrix_;
    o.left = values;
    return call("quadratic", o);
  }
  at::Tensor reduce(c10::string_view reduction = "sum",
                    c10::optional<int64_t> dim = c10::nullopt,
                    bool keepdim = false, bool exclude_self = false) const {
    auto o = matrix_;
    o.reduction = std::string(reduction);
    o.dim = dim;
    o.keepdim = keepdim;
    o.exclude_self = exclude_self;
    return call("reduce", o);
  }
  at::Tensor diagonal() const {
    return call("diagonal", matrix_);
  }
  at::Tensor block(const at::Tensor& rows, const at::Tensor& columns) const {
    auto o = matrix_;
    o.rows = rows;
    o.columns = columns;
    return call("block", o);
  }

 private:
  at::Tensor call(c10::string_view operation,
                  const PairwiseOperationOptions& options) const {
    return media_pairwise_operations(x1_, x2_, metric_, operation, options,
                                     comparison_);
  }
  at::Tensor x1_;
  c10::optional<at::Tensor> x2_;
  std::string metric_;
  ComparisonOptions comparison_;
  PairwiseOperationOptions matrix_;
};
using ImagePairwiseOperator = MediaPairwiseOperator<ImageComparisonOptions>;
using AudioPairwiseOperator = MediaPairwiseOperator<AudioComparisonOptions>;
}  // namespace torchpairwise::ops
