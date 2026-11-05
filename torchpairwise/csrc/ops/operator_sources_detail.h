#pragma once

#include <ATen/autocast_mode.h>

#include "pairwise_operations.h"
#include "streaming_reductions_detail.h"

namespace torchpairwise::ops {
struct ImageComparisonOptions;
struct AudioComparisonOptions;
struct SequenceOperatorOptions;
struct TokenOperatorOptions;
struct MMDOperatorOptions;
struct StructuredOperationOptions;
}  // namespace torchpairwise::ops

namespace torchpairwise::ops::detail::streaming {
// Internal source builders shared by native consumers. They prepare original
// collections/statistics once; tile evaluation and gradient replay stay in the
// common engine. No Python callback or persistent numerical cache is involved.
inline bool operator_autocast_enabled(const Tensor& input) {
  return (input.device().is_cpu() || input.device().is_cuda()) &&
         at::autocast::is_autocast_enabled(input.device().type());
}
PreparedComparison prepare_pairwise_operator(
    const Tensor& input, const c10::optional<Tensor>& other,
    c10::string_view metric, bool kernel, c10::string_view operation,
    const PairwiseOperationOptions& options);
PreparedComparison prepare_structured_operator(
    at::TensorList x1, const c10::optional<std::vector<Tensor>>& x2,
    c10::string_view metric, c10::string_view operation,
    const StructuredOperationOptions& options);
PreparedComparison prepare_image_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const PairwiseOperationOptions& matrix,
    const ImageComparisonOptions& comparison);
PreparedComparison prepare_audio_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const PairwiseOperationOptions& matrix,
    const AudioComparisonOptions& comparison);
PreparedComparison prepare_sequence_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const PairwiseOperationOptions& matrix,
    const SequenceOperatorOptions& comparison);
PreparedComparison prepare_token_operator(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view operation, const PairwiseOperationOptions& matrix,
    const TokenOperatorOptions& comparison);
PreparedComparison prepare_mmd_operator(const Tensor& x,
                                        const c10::optional<Tensor>& y,
                                        c10::string_view metric,
                                        c10::string_view operation,
                                        const PairwiseOperationOptions& matrix,
                                        const MMDOperatorOptions& comparison);
PreparedComparison prepare_matrix_operator(
    const Tensor& input, c10::string_view operation,
    const PairwiseOperationOptions& options);
}  // namespace torchpairwise::ops::detail::streaming
