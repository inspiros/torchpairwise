#pragma once
#include "expression_statistics_detail.h"

namespace torchpairwise::ops::detail::streaming {
using SoftmaxSignature = StatisticsOutputs(at::TensorList,
                                           TORCHPAIRWISE_EXPRESSION_TYPES);
using SoftmaxGradSignature = Tensors(const Tensor&, at::TensorList,
                                     const Tensor&, const Tensor&,
                                     TORCHPAIRWISE_EXPRESSION_TYPES);
StatisticsOutputs softmax_matmul_kernel(at::TensorList operands,
                                        TORCHPAIRWISE_EXPRESSION_DECL);
Tensors softmax_matmul_backward_kernel(const Tensor& gradient,
                                       at::TensorList operands,
                                       const Tensor& result,
                                       const Tensor& state,
                                       TORCHPAIRWISE_EXPRESSION_DECL);
Tensor expression_softmax_matmul(at::TensorList operands,
                                 const Settings& settings);
}  // namespace torchpairwise::ops::detail::streaming
