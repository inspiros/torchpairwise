#pragma once
#include "pairwise_operations.h"
#include "streaming_reductions_detail.h"
namespace torchpairwise::ops::detail::streaming {
void validate_pairwise_options(c10::string_view operation,
                               const PairwiseOperationOptions& options);
Tensor execute_prepared(PreparedComparison prepared, c10::string_view operation,
                        const PairwiseOperationOptions& options, bool autocast);
using FormSignature = Tensor(at::TensorList, const Tensor&, const Tensor&,
                             TORCHPAIRWISE_STREAMING_META_TYPES);
using FormGradSignature = Tensors(const Tensor&, at::TensorList, const Tensor&,
                                  const Tensor&,
                                  TORCHPAIRWISE_STREAMING_META_TYPES);
Tensor expression_form(at::TensorList operands, const Tensor& left,
                       const Tensor& right, const Settings& settings);
Tensor pairwise_form_kernel(at::TensorList operands, const Tensor& left,
                            const Tensor& right,
                            TORCHPAIRWISE_STREAMING_META_DECL);
Tensors pairwise_form_backward_kernel(const Tensor& gradient,
                                      at::TensorList operands,
                                      const Tensor& left, const Tensor& right,
                                      TORCHPAIRWISE_STREAMING_META_DECL);
Tensor expression_form_kernel(at::TensorList operands, const Tensor& left,
                              const Tensor& right, const Settings& settings);
Tensors expression_form_backward_kernel(const Tensor& gradient,
                                        at::TensorList operands,
                                        const Tensor& left, const Tensor& right,
                                        const Settings& settings);
inline Tensor pairwise_form(at::TensorList v, const Tensor& left,
                            const Tensor& right, const Settings& s) {
  if (s.expression)
    return expression_form(v, left, right, s);
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_pairwise_form", "")
                       .typed<FormSignature>();
  return op.call(v, left, right, TORCHPAIRWISE_STREAMING_META_ARGS);
}
}  // namespace torchpairwise::ops::detail::streaming
