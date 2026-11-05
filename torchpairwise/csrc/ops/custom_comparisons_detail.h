#pragma once
#include "custom_comparisons.h"
#include "streaming_reductions_detail.h"

namespace torchpairwise::ops::detail::custom {
using streaming::PreparedComparison;
using streaming::Settings;
using streaming::Tensor;
using streaming::Tensors;
std::string operation_names(c10::string_view forward, c10::string_view backward,
                            int64_t sample_ndim = 1);
Tensors unpack(const Tensor& packed, at::IntArrayRef shapes);
Tensor forward(const Tensor& x, const Tensor& y, const Tensor& packed,
               c10::string_view names, at::IntArrayRef shapes, int64_t dtype);
Tensors backward(const Tensor& gradient, const Tensor& x, const Tensor& y,
                 const Tensor& packed, c10::string_view names,
                 at::IntArrayRef shapes);
Tensor paired_with_vjp(const Tensor& x, const Tensor& y, const Tensor& packed,
                       c10::string_view names, at::IntArrayRef shapes,
                       int64_t dtype);
Tensor paired(const Tensor& x, const Tensor& y, const Tensor& packed,
              c10::string_view names, at::IntArrayRef shapes, int64_t dtype);
Tensor evaluate(at::TensorList local, const Settings& settings, bool paired);
PreparedComparison prepare(const Tensor& x, const c10::optional<Tensor>& y,
                           at::TensorList parameters, c10::string_view forward,
                           c10::string_view backward, int64_t sample_ndim,
                           c10::optional<int64_t> output_dtype,
                           const PairwiseOperationOptions& options,
                           bool streaming = true);
using DenseSignature = Tensor(at::TensorList, c10::string_view, at::IntArrayRef,
                              int64_t, int64_t, int64_t, int64_t);
using DenseGradSignature = Tensors(const Tensor&, at::TensorList,
                                   c10::string_view, at::IntArrayRef, int64_t,
                                   int64_t, int64_t, int64_t);
Tensor dense_forward(at::TensorList inputs, c10::string_view names,
                     at::IntArrayRef shapes, int64_t mode, int64_t rank,
                     int64_t dtype, int64_t size);
Tensors dense_backward(const Tensor& gradient, at::TensorList inputs,
                       c10::string_view names, at::IntArrayRef shapes,
                       int64_t mode, int64_t rank, int64_t dtype, int64_t size);
}  // namespace torchpairwise::ops::detail::custom
