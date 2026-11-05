#pragma once
#include <ATen/ATen.h>

namespace torchpairwise::ops::detail::streaming {
#ifdef WITH_CUDA
// Full per-pair squared-distance adjoint, reading sparse selection weights in
// registers. No dense cotangent and no source graph are constructed.
std::tuple<at::Tensor, at::Tensor> selection_sqeuclidean_backward_cuda(
    const at::Tensor& x, const at::Tensor& y, const at::Tensor& gradient,
    const at::Tensor& indices, double weight);
#endif
}  // namespace torchpairwise::ops::detail::streaming
