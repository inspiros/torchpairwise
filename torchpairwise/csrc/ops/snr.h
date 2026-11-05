#pragma once

#include <ATen/ATen.h>

namespace torchpairwise::ops {
at::Tensor _snr(const at::Tensor& x1, const at::Tensor& x2);

namespace detail {
std::tuple<at::Tensor, at::Tensor> _snr_backward(const at::Tensor& grad,
                                                 const at::Tensor& x1,
                                                 const at::Tensor& x2);
}
}  // namespace torchpairwise::ops
