#pragma once

#include <ATen/ATen.h>

namespace torchpairwise::ops {
inline void check_pointset_nn(const at::Tensor& x, const at::Tensor& y,
                              const at::Tensor& xl, const at::Tensor& yl) {
  TORCH_CHECK(x.dim() == 4 && y.dim() == 4,
              "Point-set inputs must have shape [B,N,P,C] and [B,M,Q,C]");
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Point-set inputs must share dtype and device");
  TORCH_CHECK(x.size(0) == y.size(0) && x.size(3) == y.size(3),
              "Point-set batch and feature sizes must match");
  TORCH_CHECK(x.size(2) > 0 && y.size(2) > 0 && x.size(3) > 0,
              "Point and feature dimensions must be non-empty");
  TORCH_CHECK(xl.dim() == 2 && yl.dim() == 2 && xl.size(0) == x.size(0) &&
                  xl.size(1) == x.size(1) && yl.size(0) == y.size(0) &&
                  yl.size(1) == y.size(1),
              "Lengths must match batch and sample dimensions");
  TORCH_CHECK(xl.scalar_type() == at::kLong && yl.scalar_type() == at::kLong &&
                  xl.device() == x.device() && yl.device() == x.device(),
              "Lengths must be int64 tensors on the input device");
}
}  // namespace torchpairwise::ops
