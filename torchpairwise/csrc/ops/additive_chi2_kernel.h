#pragma once

#include <ATen/ATen.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor _additive_chi2_kernel(
                const at::Tensor &x1,
                const at::Tensor &x2);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __additive_chi2_kernel_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2);
        }
    }
}
