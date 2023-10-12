#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        at::Tensor _sqmahalanobis(
                const at::Tensor &x1,
                const at::Tensor &x2,
                const at::Tensor &VI);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor, at::Tensor> __sqmahalanobis_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    const at::Tensor &VI);
        }
    }
}
