#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        at::Tensor _haversine(
                const at::Tensor &x1,
                const at::Tensor &x2);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __haversine_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2);
        }
    }
}
