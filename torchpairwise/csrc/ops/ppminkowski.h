#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        TORCHPAIRWISE_API at::Tensor _ppminkowski(
                const at::Tensor &x1,
                const at::Tensor &x2,
                double p = 2);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __ppminkowski_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    double p);
        }
    }
}
