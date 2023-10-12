#pragma once

#include <ATen/ATen.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor _braycurtis(
                const at::Tensor &x1,
                const at::Tensor &x2);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __braycurtis_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2);
        }
    }
}
