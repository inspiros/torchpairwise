#pragma once

#include <ATen/ATen.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor _sqjensenshannon(
                const at::Tensor &x1,
                const at::Tensor &x2,
                c10::optional<double> base = c10::nullopt);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __sqjensenshannon_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    c10::optional<double> base);
        }
    }
}
