#pragma once

#include <ATen/ATen.h>

namespace torchpairwise {
    namespace ops {
        std::tuple<at::Tensor, at::Tensor, at::Tensor> _directed_hausdorff(
                const at::Tensor &x1,
                const at::Tensor &x2,
                bool shuffle = false,
                c10::optional<at::Generator> generator = c10::nullopt);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __directed_hausdorff_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    bool shuffle,
                    c10::optional<at::Generator> generator);
        }
    }
}
