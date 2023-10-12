#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        TORCHPAIRWISE_API at::Tensor k_neighbors_mask(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                int64_t k = 0);

        TORCHPAIRWISE_API at::Tensor radius_neighbors_mask(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                double epsilon = 0);
    }
}
