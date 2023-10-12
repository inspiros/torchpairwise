#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        TORCHPAIRWISE_API at::Tensor cdist(
                const at::Tensor &x1,
                const at::Tensor &x2,
                c10::string_view metric = "minkowski",
                const c10::optional<at::Tensor> &w = c10::nullopt,
                const c10::optional<at::Tensor> &V = c10::nullopt,
                const c10::optional<at::Tensor> &VI = c10::nullopt,
                c10::optional<double> p = c10::nullopt,
                c10::optional<double> base = c10::nullopt,
                c10::optional<bool> shuffle = c10::nullopt,
                c10::optional<at::Generator> generator = c10::nullopt);

        TORCHPAIRWISE_API at::Tensor pdist(
                const at::Tensor &input,
                c10::string_view metric = "minkowski",
                const c10::optional<at::Tensor> &w = c10::nullopt,
                const c10::optional<at::Tensor> &V = c10::nullopt,
                const c10::optional<at::Tensor> &VI = c10::nullopt,
                c10::optional<double> p = c10::nullopt,
                c10::optional<double> base = c10::nullopt,
                c10::optional<bool> shuffle = c10::nullopt,
                c10::optional<at::Generator> generator = c10::nullopt);
    }
}
