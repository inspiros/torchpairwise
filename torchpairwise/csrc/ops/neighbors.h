#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        TORCHPAIRWISE_API at::Tensor k_neighbors_mask(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                int64_t k = 1,
                c10::string_view metric = "euclidean",
                const c10::optional<at::Tensor> &w = c10::nullopt,
                const c10::optional<at::Tensor> &V = c10::nullopt,
                const c10::optional<at::Tensor> &VI = c10::nullopt,
                c10::optional<double> p = c10::nullopt,
                c10::optional<double> base = c10::nullopt,
                c10::optional<bool> shuffle = c10::nullopt,
                c10::optional<at::Generator> generator = c10::nullopt);

        TORCHPAIRWISE_API at::Tensor radius_neighbors_mask(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt,
                double epsilon = 0,
                c10::string_view metric = "euclidean",
                const c10::optional<at::Tensor> &w = c10::nullopt,
                const c10::optional<at::Tensor> &V = c10::nullopt,
                const c10::optional<at::Tensor> &VI = c10::nullopt,
                c10::optional<double> p = c10::nullopt,
                c10::optional<double> base = c10::nullopt,
                c10::optional<bool> shuffle = c10::nullopt,
                c10::optional<at::Generator> generator = c10::nullopt);
    }
}
