#pragma once

#include <limits>
#include <ATen/ATen.h>
#include <torch/library.h>

#include "utils/scalar_type_utils.h"

namespace torchpairwise {
    namespace ops {
        void test_scalar_type() {
            auto x = at::Scalar(0.000976563);
            std::cout << "eps" << std::endl;
            std::cout << x.toBFloat16().x << std::endl;

            x = at::Scalar(std::numeric_limits<double>::infinity());
            std::cout << "inf" << std::endl;
            std::cout << x.toBFloat16().x << std::endl;
        }

        at::Tensor test_fill_indices(const at::Tensor &X, const at::Tensor &Y) {
            int64_t k = 2;
            auto neighbors_inds = at::cdist(X, Y).argsort(1, false).slice(1, 0, k + 1);
            auto output = at::zeros({X.size(0), Y.size(0)}, X.options());
            auto first_dim = at::arange(0, X.size(0), neighbors_inds.options()).view({-1, 1}).repeat({1, k + 1});
            output.index_put_({first_dim.flatten(), neighbors_inds.flatten()}, 1);
            return output;
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            m.def("torchpairwise::_test_scalar_type", TORCH_FN(test_scalar_type));
            m.def("torchpairwise::_test_fill_indices", TORCH_FN(test_fill_indices));
        }
    }
}
