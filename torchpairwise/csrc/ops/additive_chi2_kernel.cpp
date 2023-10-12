#include "additive_chi2_kernel.h"

#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor _additive_chi2_kernel(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("torchpairwise::_additive_chi2_kernel", "")
                    .typed<decltype(_additive_chi2_kernel)>();
            return op.call(x1, x2);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __additive_chi2_kernel_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::__additive_chi2_kernel_backward", "")
                                .typed<decltype(__additive_chi2_kernel_backward)>();
                return op.call(grad, x1, x2);
            }
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_additive_chi2_kernel(Tensor x1, Tensor x2) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::__additive_chi2_kernel_backward(Tensor grad, Tensor x1, Tensor x2) -> (Tensor grad_x1, Tensor grad_x2)")
            );
        }
    } // namespace ops
} // namespace torchpairwise
