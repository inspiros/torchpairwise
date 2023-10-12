#include "canberra.h"

#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor _canberra(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("torchpairwise::_canberra", "")
                    .typed<decltype(_canberra)>();
            return op.call(x1, x2);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __canberra_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::__canberra_backward", "")
                                .typed<decltype(__canberra_backward)>();
                return op.call(grad, x1, x2);
            }
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_canberra(Tensor x1, Tensor x2) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::__canberra_backward(Tensor grad, Tensor x1, Tensor x2) -> (Tensor grad_x1, Tensor grad_x2)")
            );
        }
    } // namespace ops
} // namespace torchpairwise
