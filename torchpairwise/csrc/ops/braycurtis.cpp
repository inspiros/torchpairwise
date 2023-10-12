#include "braycurtis.h"

#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor _braycurtis(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("torchpairwise::_braycurtis", "")
                    .typed<decltype(_braycurtis)>();
            return op.call(x1, x2);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __braycurtis_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::__braycurtis_backward", "")
                                .typed<decltype(__braycurtis_backward)>();
                return op.call(grad, x1, x2);
            }
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_braycurtis(Tensor x1, Tensor x2) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::__braycurtis_backward(Tensor grad, Tensor x1, Tensor x2) -> (Tensor grad_x1, Tensor grad_x2)")
            );
        }
    } // namespace ops
} // namespace torchpairwise
