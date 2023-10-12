#include "ppminkowski.h"

#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor _ppminkowski(
                const at::Tensor &x1,
                const at::Tensor &x2,
                double p) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("torchpairwise::_ppminkowski", "")
                    .typed<decltype(_ppminkowski)>();
            return op.call(x1, x2, p);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __ppminkowski_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    double p) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::__ppminkowski_backward", "")
                                .typed<decltype(__ppminkowski_backward)>();
                return op.call(grad, x1, x2, p);
            }
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_ppminkowski(Tensor x1, Tensor x2, float p=2) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::__ppminkowski_backward(Tensor grad, Tensor x1, Tensor x2, float p) -> (Tensor grad_x1, Tensor grad_x2)")
            );
        }
    } // namespace ops
} // namespace torchpairwise
