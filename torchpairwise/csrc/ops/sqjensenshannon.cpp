#include "sqjensenshannon.h"

#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor _sqjensenshannon(
                const at::Tensor &x1,
                const at::Tensor &x2,
                c10::optional<double> base) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("torchpairwise::_sqjensenshannon", "")
                    .typed<decltype(_sqjensenshannon)>();
            return op.call(x1, x2, base);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> __sqjensenshannon_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    c10::optional<double> base) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::__sqjensenshannon_backward", "")
                                .typed<decltype(__sqjensenshannon_backward)>();
                return op.call(grad, x1, x2, base);
            }
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_sqjensenshannon(Tensor x1, Tensor x2, float? base=None) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::__sqjensenshannon_backward(Tensor grad, Tensor x1, Tensor x2, float? base) -> (Tensor grad_x1, Tensor grad_x2)")
            );
        }
    } // namespace ops
} // namespace torchpairwise
