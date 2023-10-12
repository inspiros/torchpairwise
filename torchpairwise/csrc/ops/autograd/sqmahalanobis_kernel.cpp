#include "../sqmahalanobis.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        namespace {
            class SquaredMahalanobisDistancesFunction
                    : public torch::autograd::Function<SquaredMahalanobisDistancesFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &x1,
                        const torch::autograd::Variable &x2,
                        const torch::autograd::Variable &VI) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    auto output = _sqmahalanobis(x1, x2, VI);

                    ctx->save_for_backward({x1, x2, VI});

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto x1 = saved[0];
                    auto x2 = saved[1];
                    auto VI = saved[2];

                    auto grads = detail::__sqmahalanobis_backward(
                            grad_output[0], x1, x2, VI);
                    auto grad_x1 = std::get<0>(grads);
                    auto grad_x2 = std::get<1>(grads);
                    auto grad_VI = std::get<2>(grads);

                    return {
                            grad_x1,
                            grad_x2,
                            grad_VI,
                    };
                }
            };

            at::Tensor _sqmahalanobis_autograd(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    const at::Tensor &VI) {
                return SquaredMahalanobisDistancesFunction::apply(x1, x2, VI);
            }
        } // namespace

        TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_sqmahalanobis"),
                    TORCH_FN(_sqmahalanobis_autograd));
        }
    }
}
