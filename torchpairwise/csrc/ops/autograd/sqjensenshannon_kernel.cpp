#include "../sqjensenshannon.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        namespace {
            class SquaredJensenShannonDistancesFunction
                    : public torch::autograd::Function<SquaredJensenShannonDistancesFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &x1,
                        const torch::autograd::Variable &x2,
                        c10::optional<double> base) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    ctx->save_for_backward({x1, x2});
                    ctx->saved_data["base"] = base;

                    auto output = _sqjensenshannon(x1, x2, base);

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto x1 = saved[0];
                    auto x2 = saved[1];
                    auto base = ctx->saved_data["base"].toOptional<double>();

                    auto grads = detail::__sqjensenshannon_backward(
                            grad_output[0],
                            x1,
                            x2,
                            base);
                    auto grad_x1 = std::get<0>(grads);
                    auto grad_x2 = std::get<1>(grads);

                    return {
                            grad_x1,
                            grad_x2,
                            torch::autograd::Variable(),
                    };
                }
            };

            at::Tensor _sqjensenshannon_autograd(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    c10::optional<double> base) {
                return SquaredJensenShannonDistancesFunction::apply(x1, x2, base);
            }
        } // namespace

        TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_sqjensenshannon"),
                    TORCH_FN(_sqjensenshannon_autograd));
        }
    }
}
