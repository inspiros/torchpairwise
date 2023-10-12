#include "../ppminkowski.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        namespace {
            class PPowerMinkowskiFunction
                    : public torch::autograd::Function<PPowerMinkowskiFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &x1,
                        const torch::autograd::Variable &x2,
                        double p) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    auto output = _ppminkowski(x1, x2, p);

                    ctx->save_for_backward({x1, x2});
                    ctx->saved_data["p"] = p;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto x1 = saved[0];
                    auto x2 = saved[1];
                    double p = ctx->saved_data["p"].toDouble();

                    auto grads = detail::__ppminkowski_backward(
                            grad_output[0], x1, x2, p);
                    auto grad_x1 = std::get<0>(grads);
                    auto grad_x2 = std::get<1>(grads);

                    return {
                            grad_x1,
                            grad_x2,
                            torch::autograd::Variable(),
                    };
                }
            };

            at::Tensor _ppminkowski_autograd(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    double p) {
                return PPowerMinkowskiFunction::apply(x1, x2, p);
            }
        } // namespace

        TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_ppminkowski"),
                    TORCH_FN(_ppminkowski_autograd));
        }
    }
}
