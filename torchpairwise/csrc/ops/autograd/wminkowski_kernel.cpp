#include "../wminkowski.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        namespace {
            class WeightedMinkowskiFunction
                    : public torch::autograd::Function<WeightedMinkowskiFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &x1,
                        const torch::autograd::Variable &x2,
                        const torch::autograd::Variable &w,
                        double p) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    auto output = _wminkowski(x1, x2, w, p);

                    ctx->save_for_backward({x1, x2, w});
                    ctx->saved_data["p"] = p;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto x1 = saved[0];
                    auto x2 = saved[1];
                    auto w = saved[2];
                    double p = ctx->saved_data["p"].toDouble();

                    auto grads = detail::__wminkowski_backward(
                            grad_output[0], x1, x2, w, p);
                    auto grad_x1 = std::get<0>(grads);
                    auto grad_x2 = std::get<1>(grads);
                    auto grad_w = std::get<2>(grads);

                    return {
                            grad_x1,
                            grad_x2,
                            grad_w,
                            torch::autograd::Variable(),
                    };
                }
            };

            at::Tensor _wminkowski_autograd(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    const at::Tensor &w,
                    double p) {
                return WeightedMinkowskiFunction::apply(x1, x2, w, p);
            }
        } // namespace

        TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_wminkowski"),
                    TORCH_FN(_wminkowski_autograd));
        }
    }
}
