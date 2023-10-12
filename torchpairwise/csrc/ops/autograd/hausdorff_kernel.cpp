#include "../hausdorff.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace torchpairwise {
    namespace ops {
        namespace {
            class DirectedHausdorffDistancesFunction
                    : public torch::autograd::Function<DirectedHausdorffDistancesFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &x1,
                        const torch::autograd::Variable &x2,
                        bool shuffle,
                        c10::optional<at::Generator> generator) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    ctx->save_for_backward({x1, x2});
                    ctx->saved_data["shuffle"] = shuffle;
                    ctx->saved_data["generator"] = generator.has_value() ? c10::make_optional(generator->clone())
                                                                         : generator;

                    at::Tensor output, x1_indices, x2_indices;
                    std::tie(output, x1_indices, x2_indices) = _directed_hausdorff(x1, x2, shuffle, generator);
                    ctx->mark_non_differentiable({x1_indices, x2_indices});

                    return {
                            output,
                            x1_indices,
                            x2_indices,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto x1 = saved[0];
                    auto x2 = saved[1];
                    auto shuffle = ctx->saved_data["shuffle"].toBool();
                    auto generator = ctx->saved_data["generator"].toOptional<at::Generator>();

                    auto grads = detail::__directed_hausdorff_backward(
                            grad_output[0],
                            x1,
                            x2,
                            shuffle,
                            generator);
                    auto grad_x1 = std::get<0>(grads);
                    auto grad_x2 = std::get<1>(grads);

                    return {
                            grad_x1,
                            grad_x2,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            std::tuple<at::Tensor, at::Tensor, at::Tensor> _directed_hausdorff_autograd(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    bool shuffle,
                    c10::optional<at::Generator> generator) {
                auto result = DirectedHausdorffDistancesFunction::apply(x1, x2, shuffle, generator);
                return std::make_tuple(result[0], result[1], result[2]);
            }
        } // namespace

        TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_directed_hausdorff"),
                    TORCH_FN(_directed_hausdorff_autograd));
        }
    }
}
