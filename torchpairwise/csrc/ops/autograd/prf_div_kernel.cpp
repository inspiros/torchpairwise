#include <torch/autograd.h>
#include <torch/types.h>

#include "../prf_div.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            class PRFDivideFunction
                    : public torch::autograd::Function<PRFDivideFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &self,
                        const torch::autograd::Variable &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = prf_div(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({self, other});
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto self = saved[0];
                    auto other = saved[1];
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grads = detail::_prf_div_backward(
                            grad_output[0],
                            self,
                            other,
                            mode);
                    auto grad_self = std::get<0>(grads);
                    auto grad_other = std::get<1>(grads);

                    return {
                            grad_self,
                            grad_other,
                            torch::autograd::Variable(),
                    };
                }
            };

            class IZeroLeftDivideFunction
                    : public torch::autograd::Function<IZeroLeftDivideFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &self,
                        const torch::autograd::Variable &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = prf_ldiv(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({self, other});
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto self = saved[0];
                    auto other = saved[1];
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grads = detail::_prf_ldiv_backward(
                            grad_output[0],
                            self,
                            other,
                            mode);
                    auto grad_self = std::get<0>(grads);
                    auto grad_other = std::get<1>(grads);

                    return {
                            grad_self,
                            grad_other,
                            torch::autograd::Variable(),
                    };
                }
            };

            class PRFDivideScalarFunction
                    : public torch::autograd::Function<PRFDivideScalarFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &self,
                        const at::Scalar &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = prf_div(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({self});
                    ctx->saved_data["other"] = other;
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto self = saved[0];
                    auto other = ctx->saved_data["other"].toScalar();
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grad_self = detail::_prf_div_backward(
                            grad_output[0],
                            self,
                            other,
                            mode);

                    return {
                            grad_self,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class IZeroLeftDivideScalarFunction
                    : public torch::autograd::Function<IZeroLeftDivideScalarFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &self,
                        const at::Scalar &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = prf_ldiv(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({self});
                    ctx->saved_data["other"] = other;
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto self = saved[0];
                    auto other = ctx->saved_data["other"].toScalar();
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grad_self = detail::_prf_ldiv_backward(
                            grad_output[0],
                            self,
                            other,
                            mode);

                    return {
                            grad_self,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class PRFDividerScalarFunction
                    : public torch::autograd::Function<PRFDividerScalarFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const at::Scalar &self,
                        const torch::autograd::Variable &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = prf_div(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({other});
                    ctx->saved_data["self"] = self;
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto other = saved[0];
                    auto self = ctx->saved_data["self"].toScalar();
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grad_other = detail::_prf_div_backward(
                            grad_output[0],
                            self,
                            other,
                            mode);

                    return {
                            torch::autograd::Variable(),
                            grad_other,
                            torch::autograd::Variable(),
                    };
                }
            };

            class IZeroLeftDividerScalarFunction
                    : public torch::autograd::Function<IZeroLeftDividerScalarFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const at::Scalar &self,
                        const torch::autograd::Variable &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = prf_ldiv(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({other});
                    ctx->saved_data["self"] = self;
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto other = saved[0];
                    auto self = ctx->saved_data["self"].toScalar();
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grad_other = detail::_prf_ldiv_backward(
                            grad_output[0],
                            self,
                            other,
                            mode);

                    return {
                            torch::autograd::Variable(),
                            grad_other,
                            torch::autograd::Variable(),
                    };
                }
            };

            class PRFDivideInplaceFunction
                    : public torch::autograd::Function<PRFDivideInplaceFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        torch::autograd::Variable &self,
                        const torch::autograd::Variable &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    ctx->mark_dirty({self});
                    auto output = prf_div_(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({self, other});
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto self = saved[0];
                    auto other = saved[1];
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grads = detail::_prf_div__backward(
                            grad_output[0],
                            self,
                            other,
                            mode);
                    auto grad_self = std::get<0>(grads);
                    auto grad_other = std::get<1>(grads);

                    return {
                            grad_self,
                            grad_other,
                            torch::autograd::Variable(),
                    };
                }
            };

            class IZeroLeftDivideInplaceFunction
                    : public torch::autograd::Function<IZeroLeftDivideInplaceFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        torch::autograd::Variable &self,
                        const torch::autograd::Variable &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    ctx->mark_dirty({self});
                    auto output = prf_ldiv_(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({self, other});
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto self = saved[0];
                    auto other = saved[1];
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grads = detail::_prf_ldiv__backward(
                            grad_output[0],
                            self,
                            other,
                            mode);
                    auto grad_self = std::get<0>(grads);
                    auto grad_other = std::get<1>(grads);

                    return {
                            grad_self,
                            grad_other,
                            torch::autograd::Variable(),
                    };
                }
            };

            class PRFDivideInplaceScalarFunction
                    : public torch::autograd::Function<PRFDivideInplaceScalarFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        torch::autograd::Variable &self,
                        const at::Scalar &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    ctx->mark_dirty({self});
                    auto output = prf_div_(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({self});
                    ctx->saved_data["other"] = other;
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto self = saved[0];
                    auto other = ctx->saved_data["other"].toScalar();
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grad_self = detail::_prf_div__backward(
                            grad_output[0],
                            self,
                            other,
                            mode);

                    return {
                            grad_self,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class IZeroLeftDivideInplaceScalarFunction
                    : public torch::autograd::Function<IZeroLeftDivideInplaceScalarFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        torch::autograd::Variable &self,
                        const at::Scalar &other,
                        c10::string_view mode) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    ctx->mark_dirty({self});
                    auto output = prf_ldiv_(
                            self,
                            other,
                            mode);

                    ctx->save_for_backward({self});
                    ctx->saved_data["other"] = other;
                    ctx->saved_data["mode"] = mode;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {

                    auto saved = ctx->get_saved_variables();
                    auto self = saved[0];
                    auto other = ctx->saved_data["other"].toScalar();
                    auto mode = ctx->saved_data["mode"].toStringView();

                    auto grad_self = detail::_prf_ldiv__backward(
                            grad_output[0],
                            self,
                            other,
                            mode);

                    return {
                            grad_self,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };
        } // namespace

        at::Tensor prf_div_autograd(
                const at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode) {
            return PRFDivideFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv_autograd(
                const at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode) {
            return IZeroLeftDivideFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_div_Scalar_autograd(
                const at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode) {
            return PRFDivideScalarFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv_Scalar_autograd(
                const at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode) {
            return IZeroLeftDivideScalarFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_div_rScalar_autograd(
                const at::Scalar &self,
                const at::Tensor &other,
                c10::string_view mode) {
            return PRFDividerScalarFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv_rScalar_autograd(
                const at::Scalar &self,
                const at::Tensor &other,
                c10::string_view mode) {
            return IZeroLeftDividerScalarFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_div__autograd(
                at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode) {
            return PRFDivideInplaceFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv__autograd(
                at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode) {
            return IZeroLeftDivideInplaceFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_div__Scalar_autograd(
                at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode) {
            return PRFDivideInplaceScalarFunction::apply(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv__Scalar_autograd(
                at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode) {
            return IZeroLeftDivideInplaceScalarFunction::apply(
                    self,
                    other,
                    mode);
        }

        TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div"),
                    TORCH_FN(prf_div_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv"),
                    TORCH_FN(prf_ldiv_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div.Scalar"),
                    TORCH_FN(prf_div_Scalar_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv.Scalar"),
                    TORCH_FN(prf_ldiv_Scalar_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div.rScalar"),
                    TORCH_FN(prf_div_rScalar_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv.rScalar"),
                    TORCH_FN(prf_ldiv_rScalar_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div_"),
                    TORCH_FN(prf_div__autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv_"),
                    TORCH_FN(prf_ldiv__autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div_.Scalar"),
                    TORCH_FN(prf_div__Scalar_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv_.Scalar"),
                    TORCH_FN(prf_ldiv__Scalar_autograd));
        }
    }
}
