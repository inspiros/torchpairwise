#pragma once

#include <ATen/ATen.h>

namespace torchpairwise {
    namespace ops {
        at::Tensor prf_div(
                const at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode = "zero");

        at::Tensor prf_ldiv(
                const at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode = "zero");

        at::Tensor prf_div(
                const at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode = "zero");

        at::Tensor prf_ldiv(
                const at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode = "zero");

        at::Tensor prf_div(
                const at::Scalar &self,
                const at::Tensor &other,
                c10::string_view mode = "zero");

        at::Tensor prf_ldiv(
                const at::Scalar &self,
                const at::Tensor &other,
                c10::string_view mode = "zero");

        at::Tensor prf_div_(
                at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode = "zero");

        at::Tensor prf_ldiv_(
                at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode = "zero");

        at::Tensor prf_div_(
                at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode = "zero");

        at::Tensor prf_ldiv_(
                at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode = "zero");

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _prf_div_backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode = "zero");

            std::tuple<at::Tensor, at::Tensor> _prf_ldiv_backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode = "zero");

            at::Tensor _prf_div_backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode = "zero");

            at::Tensor _prf_ldiv_backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode = "zero");

            at::Tensor _prf_div_backward(
                    const at::Tensor &grad_output,
                    const at::Scalar &self,
                    const at::Tensor &other,
                    c10::string_view mode = "zero");

            at::Tensor _prf_ldiv_backward(
                    const at::Tensor &grad_output,
                    const at::Scalar &self,
                    const at::Tensor &other,
                    c10::string_view mode = "zero");

            std::tuple<at::Tensor, at::Tensor> _prf_div__backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode = "zero");

            std::tuple<at::Tensor, at::Tensor> _prf_ldiv__backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode = "zero");

            at::Tensor _prf_div__backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode = "zero");

            at::Tensor _prf_ldiv__backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode = "zero");
        }
    }
}
