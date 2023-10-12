#include <torch/types.h>

#include "prf_div.h"

namespace torchpairwise {
    namespace ops {
        at::Tensor prf_div(
                const at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_div.Tensor")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_div", "")
                                     .typed < at::Tensor(
            const at::Tensor &, const at::Tensor &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv(
                const at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_ldiv.Tensor")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_ldiv", "")
                                     .typed < at::Tensor(
            const at::Tensor &, const at::Tensor &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_div(
                const at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_div.Scalar")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_div", "Scalar")
                                     .typed < at::Tensor(
            const at::Tensor &, const at::Scalar &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv(
                const at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_ldiv.Scalar")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_ldiv", "Scalar")
                                     .typed < at::Tensor(
            const at::Tensor &, const at::Scalar &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_div(
                const at::Scalar &self,
                const at::Tensor &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_div.rScalar")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_div", "rScalar")
                                     .typed < at::Tensor(
            const at::Scalar &, const at::Tensor &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv(
                const at::Scalar &self,
                const at::Tensor &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_ldiv.rScalar")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_ldiv", "rScalar")
                                     .typed < at::Tensor(
            const at::Scalar &, const at::Tensor &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_div_(
                at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_div_.Tensor")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_div_", "")
                                     .typed < at::Tensor(at::Tensor & ,
            const at::Tensor &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv_(
                at::Tensor &self,
                const at::Tensor &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_ldiv_.Tensor")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_ldiv_", "")
                                     .typed < at::Tensor(at::Tensor & ,
            const at::Tensor &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_div_(
                at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_div_.Scalar")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_div_", "Scalar")
                                     .typed < at::Tensor(at::Tensor & ,
            const at::Scalar &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        at::Tensor prf_ldiv_(
                at::Tensor &self,
                const at::Scalar &other,
                c10::string_view mode) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.prf_div.prf_ldiv_.Scalar")
            static auto op = c10::Dispatcher::singleton()
                                     .findSchemaOrThrow("torchpairwise::prf_ldiv_", "Scalar")
                                     .typed < at::Tensor(at::Tensor & ,
            const at::Scalar &, c10::string_view)>();
            return op.call(
                    self,
                    other,
                    mode);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _prf_div_backward(
                    const at::Tensor &grad,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                using signature = std::tuple<at::Tensor, at::Tensor>(const at::Tensor &,
                                                                     const at::Tensor &,
                                                                     const at::Tensor &,
                                                                     c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_div_backward", "")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            std::tuple<at::Tensor, at::Tensor> _prf_ldiv_backward(
                    const at::Tensor &grad,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                using signature = std::tuple<at::Tensor, at::Tensor>(const at::Tensor &,
                                                                     const at::Tensor &,
                                                                     const at::Tensor &,
                                                                     c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_ldiv_backward", "")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            at::Tensor _prf_div_backward(
                    const at::Tensor &grad,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode) {
                using signature = at::Tensor(const at::Tensor &,
                                             const at::Tensor &,
                                             const at::Scalar &,
                                             c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_div_backward", "Scalar")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            at::Tensor _prf_ldiv_backward(
                    const at::Tensor &grad,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode) {
                using signature = at::Tensor(const at::Tensor &,
                                             const at::Tensor &,
                                             const at::Scalar &,
                                             c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_ldiv_backward", "Scalar")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            at::Tensor _prf_div_backward(
                    const at::Tensor &grad,
                    const at::Scalar &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                using signature = at::Tensor(const at::Tensor &,
                                             const at::Scalar &,
                                             const at::Tensor &,
                                             c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_div_backward", "rScalar")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            at::Tensor _prf_ldiv_backward(
                    const at::Tensor &grad,
                    const at::Scalar &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                using signature = at::Tensor(const at::Tensor &,
                                             const at::Scalar &,
                                             const at::Tensor &,
                                             c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_ldiv_backward", "rScalar")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            std::tuple<at::Tensor, at::Tensor> _prf_div__backward(
                    const at::Tensor &grad,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                using signature = std::tuple<at::Tensor, at::Tensor>(const at::Tensor &,
                                                                     const at::Tensor &,
                                                                     const at::Tensor &,
                                                                     c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_div__backward", "")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            std::tuple<at::Tensor, at::Tensor> _prf_ldiv__backward(
                    const at::Tensor &grad,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                using signature = std::tuple<at::Tensor, at::Tensor>(const at::Tensor &,
                                                                     const at::Tensor &,
                                                                     const at::Tensor &,
                                                                     c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_ldiv__backward", "")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            at::Tensor _prf_div__backward(
                    const at::Tensor &grad,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode) {
                using signature = at::Tensor(const at::Tensor &,
                                             const at::Tensor &,
                                             const at::Scalar &,
                                             c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_div__backward", "Scalar")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }

            at::Tensor _prf_ldiv__backward(
                    const at::Tensor &grad,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode) {
                using signature = at::Tensor(const at::Tensor &,
                                             const at::Tensor &,
                                             const at::Scalar &,
                                             c10::string_view);
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("torchpairwise::_prf_ldiv__backward", "Scalar")
                                .typed<signature>();
                return op.call(
                        grad,
                        self,
                        other,
                        mode);
            }
        } // namespace detail

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_div(Tensor input, Tensor other, str mode=\"zero\") -> Tensor"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_ldiv(Tensor input, Tensor other, str mode=\"zero\") -> Tensor"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_div.Scalar(Tensor input, Scalar other, str mode=\"zero\") -> Tensor"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_ldiv.Scalar(Tensor input, Scalar other, str mode=\"zero\") -> Tensor"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_div.rScalar(Scalar input, Tensor other, str mode=\"zero\") -> Tensor"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_ldiv.rScalar(Scalar input, Tensor other, str mode=\"zero\") -> Tensor"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_div_(Tensor(a!) input, Tensor other, str mode=\"zero\") -> Tensor(a!)"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_ldiv_(Tensor(a!) input, Tensor other, str mode=\"zero\") -> Tensor(a!)"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_div_.Scalar(Tensor(a!) input, Scalar other, str mode=\"zero\") -> Tensor(a!)"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::prf_ldiv_.Scalar(Tensor(a!) input, Scalar other, str mode=\"zero\") -> Tensor(a!)"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_div_backward(Tensor grad, Tensor self, Tensor other, str mode=\"zero\") -> (Tensor grad_self, Tensor grad_other)"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_ldiv_backward(Tensor grad, Tensor self, Tensor other, str mode=\"zero\") -> (Tensor grad_self, Tensor grad_other)"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_div_backward.Scalar(Tensor grad, Tensor self, Scalar other, str mode=\"zero\") -> Tensor grad_self"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_ldiv_backward.Scalar(Tensor grad, Tensor self, Scalar other, str mode=\"zero\") -> Tensor grad_self"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_div_backward.rScalar(Tensor grad, Scalar self, Tensor other, str mode=\"zero\") -> Tensor grad_other"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_ldiv_backward.rScalar(Tensor grad, Scalar self, Tensor other, str mode=\"zero\") -> Tensor grad_other"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_div__backward(Tensor grad, Tensor self, Tensor other, str mode=\"zero\") -> (Tensor grad_self, Tensor grad_other)"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_ldiv__backward(Tensor grad, Tensor self, Tensor other, str mode=\"zero\") -> (Tensor grad_self, Tensor grad_other)"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_div__backward.Scalar(Tensor grad, Tensor self, Scalar other, str mode=\"zero\") -> Tensor grad_self"));
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "torchpairwise::_prf_ldiv__backward.Scalar(Tensor grad, Tensor self, Scalar other, str mode=\"zero\") -> Tensor grad_self"));
        }
    } // namespace ops
} // namespace torchpairwise
