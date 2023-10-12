#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "prf_divide.h"
#include "../utils/dispatch.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            static constexpr auto default_result_type = at::kFloat;

            namespace impl {
                template<PRFDivMode mode = Zero, typename scalar_t, typename index_t>
                C10_ALWAYS_INLINE void prf_div_forward_kernel_impl(
                        index_t n_kernels,
                        const scalar_t *self,
                        const scalar_t *other,
                        scalar_t *output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        output[index] = prf_divide<mode>(self[index], other[index]);
                    }
                }

                template<PRFDivMode mode = Zero, typename scalar_t, typename index_t>
                C10_ALWAYS_INLINE void prf_div_forward_kernel_impl(
                        index_t n_kernels,
                        scalar_t self,
                        const scalar_t *other,
                        scalar_t *output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        output[index] = prf_divide<mode>(self, other[index]);
                    }
                }

                template<PRFDivMode mode = Zero, typename scalar_t, typename index_t>
                C10_ALWAYS_INLINE void prf_div_forward_kernel_impl(
                        index_t n_kernels,
                        const scalar_t *self,
                        scalar_t other,
                        scalar_t *output) {
                    if (other == static_cast<scalar_t>(0)) {
                        CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                            if constexpr (mode == Zero)
                                output[index] = static_cast<scalar_t>(0);
                            else if constexpr (mode == Identity)
                                output[index] = self[index];
                        }
                    } else {
                        CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                            output[index] = self[index] / other;
                        }
                    }
                }
            }  // namespace impl

            template<bool left = false>
            at::Tensor prf_div_forward_kernel(
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                TORCH_CHECK(self.sizes() == other.sizes(),
                            "Broadcasting semantic is not implemented for prf_div. "
                            "Expect self's shape to be identical to other's shape. Got self.shape=",
                            self.sizes(),
                            ", while other.shape=",
                            other.sizes())

                auto result_type = at::result_type(self, other);
                if (at::isIntegralType(result_type, true))
                    result_type = default_result_type;
                auto self_c = self.contiguous().to(result_type);
                auto other_c = other.contiguous().to(result_type);

                auto output = at::empty_like(self_c);
                int64_t n_kernels = output.numel();

                auto self_flatten = self_c.flatten();
                auto other_flatten = other_c.flatten();
                auto output_flatten = output.flatten();

                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::kHalf, at::kBFloat16, result_type, "prf_div", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(mode, ([&] {
                            if constexpr (left) {
                                impl::prf_div_forward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        other_flatten.data_ptr<scalar_t>(),
                                        self_flatten.data_ptr<scalar_t>(),
                                        output_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::prf_div_forward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        self_flatten.data_ptr<scalar_t>(),
                                        other_flatten.data_ptr<scalar_t>(),
                                        output_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                }));
                return output;
            }

            template<bool left = false>
            at::Tensor prf_div_Scalar_forward_kernel(
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode) {
                auto result_type = at::result_type(self, other);
                if (at::isIntegralType(result_type, true))
                    result_type = default_result_type;
                auto self_c = self.contiguous().to(result_type);

                auto output = at::empty_like(self_c);
                int64_t n_kernels = output.numel();

                auto self_flatten = self_c.flatten();
                auto output_flatten = output.flatten();

                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::kHalf, at::kBFloat16, result_type, "prf_div_Scalar", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(mode, ([&] {
                            if constexpr (left) {
                                impl::prf_div_forward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        other.to<scalar_t>(),
                                        self_flatten.data_ptr<scalar_t>(),
                                        output_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::prf_div_forward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        self_flatten.data_ptr<scalar_t>(),
                                        other.to<scalar_t>(),
                                        output_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                }));
                return output;
            }

            template<bool left = false>
            at::Tensor prf_div_rScalar_forward_kernel(
                    const at::Scalar &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                return prf_div_Scalar_forward_kernel<!left>(other, self, mode);
            }

            template<bool left = false>
            at::Tensor prf_div__forward_kernel(
                    at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                TORCH_CHECK(self.sizes() == other.sizes(),
                            "Broadcasting semantic is not implemented for prf_div_. "
                            "Expect self's shape to be identical to other's shape. Got self.shape=",
                            self.sizes(),
                            ", while other.shape=",
                            other.sizes())

                auto result_type = at::result_type(self, other);
                if (at::isIntegralType(result_type, true))
                    result_type = default_result_type;
                TORCH_CHECK(at::can_cast(result_type, self.scalar_type()),
                            "result type ",
                            result_type,
                            " can't be cast to the desired output type ",
                            self.scalar_type())
                auto self_c = self.contiguous().to(result_type);
                auto other_c = other.contiguous().to(result_type);

                bool needs_cast = !self.is_contiguous() || result_type != self.scalar_type();
                auto output = needs_cast ? at::empty_like(self_c) : self_c;
                int64_t n_kernels = self.numel();

                auto output_flatten = output.flatten();
                auto self_flatten = self_c.flatten();
                auto other_flatten = other_c.flatten();

                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::kHalf, at::kBFloat16, result_type, "prf_div_", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(mode, ([&] {
                            if constexpr (left) {
                                impl::prf_div_forward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        other_flatten.data_ptr<scalar_t>(),
                                        self_flatten.data_ptr<scalar_t>(),
                                        output.data_ptr<scalar_t>());
                            } else {
                                impl::prf_div_forward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        self_flatten.data_ptr<scalar_t>(),
                                        other_flatten.data_ptr<scalar_t>(),
                                        output.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                }));

                if (needs_cast)
                    self.copy_(output);
                return self;
            }

            template<bool left = false>
            at::Tensor prf_div__Scalar_forward_kernel(
                    at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode) {
                auto result_type = at::result_type(self, other);
                if (at::isIntegralType(result_type, true))
                    result_type = default_result_type;
                TORCH_CHECK(at::can_cast(result_type, self.scalar_type()),
                            "result type ",
                            result_type,
                            " can't be cast to the desired output type ",
                            self.scalar_type())
                auto self_c = self.contiguous().to(result_type);

                bool needs_cast = !self.is_contiguous();
                auto output = needs_cast ? at::empty_like(self_c) : self_c;
                int64_t n_kernels = self.numel();

                auto output_flatten = output.flatten();
                auto self_flatten = self_c.flatten();

                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::kHalf, at::kBFloat16, result_type, "prf_div__Scalar", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(mode, ([&] {
                            if constexpr (left) {
                                impl::prf_div_forward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        other.to<scalar_t>(),
                                        self_flatten.data_ptr<scalar_t>(),
                                        output.data_ptr<scalar_t>());
                            } else {
                                impl::prf_div_forward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        self_flatten.data_ptr<scalar_t>(),
                                        other.to<scalar_t>(),
                                        output.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                }));

                if (needs_cast)
                    self.copy_(output);
                return self;
            }

            namespace impl {
                template<PRFDivMode mode = Zero, typename scalar_t, typename index_t>
                C10_ALWAYS_INLINE void prf_div_backward_kernel_impl(
                        index_t n_kernels,
                        const scalar_t *grad_output,
                        const scalar_t *self,
                        const scalar_t *other,
                        scalar_t *grad_self,
                        scalar_t *grad_other) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        scalar_t o = other[index];
                        if (o != static_cast<scalar_t>(0)) {
                            scalar_t g_s = grad_output[index] / o;
                            grad_self[index] = g_s;
                            grad_other[index] = g_s * -self[index] / o;
                        } else {
                            if constexpr (mode == Identity)
                                grad_self[index] = grad_output[index];
                        }
                    }
                }

                template<PRFDivMode mode = Zero, typename scalar_t, typename index_t>
                C10_ALWAYS_INLINE void prf_div_backward_kernel_impl(
                        index_t n_kernels,
                        const scalar_t *grad_output,
                        scalar_t other,
                        scalar_t *grad_self) {
                    if (other != static_cast<scalar_t>(0)) {
                        CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                            grad_self[index] = grad_output[index] / other;
                        }
                    } else {
                        if constexpr (mode == Identity) {
                            CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                                grad_self[index] = grad_output[index];
                            }
                        }
                    }
                }

                template<PRFDivMode mode = Zero, typename scalar_t, typename index_t>
                C10_ALWAYS_INLINE void prf_div_backward_kernel_impl(
                        index_t n_kernels,
                        const scalar_t *grad_output,
                        scalar_t self,
                        const scalar_t *other,
                        scalar_t *grad_other) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        scalar_t o = other[index];
                        if (o != static_cast<scalar_t>(0)) {
                            grad_other[index] = grad_output[index] * -self / (o * o);
                        }
                    }
                }
            }  // namespace impl

            template<bool left = false>
            std::tuple<at::Tensor, at::Tensor> prf_div_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                auto result_type = grad_output.scalar_type();
                auto grad_output_c = grad_output.contiguous();
                auto self_c = self.contiguous().to(result_type);
                auto other_c = other.contiguous().to(result_type);

                int64_t n_kernels = grad_output.numel();
                auto grad_self = at::zeros_like(self_c);
                auto grad_other = at::zeros_like(other_c);

                auto grad_output_flatten = grad_output.flatten();
                auto self_flatten = self.flatten();
                auto other_flatten = other.flatten();
                auto grad_self_flatten = grad_self.flatten();
                auto grad_other_flatten = grad_other.flatten();

                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::kHalf, at::kBFloat16, result_type, "prf_div_backward", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(mode, ([&] {
                            if constexpr (left) {
                                impl::prf_div_backward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_flatten.data_ptr<scalar_t>(),
                                        other_flatten.data_ptr<scalar_t>(),
                                        self_flatten.data_ptr<scalar_t>(),
                                        grad_other_flatten.data_ptr<scalar_t>(),
                                        grad_self_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::prf_div_backward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_flatten.data_ptr<scalar_t>(),
                                        self_flatten.data_ptr<scalar_t>(),
                                        other_flatten.data_ptr<scalar_t>(),
                                        grad_self_flatten.data_ptr<scalar_t>(),
                                        grad_other_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                }));
                return std::make_tuple(grad_self.to(self.scalar_type()), grad_other.to(other.scalar_type()));
            }

            template<bool left = false>
            at::Tensor prf_div_backward_Scalar_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &self,
                    const at::Scalar &other,
                    c10::string_view mode) {
                auto result_type = grad_output.scalar_type();
                auto grad_output_c = grad_output.contiguous();
                auto self_c = self.contiguous().to(result_type);

                int64_t n_kernels = grad_output.numel();
                auto grad_self = at::zeros_like(self_c);

                auto grad_output_flatten = grad_output.flatten();
                auto self_flatten = self.flatten();
                auto grad_self_flatten = grad_self.flatten();

                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::kHalf, at::kBFloat16, result_type, "prf_div_Scalar_backward", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(mode, ([&] {
                            if constexpr (left) {  // TODO
                                impl::prf_div_backward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_flatten.data_ptr<scalar_t>(),
                                        other.to<scalar_t>(),
                                        self_flatten.data_ptr<scalar_t>(),
                                        grad_self_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::prf_div_backward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_flatten.data_ptr<scalar_t>(),
                                        other.to<scalar_t>(),
                                        grad_self_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                }));
                return grad_self.to(self.scalar_type());
            }

            template<bool left = false>
            at::Tensor prf_div_backward_rScalar_kernel(
                    const at::Tensor &grad_output,
                    const at::Scalar &self,
                    const at::Tensor &other,
                    c10::string_view mode) {
                auto result_type = grad_output.scalar_type();
                auto grad_output_c = grad_output.contiguous();
                auto other_c = other.contiguous().to(result_type);

                int64_t n_kernels = grad_output.numel();
                auto grad_other = at::zeros_like(other_c);

                auto grad_output_flatten = grad_output.flatten();
                auto other_flatten = other.flatten();
                auto grad_other_flatten = grad_other.flatten();

                AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::kHalf, at::kBFloat16, result_type, "prf_div_rScalar_backward", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(mode, ([&] {
                            if constexpr (left) {
                                impl::prf_div_backward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_flatten.data_ptr<scalar_t>(),
                                        self.to<scalar_t>(),
                                        grad_other_flatten.data_ptr<scalar_t>());
                            } else {
                                impl::prf_div_backward_kernel_impl<prf_div_mode, scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_flatten.data_ptr<scalar_t>(),
                                        self.to<scalar_t>(),
                                        other_flatten.data_ptr<scalar_t>(),
                                        grad_other_flatten.data_ptr<scalar_t>());
                            }
                        }));
                    }));
                }));
                return grad_other.to(other.scalar_type());
            }
        }

        TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div"),
                    TORCH_FN(prf_div_forward_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv"),
                    TORCH_FN(prf_div_forward_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div.Scalar"),
                    TORCH_FN(prf_div_Scalar_forward_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv.Scalar"),
                    TORCH_FN(prf_div_Scalar_forward_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div.rScalar"),
                    TORCH_FN(prf_div_rScalar_forward_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv.rScalar"),
                    TORCH_FN(prf_div_rScalar_forward_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div_"),
                    TORCH_FN(prf_div__forward_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv_"),
                    TORCH_FN(prf_div__forward_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_div_.Scalar"),
                    TORCH_FN(prf_div__Scalar_forward_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::prf_ldiv_.Scalar"),
                    TORCH_FN(prf_div__Scalar_forward_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_div_backward"),
                    TORCH_FN(prf_div_backward_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_ldiv_backward"),
                    TORCH_FN(prf_div_backward_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_div_backward.Scalar"),
                    TORCH_FN(prf_div_backward_Scalar_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_ldiv_backward.Scalar"),
                    TORCH_FN(prf_div_backward_Scalar_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_div_backward.rScalar"),
                    TORCH_FN(prf_div_backward_rScalar_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_ldiv_backward.rScalar"),
                    TORCH_FN(prf_div_backward_rScalar_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_div__backward"),
                    TORCH_FN(prf_div_backward_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_ldiv__backward"),
                    TORCH_FN(prf_div_backward_kernel<true>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_div__backward.Scalar"),
                    TORCH_FN(prf_div_backward_Scalar_kernel<false>));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_prf_ldiv__backward.Scalar"),
                    TORCH_FN(prf_div_backward_Scalar_kernel<true>));
        }
    }
}
