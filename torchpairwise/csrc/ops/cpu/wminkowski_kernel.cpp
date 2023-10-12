#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "signum.h"
#include "../utils/dispatch.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            namespace impl {
                template<bool backward = false, typename scalar_t, typename index_t>
                void _wminkowski_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> w,
                        scalar_t p,
                        at::TensorAccessor<scalar_t, 3> output) {
                    scalar_t r_p = 1 / p;
                    if constexpr (backward)
                        r_p -= 1;
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t val = 0;
                        for (index_t k = 0; k < x1.size(2); k++)
                            val += pow(fabs(x1[b][i][k] - x2[b][j][k]), p) * w[b][k];
                        output[b][i][j] = pow(val, r_p);
                    }
                }

                template<bool neg = false, typename scalar_t, typename index_t>
                void _wminkowski_inf_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> w,
                        at::TensorAccessor<scalar_t, 3> output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t val = fabs(x1[b][i][0] - x2[b][j][0]) * w[b][0], tmp;
                        for (index_t k = 1; k < x1.size(2); k++) {
                            tmp = fabs(x1[b][i][k] - x2[b][j][k]) * w[b][k];
                            if constexpr (neg) {
                                if (tmp < val)
                                    val = tmp;
                            } else {
                                if (tmp > val)
                                    val = tmp;
                            }
                        }
                        output[b][i][j] = val;
                    }
                }
            } // namespace impl

            at::Tensor _wminkowski_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    const at::Tensor &w,
                    double p) {
                at::CheckedFrom c = "_wminkowski_forward";
                auto args = {
                        at::TensorArg(x1, "x1", 1),
                        at::TensorArg(x2, "x2", 2),
                        at::TensorArg(w, "w", 3)};
                at::checkAllSameType(c, args);

                bool unbatched = x1.ndimension() == 2;
                TORCH_CHECK(unbatched || x1.ndimension() == 3,
                            "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || x2.ndimension() == 3,
                            "x2 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || w.ndimension() == 2,
                            "w must be 1-D (unbatched) or 2-D (batched) tensor.")
                TORCH_CHECK(unbatched || (x1.size(0) == x2.size(0) && x1.size(0) == w.size(0)),
                            "batch_size of x1, x2, and w do not match.")
                TORCH_CHECK((unbatched && x1.size(1) == x2.size(1) && x1.size(1) == w.size(0)) ||
                            (!unbatched && x1.size(2) == x2.size(2) && x1.size(1) == w.size(1)),
                            "feature dimension of x1, x2, and w do not match.")

                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                auto w_c = w.contiguous();
                if (unbatched) {
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                    w_c = w_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                int64_t n_kernels = batch_size * x1_c.size(1) * x2_c.size(1);
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1_c.options());

                if (p == 0) {
                    output.fill_(x1_c.size(2));
                } else {
                    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                                    x1_c.scalar_type(), "_wminkowski_forward_cpu", ([&] {
                        TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                            auto output_accessor =
                                    output.accessor<scalar_t, 3>();
                            if (std::isinf(p)) {
                                TORCHPAIRWISE_DISPATCH_BOOL_NAME(negative, p < 0, ([&] {
                                    impl::_wminkowski_inf_kernel_impl<negative, scalar_t, index_t>(
                                            n_kernels,
                                            x1_c.accessor<scalar_t, 3>(),
                                            x2_c.accessor<scalar_t, 3>(),
                                            w_c.accessor<scalar_t, 2>(),
                                            output_accessor);
                                }));
                            } else {
                                impl::_wminkowski_kernel_impl<false, scalar_t, index_t>(
                                        n_kernels,
                                        x1_c.accessor<scalar_t, 3>(),
                                        x2_c.accessor<scalar_t, 3>(),
                                        w_c.accessor<scalar_t, 2>(),
                                        static_cast<scalar_t>(p),
                                        output_accessor);
                            }
                        }));
                    }));
                }
                if (unbatched)
                    output.squeeze_(0);
                return output;
            }

            namespace impl {
                template<typename scalar_t, typename index_t>
                void _wminkowski_backward_x1_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> w,
                        scalar_t p,
                        at::TensorAccessor<scalar_t, 3> grad_x1) {
                    scalar_t p_minus_1 = p - 1;
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t i = index % x1.size(1);
                        index_t b = index / x1.size(1);

                        scalar_t val;
                        for (index_t j = 0; j < x2.size(1); j++) {
                            for (index_t k = 0; k < x1.size(2); k++) {
                                val = x1[b][i][k] - x2[b][j][k];
                                grad_x1[b][i][k] += grad_output[b][i][j] * pow(fabs(val), p_minus_1) * w[b][k] *
                                                    output[b][i][j] * m_signum(val);
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _wminkowski_backward_x2_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> w,
                        scalar_t p,
                        at::TensorAccessor<scalar_t, 3> grad_x2) {
                    scalar_t p_minus_1 = p - 1;
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t b = index / x2.size(1);

                        scalar_t val;
                        for (index_t i = 0; i < x1.size(1); i++) {
                            for (index_t k = 0; k < x1.size(2); k++) {
                                val = x2[b][j][k] - x1[b][i][k];
                                grad_x2[b][j][k] += grad_output[b][i][j] * pow(fabs(val), p_minus_1) * w[b][k] *
                                                    output[b][i][j] * m_signum(val);
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _wminkowski_backward_v_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> w,
                        scalar_t p,
                        at::TensorAccessor<scalar_t, 2> grad_w) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % w.size(1);
                        index_t b = index / w.size(1);

                        scalar_t val;
                        for (index_t i = 0; i < x1.size(1); i++) {
                            for (index_t j = 0; j < x2.size(1); j++) {
                                val = x2[b][j][k] - x1[b][i][k];
                                grad_w[b][k] += grad_output[b][i][j] * pow(fabs(val), p) / p *
                                                output[b][i][j];
                            }
                        }
                    }
                }

                template<bool neg = false, typename scalar_t, typename index_t>
                void _wminkowski_inf_backward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 2> w,
                        at::TensorAccessor<scalar_t, 3> grad_x1,
                        at::TensorAccessor<scalar_t, 3> grad_x2,
                        at::TensorAccessor<scalar_t, 2> grad_w) {
                    CPU_1D_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t val = x1[b][i][0] - x2[b][j][0], fabs_val = fabs(val), w_val = fabs_val * w[b][0];
                        scalar_t tmp, fabs_tmp, w_tmp;
                        index_t val_k = 0;
                        for (index_t k = 1; k < x1.size(2); k++) {
                            tmp = x1[b][i][k] - x2[b][j][k];
                            fabs_tmp = fabs(tmp);
                            w_tmp = fabs_tmp * w[b][k];
                            if constexpr (neg) {
                                if (w_tmp < w_val) {
                                    val = tmp;
                                    fabs_val = fabs_tmp;
                                    w_val = w_tmp;
                                    val_k = k;
                                }
                            } else {
                                if (w_tmp > w_val) {
                                    val = tmp;
                                    fabs_val = fabs_tmp;
                                    w_val = w_tmp;
                                    val_k = k;
                                }
                            }
                        }
                        scalar_t sgn_val = m_signum(val);
                        grad_x1[b][i][val_k] += grad_output[b][i][j] * w[b][val_k] * sgn_val;
                        grad_x2[b][j][val_k] += grad_output[b][i][j] * w[b][val_k] * -sgn_val;
                        grad_w[b][val_k] += grad_output[b][i][j] * fabs_val;
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor, at::Tensor> _wminkowski_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    const at::Tensor &w,
                    double p) {
                bool unbatched = x1.ndimension() == 2;

                auto grad_output_c = grad_output.contiguous();
                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                auto w_c = w.contiguous();
                if (unbatched) {
                    grad_output_c = grad_output_c.unsqueeze(0);
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                    w_c = w_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                int64_t n_kernels;
                auto grad_x1 = at::zeros_like(x1_c);
                auto grad_x2 = at::zeros_like(x2_c);
                auto grad_w = at::zeros_like(w_c);

                if (p != 0) {
                    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                                    grad_output_c.scalar_type(), "_wminkowski_backward_cpu", ([&] {
                        if (std::isinf(p)) {
                            n_kernels = grad_output_c.numel();
                            TORCHPAIRWISE_DISPATCH_INDEX_TYPE(std::max(n_kernels, batch_size *
                                                                          std::max(x1_c.size(1), x2_c.size(1))), ([&] {
                                TORCHPAIRWISE_DISPATCH_BOOL_NAME(negative, p < 0, ([&] {
                                    auto grad_x1_accessor =
                                            grad_x1.accessor<scalar_t, 3>();
                                    auto grad_x2_accessor =
                                            grad_x2.accessor<scalar_t, 3>();
                                    auto grad_w_accessor =
                                            grad_w.accessor<scalar_t, 2>();
                                    impl::_wminkowski_inf_backward_kernel_impl<negative, scalar_t, index_t>(
                                            n_kernels,
                                            grad_output_c.accessor<scalar_t, 3>(),
                                            x1_c.accessor<scalar_t, 3>(),
                                            x2_c.accessor<scalar_t, 3>(),
                                            w_c.accessor<scalar_t, 2>(),
                                            grad_x1_accessor,
                                            grad_x2_accessor,
                                            grad_w_accessor);
                                }));
                            }));
                        } else {
                            auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                                    grad_output.options());
                            n_kernels = output.numel();
                            TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                                auto output_accessor =
                                        output.accessor<scalar_t, 3>();
                                impl::_wminkowski_kernel_impl<true, scalar_t, index_t>(
                                        n_kernels,
                                        x1_c.accessor<scalar_t, 3>(),
                                        x2_c.accessor<scalar_t, 3>(),
                                        w_c.accessor<scalar_t, 2>(),
                                        static_cast<scalar_t>(p),
                                        output_accessor);
                            }));

                            n_kernels = batch_size * x1_c.size(1);
                            TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                                auto grad_x1_accessor =
                                        grad_x1.accessor<scalar_t, 3>();
                                impl::_wminkowski_backward_x1_kernel_impl<scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_c.accessor<scalar_t, 3>(),
                                        output.accessor<scalar_t, 3>(),
                                        x1_c.accessor<scalar_t, 3>(),
                                        x2_c.accessor<scalar_t, 3>(),
                                        w_c.accessor<scalar_t, 2>(),
                                        static_cast<scalar_t>(p),
                                        grad_x1_accessor);
                            }));

                            n_kernels = batch_size * x2_c.size(1);
                            TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                                auto grad_x2_accessor =
                                        grad_x2.accessor<scalar_t, 3>();
                                impl::_wminkowski_backward_x2_kernel_impl<scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_c.accessor<scalar_t, 3>(),
                                        output.accessor<scalar_t, 3>(),
                                        x1_c.accessor<scalar_t, 3>(),
                                        x2_c.accessor<scalar_t, 3>(),
                                        w_c.accessor<scalar_t, 2>(),
                                        static_cast<scalar_t>(p),
                                        grad_x2_accessor);
                            }));

                            n_kernels = batch_size * w_c.size(1);
                            TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                                auto grad_v_accessor =
                                        grad_w.accessor<scalar_t, 2>();
                                impl::_wminkowski_backward_v_kernel_impl<scalar_t, index_t>(
                                        n_kernels,
                                        grad_output_c.accessor<scalar_t, 3>(),
                                        output.accessor<scalar_t, 3>(),
                                        x1_c.accessor<scalar_t, 3>(),
                                        x2_c.accessor<scalar_t, 3>(),
                                        w_c.accessor<scalar_t, 2>(),
                                        static_cast<scalar_t>(p),
                                        grad_v_accessor);
                            }));
                        }
                    }));
                }
                if (unbatched) {
                    grad_x1.squeeze_(0);
                    grad_x2.squeeze_(0);
                    grad_w.squeeze_(0);
                }
                return std::make_tuple(grad_x1, grad_x2, grad_w);
            }
        }

        TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_wminkowski"),
                    TORCH_FN(_wminkowski_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::__wminkowski_backward"),
                    TORCH_FN(_wminkowski_backward_kernel));
        }
    }
}
