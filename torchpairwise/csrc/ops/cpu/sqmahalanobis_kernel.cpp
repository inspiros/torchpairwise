#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "../utils/dispatch.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            namespace impl {
                template<typename scalar_t, typename index_t>
                void _sqmahalanobis_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 3> VI,
                        at::TensorAccessor<scalar_t, 3> output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t val = 0, diff_k, diff_l;
                        for (index_t k = 0; k < x1.size(2); k++) {
                            for (index_t l = 0; l < x2.size(2); l++) {
                                diff_k = x1[b][i][k] - x2[b][j][k];
                                diff_l = x1[b][i][l] - x2[b][j][l];
                                val += diff_k * VI[b][k][l] * diff_l;
                            }
                        }
                        output[b][i][j] = val;
                    }
                }
            } // namespace impl

            at::Tensor _sqmahalanobis_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    const at::Tensor &VI) {
                at::CheckedFrom c = "_sqmahalanobis_forward";
                auto args = {
                        at::TensorArg(x1, "x1", 1),
                        at::TensorArg(x2, "x2", 2),
                        at::TensorArg(VI, "VI", 3)};
                at::checkAllSameType(c, args);

                bool unbatched = x1.ndimension() == 2;
                TORCH_CHECK(unbatched || x1.ndimension() == 3,
                            "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || x2.ndimension() == 3,
                            "x2 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || VI.ndimension() == 3,
                            "VI must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || (x1.size(0) == x2.size(0) && x1.size(0) == VI.size(0)),
                            "batch_size of x1, x2, and VI do not match.")
                TORCH_CHECK((unbatched && x1.size(1) == x2.size(1) && x1.size(1) == VI.size(0)) ||
                            (!unbatched && x1.size(2) == x2.size(2) && x1.size(2) == VI.size(1)),
                            "feature dimension of x1, x2, and VI do not match.")

                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                auto VI_c = VI.contiguous();
                if (unbatched) {
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                    VI_c = VI_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                int64_t n_kernels = batch_size * x1_c.size(1) * x2_c.size(1);
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1_c.options());

                AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                                x1_c.scalar_type(), "_sqmahalanobis_forward_cpu", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<scalar_t, 3>();
                        impl::_sqmahalanobis_forward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                VI_c.accessor<scalar_t, 3>(),
                                output_accessor);
                    }));
                }));
                if (unbatched)
                    output.squeeze_(0);
                return output;
            }

            namespace impl {
                template<typename scalar_t, typename index_t>
                void _sqmahalanobis_backward_x1_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 3> VI,
                        at::TensorAccessor<scalar_t, 3> grad_x1) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t i = index % x1.size(1);
                        index_t b = index / x1.size(1);

                        for (index_t j = 0; j < x2.size(1); j++) {
                            for (index_t k = 0; k < x1.size(2); k++) {
                                for (index_t l = 0; l < x2.size(2); l++) {
                                    grad_x1[b][i][k] +=
                                            grad_output[b][i][j] * VI[b][k][l] * (x1[b][i][l] - x2[b][j][l]);
                                    grad_x1[b][i][l] +=
                                            grad_output[b][i][j] * (x1[b][i][k] - x2[b][j][k]) * VI[b][k][l];
                                }
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _sqmahalanobis_backward_x2_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        const at::TensorAccessor<scalar_t, 3> VI,
                        at::TensorAccessor<scalar_t, 3> grad_x2) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t b = index / x2.size(1);

                        for (index_t i = 0; i < x1.size(1); i++) {
                            for (index_t k = 0; k < x1.size(2); k++) {
                                for (index_t l = 0; l < x2.size(2); l++) {
                                    grad_x2[b][j][k] +=
                                            -grad_output[b][i][j] * VI[b][k][l] * (x1[b][i][l] - x2[b][j][l]);
                                    grad_x2[b][j][l] +=
                                            -grad_output[b][i][j] * (x1[b][i][k] - x2[b][j][k]) * VI[b][k][l];
                                }
                            }
                        }
                    }
                }

                template<typename scalar_t, typename index_t>
                void _sqmahalanobis_backward_VI_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> grad_VI) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t l = index % x2.size(2);
                        index_t k = (index / x2.size(2)) % x1.size(2);
                        index_t b = index / (x2.size(2) * x1.size(2));

                        scalar_t val = 0;
                        for (index_t i = 0; i < x1.size(1); i++) {
                            for (index_t j = 0; j < x2.size(1); j++) {
                                val += grad_output[b][i][j] *
                                        (x1[b][i][k] - x2[b][j][k]) *
                                        (x1[b][i][l] - x2[b][j][l]);
                            }
                        }
                        grad_VI[b][k][l] += val;
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor, at::Tensor> _sqmahalanobis_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    const at::Tensor &VI) {
                bool unbatched = x1.ndimension() == 2;

                auto grad_output_c = grad_output.contiguous();
                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                auto VI_c = VI.contiguous();
                if (unbatched) {
                    grad_output_c = grad_output_c.unsqueeze(0);
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                    VI_c = VI_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                int64_t n_kernels;
                auto grad_x1 = at::zeros_like(x1_c);
                auto grad_x2 = at::zeros_like(x2_c);
                auto grad_VI = at::zeros_like(VI_c);

                AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                                grad_output_c.scalar_type(), "_sqmahalanobis_backward_cpu", ([&] {
                    n_kernels = batch_size * x1_c.size(1);
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x1_accessor =
                                grad_x1.accessor<scalar_t, 3>();
                        impl::_sqmahalanobis_backward_x1_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                VI_c.accessor<scalar_t, 3>(),
                                grad_x1_accessor);
                    }));

                    n_kernels = batch_size * x2_c.size(1);
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x2_accessor =
                                grad_x2.accessor<scalar_t, 3>();
                        impl::_sqmahalanobis_backward_x2_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                VI_c.accessor<scalar_t, 3>(),
                                grad_x2_accessor);
                    }));

                    n_kernels = grad_VI.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_VI_accessor =
                                grad_VI.accessor<scalar_t, 3>();
                        impl::_sqmahalanobis_backward_VI_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                grad_VI_accessor);
                    }));
                }));
                if (unbatched) {
                    grad_x1.squeeze_(0);
                    grad_x2.squeeze_(0);
                    grad_VI.squeeze_(0);
                }
                return std::make_tuple(grad_x1, grad_x2, grad_VI);
            }
        }

        TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_sqmahalanobis"),
                    TORCH_FN(_sqmahalanobis_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::__sqmahalanobis_backward"),
                    TORCH_FN(_sqmahalanobis_backward_kernel));
        }
    }
}
