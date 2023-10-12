#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "../utils/dispatch.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            namespace impl {
                template<typename scalar_t, typename index_t>
                void _haversine_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        output[b][i][j] = 2 * asin(sqrt(pow(sin((x1[b][i][0] - x2[b][j][0]) / 2), 2) +
                                                        cos(x1[b][i][0]) * cos(x2[b][j][0]) *
                                                        pow(sin((x1[b][i][1] - x2[b][j][1]) / 2), 2)));
                    }
                }
            } // namespace impl

            at::Tensor _haversine_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                at::CheckedFrom c = "_haversine_forward";
                auto args = {
                        at::TensorArg(x1, "x1", 1),
                        at::TensorArg(x2, "x2", 2)};
                at::checkAllSameType(c, args);

                bool unbatched = x1.ndimension() == 2;
                TORCH_CHECK(unbatched || x1.ndimension() == 3,
                            "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || x2.ndimension() == 3,
                            "x2 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || x1.size(0) == x2.size(0),
                            "batch_size of x1 and x2 do not match.")
                TORCH_CHECK((unbatched && x1.size(1) == 2 && x2.size(1) == 2) ||
                            (!unbatched && x1.size(2) == 2 && x2.size(2) == 2),
                            "feature dimension of x1 and x2 must be 2.")

                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                if (unbatched) {
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                int64_t n_kernels = batch_size * x1_c.size(1) * x2_c.size(1);
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1_c.options());

                AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                                x1_c.scalar_type(), "_haversine_forward_cpu", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<scalar_t, 3>();
                        impl::_haversine_forward_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                output_accessor);
                    }));
                }));
                if (unbatched)
                    output.squeeze_(0);
                return output;
            }

            namespace impl {
                template<typename scalar_t, typename index_t>
                void _haversine_backward_denom_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> denom) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        scalar_t val = pow(sin((x1[b][i][0] - x2[b][j][0]) / 2), 2) +
                                       cos(x1[b][i][0]) * cos(x2[b][j][0]) *
                                       pow(sin((x1[b][i][1] - x2[b][j][1]) / 2), 2);
                        denom[b][i][j] = sqrt(val) * sqrt(1 - val);
                    }
                }

                template<typename scalar_t, typename index_t>
                void _haversine_backward_x1_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> denom,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> grad_x1) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t i = index % x1.size(1);
                        index_t b = index / x1.size(1);

                        scalar_t grad_val, lat_val = 0, lon_val = 0;
                        for (index_t j = 0; j < x2.size(1); j++) {
                            grad_val = grad_output[b][i][j] / denom[b][i][j];
                            lat_val += grad_val * (
                                    -sin(x1[b][i][0]) * pow(sin((x1[b][i][1] - x2[b][j][1]) / 2), 2) *
                                    cos(x2[b][j][0]) + sin(x1[b][i][0] - x2[b][j][0]) / 2);
                            lon_val += grad_val * (
                                    sin((x1[b][i][1] - x2[b][j][1]) / 2) * cos(x1[b][i][0]) * cos(x2[b][j][0]) *
                                    cos((x1[b][i][1] - x2[b][j][1]) / 2));
                        }
                        grad_x1[b][i][0] += lat_val;
                        grad_x1[b][i][1] += lon_val;
                    }
                }

                template<typename scalar_t, typename index_t>
                void _haversine_backward_x2_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> grad_output,
                        const at::TensorAccessor<scalar_t, 3> denom,
                        const at::TensorAccessor<scalar_t, 3> x1,
                        const at::TensorAccessor<scalar_t, 3> x2,
                        at::TensorAccessor<scalar_t, 3> grad_x2) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t b = index / x2.size(1);

                        scalar_t grad_val, lat_val = 0, lon_val = 0;
                        for (index_t i = 0; i < x1.size(1); i++) {
                            grad_val = grad_output[b][i][j] / denom[b][i][j];
                            lat_val += grad_val * (
                                    -sin(x2[b][j][0]) * pow(sin((x1[b][i][1] - x2[b][j][1]) / 2), 2) *
                                    cos(x1[b][i][0]) - sin(x1[b][i][0] - x2[b][j][0]) / 2);
                            lon_val += grad_val * (
                                    -sin((x1[b][i][1] - x2[b][j][1]) / 2) * cos(x1[b][i][0]) * cos(x2[b][j][0]) *
                                    cos((x1[b][i][1] - x2[b][j][1]) / 2));
                        }
                        grad_x2[b][j][0] += lat_val;
                        grad_x2[b][j][1] += lon_val;
                    }
                }
            } // namespace impl

            std::tuple<at::Tensor, at::Tensor> _haversine_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                bool unbatched = x1.ndimension() == 2;

                auto grad_output_c = grad_output.contiguous();
                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                if (unbatched) {
                    grad_output_c = grad_output_c.unsqueeze(0);
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                int64_t n_kernels;
                auto grad_x1 = at::zeros_like(x1_c);
                auto grad_x2 = at::zeros_like(x2_c);

                AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                                grad_output_c.scalar_type(), "_haversine_backward_cpu", ([&] {
                    auto denom = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                           grad_output.options());
                    n_kernels = denom.numel();
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto denom_accessor =
                                denom.accessor<scalar_t, 3>();
                        impl::_haversine_backward_denom_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                denom_accessor);
                    }));

                    n_kernels = batch_size * x1_c.size(1);
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x1_accessor =
                                grad_x1.accessor<scalar_t, 3>();
                        impl::_haversine_backward_x1_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                denom.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                grad_x1_accessor);
                    }));

                    n_kernels = batch_size * x2_c.size(1);
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto grad_x2_accessor =
                                grad_x2.accessor<scalar_t, 3>();
                        impl::_haversine_backward_x2_kernel_impl<scalar_t, index_t>(
                                n_kernels,
                                grad_output_c.accessor<scalar_t, 3>(),
                                denom.accessor<scalar_t, 3>(),
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                grad_x2_accessor);
                    }));
                }));
                if (unbatched) {
                    grad_x1.squeeze_(0);
                    grad_x2.squeeze_(0);
                }
                return std::make_tuple(grad_x1, grad_x2);
            }
        }

        TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::_haversine"),
                    TORCH_FN(_haversine_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("torchpairwise::__haversine_backward"),
                    TORCH_FN(_haversine_backward_kernel));
        }
    }
}