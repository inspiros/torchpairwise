#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "binary_ops.h"
#include "reduction_ops.h"

namespace torchpairwise {
    namespace ops {
        namespace {
            namespace impl {
                template<BinaryOp op, typename input_t, typename output_t, typename index_t>
                void pairwise_binary_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<input_t, 3> x1,
                        const at::TensorAccessor<input_t, 3> x2,
                        at::TensorAccessor<output_t, 4> output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t k = index % x1.size(2);
                        index_t j = (index / x1.size(2)) % x2.size(1);
                        index_t i = (index / (x1.size(2) * x2.size(1))) % x1.size(1);
                        index_t b = index / (x1.size(2) * x2.size(1) * x1.size(1));

                        output[b][i][j][k] = static_cast<output_t>(call<op>(x1[b][i][k], x2[b][j][k]));
                    }
                }

                template<BinaryOp binary_op, ReductionOp reduction_op,
                        typename output_t, typename input_t, typename index_t>
                void pairwise_binary_reduction_forward_kernel_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<input_t, 3> x1,
                        const at::TensorAccessor<input_t, 3> x2,
                        at::TensorAccessor<output_t, 3> output) {
                    CPU_1D_PARALLEL_KERNEL_LOOP(index, n_kernels) {
                        index_t j = index % x2.size(1);
                        index_t i = (index / x2.size(1)) % x1.size(1);
                        index_t b = index / (x2.size(1) * x1.size(1));

                        output_t val = identity_value<reduction_op, output_t>();
                        for (index_t k = 0; k < x1.size(2); k++) {
                            accumulate_call<reduction_op>(
                                    &val, static_cast<output_t>(call<binary_op>(x1[b][i][k], x2[b][j][k])));
                        }
                        if constexpr (reduction_op == Mean)
                            val /= static_cast<output_t>(x1.size(2));
                        output[b][i][j] = val;
                    }
                }
            } // namespace impl

            template<BinaryOp op>
            at::Tensor pairwise_binary_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                bool unbatched = x1.ndimension() == 2;
                TORCH_CHECK(x1.ndimension() >= 2,
                            "x1 must have at least 2 dimensions.")
                TORCH_CHECK(x2.ndimension() >= 2,
                            "x2 must have at least 2 dimensions.")
                TORCH_CHECK(unbatched || x1.size(0) == x2.size(0),
                            "batch_size of x1 and x2 do not match.")
                TORCH_CHECK((unbatched && x1.size(1) == x2.size(1)) ||
                            (!unbatched && [&]() {
                                if (x1.ndimension() != x2.ndimension())
                                    return false;
                                for (const auto d: c10::irange(2, x1.ndimension())) {
                                    if (x1.size(d) != x2.size(d))
                                        return false;
                                }
                                return true;
                            }()),
                            "feature dimensions of x1 and x2 do not match.")

                auto common_type = at::result_type(x1, x2);
                auto x1_c = x1.to(common_type).contiguous();
                auto x2_c = x2.to(common_type).contiguous();
                if (unbatched) {
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                }
                x1_c = x1_c.flatten(2);
                x2_c = x2_c.flatten(2);

                int64_t batch_size = x1_c.size(0);
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1), x1_c.size(2)},
                                        x1_c.options().dtype(at::kBool));
                int64_t n_kernels = output.numel();

                TORCHPAIRWISE_DISPATCH_CONSTEXPR_BINARY_OP_TYPES(
                        op, common_type, "pairwise_binary_forward_cpu", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<bool, 4>();
                        impl::pairwise_binary_forward_kernel_impl<op, scalar_t, bool, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                output_accessor);
                    }));
                }))
                if (unbatched)
                    output.squeeze_(0);
                else {
                    std::vector<int64_t> output_shape;
                    output_shape.reserve(x1.ndimension() + 1);
                    output_shape.emplace_back(batch_size);
                    output_shape.emplace_back(x1_c.size(1));
                    output_shape.emplace_back(x2_c.size(1));
                    for (const auto d: c10::irange(2, x1.ndimension())) {
                        output_shape.emplace_back(x1.size(d));
                    }
                    output = output.view(output_shape);
                }
                return output;
            }

            template<BinaryOp binary_op, ReductionOp reduction_op>
            at::Tensor pairwise_binary_reduction_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                bool unbatched = x1.ndimension() == 2;
                TORCH_CHECK(unbatched || x1.ndimension() == 3,
                            "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || x2.ndimension() == 3,
                            "x2 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || x1.size(0) == x2.size(0),
                            "batch_size of x1 and x2 do not match.")
                TORCH_CHECK((unbatched && x1.size(1) == x2.size(1)) ||
                            (!unbatched && x1.size(2) == x2.size(2)),
                            "feature dimension of x1 and x2 do not match.")

                at::ScalarType common_type;
                if constexpr (reduction_op == All || reduction_op == Any)
                    common_type = at::ScalarType::Bool;
                else
                    common_type = at::result_type(x1, x2);
                auto x1_c = x1.to(common_type).contiguous();
                auto x2_c = x2.to(common_type).contiguous();
                if (unbatched) {
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                at::ScalarType output_type;
                if constexpr (reduction_op == All || reduction_op == Any)
                    output_type = at::kBool;
                else
                    output_type = at::kFloat;
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                        x1_c.options().dtype(output_type));
                int64_t n_kernels = output.numel();

                TORCHPAIRWISE_DISPATCH_CONSTEXPR_BINARY_REDUCTION_OP_TYPES(
                        binary_op, reduction_op, common_type, "pairwise_binary_reduction_forward_cpu", ([&] {
                    TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_accessor =
                                output.accessor<output_t, 3>();
                        impl::pairwise_binary_reduction_forward_kernel_impl<
                                binary_op, reduction_op, output_t, scalar_t, index_t>(
                                n_kernels,
                                x1_c.accessor<scalar_t, 3>(),
                                x2_c.accessor<scalar_t, 3>(),
                                output_accessor);
                    }));
                }))
                if (unbatched)
                    output.squeeze_(0);
                return output;
            }

#define DEFINE_PAIRWISE_BINARY_WITH_REDUCTION(REDUCTION_OP)                           \
template<BinaryOp binary_op>                                                          \
inline at::Tensor pairwise_binary_##REDUCTION_OP##_forward_kernel(                    \
        const at::Tensor &x1,                                                         \
        const at::Tensor &x2) {                                                       \
    return pairwise_binary_reduction_forward_kernel<binary_op, REDUCTION_OP>(x1, x2); \
}

            DEFINE_PAIRWISE_BINARY_WITH_REDUCTION(All)

            DEFINE_PAIRWISE_BINARY_WITH_REDUCTION(Any)

            DEFINE_PAIRWISE_BINARY_WITH_REDUCTION(Sum)

            DEFINE_PAIRWISE_BINARY_WITH_REDUCTION(Prod)

            DEFINE_PAIRWISE_BINARY_WITH_REDUCTION(Mean)
        } // namespace

        TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
            // ~~~~~ binary ~~~~~
            // logical
            m.impl(
                    op_schema_name<And>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<And>));
            m.impl(
                    op_schema_name<Or>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<Or>));
            m.impl(
                    op_schema_name<Xor>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<Xor>));
            // comparison
            m.impl(
                    op_schema_name<Equal>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<Equal>));
            m.impl(
                    op_schema_name<NotEqual>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<NotEqual>));
            m.impl(
                    op_schema_name<Less>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<Less>));
            m.impl(
                    op_schema_name<Greater>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<Greater>));
            m.impl(
                    op_schema_name<LessEqual>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<LessEqual>));
            m.impl(
                    op_schema_name<GreaterEqual>().c_str(),
                    TORCH_FN(pairwise_binary_forward_kernel<GreaterEqual>));

            // ~~~~~ binary reduction ~~~~~
            // logical sum
            m.impl(
                    op_schema_name<And, Sum>().c_str(),
                    TORCH_FN(pairwise_binary_Sum_forward_kernel<And>));
            m.impl(
                    op_schema_name<Or, Sum>().c_str(),
                    TORCH_FN(pairwise_binary_Sum_forward_kernel<Or>));
            m.impl(
                    op_schema_name<Xor, Sum>().c_str(),
                    TORCH_FN(pairwise_binary_Sum_forward_kernel<Xor>));
            // comparison sum
            m.impl(
                    op_schema_name<Equal, Sum>().c_str(),
                    TORCH_FN(pairwise_binary_Sum_forward_kernel<Equal>));
            m.impl(
                    op_schema_name<NotEqual, Sum>().c_str(),
                    TORCH_FN(pairwise_binary_Sum_forward_kernel<NotEqual>));
            // comparison mean
            m.impl(
                    op_schema_name<Equal, Mean>().c_str(),
                    TORCH_FN(pairwise_binary_Mean_forward_kernel<Equal>));
            m.impl(
                    op_schema_name<NotEqual, Mean>().c_str(),
                    TORCH_FN(pairwise_binary_Mean_forward_kernel<NotEqual>));
        }
    }
}
