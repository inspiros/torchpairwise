#pragma once

#include <ATen/ATen.h>

#include "common/binary_ops.h"
#include "common/reduction_ops.h"
#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        namespace detail {
            template<BinaryOp binary_op>
            inline at::Tensor _pairwise_binary(
                    const at::Tensor &x1,
                    const at::Tensor &x2);

            template<BinaryOp binary_op, ReductionOp reduction_op>
            inline at::Tensor _pairwise_binary_reduction(
                    const at::Tensor &x1,
                    const at::Tensor &x2);
        }

        // ~~~~~ binary ~~~~~
        // logical
        TORCHPAIRWISE_API at::Tensor pwand(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwor(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwxor(
                const at::Tensor &x1,
                const at::Tensor &x2);

        // comparison
        TORCHPAIRWISE_API at::Tensor pweq(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwne(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwlt(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwgt(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwle(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwge(
                const at::Tensor &x1,
                const at::Tensor &x2);

        // ~~~~~ binary reduction ~~~~~
        // logical sum
        TORCHPAIRWISE_API at::Tensor pwand_sum(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwor_sum(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwxor_sum(
                const at::Tensor &x1,
                const at::Tensor &x2);

        // comparison sum
        TORCHPAIRWISE_API at::Tensor pweq_sum(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwne_sum(
                const at::Tensor &x1,
                const at::Tensor &x2);

        // comparison mean
        TORCHPAIRWISE_API at::Tensor pweq_mean(
                const at::Tensor &x1,
                const at::Tensor &x2);

        TORCHPAIRWISE_API at::Tensor pwne_mean(
                const at::Tensor &x1,
                const at::Tensor &x2);
    }
}
