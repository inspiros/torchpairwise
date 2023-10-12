#include "haversine.h"

#include <torch/types.h>

#include "common/binary_ops.h"
#include "common/reduction_ops.h"

namespace torchpairwise {
    namespace ops {
        namespace detail {
            template<BinaryOp binary_op>
            inline at::Tensor _pairwise_binary(
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow(op_schema_name<binary_op>().c_str(), "")
                        .template typed<decltype(_pairwise_binary<binary_op>)>();
                return op.call(x1, x2);
            }

            template<BinaryOp binary_op, ReductionOp reduction_op>
            inline at::Tensor _pairwise_binary_reduction(
                    const at::Tensor &x1,
                    const at::Tensor &x2) {
                static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow(op_schema_name<binary_op, reduction_op>().c_str(), "")
                        .template typed<decltype(_pairwise_binary_reduction<binary_op, reduction_op>)>();
                return op.call(x1, x2);
            }
        }

        // ~~~~~ binary ~~~~~
        // logical
        at::Tensor pwand(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<And>(x1, x2);
        }

        at::Tensor pwor(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<Or>(x1, x2);
        }

        at::Tensor pwxor(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<Xor>(x1, x2);
        }

        // comparison
        at::Tensor pweq(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<Equal>(x1, x2);
        }

        at::Tensor pwne(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<NotEqual>(x1, x2);
        }

        at::Tensor pwlt(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<Less>(x1, x2);
        }

        at::Tensor pwgt(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<Greater>(x1, x2);
        }

        at::Tensor pwle(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<LessEqual>(x1, x2);
        }

        at::Tensor pwge(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary<GreaterEqual>(x1, x2);
        }

        // ~~~~~ binary reduction ~~~~~
        // logical sum
        at::Tensor pwand_sum(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary_reduction<And, Sum>(x1, x2);
        }

        at::Tensor pwor_sum(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary_reduction<Or, Sum>(x1, x2);
        }

        at::Tensor pwxor_sum(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary_reduction<Xor, Sum>(x1, x2);
        }

        // comparison sum
        at::Tensor pweq_sum(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary_reduction<Equal, Sum>(x1, x2);
        }

        at::Tensor pwne_sum(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary_reduction<NotEqual, Sum>(x1, x2);
        }

        // comparison mean
        at::Tensor pweq_mean(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary_reduction<Equal, Mean>(x1, x2);
        }

        at::Tensor pwne_mean(
                const at::Tensor &x1,
                const at::Tensor &x2) {
            return detail::_pairwise_binary_reduction<NotEqual, Mean>(x1, x2);
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
            // ~~~~~ binary ~~~~~
            // logical
            m.def(op_full_schema<And>().c_str());
            m.def(op_full_schema<Or>().c_str());
            m.def(op_full_schema<Xor>().c_str());
            // comparison
            m.def(op_full_schema<Equal>().c_str());
            m.def(op_full_schema<NotEqual>().c_str());
            m.def(op_full_schema<Less>().c_str());
            m.def(op_full_schema<Greater>().c_str());
            m.def(op_full_schema<LessEqual>().c_str());
            m.def(op_full_schema<GreaterEqual>().c_str());

            // ~~~~~ binary reduction ~~~~~
            // logical sum
            m.def(op_full_schema<And, Sum>().c_str());
            m.def(op_full_schema<Or, Sum>().c_str());
            m.def(op_full_schema<Xor, Sum>().c_str());
            // comparison sum
            m.def(op_full_schema<Equal, Sum>().c_str());
            m.def(op_full_schema<NotEqual, Sum>().c_str());
            // comparison mean
            m.def(op_full_schema<Equal, Mean>().c_str());
            m.def(op_full_schema<NotEqual, Mean>().c_str());
        }
    } // namespace ops
} // namespace torchpairwise
