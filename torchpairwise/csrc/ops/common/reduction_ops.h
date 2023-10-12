#pragma once

#include "binary_ops.h"

namespace torchpairwise {
    namespace ops {
        enum ReductionOp {
            // logical
            All, Any,
            // arithmetics
            Sum, Prod, Mean,
        };

        template<BinaryOp binary_op, ReductionOp reduction_op,
                bool with_namespace = false, bool with_signature = false>
        inline std::string op_name() {
            std::string ns = with_namespace ? "torchpairwise::" : "";
            std::string signature = with_signature ? "(Tensor x1, Tensor x2) -> Tensor" : "";
            static const std::string binary_prefix = op_name<binary_op, false, false>() + "_";
            switch (reduction_op) {
                case All:
                    return c10::str(ns, binary_prefix, "all", signature);
                case Any:
                    return c10::str(ns, binary_prefix, "any", signature);
                case Sum:
                    return c10::str(ns, binary_prefix, "sum", signature);
                case Prod:
                    return c10::str(ns, binary_prefix, "prod", signature);
                case Mean:
                    return c10::str(ns, binary_prefix, "mean", signature);
                default:
                    return "[unknown_op]";
            }
        }

        template<BinaryOp binary_op, ReductionOp reduction_op>
        inline std::string op_schema_name() {
            return op_name<binary_op, reduction_op, true, false>();
        }

        template<BinaryOp binary_op, ReductionOp reduction_op>
        inline std::string op_full_schema() {
            return op_name<binary_op, reduction_op, true, true>();
        }
    }
}

// dispatch macros

#define TORCHPAIRWISE_DISPATCH_BINARY_REDUCTION_OP_TYPES(BINARY_OP, REDUCTION_OP, TYPE, NAME, ...) \
if (REDUCTION_OP == All || REDUCTION_OP == Any) {                                            \
    using output_t = bool;                                                                   \
    TORCHPAIRWISE_DISPATCH_BINARY_OP_TYPES(BINARY_OP, TYPE, NAME, __VA_ARGS__)                     \
} else {                                                                                     \
    using output_t = float;                                                                  \
    TORCHPAIRWISE_DISPATCH_BINARY_OP_TYPES(BINARY_OP, TYPE, NAME, __VA_ARGS__)                     \
}

#define TORCHPAIRWISE_DISPATCH_CONSTEXPR_BINARY_REDUCTION_OP_TYPES(BINARY_OP, REDUCTION_OP, TYPE, NAME, ...) \
if constexpr (REDUCTION_OP == All || REDUCTION_OP == Any) {                                            \
    using output_t = bool;                                                                             \
    TORCHPAIRWISE_DISPATCH_CONSTEXPR_BINARY_OP_TYPES(BINARY_OP, TYPE, NAME, __VA_ARGS__)                     \
} else {                                                                                               \
    using output_t = float;                                                                            \
    TORCHPAIRWISE_DISPATCH_CONSTEXPR_BINARY_OP_TYPES(BINARY_OP, TYPE, NAME, __VA_ARGS__)                     \
}
