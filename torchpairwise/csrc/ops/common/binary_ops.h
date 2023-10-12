#pragma once

#include <c10/util/StringUtil.h>

#include "../utils/dispatch.h"

namespace torchpairwise {
    namespace ops {
        enum BinaryOp {
            // logical
            And, Or, Xor,
            // comparison
            Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual
        };

        template<BinaryOp op, bool with_namespace = false, bool with_signature = false>
        inline std::string op_name() {
            std::string ns = with_namespace ? "torchpairwise::" : "";
            std::string signature = with_signature ? "(Tensor x1, Tensor x2) -> Tensor" : "";
            switch (op) {
                case And:
                    return c10::str(ns, "pwand", signature);
                case Or:
                    return c10::str(ns, "pwor", signature);
                case Xor:
                    return c10::str(ns, "pwxor", signature);
                case Equal:
                    return c10::str(ns, "pweq", signature);
                case NotEqual:
                    return c10::str(ns, "pwne", signature);
                case Less:
                    return c10::str(ns, "pwlt", signature);
                case Greater:
                    return c10::str(ns, "pwgt", signature);
                case LessEqual:
                    return c10::str(ns, "pwle", signature);
                case GreaterEqual:
                    return c10::str(ns, "pwge", signature);
                default:
                    return "[unknown_op]";
            }
        }

        template<BinaryOp op>
        inline std::string op_schema_name() {
            return op_name<op, true, false>();
        }

        template<BinaryOp op>
        inline std::string op_full_schema() {
            return op_name<op, true, true>();
        }
    }
}

// dispatch macros

#define TORCHPAIRWISE_DISPATCH_BINARY_OP_TYPES(BINARY_OP, TYPE, NAME, ...)                                 \
if (BINARY_OP == And || BINARY_OP == Or || BINARY_OP == Xor) {                                       \
    AT_DISPATCH_BOOLEAN_TYPE(TYPE, NAME, __VA_ARGS__);                                               \
} else if (BINARY_OP == Equal || BINARY_OP == NotEqual) {                                            \
    AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, \
                               TYPE, NAME, __VA_ARGS__);                                             \
} else {                                                                                             \
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,                       \
                                   TYPE, NAME, __VA_ARGS__);                                         \
}

#define TORCHPAIRWISE_DISPATCH_CONSTEXPR_BINARY_OP_TYPES(BINARY_OP, TYPE, NAME, ...)                       \
if constexpr (BINARY_OP == And || BINARY_OP == Or || BINARY_OP == Xor) {                             \
    AT_DISPATCH_BOOLEAN_TYPE(TYPE, NAME, __VA_ARGS__);                                               \
} else if constexpr (BINARY_OP == Equal || BINARY_OP == NotEqual) {                                  \
    AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, \
                               TYPE, NAME, __VA_ARGS__);                                             \
} else {                                                                                             \
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,                       \
                               TYPE, NAME, __VA_ARGS__);                                             \
}
