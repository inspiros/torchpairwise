#pragma once

#include "cpu_helpers.h"
#include "../common/binary_ops.h"

namespace torchpairwise {
    namespace ops {
        template<BinaryOp op, typename scalar_t>
        __forceinline__ scalar_t call(scalar_t self, scalar_t other) {
            // logical
            if constexpr (op == And)
                return self & other;
            if constexpr (op == Or)
                return self | other;
            if constexpr (op == Xor)
                return self ^ other;
            // comparison
            if constexpr (op == Equal)
                return self == other;
            if constexpr (op == NotEqual)
                return self != other;
            if constexpr (op == Less)
                return self < other;
            if constexpr (op == Greater)
                return self > other;
            if constexpr (op == LessEqual)
                return self <= other;
            if constexpr (op == GreaterEqual)
                return self >= other;
        }
    }
}
