#pragma once

#include "cpu_helpers.h"
#include "../common/prf_div_mode.h"

namespace torchpairwise {
    namespace ops {
        template<PRFDivMode mode = Zero, typename T>
        __forceinline__ constexpr T prf_divide(const T &x, const T &y) {
            if constexpr (mode == Zero)
                return y != T(0) ? x / y : T(0);
            else if constexpr (mode == Identity)
                return y != T(0) ? x / y : x;
            else
                return x / y;
        }
    }
}
