#pragma once

#include "cpu_helpers.h"
#include "../utils/scalar_type_utils.h"

namespace torchpairwise {
    namespace ops {
        template<typename T>
        __forceinline__ constexpr T rel_entr(const T &x, const T &y) {
            if (std::isnan(x))
                return x;
            else if (x > T(0) && y > T(0))
                return x * log(x / y);
            else if (x == T(0) && y >= T(0))
                return 0;
            else
                return c10::CPPTypeLimits<T>::upper_bound();
        }
    }
}
