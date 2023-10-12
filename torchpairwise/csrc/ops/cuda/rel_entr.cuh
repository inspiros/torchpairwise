#pragma once

#include <c10/util/Half-inl.h>
#include <c10/util/BFloat16-inl.h>

#include "../utils/scalar_type_utils.h"

namespace torchpairwise {
    namespace ops {
        template<typename T>
        __forceinline__ __device__ bool m_isnan(const T &_X) throw() {
            if constexpr (std::is_same_v<T, c10::Half> ||
                          std::is_same_v<T, c10::BFloat16>)
                return _X.x == std::numeric_limits<T>::quiet_NaN().x;
            else
                return isnan(_X);
        }

        template<typename T>
        __forceinline__ __device__ constexpr T rel_entr(const T &x, const T &y) {
            if (m_isnan(x))
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
