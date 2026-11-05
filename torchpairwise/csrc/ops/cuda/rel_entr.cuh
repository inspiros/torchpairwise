#pragma once

#include <ATen/NumericUtils.h>
#include <c10/util/BFloat16-inl.h>
#include <c10/util/Half-inl.h>

#include "../utils/scalar_type_utils.h"

namespace torchpairwise::ops {
// Any NaN payload or sign, not only the canonical quiet NaN.
template <typename T>
__forceinline__ __device__ bool m_isnan(const T& _X) throw() {
  return at::_isnan(_X);
}

template <typename T>
__forceinline__ __device__ constexpr T rel_entr(const T& x, const T& y) {
  if (m_isnan(x))
    return x;
  else if (x > T(0) && y > T(0))
    return x * log(x / y);
  else if (x == T(0) && y >= T(0))
    return 0;
  else
    return c10::CPPTypeLimits<T>::upper_bound();
}
}  // namespace torchpairwise::ops
