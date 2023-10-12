#pragma once

#include <ATen/ScalarType.h>
#include <c10/util/Half-inl.h>
#include <c10/util/BFloat16-inl.h>

namespace c10 {
    template<typename T>
    struct is_signed : std::is_signed<T> {
    };

    template<typename T>
    struct is_unsigned : std::is_unsigned<T> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>> : std::bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>> : std::bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::BFloat16>> : std::bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::BFloat16>> : std::bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QInt32>> : std::bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QInt32>> : std::bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QInt8>> : std::bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QInt8>> : std::bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt8>> : std::bool_constant<false> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt8>> : std::bool_constant<true> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt2x4>> : std::bool_constant<false> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt2x4>> : std::bool_constant<true> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt4x2>> : std::bool_constant<false> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt4x2>> : std::bool_constant<true> {
    };

#define HALF_MIN at::Half(0xFBFF, at::Half::from_bits())
#define HALF_MAX at::Half(0x7BFF, at::Half::from_bits())
#define HALF_LB at::Half(0xFC00, at::Half::from_bits())
#define HALF_UB at::Half(0x7C00, at::Half::from_bits())
#define HALF_EPS at::Half(0x1400, at::Half::from_bits())
#define HALF_QNAN at::Half(0x7E00, at::Half::from_bits())
#define HALF_SNAN at::Half(0x7E00, at::Half::from_bits())

#define BFLOAT16_MIN at::BFloat16(0xFF7F, at::BFloat16::from_bits())
#define BFLOAT16_MAX at::BFloat16(0x7F7F, at::BFloat16::from_bits())
#define BFLOAT16_LB at::BFloat16(0xFF80, at::BFloat16::from_bits())
#define BFLOAT16_UB at::BFloat16(0x7F80, at::BFloat16::from_bits())
#define BFLOAT16_EPS at::BFloat16(0x3A80, at::BFloat16::from_bits())
#define BFLOAT16_QNAN at::BFloat16(0x7FC0, at::BFloat16::from_bits())
#define BFLOAT16_SNAN at::BFloat16(0x7FC0, at::BFloat16::from_bits())

#define AT_FORWARD_INTEGRAL_LIMITS(_, cpp_type, scalar_type)  \
    _(cpp_type, scalar_type,                                  \
    std::numeric_limits<cpp_type>::min(),                     \
    std::numeric_limits<cpp_type>::max(),                     \
    std::numeric_limits<cpp_type>::min(),                     \
    std::numeric_limits<cpp_type>::max(),                     \
    std::numeric_limits<cpp_type>::epsilon(),                 \
    0, 0)

#define AT_FORWARD_FLOATING_LIMITS(_, cpp_type, scalar_type)  \
    _(cpp_type, scalar_type,                                  \
    std::numeric_limits<cpp_type>::min(),                     \
    std::numeric_limits<cpp_type>::max(),                     \
    -std::numeric_limits<cpp_type>::infinity(),               \
    std::numeric_limits<cpp_type>::infinity(),                \
    std::numeric_limits<cpp_type>::epsilon(),                 \
    std::numeric_limits<cpp_type>::quiet_NaN(),               \
    std::numeric_limits<cpp_type>::signaling_NaN())

#define AT_FORALL_TYPES(_)  \
    AT_FORWARD_INTEGRAL_LIMITS(_, uint8_t, Byte) /* 0 */ \
    AT_FORWARD_INTEGRAL_LIMITS(_, int8_t, Char) /* 1 */  \
    AT_FORWARD_INTEGRAL_LIMITS(_, int16_t, Short) /* 2 */\
    AT_FORWARD_INTEGRAL_LIMITS(_, int, Int) /* 3 */      \
    AT_FORWARD_INTEGRAL_LIMITS(_, int64_t, Long) /* 4 */ \
    _(at::Half, Half, HALF_MIN, HALF_MAX, HALF_LB, HALF_UB, HALF_EPS, HALF_QNAN, HALF_SNAN) /* 5 */ \
    AT_FORWARD_FLOATING_LIMITS(_, float, Float) /* 6 */  \
    AT_FORWARD_FLOATING_LIMITS(_, double, Double) /* 7 */\
    AT_FORWARD_INTEGRAL_LIMITS(_, bool, Bool) /* 8 */    \
    _(at::BFloat16, BFloat16, BFLOAT16_MIN, BFLOAT16_MAX, BFLOAT16_LB, BFLOAT16_UB, BFLOAT16_EPS, BFLOAT16_QNAN, BFLOAT16_SNAN) /* 9 */

    template<c10::ScalarType T>
    struct ScalarTypeLimits;

    template<typename T>
    struct CPPTypeLimits;

#define SPECIALIZE_ScalarTypeLimits(cpp_type, scalar_type, min_value, max_value, lb_value, ub_value, eps_value, qnan_value, snan_value)  \
template <>                                                                                                                              \
struct ScalarTypeLimits<c10::ScalarType::scalar_type> {                                                                                  \
    using scalar_t = cpp_type;                                                                                                           \
    C10_NODISCARD static constexpr cpp_type(min)() noexcept {                                                                            \
        return min_value;                                                                                                                \
    }                                                                                                                                    \
    C10_NODISCARD static constexpr cpp_type(max)() noexcept {                                                                            \
        return max_value;                                                                                                                \
    }                                                                                                                                    \
    C10_NODISCARD static constexpr cpp_type(lower_bound)() noexcept {                                                                    \
        return lb_value;                                                                                                                 \
    }                                                                                                                                    \
    C10_NODISCARD static constexpr cpp_type(upper_bound)() noexcept {                                                                    \
        return ub_value;                                                                                                                 \
    }                                                                                                                                    \
    C10_NODISCARD static constexpr cpp_type(epsilon)() noexcept {                                                                        \
        return eps_value;                                                                                                                \
    }                                                                                                                                    \
    C10_NODISCARD static constexpr cpp_type(quiet_nan)() noexcept {                                                                      \
        return qnan_value;                                                                                                               \
    }                                                                                                                                    \
    C10_NODISCARD static constexpr cpp_type(signaling_nan)() noexcept {                                                                  \
        return snan_value;                                                                                                               \
    }                                                                                                                                    \
};

#define SPECIALIZE_CPPTypeLimits(cpp_type, _, min_value, max_value, lb_value, ub_value, eps_value, qnan_value, snan_value)  \
template <>                                                                                                                 \
struct CPPTypeLimits<cpp_type> {                                                                                            \
    C10_NODISCARD static constexpr cpp_type(min)() noexcept {                                                               \
        return min_value;                                                                                                   \
    }                                                                                                                       \
    C10_NODISCARD static constexpr cpp_type(max)() noexcept {                                                               \
        return max_value;                                                                                                   \
    }                                                                                                                       \
    C10_NODISCARD static constexpr cpp_type(lower_bound)() noexcept {                                                       \
        return lb_value;                                                                                                    \
    }                                                                                                                       \
    C10_NODISCARD static constexpr cpp_type(upper_bound)() noexcept {                                                       \
        return ub_value;                                                                                                    \
    }                                                                                                                       \
    C10_NODISCARD static constexpr cpp_type(epsilon)() noexcept {                                                           \
        return eps_value;                                                                                                   \
    }                                                                                                                       \
    C10_NODISCARD static constexpr cpp_type(quiet_nan)() noexcept {                                                         \
        return qnan_value;                                                                                                  \
    }                                                                                                                       \
    C10_NODISCARD static constexpr cpp_type(signaling_nan)() noexcept {                                                     \
        return snan_value;                                                                                                  \
    }                                                                                                                       \
};

    AT_FORALL_TYPES(SPECIALIZE_ScalarTypeLimits)

    AT_FORALL_TYPES(SPECIALIZE_CPPTypeLimits)

#undef AT_FORWARD_INTEGRAL_LIMITS
#undef AT_FORWARD_FLOATING_LIMITS
#undef AT_FORALL_TYPES
#undef SPECIALIZE_ScalarTypeLimits
#undef SPECIALIZE_CPPTypeLimits
}
