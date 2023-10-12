#pragma once

#include <c10/util/StringUtil.h>
#include <c10/util/Exception.h>

namespace torchpairwise {
    namespace ops {
        enum PRFDivMode {
            Zero,
            Identity,
        };

        inline PRFDivMode get_prf_div_mode(c10::string_view mode) {
            if (mode == "zero")
                return Zero;
            else if (mode == "identity") {
                return Identity;
            } else {
                TORCH_CHECK(false,
                            "mode must be either zero or identity. Got ",
                            mode)
            }
        }
    }
}

#define TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(MODE, ...)                   \
    auto _mode = get_prf_div_mode(MODE);                           \
    if (_mode == Zero) {                                           \
        static constexpr auto prf_div_mode = PRFDivMode::Zero;     \
        __VA_ARGS__();                                             \
    } else if (_mode == Identity) {                                \
        static constexpr auto prf_div_mode = PRFDivMode::Identity; \
        __VA_ARGS__();                                             \
    }
