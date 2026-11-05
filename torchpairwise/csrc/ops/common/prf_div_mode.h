#pragma once

#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>

namespace torchpairwise::ops {
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
    TORCH_CHECK(false, "mode must be either zero or identity. Got ", mode)
  }
}
}  // namespace torchpairwise::ops

#define TORCHPAIRWISE_DISPATCH_PRF_DIV_MODE(MODE, ...)         \
  auto _mode = get_prf_div_mode(MODE);                         \
  if (_mode == Zero) {                                         \
    static constexpr auto prf_div_mode = PRFDivMode::Zero;     \
    __VA_ARGS__();                                             \
  } else if (_mode == Identity) {                              \
    static constexpr auto prf_div_mode = PRFDivMode::Identity; \
    __VA_ARGS__();                                             \
  }
