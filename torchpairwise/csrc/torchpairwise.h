#pragma once

#include "macros.h"
#include "ops/ops.h"
#include "streaming.h"

namespace torchpairwise {
TORCHPAIRWISE_API int64_t cuda_version();
TORCHPAIRWISE_API std::string cuda_arch_flags();
}  // namespace torchpairwise
