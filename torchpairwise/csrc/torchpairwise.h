#pragma once

#include "ops/ops.h"
#include "macros.h"

namespace torchpairwise {
    TORCHPAIRWISE_API int64_t cuda_version();
    TORCHPAIRWISE_API std::string cuda_arch_flags();
}
