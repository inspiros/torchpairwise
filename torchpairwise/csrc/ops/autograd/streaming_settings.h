#pragma once
#include <torch/autograd.h>

#include "../streaming_reductions_detail.h"
namespace torchpairwise::ops::detail::streaming {
inline void save_settings(torch::autograd::AutogradContext* ctx,
                          const Settings& s) {
  ctx->saved_data["metric"] = s.metric;
  ctx->saved_data["operation"] = s.operation;
  ctx->saved_data["numbers"] = std::vector<int64_t>{
      s.rank,   s.k,       s.size,      s.dtype,       s.mask,
      s.kernel, s.largest, s.exclusion, s.self_cosine, s.nan_first};
  ctx->saved_data["reals"] = s.reals;
  ctx->saved_data["integers"] = s.integers;
  ctx->saved_data["flags"] = s.flags;
  ctx->saved_data["family"] = s.family ? c10::IValue(*s.family) : c10::IValue();
}
inline Settings load_settings(torch::autograd::AutogradContext* ctx) {
  auto numbers = ctx->saved_data["numbers"].toIntVector();
  return {ctx->saved_data["metric"].toStringRef(),
          ctx->saved_data["operation"].toStringRef(),
          bool(numbers[5]),
          bool(numbers[6]),
          bool(numbers[7]),
          bool(numbers[8]),
          bool(numbers[9]),
          numbers[0],
          numbers[1],
          numbers[2],
          numbers[3],
          numbers[4],
          ctx->saved_data["reals"].to<Reals>(),
          ctx->saved_data["integers"].to<Integers>(),
          ctx->saved_data["flags"].to<Flags>(),
          ctx->saved_data["family"].isNone()
              ? c10::nullopt
              : c10::optional<std::string>(
                    ctx->saved_data["family"].toStringRef())};
}
}  // namespace torchpairwise::ops::detail::streaming
