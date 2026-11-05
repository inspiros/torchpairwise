#pragma once
#include "comparison_selectors.h"
namespace torchpairwise::ops::detail {
at::Tensor select_comparison(at::TensorList x1,
                             const c10::optional<ComparisonCollection>& x2,
                             c10::string_view metric, c10::string_view mode,
                             c10::string_view role,
                             const ComparisonOptions& options);
}  // namespace torchpairwise::ops::detail
