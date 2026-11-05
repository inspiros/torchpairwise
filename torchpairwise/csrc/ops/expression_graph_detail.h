#pragma once
#include "expression_operations.h"
#include "streaming_reductions_detail.h"

namespace torchpairwise::ops::detail::streaming {
// Prepare a scalar graph once for repeated native consumers.
PreparedComparison prepare_expression_graph(const ExpressionGraph& graph);
}  // namespace torchpairwise::ops::detail::streaming
