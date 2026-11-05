#pragma once
// Sample tiling shared by native replay kernels.
//
// Tiles are bounded by an element budget rather than a fixed sample count, so
// large inputs need few dispatches while pair-feature temporaries stay small.
// Matrix-form formulas pass width=1 (no pair-feature intermediate).
#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace torchpairwise::ops::utils {
constexpr int64_t tile_element_budget = int64_t(1) << 22;

// Side of a square all-pairs output tile holding the whole budget.
inline int64_t default_tile_side() {
  return int64_t(std::sqrt(double(tile_element_budget)));
}

// Rows such that rows * columns * width stays within the budget.
inline int64_t tile_rows(int64_t columns, int64_t width, int64_t limit) {
  auto per_row = std::max<int64_t>(1, columns) * std::max<int64_t>(1, width);
  return std::clamp<int64_t>(tile_element_budget / per_row, 1,
                             std::max<int64_t>(1, limit));
}

// Square-ish [rows, columns] all-pairs tile for n x m comparisons.
inline std::pair<int64_t, int64_t> tile_shape(int64_t n, int64_t m,
                                              int64_t width) {
  auto pairs =
      std::max<int64_t>(1, tile_element_budget / std::max<int64_t>(1, width));
  auto side = std::max<int64_t>(1, int64_t(std::sqrt(double(pairs))));
  auto rows = std::clamp<int64_t>(side, 1, std::max<int64_t>(1, n));
  auto columns = std::clamp<int64_t>(pairs / rows, 1, std::max<int64_t>(1, m));
  return {rows, columns};
}

// A condensed row block compares rows [i, i + rows) with columns
// [i + 1, n); local row k keeps local columns c >= k. Returns their flattened
// row-major positions in the [rows, columns] block, which is condensed order.
inline at::Tensor condensed_block_index(int64_t rows, int64_t columns,
                                        const at::TensorOptions& options) {
  auto long_options = options.dtype(at::kLong);
  auto r = at::arange(rows, long_options).unsqueeze(-1);
  auto c = at::arange(columns, long_options).unsqueeze(0);
  return (r * columns + c).masked_select(c.ge(r));
}

// Condensed entries contributed by rows [i, i + rows) of n samples.
inline int64_t condensed_block_size(int64_t n, int64_t i, int64_t rows) {
  return rows * (n - i - 1) - rows * (rows - 1) / 2;
}
// Visits budgeted tiles in output order: mode 0 all-pairs, 1 paired, 2
// condensed. Condensed row blocks compare rows
// [i, i + count) with columns [i + 1, n); `columns` receives the block width.
template <typename Visit>
inline void visit_tiles(int64_t n, int64_t m, int64_t mode, int64_t width,
                        Visit visit) {
  if (mode == 1) {
    for (int64_t i = 0, count; i < n; i += count) {
      count = utils::tile_rows(1, width, n - i);
      visit(i, count, i, count, int64_t(0));
    }
    return;
  }
  for (int64_t i = 0, count; i < (mode == 2 ? n - 1 : n); i += count) {
    auto first = mode == 2 ? i + 1 : int64_t(0);
    auto limit = mode == 2 ? n - 1 - i : n - i;
    count = std::min(utils::tile_shape(limit, m - first, width).first, limit);
    for (int64_t j = first, other; j < m; j += other) {
      other = utils::tile_rows(count, width, m - j);
      visit(i, count, j, other, m - first);
    }
  }
}
}  // namespace torchpairwise::ops::utils
