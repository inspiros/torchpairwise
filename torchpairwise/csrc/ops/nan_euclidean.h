#pragma once
#include <ATen/ATen.h>

namespace torchpairwise::ops {
inline void check_nan_inputs(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided,
              "NaN Euclidean inputs must be dense strided tensors");
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
              "NaN Euclidean inputs must be 2D or 3D");
  TORCH_CHECK(y.dim() == x.dim(), "NaN Euclidean input ranks must match");
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "NaN Euclidean inputs must share dtype and device");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "NaN Euclidean inputs require float32 or float64 (or autocast)");
  TORCH_CHECK(x.size(-1) > 0 && x.size(-1) == y.size(-1),
              "Features must match and be non-empty");
  TORCH_CHECK(x.dim() == 2 || x.size(0) == y.size(0), "Batches must match");
}

inline void check_nan_backward(const at::Tensor& g, const at::Tensor& x,
                               const at::Tensor& y, const at::Tensor& counts) {
  check_nan_inputs(x, y);
  auto shape = x.sizes().vec();
  shape.back() = y.size(-2);
  TORCH_CHECK(
      g.layout() == at::kStrided && g.device() == x.device() &&
          g.scalar_type() == x.scalar_type() &&
          g.sizes() == at::IntArrayRef(shape),
      "Gradient must match the pairwise output shape, dtype and device");
  TORCH_CHECK(
      counts.layout() == at::kStrided && counts.device() == x.device() &&
          counts.scalar_type() == at::kLong && counts.sizes() == g.sizes(),
      "Counts must be int64 and match the output shape and device");
}

#ifdef __CUDACC__
#define TORCHPAIRWISE_NAN_HD __host__ __device__
#else
#define TORCHPAIRWISE_NAN_HD
#endif

template <typename scalar_t>
TORCHPAIRWISE_NAN_HD inline void nan_squared_value(const scalar_t* x,
                                                   const scalar_t* y, int64_t n,
                                                   int64_t m, int64_t d,
                                                   int64_t t, scalar_t& value,
                                                   int64_t& count) {
  const auto j = t % m, i = (t / m) % n, b = t / (n * m);
  value = 0;
  count = 0;
  for (int64_t k = 0; k < d; ++k) {
    const auto a = x[(b * n + i) * d + k], c = y[(b * m + j) * d + k];
    if (a == a && c == c) {
      const auto diff = a - c;
      value += diff * diff;
      ++count;
    }
  }
  // Undefined pairs are represented by (0,0) privately. The public wrapper
  // inserts NaN while choosing zero gradients for no-overlap pairs.
  if (count)
    value *= scalar_t(d) / scalar_t(count);
}

template <bool left, typename scalar_t>
TORCHPAIRWISE_NAN_HD inline scalar_t nan_squared_derivative(
    const scalar_t* g, const scalar_t* x, const scalar_t* y,
    const int64_t* counts, int64_t n, int64_t m, int64_t d, int64_t t) {
  const auto own = left ? n : m, other = left ? m : n;
  const auto k = t % d, i = (t / d) % own, b = t / (d * own);
  const auto a = (left ? x : y)[t];
  scalar_t value = 0;
  if (a != a)
    return value;
  for (int64_t j = 0; j < other; ++j) {
    const auto pair = (b * n + (left ? i : j)) * m + (left ? j : i);
    const auto count = counts[pair];
    const auto c = (left ? y : x)[(b * other + j) * d + k];
    if (count > 0 && count <= d && c == c)
      value +=
          (a - c) * (scalar_t(2) * scalar_t(d) / scalar_t(count)) * g[pair];
  }
  return value;
}
#undef TORCHPAIRWISE_NAN_HD
}  // namespace torchpairwise::ops

namespace torchpairwise::ops {
std::tuple<at::Tensor, at::Tensor> _nan_sqeuclidean(const at::Tensor& x,
                                                    const at::Tensor& y);
namespace detail {
std::tuple<at::Tensor, at::Tensor> _nan_sqeuclidean_backward(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& y,
    const at::Tensor& counts);
}
}  // namespace torchpairwise::ops
