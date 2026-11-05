#pragma once
#include <ATen/ATen.h>

namespace torchpairwise::ops {
inline void check_histogram_inputs(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.layout() == at::kStrided && y.layout() == at::kStrided,
              "Histogram inputs must be dense strided tensors");
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
              "Histogram inputs must be 2D or 3D");
  TORCH_CHECK(y.dim() == x.dim(), "Histogram input ranks must match");
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "Histogram inputs must share dtype and device");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "Histogram inputs require float32 or float64 (or autocast)");
  TORCH_CHECK(x.size(-1) > 0 && x.size(-1) == y.size(-1),
              "Histogram features must match and be non-empty");
  TORCH_CHECK(x.dim() == 2 || x.size(0) == y.size(0),
              "Histogram batches must match");
}

inline void check_histogram_grad(const at::Tensor& g, const at::Tensor& x,
                                 const at::Tensor& y) {
  check_histogram_inputs(x, y);
  auto shape = x.sizes().vec();
  shape.back() = y.size(-2);
  TORCH_CHECK(
      g.layout() == at::kStrided && g.device() == x.device() &&
          g.scalar_type() == x.scalar_type() &&
          g.sizes() == at::IntArrayRef(shape),
      "Histogram gradient must match the output shape, dtype and device");
}

#ifdef __CUDACC__
#define TORCHPAIRWISE_HIST_HD __host__ __device__
#else
#define TORCHPAIRWISE_HIST_HD
#endif

template <typename scalar_t>
TORCHPAIRWISE_HIST_HD inline scalar_t histogram_value(const scalar_t* x,
                                                      const scalar_t* y,
                                                      int64_t n, int64_t m,
                                                      int64_t d, int64_t t) {
  const auto j = t % m, i = (t / m) % n, b = t / (n * m);
  scalar_t value = 0;
  for (int64_t k = 0; k < d; ++k) {
    const auto a = x[(b * n + i) * d + k], c = y[(b * m + j) * d + k];
    value += a < c ? a : c;
  }
  return value;
}

template <bool left, typename scalar_t>
TORCHPAIRWISE_HIST_HD inline scalar_t histogram_derivative(
    const scalar_t* g, const scalar_t* x, const scalar_t* y, int64_t n,
    int64_t m, int64_t d, int64_t t) {
  const auto own = left ? n : m, other = left ? m : n;
  const auto k = t % d, i = (t / d) % own, b = t / (d * own);
  const auto a = (left ? x : y)[t];
  scalar_t value = 0;
  for (int64_t j = 0; j < other; ++j) {
    const auto c = (left ? y : x)[(b * other + j) * d + k];
    const auto weight =
        a < c ? scalar_t(1) : (a == c ? scalar_t(.5) : scalar_t(0));
    value += weight * g[(b * n + (left ? i : j)) * m + (left ? j : i)];
  }
  return value;
}
#undef TORCHPAIRWISE_HIST_HD
}  // namespace torchpairwise::ops

namespace torchpairwise::ops {
at::Tensor _histogram_intersection(const at::Tensor& x, const at::Tensor& y);
namespace detail {
std::tuple<at::Tensor, at::Tensor> _histogram_intersection_backward(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& y);
}
}  // namespace torchpairwise::ops
