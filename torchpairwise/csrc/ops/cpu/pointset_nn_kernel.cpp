#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <limits>

#include "../pointset_nn.h"

namespace torchpairwise::ops {
namespace {
std::tuple<at::Tensor, at::Tensor> pointset_nn_kernel(const at::Tensor& x,
                                                      const at::Tensor& y,
                                                      const at::Tensor& xl,
                                                      const at::Tensor& yl) {
  check_pointset_nn(x, y, xl, yl);
  auto xc = x.contiguous(), yc = y.contiguous();
  auto xlc = xl.contiguous(), ylc = yl.contiguous();
  const auto B = x.size(0), N = x.size(1), P = x.size(2), C = x.size(3);
  const auto M = y.size(1), Q = y.size(2);
  auto distances = at::empty({B, N, M, P}, x.options());
  auto indices = at::empty({B, N, M, P}, x.options().dtype(at::kLong));
  const auto *lx = xlc.const_data_ptr<int64_t>(),
             *ly = ylc.const_data_ptr<int64_t>();
  auto* idx = indices.mutable_data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointset_nn_kernel", [&] {
    const auto *a = xc.const_data_ptr<scalar_t>(),
               *b = yc.const_data_ptr<scalar_t>();
    auto* out = distances.mutable_data_ptr<scalar_t>();
    at::parallel_for(
        0, distances.numel(), 256, [&](int64_t start, int64_t end) {
          for (int64_t t = start; t < end; ++t) {
            const auto p = t % P, j = (t / P) % M;
            const auto i = (t / (P * M)) % N, batch = t / (P * M * N);
            out[t] = 0;
            idx[t] = 0;
            if (p >= lx[batch * N + i])
              continue;
            scalar_t best = std::numeric_limits<scalar_t>::infinity();
            int64_t nearest = 0;
            // Clamp private-op lengths for memory safety. Public APIs also
            // validate their values without host reads in compiled graphs.
            const auto length =
                std::max<int64_t>(0, std::min(ly[batch * M + j], Q));
            for (int64_t q = 0; q < length; ++q) {
              scalar_t squared = 0;
              for (int64_t c = 0; c < C; ++c) {
                const scalar_t diff = a[((batch * N + i) * P + p) * C + c] -
                                      b[((batch * M + j) * Q + q) * C + c];
                squared += diff * diff;
              }
              if (squared < best) {
                best = squared;
                nearest = q;
              }
            }
            out[t] = best;
            idx[t] = nearest;
          }
        });
  });
  return {distances, indices};
}
}  // namespace

TORCH_LIBRARY_IMPL(torchpairwise, CPU, m) {
  m.impl("_pointset_nn", TORCH_FN(pointset_nn_kernel));
}
}  // namespace torchpairwise::ops
