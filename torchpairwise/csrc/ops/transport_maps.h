#pragma once
#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise::ops {
struct SinkhornMapOptions {
  c10::optional<at::Tensor> source_weights, target_weights;
  double regularization = .1;
  int64_t max_iter = 100, p = 2, chunk_size = 128;
  bool displacement = false;
};
// Conditional barycentric projection of a finite-iteration balanced entropic
// transport plan, or its displacement from x. Zero-source-mass rows return 0.
TORCHPAIRWISE_API at::Tensor sinkhorn_map(
    const at::Tensor& x, const at::Tensor& y,
    const SinkhornMapOptions& options = {});
}  // namespace torchpairwise::ops
