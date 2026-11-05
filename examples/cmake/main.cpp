// Configure against an installed torchpairwise CMake package.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>

#include <iostream>

#include "torchpairwise.h"

int main() {
  at::set_num_threads(1);
  auto x = at::randn({17, 5}, at::kDouble).set_requires_grad(true);
  auto y = at::randn({23, 5}, at::kDouble).set_requires_grad(true);
  auto values = at::randn({23, 2}, at::kDouble).set_requires_grad(true);
  torchpairwise::ops::StreamingReductionOptions options;
  options.values = values;
  options.chunk_size = 8;
  options.comparison.gamma = .2;
  // Result is [17,2]; the [17,23] kernel matrix is never stored.
  auto result = std::get<0>(torchpairwise::ops::streaming_reductions(
      x, y, "rbf", "kernel_sum", true, options));
  auto gradients =
      torch::autograd::grad({result.square().mean()}, {x, y, values});
  TORCH_CHECK(result.sizes() == at::IntArrayRef({17, 2}));
  for (const auto& gradient : gradients)
    TORCH_CHECK(gradient.isfinite().all().item<bool>());
  std::cout
      << "Installed CMake consumer: kernel product and gradients passed\n";
}
