// Link LibTorch and the torchpairwise library built against the same PyTorch.
// No Python interpreter or Python registration code is used by this executable.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>

#include <iostream>

#include "torchpairwise.h"

void check(at::Device device) {
  using torchpairwise::ops::weighted_comparisons;
  auto options = at::TensorOptions().device(device).dtype(at::kDouble);
  auto x = at::rand({3, 5}, options).set_requires_grad(true);
  auto y = at::rand({4, 5}, options).set_requires_grad(true);
  // Weights introduce a batch axis and differentiate independently.
  auto w = (at::rand({2, 5}, options) + .2).set_requires_grad(true);
  auto difference = x.unsqueeze(-2) - y.unsqueeze(-3);
  auto reference =
      (difference.square().unsqueeze(0) * w.unsqueeze(-2).unsqueeze(-2))
          .sum(-1);
  auto actual = weighted_comparisons(x, y, w, "sqeuclidean");
  TORCH_CHECK(at::allclose(actual, reference), "Weighted values differ");
  auto gradients = torch::autograd::grad({actual.sum()}, {x, y, w});
  auto expected = torch::autograd::grad({reference.sum()}, {x, y, w});
  for (size_t i = 0; i < gradients.size(); ++i)
    TORCH_CHECK(at::allclose(gradients[i], expected[i]),
                "Native weighted gradient differs");
  auto matrix =
      weighted_comparisons(x, c10::nullopt, w, "minkowski", "all_pairs", 1.7);
  auto paired = weighted_comparisons(x, x, w, "minkowski", "paired", 1.7);
  TORCH_CHECK(at::allclose(paired, matrix.diagonal(0, -2, -1)),
              "Paired values differ");
  auto condensed =
      weighted_comparisons(x, c10::nullopt, w, "minkowski", "condensed", 1.7);
  auto triangle = at::stack(
      {matrix.select(-2, 0).select(-1, 1), matrix.select(-2, 0).select(-1, 2),
       matrix.select(-2, 1).select(-1, 2)},
      -1);
  TORCH_CHECK(at::allclose(condensed, triangle), "Condensed values differ");
  auto a = x.gt(.5), b = y.gt(.5);
  auto boolean = weighted_comparisons(a, b, w, "jaccard");
  TORCH_CHECK(boolean.scalar_type() == at::kDouble,
              "Boolean result must follow weight dtype");
  TORCH_CHECK(torch::autograd::grad({boolean.sum()}, {w})[0]
                  .isfinite()
                  .all()
                  .item<bool>(),
              "Weight gradients must be finite");
  bool rejected = false;
  try {
    weighted_comparisons(x, y, w, "cosine", "all_pairs", 2.);
  } catch (const c10::Error&) {
    rejected = true;
  }
  TORCH_CHECK(rejected, "Unused p must be rejected");
  std::cout
      << device
      << ": native weighted values, input/weight gradients and modes passed\n";
}
int main() {
  check(at::Device(at::kCPU));
  if (at::hasCUDA())
    check(at::Device(at::kCUDA, 0));
}
