// Link LibTorch and the matching torchpairwise extension. No Python init.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>

#include <iostream>

#include "torchpairwise.h"

void check(at::Device device) {
  using namespace torchpairwise::ops;
  auto floating = at::TensorOptions().dtype(at::kDouble).device(device);
  auto integer = floating.dtype(at::kLong);
  auto x = at::rand({2, 3, 4, 2}, floating).set_requires_grad(true);
  auto y = at::rand({1, 3, 6, 2}, floating).set_requires_grad(true);
  auto xl = at::tensor({4, 2, 3, 1, 4, 2}, integer).reshape({2, 3});
  auto yl = at::tensor({6, 3, 4}, integer).reshape({1, 3});
  PointsetComparisonOptions options;
  options.batched = true;
  options.feature_dim = 2;
  options.squared = true;
  options.x1_lengths = xl;
  options.x2_lengths = yl;
  options.chunk_size = 2;
  // The feature axis precedes two spatial point axes.
  auto xs = x.reshape({2, 3, 2, 2, 2}).movedim(-1, 2);
  auto ys = y.reshape({1, 3, 2, 3, 2}).movedim(-1, 2);
  auto result = pointset_comparisons(xs, ys, "chamfer", "all_pairs", options);
  auto ground = at::cdist(x.unsqueeze(2), y.unsqueeze(1), 2, 2).square();
  auto valid_y =
      at::arange(6, integer).view({1, 1, 1, 1, 6}).lt(yl.view({1, 1, 3, 1, 1}));
  auto valid_x =
      at::arange(4, integer).view({1, 1, 1, 4}).lt(xl.view({2, 3, 1, 1}));
  auto forward = std::get<0>(ground.masked_fill(~valid_y, INFINITY).min(-1));
  auto reverse =
      std::get<0>(ground.masked_fill(~valid_x.unsqueeze(-1), INFINITY).min(-2));
  auto reference =
      at::where(valid_x, forward, 0.).sum(-1) / xl.unsqueeze(-1) +
      at::where(valid_y.squeeze(-2), reverse, 0.).sum(-1) / yl.unsqueeze(-2);
  TORCH_CHECK(at::allclose(result, reference), "Native Chamfer values differ");
  auto gradients = torch::autograd::grad({result.sum()}, {x, y});
  auto expected = torch::autograd::grad({reference.sum()}, {x, y});
  for (size_t i = 0; i < 2; ++i)
    TORCH_CHECK(at::allclose(gradients[i], expected[i]),
                "Native Chamfer gradients differ");
  auto paired = pointset_comparisons(xs, ys, "chamfer", "paired", options);
  TORCH_CHECK(at::allclose(paired, result.diagonal(0, -2, -1)),
              "Paired values differ");
  options.x2_lengths = c10::nullopt;
  auto condensed =
      pointset_comparisons(xs, c10::nullopt, "chamfer", "condensed", options);
  auto self =
      pointset_comparisons(xs, c10::nullopt, "chamfer", "all_pairs", options);
  auto triangle = at::stack(
      {self.select(-2, 0).select(-1, 1), self.select(-2, 0).select(-1, 2),
       self.select(-2, 1).select(-1, 2)},
      -1);
  TORCH_CHECK(at::allclose(condensed, triangle), "Condensed values differ");
  options.squared = c10::nullopt;
  options.x2_lengths = yl;
  auto directed = directed_pointset_comparisons(xs, ys, "all_pairs", options);
  TORCH_CHECK(std::get<1>(directed).lt(xl.unsqueeze(-1)).all().item<bool>(),
              "Invalid source witnesses");
  TORCH_CHECK(std::get<2>(directed).lt(yl.unsqueeze(-2)).all().item<bool>(),
              "Invalid target witnesses");
  auto hausdorff =
      pointset_comparisons(xs, ys, "hausdorff", "all_pairs", options);
  TORCH_CHECK(hausdorff.isfinite().all().item<bool>(),
              "Invalid Hausdorff values");
  TORCH_CHECK(torch::autograd::grad({hausdorff.sum()}, {x, y})[0]
                  .isfinite()
                  .all()
                  .item<bool>(),
              "Invalid Hausdorff gradients");
  auto legacy = pointset_comparisons(x, y, "hausdorff");
  TORCH_CHECK(torch::autograd::grad({legacy.sum()}, {x, y})[0]
                  .isfinite()
                  .all()
                  .item<bool>(),
              "Invalid full-cloud gradients");
  bool rejected = false;
  try {
    PointsetComparisonOptions invalid;
    invalid.squared = false;
    pointset_comparisons(x, y, "hausdorff", "all_pairs", invalid);
  } catch (const c10::Error&) {
    rejected = true;
  }
  TORCH_CHECK(rejected, "Unused squared must be rejected");
  std::cout << device
            << ": native point-set values, gradients, spatial layouts, modes "
               "and witnesses passed\n";
}
int main() {
  check(at::Device(at::kCPU));
  if (at::hasCUDA())
    check(at::Device(at::kCUDA, 0));
}
