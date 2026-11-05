// Standalone dense neighbor selection over the full native vector/cloud
// catalog.
#include <ATen/ATen.h>

#include <cmath>
#include <iostream>
#include <limits>

#include "torchpairwise.h"

void check(at::Device device) {
  using namespace torchpairwise::ops;
  auto options = at::TensorOptions().dtype(at::kDouble).device(device);
  auto x =
      at::tensor({0., 1., 3.}, options).reshape({3, 1}).set_requires_grad(true);
  auto selected = builtin_neighbor_mask(x, c10::nullopt, "k", 1);
  auto expected =
      at::tensor({0, 1, 0, 1, 0, 0, 0, 1, 0}, options.dtype(at::kLong))
          .to(at::kBool)
          .reshape({3, 3});
  TORCH_CHECK(at::equal(selected, expected),
              "Native neighbor selection differs");
  TORCH_CHECK(!selected.requires_grad(),
              "Neighbor masks must not require gradients");
  auto radius =
      builtin_neighbor_mask(x, c10::nullopt, "radius", c10::nullopt, 1.);
  expected = at::tensor({0, 1, 0, 1, 0, 0, 0, 0, 0}, options.dtype(at::kLong))
                 .to(at::kBool)
                 .reshape({3, 3});
  TORCH_CHECK(at::equal(radius, expected), "Native radius selection differs");
  auto clouds = x.unsqueeze(1).repeat({1, 2, 1});
  BuiltinComparisonOptions parameters;
  parameters.squared = true;
  parameters.chunk_size = 1;
  auto cloud_mask =
      builtin_neighbor_mask(clouds, c10::nullopt, "k", 1, c10::nullopt,
                            c10::nullopt, "chamfer", parameters);
  TORCH_CHECK(at::equal(cloud_mask, selected),
              "Cloud neighbor selection differs");
  auto nonfinite =
      at::tensor({NAN, INFINITY, INFINITY, 1., 1., NAN, NAN, -INFINITY, 0.},
                 options)
          .reshape({3, 3});
  auto mask =
      neighbor_mask_from_distances(nonfinite, "k", 2, c10::nullopt, true);
  TORCH_CHECK(!mask.diagonal().any().item<bool>() &&
                  mask.sum(-1).eq(2).all().item<bool>(),
              "Excluded self must sort after infinite and NaN candidates");
  auto empty = at::empty({0, 0}, options);
  TORCH_CHECK(
      neighbor_mask_from_distances(empty, "k", 0, c10::nullopt, true).numel() ==
          0,
      "Empty self selection must remain empty");
  std::cout << device
            << ": native vector/cloud neighbor masks and non-finite selection "
               "passed\n";
}
int main() {
  check(at::Device(at::kCPU));
  if (at::hasCUDA())
    check(at::Device(at::kCUDA, 0));
}
