// Build against LibTorch and the matching native torchpairwise library.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>

#include <iostream>

#include "torchpairwise.h"

void check(at::Device device) {
  using torchpairwise::ops::streaming_reductions;
  using torchpairwise::ops::StreamingReductionOptions;
  auto options = at::TensorOptions().device(device).dtype(at::kDouble);
  auto x = at::rand({3, 2}, options).set_requires_grad(true);
  auto y = at::rand({4, 2}, options).set_requires_grad(true);
  auto payload = at::rand({4, 3}, options).set_requires_grad(true);
  StreamingReductionOptions o;
  o.chunk_size = 2;
  auto dense = x.matmul(y.transpose(-1, -2));
  for (auto operation : {"sum", "prod", "min", "max", "all", "any"}) {
    auto actual =
        std::get<0>(streaming_reductions(x, y, "linear", operation, true, o));
    at::Tensor expected;
    if (std::string(operation) == "sum")
      expected = dense.sum(-1);
    else if (std::string(operation) == "prod")
      expected = dense.prod(-1);
    else if (std::string(operation) == "min")
      expected = std::get<0>(dense.min(-1));
    else if (std::string(operation) == "max")
      expected = std::get<0>(dense.max(-1));
    else if (std::string(operation) == "all")
      expected = dense.ne(0).all(-1);
    else
      expected = dense.ne(0).any(-1);
    TORCH_CHECK(at::allclose(actual, expected), "Streaming row values differ");
    if (actual.is_floating_point()) {
      auto gradient = torch::autograd::grad({actual.sum()}, {x, y}, {}, true);
      auto reference =
          torch::autograd::grad({expected.sum()}, {x, y}, {}, true);
      for (size_t i = 0; i < gradient.size(); ++i)
        TORCH_CHECK(at::allclose(gradient[i], reference[i]),
                    "Streaming row gradients differ");
    }
  }
  o.k = 2;
  auto nearest = streaming_reductions(x, y, "linear", "neighbors", true, o);
  auto order = at::argsort(dense, true, -1, false).narrow(-1, 0, 2);
  TORCH_CHECK(at::equal(std::get<1>(nearest), order),
              "Stable neighbor indices differ");
  TORCH_CHECK(at::allclose(std::get<0>(nearest), dense.gather(-1, order)),
              "Neighbor values differ");
  o.values = payload;
  auto actual =
      std::get<0>(streaming_reductions(x, y, "linear", "kernel_sum", true, o));
  auto expected = dense.matmul(payload);
  TORCH_CHECK(at::allclose(actual, expected),
              "Streaming payload values differ");
  auto gradient = torch::autograd::grad({actual.sum()}, {x, y, payload});
  auto reference = torch::autograd::grad({expected.sum()}, {x, y, payload});
  for (size_t i = 0; i < gradient.size(); ++i)
    TORCH_CHECK(at::allclose(gradient[i], reference[i]),
                "Streaming payload gradients differ");
  std::cout << device
            << ": native streaming rows, stable neighbors and payload "
               "gradients passed\n";
}
int main() {
  check(at::Device(at::kCPU));
  if (at::hasCUDA())
    check(at::Device(at::kCUDA, 0));
}
