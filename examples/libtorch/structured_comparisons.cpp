// Link LibTorch and the matching torchpairwise extension. No Python
// initialization.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>

#include <cmath>
#include <iostream>
#include <string>

#include "torchpairwise.h"

void check(at::Device device) {
  using namespace torchpairwise::ops;
  auto options = at::TensorOptions().dtype(at::kDouble).device(device);
  auto mean = at::tensor({0., 1., 3., 2.}, options)
                  .reshape({2, 2})
                  .set_requires_grad(true);
  auto covariance =
      at::eye(2, options).repeat({2, 1, 1}).set_requires_grad(true);
  auto target =
      (at::eye(2, options) * 2).repeat({2, 1, 1}).set_requires_grad(true);
  auto gaussian =
      gaussian_comparisons(mean, covariance, mean + 1, target, "paired", true);
  auto expected = 2. + 2. * std::pow(std::sqrt(2.) - 1., 2.);
  TORCH_CHECK(at::allclose(gaussian, at::full_like(gaussian, expected)),
              "Gaussian values differ");
  for (auto gradient :
       torch::autograd::grad({gaussian.sum()}, {mean, covariance, target}))
    TORCH_CHECK(gradient.isfinite().all().item<bool>(),
                "Gaussian derivative is not finite");
  for (const auto& metric : {"log_euclidean", "affine_invariant", "stein"}) {
    auto value =
        covariance_comparisons(covariance, target, metric, "paired", true);
    double reference = std::string(metric) == "stein"
                           ? 2 * std::log(1.5 / std::sqrt(2.))
                           : 2 * std::pow(std::log(2.), 2.);
    TORCH_CHECK(at::allclose(value, at::full_like(value, reference)),
                "Covariance values differ");
    for (auto gradient :
         torch::autograd::grad({value.sum()}, {covariance, target}))
      TORCH_CHECK(gradient.isfinite().all().item<bool>(),
                  "Repeated-spectrum derivative is not finite");
  }
  auto mixture_mean =
      mean.unsqueeze(1).repeat({1, 2, 1}).set_requires_grad(true);
  auto mixture_covariance = covariance.unsqueeze(1)
                                .repeat({1, 2, 1, 1})
                                .detach()
                                .set_requires_grad(true);
  auto weights = at::ones({2, 2}, options).set_requires_grad(true);
  for (const auto& metric :
       {"overlap", "l2", "cs", "wasserstein", "entropic", "sinkhorn"}) {
    bool regularized =
        std::string(metric) == "entropic" || std::string(metric) == "sinkhorn";
    auto value = mixture_comparisons(
        mixture_mean, mixture_covariance, weights, mixture_mean + 1,
        mixture_covariance * 2, weights, metric, "paired", c10::nullopt,
        c10::nullopt, c10::nullopt,
        regularized ? c10::optional<double>(.4) : c10::nullopt,
        regularized ? c10::optional<int64_t>(8) : c10::nullopt, 2);
    TORCH_CHECK(value.isfinite().all().item<bool>(),
                "Mixture result is not finite");
    for (auto gradient : torch::autograd::grad(
             {value.sum()}, {mixture_mean, mixture_covariance, weights}))
      TORCH_CHECK(gradient.isfinite().all().item<bool>(),
                  "Mixture derivative is not finite");
    auto condensed = mixture_comparisons(mixture_mean, mixture_covariance,
                                         weights, c10::nullopt, c10::nullopt,
                                         c10::nullopt, metric, "condensed");
    TORCH_CHECK(condensed.numel() == 1,
                "Condensed mixture output has incorrect size");
  }
  bool rejected = false;
  try {
    mixture_comparisons(mixture_mean, mixture_covariance, weights, c10::nullopt,
                        c10::nullopt, c10::nullopt, "overlap", "all_pairs",
                        c10::nullopt, c10::nullopt, true);
  } catch (const c10::Error&) {
    rejected = true;
  }
  TORCH_CHECK(rejected, "Unused squared parameter was accepted");
  std::cout << device
            << ": native Gaussian, SPD and mixture values, modes and backward "
               "passed\n";
}
int main() {
  check(at::Device(at::kCPU));
  if (at::hasCUDA())
    check(at::Device(at::kCUDA, 0));
}
