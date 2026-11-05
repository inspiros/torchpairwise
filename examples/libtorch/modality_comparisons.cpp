// Link LibTorch and the matching torchpairwise extension. No Python init.
#include <ATen/ATen.h>
#include <torch/csrc/autograd/autograd.h>

#include <cmath>
#include <iostream>

#include "torchpairwise.h"

void check(at::Device device) {
  using namespace torchpairwise::ops;
  auto options = at::TensorOptions().dtype(at::kDouble).device(device);
  auto x = at::rand({3, 2, 6, 7}, options).set_requires_grad(true);
  auto y = at::rand_like(x).set_requires_grad(true);
  ImageComparisonOptions image;
  image.data_range = 1.7;
  image.chunk_size = 2;
  auto psnr = image_comparisons(x, y, "psnr", "paired", image);
  auto reference =
      20 * std::log10(1.7) - 10 * (x - y).square().mean({-3, -2, -1}).log10();
  TORCH_CHECK(at::allclose(psnr, reference), "Native PSNR differs");
  auto gradients = torch::autograd::grad({psnr.sum()}, {x, y});
  auto expected = torch::autograd::grad({reference.sum()}, {x, y});
  for (size_t i = 0; i < 2; ++i)
    TORCH_CHECK(at::allclose(gradients[i], expected[i]),
                "Native PSNR gradient differs");
  image.window_size = 3;
  image.weights = {.4, .6};
  for (auto metric : {"ssim", "multiscale_ssim"}) {
    auto all = image_comparisons(x, y, metric, "all_pairs", image);
    auto paired = image_comparisons(x, y, metric, "paired", image);
    TORCH_CHECK(at::allclose(paired, all.diagonal(0, -2, -1)),
                "Image paired mode differs");
    auto self = image_comparisons(x, c10::nullopt, metric, "all_pairs", image);
    auto condensed =
        image_comparisons(x, c10::nullopt, metric, "condensed", image);
    auto triangle = at::stack({self.select(-2, 0).select(-1, 1),
                               self.select(-2, 0).select(-1, 2),
                               self.select(-2, 1).select(-1, 2)});
    TORCH_CHECK(at::allclose(condensed, triangle),
                "Image condensed order differs");
    auto gradient = torch::autograd::grad({all.sum()}, {x, y});
    for (auto& g : gradient)
      TORCH_CHECK(g.isfinite().all().item<bool>(),
                  "Image gradient is nonfinite");
  }
  auto a = at::randn({3, 2, 24}, options).set_requires_grad(true);
  auto b = at::randn_like(a).set_requires_grad(true);
  AudioComparisonOptions audio;
  audio.n_fft = 8;
  audio.hop_length = 3;
  audio.win_length = 6;
  audio.resolutions = {8, 2, 8, 12, 3, 12, 8, 2, 8};
  audio.chunk_size = 2;
  for (auto metric : {"si_sdr", "signal_snr", "log_spectral",
                      "spectral_convergence", "multiresolution_stft"}) {
    auto all = audio_comparisons(a, b, metric, "all_pairs", audio);
    auto paired = audio_comparisons(a, b, metric, "paired", audio);
    TORCH_CHECK(at::allclose(paired, all.diagonal(0, -2, -1)),
                "Audio paired mode differs");
    for (auto& g : torch::autograd::grad({all.sum()}, {a, b}))
      TORCH_CHECK(g.isfinite().all().item<bool>(),
                  "Audio gradient is nonfinite");
  }
  auto snr = audio_comparisons(a, b, "signal_snr", "paired", audio);
  auto snr_reference =
      (10 * (b.square().sum(-1).clamp_min(audio.energy_floor).log10() -
             (a - b).square().sum(-1).clamp_min(audio.energy_floor).log10()))
          .mean(-1);
  TORCH_CHECK(at::allclose(snr, snr_reference), "Native SNR differs");
  std::cout
      << device
      << ": native image/audio values, first gradients and modes passed\n";
}
int main() {
  check(at::Device(at::kCPU));
  if (at::hasCUDA())
    check(at::Device(at::kCUDA, 0));
}
