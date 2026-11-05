#pragma once
#include <ATen/ATen.h>

#include "../macros.h"
namespace torchpairwise::ops {
struct AudioComparisonOptions {
  int64_t chunk_size = 16;
  c10::optional<bool> zero_mean;
  double energy_floor = 1e-8, log_floor = 1e-7, norm_floor = 1e-7;
  int64_t n_fft = 512;
  c10::optional<int64_t> hop_length, win_length;
  std::string window = "hann", pad_mode = "constant";
  bool center = true, normalized = false;
  std::vector<int64_t> resolutions{512,  128,  512, 1024, 256,
                                   1024, 2048, 512, 2048};
};
// Waveforms [...,N,C,T]. Condensed is supported for log_spectral only.
// Resolution triples are flattened (n_fft,hop_length,win_length).
TORCHPAIRWISE_API at::Tensor audio_comparisons(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
    c10::string_view metric, c10::string_view mode = "all_pairs",
    const AudioComparisonOptions& options = {});
}  // namespace torchpairwise::ops
