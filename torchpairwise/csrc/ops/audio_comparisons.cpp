#include "audio_comparisons.h"

#include <array>
#include <map>

#include "audio_comparisons_detail.h"
#include "media_operations_detail.h"
#include "modality_comparisons_detail.h"
#include "utils/comparison_dispatch.h"

namespace torchpairwise::ops {
namespace {
at::Tensor audio_comparisons_composite(const at::Tensor& input,
                                       const c10::optional<at::Tensor>& other,
                                       c10::string_view metric,
                                       c10::string_view mode,
                                       const AudioComparisonOptions& options);
}

namespace {
using namespace modality_detail;
struct Audio {
  int64_t kind;
  std::vector<int64_t> settings;
  std::vector<double> p;
  Audio(int64_t kind, at::IntArrayRef settings, at::ArrayRef<double> p)
      : kind(kind), settings(settings.vec()), p(p.vec()) {}
  Tensor magnitude(const Tensor& x, int64_t n, int64_t hop, int64_t win) const {
    auto window = settings[1] == 2 ? at::ones({win}, x.options())
                  : settings[1] == 0
                      ? at::hann_window(win, true, x.options())
                      : at::hamming_window(win, true, x.options());
    auto spectrum =
        at::stft(x.reshape({-1, x.size(-1)}), n, hop, win, window,
                 bool(settings[2]), settings[3] == 0 ? "constant" : "reflect",
                 bool(settings[4]), true, true);
    auto shape = x.sizes().vec();
    shape.back() = -1;
    return spectrum.abs().reshape(shape);
  }
  Features features(const Tensor& x) const {
    if (kind <= 1)
      return {settings[0] ? x - x.mean(-1, true) : x};
    std::map<std::array<int64_t, 3>, Tensor> cache;
    Features magnitudes, logs;
    for (size_t i = 5; i < settings.size(); i += 3) {
      std::array<int64_t, 3> resolution{settings[i], settings[i + 1],
                                        settings[i + 2]};
      auto found = cache.find(resolution);
      if (found == cache.end())
        found =
            cache
                .emplace(resolution, magnitude(x, resolution[0], resolution[1],
                                               resolution[2]))
                .first;
      magnitudes.push_back(found->second);
    }
    if (kind == 3)
      return magnitudes;
    for (const auto& m : magnitudes)
      logs.push_back(m.clamp_min(p[1]).log());
    if (kind == 2)
      return logs;
    magnitudes.insert(magnitudes.end(), logs.begin(), logs.end());
    return magnitudes;
  }
  Tensor convergence(const Tensor& a, const Tensor& b) const {
    auto numerator =
        at::linalg_vector_norm((a - b).flatten(-2), 2, {-1}, false);
    auto denominator =
        at::linalg_vector_norm(b.flatten(-2), 2, {-1}, false).clamp_min(p[2]);
    return numerator / denominator;
  }
  Tensor score(const Features& a, const Features& b) const {
    if (kind <= 1) {
      auto target = b[0];
      if (kind == 0) {
        auto energy = b[0].square().sum(-1, true);
        target = (a[0] * b[0]).sum(-1, true) /
                 at::where(energy.gt(0), energy, 1.) * b[0];
      }
      auto signal = target.square().sum(-1).clamp_min(p[0]);
      auto error = (a[0] - target).square().sum(-1).clamp_min(p[0]);
      return (10 * (signal.log10() - error.log10())).mean(-1);
    }
    if (kind == 2)
      return (a[0] - b[0]).abs().mean({-2, -1});
    if (kind == 3)
      return convergence(a[0], b[0]);
    Features result;
    auto levels = a.size() / 2;
    for (size_t i = 0; i < levels; ++i)
      result.push_back(convergence(a[i], b[i]) +
                       (a[levels + i] - b[levels + i]).abs().mean({-2, -1}));
    return at::stack(result).mean(0);
  }
};
Tensor audio_compare_kernel(const Tensor& x, const Tensor& y, int64_t kind,
                            int64_t mode, int64_t size,
                            at::IntArrayRef settings, at::ArrayRef<double> p) {
  Audio audio(kind, settings, p);
  return assemble(
      x, y, mode, size, 3, [&](const Tensor& t) { return audio.features(t); },
      [&](const Features& a, const Features& b) { return audio.score(a, b); });
}
Pair audio_compare_backward_kernel(const Tensor& grad, const Tensor& x,
                                   const Tensor& y, int64_t kind, int64_t mode,
                                   int64_t size, at::IntArrayRef settings,
                                   at::ArrayRef<double> p) {
  Audio audio(kind, settings, p);
  return backward(
      grad, x, y, mode, size, 3,
      [&](const Tensor& t) { return audio.features(t); },
      [&](const Features& a, const Features& b) { return audio.score(a, b); });
}
using Core = Tensor(const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
                    at::IntArrayRef, at::ArrayRef<double>);
using Grad = Pair(const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t,
                  int64_t, at::IntArrayRef, at::ArrayRef<double>);

int64_t kind_for(c10::string_view metric) {
  auto kind = detail::media_comparison_kind(metric);
  TORCH_CHECK(kind >= 3, "Unsupported audio comparison: ", metric);
  return kind - 3;
}
}  // namespace
Tensor audio_comparisons(const Tensor& input,
                         const c10::optional<Tensor>& other,
                         c10::string_view metric, c10::string_view mode,
                         const AudioComparisonOptions& options) {
  using Signature = Tensor(
      const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
      c10::string_view mode, int64_t size, c10::optional<bool> zero_mean,
      double energy, int64_t n, c10::optional<int64_t> hop,
      c10::optional<int64_t> win, c10::string_view window, bool center,
      c10::string_view padding, bool normalized, double log, double norm,
      at::IntArrayRef resolutions);
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::audio_comparisons", "")
          .typed<Signature>();
  return op.call(input, other, metric, mode, options.chunk_size,
                 options.zero_mean, options.energy_floor, options.n_fft,
                 options.hop_length, options.win_length, options.window,
                 options.center, options.pad_mode, options.normalized,
                 options.log_floor, options.norm_floor, options.resolutions);
}
namespace {
Tensor audio_comparisons_impl(
    const Tensor& x, const c10::optional<Tensor>& y, c10::string_view metric,
    c10::string_view mode, int64_t size, c10::optional<bool> zero_mean,
    double energy, int64_t n, c10::optional<int64_t> hop,
    c10::optional<int64_t> win, c10::string_view window, bool center,
    c10::string_view padding, bool normalized, double log, double norm,
    at::IntArrayRef resolutions) {
  AudioComparisonOptions options;
  options.chunk_size = size;
  options.zero_mean = zero_mean;
  options.energy_floor = energy;
  options.n_fft = n;
  options.hop_length = hop;
  options.win_length = win;
  options.window = std::string(window);
  options.center = center;
  options.pad_mode = std::string(padding);
  options.normalized = normalized;
  options.log_floor = log;
  options.norm_floor = norm;
  options.resolutions = resolutions.vec();
  return audio_comparisons_composite(x, y, metric, mode, options);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "audio_comparisons(Tensor x1, Tensor? x2, str metric, str "
      "mode='all_pairs', int chunk_size=16, bool? zero_mean=None, float "
      "energy_floor=1e-8, int n_fft=512, int? hop_length=None, int? "
      "win_length=None, str window='hann', bool center=True, str "
      "pad_mode='constant', bool normalized=False, float log_floor=1e-7, float "
      "norm_floor=1e-7, int[] "
      "resolutions=[512,128,512,1024,256,1024,2048,512,2048]) -> Tensor",
      TORCH_FN(audio_comparisons_impl));
  m.def(
      "_audio_compare(Tensor x, Tensor y, int kind, int mode, int size, int[] "
      "settings, float[] parameters) -> Tensor");
  m.def(
      "_audio_compare_backward(Tensor grad, Tensor x, Tensor y, int kind, int "
      "mode, int size, int[] settings, float[] parameters) -> (Tensor, "
      "Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("_audio_compare", TORCH_FN(audio_compare_kernel));
  m.impl("_audio_compare_backward", TORCH_FN(audio_compare_backward_kernel));
}

}  // namespace
namespace {
Tensor audio_comparisons_composite(const Tensor& input,
                                   const c10::optional<Tensor>& other,
                                   c10::string_view metric,
                                   c10::string_view mode,
                                   const AudioComparisonOptions& options) {
  auto p = detail::media::prepare_audio(input, other, metric, mode, options);
  return detail::audio::_audio_compare(p.x, p.y, p.kind, mode_for(mode),
                                       options.chunk_size, p.settings,
                                       p.parameters)
      .reshape_symint(
          output_shape(input, other.value_or(input), 3, mode_for(mode)))
      .to(p.dtype);
}
}  // namespace
}  // namespace torchpairwise::ops

namespace torchpairwise::ops::detail::media {
using namespace modality_detail;
PreparedInputs prepare_audio(const Tensor& input,
                             const c10::optional<Tensor>& other,
                             c10::string_view metric, c10::string_view mode,
                             const AudioComparisonOptions& options) {
  auto kind = kind_for(metric), pairing = mode_for(mode);
  auto y = other.value_or(input);
  TORCH_CHECK(pairing != 1 || other.has_value(),
              "Paired comparisons require two inputs");
  TORCH_CHECK(
      pairing != 2 || (!other.has_value() && kind == 2),
      "Condensed audio comparisons support log_spectral with one input");
  validate(input, y, 3, pairing, options.chunk_size);
  positive(options.energy_floor, "energy_floor");
  positive(options.log_floor, "log_floor");
  positive(options.norm_floor, "norm_floor");
  std::vector<int64_t> settings{options.zero_mean.value_or(kind == 0), 0,
                                options.center, 0, options.normalized};
  if (kind > 1) {
    TORCH_CHECK(options.window == "hann" || options.window == "hamming" ||
                    options.window == "rectangular",
                "window must be hann, hamming, or rectangular");
    TORCH_CHECK(options.pad_mode == "constant" || options.pad_mode == "reflect",
                "pad_mode must be constant or reflect");
    settings[1] = options.window == "hann"      ? 0
                  : options.window == "hamming" ? 1
                                                : 2;
    settings[3] = options.pad_mode == "constant" ? 0 : 1;
    auto resolutions =
        kind == 4
            ? options.resolutions
            : std::vector<int64_t>{options.n_fft,
                                   options.hop_length.value_or(
                                       std::max(int64_t(1), options.n_fft / 4)),
                                   options.win_length.value_or(options.n_fft)};
    TORCH_CHECK(!resolutions.empty() && resolutions.size() % 3 == 0,
                "resolutions must contain non-empty flattened triples");
    for (size_t i = 0; i < resolutions.size(); i += 3) {
      auto n = resolutions[i], hop = resolutions[i + 1],
           win = resolutions[i + 2];
      TORCH_CHECK(n > 0 && hop > 0 && win > 0, "STFT sizes must be positive");
      TORCH_CHECK(win <= n, "win_length must not exceed n_fft");
      TORCH_CHECK(
          !options.center || settings[3] != 1 || input.sym_size(-1) > n / 2,
          "Reflect padding requires T > n_fft//2");
      TORCH_CHECK(options.center || input.sym_size(-1) >= n,
                  "center=False requires T >= n_fft");
    }
    settings.insert(settings.end(), resolutions.begin(), resolutions.end());
  }
  auto dtype = input.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU);
  c10::impl::ExcludeDispatchKeyGuard no_cuda(c10::DispatchKey::AutocastCUDA);
  auto x = input.to(at::kDouble);
  y = input.is_same(y) ? x : y.to(at::kDouble);
  std::vector<double> parameters{options.energy_floor, options.log_floor,
                                 options.norm_floor};
  return {x, y, kind, 3, dtype, std::move(settings), std::move(parameters)};
}

Features audio_features(const Tensor& x, int64_t kind, at::IntArrayRef settings,
                        at::ArrayRef<double> parameters) {
  return Audio(kind, settings, parameters).features(x);
}
Tensor audio_score(const Features& a, const Features& b, int64_t kind,
                   at::IntArrayRef settings, at::ArrayRef<double> parameters) {
  return Audio(kind, settings, parameters).score(a, b);
}
}  // namespace torchpairwise::ops::detail::media
