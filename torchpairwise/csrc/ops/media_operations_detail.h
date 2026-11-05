#pragma once
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include "audio_comparisons.h"
#include "image_comparisons.h"
namespace torchpairwise::ops::detail::media {
using at::Tensor;
using Features = std::vector<Tensor>;
struct PreparedInputs {
  Tensor x, y;
  int64_t kind, rank;
  at::ScalarType dtype;
  std::vector<int64_t> settings;
  std::vector<double> parameters;
};
PreparedInputs prepare_image(const Tensor& x, const c10::optional<Tensor>& y,
                             c10::string_view metric, c10::string_view mode,
                             const ImageComparisonOptions& options);
PreparedInputs prepare_audio(const Tensor& x, const c10::optional<Tensor>& y,
                             c10::string_view metric, c10::string_view mode,
                             const AudioComparisonOptions& options);
Features image_features(const Tensor& x, int64_t kind, int64_t window,
                        at::ArrayRef<double> parameters);
Tensor image_score(const Features& a, const Features& b, int64_t kind,
                   int64_t window, at::ArrayRef<double> parameters);
Tensor image_map(const Features& a, const Features& b, int64_t comparison,
                 int64_t window, at::ArrayRef<double> parameters);
Tensor image_map_with_identity(const Features& a, const Features& b,
                               int64_t comparison, int64_t window,
                               at::ArrayRef<double> parameters,
                               const Tensor& identical);
Features audio_features(const Tensor& x, int64_t kind, at::IntArrayRef settings,
                        at::ArrayRef<double> parameters);
Tensor audio_score(const Features& a, const Features& b, int64_t kind,
                   at::IntArrayRef settings, at::ArrayRef<double> parameters);
// Packed features have shape [...,N,F]. Settings retain fixed event dimensions;
// sample/batch axes remain symbolic in the public composite preparation.
Tensor feature_pair(const Tensor& a, const Tensor& b, int64_t kind, bool paired,
                    at::IntArrayRef settings, at::ArrayRef<double> parameters);
using FeatureSignature = Tensor(const Tensor&, int64_t, at::IntArrayRef,
                                at::ArrayRef<double>);
using FeatureGradSignature = Tensor(const Tensor&, const Tensor&, int64_t,
                                    at::IntArrayRef, at::ArrayRef<double>);
inline Tensor features(const Tensor& x, int64_t kind, at::IntArrayRef settings,
                       at::ArrayRef<double> parameters) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::_media_features", "")
                       .typed<FeatureSignature>();
  return op.call(x, kind, settings, parameters);
}
}  // namespace torchpairwise::ops::detail::media
