#include "cpdist.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include "comparison_selectors_detail.h"
namespace torchpairwise::ops {
namespace {
using at::Tensor;
Tensor cdist_impl(
    const Tensor& x1, const c10::optional<Tensor>& x2, c10::string_view metric,
    c10::optional<at::Tensor> w, c10::optional<at::Tensor> V,
    c10::optional<at::Tensor> VI, c10::optional<double> p,
    c10::optional<double> base, c10::optional<bool> shuffle,
    c10::optional<at::Generator> generator, c10::optional<bool> batched,
    c10::optional<int64_t> feature_dim, c10::optional<bool> squared,
    c10::optional<at::Tensor> x1_lengths, c10::optional<at::Tensor> x2_lengths,
    c10::optional<int64_t> chunk_size, c10::optional<at::Tensor> bin_positions,
    c10::optional<at::Tensor> projections, c10::optional<double> regularization,
    c10::optional<int64_t> max_iter, c10::optional<std::string> kernel,
    c10::optional<double> length_scale, c10::optional<double> alpha,
    c10::optional<double> gamma, c10::optional<int64_t> degree,
    c10::optional<double> coef0, c10::optional<double> nu,
    c10::optional<double> data_range, c10::optional<int64_t> window_size,
    c10::optional<double> sigma, c10::optional<double> k1,
    c10::optional<double> k2, c10::optional<std::vector<double>> weights,
    c10::optional<bool> zero_mean, c10::optional<double> energy_floor,
    c10::optional<double> log_floor, c10::optional<double> norm_floor,
    c10::optional<int64_t> n_fft, c10::optional<int64_t> hop_length,
    c10::optional<int64_t> win_length, c10::optional<c10::IValue> window,
    c10::optional<bool> center, c10::optional<bool> normalized,
    c10::optional<std::string> pad_mode,
    c10::optional<std::vector<int64_t>> resolutions,
    c10::optional<int64_t> component_chunk_size,
    c10::optional<std::string> estimator, c10::optional<at::Tensor> x_lengths) {
  ComparisonOptions o;
  o.w = w;
  o.V = V;
  o.VI = VI;
  o.p = p;
  o.base = base;
  o.shuffle = shuffle;
  o.generator = generator;
  o.batched = batched;
  o.feature_dim = feature_dim;
  o.squared = squared;
  o.x1_lengths = x1_lengths;
  o.x2_lengths = x2_lengths;
  o.chunk_size = chunk_size;
  o.bin_positions = bin_positions;
  o.projections = projections;
  o.regularization = regularization;
  o.max_iter = max_iter;
  o.kernel = kernel;
  o.length_scale = length_scale;
  o.alpha = alpha;
  o.gamma = gamma;
  o.degree = degree;
  o.coef0 = coef0;
  o.nu = nu;
  o.data_range = data_range;
  o.window_size = window_size;
  o.sigma = sigma;
  o.k1 = k1;
  o.k2 = k2;
  o.weights = weights;
  o.zero_mean = zero_mean;
  o.energy_floor = energy_floor;
  o.log_floor = log_floor;
  o.norm_floor = norm_floor;
  o.n_fft = n_fft;
  o.hop_length = hop_length;
  o.win_length = win_length;
  o.window = window.value_or(c10::IValue());
  o.center = center;
  o.normalized = normalized;
  o.pad_mode = pad_mode;
  o.resolutions = resolutions;
  o.component_chunk_size = component_chunk_size;
  o.estimator = estimator;
  o.x_lengths = x_lengths;
  return detail::select_comparison(
      ComparisonCollection{x1},
      x2 ? c10::optional<ComparisonCollection>({*x2}) : c10::nullopt, metric,
      "all_pairs", "distance", o);
}
Tensor cdist_collections_impl(
    at::TensorList x1, const c10::optional<ComparisonCollection>& x2,
    c10::string_view metric, c10::optional<at::Tensor> w,
    c10::optional<at::Tensor> V, c10::optional<at::Tensor> VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator,
    c10::optional<bool> batched, c10::optional<int64_t> feature_dim,
    c10::optional<bool> squared, c10::optional<at::Tensor> x1_lengths,
    c10::optional<at::Tensor> x2_lengths, c10::optional<int64_t> chunk_size,
    c10::optional<at::Tensor> bin_positions,
    c10::optional<at::Tensor> projections, c10::optional<double> regularization,
    c10::optional<int64_t> max_iter, c10::optional<std::string> kernel,
    c10::optional<double> length_scale, c10::optional<double> alpha,
    c10::optional<double> gamma, c10::optional<int64_t> degree,
    c10::optional<double> coef0, c10::optional<double> nu,
    c10::optional<double> data_range, c10::optional<int64_t> window_size,
    c10::optional<double> sigma, c10::optional<double> k1,
    c10::optional<double> k2, c10::optional<std::vector<double>> weights,
    c10::optional<bool> zero_mean, c10::optional<double> energy_floor,
    c10::optional<double> log_floor, c10::optional<double> norm_floor,
    c10::optional<int64_t> n_fft, c10::optional<int64_t> hop_length,
    c10::optional<int64_t> win_length, c10::optional<c10::IValue> window,
    c10::optional<bool> center, c10::optional<bool> normalized,
    c10::optional<std::string> pad_mode,
    c10::optional<std::vector<int64_t>> resolutions,
    c10::optional<int64_t> component_chunk_size,
    c10::optional<std::string> estimator, c10::optional<at::Tensor> x_lengths) {
  ComparisonOptions o;
  o.w = w;
  o.V = V;
  o.VI = VI;
  o.p = p;
  o.base = base;
  o.shuffle = shuffle;
  o.generator = generator;
  o.batched = batched;
  o.feature_dim = feature_dim;
  o.squared = squared;
  o.x1_lengths = x1_lengths;
  o.x2_lengths = x2_lengths;
  o.chunk_size = chunk_size;
  o.bin_positions = bin_positions;
  o.projections = projections;
  o.regularization = regularization;
  o.max_iter = max_iter;
  o.kernel = kernel;
  o.length_scale = length_scale;
  o.alpha = alpha;
  o.gamma = gamma;
  o.degree = degree;
  o.coef0 = coef0;
  o.nu = nu;
  o.data_range = data_range;
  o.window_size = window_size;
  o.sigma = sigma;
  o.k1 = k1;
  o.k2 = k2;
  o.weights = weights;
  o.zero_mean = zero_mean;
  o.energy_floor = energy_floor;
  o.log_floor = log_floor;
  o.norm_floor = norm_floor;
  o.n_fft = n_fft;
  o.hop_length = hop_length;
  o.win_length = win_length;
  o.window = window.value_or(c10::IValue());
  o.center = center;
  o.normalized = normalized;
  o.pad_mode = pad_mode;
  o.resolutions = resolutions;
  o.component_chunk_size = component_chunk_size;
  o.estimator = estimator;
  o.x_lengths = x_lengths;
  return detail::select_comparison(x1, x2, metric, "all_pairs", "distance", o);
}
Tensor pdist_impl(
    const Tensor& x1, c10::string_view metric, c10::optional<at::Tensor> w,
    c10::optional<at::Tensor> V, c10::optional<at::Tensor> VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator,
    c10::optional<bool> batched, c10::optional<int64_t> feature_dim,
    c10::optional<bool> squared, c10::optional<at::Tensor> x1_lengths,
    c10::optional<at::Tensor> x2_lengths, c10::optional<int64_t> chunk_size,
    c10::optional<at::Tensor> bin_positions,
    c10::optional<at::Tensor> projections, c10::optional<double> regularization,
    c10::optional<int64_t> max_iter, c10::optional<std::string> kernel,
    c10::optional<double> length_scale, c10::optional<double> alpha,
    c10::optional<double> gamma, c10::optional<int64_t> degree,
    c10::optional<double> coef0, c10::optional<double> nu,
    c10::optional<double> data_range, c10::optional<int64_t> window_size,
    c10::optional<double> sigma, c10::optional<double> k1,
    c10::optional<double> k2, c10::optional<std::vector<double>> weights,
    c10::optional<bool> zero_mean, c10::optional<double> energy_floor,
    c10::optional<double> log_floor, c10::optional<double> norm_floor,
    c10::optional<int64_t> n_fft, c10::optional<int64_t> hop_length,
    c10::optional<int64_t> win_length, c10::optional<c10::IValue> window,
    c10::optional<bool> center, c10::optional<bool> normalized,
    c10::optional<std::string> pad_mode,
    c10::optional<std::vector<int64_t>> resolutions,
    c10::optional<int64_t> component_chunk_size,
    c10::optional<std::string> estimator, c10::optional<at::Tensor> x_lengths) {
  ComparisonOptions o;
  o.w = w;
  o.V = V;
  o.VI = VI;
  o.p = p;
  o.base = base;
  o.shuffle = shuffle;
  o.generator = generator;
  o.batched = batched;
  o.feature_dim = feature_dim;
  o.squared = squared;
  o.x1_lengths = x1_lengths;
  o.x2_lengths = x2_lengths;
  o.chunk_size = chunk_size;
  o.bin_positions = bin_positions;
  o.projections = projections;
  o.regularization = regularization;
  o.max_iter = max_iter;
  o.kernel = kernel;
  o.length_scale = length_scale;
  o.alpha = alpha;
  o.gamma = gamma;
  o.degree = degree;
  o.coef0 = coef0;
  o.nu = nu;
  o.data_range = data_range;
  o.window_size = window_size;
  o.sigma = sigma;
  o.k1 = k1;
  o.k2 = k2;
  o.weights = weights;
  o.zero_mean = zero_mean;
  o.energy_floor = energy_floor;
  o.log_floor = log_floor;
  o.norm_floor = norm_floor;
  o.n_fft = n_fft;
  o.hop_length = hop_length;
  o.win_length = win_length;
  o.window = window.value_or(c10::IValue());
  o.center = center;
  o.normalized = normalized;
  o.pad_mode = pad_mode;
  o.resolutions = resolutions;
  o.component_chunk_size = component_chunk_size;
  o.estimator = estimator;
  o.x_lengths = x_lengths;
  return detail::select_comparison(ComparisonCollection{x1}, c10::nullopt,
                                   metric, "condensed", "distance", o);
}
Tensor pdist_collections_impl(
    at::TensorList x1, c10::string_view metric, c10::optional<at::Tensor> w,
    c10::optional<at::Tensor> V, c10::optional<at::Tensor> VI,
    c10::optional<double> p, c10::optional<double> base,
    c10::optional<bool> shuffle, c10::optional<at::Generator> generator,
    c10::optional<bool> batched, c10::optional<int64_t> feature_dim,
    c10::optional<bool> squared, c10::optional<at::Tensor> x1_lengths,
    c10::optional<at::Tensor> x2_lengths, c10::optional<int64_t> chunk_size,
    c10::optional<at::Tensor> bin_positions,
    c10::optional<at::Tensor> projections, c10::optional<double> regularization,
    c10::optional<int64_t> max_iter, c10::optional<std::string> kernel,
    c10::optional<double> length_scale, c10::optional<double> alpha,
    c10::optional<double> gamma, c10::optional<int64_t> degree,
    c10::optional<double> coef0, c10::optional<double> nu,
    c10::optional<double> data_range, c10::optional<int64_t> window_size,
    c10::optional<double> sigma, c10::optional<double> k1,
    c10::optional<double> k2, c10::optional<std::vector<double>> weights,
    c10::optional<bool> zero_mean, c10::optional<double> energy_floor,
    c10::optional<double> log_floor, c10::optional<double> norm_floor,
    c10::optional<int64_t> n_fft, c10::optional<int64_t> hop_length,
    c10::optional<int64_t> win_length, c10::optional<c10::IValue> window,
    c10::optional<bool> center, c10::optional<bool> normalized,
    c10::optional<std::string> pad_mode,
    c10::optional<std::vector<int64_t>> resolutions,
    c10::optional<int64_t> component_chunk_size,
    c10::optional<std::string> estimator, c10::optional<at::Tensor> x_lengths) {
  ComparisonOptions o;
  o.w = w;
  o.V = V;
  o.VI = VI;
  o.p = p;
  o.base = base;
  o.shuffle = shuffle;
  o.generator = generator;
  o.batched = batched;
  o.feature_dim = feature_dim;
  o.squared = squared;
  o.x1_lengths = x1_lengths;
  o.x2_lengths = x2_lengths;
  o.chunk_size = chunk_size;
  o.bin_positions = bin_positions;
  o.projections = projections;
  o.regularization = regularization;
  o.max_iter = max_iter;
  o.kernel = kernel;
  o.length_scale = length_scale;
  o.alpha = alpha;
  o.gamma = gamma;
  o.degree = degree;
  o.coef0 = coef0;
  o.nu = nu;
  o.data_range = data_range;
  o.window_size = window_size;
  o.sigma = sigma;
  o.k1 = k1;
  o.k2 = k2;
  o.weights = weights;
  o.zero_mean = zero_mean;
  o.energy_floor = energy_floor;
  o.log_floor = log_floor;
  o.norm_floor = norm_floor;
  o.n_fft = n_fft;
  o.hop_length = hop_length;
  o.win_length = win_length;
  o.window = window.value_or(c10::IValue());
  o.center = center;
  o.normalized = normalized;
  o.pad_mode = pad_mode;
  o.resolutions = resolutions;
  o.component_chunk_size = component_chunk_size;
  o.estimator = estimator;
  o.x_lengths = x_lengths;
  return detail::select_comparison(x1, c10::nullopt, metric, "condensed",
                                   "distance", o);
}
Tensor pdist_kernel_impl(const Tensor& x, c10::string_view kernel,
                         int64_t degree, c10::optional<double> gamma,
                         double coef0) {
  ComparisonOptions o;
  if (kernel == "polynomial" || kernel == "polynomial_kernel") {
    o.degree = degree;
    o.coef0 = coef0;
  }
  if (kernel == "sigmoid" || kernel == "sigmoid_kernel")
    o.coef0 = coef0;
  o.gamma = gamma;
  return detail::select_comparison(ComparisonCollection{x}, c10::nullopt,
                                   kernel, "condensed", "kernel", o);
}
}  // namespace
Tensor cdist(const Tensor& x1, const c10::optional<Tensor>& x2,
             c10::string_view metric, const ComparisonOptions& o) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::cdist", "")
                       .typed<decltype(cdist_impl)>();
  return op.call(
      x1, x2, metric, o.w, o.V, o.VI, o.p, o.base, o.shuffle, o.generator,
      o.batched, o.feature_dim, o.squared, o.x1_lengths, o.x2_lengths,
      o.chunk_size, o.bin_positions, o.projections, o.regularization,
      o.max_iter, o.kernel, o.length_scale, o.alpha, o.gamma, o.degree, o.coef0,
      o.nu, o.data_range, o.window_size, o.sigma, o.k1, o.k2, o.weights,
      o.zero_mean, o.energy_floor, o.log_floor, o.norm_floor, o.n_fft,
      o.hop_length, o.win_length,
      o.window.isNone() ? c10::nullopt : c10::optional<c10::IValue>(o.window),
      o.center, o.normalized, o.pad_mode, o.resolutions, o.component_chunk_size,
      o.estimator, o.x_lengths);
}
Tensor cdist(at::TensorList x1, const c10::optional<ComparisonCollection>& x2,
             c10::string_view metric, const ComparisonOptions& o) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::cdist", "collections")
                       .typed<decltype(cdist_collections_impl)>();
  return op.call(
      x1, x2, metric, o.w, o.V, o.VI, o.p, o.base, o.shuffle, o.generator,
      o.batched, o.feature_dim, o.squared, o.x1_lengths, o.x2_lengths,
      o.chunk_size, o.bin_positions, o.projections, o.regularization,
      o.max_iter, o.kernel, o.length_scale, o.alpha, o.gamma, o.degree, o.coef0,
      o.nu, o.data_range, o.window_size, o.sigma, o.k1, o.k2, o.weights,
      o.zero_mean, o.energy_floor, o.log_floor, o.norm_floor, o.n_fft,
      o.hop_length, o.win_length,
      o.window.isNone() ? c10::nullopt : c10::optional<c10::IValue>(o.window),
      o.center, o.normalized, o.pad_mode, o.resolutions, o.component_chunk_size,
      o.estimator, o.x_lengths);
}
Tensor pdist(const Tensor& x1, c10::string_view metric,
             const ComparisonOptions& o) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::pdist", "")
                       .typed<decltype(pdist_impl)>();
  return op.call(
      x1, metric, o.w, o.V, o.VI, o.p, o.base, o.shuffle, o.generator,
      o.batched, o.feature_dim, o.squared, o.x1_lengths, o.x2_lengths,
      o.chunk_size, o.bin_positions, o.projections, o.regularization,
      o.max_iter, o.kernel, o.length_scale, o.alpha, o.gamma, o.degree, o.coef0,
      o.nu, o.data_range, o.window_size, o.sigma, o.k1, o.k2, o.weights,
      o.zero_mean, o.energy_floor, o.log_floor, o.norm_floor, o.n_fft,
      o.hop_length, o.win_length,
      o.window.isNone() ? c10::nullopt : c10::optional<c10::IValue>(o.window),
      o.center, o.normalized, o.pad_mode, o.resolutions, o.component_chunk_size,
      o.estimator, o.x_lengths);
}
Tensor pdist(at::TensorList x1, c10::string_view metric,
             const ComparisonOptions& o) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::pdist", "collections")
                       .typed<decltype(pdist_collections_impl)>();
  return op.call(
      x1, metric, o.w, o.V, o.VI, o.p, o.base, o.shuffle, o.generator,
      o.batched, o.feature_dim, o.squared, o.x1_lengths, o.x2_lengths,
      o.chunk_size, o.bin_positions, o.projections, o.regularization,
      o.max_iter, o.kernel, o.length_scale, o.alpha, o.gamma, o.degree, o.coef0,
      o.nu, o.data_range, o.window_size, o.sigma, o.k1, o.k2, o.weights,
      o.zero_mean, o.energy_floor, o.log_floor, o.norm_floor, o.n_fft,
      o.hop_length, o.win_length,
      o.window.isNone() ? c10::nullopt : c10::optional<c10::IValue>(o.window),
      o.center, o.normalized, o.pad_mode, o.resolutions, o.component_chunk_size,
      o.estimator, o.x_lengths);
}
Tensor cdist(const Tensor& x1, const Tensor& x2, c10::string_view metric,
             const c10::optional<Tensor>& w, const c10::optional<Tensor>& V,
             const c10::optional<Tensor>& VI, c10::optional<double> p,
             c10::optional<double> base, c10::optional<bool> shuffle,
             c10::optional<at::Generator> generator) {
  ComparisonOptions o;
  o.w = w;
  o.V = V;
  o.VI = VI;
  o.p = p;
  o.base = base;
  o.shuffle = shuffle;
  o.generator = generator;
  return cdist(x1, c10::optional<Tensor>(x2), metric, o);
}
Tensor pdist(const Tensor& x1, c10::string_view metric,
             const c10::optional<Tensor>& w, const c10::optional<Tensor>& V,
             const c10::optional<Tensor>& VI, c10::optional<double> p,
             c10::optional<double> base, c10::optional<bool> shuffle,
             c10::optional<at::Generator> generator) {
  ComparisonOptions o;
  o.w = w;
  o.V = V;
  o.VI = VI;
  o.p = p;
  o.base = base;
  o.shuffle = shuffle;
  o.generator = generator;
  return pdist(x1, metric, o);
}
Tensor pdist_kernel(const Tensor& input, c10::string_view kernel,
                    int64_t degree, c10::optional<double> gamma, double coef0) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::pdist_kernel", "")
                       .typed<decltype(pdist_kernel_impl)>();
  return op.call(input, kernel, degree, gamma, coef0);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "cdist(Tensor x1, Tensor? x2=None, str metric='minkowski', *, Tensor? "
      "w=None, Tensor? V=None, Tensor? VI=None, float? p=None, float? "
      "base=None, bool? shuffle=None, Generator? generator=None, bool? "
      "batched=None, int? feature_dim=None, bool? squared=None, Tensor? "
      "x1_lengths=None, Tensor? x2_lengths=None, int? chunk_size=None, Tensor? "
      "bin_positions=None, Tensor? projections=None, float? "
      "regularization=None, int? max_iter=None, str? kernel=None, float? "
      "length_scale=None, float? alpha=None, float? gamma=None, int? "
      "degree=None, float? coef0=None, float? nu=None, float? data_range=None, "
      "int? window_size=None, float? sigma=None, float? k1=None, float? "
      "k2=None, float[]? weights=None, bool? zero_mean=None, float? "
      "energy_floor=None, float? log_floor=None, float? norm_floor=None, int? "
      "n_fft=None, int? hop_length=None, int? win_length=None, Any? "
      "window=None, bool? center=None, bool? normalized=None, str? "
      "pad_mode=None, int[]? resolutions=None, int? component_chunk_size=None, "
      "str? estimator=None, Tensor? x_lengths=None) -> Tensor",
      TORCH_FN(cdist_impl));
  m.def(
      "cdist.collections(Tensor[] x1, Tensor[]? x2=None, str "
      "metric='minkowski', *, Tensor? w=None, Tensor? V=None, Tensor? VI=None, "
      "float? p=None, float? base=None, bool? shuffle=None, Generator? "
      "generator=None, bool? batched=None, int? feature_dim=None, bool? "
      "squared=None, Tensor? x1_lengths=None, Tensor? x2_lengths=None, int? "
      "chunk_size=None, Tensor? bin_positions=None, Tensor? projections=None, "
      "float? regularization=None, int? max_iter=None, str? kernel=None, "
      "float? length_scale=None, float? alpha=None, float? gamma=None, int? "
      "degree=None, float? coef0=None, float? nu=None, float? data_range=None, "
      "int? window_size=None, float? sigma=None, float? k1=None, float? "
      "k2=None, float[]? weights=None, bool? zero_mean=None, float? "
      "energy_floor=None, float? log_floor=None, float? norm_floor=None, int? "
      "n_fft=None, int? hop_length=None, int? win_length=None, Any? "
      "window=None, bool? center=None, bool? normalized=None, str? "
      "pad_mode=None, int[]? resolutions=None, int? component_chunk_size=None, "
      "str? estimator=None, Tensor? x_lengths=None) -> Tensor",
      TORCH_FN(cdist_collections_impl));
  m.def(
      "pdist(Tensor input, str metric='minkowski', *, Tensor? w=None, Tensor? "
      "V=None, Tensor? VI=None, float? p=None, float? base=None, bool? "
      "shuffle=None, Generator? generator=None, bool? batched=None, int? "
      "feature_dim=None, bool? squared=None, Tensor? x1_lengths=None, Tensor? "
      "x2_lengths=None, int? chunk_size=None, Tensor? bin_positions=None, "
      "Tensor? projections=None, float? regularization=None, int? "
      "max_iter=None, str? kernel=None, float? length_scale=None, float? "
      "alpha=None, float? gamma=None, int? degree=None, float? coef0=None, "
      "float? nu=None, float? data_range=None, int? window_size=None, float? "
      "sigma=None, float? k1=None, float? k2=None, float[]? weights=None, "
      "bool? zero_mean=None, float? energy_floor=None, float? log_floor=None, "
      "float? norm_floor=None, int? n_fft=None, int? hop_length=None, int? "
      "win_length=None, Any? window=None, bool? center=None, bool? "
      "normalized=None, str? pad_mode=None, int[]? resolutions=None, int? "
      "component_chunk_size=None, str? estimator=None, Tensor? x_lengths=None) "
      "-> Tensor",
      TORCH_FN(pdist_impl));
  m.def(
      "pdist.collections(Tensor[] input, str metric='minkowski', *, Tensor? "
      "w=None, Tensor? V=None, Tensor? VI=None, float? p=None, float? "
      "base=None, bool? shuffle=None, Generator? generator=None, bool? "
      "batched=None, int? feature_dim=None, bool? squared=None, Tensor? "
      "x1_lengths=None, Tensor? x2_lengths=None, int? chunk_size=None, Tensor? "
      "bin_positions=None, Tensor? projections=None, float? "
      "regularization=None, int? max_iter=None, str? kernel=None, float? "
      "length_scale=None, float? alpha=None, float? gamma=None, int? "
      "degree=None, float? coef0=None, float? nu=None, float? data_range=None, "
      "int? window_size=None, float? sigma=None, float? k1=None, float? "
      "k2=None, float[]? weights=None, bool? zero_mean=None, float? "
      "energy_floor=None, float? log_floor=None, float? norm_floor=None, int? "
      "n_fft=None, int? hop_length=None, int? win_length=None, Any? "
      "window=None, bool? center=None, bool? normalized=None, str? "
      "pad_mode=None, int[]? resolutions=None, int? component_chunk_size=None, "
      "str? estimator=None, Tensor? x_lengths=None) -> Tensor",
      TORCH_FN(pdist_collections_impl));
  m.def(
      "pdist_kernel(Tensor input, str kernel='rbf', *, int degree=3, float? "
      "gamma=None, float coef0=1.) -> Tensor",
      TORCH_FN(pdist_kernel_impl));
}
}  // namespace torchpairwise::ops
