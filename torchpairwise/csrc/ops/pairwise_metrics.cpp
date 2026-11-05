
#include "pairwise_metrics.h"

#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "additive_chi2_kernel.h"
#include "braycurtis.h"
#include "canberra.h"
#include "hausdorff.h"
#include "haversine.h"
#include "pairwise_binary.h"
#include "ppminkowski.h"
#include "prf_div.h"
#include "snr.h"
#include "sqjensenshannon.h"
#include "sqmahalanobis.h"
#include "utils/scalar_type_utils.h"
#include "utils/value_checks.h"
#include "wminkowski.h"

namespace torchpairwise::ops {
namespace {
at::Tensor euclidean_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2);
at::Tensor haversine_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2);
at::Tensor manhattan_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2);
at::Tensor cosine_distances_impl(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2);
at::Tensor linear_kernel_impl(const at::Tensor& x1,
                              const c10::optional<at::Tensor>& x2);
at::Tensor polynomial_kernel_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2,
                                  int64_t degree, c10::optional<double> gamma,
                                  double coef0);
at::Tensor sigmoid_kernel_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2,
                               c10::optional<double> gamma, double coef0);
at::Tensor rbf_kernel_impl(const at::Tensor& x1,
                           const c10::optional<at::Tensor>& x2,
                           c10::optional<double> gamma);
at::Tensor laplacian_kernel_impl(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2,
                                 c10::optional<double> gamma);
at::Tensor cosine_similarity_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2);
at::Tensor additive_chi2_kernel_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2);
at::Tensor chi2_kernel_impl(const at::Tensor& x1,
                            const c10::optional<at::Tensor>& x2, double gamma);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
directed_hausdorff_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2,
                                  bool shuffle,
                                  c10::optional<at::Generator> generator);
at::Tensor minkowski_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2,
                                    double p);
at::Tensor wminkowski_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2,
                                     double p,
                                     const c10::optional<at::Tensor>& w);
at::Tensor sqeuclidean_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2);
at::Tensor correlation_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2);
at::Tensor hamming_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2);
at::Tensor jaccard_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2);
at::Tensor kulsinski_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2);
at::Tensor kulczynski1_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2);
at::Tensor seuclidean_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2,
                                     const c10::optional<at::Tensor>& V);
at::Tensor cityblock_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2);
at::Tensor mahalanobis_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2,
                                      const c10::optional<at::Tensor>& VI);
at::Tensor chebyshev_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2);
at::Tensor braycurtis_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2);
at::Tensor canberra_distances_impl(const at::Tensor& x1,
                                   const c10::optional<at::Tensor>& x2);
at::Tensor jensenshannon_distances_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2,
                                        c10::optional<double> base);
at::Tensor yule_distances_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2);
at::Tensor dice_distances_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2);
at::Tensor rogerstanimoto_distances_impl(const at::Tensor& x1,
                                         const c10::optional<at::Tensor>& x2);
at::Tensor russellrao_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2);
at::Tensor sokalmichener_distances_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2);
at::Tensor sokalsneath_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2);
at::Tensor snr_distances_impl(const at::Tensor& x1,
                              const c10::optional<at::Tensor>& x2);
at::Tensor l1_distances_impl(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2);
at::Tensor l2_distances_impl(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2);
at::Tensor lp_distances_impl(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2, double p);
at::Tensor linf_distances_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2);
}  // namespace

namespace {
// Choose the zero subgradient at an exactly zero distance without
// evaluating sqrt's infinite derivative at that point.
at::Tensor distance_sqrt(const at::Tensor& value) {
  auto nonzero = value.ne(0);
  return at::where(nonzero,
                   at::where(nonzero, value, at::ones_like(value)).sqrt(),
                   at::zeros_like(value));
}
}  // namespace

at::Tensor euclidean_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::euclidean_distances", "")
          .typed<decltype(euclidean_distances)>();
  return op.call(x1, x2);
}

at::Tensor haversine_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::haversine_distances", "")
          .typed<decltype(haversine_distances)>();
  return op.call(x1, x2);
}

at::Tensor manhattan_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::manhattan_distances", "")
          .typed<decltype(manhattan_distances)>();
  return op.call(x1, x2);
}

at::Tensor cosine_distances(const at::Tensor& x1,
                            const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::cosine_distances", "")
                       .typed<decltype(cosine_distances)>();
  return op.call(x1, x2);
}

at::Tensor linear_kernel(const at::Tensor& x1,
                         const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::linear_kernel", "")
                       .typed<decltype(linear_kernel)>();
  return op.call(x1, x2);
}

at::Tensor polynomial_kernel(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2,
                             int64_t degree, c10::optional<double> gamma,
                             double coef0) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::polynomial_kernel", "")
          .typed<decltype(polynomial_kernel)>();
  return op.call(x1, x2, degree, gamma, coef0);
}

at::Tensor sigmoid_kernel(const at::Tensor& x1,
                          const c10::optional<at::Tensor>& x2,
                          c10::optional<double> gamma, double coef0) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::sigmoid_kernel", "")
                       .typed<decltype(sigmoid_kernel)>();
  return op.call(x1, x2, gamma, coef0);
}

at::Tensor rbf_kernel(const at::Tensor& x1, const c10::optional<at::Tensor>& x2,
                      c10::optional<double> gamma) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::rbf_kernel", "")
                       .typed<decltype(rbf_kernel)>();
  return op.call(x1, x2, gamma);
}

at::Tensor laplacian_kernel(const at::Tensor& x1,
                            const c10::optional<at::Tensor>& x2,
                            c10::optional<double> gamma) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::laplacian_kernel", "")
                       .typed<decltype(laplacian_kernel)>();
  return op.call(x1, x2, gamma);
}

at::Tensor cosine_similarity(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::cosine_similarity", "")
          .typed<decltype(cosine_similarity)>();
  return op.call(x1, x2);
}

at::Tensor additive_chi2_kernel(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::additive_chi2_kernel", "")
          .typed<decltype(additive_chi2_kernel)>();
  return op.call(x1, x2);
}

at::Tensor chi2_kernel(const at::Tensor& x1,
                       const c10::optional<at::Tensor>& x2, double gamma) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::chi2_kernel", "")
                       .typed<decltype(chi2_kernel)>();
  return op.call(x1, x2, gamma);
}

// scipy
std::tuple<at::Tensor, at::Tensor, at::Tensor> directed_hausdorff_distances(
    const at::Tensor& x1, const c10::optional<at::Tensor>& x2, bool shuffle,
    c10::optional<at::Generator> generator) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::directed_hausdorff_distances", "")
          .typed<decltype(directed_hausdorff_distances)>();
  return op.call(x1, x2, shuffle, generator);
}

at::Tensor minkowski_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2, double p) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::minkowski_distances", "")
          .typed<decltype(minkowski_distances)>();
  return op.call(x1, x2, p);
}

at::Tensor wminkowski_distances(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2, double p,
                                const c10::optional<at::Tensor>& w) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::wminkowski_distances", "")
          .typed<decltype(wminkowski_distances)>();
  return op.call(x1, x2, p, w);
}

at::Tensor sqeuclidean_distances(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::sqeuclidean_distances", "")
          .typed<decltype(sqeuclidean_distances)>();
  return op.call(x1, x2);
}

at::Tensor correlation_distances(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::correlation_distances", "")
          .typed<decltype(correlation_distances)>();
  return op.call(x1, x2);
}

at::Tensor hamming_distances(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::hamming_distances", "")
          .typed<decltype(hamming_distances)>();
  return op.call(x1, x2);
}

at::Tensor jaccard_distances(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::jaccard_distances", "")
          .typed<decltype(jaccard_distances)>();
  return op.call(x1, x2);
}

at::Tensor kulsinski_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::kulsinski_distances", "")
          .typed<decltype(kulsinski_distances)>();
  return op.call(x1, x2);
}

at::Tensor kulczynski1_distances(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::kulczynski1_distances", "")
          .typed<decltype(kulczynski1_distances)>();
  return op.call(x1, x2);
}

at::Tensor seuclidean_distances(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2,
                                const c10::optional<at::Tensor>& V) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::seuclidean_distances", "")
          .typed<decltype(seuclidean_distances)>();
  return op.call(x1, x2, V);
}

at::Tensor cityblock_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::cityblock_distances", "")
          .typed<decltype(cityblock_distances)>();
  return op.call(x1, x2);
}

at::Tensor mahalanobis_distances(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2,
                                 const c10::optional<at::Tensor>& VI) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::mahalanobis_distances", "")
          .typed<decltype(mahalanobis_distances)>();
  return op.call(x1, x2, VI);
}

at::Tensor chebyshev_distances(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::chebyshev_distances", "")
          .typed<decltype(chebyshev_distances)>();
  return op.call(x1, x2);
}

at::Tensor braycurtis_distances(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::braycurtis_distances", "")
          .typed<decltype(braycurtis_distances)>();
  return op.call(x1, x2);
}

at::Tensor canberra_distances(const at::Tensor& x1,
                              const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::canberra_distances", "")
          .typed<decltype(canberra_distances)>();
  return op.call(x1, x2);
}

at::Tensor jensenshannon_distances(const at::Tensor& x1,
                                   const c10::optional<at::Tensor>& x2,
                                   c10::optional<double> base) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::jensenshannon_distances", "")
          .typed<decltype(jensenshannon_distances)>();
  return op.call(x1, x2, base);
}

at::Tensor yule_distances(const at::Tensor& x1,
                          const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::yule_distances", "")
                       .typed<decltype(yule_distances)>();
  return op.call(x1, x2);
}

at::Tensor dice_distances(const at::Tensor& x1,
                          const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::dice_distances", "")
                       .typed<decltype(dice_distances)>();
  return op.call(x1, x2);
}

at::Tensor rogerstanimoto_distances(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::rogerstanimoto_distances", "")
          .typed<decltype(rogerstanimoto_distances)>();
  return op.call(x1, x2);
}

at::Tensor russellrao_distances(const at::Tensor& x1,
                                const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::russellrao_distances", "")
          .typed<decltype(russellrao_distances)>();
  return op.call(x1, x2);
}

at::Tensor sokalmichener_distances(const at::Tensor& x1,
                                   const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::sokalmichener_distances", "")
          .typed<decltype(sokalmichener_distances)>();
  return op.call(x1, x2);
}

at::Tensor sokalsneath_distances(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::sokalsneath_distances", "")
          .typed<decltype(sokalsneath_distances)>();
  return op.call(x1, x2);
}

at::Tensor snr_distances(const at::Tensor& x1,
                         const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::snr_distances", "")
                       .typed<decltype(snr_distances)>();
  return op.call(x1, x2);
}

// aliases
at::Tensor l1_distances(const at::Tensor& x1,
                        const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::l1_distances", "")
                       .typed<decltype(l1_distances)>();
  return op.call(x1, x2);
}

at::Tensor l2_distances(const at::Tensor& x1,
                        const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::l2_distances", "")
                       .typed<decltype(l2_distances)>();
  return op.call(x1, x2);
}

at::Tensor lp_distances(const at::Tensor& x1,
                        const c10::optional<at::Tensor>& x2, double p) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::lp_distances", "")
                       .typed<decltype(lp_distances)>();
  return op.call(x1, x2, p);
}

at::Tensor linf_distances(const at::Tensor& x1,
                          const c10::optional<at::Tensor>& x2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchpairwise::linf_distances", "")
                       .typed<decltype(linf_distances)>();
  return op.call(x1, x2);
}

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "torchpairwise::euclidean_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(euclidean_distances_impl));
  m.def(
      "torchpairwise::haversine_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(haversine_distances_impl));
  m.def(
      "torchpairwise::manhattan_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(manhattan_distances_impl));
  m.def("torchpairwise::cosine_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(cosine_distances_impl));
  m.def("torchpairwise::linear_kernel(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(linear_kernel_impl));
  m.def(
      "torchpairwise::polynomial_kernel(Tensor x1, Tensor? x2=None, int "
      "degree=3, float? gamma=None, float coef0=1) -> Tensor",
      TORCH_FN(polynomial_kernel_impl));
  m.def(
      "torchpairwise::sigmoid_kernel(Tensor x1, Tensor? x2=None, float? "
      "gamma=None, float coef0=1) -> Tensor",
      TORCH_FN(sigmoid_kernel_impl));
  m.def(
      "torchpairwise::rbf_kernel(Tensor x1, Tensor? x2=None, float? "
      "gamma=None) -> Tensor",
      TORCH_FN(rbf_kernel_impl));
  m.def(
      "torchpairwise::laplacian_kernel(Tensor x1, Tensor? x2=None, float? "
      "gamma=None) -> Tensor",
      TORCH_FN(laplacian_kernel_impl));
  m.def(
      "torchpairwise::cosine_similarity(Tensor x1, Tensor? x2=None) -> Tensor",
      TORCH_FN(cosine_similarity_impl));
  m.def(
      "torchpairwise::additive_chi2_kernel(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(additive_chi2_kernel_impl));
  m.def(
      "torchpairwise::chi2_kernel(Tensor x1, Tensor? x2=None, float gamma=1.0) "
      "-> Tensor",
      TORCH_FN(chi2_kernel_impl));
  m.def(
      "torchpairwise::directed_hausdorff_distances(Tensor x1, Tensor? x2=None, "
      "*, bool shuffle=False, Generator? generator=None) -> (Tensor output, "
      "Tensor x1_indices, Tensor x2_indices)",
      TORCH_FN(directed_hausdorff_distances_impl));
  m.def(
      "torchpairwise::minkowski_distances(Tensor x1, Tensor? x2=None, float "
      "p=2) -> Tensor",
      TORCH_FN(minkowski_distances_impl));
  m.def(
      "torchpairwise::wminkowski_distances(Tensor x1, Tensor? x2=None, float "
      "p=2, Tensor? w=None) -> Tensor",
      TORCH_FN(wminkowski_distances_impl));
  m.def(
      "torchpairwise::sqeuclidean_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(sqeuclidean_distances_impl));
  m.def(
      "torchpairwise::correlation_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(correlation_distances_impl));
  m.def(
      "torchpairwise::hamming_distances(Tensor x1, Tensor? x2=None) -> Tensor",
      TORCH_FN(hamming_distances_impl));
  m.def(
      "torchpairwise::jaccard_distances(Tensor x1, Tensor? x2=None) -> Tensor",
      TORCH_FN(jaccard_distances_impl));
  m.def(
      "torchpairwise::kulsinski_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(kulsinski_distances_impl));
  m.def(
      "torchpairwise::kulczynski1_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(kulczynski1_distances_impl));
  m.def(
      "torchpairwise::seuclidean_distances(Tensor x1, Tensor? x2=None, Tensor? "
      "V=None) -> Tensor",
      TORCH_FN(seuclidean_distances_impl));
  m.def(
      "torchpairwise::cityblock_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(cityblock_distances_impl));
  m.def(
      "torchpairwise::mahalanobis_distances(Tensor x1, Tensor? x2=None, "
      "Tensor? VI=None) -> Tensor",
      TORCH_FN(mahalanobis_distances_impl));
  m.def(
      "torchpairwise::chebyshev_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(chebyshev_distances_impl));
  m.def(
      "torchpairwise::braycurtis_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(braycurtis_distances_impl));
  m.def(
      "torchpairwise::canberra_distances(Tensor x1, Tensor? x2=None) -> Tensor",
      TORCH_FN(canberra_distances_impl));
  m.def(
      "torchpairwise::jensenshannon_distances(Tensor x1, Tensor? x2=None, "
      "float? base=None) -> Tensor",
      TORCH_FN(jensenshannon_distances_impl));
  m.def("torchpairwise::yule_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(yule_distances_impl));
  m.def("torchpairwise::dice_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(dice_distances_impl));
  m.def(
      "torchpairwise::rogerstanimoto_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(rogerstanimoto_distances_impl));
  m.def(
      "torchpairwise::russellrao_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(russellrao_distances_impl));
  m.def(
      "torchpairwise::sokalmichener_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(sokalmichener_distances_impl));
  m.def(
      "torchpairwise::sokalsneath_distances(Tensor x1, Tensor? x2=None) -> "
      "Tensor",
      TORCH_FN(sokalsneath_distances_impl));
  m.def("torchpairwise::snr_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(snr_distances_impl));
  m.def("torchpairwise::l1_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(l1_distances_impl));
  m.def("torchpairwise::l2_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(l2_distances_impl));
  m.def(
      "torchpairwise::lp_distances(Tensor x1, Tensor? x2=None, float p=2) -> "
      "Tensor",
      TORCH_FN(lp_distances_impl));
  m.def("torchpairwise::linf_distances(Tensor x1, Tensor? x2=None) -> Tensor",
        TORCH_FN(linf_distances_impl));
}
namespace {
at::Tensor euclidean_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.euclidean_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("euclidean_distances", x1, x2);
  return at::cdist(x1_, x2_, 2);
}

at::Tensor haversine_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.haversine_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("haversine_distances", x1, x2);
  return _haversine(x1_, x2_);
}

at::Tensor manhattan_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.manhattan_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("manhattan_distances", x1, x2);
  return _ppminkowski(x1_, x2_, 1);
}

at::Tensor cosine_distances_impl(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.cosine_distances")
  auto output = cosine_similarity(x1, x2).neg_().add_(1).clip_(0, 2);
  if (!x2.has_value() || x1.is_same(x2.value()))
    output.diagonal(0, -2, -1).zero_();
  return output;
}

at::Tensor linear_kernel_impl(const at::Tensor& x1,
                              const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.linear_kernel")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) = utils::check_pairwise_inputs("linear_kernel", x1, x2);
  return x1_.matmul(x2_.transpose(-1, -2));
}

at::Tensor polynomial_kernel_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2,
                                  int64_t degree, c10::optional<double> gamma,
                                  double coef0) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.polynomial_kernel")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("polynomial_kernel", x1, x2);
  auto gamma_ =
      gamma.value_or(1.0 / x1.sym_size(-1).guard_int(__FILE__, __LINE__));
  return x1_.matmul(x2_.transpose(-1, -2))
      .mul_(gamma_)
      .add_(coef0)
      .pow_(degree);
}

at::Tensor sigmoid_kernel_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2,
                               c10::optional<double> gamma, double coef0) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.sigmoid_kernel")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) = utils::check_pairwise_inputs("sigmoid_kernel", x1, x2);
  auto gamma_ =
      gamma.value_or(1.0 / x1.sym_size(-1).guard_int(__FILE__, __LINE__));
  return x1_.matmul(x2_.transpose(-1, -2)).mul_(gamma_).add_(coef0).tanh_();
}

at::Tensor rbf_kernel_impl(const at::Tensor& x1,
                           const c10::optional<at::Tensor>& x2,
                           c10::optional<double> gamma) {
  C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.rbf_kernel")
  auto gamma_ =
      gamma.value_or(1.0 / x1.sym_size(-1).guard_int(__FILE__, __LINE__));
  return sqeuclidean_distances(x1, x2).mul_(-gamma_).exp_();
}

at::Tensor laplacian_kernel_impl(const at::Tensor& x1,
                                 const c10::optional<at::Tensor>& x2,
                                 c10::optional<double> gamma) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.laplacian_kernel")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) = utils::check_pairwise_inputs("laplacian_kernel", x1, x2);
  auto gamma_ =
      gamma.value_or(1.0 / x1.sym_size(-1).guard_int(__FILE__, __LINE__));
  // Do not scale a distance saved by an autograd node in place. _ppminkowski
  // currently saves only its inputs; keep the out-of-place scaling here too.
  return _ppminkowski(x1_, x2_, 1).mul(-gamma_).exp_();
}

at::Tensor cosine_similarity_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.cosine_similarity")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("cosine_similarity", x1, x2);
  auto x1_norm = x1.norm(2, -1, true);
  auto x2_norm = x2.has_value() ? x2_.norm(2, -1, true) : x1_norm;
  auto x1_nonzero = x1_norm.gt(0);
  auto x2_nonzero = x2_norm.gt(0);
  auto x1_unit = at::where(
      x1_nonzero, x1_ / at::where(x1_nonzero, x1_norm, at::ones_like(x1_norm)),
      at::zeros_like(x1_));
  auto x2_unit = at::where(
      x2_nonzero, x2_ / at::where(x2_nonzero, x2_norm, at::ones_like(x2_norm)),
      at::zeros_like(x2_));
  return x1_unit.matmul(x2_unit.transpose(-1, -2));
}

at::Tensor additive_chi2_kernel_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.additive_chi2_kernel")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("additive_chi2_kernel", x1, x2);
  utils::check_values(x1_.ge(0).all() & x2_.ge(0).all(),
                      "All elements of x1 and x2 must be non-negative");
  return _additive_chi2_kernel(x1_, x2_);
}

at::Tensor chi2_kernel_impl(const at::Tensor& x1,
                            const c10::optional<at::Tensor>& x2, double gamma) {
  C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.chi2_kernel")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) = utils::check_pairwise_inputs("chi2_kernel", x1, x2);
  utils::check_values(x1_.ge(0).all() & x2_.ge(0).all(),
                      "All elements of x1 and x2 must be non-negative");
  return _additive_chi2_kernel(x1_, x2_).mul_(gamma).exp_();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
directed_hausdorff_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2,
                                  bool shuffle,
                                  c10::optional<at::Generator> generator) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.directed_hausdorff_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) = utils::check_pairwise_inputs<false, true>(
      "directed_hausdorff_distances", x1, x2);
  return _directed_hausdorff(x1_, x2_, shuffle, generator);
}

at::Tensor minkowski_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2,
                                    double p) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.minkowski_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("minkowski_distances", x1, x2);
  // Retain cdist's validation and the existing zero/Euclidean paths.
  if (!(p > 0) || p == 2)
    return at::cdist(x1_, x2_, p);
  auto value = _ppminkowski(x1_, x2_, p);
  if (std::isinf(p))
    return value;
  // 0^(1/p) = 0 for p > 0, so the root needs no mask without autograd.
  if (!at::GradMode::is_enabled() ||
      !(x1_.requires_grad() || x2_.requires_grad()))
    return value.pow_(1. / p);
  // Mask before the root as well as after it: the derivative of the root
  // must never be evaluated at zero, even when the upstream gradient is zero.
  auto zero = value.eq(0);
  return at::where(zero, 1., value).pow(1. / p).masked_fill_(zero, 0);
}

at::Tensor wminkowski_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2,
                                     double p,
                                     const c10::optional<at::Tensor>& w) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.wminkowski_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("wminkowski_distances", x1, x2);
  TORCH_CHECK(p > 0 && !std::isnan(p), "Weighted Minkowski p must be positive");
  if (w.has_value() && std::isinf(p)) {
    auto mask = w->gt(0).to(x1_.scalar_type()).unsqueeze(-2);
    return _ppminkowski(x1_ * mask, x2_ * mask, p) + w->sum() * 0;
  }
  return w.has_value() ? _wminkowski(x1_, x2_, w.value(), p)
                       : minkowski_distances_impl(x1_, x2_, p);
}

at::Tensor sqeuclidean_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.sqeuclidean_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("sqeuclidean_distances", x1, x2);
  return _ppminkowski(x1_, x2_, 2);
}

at::Tensor correlation_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.correlation_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("correlation_distances", x1, x2);
  // centering
  x1_ = x1_ - x1_.mean(-1, true);
  x2_ = x2.has_value() ? x2_ - x2_.mean(-1, true) : x1_;

  auto x1_norm = x1_.norm(2, -1, true);
  auto x2_norm = x2.has_value() ? x2_.norm(2, -1, true) : x1_norm;
  auto denom = x1_norm.mul(x2_norm.transpose(-1, -2));
  return (x1_.matmul(x2_.transpose(-1, -2)) / denom).neg_().add_(1);
}

at::Tensor hamming_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.hamming_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("hamming_distances", x1, x2);
  return pwne_mean(x1_, x2_);
}

at::Tensor jaccard_distances_impl(const at::Tensor& x1,
                                  const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.jaccard_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("jaccard_distances", x1, x2);
  return prf_div(pwxor_sum(x1_, x2_), pwor_sum(x1_, x2_));
}

at::Tensor kulsinski_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  TORCH_WARN_DEPRECATION(
      "kulsinski_distances is deprecated in SciPy 1.9.0 and might "
      "be removed in the future. Please use kulczynski1_distances.")
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.kulsinski_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("kulsinski_distances", x1, x2);
  auto S = pwxor_sum(x1_, x2_);
  return (S - pwand_sum(x1_, x2_) +
          x1_.sym_size(-1).guard_int(__FILE__, __LINE__)) /
         (S + x1_.sym_size(-1).guard_int(__FILE__, __LINE__));
}

at::Tensor kulczynski1_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.kulczynski1_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("kulczynski1_distances", x1, x2);
  return pwand_sum(x1_, x2_) / pwxor_sum(x1_, x2_);
}

at::Tensor seuclidean_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2,
                                     const c10::optional<at::Tensor>& V) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.seuclidean_distances")
  TORCH_CHECK(!x2.has_value() || V.has_value(),
              "V is required for seuclidean when x2 is passed.")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("euclidean_distances", x1, x2);
  at::Tensor V_;
  if (V.has_value()) {
    V_ = V.value();
  } else {
    at::NoGradGuard no_grad_guard;
    V_ = x1_.var(at::IntArrayRef{-2});
  }
  utils::check_values(at::isfinite(V_).logical_and(V_.gt(0)).all(),
                      "V must contain positive finite variances");
  return _wminkowski(x1_, x2_, V_.reciprocal(), 2);
}

at::Tensor cityblock_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.cityblock_distances")
  return manhattan_distances(x1, x2);
}

at::Tensor mahalanobis_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2,
                                      const c10::optional<at::Tensor>& VI) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.mahalanobis_distances")
  TORCH_CHECK(!x2.has_value() || VI.has_value(),
              "VI is required for manhalanobis when x2 is passed.")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("mahalanobis_distances", x1, x2);
  at::Tensor VI_;
  if (VI.has_value()) {
    VI_ = VI.value();
  } else {
    at::NoGradGuard no_grad_guard;
    if (x1_.ndimension() == 2)
      VI_ = at::cov(x1_.transpose(-1, -2)).inverse().transpose_(-1, -2);
    else {
      auto cov =
          at::empty({x1_.size(0), x1_.size(2), x1_.size(2)}, x1_.options());
      for (const auto b : c10::irange(x1_.size(0))) {
        cov.index_put_({b}, at::cov(x1_[b].transpose(-1, -2)));
      }
      VI_ = cov.inverse().transpose_(-1, -2);
    }
  }
  return distance_sqrt(_sqmahalanobis(x1_, x2_, VI_));
}

at::Tensor chebyshev_distances_impl(const at::Tensor& x1,
                                    const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.chebyshev_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("chebyshev_distances", x1, x2);
  return _ppminkowski(x1_, x2_, c10::CPPTypeLimits<double>::upper_bound());
}

at::Tensor braycurtis_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.braycurtis_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("braycurtis_distances", x1, x2);
  return _braycurtis(x1_, x2_);
}

at::Tensor canberra_distances_impl(const at::Tensor& x1,
                                   const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.canberra_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("canberra_distances", x1, x2);
  return _canberra(x1_, x2_);
}

at::Tensor jensenshannon_distances_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2,
                                        c10::optional<double> base) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.jensenshannon_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs("jensenshannon_distances", x1, x2);
  utils::ValueChecks checks;
  for (const auto& value : {x1_, x2_})
    checks.require((value.ge(0).all() & value.sum(-1).gt(0).all()),
                   "Jensen-Shannon inputs must be non-negative with positive "
                   "row mass");
  checks.finish();
  // normalize 0-1
  x1_ = x1_ / x1_.sum(-1, true);
  x2_ = x2.has_value() ? x2_ / x2_.sum(-1, true) : x1_;
  return distance_sqrt(_sqjensenshannon(x1_, x2_, base).clamp_min(0));
}

at::Tensor yule_distances_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.yule_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("yule_distances", x1, x2);
  auto half_R = pwand_sum(x1_, ~x2_) * pwand_sum(~x1_, x2_);
  return prf_div(2 * half_R,
                 (pwand_sum(x1_, x2_) * pwand_sum(~x1_, ~x2_) + half_R));
}

at::Tensor dice_distances_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.dice_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("dice_distances", x1, x2);
  auto R = pwxor_sum(x1_, x2_);
  return R / (2 * pwand_sum(x1_, x2_) + R);
}

at::Tensor rogerstanimoto_distances_impl(const at::Tensor& x1,
                                         const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.rogerstanimoto_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("rogerstanimoto_distances", x1, x2);
  auto R = 2 * pwxor_sum(x1_, x2_);
  auto S = pweq_sum(x1_, x2_);
  return R / (S + R);
}

at::Tensor russellrao_distances_impl(const at::Tensor& x1,
                                     const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.russellrao_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("russellrao_distances", x1, x2);
  return (x1.sym_size(-1).guard_int(__FILE__, __LINE__) - pwand_sum(x1_, x2_)) /
         x1.sym_size(-1).guard_int(__FILE__, __LINE__);
}

at::Tensor sokalmichener_distances_impl(const at::Tensor& x1,
                                        const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.sokalmichener_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("sokalmichener_distances", x1, x2);
  auto R = 2 * pwxor_sum(x1_, x2_);
  auto S = pweq_sum(x1_, x2_);
  return R / (S + R);
}

at::Tensor sokalsneath_distances_impl(const at::Tensor& x1,
                                      const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.sokalsneath_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) =
      utils::check_pairwise_inputs<true>("sokalsneath_distances", x1, x2);
  auto R = 2 * pwxor_sum(x1_, x2_);
  // intentionally returns nan for entirely different pairs
  return R / (pwand_sum(x1_, x2_) + R);
}

at::Tensor snr_distances_impl(const at::Tensor& x1,
                              const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.snr_distances")
  at::Tensor x1_, x2_;
  std::tie(x1_, x2_) = utils::check_pairwise_inputs("snr_distances", x1, x2);
  // Compute noise/signal directly so zero noise has a finite zero gradient.
  return _snr(x1_, x2_);
}

at::Tensor l1_distances_impl(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.l1_distances")
  return manhattan_distances(x1, x2);
}

at::Tensor l2_distances_impl(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.l2_distances")
  return euclidean_distances(x1, x2);
}

at::Tensor lp_distances_impl(const at::Tensor& x1,
                             const c10::optional<at::Tensor>& x2, double p) {
  C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.lp_distances")
  return minkowski_distances(x1, x2, p);
}

at::Tensor linf_distances_impl(const at::Tensor& x1,
                               const c10::optional<at::Tensor>& x2) {
  C10_LOG_API_USAGE_ONCE(
      "torchpairwise.csrc.ops.pairwise_metrics.linf_distances")
  return chebyshev_distances(x1, x2);
}
}  // namespace
}  // namespace torchpairwise::ops
