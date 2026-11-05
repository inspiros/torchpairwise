#include "linear_preconditioning.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <cmath>

#include "utils/value_checks.h"

namespace torchpairwise::ops {
namespace {
using State = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
void check_matrix(const at::Tensor& value) {
  TORCH_CHECK(
      value.layout() == at::kStrided && value.dim() >= 2 &&
          (value.scalar_type() == at::kFloat ||
           value.scalar_type() == at::kDouble),
      "Low-rank setup requires strided float32/float64 [...,N,L] tensors");
}
void check_setup(const at::Tensor& value) {
  if (at::GradMode::is_enabled())
    TORCH_CHECK(!value.requires_grad(),
                "Low-rank setup is inference-only; use torch.no_grad() or "
                "detach inputs explicitly");
  if (value.is_cuda())
    TORCH_CHECK(!c10::impl::VirtualGuardImpl(value.device().type())
                     .getStream(value.device())
                     .is_capturing(),
                "Low-rank setup does not support CUDA graph capture");
}
void check_shift(const at::Tensor& value, const at::Tensor& shift) {
  TORCH_CHECK(shift.layout() == at::kStrided && shift.dim() == 0 &&
                  shift.scalar_type() == value.scalar_type() &&
                  shift.device() == value.device(),
              "Low-rank shift must be a scalar tensor matching dtype/device");
  check_setup(value);
  check_setup(shift);
  utils::ValueChecks checks;
  checks.require(value.isfinite().all(), "Low-rank inputs must be finite");
  checks.require(
      (shift.isfinite() & shift.gt(0) & shift.reciprocal().isfinite()).all(),
      "Low-rank shift must be finite, positive and have a finite reciprocal");
  checks.finish();
}
State factor_features(const at::Tensor& features, const at::Tensor& shift) {
  // Thin factors also handle dependent columns. Retain double orthogonality
  // when returning float32 actions; never construct the complementary
  // projector.
  auto [u, s, vh] = at::linalg_svd(features.to(at::kDouble), false);
  auto spectrum = s.square();
  utils::check_values((u.isfinite().all() & spectrum.isfinite().all()),
                      "Low-rank factorization produced nonfinite state");
  // SVD commonly returns column-major U. Make the custom op's output strides
  // deterministic across backends and consistent with its Fake implementation.
  return {u.clone(at::MemoryFormat::Contiguous), spectrum, shift.clone()};
}
State linear_lowrank_setup_impl(const at::Tensor& features,
                                const at::Tensor& shift) {
  check_matrix(features);
  check_shift(features, shift);
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  return factor_features(features, shift);
}
State linear_nystrom_setup_impl(const at::Tensor& cross,
                                const at::Tensor& landmarks,
                                const at::Tensor& shift, double jitter) {
  check_matrix(cross);
  check_matrix(landmarks);
  TORCH_CHECK(cross.scalar_type() == landmarks.scalar_type() &&
                  cross.device() == landmarks.device(),
              "Nyström cross and landmark matrices must share dtype/device");
  TORCH_CHECK(landmarks.sym_size(-2) == landmarks.sym_size(-1) &&
                  cross.sym_size(-1) == landmarks.sym_size(-1),
              "Nyström requires cross [...,N,L] and landmarks [...,L,L]");
  TORCH_CHECK(std::isfinite(jitter) && jitter >= 0,
              "Nyström jitter must be finite and nonnegative");
  check_shift(cross, shift);
  check_setup(landmarks);
  utils::check_values(landmarks.isfinite().all(),
                      "Nyström landmark matrix must be finite");
  at::NoGradGuard no_grad;
  c10::impl::ExcludeDispatchKeyGuard no_cpu(c10::DispatchKey::AutocastCPU),
      no_cuda(c10::DispatchKey::AutocastCUDA);
  auto w = landmarks.to(at::kDouble);
  auto eps = cross.scalar_type() == at::kFloat ? 1.1920928955078125e-7
                                               : 2.220446049250313e-16;
  // Elementwise scale keeps empty landmark/batch axes valid without max().
  utils::check_values(
      (w - w.transpose(-1, -2))
          .abs()
          .le(32 * eps *
              at::maximum(w.abs(), w.transpose(-1, -2).abs()).clamp_min(1.))
          .all(),
      "Nyström landmark matrix must be symmetric");
  auto regularized = (w * .5 + w.transpose(-1, -2) * .5).clone();
  regularized.diagonal(0, -2, -1).add_(jitter);
  auto [l, info] = at::linalg_cholesky_ex(regularized);
  utils::check_values(
      info.eq(0).all(),
      "Nyström landmark matrix plus jitter must be positive definite; choose "
      "explicit jitter for dependent landmarks");
  // F = C L^-T, hence F F.T = C (W+jitter I)^-1 C.T.
  auto batch = at::infer_size_symint(
      cross.sym_sizes().slice(0, cross.dim() - 2),
      landmarks.sym_sizes().slice(0, landmarks.dim() - 2));
  auto shape = batch;
  shape.insert(shape.end(), {cross.sym_size(-2), cross.sym_size(-1)});
  auto features =
      at::linalg_solve_triangular(
          l, cross.to(at::kDouble).expand_symint(shape).transpose(-1, -2),
          false)
          .transpose(-1, -2);
  utils::check_values(features.isfinite().all(),
                      "Nyström whitening produced nonfinite features");
  return factor_features(features, shift);
}
TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def(
      "linear_lowrank_setup(Tensor features, Tensor shift) -> (Tensor, Tensor, "
      "Tensor)");
  m.def(
      "linear_nystrom_setup(Tensor cross, Tensor landmarks, Tensor shift, "
      "float jitter=0.) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(torchpairwise, CompositeExplicitAutograd, m) {
  m.impl("linear_lowrank_setup", TORCH_FN(linear_lowrank_setup_impl));
  m.impl("linear_nystrom_setup", TORCH_FN(linear_nystrom_setup_impl));
}
}  // namespace
LinearOperator lowrank_inverse(const at::Tensor& features,
                               const at::Tensor& shift) {
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_lowrank_setup", "")
          .typed<decltype(linear_lowrank_setup_impl)>();
  auto [u, spectrum, alpha] = handle.call(features, shift);
  return lowrank_inverse_operator({u, spectrum, alpha});
}
LinearOperator lowrank_inverse(const at::Tensor& features, double shift) {
  return lowrank_inverse(features,
                         at::scalar_tensor(shift, features.options()));
}
LinearOperator nystrom_inverse(const at::Tensor& cross,
                               const at::Tensor& landmarks,
                               const at::Tensor& shift, double jitter) {
  static auto handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchpairwise::linear_nystrom_setup", "")
          .typed<decltype(linear_nystrom_setup_impl)>();
  auto [u, spectrum, alpha] = handle.call(cross, landmarks, shift, jitter);
  return lowrank_inverse_operator({u, spectrum, alpha});
}
LinearOperator nystrom_inverse(const at::Tensor& cross,
                               const at::Tensor& landmarks, double shift,
                               double jitter) {
  return nystrom_inverse(cross, landmarks,
                         at::scalar_tensor(shift, cross.options()), jitter);
}
}  // namespace torchpairwise::ops
