#pragma once
// Symmetric eigendecomposition safe for exactly zero CUDA matrices.
//
// cuSOLVER-backed linalg_eigh can return NaN eigenvalues for an exactly zero
// matrix when its device workspace holds stale NaN bits from earlier
// allocations. Such matrices are decomposed as the identity, keeping identity
// eigenvectors, and their spectrum is restored to zero.
#include <ATen/ATen.h>

#include <tuple>

namespace torchpairwise::ops::utils {
inline std::tuple<at::Tensor, at::Tensor> eigh(const at::Tensor& x,
                                               c10::string_view uplo = "L") {
  if (!x.is_cuda())
    return at::linalg_eigh(x, uplo);
  auto zero = x.eq(0).all(-1).all(-1).unsqueeze(-1);
  auto identity = at::eye_symint(x.sym_size(-1), x.options());
  auto [values, vectors] =
      at::linalg_eigh(at::where(zero.unsqueeze(-1), identity, x), uplo);
  return {at::where(zero, 0., values), vectors};
}
}  // namespace torchpairwise::ops::utils
