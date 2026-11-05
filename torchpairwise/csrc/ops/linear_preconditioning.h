#pragma once
#include "linear_operations.h"

namespace torchpairwise::ops {
// Setup is inference-only; stores a double thin SVD, never an N-by-N inverse.
// Fixed SPD inverse action of F F.T + shift I, F [...,N,L].
TORCHPAIRWISE_API LinearOperator lowrank_inverse(const at::Tensor& features,
                                                 const at::Tensor& shift);
TORCHPAIRWISE_API LinearOperator lowrank_inverse(const at::Tensor& features,
                                                 double shift);
// C [...,N,L], W [...,L,L]. Approximates (K + shift I)^-1 using
// C (W + jitter I)^-1 C.T. W + jitter I must be symmetric positive definite.
TORCHPAIRWISE_API LinearOperator nystrom_inverse(const at::Tensor& cross,
                                                 const at::Tensor& landmarks,
                                                 const at::Tensor& shift,
                                                 double jitter = 0.);
TORCHPAIRWISE_API LinearOperator nystrom_inverse(const at::Tensor& cross,
                                                 const at::Tensor& landmarks,
                                                 double shift,
                                                 double jitter = 0.);
}  // namespace torchpairwise::ops
