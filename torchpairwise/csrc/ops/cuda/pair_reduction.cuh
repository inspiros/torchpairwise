#pragma once
// Chunked pair reductions for legacy CUDA backward kernels.
//
// Each thread accumulates one output element over a contiguous chunk of the
// reduced sample axis, in the operator math type. Chunk partials are summed
// with ATen: no atomics, deterministic results, and enough threads to fill
// the device even when one side of the comparison is small.
#include <ATen/ATen.h>
#include <ATen/OpMathType.h>

#include <algorithm>

#include "cuda_helpers.h"

namespace torchpairwise::ops::cuda_detail {
template <typename scalar_t>
using Accessor3 =
    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits,
                                    int64_t>;
template <typename scalar_t>
using Accessor2 =
    at::GenericPackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits,
                                    int64_t>;

template <typename Term>
C10_LAUNCH_BOUNDS_1(256)
__global__
    void pair_reduction_kernel(int64_t n_kernels, int64_t batch, int64_t rows,
                               int64_t width, int64_t others, int64_t chunk,
                               Term term, typename Term::acc_t* partial) {
  CUDA_1D_KERNEL_LOOP_T(index, n_kernels, int64_t) {
    int64_t e = index % width;
    int64_t r = (index / width) % rows;
    int64_t b = (index / (width * rows)) % batch;
    int64_t start = index / (width * rows * batch) * chunk;
    int64_t stop = start + chunk < others ? start + chunk : others;
    typename Term::acc_t value = 0;
    for (int64_t o = start; o < stop; ++o)
      value += term(b, r, o, e);
    partial[index] = value;
  }
}

// Returns sum_o term(b, r, o, e) with shape [batch, rows, width], in the
// term's accumulation dtype.
template <typename Term>
at::Tensor pair_reduction(const Term& term, int64_t batch, int64_t rows,
                          int64_t width, int64_t others,
                          const at::TensorOptions& options) {
  using acc_t = typename Term::acc_t;
  auto acc_options = options.dtype(c10::CppTypeToScalarType<acc_t>::value);
  auto outputs = batch * rows * width;
  if (outputs == 0 || others == 0)
    return at::zeros({batch, rows, width}, acc_options);
  // Split the reduced axis only while the outputs alone underfill the device,
  // keeping at least 64 terms per thread.
  constexpr int64_t target_threads = int64_t(1) << 18;
  int64_t chunks = std::clamp<int64_t>(target_threads / outputs, 1,
                                       std::max<int64_t>(1, others / 64));
  int64_t chunk = (others + chunks - 1) / chunks;
  chunks = (others + chunk - 1) / chunk;
  auto partial = at::empty({chunks, batch, rows, width}, acc_options);
  constexpr unsigned int threads = 256;
  auto n_kernels = partial.numel();
  pair_reduction_kernel<Term><<<GET_BLOCKS(threads, n_kernels), threads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      n_kernels, batch, rows, width, others, chunk, term,
      static_cast<acc_t*>(partial.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return chunks == 1 ? partial.select(0, 0) : partial.sum(0);
}

// Gradient entry for row r of the reduced side's partner o.
template <typename scalar_t>
__device__ __forceinline__ scalar_t pair_entry(const Accessor3<scalar_t>& g,
                                               bool transpose, int64_t b,
                                               int64_t r, int64_t o) {
  return transpose ? g[b][o][r] : g[b][r][o];
}

// sum_o g * (own_r - other_o), the exact difference sum shared by squared
// Euclidean (p=2) and Mahalanobis gradients.
template <typename scalar_t>
struct DifferenceTerm {
  using acc_t = at::opmath_type<scalar_t>;
  Accessor3<scalar_t> grad, own, other;
  bool transpose;
  __device__ acc_t operator()(int64_t b, int64_t r, int64_t o,
                              int64_t e) const {
    return acc_t(pair_entry(grad, transpose, b, r, o)) *
           (acc_t(own[b][r][e]) - acc_t(other[b][o][e]));
  }
};
}  // namespace torchpairwise::ops::cuda_detail
