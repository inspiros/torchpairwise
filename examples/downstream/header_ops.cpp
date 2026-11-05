// A downstream operator built on the native C++ API. It includes the headers
// from torchpairwise.get_include() and links torchpairwise.library_path(),
// so it must be rebuilt whenever torchpairwise or PyTorch changes.
#include <torch/library.h>
#include <torchpairwise.h>

namespace {
// Same prediction as dispatcher_ops.cpp, streamed over column chunks: the
// [N,M] kernel matrix is never formed, in the forward or the backward pass.
at::Tensor rbf_predict(const at::Tensor& x, const at::Tensor& centers,
                       const at::Tensor& coefficients, double gamma,
                       int64_t chunk_size) {
  namespace s = torchpairwise::streaming;
  s::PairwiseOptions options;
  options.comparison.gamma = gamma;
  options.chunk_size = chunk_size;
  return s::kernel(x, centers, "rbf", options).matmul(coefficients);
}
}  // namespace

TORCH_LIBRARY(header_consumer, m) {
  m.def(
      "rbf_predict(Tensor x, Tensor centers, Tensor coefficients, float gamma,"
      " int chunk_size) -> Tensor",
      &rbf_predict);
}
