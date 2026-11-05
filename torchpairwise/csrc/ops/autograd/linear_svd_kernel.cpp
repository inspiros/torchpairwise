#include <torch/library.h>

// The native inference frontend rejects gradient-bearing inputs explicitly.
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_svd", torch::CppFunction::makeFallthrough());
}
