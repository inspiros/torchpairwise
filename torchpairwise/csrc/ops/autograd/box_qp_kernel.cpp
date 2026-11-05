#include <torch/library.h>

TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  // The native inference frontend rejects gradient-bearing requests explicitly.
  m.impl("box_qp", torch::CppFunction::makeFallthrough());
}
