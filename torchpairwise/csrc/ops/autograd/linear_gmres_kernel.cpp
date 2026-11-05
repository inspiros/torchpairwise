#include <torch/library.h>

// The native numerical entry rejects gradient-bearing operands explicitly.
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_gmres", torch::CppFunction::makeFallthrough());
}
