#include <torch/library.h>

// The numerical entry rejects gradient requests before its no-grad iteration.
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_lsqr", torch::CppFunction::makeFallthrough());
}
