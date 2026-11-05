#include <torch/library.h>

// Setup rejects gradient-bearing inputs before computing frozen spectral state.
// Applying that state uses CompositeImplicitAutograd and supports RHS
// gradients.
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_lowrank_setup", torch::CppFunction::makeFallthrough());
  m.impl("linear_nystrom_setup", torch::CppFunction::makeFallthrough());
}
