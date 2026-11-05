#include <torch/library.h>

TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_minres", torch::CppFunction::makeFallthrough());
}
