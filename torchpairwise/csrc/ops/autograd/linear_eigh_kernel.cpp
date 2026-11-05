#include <torch/library.h>

// Inference-only execution rejects gradient requests in the native frontend.
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_eigh", torch::CppFunction::makeFallthrough());
}
