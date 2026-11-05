#include <torch/library.h>

// The native solver explicitly rejects gradient requests before entering its
// no-grad iteration. Inference dispatch must not acquire a fallback grad_fn.
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("linear_cg", torch::CppFunction::makeFallthrough());
  m.impl("linear_pcg", torch::CppFunction::makeFallthrough());
}
