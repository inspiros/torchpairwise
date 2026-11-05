#include <torch/library.h>
namespace torchpairwise::ops {
TORCH_LIBRARY_IMPL(torchpairwise, Autograd, m) {
  m.impl("_token_compare", torch::CppFunction::makeFallthrough());
}
}  // namespace torchpairwise::ops
