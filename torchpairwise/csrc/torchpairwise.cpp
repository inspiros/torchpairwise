#ifdef USE_PYTHON
#include <Python.h>
#endif  // USE_PYTHON

#include <torch/script.h>

#include "torchpairwise.h"

#ifdef WITH_CUDA

#include <cuda.h>

#endif

// If we are in a Windows environment, we need to define
// initialization functions for the _C extension
#ifdef _WIN32
#ifdef USE_PYTHON
PyMODINIT_FUNC PyInit__C(void) {
  // No need to do anything.
  // extension.py will run on load
  return nullptr;
}
#endif  // USE_PYTHON
#endif  // _WIN32

namespace torchpairwise {
int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

std::string cuda_arch_flags() {
#if defined(WITH_CUDA) && defined(TORCHPAIRWISE_CUDA_ARCH_FLAGS)
  return TORCHPAIRWISE_CUDA_ARCH_FLAGS;
#else
  // CPU-only or a custom build without resolved architecture metadata.
  return {};
#endif
}

TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
  m.def("_cuda_version", &cuda_version);
  m.def("_cuda_arch_flags", &cuda_arch_flags);
}
}  // namespace torchpairwise
