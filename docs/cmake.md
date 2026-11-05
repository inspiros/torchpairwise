# Standalone C++ Installation

Build a shared torchpairwise library against an existing LibTorch installation.
Python development headers, Python embedding libraries, and a Python interpreter
at runtime are not required. Some LibTorch CUDA CMake distributions use a Python
interpreter during configuration; that is an upstream tooling dependency.
The native source manifest is shared with setuptools; both builds use the same algorithms, dispatcher schemas, and autograd code.

Requirements: CMake 3.25 or newer, a C++20 compiler, and compatible LibTorch.
CUDA builds additionally need a CUDA-enabled LibTorch and a compatible CUDA
compiler/toolkit supporting C++20. OpenMP is enabled by default for specialized
CPU loops; use `-DTORCHPAIRWISE_WITH_OPENMP=OFF` for serial loops.

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_INSTALL_PREFIX=/path/to/torchpairwise-install \
  -DTORCHPAIRWISE_BUILD_TESTS=ON
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
cmake --install build/native
```

The default omits specialized CUDA kernels. Add these options for CUDA, choosing
architectures supported by your users' GPUs and the selected toolchain:

```sh
TORCH_CUDA_ARCH_LIST="7.5" cmake -S . -B build/native-cuda \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/cuda-libtorch \
  -DCMAKE_INSTALL_PREFIX=/path/to/torchpairwise-cuda-install \
  -DTORCHPAIRWISE_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DTORCHPAIRWISE_BUILD_TESTS=ON
```

Set both architecture controls consistently: some LibTorch CMake releases use
`TORCH_CUDA_ARCH_LIST` internally and override `CMAKE_CUDA_ARCHITECTURES`.

`torchpairwise::cuda_arch_flags()` reports space-separated `sm_*` (native code)
and `compute_*` (PTX) targets from resolved gencode flags and numeric CMake
architectures, including `-real`/`-virtual` suffixes. CPU-only builds return an
empty string. Unresolved custom controls (`native`, `all`, custom `-arch`/`-code`,
or configuration-specific architecture flags) also return an empty string.

A build without specialized CUDA kernels can use a CUDA-enabled LibTorch, but
will still inherit that LibTorch distribution's CUDA dependencies. ATen-only
compositions may then run on CUDA, while omitted custom CUDA kernels are unavailable. For an installation
without CUDA dependencies, use CPU-only LibTorch. CUDA CMake tests require a
visible compatible GPU; compilation itself does not require one.

The install contains the shared library, native headers, and an exported CMake
package. Consume it through the exported target:

```cmake
find_package(torchpairwise CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE torchpairwise::torchpairwise)
```

```cpp
#include <torchpairwise.h>
auto distances = torchpairwise::ops::euclidean_distances(X, Y);
```

Configure the [external consumer example](../examples/cmake/) with both prefixes:

```sh
cmake -S examples/cmake -B build/consumer \
  -DCMAKE_PREFIX_PATH="/path/to/torchpairwise-install;/path/to/libtorch"
cmake --build build/consumer --parallel 2
./build/consumer/kernel_product
```

The target carries include paths, C++20, and LibTorch linkage. Use a LibTorch
with the same major/minor version and C++ ABI as the build; the installed config
rejects a different LibTorch minor. Matching the minor alone does not guarantee
compiler, ABI, or CUDA compatibility. Configure the platform loader to find
both libraries after moving an installation: on Linux, use application RPATH
or `LD_LIBRARY_PATH`; on Windows, arrange the DLL search path. LibTorch and CUDA
runtimes are dependencies, not bundled into this install.

Opening the repository's `CMakeLists.txt` in CLion provides the native target.
Set the LibTorch prefix in your local CMake profile and enable ClangFormat so
CLion reads the root `.clang-format`. Keep machine-specific profiles in the IDE
or ignored `CMakeUserPresets.json`; do not commit local installation paths.

This is a separate native build; Python wheels continue to use setuptools and
their torch-minor-specific extension. Do not load the standalone library and the
wheel extension into the same process: each registers the same operator names.
