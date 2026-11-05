<div class="eyebrow">Getting started</div>

# Install and Build

Use a compatible PyTorch or LibTorch installation. Native extensions must match
the PyTorch major/minor, C++ ABI and platform toolchain. CUDA execution also
requires a compatible CUDA build and NVIDIA driver.

## Python Source Installation

These examples target TorchPairwise 1.0. Install PyTorch for your platform first,
then build this checkout against that installation:

```sh
git clone https://github.com/inspiros/torchpairwise.git
cd torchpairwise
python -m pip install -e . --no-build-isolation
```

The build detects CUDA when a compatible toolkit is available. See the
[source build guide](https://github.com/inspiros/torchpairwise/blob/master/docs/building.md)
for CPU-only builds, toolchain options and matching architectures.

## Standalone C++

Build and install the CMake package against your LibTorch prefix:

```sh
cmake -S . -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_INSTALL_PREFIX=/path/to/torchpairwise-install
cmake --build build/native --parallel 2
cmake --install build/native
```

For specialized CUDA kernels, enable `TORCHPAIRWISE_WITH_CUDA=ON` and choose
compatible CUDA architectures. A build linked to CUDA LibTorch inherits its
CUDA dependencies even if custom kernels are omitted. See the
[CMake guide](https://github.com/inspiros/torchpairwise/blob/master/docs/cmake.md).

Consume the exported target:

```cmake
find_package(torchpairwise CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE torchpairwise::torchpairwise)
```

Include `<torchpairwise.h>`. The installed target carries native headers, C++20
and LibTorch linkage. Arrange runtime library search paths for both LibTorch and
torchpairwise. No Python runtime is needed for built-in C++ execution.
