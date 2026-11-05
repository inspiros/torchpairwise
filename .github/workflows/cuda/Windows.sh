#! /bin/bash

set -e
set -x

# Install CUDA 13.2 (matches the torch cu132 wheels). CUDA 13 replaced the
# thrust subpackage with cccl and ships crt/nvvm separately from nvcc.
echo "Installing CUDA 13.2"
curl -L -o cuda.exe https://developer.download.nvidia.com/compute/cuda/13.2.2/network_installers/cuda_13.2.2_windows_network.exe
./cuda.exe -s nvcc_13.2 crt_13.2 nvvm_13.2 cudart_13.2 cccl_13.2 cublas_dev_13.2 curand_dev_13.2 cusolver_dev_13.2 cusparse_dev_13.2
rm cuda.exe

export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin":$PATH
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"

# test
nvcc -V
