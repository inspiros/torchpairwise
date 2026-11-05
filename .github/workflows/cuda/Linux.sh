#! /bin/bash

set -e
set -x

# Install CUDA 13.2 (matches the torch cu132 wheels):
echo "Installing CUDA 13.2"

# Must match the runner image (ubuntu-22.04).
OS=ubuntu2204

wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get -y install cuda-nvcc-13-2 cuda-cccl-13-2 cuda-cudart-dev-13-2 libcublas-dev-13-2 libcurand-dev-13-2 libcusolver-dev-13-2 libcusparse-dev-13-2
sudo apt clean

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda"

# test
nvcc -V
