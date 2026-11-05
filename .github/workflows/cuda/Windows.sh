#! /bin/bash

set -e
set -x

# Install CUDA 13.2 (matches the torch cu132 wheels). CUDA 13 ships crt/nvvm
# separately from nvcc. The 13.2 Windows installer still names CCCL thrust_13.2
# (cccl_13.x only appears in later releases), and fails on unknown names.
echo "Installing CUDA 13.2"
CUDA_ROOT="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
curl -L -o cuda.exe https://developer.download.nvidia.com/compute/cuda/13.2.2/network_installers/cuda_13.2.2_windows_network.exe
status=0
./cuda.exe -s nvcc_13.2 crt_13.2 nvvm_13.2 cudart_13.2 thrust_13.2 cublas_dev_13.2 curand_dev_13.2 cusolver_dev_13.2 cusparse_dev_13.2 \
  -loglevel:6 -log:"$PWD/cuda_install_logs" || status=$?
rm cuda.exe
if [ ! -x "$CUDA_ROOT/bin/nvcc.exe" ]; then
  echo "CUDA install failed (installer exit code $status)"
  cat cuda_install_logs/LOG.setup.exe.log cuda_install_logs/LOG.RunDll32.exe.log 2>/dev/null || ls -R cuda_install_logs || true
  exit 1
fi
rm -rf cuda_install_logs

export PATH="$CUDA_ROOT/bin":$PATH
export CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"

# test
nvcc -V
