#! /bin/bash
# Builds and repairs the Linux wheel inside quay.io/pypa/manylinux_2_28_x86_64,
# the glibc baseline of PyTorch's own Linux wheels. Run from the checkout root.
#   TORCH_VERSION  e.g. 2.15.0
#   TORCH_VARIANT  cu132 or cpu (download.pytorch.org index name)
#   PYTHON_TAG     interpreter under /opt/python (default cp311-cp311)

set -euxo pipefail
: "${TORCH_VERSION:?}" "${TORCH_VARIANT:?}"
PYTHON_TAG="${PYTHON_TAG:-cp311-cp311}"
export PATH="/opt/python/${PYTHON_TAG}/bin:${PATH}"

if [ "${TORCH_VARIANT}" != cpu ]; then
  # CUDA 13.2 (matches the torch cu132 wheels): compiler and the libraries
  # the extension links, from NVIDIA's EL8 repository.
  curl -fsSL -o /etc/yum.repos.d/cuda-rhel8.repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
  dnf install -y cuda-nvcc-13-2 cuda-cccl-13-2 cuda-cudart-devel-13-2 \
    libcublas-devel-13-2 libcurand-devel-13-2 libcusolver-devel-13-2 \
    libcusparse-devel-13-2
  dnf clean all
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH="${CUDA_HOME}/bin:${PATH}"
  nvcc -V
  source .github/workflows/cuda/Linux-env.sh
else
  export TORCHPAIRWISE_BUILD_CUDA=0
fi
gcc --version | head -n 1

python -m pip install --upgrade setuptools wheel ninja auditwheel
python -m pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

python setup.py bdist_wheel --dist-dir=dist
python tools/packaging/audit_torch_extension.py repair --plat manylinux_2_28_x86_64 -w wheelhouse dist/*.whl
rm -r dist
