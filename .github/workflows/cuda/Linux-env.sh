#! /bin/bash

export FORCE_CUDA=1
# CUDA 13 supports sm_75 (Turing) and newer; keep PTX for future GPUs.
export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.6 9.0 10.0 12.0+PTX'
