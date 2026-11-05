# Changelog

## 1.0.0

This release overhauls the 0.3 API and implementation. See the
[migration guide](docs/migration.md) for behavior changes and rebuilding native
extensions against your installed PyTorch.

- Expand the comparison catalog to vectors, distributions, point sets, images,
  waveforms and other structured inputs, with all-pairs, paired, condensed and
  single modes where mathematically supported.
- Add lazy comparisons, streamed reductions, selection, matrix products and
  operator algebra, including an optional KeOps backend.
- Add matrix-free solvers, spectral estimates and selected matrix-function
  actions for kernel and other pairwise algorithms.
- Support custom comparison callables, explicit backward functions and reused
  encoder features, with Python and C++ interfaces.
- Extend native C++/LibTorch coverage and PyTorch compilation/export integration;
  improve batching, mixed precision, gradients and CUDA stream correctness.
- Add Python/C++ documentation, an algorithm example gallery, CMake integration
  and downstream extension examples.
- Refresh build and test workflows for current PyTorch releases and platform
  wheels. Platform results depend on the corresponding CI runs; CUDA wheel
  loading on hosted CPU runners does not validate GPU execution.

Compatibility changes include stricter input validation, corrected kernel gamma
and weighted Minkowski conventions, and a unified `lazy` API. Numerical and
deployment limitations remain documented in the relevant API guides.
