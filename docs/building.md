# Building and Validating from Source

Install a compatible PyTorch first, then build against that same environment:

```bash
python -m pip install setuptools wheel ninja pytest pytest-xdist scipy scikit-learn
python -m pip install --no-build-isolation --no-deps .
TORCHINDUCTOR_COMPILE_THREADS=1 python -m pytest -n 2
```

An isolated PEP 517 build may select a different PyTorch from the one in your
runtime. Native wheel metadata is restricted to the PyTorch minor used to build
it. Rebuild when changing that minor. Source dependencies accept PyTorch 2.12
through 2.15, with Python 3.10+ subject to the selected torch version's own
interpreter requirements. Executed validation covers **2.12.1 on CPU/CUDA** and
**2.15.0 on CPU**; intermediate minors and latest-PyTorch CUDA have not been
validated. An accepted dependency range is not a binary ABI or test guarantee.

PyTorch selects the C++ standard required by its headers (C++20 for 2.15).

A CUDA build requires a toolkit/compiler as well as a compatible PyTorch wheel,
GPU architecture and installed driver. Set `CUDA_HOME` when necessary. A newer
user-space toolkit cannot replace a driver that is too old for a wheel's runtime.

- `TORCHPAIRWISE_BUILD_CUDA=0` selects a CPU-only extension.
- `TORCHPAIRWISE_BUILD_CUDA=1` explicitly selects CUDA; `FORCE_CUDA=1` is retained
  for compatibility. A missing toolkit produces a clear error.
- Default `auto` builds CUDA if both the toolkit and a CUDA-enabled torch are
  present, even when GPUs are hidden at build time.
- `TORCH_CUDA_ARCH_LIST` selects target architectures; use the actual development
  GPU's architecture for a faster local build, and a deliberate list for releases.
- `TORCHPAIRWISE_PY_LIMITED_API=1` builds against the CPython stable ABI
  (`_C.abi3`, wheel tag `cp310-abi3`), so one binary serves every Python version
  that the same torch build supports. Release wheels use it; it is off by
  default so in-place builds for several interpreters keep separate files.
- `MAX_JOBS` limits compiler parallelism. `NVCC_FLAGS` accepts shell-quoted flags.
- `DEBUG=1` builds with debugging flags.

`torchpairwise.extension.cuda_arch_list()` reports this extension's compiled
targets, for example `['compute_75', 'sm_75']` for a `7.5+PTX` build. `sm_*`
means native GPU code and `compute_*` means embedded PTX. This is independent of
PyTorch's own target list and the GPU visible when the function is called.
CPU-only builds and custom builds with unresolved target metadata return `[]`.
Standard setuptools builds record the resolved nvcc gencode targets; custom
`-arch`/`-code` controls are reported as unknown rather than inferred.

CPU tests run in pull-request CI; CUDA tests run when an accessible GPU is present.
For a CUDA validation record, include source hashes, Python/torch/toolkit/compiler
versions, GPU/driver/architecture, commands and pytest results. The suite includes
non-default stream ordering, native first-order gradients, independent numerical
references, batch broadcasting and paired comparisons.

Tests use two processes with `-n 2`; each worker limits PyTorch to one CPU
thread. Keep worker counts bounded on shared GPUs. `--device cpu` or
`--device cuda` selects the device cases before scheduling. For a quick numerical
and regression check, omit the compiler suite:

```bash
python -m pytest --device cpu -n 2 -m 'not compiler'
```

Run without `-m` for full validation. Compiler checks cover representative public
routes plus each native primitive's operator contract, avoiding a full cross
product of metrics, dtypes, comparison modes and compiler backends.

Optional KeOps support: `python -m pip install "torchpairwise[keops]"`.

## Release Wheels

PyPI wheels follow the latest PyTorch's PyPI defaults: torch 2.15.0 with
CUDA 13.2 on Linux, and CPU-only on macOS and Windows, for Python 3.11 to 3.15.
Each release also builds a Windows CUDA 13.2 wheel, versioned with the local
label `+cu132` (for example `0.4.0+cu132`). PyPI does not accept local versions,
so this wheel is a release download rather than a PyPI package. Install it next
to torch from PyTorch's cu132 index:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install --no-deps torchpairwise-<version>+cu132-cp311-abi3-win_amd64.whl
```

CUDA 13 wheels need an NVIDIA driver of release 580 or newer and a Turing
(sm_75) or newer GPU. Each platform and variant gets one stable-ABI wheel,
tested on every supported Python; CI runners have no GPU, so the CUDA wheels
are tested for loading and CPU results. For another torch minor, another CUDA variant or an
older driver, install a matching PyTorch and build from source as above:

```bash
python -m pip install --no-build-isolation --no-binary torchpairwise torchpairwise
```

See [migration from the published release](migration.md) for changed behavior and
new public comparison modes.
See [binary and wheel size](binary-size.md) for measured development artifact
sizes, debug information and the release GPU architecture tradeoff.
