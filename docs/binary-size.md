# Native Binary and Wheel Size

Metric count does not directly predict wheel size. Built-in comparisons use
native C++/CUDA operators or C++ compositions of existing PyTorch operators,
with thin Python adapters. An ATen composition adds host code without
recompiling those PyTorch operators into the extension. PyTorch remains an
external dependency, not a bundled runtime in this wheel.

Moving that composition to C++ adds host code, schemas and native autograd
orchestration. It still calls PyTorch's already-built operators; it does not
duplicate their CUDA kernels in our extension. C++ portability therefore has
a measurable cost without requiring a new CPU/CUDA kernel for every metric.

New native operators add code for supported scalar types, CPU/CUDA execution
and backward paths. CUDA builds may embed several GPU machine-code targets
and PTX. More target architectures increase size and compilation work, with
shared host code and metadata preventing a simple linear size estimate.
Optional `torch.compile` code is generated in the user's runtime cache; it is
not precompiled into the published extension by this package.

## Development Artifact Measurement

Measured on 2026-10-29: Linux x86-64, CPython 3.10, PyTorch 2.12.1, CUDA toolkit
12.1, GPU target sm_75. These are development artifacts, not a promised size
for a release covering several architectures or a different toolchain.

| Artifact | Size |
| --- | ---: |
| Existing wheel download | 39.02 MiB |
| Extension installed from that wheel | 114.97 MiB |
| Extension after removing debug sections, disposable copy | 19.48 MiB |
| Compressed extension after removing debug sections | 2.90 MiB |
| Other compressed wheel contents | 0.16 MiB |

The subsequent Gaussian-mixture/sequence/audio/token additions, feature wrapper
and first streaming reductions increased the complete wheel by 32,338 bytes,
to 39.05 MiB, with an identical native extension. A separately repacked candidate
with `--strip-debug` is 3.11 MiB. Both complete wheels passed 653 isolated
installed-package CPU/CUDA checks. This remains a single-sm_75 development
build; a release with several GPU architectures needs its own measurement.

Removing debug sections from this copy reduced its compressed extension by
about 92.5%. A complete repacked wheel would add its metadata and other files;
the compressed-extension row is not itself a wheel size. Import, native
distance forward/backward and Chamfer forward/backward passed on CPU and CUDA
using a disposable debug-stripped package. The original binary was preserved.
The initial smoke check and subsequent installed-wheel validation do not replace
testing future release builds with their chosen architecture/toolchain matrix.

## First C++ Composition Migration

The first 23-function vector/probability/histogram/radial migration uses shared
C++ ATen execution and native first-backward replay, including all-pairs,
paired and symmetric condensed modes. On the same development toolchain:

| Artifact | Before | After | Increase |
| --- | ---: | ---: | ---: |
| Wheel including development debug sections | 40,955,250 B | 43,989,979 B | 3,034,729 B |
| Wheel after `--strip-debug` | 3,266,571 B | 3,422,436 B | 155,865 B |
| Installed extension after `--strip-debug` | 20,428,776 B | 21,121,184 B | 692,408 B |

The stripped wheel grows by about 4.8%, or 152 KiB. This includes the first
shared native comparison/replay infrastructure and schemas, so it is not a
per-metric growth rate to extrapolate across the remaining families. Native
CUDA source and GPU targets are unchanged in this migration. Original binaries
are preserved; stripping is applied only to separate artifact copies. These
measurements do not include PyTorch's separate installation size.

## Weighted C++ Migration

The next migration adds shared native execution and first backward for 19
weighted metrics, plus the legacy weighted-Minkowski entry name and aliases.
All-pairs, paired and condensed modes use the same C++ entry point. With the
same development toolchain and sm_75 target:

| Artifact | Before | After | Increase |
| --- | ---: | ---: | ---: |
| Wheel including development debug sections | 43,989,979 B | 45,882,091 B | 1,892,112 B |
| Wheel after `--strip-debug` | 3,422,436 B | 3,487,108 B | 64,672 B |
| Installed extension after `--strip-debug` | 21,121,184 B | 21,291,656 B | 170,472 B |

The stripped wheel grows by about 1.9%, or 63 KiB. This also removes the two
superseded Python formula modules. It adds host ATen/replay code and a typed
schema, reusing the existing optimized Minkowski primitives; no CUDA kernel or
GPU architecture is added. These figures remain specific to this toolchain.

## Nearest-Point Set Migration

Shared C++ layout/length/mode orchestration and nearest-point backward for
Chamfer and directed/symmetric Hausdorff add the following on the same toolchain:

| Artifact | Before | After | Increase |
| --- | ---: | ---: | ---: |
| Wheel including development debug sections | 45,882,091 B | 49,170,693 B | 3,288,602 B |
| Wheel after `--strip-debug` | 3,487,108 B | 3,577,850 B | 90,742 B |
| Installed extension after `--strip-debug` | 21,291,656 B | 21,516,000 B | 224,344 B |

The stripped wheel grows by about 2.6%, or 89 KiB. The existing native nearest
search and optimized Hausdorff forward kernels are reused; no CUDA kernel or
GPU architecture is added. This measurement also includes native replay and
witness-gradient orchestration, with Python-only built-in formulas removed.
It remains a single-sm_75 development artifact, not a multi-architecture release
size guarantee.

## Remaining Built-in C++ Migration

The remaining native migration covers energy/MMD and transport, Gaussian/SPD
and mixture comparisons, image/audio scores, sequence/token distances, generic
comparison selection, streaming reductions and dense neighbor selection. It
uses shared C++ ATen algorithms and native backward implementations with the
same Linux/CPython 3.10/PyTorch 2.12.1/CUDA toolkit 12.1/sm_75 development target:

| Artifact | Nearest-point stage | Complete built-in stage | Increase |
| --- | ---: | ---: | ---: |
| Wheel including development debug sections | 49,170,693 B | 64,253,636 B | 15,082,943 B |
| Wheel after `--strip-debug` | 3,577,850 B | 4,142,051 B | 564,201 B |
| Installed extension after `--strip-debug` | 21,516,000 B | 23,022,696 B | 1,506,696 B |

The stripped wheel is about 3.95 MiB, an increase of about 551 KiB (15.8%) for
this entire migration. Its installed extension is about 21.96 MiB. The larger
unstripped increase is mostly debug information. Stripping is performed on a
separate copy; the validated source-tree binary is preserved. These figures
exclude PyTorch and its CUDA dependencies. They do not predict the size of a
future release targeting several GPU architectures or another toolchain.

## C++ Convention and Dispatcher Unification

Separating native autograd into `ops/autograd`, replacing legacy functors with
free functions, and using typed public dispatcher wrappers adds the following
on the same development toolchain and sm_75 target:

| Artifact | Before | After | Increase |
| --- | ---: | ---: | ---: |
| Wheel including development debug sections | 64,253,636 B | 77,387,433 B | 13,133,797 B |
| Wheel after `--strip-debug` | 4,142,051 B | 4,306,625 B | 164,574 B |
| Installed extension after `--strip-debug` | 23,022,696 B | 23,438,304 B | 415,608 B |

The stripped wheel is about 4.11 MiB, growing by 4.0%; the stripped installed
extension grows by 1.8%. This includes shared native name routing and the private
histogram/NaN autograd migration. The larger unstripped increase includes debug
information from the reorganized translation units. Operator schemas, numerical
contracts and GPU targets are preserved. The original extension remains intact;
stripping applies only to the separate packaging copy.

## Release Approach

Choose supported GPU targets explicitly using `TORCH_CUDA_ARCH_LIST`, rather
than publishing a wheel accidentally targeted only at the builder's GPU.
The target list must match the selected PyTorch/CUDA/toolchain support policy.
CPU-only builds can use `TORCHPAIRWISE_BUILD_CUDA=0`; CUDA builds can use
`TORCHPAIRWISE_BUILD_CUDA=1` with a compatible toolkit.

Avoid publishing development debug information by accident. On Linux,
`strip --strip-debug` on the built extension is a measured packaging option;
preserve the original artifact for debugging, then test the actual packaged
wheel after stripping. Retaining symbols separately can help diagnose native
failures. Other platforms require their own symbol-handling tools. More
aggressive symbol stripping needs separate compatibility validation.

Record compressed wheel size and installed extension size in release artifact
reports, together with target architectures, compiler flags and toolchain.
Compare equivalent build settings when assessing growth from a new operator.
Prefer shared implementations or PyTorch compositions where appropriate;
dedicated native kernels should earn their size through measured benefits.
Per-family optional binaries are a possible later tradeoff, but would increase
build, installation and compatibility complexity. Current measurements do not
justify that split.
