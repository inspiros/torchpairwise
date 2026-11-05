# Repository guidance

- When delegating, default to `gpt-6.1-sol`. Use `gpt-6-astra` only when the
  maintainer explicitly requests it; do not automatically start Astra reviews
  for new additions.
- Disposable development scripts, experiments, and benchmark output belong in
  `develop/` (ignored).
- Unit and regression tests belong in `tests/`; runnable examples belong in
  `examples/`; user documentation belongs in `docs/`.
- Built-in distances, kernels and reductions must be usable from standalone
  C++/LibTorch as well as Python. Implement shared algorithms in C++ with
  ATen operations where appropriate; bespoke CUDA kernels are optional
  optimizations. Register typed dispatcher schemas and expose thin Python
  bindings supporting torch.compile and torch.export; TorchScript compatibility
  is optional. Python-only orchestration must
  not silently become a dependency of built-in forward or backward execution.
  Arbitrary user-provided Python callbacks and encoder factories have separate
  deployment requirements. Existing Python-only implementations are migration
  work, not evidence that this requirement is already satisfied.
- Native code follows the root `.clang-format` and torchvision-style layers:
  schemas, typed dispatcher wrappers and numerical forward/backward code in
  `csrc/ops`; autograd Functions and Autograd registrations in `ops/autograd`;
  actual specialized kernels in `ops/cpu` and `ops/cuda`. Prefer shared family
  files, snake_case functions, descriptive `*Function` autograd classes and
  exported free functions rather than functors. Public operator wrappers use
  cached typed dispatcher handles; registered implementations must not call
  their own public wrapper. Generic selectors reuse the canonical native
  comparison catalog. Preserve existing serialized operator names.
- Maintainer-only plans, audit notes, and agent handoffs belong in `.maintainer/`
  (ignored). Read `.maintainer/resume.md` first: it holds the current state only
  and is overwritten at each handoff. Finished or superseded notes go to
  `.maintainer/archive/`, generated logs and manifests to `.maintainer/artifacts/`.
  The initial overhaul audit is `.maintainer/archive/overhaul-plan.md`.
- Private environment and access information belongs in `.private/` (ignored).
  Read `.private/cuda-development.md` before connecting to the CUDA development
  environment, synchronizing source, building, or running GPU validation. Keep
  machine names, account details, and private paths out of tracked documentation.
  If that file is absent in another checkout, obtain the environment details from
  the maintainer before attempting remote validation.
- CUDA builds and runtime validation use the authorized lab environment. Reuse
  existing environments where compatible; follow the private notes for placement
  of any unavoidable new environment and its uv cache. Do not replace or upgrade
  shared environments or system drivers/toolkits as a routine project action.
  The lab hosts share disks and the maintainer has no sudo access: use one shared
  checkout/environment, and plan user-space tooling without privileged installs.
- Routine implementation validation uses the authorized primary CPU/CUDA
  environment. Leave latest-CPU validation on PyTorch 2.15 to the review stage;
  do not routinely rebuild or test that environment during implementation.
- Do not routinely run standalone C++/LibTorch tests or CTest. The maintainer
  delegates that validation to Claude during review. Continue normal Python
  CPU/CUDA and compile/export checks; retain native API and build support,
  and report standalone validation as deferred unless explicitly requested for
  the current task. Installed-wheel building/installation/validation is also
  review-only; do not repeat it during routine implementation.
- Keep tests focused on distinct behavior and regressions. Avoid multiplying
  aliases, equivalent wrappers and shared execution paths into full test grids.
  Run relevant primary CPU/CUDA tests in parallel; reserve full-suite runs for
  changes that affect shared execution or integration.
- Distinguish source inspection from executed tests. Record the exact revision,
  Python/PyTorch/CUDA/compiler versions, GPU architecture, commands, and results
  for remote validation. Never describe an unbuilt extension as CUDA-tested.
