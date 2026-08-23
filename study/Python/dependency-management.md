# How dependencies are managed in the sglang repo

Date: 2026-08-12 — based on reading [python/pyproject.toml](../../python/pyproject.toml),
[python/setup.py](../../python/setup.py), the platform pyproject variants, the Docker / CI
install scripts, and the release tooling.

## There is no lockfile

Nothing in this repo is locked — no `uv.lock`, and no `requirements.txt` outside
[experimental/sgl-router/tests/e2e/](../../experimental/sgl-router/tests/e2e/).
Python dependency management is: one hand-maintained `dependencies` list per
platform, pinning discipline encoded as inline comments, and CI as the
resolution test. The Rust side **is** locked — [rust/Cargo.lock](../../rust/Cargo.lock)
is committed.

## Four separately-versioned artifacts

| Artifact | Manifest | Build backend |
| --- | --- | --- |
| `sglang` | [python/pyproject.toml](../../python/pyproject.toml) | setuptools + setuptools-rust + setuptools-scm |
| `sglang-kernel` (AOT CUDA kernels) | [python/sglang/kernels/aot/pyproject.toml](../../python/sglang/kernels/aot/pyproject.toml) | scikit-build-core (CMake) |
| Rust PyO3 extensions | [rust/Cargo.toml](../../rust/Cargo.toml) workspace | cargo, built *into* the sglang wheel |
| sgl-router / model-gateway | [experimental/sgl-router/](../../experimental/sgl-router/), [sgl-model-gateway/](../../sgl-model-gateway/) | separate |

`sglang` depends on `sglang-kernel` as an ordinary version pin
([pyproject.toml:73](../../python/pyproject.toml#L73)) even though it is built from this
same tree — the kernel package releases on its own cadence via
[scripts/release/bump_kernel_version.py](../../scripts/release/bump_kernel_version.py).

## The `cp pyproject_X.toml pyproject.toml` swap

The checked-in [pyproject.toml](../../python/pyproject.toml) is **the CUDA one**. Other
platforms overwrite it before installing:

| Variant | Consumers |
| --- | --- |
| [pyproject_cpu.toml](../../python/pyproject_cpu.toml) | [xeon.Dockerfile:41](../../docker/xeon.Dockerfile#L41), [arm64.Dockerfile:43](../../docker/arm64.Dockerfile#L43) — publishes under a *different distribution name*, `sglang-cpu` |
| [pyproject_other.toml](../../python/pyproject_other.toml) | ROCm, MUSA, Apple Metal, MThreads — [rocm.Dockerfile:322](../../docker/rocm.Dockerfile#L322), [amd_ci_install_dependency.sh:133](../../scripts/ci/amd/amd_ci_install_dependency.sh#L133) |
| [pyproject_npu.toml](../../python/pyproject_npu.toml) | [npu.Dockerfile:96](../../docker/npu.Dockerfile#L96) |
| [pyproject_xpu.toml](../../python/pyproject_xpu.toml) | [xpu.Dockerfile:77](../../docker/xpu.Dockerfile#L77) |

So the dependency list is duplicated 5×, and the variants drift deliberately:

- CUDA pins `torch==2.13.0`; CPU and XPU pin `2.12.0`.
- XPU uses `+xpu` local-version wheels off Intel's index (`torch==2.12.0+xpu`,
  `torchvision==0.27.0+xpu`, ...).
- NPU pins `torchao==0.9.0` against CUDA's `0.17.0`.
- The entire NVIDIA block — `cuda-python`, `cuda-tile`, `flashinfer_python[cu13]`,
  `humming-kernels[cu13]`, `nvidia-cutlass-dsl`, `nvidia-mathdx`, `nvidia-ml-py`,
  `quack-kernels`, `sgl-deep-ep`, `sgl-deep-gemm`, `sglang-kernel`, `tilelang`,
  `flash-attn-4`, `numba` — is simply absent from the non-CUDA variants.

`pyproject_other.toml` is structurally different from the rest: it has an
almost-empty base `dependencies` and layers everything through self-referencing
extras (`runtime_base` → `runtime_common` → `srt_empty` / `srt_hip` /
`diffusion_hip`). That lets an out-of-tree plugin run
`pip install -e ".[srt_empty]"` and get sglang with **no torch** in the chain,
then bring its own `torch_npu` / `torch_musa`.

## Extras as composition layers

In the CUDA file: `test` → `dev` → `all`, plus feature extras (`diffusion`,
`diffusion-qvg`, `tracing`, `http2`, `ray`, `fastokens`, `runai`,
`checkpoint-engine`). They compose by self-reference —
`dev = ["sglang[test]"]`, and `test` itself pulls `sglang[fastokens]`.

## Pinning conventions

- Alphabetical order is a stated rule ([pyproject.toml:17](../../python/pyproject.toml#L17)).
- Exact pins carry a *reason comment* inline. Examples:
  [`cuda-tile==1.6.0rc5`](../../python/pyproject.toml#L27-L28) — "1.6.0rc6 shipped no cp310
  linux x86_64 wheel"; the `compressed-tensors==0.15.0` note in
  [pyproject_other.toml](../../python/pyproject_other.toml#L113-L116) about pip backtracking
  into an unbuildable ancient setuptools sdist.
- Platform availability uses PEP 508 markers rather than more variants where
  possible — see the `aarch64` / `arm64` guards on `av`, `decord2`, `torchcodec`,
  `st_attn`, `vsa`.
- Cross-file pin coupling is flagged in comments, e.g. flashinfer must "keep it
  aligned with jit-cache version in Dockerfile"
  ([pyproject.toml:36](../../python/pyproject.toml#L36)).

## Things that can't live in pyproject

- **Git-only deps.** PyPI rejects distribution metadata carrying a direct URL
  requirement, so `sgl-eval` is deliberately *not* declared and is installed by
  [ci_install_dependency.sh:726](../../scripts/ci/cuda/ci_install_dependency.sh#L726)
  instead — with `antlr4-python3-runtime==4.9.3` declared in the `test` extra
  purely to keep that later install compatible
  ([pyproject.toml:167-173](../../python/pyproject.toml#L167-L173)). The XPU variant ignores
  this rule and uses `sgl-kernel @ git+https://...sgl-kernel-xpu.git`; it isn't
  uploaded to PyPI.
- **Alternate wheel indexes.** Resolved at install time, not declared:
  `download.pytorch.org/whl/cu1XX`, `docs.sglang.ai/whl/cu1XX` (sglang-kernel,
  sgl-deep-gemm), `flashinfer.ai/whl`. `[[tool.uv.index]]` only pins PyPI as the
  default index.
- **Build-time deps of sdists.** `[tool.uv.extra-build-dependencies]` injects
  `setuptools, torch` for `st-attn` and `vsa`, which need torch importable to build.
- **Hugging Face Hub kernels.** `[tool.kernels.dependencies]` at
  [pyproject.toml:254](../../python/pyproject.toml#L254) pulls
  `kernels-community/sgl-flash-attn3` through the `kernels` library — a dependency
  channel that isn't pip at all.

## Version and Rust-extension wiring in setup.py

Version is dynamic: setuptools-scm derives it from git via a custom
`git_describe_command` ([tools/get_version_tag.py](../../scripts/release/get_version_tag.py)),
writing `sglang/_version.py`, with `fallback_version = "0.0.0.dev0"` so editable
installs work without `.git` metadata.

Rust extensions are **not** listed anywhere in pyproject. [setup.py](../../python/setup.py)
shells out to `cargo metadata` against [rust/Cargo.toml](../../rust/Cargo.toml)
and builds one PyO3 module per crate declaring:

```toml
[package.metadata.sglang]
python-module = "sglang.srt.<pkg>._core"
```

Two filters narrow the discovered set:

1. `[tool.sglang] rust-extensions` in the *active* pyproject — how
   `pyproject_other.toml` builds only the multimodal crate (grpc needs
   proto/tonic and is intentionally CUDA-only).
2. `SGLANG_BUILD_RUST_EXTS` env var at build time — `all` / `none` /
   comma-separated substrings.

Adding a crate therefore needs no pyproject edit. Rust dep versions themselves
are centralized in `[workspace.dependencies]` and members pull them with
`dep = { workspace = true }`.

## Where the real resolution happens

For CUDA users the [docker/Dockerfile](../../docker/Dockerfile) is the de-facto
lock: per-CU-version wheel swaps, `--force-reinstall --no-deps` for kernel
wheels, a `constraints.txt` at [line 557](../../docker/Dockerfile#L557), and a
final `pip install --no-deps -e "python[${BUILD_TYPE}]"`.
[scripts/ci/cuda/ci_install_dependency.sh](../../scripts/ci/cuda/ci_install_dependency.sh)
mirrors that with `uv`, keyed on `CU_VERSION` (default `cu130`).

End users get the simpler path in
[docs/docs/get-started/install.mdx](../../docs/docs/get-started/install.mdx):
`uv pip install --prerelease=allow sglang`, with documented follow-up
`--force-reinstall` commands to move the torch stack from cu13 to cu12.

## Automation and guardrails

Because versions are duplicated across pyprojects, Dockerfiles, and docs, bumps
are scripted (see [scripts/release/README.md](../../scripts/release/README.md)):

- [bump_sglang_version.py](../../scripts/release/bump_sglang_version.py) — touches
  the Makefile, 3 pyprojects, rocm.Dockerfile, and 3 doc pages.
- [bump_kernel_version.py](../../scripts/release/bump_kernel_version.py) — the 4
  AOT-kernel pyprojects plus `sgl_kernel/version.py`.
- [bump_docs_install_version.py](../../scripts/release/bump_docs_install_version.py) —
  run by a bot workflow on release-tag push.

Two registered tests police the manifests:

- [test_get_version_tag.py](../../test/registered/unit/tools/test_get_version_tag.py)
  — checks all four pyproject variants.
- [test_srt_empty_deps.py](../../test/registered/core/test_srt_empty_deps.py)
  — asserts `runtime_base` in `pyproject_other.toml` stays torch-free.

`.pre-commit-config.yaml` adds `check-toml` (syntax only, not semantics).

## Practical upshot when adding a dependency

1. Edit [python/pyproject.toml](../../python/pyproject.toml) in alphabetical position, with
   a comment justifying any exact pin.
2. Decide consciously whether it also belongs in the four platform variants —
   nothing propagates automatically.
3. Nothing will catch an omission until that platform's CI job or Docker build
   runs; there's no lockfile or resolver check to lean on.

See also [SETUP_ENV.md](../../studyIterations/SETUP_ENV.md) for the local
editor environment (which installs none of this).
