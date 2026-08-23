# Setting up a local Python environment

Asked `Claude Code`:

    I cloned this repo, opened in VS Code, I can see it contains Python code and pyproject.toml, and the outside shell's
    default Python intepretor /opt/homebrew/bin/python3 was picked up; Create a script under studyIterations to set up
    Python environment based on python/pyproject.toml, a MD file under the same directory to document it.

## Set Python environment

`setup_env.sh` (same directory) creates a project-local virtualenv and installs
SGLang into it as an editable install, using the pyproject that matches your
machine. The script lives here, but the package it installs — and every
pyproject variant it chooses between — lives in the repo's `python/` directory;
it resolves that from the repo root, so it works from any cwd.

```bash
# from the repo root
./studyIterations/setup_env.sh
source .venv/bin/activate
```

That gives you `<repo>/.venv`, which VS Code picks up automatically instead of
whatever `python3` happens to be first on your `PATH`.

### IDE: VS Code

VS Code will automatically detect the new Python env, when asked, click Yes to use it;
Then reload the window by
- Shift + Cmd + P / Developer: Reload Window
- Or restarting VS Code

### IDE: IntelliJ

IntelliJ configures the SDKs at application level, not at project level, this makes the tool not be able to configure the Python SDK as above.

Manually configure project specific Python SDK:
- Go to File / Project Structure... / Project Settings / Project / SDK
- If the Python SDK based on .venv/bin/python has not been created
  1. Click the + sigh at top to add a SDK
  2. Add Python SDK from disk...
  3. Existing environment
  4. Click the ... to navigate .venv/bin
  5. Select python

There are still errors like

    python/sglang/srt/managers/multi_tokenizer_mixin.py, IntelliJ editor displays error that the sglang package is not found

CC's Diagnosis: this is an IntelliJ/PyCharm resolution issue, not a real packaging problem.

Why IntelliJ complains while VS Code doesn't: sglang is installed editable into .venv via
a dynamic editable-install hook — site-packages contains only:

  __editable__.sglang-0.5.18...pth → executes __editable___sglang_..._finder.py
  the actual sources live at sglang and are found at runtime by that import hook
  Pylance resolves it because it honors the editable hook (and your working set).
  IntelliJ's indexer only looks for a real sglang/ directory (or a plain path .pth)
  under site-packages — there isn't one, so it reports "package not found" at the from sglang.srt... imports.

Fix in IntelliJ:

- Mark the source root so IntelliJ sees the package (most reliable for editable installs):
  right-click python in the Project view → Mark Directory as → Sources Root.
- Go to File / Reload All from Disk

That makes IntelliJ resolve sglang from the checkout directly, matching the editable install. (Reinstalling non-editable is not recommended — you'd lose live edits.)

## Why a script and not just `pip install -e python`

Two things make a plain `pip install -e python` wrong on most machines:

1. **`python/pyproject.toml` is the CUDA/Linux build.** It pins
   `torch==2.13.0`, `flashinfer_python[cu13]`, `flash-attn-4`, `sgl-deep-ep`
   and friends — none of which resolve on macOS. SGLang keeps one pyproject per
   platform in `python/` (`pyproject_other.toml` for MPS/ROCm/HPU/MUSA,
   plus `pyproject_cpu.toml`, `pyproject_npu.toml`, `pyproject_xpu.toml`), and
   `setup.py` only ever reads `python/pyproject.toml`. The upstream docs tell
   you to `mv pyproject_other.toml pyproject.toml`, which permanently dirties
   the checkout. The script copies the right variant into place, installs, then
   restores the original file (also on Ctrl-C / failure), so `git status` stays
   clean.
2. **The interpreter version matters.** Homebrew's `python3` is currently
   3.14, and the platform pins (`torch==2.11.0` on macOS) have no 3.14 wheels.
   The script creates the venv on Python 3.12, matching what the Apple Silicon
   docs use.

## What it does

1. Detects a platform *variant* — `mps` on Apple Silicon, `cuda` if
   `nvidia-smi` exists, `hip` for ROCm, `npu`, else `cpu`.
2. Creates `<repo>/.venv` with `uv venv --python 3.12 --seed`, falling back to
   `python3.12 -m venv` when `uv` is not installed. An existing venv is reused
   unless you pass `--recreate`.
3. Temporarily swaps in the variant's pyproject, runs
   `uv pip install -e "python[<extras>]"`, then restores `pyproject.toml`.
4. Writes `.vscode/settings.json` (gitignored, skipped if it already exists)
   pointing `python.defaultInterpreterPath` at the venv.
5. Smoke-tests the result with `import sglang`.

Expect the first real run to take several minutes: it downloads a few GB of
wheels (torch, MLX, diffusers, …) and compiles the `sglang-mm` Rust crate.
`--dry-run` resolves the dependency set and prints the install plan without
downloading anything — worth doing first on a new machine or after a
dependency bump. On Apple Silicon it should land on Python 3.12 with torch
2.11 + MLX and leave `pyproject.toml` untouched.

### Variant → pyproject → default extras

| Variant | pyproject file         | Default extras | Notes                                           |
|---------|------------------------|----------------|-------------------------------------------------|
| `mps`   | `pyproject_other.toml` | `all_mps`      | Apple Silicon; torch 2.11 + MLX                 |
| `cuda`  | `pyproject.toml`       | `all`          | NVIDIA, Linux                                   |
| `hip`   | `pyproject_other.toml` | `all_hip`      | AMD ROCm                                        |
| `cpu`   | `pyproject_cpu.toml`   | `dev`          | Intel Xeon CPU serving                          |
| `npu`   | `pyproject_npu.toml`   | `all_npu`      | Ascend NPU                                      |
| `xpu`   | `pyproject_xpu.toml`   | `all`          | Intel XPU                                       |
| `empty` | `pyproject_other.toml` | `srt_empty`    | Pure-Python subset, no torch; installs anywhere |

Override any of it: `--variant`, `--extras`. For example `--extras dev_mps`
adds the test dependencies (pytest, accelerate, pandas, …) on macOS, and
`--extras srt_mps` skips the diffusion stack.

## Options

```
--variant <name>   auto (default), cuda, mps, cpu, hip, npu, xpu, empty
--python <ver>     Python version for the venv (default: 3.12)
--venv <path>      Venv location (default: <repo>/.venv)
--extras <list>    Comma-separated extras, overriding the variant default
--no-rust          Skip the Rust extension modules (SGLANG_BUILD_RUST_EXTS=none)
--no-vscode        Do not write .vscode/settings.json
--recreate         Delete and recreate an existing venv
--dry-run          Resolve dependencies without installing
```

### Rust extensions

`setup.py` discovers PyO3 extension crates in `../rust` through
`cargo metadata`, so the install shells out to `cargo` unless you opt out. On
non-CUDA platforms only `sglang-mm` (`sglang.srt.multimodal._core`) is built.
The extensions are optional at runtime — only the inkling multimodal processor
and the Rust server import them — so `--no-rust` is a fine escape hatch if you
have no toolchain or the build fails. The workspace pins Rust 1.92 in
`rust/rust-toolchain.toml`; `rustup` fetches it on demand.

## Using the environment

```bash
source .venv/bin/activate
python -c "import sglang; print(sglang.__version__)"
```

In VS Code: `Cmd+Shift+P` → **Python: Select Interpreter** →
`<repo>/.venv/bin/python`. Reload the window if an old interpreter is still
cached. The generated `.vscode/settings.json` also sets
`python.analysis.extraPaths` to `python/` so Pylance resolves `sglang.*`
through the source tree rather than the editable-install shim.

Because the install is editable, edits under `python/sglang/` take effect
immediately. Re-run the script after pulling changes that touch dependency
lists; changes to the Rust crates need a re-run too (or a manual
`uv pip install -e python --no-deps` with the right pyproject in place).

## Auto-activating the venv with direnv

Forgetting `source .venv/bin/activate` is how you end up running Homebrew's
`python3` against the source tree. [direnv](https://direnv.net/) activates the
venv when you `cd` into the repo and unwinds it when you leave. Setup on a new
machine:

```bash
# 1. Install and hook it into your shell (~/.zshrc), then restart the shell
brew install direnv
eval "$(direnv hook zsh)"     # oh-my-zsh: add it to plugins=(git direnv) instead
                              # bash: eval "$(direnv hook bash)" as the LAST line of ~/.bashrc

# 2. Write <repo>/.envrc
cat > .envrc <<'EOF'
if [ ! -x .venv/bin/python ]; then
  log_error "No .venv found — run ./studyIterations/setup_env.sh first"
else
  export VIRTUAL_ENV="$PWD/.venv"
  export VIRTUAL_ENV_PROMPT="(sglang)"
  PATH_add "$VIRTUAL_ENV/bin"
fi
EOF

# 3. Keep .envrc out of git, then approve it
echo '.envrc' >> .git/info/exclude
direnv allow
```

Verify with `cd .. && cd -`, then `which python` — it should print
`<repo>/.venv/bin/python`.

- **`PATH_add`, not `source .venv/bin/activate`.** direnv restores the
  environment it captured on entry, so `PATH_add` unwinds cleanly on `cd` out;
  sourcing `activate` also leaves a stale `deactivate` function behind.
- **`.envrc` is not checked in.** Upstream ships none and `.gitignore` only
  covers `.venv`, so committing it would dirty a fork. `.git/info/exclude` is
  the per-clone equivalent — repeat that step on every clone.
- **Re-run `direnv allow` after every edit to `.envrc`**, or it stays blocked.
  That refusal is the security model, not a bug.

## Apple Silicon specifics

Launching a server on macOS needs the MLX backend and no CUDA graphs:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path mlx-community/Qwen3-0.6B-4bit \
  --disable-cuda-graph
```

Building the optional native Metal kernels additionally requires the Xcode
Command Line Tools (`xcode-select --install`) and is a separate step:

```bash
python python/sglang/kernels/aot/setup_metal.py install
```

See `docs/docs/hardware-platforms/apple_metal.mdx` for the full platform guide.

## Troubleshooting

- **`No solution found` / resolution errors on macOS** — you are almost
  certainly resolving against the CUDA `pyproject.toml`. Make sure you ran the
  script (which swaps the file) rather than `pip install -e python` directly.
- **`cargo metadata failed` or a Rust compile error** — re-run with
  `--no-rust`.
- **Wrong interpreter in VS Code** — check the status bar; `.venv` at the
  workspace root is auto-discovered, but a stale
  `python.defaultInterpreterPath` in user settings can win. Selecting the
  interpreter explicitly fixes it.
- **`python3.12 not on PATH`** — install `uv`
  (<https://docs.astral.sh/uv/>), which downloads a matching interpreter
  itself, or pass `--python <version you have>` (must satisfy
  `requires-python = ">=3.10"`, and wheels must exist for the pinned torch).
