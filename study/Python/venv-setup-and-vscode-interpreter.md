# Python venvs in this sglang clone + pointing VS Code at one

**Repo:** `/Users/WChen/AI/sglangTry/sglang`
**Machine:** macOS arm64 (M3 Pro), no CUDA
**Current state as of 2026-08-17:** VS Code points at the **root `.venv` (3.14.6)**.

Companion notes: [pre-commit-setup.md](pre-commit-setup.md),
`python/study/vscode-pylance-env-setup.md`.

## Why this note exists

Switching the VS Code interpreter lit up ~20 unresolved-import errors in
`python/sglang/srt/entrypoints/http_server.py` (line 43 `import aiohttp` and
everything below it). Cause: the interpreter had been switched to a venv that
never had those packages. Nothing was wrong with the code.

## The two venvs in this clone

Both are gitignored (`.gitignore:124`), so a fresh clone has neither.

| venv | Python | created by | contents |
| --- | --- | --- | --- |
| `./.venv` (repo root) | 3.14.6 | uv | 91 packages — the working set, built up one import error at a time |
| `./python/.venv` | 3.11.14 | uv | 42 packages — the bare original set (torch, numpy, msgspec, pydantic, pyzmq, jinja2) |

`sglang` itself is **not** pip-installed in either.

## Why `pip install -e python` is not the answer here

`python/pyproject.toml` lists CUDA-only dependencies — `flash-attn-4`,
`flashinfer_python[cu13]`, `cuda-python>=13.0`, `nvidia-cutlass-dsl[cu13]`,
`humming-kernels[cu13]` — none of which build on macOS arm64. So deps get
installed by hand, one import error at a time.

On a **Linux + CUDA** box, ignore this whole note and just
`uv pip install -e python`.

## How VS Code resolves the interpreter

Three layers, in increasing precedence:

| Layer | Where | Notes |
| --- | --- | --- |
| 1. `python.defaultInterpreterPath` | `.vscode/settings.json` | Only a **fallback**. Ignored once layer 2 exists. |
| 2. Workspace state | VS Code internal storage | Set by **Python: Select Interpreter**. **Wins.** |
| 3. `python.analysis.extraPaths` | `.vscode/settings.json` | Pylance-only; unrelated to which interpreter runs |

This is the gotcha: editing `settings.json` alone often changes nothing, because
a previous pick from the interpreter picker is still stored in workspace state.
Always re-pick in the picker, then reload the window.

`.vscode/settings.json` is gitignored (`.gitignore:226`) — recreate it per machine.

## Full setup on a fresh clone (targeting `python/.venv`)

```bash
# 0. prereqs: uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. clone
git clone https://github.com/sgl-project/sglang.git ~/AI/sglangTry/sglang
cd ~/AI/sglangTry/sglang
export REPO="$PWD"

# 2. create the venv  (3.11 = best wheel coverage; pyproject requires >=3.10)
uv venv --python 3.11 "$REPO/python/.venv"

# 3. install deps  (verified 2026-08-17: resolves to 91 packages on 3.11.14,
#    same versions as the working root .venv)
VIRTUAL_ENV="$REPO/python/.venv" uv pip install \
  torch torchvision transformers==5.12.1 \
  aiohttp fastapi uvicorn uvloop starlette requests orjson pydantic msgspec numpy pyzmq \
  compressed-tensors einops gguf sentencepiece xgrammar partial_json_parser openai \
  dill ipython packaging pillow psutil pybase64 setproctitle tqdm

# 4. make `import sglang` work for the interpreter itself
"$REPO/python/.venv/bin/python" -c "
import sysconfig, pathlib, os
pathlib.Path(sysconfig.get_paths()['purelib'], 'sglang-src.pth').write_text(os.environ['REPO'] + '/python\n')
"

# 5. VS Code settings (gitignored — create by hand)
mkdir -p "$REPO/.vscode"
cat > "$REPO/.vscode/settings.json" <<EOF
{
  "python.defaultInterpreterPath": "$REPO/python/.venv/bin/python",
  "python.analysis.extraPaths": ["$REPO/python"]
}
EOF

# 6. verify
"$REPO/python/.venv/bin/python" -c "
import importlib.util as u
mods='sglang aiohttp fastapi uvicorn uvloop starlette orjson requests numpy torch transformers setproctitle zmq'.split()
missing=[m for m in mods if not u.find_spec(m)]
print('MISSING:', missing) if missing else print('all imports resolve')
"
```

Then in VS Code:

1. `Cmd+Shift+P` → **Python: Select Interpreter** → pick `./python/.venv/bin/python`
2. `Cmd+Shift+P` → **Developer: Reload Window**

Status bar should read `3.11.14 ('.venv': venv)`.

To target the **root `.venv`** instead, swap `$REPO/python/.venv` for `$REPO/.venv`
in steps 2/3/4/5 and pick that one in the picker.

## Two ways to make `sglang` importable

| Mechanism | Helps Pylance | Helps the interpreter at runtime |
| --- | --- | --- |
| `python.analysis.extraPaths` in `settings.json` | yes | **no** |
| `sglang-src.pth` in site-packages (step 4) | yes | yes |
| `export PYTHONPATH=$REPO/python` | no | yes, per shell only |

Use the `.pth` **and** `extraPaths`. The `.pth` is a one-line file containing an
absolute path; Python adds every path in a `.pth` to `sys.path` at startup.

## Gotchas

- **Pin `transformers==5.12.1`.** 5.15.0 breaks import — `python/sglang/srt/configs/qwen3_asr.py`
  re-registers a `qwen3_asr` config that newer transformers owns.
- `fastapi`, `uvicorn`, `uvloop`, `setproctitle` were missing from **both** venvs
  until 2026-08-17; they are direct imports of `http_server.py`. Installed into
  the root `.venv` on that date.
- This setup gets you a working **editor** plus CPU module imports, not a runnable
  server. Importing anything reaching `layers.moe` still dies inside
  `torch._inductor` because `python/sglang/_triton_stub.py` returns a mock
  *module* for `triton.compiler.CompiledKernel` where torch needs a class.
  `TORCH_COMPILE_DISABLE=1` does not help.
- The root `.venv` also carries an editable install of `rust/sglang-server`
  (`uv pip install -e rust/sglang-server`). Not needed for Pylance.
- `studyRun/venvs/mps-py312` (the MLX Metal serving env referenced in older notes)
  no longer exists on disk as of 2026-08-17.
