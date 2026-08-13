# VS Code / Pylance setup for this sglang checkout

Date: 2026-08-12

## The symptom

Pylance reported an error on [io_struct.py:79](../sglang/srt/managers/io_struct.py#L79):

```
No parameter named "tag"
```

```python
class BaseReq(msgspec.Struct, tag=True, kw_only=True, array_like=True):
```

Similar squiggles appeared on the `torch` / `zmq` / `pydantic` / `numpy` imports
at [io_struct.py:46-51](../sglang/srt/managers/io_struct.py#L46-L51).

## The cause

Not a code bug. `tag`, `kw_only`, and `array_like` are genuine class-level
keyword arguments of `msgspec.Struct` — msgspec declares them in the
`Struct.__init_subclass__` signature inside its bundled type stub
(`msgspec/__init__.pyi`, verified on 0.21.1):

```python
tag: Union[None, bool, str, int, Callable[[str], Union[str, int]]] = None,
```

The project simply had no Python environment:

- `python3` resolved to `/opt/homebrew/bin/python3`, where `import msgspec`
  raised `ModuleNotFoundError`
- no `.venv`, no conda env, no `.vscode/settings.json` selecting an interpreter

With msgspec unresolvable, Pylance can't load that stub and falls back to
`object.__init_subclass__`, which accepts no keyword arguments — hence
"No parameter named `tag`".

## The fix that was applied

Created a venv at the repo root and installed the deps needed for static
analysis of this file:

```bash
cd /Users/WChen/AI/sglangTry/sglang
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install msgspec pydantic numpy pyzmq torch
```

Resulting versions (Python 3.14.6, macOS arm64):

| package | version |
| --- | --- |
| msgspec | 0.21.1 |
| pydantic | 2.13.4 |
| numpy | 2.5.2 |
| pyzmq | 27.1.0 |
| torch | 2.13.0 (cp314 macosx_14_0_arm64 wheel) |

Then wrote `.vscode/settings.json` (the `.vscode/` directory is gitignored —
see [.gitignore:226](../../.gitignore#L226)):

```json
{
  "python.defaultInterpreterPath": "/Users/WChen/AI/sglangTry/sglang/.venv/bin/python",
  "python.analysis.extraPaths": ["/Users/WChen/AI/sglangTry/sglang/python"]
}
```

`extraPaths` matters because the `sglang` package itself is **not** installed —
it lets Pylance resolve the `from sglang.srt...` imports off the source tree.

If VS Code's workspace root is `sglang/python/` rather than `sglang/`, that
settings file won't be read; select the interpreter manually via
`Cmd+Shift+P` → *Python: Select Interpreter* → `.venv/bin/python`.

## Notes / caveats

- This venv is for **editor analysis only**, not for running sglang. A full
  `pip install -e python` pulls CUDA-only pieces (`sgl-kernel`) that don't build
  on macOS arm64; `python/pyproject_cpu.toml` is the variant to start from if a
  runnable local install is ever wanted.
- Reloading the VS Code window (`Cmd+Shift+P` → *Developer: Reload Window*)
  after switching interpreters is sometimes needed before Pylance re-analyzes.
