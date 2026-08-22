# Why Black didn't reformat my joined line on commit

**Date:** 2026-08-16
**Repo:** `/Users/WChen/AI/sglangTry/sglang`
**Branch:** `localRun260814_A`
**Commit in question:** `194c5c7064` — "Trigger the formatter Black pre-commit hook"

> **Status update (2026-08-17):** the Fix section below has since been carried
> out — `pre-commit` was installed via `uv tool` and `pre-commit install` was run
> — and then the git hook was deliberately removed again with
> `pre-commit uninstall`. Commits in this clone currently run no hooks. For the
> current state and the full enable/disable/run reference, see
> [pre-commit-setup.md](pre-commit-setup.md). This document remains accurate as
> the record of the original diagnosis.

## The situation

In IntelliJ I intentionally joined this 3-line statement in
`python/sglang/srt/entrypoints/engine.py` into a single line and committed it,
expecting the Black pre-commit hook to split it back:

```python
# before (Black's formatting)
scheduler_init_result.engine_info_bootstrap_server = (
    engine_info_bootstrap_server
)

# after my edit — line 1128, 89 characters
scheduler_init_result.engine_info_bootstrap_server = engine_info_bootstrap_server
```

Black did not reformat it.

## The answer

The editor is irrelevant — IntelliJ vs VS Code makes no difference.
**The pre-commit hooks were never installed in this clone**, so nothing ran at
commit time.

### Evidence

1. `.git/hooks/` contains only the 14 stock `*.sample` files. There is no
   executable `pre-commit` hook, and `core.hooksPath` is unset. Git had nothing
   to invoke.

   ```bash
   $ ls .git/hooks/ | grep -v '\.sample$'
   NO active hooks
   $ git config --get core.hooksPath
   (empty)
   ```

2. `pre-commit` isn't on PATH at all, and neither is `black` — not globally, not
   in `python/studyIterations/venvs/mps-py312/bin/`.

   ```bash
   $ which pre-commit
   pre-commit not found
   $ which black
   black not found
   ```

3. The commit went through untouched. `git show 194c5c7064` records exactly the
   3-lines→1-line change with no follow-up reformat:

   ```diff
   -        scheduler_init_result.engine_info_bootstrap_server = (
   -            engine_info_bootstrap_server
   -        )
   +        scheduler_init_result.engine_info_bootstrap_server = engine_info_bootstrap_server
   ```

## The key misconception

`.pre-commit-config.yaml` is **just a config file**. Checking it into the repo
does nothing on its own. It only takes effect after you run `pre-commit install`,
which writes a `.git/hooks/pre-commit` shim that shells out to the `pre-commit`
tool. Until then, committing is a plain git commit — which is what happened.

## Second gotcha: what a working hook would actually have done

Even a working hook wouldn't have produced the commit I expected. `pre-commit`
hooks that *modify* files **fail** the commit and leave the reformatted file
unstaged. I would have seen:

```
black....................................................................Failed
- files were modified by this hook
```

...no commit created, and I'd have to `git add` + re-commit. **Black never
silently amends a commit.**

## Fix

```bash
uv tool install pre-commit          # -> ~/.local/bin/pre-commit
cd /Users/WChen/AI/sglangTry/sglang
pre-commit install                  # writes .git/hooks/pre-commit
```

> **Why not `pip install` into the venv?** `python/studyIterations/venvs/mps-py312/`
> was created by `uv venv`, and **uv does not install pip into the venvs it
> creates** — there is no `bin/pip`. You install into such a venv with the
> external `uv` binary (`/Users/WChen/.local/bin/uv`, version 0.9.26, matching
> the `uv = 0.9.26` line in `pyvenv.cfg`).
>
> Beyond that, `pre-commit` is a *developer tool*, not a runtime dependency of
> sglang, so it doesn't belong in the serving venv at all. `uv tool install`
> gives it its own isolated environment and puts the executable on PATH.
>
> PATH placement matters here: `pre-commit install` writes a
> `.git/hooks/pre-commit` shim that later shells out to `pre-commit`, so the
> command must be durably resolvable. An ephemeral `uvx pre-commit` works for a
> one-off `run`, but leaves the installed hook pointing at a cache entry that
> can be garbage-collected.

Then to reformat the line already committed:

```bash
pre-commit run --files python/sglang/srt/entrypoints/engine.py
```

### Note: Black is deliberately NOT pip-installed

Only `pre-commit` gets installed above. **You never `pip install black` for this
workflow.** `pre-commit` is a hook *manager*: on first run it reads each
`repo:` / `rev:` pair in `.pre-commit-config.yaml`, clones it, and builds an
**isolated virtualenv per hook** under `~/.cache/pre-commit/`. So `psf/black` at
`rev: 26.1.0` gets its own environment there, pinned to that exact version and
entirely separate from the `mps-py312` venv.

That isolation is the whole point of the pinning: every contributor gets
byte-identical formatting regardless of what happens to be installed in their
project venv. It also means `which black` will *still* report "not found" after
the fix — Black lives in the cache, not on PATH. That is expected, not a
symptom of a broken install.

Consequently the first `pre-commit run` is slow: it builds environments for
black 26.1.0, isort 7.0.0, ruff, clang-format, etc. (a minute or two). Later
runs reuse the cache and are fast. On this machine `~/.cache/pre-commit/` exists
(from Dec 2025) but holds only `db.db`, a `README`, and stale `patch*` files —
no hook environments — so the first run will do the full build.

Once it runs, it rewrites line 1128 back to the parenthesized 3-line form, since
89 > Black's default line length of 88.

### Alternative: run Black standalone

If you'd rather invoke Black directly (e.g. to wire it into IntelliJ's
*Settings → Tools → Black*), install it explicitly and match the pinned version
so you don't fight the hook:

```bash
uv tool install 'black==26.1.0'     # -> ~/.local/bin/black
black python/sglang/srt/entrypoints/engine.py
```

(Again: not `pip install` — the venv has no pip. If you specifically want Black
*inside* the serving venv rather than as a standalone tool, use
`uv pip install --python python/studyIterations/venvs/mps-py312 'black==26.1.0'`.)

This is independent of the pre-commit cache — the two can coexist, but they only
agree if the versions match.

## Background: why the parentheses exist at all

They are not semantically required — `(x)` is grouping, not a tuple (a 1-tuple
needs `(x,)`). They are a **line-continuation device**:

- Python only allows a statement to span multiple physical lines via implicit
  line joining inside `()` / `[]` / `{}`, or via an explicit trailing backslash.
- A plain assignment has no natural bracket, so Black wraps the right-hand side
  in "invisible parens". Black never emits `\` (backslashes are
  whitespace-fragile — a single trailing space after one is a syntax error).

The arithmetic that forces the wrap: 8-space indent + `scheduler_init_result.engine_info_bootstrap_server`
(50 chars) + ` = ` (3) + `engine_info_bootstrap_server` (28) = **89 characters**,
one over Black's default limit of 88.

The repo runs `psf/black` 26.1.0 via `.pre-commit-config.yaml` lines 51–55 with no
`line-length` override anywhere in `pyproject.toml`, so the default 88 is in force.
