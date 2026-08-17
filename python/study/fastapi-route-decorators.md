# How `@app.*` decorators create SGLang's HTTP endpoints, and when they run

**File:** `python/sglang/srt/entrypoints/http_server.py`
**Question answered:** when are the decorations that add HTTP endpoints processed?

## Short answer

**At module import time** — long before `launch_server()` (line 2718) is ever
called. By the time `launch_server` runs, the routing table is already complete
and frozen; that function wires up the *engine*, not the *routes*.

## Why: a decorator is not deferred machinery

```python
@app.get("/health")
def health(...): ...
```

is pure syntax sugar for:

```python
def health(...): ...
health = app.get("/health")(health)
```

Both lines execute the moment the interpreter reaches that `def` statement, which
happens while `sglang.srt.entrypoints.http_server` is being imported.
`app.get("/health")` returns a registrar closure; calling it appends an
`APIRoute` to `app.router.routes` and returns the original function unchanged.

Nothing about this is lazy. There is no "route registration phase" at startup.

## The import-time sequence in this module

Everything at module level runs top to bottom, exactly once per process:

| Lines | What happens |
| --- | --- |
| 455-458 | `app = FastAPI(lifespan=lifespan, openapi_url=...)` — must exist before any decorator can name it |
| 459 | `app.router.route_class = ORJSONRoute` |
| 460-473 | `app.add_middleware(CORSMiddleware)`, plus conditional `RequestDecompressionMiddleware` |
| 480, 485 | `app.include_router(v1_loads_router)` / `include_router(elastic_ep_router)` — splices in routes defined in other modules |
| 524-2020 | the **84** `@app.*` decorators: 2 `@app.exception_handler`, then every route from `/health` to `/invocations` and the Vertex route |
| 2718 | `def launch_server(...)` — merely *binds a name*; the body has not run |

Ordering inside this list is load-bearing. `route_class = ORJSONRoute` is set on
line 459, **before** any decorator executes, so every route built afterwards picks
up the orjson response class. Move it below the decorators and it would silently
apply to nothing.

## Consequence 1: decorator *arguments* are evaluated at import too

The sharpest example is line 2020:

```python
@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate"))
```

That `os.environ.get(...)` call runs during import. Setting `AIP_PREDICT_ROUTE`
*after* the module has been imported has no effect whatsoever — the path string
is baked into the `APIRoute` object. The env var must be set before the first
import of `http_server` (i.e. in the process environment at launch), which is
exactly how Vertex AI supplies it.

This generalizes: any expression inside a decorator's parentheses — env lookups,
config reads, feature flags — is frozen at import.

## Consequence 2: `launch_server` supplies state, not routes

`launch_server` can call `Engine._launch_subprocesses(...)` (line 2749) and then
start serving because the HTTP *surface* was defined at import, while the *state*
the handlers need is injected later:

```python
# _launch_server_impl, line 2479 — "Called by launch_server after subprocesses have been launched."
set_global_state(
    _GlobalState(
        tokenizer_manager=tokenizer_manager,
        template_manager=template_manager,
        scheduler_info=scheduler_infos[0],
    )
)
```

Handlers read that module global (`_global_state`, line 204) at request time. So
the lifecycle is:

```
import  ->  routes registered on `app`, handlers reference _global_state (still None)
launch  ->  subprocesses spawned, set_global_state(...) fills it in
serve   ->  requests arrive, handlers dereference _global_state
```

## Consequence 3: this split is what makes multi-worker mode work

In multi-tokenizer mode Granian is handed an **import string** rather than the
object (line 2422):

```python
app if tokenizer_worker_num == 1 else "sglang.srt.entrypoints.http_server:app"
```

Each worker process re-imports the module, re-runs all 84 decorators, and builds
its own fully-populated `app`. Per-worker state then arrives through the
`lifespan` handler (line 270), which calls `init_multi_tokenizer()` -> its own
`set_global_state(...)` at line 258. Routes are cheap to rebuild because they are
just import-time side effects; state is what must be per-process.

## The one genuine runtime addition

Routes are import-time only, but **middleware is not strictly so**. Inside
`_launch_server_impl` (line 2492):

```python
if server_args.enable_metrics:
    add_prometheus_track_response_middleware(app)
```

This runs at launch, after import, because it depends on `server_args`. It works
because Starlette builds its middleware stack lazily on first request, not at
`FastAPI()` construction. Adding a *route* this late would also technically work,
but nothing in this module does it — the pattern here is
"routes at import, middleware may be conditional at launch."

## Quick way to see the table yourself

```python
from sglang.srt.entrypoints.http_server import app
for r in app.routes:
    print(getattr(r, "methods", None), r.path)
```

Note that merely importing the module is enough to populate this — no server, no
engine, no `launch_server` call required. That is the whole point.
