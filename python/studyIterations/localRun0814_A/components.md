# SGLang components, as they actually run on this MacBook

Phase 2 deliverable for [localRunPlan.md](localRunPlan.md).

Everything here is written against **this machine's running server** — Apple M3 Pro,
36 GB, `Qwen3-0.6B` bf16, MLX Metal backend, `tp=dp=pp=1`. Every claim is either a
`file:line` citation into the repo at `07821e9d56` or output captured from a live run.
Where the code says one thing and the runtime does another, the runtime wins and the
discrepancy is called out.

---

## 0. The short answer

Four OS processes, of which **three are SGLang components**:

```
pid 77398  ppid 1       139 MB   23 threads   python -m sglang.launch_server
   │                                          └─ uvicorn/FastAPI + TokenizerManager
   ├── pid 77530          9 MB    1 thread    multiprocessing.resource_tracker
   │                                          └─ stdlib bookkeeping, NOT an sglang component
   ├── pid 77531        172 MB   56 threads   sglang::scheduler
   │                                          └─ Scheduler + MlxTpModelWorker
   │                                             + MlxModelRunner (+ MlxModelRunnerStub)
   └── pid 77532        103 MB    7 threads   sglang::detokenizer
                                              └─ DetokenizerManager
```

The child names come from `setproctitle`, set at
[scheduler.py:4965](../../python/sglang/srt/managers/scheduler.py#L4965) and
[detokenizer_manager.py:521](../../python/sglang/srt/managers/detokenizer_manager.py#L521).

RSS is measured at idle. Under load the scheduler climbs to ~2.7 GB — it holds the model
and the 3.5 GB MLX KV pool; MLX releases cache between bursts, which is why idle RSS looks
modest.

**ModelRunner count: exactly one real one.** See §6 — the answer has a wrinkle.

---

## 1. How the processes come to exist

`python -m sglang.launch_server` → `Engine._launch_subprocesses`
([engine.py:1052](../../python/sglang/srt/entrypoints/engine.py#L1052)), whose docstring
states the split directly:

> "Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the
> DetokenizerManager in another subprocess."

The two children are ordinary `mp.Process` forks —
[engine.py:898](../../python/sglang/srt/entrypoints/engine.py#L898) (`run_scheduler_process`)
and [engine.py:987](../../python/sglang/srt/entrypoints/engine.py#L987)
(`run_detokenizer_process`). `resource_tracker` is spawned by the stdlib the first time
`multiprocessing` is used; nothing in SGLang asks for it.

**Why this matters:** the scheduler is a *separate process*, so a `kill` on the parent pid
orphans it holding the model, the KV pool and the ZMQ sockets. That is the trap
`bin/stop_sglang.sh` exists to avoid (Phase 7).

---

## 2. API server

### 2.1 The Python path (default)

FastAPI + uvicorn, **in the main process**, in-process with the TokenizerManager, at
[entrypoints/http_server.py](../../python/sglang/srt/entrypoints/http_server.py). Roughly
40 routes; the ones this study uses:

| Route | Line |
|---|---|
| `POST /generate` | [http_server.py:874](../../python/sglang/srt/entrypoints/http_server.py#L874) |
| `POST /v1/chat/completions` | [http_server.py:1702](../../python/sglang/srt/entrypoints/http_server.py#L1702) |
| `GET /health` | [http_server.py:646](../../python/sglang/srt/entrypoints/http_server.py#L646) |
| `GET /get_server_info` | [http_server.py:771](../../python/sglang/srt/entrypoints/http_server.py#L771) |
| `GET /metrics` | mounted via [utils/common.py:2398](../../python/sglang/srt/utils/common.py#L2398) |

Every generate-shaped route funnels into the same coroutine —
`_global_state.tokenizer_manager.generate_request(...)`. The OpenAI layer is a translation
shim, not a second pipeline.

### 2.2 The Rust path — "how does the Rust server run?"

**It is not a separate server you start. It is a Rust extension module that the Scheduler
starts, from inside the scheduler process.** That inversion is the interesting part.

- The crate is [rust/sglang-server](../../rust/sglang-server), built by **maturin/pyo3**
  into a Python extension module named **`_core`**
  ([pyproject.toml](../../rust/sglang-server/pyproject.toml), `module-name = "_core"`).
- Its `src/` contains `api_server.rs`, `tokenizer_manager.rs`, `detokenizer.rs`, `ring.rs`,
  `fsm.rs` — so it replaces the HTTP server **and** the tokenizer/detokenizer layer.
- It is gated by the env var `SGLANG_RUST_SERVER`
  ([environ.py:1432](../../python/sglang/srt/environ.py#L1432)).
- It is launched from
  [scheduler.py:1983](../../python/sglang/srt/managers/scheduler.py#L1983)
  (`maybe_init_rust_server`), gated on rank 0 by `_hosts_rust_server()`
  ([scheduler.py:1977](../../python/sglang/srt/managers/scheduler.py#L1977)); the plumbing
  lives in [managers/rust_server.py](../../python/sglang/srt/managers/rust_server.py).

#### I built it and ran it

It was **not** built in this checkout, and the plan rated it medium-high risk. It built
cleanly:

```
$ VIRTUAL_ENV=studyIterations/venvs/mps-py312 maturin develop --release
🐍 Found CPython 3.12 at .../mps-py312/bin/python
   Compiling sglang-server v0.1.0
    Finished `release` profile [optimized] target(s) in 1m 28s
🛠 Installed sglang-server-0.1.0
```

Rust 1.92 was already installed (the crate pins it; `cargo` on PATH was 1.90 — `rustup`
resolves the pin automatically, so no manual toolchain work was needed).

> **Gotcha worth recording:** the first build produced a **CPython 3.14** wheel and
> installed `_core` into the repo's unrelated `.venv`, because `VIRTUAL_ENV` was inherited
> from the shell. `maturin` trusts `VIRTUAL_ENV` over cwd. Set it explicitly.

Running with `SGLANG_RUST_SERVER=1`, it serves — and the Rust tracing logs interleave with
the Python ones:

```
INFO sglang_server::tokenizer: loaded tokenizer path=studyIterations/models/Qwen3-0.6B
INFO sglang_server::api_server::openai: loaded OpenAI chat template
[2026-08-14 18:00:09] SGLANG_RUST_SERVER enabled, Rust server listen on 127.0.0.1:30000
INFO sglang_server::api_server::log: 127.0.0.1 - "GET /health HTTP/1.1" 200 OK
```

`POST /generate` and `POST /v1/chat/completions` both return correct output.

#### What changes, measured

**The DetokenizerManager process disappears.**

| | Python path | Rust path |
|---|---|---|
| processes | 4 | **3** |
| `sglang::detokenizer` | present (7 threads) | **absent** |
| `sglang::scheduler` threads | 56 | **66** |
| `GET /health` | 200 | 200 |
| `GET /v1/models` | 200 | 200 |
| `POST /generate` | 200 | 200 |
| `POST /v1/chat/completions` | 200 | 200 |
| **`GET /metrics`** | **200** | **404** |
| **`GET /get_server_info`** | **200** | **404** |

Detokenization moves from a Python *process* into Rust *threads* inside the scheduler
process (`detokenizer.rs`), which is why the thread count rises by ~10 as one process
vanishes.

**The trade-off that matters for this study: the Rust server has no `/metrics`.** Enabling
it silently breaks Prometheus, Grafana and the TUI — all three depend on that endpoint.
`/get_server_info` is also absent, so `bin/status.sh` loses its server-info section. This
is why the study run stays on the Python path by default; the Rust path is a thing to
switch to deliberately, not a free speedup.

---

## 3. Tokenizer & TokenizerManager

Lives **in the main process**, in-process with FastAPI — no IPC hop between the HTTP
handler and the tokenizer.

Socket setup, [tokenizer_manager.py:537-549](../../python/sglang/srt/managers/tokenizer_manager.py#L537-L549):

```python
context, zmq.PULL, port_args.tokenizer_ipc_name, True        # bind   <- results in
context, zmq.PUSH, port_args.scheduler_input_ipc_name, True  # bind   -> requests out
```

The trailing `True` means **bind**. Requests leave via `_send_one_request`
([tokenizer_manager.py:1586](../../python/sglang/srt/managers/tokenizer_manager.py#L1586))
→ `sock_send(self.send_to_scheduler, obj)`
([tokenizer_manager.py:560](../../python/sglang/srt/managers/tokenizer_manager.py#L560)).

Responsibilities: assign the `rid`, tokenize, hold the per-request future, and reassemble
streamed output arriving back on `tokenizer_ipc_name`.

---

## 4. Scheduler

Own process. Entry point `run_scheduler_process`
([scheduler.py:4985](../../python/sglang/srt/managers/scheduler.py#L4985)); event loops at
[scheduler.py:1709](../../python/sglang/srt/managers/scheduler.py#L1709)
(`event_loop_normal`) and
[scheduler.py:1744](../../python/sglang/srt/managers/scheduler.py#L1744)
(`event_loop_overlap`).

**On this machine `event_loop_overlap` is the live one.** `/get_server_info` reports
`disable_overlap_schedule=False`, because the forced-disable at
[server_args.py:4307-4310](../../python/sglang/srt/server_args.py#L4307-L4310) only fires
when `use_mlx()` is *false*; with `SGLANG_USE_MLX=1` overlap stays on via MLX `async_eval()`.

The per-iteration lines in the log are the scheduler narrating itself:

```
Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 0, token usage: 0.00,
               #running-req: 0, #queue-req: 0, cuda graph: False
Decode batch,  #running-req: 1, #token: 4, token usage: 0.00, gen throughput (token/s): 0.01
```

Cache: `Init Unified Radix Cache … Tree Core: UnifiedTreeCore`.

**Admission control:** `max_running_requests` defaults to 4096 here, so nothing ever
queues — `num_queue_reqs` is permanently 0 in normal use. Launching with
`--max-running-requests 2` under 10-way load makes the mechanism visible:

```
running=2  queued=8 → 6 → 4 → 2
```

---

## 5. TpModelWorker

Owned by the Scheduler, **inside the scheduler process** — not a separate process.
Selected at [scheduler.py:910-917](../../python/sglang/srt/managers/scheduler.py#L910-L917):

```python
if use_mlx():
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    self.tp_worker = MlxTpModelWorker(**worker_kwargs)
else:
    from sglang.srt.managers.tp_worker import TpModelWorker
    self.tp_worker = TpModelWorker(**worker_kwargs)
```

So on this box it is **`MlxTpModelWorker`**
([mlx/tp_worker.py:70](../../python/sglang/srt/hardware_backend/mlx/tp_worker.py#L70)),
which subclasses the generic `TpModelWorker` purely to keep the scheduler's integration
surface unchanged.

---

## 6. ModelRunner — "how many?"

**One real ModelRunner, plus one stub — and the stub explains three confusing metrics.**

The count rule: **one ModelRunner per (TP × PP × DP) rank**, each owned by one
TpModelWorker inside one scheduler process. Here `tp=dp=pp=1`, one `sglang::scheduler`
process → **one**.

But `MlxTpModelWorker._init_model_runner`
([mlx/tp_worker.py:79-116](../../python/sglang/srt/hardware_backend/mlx/tp_worker.py#L79-L116))
builds **two** objects:

```python
self._mlx_runner   = MlxModelRunner(**init_kwargs)      # the real inference engine
self._model_runner = MlxModelRunnerStub(...)            # torch-side placeholder
```

Its own docstring is explicit: it "replaces the standard ModelRunner with
MlxModelRunnerStub (no PyTorch weights, zero-memory KV cache) and delegates all forward
passes to a native MlxModelRunner."

Confirmed in the startup log:

```
Initializing MlxModelRunner for end-to-end MLX inference
MLX model loaded in 0.74s
MLX stub: skipping PyTorch model weight loading (inference runs through MLX)
MLX stub: initialized minimal pools (max_total_num_tokens=32768, zero GPU KV cache allocation)
Engine startup timings (s): load_weight=0.00, ... scheduler_e2e=2.88, tokenizer_e2e=8.80
```

`load_weight=0.00` on the torch side while MLX loads in 0.74 s — the PyTorch loader is
genuinely bypassed, not merely fast.

### The stub is the root cause of three misleading readings

Independently noticed across Phases 1 and 3, all one bug class — **they read the stub, not
the real runner**:

| Reading | Reports | Reality |
|---|---|---|
| `/get_server_info` `attention_backend` | `torch_native` | MLX serves attention via `MlxAttentionKVPool` |
| `sglang:kv_cache_memory_usage_gb` | `0.0` | pool is **3,584 MB** |
| `sglang:weight_memory_usage_gb` | `0.0` | weights loaded, 0.74 s via MLX |

Anything asking torch "how much did you allocate?" correctly answers zero. The Grafana
dashboard therefore counts KV **tokens** (`kv_used_tokens` / `kv_evictable_tokens` /
`kv_available_tokens`), which the scheduler computes directly and which are correct.

**How the count would change:** `--tp-size 2` → 2 scheduler processes, 2 workers, 2
ModelRunners. `--dp-size 2` → 2 replicas plus a `sglang::data_parallel_controller`
process. Neither is useful on a single Metal GPU, so both stay at 1 here.

---

## 7. IPC topology

Single-node SGLang wires its processes with **ZMQ over `ipc://` unix domain sockets**, not
TCP — the paths are `tempfile` names created at
[server_args.py:9747-9758](../../python/sglang/srt/server_args.py#L9747-L9758).

```
                    HTTP :30000  (the only TCP socket)
                         │
        ┌────────────────▼─────────────────┐
        │  main process (pid 77398)        │
        │  uvicorn/FastAPI                 │
        │  TokenizerManager                │
        └──┬────────────────────────▲──────┘
           │ PUSH                   │ PULL  (binds tokenizer_ipc_name)
           │ scheduler_input_ipc    │
           ▼ (binds)                │
        ┌──────────────────┐        │
        │ sglang::scheduler│        │
        │  (pid 77531)     │        │
        │  Scheduler       │        │
        │  MlxTpModelWorker│        │
        │  MlxModelRunner  │        │
        └──┬───────────────┘        │
           │ PUSH detokenizer_ipc   │
           ▼                        │
        ┌──────────────────┐        │
        │sglang::detokenizer│───────┘  PUSH (connects to tokenizer_ipc_name)
        │  (pid 77532)      │
        └───────────────────┘
```

Live `lsof` shows exactly **three** bound unix sockets:

```
pid 77398:  /var/folders/.../T/tmp1o4u1nts
            /var/folders/.../T/tmp5_srftfp
pid 77532:  /var/folders/.../T/tmpoxrn0n8w
```

**The scheduler shows zero — and that is correct, not a measurement failure.** `lsof -U`
lists sockets a process *binds*. The main process binds two (`tokenizer_ipc_name`,
`scheduler_input_ipc_name`, both `True` at
[tokenizer_manager.py:537-541](../../python/sglang/srt/managers/tokenizer_manager.py#L537-L541));
the detokenizer binds one (`detokenizer_ipc_name`, `True` at
[detokenizer_manager.py:114](../../python/sglang/srt/managers/detokenizer_manager.py#L114))
and *connects* to `tokenizer_ipc_name` (`False` at
[detokenizer_manager.py:120-121](../../python/sglang/srt/managers/detokenizer_manager.py#L120-L121)).
The scheduler only ever connects. 2 + 1 + 0 = 3. ✅

**Consequence for Phase 5:** unix domain sockets carry no packets, so `tcpdump` can see the
HTTP on `:30000` and nothing else.

---

## 8. One request, end to end

Real trace, `rid = 2a33f8cb006f43d18106420a410b6414`, `POST /generate`, 12 output tokens.

| # | Where | Evidence |
|---|---|---|
| 1 | `POST /generate` → FastAPI | [http_server.py:874](../../python/sglang/srt/entrypoints/http_server.py#L874) |
| 2 | TokenizerManager assigns rid, tokenizes | `{"event": "request.received", "rid": "2a33f8cb…"}` |
| 3 | ZMQ PUSH → scheduler | [tokenizer_manager.py:1586](../../python/sglang/srt/managers/tokenizer_manager.py#L1586) |
| 4 | Scheduler admits, prefills | `Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 0` |
| 5 | MlxTpModelWorker → MlxModelRunner | forward passes on Metal |
| 6 | Scheduler decodes | `Decode batch, #running-req: 1, #token: 4` |
| 7 | ZMQ PUSH → detokenizer | [scheduler.py:4847](../../python/sglang/srt/managers/scheduler.py#L4847) |
| 8 | Detokenizer → TokenizerManager | [detokenizer_manager.py:173](../../python/sglang/srt/managers/detokenizer_manager.py#L173) |
| 9 | HTTP 200 | `POST /generate HTTP/1.1" 200 OK` |
| 10 | Timing summary | see below |

```
ReqTimeStats(rid=2a33f8cb006f43d18106420a410b6414, input_len=3, cached_input_len=0,
             output_len=12, attempts=0, type=unified):
             queue_duration=0.14ms, forward_duration=1670.96ms
```

`queue_duration=0.14 ms` vs `forward_duration=1671 ms` — with no contention the scheduler
is ~0.008 % of the request. All the time is Metal compute. That ratio is what
`--max-running-requests 2` inverts.

---

## 9. Things that surprised me

1. **The Rust server is started by the Scheduler, not by the entrypoint.** Every mental
   model of "the web server launches the workers" is backwards here.
2. **Enabling the Rust server deletes a process and kills `/metrics`.** A faster front end
   costs you the entire observability stack.
3. **The MLX backend keeps a torch stub ModelRunner alive purely for the scheduler's
   benefit**, and that stub is what three "0 GB / torch_native" readings are reporting.
4. **`lsof` showing no sockets for the scheduler is the correct answer**, because it only
   ever connects. Easy to misread as a broken measurement.
5. **`load_weight=0.00` next to a working model** is not a bug — startup is dominated by
   `tokenizer_e2e=8.80 s` (importing transformers and the model registry), not by weights.
