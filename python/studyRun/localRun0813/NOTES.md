# SGLang on a MacBook Pro M3 — what runs, and where

Everything below was observed on this machine, serving `Qwen/Qwen3-0.6B`
(bf16) through the MLX/Metal backend from this checkout at `07821e9d56`.

- [Quick start](#quick-start)
- [What the setup looks like](#what-the-setup-looks-like)
- [Process topology](#process-topology)
- [The components](#the-components)
- [A request, end to end](#a-request-end-to-end)
- [The Rust api-server](#the-rust-api-server)
- [What Apple Silicon changes](#what-apple-silicon-changes)
- [Numbers from this box](#numbers-from-this-box)
- [Local gotchas](#local-gotchas)

## Quick start

```bash
cd studyRun/localRun0813
./start.sh              # boot the server, wait for /health, print the process tree
./test.sh               # exercise /generate, /v1/chat/completions, SSE, 4-way concurrency
./.venv/bin/python monitor.py   # live process + metrics view (Ctrl-C to quit)
./status.sh             # one-shot snapshot
./stop.sh               # SIGTERM the whole tree, escalate to SIGKILL
```

Other entry points:

| command | what it does |
|---|---|
| `./start.sh --fg` | run in the foreground so the log goes to your terminal |
| `./start.sh --rust` | swap the Python api-server for the embedded Rust one |
| `STUDY_QUANT=mlx_q4 ./start.sh` | quantize the weights to 4-bit at load time |
| `STUDY_LOG_REQUESTS=1 ./start.sh` | add per-request `Receive:` / `Finish:` lines |
| `./stop.sh --force` | skip SIGTERM, go straight to SIGKILL |

All knobs live in [config.sh](config.sh). Logs land in `logs/server.log`, the
pid in `run/server.pid`.

## What the setup looks like

**Interpreter.** `studyRun/localRun0813/.venv` (CPython 3.12.8). The repo's own `.venv` is
Python 3.14 built against `python/pyproject.toml`, which pins CUDA-only wheels
(`flashinfer`, `cuda-python`, `sgl-deep-gemm`) and cannot be installed on
macOS. The Apple Silicon path uses a different pyproject:

```bash
uv venv -p 3.12 studyRun/localRun0813/.venv
cp python/pyproject_other.toml python/pyproject.toml     # temporarily
SGLANG_BUILD_RUST_EXTS=none uv pip install --python studyRun/localRun0813/.venv/bin/python -e "python[srt_mps]"
git checkout python/pyproject.toml                        # put it back
```

`[srt_mps]` is `mlx` + `mlx-lm` + `torch==2.11.0` + the torch-free
`runtime_common` set. Python 3.12 rather than 3.14 because `outlines-core
0.1.26` ships no cp314 wheel and its PyO3 0.22 build rejects 3.14.

**Model.** `studyRun/models/Qwen3-0.6B`, 1.5 GB of bf16 safetensors. Fetched
from ModelScope, not Hugging Face — see [Local gotchas](#local-gotchas).

**TLS.** `studyRun/localRun0813/.venv/.../sitecustomize.py` calls `truststore.inject_into_ssl()`
so Python verifies against the macOS keychain. Without it every HTTPS call
fails with `self-signed certificate in certificate chain`, because the
corporate proxy re-signs traffic with a root CA that `certifi` does not carry.

## Process topology

Three OS processes, observed via `./status.sh`:

```
  15143   ppid=1        api-server + TokenizerManager     <- python -m sglang.launch_server
  15354   ppid=15143    scheduler                         <- setproctitle "sglang::scheduler"
  15355   ppid=15143    detokenizer                       <- setproctitle "sglang::detokenizer"
```

That mapping comes straight from `launch_server`'s docstring in
[http_server.py:2718](../../python/sglang/srt/entrypoints/http_server.py#L2718),
and the fan-out happens in `Engine._launch_subprocesses`
([engine.py:1125](../../python/sglang/srt/entrypoints/engine.py#L1125)):
scheduler processes first, then detokenizer(s), then the TokenizerManager in
the main process.

The children rename themselves with `setproctitle`
([scheduler.py:4965](../../python/sglang/srt/managers/scheduler.py#L4965),
[detokenizer_manager.py:521](../../python/sglang/srt/managers/detokenizer_manager.py#L521)),
which is exactly what `status.sh` and `monitor.py` key off. They also install
`kill_itself_when_parent_died()`, so they normally die with the parent —
`stop.sh` still sweeps for orphans, because a SIGKILL'd parent can leave them.

**Counts, in general.** With `--tp-size 1 --pp-size 1 --dp-size 1` (this run)
you get one of everything. The scaling rules:

| component | how many |
|---|---|
| api-server + TokenizerManager | 1 per node, in the main process (`--tokenizer-worker-num N` puts a `MultiTokenizerRouter` in front of N of them) |
| Scheduler | `tp_size × pp_size` processes per node, one `mp.Process` each; `+1` DP controller when `dp_size > 1` |
| DetokenizerManager | 1 process (`--detokenizer-worker-num N` for more, fronted by a router) |
| TpModelWorker | exactly 1 per scheduler, plus 1 draft worker when speculative decoding is on |
| ModelRunner | exactly 1 per TpModelWorker ([tp_worker.py:454](../../python/sglang/srt/managers/tp_worker.py#L454)) — so `tp_size × pp_size` in total, `×2` with a draft model, and `speculative_num_steps` of them for multi-layer EAGLE |

So on this box: **1 TpModelWorker, 1 ModelRunner** — well, 1½; see
[the MLX section](#what-apple-silicon-changes).

**IPC.** ZMQ over `ipc://` unix sockets, one per hop, named after
`NamedTemporaryFile`s under `$TMPDIR`
([server_args.py:9749](../../python/sglang/srt/server_args.py#L9749)):
`tokenizer_ipc_name`, `scheduler_input_ipc_name`, `detokenizer_ipc_name`,
`rpc_ipc_name`, `metrics_ipc_name`. `--enable-dp-attention` switches these to
`tcp://` so the same code works across nodes.

## The components

### api-server (Python, default)

`python -m sglang.launch_server` → `launch_server()` → after the subprocesses
are up, `_setup_and_run_http_server()` runs a FastAPI app under uvicorn in the
main process. It owns the two request dialects:

- **native**: `/generate`, `/health`, `/get_model_info`, `/flush_cache`,
  `/update_weights_from_disk`, `/metrics`, …
- **OpenAI-compatible**: `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/models` — these live under
  [`entrypoints/openai/`](../../python/sglang/srt/entrypoints/openai/) and lower
  onto the same `GenerateReqInput` the native route builds.

The handler is thin: parse, validate, hand to `TokenizerManager`, stream back.
No model state lives here.

`/health` is not free — it submits a real 1-token generation
(`HEALTH_CHECK_<uuid>` shows up in the scheduler log), which is why `start.sh`
polls it to decide readiness and `monitor.py` deliberately does not.

### TokenizerManager

[managers/tokenizer_manager.py](../../python/sglang/srt/managers/tokenizer_manager.py).
Runs **in the api-server process**, as asyncio coroutines — not a subprocess.
Per request it:

1. runs the HF tokenizer (`init_tokenizer_and_processor`), or the chat
   template + multimodal processor for chat requests;
2. builds a `TokenizedGenerateReqInput` and ZMQ-pushes it to the scheduler;
3. registers an asyncio future keyed by request id (`rid`);
4. consumes detokenized chunks arriving from the DetokenizerManager and
   resolves/streams that future;
5. keeps the Prometheus counters that back `/metrics` (TTFT, ITL, e2e latency,
   prompt/generation token totals).

It is also the control path: aborts, `/flush_cache`, and weight updates all
round-trip through the same socket pair.

### Scheduler

[managers/scheduler.py](../../python/sglang/srt/managers/scheduler.py), one per
`(tp_rank, pp_rank)`, running `run_scheduler_process`. This is the busy one —
`monitor.py` shows it at ~55-70 % CPU even when idle, because the event loop
polls its ZMQ socket continuously.

Loop selection happens at
[scheduler.py:4893](../../python/sglang/srt/managers/scheduler.py#L4893):
`event_loop_pdmux` → `event_loop_pp` → **`event_loop_overlap_mlx`** →
`event_loop_overlap` → `event_loop_normal`. This run lands on
`event_loop_overlap_mlx`, since `enable_overlap_mlx = not
disable_overlap_schedule and use_mlx()`.

Each iteration: drain new requests off ZMQ → `get_next_batch_to_run()` decides
prefill vs decode and how many requests fit → `tp_worker.forward_batch_generation()`
→ `process_batch_result()` updates per-request state → finished/streamed tokens
go out to the DetokenizerManager. Owned state:

- the **waiting queue** and the **running batch** (`--schedule-policy fcfs` here);
- the **radix cache** for prefix reuse (`Init Unified Radix Cache … UnifiedTreeCore`);
- the **token→KV slot allocator** and `req_to_token_pool`;
- retraction/preemption when KV runs short.

Its `Prefill batch, …` / `Decode batch, …` lines (every
`--decode-log-interval` steps) are the single most useful thing in the log:

```
Prefill batch, #new-seq: 1, #new-token: 18, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, ...
Decode batch,  #running-req: 1, #token: 19, token usage: 0.00, cuda graph: False, gen throughput (token/s): 97.25, #queue-req: 0
```

### TpModelWorker

[managers/tp_worker.py](../../python/sglang/srt/managers/tp_worker.py). A thin
adapter between scheduler bookkeeping and the executor: it owns the
`ModelRunner`, exposes `forward_batch_generation(ForwardBatch)`, and holds the
worker-local view of the memory pools. On a multi-GPU box it is also where the
tensor-parallel rank lives — the collective ops are inside the model, but the
worker is the per-rank object the scheduler talks to. Constructed in
`Scheduler.init_tp_model_worker()`
([scheduler.py:901](../../python/sglang/srt/managers/scheduler.py#L901)), which is
also the `use_mlx()` fork point.

### ModelRunner

[model_executor/model_runner.py](../../python/sglang/srt/model_executor/model_runner.py).
The executor. It loads weights, builds the attention backend, sizes and
allocates the KV pool, captures CUDA graphs (skipped here), and runs
`forward(ForwardBatch)` → logits → sampler → next token ids.

One per TpModelWorker. It is a *frozen core file* in this repo — see
`.claude/rules/modify-component-must-read.md` before editing it.

### DetokenizerManager

[managers/detokenizer_manager.py](../../python/sglang/srt/managers/detokenizer_manager.py),
its own process. Takes `BatchTokenIDOutput` from the scheduler, runs
incremental detokenization (holding the per-request decode state so partial
UTF-8 and multi-token graphemes stream correctly), and pushes
`BatchStrOutput` back to the TokenizerManager. It is a separate process so
that detokenization — pure Python string work — never blocks the scheduler
between forward passes.

## A request, end to end

`POST /generate` with `{"text": "...", "sampling_params": {...}}`:

```
curl
 └─► [proc 1] uvicorn/FastAPI  → GenerateReqInput
      └─► TokenizerManager     → HF tokenize → TokenizedGenerateReqInput
           └─ zmq ipc ─►
[proc 2]    Scheduler          → waiting queue → radix prefix match → ScheduleBatch
             └─► TpModelWorker → ForwardBatch
                  └─► ModelRunner (MlxModelRunner) → Metal kernels → logits → sample
             ◄── next token ids
             └─ zmq ipc ─►
[proc 3]         DetokenizerManager → incremental detokenize → BatchStrOutput
                  └─ zmq ipc ─►
[proc 1]              TokenizerManager → resolve/stream the asyncio future
                       └─► FastAPI → SSE chunk or final JSON
```

The `meta_info` on every response carries the timestamps for each hop —
`request_received_ts`, `api_server_dispatch_finish_ts`, `queue_time`,
`forward_entry_time`, `prefill_finished_time`, `request_finished_ts`,
`response_sent_to_client_ts` — which is the cheapest way to see where time
actually went.

## The Rust api-server

`SGLANG_RUST_SERVER=1` (what `./start.sh --rust` sets) replaces the Python
api-server, TokenizerManager **and** DetokenizerManager with Rust threads
running **inside the scheduler process**. The docstring at
[managers/rust_server.py:1](../../python/sglang/srt/managers/rust_server.py#L1)
says it plainly, and `_launch_subprocesses` takes an early return that skips
the detokenizer subprocess and the tokenizer manager entirely
([engine.py:1168](../../python/sglang/srt/entrypoints/engine.py#L1168)).

Observed here — two processes instead of three, and no `sglang::detokenizer`:

```
  14471   ppid=14380    launcher (Rust api-server lives in scheduler)
  14566   ppid=14471    scheduler
```

The Rust side mirrors the Python module layout one-for-one:
[`api_server/`](../../rust/sglang-server/src/api_server/) (axum; `native_api.rs`
and `openai/`), [`tokenizer_manager.rs`](../../rust/sglang-server/src/tokenizer_manager.rs),
[`detokenizer.rs`](../../rust/sglang-server/src/detokenizer.rs), and
[`ring.rs`](../../rust/sglang-server/src/ring.rs) — a lock-free ring buffer that
replaces the ZMQ hop, since producer and consumer are now threads in one
process. `Scheduler.maybe_init_rust_server()` swaps `self.recv_from_tokenizer`
to the `RustServer` object so the event loop drains that ring instead of a
socket ([scheduler.py:1998](../../python/sglang/srt/managers/scheduler.py#L1998)).

It logs through `tracing` rather than Python `logging`, so the two styles
interleave in `server.log`:

```
2026-08-13T16:09:25.686272Z  INFO sglang_server::tokenizer: loaded tokenizer path=...
[2026-08-13 11:09:25] SGLANG_RUST_SERVER enabled, Rust server listen on 127.0.0.1:30000
```

**Building it.** The extension is `sglang.srt.server._core`, a PyO3 cdylib from
`rust/sglang-server`. `pyproject_other.toml`'s `[tool.sglang] rust-extensions =
["multimodal"]` allowlist excludes it, and `SGLANG_BUILD_RUST_EXTS` can only
narrow that list, never widen it — so it was built by hand:

```bash
cd rust
RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
  PYO3_PYTHON=../studyRun/localRun0813/.venv/bin/python \
  cargo build --release -p sglang-server
mkdir -p ../python/sglang/srt/server
cp target/release/libsglang_server.dylib ../python/sglang/srt/server/_core.so
```

The `RUSTFLAGS` are what setuptools-rust would pass: a Python extension module
must leave `Py*` symbols unresolved at link time, and without them the macOS
linker fails with `symbol(s) not found for architecture arm64`. The
`sglang/srt/server/` directory needs no `__init__.py` (implicit namespace
package), and `*.so` / `*.dylib` are gitignored, so the checkout stays clean.
`rm -rf python/sglang/srt/server rust/target` undoes all of it.

**Caveat:** the Rust router is a subset — `/generate`, `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/health`, `/health_generate`, `/model_info`,
`/server_info`, and nothing else. In particular there is no `/metrics`, so
`monitor.py`'s metrics pane stays empty and only the process pane is live.

## What Apple Silicon changes

`SGLANG_USE_MLX=1` routes execution through
[`hardware_backend/mlx/`](../../python/sglang/srt/hardware_backend/mlx/) instead
of PyTorch. Without it SGLang falls back to `torch.mps`, which implements far
fewer of the ops these models need. The pieces:

- **`MlxTpModelWorker`** subclasses `TpModelWorker` and overrides
  `_init_model_runner` to build *two* objects:
  - **`MlxModelRunnerStub`** — a real `ModelRunner` subclass that creates only
    the CPU-side bookkeeping the scheduler needs (`req_to_token_pool`,
    `token_to_kv_pool_allocator`) and allocates **zero** GPU KV memory. Its
    `_DummyKVCache` satisfies the interface and raises on any buffer access.
  - **`MlxModelRunner`** — the thing that actually runs, loading weights via
    `mlx_lm.load()` and keeping KV in `MlxAttentionKVPool`.

  So "how many ModelRunners" has a footnote on this backend: one
  scheduler-facing stub plus one MLX executor, per worker. The log says so
  out loud:

  ```
  MLX stub: skipping PyTorch model weight loading (inference runs through MLX)
  MLX stub: initialized minimal pools (max_total_num_tokens=108279, ..., zero GPU KV cache allocation)
  ```

- **Overlap scheduling** is MLX-native. `event_loop_overlap_mlx`
  ([mlx/scheduler_mixin.py](../../python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py))
  keeps two lazy MLX graphs queued on the GPU while the CPU does scheduling
  bookkeeping on the older one; chained decodes let step N+1 read step N's
  still-lazy writes through MLX's dependency tracking, so the GPU never idles
  between steps. This is the Metal equivalent of the CUDA-graph + overlap trick.

- **KV pool sizing** is dynamic against unified memory rather than a fixed
  VRAM fraction:

  ```
  Wired memory limit set to 28.1 GB
  Auto-sized attention KV pool: sys_available=13.14 GB, mlx_limit=28.1 GB,
    mlx_used=1.11 GB, kv_budget=11.57 GB, bytes_per_slot=114688, pool_size=108279
  MlxAttentionKVPool: 108280 slots x 28 layers x 8 heads x 128 dim, dtype=bfloat16, ~11843.1 MB
  ```

  `max_total_num_tokens` therefore varies run to run (83 980 / 108 279 /
  116 957 across three boots) depending on what else the Mac is doing.

- **`--disable-cuda-graph` is mandatory** and the attention backend resolves to
  `torch_native`; the MLX path patches attention with `MLXAttentionWrapper`
  underneath regardless.

- **Quantization**: `--quantization mlx_q4` / `mlx_q8` quantizes in memory at
  load time via `mlx_lm.utils.quantize_model`; pre-quantized
  `mlx-community/*-4bit` repos load directly. `STUDY_QUANT` in `config.sh`
  wires this up.

## Numbers from this box

M3 MacBook Pro, 36 GB, Qwen3-0.6B bf16, single request:

| | |
|---|---|
| cold start to `/health` ok | 12-20 s (~27 s of that is tokenizer/process setup, 3.4 s scheduler e2e, 0.7 s MLX weight load) |
| decode throughput | 95-99 tok/s steady state |
| avg TTFT | ~0.9 s (first request, includes MLX warmup); ~60 ms warm |
| avg inter-token latency | ~12 ms |
| RSS: api-server / scheduler / detokenizer | ~130 MB / 2.7 GB / ~100 MB |
| scheduler CPU when idle | 55-70 % of one core (the event loop polls) |

`max_total_num_tokens` around 108 k means the 0.6 B model can hold ~100 k
tokens of KV — context length is the binding constraint (40 960), not memory.

## Local gotchas

**The Hugging Face CDN is blocked on this network.** `huggingface.co` itself
resolves and serves small files, but every LFS/Xet redirect lands on
`us.aws.cdn.hf.co`, which the corporate proxy answers with an HTML block page
under a "Generative AI Block" policy — surfacing as a confusing
`403 Forbidden` from `hf_hub`. `hf-mirror.com` is blocked the same way.
ModelScope is not, so:

```bash
curl -L --retry 20 --retry-all-errors -C - -o model.safetensors \
  "https://www.modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/model.safetensors"
```

`config.sh` sets `HF_HUB_OFFLINE=1` and points `--model-path` at the local
directory so nothing tries the hub at startup.

**Do not name your own env vars `SGL_*`.** `sglang.srt.environ` rewrites every
`SGL_*` variable in the environment to `SGLANG_*` at import time and emits a
`DeprecationWarning` per variable, per process. An early version of these
scripts used `SGL_PORT`, `SGL_MODEL_PATH`, … and produced 30 warnings before
the first real log line. They are `STUDY_*` now.

**`--log-requests` is verbose at every level.** All four levels print the whole
`GenerateReqInput` dataclass (~70 fields); the level only changes which fields
are skipped. Two of those per request buries the scheduler lines, so it is off
by default and gated behind `STUDY_LOG_REQUESTS=1`.

**Metal kernels in `sgl-kernel` were not built.** That needs `xcrun --find
metal`, which the bare Command Line Tools do not provide (full Xcode does).
It is optional — MLX ships its own kernels — but `SGLANG_MLX_USE_CUSTOM_ROPE=1`
and `SGLANG_MLX_FUSE_SWIGLU=1` from the
[Apple Metal doc](../../docs/docs/hardware-platforms/apple_metal.mdx) will not
have anything to load.
