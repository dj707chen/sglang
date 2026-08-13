# SGLang `rust/` — Repository Overview

> Study note captured from a Claude Code session.
> Working directory: `/Users/WChen/AI/sglangTry/sglang/rust`
> Branch: `main` @ `773806229`

---

## The ask

> Help me understand this repository, use diagrams if necessary

---

## The response

### Big picture

SGLang is a Python LLM serving engine. `rust/` is a **cargo workspace of three PyO3 extension modules** that replace hot-path Python components in-process. They aren't standalone binaries — each compiles to a `cdylib` that gets imported as a Python module inside the sglang wheel.

```
sglang/
├── python/sglang/       the engine (scheduler, model runner, attention, ...)
├── proto/               sglang.proto  ──┐ (consumed by rust/sglang-grpc/build.rs)
├── rust/                 ← you are here │
│   ├── sglang-server/   → sglang.srt.server._core      HTTP frontend
│   ├── sglang-grpc/     → sglang.srt.grpc._core     ←──┘ gRPC frontend
│   └── sglang-mm/       → sglang.srt.multimodal._core   image preprocessing
└── sgl-model-gateway/   separate Rust project (router/gateway), NOT in this workspace
```

The mapping crate → Python module is declared in each `Cargo.toml` under
`[package.metadata.sglang] python-module`, and `python/setup.py` auto-discovers the
workspace and turns each member into a `setuptools-rust` `RustExtension`.

### The three crates

| Crate | Replaces | Shape |
|---|---|---|
| `sglang-server/` | Python api-server + `TokenizerManager` + `DetokenizerManager` | `cdylib` + `rlib`, axum/tokio, biggest crate |
| `sglang-grpc/` | Python gRPC frontend | `cdylib`, tonic; calls *back into* Python via a `PyBridge` |
| `sglang-mm/` | HF/PIL image preprocessing | `cdylib` (Python path) **+ `rlib`** (linked into sglang-server) |

`sglang-mm` is the interesting dual-build: features `python` and `parallel` are
deliberately non-default so that when `sglang-server` depends on it, cargo's feature
unification can't leak PyO3 or rayon into the pure-Rust core
(`sglang-mm/Cargo.toml:26-44`).

### sglang-server — the main event

The whole thing runs **inside the Python scheduler process**, on its own threads.
Stages 1–5 never touch a `PyObject`, so they run concurrently with the scheduler
without fighting for the GIL. Only three methods cross the boundary.

```
                    ┌──────────────── Rust threads (no GIL) ────────────────┐
  HTTP client       │                                                       │
      │             │  ┌─────────────┐                                      │
      └── axum ────►│  │ api_server  │  tokio multi-thread, pinned core set │
                    │  └──────┬──────┘                                      │
                    │         │ flume: TmEvent::Ingress                     │
                    │         ▼                                             │
                    │  ┌─────────────┐        ┌───────────────┐             │
                    │  │ tm::ingress │◄──────►│ tokenizer pool│ N pinned    │
                    │  │  (FSM)      │        └───────────────┘ threads     │
                    │  │             │        ┌───────────────┐             │
                    │  │             │◄──────►│  mm workers   │ K threads   │
                    │  └──────┬──────┘        └───────┬───────┘ (sglang-mm) │
                    │         │                       │                     │
                    │   ingress ring            mm_sidecar (rid → features) │
                    └─────────┼───────────────────────┼─────────────────────┘
                              │                       │
   ═══ GIL boundary ══════════▼═══════════════════════▼═════════════════════
                       recv_requests()            take_mm()
                    ┌───────────────────────────────────────────┐
                    │        Python Scheduler event loop        │
                    │   (model runner, batching, CUDA launch)   │
                    └───────────────────────┬───────────────────┘
                              push_batch() / push_result() / push_error()
   ═══════════════════════════▼═════════════════════════════════════════════
                    ┌─────────────┐
                    │ egress ring │
                    └──────┬──────┘
                    ┌──────▼──────┐      ┌──────────────────┐
                    │ tm::egress  │─────►│ detok shards × M │ pinned, one
                    └─────────────┘      └────────┬─────────┘ rid→sink map each
                                                  │ EgressSink
                                                  ▼  SSE / unary JSON → client
```

Key files: `src/runtime.rs` (thread layout + core pinning), `src/ring.rs` (the two
boundary queues), `src/lib.rs` (the PyO3 surface).

#### The GIL story is the design constraint

Nearly every non-obvious decision in this crate traces back to GIL cost, and the
comments say so explicitly:

- `recv_requests` / `push_batch` run **GIL-held on purpose** — detaching costs a 5 ms
  interpreter switch-interval wait to cover ~0.2 µs of work (`src/lib.rs:154-176`).
- `push_frame` only detaches on the slow path (full ring), where genuine blocking makes
  it pay (`src/lib.rs:290-301`).
- Ingress is **columnar**: msgpack header + raw little-endian int64 `ids` blob, so the
  big token tensor never goes through msgpack, and cells are copied straight into one
  `PyBytes`.
- `take_mm` moves Rust `Vec`s into numpy arrays with zero copy; hashes are precomputed
  on worker threads so no per-byte work happens on the scheduler loop.

#### Request lifecycle FSM

The state lives *inside* the owned request struct, so transitions are in-place
mutations by whichever stage owns it — no locks anywhere (`src/fsm.rs`).

```
Received → Validating ─┬─ Control ──────────────────────────► PreSendValidating
                       └─ Generate → Normalizing ─┬─ HasMultimodal ──► Encoding ──┐
                                                  ├─ NeedsTokenize ──► Tokenizing─┤
                                                  └─ AlreadyTokenized ────────────┤
                                                                                  ▼
                                        Queued ◄──────────────────── PreSendValidating
                                          │ (pushed to ingress ring)
                                          ▼
                                  Streaming{chunks_sent} → Finalizing → Completed
                       from any state: → Failed(Error) | Aborted
```

Ingress edges are driven in `src/tokenizer_manager/ingress.rs`, egress edges in
`src/tokenizer_manager/egress.rs` + `src/detokenizer.rs`.

#### HTTP surface

`src/api_server.rs` mounts per-module routers:

- native: `/generate`, `/health`, `/health_generate`, `/server_info`, `/get_model_info`
- OpenAI: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/v1/models/{model}`
  — `src/api_server/openai.rs` is a deliberately thin adapter over Dynamo's OpenAI
  types/renderer/parsers; `ChunkEvent` stays the single backend output type for both
  unary and streaming
- PD disaggregation bootstrap: `/route`, `/register_dp_rank`, `/query_dp_ranks`
  (conditionally mounted)

Note the honest `TODO(auth)` at `src/api_server.rs:60-62`: a configured `api_key` does
**not** yet protect these routes, unlike the Python server.

### Multimodal path

Two entry points into the same crate:

```
Python processor (Inkling)  ──► sglang.srt.multimodal._core  ──┐
                                                               ├─► sglang-mm
sglang-server mm workers  ──► sglang_mm rlib (no pyo3) ────────┘
```

The design rule in `sglang-mm/src/pipeline.rs` is **families produce data, the driver
owns control flow**: a model family implements `MmFamilyProcessor` (decode → tensors +
a declarative `TokenLayout`), and `sglang-mm/src/driver.rs` mechanically applies
expansion, offsets, and positions — so every family gets identical failure semantics
for free. Currently `qwen_vl` and `inkling`.

Python selects a family by serializing a spec JSON; the matching table lives on the
Python side in `python/sglang/srt/managers/rust_server.py` (`NATIVE_MM_FAMILIES`), and
Rust parses it in `registry::pipeline_from_spec`. Features are handed back either
inline (single rank, zero-copy numpy) or via POSIX shm segments when the scheduler
broadcasts across TP ranks.

### sglang-grpc — opposite direction

Unlike sglang-server (Rust owns the frontend, calls into Python), sglang-grpc is Rust
**wrapping** Python: `start_server(host, port, runtime_handle)` spawns a tonic server on
its own Tokio runtime in a background thread, and `src/bridge.rs`'s `PyBridge` calls
`submit_generate` / `submit_embed` / `abort` on the Python `RuntimeHandle`. Protos
compile from `proto/sglang/runtime/v1/sglang.proto` via `tonic-build`, with a vendored
`protoc` fallback so `cargo clippy` works without a system protobuf.

### Build & enablement

- Toolchain pinned to Rust **1.92, edition 2024** (`rust-toolchain.toml`).
- Workspace-level `[workspace.dependencies]` is the single version source; release
  profile is `lto = true, codegen-units = 1`, and every crate sets `debug = false` so
  even editable installs get optimized builds.
- Runtime gate: `SGLANG_RUST_SERVER` (default `False`,
  `python/sglang/srt/environ.py:1390`). When on, the scheduler holds an
  `Optional[RustServer]` and delegates `recv_requests` / `push_generation` / idle
  parking to it.
- CI: `pr-test-rust.yml`, `pr-test-rust-exts.yml`, `pr-benchmark-rust.yml`, plus
  ext-build/cache-seeding workflows.

One notable pin: `regex-syntax = "=0.8.11"` **exactly** — this crate's accepted grammar
defines the "anything Rust admits, Python can compile" invariant in `message::sampling`,
and a minor bump could silently reopen a scheduler-killing hole
(`sglang-server/Cargo.toml:53-56`).

### Where I'd start reading

1. `sglang-server/src/lib.rs` — the entire Python-facing contract in ~330 lines, with
   the perf rationale inline.
2. `sglang-server/src/runtime.rs` — how threads, channels, and core pinning are wired.
3. `sglang-server/src/fsm.rs` + `src/tokenizer_manager/ingress.rs` — the request
   lifecycle.
4. `python/sglang/srt/managers/rust_server.py` — the mirror image from Python.

This codebase is unusually well-commented; module headers (`//!`) carry the design
intent and the *reasons* for the odd choices. When something looks strange, the comment
above it almost always explains the benchmark that forced it.
