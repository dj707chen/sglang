# SGLang `python/` — Repository Overview

> Study note captured from a Claude Code session.
> Working directory: `/Users/WChen/AI/sglangTry/sglang/python`
> Branch: `study` @ `687967c70d`
> Model: Claude Opus 5 (1M context)

---

## The ask

> Help me understand this repository, use diagrams if necessary, save my ask and your response in a MD file with your model name at end of file name under study directory.

---

## The response

### Big picture

SGLang is a **high-performance LLM/multimodal serving engine**. This repo is a monorepo; `python/` holds the engine itself — everything from the HTTP surface down to CUDA-graph-captured model forwards.

```
sglang/
├── python/               ← you are here (the engine)
│   └── sglang/
│       ├── srt/          "SGLang RunTime" — 1585 .py files, the actual server
│       ├── lang/         the frontend DSL (gen/select/fork, structured programs)
│       ├── cli/          `sglang serve` / `generate` / `killall`
│       ├── test/         test harness + runners
│       └── bench_*.py    serving / offline / one-batch benchmarks
├── rust/                 3 PyO3 crates that replace hot-path Python in-process
├── sgl-model-gateway/    separate Rust router/gateway
├── sgl-kernel/ (3rdparty) AOT CUDA/C++ kernels
├── proto/                sglang.proto (gRPC frontend)
└── docs/ test/ benchmark/ docker/ scripts/
```

Two distinct usage modes share the same core:

| Mode | Entry | Shape |
|---|---|---|
| **Server** | `sglang.launch_server` → `srt/entrypoints/http_server.py` | FastAPI, OpenAI-compatible |
| **Offline / embedded** | `sglang.Engine(...)` → `srt/entrypoints/engine.py` | Python API, used by RL frameworks (verl, slime, Miles, AReaL) |

`Engine` is the real object. `http_server.py` is a thin FastAPI shell wrapping the same `TokenizerManager`.

---

### The process topology

This is the single most important thing to internalize. SGLang is **not** one process — it's a small pipeline of processes connected by **ZeroMQ**, so Python GIL contention in tokenization never stalls the GPU loop.

```
 HTTP client
     │
     │  POST /v1/chat/completions
     ▼
┌──────────────────────────────────────────────────────────┐
│  Process 0 — main / "tokenizer" process                  │
│                                                          │
│   http_server.py  (FastAPI + uvicorn, asyncio)           │
│        │                                                 │
│        ▼                                                 │
│   entrypoints/openai/*  serving_chat / completions /     │
│        │                embedding / responses / score    │
│        ▼                                                 │
│   TokenizerManager  (managers/tokenizer_manager.py)      │
│     • HF tokenizer → token ids                           │
│     • chat template / multimodal preprocessing           │
│     • one asyncio future per request id                  │
└────────┬─────────────────────────────────────────────────┘
         │ ZMQ PUSH  scheduler_input_ipc_name
         │ (msgspec-encoded TokenizedGenerateReqInput)
         ▼
┌──────────────────────────────────────────────────────────┐
│  Process 1..N — Scheduler, one per TP rank               │
│                                                          │
│   Scheduler  (managers/scheduler.py, ~5k LOC)            │
│     event_loop_overlap()  ← the heartbeat                │
│        │                                                 │
│        ▼                                                 │
│   TpModelWorker → ModelRunner → model forward (GPU)      │
└────────┬─────────────────────────────────────────────────┘
         │ ZMQ PUSH  detokenizer_ipc_name  (output token ids)
         ▼
┌──────────────────────────────────────────────────────────┐
│  Process N+1 — DetokenizerManager                        │
│     incremental detokenization (handles partial UTF-8)   │
└────────┬─────────────────────────────────────────────────┘
         │ ZMQ PUSH  tokenizer_ipc_name  (text deltas)
         ▼
   back to TokenizerManager → resolves the asyncio future → SSE stream
```

Channels are declared in one place — `PortArgs` in [server_args.py:9668](../../python/sglang/srt/server_args.py#L9668): `tokenizer_ipc_name`, `scheduler_input_ipc_name`, `detokenizer_ipc_name`, `rpc_ipc_name`, `metrics_ipc_name`. Single-node uses `ipc://` unix sockets; multi-node switches to `tcp://`.

Spawning lives in `Engine._launch_subprocesses` ([engine.py:1036](../../python/sglang/srt/entrypoints/engine.py#L1036)). Variants it handles: `node_rank >= 1` (no tokenizer at all, just block on schedulers), `tokenizer_worker_num > 1` (a `MultiTokenizerRouter`), `SGLANG_RUST_SERVER=1` (the Rust crate hosts HTTP + tokenize + detokenize inside the rank-0 scheduler, and the Python processes above are skipped entirely).

---

### The scheduler event loop — where everything happens

`Scheduler` is the heart. Its `__init__` is pure orchestration: ~40 `init_*` methods (`init_memory_pools`, `init_all_attention_backends`, `init_all_cuda_graphs`, `init_disaggregation`, `init_overlap`, …). The repo enforces this style — see `.claude/skills/large-class-style/`.

Two loops, dispatched by `run_event_loop()` ([scheduler.py:1653](../../python/sglang/srt/managers/scheduler.py#L1653)):

```
event_loop_normal()                    event_loop_overlap()   ← default
─────────────────────                  ────────────────────────────────
while True:                            while True:
  recv_requests()                        recv_requests()
  process_input_requests()               process_input_requests()
  batch = get_next_batch_to_run()        batch = get_next_batch_to_run()
  result = run_batch(batch)   ─┐         result = run_batch(batch)   ← launch, don't wait
  process_batch_result(...)   ─┘         _apply_war_barrier()
                                         result_queue.append(...)
  (CPU blocks on GPU)                    process_batch_result(last_batch)  ← CPU work
                                         launch_batch_sample_if_needed()
                                       (CPU of step N overlaps GPU of step N)
```

The overlap loop is the "zero-overhead CPU scheduler" from the v0.4 blog: batch construction, radix-cache bookkeeping, and result processing for step *N-1* run on the CPU while step *N*'s kernels are in flight on `forward_stream`. The scheduler owns a separate `schedule_stream`, and `_apply_war_barrier()` inserts a write-after-read fence so the next step's writes to shared buffers can't race the current forward's reads.

#### What `get_next_batch_to_run` decides

```
                    ┌─────────────────────┐
   new requests ───►│  waiting_queue      │
                    └──────────┬──────────┘
                               │  SchedulePolicy: lpm | fcfs | lof | dfs-weight | priority
                               │  PrefillAdder: budget by tokens + #reqs + KV headroom
                               ▼
              ┌────────────────────────────────┐
              │ get_new_batch_prefill()        │  EXTEND batch (chunked if too long)
              └───────────────┬────────────────┘
                              │  if no prefill fits ↓
              ┌────────────────────────────────┐
              │ update_running_batch()         │  DECODE batch (1 token/req)
              │   • retract reqs on OOM        │
              │   • filter finished            │
              └───────────────┬────────────────┘
                              ▼
                        ScheduleBatch  ──►  run_batch()
```

Prefill is preferred over decode when it fits — that's continuous batching. Under KV pressure, running requests are **retracted** (their KV freed, pushed back to the queue) rather than the server OOM-ing. `ForwardMode` ∈ `{EXTEND, DECODE, MIXED, IDLE, TARGET_VERIFY, DRAFT_EXTEND, PREBUILT, SPLIT_PREFILL, DLLM_EXTEND}` ([forward_batch_info.py:98](../../python/sglang/srt/model_executor/forward_batch_info.py#L98)).

There is a repo rule worth knowing: **`ScheduleBatch` fields are rebound, never mutated in place** (`.claude/rules/schedule-batch-out-of-place-mutation.md`), because the overlap loop holds `batch.copy()` snapshots in `result_queue`.

---

### RadixAttention — the signature feature

`mem_cache/radix_cache.py`. KV cache is not per-request; it's a **radix tree keyed on token-id prefixes**, shared across all requests.

```
                    ┌── root ──┐
                    │          │
        "You are a helpful"   "Translate the"
              │                     │
        ┌─────┴─────┐          ┌────┴────┐
   " assistant."  " bot."   " following"  " text"
     ▲                          ▲
     │                          │
   req A, req B share          req C
   this prefix's KV            (lock_ref pins live nodes)
```

- `match_prefix()` → longest cached prefix, so prefill only computes the suffix.
- `inc_lock_ref` / `dec_lock_ref` pin nodes belonging to in-flight requests.
- `evict()` LRU-evicts unlocked leaves when the token pool runs low.

Two pools underneath: `req_to_token_pool` (request slot → its token positions) and `token_to_kv_pool` (paged KV blocks). Variants of the cache exist for the harder cases — `hiradix_cache.py` (tiered CPU/disk offload), `swa_radix_cache.py` (sliding-window models), `mamba_radix_cache.py` (hybrid SSM state), `radix_cache_cpp.py` (the C++ tree in `cpp_radix_tree/`).

---

### The execution stack below the scheduler

```
Scheduler.run_batch(batch)
   │
   ▼
TpModelWorker  (managers/tp_worker.py)
   │  • owns the ModelRunner, handles weight updates / LoRA / IPC RPCs
   ▼
ModelRunner  (model_executor/model_runner.py, ~2k LOC — a FROZEN core file)
   │  • load_model, init memory pools, init attention backend,
   │    capture CUDA graphs, forward()
   ▼
ForwardBatch  (model_executor/forward_batch_info.py)
   │  the GPU-side view: input_ids, positions, out_cache_loc,
   │  attn_backend metadata, mm inputs
   ▼
srt/models/<arch>.py     216 model files: llama.py, qwen3_moe.py,
   │                     deepseek_v2.py, glm4.py, gemma3.py, ...
   ▼
srt/layers/              RadixAttention, linear, layernorm, rotary,
                         moe/, quantization/, sampler, logits_processor
   ▼
srt/layers/attention/<backend>.py
                         flashinfer, flashattention (FA3), triton, flashmla,
                         trtllm_mha/mla, aiter (ROCm), torch_native, xpu, ...
```

Two notes:
- `model_runner.py` is listed as a **frozen core file** in `.claude/rules/modify-component-must-read.md` — edits require reading the `large-class-style` skill first.
- Attention backends are pluggable via `attention_registry.py`; each implements `init_forward_metadata`, `forward_extend`, `forward_decode` against `base_attn_backend.py`.

---

### Feature map — where to look for what

| Feature | Location |
|---|---|
| Continuous batching / chunked prefill | `managers/schedule_policy.py`, `scheduler.py` |
| Prefix caching (RadixAttention) | `mem_cache/radix_cache.py` + variants |
| Paged KV / allocators | `mem_cache/memory_pool.py`, `mem_cache/allocator/` |
| Speculative decoding | `speculative/` — EAGLE (v1/v2/multi-layer), DFlash, n-gram, standalone, frozen-KV MTP |
| PD disaggregation | `disaggregation/` — `prefill.py`, `decode.py`, transports: mooncake / nixl / mori / ascend |
| Tensor / pipeline / expert / data parallel | `distributed/`, `layers/dp_attention.py`, `scheduler_pp_mixin.py`, `eplb/` |
| MoE + expert-parallel load balancing | `layers/moe/`, `eplb/`, `elastic_ep/` |
| Quantization | `layers/quantization/` — fp8, fp4/mxfp4/nvfp4, int8, awq, gptq, marlin, bitsandbytes, gguf, modelopt |
| Structured / constrained output | `constrained/` (xgrammar, outlines, llguidance) |
| LoRA (multi-adapter batching) | `lora/` |
| Multimodal | `multimodal/`, `managers/mm_utils.py`, `managers/multimodal_processor.py` |
| Tool calling / reasoning parsers | `function_call/`, `parser/` |
| Diffusion (image/video) | `sglang/multimodal_gen/`, `srt/models/` diffusion entries |
| RL weight sync | `weight_sync/`, `checkpoint_engine/`, `Engine.update_weights_from_*` |
| Config surface | `server_args.py` (~9.8k LOC), `arg_groups/`, `environ.py` |
| Runtime/global state discipline | `runtime_context.py` (+ the `sglang-runtime-context` skill) |

---

### Request lifecycle, end to end

```
1.  POST /v1/chat/completions
2.  serving_chat.py → chat template → TokenizerManager.generate_request()
3.  tokenize → TokenizedGenerateReqInput → ZMQ → Scheduler
4.  Scheduler.handle_generate_request() → Req → waiting_queue
5.  get_next_batch_to_run() → PrefillAdder picks reqs within budget
      └─ radix match_prefix() shortens the extend
6.  run_batch(EXTEND) → TpModelWorker → ModelRunner.forward() → logits
7.  Sampler → next token; KV written into token_to_kv_pool
8.  process_batch_result() → stream token via ZMQ → DetokenizerManager
9.  DetokenizerManager → text delta → ZMQ → TokenizerManager
10. TokenizerManager resolves the per-request asyncio queue → SSE chunk out
11. loop 6–10 in DECODE mode until EOS / max_new_tokens / abort
12. cache_finished_req() → KV inserted into the radix tree for future reuse
```

---

### Reading order if you're new

1. `srt/entrypoints/engine.py` — `Engine.__init__`, `_launch_subprocesses`. Gives you the process picture.
2. `srt/managers/scheduler.py` — `event_loop_overlap`, `get_next_batch_to_run`, `run_batch`. The heartbeat.
3. `srt/managers/schedule_batch.py` — `Req` and `ScheduleBatch`, the two core data structures.
4. `srt/mem_cache/radix_cache.py` — `match_prefix`, `cache_finished_req`, `evict`.
5. `srt/model_executor/model_runner.py` + `forward_batch_info.py` — the GPU boundary.
6. `srt/models/llama.py` — the simplest complete model, the template for all 216.

### House rules living in `.claude/`

Worth skimming before any edit — they are enforced conventions, not suggestions:
`rules/general-code-style.md` (stateless, immutable, <100 LOC functions, <2k LOC files, no mixins, keyword args), `rules/no-dataclasses.md` (use `msgspec.Struct`), `rules/no-getattr-defensive.md`, `rules/schedule-batch-out-of-place-mutation.md`, `rules/forward-batch-init-new-purity.md`, plus the skills index in `rules/modify-component-must-read.md`.
