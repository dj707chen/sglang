# localRun Plan — SGLang on MacBook Pro M3 Pro

Plan for [localRunGrafanaTUI.md](Prompt_localRunGrafanaTUI).

Status: **ALL PHASES COMPLETE** (2026-08-14) — 1, 6, 7, 3, 4, 2, 5.

Stack up: sglang `:30000`, Prometheus `:9090`, Grafana `:3000`, node_exporter `:9100`.
TUI: `./bin/tui.sh`. Component deep-dive: **[components.md](components.md)**.
Q1 bf16, Q2 Homebrew, Q3 node_exporter — answered. **Q4 answered by circumstance: no
source shim was needed and none was written; the repo is untouched.**

Grafana dashboard layout was reviewed and approved by the user on 2026-08-14.

**Two things need your `sudo` if you want them** (I deliberately did not escalate):
packet-level capture via `bin/capture_http.sh`. The non-root `bin/tap.sh` already records
the same HTTP traffic and is working.

**One thing is not achievable on this machine:** raw ZMQ inter-process frame capture — see
Phase 5, Tier B. Structural, not a missing tool.

Each phase below has a `Decisions made` and `Implementation details` block that I fill in
as I execute it. Everything currently marked `(pending)`.

---

## 0. Environment audit (already done, read-only)

Evidence gathered before writing this plan — this shapes every decision below.

| Thing | Finding |
|---|---|
| Machine | Apple M3 Pro, 36 GB unified memory, macOS 26.6.1 |
| Repo | `/Users/WChen/AI/sglangTry/sglang` @ `07821e9d56`, branch `localRun260814_A` |
| Venv | `studyIterations/venvs/mps-py312` — Python 3.12.8, uv-created (**no `pip` module**, use `uv pip --python`) |
| SGLang | `0.5.18.dev494+g07821e9d5`, installed **editable** from `python/` — source edits take effect live |
| torch | 2.11.0, `torch.backends.mps.is_available() == True` |
| MLX | `mlx` 0.32.0, `mlx-metal` 0.32.0, `mlx-lm` 0.31.3 |
| Model | `studyIterations/models/Qwen3-0.6B` already present (bf16, 1.5 GB) |
| Also cached | `~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit` |
| Rust | cargo 1.90.0 installed; repo pins toolchain **1.92** (`rust/rust-toolchain.toml`) |
| Rust server | `rust/sglang-server` = maturin/pyo3 ext module `_core`. **Not built** (`import _core` fails) |
| Prometheus | not installed |
| Grafana | not installed |
| Docker | Rancher Desktop at `~/.rd/bin/docker` |
| tcpdump | `/usr/sbin/tcpdump` present; no `tshark`/`wireshark` |
| Support libs | `prometheus-client`, `pyzmq`, `psutil`, `setproctitle`, `rich` 15.0, `fastapi`, `uvicorn` all present. **No `textual`.** |

Key discovery: **SGLang ships a first-class Apple Metal backend** —
`python/sglang/srt/hardware_backend/mlx/` (own `model_runner.py`, `tp_worker.py`,
`scheduler_mixin.py`, KV-cache layout, sampling, profiler) documented at
[docs/docs/hardware-platforms/apple_metal.mdx](../../docs/docs/hardware-platforms/apple_metal.mdx).
This is not a torch-MPS fallback hack; it is a separate hardware backend selected by
`SGLANG_USE_MLX=1`. That makes this machine a legitimate place to study the runtime.

Second key discovery affecting Phase 5: single-node SGLang wires its inter-process ZMQ
over **`ipc://` unix domain sockets**, not TCP
([server_args.py:9747-9758](../../python/sglang/srt/server_args.py#L9747-L9758)).
Unix sockets are invisible to `tcpdump`. Details and workaround in Phase 5.

---

## 1. Model selection & serving

### Selection reasoning

I am **keeping `Qwen3-0.6B` (bf16)** as the primary model. Criteria and why it wins:

1. **Must have a native SGLang model implementation.** `Qwen3ForCausalLM` is supported by
   both the CUDA and MLX paths, so nothing about the run is exotic.
2. **Must fit with room to spare.** 0.6 B bf16 ≈ 1.2 GB weights. On 36 GB unified memory
   that leaves the KV pool, the OS, Grafana/Prometheus and my own tooling completely
   unconstrained — I want headroom because the *point* is observing the components, not
   maximizing tokens/s.
3. **Fast startup.** I will be restarting the server many times across phases. A 0.6 B
   model loads in seconds; a 7 B would make each iteration painful.
4. **Non-trivial architecture.** Qwen3 has GQA (16 q-heads / 8 kv-heads, head_dim 128,
   28 layers) and tied embeddings — so KV-cache and attention metrics are actually
   interesting to watch, unlike a toy GPT-2.
5. **Already downloaded** — from **ModelScope** (`configuration.json` with
   `{"framework": "pytorch", ...}` is the ModelScope marker; the HF mirror does not ship
   that file). I will record the exact provenance in the run log rather than re-download.

Fallback / A-B comparison model: `mlx-community/Qwen3-0.6B-4bit`, already in the HF cache.
Useful to demo the MLX pre-quantized load path against bf16 without a new download.

### Serving

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path studyIterations/models/Qwen3-0.6B \
  --host 127.0.0.1 --port 30000 \
  --disable-cuda-graph \
  --enable-metrics \
  ...logging flags from Phase 6...
```

- Model downloads, if any, go to `studyIterations/models/`.
- `studyIterations/` gets a `.gitignore` (`models/`, `venvs/`, `logs/`, `pcap/`, `data/`) so
  weights and captures never enter git. The plan/doc/scripts **do** stay tracked.

**Open question for you (Q1):** bf16 as primary, or should I make the 4-bit MLX build the
primary and bf16 the comparison? I default to bf16 — fewer moving parts when the goal is
understanding components.

### ✅ Decisions made (executed 2026-08-14 16:36)

1. **Primary model = `Qwen3-0.6B` bf16** from `studyIterations/models/`, on the Q1 default. Not
   re-downloaded — verified the existing copy instead (see below).
2. **Provenance recorded: ModelScope mirror of upstream `Qwen/Qwen3-0.6B`.** The README is
   the upstream Qwen model card (its links point at `huggingface.co/Qwen/Qwen3-0.6B`), but
   the directory carries `configuration.json`
   (`{"framework": "pytorch", "task": "text-generation", "allow_remote": true}`), which
   only ModelScope emits. So: content is upstream Qwen, delivery channel was ModelScope.
3. **Launched with the minimum viable flag set**, deliberately *not* the full Phase 6
   logging set — I wanted a clean baseline where any failure is unambiguous. Logging flags
   land in Phase 6, `--enable-metrics` in Phase 3.
4. **Left the server running** after the phase so Phases 6/7/3/4 can attach to it rather
   than pay startup again.
5. **`studyIterations/.gitignore` written and verified with `git check-ignore -v`** rather than
   assumed: `models/`, `venvs/`, and per-run `*/logs/ */pcap/ */run/ */data/` are ignored;
   `localRunPlan.md`, `CC_conv.md` and the scripts stay tracked.

### ✅ Implementation details

**Model verification** (`safetensors.safe_open`, no full load): 1,503,300,328 bytes,
**311 tensors**, `model.embed_tokens.weight` = `bfloat16 (151936, 1024)`. Config confirms
28 layers, hidden 1024, **16 q-heads / 8 kv-heads (GQA)**, head_dim 128, vocab 151936.
Tensor count reconciles exactly: 28 layers × 11 tensors + `embed_tokens` + `lm_head` +
`model.norm` = 311.

**Launch command actually used:**
```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path studyIterations/models/Qwen3-0.6B \
  --host 127.0.0.1 --port 30000 \
  --disable-cuda-graph
```
PID → `run/sglang.pid`, stdout+stderr → `logs/server-20260814-163600.log`.

**Result: healthy in 14 s.** Key log evidence that the MLX backend is genuinely in play:
```
Initializing MlxModelRunner for end-to-end MLX inference
MLX model loaded in 0.74s
Wired memory limit set to 28.1 GB
MLX stub: skipping PyTorch model weight loading (inference runs through MLX)
MlxAttentionKVPool: 112665 slots x 28 layers x 8 heads x 128 dim, dtype=bfloat16, ~12322.7 MB
Init Unified Radix Cache. Tree Core: UnifiedTreeCore
Engine startup timings (s): load_weight=0.00, kv_cache_allocation=0.00,
  scheduler_e2e=2.88, tokenizer_e2e=8.80
```
Note the weights load in **0.74 s** via MLX while the torch-side `load_weight` is `0.00` —
the PyTorch loader is genuinely bypassed, not merely idle. Startup is dominated by
`tokenizer_e2e=8.80 s` (importing transformers + the model registry), not by the model.

**Smoke tests — all three pass:**

| Endpoint | Result |
|---|---|
| `POST /generate` | 48 tokens, coherent text, `temperature=0` |
| `POST /v1/chat/completions` | `finish_reason=length`, `usage={prompt:15, completion:64, total:79}` |
| `POST /v1/chat/completions` (`stream:true`) | proper SSE `chat.completion.chunk` deltas |

**Live process tree — this is the Phase 2 evidence for "ModelRunner, how many?":**
```
34866  main            107 MB   python -m sglang.launch_server   ← FastAPI/uvicorn + TokenizerManager
├─ 35005 (unnamed)       8 MB   multiprocessing.resource_tracker ← stdlib, NOT an SGLang component
├─ 35006 sglang::scheduler 1377 MB, 59% CPU                      ← Scheduler + TpModelWorker + MlxModelRunner
└─ 35007 sglang::detokenizer 82 MB                               ← DetokenizerManager
```
`/get_server_info` confirms `tp_size=1, dp_size=1, pp_size=1, device=mps`. So:
**exactly one ModelRunner**, living inside the single `sglang::scheduler` process — and it
is `hardware_backend/mlx/model_runner.py::MlxModelRunner`, not the CUDA
`model_executor/model_runner.py`. Answer confirmed empirically, not inferred.

`disable_overlap_schedule=False` — overlap scheduling **stays on** under MLX via
`async_eval()`, exactly as `apple_metal.mdx` claims and as
[server_args.py:4307-4310](../../python/sglang/srt/server_args.py#L4307-L4310) implements
(the forced-disable only applies when `use_mlx()` is false).

### ⚠️ Five findings that change later phases

1. **The KV pool auto-sized to 12.03 GB / 112,664 tokens** (`kv_budget=12.03 GB` from
   `sys_available=13.67 GB`). That is SGLang claiming ~89% of free memory for a 0.6 B
   model. It works now, but Phase 3 adds Prometheus + Grafana to the same 36 GB of
   *unified* memory. **Action: cap it in the Phase 7 launch script** (`--max-total-tokens`
   ≈ 32768, still ~10× more than these experiments need) so the observability stack cannot
   be starved by the thing it is observing.
2. **`/get_server_info` reports `attention_backend = torch_native`, which is misleading** —
   attention actually runs through `MlxAttentionKVPool`. The field reflects the torch-side
   stub, not the live path. **Do not put this field on the Grafana dashboard as-is**; it
   would state the opposite of the truth.
3. **Qwen3 emits `<think>` blocks and they land in `content`**, with
   `reasoning_content: null` and `reasoning_tokens: 0`. Without `--reasoning-parser qwen3`
   the token accounting conflates reasoning with answer. **Action: add that flag in Phase
   6**, otherwise the Grafana token panels measure a blend of two different things.
4. **This ModelScope mirror ships a redundant `lm_head.weight`** despite
   `tie_word_embeddings: true` — both `lm_head.weight` and `model.embed_tokens.weight` are
   present, ~311 MB of duplicate. Upstream HF omits it. Harmless (loaders tolerate it) but
   it explains why a "0.6 B" bf16 file is 1.5 GB rather than ~1.2 GB.
5. **`/dev/shm` does not exist on macOS** → a benign startup error,
   `load snapshot writer init failed: [Errno 2] ... '/dev/shm/sglang_loads_*.shm'`. It is
   the DP-load-snapshot path and is irrelevant at `dp_size=1`, but **it is a real
   constraint on Phase 5 option B1** (`--enable-dp-attention`), which leans on that
   machinery. Lowers my confidence in B1 on this platform.

Also worth knowing: **`/health` returns 503 until the warmup `/generate` completes** — so
the Phase 7 `start_sglang.sh` readiness probe must poll `/health` and tolerate 503, not
treat the first non-200 as failure.

---

## 2. Component deep-dive

Deliverable: `components.md` — a walkthrough written **against this specific running
process tree**, with `file:line` citations into the repo and real log/PID evidence
captured from my own run. Not a generic architecture essay.

Sections, mapped to your list:

### 2.1 API server — and "how does the Rust server run?"
There are **two** HTTP frontends in this repo and they are worth contrasting directly:

- **Python path (default):** FastAPI/uvicorn in
  [python/sglang/srt/entrypoints/http_server.py](../../python/sglang/srt/entrypoints/http_server.py) —
  ~40 routes (`/generate`, `/v1/chat/completions`, `/get_server_info`, `/health`,
  `/get_load`, `/metrics`…). Runs in the **main process**, in-process with TokenizerManager.
- **Rust path (opt-in):** `rust/sglang-server`, a **pyo3 extension module named `_core`**
  built with maturin, gated by the env var `SGLANG_RUST_SERVER`
  ([environ.py:1432](../../python/sglang/srt/environ.py#L1432)), driven from
  [python/sglang/srt/managers/rust_server.py](../../python/sglang/srt/managers/rust_server.py).
  Its `src/` has `api_server.rs`, `tokenizer_manager.rs`, `detokenizer.rs`, `ring.rs`,
  `fsm.rs` — i.e. it replaces the HTTP server **and** the tokenizer/detokenizer management layer
  with Rust, and it is **started by the Scheduler on rank 0**
  ([scheduler.py:1977-1990](../../python/sglang/srt/managers/scheduler.py#L1977-L1990)),
  not by the Python entrypoint. That inversion is the interesting bit and I want to
  demonstrate it live.

  Plan: `cargo`/maturin-build `_core` into the venv, launch with `SGLANG_RUST_SERVER=1`,
  and show the **process-tree and ZMQ-topology difference** side by side with the Python
  path. Marked **best-effort**: it pins Rust 1.92 (have 1.90 → `rustup` update needed) and
  the crate may never have been built on macOS/arm64. If it does not build in reasonable
  time I will document the architecture from source + the build error rather than burn
  hours on it, and say so plainly.

### 2.2 Tokenizer / TokenizerManager
In-process with the API server. Owns request IDs, the ZMQ send to the scheduler, the
receive loop from the detokenizer, and the per-request future map. I will trace one
request end-to-end with the actual IDs from a live run.

### 2.3 Scheduler
Separate **process** (`setproctitle` name `sglang::scheduler`,
[scheduler.py:4965](../../python/sglang/srt/managers/scheduler.py#L4965)). Event loop,
waiting queue → running batch, prefill/decode phase selection, radix cache, and the
`decode_log_interval` stats line. This is the component the TUI and Grafana will focus on.
Note: on MLX without `use_mlx()` the runtime forces `disable_overlap_schedule`
([server_args.py:4307-4310](../../python/sglang/srt/server_args.py#L4307-L4310)) — with
`SGLANG_USE_MLX=1` overlap **stays on** via MLX `async_eval()`. I will show both.

### 2.4 TpModelWorker
Lives inside the scheduler process. On this machine the MLX variant
`hardware_backend/mlx/tp_worker.py` is used instead of the CUDA one — I will confirm which
class is actually instantiated at runtime rather than assume.

### 2.5 ModelRunner — "how many?"
Direct answer with evidence: **one ModelRunner per (TP rank × PP rank × DP rank)**, each
owned by one TpModelWorker inside one scheduler process. With `--tp-size 1 --dp-size 1`
on this laptop that is **exactly one**, and it is
`hardware_backend/mlx/model_runner.py`, not the CUDA `model_executor/model_runner.py`.
I will prove the count by dumping the live process tree + a one-shot introspection, and
then show how the count changes on paper for TP=2/DP=2.

Also delivered: a **process/IPC topology diagram** (ASCII + a Mermaid version) generated
from the actual `lsof`/`ps` output of my run.

### ✅ Decisions made (executed 2026-08-14 18:00)

Deliverable written: **[components.md](components.md)**. Every claim is either a
`file:line` citation at `07821e9d56` or output captured from the live server; where code
and runtime disagree, the runtime wins and the discrepancy is called out.

1. **I built and ran the Rust server.** The plan rated this medium-high risk and reserved
   the right to document it from source. That turned out to be unnecessary — see below.
2. **Delivered the ASCII topology diagram; dropped the Mermaid version.** The ASCII diagram
   already carries bind/connect direction and pids, and a second rendering of the same
   graph would be a maintenance liability with no extra information. Say the word if you
   want Mermaid for embedding elsewhere.

### ✅ Implementation details

**The Rust server built cleanly in 1m28s** — Rust 1.92 was already installed via `rustup`
(the crate pins it; `cargo` on PATH is 1.90 and `rustup` resolves the pin automatically).

> **Gotcha recorded in components.md:** the first build emitted a **CPython 3.14** wheel and
> installed `_core` into the repo's unrelated `.venv`, because `maturin` trusts an inherited
> `VIRTUAL_ENV` over cwd. Rebuilt with `VIRTUAL_ENV` set explicitly.

**Answering "how does the Rust server run?" concretely:** it is **not** a separate server
you launch. It is a maturin/pyo3 extension module (`_core`) that the **Scheduler** starts
from *inside the scheduler process*
([scheduler.py:1983](../../python/sglang/srt/managers/scheduler.py#L1983), gated by
`_hosts_rust_server()` at
[scheduler.py:1977](../../python/sglang/srt/managers/scheduler.py#L1977)). The usual mental
model — web server launches workers — is backwards here.

**Measured Python vs Rust, same model, same flags:**

| | Python path | Rust path |
|---|---|---|
| processes | 4 | **3** |
| `sglang::detokenizer` | present (7 threads) | **absent** |
| `sglang::scheduler` threads | 56 | **66** |
| `POST /generate`, `POST /v1/chat/completions` | 200 | 200 |
| `GET /health`, `GET /v1/models` | 200 | 200 |
| **`GET /metrics`** | **200** | **404** |
| **`GET /get_server_info`** | **200** | **404** |

Detokenization moves from a Python *process* into Rust *threads* inside the scheduler
(`detokenizer.rs`) — one process vanishes, ~10 threads appear.

**⚠️ The finding that matters most: the Rust server has no `/metrics`.** Turning it on
silently breaks Prometheus, Grafana *and* the TUI, and `status.sh` loses its server-info
section. It is a deliberate switch, not a free speedup. The study run stays on the Python
path by default; the server was restored to it and `/metrics` re-verified at 200.

**The ModelRunner answer has a wrinkle worth the whole phase.** `MlxTpModelWorker`
constructs **two** objects
([mlx/tp_worker.py:79-116](../../python/sglang/srt/hardware_backend/mlx/tp_worker.py#L79-L116)):
a real `MlxModelRunner` and a torch-side `MlxModelRunnerStub` ("no PyTorch weights,
zero-memory KV cache"). So: **one real ModelRunner** per (TP×PP×DP) rank — one here — plus
a stub that exists only to satisfy the scheduler's integration surface.

**That stub is the single root cause of three separately-discovered anomalies:**

| Reading | Reports | Reality |
|---|---|---|
| `attention_backend` (Phase 1) | `torch_native` | MLX serves attention |
| `kv_cache_memory_usage_gb` (Phase 3) | `0.0` | pool is 3,584 MB |
| `weight_memory_usage_gb` (Phase 3) | `0.0` | weights load in 0.74 s |

All three query torch, which correctly answers "I allocated nothing". Excluding them from
the dashboard in Phase 3 was right, and now it is explained rather than merely observed.

**End-to-end trace captured** for a real request (`rid=2a33f8cb…`), ten hops from FastAPI
through both ZMQ boundaries and back, with:
```
queue_duration=0.14ms, forward_duration=1670.96ms
```
Uncontended, the scheduler is ~0.008 % of request latency; all of it is Metal compute.

**IPC topology verified, including a result that looks like a bug and isn't.** `lsof` shows
exactly three bound unix sockets — two on the main process, one on the detokenizer, **zero
on the scheduler**. That is correct: `lsof -U` lists sockets a process *binds*, and the
scheduler only ever *connects*. The bind/connect split is visible in the source as the
trailing boolean at
[tokenizer_manager.py:537-541](../../python/sglang/srt/managers/tokenizer_manager.py#L537-L541)
and [detokenizer_manager.py:114-121](../../python/sglang/srt/managers/detokenizer_manager.py#L114-L121).
2 + 1 + 0 = 3. ✅

---

## 3. Prometheus + Grafana

SGLang already exposes a Prometheus registry at `GET /metrics` on the serving port when
`--enable-metrics` is passed
([utils/common.py:2398](../../python/sglang/srt/utils/common.py#L2398)) — the scheduler
ships its stats to the frontend over a dedicated `metrics_ipc_name` ZMQ socket, so the
numbers are real scheduler state, not HTTP-layer guesses.

**Recommended install: Homebrew, not Docker.** `brew install prometheus grafana` +
`brew services`. Reasons: they run as native arm64 processes so they show up in the same
`ps` tree I am already teaching you to read; no Rancher Desktop VM competing for the same
unified memory the model is using; and scraping `127.0.0.1:30000` needs no
`host.docker.internal` bridging. Docker would work but adds a VM between me and every
observation. **(Q2 — say the word if you prefer the Docker Compose variant instead.)**

Ports: SGLang `30000`, Prometheus `9090`, Grafana `3000`.

Deliverables:
- `obs/prometheus.yml` — 1 s–2 s scrape of the SGLang job (fast, because a 0.6 B model
  decodes so quickly that a 15 s default scrape would alias away everything interesting).
- `obs/grafana/` — provisioned datasource + a **dashboard JSON committed to the repo**, so
  it survives a Grafana reinstall and is reviewable as a diff. Panels grouped by the
  components in Phase 2: request rate / TTFT / ITL / e2e latency histograms; running vs
  queued requests; token throughput (prefill and decode split); KV-cache utilization;
  radix-cache hit rate.
- Optionally a `node_exporter` job for host CPU/memory context. **(Q3 — include host
  metrics, or keep the dashboard purely SGLang?)** I lean toward including it; unified
  memory pressure is the whole story on a laptop.

### ✅ Decisions made (executed 2026-08-14 17:06)

**Q2 → Homebrew** and **Q3 → yes, include node_exporter**, both as you chose.
Installed `prometheus 3.13.2`, `grafana 13.1.3`, `node_exporter 1.12.1`.

1. **Homebrew for *installation*, but NOT `brew services` for *lifecycle*.** This is a
   deviation from the plan's wording and worth stating plainly. `brew services` installs
   login-time launchd agents and reads `/opt/homebrew/etc/*.yml` — that would put this
   study run's config outside the repo and leave three daemons starting on every boot,
   which you never asked for. Instead `bin/start_obs.sh` runs all three binaries directly
   against configs in `obs/`, with pidfiles in `run/` and data in `obs/data/`
   (gitignored). `bin/stop_obs.sh` only kills pids it recorded, so it cannot take down an
   unrelated Grafana.
2. **Scrape intervals: 1 s for sglang, 5 s for the host, 15 s for Prometheus itself.**
   The 15 s default would alias away everything interesting — a 128-token response
   completes in ~1–2 s here, so a whole request lifecycle can fall between two scrapes.
   1 s is the floor worth using: the scheduler only refreshes its gauges every
   `--decode-log-interval` (20) iterations, so faster buys nothing but CPU.
3. **Dashboard is generated by a checked-in Python script**
   (`obs/grafana/dashboards/build_dashboard.py` → `sglang-runtime.json`) rather than
   hand-authored or exported from the Grafana UI. Overlapping `gridPos` is a *silent*
   failure — Grafana just stacks panels — so the generator is paired with a grid-overlap
   assertion. A UI export would also bake in machine-specific ids and be unreviewable as
   a diff.
4. **Colour follows the validated dark-mode categorical palette**, checked with the
   `dataviz` validator (lightness band, chroma floor, CVD separation, normal-vision
   floor, contrast — all PASS against surface `#1a1a19`). Quantile panels (p50/p95/p99)
   use a **single-hue sequential ramp** instead of categorical hues, because quantiles are
   ordinal, not distinct entities. "Available" KV space is neutral gray — free space is
   not a peer category of used space. No panel uses two y-axes.

### ✅ Implementation details

**Files:** `obs/prometheus.yml`, `obs/grafana/grafana.ini`,
`obs/grafana/provisioning/{datasources,dashboards}/*.yml`,
`obs/grafana/dashboards/build_dashboard.py` + generated `sglang-runtime.json`,
`bin/start_obs.sh`, `bin/stop_obs.sh`.

Provisioning files cannot use relative paths, but hardcoding an absolute path into a
tracked file would break on any other machine — so the templates carry `__DASHBOARD_DIR__`
/ `__HOME_DASHBOARD__` placeholders that `start_obs.sh` substitutes into a rendered copy
under `obs/data/`. The tracked files stay machine-independent.

**Dashboard: 19 panels in 6 rows** — at-a-glance stat tiles; request flow; latency
(TTFT / ITL / e2e / queue-time quantiles); throughput by mode; KV pool occupancy and
prefix-cache hit rate; host CPU / memory / swap.

**Verification: all 32 panel queries were run against the Prometheus API under real load
(24 concurrent requests). 0 broken, 0 empty.** Representative peaks:

| Panel | Peak observed |
|---|---|
| Generation throughput | 436.7 tok/s |
| TTFT p50 / p99 | 1.5 s / 2.48 s |
| Inter-token latency p50 | 0.0175 s |
| E2E latency p50 / p99 | 2.75 s / 4.94 s |
| Token throughput (decode) | 105.5 tok/s |
| Prefix cache hit rate (windowed) | 36.3 % |
| KV evictable / available | 382 / 32,760 tokens |

Grafana confirms the dashboard provisioned into folder "SGLang Study" with all 19 panels,
and the datasource health check returns `OK - Successfully queried the Prometheus API`.

### ⚠️ Findings

1. **The default latency buckets are unusable on this machine — fixed.** `sglang`'s
   default TTFT *and* e2e buckets both start at **0.1 s**; they are tuned for large models
   on datacenter GPUs. Measured TTFT here is ~0.026 s idle, so every fast request piled
   into the first bucket and `histogram_quantile()` could not resolve anything below
   100 ms — p50 TTFT would have been a flat, meaningless `0.1`. Added
   `--bucket-time-to-first-token` (0.005 s floor) and `--bucket-e2e-request-latency`
   (0.05 s floor) to `bin/env.sh`, and **verified the new boundaries appear in `/metrics`**.
   ITL was left alone: its default already starts at 0.002 s and the measured ~0.017 s mean
   lands mid-range.
2. **Two memory metrics read `0.0` under MLX and are excluded from the dashboard.**
   `sglang:kv_cache_memory_usage_gb` and `sglang:weight_memory_usage_gb` measure *torch-side*
   allocation, which the MLX backend bypasses — so a "KV cache GB" panel would have
   confidently displayed **0 GB while the pool actually held 3.5 GB**. This is the same
   class of artifact as `attention_backend=torch_native` from Phase 1. The KV panel counts
   **tokens** instead, which the scheduler computes directly and which are correct.
3. **Host metrics immediately earned their place (Q3 was the right call).** During load the
   host showed **10.2 GB compressed memory and 1.34 GB swap in use**. That is real memory
   pressure on a 36 GB machine, it is completely invisible from sglang's own metrics, and
   it retroactively justifies the Phase 1 decision to cap the KV pool.
4. **`node_exporter` on macOS exposes different metrics than on Linux.** There is no
   `node_memory_MemAvailable_bytes`; darwin provides `node_memory_{wired,active,compressed,
   free,inactive,purgeable,internal}_bytes` plus swap. Every name in the dashboard was read
   off the live endpoint rather than copied from a Linux dashboard.
5. **`num_queue_reqs` is always 0 in normal operation — and that is correct, not a broken
   panel.** The default `max_running_requests=4096` means the scheduler admits everything
   immediately, so nothing ever waits. Rather than leave the panel unverified I proved it
   by restarting with `--max-running-requests 2` under 10-way load:
   ```
   running=2  queued=8
   running=2  queued=6
   running=2  queued=4
   running=2  queued=2
   ```
   Admission control pinning the running batch at 2 while the queue drains. **To make the
   queue panel show anything, launch with
   `./bin/start_sglang.sh --max-running-requests 2`.**

### ⚠️ Not verified: the rendered pixels

I checked the dashboard **structurally** (no `gridPos` overlaps, unique panel ids, 24-col
bounds) and **by data** (all 32 queries return non-empty series). I have **not** looked at
the rendered dashboard — Grafana's `/render` endpoint returns HTTP 500 because the
image-renderer plugin is not installed, and I did not install it unasked. So label
collisions, legend overflow or awkward panel heights would not have been caught.
**Please eyeball <http://127.0.0.1:3000> and tell me what needs adjusting** — or say the
word and I will install `grafana-image-renderer` so I can check it myself.

---

## 4. TUI

Grafana answers "what are the trends"; the TUI answers "what are the *processes* doing
right now". They are deliberately not the same view.

Build: `tui/sglang_tui.py`, a `rich`-based live dashboard (`rich` 15.0 already in the venv;
**no new dependency**, and no `textual` install needed).

Panels:
1. **Process tree** — main / `sglang::scheduler` / `sglang::detokenizer` discovered by
   `setproctitle` name via `psutil`, with per-process CPU%, RSS, thread count, and uptime.
   This makes the multi-process architecture *visible*, which is the actual ask in item 2.
2. **Scheduler state** — running / queued / KV-cache usage / cache-hit rate, parsed from
   the `/metrics` scrape.
3. **Throughput** — prefill and decode tokens/s with a small sparkline history.
4. **Latency** — TTFT / ITL / e2e, p50 & p99 from the histogram buckets.
5. **IPC topology** — the ZMQ unix sockets each process holds, from `lsof`. Static, but it
   is the clearest possible illustration of who talks to whom.
6. **Recent requests** — tail of the structured request log from Phase 6.

Refresh ~1 Hz, read-only, no writes to the server. Runs in its own terminal; `q` quits.

### ✅ Decisions made (executed 2026-08-14 17:28)

Built as planned — `tui/sglang_tui.py` + `bin/tui.sh`, `rich` only, **no new
dependencies**, all six panels. Four decisions worth recording:

1. **Added a non-interactive mode (`--once` / `--iterations N`).** Not in the original
   plan. A `rich.Live` full-screen app cannot be verified without a TTY, and "it launched
   without crashing" is not verification. These flags render N frames to stdout, which is
   how every screenshot in this document was produced and how the bugs below were caught.
2. **The TUI parses JSON, not text** — using the Phase 6 `--log-requests-format json`
   finding. No regex against human-readable log lines.
3. **`lsof` is refreshed every 15th tick, not every tick.** It costs >100 ms and the
   socket set is static once the server is up; polling it at 1 Hz would make the whole
   TUI stutter.
4. **The scheduler panel shows cache hit rate twice** — cumulative *and* windowed. See
   finding 4 below; showing only the cumulative gauge would have been actively misleading.

### ✅ Implementation details

Six panels, laid out with fixed heights for the top two rows and a flexible bottom row:

| Panel | Source | Notes |
|---|---|---|
| Processes | `psutil` | pid, component, RSS, CPU%, threads, uptime, total |
| Scheduler | `/metrics` | running/queued, KV stacked bar, cache hit ×2 |
| Throughput | `/metrics` rates | decode / prefill-compute / prefill-cache + sparklines |
| Latency | `/metrics` histograms | p50/p95/p99 computed locally by bucket interpolation |
| IPC topology | static + `lsof` | ASCII wiring diagram + live unix socket paths |
| Recent requests | JSON request log | health checks filtered, real durations |

Verified under 14-way concurrent load. Representative live frame:

```
   PID COMPONENT                                RSS   CPU  THR      UP
 77398 HTTP server + TokenizerManager          447M    1%   23  17m12s
 77530 └─ resource_tracker (stdlib)             10M    0%    1  17m06s
 77531 └─ Scheduler +TpWorker +ModelRunner   1,609M   58%   57  17m06s
 77532 └─ DetokenizerManager                   344M    0%    7  17m06s
       total                                 2,409M

 tokens/s              now  last 30s        │            p50      p95      p99
 decode              143.8  █▁              │ TTFT     817.4ms   1.41s    2.23s
 prefill (compute)    10.4  █▁              │ inter-t   17.6ms  22.8ms   33.4ms
 prefill (cache hit)   4.7  █▁              │ e2e        2.43s   2.94s    2.99s
 requests/s           1.42  ▁█              │ queue      6.5ms   9.7ms    9.9ms
```

The process panel is the payoff for item 2 of the request: it shows the four-process
architecture at a glance, with the scheduler visibly holding ~1.6 GB and ~58% CPU while
the other three idle — and with the `resource_tracker` explicitly labelled *stdlib, not
sglang* so the component count cannot be misread as four.

### ⚠️ Four bugs found by actually rendering it

Every one of these would have survived a "does it launch?" check.

1. **Only 1 of 4 processes was detected.** On macOS `psutil.name()` returns `"Python"` for
   the forked workers — `setproctitle` rewrites **argv**, which surfaces in `cmdline`, not
   in `name`. My matcher only looked at `name`, so it found the parent and silently missed
   the scheduler and detokenizer. Now matches against `name + " " + cmdline`.
2. **The `resource_tracker` was missing**, because I folded the sibling sweep inside an
   `if parent not in hits` guard — and the parent is *always* matched directly (its cmdline
   contains `sglang.launch_server`), so the sweep never ran. Split into two passes.
3. **Every request showed `DUR 0.00s`.** I computed duration as wall-clock at parse time,
   but a single poll normally reads *both* the `request.received` and `request.finished`
   lines for a request, so it was measuring `now - now`. Now uses the log's own ISO
   timestamps. Durations went from a uniform `0.00s` to `2.22s / 2.23s / 2.23s`, which
   cross-checks against the independently-measured e2e p50 of 2.43 s.
4. **`prefix cache hit` read `0.0%` during traffic that Grafana scored at 36%.**
   `sglang:cache_hit_rate` is a scheduler gauge that decays to 0 between log intervals.
   The panel now shows cumulative *and* a windowed rate computed from
   `prefill_effective_tokens_total`, the same expression the Grafana panel uses.

Two smaller layout fixes: `rich` collapsed the Scheduler panel's label column to zero
width (labels vanished entirely) until both columns got explicit ratios, and long
component names wrapped to three lines until the role column got
`no_wrap` + `overflow="ellipsis"`.

### Notes on honest gaps

- A request already in flight when the TUI starts shows `DUR —`, because the tail seeks to
  EOF on startup and never saw its `request.received`. That is correct behaviour, and the
  em-dash is deliberate rather than a fabricated 0.
- `cache hit (windowed)` shows `—` when no prefill happened in the sample window. Also
  deliberate: there is no rate to report, and 0.0% would be a different claim.
- Throughput rates depend on the scheduler refreshing its counters, which it does every
  `--decode-log-interval` (20) iterations — so at very low load the rate can legitimately
  read 0 between updates.

---

## 5. Network traffic recording

This splits into two tiers because of the `ipc://` finding in Phase 0. I want to be up
front that tier B is the hard one.

### Tier A — client ↔ API server HTTP (fully achievable)
`net/capture_http.sh`: `tcpdump -i lo0 -s 0 -w pcap/<ts>.pcap 'tcp port 30000'`.
Requires **`sudo`** — the script will prompt, and I will not run privileged capture
without telling you. Decoding: a small `net/decode_pcap.py` that reassembles the loopback
TCP stream and prints request/response lines + SSE token chunks, so you can literally
watch streaming deltas arrive. `tshark` stays **optional** (`brew install wireshark` is a
large install for a decode I can do in ~100 lines).

Also recorded here: the Prometheus → SGLang `/metrics` scrapes (same port, same capture)
and Grafana → Prometheus on 9090, so the whole observability loop is on tape.

### Tier B — internal ZMQ IPC (best-effort, two options)
Unix domain sockets carry no packets `tcpdump` can see. Options, in the order I would try:

- **B1 — force ZMQ onto TCP loopback.** `--enable-dp-attention` switches SGLang from
  `ipc://` to `tcp://127.0.0.1:<port_base+N>`
  ([server_args.py:9760-9790](../../python/sglang/srt/server_args.py#L9760-L9790)), which
  makes every tokenizer↔scheduler↔detokenizer message capturable on `lo0`. Risk: DP
  attention may be unsupported or nonsensical on the MLX backend. I will *try* it, and if
  it fails I will say so rather than pretend.
- **B2 — application-level tap.** A tiny ZMQ monitor/proxy or a logging shim on the
  send/recv call sites, giving decoded Python-object-level messages instead of raw frames.
  Strictly more readable than packets, strictly less "real network traffic".

**(Q4)** If B1 fails, do you want B2 (invasive: a temporary shim in `python/sglang/srt/`,
which I would keep on a scratch branch and not commit), or should I stop at documenting
the socket topology via `lsof` and leave the source untouched? My default is the
non-invasive option unless you say otherwise.

### ✅ Decisions made (executed 2026-08-14 18:22)

**I cannot run packet capture on this machine.** `sudo` requires a password, `/dev/bpf*` is
`crw------- root:wheel`, and this account is not in `access_bpf`. Rather than leave the
phase blocked on an interactive password prompt, I split Tier A in two:

1. **Built a non-root application-layer tap (`net/http_tap.py`, `bin/tap.sh`) — this works
   today and is the primary deliverable.** A recording TCP proxy on `:30001` that forwards
   to `:30000` and writes every exchange to `pcap/http-<ts>.jsonl` (structured) and
   `.raw` (bytes, both directions).
2. **Wrote the `sudo` path for you to run** — `bin/capture_http.sh` (tcpdump on `lo0`) plus
   `net/decode_pcap.py`, a dependency-free pcap decoder. **I verified the decoder without
   root** by synthesising a pcap from real captured bytes (`net/make_test_pcap.py`).
3. **Did not install `tshark`.** `brew install wireshark` is a large install for a decode
   that fits in ~200 lines, and the decoder is now tested.
4. **Q4 answered by circumstance: no source shim was needed, and none was written.** The
   repo is untouched.

### ✅ Implementation details

**Tier A1 — the non-root tap works, and produced a result the metrics can't.** Verified on
real traffic:

```
POST /generate            -> 200  ttfb=0.3441s total=0.3442s in=217B out=878B
POST /v1/chat/completions -> 200  ttfb=0.0404s total=5.3636s in=341B out=9998B sse=32
```

For streaming responses it records **per-chunk arrival times**:

```
first 8 chunk arrival times (s): [0.0405, 0.063, 0.0735, 0.085, 0.0967, 0.1086, 0.1203, 0.1322]
inter-chunk gaps (s):            [0.0225, 0.0105, 0.0115, 0.0117, 0.0119, 0.0117, 0.0119, 0.0116]
```

Those ~11.7 ms gaps are an **independent, wire-level measurement of inter-token latency**,
derived from arrival timestamps rather than from the server's own histogram — and they
corroborate the `sglang:inter_token_latency_seconds` p50 of ~17 ms seen under heavier
concurrency. A pcap would not give this without extra work; the tap gives it for free.

Trade-off, stated plainly: the tap loses the TCP/IP layer (handshakes, window sizes,
retransmits) and only sees traffic addressed to it. Use `capture_http.sh` when the
TCP layer is the object of study.

**Tier A2 — the pcap decoder is verified.** It handles pcap and pcapng containers, macOS
`DLT_NULL` loopback *and* Ethernet framing, IPv4/IPv6, TCP reassembly with retransmit
dedup by sequence number, HTTP message framing, and SSE event splitting:

```
--- connection 127.0.0.1:51101 -> 127.0.0.1:30000 ---
  REQ  POST /v1/chat/completions HTTP/1.1   (211 B)
  RESP HTTP/1.1 200 OK   (338 B)
       SSE: 5 events
```

`capture_http.sh` filters ports 30000, 9090 and 9100, so a capture records the **whole
observability loop** — client↔sglang, Prometheus↔sglang scrapes, Grafana↔Prometheus,
Prometheus↔node_exporter — not just client traffic.

### Tier B — internal ZMQ: one partial success, one honest failure

**B0 (`--enable-forward-pass-metrics`) — transport works, payload is empty on MLX.**

The plumbing is real and I proved it end to end. Launching with
`--forward-pass-metrics-ipc-name ipc:///tmp/sglang-fpm` binds a ZMQ **PUB** socket
(`FPM: ZMQ PUB bound on ipc:///tmp/sglang-fpm.0`), and `net/zmq_fpm_tap.py` subscribes,
decodes the multipart `(topic, seq_uint64_be, msgspec-msgpack)` frames, and writes JSONL.
**14 messages received in 14.8 s.**

But **all 14 were idle heartbeats** — `wall_time: 0.0` and every counter zero, at exactly
1/s — despite five concurrent requests running the whole time. Root cause, traced to a
single early return:

```python
# metrics_reporter.py:981-986
if self.scheduler._fpm_uses_device_timer:
    self.forward_pass_device_timer._report()
    wall_time = self.scheduler._fpm_gpu_time_acc
    self.scheduler._fpm_gpu_time_acc = 0.0
    if wall_time == 0.0:
        return          # <-- every per-iteration emit dies here
```

`wall_time` comes from `DeviceTimer`, which is implemented with **`torch.cuda.Event`**
([utils/device_timer.py:87-98](../../python/sglang/srt/utils/device_timer.py#L87-L98)), and
every `device_timer_ctx(...)` call site lives under `model_executor/` — `model_runner.py`,
`eager_runner.py`, `decode_cuda_graph_runner.py`. **There are zero call sites under
`hardware_backend/mlx/`.** So on MLX nothing ever wraps a forward, the accumulator stays
0.0, and each iteration returns early. Only the heartbeat escapes.

**This is the fourth instance of the same root cause** as the three in Phase 2 §6:
instrumentation bound to the torch path, which MLX bypasses. On a CUDA box this tap would
carry real per-iteration data; the script is ready for that.

**B1 (`--enable-dp-attention`) — failed, and I initially misread it as a success.**

The server started fine (16 s, no crash — my `/dev/shm` concern was unfounded) and the
scheduler suddenly had **eight TCP listeners** where before it bound nothing. I took that
as the ZMQ moving to TCP. **It was not.** Checking the other processes showed the ipc://
sockets still in place, in exactly the old 2 + 1 + 0 pattern:

```
pid 58043 (main):         2 bound unix sockets
pid 58164 (scheduler):    0 bound unix sockets
pid 58165 (detokenizer):  1 bound unix socket
```

and the *derived* ZMQ ports the TCP branch would have used (`port_base` = 30000 + 233 + 1
= 30234 onward) were all **free**. The scheduler's new TCP listeners are
`torch.distributed`/gloo sockets, which appear because dist init runs — not ZMQ.

Why: the run's own `server_args` dump shows **`enable_dp_attention=False`** even though I
passed `--enable-dp-attention`. It is a `resolvable=True` argument
([server_args.py:1140-1147](../../python/sglang/srt/server_args.py#L1140-L1147)) and gets
resolved off for this model — its help text says DP attention supports "DeepSeek-V2 and
Qwen 2/3 **MoE**" models, and Qwen3-0.6B is dense. So `PortArgs.init_new` took the `if not
enable_dp_attention` branch and everything stayed on `ipc://`.

Confirming this on an MoE model would mean downloading one, which is out of scope for this
laptop and not worth it for a capture technique that is a means, not an end.

### Conclusion on item 4 of the request

| Traffic | Status |
|---|---|
| Client ↔ API server HTTP, incl. SSE token-by-token | ✅ **recorded** (non-root tap, working now) |
| Prometheus ↔ sglang, Grafana ↔ Prometheus, ↔ node_exporter | ✅ script ready (`capture_http.sh`, needs your `sudo`) |
| Packet-level TCP/IP of the above | ✅ script + verified decoder, ⏳ needs your `sudo` to run |
| Internal ZMQ, per-iteration scheduler telemetry | ⚠️ transport proven, **payload empty on MLX** (torch-bound timer) |
| Internal ZMQ, raw inter-process frames | ❌ **not achievable** here — unix sockets, and the TCP path needs an MoE model |

The one genuinely closed door is raw ZMQ frame capture, for a reason that is structural
rather than a missing tool: single-node SGLang deliberately uses unix domain sockets, which
carry no packets. Everything else is either captured or one `sudo` away.

---

## 6. Log level — detailed but not overwhelming

Proposed baseline, tuned so that you see per-request lifecycle and per-iteration scheduler
state without drowning in per-layer chatter:

| Flag | Value | Why |
|---|---|---|
| `--log-level` | `info` | `debug` on the scheduler loop is genuinely unreadable at 0.6 B decode speed |
| `--log-level-http` | `warning` | suppresses one uvicorn access line per Prometheus scrape — at a 1 s scrape that is otherwise 3600 useless lines/hour |
| `--log-requests` | on | the per-request lifecycle, which is the thing worth reading |
| `--log-requests-level` | `1` | ids + lengths + finish reason, **not** full prompt/output text |
| `--decode-log-interval` | `20` | scheduler stats line ~1×/s at laptop decode rates |
| `--enable-request-time-stats-logging` | on | per-stage timings, feeds the TUI's latency panel |
| `--show-time-cost` | off initially | opt-in when profiling a specific phase |

Everything goes to `logs/server-<ts>.log` via tee, so the TUI can tail it and you keep a
scrollback. I will verify the exact accepted values of `--log-requests-level` against
`server_args.py` before writing them into the launch script rather than trusting this
table.

### ✅ Decisions made (executed 2026-08-14 16:56)

I did verify the table against `server_args.py`, and **two entries were wrong**:

1. **`--log-level-http warning` was the wrong tool.** SGLang has
   **`--uvicorn-access-log-exclude-prefixes`**
   ([server_args.py:1499-1508](../../python/sglang/srt/server_args.py#L1499-L1508)), which
   suppresses access logs *by path prefix*. That is strictly better: passing
   `/metrics /health` kills exactly the scrape noise while **keeping** access logs for real
   requests, whereas `--log-level-http warning` would have thrown away both. Note the
   default is `()` — empty — so it must be passed explicitly.
   **Adopted `--uvicorn-access-log-exclude-prefixes /metrics /health`; dropped
   `--log-level-http`.**
2. **My description of `--log-requests-level 1` was slightly off.** The real semantics are
   `0` = metadata without sampling params, `1` = metadata **+ sampling params**, `2` =
   + partial input/output, `3` = every input/output (default is **2**). Level 1 is still
   the right choice — it is the highest level that logs **no prompt or output text** — but
   it is "metadata + sampling params", not "ids + lengths + finish reason".

Three additions the audit turned up that were not in the original table:

3. **`--log-requests-format json` + `--log-requests-target stdout <dir>`.** Requests can be
   emitted as structured JSON to *both* stdout and a file. The filename is deterministic:
   `<dir>/<hostname>_<rank>.log`, hourly-rotating
   ([log_utils.py:35-45](../../python/sglang/srt/utils/log_utils.py#L35-L45)). **This
   changes Phase 4** — the TUI can parse JSON instead of regexing human-readable text.
4. **`--reasoning-parser qwen3`** — confirmed `qwen3` is a valid choice (26 parsers
   registered in `ReasoningParser.DetectorMap`). Adopted, per the Phase 1 finding.
5. **`--crash-dump-folder`** — dumps the last 5 minutes of requests on a crash. Cheap
   insurance while experimenting; pointed at `logs/crash/`.

**Deferred, but noted as significant: `--enable-forward-pass-metrics`**
([server_args.py:1610-1614](../../python/sglang/srt/server_args.py#L1610-L1614)) publishes
**per-iteration forward-pass metrics over a dedicated ZMQ IPC endpoint** that external
consumers can subscribe to. This is a *sanctioned* way to observe internal per-iteration
state without patching source — so it is a serious candidate for **Phase 5 tier B**, and
better than either B1 or B2. Revisit at Phase 5.

### ✅ Implementation details

Final flag set (lives in `bin/start_sglang.sh`, values in `bin/env.sh`):

```
--log-level info
--log-requests --log-requests-level 1
--log-requests-format json
--log-requests-target stdout <run>/logs/requests
--uvicorn-access-log-exclude-prefixes /metrics /health
--decode-log-interval 20
--enable-request-time-stats-logging
--reasoning-parser qwen3
--crash-dump-folder <run>/logs/crash
```

**Measured signal-to-noise** on a full startup + smoke test + 10 deliberate
`/metrics`+`/health` probes:

| Category | Lines |
|---|---|
| startup/banner ("other") | 70 |
| request JSON | 24 |
| Prefill batch | 12 |
| uvicorn access (real requests only) | 7 |
| Decode batch | 4 |
| **`/metrics` or `/health` access lines** | **0** ✅ |
| **Total** | **117** |

Zero scrape-noise lines from ten probes, while `POST /generate` and `POST /v1/chat`
still produce access logs. That is the intended trade exactly.

Request log entries are clean structured JSON with `request.received` / `request.finished`
events, `rid`, and `sampling_params`, and **no prompt text**:
```json
{"timestamp": "2026-08-14T16:56:53.632243", "event": "request.received",
 "rid": "46cda157688c4dbcb93bf71ef82c9b99",
 "obj": {"sampling_params": {"temperature": 0, "max_new_tokens": 8}, ...}}
```

**`--reasoning-parser qwen3` verified working.** Same prompt, before and after:

| | before | after |
|---|---|---|
| `content` | `'<think>\nOkay, the user wants...'` | `''` |
| `reasoning_content` | `null` | `'\nOkay, the user wants...'` |
| `reasoning_tokens` | `0` | `32` |

The `<think>` block is now split out, so Phase 3's token panels can distinguish reasoning
from answer instead of blending them.

**⚠️ One residual noise source for Phase 4:** `/health` probes generate real requests with
`rid` prefixed `HEALTH_CHECK_`, and those *do* appear in the request log (the access-log
exclusion only covers uvicorn's HTTP layer, not the request logger). The TUI's "recent
requests" panel must filter `rid.startswith("HEALTH_CHECK_")` or it will show nothing but
health checks.

---

## 7. Control scripts

All under `studyIterations/iter260814/bin/`, all `set -euo pipefail`, all safe to re-run.

| Script | Does |
|---|---|
| `env.sh` | single source of truth: venv path, model path, ports, log dir. Sourced by everything else. |
| `start_sglang.sh` | launch the server with the Phase 1 + Phase 6 flags, write `run/sglang.pid`, tee to `logs/`, wait for `/health` before returning 0 |
| `start_obs.sh` | `brew services start prometheus grafana` (or bring up the compose file), wait for both to answer, print the dashboard URL |
| `stop_sglang.sh` | graceful `SIGTERM` to the process **tree** (scheduler + detokenizer are children and *will* orphan if you kill only the parent), escalate to `SIGKILL` after a timeout, then verify port 30000 is released |
| `stop_obs.sh` | stop Prometheus/Grafana |
| `stop_all.sh` | the panic button: everything, including a stray-`sglang::`-process sweep |
| `restart_all.sh` | `stop_all` → `start_obs` → `start_sglang` → `status` |
| `status.sh` | one-shot text summary: PIDs, ports, health endpoints, disk used by logs/pcaps |
| `tui.sh` | run the Phase 4 TUI against the running server |
| `capture_http.sh` | Phase 5 tier A capture (prompts for sudo) |
| `smoke.sh` | one `/generate` + one `/v1/chat/completions` to prove the thing works |

The orphaned-child hazard in `stop_sglang.sh` is the one real correctness trap here —
`kill $(cat pid)` alone leaves `sglang::scheduler` holding the GPU and the ZMQ sockets, and
the next start then fails on a port/socket collision. The script will kill the tree and
verify.

### ✅ Decisions made (executed 2026-08-14 16:56)

**Written and verified: `env.sh`, `start_sglang.sh`, `stop_sglang.sh`, `stop_all.sh`,
`restart_all.sh`, `status.sh`, `smoke.sh`.** Deferred to their own phases:
`start_obs.sh`/`stop_obs.sh` (Phase 3), `tui.sh` (Phase 4), `capture_http.sh` (Phase 5).
`restart_all.sh` and `stop_all.sh` already call the obs scripts *conditionally*
(`[[ -x ... ]]`), so they light up automatically when Phase 3 lands — no edit needed then.

1. **`env.sh` is the single source of truth** and resolves `RUN_DIR`/`REPO_ROOT` from
   `BASH_SOURCE`, so every script works from any cwd.
2. **Wrote for bash 3.2, not bash 5.** `/bin/bash` on macOS is 3.2.57 while
   `/opt/homebrew/bin/bash` is 5.3.3 — `#!/usr/bin/env bash` picks whichever won the PATH
   race. My first draft of `stop_sglang.sh` used `mapfile` (bash 4+), which would have
   worked when I tested it and then broken silently under a different PATH. Replaced with a
   portable `while IFS= read -r` loop. **Both versions now `bash -n` clean.**
3. **Capped the KV pool at `--max-total-tokens 32768`**, acting on the Phase 1 finding.
4. **`--enable-metrics` is on from now on**, since both Phase 3 and Phase 4 need `/metrics`
   and it costs nothing.

### ✅ Implementation details

**`sglang_pids()` in `env.sh` is the core of the stop logic.** It unions three sources —
the pidfile's process *and its children*, anything matching the `sglang::` setproctitle
names, and any stray `sglang.launch_server` parent that outlived its pidfile — then
deduplicates. `stop_sglang.sh` sends `SIGTERM` to the whole set, waits 15 s, escalates to
`SIGKILL`, then **verifies** both that no pid survived and that port 30000 was released,
rather than assuming.

**Proof the orphan hazard is real and handled** — before/after on a live server:
```
=== BEFORE ===  34866 launch_server | 35006 sglang::scheduler | 35007 sglang::detokenizer
[16:56:21] stopping 4 process(es): 34866 35005 35006 35007
[16:56:28] port 30000 released
[16:56:28] stopped cleanly
=== AFTER ===   (none)   port 30000 free
```
It found **4** processes, not the 1 in the pidfile — the two `sglang::` children plus the
stdlib `multiprocessing.resource_tracker`. A naive `kill $(cat pid)` would have left the
scheduler alive holding 1.4 GB and the ZMQ sockets. Clean teardown took ~7 s.

**`restart_all.sh` round-trips in ~20 s** (stop → start → ready in 17 s → smoke pass):
```
[16:56:38] === 1/4  stopping everything ===
[16:56:38] === 2/4  observability stack not set up yet (Phase 3), skipping ===
[16:56:38] === 3/4  starting sglang ===
[16:56:55] READY in 17s
[16:56:59] smoke test PASSED
```

**The KV cap took effect at the pool level**, not just in the reported config:
```
before: MlxAttentionKVPool: 112665 slots x 28 x 8 x 128, ~12322.7 MB
after : MlxAttentionKVPool:  32769 slots x 28 x 8 x 128,  ~3584.1 MB
```
A **3.4× reduction**, freeing ~8.7 GB of unified memory for Prometheus, Grafana and the
TUI. `/get_server_info` confirms `max_total_num_tokens = 32768`.

**`status.sh`** prints the process table (with RSS in MB), colour-coded UP/down probes for
all four endpoints, parsed server info, disk usage, and the paths of the current server and
request logs. `start_sglang.sh` calls it automatically on success.

**`start_sglang.sh` readiness** polls `/health` for a 200 with a 180 s deadline, while also
checking the process is still alive each second — so a crash during startup fails fast with
the last 30 log lines instead of hanging until timeout. This is the Phase 1 "503 during
warmup" finding, handled.

**`smoke.sh`** exercises `/generate`, `/v1/chat/completions` and streaming SSE, and exits
non-zero on any failure so it works as a gate in `restart_all.sh`.

**`stop_all.sh`** additionally detects a `tcpdump` left running from Phase 5 and prints the
`sudo pkill` needed — it deliberately does **not** try to sudo on your behalf.

---

## Proposed layout

```
studyIterations/
├── .gitignore                  # models/ venvs/ logs/ pcap/ run/ data/
├── models/Qwen3-0.6B/          # already present (ModelScope)
├── venvs/mps-py312/            # already present
└── iter260814/
    ├── localRunGrafanaTUI.md   # your request
    ├── localRunPlan.md         # this file, updated per phase
    ├── components.md           # Phase 2 deliverable
    ├── runbook.md              # how to drive all of the above
    ├── bin/                    # Phase 7 scripts
    ├── obs/                    # prometheus.yml, grafana provisioning + dashboard JSON
    ├── tui/sglang_tui.py       # Phase 4
    ├── net/                    # capture + decode
    ├── logs/                   # gitignored
    ├── pcap/                   # gitignored
    └── run/                    # pidfiles, gitignored
```

## Execution order

`1 → 6 → 7 → 3 → 4 → 2 → 5`

Rationale: get a server up and its logging sane first (1, 6); wrap it in start/stop scripts
immediately (7) so every later phase can restart cleanly; then metrics (3) and the TUI (4),
which give me the instruments to *observe* the components; write the component deep-dive
(2) once I can point at live evidence instead of just source; network capture (5) last
because it is the most likely to hit a macOS-specific wall.

I will update this file's `Decisions made` / `Implementation details` blocks at the end of
each phase.

## Risks, stated up front

| Risk | Likelihood | Handling |
|---|---|---|
| MLX backend fails on this Qwen3 build | low | documented, supported path; fall back to `--device cpu --attention-backend torch_native` |
| `rust/sglang-server` won't build on macOS arm64 (needs rustup 1.92) | **medium-high** | best-effort; document architecture from source + the failure if so |
| ZMQ IPC not capturable (Phase 5 tier B) | **high** | tiers A/B1/B2 above; Q4 |
| `brew install grafana` is slow | low | one-time; Docker fallback available |
| tcpdump needs sudo | certain | script prompts; you stay in control |

## Questions before I start

- **Q1** Primary model: Qwen3-0.6B **bf16** (my default) or the 4-bit MLX build?
- **Q2** Prometheus/Grafana via **Homebrew** (my default) or Docker Compose?
- **Q3** Include `node_exporter` host metrics on the dashboard? (I lean yes.)
- **Q4** If ZMQ-over-TCP (B1) fails, allow a temporary source-level IPC shim (B2), or stay
  non-invasive? (I default to non-invasive.)

None of these block me from starting Phase 1 — I can begin on the defaults and adjust.
